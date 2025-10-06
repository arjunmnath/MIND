import os
from dataclasses import asdict
from typing import List, Tuple

import fsspec
import mlflow
import torch
from config_classes import Snapshot, TrainingConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils import upload_to_s3


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        loss_fn,
        optimizer,
        metrices,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.metrices = metrices
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.acc = torch.accelerator.current_accelerator()
        self.device: torch.device = torch.device(f"{self.acc}:{self.local_rank}")
        self.device_type = self.device.type

        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (
            self._prepare_dataloader(test_dataset) if test_dataset else None
        )

        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = self.config.save_every

        if self.config.use_amp:
            self.scaler = GradScaler(self.device_type)
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        upload_to_s3(snapshot, self.config.snapshot_path)

        print(f"Snapshot saved at epoch {epoch}")

    def _run_batch(
        self,
        history: torch.Tensor,
        clicks: torch.Tensor,
        non_clicks: torch.Tensor,
        train: bool = True,
    ) -> Tuple[List[float], float]:
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type=self.device_type,
            dtype=torch.float16,
            enabled=(self.config.use_amp),
        ):
            indexes, relevance, target = self.model(history, clicks, non_clicks)
            loss = self.loss_fn(relevance, target)
            metrices = [
                metric(relevance, target, indexes=indexes) for metric in self.metrices
            ]

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.optimizer.step()

        return metrices, loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if train:
            dataloader.sampler.set_epoch(epoch)
        for iter, (history, clicks, non_clicks) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            history = history.to(self.local_rank)
            clicks = clicks.to(self.local_rank)
            non_clicks = non_clicks.to(self.local_rank)
            torch.cuda.empty_cache()
            if train:
                self.model.train()
            else:
                self.model.eval()
            metrices, batch_loss = self._run_batch(history, clicks, non_clicks, train)
            if iter % 100 == 0:
                print(
                    f"[RANK{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f} |"
                    f" auc: {metrices[0]:.5f} | ndcg@5: {metrices[1]:.4f} | ndcg@10: {metrices[2]:.4f}"
                )
                mlflow.log_metric("loss", batch_loss, step=iter)
                mlflow.log_metric("auc", metrices[0], step=iter)
                mlflow.log_metric("ndcg_5", metrices[1], step=iter)
                mlflow.log_metric("ndcg_10", metrices[2], step=iter)

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
