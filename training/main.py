import os
import sys
from pathlib import Path

import hydra
import mlflow
import torch
from config_classes import DataConfig, MLFlowConfig, OptimizerConfig, TrainingConfig
from dataset import Mind
from models import *
from models.models import InfoNCE, TwoTowerRecommendation
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG
from trainer import Trainer
from utils import create_optimizer


def verify_min_gpu_count(min_gpus: int = 1) -> bool:
    """
    Verifies if there are enough GPUs available for training.

    Args:
        min_gpus (int): Minimum number of GPUs required.

    Returns:
        bool: True if the required number of GPUs is available, False otherwise.
    """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def ddp_setup():
    """
    Sets up Distributed Data Parallel (DDP) for multi-GPU training.

    This function initializes the process group and sets the device for each process.
    """
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)


def get_train_objs(data_cfg: DataConfig, opt_cfg: OptimizerConfig):
    """
    Initializes training objects including datasets, model, loss function, and metrics.

    Args:
        data_cfg (DataConfig): Configuration object containing data-related settings.
        opt_cfg (OptimizerConfig): Configuration object containing optimizer-related settings.

    Returns:
        tuple: Contains the model, optimizer, loss function, metrics, training dataset, and testing dataset.
    """
    data_dir = Path(data_cfg.data_dir)
    embed_dir = Path(data_cfg.embed_dir)
    train_dataset = Mind(
        dataset_dir=data_dir / "train",
        precompute=data_cfg.precompute,
        embed_dir=embed_dir / "train",
    )
    test_dataset = Mind(
        dataset_dir=data_dir / "test",
        precompute=data_cfg.precompute,
        embed_dir=embed_dir / "test",
    )
    loss_fn = InfoNCE()
    auc_roc = RetrievalAUROC()
    ndcg_5 = RetrievalNormalizedDCG(top_k=5)
    ndcg_10 = RetrievalNormalizedDCG(top_k=10)
    model = TwoTowerRecommendation()
    optimizer = create_optimizer(model, opt_cfg)

    return (
        model,
        optimizer,
        loss_fn,
        [auc_roc, ndcg_5, ndcg_10],
        train_dataset,
        test_dataset,
    )


@hydra.main(version_base=None, config_path=".", config_name="mind_train_cfg.yaml")
def main(cfg: DictConfig):
    ddp_setup()

    # configs
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainingConfig(**cfg["trainer_config"])
    mlflow_cfg = MLFlowConfig(**cfg["mlflow"])

    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment("/Users/dev.arjunmnath@gmail.com/mind-recommendation-system")

    model, optimizer, loss_fn, metrices, train_data, test_data = get_train_objs(
        data_cfg, opt_cfg
    )

    trainer = Trainer(
        config=trainer_cfg,
        model=model,
        loss_fn=loss_fn,
        metrices=metrices,
        optimizer=optimizer,
        train_dataset=train_data,
        test_dataset=test_data,
    )
    with mlflow.start_run() as run:
        params = {
            "learning_rate": opt_cfg.learning_rate,
            "batch_size": trainer_cfg.batch_size,
            "loss_function": loss_fn.__class__.__name__,
            "metrics": [metric.__class__.__name__ for metric in metrices],
            "optimizer": "AdamW",
        }
        mlflow.log_params(params)
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        trainer.train()
        model_info = mlflow.pytorch.log_model(model, name="model")
        print(model_info)
    destroy_process_group()


if __name__ == "__main__":
    _min_gpu_count = 1
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(
            f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting."
        )
        sys.exit()
    main()
