import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

from config_classes import DataConfig, OptimizerConfig, TrainingConfig
from dataset import Mind
from models import *
from models.models import TwoTowerRecommendation
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
    has_gpu = torch.backends.mps.is_available()
    gpu_count = torch.mps.device_count()
    return has_gpu and gpu_count >= min_gpus


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
    auc_roc = RetrievalAUROC()
    ndcg_5 = RetrievalNormalizedDCG(top_k=5)
    ndcg_10 = RetrievalNormalizedDCG(top_k=10)
    model = TwoTowerRecommendation()
    optimizer = create_optimizer(model, opt_cfg)
    return (
        model,
        optimizer,
        [auc_roc, ndcg_5, ndcg_10],
        train_dataset,
        test_dataset,
    )


@hydra.main(version_base=None, config_path=".", config_name="mind_train_cfg.yaml")
def main(cfg: DictConfig):
    # configs
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainingConfig(**cfg["trainer_config"])

    model, optimizer, metrices, train_data, test_data = get_train_objs(
        data_cfg, opt_cfg
    )
    trainer = Trainer(
        config=trainer_cfg,
        model=model,
        metrices=metrices,
        optimizer=optimizer,
        train_dataset=train_data,
        test_dataset=test_data,
        use_ddp=False,
    )
    trainer.train()


if __name__ == "__main__":
    main()
