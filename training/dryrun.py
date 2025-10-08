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
    # Use MPS device for training on Apple GPU
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # configs
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainingConfig(**cfg["trainer_config"])
    mlflow_cfg = MLFlowConfig(**cfg["mlflow"])

    if torch.backends.mps.is_available():
        print("Using MPS for training")
    else:
        print("MPS not available, falling back to CPU")

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

    trainer.train()


if __name__ == "__main__":
    import os

    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    if not verify_min_gpu_count(min_gpus=1):
        print(f"Unable to locate sufficient 1 GPUs to run this example. Exiting.")
        sys.exit()
    main()
