import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from config_classes import Snapshot
from dataset import Mind
from models import *
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG
from tqdm import tqdm
from utils import plot_attention_scores, upload_to_s3

logger = logging.getLogger(__name__)

model = TwoTowerRecommendation()
# model = model.to('mps')
batch_size = 2
optimizer = optim.AdamW(model.parameters(), lr=0.01)

h = torch.randn(batch_size, 558, 768)
c = torch.randn(batch_size, 35, 768)
nc = torch.randn(batch_size, 297, 768)
h[0, 5:, :] = 0
c[0, 1:, :] = 0
nc[0, 1:, :] = 0
h[1, 4:, :] = 0
c[1, 1:, :] = 0
nc[1, 2:, :] = 0
dataset = TensorDataset(h, c, nc)
loader = DataLoader(dataset, batch_size=64)


# data_dir = Path("./dataset")
# embed_dir = Path("./model_binaries")
# train_dataset = Mind(
#     dataset_dir=data_dir / "train",
#     precompute=True,
#     embed_dir=embed_dir / "train",
# )
# test_dataset = Mind(
#     dataset_dir=data_dir / "test",
#     precompute=True,
#     embed_dir=embed_dir / "test",
# )
# loader = DataLoader(train_dataset, batch_size=batch_size)
#
crit = [
    RetrievalAUROC(),
    RetrievalNormalizedDCG(top_k=5),
    RetrievalNormalizedDCG(top_k=10),
]


# Setup logging
from logging_config import setup_logging

setup_logging()

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for iter, (history, clicks, non_clicks) in enumerate(loader):
        (
            loss,
            preds,
            target,
            indexes,
            attn_scores,
            seq_len,
        ) = model(history, clicks, non_clicks, log=True)
        metrices = [metric(preds, target, indexes=indexes) for metric in crit]
        # print(
        #     f"[Step {epoch}:{iter} | train Loss {loss:.5f} |"
        #     f" auc: {metrices[0]:.5f} | ndcg@5: {metrices[1]:.4f} | ndcg@10: {metrices[2]:.4f}"
        # )
        logger.debug(f"Predictions: {preds}, Targets: {target}")
        loss.backward()
        optimizer.step()
        if epoch == num_epochs - 1:
            plot_attention_scores(
                attn_scores,
                title="Multi-Head Attention Scores",
                non_padded_size=seq_len,
            )
