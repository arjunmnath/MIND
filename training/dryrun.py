from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import Mind
from models import *
from utils import plot_attention_scores

model = TwoTowerRecommendation()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
h = torch.randn(2, 558, 768)
c = torch.randn(2, 35, 768)
nc = torch.randn(2, 297, 768)
h[:, 5:, :] = 0
c[:, 1:, :] = 0
nc[:, 2:, :] = 0
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
# loader = DataLoader(train_dataset, batch_size=64)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for history, clicks, non_clicks in loader:
        (
            loss,
            user_repr,
            impressions,
            labels,
            samples_per_batch,
            attn_scores,
            seq_len,
        ) = model(history, clicks, non_clicks)
        # print((user_repr * impressions).sum(dim=-1), labels, seq_len)
        print(loss)
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.abs().max())  # Check the max gradient
        optimizer.step()
        if epoch == num_epochs - 1:
            plot_attention_scores(
                attn_scores,
                title="Multi-Head Attention Scores",
                non_padded_size=seq_len,
            )
