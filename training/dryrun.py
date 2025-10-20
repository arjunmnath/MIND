from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Mind
from models import *
from utils import plot_attention_scores

writer = SummaryWriter(log_dir="runs/experiment_name")

model = TwoTowerRecommendation()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# h = torch.randn(1, 558, 768)
# c = torch.randn(1, 35, 768)
# nc = torch.randn(1, 297, 768)
# h[:, 5:, :] = 0
# c[:, 1:, :] = 0
# nc[:, 2:, :] = 0
[h, c, nc] = [val.unsqueeze(0) for _, val in torch.load("mock.pth", weights_only=False)]
print(h.shape, c.shape, nc.shape)
dataset = TensorDataset(h, c, nc)
loader = DataLoader(dataset)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for history, clicks, non_clicks in loader:
        loss, user_repr, impressions, labels, attn_scores, seq_len = model(
            history, clicks, non_clicks
        )
        # print(user_repr @ impressions.mT, labels, seq_len)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(name, param.grad.abs().max())  # Check the max gradient
        optimizer.step()
        if epoch == num_epochs - 1:
            plot_attention_scores(
                attn_scores,
                title="Multi-Head Attention Scores",
                non_padded_size=seq_len,
            )
