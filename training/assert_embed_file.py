import logging
import torch
from pathlib import Path

from logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

root = Path('./model_binaries')

train_path = root / 'train' / 'embedding.pth'
test_path = root / 'test' / 'embedding.pth'
assert(train_path.exists() and test_path.exists())
train_embed = torch.load(train_path)
test_embed = torch.load(test_path)

logger.info("Checking for nan values in training embeddings...")
for key, tensor in train_embed.items():
    assert(not torch.isnan(tensor).any()), f"\narticle {key} contains nan values..."
logger.info("Passed")

logger.info("Checking for nan values in testing embeddings...")
for key, tensor in test_embed.items():
    assert(not torch.isnan(tensor).any()), f"\narticle {key} contains nan values..."
logger.info("Passed")
