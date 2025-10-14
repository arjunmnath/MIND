import torch
from pathlib import Path

root = Path('./model_binaries')

train_path = root / 'train' / 'embedding.pth'
test_path = root / 'test' / 'embedding.pth'
assert(train_path.exists() and test_path.exists())
train_embed = torch.load(train_path)
test_embed = torch.load(test_path)

print("checking for nan values in training embeddings: ", end='')
for key, tensor in train_embed.items():
    assert(not torch.isnan(tensor).any()), f"\narticle {key} contains nan values..."
print("Passed")
print("checking for nan values in testing embeddings: ", end='')
for key, tensor in test_embed.items():
    assert(not torch.isnan(tensor).any()), f"\narticle {key} contains nan values..."
print("Passed")
