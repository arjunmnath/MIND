import logging
import sys

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

assert len(sys.argv) == 2
news_csv = sys.argv[1]
batch_size = 512

df = pd.read_csv(news_csv)
contents = df["content"].tolist()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
logger.info(f"Device: {device}")

model = SentenceTransformer("google/embeddinggemma-300m", device=device)

content_embeddings = []
batch_count = len(contents) // batch_size + (
    1 if len(contents) % batch_size != 0 else 0
)
progress_bar = tqdm(range(batch_count), desc="Processing Batches")

for i in progress_bar:
    batch = contents[i * batch_size : (i + 1) * batch_size]
    batch_embeddings = model.encode(batch, device=device, convert_to_tensor=True)
    if torch.isnan(batch_embeddings).any():
        logger.warning("produced batch embedding has nan values")
    else:
        content_embeddings.append(batch_embeddings)

content_embeddings = torch.cat(content_embeddings, dim=0)
tensor_data = dict(zip(df["news_id"].tolist(), content_embeddings.cpu()))
torch.save(tensor_data, "news-gemma-embedding-small-train.pth")
