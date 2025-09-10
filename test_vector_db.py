import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import torch

headers = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities"
]

news_path = 'dataset/large_dev/news.tsv' 
df = pd.read_csv(news_path.strip(), sep="\t", names=headers, quoting=3)
print(f"Loaded {len(df)} news articles")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("google/embeddinggemma-300m", device=device)

qdrant = QdrantClient(
    url="https://152b7e4a-5706-4c3f-b740-0145473bde0e.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.a6TN0Np76IAy4Dl7NGmLZY_kY9uQieBVJsJP7V8J7i0",
)

collection_name = "news_embeddings"

query = input("search query: ") 
query_emb = model.encode(query, device=device)
query_bias_emb = model.encode("liked by teenagers", device=device)

query_vec = query_emb + query_bias_emb
conditioned = qdrant.search(
    collection_name=collection_name,
    query_vector=query_emb,
    limit=5
)
non_conditioned = qdrant.search(
    collection_name=collection_name,
    query_vector=query_vec,
    limit=5
)

print("\n🔍 Top 5 Relevant News Articles(conditioned):")
for r in conditioned:
    news_idx = df.index[df["news_id"] == r.payload["news_id"]][0]
    print(f"Score: {r.score:.4f} | Title: {df.iloc[news_idx]['title']}")

print("\n🔍 Top 5 Relevant News Articles(unconditioned):")
for r in non_conditioned:
    news_idx = df.index[df["news_id"] == r.payload["news_id"]][0]
    print(f"Score: {r.score:.4f} | Title: {df.iloc[news_idx]['title']}")
