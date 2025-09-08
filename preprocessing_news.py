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

news_path = input("news.tsv path: ")
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

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=768,  
        distance=models.Distance.COSINE
    )
)

df["combined_text"] = (
    df["category"].fillna("") + " " +
    df["subcategory"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["abstract"].fillna("")
)

texts = df["combined_text"].tolist()
batch_size = 128
all_embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Embeddings"):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch, show_progress_bar=False, device=device)
    all_embeddings.extend(embeddings)

points = [
    models.PointStruct(
        id=i,
        vector=all_embeddings[i],
        payload={
            "news_id": df.iloc[i]["news_id"],
        }
    )
    for i in range(len(all_embeddings))
]

qdrant.upsert(collection_name=collection_name, points=points)
print(f"✅ Stored {len(points)} embeddings in Qdrant Cloud.")
query = "Latest updates on space exploration and Mars missions"
query_emb = model.encode_query(query, device=device)

results = qdrant.search(
    collection_name=collection_name,
    query_vector=query_emb,
    limit=5
)

print("\n🔍 Top 5 Relevant News Articles:")
for r in results:
    print(f"Score: {r.score:.4f} | Title: {r.payload['title']}")
