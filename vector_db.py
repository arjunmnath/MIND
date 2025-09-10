from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://152b7e4a-5706-4c3f-b740-0145473bde0e.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.a6TN0Np76IAy4Dl7NGmLZY_kY9uQieBVJsJP7V8J7i0",
)
print(qdrant_client.get_collections())
