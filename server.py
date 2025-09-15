import os

import numpy as np
import onnxruntime as ort
import psycopg2
from flask import Flask, jsonify, request
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from transformers import AutoTokenizer

from creds import *

app = Flask(__name__)
CORS(app)
MAX_LEN = 2048
collection_name = "news_embeddings"
model_id = "onnx-community/embeddinggemma-300m-ONNX"
model_path = hf_hub_download(model_id, subfolder="onnx", filename="model.onnx")
hf_hub_download(model_id, subfolder="onnx", filename="model.onnx_data")
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

qdrant = QdrantClient(
    url="https://152b7e4a-5706-4c3f-b740-0145473bde0e.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.a6TN0Np76IAy4Dl7NGmLZY_kY9uQieBVJsJP7V8J7i0",
)

prefixes = {
    "query": "task: search result | query: ",
    "document": "title: ::title:: | text: ::text::",
}
try:
    qdrant.get_collection(collection_name)
except:
    qdrant.create_collection(
        collection_name, vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )


def fetch_articles():
    """Fetch all articles from the articles table."""
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, content, url, source, region, published_date FROM articles;"
    )
    articles = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": row[0],
            "title": row[1],
            "url": row[3],
            "source": row[4],
            "region": row[5],
            "published_date": row[6],
        }
        for row in articles
    ]


def get_article_embedding(article_id):
    """
    Fetch the embedding for a given article_id from article_embeddings
    and return it as a NumPy array.
    """
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    sql = "SELECT embedding FROM article_embeddings WHERE article_id = %s;"
    cur.execute(sql, (article_id,))
    result = cur.fetchone()

    cur.close()
    conn.close()

    if result is None:
        print(f"⚠️ No embedding found for article_id {article_id}")
        return None

    embedding_str = result[0]
    embedding = np.array([float(x) for x in embedding_str.strip("[]").split(",")])
    return embedding


def fetch_similar_articles(query_embedding, k=10):
    """
    Fetch top-k most similar articles given an embedding vector.
    Uses cosine distance with pgvector.
    """
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    embedding_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"
    sql = """
        SELECT a.id, a.title, a.content, a.url, a.source, a.region, a.published_date,
               1 - (ae.embedding <=> %s::vector) AS cosine_similarity
        FROM article_embeddings ae
        JOIN articles a ON ae.article_id = a.id
        ORDER BY 1 - (ae.embedding <=> %s::vector) DESC
        LIMIT %s;
    """
    cur.execute(sql, (embedding_str, embedding_str, k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": row[0],
            "title": row[1],
            "url": row[3],
            "source": row[4],
            "region": row[5],
            "published_date": row[6],
            "distance": row[7],
        }
        for row in results
    ]


def chunk_text(text, max_len=MAX_LEN):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(tokens), max_len):
        yield tokenizer.decode(tokens[i : i + max_len])


def preprocess_text(title, text):
    txt = prefixes["document"].replace("::title::", title).replace("::text::", text)
    text = txt[:MAX_LEN]
    inputs = tokenizer([text], padding=True, return_tensors="np")
    return inputs.data


def get_embedding_from_onnx(title, text):
    inputs = preprocess_text(title, text)
    output_name = "sentence_embedding"
    result = session.run([output_name], inputs)
    return result[0]


@app.route("/index", methods=["POST"])
def index_article():
    data = request.get_json()
    title = data.get("title")
    text = data.get("text")
    if not title or not text:
        return jsonify({"error": "Title and text are required"}), 400
    try:
        embedding = get_embedding_from_onnx(title, text)
        article_id = f"article_{hash(title)}"
        """qdrant.upsert(
            collection_name=collection_name,
            points=[(article_id, embedding.tolist())]  
        )
        """
        return (
            jsonify(
                {
                    "message": "Article indexed successfully",
                    "id": article_id,
                    "vec": embedding.tolist(),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def search_articles():
    data = request.get_json()
    impressions = data.get("impressions")
    if not impressions:
        return jsonify({"error": "impressions is required"}), 400

    try:
        embedding = [get_article_embedding(id) for id in impressions]
        stacked_embed = np.vstack(embedding)
        query_vector = np.mean(stacked_embed, axis=0)
        results = fetch_similar_articles(query_vector, 10)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/news", methods=["GET"])
def get_news_article():
    return jsonify(fetch_articles()), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5900)
