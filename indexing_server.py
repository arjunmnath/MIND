import os
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

app = Flask(__name__)
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
        collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)  
    )

def preprocess_text(title, text):
    txt = prefixes['document'].replace('::title::', title).replace('::text::', text)
    inputs = tokenizer([text], padding=True, return_tensors="np")
    return inputs.data

def get_embedding_from_onnx(title, text):
    inputs = preprocess_text(title, text)
    output_name = "sentence_embedding"
    result = session.run([output_name], inputs)
    return result[0]

@app.route('/index', methods=['POST'])
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
        return jsonify({"message": "Article indexed successfully", "id": article_id, "vec": embedding.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_articles():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        query_embedding = get_embedding_from_onnx(query)
        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=5  
        )
        results = [{"id": result.id, "similarity": result.score} for result in search_results]

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5900)
