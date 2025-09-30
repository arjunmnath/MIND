from flask import Flask, jsonify, request

# import onnxruntime as ort

app = Flask(__name__)

@app.route("/news", methods=["GET"])
def get_news_article():
    return jsonify(fetch_articles()), 200

