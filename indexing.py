import json

import numpy as np
import psycopg2

import server
from creds import *
from server import get_embedding_from_onnx
from utlis import fetch_article


def does_similar_article_exist(query_embedding, k=10):
    """
    Checks with there exist any similar articles on the database
    """
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    embedding_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"
    sql = """
        SELECT a.id, 
               1 - (ae.embedding <=> %s::vector) AS cosine_similarity
        FROM article_embeddings ae
        JOIN articles a ON ae.article_id = a.id
        WHERE 1 - (ae.embedding <=> %s::vector) >= 0.95
        ORDER BY cosine_similarity DESC
        LIMIT %s;
    """
    cur.execute(sql, (embedding_str, embedding_str, k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return len(results) > 0


def insert_embedding(article_id, embedding, model_version="v1.1"):
    """Insert article embedding into article_embeddings table."""
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"
    cur.execute(
        """
        INSERT INTO article_embeddings (article_id, embedding, model_version)
        VALUES (%s, %s, %s)
    """,
        (article_id, embedding_str, model_version),
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_article(article_json):
    embed = get_embedding_from_onnx(
        article_json["title"], article_json["content"] + article_json["published_date"]
    )[0]
    if does_similar_article_exist(embed):
        print(f"🚨 Skipping similar article, title {article_json['title']}")
        return
    sql = """
        INSERT INTO articles (title, content, url, published_date, source, region, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        RETURNING id;
    """
    try:
        conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
        )
        cur = conn.cursor()
        cur.execute(
            sql,
            (
                article_json.get("title"),
                article_json.get("content"),
                article_json.get("url"),
                article_json.get("published_date"),
                article_json.get("source"),
                article_json.get("region"),
            ),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        print(f"✅ Article inserted with id {new_id}", end="\r")
        cur.close()
        conn.close()
        insert_embedding(new_id, embed)
        return new_id
    except Exception as e:
        print("❌ Error inserting article:", e)
        return None


if __name__ == "__main__":
    with open("scraper/articles.json") as f:
        articles = json.load(f)
        for article in articles:
            insert_article(article)
