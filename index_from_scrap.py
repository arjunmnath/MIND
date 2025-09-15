import psycopg2
import numpy as np
import indexing_server

USER = "postgres.xqwyfuegdbpamsngmlbw"
PASSWORD = "curatraliasnwsfusion"
HOST = "aws-1-ap-south-1.pooler.supabase.com"
PORT = 6543
DBNAME = "postgres"


def fetch_articles():
    """Fetch all articles from the articles table."""
    conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, url, source, region, published_date FROM articles;")
    articles = cur.fetchall()
    cur.close()
    conn.close()
    return articles
def insert_embedding(article_id, embedding, model_version="v1.1"):
    """Insert article embedding into article_embeddings table."""
    conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()

    # Convert numpy array to PostgreSQL vector string
    embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"

    cur.execute("""
        INSERT INTO article_embeddings (article_id, embedding, model_version)
        VALUES (%s, %s, %s)
    """, (article_id, embedding_str, model_version))

    conn.commit()
    cur.close()
    conn.close()

def process_and_index_articles(index_article):
    """Fetch all articles, generate embeddings, and store them."""
    articles = fetch_articles()
    for article in articles:
        article_id = article[0]
        article_data = {
            "id": article[0],
            "title": article[1],
            "content": article[2],
            "url": article[3],
            "source": article[4],
            "region": article[5],
            "published_date": article[6]
        }
        # Call your existing embedding function
        embedding = indexing_server.get_embedding_from_onnx(article_data['title'], article_data['content'])
        if isinstance(embedding, np.ndarray):
            insert_embedding(article_id, embedding)
            print(f"✅ Indexed article {article_id}")
        else:
            print(f"⚠️ Skipped article {article_id}, embedding not valid")
