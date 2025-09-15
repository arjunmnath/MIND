import psycopg2

from creds import *


def fetch_article(article_id):
    """Fetch all articles from the articles table."""
    conn = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, content, url, source, region, published_date FROM articles WHERE id=%s;",
        (article_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return {
        "id": row[0],
        "title": row[1],
        "content": row[2],
        "url": row[3],
        "source": row[4],
        "region": row[5],
        "published_date": row[6],
    }
