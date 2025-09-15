import os
import json
import psycopg2

USER = "postgres.xqwyfuegdbpamsngmlbw"
PASSWORD = "curatraliasnewsfusion"
HOST = "aws-1-ap-south-1.pooler.supabase.com"
PORT = 6543
DBNAME = "postgres"


schema_sql = """
-- Drop tables in correct order (respecting foreign keys)
DROP TABLE IF EXISTS user_interactions CASCADE;
DROP TABLE IF EXISTS article_embeddings CASCADE;
DROP TABLE IF EXISTS articles CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;

-- user table
CREATE TABLE "user" (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email VARCHAR,
    role VARCHAR,
    preferences JSONB,
    created_at TIMESTAMPTZ
);

-- articles table (updated)
CREATE TABLE articles (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ,
    title TEXT,
    content TEXT,
    url VARCHAR,
    updated_at TIMESTAMPTZ,
    source TEXT,
    region VARCHAR,
    published_date TIMESTAMPTZ
);

-- user_interactions table
CREATE TABLE user_interactions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ,
    article_id BIGINT,
    type VARCHAR,
    timestamp TIMESTAMPTZ,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

-- article_embeddings table
CREATE TABLE article_embeddings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    article_id BIGINT,
    embedding VECTOR,
    model_version VARCHAR,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

-- categories table
CREATE TABLE categories (
    cat_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR,
    description TEXT
);
"""


def reset_schema():
    try:
        conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
        )
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
        print("✅ Schema dropped and recreated successfully!")
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Error:", e)


def insert_article(article_json):
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
                article_json.get(
                    "published_date"
                ),  # should be a valid timestamp string (e.g. "2025-09-15T10:00:00Z")
                article_json.get("source"),
                article_json.get("region"),
            ),
        )

        new_id = cur.fetchone()[0]
        conn.commit()

        print(f"✅ Article inserted with id {new_id}")

        cur.close()
        conn.close()

        return new_id
    except Exception as e:
        print("❌ Error inserting article:", e)
        return None


if __name__ == "__main__":
    with open("scraper/articles.json") as f:
        articles = json.load(f)
        for article in articles:
            insert_article(article)
