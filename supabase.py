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



