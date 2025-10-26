import json
import os

import psycopg2

USER = "postgres.xqwyfuegdbpamsngmlbw"
PASSWORD = "curatraliasnewsfusion"
HOST = "aws-1-ap-south-1.pooler.supabase.com"
PORT = 6543
DBNAME = "postgres"

schema_sql = """

-- Drop all tables in the current schema (public)
DO $$ 
DECLARE
    r RECORD;
BEGIN
    -- Loop through all tables and drop them
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || r.tablename || ' CASCADE';
    END LOOP;
END $$;
-- Table to store article embeddings
CREATE TABLE public.article_embeddings (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  article_id bigint,
  embedding float8[],  -- Storing embeddings as arrays of floats
  model_version character varying,
  projected_embedding float8[],  -- Storing projected embeddings
  CONSTRAINT article_embeddings_pkey PRIMARY KEY (id),
  CONSTRAINT article_embeddings_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id)
);

-- Table to store articles
CREATE TABLE public.articles (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  created_at timestamp with time zone,
  title text,
  content text,
  url character varying,
  updated_at timestamp with time zone,
  source text,
  region character varying,
  published_date timestamp with time zone,
  CONSTRAINT articles_pkey PRIMARY KEY (id)
);

-- Table to store categories
CREATE TABLE public.categories (
  cat_id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  name character varying,
  description text,
  CONSTRAINT categories_pkey PRIMARY KEY (cat_id)
);

-- Table to store users
CREATE TABLE public.user (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  username character varying UNIQUE NOT NULL,  -- Added username for user identification
  email character varying UNIQUE,  -- Keeping email optional, assuming username is primary for login
  password character varying NOT NULL,  -- Added password for authentication
  role character varying,  -- Assuming roles like 'admin', 'user', etc.
  preferences jsonb,  -- User-specific settings stored as JSON
  created_at timestamp with time zone,
  CONSTRAINT user_pkey PRIMARY KEY (id)
);

-- Table to store user interactions with articles (e.g., clicks, shares)
CREATE TABLE public.user_interactions (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  created_at timestamp with time zone,
  user_id bigint,  -- Added reference to user
  article_id bigint,
  type character varying,  -- Type of interaction (e.g., 'click', 'share')
  timestamp timestamp with time zone,
  CONSTRAINT user_interactions_pkey PRIMARY KEY (id),
  CONSTRAINT user_interactions_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id),
  CONSTRAINT user_interactions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.user(id)
);

-- Table to manage the many-to-many relationship between articles and categories
CREATE TABLE public.article_categories (
  article_id bigint,
  cat_id bigint,
  CONSTRAINT article_categories_pkey PRIMARY KEY (article_id, cat_id),
  CONSTRAINT article_categories_article_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id),
  CONSTRAINT article_categories_category_fkey FOREIGN KEY (cat_id) REFERENCES public.categories(cat_id)
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
if __name__ == "__main__":
    reset_schema()
