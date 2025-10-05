import sys

import pandas as pd
from tqdm import tqdm

assert len(sys.argv) == 2
news_csv = sys.argv[1]

headers = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
df = pd.read_csv(news_csv, sep="\t", names=headers, quoting=3)
df["title"] = df["title"].fillna(" ")
df["abstract"] = df["abstract"].fillna(" ")
df["content"] = "Title: " + df["title"] + " Abstract: " + df["abstract"]
df = df.drop(
    columns=[
        "category",
        "title",
        "abstract",
        "subcategory",
        "url",
        "title_entities",
        "abstract_entities",
    ]
)
df.to_csv("../dataset/processed_news.csv", index=False)
