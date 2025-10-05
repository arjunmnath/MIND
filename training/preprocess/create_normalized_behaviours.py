import sys

import pandas as pd
from tqdm import tqdm

assert len(sys.argv) == 2
behaviour_csv = sys.argv[1]

headers = ["id", "user_id", "timestamp", "history", "impressions"]
df = pd.read_csv(behaviour_csv, sep="\t", names=headers, quoting=3)


def get_clicks(impressions):
    impressions = impressions.split(" ")
    impressions = [impression.split("-") for impression in impressions]
    return " ".join(
        [article_id for article_id, interaction in impressions if interaction == "1"]
    )


def get_non_clicks(impressions):
    impressions = impressions.split(" ")
    impressions = [impression.split("-") for impression in impressions]
    return " ".join(
        [article_id for article_id, interaction in impressions if interaction == "0"]
    )


df["clicks"] = df["impressions"].apply(get_clicks)
df["non_clicks"] = df["impressions"].apply(get_non_clicks)
df = df.drop(columns=["id", "timestamp", "impressions"])
df.to_csv("../dataset/normalized_behaviours.csv", index=False)
