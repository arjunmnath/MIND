from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Mind(Dataset):
    def __init__(self, behaviour_path, precompute=False, embedding_path=None):
        if precompute:
            assert (
                embedding_path is not None
            ), "expected patht to precompute news embeddings"
            self.embed = torch.load(embedding_path)
        self.df = pd.read_csv(behaviour_path)
        print(self.df.dropna())
        # self.df["clicks_embed"] = self.df["clicks"].apply(
        #     partial(self.get_click_embedings)
        # )
        # self.df["non_clicks_embed"] = self.df["non_clicks"].apply(
        #     partial(self.get_click_embedings)
        # )
        # self.df["history_embed"] = self.df["history"].apply(
        #     partial(self.get_click_embedings)
        # )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def get_click_embedings(self, ids: str) -> Union[None, torch.Tensor]:
        id_list = ids.strip().split(" ")
        embeds = []
        try:
            for article_id in id_list:
                embeds.append(self.embed[article_id])
        except KeyError:
            return None
        return torch.stack(embeds, dim=0)


if __name__ == "__main__":
    train_behavior_csv = "../dataset/normalized_behaviours.csv"
    train_news_csv = "../dataset/processed_news.csv"
    mind = Mind(
        train_behavior_csv,
        precompute=True,
        embedding_path="../model_binaries/news-gemma-embedding-small-train.pth",
    )
    i = 0
    for a in mind:
        i += 1
        if i == 1:
            break
