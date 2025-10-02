from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

tqdm.pandas()


class Mind(Dataset):
    """
    A PyTorch Dataset class for managing user behavior data and their corresponding
    embeddings or news content, with options for precomputing or using raw news data.

    Args:
        behaviour_path (str): Path to the CSV file containing user behavior data.
        precompute (bool): Whether to use precomputed embeddings (default is False).
        embedding_path (Union[str, None]): Path to the precomputed embeddings (if precompute is True).
        news_path (Union[str, None]): Path to the CSV file containing news articles (if precompute is False).

    Attributes:
        df (pd.DataFrame): DataFrame holding user behavior data.
        precompute (bool): Whether precomputed embeddings are used.
        embed (dict): Dictionary containing precomputed embeddings (if precompute is True).
        news (pd.DataFrame): DataFrame containing news article data (if precompute is False).
    """

    def __init__(
        self,
        behaviour_path: str,
        precompute: bool = False,
        embedding_path: Union[str, None] = None,
        news_path: Union[str, None] = None,
    ) -> None:
        """
        Initializes the Mind dataset by loading behavior data and either embedding data or news data.

        Args:
            behaviour_path (str): Path to the user behavior CSV.
            precompute (bool): Flag to indicate if precomputed embeddings should be used.
            embedding_path (Union[str, None]): Path to precomputed embeddings file (required if precompute is True).
            news_path (Union[str, None]): Path to news CSV (required if precompute is False).
        """
        self.df = pd.read_csv(behaviour_path)
        self.precompute = precompute
        if precompute:
            assert embedding_path is not None and embedding_path.endswith(
                ".pth"
            ), "Expected path to precompute news embeddings"
            self.embed = torch.load(embedding_path)
            print("Merging with precomputed embeddings: ")
            (
                self.df["clicks_ready"],
                self.df["non_clicks_ready"],
                self.df["history_ready"],
            ) = zip(
                *self.df[["clicks", "non_clicks", "history"]].progress_apply(
                    self._get_batch_embeddings, axis=1
                )
            )
        else:
            assert news_path is not None, "Expected path to news csv..."
            self.news = pd.read_csv(news_path)
            self.news.set_index("news_id", inplace=True)
            print("Merging with news.csv: ")
            (
                self.df["clicks_ready"],
                self.df["non_clicks_ready"],
                self.df["history_ready"],
            ) = zip(
                *self.df[["clicks", "non_clicks", "history"]].progress_apply(
                    self._get_batch_news, axis=1
                )
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: Union[int, slice]):
        """
        Retrieves a sample (or a batch of samples) from the dataset given an index (or a slice).

        Args:
            idx (Union[int, slice]): The index or slice of the samples to retrieve.

        Returns:
            Union[tuple, List[tuple]]: A tuple containing the preprocessed history, clicks, and non-clicks data
                                       if an integer index is given, or a list of tuples for a slice.
        """
        items = self.df.iloc[idx]
        return (
            items["history_ready"],
            items["clicks_ready"],
            items["non_clicks_ready"],
        )

    def _get_batch_embeddings(self, row) -> tuple:
        """
        Processes a row of the dataframe and returns the embeddings for clicks, non-clicks, and history.

        Args:
            row (pd.Series): A row from the dataframe containing the behavior data.

        Returns:
            tuple: A tuple containing the embeddings for clicks, non-clicks, and history.
        """
        return (
            self._get_embeddings(row["clicks"]),
            self._get_embeddings(row["non_clicks"]),
            self._get_embeddings(row["history"]),
        )

    def _get_batch_news(self, row) -> tuple:
        """
        Processes a row of the dataframe and returns the news content for clicks, non-clicks, and history.

        Args:
            row (pd.Series): A row from the dataframe containing the behavior data.

        Returns:
            tuple: A tuple containing the news content for clicks, non-clicks, and history.
        """
        return (
            self._get_news_content(row["clicks"]),
            self._get_news_content(row["non_clicks"]),
            self._get_news_content(row["history"]),
        )

    def _get_news_content(self, ids: str) -> List[str]:
        """
        Retrieves the content of the news articles corresponding to the given IDs.

        Args:
            ids (str): A string of space-separated article IDs.

        Returns:
            List[str]: A list of content for the corresponding articles.
        """
        if isinstance(ids, float) and pd.isna(ids):
            return []
        id_list = ids.strip().split(" ")
        news = [self.news.loc[article_id]["content"] for article_id in id_list]
        return news

    def _get_embeddings(self, ids: str) -> Union[None, torch.Tensor]:
        """
        Retrieves the embeddings for the given article IDs.

        Args:
            ids (str): A string of space-separated article IDs.

        Returns:
            torch.Tensor: A tensor of embeddings for the given article IDs.
            None: If no valid embeddings are found.
        """
        if isinstance(ids, float) and pd.isna(ids):
            return torch.tensor([])
        id_list = ids.strip().split(" ")
        try:
            embeds = [
                self.embed[article_id]
                for article_id in id_list
                if article_id in self.embed
            ]
            if embeds:
                return torch.stack(embeds, dim=0)
        except KeyError:
            return torch.tensor([])
        return torch.tensor([])

