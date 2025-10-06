from functools import partial
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from models import TwoTowerRecommendation
from models.models import InfoNCE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

tqdm.pandas()


class Mind(Dataset):
    """
    A PyTorch Dataset class for managing user behavior data and their corresponding
    embeddings or news content, with options for precomputing or using raw news data.

    Args:
        dataset_dir (Path): Directory path containing the dataset files.
        precompute (bool): Flag to indicate if precomputed embeddings should be used.
        embed_dir(Union[Path, None]): Path to precomputed embeddings file (required if precompute is True).

    Attributes:
        df (pd.DataFrame): DataFrame holding user behavior data.
        precompute (bool): Whether precomputed embeddings are used.
        embed (dict): Dictionary containing precomputed embeddings (if precompute is True).
        news (pd.DataFrame): DataFrame containing news article data (if precompute is False).
    """

    def __init__(
        self,
        dataset_dir: Path,
        precompute: bool = False,
        embed_dir: Union[Path, None] = None,
    ) -> None:
        """
        Initializes the Mind dataset by loading behavior data and either embedding data or news data.

        Args:
            dataset_dir (Path): Directory path containing the dataset files.
            precompute (bool): Flag to indicate if precomputed embeddings should be used.
            embedding_path (Union[Path, None]): Path to precomputed embeddings file (required if precompute is True).
        """
        behaviour_path = dataset_dir / "normalized_behaviours.csv"
        news_path = dataset_dir / "processed_news.csv"
        assert (
            behaviour_path.exists() and news_path.exists()
        ), "behaviour_path or news_path does not exist"
        self.df = pd.read_csv(behaviour_path.absolute().as_posix())
        self.precompute = precompute
        self.click_padding = 35
        self.history_padding = 558
        self.non_click_padding = 297
        if precompute:
            assert embed_dir is not None, "embedding directory is required"
            embed_path = embed_dir / "embedding.pth"
            assert embed_path.exists(), "embeeding file does not exist"
            self.embed = torch.load(embed_path.absolute().as_posix())
        else:
            self.news = pd.read_csv(news_path.absolute().as_posix())
            self.news.set_index("news_id", inplace=True)

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
        items = (
            self.df.iloc[idx : idx + 1] if isinstance(idx, int) else self.df.iloc[idx]
        )
        transform = self._get_embeddings if self.precompute else self._get_batch_news

        clicks_transformed = items["clicks"].apply(
            partial(transform, padding_size=self.click_padding)
        )
        non_clicks_transformed = items["non_clicks"].apply(
            partial(transform, padding_size=self.non_click_padding)
        )
        history_transformed = items["history"].apply(
            partial(transform, padding_size=self.history_padding)
        )

        # Stack the tensors to create a batch (if needed)
        clicks = torch.stack([clicks for clicks in clicks_transformed])
        non_clicks = torch.stack([non_clicks for non_clicks in non_clicks_transformed])
        history = torch.stack([history for history in history_transformed])

        return history.squeeze(), clicks.squeeze(), non_clicks.squeeze()

    def _get_batch_news(self, row) -> List:
        """
        Processes a row of the dataframe and returns the news content for clicks, non-clicks, and history.

        Args:
            row (pd.Series): A row from the dataframe containing the behavior data.

        Returns:
            List: A List containing the news content for clicks, non-clicks, and history.
        """
        return [
            self._get_news_content(row["clicks"]),
            self._get_news_content(row["non_clicks"]),
            self._get_news_content(row["history"]),
        ]

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

    def _get_embeddings(self, ids: str, padding_size: int) -> Union[None, torch.Tensor]:
        """
        Retrieves the embeddings for the given article IDs.

        Args:
            ids (str): A string of space-separated article IDs.
            padding_size (int): The padding to be applied on the sequence dimension.

        Returns:
            torch.Tensor: A tensor of embeddings for the given article IDs.
            None: If no valid embeddings are found.
        """
        padded_tensor = torch.zeros(padding_size, 768)
        if isinstance(ids, float) and pd.isna(ids):
            return padded_tensor
        id_list = ids.strip().split(" ")
        try:
            i = 0
            for article_id in id_list:
                if article_id in self.embed:
                    padded_tensor[i] = self.embed[article_id]
                    i += 1
        except KeyError:
            return padded_tensor
        return padded_tensor


if __name__ == "__main__":
    from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

    dataset = Mind(
        Path("./dataset") / "test",
        True,
        Path("./model_binaries") / "test",
    )
    loader = DataLoader(dataset, batch_size=64)
    model = TwoTowerRecommendation()
    ndcg = RetrievalNormalizedDCG()
    loss = torch.nn.CosineEmbeddingLoss()
    for iter, (history, clicks, non_clicks) in enumerate(loader):
        indexes, relevance, target = model(history, clicks, non_clicks)
        print(ndcg(relevance, target, indexes=indexes))
        print(loss(relevance, target))
        break
