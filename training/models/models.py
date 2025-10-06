from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torchinfo import summary

from .user_encoder import UserEncoder


class NewsEncoder(nn.Module):
    """
    A model that encodes and projects text into a normalized vector space.

    Uses a pre-trained SentenceTransformer for encoding, followed by a feed-forward
    network to project the embeddings to a lower-dimensional space. Outputs are
    normalized vectors.

    Args:
        embed_dim (int): Dimension of the input embeddings (default 768).
        hidden_dim (int): Hidden dimension for the projection layer (default 128).
        dropout (float): Dropout rate (default 0.1).
        is_vector_input (bool): Whether the input is pre-encoded vectors (default True).

    Methods:
        forward(contents): Encodes and projects the input, returning normalized vectors.
    """

    def __init__(self, embed_dim=768, dropout=0.1, is_vector_input=True):
        super(NewsEncoder, self).__init__()

        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.is_vector_input = is_vector_input

    def forward(self, contents):
        """
        Encodes and projects the input into a normalized vector.
        Args:
            contents (Union[List[str], torch.Tensor]): Input text or pre-encoded vectors.

        Returns:
            torch.Tensor: Normalized projected vectors.
        """
        batch_embeddings = (
            self.model.encode(
                contents, device=next(self.parameters()).device, convert_to_tensor=True
            )
            if not self.is_vector_input
            else contents
        )  # shape: [batch_size, 768]
        batch_embeddings = batch_embeddings.clone().detach()
        projections = self.project(batch_embeddings)  # shape: [batch_size, 768]
        return F.normalize(projections, p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = UserEncoder(768)
        self.news_tower = NewsEncoder()

    def forward(self, history, clicks, non_clicks):
        """
        history: [batch_size, num_history, 768]
        clicks: [batch_size, num_clicks, 768]
        non_clicks: [batch_size, num_non_clicks, 768]
        """
        history = self.news_tower(history)
        clicks = self.news_tower(clicks)
        non_clicks = self.news_tower(non_clicks)

        user_repr, _ = self.user_tower(history)  # dim: [batch_size, 768]
        user_repr = F.normalize(
            user_repr.unsqueeze(1), p=2, dim=-1
        )  # dim: [batch_size, 1, 768]
        relevance_clicks = torch.bmm(user_repr, clicks.transpose(1, 2)).squeeze(
            1
        )  # dims: [batch_size, num_clicks]
        relevance_non_clicks = torch.bmm(user_repr, non_clicks.transpose(1, 2)).squeeze(
            1
        )  # dims: [batch_size, num_non_clicks]

        target_clicks = torch.ones_like(relevance_clicks)  # [batch_size, num_clicks]
        target_non_clicks = torch.zeros_like(
            relevance_non_clicks
        )  # [batch_size, num_non_clicks]

        print(relevance_non_clicks, relevance_clicks, sep="\n")
        relevance = torch.cat(
            [relevance_clicks, relevance_non_clicks], dim=1
        )  # dims: [batch_size, num_clicks + num_non_clicks]
        target = torch.cat(
            [target_clicks, target_non_clicks], dim=1
        )  # dims: [batch_size, num_clicks + num_non_clicks]

        indexes = torch.cat(
            [torch.arange(relevance.size(0)).unsqueeze(1)] * relevance.size(1), dim=1
        )  # dims: [batch_size, num_clicks + num_non_clicks]

        return indexes.flatten(), relevance.flatten(), target.flatten()


class InfoNCE(nn.Module):
    """Implementation of the InfoNCE (Information Noise Contrastive Estimation) loss function.

    Paper: https://arxiv.org/pdf/1807.03748#page=3

    This loss function is commonly used in contrastive learning, where the goal is to
    distinguish between similar and dissimilar pairs of data points. The loss encourages
    the model to assign a higher similarity score to positive pairs and a lower similarity
    score to negative pairs.

    Args:
        temperature (float, optional): A scaling factor that adjusts the distribution
            of similarities. Default is 0.07. Lower values make the model more sensitive
            to small differences in similarity, while higher values make the model more
            lenient.
    """

    def __init__(self, temperature=0.07):
        """
        Initializes the InfoNCE loss module with the given temperature.

        Args:
            temperature (float, optional): A scaling factor for similarity logits. Default is 0.07.
        """
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, similarities: torch.Tensor, target: torch.Tensor):
        """
        Compute the InfoNCE loss.

        The loss is computed by scaling the similarity scores (logits) by the temperature
        factor and applying binary cross-entropy loss between the logits and the target labels.

        Args:
            similarities (torch.Tensor): A tensor containing the similarity scores (logits).
            target (torch.Tensor): A tensor containing the binary target labels (0 or 1).

        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        logits = similarities / self.temperature
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss


if __name__ == "__main__":
    from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

    model = TwoTowerRecommendation()
    batch_size = 2
    embed_dims = 768
    history = torch.randn(batch_size, 6, embed_dims)
    clicks = torch.randn(batch_size, 1, embed_dims)
    non_clicks = torch.randn(batch_size, 3, embed_dims)
    indexes, preds, target = model(history, clicks, non_clicks)
    loss_fn = InfoNCE()
    print(loss_fn(preds, target))
