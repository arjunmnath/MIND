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
        if not is_vector_input:
            self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
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
        if torch.isnan(batch_embeddings).any():
            print(f"NaN detected before projection in NewsEncoder")
        projections = self.project(batch_embeddings)  # shape: [batch_size, 768]
        if torch.isnan(batch_embeddings).any():
            print(f"NaN detected after projection in NewsEncoder")
        return F.normalize(projections, p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = UserEncoder(768)
        self.news_tower = NewsEncoder()

    def forward(self, history, clicks, non_clicks):
        """
        history: [batch_size, history_pad_size, 768]
        clicks: [batch_size, clicks_pad_size, 768]
        non_clicks: [batch_size, non_clicks_pad_size, 768]
        """
        history = self.news_tower(history)
        clicks = self.news_tower(clicks)
        non_clicks = self.news_tower(non_clicks)
        user_repr, _ = self.user_tower(history)  # dim: [batch_size, 768]
        user_repr = F.normalize(
            user_repr.unsqueeze(1), p=2, dim=-1
        )  # dim: [batch_size, 1, 768]

        relevance_clicks_padded = torch.bmm(user_repr, clicks.transpose(1, 2)).squeeze(
            1
        )  # dims: [batch_size, click_pad_size]
        relevance_non_clicks_padded = torch.bmm(
            user_repr, non_clicks.transpose(1, 2)
        ).squeeze(
            1
        )  # dims: [batch_size, non_click_pad_size]
        batch_size, clicks_pad_size = relevance_clicks_padded.shape
        non_click_pad_size = relevance_non_clicks_padded.shape[1]

        # Create proper padding masks - check if the input embeddings are all zeros (padded)
        clicks_padding_mask = clicks.sum(dim=-1) == 0  # [batch_size, click_pad_size]
        non_clicks_padding_mask = (
            non_clicks.sum(dim=-1) == 0
        )  # [batch_size, non_click_pad_size]

        # Remove padded positions
        relevance_clicks = relevance_clicks_padded[
            ~clicks_padding_mask
        ]  # [click_count]
        relevance_non_clicks = relevance_non_clicks_padded[
            ~non_clicks_padding_mask
        ]  # [non_click_count]

        target_clicks = torch.ones_like(relevance_clicks)  # [click_count]
        target_non_clicks = torch.zeros_like(relevance_non_clicks)  # [num_non_clicks]
        relevance = torch.cat(
            [relevance_clicks, relevance_non_clicks], dim=0
        )  # dims: [num_clicks + num_non_clicks]
        target = torch.cat(
            [target_clicks, target_non_clicks]
        )  # dims: [num_clicks + num_non_clicks]

        # Create proper indexes for metrics - each sample gets a unique index
        num_clicks = relevance_clicks.shape[0]
        num_non_clicks = relevance_non_clicks.shape[0]

        # Create indexes: clicks get batch indices, non_clicks get batch indices
        click_batch_indices = []
        non_click_batch_indices = []

        for batch_idx in range(batch_size):
            # Count actual clicks and non-clicks for this batch
            batch_clicks = (~clicks_padding_mask[batch_idx]).sum().item()
            batch_non_clicks = (~non_clicks_padding_mask[batch_idx]).sum().item()

            click_batch_indices.extend([batch_idx] * batch_clicks)
            non_click_batch_indices.extend([batch_idx] * batch_non_clicks)

        indexes = torch.tensor(
            click_batch_indices + non_click_batch_indices, device=history.device
        )
        return indexes, relevance, target


class InfoNCE(nn.Module):
    """Partial Implementation of the InfoNCE (Information Noise Contrastive Estimation) loss function.

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

    def __init__(self, temperature=0.5):
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
            similarities (torch.Tensor): A flat tensor (1-D) containing the similarity scores (logits).
            target (torch.Tensor): A flat tensor (1-D) containing the binary target labels (0 or 1).

        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        logits = similarities / self.temperature
        loss = F.binary_cross_entropy_with_logits(logits, target)
        print(F.sigmoid(logits), target, loss, sep=" ::: ", end="\r")
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
    preds += 1
    print(preds / 0.5, target)
    loss_fn = InfoNCE()
    print(loss_fn(preds, target))
