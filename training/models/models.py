import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torchinfo import summary

try:
    from .user_encoder import UserEncoder

except ImportError:
    from user_encoder import UserEncoder


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

    def __init__(self, embed_dim=768, dropout=0.1):
        super(NewsEncoder, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, contents: torch.Tensor) -> torch.Tensor:
        """
        Encodes and projects the input into a normalized vector.
        Args:
            contents (torch.Tensor]): Input pre-encoded vectors.
        Returns:
            torch.Tensor: Normalized projected vectors.
        """
        projections = self.project(contents)  # shape: [batch_size, 768]
        return F.normalize(projections, p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = UserEncoder(768)
        self.news_tower = NewsEncoder()
        self._cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, history, clicks, non_clicks):
        """
        history: [batch_size, history_pad_size, 768]
        clicks: [batch_size, clicks_pad_size, 768]
        non_clicks: [batch_size, non_clicks_pad_size, 768]
        """
        history = self.news_tower(history)
        clicks = self.news_tower(clicks)  # dim: [batch_size, clicks_pad_size, 768]
        non_clicks = self.news_tower(non_clicks)
        user_repr, _ = self.user_tower(history)  # dim: [batch_size, 768]
        user_repr = F.normalize(
            user_repr.unsqueeze(1), p=2, dim=-1
        )  # dim: [batch_size, 1, 768]
        click_pad_size = clicks.shape[1]
        non_click_pad_size = non_clicks.shape[1]
        labels_positive = torch.ones(
            clicks.shape[0], clicks.shape[1], device=user_repr.device
        )
        lables_negative = torch.full(
            (non_clicks.shape[0], non_clicks.shape[1]), -1, device=user_repr.device
        )
        impressions = torch.cat([clicks, non_clicks], dim=1)
        labels = torch.cat([labels_positive, lables_negative], dim=1)
        loss = self._cosine_loss(
            user_repr.expand(-1, click_pad_size + non_click_pad_size, 768).reshape(
                -1, 768
            ),
            impressions.reshape(-1, 768),
            labels.flatten(),
        )

        assert not torch.isnan(history).any(), "NaNs found in history"
        assert not torch.isnan(clicks).any(), "NaNs found in clicks"
        assert not torch.isnan(non_clicks).any(), "NaNs found in non_clicks"
        assert not torch.isnan(loss).any(), "NaNs found in loss"
        return (loss, user_repr, impressions, labels)


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
        return loss


if __name__ == "__main__":
    import sys

    from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        user_encoder = UserEncoder(768)
        news_encoder = NewsEncoder()
        news_input = torch.randn(32, 768)  # [batch_size, embed_dim]
        user_input = torch.randn(32, 6, 768)  # [batch_size, n_history, embed_dim]
        torch.onnx.export(
            user_encoder,
            user_input,
            "user_encoder.onnx",
            verbose=True,
            input_names=["history"],
            output_names=["user_representation"],
            dynamo=True
        )
        torch.onnx.export(
            news_encoder,
            news_input,
            "news_encoder.onnx",
            verbose=True,
            input_names=["news_embeddings"],
            output_names=["news_representation"],
            dynamo=True
        )
        exit(0)
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
