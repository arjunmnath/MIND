import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torchinfo import summary

# try:
#     from .user_encoder import UserEncoder
#
# except ImportError:
#     from user_encoder import UserEncoder
#


class UserEncoder(nn.Module):
    """
    Complete User Encoder module combining multi-head self-attention.
    """

    def __init__(self, news_embed_dim, num_heads=16):
        super().__init__()

        assert (
            news_embed_dim % num_heads == 0
        ), f"news_embed_dim ({news_embed_dim}) must be divisible by num_heads ({num_heads})"

        # Define the multi-head self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim, num_heads=num_heads
        )

        # Optionally, use a linear layer to project the output to the desired representation size
        self.fc = nn.Linear(news_embed_dim, news_embed_dim)

    def forward(self, news_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            news_embeds: (batch_size, num_news, news_embed_dim)
                        - embeddings of news browsed by users
        Returns:
            user_repr: (batch_size, news_embed_dim) - user representation
            attention_weights: (batch_size, num_news) - news importance weights
        """
        news_embeds = news_embeds.transpose(
            0, 1
        )  # (num_news, batch_size, news_embed_dim)
        attn_output, attn_weights = self.attention(
            news_embeds,
            news_embeds,
            news_embeds,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_output = attn_output * ((mask.mT.unsqueeze(2)).float())
        user_repr = attn_output.mean(dim=0)
        user_repr = self.fc(user_repr)
        return user_repr, attn_weights


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

    Methods:
        forward(contents): Encodes and projects the input, returning normalized vectors.
    """

    def __init__(self, embed_dim=768, dropout=0.1):
        super(NewsEncoder, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, contents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes and projects the input into a normalized vector.
        Args:
            contents (torch.Tensor]): Input pre-encoded vectors.
        Returns:
            torch.Tensor: Normalized projected vectors.
        """
        projections = self.project(contents)  # shape: [batch_size, 768]
        projections = projections * mask.unsqueeze(2).float()
        return F.normalize(contents + projections, p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = UserEncoder(768)
        self.news_tower = NewsEncoder()
        self._cosine_loss = nn.CosineEmbeddingLoss()
        self._infonce = InfoNCE()
        self.apply(self.init_weights)

    def forward(self, history, clicks, non_clicks):
        """
        history: [batch_size, history_pad_size, 768]
        clicks: [batch_size, clicks_pad_size, 768]
        non_clicks: [batch_size, non_clicks_pad_size, 768]
        """
        batch_size = clicks.shape[0]
        history_mask = history.sum(dim=-1) != 0
        click_mask = clicks.sum(dim=-1) != 0
        non_click_mask = non_clicks.sum(dim=-1) != 0
        seq_len = history_mask.long().sum().item()
        click_seq_len = click_mask.long().sum().item()
        non_click_seq_len = non_click_mask.long().sum().item()
        history = self.news_tower(
            history, mask=history_mask
        )  # dim: [batch_size, seq_len, 768]
        clicks = self.news_tower(
            clicks, mask=click_mask
        )  # dim: [batch_size, num_clicks, 768]
        non_clicks = self.news_tower(
            non_clicks, mask=non_click_mask
        )  # dim: [batch_size, num_non_clicks, 768]
        user_repr, attn_scores = self.user_tower(
            history, mask=history_mask
        )  # dim: [batch_size, 768]
        user_repr = F.normalize(
            user_repr.unsqueeze(1), p=2, dim=-1
        )  # dim: [batch_size, 1, 768]
        labels_positive = torch.ones(batch_size, click_seq_len, device=user_repr.device)
        lables_negative = torch.full(
            (batch_size, non_click_seq_len), -1, device=user_repr.device
        )
        impressions = torch.cat(
            [
                clicks[click_mask].view(1, -1, 768),
                non_clicks[non_click_mask].view(1, -1, 768),
            ],
            dim=1,
        )
        labels = torch.cat([labels_positive, lables_negative], dim=1)
        loss1 = self._cosine_loss(
            user_repr.expand(-1, click_seq_len + non_click_seq_len, 768).reshape(
                -1, 768
            ),
            impressions.reshape(-1, 768),
            labels.flatten(),
        )
        # loss2 = self._infonce(user_repr, impressions, (labels + 1) / 2)
        return (loss1, user_repr, impressions, labels, attn_scores, seq_len)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class InfoNCE(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss.
    Paper: https://arxiv.org/abs/1807.03748
    """

    def __init__(self, temperature=0.5):
        """
        Initializes the InfoNCE loss module with the given temperature.

        Args:
            temperature (float, optional): A scaling factor for similarity logits. Default is 0.07.
        """
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, key: torch.Tensor, labels: torch.Tensor):
        """
        Compute the InfoNCE loss.

        Assumes positive pairs are aligned by index (query[i] matches key[i])

        Args:
            query: Tensor of shape (batch_size, embedding_dim)
            key: Tensor of shape (batch_size, embedding_dim)
            labels: (batch_size, num_impressions) - binary matrix where 1=positive, 0=negative
        Returns:
            torch.Tensor: The computed InfoNCE loss.

        """
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        logits = (query @ key.mT).squeeze(1)
        # sigmoid_logits = (torch.sigmoid(logits) - 0.5) * 2
        # sigmoid_logits = torch.clamp(sigmoid_logits, min=-1, max=1)
        loss = -torch.mean(
            labels.float() * torch.log(logits)
            + (1 - labels.float()) * torch.log(1 - logits)
        )
        return loss


if __name__ == "__main__":
    import sys

    from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        user_encoder = UserEncoder(768)
        news_encoder = NewsEncoder()
        news_input = torch.randn(32, 768)  # [batch_size, embed_dim]
        user_input = torch.randn(32, 6, 768)  # [batch_size, n_history, embed_dim]
        news_dynamic_shapes = {
            "contents": {
                0: "batch_size",
            },
        }
        torch.onnx.export(
            user_encoder,
            user_input,
            "user_encoder.onnx",
            verbose=True,
            input_names=["history"],
            output_names=["user_representation"],
            dynamo=True,
        )
        torch.onnx.export(
            news_encoder,
            news_input,
            "news_encoder.onnx",
            verbose=True,
            dynamic_shapes=news_dynamic_shapes,
            input_names=["news_embeddings"],
            output_names=["news_representation"],
            dynamo=True,
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
