from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


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

    def __init__(
        self, embed_dim=768, hidden_dim=128, dropout=0.1, is_vector_input=True
    ):
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


class UserEncoder(nn.Module):
    """
    paper: Neural News Recommendation with Multi-Head Self-Attention (https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf)
    """

    def __init__(self, embed_dim=768, hidden_dim=128, dropout=0.1, agg="attention"):
        super(UserEncoder, self).__init__()
        self.agg_type = agg
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        if agg == "attention":
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        elif agg == "gru":
            self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(self, history_embs):
        if self.agg_type == "mean":
            out = history_embs.mean(dim=1)
        elif self.agg_type == "gru":
            _, h = self.gru(history_embs)
            out = h.squeeze(0)
        elif self.agg_type == "attention":
            attn_out, _ = self.attn(history_embs, history_embs, history_embs)
            out = attn_out
        else:
            raise ValueError("Unknown aggregator")
        attn_out, _ = self.attn(history_embs, history_embs, history_embs)
        out = attn_out
        return F.normalize(self.project(out), p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = UserEncoder()
        self.news_tower = NewsEncoder()

    def forward(self, history, clicks, non_clicks):
        indexes = []
        relevance = []
        target = []
        return indexes, relevance, target


def infonce_loss(anchor, positive, negative, temperature=0.07):
    """
    Paper: https://arxiv.org/pdf/1807.03748#page=3

    Args:
    - anchor: Tensor of anchor embeddings, shape (batch_size, embedding_dim)
    - positive: Tensor of positive embeddings, shape (batch_size, embedding_dim)
    - negative: Tensor of negative embeddings, shape (batch_size, num_negatives, embedding_dim)
    - temperature: Temperature scaling parameter (default=0.07)
    """

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=2)

    # Calculate similarity scores (dot products)
    pos_sim = torch.bmm(
        anchor.unsqueeze(1), positive.unsqueeze(2)
    ).squeeze()  # (batch_size,)
    neg_sim = torch.bmm(
        anchor.unsqueeze(1), negative.transpose(1, 2)
    )  # (batch_size, num_negatives)

    # Combine positive and negative similarities
    logits = torch.cat(
        [pos_sim.unsqueeze(1), neg_sim], dim=1
    )  # (batch_size, num_negatives + 1)

    # Apply temperature scaling
    logits = logits / temperature

    # Labels for InfoNCE (positive sample is the first one)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss
