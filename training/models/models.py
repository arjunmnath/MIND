import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatingUserEncoder(nn.Module):
    """
    Aggregating User Encoder using Attention Pooling (batch_first).

    This module processes a user's browsed news as a "bag of interests"
    to generate a single, aggregated user embedding. It uses a special [CLS] token
    and non-causal self-attention to summarize the entire history.
    This version uses batch_first=True for more intuitive tensor handling.
    """

    def __init__(self, news_embed_dim, max_seq_len=512, num_heads=16, dropout=0.3):
        super().__init__()
        assert (
            news_embed_dim % num_heads == 0
        ), f"news_embed_dim ({news_embed_dim}) must be divisible by num_heads ({num_heads})"

        self.cls_token = nn.Parameter(torch.randn(1, 1, news_embed_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # The key change is here!
        )
        self.position_encoding = nn.Embedding(max_seq_len + 1, news_embed_dim)
        self.project = nn.Sequential(
            nn.Linear(news_embed_dim, news_embed_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(news_embed_dim),
            nn.ReLU(),
        )

    def forward(
        self, news_embeds: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            news_embeds: (batch_size, num_news, news_embed_dim)
            padding_mask: (batch_size, num_news) - True indicates a padded item.

        Returns:
            user_embedding: (batch_size, news_embed_dim)
            attention_weights: (batch_size, num_heads, num_news+1, num_news+1)
        """
        batch_size, n_history, embed_dim = news_embeds.shape
        device = news_embeds.device

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        news_embeds = torch.cat((cls_tokens, news_embeds), dim=1)

        seq_len = n_history + 1

        time_steps = torch.arange(seq_len, device=device)
        pos_encodings = self.position_encoding(time_steps).unsqueeze(0)
        news_embeds = news_embeds + pos_encodings

        if padding_mask is not None:
            cls_padding_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=device
            )
            padding_mask = torch.cat((cls_padding_mask, padding_mask), dim=1)
        attn_output, attn_weights = self.attention(
            query=news_embeds,
            key=news_embeds,
            value=news_embeds,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        projected_output = self.project(attn_output)
        user_embedding = projected_output[:, 0, :]
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)

        return user_embedding, attn_weights


class UserEncoder(nn.Module):
    """
    Complete User Encoder module combining multi-head self-attention.
    """

    def __init__(self, news_embed_dim, max_seq_len=558, num_heads=16, dropout=0.3):
        super().__init__()

        assert (
            news_embed_dim % num_heads == 0
        ), f"news_embed_dim ({news_embed_dim}) must be divisible by num_heads ({num_heads})"
        self.attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.position_encoding = nn.Embedding(max_seq_len, news_embed_dim)
        self.project = nn.Sequential(
            nn.Linear(news_embed_dim, news_embed_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(news_embed_dim),
            nn.ReLU(),
        )
        self.temporal_decay = nn.Embedding(max_seq_len, 1)

    def forward(
        self, news_embeds: torch.tensor, mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Args:
            news_embeds: (batch_size, num_news, news_embed_dim) - embeddings of news browsed by users
        Returns:
            user_repr: (batch_size, news_embed_dim) - user representation
            attention_weights: (batch_size, num_news) - news importance weights
        """
        n_history = news_embeds.shape[1]
        time_steps = torch.arange(n_history, device=news_embeds.device)

        pos_encodings = self.position_encoding(time_steps).unsqueeze(0)
        news_embeds = news_embeds + pos_encodings
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
        attn_output = (
            attn_output * ((mask.mT.unsqueeze(2)).float())
            if mask is not None
            else attn_output
        )
        weighted_sum = torch.sum(
            attn_output * self.temporal_decay(time_steps).unsqueeze(2),
            dim=0,
        )
        user_repr = self.project(weighted_sum)
        user_repr = F.normalize(user_repr.unsqueeze(1), p=2, dim=-1)
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

    def forward(
        self, contents: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encodes and projects the input into a normalized vector.
        Args:
            contents (torch.Tensor]): Input pre-encoded vectors.
        Returns:
            torch.Tensor: Normalized projected vectors.
        """
        projections = self.project(contents)  # shape: [batch_size, 768]
        projections = (
            projections * mask.unsqueeze(2).float() if mask is not None else projections
        )
        return F.normalize(contents + projections, p=2, dim=-1)


class TwoTowerRecommendation(nn.Module):
    def __init__(self):
        super(TwoTowerRecommendation, self).__init__()
        self.user_tower = AggregatingUserEncoder(768)
        self.news_tower = NewsEncoder()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, history, clicks, non_clicks, log=False):
        """
        history: [batch_size, history_pad_size, 768]
        clicks: [batch_size, clicks_pad_size, 768]
        non_clicks: [batch_size, non_clicks_pad_size, 768]
        """
        assert history.ndim == 3 and clicks.ndim == 3 and non_clicks.ndim == 3
        batch_size = clicks.shape[0]

        history_mask = history.sum(dim=-1) != 0
        click_mask = clicks.sum(dim=-1) != 0
        non_click_mask = non_clicks.sum(dim=-1) != 0

        seq_len = history_mask[0].long().sum().item()
        click_seq_len = click_mask.long().sum().item()
        non_click_seq_len = non_click_mask.long().sum().item()

        clicks_per_sample = click_mask.sum(dim=-1)
        non_clicks_per_sample = non_click_mask.sum(dim=-1)
        samples_per_batch = clicks_per_sample + non_clicks_per_sample

        history = self.news_tower(history, mask=history_mask)
        clicks = self.news_tower(clicks, mask=click_mask)
        non_clicks = self.news_tower(non_clicks, mask=non_click_mask)
        user_repr, attn_scores = self.user_tower(history, padding_mask=~history_mask)
        user_repr = user_repr.unsqueeze(1)
        pos_keys = (user_repr @ clicks.mT).squeeze()
        neg_keys = (user_repr @ non_clicks.mT).squeeze()
        neg_keys[~non_click_mask] = -torch.inf
        repeated_neg_keys = torch.repeat_interleave(neg_keys, clicks_per_sample, dim=0)
        pos_keys_flattened = pos_keys[click_mask]
        result = torch.cat((pos_keys_flattened.unsqueeze(1), repeated_neg_keys), dim=1)
        loss = self._loss(
            result,
            torch.zeros(result.size(0), dtype=torch.long, device=user_repr.device),
        )
        if not log:
            return loss

        labels_positive = torch.ones(click_seq_len, device=user_repr.device)
        lables_negative = torch.full((non_click_seq_len,), -1, device=user_repr.device)

        index_positive = torch.arange(
            batch_size, device=user_repr.device, dtype=torch.long
        ).repeat_interleave(clicks_per_sample)
        index_negative = torch.arange(
            batch_size, device=user_repr.device, dtype=torch.long
        ).repeat_interleave(non_clicks_per_sample)

        impressions = torch.cat(
            [
                clicks[click_mask].view(-1, 768),
                non_clicks[non_click_mask].view(-1, 768),
            ],
            dim=0,
        )
        labels = torch.cat([labels_positive, lables_negative], dim=0)
        indexes = torch.cat([index_positive, index_negative], dim=0)
        user_repr = user_repr.squeeze(1).repeat_interleave(samples_per_batch, dim=0)
        return (
            loss,
            (user_repr * impressions).sum(dim=-1),
            (labels + 1) / 2,
            indexes,
            attn_scores,
            seq_len,
        )
