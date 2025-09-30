import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class NewsTower(nn.Module):
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
        super(NewsTower, self).__init__()
        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
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


class UserTower(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128, dropout=0.1, agg="attention"):
        super(UserTower, self).__init__()
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
        # if self.agg_type == "mean":
        #     out = history_embs.mean(dim=1)
        # elif self.agg_type == "gru":
        #     _, h = self.gru(history_embs)
        #     out = h.squeeze(0)
        # elif self.agg_type == "attention":
        #     attn_out, _ = self.attn(history_embs, history_embs, history_embs)
        #     out = attn_out
        # else:
        #     raise ValueError("Unknown aggregator")
        attn_out, _ = self.attn(history_embs, history_embs, history_embs)
        out = attn_out
        return F.normalize(self.project(out), p=2, dim=-1)


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


# class TwoTowerRecommendation(nn.Module):
#     def __init__(self, user_tower, news_tower):
#         super(TwoTowerRecommendation, self).__init__()
#         self.user_tower = user_tower
#         self.news_tower = news_tower
#
#     def forward(self, user_history_embeddings, contents, labels):
#         news_embedding = self.news_tower(contents)
#         user_embedding = self.user_tower(user_history_embeddings)
#         similarity = F.cosine_similarity(user_embedding, news_embedding, dim=-1)
#         return similarity
#

contents = [
    "Breaking News: Economy Soars"
    + "The economy experienced a major surge today due to..."
]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

# news_tower = NewsTower(is_vector_input=False).to(device)
# print(torch.var(news_tower(contents)))
#
user_history_embeddings = torch.randn(1, 5, 768).to(device)
user_tower = UserTower(embed_dim=768, hidden_dim=128).to(device)
print(user_tower(user_history_embeddings).shape)

# user_embed = user_tower(user_history_embeddings).detach()
# news_embed = news_tower(contents).detach()
# labels = torch.ones(1).to(device)
# loss = nn.CosineEmbeddingLoss(margin=0.0)
# similarity_score = F.cosine_similarity(user_embed, news_embed, dim=-1)
# print(similarity_score, loss(user_embed, news_embed, labels))
