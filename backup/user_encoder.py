import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsLevelMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for news articles.
    Computes attention between news items browsed by the same user.
    """

    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Parameters for each attention head
        self.Q = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_heads)]
        )
        self.V = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_heads)]
        )

    def forward(self, news_embeds):
        """
        Args:
            news_embeds: (batch_size, num_news, input_dim) - news representations

        Returns:
            multi_head_repr: (batch_size, num_news, input_dim * num_heads)
        """
        batch_size, M, dim = news_embeds.shape
        head_outputs = []

        for k in range(self.num_heads):
            # Get the linear layers for this head
            Q_layer = self.Q[k]
            V_layer = self.V[k]
            # Apply Q transformation for the k-th head
            Q_k = Q_layer(news_embeds)  # (batch, M, dim)
            # Compute attention scores: β^k_{i,j} = exp(r_i^T Q_k^T r_j) / Σ_m exp(r_i^T Q_k^T r_m)
            # Shape: (batch, M, M)
            scores = torch.bmm(Q_k, news_embeds.transpose(1, 2))  # (batch, M, M)
            attention_weights = F.softmax(scores, dim=2)  # (batch, M, M)

            # Compute weighted sum: h^n_{i,k} = V_k(Σ_j β^k_{i,j} r_j)
            weighted_news = torch.bmm(attention_weights, news_embeds)  # (batch, M, dim)
            head_repr = V_layer(weighted_news)  # (batch, M, dim)
            head_outputs.append(head_repr)

        # Concatenate all heads: h^n_i = [h^n_{i,1}; h^n_{i,2}; ...; h^n_{i,h}]
        multi_head_repr = torch.cat(head_outputs, dim=2)  # (batch, M, dim * num_heads)

        return multi_head_repr


class AdditiveNewsAttention(nn.Module):
    """
    Additive attention mechanism to select important news for user representation.
    """

    def __init__(self, news_dim):
        super().__init__()
        self.V_n = nn.Linear(news_dim, news_dim)
        self.v_n = nn.Linear(news_dim, 1, bias=False)
        self.q_n = nn.Parameter(torch.randn(news_dim))

    def forward(self, news_repr):
        """
        Args:
            news_repr: (batch_size, num_news, news_dim) - multi-head news representations

        Returns:
            user_repr: (batch_size, news_dim) - final user representation
            attention_weights: (batch_size, num_news) - attention weights for interpretability
        """
        # Equation (8): a^n_i = q_n^T tanh(V_n × h^n_i + v_n)
        transformed = self.V_n(news_repr)  # (batch, N, news_dim)

        # Add v_n (broadcast q_n across batch and news dimensions)
        q_n_expanded = self.q_n.unsqueeze(0).unsqueeze(0)  # (1, 1, news_dim)
        combined = torch.tanh(transformed + q_n_expanded)  # (batch, N, news_dim)

        # Compute attention scores
        attention_scores = self.v_n(combined).squeeze(-1)  # (batch, N)

        # Equation (9): α^n_i = exp(a^n_i) / Σ_j exp(a^n_j)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, N)

        # Equation (10): u = Σ_i α^n_i h^n_i
        user_repr = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, N)
            news_repr,  # (batch, N, news_dim)
        ).squeeze(
            1
        )  # (batch, news_dim)

        return user_repr, attention_weights


class UserEncoder(nn.Module):
    """
    Complete User Encoder module combining multi-head self-attention and additive attention.
    """

    def __init__(self, news_embed_dim, num_heads=16):
        super().__init__()

        # Layer 1: News-level multi-head self-attention
        self.news_self_attention = NewsLevelMultiHeadSelfAttention(
            input_dim=news_embed_dim, num_heads=num_heads
        )

        # Layer 2: Additive news attention
        multi_head_dim = news_embed_dim * num_heads
        self.additive_attention = AdditiveNewsAttention(multi_head_dim)

    def forward(self, news_embeds):
        """
        Args:
            news_embeds: (batch_size, num_news, news_embed_dim)
                        - embeddings of news browsed by users

        Returns:
            user_repr: (batch_size, news_embed_dim * num_heads) - user representation
            attention_weights: (batch_size, num_news) - news importance weights
        """
        # Apply multi-head self-attention on news
        news_multi_head_repr = self.news_self_attention(news_embeds)

        # Apply additive attention to get user representation
        user_repr, attention_weights = self.additive_attention(news_multi_head_repr)

        return user_repr, attention_weights


# Example usage
if __name__ == "__main__":
    batch_size = 4
    news_embed_dim = 256
    num_heads = 16

    # Create sample news embeddings
    news_embeds = torch.randn(batch_size, 10, news_embed_dim)

    # Initialize user encoder
    user_encoder = UserEncoder(news_embed_dim, num_heads)

    # Get user representation
    user_repr, attention_weights = user_encoder(news_embeds)

    print(f"Input news embeddings shape: {news_embeds.shape}")
    print(f"User representation shape: {user_repr.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nSample attention weights (should sum to 1):")
    print(attention_weights[0])
    print(f"Sum: {attention_weights[0].sum()}")

