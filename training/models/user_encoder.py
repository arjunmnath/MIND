import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsLevelMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for news articles.
    Computes attention between news items browsed by the same user.
    The embedding dimension is split across heads.
    """

    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads  # d_k in your description

        # Single linear layers that project to all heads at once
        self.W_Q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_V = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, news_embeds):
        """
        Args:
            news_embeds: (batch_size, num_news, input_dim) - news representations

        Returns:
            multi_head_repr: (batch_size, num_news, input_dim)
        """
        batch_size, M, dim = news_embeds.shape

        # Step 1: Linear transformations to get Q and V for all heads
        # Q, V shape: (batch_size, num_news, input_dim)
        Q = self.W_Q(news_embeds)
        V = self.W_V(news_embeds)

        # Step 2: Reshape to split into heads
        # (batch, M, input_dim) -> (batch, M, num_heads, head_dim) -> (batch, num_heads, M, head_dim)
        Q = Q.view(batch_size, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Also reshape original embeddings for attention computation
        news_embeds_heads = news_embeds.view(
            batch_size, M, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # Shape: (batch, num_heads, M, head_dim)

        # Step 3: Compute attention for each head
        # Attention scores: Q @ news_embeds^T
        # (batch, num_heads, M, head_dim) @ (batch, num_heads, head_dim, M)
        # -> (batch, num_heads, M, M)
        scores = torch.matmul(Q, news_embeds_heads.transpose(-2, -1))

        # Apply softmax to get attention weights β^k_{i,j}
        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, M, M)

        # Step 4: Apply attention weights to get weighted news
        # (batch, num_heads, M, M) @ (batch, num_heads, M, head_dim)
        # -> (batch, num_heads, M, head_dim)
        weighted_news = torch.matmul(attention_weights, news_embeds_heads)

        # Step 5: Apply V transformation
        # (batch, num_heads, M, head_dim) @ (batch, num_heads, M, head_dim)
        # We need to apply V to weighted_news
        # Actually V transformation is already applied, so we use V matrix
        head_outputs = torch.matmul(
            attention_weights, V
        )  # (batch, num_heads, M, head_dim)

        # Step 6: Concatenate heads back together
        # (batch, num_heads, M, head_dim) -> (batch, M, num_heads, head_dim) -> (batch, M, input_dim)
        multi_head_repr = (
            head_outputs.transpose(1, 2).contiguous().view(batch_size, M, dim)
        )

        return multi_head_repr


class AdditiveNewsAttention(nn.Module):
    """
    Additive attention mechanism to select important news for user representation.
    """

    def __init__(self, news_dim):
        super().__init__()
        self.V_n = nn.Linear(news_dim, news_dim)
        self.v_n = nn.Parameter(torch.zeros(news_dim))
        self.q_n = nn.Parameter(torch.zeros(news_dim))
        
        # Initialize parameters properly
        nn.init.xavier_uniform_(self.V_n.weight)
        nn.init.zeros_(self.V_n.bias)
        nn.init.normal_(self.v_n, mean=0.0, std=0.02)
        nn.init.normal_(self.q_n, mean=0.0, std=0.02)

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

        # Add v_n (broadcast across batch and news dimensions)
        v_n_expanded = self.v_n.unsqueeze(0).unsqueeze(0)  # (1, 1, news_dim)
        combined = torch.tanh(transformed + v_n_expanded)  # (batch, N, news_dim)

        # Multiply by q_n and sum across the last dimension
        q_n_expanded = self.q_n.unsqueeze(0).unsqueeze(0)  # (1, 1, news_dim)
        attention_scores = (combined * q_n_expanded).sum(dim=-1)  # (batch, N)

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
    Paper: https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf
    Complete User Encoder module combining multi-head self-attention and additive attention.
    The embedding dimension is split across attention heads as per standard transformer architecture.
    """

    def __init__(self, news_embed_dim, num_heads=16):
        super().__init__()

        assert (
            news_embed_dim % num_heads == 0
        ), f"news_embed_dim ({news_embed_dim}) must be divisible by num_heads ({num_heads})"

        # Layer 1: News-level multi-head self-attention
        self.news_self_attention = NewsLevelMultiHeadSelfAttention(
            input_dim=news_embed_dim, num_heads=num_heads
        )

        # Layer 2: Additive news attention
        # Note: output dimension stays the same (not multiplied by num_heads)
        self.additive_attention = AdditiveNewsAttention(news_embed_dim)

    def forward(self, news_embeds):
        """
        Args:
            news_embeds: (batch_size, num_news, news_embed_dim)
                        - embeddings of news browsed by users

        Returns:
            user_repr: (batch_size, news_embed_dim) - user representation
            attention_weights: (batch_size, num_news) - news importance weights
        """
        # Apply multi-head self-attention on news
        news_multi_head_repr = self.news_self_attention(news_embeds)

        # Apply additive attention to get user representation
        user_repr, attention_weights = self.additive_attention(news_multi_head_repr)

        return user_repr, attention_weights
