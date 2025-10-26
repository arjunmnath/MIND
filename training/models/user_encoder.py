import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for two-tower recommendation systems.
    Supports in-batch negatives + sampled hard negatives.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Scaling factor for logits (lower = sharper distributions)
            use_in_batch_negatives: Whether to use other samples' positives as negatives
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, pos_emb, neg_emb):
        """
        Args:
            user_emb: (batch_size, n_positive, emb_dim) - user tower output
            pos_emb: (batch_size, n_positive, emb_dim) - positive article embeddings
            neg_emb: (batch_size, num_neg, emb_dim) - sampled negative embeddings (optional)

        Returns:
            loss: scalar tensor
            metrics: dict with accuracy and margin stats
        """
        batch_size = user_emb.shape[0]
        user_emb = F.normalize(user_emb, dim=-1)  # (batch_size, 768)
        pos_emb = F.normalize(pos_emb, dim=-1)  # (batch_size, N, 768)
        neg_emb = F.normalize(neg_emb, dim=-1)  # (batch_size, N, 768)
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1) / self.temperature

        logits = pos_scores.unsqueeze(1)  # (batch_size, 1)
        neg_scores = (
            torch.sum(user_emb.unsqueeze(1) * neg_emb, dim=-1) / self.temperature
        )

        logits = torch.cat([logits, neg_scores], dim=1)

        # InfoNCE loss: positive should rank first
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_emb.device)
        loss = F.cross_entropy(logits, labels)

        # Compute metrics for monitoring
        with torch.no_grad():
            # Accuracy: % of samples where positive scores highest
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == 0).float().mean()

            # Average positive score
            avg_pos_score = pos_scores.mean()

            # Average negative score (if negatives exist)
            if logits.shape[1] > 1:
                neg_logits = logits[:, 1:]  # all negatives
                avg_neg_score = neg_logits.max(dim=1)[
                    0
                ].mean()  # hardest negative per sample
                margin = avg_pos_score - avg_neg_score
            else:
                avg_neg_score = torch.tensor(0.0)
                margin = torch.tensor(0.0)

            metrics = {
                "accuracy": accuracy.item(),
                "avg_pos_score": avg_pos_score.item(),
                "avg_neg_score": avg_neg_score.item(),
                "margin": margin.item(),
            }

        return loss, metrics


class InfoNCEWithHardNegativeMining(nn.Module):
    """
    InfoNCE with online hard negative mining.
    Selects hardest K negatives from a larger pool.
    """

    def __init__(
        self, temperature=0.07, use_in_batch_negatives=True, top_k_hard_negatives=5
    ):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        self.top_k = top_k_hard_negatives

    def forward(self, user_emb, pos_emb, neg_emb_pool):
        """
        Args:
            user_emb: (batch_size, emb_dim)
            pos_emb: (batch_size, emb_dim)
            neg_emb_pool: (batch_size, pool_size, emb_dim) - larger pool of negatives

        Selects top_k hardest negatives from the pool based on similarity.
        """
        batch_size = user_emb.shape[0]

        # Normalize
        user_emb = F.normalize(user_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb_pool = F.normalize(neg_emb_pool, dim=-1)

        # Compute similarities with all negatives in pool
        # (batch_size, pool_size)
        neg_similarities = torch.sum(user_emb.unsqueeze(1) * neg_emb_pool, dim=-1)

        # Select top-k hardest negatives (highest similarity = hardest)
        _, top_k_indices = torch.topk(neg_similarities, k=self.top_k, dim=1)

        # Gather hard negatives (batch_size, top_k, emb_dim)
        batch_indices = torch.arange(batch_size, device=user_emb.device).unsqueeze(1)
        hard_neg_emb = neg_emb_pool[batch_indices, top_k_indices]

        # Now use standard InfoNCE with selected hard negatives
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1) / self.temperature
        logits = pos_scores.unsqueeze(1)

        if self.use_in_batch_negatives:
            in_batch_scores = torch.matmul(user_emb, pos_emb.T) / self.temperature
            mask = torch.eye(batch_size, device=user_emb.device).bool()
            in_batch_scores = in_batch_scores.masked_fill(mask, float("-inf"))
            logits = torch.cat([logits, in_batch_scores], dim=1)

        # Add hard negatives
        hard_neg_scores = (
            torch.sum(user_emb.unsqueeze(1) * hard_neg_emb, dim=-1) / self.temperature
        )
        logits = torch.cat([logits, hard_neg_scores], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=user_emb.device)
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            accuracy = (torch.argmax(logits, dim=1) == 0).float().mean()
            metrics = {
                "accuracy": accuracy.item(),
                "avg_pos_score": pos_scores.mean().item(),
                "avg_hard_neg_score": hard_neg_scores.max(dim=1)[0].mean().item(),
            }

        return loss, metrics


# ============================================================================
# Usage Example
# ============================================================================


def example_usage():
    """Example of how to use InfoNCE loss in training loop"""

    # Hyperparameters
    batch_size = 256
    emb_dim = 128
    num_sampled_negatives = 10

    # Initialize loss
    criterion = InfoNCELoss(
        temperature=0.07, use_in_batch_negatives=True  # Recommended!
    )

    # Dummy data (replace with your actual model outputs)
    user_emb = torch.randn(batch_size, emb_dim)  # From user tower
    pos_article_emb = torch.randn(batch_size, emb_dim)  # From article tower (positives)
    neg_article_emb = torch.randn(
        batch_size, num_sampled_negatives, emb_dim
    )  # Sampled negatives

    # Compute loss
    loss, metrics = criterion(user_emb, pos_article_emb, neg_article_emb)

    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Margin (pos - neg): {metrics['margin']:.4f}")

    # Backward pass
    loss.backward()

    # With only in-batch negatives (no sampled negatives)
    loss_in_batch_only, metrics = criterion(user_emb, pos_article_emb, neg_emb=None)
    logger.info(f"Loss (in-batch only): {loss_in_batch_only.item():.4f}")

    # With hard negative mining
    criterion_hnm = InfoNCEWithHardNegativeMining(
        temperature=0.07, use_in_batch_negatives=True, top_k_hard_negatives=5
    )

    # Provide larger pool of negatives (e.g., 50), automatically selects hardest 5
    neg_pool = torch.randn(batch_size, 50, emb_dim)
    loss_hnm, metrics_hnm = criterion_hnm(user_emb, pos_article_emb, neg_pool)
    logger.info(f"Loss (with hard neg mining): {loss_hnm.item():.4f}")


if __name__ == "__main__":
    example_usage()


# ============================================================================
# Training Tips
# ============================================================================
"""
RECOMMENDATIONS:

1. Temperature tuning:
   - Start with 0.07 (common default)
   - Lower (0.01-0.05) = sharper, more aggressive
   - Higher (0.1-0.3) = softer, more forgiving
   - If training is unstable, increase temperature

2. Batch size:
   - Larger is better for in-batch negatives (256-512 recommended)
   - With batch_size=256, you get 255 free negatives per sample!

3. Negative sampling strategy:
   - In-batch negatives: Always use (free & hard)
   - Sampled negatives: 5-10 per sample
   - Mix: 50% random, 30% popularity-biased, 20% hard (similar embeddings)

4. Learning rate:
   - Start lower than with CosineEmbeddingLoss (1e-4 to 5e-4)
   - InfoNCE gradients can be larger
   - Use warmup (1000-2000 steps)

5. Monitoring:
   - Watch 'accuracy' metric: should approach 0.9+ as training progresses
   - Watch 'margin': should increase (positives getting higher scores than negatives)
   - If accuracy stays low (<0.5), decrease temperature or check data quality

6. Comparing to your current setup:
   - InfoNCE should give MORE stable gradients
   - Metrics should be less noisy
   - Training should converge faster
"""
