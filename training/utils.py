import io
import random
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch

from config_classes import OptimizerConfig


def upload_to_s3(obj, dst):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    dst = urlparse(dst)
    boto3.client("s3").upload_fileobj(buffer, dst.netloc, dst.path.lstrip("/"))


def evaluate(dataloader, model, loss_fn, auc_roc, ndcg_5, ndcg_10, epoch, device):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    eval_auc = 0
    eval_ndcg_5 = 0
    eval_ndcg_10 = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_auc += auc_roc(pred, y)
            eval_ndcg_5 += ndcg_5(pred, y)
            eval_ndcg_10 += ndcg_10(pred, y)

    eval_loss /= num_batches
    eval_auc /= num_batches
    eval_ndcg_5 /= num_batches
    eval_ndcg_10 /= num_batches
    mlflow.log_metric("eval_loss", eval_loss, step=epoch)
    mlflow.log_metric("eval_auc", eval_auc, step=epoch)
    mlflow.log_metric("eval_ndcg_5", eval_ndcg_5, step=epoch)
    mlflow.log_metric("eval_ndcg_10", eval_ndcg_10, step=epoch)
    print(
        f"Eval metrics: AUC ROC: {eval_auc:.4f}, Eval NDCG@5: {eval_ndcg_5:.4f}, Eval NDCG@10: {eval_ndcg_10:.4f}, Avg loss: {eval_loss:.4f} "
    )


def create_optimizer(model: torch.nn.Module, opt_config: OptimizerConfig):
    """
    This function separates out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("in_proj_weight"):
                # MHA projection layer
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith("pos_emb"):
                # positional embedding shouldn't be decayed
                no_decay.add(fpn)
            elif pn in ["q_n", "v_n"]:
                # attention parameters should not be decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": opt_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95)
    )
    return optimizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_attention_scores(
    attn_weights,
    non_padded_size,
    head_idx=None,
    figsize=(10, 8),
    save_path=None,
    title=None,
):
    """
    Plot attention scores from nn.MultiheadAttention for a single sample.

    Args:
        attn_weights: Attention weights tensor of shape (batch, num_heads, seq_len, seq_len)
                     or (num_heads, seq_len, seq_len) if batch already removed
        head_idx: Index of specific attention head to plot. If None, plots all heads.
        figsize: Figure size for the plot
        save_path: Path to save the figure. If None, displays the plot.
        title: Custom title for the plot

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Convert to numpy and handle batch dimension
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    # Remove batch dimension if present
    if attn_weights.ndim == 4:
        attn_weights = attn_weights[0]  # Take first sample
    attn_weights = attn_weights[:, :non_padded_size, :non_padded_size]
    num_heads, seq_len, _ = attn_weights.shape
    # Plot specific head or all heads
    if head_idx is not None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(attn_weights[head_idx], cmap="viridis", aspect="auto")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title or f"Attention Scores - Head {head_idx}")
        plt.colorbar(im, ax=ax, label="Attention Weight")
    else:
        # Plot all heads in a grid
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        for i in range(num_heads):
            im = axes[i].imshow(attn_weights[i], cmap="viridis", aspect="auto")
            axes[i].set_xlabel("Key Position")
            axes[i].set_ylabel("Query Position")
            axes[i].set_title(f"Head {i}")
            plt.colorbar(im, ax=axes[i], label="Attn Weight")

        # Hide extra subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis("off")

        if title:
            fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

    return fig, axes if head_idx is None else ax
