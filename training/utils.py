import mlflow
import torch


def train(
    dataloader,
    model,
    loss_fn,
    auc_roc,
    ndcg_5,
    ndcg_10,
    optimizer,
    epoch,
    device,
    log_every,
):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        auc = auc_roc(pred, y)
        ndcg_5 = ndcg_5(pred, y)
        ndcg_10 = ndcg_10(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % log_every == 0:
            loss_value = loss.item()
            current = batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", loss_value, step=step)
            mlflow.log_metric("auc", auc, step=step)
            mlflow.log_metric("ndcg_5", ndcg_5, step=step)
            mlflow.log_metric("ndcg_10", ndcg_10, step=step)
            print(
                f"loss: {loss_value:.4f} auc: {auc:.4f} ndcg@5: {ndcg_5:.4f} ndcg@10: {ndcg_10:.4f} [{current} / {len(dataloader)}]"
            )


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
