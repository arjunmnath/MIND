import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

from dataset import Mind
from models import *
from utils import evaluate, train

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

mlflow.login()
mlflow.set_experiment("/Users/dev.arjunmnath@gmail.com/mind-recommendation-system")


def main(args):
    epochs = args.epochs
    train_dir = Path(args.data_dir) / "train"
    test_dir = Path(args.data_dir) / "test"

    train_dataset = Mind(
        train_dir.absolute().as_posix(),
        precompute=args.precompute,
        embedding_path=args.embed_path,
    )
    test_dataset = Mind(
        test_dir.absolute().as_posix(),
        precompute=args.precompute,
        embedding_path=args.embed_path,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    auc_roc = RetrievalAUROC()
    ndcg_5 = RetrievalNormalizedDCG(top_k=5)
    ndcg_10 = RetrievalNormalizedDCG(top_k=10)
    model = NewsTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    with mlflow.start_run() as run:
        params = {
            "epochs": epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "loss_function": loss_fn.__class__.__name__,
            "metrics": [
                auc_roc.__class__.__name__,
                ndcg_5.__class__.__name__,
                ndcg_10.__class__.__name__,
            ],
            "optimizer": "AdamW",
        }
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(
                train_dataloader,
                model,
                loss_fn,
                auc_roc,
                ndcg_5,
                ndcg_10,
                optimizer,
                epoch=t,
                device=device,
                log_every=100,
            )
            evaluate(
                test_dataloader,
                model,
                loss_fn,
                auc_roc,
                ndcg_5,
                ndcg_10,
                epoch=t,
                device=device,
            )

        model_info = mlflow.pytorch.log_model(model, name="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Model Training and Evaluation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval (in batches) for logging metrics.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./model.pth",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset", help="Directory for training data"
    )
    parser.add_argument(
        "--precompute",
        type=bool,
        default=True,
        help="Use precomputed news embeddings for training",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default="./model_binaries",
        help="Path to PyTorch .pth file containing precomputed Embeddings (required only if precompute is True)",
    )
    args = parser.parse_args()
    main(args)
