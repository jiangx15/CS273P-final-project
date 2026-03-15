"""Evaluation script for saved checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from baseline_logistic import train_logistic_baseline
from dataset import BankMarketingDataset
from metrics import compute_classification_metrics
from model_mlp import MLPBaseline
from model_tabtransformer import TabTransformer
from preprocess import preprocess_data
from utils import ensure_dir, get_device, load_yaml_config, set_seed


def load_embedding_test_data(processed_dir: str | Path, batch_size: int) -> DataLoader:
    """Load the embedding-based test split."""
    data = np.load(Path(processed_dir) / "embedding_data.npz")
    dataset = BankMarketingDataset(data["x_cat_test"], data["x_num_test"], data["y_test"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def rebuild_model(checkpoint: Dict[str, object]) -> nn.Module:
    """Recreate a torch model from checkpoint metadata."""
    metadata = checkpoint["metadata"]
    model_name = checkpoint["model_name"]
    model_config = checkpoint["model_config"]
    cardinalities = list(metadata["categorical_cardinalities"].values())
    num_numeric = len(metadata["numerical_features"])

    if model_name == "mlp":
        model = MLPBaseline(cardinalities, num_numeric, model_config["embedding_dim"], model_config["dropout"])
    elif model_name == "tabtransformer":
        model = TabTransformer(
            cardinalities,
            num_numeric,
            model_config["embedding_dim"],
            model_config["nhead"],
            model_config["num_layers"],
            model_config["feedforward_dim"],
            model_config["dropout"],
        )
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_torch_checkpoint(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    results_dir: str | Path,
    batch_size: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a saved PyTorch checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = rebuild_model(checkpoint).to(get_device())
    model.eval()
    dataloader = load_embedding_test_data(processed_dir, batch_size)
    device = get_device()
    metadata = checkpoint["metadata"]
    expected_num_cat = len(metadata["categorical_cardinalities"])
    expected_num_num = len(metadata["numerical_features"])
    sample_x_cat, sample_x_num, _ = dataloader.dataset[0]

    if sample_x_cat.shape[0] != expected_num_cat or sample_x_num.shape[0] != expected_num_num:
        raise ValueError(
            "Processed evaluation data does not match the checkpoint feature layout. "
            f"Checkpoint expects {expected_num_cat} categorical and {expected_num_num} numerical features, "
            f"but processed data has {sample_x_cat.shape[0]} categorical and {sample_x_num.shape[0]} numerical features. "
            "Regenerate processed data with the same `include_duration` setting used during training before evaluation."
        )

    y_true_list = []
    y_prob_list = []
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            logits = model(x_cat.to(device), x_num.to(device))
            y_prob_list.append(torch.sigmoid(logits).cpu().numpy())
            y_true_list.append(y.numpy())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    metrics = compute_classification_metrics(y_true, y_prob)
    metrics["model"] = checkpoint["model_name"]
    save_evaluation_artifacts(y_true, y_prob, metrics, results_dir)
    return metrics, y_true, y_prob


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    results_dir: str | Path,
) -> None:
    """Save ROC curve and confusion matrix."""
    results_path = ensure_dir(results_dir)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path / "roc_curve.png", dpi=200)
    plt.close()

    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_path / "confusion_matrix.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--model", type=str, choices=["logistic", "mlp", "tabtransformer"], default=None)
    parser.add_argument("--include-duration", action="store_true")
    parser.add_argument("--exclude-duration", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["training"]["seed"])
    include_duration = config["experiment"]["include_duration"]
    if args.include_duration:
        include_duration = True
    if args.exclude_duration:
        include_duration = False

    preprocess_data(
        csv_path=config["paths"]["data_csv"],
        output_dir=config["paths"]["processed_dir"],
        seed=config["training"]["seed"],
        include_duration=include_duration,
    )

    if args.model == "logistic":
        metrics = train_logistic_baseline(
            processed_dir=config["paths"]["processed_dir"],
            results_dir=config["paths"]["results_dir"],
            include_duration=include_duration,
        )
        print(pd.DataFrame([metrics]))
        return

    metrics, _, _ = evaluate_torch_checkpoint(
        checkpoint_path=args.checkpoint,
        processed_dir=config["paths"]["processed_dir"],
        results_dir=config["paths"]["results_dir"],
        batch_size=config["training"]["batch_size"],
    )
    print(pd.DataFrame([metrics]))


if __name__ == "__main__":
    main()
