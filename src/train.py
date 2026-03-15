"""Training script for logistic regression, MLP, and TabTransformer."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_logistic import train_logistic_baseline
from dataset import BankMarketingDataset
from metrics import compute_classification_metrics
from model_mlp import MLPBaseline
from model_tabtransformer import TabTransformer
from preprocess import preprocess_data
from utils import ensure_dir, get_device, load_json, load_yaml_config, set_seed


def build_dataloaders(processed_dir: str | Path, batch_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
    """Create dataloaders from saved embedding-formatted numpy arrays."""
    data = np.load(Path(processed_dir) / "embedding_data.npz")
    train_dataset = BankMarketingDataset(data["x_cat_train"], data["x_num_train"], data["y_train"])
    valid_dataset = BankMarketingDataset(data["x_cat_valid"], data["x_num_valid"], data["y_valid"])
    test_dataset = BankMarketingDataset(data["x_cat_test"], data["x_num_test"], data["y_test"])
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "valid": DataLoader(valid_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }
    return loaders, {key: data[key] for key in data.files}


def create_model(model_name: str, metadata: Dict[str, object], config: Dict[str, object]) -> nn.Module:
    """Instantiate a model from metadata and config."""
    cardinalities = list(metadata["categorical_cardinalities"].values())
    num_numeric_features = len(metadata["numerical_features"])

    if model_name == "mlp":
        return MLPBaseline(
            categorical_cardinalities=cardinalities,
            num_numeric_features=num_numeric_features,
            embedding_dim=config["model"]["embedding_dim"],
            dropout=config["model"]["dropout"],
        )
    if model_name == "tabtransformer":
        return TabTransformer(
            categorical_cardinalities=cardinalities,
            num_numeric_features=num_numeric_features,
            embedding_dim=config["model"]["embedding_dim"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            feedforward_dim=config["model"]["feedforward_dim"],
            dropout=config["model"]["dropout"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run a single train or eval epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for x_cat, x_num, y in tqdm(dataloader, leave=False):
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.detach().cpu().numpy())
        total_loss += float(loss.item()) * y.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(all_targets), np.concatenate(all_probs)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: Dict[str, object],
    model_name: str,
    config: Dict[str, object],
) -> None:
    """Persist the trained model checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_name": model_name,
            "model_config": config["model"],
        },
        path,
    )


def train_torch_model(
    model_name: str,
    config: Dict[str, object],
    include_duration: bool,
) -> Dict[str, float]:
    """Train a PyTorch model and return test metrics."""
    seed = config["training"]["seed"]
    set_seed(seed)
    processed_dir = Path(config["paths"]["processed_dir"])
    results_dir = ensure_dir(config["paths"]["results_dir"])
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    metadata = load_json(processed_dir / "metadata.json")
    loaders, _ = build_dataloaders(processed_dir, batch_size=config["training"]["batch_size"])
    device = get_device()

    model = create_model(model_name, metadata, config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_loss = float("inf")
    patience = config["training"]["patience"]
    patience_counter = 0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, y_train, p_train = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, y_val, p_val = run_epoch(model, loaders["valid"], criterion, device)

        train_metrics = compute_classification_metrics(y_train, p_train)
        val_metrics = compute_classification_metrics(y_val, p_val)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_record)
        print(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = checkpoint_dir / f"{model_name}_{'with' if include_duration else 'without'}_duration.pt"
    save_checkpoint(checkpoint_path, model, metadata, model_name, config)
    save_checkpoint(checkpoint_dir / "best_model.pt", model, metadata, model_name, config)

    test_loss, y_test, p_test = run_epoch(model, loaders["test"], criterion, device)
    test_metrics = compute_classification_metrics(y_test, p_test)
    test_metrics.update(
        {
            "model": model_name,
            "test_loss": test_loss,
            "include_duration": include_duration,
            "checkpoint": str(checkpoint_path),
        }
    )

    pd.DataFrame(history).to_csv(results_dir / f"{model_name}_history.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(results_dir / f"{model_name}_metrics.csv", index=False)
    return test_metrics


def prepare_processed_data(config: Dict[str, object], include_duration: bool) -> None:
    """Generate processed arrays from raw CSV."""
    preprocess_data(
        csv_path=config["paths"]["data_csv"],
        output_dir=config["paths"]["processed_dir"],
        seed=config["training"]["seed"],
        include_duration=include_duration,
    )


def train_single_model(model_name: str, config: Dict[str, object], include_duration: bool) -> Dict[str, float]:
    """Train one model according to the requested backend."""
    prepare_processed_data(config, include_duration)
    if model_name == "logistic":
        return train_logistic_baseline(
            processed_dir=config["paths"]["processed_dir"],
            results_dir=config["paths"]["results_dir"],
            include_duration=include_duration,
        )
    return train_torch_model(model_name, config, include_duration)


def run_all_experiments(config: Dict[str, object]) -> pd.DataFrame:
    """Run all model comparisons and duration ablation."""
    experiment_rows: List[Dict[str, float]] = []
    for include_duration in [True, False]:
        for model_name in ["logistic", "mlp", "tabtransformer"]:
            print(f"Running model={model_name}, include_duration={include_duration}")
            result = train_single_model(model_name, config, include_duration)
            experiment_rows.append(result)

    results_df = pd.DataFrame(experiment_rows)
    results_df.to_csv(Path(config["paths"]["results_dir"]) / "experiment_results.csv", index=False)
    print(results_df[["model", "include_duration", "accuracy", "f1", "roc_auc"]])
    return results_df


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train tabular bank marketing models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, choices=["logistic", "mlp", "tabtransformer", "all"], required=True)
    parser.add_argument("--include-duration", action="store_true")
    parser.add_argument("--exclude-duration", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    include_duration = config["experiment"]["include_duration"]
    if args.include_duration:
        include_duration = True
    if args.exclude_duration:
        include_duration = False

    ensure_dir(config["paths"]["results_dir"])
    ensure_dir(config["paths"]["checkpoint_dir"])
    ensure_dir(config["paths"]["processed_dir"])

    if args.model == "all":
        run_all_experiments(config)
    else:
        result = train_single_model(args.model, config, include_duration)
        summary = pd.DataFrame([result])
        summary.to_csv(Path(config["paths"]["results_dir"]) / "metrics.csv", index=False)
        print(summary[["model", "accuracy", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
