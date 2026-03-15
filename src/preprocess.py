"""Data preprocessing for the bank marketing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import ensure_dir, load_yaml_config, save_json, set_seed


def read_bank_marketing_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read the bank marketing CSV with delimiter auto-detection."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [column.strip().strip('"') for column in df.columns]

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip().str.strip('"')

    return df


def detect_feature_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    """Infer categorical and numerical columns from pandas dtypes."""
    feature_df = df.drop(columns=[target_column])
    categorical_features = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_features = [column for column in feature_df.columns if column not in categorical_features]
    return categorical_features, numerical_features


def encode_target(series: pd.Series) -> np.ndarray:
    """Map yes/no labels to 1/0."""
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).to_numpy(dtype=np.int64)


def stratified_split(
    df: pd.DataFrame,
    target: np.ndarray,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into 70/15/15 train/validation/test."""
    train_df, temp_df, y_train, y_temp = train_test_split(
        df,
        target,
        test_size=0.30,
        stratify=target,
        random_state=seed,
    )
    valid_df, test_df, y_valid, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return train_df, valid_df, test_df, y_train, y_valid, y_test


def label_encode_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, Dict[str, int]]]:
    """Label-encode categorical features based on the train split."""
    encoders: Dict[str, Dict[str, int]] = {}
    cardinalities: Dict[str, int] = {}

    def transform_frame(frame: pd.DataFrame) -> np.ndarray:
        encoded_columns: List[np.ndarray] = []
        for column in categorical_features:
            mapping = encoders[column]
            values = frame[column].astype(str).fillna("__missing__")
            encoded = values.map(lambda item: mapping.get(item, mapping["__unknown__"])).to_numpy(dtype=np.int64)
            encoded_columns.append(encoded)
        if not encoded_columns:
            return np.zeros((len(frame), 0), dtype=np.int64)
        return np.column_stack(encoded_columns)

    for column in categorical_features:
        train_values = train_df[column].astype(str).fillna("__missing__")
        unique_values = sorted(train_values.unique().tolist())
        mapping = {"__unknown__": 0}
        mapping.update({value: index + 1 for index, value in enumerate(unique_values)})
        encoders[column] = mapping
        cardinalities[column] = len(mapping)

    return (
        transform_frame(train_df),
        transform_frame(valid_df),
        transform_frame(test_df),
        cardinalities,
        encoders,
    )


def scale_numeric_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize numerical features using train statistics."""
    if not numerical_features:
        empty = np.zeros((len(train_df), 0), dtype=np.float32)
        return empty, np.zeros((len(valid_df), 0), dtype=np.float32), np.zeros((len(test_df), 0), dtype=np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[numerical_features]).astype(np.float32)
    x_valid = scaler.transform(valid_df[numerical_features]).astype(np.float32)
    x_test = scaler.transform(test_df[numerical_features]).astype(np.float32)
    return x_train, x_valid, x_test


def one_hot_encode_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a one-hot encoder for the logistic regression baseline."""
    if not categorical_features:
        empty = np.zeros((len(train_df), 0), dtype=np.float32)
        return empty, np.zeros((len(valid_df), 0), dtype=np.float32), np.zeros((len(test_df), 0), dtype=np.float32)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    x_train = encoder.fit_transform(train_df[categorical_features]).astype(np.float32)
    x_valid = encoder.transform(valid_df[categorical_features]).astype(np.float32)
    x_test = encoder.transform(test_df[categorical_features]).astype(np.float32)
    return x_train, x_valid, x_test


def preprocess_data(
    csv_path: str | Path,
    output_dir: str | Path,
    target_column: str = "y",
    seed: int = 42,
    include_duration: bool = True,
) -> Dict[str, object]:
    """Run the full preprocessing pipeline and persist processed arrays."""
    set_seed(seed)
    df = read_bank_marketing_csv(csv_path)

    if not include_duration and "duration" in df.columns:
        df = df.drop(columns=["duration"])

    categorical_features, numerical_features = detect_feature_types(df, target_column)
    target = encode_target(df[target_column])
    feature_df = df.drop(columns=[target_column])

    train_df, valid_df, test_df, y_train, y_valid, y_test = stratified_split(feature_df, target, seed)
    x_cat_train, x_cat_valid, x_cat_test, cardinalities, encoders = label_encode_splits(
        train_df, valid_df, test_df, categorical_features
    )
    x_num_train, x_num_valid, x_num_test = scale_numeric_splits(train_df, valid_df, test_df, numerical_features)
    x_oh_train, x_oh_valid, x_oh_test = one_hot_encode_splits(train_df, valid_df, test_df, categorical_features)

    output_path = ensure_dir(output_dir)
    np.savez_compressed(
        output_path / "embedding_data.npz",
        x_cat_train=x_cat_train,
        x_cat_valid=x_cat_valid,
        x_cat_test=x_cat_test,
        x_num_train=x_num_train,
        x_num_valid=x_num_valid,
        x_num_test=x_num_test,
        y_train=y_train.astype(np.float32),
        y_valid=y_valid.astype(np.float32),
        y_test=y_test.astype(np.float32),
    )
    np.savez_compressed(
        output_path / "logistic_data.npz",
        x_train=np.concatenate([x_oh_train, x_num_train], axis=1),
        x_valid=np.concatenate([x_oh_valid, x_num_valid], axis=1),
        x_test=np.concatenate([x_oh_test, x_num_test], axis=1),
        y_train=y_train.astype(np.float32),
        y_valid=y_valid.astype(np.float32),
        y_test=y_test.astype(np.float32),
    )

    metadata = {
        "target_column": target_column,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "categorical_cardinalities": cardinalities,
        "categorical_encoders": encoders,
        "include_duration": include_duration,
        "seed": seed,
        "num_train": int(len(train_df)),
        "num_valid": int(len(valid_df)),
        "num_test": int(len(test_df)),
    }
    save_json(metadata, output_path / "metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preprocess the bank marketing dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
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

    data_path = args.data_path or config["paths"]["data_csv"]
    output_dir = args.output_dir or config["paths"]["processed_dir"]
    seed = args.seed if args.seed is not None else config["training"]["seed"]

    metadata = preprocess_data(
        csv_path=data_path,
        output_dir=output_dir,
        seed=seed,
        include_duration=include_duration,
    )
    print("Preprocessing complete.")
    print(f"Categorical features: {metadata['categorical_features']}")
    print(f"Numerical features: {metadata['numerical_features']}")


if __name__ == "__main__":
    main()
