"""Logistic regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from metrics import compute_classification_metrics
from utils import ensure_dir


def train_logistic_baseline(
    processed_dir: str | Path,
    results_dir: str | Path,
    include_duration: bool = True,
) -> Dict[str, float]:
    """Train and evaluate the sklearn logistic regression baseline."""
    data = np.load(Path(processed_dir) / "logistic_data.npz")
    model = LogisticRegression(max_iter=1000)
    model.fit(data["x_train"], data["y_train"].astype(int))

    y_prob = model.predict_proba(data["x_test"])[:, 1]
    metrics = compute_classification_metrics(data["y_test"], y_prob)
    metrics["model"] = "logistic"
    metrics["include_duration"] = include_duration

    ensure_dir(results_dir)
    pd.DataFrame([metrics]).to_csv(Path(results_dir) / "logistic_metrics.csv", index=False)
    return metrics
