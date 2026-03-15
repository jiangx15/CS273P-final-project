"""Embedding-based MLP baseline."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """MLP baseline with embeddings for categorical features."""

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        num_numeric_features: int,
        embedding_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinalities]
        )

        input_dim = len(categorical_cardinalities) * embedding_dim + num_numeric_features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embeddings = [layer(x_cat[:, idx]) for idx, layer in enumerate(self.embedding_layers)]
        cat_features = torch.cat(embeddings, dim=1) if embeddings else x_num.new_zeros((x_num.size(0), 0))
        features = torch.cat([cat_features, x_num], dim=1)
        return self.classifier(features).squeeze(1)
