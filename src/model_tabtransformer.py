"""TabTransformer model implementation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TabTransformer(nn.Module):
    """TabTransformer for tabular binary classification."""

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        num_numeric_features: int,
        embedding_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_cardinalities]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        input_dim = len(categorical_cardinalities) * embedding_dim + num_numeric_features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embeddings = [layer(x_cat[:, idx]).unsqueeze(1) for idx, layer in enumerate(self.embedding_layers)]
        if embeddings:
            x = torch.cat(embeddings, dim=1)
            x = self.transformer(x)
            cat_features = x.flatten(start_dim=1)
        else:
            cat_features = x_num.new_zeros((x_num.size(0), 0))
        features = torch.cat([cat_features, x_num], dim=1)
        return self.classifier(features).squeeze(1)
