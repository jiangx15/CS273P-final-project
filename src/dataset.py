"""Dataset definitions for bank marketing experiments."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BankMarketingDataset(Dataset):
    """Simple tabular dataset for mixed categorical and numerical inputs."""

    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> None:
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_cat[index], self.x_num[index], self.y[index]
