"""Small PyTorch helpers for tabular deep learning experiments.

The notebooks keep the story and analysis, while this module keeps repeated
training code tidy enough to reuse for both regression and classification.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TaskName = Literal["regression", "binary"]


def seed_everything(seed: int = 42) -> None:
    """Make common random sources deterministic for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy_dense(matrix) -> np.ndarray:
    """Convert scipy sparse/pandas/numpy inputs into float32 dense arrays."""
    if hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def add_numeric_summary_features(df):
    """Add row-level summary features for numeric tabular dataframes."""
    result = df.copy()
    numeric = result.select_dtypes(include=["number"])
    if numeric.empty:
        return result
    result["row_mean"] = numeric.mean(axis=1)
    result["row_std"] = numeric.std(axis=1).fillna(0)
    result["row_min"] = numeric.min(axis=1)
    result["row_max"] = numeric.max(axis=1)
    result["row_missing_count"] = numeric.isna().sum(axis=1)
    return result


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        TabularDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class MLP(nn.Module):
    """Configurable fully connected network for tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
        batch_norm: bool = True,
        output_dim: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1, batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(width, width)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width)])
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        self.block = nn.Sequential(*layers)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualMLP(nn.Module):
    """Residual MLP that keeps a stable hidden width across blocks."""

    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU())
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(width, dropout=dropout, batch_norm=batch_norm)
                for _ in range(n_blocks)
            ]
        )
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_valid_loss: float
    best_epoch: int


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            pred = model(xb)
            loss = criterion(pred, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = xb.shape[0]
        running_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size

    return running_loss / max(total, 1)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    task: TaskName,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 30,
    patience: int = 5,
    device: torch.device | None = None,
    pos_weight: float | None = None,
) -> TrainResult:
    """Train with early stopping on validation loss."""
    device = device or get_device()
    model.to(device)

    if task == "binary":
        weight_tensor = None
        if pos_weight is not None:
            weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(model, train_loader, criterion, device, optimizer)
        valid_loss = _epoch_loss(model, valid_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(history=history, best_valid_loss=best_loss, best_epoch=best_epoch)


@torch.no_grad()
def predict_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_device()
    model.to(device)
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).detach().cpu().numpy())
    return np.vstack(preds).reshape(-1)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))
