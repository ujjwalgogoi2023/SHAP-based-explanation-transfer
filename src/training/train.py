"""
Training loop for the CNN-LSTM classifier with early stopping and
class-weighted loss (attack classes are heavily imbalanced).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loss(y_train: np.ndarray, n_classes: int, weighted: bool, device: torch.device) -> nn.Module:
    if not weighted:
        return nn.CrossEntropyLoss()
    classes = np.arange(n_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights_t)


def run_epoch(model, loader: DataLoader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, n_correct, n_total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n_correct += (logits.argmax(dim=1) == y).sum().item()
            n_total += len(y)
    return total_loss / n_total, n_correct / n_total


def train_model(model, train_loader, val_loader, y_train, n_classes, cfg, device):
    criterion = make_loss(y_train, n_classes, cfg["train"]["class_weighted_loss"], device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    model.to(device)

    best_val_loss = float("inf")
    best_state = None
    patience = cfg["train"]["early_stopping_patience"]
    bad_epochs = 0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"epoch {epoch + 1}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
