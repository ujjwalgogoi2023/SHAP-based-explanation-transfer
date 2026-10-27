"""
Level-1 detection performance metrics (proposal Sec 4.1).

Computed per attack class and as macro averages, on both the source test
split and the target dataset (no retraining).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def false_positive_rate_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fpr = np.zeros(n_classes)
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr[c] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr


def compute_detection_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: list
) -> dict:
    n_classes = len(class_names)

    per_class = pd.DataFrame(
        {
            "precision": precision_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0),
            "recall": recall_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0),
            "f1": f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0),
            "fpr": false_positive_rate_per_class(y_true, y_pred, n_classes),
        },
        index=class_names,
    )

    try:
        y_true_onehot = np.eye(n_classes)[y_true]
        auc_roc = roc_auc_score(y_true_onehot, y_proba, average="macro", multi_class="ovr")
        auc_pr = average_precision_score(y_true_onehot, y_proba, average="macro")
    except ValueError:
        auc_roc, auc_pr = float("nan"), float("nan")

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fpr_macro": false_positive_rate_per_class(y_true, y_pred, n_classes).mean(),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_roc_macro": auc_roc,
        "auc_pr_macro": auc_pr,
    }

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def predict(model, loader, device) -> tuple:
    import torch

    model.eval()
    all_preds, all_proba, all_true = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
            all_preds.append(proba.argmax(axis=1))
            all_true.append(y.numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_preds),
        np.concatenate(all_proba),
    )
