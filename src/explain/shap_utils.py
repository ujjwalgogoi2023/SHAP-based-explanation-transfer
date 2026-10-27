"""
SHAP wrapper producing per-class feature-importance rankings
(proposal Sec 4.2 / Sec 2 Phase 5 — the novel contribution).

For each unified attack class, SHAP values are computed for the model's
class-c logit over samples whose true label is c, then averaged (mean |SHAP|)
per feature to produce a ranking E_source or E_target.

IMPORTANT — choice of explainer
-------------------------------
`shap.DeepExplainer` (DeepSHAP) does NOT implement the DeepLIFT rescale rule
for `nn.LSTM`; it emits `UserWarning: unrecognized nn.Module: LSTM` and falls
back to a plain gradient for that module. Attributions for a CNN-LSTM are
therefore not true DeepSHAP values and the additivity property does not hold.

Since the whole contribution of this study rests on the attributions being
correct, the default here is `shap.GradientExplainer` (Expected Gradients),
which is architecture-agnostic and handles recurrent layers correctly.
`DeepExplainer` remains selectable via `explainer_type="deep"` so the two can
be compared — a useful robustness appendix, since a finding that survives both
attribution methods is considerably stronger.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import shap
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install shap") from e


def _sample_background(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def _to_seq_tensor(X: np.ndarray, device) -> torch.Tensor:
    # (n, n_features) -> (n, seq_len=1, n_features)
    return torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)


def compute_shap_rankings(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    class_names: list,
    feature_names: list,
    device,
    background_samples: int = 200,
    explain_samples_per_class: int = 500,
    seed: int = 42,
    explainer_type: str = "gradient",
) -> dict:
    """Returns {class_name: {"ranking": [feature,...] sorted desc,
                              "importance": {feature: mean_abs_shap}}}

    explainer_type: "gradient" (default, correct for LSTM) or "deep".
    """
    model.eval().to(device)

    bg = _sample_background(X_background, background_samples, seed)
    bg_tensor = _to_seq_tensor(bg, device)

    if explainer_type == "gradient":
        explainer = shap.GradientExplainer(model, bg_tensor)
    elif explainer_type == "deep":
        explainer = shap.DeepExplainer(model, bg_tensor)
    else:
        raise ValueError(f"explainer_type must be 'gradient' or 'deep', got {explainer_type!r}")

    results = {}
    rng = np.random.default_rng(seed)

    for class_idx, class_name in enumerate(class_names):
        class_mask = y_explain == class_idx
        class_X = X_explain[class_mask]
        if len(class_X) == 0:
            results[class_name] = {"ranking": list(feature_names), "importance": {f: 0.0 for f in feature_names}}
            continue

        n = min(explain_samples_per_class, len(class_X))
        idx = rng.choice(len(class_X), size=n, replace=False)
        sample = class_X[idx]
        sample_tensor = _to_seq_tensor(sample, device)

        # check_additivity is a DeepExplainer-only kwarg; GradientExplainer
        # (Expected Gradients) does not accept it.
        if explainer_type == "deep":
            shap_values = explainer.shap_values(sample_tensor, check_additivity=False)
        else:
            shap_values = explainer.shap_values(sample_tensor)
        class_shap = _extract_class_shap(shap_values, class_idx)  # (n, seq_len, n_features)
        class_shap = np.asarray(class_shap).reshape(n, -1, len(feature_names))
        mean_abs = np.abs(class_shap).mean(axis=(0, 1))  # (n_features,)

        importance = dict(zip(feature_names, mean_abs.tolist()))
        ranking = sorted(feature_names, key=lambda f: importance[f], reverse=True)
        results[class_name] = {"ranking": ranking, "importance": importance}

    return results


def _extract_class_shap(shap_values, class_idx: int):
    """Normalise across shap library versions: some return a list of
    per-class arrays, others a single array with a trailing class axis."""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    arr = np.asarray(shap_values)
    if arr.ndim == 4:  # (n, seq_len, n_features, n_classes)
        return arr[..., class_idx]
    return arr
