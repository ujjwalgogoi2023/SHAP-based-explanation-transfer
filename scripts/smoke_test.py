"""End-to-end smoke test on synthetic data.

Verifies shapes, imports, and metric math without needing the real datasets
downloaded. Run: python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FlowDataset, make_dataloaders
from src.data.schema import SHARED_FEATURES
from src.eval.baselines import random_permutation_baseline, summarise_across_seeds, wilcoxon_vs_random
from src.eval.metrics import compute_detection_metrics, predict
from src.explain.consistency import compute_all_class_consistency, explanation_consistency_score
from src.models.cnn_lstm import build_model
from src.training.train import get_device, train_model
from src.utils.seed import set_seed

CLASS_NAMES = ["Benign", "DoS/DDoS", "Reconnaissance/Probe", "Exploits"]


def make_centers(n_features, n_classes, seed=0):
    """Class centres must be SHARED across train/val/test, otherwise the label
    semantics differ per split and validation accuracy collapses to zero."""
    return np.random.default_rng(seed).normal(0, 2, size=(n_classes, n_features))


def synth(n, centers, seed):
    n_classes, n_features = centers.shape
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(0, 1, size=(n, n_features))
    return X.astype(np.float32), y.astype(np.int64)


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = 3
    cfg["train"]["device"] = "cpu"
    cfg["shap"]["background_samples"] = 50
    cfg["shap"]["explain_samples_per_class"] = 40

    n_features = len(SHARED_FEATURES)
    n_classes = len(CLASS_NAMES)
    device = get_device(cfg["train"]["device"])
    set_seed(42)

    centers = make_centers(n_features, n_classes, seed=0)
    X_train, y_train = synth(2000, centers, 1)
    X_val, y_val = synth(400, centers, 2)
    X_test, y_test = synth(400, centers, 3)
    # "target" dataset: same class semantics, shifted/scaled feature
    # distribution — mimicking the same attack captured in a different testbed
    X_tgt, y_tgt = synth(600, centers, 99)
    X_tgt = X_tgt * 1.4 + 0.7

    splits = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
              "X_test": X_test, "y_test": y_test}
    train_loader, val_loader, test_loader = make_dataloaders(splits, cfg["data"]["sequence_length"], cfg["train"]["batch_size"])

    print("[1/6] Building + training model ...")
    model = build_model(cfg, n_features, n_classes)
    model = train_model(model, train_loader, val_loader, y_train, n_classes, cfg, device)

    print("[2/6] Source detection metrics ...")
    yt, yp, ypr = predict(model, test_loader, device)
    src = compute_detection_metrics(yt, yp, ypr, CLASS_NAMES)
    print("   source accuracy:", round(src["summary"]["accuracy"], 4),
          "| macro F1:", round(src["summary"]["f1_macro"], 4),
          "| MCC:", round(src["summary"]["mcc"], 4))
    # The synthetic task is well-separated, so anything near chance (0.25)
    # means training or the label plumbing is broken, not that the task is hard.
    assert src["summary"]["accuracy"] > 0.70, f"training appears broken: {src['summary']}"

    print("[3/6] Target (transfer) detection metrics ...")
    tgt_loader = DataLoader(FlowDataset(X_tgt, y_tgt, cfg["data"]["sequence_length"]),
                            batch_size=cfg["train"]["batch_size"], shuffle=False)
    yt2, yp2, ypr2 = predict(model, tgt_loader, device)
    tgt = compute_detection_metrics(yt2, yp2, ypr2, CLASS_NAMES)
    print("   target accuracy:", round(tgt["summary"]["accuracy"], 4),
          "| macro F1:", round(tgt["summary"]["f1_macro"], 4))
    assert src["per_class"].shape == (n_classes, 4)
    assert src["confusion_matrix"].shape == (n_classes, n_classes)

    print("[4/6] SHAP rankings ...")
    from src.explain.shap_utils import compute_shap_rankings

    source_rankings = compute_shap_rankings(
        model, X_train, X_test, y_test, CLASS_NAMES, SHARED_FEATURES, device,
        cfg["shap"]["background_samples"], cfg["shap"]["explain_samples_per_class"], seed=42)
    target_rankings = compute_shap_rankings(
        model, X_train, X_tgt, y_tgt, CLASS_NAMES, SHARED_FEATURES, device,
        cfg["shap"]["background_samples"], cfg["shap"]["explain_samples_per_class"], seed=42)
    for cls in CLASS_NAMES:
        assert len(source_rankings[cls]["ranking"]) == n_features
        assert set(source_rankings[cls]["ranking"]) == set(SHARED_FEATURES)
    print("   top-5 source (Benign):", source_rankings["Benign"]["ranking"][:5])
    print("   top-5 target (Benign):", target_rankings["Benign"]["ranking"][:5])

    print("[5/6] Consistency metrics ...")
    consistency = compute_all_class_consistency(source_rankings, target_rankings, SHARED_FEATURES, cfg["shap"]["top_k"])
    for cls, m in consistency.items():
        print(f"   {cls:24s} rho={m['spearman_rho']:+.3f} tau={m['kendall_tau']:+.3f} "
              f"J@5={m['jaccard@5']:.3f} RBO={m['rbo']:.3f} ECS={m['explanation_consistency_score']:.3f}")
        assert -1.0 <= m["spearman_rho"] <= 1.0
        assert 0.0 <= m["jaccard@5"] <= 1.0
        assert 0.0 <= m["rbo"] <= 1.0
        assert 0.0 <= m["explanation_consistency_score"] <= 1.0

    print("[6/6] Baselines + Wilcoxon ...")
    obs, rand = [], []
    for cls in CLASS_NAMES:
        o = explanation_consistency_score(
            source_rankings[cls]["ranking"], target_rankings[cls]["ranking"],
            source_rankings[cls]["importance"], target_rankings[cls]["importance"], SHARED_FEATURES)
        r = random_permutation_baseline(
            source_rankings[cls]["ranking"], source_rankings[cls]["importance"], SHARED_FEATURES, seed=42)
        obs.append(o["explanation_consistency_score"])
        rand.append(r["explanation_consistency_score"])
    print("   observed:", summarise_across_seeds(obs))
    print("   random  :", summarise_across_seeds(rand))
    print("   wilcoxon:", wilcoxon_vs_random(obs, rand))

    # Identity check: a ranking compared against itself must score 1.0
    perfect = explanation_consistency_score(
        SHARED_FEATURES, SHARED_FEATURES,
        {f: 1.0 for f in SHARED_FEATURES}, {f: 1.0 for f in SHARED_FEATURES}, SHARED_FEATURES)
    assert abs(perfect["spearman_rho"] - 1.0) < 1e-9, perfect
    assert abs(perfect["jaccard@5"] - 1.0) < 1e-9, perfect
    assert abs(perfect["rbo"] - 1.0) < 1e-9, perfect
    assert abs(perfect["explanation_consistency_score"] - 1.0) < 1e-9, perfect

    # Reversed ranking must score near the floor
    reversed_ranking = list(reversed(SHARED_FEATURES))
    worst = explanation_consistency_score(
        SHARED_FEATURES, reversed_ranking,
        {f: 1.0 for f in SHARED_FEATURES}, {f: 1.0 for f in SHARED_FEATURES}, SHARED_FEATURES)
    assert abs(worst["spearman_rho"] + 1.0) < 1e-9, worst
    assert worst["explanation_consistency_score"] < perfect["explanation_consistency_score"]

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
