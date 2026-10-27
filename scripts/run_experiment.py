"""Headless end-to-end run: train all seeds, extract SHAP, score consistency.

Equivalent to notebooks 02-04 but runnable from the CLI so results can be
generated without babysitting a kernel.

    python scripts/run_experiment.py --config configs/default.yaml
"""
import argparse
import itertools
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FlowDataset, make_dataloaders
from src.data.schema import CLASS_CORRESPONDENCE, SHARED_FEATURES
from src.eval.baselines import random_permutation_baseline, same_dataset_upper_bound, summarise_across_seeds, wilcoxon_vs_random
from src.eval.metrics import compute_detection_metrics, predict
from src.explain.consistency import compute_all_class_consistency, explanation_consistency_score
from src.explain.shap_utils import compute_shap_rankings
from src.models.cnn_lstm import build_model
from src.training.train import get_device, train_model
from src.utils.seed import set_seed


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["data"]["processed_dir"], "class_names.yaml")) as f:
        class_names = yaml.safe_load(f)

    src_name = cfg["data"]["source_dataset"]
    tgt_name = cfg["data"]["target_dataset"]
    source = np.load(os.path.join(cfg["data"]["processed_dir"], f"{src_name}_source.npz"))
    target = np.load(os.path.join(cfg["data"]["processed_dir"], f"{tgt_name}_target.npz"))

    n_features, n_classes = source["X_train"].shape[1], len(class_names)
    device = get_device(cfg["train"]["device"])
    seq_len = cfg["data"]["sequence_length"]
    seeds = cfg["seeds"][: cfg["n_seeds"]]

    for d in [cfg["output"]["checkpoints_dir"], cfg["output"]["tables_dir"], cfg["output"]["figures_dir"]]:
        os.makedirs(d, exist_ok=True)

    print(f"device={device}  features={n_features}  classes={class_names}  seeds={seeds}\n")

    detection_rows, rankings_by_seed = [], {}

    for seed in seeds:
        print(f"{'=' * 70}\nseed {seed}\n{'=' * 70}")
        set_seed(seed)

        splits = {k: source[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}
        train_loader, val_loader, test_loader = make_dataloaders(splits, seq_len, cfg["train"]["batch_size"])

        model = build_model(cfg, n_features, n_classes)
        model = train_model(model, train_loader, val_loader, source["y_train"], n_classes, cfg, device)
        torch.save(model.state_dict(), os.path.join(cfg["output"]["checkpoints_dir"], f"cnn_lstm_seed{seed}.pt"))

        yt, yp, ypr = predict(model, test_loader, device)
        src_m = compute_detection_metrics(yt, yp, ypr, class_names)

        tgt_loader = DataLoader(FlowDataset(target["X"], target["y"], seq_len),
                                batch_size=cfg["train"]["batch_size"], shuffle=False)
        yt2, yp2, ypr2 = predict(model, tgt_loader, device)
        tgt_m = compute_detection_metrics(yt2, yp2, ypr2, class_names)

        print(f"  source  acc={src_m['summary']['accuracy']:.4f} macroF1={src_m['summary']['f1_macro']:.4f} MCC={src_m['summary']['mcc']:.4f}")
        print(f"  target  acc={tgt_m['summary']['accuracy']:.4f} macroF1={tgt_m['summary']['f1_macro']:.4f} MCC={tgt_m['summary']['mcc']:.4f}")

        detection_rows.append({"seed": seed, "split": "source", **src_m["summary"]})
        detection_rows.append({"seed": seed, "split": "target", **tgt_m["summary"]})

        print("  extracting SHAP rankings ...")
        kw = dict(class_names=class_names, feature_names=SHARED_FEATURES, device=device,
                  background_samples=cfg["shap"]["background_samples"],
                  explain_samples_per_class=cfg["shap"]["explain_samples_per_class"],
                  seed=seed, explainer_type=cfg["shap"]["explainer_type"])
        src_rank = compute_shap_rankings(model, source["X_train"], source["X_test"], source["y_test"], **kw)
        tgt_rank = compute_shap_rankings(model, source["X_train"], target["X"], target["y"], **kw)

        # Condition 3: a model trained NATIVELY on the target. Needed because
        # the transferred model can sit at chance on the target, and SHAP
        # values of a chance-level model explain nothing. Comparing
        # source-model-on-source vs target-model-on-target is the like-for-like
        # test of whether the same features matter for the same attack.
        print("  training native target model ...")
        set_seed(seed)
        tgt_splits = {k: target[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}
        tl, vl, tel = make_dataloaders(tgt_splits, seq_len, cfg["train"]["batch_size"])
        tgt_model = build_model(cfg, n_features, n_classes)
        tgt_model = train_model(tgt_model, tl, vl, target["y_train"], n_classes, cfg, device)
        torch.save(tgt_model.state_dict(),
                   os.path.join(cfg["output"]["checkpoints_dir"], f"cnn_lstm_native_target_seed{seed}.pt"))

        yt3, yp3, ypr3 = predict(tgt_model, tel, device)
        nat_m = compute_detection_metrics(yt3, yp3, ypr3, class_names)
        print(f"  native target  acc={nat_m['summary']['accuracy']:.4f} macroF1={nat_m['summary']['f1_macro']:.4f} MCC={nat_m['summary']['mcc']:.4f}")
        detection_rows.append({"seed": seed, "split": "native_target", **nat_m["summary"]})

        nat_rank = compute_shap_rankings(tgt_model, target["X_train"], target["X_test"], target["y_test"], **kw)
        rankings_by_seed[seed] = {"source": src_rank, "target": tgt_rank, "native_target": nat_rank}
        with open(os.path.join(cfg["output"]["tables_dir"], f"shap_rankings_seed{seed}.pkl"), "wb") as f:
            pickle.dump(rankings_by_seed[seed], f)

    det_df = pd.DataFrame(detection_rows)
    det_df.to_csv(os.path.join(cfg["output"]["tables_dir"], "detection_per_seed.csv"), index=False)

    print(f"\n{'=' * 70}\nDETECTION PERFORMANCE (mean +/- std over {len(seeds)} seeds)\n{'=' * 70}")
    det_summary = det_df.drop(columns=["seed"]).groupby("split").agg(["mean", "std"])
    for metric in ["accuracy", "f1_macro", "mcc", "auc_pr_macro"]:
        s, t = det_summary.loc["source", metric], det_summary.loc["target", metric]
        print(f"  {metric:16s} source {s['mean']:.4f}+/-{s['std']:.4f}   target {t['mean']:.4f}+/-{t['std']:.4f}")
    det_summary.to_csv(os.path.join(cfg["output"]["tables_dir"], "detection_summary.csv"))

    # ---- consistency -------------------------------------------------------
    rows = []
    for seed in seeds:
        # transfer: same (source-trained) model, explained on each dataset
        for cls, m in compute_all_class_consistency(
            rankings_by_seed[seed]["source"], rankings_by_seed[seed]["target"],
            SHARED_FEATURES, cfg["shap"]["top_k"]
        ).items():
            rows.append({"seed": seed, "condition": "transfer", "class": cls,
                         "correspondence": CLASS_CORRESPONDENCE.get(cls, "?"), **m})
        # native: two competent models, one per dataset
        for cls, m in compute_all_class_consistency(
            rankings_by_seed[seed]["source"], rankings_by_seed[seed]["native_target"],
            SHARED_FEATURES, cfg["shap"]["top_k"]
        ).items():
            rows.append({"seed": seed, "condition": "native", "class": cls,
                         "correspondence": CLASS_CORRESPONDENCE.get(cls, "?"), **m})
    cons_df = pd.DataFrame(rows)
    cons_df.to_csv(os.path.join(cfg["output"]["tables_dir"], "consistency_per_seed.csv"), index=False)

    print(f"\n{'=' * 70}\nCROSS-DATASET EXPLANATION CONSISTENCY\n{'=' * 70}")
    print("  transfer = one source-trained model explained on both datasets")
    print("  native   = source-trained model vs target-trained model (both competent)\n")
    per_class = cons_df.groupby(["condition", "correspondence", "class"]).agg(
        rho=("spearman_rho", "mean"), rho_sd=("spearman_rho", "std"),
        ecs=("explanation_consistency_score", "mean"), ecs_sd=("explanation_consistency_score", "std"))
    print(per_class.round(4).to_string())
    per_class.to_csv(os.path.join(cfg["output"]["tables_dir"], "consistency_summary.csv"))

    # ---- anchors -----------------------------------------------------------
    ub_rows = []
    for a, b in itertools.combinations(seeds, 2):
        for cls, m in same_dataset_upper_bound(
            rankings_by_seed[a]["source"], rankings_by_seed[b]["source"], SHARED_FEATURES, cfg["shap"]["top_k"]
        ).items():
            ub_rows.append({"pair": f"{a}v{b}", "class": cls, **m})
    ub_df = pd.DataFrame(ub_rows)
    ub_df.to_csv(os.path.join(cfg["output"]["tables_dir"], "upper_bound.csv"), index=False)

    rnd = []
    obs = {"transfer": [], "native": []}
    for seed in seeds:
        for cls in class_names:
            s = rankings_by_seed[seed]["source"][cls]
            rnd.append(random_permutation_baseline(s["ranking"], s["importance"], SHARED_FEATURES, seed,
                                                   cfg["shap"]["top_k"])
                       ["explanation_consistency_score"])
            for cond, key in [("transfer", "target"), ("native", "native_target")]:
                o = rankings_by_seed[seed][key][cls]
                obs[cond].append(explanation_consistency_score(
                    s["ranking"], o["ranking"], s["importance"], o["importance"],
                    SHARED_FEATURES, cfg["shap"]["top_k"])["explanation_consistency_score"])

    ub_mean, lb_mean = float(ub_df["explanation_consistency_score"].mean()), float(np.mean(rnd))

    print(f"\n{'=' * 70}\nANCHORED INTERPRETATION\n{'=' * 70}")
    print(f"  upper bound (same dataset, reseeded) ECS = {ub_mean:.4f}")
    print(f"  lower bound (random permutation)     ECS = {lb_mean:.4f}\n")

    headline = {"upper_bound_ecs": ub_mean, "lower_bound_ecs": lb_mean, "conditions": {}}
    for cond in ["transfer", "native"]:
        m = float(np.mean(obs[cond]))
        norm = (m - lb_mean) / (ub_mean - lb_mean) if ub_mean > lb_mean else float("nan")
        w = wilcoxon_vs_random(obs[cond], rnd)
        print(f"  [{cond}] observed ECS = {m:.4f}   normalised = {norm:.3f}")
        print(f"           wilcoxon vs random: p={w['p_value']:.4g} significant={w['significant']}")
        headline["conditions"][cond] = {"observed_ecs": m, "normalised_consistency": float(norm), "wilcoxon": w}

    print("\n  normalised: 1.0 = transfers like a same-dataset reseed, 0.0 = random")
    print("  The 'native' row is the interpretable one -- 'transfer' is confounded")
    print("  by how badly the source model performs on the target.")

    headline["detection"] = {f"{s}_f1_macro": float(det_df[det_df.split == s].f1_macro.mean())
                             for s in ["source", "target", "native_target"]}
    headline["detection"]["target_mcc"] = float(det_df[det_df.split == "target"].mcc.mean())
    with open(os.path.join(cfg["output"]["tables_dir"], "headline.yaml"), "w") as f:
        yaml.safe_dump(headline, f)
    print(f"\nAll tables written to {cfg['output']['tables_dir']}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    main(p.parse_args().config)
