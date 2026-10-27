"""Re-run the explanation half of the study with a different SHAP explainer.

    python scripts/run_explainer_variant.py --explainer deep

Loads the checkpoints `run_experiment.py` already wrote, extracts per-class
rankings with the chosen explainer, and recomputes consistency, both anchors
and the significance test into a separate tables directory. Nothing is
retrained: the models are identical to the main run, so any difference in the
results is attributable to the attribution method and to nothing else. That is
the whole point of the comparison, and it is also why this is a separate script
rather than a flag on `run_experiment.py` -- that script's job is to train from
scratch, and teaching it to silently skip training would be a trap.

Detection metrics do not depend on the explainer and are carried over from the
main run's `detection_per_seed.csv`.

On `--explainer deep`: `shap.DeepExplainer` does not implement the DeepLIFT
rescale rule for `nn.LSTM`. It emits `UserWarning: unrecognized nn.Module:
LSTM` and falls back to a plain gradient for that layer, so its output is not a
valid DeepSHAP value for this architecture. The warnings are counted and
recorded in `provenance.yaml` rather than suppressed, because the count is part
of the finding.
"""
import argparse
import itertools
import os
import pickle
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.schema import CLASS_CORRESPONDENCE, SHARED_FEATURES
from src.eval.baselines import random_permutation_baseline, same_dataset_upper_bound, wilcoxon_vs_random
from src.explain.consistency import compute_all_class_consistency, explanation_consistency_score
from src.explain.shap_utils import compute_shap_rankings
from src.models.cnn_lstm import build_model
from src.training.train import get_device
from src.utils.seed import set_seed

LSTM_WARNING = "unrecognized nn.Module: LSTM"


def main(config_path: str, explainer: str, out_dir: str | None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["data"]["processed_dir"], "class_names.yaml")) as f:
        class_names = yaml.safe_load(f)

    out_dir = out_dir or f"{cfg['output']['results_dir']}/tables_{explainer}"
    os.makedirs(out_dir, exist_ok=True)

    src_name, tgt_name = cfg["data"]["source_dataset"], cfg["data"]["target_dataset"]
    processed = cfg["data"]["processed_dir"]
    source = np.load(os.path.join(processed, f"{src_name}_source.npz"))
    target = np.load(os.path.join(processed, f"{tgt_name}_target.npz"))

    seeds = cfg["seeds"][: cfg["n_seeds"]]
    top_k = cfg["shap"]["top_k"]
    device = get_device(cfg["train"]["device"])
    n_features, n_classes = source["X_train"].shape[1], len(class_names)
    ckpt_dir = cfg["output"]["checkpoints_dir"]

    print(f"explainer={explainer}  device={device}  seeds={seeds}  -> {out_dir}/")

    def load(template, seed):
        path = os.path.join(ckpt_dir, template.format(seed=seed))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing -- run scripts/run_experiment.py first")
        model = build_model(cfg, n_features, n_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device).eval()

    kw = dict(class_names=class_names, feature_names=SHARED_FEATURES, device=device,
              background_samples=cfg["shap"]["background_samples"],
              explain_samples_per_class=cfg["shap"]["explain_samples_per_class"],
              explainer_type=explainer)

    rankings_by_seed = {}
    lstm_warnings = 0
    other_warnings: dict[str, int] = {}
    t_start = time.time()

    for seed in seeds:
        print(f"\n{'=' * 70}\nseed {seed}\n{'=' * 70}")
        set_seed(seed)
        src_model = load("cnn_lstm_seed{seed}.pt", seed)
        tgt_model = load("cnn_lstm_native_target_seed{seed}.pt", seed)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.time()
            # Identical calls to run_experiment.py, so only the explainer differs:
            # the target ranking keeps its background from the SOURCE training
            # split, because it explains the source-trained model.
            src_rank = compute_shap_rankings(src_model, source["X_train"], source["X_test"],
                                             source["y_test"], seed=seed, **kw)
            tgt_rank = compute_shap_rankings(src_model, source["X_train"], target["X"],
                                             target["y"], seed=seed, **kw)
            nat_rank = compute_shap_rankings(tgt_model, target["X_train"], target["X_test"],
                                             target["y_test"], seed=seed, **kw)
            dt = time.time() - t0

        for w in caught:
            msg = str(w.message)
            if LSTM_WARNING in msg:
                lstm_warnings += 1
            else:
                key = f"{w.category.__name__}: {msg[:120]}"
                other_warnings[key] = other_warnings.get(key, 0) + 1

        print(f"  {dt:.0f}s   LSTM warnings so far: {lstm_warnings}")
        for cls in class_names:
            print(f"    {cls:22s} top3={src_rank[cls]['ranking'][:3]} -> {nat_rank[cls]['ranking'][:3]}")

        rankings_by_seed[seed] = {"source": src_rank, "target": tgt_rank, "native_target": nat_rank}
        with open(os.path.join(out_dir, f"shap_rankings_seed{seed}.pkl"), "wb") as f:
            pickle.dump(rankings_by_seed[seed], f)

    # ---- consistency -------------------------------------------------------
    rows = []
    for seed in seeds:
        for cond, key in [("transfer", "target"), ("native", "native_target")]:
            for cls, m in compute_all_class_consistency(
                rankings_by_seed[seed]["source"], rankings_by_seed[seed][key],
                SHARED_FEATURES, top_k
            ).items():
                rows.append({"seed": seed, "condition": cond, "class": cls,
                             "correspondence": CLASS_CORRESPONDENCE.get(cls, "?"), **m})
    cons_df = pd.DataFrame(rows)
    cons_df.to_csv(os.path.join(out_dir, "consistency_per_seed.csv"), index=False)

    per_class = cons_df.groupby(["condition", "correspondence", "class"]).agg(
        rho=("spearman_rho", "mean"), rho_sd=("spearman_rho", "std"),
        ecs=("explanation_consistency_score", "mean"), ecs_sd=("explanation_consistency_score", "std"))
    print(f"\n{'=' * 70}\nCONSISTENCY ({explainer})\n{'=' * 70}")
    print(per_class.round(4).to_string())
    per_class.to_csv(os.path.join(out_dir, "consistency_summary.csv"))

    # ---- anchors -----------------------------------------------------------
    ub_rows = []
    for a, b in itertools.combinations(seeds, 2):
        for cls, m in same_dataset_upper_bound(rankings_by_seed[a]["source"],
                                               rankings_by_seed[b]["source"],
                                               SHARED_FEATURES, top_k).items():
            ub_rows.append({"pair": f"{a}v{b}", "class": cls, **m})
    ub_df = pd.DataFrame(ub_rows)
    ub_df.to_csv(os.path.join(out_dir, "upper_bound.csv"), index=False)

    rnd, obs = [], {"transfer": [], "native": []}
    for seed in seeds:
        for cls in class_names:
            s = rankings_by_seed[seed]["source"][cls]
            rnd.append(random_permutation_baseline(s["ranking"], s["importance"],
                                                   SHARED_FEATURES, seed, top_k)
                       ["explanation_consistency_score"])
            for cond, key in [("transfer", "target"), ("native", "native_target")]:
                o = rankings_by_seed[seed][key][cls]
                obs[cond].append(explanation_consistency_score(
                    s["ranking"], o["ranking"], s["importance"], o["importance"],
                    SHARED_FEATURES, top_k)["explanation_consistency_score"])

    ub_mean, lb_mean = float(ub_df["explanation_consistency_score"].mean()), float(np.mean(rnd))
    headline = {"explainer": explainer, "upper_bound_ecs": ub_mean,
                "lower_bound_ecs": lb_mean, "conditions": {}}

    print(f"\n{'=' * 70}\nANCHORED INTERPRETATION ({explainer})\n{'=' * 70}")
    print(f"  upper bound = {ub_mean:.4f}   lower bound = {lb_mean:.4f}\n")
    for cond in ["transfer", "native"]:
        m = float(np.mean(obs[cond]))
        norm = (m - lb_mean) / (ub_mean - lb_mean) if ub_mean > lb_mean else float("nan")
        w = wilcoxon_vs_random(obs[cond], rnd)
        print(f"  [{cond}] ECS = {m:.4f}   normalised = {norm:.3f}   p = {w['p_value']:.4g}")
        headline["conditions"][cond] = {"observed_ecs": m, "normalised_consistency": float(norm),
                                        "wilcoxon": w}

    det_path = os.path.join(cfg["output"]["tables_dir"], "detection_per_seed.csv")
    if os.path.exists(det_path):
        det_df = pd.read_csv(det_path)
        headline["detection"] = {f"{s}_f1_macro": float(det_df[det_df.split == s].f1_macro.mean())
                                 for s in ["source", "target", "native_target"]}
        headline["detection"]["note"] = "carried over from the main run; independent of the explainer"

    with open(os.path.join(out_dir, "headline.yaml"), "w") as f:
        yaml.safe_dump(headline, f)

    provenance = {
        "explainer": explainer,
        "checkpoints_reused_from": ckpt_dir,
        "retrained": False,
        "seeds": seeds,
        "top_k": top_k,
        "background_samples": cfg["shap"]["background_samples"],
        "explain_samples_per_class": cfg["shap"]["explain_samples_per_class"],
        "lstm_fallback_warnings": lstm_warnings,
        "other_warnings": other_warnings or None,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "provenance.yaml"), "w") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)

    print(f"\n  '{LSTM_WARNING}' warnings: {lstm_warnings}")
    if other_warnings:
        for k, n in other_warnings.items():
            print(f"  x{n}  {k}")
    print(f"\nAll tables written to {out_dir}/  ({provenance['runtime_seconds']:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--explainer", default="deep", choices=["gradient", "deep"])
    p.add_argument("--out-dir", default=None,
                   help="default: results/tables_<explainer>")
    a = p.parse_args()
    main(a.config, a.explainer, a.out_dir)
