"""Compare the two attribution methods and emit the appendix table.

    python scripts/compare_explainers.py

Reads the main run (`results/tables/`, Expected Gradients) and the variant run
(`results/tables_deep/`, DeepSHAP) and prints both a console summary and the
LaTeX tabular body used in the paper's appendix, so the table in the paper is
generated from the artefacts rather than typed out.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml

from src.data.schema import UNIFIED_CLASSES

SHORT = {"Reconnaissance/Probe": "Recon./Probe", "DoS/DDoS": "DoS/DDoS",
         "Benign": "Benign", "Exploits": "Exploits"}


def load(d):
    with open(os.path.join(d, "headline.yaml")) as f:
        head = yaml.safe_load(f)
    cons = pd.read_csv(os.path.join(d, "consistency_per_seed.csv"))
    ub = pd.read_csv(os.path.join(d, "upper_bound.csv"))
    return head, cons, ub


def per_class(cons, ub, head, cond):
    """Per-class ECS and consistency normalised against that class's own
    upper bound and the pooled random floor."""
    lb = head["lower_bound_ecs"]
    ubc = ub.groupby("class")["explanation_consistency_score"].mean()
    out = {}
    for cls in [c for c in UNIFIED_CLASSES if c in ubc.index]:
        s = cons[(cons.condition == cond) & (cons["class"] == cls)]["explanation_consistency_score"]
        out[cls] = (s.mean(), s.std(), (s.mean() - lb) / (ubc[cls] - lb))
    return out


def main(grad_dir, deep_dir):
    for d in (grad_dir, deep_dir):
        if not os.path.exists(os.path.join(d, "headline.yaml")):
            sys.exit(f"missing {d}/headline.yaml -- run scripts/run_explainer_variant.py first")

    g_head, g_cons, g_ub = load(grad_dir)
    d_head, d_cons, d_ub = load(deep_dir)

    prov_path = os.path.join(deep_dir, "provenance.yaml")
    prov = yaml.safe_load(open(prov_path)) if os.path.exists(prov_path) else {}

    print("=" * 78)
    print("POOLED, native condition (the interpretable one)")
    print("=" * 78)
    print(f"{'':28s} {'Expected Gradients':>20s} {'DeepSHAP':>20s}")
    rows = [
        ("upper bound ECS", g_head["upper_bound_ecs"], d_head["upper_bound_ecs"]),
        ("random floor ECS", g_head["lower_bound_ecs"], d_head["lower_bound_ecs"]),
        ("native ECS", g_head["conditions"]["native"]["observed_ecs"],
         d_head["conditions"]["native"]["observed_ecs"]),
        ("native normalised", g_head["conditions"]["native"]["normalised_consistency"],
         d_head["conditions"]["native"]["normalised_consistency"]),
        ("transfer ECS", g_head["conditions"]["transfer"]["observed_ecs"],
         d_head["conditions"]["transfer"]["observed_ecs"]),
        ("transfer normalised", g_head["conditions"]["transfer"]["normalised_consistency"],
         d_head["conditions"]["transfer"]["normalised_consistency"]),
    ]
    for label, a, b in rows:
        print(f"  {label:26s} {a:20.4f} {b:20.4f}")
    for cond in ("native", "transfer"):
        gp = g_head["conditions"][cond]["wilcoxon"]["p_value"]
        dp = d_head["conditions"][cond]["wilcoxon"]["p_value"]
        print(f"  {cond + ' Wilcoxon p':26s} {gp:20.3g} {dp:20.3g}")

    g_pc = per_class(g_cons, g_ub, g_head, "native")
    d_pc = per_class(d_cons, d_ub, d_head, "native")

    print()
    print("=" * 78)
    print("PER CLASS, native condition: ECS (normalised)")
    print("=" * 78)
    for cls in g_pc:
        ge, gs, gn = g_pc[cls]
        de, ds, dn = d_pc[cls]
        print(f"  {cls:22s} grad {ge:.3f}+-{gs:.3f} ({gn:.2f})   deep {de:.3f}+-{ds:.3f} ({dn:.2f})")

    if prov:
        print()
        print(f"LSTM fallback warnings during the deep run: {prov.get('lstm_fallback_warnings')}")
        print(f"checkpoints reused, retrained = {prov.get('retrained')}")

    print()
    print("=" * 78)
    print("LaTeX tabular body for the appendix (paste into main.tex)")
    print("=" * 78)
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{\textbf{Expected Gradients}} & "
          r"\multicolumn{2}{c}{\textbf{DeepSHAP}} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"\textbf{Class} & ECS & Norm. & ECS & Norm. \\")
    print(r"\midrule")
    for cls in g_pc:
        ge, gs, gn = g_pc[cls]
        de, ds, dn = d_pc[cls]
        print(f"{SHORT.get(cls, cls)} & ${ge:.3f}$ & ${gn:.2f}$ & ${de:.3f}$ & ${dn:.2f}$ \\\\")
    print(r"\midrule")
    gh, dh = g_head["conditions"]["native"], d_head["conditions"]["native"]
    print(f"Pooled & ${gh['observed_ecs']:.3f}$ & ${gh['normalised_consistency']:.3f}$ & "
          f"${dh['observed_ecs']:.3f}$ & ${dh['normalised_consistency']:.3f}$ \\\\")
    print(f"Upper bound & ${g_head['upper_bound_ecs']:.3f}$ & $1.00$ & "
          f"${d_head['upper_bound_ecs']:.3f}$ & $1.00$ \\\\")
    print(f"Random floor & ${g_head['lower_bound_ecs']:.3f}$ & $0.00$ & "
          f"${d_head['lower_bound_ecs']:.3f}$ & $0.00$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gradient-dir", default="results/tables")
    p.add_argument("--deep-dir", default="results/tables_deep")
    a = p.parse_args()
    main(a.gradient_dir, a.deep_dir)
