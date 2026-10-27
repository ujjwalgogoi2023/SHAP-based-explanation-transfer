"""Stage 6 — holding the EDA findings against what the model actually did.

The parent project trained a hybrid CNN-LSTM on CICIDS2017 across five seeds,
applied it unchanged to UNSW-NB15, and compared per-class SHAP rankings. Those
results are read here as held-out evidence: the EDA was carried out on the
data alone, so if its diagnostics line up with the modelling outcome, the
diagnostics were doing their job.

This stage computes nothing new about the benchmarks. It only joins what
Stages 2-5 measured to `results/tables/`, which this project did not produce.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.eda.common import model_results
from src.eda.settings import CLASS_COLOUR, MODEL_TAB_DIR, TAB_DIR, UNIFIED_CLASSES, savefig, savetable


def _eda_class_shift() -> pd.Series:
    cs = pd.read_csv(os.path.join(TAB_DIR, "t19_class_conditional_shift.csv"))
    return cs.groupby("class").smd.apply(lambda s: s.abs().mean())


def _eda_ranking_agreement() -> pd.Series:
    ra = pd.read_csv(os.path.join(TAB_DIR, "t17_ranking_agreement.csv"))
    return ra.set_index("class").spearman_rho


def fig_synthesis() -> pd.DataFrame:
    res = model_results()
    det = pd.read_csv(os.path.join(MODEL_TAB_DIR, "detection_per_class.csv"))
    cons = res["consistency"]
    head = res["headline"]

    shift = _eda_class_shift()
    agree = _eda_ranking_agreement()

    per_class = (det[det.condition == "transfer"]
                 .groupby("class").f1.agg(["mean", "std"])
                 .rename(columns={"mean": "transfer_f1", "std": "transfer_f1_sd"}))
    # The per-class table names the three conditions source / transfer /
    # native; the summary table calls the same three source / target /
    # native_target. Both are read here, so the mapping is spelled out.
    native = (det[det.condition == "native"]
              .groupby("class").f1.mean().rename("native_target_f1"))
    source = (det[det.condition == "source"]
              .groupby("class").f1.mean().rename("source_f1"))
    ecs = cons[cons.condition == "native"].set_index("class").ecs.rename("shap_ecs_native")

    joined = (pd.concat([shift.rename("eda_mean_abs_smd"), agree.rename("eda_mi_rank_rho"),
                         source, native, per_class, ecs], axis=1)
              .reindex(UNIFIED_CLASSES))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))

    # (a) what happened to detection.
    ax = axes[0]
    d = res["detection"]
    order = ["source", "native_target", "target"]
    labels = ["trained and tested\non CICIDS2017", "trained and tested\non UNSW-NB15",
              "CICIDS2017 model\napplied to UNSW-NB15"]
    means = [d.loc[o, ("f1_macro", "mean")] for o in order]
    sds = [d.loc[o, ("f1_macro", "std")] for o in order]
    colours = ["#1f6feb", "#d1495b", "#6c757d"]
    bars = ax.bar(range(3), means, yerr=sds, capsize=4, color=colours, width=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.035, f"{m:.3f}", ha="center",
                fontsize=9, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("macro F1 (mean of 5 seeds, $\\pm$1 SD)")
    ax.set_ylim(0, 1.22)
    ax.set_title("(a) What the shift cost the model", fontsize=9.5)
    ax.annotate("", xy=(2, means[2] + 0.06), xytext=(0, means[0] + 0.09),
                arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.4,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(1.0, 1.14, f"$-${means[0] - means[2]:.2f} macro F1", ha="center", fontsize=9,
            color="#c1121f", weight="bold")

    # (b) class-level shift against class-level transfer performance.
    ax = axes[1]
    for cls in UNIFIED_CLASSES:
        r = joined.loc[cls]
        ax.scatter(r.eda_mean_abs_smd, r.transfer_f1, s=150, color=CLASS_COLOUR[cls],
                   edgecolor="#333", zorder=3, label=cls)
        ax.annotate(cls.replace("Reconnaissance/", "Recon./"),
                    (r.eda_mean_abs_smd, r.transfer_f1), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7.4)
    rho = stats.spearmanr(joined.eda_mean_abs_smd, joined.transfer_f1).statistic
    ax.set_xlabel("EDA: mean |SMD| within the class (Fig. 13b)")
    ax.set_ylabel("model: transfer F1 for that class")
    ax.set_title("(b) Shift measured before training,\nagainst transfer performance after",
                 fontsize=9.5)
    ax.text(0.97, 0.95, f"Spearman $\\rho$ = {rho:+.2f}  (n = 4)", transform=ax.transAxes,
            ha="right", fontsize=8)
    ax.set_ylim(-0.05, max(joined.transfer_f1) * 1.45)

    # (c) the model-free ranking disagreement against the SHAP disagreement.
    ax = axes[2]
    for cls in UNIFIED_CLASSES:
        r = joined.loc[cls]
        ax.scatter(r.eda_mi_rank_rho, r.shap_ecs_native, s=150, color=CLASS_COLOUR[cls],
                   edgecolor="#333", zorder=3)
        ax.annotate(cls.replace("Reconnaissance/", "Recon./"),
                    (r.eda_mi_rank_rho, r.shap_ecs_native), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7.4)
    ax.axhline(head["lower_bound_ecs"], color="#c1121f", ls="--", lw=1)
    ax.text(0.02, head["lower_bound_ecs"] + 0.004, "random-permutation floor", fontsize=7,
            color="#c1121f", transform=ax.get_yaxis_transform())
    ax.axhline(head["upper_bound_ecs"], color="#2a9d8f", ls="--", lw=1)
    ax.text(0.02, head["upper_bound_ecs"] + 0.004, "same-dataset reseed ceiling", fontsize=7,
            color="#2a9d8f", transform=ax.get_yaxis_transform())
    rho2 = stats.spearmanr(joined.eda_mi_rank_rho, joined.shap_ecs_native).statistic
    ax.set_xlabel("EDA: agreement of MI feature rankings (Fig. 10c)")
    ax.set_ylabel("model: SHAP explanation consistency")
    ax.set_title("(c) The benchmarks disagreed about features\nbefore any model was fitted",
                 fontsize=9.5)
    ax.text(0.97, 0.06, f"Spearman $\\rho$ = {rho2:+.2f}  (n = 4)", transform=ax.transAxes,
            ha="right", fontsize=8)
    ax.set_ylim(head["lower_bound_ecs"] - 0.04, head["upper_bound_ecs"] + 0.05)

    fig.suptitle("Figure 16 — EDA diagnostics against the modelling outcome they anticipate")
    fig.tight_layout()
    savefig(fig, "f16_synthesis")
    plt.close(fig)

    joined.index.name = "class"
    savetable(joined, "t24_eda_vs_model")
    print(joined.to_string())
    print(f"\nrho(shift, transfer F1) = {rho:+.3f}   rho(MI agreement, SHAP ECS) = {rho2:+.3f}")
    return joined


def findings_table() -> pd.DataFrame:
    """The report's closing table: each finding, the evidence, the consequence."""
    q = pd.read_csv(os.path.join(TAB_DIR, "t09_quality_scorecard.csv")).set_index("dataset")
    imb = pd.read_csv(os.path.join(TAB_DIR, "t08_imbalance.csv")).set_index("dataset")
    uni = pd.read_csv(os.path.join(TAB_DIR, "t13_univariate_summary.csv")).set_index("dataset")
    sh = pd.read_csv(os.path.join(TAB_DIR, "t23_shift_headline.csv")).iloc[0]
    vif = pd.read_csv(os.path.join(TAB_DIR, "t15_vif.csv")).set_index("feature")
    agree = pd.read_csv(os.path.join(TAB_DIR, "t17_ranking_agreement.csv"))

    rows = [
        ("Structural defects in the published files",
         f"{q.loc['cicids2017', 'columns_with_leading_space']} CICIDS2017 column names carry a "
         f"leading space, one name is used twice, {q.loc['cicids2017', 'constant_columns']} columns "
         f"are constant and {q.loc['cicids2017', 'columns_over_99pct_zero']} are >99% zero",
         "Exact-name lookups fail silently; zero-variance columns contribute nothing but are "
         "counted as features"),
        ("Impossible values survive in physical quantities",
         f"{q.loc['cicids2017', 'columns_with_negatives']} CICIDS2017 columns contain negative "
         f"values, including Flow Duration (min -13) and Fwd Header Length (min -3.2e10)",
         "Rows have to be screened on physical plausibility, not only on NaN"),
        ("Infinite and missing rates",
         "Flow Bytes/s: 1,358 NaN and 1,509 infinite; Flow Packets/s: 2,867 infinite, "
         "all from division by a zero flow duration",
         "Rate features must be recomputed from primitives rather than read from the file"),
        ("Duplicate records",
         f"{q.loc['cicids2017', 'duplicate_pct']:.1f}% of CICIDS2017 and "
         f"{q.loc['unsw_nb15', 'duplicate_pct']:.1f}% of UNSW-NB15 records are exact duplicates",
         "A random split leaks identical rows between train and test and inflates reported scores"),
        ("Label encoding is not UTF-8",
         "3 of 15 CICIDS2017 labels contain cp1252 byte 0x96; the matched string depends on the "
         "read encoding",
         "Labels must be canonicalised before mapping, or whole attack classes vanish silently"),
        ("Severe class imbalance",
         f"CICIDS2017 majority:minority = {imb.loc['cicids2017', 'imbalance_ratio_raw_labels']:,.0f}:1 "
         f"across raw labels (Heartbleed n=11); UNSW-NB15 = "
         f"{imb.loc['unsw_nb15', 'imbalance_ratio_raw_labels']:,.0f}:1",
         "Accuracy is uninformative; per-class metrics and class weighting are mandatory"),
        ("Extreme skew over nine orders of magnitude",
         f"median |skew| = {uni.loc['cicids2017', 'median_abs_skew_raw']:.1f} on CICIDS2017, "
         f"falling to {uni.loc['cicids2017', 'median_abs_skew_log']:.2f} after log1p",
         "Any scale-sensitive step (standardisation, PCA, distance) needs the transform first"),
        ("Outlier rules disagree by an order of magnitude",
         f"Tukey flags {uni.loc['cicids2017', 'mean_tukey_pct']:.1f}% of CICIDS2017 records on "
         f"average, the three-sigma rule {uni.loc['cicids2017', 'mean_z3_pct']:.1f}%",
         "Neither can be used as an automatic drop rule; clipping at fixed quantiles is reported "
         "instead"),
        ("The feature set is largely redundant",
         f"max VIF = {vif.cicids2017.max():,.0f} (CICIDS2017) and {vif.unsw_nb15.max():,.0f} "
         f"(UNSW-NB15); {sh['pcs_for_90pct_variance']:.0f} principal components carry 90% of the "
         f"variance of 12 features",
         "Attribution is split across interchangeable features, which depresses rank stability "
         "independently of any dataset effect"),
        ("Every feature has shifted between the benchmarks",
         f"all 12 features exceed the PSI 0.25 'major shift' threshold; mean KS D = "
         f"{sh['mean_ks']:.2f}, mean JS divergence = {sh['mean_jsd_bits']:.2f} bits",
         "A source-fitted scaler maps the target into the wrong region of feature space"),
        ("The shift is not an artefact of class mixture",
         f"mean |SMD| within class = {sh['mean_abs_smd_within_class']:.2f}, against Cohen's 0.8 "
         f"for a large effect",
         "The same attack class genuinely looks different in the two testbeds"),
        ("The corpora are almost perfectly separable",
         f"a gradient-boosted classifier recovers the source benchmark with AUC = "
         f"{sh['domain_auc']:.5f} from the 12 shared features alone",
         "A detector is free to key on corpus identity rather than attack behaviour"),
        ("The benchmarks disagree about which features matter, before any model",
         "mutual-information feature rankings agree at Spearman rho = "
         + ", ".join(f"{r.spearman_rho:+.2f} ({r['class']})" for _, r in agree.iterrows()),
         "Explanation instability is visible in the data itself, not only in the attributions"),
    ]
    out = pd.DataFrame(rows, columns=["finding", "evidence", "consequence for modelling"])
    savetable(out, "t25_findings", index=False)
    return out


def run() -> None:
    fig_synthesis()
    findings_table()


if __name__ == "__main__":
    from src.eda.settings import apply_style
    apply_style()
    run()
