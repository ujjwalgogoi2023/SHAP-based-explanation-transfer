"""Stage 3 — univariate structure of the twelve shared features.

The question this stage has to settle before anything multivariate is run is
what scale the features live on. All twelve are non-negative, zero-inflated
and span several orders of magnitude, which rules out reading a mean, a
standard deviation or a 3-sigma outlier rule at face value.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from src.eda.common import describe, load_both, outlier_rates
from src.eda.settings import (
    DATASETS,
    DS_COLOUR,
    FEATURE_SHORT,
    FEATURE_UNITS,
    PRETTY,
    savefig,
    savetable,
)


def _log_axis(ax, hi: float) -> None:
    """Tick a log1p-transformed axis at decades of the original unit, so the
    reader never has to mentally invert the transform."""
    decades = [0.0] + [10.0 ** k for k in range(0, 12)]
    pos, lab = [], []
    for v in decades:
        t = np.log1p(v)
        if t > hi * 1.001:
            break
        pos.append(t)
        lab.append("0" if v == 0 else ("1" if v == 1 else f"$10^{{{int(np.log10(v))}}}$"))
    if len(pos) > 6:  # thin to every other decade
        keep = [0] + list(range(1, len(pos), max(1, (len(pos) - 1) // 5)))
        pos, lab = [pos[i] for i in keep], [lab[i] for i in keep]
    ax.set_xticks(pos)
    ax.set_xticklabels(lab)


def fig_distributions(raw: dict) -> None:
    """Twelve marginal distributions, both benchmarks on the same axes.

    Plotted on a log1p x-axis: on the native scale every panel would be a
    single bar at the left edge with an invisible tail, which is exactly the
    reading failure the rest of this section is about.
    """
    feats = list(raw["cicids2017"][0].columns)
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.6))

    for ax, f in zip(axes.ravel(), feats):
        series = {d: np.log1p(np.clip(raw[d][0][f].to_numpy(), 0, None)) for d in DATASETS}
        lo = min(s.min() for s in series.values())
        hi = max(np.percentile(s, 99.9) for s in series.values())
        bins = np.linspace(lo, max(hi, lo + 1e-6), 60)
        body_peak, zero_share = 0.0, {}
        for d in DATASETS:
            h, _, _ = ax.hist(series[d], bins=bins, density=True, alpha=0.55,
                              color=DS_COLOUR[d], label=PRETTY[d])
            body_peak = max(body_peak, float(h[1:].max()) if h.size > 1 else float(h.max()))
            zero_share[d] = 100.0 * float((series[d] == 0).mean())

        # The spike at zero is often an order of magnitude taller than
        # everything else, so the y-axis is set by the body of the
        # distribution and the share sitting at exactly zero is written in.
        ax.set_ylim(0, body_peak * 1.45 if body_peak > 0 else None)
        ax.set_title(f"{f}  [{FEATURE_UNITS[f]}]", fontsize=9)
        ax.set_yticks([])
        if max(zero_share.values()) >= 1:
            ax.annotate(" ".join(f"{PRETTY[d][:4]} {zero_share[d]:.0f}%" for d in DATASETS),
                        xy=(0.015, 0.985), xycoords="axes fraction", va="top", ha="left",
                        fontsize=6.6, color="#333",
                        bbox=dict(fc="white", ec="#ccc", lw=0.5, pad=1.4))
        _log_axis(ax, bins[-1])

    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Figure 4 — Marginal distributions of the twelve shared features "
                 "(log1p scale; inset shows the share of records at exactly zero)")
    fig.tight_layout()
    savefig(fig, "f04_marginal_distributions")
    plt.close(fig)


def fig_transform_case(raw: dict) -> pd.DataFrame:
    """Skewness and kurtosis before and after log1p, plus one worked example."""
    rows = []
    for d in DATASETS:
        X = raw[d][0]
        for f in X.columns:
            v = X[f].to_numpy()
            lv = np.log1p(np.clip(v, 0, None))
            rows.append({
                "dataset": d, "feature": f,
                "skew_raw": float(stats.skew(v)),
                "skew_log": float(stats.skew(lv)),
                "kurtosis_raw": float(stats.kurtosis(v)),
                "kurtosis_log": float(stats.kurtosis(lv)),
                "orders_of_magnitude": float(np.log10(max(v.max(), 1) + 1)),
            })
    shape = pd.DataFrame(rows)

    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1.2], hspace=0.45, wspace=0.3)

    for r, d in enumerate(DATASETS):
        sub = shape[shape.dataset == d].set_index("feature")
        ax = fig.add_subplot(gs[r, 0])
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub.skew_raw.abs(), height=0.38, color="#adb5bd", label="raw")
        ax.barh(y + 0.2, sub.skew_log.abs(), height=0.38, color=DS_COLOUR[d], label="after log1p")
        ax.set_yticks(y)
        ax.set_yticklabels([FEATURE_SHORT[i] for i in sub.index], fontsize=7.5)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("|skewness| (symlog)")
        ax.set_title(f"({'ac'[r]}) {PRETTY[d]}: skewness", fontsize=9.5)
        if r == 0:
            ax.legend(loc="lower right", fontsize=7.5)

        ax = fig.add_subplot(gs[r, 1])
        ax.barh(y - 0.2, sub.kurtosis_raw.abs(), height=0.38, color="#adb5bd")
        ax.barh(y + 0.2, sub.kurtosis_log.abs(), height=0.38, color=DS_COLOUR[d])
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("|excess kurtosis| (symlog)")
        ax.set_title(f"({'bd'[r]}) {PRETTY[d]}: tail weight", fontsize=9.5)

    # Worked example on the worst-behaved feature, so the summary bars above
    # are anchored to something a reader can see.
    worst = shape.loc[shape.skew_raw.abs().idxmax()]
    v = raw[worst.dataset][0][worst.feature].to_numpy()
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(v, bins=60, color="#adb5bd")
    ax.set_yscale("log")
    ax.set_title(f"(e) {worst.feature} on {PRETTY[worst.dataset]}\nnative scale", fontsize=9)
    ax.set_xlabel(FEATURE_UNITS[worst.feature])
    ax.set_ylabel("records (log)")
    ax.text(0.97, 0.9, f"skew = {worst.skew_raw:,.0f}", transform=ax.transAxes,
            ha="right", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.log1p(np.clip(v, 0, None)), bins=60, color=DS_COLOUR[worst.dataset])
    ax.set_yscale("log")
    ax.set_title("(f) same feature after log1p", fontsize=9)
    ax.set_xlabel(f"log1p({FEATURE_UNITS[worst.feature]})")
    ax.set_ylabel("records (log)")
    ax.text(0.97, 0.9, f"skew = {worst.skew_log:,.2f}", transform=ax.transAxes,
            ha="right", fontsize=8)

    fig.suptitle("Figure 5 — Distributional shape before and after the log1p transform")
    savefig(fig, "f05_transform_justification")
    plt.close(fig)
    return shape


def fig_outliers_and_zeros(raw: dict) -> pd.DataFrame:
    rows = []
    for d in DATASETS:
        o = outlier_rates(raw[d][0])
        o.insert(0, "dataset", d)
        desc = describe(raw[d][0])[["feature", "zero_pct"]]
        rows.append(o.merge(desc, on="feature"))
    out = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    feats = list(raw["cicids2017"][0].columns)
    y = np.arange(len(feats))

    panels = [
        ("tukey_pct", "(a) Tukey rule (1.5 x IQR), native scale"),
        ("z3_pct", "(b) Three-sigma rule, native scale"),
        ("tukey_log_pct", "(c) Tukey rule after log1p"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for k, d in enumerate(DATASETS):
            sub = out[out.dataset == d].set_index("feature").reindex(feats)
            ax.barh(y + (k - 0.5) * 0.38, sub[col], height=0.36,
                    color=DS_COLOUR[d], label=PRETTY[d])
        ax.set_yticks(y)
        ax.set_yticklabels([FEATURE_SHORT[f] for f in feats] if col == "tukey_pct" else [],
                           fontsize=7.5)
        ax.set_xlabel("records flagged (%)")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, max(35, out[col].max() * 1.1))
    axes[0].legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Figure 6 — Outlier burden under two screening rules, before and after transform")
    fig.tight_layout()
    savefig(fig, "f06_outlier_burden")
    plt.close(fig)
    return out


def fig_ecdf(raw: dict) -> None:
    """Empirical CDFs, which is where the cross-dataset gap first becomes
    visible without any test statistic attached to it."""
    feats = list(raw["cicids2017"][0].columns)
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.4))

    for ax, f in zip(axes.ravel(), feats):
        hi = 0.0
        for d in DATASETS:
            v = np.sort(np.log1p(np.clip(raw[d][0][f].to_numpy(), 0, None)))
            # Thin to 4000 plotting points; the curve is a step function over
            # millions of records and the file size is otherwise absurd.
            idx = np.linspace(0, v.size - 1, min(4000, v.size)).astype(int)
            ax.plot(v[idx], idx / (v.size - 1), color=DS_COLOUR[d], lw=1.6, label=PRETTY[d])
            hi = max(hi, float(v[-1]))
        ax.set_title(f, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        _log_axis(ax, hi)
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("F(x)")

    fig.suptitle("Figure 7 — Empirical cumulative distributions, CICIDS2017 against UNSW-NB15")
    fig.tight_layout()
    savefig(fig, "f07_ecdf")
    plt.close(fig)


def run() -> None:
    raw = load_both(logged=False)

    desc = []
    for d in DATASETS:
        t = describe(raw[d][0])
        t.insert(0, "dataset", d)
        desc.append(t)
    desc = pd.concat(desc, ignore_index=True)
    savetable(desc, "t10_descriptive_statistics", index=False)

    fig_distributions(raw)
    shape = fig_transform_case(raw)
    savetable(shape, "t11_distribution_shape", index=False)
    out = fig_outliers_and_zeros(raw)
    savetable(out, "t12_outlier_rates", index=False)
    fig_ecdf(raw)

    summary = (shape.groupby("dataset")
               .agg(median_abs_skew_raw=("skew_raw", lambda s: s.abs().median()),
                    median_abs_skew_log=("skew_log", lambda s: s.abs().median()),
                    max_abs_skew_raw=("skew_raw", lambda s: s.abs().max()),
                    max_orders_of_magnitude=("orders_of_magnitude", "max")))
    summary = summary.join(out.groupby("dataset").agg(
        mean_tukey_pct=("tukey_pct", "mean"),
        mean_z3_pct=("z3_pct", "mean"),
        mean_tukey_log_pct=("tukey_log_pct", "mean"),
        mean_zero_pct=("zero_pct", "mean")))
    savetable(summary, "t13_univariate_summary")
    print(summary.T.to_string())


if __name__ == "__main__":
    from src.eda.settings import apply_style
    apply_style()
    run()
