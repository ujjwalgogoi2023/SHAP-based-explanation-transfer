"""Stage 4 — dependence between features, and between features and class.

Two things are established here. First, how much of the twelve-dimensional
feature space is actually independent information, which bears directly on how
many features an attribution method has to spread credit across. Second,
whether the two benchmarks agree about *which* features carry class
information — a question that can be asked of the data alone, with no model
and no attribution method in the way.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from src.eda.common import load_both, mutual_information
from src.eda.settings import (
    CLASS_COLOUR,
    DATASETS,
    DS_COLOUR,
    FEATURE_SHORT,
    PRETTY,
    UNIFIED_CLASSES,
    savefig,
    savetable,
)


def _spearman(X: pd.DataFrame) -> pd.DataFrame:
    """Spearman rather than Pearson: after Section 3, a linear correlation
    coefficient on these marginals would be reporting the influence of a few
    extreme flows."""
    return X.corr(method="spearman")


def fig_correlation(logged: dict) -> dict[str, pd.DataFrame]:
    corrs = {d: _spearman(logged[d][0]) for d in DATASETS}
    diff = corrs["unsw_nb15"] - corrs["cicids2017"]
    feats = list(corrs["cicids2017"].columns)
    short = [FEATURE_SHORT[f] for f in feats]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    panels = [("cicids2017", corrs["cicids2017"], "RdBu_r", (-1, 1),
               "(a) CICIDS2017 (Spearman $\\rho$)"),
              ("unsw_nb15", corrs["unsw_nb15"], "RdBu_r", (-1, 1),
               "(b) UNSW-NB15 (Spearman $\\rho$)"),
              ("diff", diff, "PuOr_r", (-1.2, 1.2),
               "(c) UNSW-NB15 minus CICIDS2017")]

    for ax, (key, M, cmap, (lo, hi), title) in zip(axes, panels):
        im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(len(feats)))
        ax.set_yticks(range(len(feats)))
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticklabels(short if key == "cicids2017" else [], fontsize=7)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        for i in range(len(feats)):
            for j in range(len(feats)):
                v = M.iloc[i, j]
                if abs(v) >= 0.45 and i != j:
                    ax.text(j, i, f"{v:+.1f}".replace("0.", "."), ha="center", va="center",
                            fontsize=5.6, color="white" if abs(v) > 0.72 else "#222")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle("Figure 8 — Rank correlation among the twelve shared features, "
                 "and how it differs between benchmarks")
    fig.tight_layout()
    savefig(fig, "f08_correlation")
    plt.close(fig)

    long = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            long.append({"feature_a": a, "feature_b": b,
                         "rho_cicids2017": corrs["cicids2017"].loc[a, b],
                         "rho_unsw_nb15": corrs["unsw_nb15"].loc[a, b],
                         "abs_difference": abs(diff.loc[a, b])})
    pairs = pd.DataFrame(long).sort_values("abs_difference", ascending=False)
    savetable(pairs, "t14_correlation_pairs", index=False)
    return corrs


def _vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors from the inverse correlation matrix, which
    is equivalent to the regression definition and avoids fitting twelve
    regressions over millions of rows."""
    R = X.corr(method="pearson").to_numpy()
    inv = np.linalg.pinv(R)
    return pd.Series(np.diag(inv), index=X.columns)


def fig_redundancy(logged: dict, corrs: dict) -> pd.DataFrame:
    vifs = pd.DataFrame({d: _vif(logged[d][0]) for d in DATASETS})
    feats = list(vifs.index)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    ax = axes[0]
    y = np.arange(len(feats))
    for k, d in enumerate(DATASETS):
        ax.barh(y + (k - 0.5) * 0.38, vifs[d], height=0.36,
                color=DS_COLOUR[d], label=PRETTY[d])
    ax.axvline(5, color="#444", ls="--", lw=1)
    ax.axvline(10, color="#c1121f", ls="--", lw=1)
    ax.text(5, -1.0, "VIF 5", fontsize=7, color="#444", ha="center")
    ax.text(10, -1.6, "VIF 10", fontsize=7, color="#c1121f", ha="center")
    ax.set_ylim(-2.0, len(feats) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_SHORT[f] for f in feats], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("variance inflation factor (log)")
    ax.set_title("(a) Multicollinearity", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)

    # Correlation-distance dendrograms: the question is not only how strong the
    # redundancy is but whether the two benchmarks group the features the same
    # way. They do not.
    for ax, d in zip(axes[1:], DATASETS):
        dist = (1 - corrs[d].abs()).to_numpy(copy=True)
        np.fill_diagonal(dist, 0.0)
        Z = hierarchy.linkage(squareform(dist, checks=False), method="average")
        hierarchy.dendrogram(Z, labels=[FEATURE_SHORT[f] for f in feats], ax=ax,
                             color_threshold=0.55, above_threshold_color="#9a9a9a",
                             leaf_font_size=7.5, leaf_rotation=90)
        ax.set_title(f"({'bc'[d == 'unsw_nb15']}) {PRETTY[d]}: feature clustering", fontsize=9.5)
        ax.set_ylabel("1 - |Spearman $\\rho$|")
        ax.set_ylim(0, 1)
        ax.grid(False)

    fig.suptitle("Figure 9 — Redundancy within the feature set")
    fig.tight_layout()
    savefig(fig, "f09_redundancy")
    plt.close(fig)

    vifs.index.name = "feature"
    savetable(vifs, "t15_vif")
    return vifs


def class_mutual_information(logged: dict) -> tuple[pd.DataFrame, dict]:
    """Normalised MI between each feature and each class, one-vs-rest."""
    overall, per_class = {}, {}
    for d in DATASETS:
        X, y = logged[d]
        overall[d] = mutual_information(X, y)
        mat = {}
        for cls in UNIFIED_CLASSES:
            binary = pd.Series(np.where((y == cls).to_numpy(), cls, "other"))
            # mutual_information expects the unified categories, so compute the
            # binary case directly with the same histogram estimator.
            mat[cls] = _binary_mi(X, (y == cls).to_numpy())
        per_class[d] = pd.DataFrame(mat)
    return pd.DataFrame(overall), per_class


def _binary_mi(X: pd.DataFrame, is_class: np.ndarray, n_bins: int = 32) -> pd.Series:
    p1 = float(is_class.mean())
    if p1 in (0.0, 1.0):
        return pd.Series(0.0, index=X.columns)
    hy = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
    out = {}
    codes = is_class.astype(int)
    for col in X.columns:
        v = X[col].to_numpy(dtype="float64")
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            out[col] = 0.0
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.digitize(v, edges[1:-1])
        joint = np.zeros((b.max() + 1, 2))
        np.add.at(joint, (b, codes), 1.0)
        joint /= joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        mi = float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))
        out[col] = mi / hy
    return pd.Series(out)


def fig_association(overall: pd.DataFrame, per_class: dict) -> pd.DataFrame:
    feats = list(overall.index)
    short = [FEATURE_SHORT[f] for f in feats]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    vmax = max(per_class[d].to_numpy().max() for d in DATASETS)
    for ax, d in zip(axes[:2], DATASETS):
        M = per_class[d].reindex(index=feats, columns=UNIFIED_CLASSES)
        im = ax.imshow(M.to_numpy(), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(UNIFIED_CLASSES)))
        ax.set_xticklabels(["Benign", "DoS/\nDDoS", "Recon./\nProbe", "Exploits"], fontsize=7.5)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(short if d == "cicids2017" else [], fontsize=7.5)
        ax.set_title(f"({'ab'[d == 'unsw_nb15']}) {PRETTY[d]}", fontsize=9.5)
        ax.grid(False)
        for i in range(len(feats)):
            for j in range(len(UNIFIED_CLASSES)):
                v = M.iloc[i, j]
                ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=6,
                        color="white" if v < vmax * 0.55 else "#111")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    # Do the two benchmarks rank the features the same way? This is the
    # model-free version of the question the parent project asks of SHAP.
    ax = axes[2]
    rows = []
    for cls in UNIFIED_CLASSES:
        a = per_class["cicids2017"][cls].rank(ascending=False)
        b = per_class["unsw_nb15"][cls].rank(ascending=False)
        rho = stats.spearmanr(a, b).statistic
        top3_a = set(per_class["cicids2017"][cls].nlargest(3).index)
        top3_b = set(per_class["unsw_nb15"][cls].nlargest(3).index)
        rows.append({"class": cls, "spearman_rho": rho,
                     "jaccard_top3": len(top3_a & top3_b) / len(top3_a | top3_b),
                     "top3_cicids2017": ", ".join(sorted(top3_a)),
                     "top3_unsw_nb15": ", ".join(sorted(top3_b))})
    agree = pd.DataFrame(rows)

    x = np.arange(len(agree))
    ax.bar(x - 0.19, agree.spearman_rho, width=0.36, color="#264653",
           label="Spearman $\\rho$ of the\nfull 12-feature ranking")
    ax.bar(x + 0.19, agree.jaccard_top3, width=0.36, color="#e9c46a",
           label="Jaccard overlap of\nthe top 3 features")
    ax.axhline(0, color="#333", lw=0.9)
    for xi, v in zip(x - 0.19, agree.spearman_rho):
        ax.text(xi, v + (0.03 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.19, agree.jaccard_top3):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Benign", "DoS/\nDDoS", "Recon./\nProbe", "Exploits"], fontsize=7.5)
    ax.set_ylim(-0.75, 1.15)
    ax.set_ylabel("agreement between benchmarks")
    ax.set_title("(c) Do the benchmarks agree on which\nfeatures carry the class?", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=6.8)

    fig.suptitle("Figure 10 — Feature-to-class association (normalised mutual information, "
                 "one class against the rest)")
    fig.tight_layout()
    savefig(fig, "f10_association")
    plt.close(fig)

    savetable(agree, "t17_ranking_agreement", index=False)
    print(agree[["class", "spearman_rho", "jaccard_top3"]].to_string(index=False))
    return agree


def fig_class_conditional(logged: dict, overall: pd.DataFrame) -> None:
    """Class-conditional distributions for the features that carry the most
    class information, shown as boxplots on the logged scale."""
    top = (overall.mean(axis=1).nlargest(4).index.tolist())
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.0), sharey="row")

    for r, d in enumerate(DATASETS):
        X, y = logged[d]
        for c, f in enumerate(top):
            ax = axes[r, c]
            data = [X.loc[(y == cls).to_numpy(), f].to_numpy() for cls in UNIFIED_CLASSES]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                            medianprops=dict(color="#111", lw=1.3))
            for patch, cls in zip(bp["boxes"], UNIFIED_CLASSES):
                patch.set_facecolor(CLASS_COLOUR[cls])
                patch.set_alpha(0.85)
                patch.set_edgecolor("#444")
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(["Ben", "DoS", "Rec", "Exp"], fontsize=7.5)
            ax.set_title(f"{f}" if r == 0 else "", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{PRETTY[d]}\nlog1p value", fontsize=8.5)

    fig.suptitle("Figure 11 — Class-conditional distributions of the four most "
                 "informative features (whiskers at 1.5 x IQR, outliers hidden)")
    fig.tight_layout()
    savefig(fig, "f11_class_conditional")
    plt.close(fig)


def run() -> None:
    logged = load_both(logged=True)
    corrs = fig_correlation(logged)
    fig_redundancy(logged, corrs)
    overall, per_class = class_mutual_information(logged)
    overall.index.name = "feature"
    savetable(overall, "t16_mutual_information")
    for d in DATASETS:
        pc = per_class[d].copy()
        pc.index.name = "feature"
        savetable(pc, f"t16b_mutual_information_per_class_{d}")
    fig_association(overall, per_class)
    fig_class_conditional(logged, overall)
    print(overall.sort_values("cicids2017", ascending=False).to_string())


if __name__ == "__main__":
    from src.eda.settings import apply_style
    apply_style()
    run()
