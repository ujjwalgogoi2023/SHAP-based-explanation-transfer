"""Stage 5 — how far apart the two benchmarks are.

Everything so far has described each benchmark on its own terms. This stage
quantifies the gap between them, which is the question that decides whether a
model or an explanation fitted on one can be read on the other. Three views:
marginal divergence per feature, class-conditional divergence, and a direct
test of separability — a classifier trained to recover which benchmark a flow
came from.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.eda.common import jensen_shannon, load_both, population_stability_index, stratified_sample
from src.eda.settings import (
    CLASS_COLOUR,
    DATASETS,
    DS_COLOUR,
    FEATURE_SHORT,
    PRETTY,
    RANDOM_STATE,
    UNIFIED_CLASSES,
    savefig,
    savetable,
)

# KS is computed on a capped random draw from each side. The statistic itself
# is a distance between ECDFs and is stable well before 200k points, while its
# p-value at 2.8 million is zero for any difference at all and carries no
# information; the effect size is what gets reported.
KS_SAMPLE = 200_000


def marginal_shift(logged: dict) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    Xc, Xu = logged["cicids2017"][0], logged["unsw_nb15"][0]
    rows = []
    for f in Xc.columns:
        a_full = Xc[f].to_numpy()
        b_full = Xu[f].to_numpy()
        a = rng.choice(a_full, min(KS_SAMPLE, a_full.size), replace=False)
        b = rng.choice(b_full, min(KS_SAMPLE, b_full.size), replace=False)
        ks = stats.ks_2samp(a, b, method="asymp")
        pooled_sd = np.sqrt((a_full.var(ddof=1) + b_full.var(ddof=1)) / 2)
        rows.append({
            "feature": f,
            "ks_statistic": float(ks.statistic),
            "jensen_shannon_bits": jensen_shannon(a, b),
            "psi": population_stability_index(a_full, b_full),
            "standardised_mean_diff": float((b_full.mean() - a_full.mean()) / pooled_sd)
            if pooled_sd > 0 else np.nan,
            "median_cicids2017": float(np.median(a_full)),
            "median_unsw_nb15": float(np.median(b_full)),
        })
    out = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)
    out["psi_verdict"] = pd.cut(out.psi, [-np.inf, 0.1, 0.25, np.inf],
                                labels=["stable", "moderate shift", "major shift"])
    return out


def fig_marginal_shift(shift: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    feats = shift.feature.tolist()
    y = np.arange(len(feats))[::-1]

    ax = axes[0]
    ax.hlines(y, 0, shift.ks_statistic, color="#8d99ae", lw=1.4)
    ax.scatter(shift.ks_statistic, y, s=46, color="#2b2d42", zorder=3)
    for yi, v in zip(y, shift.ks_statistic):
        ax.text(v + 0.018, yi, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_SHORT[f] for f in feats], fontsize=7.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Kolmogorov-Smirnov statistic D")
    ax.set_title("(a) Marginal divergence", fontsize=9.5)
    ax.axvline(0.1, color="#aaa", ls=":", lw=1)

    ax = axes[1]
    colours = {"stable": "#2a9d8f", "moderate shift": "#e9c46a", "major shift": "#e63946"}
    ax.barh(y, shift.psi, color=[colours[str(v)] for v in shift.psi_verdict])
    ax.axvline(0.1, color="#444", ls="--", lw=1)
    ax.axvline(0.25, color="#c1121f", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("population stability index")
    ax.set_title("(b) PSI against the usual thresholds", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, [f"{k} ({(shift.psi_verdict.astype(str) == k).sum()})"
                        for k in colours], loc="lower right", fontsize=7)

    ax = axes[2]
    ax.barh(y, shift.jensen_shannon_bits, color="#577590")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jensen-Shannon divergence (bits)")
    ax.set_title("(c) Divergence, 0 = identical, 1 = disjoint", fontsize=9.5)
    for yi, v in zip(y, shift.jensen_shannon_bits):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=7.5)

    fig.suptitle("Figure 12 — Per-feature distribution shift between CICIDS2017 and UNSW-NB15")
    fig.tight_layout()
    savefig(fig, "f12_marginal_shift")
    plt.close(fig)


def class_conditional_shift(logged: dict) -> pd.DataFrame:
    """Standardised mean difference per class and feature.

    The marginal gap could in principle be an artefact of the different class
    mixtures (Figure 3c). Computing the gap inside each class removes that
    explanation: if a class still moves, the traffic itself differs.
    """
    Xc, yc = logged["cicids2017"]
    Xu, yu = logged["unsw_nb15"]
    rows = []
    for cls in UNIFIED_CLASSES:
        mc, mu = (yc == cls).to_numpy(), (yu == cls).to_numpy()
        for f in Xc.columns:
            a, b = Xc.loc[mc, f].to_numpy(), Xu.loc[mu, f].to_numpy()
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows.append({"class": cls, "feature": f,
                         "smd": float((b.mean() - a.mean()) / sd) if sd > 0 else np.nan,
                         "n_cicids2017": int(mc.sum()), "n_unsw_nb15": int(mu.sum())})
    return pd.DataFrame(rows)


def fig_class_shift(cshift: pd.DataFrame) -> None:
    piv = cshift.pivot(index="feature", columns="class", values="smd")[UNIFIED_CLASSES]
    order = piv.abs().mean(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8),
                            gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    # Clipped at the 85th percentile of |SMD|: one cell (fwd_pkt_sz under
    # Reconnaissance) is large enough to flatten the colour scale for every
    # other cell. The printed numbers are unclipped.
    lim = max(2.0, float(np.nanpercentile(np.abs(piv.to_numpy()), 85)))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Benign", "DoS/\nDDoS", "Recon./\nProbe", "Exploits"], fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([FEATURE_SHORT[f] for f in piv.index], fontsize=7.8)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > lim * 0.55 else "#111")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                 label="standardised mean difference\n(UNSW-NB15 minus CICIDS2017)")
    ax.set_title("(a) Shift inside each class", fontsize=9.5)

    ax = axes[1]
    mag = piv.abs().mean(axis=0)
    bars = ax.bar(range(4), mag, color=[CLASS_COLOUR[c] for c in UNIFIED_CLASSES], width=0.6)
    for b, v in zip(bars, mag):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.8, color="#c1121f", ls="--", lw=1)
    ax.text(-0.44, 0.84, "Cohen 'large'", fontsize=7, color="#c1121f", ha="left")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Benign", "DoS/\nDDoS", "Recon./\nProbe", "Exploits"], fontsize=8)
    ax.set_ylabel("mean |SMD| over the twelve features")
    ax.set_title("(b) Which classes move most", fontsize=9.5)
    ax.set_ylim(0, max(mag) * 1.25)

    fig.suptitle("Figure 13 — Class-conditional shift: the gap is not an artefact "
                 "of the different class mixtures")
    fig.tight_layout()
    savefig(fig, "f13_class_shift")
    plt.close(fig)


def domain_classifier(logged: dict, n_per_side: int = 60_000) -> dict:
    """Train a classifier to recover which benchmark a flow came from.

    This is the sharpest single summary of the gap. If the two corpora were
    samples of the same traffic the task would be impossible and the AUC would
    sit at 0.5; the further above that it lands, the more any model fitted on
    one benchmark is free to key on features that merely identify the corpus.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    parts, domain = [], []
    for k, d in enumerate(DATASETS):
        X = logged[d][0]
        idx = rng.choice(X.shape[0], min(n_per_side, X.shape[0]), replace=False)
        parts.append(X.iloc[np.sort(idx)])
        domain.append(np.full(len(idx), k))
    X = pd.concat(parts, ignore_index=True)
    z = np.concatenate(domain)

    X_tr, X_te, z_tr, z_te = train_test_split(
        X, z, test_size=0.3, random_state=RANDOM_STATE, stratify=z)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         random_state=RANDOM_STATE)
    clf.fit(X_tr, z_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(z_te, proba))
    fpr, tpr, _ = roc_curve(z_te, proba)

    perm = permutation_importance(clf, X_te, z_te, n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    # Single-feature AUC: how much of that separability one feature carries on
    # its own, with no interactions available to it.
    solo = {}
    for f in X.columns:
        a = roc_auc_score(z_te, X_te[f].to_numpy())
        solo[f] = max(a, 1 - a)  # direction-free
    solo = pd.Series(solo).sort_values(ascending=False)

    return {"auc": auc, "fpr": fpr, "tpr": tpr, "importance": importance,
            "solo_auc": solo, "n": len(X)}


def fig_domain(dom: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.15, 1.15]})

    ax = axes[0]
    ax.plot(dom["fpr"], dom["tpr"], color="#2b2d42", lw=2)
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax.fill_between(dom["fpr"], dom["tpr"], dom["fpr"], color="#2b2d42", alpha=0.12)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("(a) Recovering the source benchmark", fontsize=9.5)
    ax.text(0.97, 0.12, f"AUC = {dom['auc']:.5f}", ha="right", fontsize=10, weight="bold",
            transform=ax.transAxes)
    ax.text(0.97, 0.04, "0.5 = indistinguishable", ha="right", fontsize=7.5, color="#666",
            transform=ax.transAxes)

    ax = axes[1]
    imp = dom["importance"][::-1]
    ax.barh(range(len(imp)), imp, color="#457b9d")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([FEATURE_SHORT[f] for f in imp.index], fontsize=7.8)
    ax.set_xlabel("drop in AUC when permuted")
    ax.set_title("(b) Which features give the corpus away", fontsize=9.5)

    ax = axes[2]
    solo = dom["solo_auc"][::-1]
    ax.barh(range(len(solo)), solo, color="#8ecae6")
    ax.axvline(0.5, color="#444", ls="--", lw=1)
    ax.set_yticks(range(len(solo)))
    ax.set_yticklabels([FEATURE_SHORT[f] for f in solo.index], fontsize=7.8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC from that feature alone")
    ax.set_title("(c) Single-feature separability", fontsize=9.5)
    for i, v in enumerate(solo):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=7)

    fig.suptitle("Figure 14 — A classifier can tell the two benchmarks apart almost perfectly")
    fig.tight_layout()
    savefig(fig, "f14_domain_separability")
    plt.close(fig)


def fig_joint_space(logged: dict) -> pd.DataFrame:
    """Where the two corpora sit relative to each other in two dimensions.

    PCA is fitted on CICIDS2017 alone and UNSW-NB15 is projected into it,
    mirroring how the parent project's source-fitted scaler is applied to the
    target: the picture then shows what the target looks like through the
    source's coordinate system, not through a compromise of the two.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    Xc, yc = logged["cicids2017"]
    Xu, yu = logged["unsw_nb15"]

    scaler = StandardScaler().fit(Xc)
    pca = PCA(n_components=min(12, Xc.shape[1])).fit(scaler.transform(Xc))
    Pc = pca.transform(scaler.transform(Xc))
    Pu = pca.transform(scaler.transform(Xu))

    evr = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })

    fig = plt.figure(figsize=(13.4, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, color="#457b9d")
    ax.plot(range(1, pca.n_components_ + 1), evr.cumulative, color="#e63946", marker="o", ms=3.5)
    ax.axhline(0.9, color="#888", ls=":", lw=1)
    k90 = int(np.searchsorted(evr.cumulative, 0.9) + 1)
    ax.text(pca.n_components_, 0.92, f"{k90} PCs reach 90%", ha="right", fontsize=7.5)
    ax.set_xlabel("component")
    ax.set_ylabel("variance explained")
    ax.set_title("(a) Scree, fitted on CICIDS2017", fontsize=9.5)
    ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(gs[0, 1:])
    load = pd.DataFrame(pca.components_[:3].T, index=Xc.columns,
                        columns=["PC1", "PC2", "PC3"])
    im = ax.imshow(load.to_numpy().T, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(Xc.columns)))
    ax.set_xticklabels([FEATURE_SHORT[f] for f in Xc.columns], rotation=90, fontsize=7.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["PC1", "PC2", "PC3"])
    ax.grid(False)
    for i in range(3):
        for j in range(len(Xc.columns)):
            ax.text(j, i, f"{load.iloc[j, i]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(load.iloc[j, i]) > 0.4 else "#111")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, label="loading")
    ax.set_title("(b) Loadings of the first three components", fontsize=9.5)

    # Balanced samples keep the scatter honest; an 80% benign corpus plotted
    # proportionally is a picture of benign traffic.
    Sc, sc_y = stratified_sample(pd.DataFrame(Pc[:, :2], columns=["PC1", "PC2"]), yc, 4000)
    Su, su_y = stratified_sample(pd.DataFrame(Pu[:, :2], columns=["PC1", "PC2"]), yu, 4000)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(Sc.PC1, Sc.PC2, s=4, alpha=0.3, color=DS_COLOUR["cicids2017"],
               label=PRETTY["cicids2017"], linewidths=0)
    ax.scatter(Su.PC1, Su.PC2, s=4, alpha=0.3, color=DS_COLOUR["unsw_nb15"],
               label=PRETTY["unsw_nb15"], linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(c) Both corpora in the source PCA space", fontsize=9.5)
    leg = ax.legend(markerscale=3, fontsize=7.5, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)

    # t-SNE on the pooled balanced sample: PCA is linear and could in principle
    # understate the overlap, so a non-linear embedding is used as a check.
    pooled = np.vstack([Pc[:, :6][np.sort(np.random.default_rng(RANDOM_STATE)
                        .choice(Pc.shape[0], 2500, replace=False))],
                        Pu[:, :6][np.sort(np.random.default_rng(RANDOM_STATE)
                        .choice(Pu.shape[0], 2500, replace=False))]])
    dom_lab = np.r_[np.zeros(2500, int), np.ones(2500, int)]
    emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=RANDOM_STATE,
               max_iter=750).fit_transform(pooled)

    ax = fig.add_subplot(gs[1, 1])
    for k, d in enumerate(DATASETS):
        m = dom_lab == k
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.4, color=DS_COLOUR[d],
                   label=PRETTY[d], linewidths=0)
    ax.set_title("(d) t-SNE of a pooled random sample", fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    leg = ax.legend(markerscale=3, fontsize=7.5, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)

    ax = fig.add_subplot(gs[1, 2])
    for cls in UNIFIED_CLASSES:
        m = (sc_y == cls).to_numpy()
        ax.scatter(Sc.PC1[m], Sc.PC2[m], s=5, alpha=0.45, color=CLASS_COLOUR[cls],
                   label=cls, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_title("(e) CICIDS2017 only, by class", fontsize=9.5)
    leg = ax.legend(markerscale=3, fontsize=6.8, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)

    fig.suptitle("Figure 15 — Principal-component structure and the position of the "
                 "target corpus within it")
    savefig(fig, "f15_pca_structure")
    plt.close(fig)

    savetable(load, "t21_pca_loadings")
    return evr


def run() -> None:
    logged = load_both(logged=True)

    shift = marginal_shift(logged)
    savetable(shift, "t18_marginal_shift", index=False)
    fig_marginal_shift(shift)
    print(shift[["feature", "ks_statistic", "psi", "jensen_shannon_bits", "psi_verdict"]]
          .to_string(index=False))

    cshift = class_conditional_shift(logged)
    savetable(cshift, "t19_class_conditional_shift", index=False)
    fig_class_shift(cshift)

    dom = domain_classifier(logged)
    fig_domain(dom)
    savetable(pd.DataFrame({"permutation_importance": dom["importance"],
                            "solo_auc": dom["solo_auc"]}), "t20_domain_classifier")
    print(f"\ndomain classifier AUC = {dom['auc']:.4f} on {dom['n']:,} flows")

    evr = fig_joint_space(logged)
    savetable(evr, "t22_pca_variance", index=False)

    headline = pd.DataFrame([{
        "mean_ks": shift.ks_statistic.mean(),
        "features_with_major_psi": int((shift.psi > 0.25).sum()),
        "features_with_ks_over_0.3": int((shift.ks_statistic > 0.3).sum()),
        "mean_jsd_bits": shift.jensen_shannon_bits.mean(),
        "domain_auc": dom["auc"],
        "mean_abs_smd_within_class": cshift.smd.abs().mean(),
        "pcs_for_90pct_variance": int(np.searchsorted(evr.cumulative, 0.9) + 1),
    }])
    savetable(headline, "t23_shift_headline", index=False)
    print(headline.T.to_string())


if __name__ == "__main__":
    from src.eda.settings import apply_style
    apply_style()
    run()
