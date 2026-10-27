"""
Level-2 explanation consistency metrics (proposal Sec 4.2 — core contribution).

Given per-class SHAP rankings E1 (source) and E2 (target), quantify how far
they agree: Spearman rho, Kendall tau, Jaccard@k, Rank-Biased Overlap, mean
absolute deviation of attribution magnitude, and an aggregate Explanation
Consistency Score (ECS).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _ranks_from_ranking(ranking: list, feature_names: list) -> np.ndarray:
    """Rank position (0 = most important) for each feature in feature_names,
    following the order given in `ranking`."""
    rank_of = {f: i for i, f in enumerate(ranking)}
    return np.array([rank_of[f] for f in feature_names])


def spearman_rho(ranking1: list, ranking2: list, feature_names: list) -> float:
    r1 = _ranks_from_ranking(ranking1, feature_names)
    r2 = _ranks_from_ranking(ranking2, feature_names)
    rho, _ = spearmanr(r1, r2)
    return float(rho)


def kendall_tau(ranking1: list, ranking2: list, feature_names: list) -> float:
    r1 = _ranks_from_ranking(ranking1, feature_names)
    r2 = _ranks_from_ranking(ranking2, feature_names)
    tau, _ = kendalltau(r1, r2)
    return float(tau)


def jaccard_at_k(ranking1: list, ranking2: list, k: int) -> float:
    """Overlap of the top-k features.

    WARNING: k must be meaningfully smaller than the number of features or this
    metric is degenerate. At k >= n_features both slices are the whole feature
    set, so Jaccard is identically 1.0 no matter how strongly the rankings
    disagree. Even at k = n_features - 3 the floor is already ~0.54. With the 13
    harmonised features here, only k <= 6 or so actually discriminates. See
    `validate_top_k`.
    """
    top1, top2 = set(ranking1[:k]), set(ranking2[:k])
    union = top1 | top2
    if not union:
        return 0.0
    return len(top1 & top2) / len(union)


def validate_top_k(top_k, n_features: int) -> list:
    """Drop k values that cannot discriminate, so a constant 1.0 does not get
    averaged into the Explanation Consistency Score and inflate it."""
    import warnings

    usable = [k for k in top_k if k < n_features]
    dropped = [k for k in top_k if k >= n_features]
    if dropped:
        warnings.warn(
            f"Dropping top-k values {dropped}: with only {n_features} features, "
            f"Jaccard@k is identically 1.0 for k >= {n_features} regardless of "
            f"how much the rankings disagree, which would inflate the ECS.",
            stacklevel=2,
        )
    if not usable:
        raise ValueError(f"No usable top-k values: all of {top_k} are >= n_features={n_features}")
    return usable


def rank_biased_overlap(ranking1: list, ranking2: list, p: float = 0.9) -> float:
    """Webber et al. (2010) RBO, extrapolated form. Tolerates non-conjoint
    lists (features present in one ranking but not the other)."""
    s, l = (ranking1, ranking2) if len(ranking1) <= len(ranking2) else (ranking2, ranking1)
    short_len, long_len = len(s), len(l)
    if short_len == 0:
        return 0.0

    ss, ls = set(), set()
    overlap = 0
    rbo_sum = 0.0
    for d in range(1, long_len + 1):
        if d <= short_len:
            ss.add(s[d - 1])
        ls.add(l[d - 1])
        overlap = len(ss & ls)
        agreement_d = overlap / d
        rbo_sum += (p ** (d - 1)) * agreement_d

    # Extrapolated RBO (Webber et al. 2010, eq. 32 approximation): weights
    # the observed overlap up to the shorter list's length, then extrapolates
    # the tail assuming the final overlap ratio holds for the remainder.
    rbo = (1 - p) * rbo_sum + (p ** short_len) * (overlap / short_len if short_len > 0 else 0.0)
    return float(min(max(rbo, 0.0), 1.0))


def mean_abs_deviation(importance1: dict, importance2: dict, feature_names: list) -> float:
    v1 = np.array([importance1.get(f, 0.0) for f in feature_names])
    v2 = np.array([importance2.get(f, 0.0) for f in feature_names])
    v1 = v1 / (v1.sum() + 1e-12)
    v2 = v2 / (v2.sum() + 1e-12)
    return float(np.mean(np.abs(v1 - v2)))


def explanation_consistency_score(
    ranking1: list,
    ranking2: list,
    importance1: dict,
    importance2: dict,
    feature_names: list,
    top_k=(5, 10, 20),
) -> dict:
    rho = spearman_rho(ranking1, ranking2, feature_names)
    tau = kendall_tau(ranking1, ranking2, feature_names)
    usable_k = validate_top_k(top_k, len(feature_names))
    jaccards = {f"jaccard@{k}": jaccard_at_k(ranking1, ranking2, k) for k in usable_k}
    rbo = rank_biased_overlap(ranking1, ranking2)
    mad = mean_abs_deviation(importance1, importance2, feature_names)

    # Aggregate: mean of normalised-to-[0,1] agreement metrics; MAD enters as
    # (1 - MAD) since it is a disagreement measure. Spearman/Kendall are
    # rescaled from [-1,1] to [0,1] first.
    rho_norm = (rho + 1) / 2
    tau_norm = (tau + 1) / 2
    jaccard_mean = np.mean(list(jaccards.values()))
    ecs = float(np.mean([rho_norm, tau_norm, jaccard_mean, rbo, 1 - min(mad, 1.0)]))

    return {
        "spearman_rho": rho,
        "kendall_tau": tau,
        **jaccards,
        "rbo": rbo,
        "mad": mad,
        "explanation_consistency_score": ecs,
    }


def compute_all_class_consistency(source_rankings: dict, target_rankings: dict, feature_names: list, top_k=(5, 10, 20)) -> dict:
    """source_rankings / target_rankings: {class_name: {"ranking":..., "importance":...}}"""
    out = {}
    for class_name in source_rankings:
        if class_name not in target_rankings:
            continue
        out[class_name] = explanation_consistency_score(
            source_rankings[class_name]["ranking"],
            target_rankings[class_name]["ranking"],
            source_rankings[class_name]["importance"],
            target_rankings[class_name]["importance"],
            feature_names,
            top_k,
        )
    return out
