"""
Statistical validation anchors (proposal Sec 4.3).

- Upper bound: explanation consistency between two independently seeded
  models trained on the SAME dataset (source vs source-reseeded).
- Lower bound: explanation consistency between a real ranking and a randomly
  permuted ranking (chance agreement).
- Wilcoxon signed-rank test: is observed cross-dataset consistency
  significantly greater than the random-permutation baseline, across seeds?
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from src.explain.consistency import explanation_consistency_score


def random_permutation_baseline(ranking: list, importance: dict, feature_names: list, seed: int,
                                top_k=(5, 10, 20)) -> dict:
    """Chance agreement: the ranking against a random permutation of itself.

    PASS THE SAME top_k USED FOR THE OBSERVED SCORES. The ECS averages a
    Jaccard@k term, and Jaccard@k has a floor that rises steeply with k once k
    approaches the feature count (at 12 features, Jaccard@10 cannot fall below
    0.67). Leaving this at the default while the observed values are scored at
    k=(3,5,7) builds the floor out of a different metric than the thing being
    compared to it, which inflates the floor and understates how far observed
    consistency sits above chance.

    Note that only the rank ORDER is permuted; the per-feature attribution
    magnitudes are carried over unchanged, so the MAD term contributes its
    maximum to the floor. The floor is conservative (high) by construction and
    the normalised consistency derived from it is correspondingly conservative.
    """
    rng = np.random.default_rng(seed)
    permuted_ranking = list(rng.permutation(ranking))
    permuted_importance = {f: importance[f] for f in permuted_ranking}
    return explanation_consistency_score(ranking, permuted_ranking, importance, permuted_importance,
                                         feature_names, top_k)


def same_dataset_upper_bound(rankings_seed_a: dict, rankings_seed_b: dict, feature_names: list, top_k=(5, 10, 20)) -> dict:
    """rankings_seed_a / rankings_seed_b: {class_name: {"ranking", "importance"}}
    from two independently seeded models trained on the same dataset."""
    out = {}
    for class_name in rankings_seed_a:
        if class_name not in rankings_seed_b:
            continue
        out[class_name] = explanation_consistency_score(
            rankings_seed_a[class_name]["ranking"],
            rankings_seed_b[class_name]["ranking"],
            rankings_seed_a[class_name]["importance"],
            rankings_seed_b[class_name]["importance"],
            feature_names,
            top_k,
        )
    return out


def wilcoxon_vs_random(observed_ecs: list, random_ecs: list) -> dict:
    """Paired Wilcoxon signed-rank test: does observed cross-dataset ECS
    (across seeds/classes) exceed the random-permutation baseline?

    Note on sample size: a one-sided signed-rank test cannot produce p < 0.05
    with fewer than 6 pairs (the smallest attainable p at n=5 is 1/2^5 = 0.031,
    at n=4 it is 0.0625). Pair over seeds x classes — 5 seeds x 4 classes = 20
    pairs — rather than over classes alone, or the test is underpowered by
    construction regardless of effect size.
    """
    observed_ecs = np.asarray(observed_ecs)
    random_ecs = np.asarray(random_ecs)
    if len(observed_ecs) != len(random_ecs):
        raise ValueError("observed_ecs and random_ecs must be paired (same length)")

    n_pairs = len(observed_ecs)
    underpowered = n_pairs < 6

    diffs = observed_ecs - random_ecs
    if np.allclose(diffs, 0):
        return {"statistic": float("nan"), "p_value": 1.0, "significant": False,
                "n_pairs": n_pairs, "underpowered": underpowered}

    stat, p = wilcoxon(observed_ecs, random_ecs, alternative="greater")
    return {"statistic": float(stat), "p_value": float(p),
            "significant": bool(p < 0.05), "n_pairs": n_pairs,
            "underpowered": underpowered}


def summarise_across_seeds(per_seed_values: list) -> dict:
    arr = np.asarray(per_seed_values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n_seeds": len(arr)}
