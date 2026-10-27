"""Shared loading and statistics helpers for the analysis stages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.ingest import read_cache
from src.eda.settings import DATASETS, RANDOM_STATE, UNIFIED_CLASSES

_CACHE: dict[str, tuple] = {}


def load(dataset: str, logged: bool = False):
    """Return (features, class labels) for one benchmark.

    `logged=True` applies log1p after clipping negatives to zero. Every
    scale-sensitive analysis in the report uses the logged view, and Section 4
    shows the skewness evidence that makes that the right choice; the raw view
    is kept for descriptive statistics, which have to be reported in the units
    the sensors actually recorded.
    """
    if dataset not in _CACHE:
        _CACHE[dataset] = read_cache(dataset)
    X, y, raw_y = _CACHE[dataset]
    if logged:
        X = np.log1p(X.clip(lower=0.0))
    return X, y


def load_both(logged: bool = False) -> dict[str, tuple]:
    return {d: load(d, logged=logged) for d in DATASETS}


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int,
                      seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """Class-balanced subsample, capped by the rarest class.

    Used wherever an algorithm is superlinear in n (t-SNE, kernel estimates).
    Balanced rather than proportional, because a proportional draw from a
    corpus that is 80% benign tells us about benign traffic and nothing else.
    """
    rng = np.random.default_rng(seed)
    per_class = max(1, n // len(UNIFIED_CLASSES))
    idx = []
    for cls in UNIFIED_CLASSES:
        pool = np.flatnonzero((y == cls).to_numpy())
        if pool.size == 0:
            continue
        take = min(per_class, pool.size)
        idx.append(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def describe(X: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics, including the shape measures the report needs
    (skew, excess kurtosis, zero share, IQR) rather than only the five-number
    summary pandas gives by default."""
    from scipy import stats

    rows = []
    for col in X.columns:
        v = X[col].to_numpy(dtype="float64")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append({
            "feature": col,
            "n": v.size,
            "mean": v.mean(),
            "std": v.std(ddof=1),
            "min": v.min(),
            "q1": q1,
            "median": med,
            "q3": q3,
            "p99": np.percentile(v, 99),
            "max": v.max(),
            "iqr": q3 - q1,
            "skew": float(stats.skew(v)),
            "excess_kurtosis": float(stats.kurtosis(v)),
            "zero_pct": 100.0 * float((v == 0).mean()),
            "cv": v.std(ddof=1) / v.mean() if v.mean() else np.nan,
        })
    return pd.DataFrame(rows)


def outlier_rates(X: pd.DataFrame) -> pd.DataFrame:
    """Two outlier screens per feature, on the raw and the logged view.

    Tukey's rule and the 3-sigma rule disagree badly on heavy-tailed data, and
    the disagreement is itself the finding: on a log-normal-ish variable the
    3-sigma rule flags almost nothing while Tukey's flags a double-digit
    percentage, so neither can be used as a mechanical drop rule here.
    """
    rows = []
    for col in X.columns:
        v = X[col].to_numpy(dtype="float64")
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        tukey = float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean())
        sd = v.std(ddof=1)
        z = float((np.abs(v - v.mean()) > 3 * sd).mean()) if sd > 0 else 0.0

        lv = np.log1p(np.clip(v, 0, None))
        lq1, lq3 = np.percentile(lv, [25, 75])
        liqr = lq3 - lq1
        ltukey = float(((lv < lq1 - 1.5 * liqr) | (lv > lq3 + 1.5 * liqr)).mean())

        rows.append({
            "feature": col,
            "tukey_pct": 100 * tukey,
            "z3_pct": 100 * z,
            "tukey_log_pct": 100 * ltukey,
            "max_over_p99": (v.max() / np.percentile(v, 99)) if np.percentile(v, 99) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def quantile_bins(a: np.ndarray, b: np.ndarray, n_bins: int = 40) -> np.ndarray:
    """Bin edges built from the pooled sample so both datasets are binned
    identically — otherwise a divergence figure would partly measure the
    binning rather than the distributions."""
    pooled = np.concatenate([a, b])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        edges = np.array([pooled.min(), pooled.mean(), pooled.max() + 1e-9])
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def _hist(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(v, bins=edges)
    p = counts.astype("float64")
    return p / p.sum() if p.sum() else p


def jensen_shannon(a: np.ndarray, b: np.ndarray, n_bins: int = 40) -> float:
    """JS divergence in bits; 0 identical, 1 disjoint."""
    edges = quantile_bins(a, b, n_bins)
    p, q = _hist(a, edges), _hist(b, edges)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def population_stability_index(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """PSI with the conventional reading: <0.1 stable, 0.1-0.25 moderate
    shift, >0.25 major shift. Bins come from the reference (first) sample,
    which is the convention the thresholds were calibrated under."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    p, q = _hist(a, edges), _hist(b, edges)
    eps = 1e-6
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def mutual_information(X: pd.DataFrame, y: pd.Series, n_bins: int = 32) -> pd.Series:
    """Histogram mutual information I(feature; class) in bits.

    Quantile binning rather than sklearn's kNN estimator: at 2.8 million rows
    the kNN estimator is the slowest thing in the pipeline by two orders of
    magnitude, and for ranking twelve features against each other a binned
    estimate with a fixed bin count is both adequate and reproducible.
    """
    codes = pd.Categorical(y, categories=UNIFIED_CLASSES).codes
    py = np.bincount(codes[codes >= 0], minlength=len(UNIFIED_CLASSES)).astype("float64")
    py /= py.sum()
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))

    out = {}
    for col in X.columns:
        v = X[col].to_numpy(dtype="float64")
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            out[col] = 0.0
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.digitize(v, edges[1:-1])
        joint = np.zeros((b.max() + 1, len(UNIFIED_CLASSES)))
        np.add.at(joint, (b, codes), 1.0)
        joint /= joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        pyj = joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        mi = float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ pyj)[nz])))
        out[col] = mi / hy if hy > 0 else 0.0  # normalised: share of class entropy explained
    return pd.Series(out, name="normalised_mi")


def model_results() -> dict:
    """Headline numbers carried over from the parent modelling project, used
    only in the closing section where the EDA findings are checked against
    what the model actually did."""
    import os

    import yaml

    from src.eda.settings import MODEL_TAB_DIR

    with open(os.path.join(MODEL_TAB_DIR, "headline.yaml")) as f:
        head = yaml.safe_load(f)
    cons = pd.read_csv(os.path.join(MODEL_TAB_DIR, "consistency_summary.csv"))
    det = pd.read_csv(os.path.join(MODEL_TAB_DIR, "detection_summary.csv"), header=[0, 1], index_col=0)
    return {"headline": head, "consistency": cons, "detection": det}
