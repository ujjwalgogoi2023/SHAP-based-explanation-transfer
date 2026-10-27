"""
Preprocessing pipeline: raw dataset CSVs -> harmonised, normalised,
train/val/test splits saved as parquet.

Usage:
    python -m src.data.preprocess --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.schema import (
    SHARED_FEATURES,
    build_feature_matrix,
    harmonise_cicids_labels,
    harmonise_unsw_labels,
    CICIDS_LABEL_COL,
    UNSW_LABEL_COL,
)


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list


def _read_all_csvs(directory: str, pattern: str = "*.csv") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files matching '{pattern}' found in {directory}. Download the raw "
            f"dataset there first (see README.md for the source links)."
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False, encoding="latin1")
        df.columns = [str(c).lstrip("﻿") for c in df.columns]
        frames.append(df)

    # Concatenating files with different schemas silently produces a frame full
    # of NaN columns rather than an error, so refuse it explicitly.
    col_counts = {len(f.columns) for f in frames}
    if len(col_counts) > 1:
        detail = ", ".join(f"{os.path.basename(p)}={len(f.columns)}" for p, f in zip(paths, frames))
        raise ValueError(
            f"Inconsistent column counts across files in {directory}: {detail}. "
            f"Narrow the glob pattern so only one dataset variant is loaded."
        )
    return pd.concat(frames, ignore_index=True)


def load_cicids2017(directory: str) -> tuple[pd.DataFrame, pd.Series]:
    df = _read_all_csvs(directory)
    df.columns = [c if c.startswith(" ") else c for c in df.columns]  # keep as-is; schema.py handles whitespace
    raw_labels = df[CICIDS_LABEL_COL] if CICIDS_LABEL_COL in df.columns else df[CICIDS_LABEL_COL.strip()]
    features = build_feature_matrix(df, "cicids2017")
    labels = harmonise_cicids_labels(raw_labels)
    return features, labels


def load_unsw_nb15(directory: str) -> tuple[pd.DataFrame, pd.Series]:
    # Only the named train/test partition is usable. The same distribution also
    # ships UNSW-NB15_1..4.csv (49 columns, NO header row) plus metadata files
    # (NUSW-NB15_features.csv, UNSW-NB15_LIST_EVENTS.csv). A bare *.csv glob
    # would concatenate all of them into garbage, so match explicitly.
    df = _read_all_csvs(directory, pattern="UNSW_NB15_*-set.csv")
    df.columns = [c.strip() for c in df.columns]
    features = build_feature_matrix(df, "unsw_nb15")
    labels = harmonise_unsw_labels(df[UNSW_LABEL_COL])
    return features, labels


def clean(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # Drop rows whose attack category has no defensible counterpart in the
    # other dataset (mapped to None in schema.py) alongside rows with
    # non-finite features.
    excluded = labels.isna()
    if excluded.any():
        print(f"  dropping {excluded.sum():,} rows in excluded attack categories")

    finite = features.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    mask = finite & ~excluded

    features = features.loc[mask].reset_index(drop=True)
    labels = labels.loc[mask].reset_index(drop=True)
    features = features.clip(lower=features.quantile(0.001), upper=features.quantile(0.999), axis=1)
    return features, labels


def log_transform(features: pd.DataFrame) -> pd.DataFrame:
    """log1p every feature.

    All 12 harmonised features are non-negative and heavy-tailed by several
    orders of magnitude — byte volumes, packet rates and inter-arrival times
    span 1e0 to 1e9 on both datasets. Standardising such features leaves the
    mean and variance dominated by a handful of extreme flows, which both
    destabilises training and makes the source-fitted scaler transfer badly to
    the target.

    This is a fixed, data-independent transform (no parameters estimated from
    either dataset), so unlike the scaler it introduces no train/target
    leakage and can safely be applied to both sides before fitting.
    """
    clipped = features.clip(lower=0.0)  # a few CICIDS IAT values are negative
    return np.log1p(clipped)


def subsample_per_class(features: pd.DataFrame, labels: pd.Series, cap: int, seed: int):
    rng = np.random.default_rng(seed)
    keep_idx = []
    for cls in labels.unique():
        idx = labels[labels == cls].index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    keep_idx.sort()
    return features.loc[keep_idx].reset_index(drop=True), labels.loc[keep_idx].reset_index(drop=True)


def make_splits(features: pd.DataFrame, labels: pd.Series, class_names: list, test_size: float, val_size: float, seed: int) -> SplitData:
    label_ids = labels.map({c: i for i, c in enumerate(class_names)}).to_numpy()
    X = features.to_numpy(dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, label_ids, test_size=(test_size + val_size), random_state=seed, stratify=label_ids
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=seed, stratify=y_temp
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test, class_names)


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)

    print("Loading CICIDS2017 ...")
    cic_features, cic_labels = load_cicids2017(cfg["data"]["cicids2017_dir"])
    cic_features, cic_labels = clean(cic_features, cic_labels)
    cic_features, cic_labels = subsample_per_class(
        cic_features, cic_labels, cfg["data"]["subsample_per_class"], cfg["seed"]
    )
    print(f"CICIDS2017: {len(cic_features)} rows after cleaning/subsampling")

    print("Loading UNSW-NB15 ...")
    unsw_features, unsw_labels = load_unsw_nb15(cfg["data"]["unsw_nb15_dir"])
    unsw_features, unsw_labels = clean(unsw_features, unsw_labels)
    unsw_features, unsw_labels = subsample_per_class(
        unsw_features, unsw_labels, cfg["data"]["subsample_per_class"], cfg["seed"]
    )
    print(f"UNSW-NB15: {len(unsw_features)} rows after cleaning/subsampling")

    class_names = sorted(set(cic_labels.unique()) | set(unsw_labels.unique()))

    source_name = cfg["data"]["source_dataset"]
    target_name = cfg["data"]["target_dataset"]
    datasets = {"cicids2017": (cic_features, cic_labels), "unsw_nb15": (unsw_features, unsw_labels)}
    source_features, source_labels = datasets[source_name]
    target_features, target_labels = datasets[target_name]

    # Compress the heavy tails first (parameter-free, applied to both sides),
    # then fit normalisation on the SOURCE dataset only and apply it unchanged
    # to the target (proposal Sec 3.2: "so that the transfer condition is
    # honest").
    source_logged = log_transform(source_features[SHARED_FEATURES])
    target_logged = log_transform(target_features[SHARED_FEATURES])

    scaler = StandardScaler()
    scaler.fit(source_logged)

    source_scaled = pd.DataFrame(scaler.transform(source_logged), columns=SHARED_FEATURES)
    target_scaled = pd.DataFrame(scaler.transform(target_logged), columns=SHARED_FEATURES)

    # A feature that is near-constant on the target after scaling carries no
    # rank information there, so its SHAP ranking would be arbitrary and would
    # register as spurious "explanation inconsistency". Surface it rather than
    # letting it quietly inflate the headline result.
    target_std = target_scaled.std()
    degenerate = target_std[target_std < 0.05]
    if len(degenerate) > 0:
        print("  WARNING: near-constant on target after scaling (SHAP rank unreliable):")
        for feat, sd in degenerate.items():
            print(f"    {feat}: std={sd:.4f}")

    source_splits = make_splits(
        source_scaled, source_labels, class_names,
        cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"],
    )
    target_label_ids = target_labels.map({c: i for i, c in enumerate(class_names)}).to_numpy()

    # The target is evaluated whole under transfer (no retraining), but it is
    # ALSO split so a separate model can be trained natively on it. That second
    # model is what makes the study interpretable: a source-trained model can
    # collapse to chance on the target (observed MCC ~0.01), and the SHAP
    # values of a model that predicts noise are not explanations of anything.
    # Comparing source-model-on-source against target-model-on-target asks the
    # actual research question -- do the same features matter for the same
    # attack in two datasets, when BOTH models are competent -- instead of
    # confounding it with model failure.
    target_splits = make_splits(
        target_scaled, target_labels, class_names,
        cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"],
    )

    out_dir = cfg["data"]["processed_dir"]
    np.savez(
        os.path.join(out_dir, f"{source_name}_source.npz"),
        X_train=source_splits.X_train, y_train=source_splits.y_train,
        X_val=source_splits.X_val, y_val=source_splits.y_val,
        X_test=source_splits.X_test, y_test=source_splits.y_test,
    )
    np.savez(
        os.path.join(out_dir, f"{target_name}_target.npz"),
        X=target_scaled.to_numpy(dtype=np.float32), y=target_label_ids,
        X_train=target_splits.X_train, y_train=target_splits.y_train,
        X_val=target_splits.X_val, y_val=target_splits.y_val,
        X_test=target_splits.X_test, y_test=target_splits.y_test,
    )
    with open(os.path.join(out_dir, "class_names.yaml"), "w") as f:
        yaml.safe_dump(class_names, f)
    print(f"Saved processed splits to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)
