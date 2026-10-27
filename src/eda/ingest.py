"""Stage 1 — ingestion and the raw data-quality audit.

The two benchmarks are read file by file rather than concatenated up front:
CICIDS2017 ships eight CSVs totalling ~0.9 GB, and the audit statistics the
report needs (per-column missingness, non-finite values, zero inflation,
negative values, cardinality) are all additive over files, so there is no
reason to hold the whole thing in memory at once.

Two products come out of this stage:

1. `audit_<dataset>.csv` — one row per RAW column, before any harmonisation.
   This is the part of the EDA that has to happen on untouched data: once the
   twelve-feature matrix is built, a missing value has already become a
   dropped row and the evidence is gone.
2. `cache/<dataset>_harmonised.npz` — the twelve shared features on their
   original scale (no log, no scaling, no clipping) with the raw and unified
   labels alongside, so every later stage reads a small file instead of the
   CSVs.

Cleaning is applied *after* caching and is recorded as a row-attrition ledger,
so the report can state exactly how many records each decision removed.
"""
from __future__ import annotations

import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.data.schema import (
    FEATURE_SPECS,
    SHARED_FEATURES,
    build_feature_matrix,
    canonical_label,
    harmonise_cicids_labels,
    harmonise_unsw_labels,
)
from src.eda.settings import CACHE_DIR, CICIDS_DIR, TAB_DIR, UNIFIED_CLASSES, UNSW_DIR

MAX_TRACKED_UNIQUES = 64

CICIDS_PATTERN = "*.pcap_ISCX.csv"
UNSW_PATTERN = "UNSW_NB15_*-set.csv"

RAW_LABEL_COL = {"cicids2017": "Label", "unsw_nb15": "attack_cat"}

# Excluded from the duplicate-row hash. UNSW-NB15 carries a surrogate row id,
# which makes every record unique by construction and would report zero
# duplicates no matter what the traffic looked like.
ID_LIKE = {"id", "index", "unnamed: 0"}

DROPPED = "(dropped)"


class ColumnAudit:
    """Additive per-column quality counters, accumulated across files."""

    def __init__(self) -> None:
        self.n_rows = 0
        self.dtypes: dict[str, Counter] = {}
        self.missing: Counter = Counter()
        self.non_finite: Counter = Counter()
        self.zeros: Counter = Counter()
        self.negative: Counter = Counter()
        self.uniques: dict[str, set] = {}
        self.overflow: set[str] = set()
        self.minimum: dict[str, float] = {}
        self.maximum: dict[str, float] = {}

    def update(self, df: pd.DataFrame) -> None:
        self.n_rows += len(df)
        for col in df.columns:
            s = df[col]
            self.dtypes.setdefault(col, Counter())[str(s.dtype)] += 1
            self.missing[col] += int(s.isna().sum())

            if pd.api.types.is_numeric_dtype(s):
                arr = s.to_numpy(dtype="float64", na_value=np.nan)
                finite = np.isfinite(arr)
                self.non_finite[col] += int((~finite & ~np.isnan(arr)).sum())
                vals = arr[finite]
                self.zeros[col] += int((vals == 0).sum())
                self.negative[col] += int((vals < 0).sum())
                if vals.size:
                    lo, hi = float(vals.min()), float(vals.max())
                    self.minimum[col] = min(self.minimum.get(col, lo), lo)
                    self.maximum[col] = max(self.maximum.get(col, hi), hi)
                observed = np.unique(vals)
            else:
                observed = pd.unique(s.dropna())

            # Cardinality is tracked exactly up to a cap: the point is to find
            # columns that are constant or near-binary, and anything past the
            # cap is "high cardinality" regardless of the exact figure.
            if col not in self.overflow:
                seen = self.uniques.setdefault(col, set())
                seen.update(np.asarray(observed)[: MAX_TRACKED_UNIQUES + 1].tolist())
                if len(seen) > MAX_TRACKED_UNIQUES:
                    self.overflow.add(col)
                    self.uniques.pop(col, None)

    def to_frame(self) -> pd.DataFrame:
        cols = list(self.dtypes.keys())
        rows = []
        for c in cols:
            n_uniq = np.nan if c in self.overflow else len(self.uniques.get(c, ()))
            rows.append({
                "column": c,
                "dtype": "/".join(sorted(self.dtypes[c])),
                "missing_n": self.missing[c],
                "missing_pct": 100.0 * self.missing[c] / self.n_rows,
                "non_finite_n": self.non_finite[c],
                "zero_pct": 100.0 * self.zeros[c] / self.n_rows,
                "negative_n": self.negative[c],
                "n_unique": n_uniq,
                "min": self.minimum.get(c, np.nan),
                "max": self.maximum.get(c, np.nan),
            })
        out = pd.DataFrame(rows)
        out["constant"] = out["n_unique"] <= 1
        return out


def _file_list(dataset: str) -> list[str]:
    directory, pattern = (
        (CICIDS_DIR, CICIDS_PATTERN) if dataset == "cicids2017" else (UNSW_DIR, UNSW_PATTERN)
    )
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")
    return paths


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CICFlowMeter writes most column names with a leading space and some
    releases add a UTF-8 BOM to the first one. Both are invisible in a
    spreadsheet and both make an exact-name lookup fail, so strip them here and
    report the fact rather than carrying the whitespace through the analysis."""
    # The BOM has to be matched in both encodings: these files are read as
    # latin1 (the CICIDS labels are cp1252), so a UTF-8 BOM arrives as the
    # three characters 'ï»¿' rather than as U+FEFF.
    df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def ingest(dataset: str, verbose: bool = True) -> dict:
    """Read one benchmark, audit its raw columns, and build the shared matrix."""
    paths = _file_list(dataset)
    audit = ColumnAudit()
    inventory, feature_frames, raw_labels = [], [], []
    leading_space_cols: set[str] = set()
    repeated_names: list[str] = []
    row_hashes: set[int] = set()
    duplicate_rows = 0

    for p in paths:
        df = pd.read_csv(p, low_memory=False, encoding="latin1")
        leading_space_cols.update(c for c in df.columns if str(c) != str(c).strip())
        # pandas silently de-duplicates repeated header names by appending
        # '.1'; CICIDS2017 ships 'Fwd Header Length' twice, so the suffix is
        # the only surviving trace of a header-level defect.
        repeated_names = sorted({c[:-2] for c in map(str, df.columns)
                                 if re.fullmatch(r".*\.\d", c)} | set(repeated_names))
        df = _normalise_columns(df)
        audit.update(df)

        # Exact-duplicate flows, counted across the whole benchmark rather than
        # within each file: CICIDS2017's capture days are separate CSVs, so a
        # within-file check would miss anything repeated across days.
        payload = df.drop(columns=[c for c in df.columns if c.lower() in ID_LIKE])
        hashed = pd.util.hash_pandas_object(payload, index=False).to_numpy()
        before = len(row_hashes)
        row_hashes.update(hashed.tolist())
        duplicate_rows += len(hashed) - (len(row_hashes) - before)

        label_col = RAW_LABEL_COL[dataset]
        raw = df[label_col].astype("string").fillna("Normal")
        raw_labels.append(raw)
        feature_frames.append(build_feature_matrix(df, dataset))

        inventory.append({
            "dataset": dataset,
            "file": os.path.basename(p),
            "size_mb": os.path.getsize(p) / 1e6,
            "rows": len(df),
            "columns": df.shape[1],
        })
        if verbose:
            print(f"    {os.path.basename(p):<58s} {len(df):>9,d} rows x {df.shape[1]:>3d} cols")
        del df

    features = pd.concat(feature_frames, ignore_index=True)
    raw_label = pd.concat(raw_labels, ignore_index=True)
    harmoniser = harmonise_cicids_labels if dataset == "cicids2017" else harmonise_unsw_labels
    unified = harmoniser(raw_label)

    return {
        "features": features,
        "raw_label": raw_label,
        "unified_label": unified,
        "audit": audit.to_frame(),
        "inventory": pd.DataFrame(inventory),
        "leading_space_cols": sorted(leading_space_cols),
        "repeated_column_names": repeated_names,
        "duplicate_rows": duplicate_rows,
        "n_rows": len(features),
    }


def attrition_ledger(bundle: dict) -> pd.DataFrame:
    """Row attrition, decision by decision, in the order the pipeline applies
    them. Reported rather than silently absorbed, because two of the three
    steps change the class balance and one changes it by a factor of four."""
    f, unified = bundle["features"], bundle["unified_label"]
    n0 = len(f)

    excluded = unified.isna()
    non_finite = ~np.isfinite(f.to_numpy(dtype="float64")).all(axis=1)
    non_finite = pd.Series(non_finite, index=f.index)

    rows = [
        {"step": "raw records read", "removed": 0, "remaining": n0},
        {"step": "attack categories with no counterpart",
         "removed": int(excluded.sum()), "remaining": int((~excluded).sum())},
        {"step": "non-finite feature values",
         "removed": int((non_finite & ~excluded).sum()),
         "remaining": int((~excluded & ~non_finite).sum())},
    ]
    kept = ~excluded & ~non_finite
    rows.append({"step": "analysed records", "removed": 0, "remaining": int(kept.sum())})
    out = pd.DataFrame(rows)
    out["pct_of_raw"] = 100.0 * out["remaining"] / n0
    return out


def clean(bundle: dict) -> dict:
    """Drop excluded categories and non-finite rows. No clipping and no
    transform: those are modelling choices, and this stage's job is to hand the
    later stages the data as the sensors recorded it."""
    f, unified, raw = bundle["features"], bundle["unified_label"], bundle["raw_label"]
    keep = unified.notna() & np.isfinite(f.to_numpy(dtype="float64")).all(axis=1)
    return {
        "features": f.loc[keep].reset_index(drop=True),
        "unified_label": unified.loc[keep].reset_index(drop=True),
        "raw_label": raw.loc[keep].reset_index(drop=True),
    }


def label_frequency(bundle: dict) -> pd.DataFrame:
    """Raw label frequencies with the unified class each one maps to, so the
    taxonomy decision is auditable instead of asserted."""
    raw, unified = bundle["raw_label"], bundle["unified_label"]
    tab = pd.DataFrame({"raw_label": raw, "unified_class": unified})
    out = (tab.groupby("raw_label", dropna=False)
              .agg(n=("unified_class", "size"),
                   unified_class=("unified_class", lambda s: s.iloc[0]))
              .reset_index())
    out["unified_class"] = out["unified_class"].fillna(DROPPED)
    out["pct"] = 100.0 * out["n"] / out["n"].sum()
    out["canonical"] = out["raw_label"].map(canonical_label)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def cache_path(dataset: str) -> str:
    return os.path.join(CACHE_DIR, f"{dataset}_harmonised.npz")


def write_cache(dataset: str, cleaned: dict) -> None:
    classes = UNIFIED_CLASSES
    raw_levels = sorted(cleaned["raw_label"].unique().tolist())
    np.savez_compressed(
        cache_path(dataset),
        X=cleaned["features"][SHARED_FEATURES].to_numpy(dtype=np.float64),
        y=cleaned["unified_label"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int8),
        raw_y=cleaned["raw_label"].map({c: i for i, c in enumerate(raw_levels)}).to_numpy(np.int16),
        features=np.array(SHARED_FEATURES, dtype=object),
        classes=np.array(classes, dtype=object),
        raw_levels=np.array(raw_levels, dtype=object),
    )


def read_cache(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    d = np.load(cache_path(dataset), allow_pickle=True)
    feats = [str(x) for x in d["features"]]
    classes = [str(x) for x in d["classes"]]
    raw_levels = [str(x) for x in d["raw_levels"]]
    X = pd.DataFrame(d["X"], columns=feats)
    y = pd.Series(pd.Categorical.from_codes(d["y"], categories=classes), name="class")
    raw_y = pd.Series(pd.Categorical.from_codes(d["raw_y"], categories=raw_levels), name="raw_label")
    return X, y, raw_y


def feature_dictionary() -> pd.DataFrame:
    """The data dictionary: what each shared feature means and which raw
    columns it is derived from on each side. Needed because not one of the
    twelve is a column rename."""
    source_cols = {
        "duration": ("Flow Duration (microsec)", "dur (sec)"),
        "fwd_packets": ("Total Fwd Packets", "spkts"),
        "bwd_packets": ("Total Backward Packets", "dpkts"),
        "fwd_bytes": ("Total Length of Fwd Packets", "sbytes"),
        "bwd_bytes": ("Total Length of Bwd Packets", "dbytes"),
        "flow_byte_rate": ("(fwd+bwd bytes) / Flow Duration", "(sbytes+dbytes) / dur"),
        "flow_packet_rate": ("(fwd+bwd packets) / Flow Duration", "(spkts+dpkts) / dur"),
        "avg_fwd_pkt_size": ("Fwd bytes / (Fwd packets + 1)", "smean"),
        "avg_bwd_pkt_size": ("Bwd bytes / (Bwd packets + 1)", "dmean"),
        "fwd_iat_mean": ("Fwd IAT Mean (microsec)", "sinpkt (millisec)"),
        "bwd_iat_mean": ("Bwd IAT Mean (microsec)", "dinpkt (millisec)"),
        "down_up_ratio": ("Bwd bytes / (Fwd bytes + 1)", "dbytes / (sbytes + 1)"),
    }
    rows = []
    for name, spec in FEATURE_SPECS.items():
        cic, unsw = source_cols[name]
        rows.append({
            "feature": name,
            "description": spec[0],
            "cicids2017_derivation": cic,
            "unsw_nb15_derivation": unsw,
            "identical_column": "no",
        })
    return pd.DataFrame(rows)


def run() -> dict:
    """Execute the stage and write its artefacts."""
    bundles, cleaned = {}, {}
    inventories, audits, ledgers, labels = [], [], [], []

    for dataset in ("cicids2017", "unsw_nb15"):
        print(f"  reading {dataset} ...")
        b = ingest(dataset)
        bundles[dataset] = b

        a = b["audit"].copy()
        a.insert(0, "dataset", dataset)
        audits.append(a)
        inventories.append(b["inventory"])

        led = attrition_ledger(b)
        led.insert(0, "dataset", dataset)
        ledgers.append(led)

        lf = label_frequency(b)
        lf.insert(0, "dataset", dataset)
        labels.append(lf)

        c = clean(b)
        cleaned[dataset] = c
        write_cache(dataset, c)
        print(f"    kept {len(c['features']):,} of {b['n_rows']:,} records "
              f"({100 * len(c['features']) / b['n_rows']:.1f}%); "
              f"{b['duplicate_rows']:,} exact duplicate rows seen")

    inventory = pd.concat(inventories, ignore_index=True)
    inventory.to_csv(os.path.join(TAB_DIR, "t01_file_inventory.csv"), index=False)
    pd.concat(audits, ignore_index=True).to_csv(
        os.path.join(TAB_DIR, "t02_raw_column_audit.csv"), index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(
        os.path.join(TAB_DIR, "t03_attrition_ledger.csv"), index=False)
    pd.concat(labels, ignore_index=True).to_csv(
        os.path.join(TAB_DIR, "t04_label_frequency.csv"), index=False)
    feature_dictionary().to_csv(
        os.path.join(TAB_DIR, "t05_feature_dictionary.csv"), index=False)

    notes = pd.DataFrame([
        {"dataset": d,
         "files": len(bundles[d]["inventory"]),
         "raw_rows": bundles[d]["n_rows"],
         "raw_columns": int(bundles[d]["inventory"]["columns"].iloc[0]),
         "columns_with_leading_space": len(bundles[d]["leading_space_cols"]),
         "repeated_column_names": "; ".join(bundles[d]["repeated_column_names"]) or "none",
         "exact_duplicate_rows": bundles[d]["duplicate_rows"],
         "analysed_rows": len(cleaned[d]["features"])}
        for d in ("cicids2017", "unsw_nb15")
    ])
    notes.to_csv(os.path.join(TAB_DIR, "t06_ingest_notes.csv"), index=False)
    print(notes.to_string(index=False))
    return {"bundles": bundles, "cleaned": cleaned}


if __name__ == "__main__":
    run()
