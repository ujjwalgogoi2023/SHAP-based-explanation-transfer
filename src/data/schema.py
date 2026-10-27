"""
Feature and label harmonisation schema for CICIDS2017 <-> UNSW-NB15.

The two datasets do not share a feature schema (CICFlowMeter vs Argus/Zeek
derived features), so every shared feature below is defined as a *derivation*
computed from each dataset's raw columns rather than a raw column rename.
Where a feature can only be approximated on one side (e.g. TCP flag counts,
which UNSW-NB15 does not expose directly), that is called out explicitly in
`APPROXIMATE_FEATURES` — this is a documented modelling decision, not a bug.

Only features computable from *both* datasets' publicly documented raw
columns are included, per the proposal's harmonisation strategy (Sec 3.2):
connection duration, forward/backward packet counts, byte volumes,
packet-rate statistics, and TCP flag counts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

EPS = 1e-6
MICROSECONDS_PER_SECOND = 1e6
MILLISECONDS_PER_SECOND = 1e3


def canonical_label(s: str) -> str:
    """Collapse a raw label to a comparison-safe canonical form.

    CICIDS2017 ships the Web Attack labels with a cp1252 en-dash (byte 0x96)
    that is not valid UTF-8. Depending on the encoding used to read the CSV it
    surfaces as '\\x96', '\\ufffd', or an en-dash, so matching on the literal
    string is unreliable and silently differs across machines. Reducing to
    [A-Za-z0-9 ] makes the lookup encoding-independent.
    """
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

# ---------------------------------------------------------------------------
# Raw column name references (for documentation / sanity checks only).
# CICFlowMeter output ships with leading spaces on most column names.
# ---------------------------------------------------------------------------
CICIDS_LABEL_COL = " Label"
UNSW_LABEL_COL = "attack_cat"
UNSW_BINARY_LABEL_COL = "label"


def _cicids_col(df: pd.DataFrame, name: str) -> pd.Series:
    """CICIDS2017 CSVs are inconsistent about leading whitespace across
    dataset releases; resolve the column regardless of stray spaces."""
    if name in df.columns:
        return df[name]
    stripped = {c.strip(): c for c in df.columns}
    if name.strip() in stripped:
        return df[stripped[name.strip()]]
    raise KeyError(f"Column '{name}' not found in CICIDS2017 dataframe")


# ---------------------------------------------------------------------------
# Shared harmonised feature vector.
# Each entry: name -> (description, cicids_fn, unsw_fn, approximate)
# ---------------------------------------------------------------------------

def _cicids_duration(df):
    return _cicids_col(df, " Flow Duration").astype(float) / 1e6  # microsec -> sec


def _unsw_duration(df):
    return df["dur"].astype(float)


def _cicids_fwd_packets(df):
    return _cicids_col(df, " Total Fwd Packets").astype(float)


def _unsw_fwd_packets(df):
    return df["spkts"].astype(float)


def _cicids_bwd_packets(df):
    return _cicids_col(df, " Total Backward Packets").astype(float)


def _unsw_bwd_packets(df):
    return df["dpkts"].astype(float)


def _cicids_fwd_bytes(df):
    return _cicids_col(df, "Total Length of Fwd Packets").astype(float)


def _unsw_fwd_bytes(df):
    return df["sbytes"].astype(float)


def _cicids_bwd_bytes(df):
    return _cicids_col(df, " Total Length of Bwd Packets").astype(float)


def _unsw_bwd_bytes(df):
    return df["dbytes"].astype(float)


# Rates are recomputed from primitives on BOTH sides rather than read from each
# dataset's own rate column. CICFlowMeter's "Flow Packets/s" is total packets
# per second, but UNSW-NB15's `rate` is a different quantity — it correlates
# only r=0.77 with (spkts+dpkts)/dur on the real data, so pairing the two
# columns would compare semantically different measures and attribute the
# difference to the model. Deriving both identically removes that confound.

def _cicids_flow_byte_rate(df):
    total_bytes = _cicids_fwd_bytes(df) + _cicids_bwd_bytes(df)
    return total_bytes / (_cicids_duration(df) + EPS)


def _unsw_flow_byte_rate(df):
    total_bytes = df["sbytes"].astype(float) + df["dbytes"].astype(float)
    return total_bytes / (df["dur"].astype(float) + EPS)


def _cicids_flow_packet_rate(df):
    total_pkts = _cicids_fwd_packets(df) + _cicids_bwd_packets(df)
    return total_pkts / (_cicids_duration(df) + EPS)


def _unsw_flow_packet_rate(df):
    total_pkts = df["spkts"].astype(float) + df["dpkts"].astype(float)
    return total_pkts / (df["dur"].astype(float) + EPS)


# Denominator guards use +1 on counts rather than +EPS. Dividing by a 1e-6
# epsilon turns any zero denominator into a ~1e6-fold spike: on the real data
# that gave down_up_ratio a CICIDS std of 1.4e6 against an IQR of 4.3, and
# after the source-fitted scaler the feature collapsed to a constant on the
# target (std 0.00), making its SHAP rank there arbitrary noise. Counts and
# byte volumes are integers, so +1 is both bounded and interpretable.

def _cicids_avg_fwd_pkt_size(df):
    fwd_bytes = _cicids_fwd_bytes(df)
    fwd_pkts = _cicids_fwd_packets(df)
    return fwd_bytes / (fwd_pkts + 1.0)


def _unsw_avg_fwd_pkt_size(df):
    return df["smean"].astype(float)


def _cicids_avg_bwd_pkt_size(df):
    bwd_bytes = _cicids_bwd_bytes(df)
    bwd_pkts = _cicids_bwd_packets(df)
    return bwd_bytes / (bwd_pkts + 1.0)


def _unsw_avg_bwd_pkt_size(df):
    return df["dmean"].astype(float)


# CICFlowMeter reports inter-arrival times in MICROseconds; UNSW-NB15 reports
# sinpkt/dinpkt in MILLIseconds. Both are converted to seconds. Leaving the
# CICIDS side raw (an earlier bug here) put the two datasets ~437,000x apart on
# this feature, which would have dominated the normalised input and wrecked
# both transfer performance and the SHAP rankings.

def _cicids_fwd_iat_mean(df):
    return _cicids_col(df, "Fwd IAT Mean").astype(float) / MICROSECONDS_PER_SECOND


def _unsw_fwd_iat_mean(df):
    return df["sinpkt"].astype(float) / MILLISECONDS_PER_SECOND


def _cicids_bwd_iat_mean(df):
    return _cicids_col(df, "Bwd IAT Mean").astype(float) / MICROSECONDS_PER_SECOND


def _unsw_bwd_iat_mean(df):
    return df["dinpkt"].astype(float) / MILLISECONDS_PER_SECOND


def _cicids_down_up_ratio(df):
    fwd_bytes = _cicids_fwd_bytes(df)
    bwd_bytes = _cicids_bwd_bytes(df)
    return bwd_bytes / (fwd_bytes + 1.0)


def _unsw_down_up_ratio(df):
    return df["dbytes"].astype(float) / (df["sbytes"].astype(float) + 1.0)


# DROPPED: fin_flag_indicator.
# The proposal listed TCP flag counts among the shared features, but the two
# datasets do not measure the same thing and the empirical distributions are
# not comparable:
#     CICIDS  mean=0.0032   99.22% of rows exactly zero  (near-constant)
#     UNSW    mean=0.4547   54.53% of rows exactly zero  (balanced binary)
# CICFlowMeter's FIN Flag Count counts FIN flags observed mid-flow, whereas
# UNSW-NB15 exposes no per-packet flags at all — only a terminating connection
# `state`, where FIN means the connection closed cleanly. A feature that is
# constant on one dataset carries no rank information there, so any
# cross-dataset difference in its SHAP rank would reflect the encoding rather
# than explanation instability. Including it would manufacture exactly the
# effect this study is trying to measure. Report as a limitation: no TCP
# flag-level feature is jointly available across these two benchmarks.

FEATURE_SPECS = {
    "duration": ("Flow duration in seconds", _cicids_duration, _unsw_duration, False),
    "fwd_packets": ("Forward packet count", _cicids_fwd_packets, _unsw_fwd_packets, False),
    "bwd_packets": ("Backward packet count", _cicids_bwd_packets, _unsw_bwd_packets, False),
    "fwd_bytes": ("Forward byte volume", _cicids_fwd_bytes, _unsw_fwd_bytes, False),
    "bwd_bytes": ("Backward byte volume", _cicids_bwd_bytes, _unsw_bwd_bytes, False),
    "flow_byte_rate": ("Bytes/sec over the flow", _cicids_flow_byte_rate, _unsw_flow_byte_rate, False),
    "flow_packet_rate": ("Packets/sec over the flow", _cicids_flow_packet_rate, _unsw_flow_packet_rate, False),
    "avg_fwd_pkt_size": ("Mean forward packet size", _cicids_avg_fwd_pkt_size, _unsw_avg_fwd_pkt_size, False),
    "avg_bwd_pkt_size": ("Mean backward packet size", _cicids_avg_bwd_pkt_size, _unsw_avg_bwd_pkt_size, False),
    "fwd_iat_mean": ("Mean forward inter-arrival time", _cicids_fwd_iat_mean, _unsw_fwd_iat_mean, False),
    "bwd_iat_mean": ("Mean backward inter-arrival time", _cicids_bwd_iat_mean, _unsw_bwd_iat_mean, False),
    "down_up_ratio": ("Backward/forward byte ratio", _cicids_down_up_ratio, _unsw_down_up_ratio, False),
}

SHARED_FEATURES = list(FEATURE_SPECS.keys())
APPROXIMATE_FEATURES = [k for k, v in FEATURE_SPECS.items() if v[3]]

# ---------------------------------------------------------------------------
# Unified label taxonomy
# ---------------------------------------------------------------------------
# A cross-dataset comparison is only meaningful when the two sides of a class
# denote the same attack behaviour. An earlier draft of this map collapsed 8
# CICIDS labels and 7 UNSW labels into a single "Exploits" bucket, which put
# credential brute-forcing, botnet C2, protocol fuzzing and block-cipher
# attacks under one name. Comparing SHAP rankings for such a bucket compares
# different attacks that happen to share a label we invented, so a low
# consistency score would be explained by the taxonomy rather than by anything
# about SHAP.
#
# Categories with no defensible counterpart in the other dataset are therefore
# mapped to None and DROPPED, rather than being silently absorbed:
#
#   CICIDS excluded:
#     Bot           botnet C2 traffic; UNSW-NB15 has no botnet category
#     FTP-Patator   credential brute force; no UNSW counterpart
#     SSH-Patator   credential brute force; no UNSW counterpart
#   UNSW excluded:
#     Generic       block-cipher attacks, largely synthetic; 58,871 records
#                   with no CICIDS analogue — the single biggest source of
#                   contamination in the old bucket
#     Fuzzers       protocol fuzzing; closer to probing than exploitation
#     Analysis      mixed spam / port-scan / HTML penetration
#     Worms         self-propagating; n=174 and no CICIDS counterpart
#
# CLASS_CORRESPONDENCE records how tight each surviving pairing is, so the
# paper can report per-class results with the appropriate confidence.

UNIFIED_CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance/Probe", "Exploits"]

CLASS_CORRESPONDENCE = {
    "Benign":               "tight",   # background traffic on both sides
    "DoS/DDoS":             "tight",   # volumetric denial of service on both
    "Reconnaissance/Probe": "tight",   # PortScan <-> Reconnaissance
    "Exploits":             "loose",   # see note below; report with caveat
}

# The Exploits pairing is the weakest of the four and is deliberately narrowed
# to genuine exploitation of a software vulnerability. Even narrowed it is
# unbalanced (~2.2k CICIDS vs ~48k UNSW records) and the two sides emphasise
# different vulnerability classes — web-application flaws on the CICIDS side,
# service/protocol exploits on the UNSW side. Treat per-class results for
# Exploits as exploratory and say so in the write-up.

CICIDS_LABEL_MAP = {
    "BENIGN": "Benign",

    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DDoS": "DoS/DDoS",

    "PortScan": "Reconnaissance/Probe",

    # Genuine exploitation of a vulnerability. Encoding variants of the Web
    # Attack labels are handled by canonical_label(), but both spellings are
    # listed for readability.
    "Web Attack – Brute Force": "Exploits",
    "Web Attack – XSS": "Exploits",
    "Web Attack – Sql Injection": "Exploits",
    "Web Attack Brute Force": "Exploits",
    "Web Attack XSS": "Exploits",
    "Web Attack Sql Injection": "Exploits",
    "Heartbleed": "Exploits",
    "Infiltration": "Exploits",

    # Excluded — no defensible UNSW-NB15 counterpart.
    "Bot": None,
    "FTP-Patator": None,
    "SSH-Patator": None,
}

UNSW_LABEL_MAP = {
    "Normal": "Benign",

    "DoS": "DoS/DDoS",

    "Reconnaissance": "Reconnaissance/Probe",

    "Exploits": "Exploits",
    "Shellcode": "Exploits",
    "Backdoor": "Exploits",
    "Backdoors": "Exploits",

    # Excluded — no defensible CICIDS2017 counterpart.
    "Generic": None,
    "Fuzzers": None,
    "Analysis": None,
    "Worms": None,
}


# Canonicalised lookups, built once so encoding variants all resolve.
_CICIDS_CANON = {canonical_label(k): v for k, v in CICIDS_LABEL_MAP.items()}
_UNSW_CANON = {canonical_label(k): v for k, v in UNSW_LABEL_MAP.items()}


def _harmonise(raw: pd.Series, canon_map: dict, dataset: str) -> pd.Series:
    """Map raw labels to the unified taxonomy.

    Returns NaN for categories deliberately excluded (mapped to None); the
    caller is expected to drop those rows. A label that is absent from the map
    entirely is an error, not an exclusion — otherwise a renamed or newly
    added class would be dropped silently.
    """
    canon = raw.map(canonical_label)
    unrecognised = sorted(set(canon.unique()) - set(canon_map.keys()))
    if unrecognised:
        raise ValueError(
            f"Unrecognised {dataset} labels (not in the map as either a class "
            f"or an explicit exclusion): {unrecognised}"
        )
    return canon.map(canon_map)


def harmonise_cicids_labels(raw_labels: pd.Series) -> pd.Series:
    return _harmonise(raw_labels, _CICIDS_CANON, "CICIDS2017")


def harmonise_unsw_labels(raw_attack_cat: pd.Series) -> pd.Series:
    canon_ready = raw_attack_cat.astype(str).replace({"nan": "Normal", "": "Normal"})
    return _harmonise(canon_ready, _UNSW_CANON, "UNSW-NB15")


def build_feature_matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Compute the shared harmonised feature matrix for one dataset.

    dataset: 'cicids2017' or 'unsw_nb15'
    """
    if dataset == "cicids2017":
        fn_index = 1
    elif dataset == "unsw_nb15":
        fn_index = 2
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    out = {}
    for name, spec in FEATURE_SPECS.items():
        fn = spec[fn_index]
        out[name] = fn(df)
    feature_df = pd.DataFrame(out)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df
