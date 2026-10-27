"""Paths, constants and the shared matplotlib style for the EDA.

Everything the EDA writes goes under `eda_results/`, kept separate from
`results/`, which holds the modelling output carried over from the parent
project. Nothing in this package writes to `results/`.
"""
from __future__ import annotations

import os

import matplotlib as mpl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CICIDS_DIR = "C:/Users/ujjwa/Downloads/archive"
UNSW_DIR = "C:/Users/ujjwa/Downloads/archive (1)"

CACHE_DIR = os.path.join(ROOT, "eda_results", "cache")
FIG_DIR = os.path.join(ROOT, "eda_results", "figures")
TAB_DIR = os.path.join(ROOT, "eda_results", "tables")
MODEL_TAB_DIR = os.path.join(ROOT, "results", "tables")

for _d in (CACHE_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

DATASETS = ("cicids2017", "unsw_nb15")
PRETTY = {"cicids2017": "CICIDS2017", "unsw_nb15": "UNSW-NB15"}

# One colour per dataset, used consistently in every figure so a reader can
# follow a dataset across the whole report without re-reading the legend.
DS_COLOUR = {"cicids2017": "#1f6feb", "unsw_nb15": "#d1495b"}
CLASS_COLOUR = {
    "Benign": "#2a9d8f",
    "DoS/DDoS": "#e76f51",
    "Reconnaissance/Probe": "#457b9d",
    "Exploits": "#c77dff",
}

UNIFIED_CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance/Probe", "Exploits"]

# Short axis labels. The full names are carried in the tables; figures with
# twelve ticks need something that fits.
FEATURE_SHORT = {
    "duration": "dur",
    "fwd_packets": "fwd_pkts",
    "bwd_packets": "bwd_pkts",
    "fwd_bytes": "fwd_bytes",
    "bwd_bytes": "bwd_bytes",
    "flow_byte_rate": "byte_rate",
    "flow_packet_rate": "pkt_rate",
    "avg_fwd_pkt_size": "fwd_pkt_sz",
    "avg_bwd_pkt_size": "bwd_pkt_sz",
    "fwd_iat_mean": "fwd_iat",
    "bwd_iat_mean": "bwd_iat",
    "down_up_ratio": "dn_up_ratio",
}

FEATURE_UNITS = {
    "duration": "s",
    "fwd_packets": "packets",
    "bwd_packets": "packets",
    "fwd_bytes": "bytes",
    "bwd_bytes": "bytes",
    "flow_byte_rate": "bytes/s",
    "flow_packet_rate": "packets/s",
    "avg_fwd_pkt_size": "bytes",
    "avg_bwd_pkt_size": "bytes",
    "fwd_iat_mean": "s",
    "bwd_iat_mean": "s",
    "down_up_ratio": "ratio",
}

RANDOM_STATE = 42


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
    })


def savefig(fig, name: str) -> str:
    """Write a figure as 300-dpi PNG (for the report) and PDF (for LaTeX).

    Any in-figure title is cleared first: the report numbers and captions every
    figure itself, and a paper never carries the figure number twice. The stage
    modules still set one, so a figure read on its own stays self-describing.
    """
    if fig._suptitle is not None:
        fig.suptitle("")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    print(f"  figure -> {os.path.relpath(png, ROOT)}")
    return png


def savetable(df, name: str, **kwargs) -> str:
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, **kwargs)
    print(f"  table  -> {os.path.relpath(path, ROOT)}")
    return path
