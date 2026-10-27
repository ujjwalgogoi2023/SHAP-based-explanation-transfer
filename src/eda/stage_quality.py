"""Stage 2 — data-quality findings, drawn from the raw column audit.

Nothing here touches the harmonised matrix. The point of the stage is that
several defects are only visible before harmonisation: a column that is
constant disappears once it is dropped, an impossible negative duration
disappears once the row is filtered, and a mojibake label disappears once it
has been canonicalised and mapped.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.settings import DS_COLOUR, FIG_DIR, PRETTY, TAB_DIR, savefig, savetable


def _audit() -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB_DIR, "t02_raw_column_audit.csv"))


def _notes() -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB_DIR, "t06_ingest_notes.csv"))


def _ledger() -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB_DIR, "t03_attrition_ledger.csv"))


def fig_quality_panel(audit: pd.DataFrame) -> None:
    """Four defect families, one panel each, all on raw columns."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    # (a) missing and non-finite values — CICIDS only; UNSW has none at all.
    ax = axes[0, 0]
    sub = audit[(audit.missing_n > 0) | (audit.non_finite_n > 0)].copy()
    sub = sub.sort_values("non_finite_n", ascending=True)
    ypos = np.arange(len(sub))
    ax.barh(ypos - 0.2, sub.missing_n, height=0.38, color="#f4a261", label="missing (NaN)")
    ax.barh(ypos + 0.2, sub.non_finite_n, height=0.38, color="#e63946", label="infinite")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.column}\n({PRETTY[r.dataset]})" for r in sub.itertuples()])
    ax.set_xlabel("records affected")
    ax.set_title("(a) Missing and infinite values")
    ax.legend(loc="lower right")
    for i, r in enumerate(sub.itertuples()):
        ax.text(max(r.missing_n, r.non_finite_n) * 1.03, i, f"{r.non_finite_n:,}",
                va="center", fontsize=7.5)
    ax.set_xlim(0, sub[["missing_n", "non_finite_n"]].to_numpy().max() * 1.18)

    # (b) zero inflation across every numeric column, with the degenerate
    # columns named. A histogram rather than a bar per column: 79 + 45 columns
    # do not fit, and the shape of the distribution is the finding.
    ax = axes[0, 1]
    for ds, grp in audit.groupby("dataset"):
        ax.hist(grp.zero_pct.dropna(), bins=np.arange(0, 105, 5), alpha=0.65,
                color=DS_COLOUR[ds], label=f"{PRETTY[ds]} ({len(grp)} columns)")
    ax.axvline(99, color="#333", ls="--", lw=1)
    ax.text(97, ax.get_ylim()[1] * 0.92, "99%", ha="right", fontsize=7.5, color="#333")
    ax.set_xlabel("share of records equal to exactly zero (%)")
    ax.set_ylabel("number of columns")
    ax.set_title("(b) Zero inflation by column")
    ax.legend(loc="upper center")

    # (c) negative values in quantities that cannot be negative.
    ax = axes[1, 0]
    neg = audit[audit.negative_n > 0].sort_values("negative_n")
    ax.barh(np.arange(len(neg)), neg.negative_n, color="#9d4edd")
    ax.set_yticks(np.arange(len(neg)))
    ax.set_yticklabels(neg.column, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("records with a negative value (log scale)")
    ax.set_title("(c) Negative values in non-negative quantities")
    for i, r in enumerate(neg.itertuples()):
        ax.text(r.negative_n * 1.15, i, f"min {r.min:,.0f}", va="center", fontsize=6.8, color="#444")
    ax.set_xlim(left=5, right=neg.negative_n.max() * 25)

    # (d) structural redundancy: constant and near-constant columns carry no
    # information at all, yet are shipped as features.
    ax = axes[1, 1]
    cic = audit[audit.dataset == "cicids2017"]
    unsw = audit[audit.dataset == "unsw_nb15"]
    cats = ["total\ncolumns", "constant\n(1 value)", ">99%\nzero", "duplicated\nname"]
    cvals = [len(cic), int(cic["constant"].sum()), int((cic.zero_pct > 99).sum()), 1]
    uvals = [len(unsw), int(unsw["constant"].sum()), int((unsw.zero_pct > 99).sum()), 0]
    x = np.arange(len(cats))
    ax.bar(x - 0.19, cvals, width=0.36, color=DS_COLOUR["cicids2017"], label=PRETTY["cicids2017"])
    ax.bar(x + 0.19, uvals, width=0.36, color=DS_COLOUR["unsw_nb15"], label=PRETTY["unsw_nb15"])
    for xi, v in zip(x - 0.19, cvals):
        ax.text(xi, v + 1, str(v), ha="center", fontsize=8)
    for xi, v in zip(x + 0.19, uvals):
        ax.text(xi, v + 1, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("columns")
    ax.set_title("(d) Degenerate and malformed columns")
    ax.legend()
    ax.set_ylim(0, max(cvals) * 1.2)

    fig.suptitle("Figure 1 — Data-quality defects in the raw benchmark files")
    fig.tight_layout()
    savefig(fig, "f01_data_quality")
    plt.close(fig)


def fig_attrition(notes: pd.DataFrame, ledger: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # (a) attrition waterfall per dataset.
    ax = axes[0]
    width = 0.36
    for k, (ds, grp) in enumerate(ledger.groupby("dataset")):
        grp = grp[grp.step != "analysed records"]
        base = grp.remaining.iloc[0]
        xs = np.arange(len(grp)) + (k - 0.5) * width
        pct = 100 * grp.remaining / base
        ax.bar(xs, pct, width=width, color=DS_COLOUR[ds], label=PRETTY[ds])
        for xi, v, n in zip(xs, pct, grp.remaining):
            ax.text(xi, v + 2.5, f"{n / 1e6:.2f}M" if n > 1e6 else f"{n / 1e3:.0f}k",
                    ha="center", fontsize=7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["as read", "after dropping\nunmatched classes", "after dropping\nnon-finite rows"],
                       fontsize=7.5)
    ax.set_ylabel("records remaining (% of raw)")
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(a) Record attrition")
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)

    # (b) exact duplicates.
    ax = axes[1]
    pct = 100 * notes.exact_duplicate_rows / notes.raw_rows
    x = np.arange(len(notes))
    ax.bar(x, pct, color=[DS_COLOUR[d] for d in notes.dataset], width=0.55)
    for xi, (p, n) in enumerate(zip(pct, notes.exact_duplicate_rows)):
        ax.text(xi, p + 0.8, f"{p:.1f}%\n({n:,})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[d] for d in notes.dataset])
    ax.set_ylabel("share of records (%)")
    ax.set_ylim(0, max(pct) * 1.35)
    ax.set_title("(b) Exact duplicate records")

    # (c) the label-encoding defect, shown as the byte it actually is.
    ax = axes[2]
    ax.axis("off")
    ax.set_title("(c) Label encoding defect (CICIDS2017)")
    lines = [
        ("bytes on disk", "Web Attack <0x96> Brute Force"),
        ("read as UTF-8", "UnicodeDecodeError"),
        ("read as latin-1", "Web Attack <0xC2><0x96> Brute Force"),
        ("read as cp1252", "Web Attack – Brute Force"),
        ("after canonicalising", "web attack brute force"),
    ]
    for i, (lhs, rhs) in enumerate(lines):
        yy = 0.92 - i * 0.145
        ax.text(0.0, yy, lhs, fontsize=8.2, color="#555", transform=ax.transAxes)
        ax.text(0.40, yy, rhs, fontsize=8.2, transform=ax.transAxes,
                fontweight="bold" if i == len(lines) - 1 else "normal")
    ax.text(0.0, 0.0,
            "Three of the 15 CICIDS2017 labels carry byte 0x96 — a valid cp1252\n"
            "en-dash, but not valid UTF-8. The exact string a label lookup has to\n"
            "match therefore depends on the encoding the file is read with.",
            fontsize=7.0, color="#444", transform=ax.transAxes, va="bottom")

    fig.suptitle("Figure 2 — Record attrition, duplication and label encoding")
    fig.tight_layout()
    savefig(fig, "f02_attrition_duplication")
    plt.close(fig)


def _readable(label: str) -> str:
    """Show the corrupted CICIDS labels as the byte they actually contain,
    rather than as whichever replacement glyph the font happens to pick."""
    import re
    corrupt = "[" + "".join(chr(c) for c in (0x96, 0xC2, 0xFFFD, 0x2013)) + "]+"
    return re.sub(corrupt, "<0x96>", str(label))


def fig_class_composition() -> None:
    lf = pd.read_csv(os.path.join(TAB_DIR, "t04_label_frequency.csv"))
    from src.eda.settings import CLASS_COLOUR
    palette = dict(CLASS_COLOUR)
    palette["(dropped)"] = "#b0b0b0"

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    for ax, ds in zip(axes[:2], ("cicids2017", "unsw_nb15")):
        sub = lf[lf.dataset == ds].sort_values("n")
        colours = [palette.get(c, "#999") for c in sub.unified_class]
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub.n, color=colours)
        ax.set_yticks(ypos)
        # Render the mojibake label as it reaches a naive reader.
        ax.set_yticklabels([s.replace("\u0096", "?") for s in sub.raw_label], fontsize=7.4)
        ax.set_xscale("log")
        ax.set_xlabel("records (log scale)")
        ax.set_title(f"({'ab'[ds == 'unsw_nb15']}) {PRETTY[ds]} raw labels")
        for i, r in enumerate(sub.itertuples()):
            ax.text(r.n * 1.25, i, f"{r.n:,}", va="center", fontsize=6.8, color="#333")
        ax.set_xlim(right=sub.n.max() * 12)

    ax = axes[2]
    kept = lf[lf.unified_class != "(dropped)"]
    piv = (kept.pivot_table(index="unified_class", columns="dataset", values="n", aggfunc="sum")
               .reindex(["Benign", "DoS/DDoS", "Reconnaissance/Probe", "Exploits"]))
    share = 100 * piv / piv.sum()
    x = np.arange(len(piv))
    ax.bar(x - 0.19, share["cicids2017"], width=0.36,
           color=DS_COLOUR["cicids2017"], label=PRETTY["cicids2017"])
    ax.bar(x + 0.19, share["unsw_nb15"], width=0.36,
           color=DS_COLOUR["unsw_nb15"], label=PRETTY["unsw_nb15"])
    for xi, v, n in zip(x - 0.19, share["cicids2017"], piv["cicids2017"]):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=7.2)
    for xi, v, n in zip(x + 0.19, share["unsw_nb15"], piv["unsw_nb15"]):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Benign", "DoS/\nDDoS", "Recon./\nProbe", "Exploits"], fontsize=8)
    ax.set_ylabel("share of analysed records (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(c) Unified classes: prior shift")
    ax.legend(loc="upper right")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette]
    fig.suptitle("Figure 3 — Label inventory, taxonomy mapping and class balance")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(legend_handles, list(palette), ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=8,
               title="bar colour = unified class the raw label maps to")
    savefig(fig, "f03_class_composition")
    plt.close(fig)

    savetable(piv.assign(cicids_pct=share["cicids2017"], unsw_pct=share["unsw_nb15"]),
              "t07_class_balance")

    imb = pd.DataFrame({
        "dataset": ["cicids2017", "unsw_nb15"],
        "majority_class_pct": [share["cicids2017"].max(), share["unsw_nb15"].max()],
        "minority_class_pct": [share["cicids2017"].min(), share["unsw_nb15"].min()],
        "imbalance_ratio_unified": [piv["cicids2017"].max() / piv["cicids2017"].min(),
                                    piv["unsw_nb15"].max() / piv["unsw_nb15"].min()],
    })
    for ds in ("cicids2017", "unsw_nb15"):
        raw = lf[lf.dataset == ds]
        imb.loc[imb.dataset == ds, "imbalance_ratio_raw_labels"] = raw.n.max() / raw.n.min()
        imb.loc[imb.dataset == ds, "n_raw_labels"] = len(raw)
        imb.loc[imb.dataset == ds, "n_labels_dropped"] = int((raw.unified_class == "(dropped)").sum())
    savetable(imb, "t08_imbalance", index=False)
    print(imb.to_string(index=False))


def run() -> None:
    audit, notes, ledger = _audit(), _notes(), _ledger()
    fig_quality_panel(audit)
    fig_attrition(notes, ledger)
    fig_class_composition()

    # A compact quality scorecard for the report body.
    rows = []
    for ds, grp in audit.groupby("dataset"):
        n = notes.loc[notes.dataset == ds].iloc[0]
        rows.append({
            "dataset": ds,
            "records": int(n.raw_rows),
            "columns": int(n.raw_columns),
            "columns_with_leading_space": int(n.columns_with_leading_space),
            "duplicated_column_names": n.repeated_column_names,
            "constant_columns": int(grp["constant"].sum()),
            "columns_over_99pct_zero": int((grp.zero_pct > 99).sum()),
            "columns_with_missing": int((grp.missing_n > 0).sum()),
            "columns_with_infinite": int((grp.non_finite_n > 0).sum()),
            "columns_with_negatives": int((grp.negative_n > 0).sum()),
            "exact_duplicate_records": int(n.exact_duplicate_rows),
            "duplicate_pct": 100 * n.exact_duplicate_rows / n.raw_rows,
        })
    score = pd.DataFrame(rows)
    savetable(score, "t09_quality_scorecard", index=False)
    print(score.T.to_string())


if __name__ == "__main__":
    from src.eda.settings import apply_style
    apply_style()
    run()
