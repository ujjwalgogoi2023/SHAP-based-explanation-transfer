"""
Figures for the cross-dataset SHAP consistency study.

Reads the artefacts written by `run_experiment.py` (per-seed SHAP ranking
pickles, the consistency and detection tables, and the model checkpoints) and
writes publication figures to `results/figures/`.

    python scripts/make_figures.py --config configs/default.yaml

Figures produced
----------------
fig1_shap_importance_by_class  per-class SHAP importance share, three conditions
fig2_rank_shift_slopegraph     where each feature moves between source and target
fig3_consistency_anchored      ECS against the random floor and the reseed ceiling
fig4_confusion_matrices        row-normalised confusion, averaged over seeds

Every figure is seed-averaged: marks are means over the five seeds and error
bars are the standard deviation across them. A single-seed figure would not be
reportable given the spread between seeds on the loose class.

Figure 4 needs torch and the checkpoints; if either is missing the other three
are still written and the reason is printed.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.data.schema import SHARED_FEATURES, UNIFIED_CLASSES

# ---------------------------------------------------------------------------
# Palette. Three categorical slots (blue / orange / aqua), validated as a
# colourblind-safe all-pairs set -- which is also why the conditions stop at
# three. Colour tracks the condition everywhere: a condition keeps its hue
# across every figure, so identity carries between them.
# ---------------------------------------------------------------------------
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#a8a79f"
SURFACE = "#ffffff"

COND_COLOUR = {"source": "#2a78d6", "transfer": "#eb6834", "native": "#1baf7a"}
COND_KEY = {"source": "source", "transfer": "target", "native": "native_target"}
COND_LABEL = {
    "source": "CICIDS2017 - source model on source data",
    "transfer": "UNSW-NB15 - same model, weights frozen",
    "native": "UNSW-NB15 - model trained natively on target",
}
DETECTION_SPLIT = {"source": "source", "transfer": "target", "native": "native_target"}

# Single-hue sequential ramp, light -> dark, for the confusion matrices.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


def nice_feature(name: str) -> str:
    return name.replace("_", " ")


def class_tag(cls: str) -> str:
    return "loose" if cls == "Exploits" else "tight"


def style_axis(ax, grid_axis=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=INK_3, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, color=INK_3)


def save(fig, figures_dir: str, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rankings(tables_dir: str, seeds: list) -> dict:
    out = {}
    for seed in seeds:
        path = os.path.join(tables_dir, f"shap_rankings_seed{seed}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing -- run scripts/run_experiment.py first")
        with open(path, "rb") as f:
            out[seed] = pickle.load(f)
    return out


def importance_shares(rankings, seeds, cond, cls, features) -> np.ndarray:
    """(n_seeds, n_features) mean|SHAP| renormalised to sum to 1 within a class.

    Raw attribution magnitudes are in logit units and are not comparable
    between two separately trained models, so only each feature's share of the
    class's total attribution can be put side by side.
    """
    key = COND_KEY[cond]
    rows = []
    for seed in seeds:
        imp = rankings[seed][key][cls]["importance"]
        v = np.array([imp[f] for f in features], dtype=float)
        total = v.sum()
        rows.append(v / total if total > 0 else v)
    return np.vstack(rows)


def mean_ranks(rankings, seeds, cond, cls, features) -> np.ndarray:
    """Mean rank position (1 = most important) across seeds."""
    key = COND_KEY[cond]
    rows = []
    for seed in seeds:
        order = rankings[seed][key][cls]["ranking"]
        pos = {f: i + 1 for i, f in enumerate(order)}
        rows.append([pos[f] for f in features])
    return np.vstack(rows).astype(float).mean(axis=0)


def ordered_classes(class_names: list) -> list:
    """Display order: the three tight classes first, loose Exploits last."""
    return [c for c in UNIFIED_CLASSES if c in class_names]


def spread_labels(values: np.ndarray, min_sep: float) -> np.ndarray:
    """Label positions pushed apart to at least min_sep, order preserved.

    Several features share almost the same mean rank, so their labels collide
    if drawn at the true position. The marks stay put; only the text moves.
    """
    values = np.asarray(values, dtype=float)
    idx = np.argsort(values)
    pos = values[idx].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_sep)
    pos += values.mean() - pos.mean()          # recentre the nudged block
    out = np.empty_like(values)
    out[idx] = pos
    return out


# ---------------------------------------------------------------------------
# Figure 1 -- per-class SHAP importance, three conditions
# ---------------------------------------------------------------------------
def fig_shap_importance(rankings, seeds, class_names, features, figures_dir):
    classes = ordered_classes(class_names)
    conds = ["source", "transfer", "native"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    bar_h = 0.26
    offsets = {"source": bar_h, "transfer": 0.0, "native": -bar_h}

    for ax, cls in zip(axes.ravel(), classes):
        # Feature order is fixed by the source ranking, so every panel reads
        # top-down as "what the source model said mattered".
        src = importance_shares(rankings, seeds, "source", cls, features)
        order = np.argsort(src.mean(axis=0))          # ascending; barh draws upward
        feats = [features[i] for i in order]
        y = np.arange(len(feats))

        for cond in conds:
            vals = importance_shares(rankings, seeds, cond, cls, features)[:, order]
            ax.barh(y + offsets[cond], vals.mean(axis=0), height=bar_h,
                    xerr=vals.std(axis=0), color=COND_COLOUR[cond],
                    edgecolor=SURFACE, linewidth=1.0,
                    error_kw=dict(ecolor=INK_3, elinewidth=0.9, capsize=0),
                    label=COND_LABEL[cond], zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels([nice_feature(f) for f in feats])
        ax.set_ylim(-0.6, len(feats) - 0.4)
        ax.set_xlabel("share of the class's total mean |SHAP|")
        ax.set_title(f"{cls}   -   {class_tag(cls)} correspondence",
                     loc="left", color=INK, pad=8)
        style_axis(ax, grid_axis="x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Where the model says the evidence is, per class",
                 x=0.012, ha="left", fontsize=14, color=INK, y=1.005)
    fig.text(0.012, 0.972,
             f"Mean over {len(seeds)} seeds; error bars are +/-1 SD. Shares, not raw "
             "attributions: magnitudes from two separately trained models are not comparable.",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    save(fig, figures_dir, "fig1_shap_importance_by_class")


# ---------------------------------------------------------------------------
# Figure 2 -- rank movement between source and target
# ---------------------------------------------------------------------------
def fig_rank_shift(rankings, seeds, class_names, features, figures_dir):
    classes = ordered_classes(class_names)
    n_feat = len(features)

    fig, axes = plt.subplots(1, len(classes), figsize=(4.1 * len(classes), 6.8))
    axes = np.atleast_1d(axes)

    for ax, cls in zip(axes, classes):
        left = mean_ranks(rankings, seeds, "source", cls, features)
        right = mean_ranks(rankings, seeds, "native", cls, features)
        top3 = set(np.argsort(left)[:3].tolist())
        left_text = spread_labels(left, 0.42)
        right_text = spread_labels(right, 0.42)

        for i, f in enumerate(features):
            hot = i in top3
            colour = COND_COLOUR["source"] if hot else INK_3
            ax.plot([0, 1], [left[i], right[i]], color=colour,
                    linewidth=2.0 if hot else 1.2, alpha=1.0 if hot else 0.7,
                    solid_capstyle="round", zorder=3 if hot else 2)
            for x, r in ((0, left[i]), (1, right[i])):
                ax.plot([x], [r], marker="o", markersize=5, color=colour,
                        markeredgecolor=SURFACE, markeredgewidth=1.2,
                        zorder=4 if hot else 3)
            # Leader from the mark to its nudged label, so a label that had to
            # move still reads as belonging to its own dot.
            ax.plot([-0.06, -0.015], [left_text[i], left[i]], color=colour,
                    linewidth=0.7, alpha=0.65, zorder=2)
            ax.plot([1.015, 1.06], [right[i], right_text[i]], color=colour,
                    linewidth=0.7, alpha=0.65, zorder=2)
            ax.text(-0.075, left_text[i], nice_feature(f), ha="right", va="center",
                    fontsize=8, color=INK if hot else INK_2)
            ax.text(1.075, right_text[i], nice_feature(f), ha="left", va="center",
                    fontsize=8, color=INK if hot else INK_2)

        ax.set_xlim(-0.82, 1.82)
        ax.set_ylim(n_feat + 0.9, 0.1)          # rank 1 at the top
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CICIDS2017\nsource model", "UNSW-NB15\nnative model"], fontsize=9)
        ax.set_yticks(range(1, n_feat + 1))
        if ax is axes[0]:
            ax.set_ylabel("mean SHAP rank (1 = most important)")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{cls}   -   {class_tag(cls)}", loc="left", color=INK, pad=8)
        style_axis(ax, grid_axis="y")
        ax.spines["bottom"].set_visible(False)

    legend_handles = [
        Line2D([], [], color=COND_COLOUR["source"], linewidth=2.0, marker="o",
               markersize=5, markeredgecolor=SURFACE,
               label="top 3 on the source dataset - the features a practitioner would report"),
        Line2D([], [], color=INK_3, linewidth=1.2, marker="o", markersize=5,
               markeredgecolor=SURFACE, label="remaining features"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("A feature's rank does not survive the move between datasets",
                 x=0.012, ha="left", fontsize=14, color=INK, y=1.01)
    fig.text(0.012, 0.962,
             "Both models are competent on the data they are explained over, so a crossing line is "
             f"explanation instability rather than a broken classifier. Ranks are means over {len(seeds)} seeds.",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])
    save(fig, figures_dir, "fig2_rank_shift_slopegraph")


# ---------------------------------------------------------------------------
# Figure 3 -- ECS against both anchors
# ---------------------------------------------------------------------------
def fig_consistency_anchored(cons_df, headline, class_names, figures_dir):
    # Reversed so the panel reads top-down as Benign -> ... -> Exploits.
    classes = ordered_classes(class_names)[::-1]
    conds = ["transfer", "native"]
    lb = float(headline["lower_bound_ecs"])
    ub = float(headline["upper_bound_ecs"])
    span = ub - lb

    stats = {}
    for cls in classes:
        for cond in conds:
            sub = cons_df[(cons_df["condition"] == cond) & (cons_df["class"] == cls)]
            stats[(cls, cond)] = (float(sub["explanation_consistency_score"].mean()),
                                  float(sub["explanation_consistency_score"].std()))
    # One column for the value labels, clear of the widest whisker, so the
    # numbers line up as a column instead of floating at random x.
    widest = max(m + sd for m, sd in stats.values())
    label_x = max(widest, ub) + 0.06 * span
    x_lo = min(lb, min(m - sd for m, sd in stats.values())) - 0.08 * span
    x_hi = label_x + 0.28 * span          # right margin holds the value column

    # hspace is deliberately not passed here: tight_layout recomputes it and
    # warns that the axes are incompatible if it is set on the gridspec.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[4, 1.35]))

    def anchors(a):
        # The band is the whole point: a bare ECS of 0.73 is uninterpretable
        # without knowing chance is 0.64 and the realistic ceiling 0.82.
        a.axvspan(lb, ub, color=INK_3, alpha=0.18, zorder=0)
        for x in (lb, ub):
            a.axvline(x, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        style_axis(a)

    # Dots, not bars: ECS has a non-zero floor, so a bar from zero would be
    # mostly dead ink and a truncated bar would overstate the differences.
    offs = {"transfer": 0.17, "native": -0.17}
    for i, cls in enumerate(classes):
        for cond in conds:
            m, sd = stats[(cls, cond)]
            y = i + offs[cond]
            ax.plot([lb, m], [y, y], color=COND_COLOUR[cond], linewidth=2.0,
                    solid_capstyle="round", zorder=3)
            # Capped whisker: the caps are what stop the SD reading as a
            # continuation of the stem.
            ax.errorbar([m], [y], xerr=[sd], fmt="o", markersize=9,
                        color=COND_COLOUR[cond], markeredgecolor=SURFACE,
                        markeredgewidth=1.6, ecolor=INK_3, elinewidth=1.0,
                        capsize=3.5, capthick=1.0, zorder=4)
            ax.text(label_x, y, f"{m:.3f}", va="center", ha="left",
                    fontsize=8.5, color=INK_2)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c}\n{class_tag(c)}" for c in classes])
    ax.set_ylim(-0.62, len(classes) - 0.26)
    anchors(ax)
    ax.text(lb, len(classes) - 0.4, " random floor", fontsize=8.5, color=INK_2, va="bottom")
    ax.text(ub, len(classes) - 0.4, " same-dataset ceiling", fontsize=8.5, color=INK_2, va="bottom")
    ax.text(label_x, len(classes) - 0.4, "ECS", fontsize=8.5, color=INK_2, va="bottom", ha="left")

    row_labels = []
    for j, cond in enumerate(conds):
        c = headline["conditions"][cond]
        obs, norm = float(c["observed_ecs"]), float(c["normalised_consistency"])
        p = float(c["wilcoxon"]["p_value"])
        y = 0.5 - j * 0.7
        ax2.plot([lb, obs], [y, y], color=COND_COLOUR[cond], linewidth=2.0,
                 solid_capstyle="round", zorder=3)
        ax2.plot([obs], [y], marker="o", markersize=9, color=COND_COLOUR[cond],
                 markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        ax2.text(label_x, y, f"{obs:.3f}", va="center", ha="left",
                 fontsize=8.5, color=INK_2)
        # The normalised figure and the test go on the row label rather than
        # into the plot area, which would otherwise have to stretch to hold them.
        row_labels.append(f"{cond}\n{norm:.0%} to ceiling\nWilcoxon p = {p:.4f}")
    ax2.set_yticks([0.5, -0.2])
    ax2.set_yticklabels(row_labels)
    ax2.set_ylim(-0.78, 1.05)
    anchors(ax2)
    ax2.set_xlabel("Explanation Consistency Score")
    ax2.set_title("pooled over classes and seeds", loc="left", color=INK_2, fontsize=9.5, pad=6)

    ax.set_xlim(x_lo, x_hi)

    fig.legend(handles=[Line2D([], [], color=COND_COLOUR[c], linewidth=2.0, marker="o",
                               markersize=8, markeredgecolor=SURFACE, label=COND_LABEL[c])
                        for c in conds],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Explanations land closer to chance than to a same-dataset reseed",
                 x=0.012, ha="left", fontsize=14, color=INK, y=1.015)
    fig.text(0.012, 0.952,
             "Dots are means over seeds, whiskers +/-1 SD, each stem drawn from the random floor. "
             "The native row is the interpretable one - the transfer row is confounded by how badly "
             "the source model performs on the target.",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, figures_dir, "fig3_consistency_anchored")


# ---------------------------------------------------------------------------
# Figure 4 -- confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion_matrices(cfg, class_names, seeds, det_df, figures_dir):
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import FlowDataset
    from src.eval.metrics import predict
    from src.models.cnn_lstm import build_model
    from src.training.train import get_device

    processed = cfg["data"]["processed_dir"]
    ckpt_dir = cfg["output"]["checkpoints_dir"]
    source = np.load(os.path.join(processed, f"{cfg['data']['source_dataset']}_source.npz"))
    target = np.load(os.path.join(processed, f"{cfg['data']['target_dataset']}_target.npz"))

    seq_len = cfg["data"]["sequence_length"]
    batch = cfg["train"]["batch_size"]
    device = get_device(cfg["train"]["device"])
    n_features, n_classes = source["X_train"].shape[1], len(class_names)

    # (condition, X, y, checkpoint template). "transfer" reuses the source
    # checkpoint over the whole target set -- the frozen-weights condition,
    # matching how run_experiment.py scores it.
    panels = [
        ("source", source["X_test"], source["y_test"], "cnn_lstm_seed{seed}.pt"),
        ("transfer", target["X"], target["y"], "cnn_lstm_seed{seed}.pt"),
        ("native", target["X_test"], target["y_test"], "cnn_lstm_native_target_seed{seed}.pt"),
    ]

    mats = {}
    for cond, X, y, template in panels:
        per_seed = []
        for seed in seeds:
            ckpt = os.path.join(ckpt_dir, template.format(seed=seed))
            model = build_model(cfg, n_features, n_classes)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)
            loader = DataLoader(FlowDataset(X, y, seq_len), batch_size=batch, shuffle=False)
            y_true, y_pred, _ = predict(model, loader, device)
            cm = np.zeros((n_classes, n_classes), dtype=float)
            np.add.at(cm, (y_true, y_pred), 1.0)
            row = cm.sum(axis=1, keepdims=True)
            per_seed.append(np.divide(cm, row, out=np.zeros_like(cm), where=row > 0))
        mats[cond] = np.mean(per_seed, axis=0)
        print(f"  confusion [{cond}] averaged over {len(seeds)} seeds")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    short = [c.replace("Reconnaissance/Probe", "Recon/Probe") for c in class_names]

    for ax, panel in zip(axes, panels):
        cond = panel[0]
        cm = mats[cond]
        ax.imshow(cm, cmap=BLUE_CMAP, vmin=0, vmax=1, aspect="equal")
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                        color=SURFACE if v > 0.55 else INK)
        f1 = det_df[det_df["split"] == DETECTION_SPLIT[cond]]["f1_macro"].mean()
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(short, rotation=30, ha="right")
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(short if ax is axes[0] else [""] * n_classes)
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("true")
        ax.set_title(f"{COND_LABEL[cond]}\nmacro F1 = {f1:.3f}", loc="left",
                     color=INK, fontsize=10, pad=8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_color(INK_3)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(length=0)

    fig.suptitle("The frozen model does not merely lose accuracy on the target - it collapses",
                 x=0.012, ha="left", fontsize=14, color=INK, y=1.04)
    fig.text(0.012, 0.955,
             "Row-normalised: each row is what actually happened to that true class, averaged over "
             f"{len(seeds)} seeds. Rows sum to 1.",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, figures_dir, "fig4_confusion_matrices")


# ---------------------------------------------------------------------------
def main(config_path: str, skip_confusion: bool):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["data"]["processed_dir"], "class_names.yaml")) as f:
        class_names = yaml.safe_load(f)

    tables_dir = cfg["output"]["tables_dir"]
    figures_dir = cfg["output"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    seeds = cfg["seeds"][: cfg["n_seeds"]]
    rankings = load_rankings(tables_dir, seeds)
    cons_df = pd.read_csv(os.path.join(tables_dir, "consistency_per_seed.csv"))
    det_df = pd.read_csv(os.path.join(tables_dir, "detection_per_seed.csv"))
    with open(os.path.join(tables_dir, "headline.yaml")) as f:
        headline = yaml.safe_load(f)

    print(f"seeds={seeds}  classes={class_names}  -> {figures_dir}/")

    fig_shap_importance(rankings, seeds, class_names, SHARED_FEATURES, figures_dir)
    fig_rank_shift(rankings, seeds, class_names, SHARED_FEATURES, figures_dir)
    fig_consistency_anchored(cons_df, headline, class_names, figures_dir)

    if skip_confusion:
        print("  skipping fig4 (--skip-confusion)")
    else:
        try:
            fig_confusion_matrices(cfg, class_names, seeds, det_df, figures_dir)
        except (ImportError, FileNotFoundError, OSError) as e:
            print(f"  skipping fig4_confusion_matrices: {type(e).__name__}: {e}")

    print("done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-confusion", action="store_true",
                   help="skip figure 4, which needs torch and the checkpoints")
    a = p.parse_args()
    main(a.config, a.skip_confusion)
