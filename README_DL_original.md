# On the Cross-Dataset Consistency of SHAP-Based Explanations in Hybrid CNN-LSTM Intrusion Detection Systems

Published IDS work routinely presents SHAP feature-importance rankings as insight into
how attacks are detected, assuming that an important feature reflects a genuine attack
property rather than an artefact of the training dataset. This project tests that
assumption: a hybrid CNN-LSTM is trained on one benchmark, SHAP rankings are extracted,
and the same model is applied to a second benchmark captured in a different testbed.
Divergent rankings indicate the explanation is keyed to dataset-specific artefacts.

## Layout

```
configs/default.yaml        all hyperparameters, seeds, paths
src/data/schema.py          harmonisation: shared feature vector + unified label taxonomy
src/data/preprocess.py      raw CSVs -> cleaned, harmonised, source-fitted-normalised splits
src/data/dataset.py         PyTorch Dataset / DataLoaders
src/models/cnn_lstm.py      hybrid CNN-LSTM classifier
src/training/train.py       training loop, class-weighted loss, early stopping
src/eval/metrics.py         Level-1 detection metrics (Sec 4.1)
src/explain/shap_utils.py   per-class SHAP rankings
src/explain/consistency.py  Level-2 consistency metrics (Sec 4.2) — core contribution
src/eval/baselines.py       upper/lower bounds + Wilcoxon (Sec 4.3)
notebooks/01..04            run these in order
scripts/smoke_test.py       end-to-end verification on synthetic data
scripts/make_figures.py     results/tables -> results/figures, after the run
scripts/run_explainer_variant.py  re-explain the same checkpoints, other explainer
scripts/compare_explainers.py     gradient vs deep, emits the appendix table
paper/main.tex              the write-up; paper/verify_numbers.py checks it
```

## Setup

```bash
pip install -r requirements.txt
```

**Anaconda on Windows:** pip-installed torch and Anaconda's MKL both ship
`libiomp5md.dll`, which aborts at import with `OMP: Error #15`. Set this before running:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Verify the pipeline before touching real data

```bash
python scripts/smoke_test.py
```

Trains on synthetic well-separated data, runs SHAP, computes every consistency metric,
and asserts the positive control: when the target genuinely shares class semantics with
the source, ECS must land near 0.9 rather than near the 0.63 random-permutation floor.
If that control fails, the metric is broken and any real result would be meaningless.

## Datasets

Point `configs/default.yaml` at the download directories.

| Dataset | Files used | Source |
|---|---|---|
| CICIDS2017 | all 8 `*.pcap_ISCX.csv` | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 | `UNSW_NB15_training-set.csv` + `testing-set.csv` only | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

Both are free and require no licence agreement.

**The UNSW glob is deliberately narrow.** That distribution also ships
`UNSW-NB15_1..4.csv` (49 columns, *no header row*) and small metadata files. A bare
`*.csv` glob concatenates all of them into a mostly-NaN frame without raising, so
`preprocess.py` matches `UNSW_NB15_*-set.csv` and rejects mismatched column counts.

### Class taxonomy

Four unified classes, with the mapping tightness recorded in `CLASS_CORRESPONDENCE`:

| Class | CICIDS2017 | UNSW-NB15 | Match |
|---|---|---|---|
| Benign | BENIGN | Normal | tight |
| DoS/DDoS | DoS Hulk/GoldenEye/slowloris/Slowhttptest, DDoS | DoS | tight |
| Reconnaissance/Probe | PortScan | Reconnaissance | tight |
| Exploits | Web Attack ×3, Heartbleed, Infiltration | Exploits, Shellcode, Backdoor | **loose** |

**Categories with no counterpart are dropped, not bucketed.** Excluded: CICIDS `Bot`,
`FTP-Patator`, `SSH-Patator`; UNSW `Generic` (58,871 records of block-cipher attacks
with no CICIDS analogue), `Fuzzers`, `Analysis`, `Worms`. An earlier draft collapsed
all of these into "Exploits", which put credential brute-forcing, botnet C2, protocol
fuzzing and cipher attacks under one invented label — a low consistency score there
would have been explained by the taxonomy, not by SHAP.

Even narrowed, Exploits remains unbalanced (2,227 CICIDS vs 48,365 UNSW) and the two
sides emphasise different vulnerability classes. **Treat Exploits as exploratory and
say so.** The three tight classes carry the primary result.

## Running

Notebooks in order — `01_preprocessing` → `02_train_source` → `03_shap_source` →
`04_cross_dataset_eval`. Notebook 02 trains all 5 seeds and checkpoints each; 03
extracts rankings on both datasets per seed; 04 produces the headline consistency
table, the baselines, and the significance test.

`scripts/run_experiment.py` does all of 02–04 headlessly. Then:

```bash
python scripts/make_figures.py
```

reads the per-seed ranking pickles and the tables and writes four figures to
`results/figures/`, each as both PNG (300 dpi) and PDF:

| Figure | What it shows |
|---|---|
| `fig1_shap_importance_by_class` | per-class attribution share, all three conditions |
| `fig2_rank_shift_slopegraph` | where each feature's rank moves between datasets |
| `fig3_consistency_anchored` | ECS against the random floor and the reseed ceiling |
| `fig4_confusion_matrices` | row-normalised confusion for all three conditions |

Everything is averaged over the five seeds with ±1 SD shown. Figure 4 reloads the
checkpoints, so it needs torch; the other three need only the tables and run with
`--skip-confusion` if torch is unavailable.

## The robustness appendix: the other explainer

A finding that survives two attribution methods is much harder to dismiss, so
the same checkpoints are re-explained with `shap.DeepExplainer`:

```bash
python scripts/run_explainer_variant.py --explainer deep
python scripts/compare_explainers.py
```

The first writes rankings, consistency, both anchors and the significance test
to `results/tables_deep/`, plus a `provenance.yaml` recording that nothing was
retrained and how many `unrecognized nn.Module: LSTM` warnings were emitted.
**Nothing is retrained** — the models are byte-identical to the main run, so
any difference in the result is attributable to the attribution method and to
nothing else. The second prints the gradient-vs-deep comparison and the LaTeX
body of the paper's appendix table.

This is a separate script rather than a flag on `run_experiment.py` on purpose:
that script's job is to train from scratch, and teaching it to silently skip
training would be a trap for whoever runs it next.

## The paper

`paper/main.tex` (IEEEtran, two-column) pulls its figures straight from
`results/figures/` via `\graphicspath`, so regenerating the figures updates the
paper with no copying. Build it with a LaTeX distribution:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Every number in the prose is checked against the tables**, not transcribed by
hand. After any re-run of the experiment:

```bash
python paper/verify_numbers.py
```

pairs each headline value computed from `results/tables/` with the exact string
that must appear in `main.tex` and exits non-zero if any has drifted. Run it
before rebuilding the PDF.

`paper/refs.bib` contains 25 entries, every one checked against the publisher's
registered metadata (Crossref) and, where reachable, the paper's own abstract.
The former `todo:` placeholders standing in for the SHAP-IDS, CNN-LSTM-IDS and
cross-dataset work this study audits have been filled in: Wang et al. (IEEE
Access 2020) and Oseni et al. (IEEE T-ITS 2023) for the SHAP-IDS claim,
Halbouni et al. (IEEE Access 2022) for the CNN-LSTM architecture, and Apruzzese
et al. (IEEE TNSM 2022) with Catillo et al. (Software Quality Journal 2022) for
cross-dataset transfer. No placeholder entries remain.

## Design decisions that affect the result

**Normalisation is fitted on the source dataset only** and applied unchanged to the
target (`preprocess.py`). Fitting on the target would leak target statistics into the
transfer condition and inflate both detection and consistency scores.

**GradientExplainer, not DeepExplainer, is the default.** `shap.DeepExplainer` does not
implement the DeepLIFT rescale rule for `nn.LSTM` — it warns `unrecognized nn.Module:
LSTM` and silently degrades to a plain gradient for that layer, so its output is not a
valid DeepSHAP value for this architecture. Since the entire contribution rests on the
attributions being correct, the default is `shap.GradientExplainer` (Expected
Gradients), which handles recurrent layers properly. `explainer_type="deep"` is still
available, and running both makes a strong robustness appendix — a finding that holds
under two attribution methods is much harder to dismiss.

**12 shared features, not 78 or 49.** CICIDS2017 (CICFlowMeter) and UNSW-NB15
(Argus/Zeek) share no feature schema, so each shared feature in `schema.py` is a
*derivation* computed from both datasets' raw columns, not a column rename. Only
features computable from both are included.

*No TCP flag feature survives.* The proposal listed flag counts among the shared
features, but the two datasets do not measure the same thing: CICIDS's FIN Flag Count
is 0 for 99.2% of flows, while UNSW's nearest proxy (`state == FIN`) is true for 45.5%.
A feature that is near-constant on one dataset carries no rank information there, so
any cross-dataset difference in its SHAP rank would reflect the encoding rather than
explanation instability — it would manufacture the very effect being measured. Report
as a limitation.

**Rates are recomputed from primitives on both sides.** UNSW's `rate` column
correlates only r=0.77 with (spkts+dpkts)/dur, so it is not the same quantity as
CICFlowMeter's `Flow Packets/s`. Both are derived identically from packets ÷ duration.

**Features are log1p'd before scaling.** All 12 are non-negative and span ~9 orders of
magnitude. Without it the scaler is dominated by a few extreme flows and transfers
badly. The transform is parameter-free, so unlike the scaler it introduces no leakage.

**Ratio denominators use +1, not +epsilon.** `bwd_bytes / (fwd_bytes + 1e-6)` turns a
zero denominator into a ~1e6 spike; on the real data that gave `down_up_ratio` a CICIDS
std of 1.4e6 against an IQR of 4.3, and the feature collapsed to std 0.00 on the target
after scaling. `preprocess.py` now warns whenever any feature lands below std 0.05 on
the target, since its SHAP rank would be arbitrary.

**Labels are canonicalised before mapping.** CICIDS ships the Web Attack labels with a
cp1252 en-dash that is not valid UTF-8; matching the literal string fails or succeeds
depending on the read encoding. `canonical_label()` reduces to `[A-Za-z0-9 ]` first.

**The random floor is scored with the same top-k as everything else.**
`random_permutation_baseline` takes a `top_k` argument; leaving it at the
default while the observed scores use the configured `[3, 5, 7]` builds the
floor out of `jaccard@{5,10}`, and with 12 features `jaccard@10` cannot fall
below ~0.67. That inflated the floor from 0.599 to 0.636 and understated how
far observed consistency sits above chance — normalised native consistency
read 0.246 instead of 0.370, and the Wilcoxon p read 0.013 instead of 5.2e-5.
Both the call in `run_experiment.py` and the one in notebook 04 now pass
`cfg["shap"]["top_k"]`. Note also that the baseline permutes only the rank
*order*, carrying attribution magnitudes over unchanged, so the MAD term
contributes its maximum to the floor; the floor is conservative by
construction and so is every normalised figure derived from it.

**Pair the Wilcoxon test over seeds × classes, not classes alone.** A one-sided
signed-rank test cannot reach p < 0.05 with fewer than 6 pairs regardless of effect
size. 5 seeds × 4 classes = 20 pairs is adequate; 4 classes alone is not.

## Interpreting the outcome

Per the proposal, either direction is publishable. Low consistency identifies a
methodological weakness in existing work; high consistency validates an assumption the
field currently relies on without evidence. The two anchors in `baselines.py` are what
make either claim interpretable — a raw correlation figure without the same-dataset
upper bound and the random lower bound cannot support either conclusion.
