"""One-off generator for the notebooks/ files (nbformat v4 JSON, no nbformat
package dependency). Run once: python scripts/make_notebooks.py"""
import json
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 01_preprocessing.ipynb
# ---------------------------------------------------------------------------
nb01 = [
    md("""# 01 — Preprocessing: CICIDS2017 + UNSW-NB15 harmonisation

Run this first, on Colab or Kaggle. It downloads/expects the raw CSVs, harmonises
both datasets into the shared 13-feature vector and 4-class taxonomy defined in
`src/data/schema.py`, and writes processed splits to `data/processed/`.

**Before running:** upload or download the raw CSVs into:
- `data/raw/cicids2017/*.csv` (8 daily capture CSVs from the CICIDS2017 release)
- `data/raw/unsw_nb15/*.csv` (the UNSW-NB15 CSV set, e.g. `UNSW-NB15_1.csv`..`_4.csv`
  or the pre-split train/test files — either works, the pipeline concatenates and re-splits)

Dataset sources (free, no licence required):
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""),
    code("""# If running on Colab, uncomment:
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/DL   # adjust to wherever you cloned/copied this repo

# If running locally under Anaconda on Windows, this must be set BEFORE torch is
# imported — pip torch and Anaconda MKL both ship libiomp5md.dll (OMP: Error #15).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

!pip install -q -r requirements.txt"""),
    code("""import sys, os
sys.path.append(os.path.abspath('..'))  # so `import src...` works from notebooks/"""),
    code("""from src.data.preprocess import run
run('../configs/default.yaml')"""),
    md("""## Sanity check the processed output"""),
    code("""import numpy as np, yaml

with open('../data/processed/class_names.yaml') as f:
    class_names = yaml.safe_load(f)
print('Unified classes:', class_names)

source = np.load('../data/processed/cicids2017_source.npz')
target = np.load('../data/processed/unsw_nb15_target.npz')
print('Source train/val/test:', source['X_train'].shape, source['X_val'].shape, source['X_test'].shape)
print('Target (whole, for transfer eval):', target['X'].shape)
print('Source train class balance:', np.bincount(source['y_train']))
print('Target class balance:', np.bincount(target['y']))"""),
]
write_nb("01_preprocessing.ipynb", nb01)

# ---------------------------------------------------------------------------
# 02_train_source.ipynb
# ---------------------------------------------------------------------------
nb02 = [
    md("""# 02 — Train CNN-LSTM on the source dataset, evaluate on source + target

Trains across all seeds in `configs/default.yaml` (`n_seeds`, default 5) for the
statistical-validation protocol in Sec 4.3 of the proposal. Each seed's model is
checkpointed; Level-1 detection metrics (Sec 4.1) are computed on the source test
split and on the full target dataset with **no retraining**.
"""),
    code("""import sys, os
sys.path.append(os.path.abspath('..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Anaconda/Windows; harmless elsewhere

import yaml
import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.training.train import get_device, train_model
from src.eval.metrics import compute_detection_metrics, predict
from src.data.dataset import make_dataloaders, FlowDataset
from src.models.cnn_lstm import build_model
from torch.utils.data import DataLoader

with open('../configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
with open('../data/processed/class_names.yaml') as f:
    class_names = yaml.safe_load(f)

device = get_device(cfg['train']['device'])
print('device:', device)

source = np.load('../data/processed/cicids2017_source.npz')
target = np.load('../data/processed/unsw_nb15_target.npz')
n_features = source['X_train'].shape[1]
n_classes = len(class_names)
os.makedirs('../checkpoints', exist_ok=True)"""),
    code("""results_per_seed = []

for seed in cfg['seeds'][:cfg['n_seeds']]:
    print(f'\\n=== seed {seed} ===')
    set_seed(seed)

    splits = {'X_train': source['X_train'], 'y_train': source['y_train'],
              'X_val': source['X_val'], 'y_val': source['y_val'],
              'X_test': source['X_test'], 'y_test': source['y_test']}
    train_loader, val_loader, test_loader = make_dataloaders(
        splits, cfg['data']['sequence_length'], cfg['train']['batch_size'])

    model = build_model(cfg, n_features, n_classes)
    model = train_model(model, train_loader, val_loader, source['y_train'], n_classes, cfg, device)

    torch.save(model.state_dict(), f'../checkpoints/cnn_lstm_seed{seed}.pt')

    # Source test evaluation
    y_true, y_pred, y_proba = predict(model, test_loader, device)
    source_metrics = compute_detection_metrics(y_true, y_pred, y_proba, class_names)
    print('Source test accuracy:', source_metrics['summary']['accuracy'])

    # Target (cross-dataset) evaluation, no retraining
    target_ds = FlowDataset(target['X'], target['y'], cfg['data']['sequence_length'])
    target_loader = DataLoader(target_ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    yt_true, yt_pred, yt_proba = predict(model, target_loader, device)
    target_metrics = compute_detection_metrics(yt_true, yt_pred, yt_proba, class_names)
    print('Target (transfer) accuracy:', target_metrics['summary']['accuracy'])

    results_per_seed.append({
        'seed': seed,
        'source_summary': source_metrics['summary'],
        'target_summary': target_metrics['summary'],
    })"""),
    md("""## Aggregate across seeds (mean ± std) — H1 check"""),
    code("""rows = []
for split_name in ['source_summary', 'target_summary']:
    df = pd.DataFrame([r[split_name] for r in results_per_seed])
    agg = df.agg(['mean', 'std']).T
    agg.columns = [f'{split_name}_mean', f'{split_name}_std']
    rows.append(agg)
summary_table = pd.concat(rows, axis=1)
summary_table.to_csv('../results/tables/detection_performance_summary.csv')
summary_table"""),
]
write_nb("02_train_source.ipynb", nb02)

# ---------------------------------------------------------------------------
# 03_shap_source.ipynb
# ---------------------------------------------------------------------------
nb03 = [
    md("""# 03 — SHAP explanation extraction (source-trained model, both datasets)

For each seed's trained model, compute per-class DeepSHAP rankings on:
- the **source** test split -> E1
- the **target** dataset -> E2 (same model, no retraining)

Rankings are pickled to `results/tables/shap_rankings_seed{seed}.pkl` for
notebook 04 to consume.

**Explainer choice matters here.** `shap.DeepExplainer` does not implement DeepLIFT
for `nn.LSTM` — it warns `unrecognized nn.Module: LSTM` and degrades to a plain
gradient for that layer, so its output is not a valid DeepSHAP value for a CNN-LSTM.
The config defaults to `explainer_type: gradient` (Expected Gradients), which handles
recurrent layers correctly. Re-run with `deep` to produce the robustness appendix.
"""),
    code("""import sys, os, pickle
sys.path.append(os.path.abspath('..'))

import yaml
import numpy as np
import torch

from src.training.train import get_device
from src.models.cnn_lstm import build_model
from src.explain.shap_utils import compute_shap_rankings
from src.data.schema import SHARED_FEATURES

with open('../configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
with open('../data/processed/class_names.yaml') as f:
    class_names = yaml.safe_load(f)

device = get_device(cfg['train']['device'])
source = np.load('../data/processed/cicids2017_source.npz')
target = np.load('../data/processed/unsw_nb15_target.npz')
n_features = source['X_train'].shape[1]
n_classes = len(class_names)
os.makedirs('../results/tables', exist_ok=True)"""),
    code("""for seed in cfg['seeds'][:cfg['n_seeds']]:
    print(f'\\n=== seed {seed} ===')
    model = build_model(cfg, n_features, n_classes)
    model.load_state_dict(torch.load(f'../checkpoints/cnn_lstm_seed{seed}.pt', map_location=device))
    model.to(device)

    source_rankings = compute_shap_rankings(
        model, X_background=source['X_train'], X_explain=source['X_test'], y_explain=source['y_test'],
        class_names=class_names, feature_names=SHARED_FEATURES, device=device,
        background_samples=cfg['shap']['background_samples'],
        explain_samples_per_class=cfg['shap']['explain_samples_per_class'], seed=seed,
        explainer_type=cfg['shap']['explainer_type'],
    )
    target_rankings = compute_shap_rankings(
        model, X_background=source['X_train'], X_explain=target['X'], y_explain=target['y'],
        class_names=class_names, feature_names=SHARED_FEATURES, device=device,
        background_samples=cfg['shap']['background_samples'],
        explain_samples_per_class=cfg['shap']['explain_samples_per_class'], seed=seed,
        explainer_type=cfg['shap']['explainer_type'],
    )

    with open(f'../results/tables/shap_rankings_seed{seed}.pkl', 'wb') as f:
        pickle.dump({'source': source_rankings, 'target': target_rankings}, f)
    print('Saved rankings for seed', seed)"""),
]
write_nb("03_shap_source.ipynb", nb03)

# ---------------------------------------------------------------------------
# 04_cross_dataset_eval.ipynb
# ---------------------------------------------------------------------------
nb04 = [
    md("""# 04 — Cross-dataset explanation consistency (H2, H3 + statistical validation)

Loads the per-seed SHAP rankings from notebook 03 and computes:
- Spearman rho, Kendall tau, Jaccard@k, RBO, MAD, and the aggregate ECS per class
- the same-dataset upper bound (two reseeded source models) and random-permutation
  lower bound (Sec 4.3)
- a Wilcoxon signed-rank test of observed ECS vs the random baseline
"""),
    code("""import sys, os, pickle
sys.path.append(os.path.abspath('..'))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.explain.consistency import compute_all_class_consistency, explanation_consistency_score
from src.eval.baselines import random_permutation_baseline, wilcoxon_vs_random, summarise_across_seeds
from src.data.schema import SHARED_FEATURES

with open('../configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
with open('../data/processed/class_names.yaml') as f:
    class_names = yaml.safe_load(f)

seeds = cfg['seeds'][:cfg['n_seeds']]
per_seed = {}
for seed in seeds:
    with open(f'../results/tables/shap_rankings_seed{seed}.pkl', 'rb') as f:
        per_seed[seed] = pickle.load(f)"""),
    md("""## Cross-dataset consistency per class, per seed"""),
    code("""rows = []
for seed in seeds:
    consistency = compute_all_class_consistency(
        per_seed[seed]['source'], per_seed[seed]['target'], SHARED_FEATURES, cfg['shap']['top_k']
    )
    for cls, metrics in consistency.items():
        rows.append({'seed': seed, 'class': cls, **metrics})
consistency_df = pd.DataFrame(rows)
consistency_df.to_csv('../results/tables/cross_dataset_consistency_per_seed.csv', index=False)
consistency_df"""),
    md("""## Mean ± std across seeds, per class (headline table)"""),
    code("""summary = consistency_df.groupby('class').agg(['mean', 'std'])
summary.to_csv('../results/tables/cross_dataset_consistency_summary.csv')
summary"""),
    md("""## Random-permutation lower bound + Wilcoxon significance test"""),
    code("""random_ecs, observed_ecs = [], []
for seed in seeds:
    for cls in class_names:
        if cls not in per_seed[seed]['source']:
            continue
        obs = explanation_consistency_score(
            per_seed[seed]['source'][cls]['ranking'], per_seed[seed]['target'][cls]['ranking'],
            per_seed[seed]['source'][cls]['importance'], per_seed[seed]['target'][cls]['importance'],
            SHARED_FEATURES, cfg['shap']['top_k'],
        )
        rand = random_permutation_baseline(
            per_seed[seed]['source'][cls]['ranking'], per_seed[seed]['source'][cls]['importance'],
            SHARED_FEATURES, seed,
        )
        observed_ecs.append(obs['explanation_consistency_score'])
        random_ecs.append(rand['explanation_consistency_score'])

# Pairs are seeds x classes (5 x 4 = 20). Do NOT collapse to classes alone:
# a one-sided signed-rank test cannot reach p < 0.05 with fewer than 6 pairs.
test_result = wilcoxon_vs_random(observed_ecs, random_ecs)
print('Observed ECS:', summarise_across_seeds(observed_ecs))
print('Random baseline ECS:', summarise_across_seeds(random_ecs))
print('Wilcoxon signed-rank (observed > random):', test_result)
if test_result.get('underpowered'):
    print('WARNING: fewer than 6 pairs — this test cannot reach significance.')"""),
    md("""## Same-dataset upper bound (Sec 4.3)

Consistency between two *independently seeded* models trained on the **same**
dataset. This is the ceiling: cross-dataset consistency can never sensibly exceed
the agreement two runs on identical data achieve with each other. Without this
anchor a raw rho of, say, 0.55 is uninterpretable — it could be near-perfect
transfer or near-total failure depending on where the ceiling sits."""),
    code("""from src.eval.baselines import same_dataset_upper_bound
from itertools import combinations

upper_rows = []
for seed_a, seed_b in combinations(seeds, 2):
    ub = same_dataset_upper_bound(
        per_seed[seed_a]['source'], per_seed[seed_b]['source'], SHARED_FEATURES, cfg['shap']['top_k'])
    for cls, m in ub.items():
        upper_rows.append({'seed_pair': f'{seed_a}v{seed_b}', 'class': cls, **m})
upper_df = pd.DataFrame(upper_rows)
upper_df.to_csv('../results/tables/same_dataset_upper_bound.csv', index=False)

print('Upper bound (same dataset, reseeded)  ECS:',
      round(upper_df['explanation_consistency_score'].mean(), 4),
      '+/-', round(upper_df['explanation_consistency_score'].std(), 4))
print('Observed  (cross-dataset)             ECS:',
      round(consistency_df['explanation_consistency_score'].mean(), 4),
      '+/-', round(consistency_df['explanation_consistency_score'].std(), 4))
print('Lower bound (random permutation)      ECS:',
      round(np.mean(random_ecs), 4))"""),
    md("""### Headline framing

Report cross-dataset consistency **as a fraction of the achievable ceiling**, not as a
bare correlation. `(observed - random) / (upper_bound - random)` is the quantity that
actually answers the research question."""),
    code("""obs_mean = consistency_df['explanation_consistency_score'].mean()
ub_mean = upper_df['explanation_consistency_score'].mean()
lb_mean = float(np.mean(random_ecs))

normalised = (obs_mean - lb_mean) / (ub_mean - lb_mean) if ub_mean > lb_mean else float('nan')
print(f'Normalised cross-dataset consistency: {normalised:.3f}')
print('  1.0 = explanations transfer as well as a same-dataset reseed')
print('  0.0 = explanations transfer no better than random')"""),
    md("""## Plot: consistency by attack class (H3)"""),
    code("""fig, ax = plt.subplots(figsize=(8, 5))
means = consistency_df.groupby('class')['explanation_consistency_score'].mean()
stds = consistency_df.groupby('class')['explanation_consistency_score'].std()
ax.bar(means.index, means.values, yerr=stds.values, capsize=4)
ax.set_ylabel('Explanation Consistency Score')
ax.set_title('Cross-dataset SHAP consistency by attack class')
plt.xticks(rotation=20)
plt.tight_layout()
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/consistency_by_class.png', dpi=150)
plt.show()"""),
]
write_nb("04_cross_dataset_eval.ipynb", nb04)
