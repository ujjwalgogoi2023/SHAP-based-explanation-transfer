"""Verify src/data/schema.py assumptions against the REAL downloaded CSVs.

Checks: (a) every referenced column exists, (b) label vocabularies match the
maps, (c) unit assumptions hold empirically, (d) derived features are
physically plausible and on comparable scales across the two datasets.
"""
import os
import sys
import glob
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CIC_DIR = r"C:\Users\ujjwa\Downloads\archive"
UNSW_DIR = r"C:\Users\ujjwa\Downloads\archive (1)"
NROWS = 200_000

from src.data.schema import (
    CICIDS_LABEL_MAP, UNSW_LABEL_MAP, SHARED_FEATURES, FEATURE_SPECS,
    build_feature_matrix,
)

def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


hr("1. CICIDS2017 — column presence")
cic_files = sorted(glob.glob(os.path.join(CIC_DIR, "*.csv")))
print(f"{len(cic_files)} CSV files found")
cic = pd.concat([pd.read_csv(f, nrows=NROWS // len(cic_files), encoding="latin1", low_memory=False)
                 for f in cic_files], ignore_index=True)
print(f"sampled shape: {cic.shape}")

needed_cic = [" Flow Duration", " Total Fwd Packets", " Total Backward Packets",
              "Total Length of Fwd Packets", " Total Length of Bwd Packets",
              "Flow Bytes/s", " Flow Packets/s", " Fwd IAT Mean", " Bwd IAT Mean",
              "FIN Flag Count", " Down/Up Ratio", " Label"]
stripped = {c.strip(): c for c in cic.columns}
for c in needed_cic:
    ok = c in cic.columns or c.strip() in stripped
    print(f"  {'OK ' if ok else 'MISSING'} {c!r}")

hr("2. CICIDS2017 — label vocabulary vs CICIDS_LABEL_MAP")
lab_col = " Label" if " Label" in cic.columns else stripped["Label"]
actual = cic[lab_col].astype(str).str.strip().value_counts()
print(actual.to_string())
print()
unmapped = [l for l in actual.index if l not in CICIDS_LABEL_MAP]
print("UNMAPPED labels (would crash preprocess):", unmapped if unmapped else "none")
unused = [k for k in CICIDS_LABEL_MAP if k not in set(actual.index)]
print("Map keys not seen in this sample:", unused)

hr("3. UNSW-NB15 — which files are usable?")
for f in sorted(glob.glob(os.path.join(UNSW_DIR, "*.csv"))):
    head = pd.read_csv(f, nrows=1, encoding="latin1", low_memory=False)
    cols = [str(c).strip().lstrip("\ufeff") for c in head.columns]
    has_named = "dur" in cols and "attack_cat" in cols
    print(f"  {'USABLE  ' if has_named else 'SKIP    '} {os.path.basename(f):32s} "
          f"ncols={len(cols):3d} first={cols[0][:18]!r}")

unsw_files = [os.path.join(UNSW_DIR, "UNSW_NB15_training-set.csv"),
              os.path.join(UNSW_DIR, "UNSW_NB15_testing-set.csv")]
unsw = pd.concat([pd.read_csv(f, encoding="latin1", low_memory=False) for f in unsw_files],
                 ignore_index=True)
unsw.columns = [str(c).strip().lstrip("\ufeff") for c in unsw.columns]
print(f"\ncombined training+testing shape: {unsw.shape}")

needed_unsw = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
               "sinpkt", "dinpkt", "smean", "dmean", "state", "attack_cat"]
for c in needed_unsw:
    print(f"  {'OK ' if c in unsw.columns else 'MISSING'} {c!r}")

hr("4. UNSW-NB15 — attack_cat vocabulary vs UNSW_LABEL_MAP")
acat = unsw["attack_cat"].astype(str).str.strip().value_counts()
print(acat.to_string())
print()
unmapped_u = [l for l in acat.index if l not in UNSW_LABEL_MAP and l.lower() != "nan"]
print("UNMAPPED attack_cat (would crash preprocess):", unmapped_u if unmapped_u else "none")

hr("5. UNIT CHECKS — the assumptions most likely to be silently wrong")

d_cic = pd.to_numeric(cic[stripped.get("Flow Duration", " Flow Duration")], errors="coerce")
print("CICIDS ' Flow Duration' raw   median = %12.1f  (assumed MICROseconds)" % d_cic.median())
print("   -> /1e6 gives seconds      median = %12.4f s" % (d_cic.median() / 1e6))
print("UNSW   'dur' raw              median = %12.4f  (documented seconds)" % unsw["dur"].median())

print()
iat_c = pd.to_numeric(cic[stripped.get("Fwd IAT Mean", " Fwd IAT Mean")], errors="coerce")
print("CICIDS ' Fwd IAT Mean' raw    median = %12.1f" % iat_c.median())
print("UNSW   'sinpkt' raw           median = %12.4f  (documented ms)" % unsw["sinpkt"].median())
print("UNSW   sinpkt/1000 -> s       median = %12.6f" % (unsw["sinpkt"].median() / 1000))
print("CICIDS iat/1e6     -> s       median = %12.6f" % (iat_c.median() / 1e6))
print(">>> schema.py currently divides sinpkt by 1000 but leaves CICIDS IAT RAW.")

print()
rate_calc = (unsw["spkts"] + unsw["dpkts"]) / (unsw["dur"] + 1e-6)
print("UNSW 'rate' vs (spkts+dpkts)/dur :")
print("   rate median      = %12.4f" % unsw["rate"].median())
print("   recomputed median= %12.4f" % rate_calc.median())
print("   corr             = %12.4f" % unsw["rate"].corr(rate_calc))

print()
print("UNSW 'state' values:", unsw["state"].value_counts().head(8).to_dict())
print("   fraction state=='FIN' =", round((unsw["state"].astype(str).str.upper() == "FIN").mean(), 4))
fin_c = pd.to_numeric(cic[stripped.get("FIN Flag Count", "FIN Flag Count")], errors="coerce")
print("CICIDS 'FIN Flag Count' values:", fin_c.value_counts().head(5).to_dict())

hr("6. DERIVED FEATURE COMPARISON (the thing that actually matters)")
cic_feat = build_feature_matrix(cic, "cicids2017").replace([np.inf, -np.inf], np.nan)
unsw_feat = build_feature_matrix(unsw, "unsw_nb15").replace([np.inf, -np.inf], np.nan)

comp = pd.DataFrame({
    "cic_median": cic_feat.median(),
    "unsw_median": unsw_feat.median(),
    "cic_nan%": (cic_feat.isna().mean() * 100).round(2),
    "unsw_nan%": (unsw_feat.isna().mean() * 100).round(2),
})
comp["ratio"] = (comp["cic_median"] / comp["unsw_median"].replace(0, np.nan)).round(3)
print(comp.to_string())
print()
print("A |ratio| far from 1 (say >100x or <0.01x) means the two datasets are NOT")
print("on the same scale for that feature -- a unit bug or a semantic mismatch.")
