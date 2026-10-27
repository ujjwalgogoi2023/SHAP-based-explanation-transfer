"""Run the whole exploratory analysis end to end.

    python scripts/run_eda.py              # everything
    python scripts/run_eda.py --from 3     # skip ingestion, reuse the cache

Stage 1 reads ~0.9 GB of raw CSVs and takes a few minutes; every later stage
reads the cached harmonised matrix instead and takes seconds to a minute.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda import (  # noqa: E402
    ingest,
    stage_multivariate,
    stage_quality,
    stage_shift,
    stage_synthesis,
    stage_univariate,
)
from src.eda.settings import apply_style  # noqa: E402

STAGES = [
    (1, "ingestion and raw data-quality audit", ingest.run),
    (2, "data-quality findings", stage_quality.run),
    (3, "univariate structure", stage_univariate.run),
    (4, "dependence and feature-class association", stage_multivariate.run),
    (5, "cross-benchmark distribution shift", stage_shift.run),
    (6, "synthesis against the modelling outcome", stage_synthesis.run),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="first stage to run (1-6); earlier stages must have run before")
    ap.add_argument("--only", type=int, default=None, help="run a single stage")
    args = ap.parse_args()

    apply_style()
    for num, name, fn in STAGES:
        if args.only is not None and num != args.only:
            continue
        if args.only is None and num < args.start:
            continue
        print(f"\n=== Stage {num}: {name} " + "=" * (46 - len(name)))
        t0 = time.time()
        fn()
        print(f"  [{time.time() - t0:.1f}s]")
    print("\nDone. Figures in eda_results/figures, tables in eda_results/tables.")


if __name__ == "__main__":
    main()
