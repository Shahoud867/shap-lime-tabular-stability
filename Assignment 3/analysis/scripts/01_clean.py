"""Stage 01: Data verification, screening (missingness, >1.5*IQR outliers, skewness), and power analysis.

Usage:
    python scripts/01_clean.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from data import download_data  # noqa: E402
from src.data import (  # noqa: E402
    compute_data_screening,
    get_exposure_column,
    load_config,
    load_dataset,
    make_part1_split,
    make_part2_split,
    resolve_path,
)
from src.stats import power_analysis_correlation  # noqa: E402


def main(config_path: Path, force_download: bool = False) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[str] = list(cfg.get("datasets", ["hmeq", "sgcs", "taiwan"]))
    missing_clean = any(not (processed_dir / f"{ds}_clean.csv").exists() for ds in datasets)
    if force_download or missing_clean:
        print("Downloading / refreshing raw datasets and codebooks...")
        download_data.main(force_download=force_download)

    iqr_mult = float(cfg["data"].get("outlier_iqr_multiplier", 1.5))
    part1_test_n = int(cfg["data"].get("part1_test_size", 300))
    part2_test_frac = float(cfg["data"].get("part2_test_frac", 0.30))

    screening_frames: list[pd.DataFrame] = []
    profile_rows: list[dict[str, Any]] = []

    role_map = {
        "hmeq": "Base-paper benchmark 1 (cost + adverse-action case study)",
        "sgcs": "Base-paper benchmark 2 (small credit sample)",
        "taiwan": "Independent main credit sample",
    }

    for ds in datasets:
        df, cb = load_dataset(ds, cfg)
        scr = compute_data_screening(df, cb, iqr_multiplier=iqr_mult)
        scr.insert(0, "dataset", ds)
        scr.to_csv(results_dir / f"data_screening_{ds}.csv", index=False)
        screening_frames.append(scr)

        p1_tr, p1_te = make_part1_split(df, test_size=part1_test_n, seed=seed)
        p2_tr, p2_te = make_part2_split(df, test_frac=part2_test_frac, seed=seed)

        exp_col = get_exposure_column(cb)
        exp_vals = pd.to_numeric(df[exp_col], errors="coerce").dropna()
        n_miss_cells = int(df.drop(columns=["target"]).isna().sum().sum())
        total_feat_cells = int(len(df) * (df.shape[1] - 1))

        n_pert = int(cb.loc[cb["variable"] != "target", "perturbable"].astype(bool).sum())
        n_def = int((df["target"] == 1).sum())

        profile_rows.append({
            "dataset": ds,
            "role": role_map.get(ds, "Credit sample"),
            "n_rows": int(len(df)),
            "n_features": int(df.shape[1] - 1),
            "n_perturbable_features": n_pert,
            "n_defaults": n_def,
            "default_rate_pct": round(float(df["target"].mean() * 100.0), 3),
            "total_missing_cells": n_miss_cells,
            "missing_pct_overall": round(100.0 * n_miss_cells / max(1, total_feat_cells), 3),
            "part1_train_pool_n": int(len(p1_tr)),
            "part1_test_n": int(len(p1_te)),
            "part2_train_n": int(len(p2_tr)),
            "part2_test_n": int(len(p2_te)),
            "exposure_col": exp_col,
            "exposure_median": round(float(exp_vals.median()), 2),
            "exposure_iqr": round(float(np.percentile(exp_vals, 75) - np.percentile(exp_vals, 25)), 2),
        })

    all_scr = pd.concat(screening_frames, ignore_index=True)
    all_scr.to_csv(results_dir / "data_screening.csv", index=False)
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(results_dir / "sample_profile.csv", index=False)

    # A priori power analysis (G*Power exact bivariate normal correlation)
    power_res = power_analysis_correlation(
        rho=0.20,
        alpha=float(cfg["stats"].get("alpha", 0.05)),
        power_target=0.80,
        two_sided=True,
    )
    with (results_dir / "power_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(power_res, fh, indent=2)

    print("=== Stage 01: Data Screening & Sample Profile ===")
    print(profile_df.to_string(index=False))
    print("\nTop skewed / missing / outlier features:")
    interesting = all_scr[(all_scr["missing_count"] > 0) | (all_scr["outlier_pct"] > 5.0) | (all_scr["skewness"].abs() > 2.0)]
    print(interesting[["dataset", "variable", "type", "missing_pct", "outlier_pct", "skewness"]].head(20).to_string(index=False))
    print(f"\nPower analysis (rho=0.20, alpha=0.05, power=0.80): n_exact={power_res['n_exact']}, n_fisher_z={power_res['n_fisher_z']}")
    print(f"Saved outputs to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 01: Data cleaning verification & screening.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml")
    parser.add_argument("--download", action="store_true", help="Force re-download raw data.")
    args = parser.parse_args()
    main(args.config, force_download=args.download)
