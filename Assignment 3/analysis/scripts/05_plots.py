"""Stage 05: Generate all publication-grade results and robustness figures (Figs. 3-7).

Reads empirical outputs from Assignment 3/analysis/results/ (no hardcoded values),
renders vector PDF and 400-DPI PNG plots via src/plots.py, and copies final files
into Assignment 3/paper/figures/.

Usage:
    python scripts/05_plots.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.plots import (
    plot_astar_baselines_pareto,
    plot_double_dissociation,
    plot_lodo_risk_coverage,
    plot_perturbation_vs_floor,
    plot_replication_curves,
    plot_robustness_ablation,
    plot_roc_comparison,
)


def _load_table(results_dir: Path, candidates: list[str]) -> pd.DataFrame | None:
    """Load the first existing parquet or csv file from candidate relative paths."""
    for rel in candidates:
        p = results_dir / rel
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".csv":
                return pd.read_csv(p)
    return None


def _load_json(results_dir: Path, candidates: list[str]) -> dict[str, Any] | None:
    for rel in candidates:
        p = results_dir / rel
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return None


def main(config_path: Path | None = None, results_dir_override: Path | None = None) -> None:
    cfg: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    results_dir = results_dir_override or Path(
        cfg.get("paths", {}).get("results_dir", ANALYSIS_ROOT / "results")
    )
    if not results_dir.is_absolute():
        results_dir = (ANALYSIS_ROOT / results_dir).resolve()

    fig_out_dir = results_dir / "figures"
    paper_fig_dir = PROJECT_ROOT / "paper" / "figures"
    extra_dirs = [paper_fig_dir]

    generated: list[str] = []
    skipped: list[str] = []

    # 1. Fig 3: Part 1 Replication curves (CoV & SRA vs. pi)
    rep_df = _load_table(results_dir, [
        "part1_replication_metrics.parquet",
        "part1_replication_metrics.csv",
        "replication_summary.parquet",
        "replication_summary.csv",
        "part1_summary.parquet",
        "part1_summary.csv",
    ])
    if rep_df is not None and not rep_df.empty:
        pdf_p, png_p = plot_replication_curves(rep_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 3: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 3 (missing Part 1 replication metrics in results/)")

    # 2. Fig 4: Perturbed DeltaE vs. Noise Floor (H2)
    ext_inst_df = _load_table(results_dir, [
        "part2_instance_summary.parquet",
        "part2_instance_summary.csv",
        "part2_draws.parquet",
        "part2_draws.csv",
        "extension_instances.parquet",
        "extension_instances.csv",
    ])
    if ext_inst_df is not None and not ext_inst_df.empty:
        pdf_p, png_p = plot_perturbation_vs_floor(ext_inst_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 4: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 4 (missing Part 2 extension instance/draw metrics in results/)")

    # 3. Fig 5: ROC curves (F(x) vs baselines & |Delta p|, H3 & H5)
    roc_df = _load_table(results_dir, [
        "part2_instance_summary.parquet",
        "part2_instance_summary.csv",
        "part2_roc_inputs.parquet",
        "part2_roc_inputs.csv",
        "extension_instances.parquet",
        "extension_instances.csv",
    ])
    if roc_df is not None and not roc_df.empty:
        pdf_p, png_p = plot_roc_comparison(roc_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 5: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 5 (missing Part 2 ROC instance summary in results/)")

    # 4. Fig 6: Double dissociation (H4b) & LIME discretisation intervention (H4c)
    dissoc_df = _load_table(results_dir, [
        "h4b_double_dissociation.parquet",
        "h4b_double_dissociation.csv",
        "double_dissociation.parquet",
        "double_dissociation.csv",
    ])
    h4c_df = _load_table(results_dir, [
        "h4c_discretisation.parquet",
        "h4c_discretisation.csv",
        "r5_lime_discretisation.parquet",
        "r5_lime_discretisation.csv",
    ])
    if dissoc_df is not None and not dissoc_df.empty:
        pdf_p, png_p = plot_double_dissociation(dissoc_df, h4c_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 6: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 6 (missing H4b double-dissociation results in results/)")

    # 5. Fig 7: Robustness — Nested Models M0->M3 & Ablation (R1) + Scale s Sensitivity (R3)
    nested_df = _load_table(results_dir, [
        "r1_nested_ablation.parquet",
        "r1_nested_ablation.csv",
        "robustness_r1_nested.parquet",
        "robustness_r1_nested.csv",
    ])
    scale_df = _load_table(results_dir, [
        "r3_scale_sensitivity.parquet",
        "r3_scale_sensitivity.csv",
        "robustness_r3_scale.parquet",
        "robustness_r3_scale.csv",
    ])
    if nested_df is not None and not nested_df.empty:
        pdf_p, png_p = plot_robustness_ablation(nested_df, scale_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 7: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 7 (missing R1 nested/ablation results in results/)")

    # 6. Fig 9 (Phase 2 A*): Baseline Pareto frontier (Exp A1) & Stabilization trade-off (Exp A3)
    astar_dir = results_dir / "astar"
    a1_pred_df = _load_table(astar_dir, ["astar_exp_a1_predictors.csv"])
    a3_mit_df = _load_table(astar_dir, ["astar_exp_a3_mitigation.csv"])
    if a1_pred_df is not None and not a1_pred_df.empty:
        pdf_p, png_p = plot_astar_baselines_pareto(a1_pred_df, a3_mit_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 9: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 9 (missing Phase 2 results/astar/astar_exp_a1_predictors.csv)")

    # 7. Fig 10 (Phase 2 A*): Selective Risk-Coverage (Exp A1/A2) & LODO domain transfer (Exp A2)
    a1_inst_df = _load_table(astar_dir, ["astar_exp_a1_instance_scores.parquet"])
    a2_lodo_df = _load_table(astar_dir, ["astar_exp_a2_lodo.csv"])
    if a1_inst_df is not None and not a1_inst_df.empty:
        pdf_p, png_p = plot_lodo_risk_coverage(a1_inst_df, a2_lodo_df, fig_out_dir, extra_dirs=extra_dirs)
        generated.append(f"Fig 10: {pdf_p.name}, {png_p.name}")
    else:
        skipped.append("Fig 10 (missing Phase 2 results/astar/astar_exp_a1_instance_scores.parquet)")

    print("=== 05_plots.py Summary ===")
    for item in generated:
        print(f"  [OK] {item}")
    for item in skipped:
        print(f"  [SKIP] {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Assignment 3 results & robustness figures.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml", help="Path to YAML config.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for results directory.")
    args = parser.parse_args()
    main(config_path=args.config, results_dir_override=args.results_dir)
