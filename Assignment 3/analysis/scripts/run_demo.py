"""Live Demo Script (< 3 minutes): Applicant Fragility Audit & Optional End-to-End Pipeline.

Demonstrates the real-time adverse-action fragility gate on HMEQ + XGB:
  1. Trains XGBoost on the HMEQ training split (seed=42).
  2. Selects the case-study rejected applicant (or a user-specified --applicant index).
  3. Prints the applicant's predicted probability, decision, and top-4 TreeSHAP reasons.
  4. Computes ex-ante closed-form fragility F_tree(x) without running any perturbations.
  5. Runs R=25 live prediction-preserving perturbations with exact interventional TreeSHAP.
  6. Reports how often the top-4 reason codes flip while the decision stays unchanged.
  7. Prints the fragility gate decision (manual review vs. issue notice) using the LODO cut-off.

Also supports `--run-pipeline-config config_demo.yaml` to run Stages 01-05 end-to-end.

Usage:
    python scripts/run_demo.py --dataset hmeq --applicant auto
    python scripts/run_demo.py --run-pipeline-config config_demo.yaml
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.crossings import count_tree_crossings, extract_tree_splits  # noqa: E402
from src.data import Preprocessor, load_config, load_dataset, make_part2_split, resolve_path  # noqa: E402
from src.explain import explain_tree_shap, sample_background  # noqa: E402
from src.fragility import baseline_margin, compute_f_tree  # noqa: E402
from src.metrics import jaccard_distance_topk, top_k_indices  # noqa: E402
from src.models import build_estimator, fit_estimator, tune_hyperparameters  # noqa: E402
from src.perturb import filter_prediction_preserving, generate_perturbations  # noqa: E402


def run_live_applicant_demo(
    config_path: Path,
    dataset: str = "hmeq",
    applicant_arg: str = "auto",
    s_scale: float = 0.05,
    n_draws: int = 25,
) -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    tau = float(cfg["part2"].get("tau", 0.05))
    top_k = int(cfg["part2"].get("top_k", 4))

    print(f"=== Live Adverse-Action Fragility Demo ({dataset.upper()} + XGBoost) ===")
    df, cb = load_dataset(dataset, cfg)
    train_df, test_df = make_part2_split(df, test_frac=float(cfg["data"].get("part2_test_frac", 0.30)), seed=seed)
    prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_df)
    X_tr, y_tr = prep.transform(train_df), train_df["target"].to_numpy(dtype=int)
    X_te = prep.transform(test_df)
    feature_names = prep.feature_names_
    threshold = float(np.mean(y_tr))

    # 1. Train XGBoost (load tuned params from results if cached, else quick tune)
    perf_path = results_dir / "part2_model_performance.csv"
    best_p: dict[str, Any] | None = None
    if perf_path.exists():
        import pandas as pd
        pf = pd.read_csv(perf_path)
        match = pf[(pf["dataset"] == dataset) & (pf["model"] == "xgb")]
        if not match.empty:
            best_p = json.loads(str(match.iloc[0]["best_params"]))
    if best_p is None:
        best_p = tune_hyperparameters("boost", X_tr, y_tr, cv_folds=3, seed=seed + 2, n_jobs=1)

    est = build_estimator("boost", params=best_p, seed=seed + 2, n_jobs=1)
    fit_estimator(est, "boost", X_tr, y_tr)
    bg = sample_background(X_tr, n=int(cfg["part2"].get("tree_shap_background_size", 100)), seed=seed)
    ts = extract_tree_splits(est, feature_names=feature_names)

    # 2. Pick applicant ID
    if applicant_arg == "auto":
        cs_path = results_dir / "case_study_hmeq.json"
        if dataset == "hmeq" and cs_path.exists():
            with cs_path.open("r", encoding="utf-8") as fh:
                cs = json.load(fh)
            app_idx = int(cs["high_fragility_applicant"]["instance_id"])
        else:
            p_te = est.predict_proba(X_te)[:, 1]
            rej_ids = np.flatnonzero(p_te >= threshold)
            f_vals = [
                compute_f_tree(X_te[i], ts, s_scale, prep.feature_sds_, prep.perturbable_mask_, weighted=False)
                for i in rej_ids
            ]
            app_idx = int(rej_ids[int(np.argmax(f_vals))])
    else:
        app_idx = int(applicant_arg)

    x_app = X_te[app_idx]
    p_app = float(est.predict_proba(x_app.reshape(1, -1))[0, 1])
    decision = "REJECT (Default Risk)" if p_app >= threshold else "APPROVE"

    # 3. Baseline TreeSHAP + Top-4 reasons + Closed-form F_tree(x)
    phi_base, _ = explain_tree_shap(est, bg, x_app.reshape(1, -1))
    phi_base = phi_base[0]
    top4 = top_k_indices(phi_base, k=top_k)
    f_tree_val = compute_f_tree(x_app, ts, s_scale, prep.feature_sds_, prep.perturbable_mask_, weighted=False)
    margin_val = baseline_margin(p_app, threshold)

    print(f"\n[Step 1-2] Test Applicant #{app_idx} | p(default)={p_app:.4f} (threshold t={threshold:.4f}) -> {decision}")
    print("  Baseline Top-4 Adverse-Action Reasons (Exact TreeSHAP):")
    for rank, j in enumerate(top4, start=1):
        print(f"    {rank}. {feature_names[j]:18s} phi = {phi_base[j]:+.4f}")

    print(f"\n[Step 3] Ex-Ante Fragility Score (0 perturbations): F_tree(x; s={s_scale}) = {f_tree_val:.4f} | Margin = {margin_val:.4f}")

    # 4. Run 25 live perturbations with exact TreeSHAP
    xp = generate_perturbations(
        x_app, s=s_scale, feature_sds=prep.feature_sds_,
        perturbable_mask=prep.perturbable_mask_,
        clip_low=prep.clip_low_, clip_high=prep.clip_high_,
        n_draws=n_draws, seed=seed + 100000 * 2 + int(s_scale * 1000) * 1000 + app_idx,
        clip=True,
    )
    f_info = filter_prediction_preserving(est, x_app, xp, threshold=threshold, tau=tau)
    phi_ps, _ = explain_tree_shap(est, bg, xp)
    ct = count_tree_crossings(x_app, xp, ts, per_tree=True)

    flips = [int(jaccard_distance_topk(phi_base, phi_ps[r], k=top_k) > 1e-12) for r in range(n_draws)]
    pres = f_info["preserved"]
    n_pres = int(np.sum(pres))
    flip_rate_pres = float(np.mean(np.asarray(flips)[pres])) if n_pres > 0 else float(np.mean(flips))

    print(f"\n[Step 4-5] Live Perturbation Audit (R={n_draws}, s={s_scale}, tau={tau}):")
    print(f"  Prediction-preserving draws : {n_pres}/{n_draws} (pass rate = {f_info['pass_rate']*100:.1f}%)")
    print(f"  Mean split crossings/tree   : {float(np.mean(ct)):.4f} (predicted F_tree = {f_tree_val:.4f})")
    print(f"  Max |Delta p| across draws  : {float(np.max(f_info['abs_dp'])):.4f}")
    print(f"  Top-4 reason-code flip rate : {flip_rate_pres*100:.1f}% (decision unchanged in {int(np.sum(f_info['y_primes'] == f_info['y_x']))}/{n_draws} draws)")

    # 5. LODO Gate Decision
    lodo_path = results_dir / "r7_lodo_gate.json"
    cutoff = None
    if lodo_path.exists():
        with lodo_path.open("r", encoding="utf-8") as fh:
            lodo = json.load(fh)
        match_l = [row for row in lodo if row["held_out_dataset"] == dataset]
        if match_l:
            cutoff = float(match_l[0]["dev_cutoff_f_tree"])
    if cutoff is None:
        cutoff = 0.05  # Fallback only if LODO results not yet cached

    gate_action = "ROUTE TO MANUAL REVIEW / STABILIZED REASONS" if f_tree_val >= cutoff else "ISSUE AUTOMATED NOTICE"
    print(f"\n[Step 6] Fragility Gate Policy (LODO cut-off = {cutoff:.4f}):")
    print(f"  F_tree ({f_tree_val:.4f}) {'>=' if f_tree_val >= cutoff else '<'} {cutoff:.4f}  ==>  {gate_action}")
    print(f"\nDemo finished in {time.time() - t0:.2f}s.")

    return {
        "applicant_id": app_idx,
        "p_default": p_app,
        "threshold": threshold,
        "f_tree": f_tree_val,
        "lodo_cutoff": cutoff,
        "gate_action": gate_action,
        "pass_rate": f_info["pass_rate"],
        "flip_rate": flip_rate_pres,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def run_end_to_end_pipeline(config_path: Path) -> None:
    """Execute Stages 01 -> 05 sequentially with the specified config (e.g., config_demo.yaml)."""
    t0 = time.time()
    py = sys.executable
    scripts = [
        "01_clean.py",
        "02_replicate.py",
        "03_extend.py",
        "04_robustness.py",
        "05_plots.py",
    ]
    for s_name in scripts:
        s_path = ANALYSIS_ROOT / "scripts" / s_name
        print(f"\n>>> Running {s_name} --config {config_path} <<<", flush=True)
        subprocess.run([py, str(s_path), "--config", str(config_path)], check=True)
    print(f"\n=== End-to-End Demo Pipeline Complete in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 Live Demo & End-to-End Runner.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml")
    parser.add_argument("--dataset", type=str, default="hmeq")
    parser.add_argument("--applicant", type=str, default="auto")
    parser.add_argument("--run-pipeline-config", type=Path, default=None, help="Run Stages 01-05 end-to-end with this config.")
    args = parser.parse_args()

    if args.run_pipeline_config is not None:
        run_end_to_end_pipeline(args.run_pipeline_config)
    else:
        run_live_applicant_demo(args.config, dataset=args.dataset, applicant_arg=args.applicant)
