"""Stage 06: Phase 2 A*-Conference Elevation Experiments (Exp A1-A4).

Evaluates Proposition P2 leaf-value-jump fragility (F_jump), scale-free
training-distribution calibration (F_cal) for Leave-One-Dataset-Out (LODO)
gating, SOTA A* baselines (S-LIME, GPEC, ST-SHAP, Epistemic UQ, AXOM), and
cross-domain non-credit mechanism replication (H4b/H4c) across all 6 benchmarks:
    Credit:     hmeq, sgcs, taiwan
    Non-credit: bank, adult, heart

All outputs are written cleanly inside `Assignment 3/analysis/results/astar/`
(without overwriting v1.0 Phase 1 results).

Usage:
    python scripts/06_a_star_extension.py --config config.yaml
    python scripts/06_a_star_extension.py --config config.yaml --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.baselines import (  # noqa: E402
    axom_treeshap_values,
    epistemic_uq_score,
    gpec_boundary_uncertainty,
    s_lime_explain,
    st_shap_values,
)
from src.crossings import (  # noqa: E402
    count_bin_crossings,
    count_tree_crossings,
    extract_lime_bin_edges,
    extract_tree_splits,
)
from src.data import (  # noqa: E402
    Preprocessor,
    load_config,
    load_dataset,
    make_part2_split,
    resolve_path,
)
from src.explain import (  # noqa: E402
    build_lime_explainer,
    explain_lime,
    explain_tree_shap,
    sample_background,
)
from src.fragility import (  # noqa: E402
    baseline_margin,
    baseline_tree_variance,
    calibrate_fragility_scores,
    compute_f_bin,
    compute_f_jump_batch,
    compute_f_tree,
)
from src.metrics import (  # noqa: E402
    jaccard_distance_topk,
    normalized_l1_distance,
    reason_code_change,
    spearman_distance,
)
from src.models import (  # noqa: E402
    build_estimator,
    canonical_model_name,
    evaluate_model,
    fit_estimator,
    tune_hyperparameters,
)
from src.perturb import filter_prediction_preserving, generate_perturbations  # noqa: E402
from src.stats import (  # noqa: E402
    cluster_bootstrap_auroc,
    cluster_bootstrap_aurc,
    delong_paired_auroc_test,
    holm_bonferroni,
    joint_logistic_cluster_robust,
    paired_aurc_diff_test,
    paired_bootstrap_auroc_diff,
    paired_wilcoxon_test,
    spearman_with_ci,
)


def stratified_sample_indices(
    y_hat: np.ndarray,
    sample_n: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `sample_n` test indices stratified 50/50 by predicted class with population weights."""
    rng = np.random.default_rng(seed)
    n_total = len(y_hat)
    pos_idx = np.flatnonzero(y_hat == 1)
    neg_idx = np.flatnonzero(y_hat == 0)

    n_target = min(int(sample_n), n_total)
    n_half = n_target // 2
    n_pos_take = min(len(pos_idx), n_half)
    n_neg_take = min(len(neg_idx), n_target - n_pos_take)
    if n_pos_take + n_neg_take < n_target:
        rem = n_target - (n_pos_take + n_neg_take)
        if len(pos_idx) - n_pos_take >= rem:
            n_pos_take += rem
        else:
            n_neg_take = min(len(neg_idx), n_neg_take + rem)

    chosen_pos = rng.choice(pos_idx, size=n_pos_take, replace=False) if n_pos_take > 0 else np.empty(0, dtype=int)
    chosen_neg = rng.choice(neg_idx, size=n_neg_take, replace=False) if n_neg_take > 0 else np.empty(0, dtype=int)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    chosen.sort()

    p_pop_pos = len(pos_idx) / max(1, n_total)
    p_pop_neg = len(neg_idx) / max(1, n_total)
    p_smp_pos = max(1, n_pos_take) / max(1, len(chosen))
    p_smp_neg = max(1, n_neg_take) / max(1, len(chosen))
    w_pos = p_pop_pos / p_smp_pos
    w_neg = p_pop_neg / p_smp_neg
    weights = np.where(y_hat[chosen] == 1, w_pos, w_neg).astype(float)
    if np.mean(weights) > 0:
        weights /= np.mean(weights)
    return chosen, weights


def _load_or_tune_model(
    ds: str,
    m_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cached_perf_df: pd.DataFrame | None,
    seed: int,
    workers: int,
) -> tuple[Any, dict[str, Any]]:
    """Reuse v1.0 tuned hyperparameters when cached, otherwise run 3-fold CV tuning."""
    canon = canonical_model_name(m_name)
    best_p: dict[str, Any] | None = None
    if cached_perf_df is not None:
        hit = cached_perf_df[(cached_perf_df["dataset"] == ds) & (cached_perf_df["model"] == m_name)]
        if not hit.empty:
            best_p = json.loads(str(hit.iloc[0]["best_params"]))

    if best_p is None:
        best_p = tune_hyperparameters(
            canon, X_tr, y_tr, cv_folds=3, seed=seed, n_jobs=max(1, min(workers, 8))
        )

    est = build_estimator(canon, params=best_p, seed=seed, n_jobs=1)
    fit_estimator(est, canon, X_tr, y_tr)
    return est, best_p


def _treeshap_chunk(
    est: Any,
    bg: np.ndarray,
    X_rows: np.ndarray,
) -> np.ndarray:
    sv, _ = explain_tree_shap(est, bg, X_rows, model_output="raw")
    return sv


def _parallel_treeshap(
    pool: Parallel,
    est: Any,
    bg: np.ndarray,
    X_all: np.ndarray,
    n_chunks: int,
) -> np.ndarray:
    """Compute exact interventional TreeSHAP in parallel row chunks."""
    if X_all.shape[0] <= 256 or n_chunks <= 1:
        sv, _ = explain_tree_shap(est, bg, X_all, model_output="raw")
        return sv
    chunks = np.array_split(X_all, min(n_chunks, X_all.shape[0]))
    parts = pool(delayed(_treeshap_chunk)(est, bg, ch) for ch in chunks if len(ch) > 0)
    return np.vstack(parts)


def _lime_worker_batch(
    X_tr: np.ndarray,
    feature_names: list[str],
    est: Any,
    X_inst_chunk: np.ndarray,
    inst_ids: np.ndarray,
    xp_by_scale: dict[float, np.ndarray],
    k_floor: int,
    num_samples: int,
    discretize: bool,
    base_seed: int,
) -> dict[int, dict[str, Any]]:
    """Run standard LIME or LIME-disc-off baseline, K noise repeats, and perturbed calls."""
    le = build_lime_explainer(
        X_tr, feature_names=feature_names, discretize_continuous=discretize, seed=base_seed
    )
    out: dict[int, dict[str, Any]] = {}
    for li, inst_id in enumerate(inst_ids):
        x_row = X_inst_chunk[li : li + 1]
        phi_base = explain_lime(
            le, est, x_row, num_samples=num_samples, seed=base_seed + 900000 + int(inst_id) * 101
        )[0]
        floor_phis = (
            np.vstack([
                explain_lime(
                    le, est, x_row, num_samples=num_samples, seed=base_seed + 100000 + int(inst_id) * 100 + kk
                )
                for kk in range(k_floor)
            ])
            if k_floor > 0
            else np.empty((0, X_inst_chunk.shape[1]), dtype=float)
        )
        pert_phis: dict[float, np.ndarray] = {}
        for s_val, xp_chunk in xp_by_scale.items():
            pert_phis[s_val] = explain_lime(
                le,
                est,
                xp_chunk[li],
                num_samples=num_samples,
                seed=base_seed + 200000 + int(round(s_val * 1000)) * 10000 + int(inst_id) * 100,
            )
        out[int(inst_id)] = {
            "phi_base": phi_base,
            "floor_phis": floor_phis,
            "pert_phis": pert_phis,
        }
    return out


def _slime_worker_chunk(
    X_tr: np.ndarray,
    feature_names: list[str],
    est: Any,
    X_inst_chunk: np.ndarray,
    inst_ids: np.ndarray,
    xp_chunk: np.ndarray,
    k_floor: int,
    n_init: int,
    n_max: int,
    top_k: int,
    base_seed: int,
) -> dict[int, dict[str, Any]]:
    """Run S-LIME (Zhou et al., KDD 2021) base, K repeats, and perturbed explanations."""
    le = build_lime_explainer(
        X_tr, feature_names=feature_names, discretize_continuous=True, seed=base_seed
    )
    out: dict[int, dict[str, Any]] = {}
    for li, inst_id in enumerate(inst_ids):
        x_row = X_inst_chunk[li : li + 1]
        phi_base, nu_base = s_lime_explain(
            le,
            est,
            x_row,
            n_init=n_init,
            n_max=n_max,
            top_k=top_k,
            seed=base_seed + 910000 + int(inst_id) * 101,
            return_n_samples=True,
        )
        floor_list: list[np.ndarray] = []
        for kk in range(k_floor):
            fp = s_lime_explain(
                le,
                est,
                x_row,
                n_init=n_init,
                n_max=n_max,
                top_k=top_k,
                seed=base_seed + 310000 + int(inst_id) * 100 + kk,
                return_n_samples=False,
            )
            floor_list.append(fp[0])
        pert_phis, nu_pert = s_lime_explain(
            le,
            est,
            xp_chunk[li],
            n_init=n_init,
            n_max=n_max,
            top_k=top_k,
            seed=base_seed + 510000 + int(inst_id) * 100,
            return_n_samples=True,
        )
        out[int(inst_id)] = {
            "phi_base": phi_base[0],
            "floor_phis": np.vstack(floor_list) if floor_list else np.empty((0, X_inst_chunk.shape[1])),
            "pert_phis": pert_phis,
            "n_used_mean": float(np.mean(np.concatenate([nu_base, nu_pert]))),
        }
    return out


def _axom_chunk(
    est: Any,
    bg: np.ndarray,
    X_rows: np.ndarray,
    threshold: float,
) -> np.ndarray:
    sv = axom_treeshap_values(est, bg, X_rows, threshold=threshold, return_agree_counts=False)
    return np.asarray(sv, dtype=float)


def _st_shap_chunk(
    est: Any,
    bg: np.ndarray,
    X_rows: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    sv, _ = st_shap_values(
        lambda X: est.predict_proba(X)[:, 1],
        bg,
        X_rows,
        budget=budget,
        seed=seed,
    )
    return sv


def _pairwise_distance_summary(
    phi_base: np.ndarray,
    pert_phis: np.ndarray,
    pres_mask: np.ndarray,
    top_k: int,
    y_cutoff: float,
    floor_phis: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute perturbation distances & reason-code flip rate relative to phi_base."""
    r_cnt = pert_phis.shape[0]
    dk_all = np.array([jaccard_distance_topk(phi_base, pert_phis[r], k=top_k) for r in range(r_cnt)])
    dr_all = np.array([spearman_distance(phi_base, pert_phis[r]) for r in range(r_cnt)])
    d1_all = np.array([normalized_l1_distance(phi_base, pert_phis[r]) for r in range(r_cnt)])

    rc_all = np.array([reason_code_change(phi_base, pert_phis[r], k=top_k, noise_floor_q=0.0) for r in range(r_cnt)], dtype=float)

    idx = np.where(pres_mask)[0]
    if idx.size == 0:
        idx = np.arange(r_cnt)

    noise_floor_dk = np.nan
    noise_floor_rc = np.nan
    if floor_phis is not None and len(floor_phis) > 0:
        f_dk = [jaccard_distance_topk(phi_base, floor_phis[kk], k=top_k) for kk in range(len(floor_phis))]
        f_rc = [reason_code_change(phi_base, floor_phis[kk], k=top_k, noise_floor_q=0.0) for kk in range(len(floor_phis))]
        noise_floor_dk = float(np.mean(f_dk))
        noise_floor_rc = float(np.mean(f_rc))

    u_bar = float(np.mean(rc_all[idx]))
    return {
        "d_k": float(np.mean(dk_all[idx])),
        "d_rho": float(np.mean(dr_all[idx])),
        "d_1": float(np.mean(d1_all[idx])),
        "u_bar": u_bar,
        "y_instab": float(u_bar >= y_cutoff),
        "noise_floor_dk": noise_floor_dk,
        "noise_floor_rc": noise_floor_rc,
    }


def _evaluate_predictor_slice(
    df_slice: pd.DataFrame,
    predictor_cols: list[str],
    runtime_map: dict[str, tuple[float, float]],
    n_boot: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Compute AUROC, AURC/E-AURC, selective risks, and Spearman rho for each predictor."""
    rows: list[dict[str, Any]] = []
    valid_base = df_slice["y_instab"].notna() & df_slice["d_k"].notna() & df_slice["u_bar"].notna()
    sub = df_slice.loc[valid_base].copy()
    if sub.empty:
        return rows

    y = sub["y_instab"].to_numpy(dtype=int)
    dk = sub["d_k"].to_numpy(dtype=float)
    ubar = sub["u_bar"].to_numpy(dtype=float)
    clusters = sub["cluster_id"].to_numpy()
    weights = sub["pop_weight"].to_numpy(dtype=float)

    for pred in predictor_cols:
        if pred not in sub.columns:
            continue
        sc = sub[pred].to_numpy(dtype=float)
        fin = np.isfinite(sc)
        if int(np.sum(fin)) < 20 or len(np.unique(y[fin])) < 2:
            continue

        auc_res = cluster_bootstrap_auroc(
            y[fin], sc[fin], clusters[fin], weights=weights[fin], n_boot=n_boot, seed=seed, alpha=alpha
        )
        aurc_res = cluster_bootstrap_aurc(
            y[fin], sc[fin], clusters[fin], n_boot=n_boot, seed=seed, alpha=alpha
        )
        rho_dk = spearman_with_ci(sc[fin], dk[fin], alpha=alpha)
        rho_ubar = spearman_with_ci(sc[fin], ubar[fin], alpha=alpha)
        fit_ms, infer_ms = runtime_map.get(pred, (np.nan, np.nan))

        rows.append({
            "predictor": pred,
            "auroc": auc_res["auroc"],
            "auroc_ci_low": auc_res["ci_low"],
            "auroc_ci_high": auc_res["ci_high"],
            "aurc": aurc_res["aurc"],
            "aurc_ci_low": aurc_res["aurc_ci_low"],
            "aurc_ci_high": aurc_res["aurc_ci_high"],
            "e_aurc": aurc_res["e_aurc"],
            "e_aurc_ci_low": aurc_res["e_aurc_ci_low"],
            "e_aurc_ci_high": aurc_res["e_aurc_ci_high"],
            "base_risk": aurc_res["base_risk"],
            "oracle_aurc": aurc_res["oracle_aurc"],
            "risk_at_cov_90": aurc_res.get("risk_at_cov_90", np.nan),
            "risk_at_cov_70": aurc_res.get("risk_at_cov_70", np.nan),
            "risk_at_cov_50": aurc_res.get("risk_at_cov_50", np.nan),
            "risk_at_cov_30": aurc_res.get("risk_at_cov_30", np.nan),
            "spearman_rho_dk": rho_dk["rho"],
            "spearman_dk_ci_low": rho_dk["ci_low"],
            "spearman_dk_ci_high": rho_dk["ci_high"],
            "spearman_rho_ubar": rho_ubar["rho"],
            "fit_ms_total": fit_ms,
            "infer_ms_per_inst": infer_ms,
            "n_obs": int(np.sum(fin)),
            "n_clusters": int(auc_res["n_clusters"]),
            "instability_rate": float(np.mean(y[fin])),
        })
    return rows


def _pairwise_predictor_tests(
    df_slice: pd.DataFrame,
    focal_col: str,
    baselines: list[str],
    n_boot: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Run DeLong + paired cluster-bootstrap AUROC and AURC comparisons against baselines."""
    sub = df_slice[df_slice["y_instab"].notna()].copy()
    rows: list[dict[str, Any]] = []
    if sub.empty or focal_col not in sub.columns:
        return rows

    y = sub["y_instab"].to_numpy(dtype=int)
    f_sc = sub[focal_col].to_numpy(dtype=float)
    clusters = sub["cluster_id"].to_numpy()
    weights = sub["pop_weight"].to_numpy(dtype=float)

    for b_name in baselines:
        if b_name not in sub.columns:
            continue
        b_sc = sub[b_name].to_numpy(dtype=float)
        fin = np.isfinite(f_sc) & np.isfinite(b_sc)
        if int(np.sum(fin)) < 20 or len(np.unique(y[fin])) < 2:
            continue

        dl = delong_paired_auroc_test(y[fin], f_sc[fin], b_sc[fin], alternative="two-sided")
        pb = paired_bootstrap_auroc_diff(
            y[fin], f_sc[fin], b_sc[fin], cluster_ids=clusters[fin],
            weights=weights[fin], n_boot=n_boot, seed=seed, alpha=alpha, alternative="two-sided",
        )
        pa = paired_aurc_diff_test(
            y[fin], f_sc[fin], b_sc[fin], cluster_ids=clusters[fin],
            n_boot=n_boot, seed=seed, alpha=alpha, alternative="two-sided",
        )
        rows.append({
            "focal": focal_col,
            "baseline": b_name,
            "auroc_focal": dl["auc_1"],
            "auroc_baseline": dl["auc_2"],
            "auroc_diff": pb["diff"],
            "auroc_diff_ci_low": pb["ci_low"],
            "auroc_diff_ci_high": pb["ci_high"],
            "delong_p": dl["pvalue"],
            "boot_auroc_p": pb["pvalue"],
            "aurc_focal": pa["aurc_a"],
            "aurc_baseline": pa["aurc_b"],
            "aurc_diff": pa["diff"],
            "aurc_diff_ci_low": pa["ci_low"],
            "aurc_diff_ci_high": pa["ci_high"],
            "boot_aurc_p": pa["pvalue"],
            "n_obs": int(np.sum(fin)),
            "n_clusters": int(pb["n_clusters"]),
        })
    return rows


def _stats_slice_worker(
    scope_ds: str,
    scope_m: str,
    sub_df: pd.DataFrame,
    rmap: dict[str, tuple[float, float]],
    predictors: list[str],
    baselines: list[str],
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Evaluate predictors, pairwise tests, and sigma ablation for a single slice."""
    p_rows = [
        {"scope_dataset": scope_ds, "model": scope_m, **r}
        for r in _evaluate_predictor_slice(sub_df, predictors, rmap, n_boot=n_boot, seed=seed, alpha=alpha)
    ]
    pair_rows = [
        {"scope_dataset": scope_ds, "model": scope_m, **r}
        for r in _pairwise_predictor_tests(sub_df, "f_jump", baselines, n_boot=n_boot, seed=seed, alpha=alpha)
    ]
    sig_row: dict[str, Any] | None = None
    sub_v = sub_df[sub_df["y_instab"].notna()].copy()
    if not sub_v.empty and len(np.unique(sub_v["y_instab"])) > 1:
        y_v = sub_v["y_instab"].to_numpy(int)
        cl_v = sub_v["cluster_id"].to_numpy()
        w_v = sub_v["pop_weight"].to_numpy(float)
        dl_ens = delong_paired_auroc_test(y_v, sub_v["f_jump"].to_numpy(float), sub_v["f_jump_ens"].to_numpy(float))
        aurc_t = cluster_bootstrap_aurc(y_v, sub_v["f_jump"].to_numpy(float), cl_v, n_boot=min(n_boot, 1000), seed=seed)
        aurc_e = cluster_bootstrap_aurc(y_v, sub_v["f_jump_ens"].to_numpy(float), cl_v, n_boot=min(n_boot, 1000), seed=seed)
        aurc_r = cluster_bootstrap_aurc(y_v, sub_v["f_jump_range"].to_numpy(float), cl_v, n_boot=min(n_boot, 1000), seed=seed)
        auc_r = cluster_bootstrap_auroc(y_v, sub_v["f_jump_range"].to_numpy(float), cl_v, weights=w_v, n_boot=min(n_boot, 1000), seed=seed)
        sig_row = {
            "scope_dataset": scope_ds, "model": scope_m,
            "auroc_tree": dl_ens["auc_1"], "auroc_ensemble": dl_ens["auc_2"], "auroc_range": auc_r["auroc"],
            "diff_tree_minus_ens": dl_ens["diff"], "delong_p_tree_vs_ens": dl_ens["pvalue"],
            "aurc_tree": aurc_t["aurc"], "aurc_ensemble": aurc_e["aurc"], "aurc_range": aurc_r["aurc"],
            "n_obs": int(len(y_v)),
        }
    return p_rows, pair_rows, sig_row


def _compute_f_jump_chunked(
    X: np.ndarray,
    tree_splits: Any,
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
    jump_mode: str = "mean",
    sigma_mode: str = "tree",
    chunk_size: int = 50,
) -> np.ndarray:
    """Chunked compute_f_jump_batch to bound peak memory on deep RF ensembles."""
    if len(X) <= chunk_size:
        return compute_f_jump_batch(
            X, tree_splits, s=s, feature_sds=feature_sds,
            perturbable_mask=perturbable_mask, jump_mode=jump_mode, sigma_mode=sigma_mode,
        )
    parts = [
        compute_f_jump_batch(
            X[i : i + chunk_size], tree_splits, s=s, feature_sds=feature_sds,
            perturbable_mask=perturbable_mask, jump_mode=jump_mode, sigma_mode=sigma_mode,
        )
        for i in range(0, len(X), chunk_size)
    ]
    return np.concatenate(parts)


def _youden_cutoff(scores: np.ndarray, y: np.ndarray) -> float:
    """Select threshold maximizing balanced sensitivity + specificity on training domain."""
    fin = np.isfinite(scores)
    s, yt = scores[fin], y[fin]
    if len(np.unique(yt)) < 2 or len(s) < 10:
        return float(np.median(s)) if len(s) else 0.5
    qs = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 91)))
    best_j = -np.inf
    best_t = float(qs[len(qs) // 2])
    pos = yt == 1
    neg = yt == 0
    for q in qs:
        pred = s >= q
        tpr = float(np.mean(pred[pos]))
        tnr = float(np.mean(~pred[neg]))
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j = j
            best_t = float(q)
    return best_t


def run_lodo_experiment(inst_df: pd.DataFrame, datasets: list[str], models: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exp A2: Calibrated LODO gating across all 6 datasets."""
    lodo_rows: list[dict[str, Any]] = []

    def _eval_fold(sub: pd.DataFrame, ds_hold: str, model_label: str, score_col: str, method_label: str, fixed_cutoff: float | None = None) -> dict[str, Any]:
        tr_sub = sub[sub["dataset"] != ds_hold]
        te_sub = sub[sub["dataset"] == ds_hold]
        tr_sc = tr_sub[score_col].to_numpy(dtype=float)
        tr_y = tr_sub["y_instab"].to_numpy(dtype=int)
        te_sc = te_sub[score_col].to_numpy(dtype=float)
        te_y = te_sub["y_instab"].to_numpy(dtype=int)

        cutoff = float(fixed_cutoff) if fixed_cutoff is not None else _youden_cutoff(tr_sc, tr_y)
        flag = te_sc >= cutoff
        flag_rate = float(np.mean(flag))
        base_rate = float(np.mean(te_y))
        acc_mask = ~flag
        acc_rate = float(np.mean(te_y[acc_mask])) if np.any(acc_mask) else np.nan
        risk_red = float((base_rate - acc_rate) / base_rate * 100.0) if (base_rate > 0 and np.isfinite(acc_rate)) else np.nan

        prec = float(precision_score(te_y, flag, zero_division=0)) if np.any(flag) else np.nan
        rec = float(recall_score(te_y, flag, zero_division=0))
        f1 = float(f1_score(te_y, flag, zero_division=0))
        bacc = float(balanced_accuracy_score(te_y, flag)) if len(np.unique(te_y)) > 1 else np.nan

        auroc_res = cluster_bootstrap_auroc(te_y, te_sc, te_sub["cluster_id"].to_numpy(), n_boot=500, seed=42)
        aurc_res = cluster_bootstrap_aurc(te_y, te_sc, te_sub["cluster_id"].to_numpy(), n_boot=500, seed=42)

        return {
            "held_out_dataset": ds_hold,
            "model": model_label,
            "method": method_label,
            "cutoff_type": f"fixed_{fixed_cutoff}" if fixed_cutoff is not None else "youden_lodo",
            "cutoff_value": cutoff,
            "flag_rate": flag_rate,
            "accepted_rate": 1.0 - flag_rate,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_acc": bacc,
            "base_instab_rate": base_rate,
            "accepted_instab_rate": acc_rate,
            "risk_reduction_pct": risk_red,
            "auroc": auroc_res["auroc"],
            "aurc": aurc_res["aurc"],
            "n_test": int(len(te_y)),
        }

    methods_spec = [
        ("f_unw", "F_unw_raw", None),
        ("f_jump", "F_jump_raw", None),
        ("f_cal_quant", "F_cal_quant_youden", None),
        ("f_cal_rz", "F_cal_rz_youden", None),
        ("f_cal_quant", "F_cal_quant@0.50", 0.50),
        ("f_cal_quant", "F_cal_quant@0.70", 0.70),
        ("f_cal_quant", "F_cal_quant@0.80", 0.80),
        ("gpec_cal_quant", "GPEC_cal_quant", None),
        ("epistemic_cal_quant", "Epistemic_cal_quant", None),
    ]

    valid = inst_df[inst_df["y_instab"].notna()].copy()
    for m_label in models + ["POOLED_RF_XGB"]:
        sub = valid if m_label == "POOLED_RF_XGB" else valid[valid["model"] == m_label]
        for ds_hold in datasets:
            if (sub["dataset"] == ds_hold).sum() == 0:
                continue
            for col, m_name, fixed_c in methods_spec:
                if col in sub.columns and sub[col].notna().any():
                    lodo_rows.append(_eval_fold(sub, ds_hold, m_label, col, m_name, fixed_c))

    lodo_df = pd.DataFrame(lodo_rows)
    summary: dict[str, Any] = {}
    if not lodo_df.empty:
        pooled_sub = lodo_df[lodo_df["model"] == "POOLED_RF_XGB"]
        for m_name, grp in pooled_sub.groupby("method"):
            summary[str(m_name)] = {
                "mean_flag_rate": float(grp["flag_rate"].mean()),
                "min_flag_rate": float(grp["flag_rate"].min()),
                "max_flag_rate": float(grp["flag_rate"].max()),
                "n_degenerate_folds": int(((grp["flag_rate"] <= 0.01) | (grp["flag_rate"] >= 0.99)).sum()),
                "mean_balanced_acc": float(grp["balanced_acc"].mean()),
                "mean_f1": float(grp["f1"].mean()),
                "mean_risk_reduction_pct": float(grp["risk_reduction_pct"].mean()),
                "mean_auroc": float(grp["auroc"].mean()),
                "mean_aurc": float(grp["aurc"].mean()),
            }
    return lodo_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 06: Phase 2 A*-Conference Elevation Experiments (Exp A1-A4).")
    parser.add_argument("--config", type=str, default=str(ANALYSIS_ROOT / "config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test on 2 datasets (n=30, R=5) into results/astar_smoke/.")
    args = parser.parse_args()

    t_start = time.time()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    n_jobs = int(cfg.get("n_jobs", -1))
    cpu_cnt = os.cpu_count() or 4
    workers = min(12, max(1, cpu_cnt if n_jobs < 0 else n_jobs))

    astar_cfg = cfg["astar"]
    datasets = list(astar_cfg["datasets"])
    non_credit_datasets = set(astar_cfg["non_credit_datasets"])
    models = list(astar_cfg["models"])
    sample_n = int(astar_cfg["sample_n"])
    scales = [float(x) for x in astar_cfg["scales_s"]]
    R = int(astar_cfg["perturbation_r"])
    tau = float(astar_cfg["tau"])
    top_k = int(astar_cfg["top_k"])
    y_cutoff = float(astar_cfg["instability_cutoff"])
    slime_n_init = int(astar_cfg["slime_n_init"])
    slime_n_max = int(astar_cfg["slime_n_max"])
    bg_size = int(cfg["part2"]["tree_shap_background_size"])
    lime_num_samples = int(cfg["part2"]["lime_num_samples"])
    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    alpha = float(cfg["stats"]["alpha"])

    out_subdir = "astar_smoke" if args.smoke else "astar"
    out_dir = resolve_path(cfg["paths"]["results_dir"]) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exp A3 uses primary scale s=0.05 on a 100-instance stratified subsample with K_floor=3, R_a3=5
    r_a3 = 5
    k_a3 = 2 if args.smoke else 3
    n_a3_target = 30 if args.smoke else 100
    s_a3 = float(scales[-1])
    if args.smoke:
        datasets = ["heart", "sgcs"]
        non_credit_datasets = {"heart"}
        sample_n = 30
        R = 5
        n_boot = 300

    results_root = resolve_path(cfg["paths"]["results_dir"])
    cached_perf_path = results_root / "part2_model_performance.csv"
    cached_perf_df = pd.read_csv(cached_perf_path) if cached_perf_path.exists() else None

    print(
        f"=== Stage 06: A* Extension ({'SMOKE' if args.smoke else 'FULL'}) | "
        f"datasets={datasets}, models={models}, n={sample_n}, R={R}, workers={workers} ==="
    )

    model_perf_rows: list[dict[str, Any]] = []
    a1_instance_rows: list[dict[str, Any]] = []
    a1_runtime_records: list[dict[str, Any]] = []
    a3_rows: list[dict[str, Any]] = []
    a3_pair_inputs: list[dict[str, Any]] = []
    a4_draws_rows: list[dict[str, Any]] = []
    a4_h4c_rows: list[dict[str, Any]] = []

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with Parallel(n_jobs=workers) as pool:
        for ds_idx, ds in enumerate(datasets):
            ckpt_file = ckpt_dir / f"ckpt_{ds}.pkl"
            if ckpt_file.exists():
                with ckpt_file.open("rb") as fh:
                    ck = pickle.load(fh)
                model_perf_rows.extend(ck["model_perf_rows"])
                a1_instance_rows.extend(ck["a1_instance_rows"])
                a1_runtime_records.extend(ck["a1_runtime_records"])
                a3_rows.extend(ck["a3_rows"])
                a3_pair_inputs.extend(ck["a3_pair_inputs"])
                a4_draws_rows.extend(ck["a4_draws_rows"])
                a4_h4c_rows.extend(ck["a4_h4c_rows"])
                print(f"  [A*] Dataset {ds} loaded from checkpoint ({ckpt_file.name})")
                continue

            _starts = (
                len(model_perf_rows),
                len(a1_instance_rows),
                len(a1_runtime_records),
                len(a3_rows),
                len(a3_pair_inputs),
                len(a4_draws_rows),
                len(a4_h4c_rows),
            )
            t_ds = time.time()
            df, cb = load_dataset(ds, cfg)
            train_df, test_df = make_part2_split(
                df, test_frac=float(cfg["data"]["part2_test_frac"]), seed=seed
            )
            prep = Preprocessor(cb).fit(train_df)
            X_tr = prep.transform(train_df)
            y_tr = train_df["target"].to_numpy(dtype=int)
            X_te = prep.transform(test_df)
            y_te = test_df["target"].to_numpy(dtype=int)
            feature_names = prep.feature_names_
            threshold = float(np.mean(y_tr))
            bg = sample_background(X_tr, n=bg_size, seed=seed + ds_idx)
            bg_st = bg[: min(25, len(bg))]

            # Deterministic training reference cohort (<= 2000 rows) for strictly leak-free F_cal
            rng_ref = np.random.default_rng(seed + 777 + ds_idx)
            n_ref = min(2000, len(X_tr))
            ref_idx = rng_ref.choice(len(X_tr), size=n_ref, replace=False) if len(X_tr) > n_ref else np.arange(len(X_tr))
            X_ref = X_tr[ref_idx]

            # Build LIME bin edges once for F_bin / C_bin
            le_on_ref = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed)
            bin_edges = extract_lime_bin_edges(le_on_ref)

            for m_idx, m_name in enumerate(models):
                t_m = time.time()
                canon = canonical_model_name(m_name)
                est, best_p = _load_or_tune_model(ds, m_name, X_tr, y_tr, cached_perf_df, seed=seed, workers=workers)

                p_te = est.predict_proba(X_te)[:, 1]
                y_hat_te = (p_te >= threshold).astype(int)
                perf = evaluate_model(y_te, p_te, threshold=threshold)
                model_perf_rows.append({
                    "dataset": ds,
                    "domain": "non_credit" if ds in non_credit_datasets else "credit",
                    "model": m_name,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                    "n_features": int(X_tr.shape[1]),
                    "n_perturbable": int(prep.perturbable_mask_.sum()),
                    "threshold_prior": threshold,
                    **perf,
                    "best_params": json.dumps(best_p),
                })

                samp_idx, pop_weights = stratified_sample_indices(
                    y_hat_te, sample_n=sample_n, seed=seed + ds_idx * 10 + m_idx
                )
                X_samp = X_te[samp_idx]
                y_samp = y_te[samp_idx]
                p_samp = p_te[samp_idx]
                yhat_samp = y_hat_te[samp_idx]
                n_samp = len(samp_idx)

                # --- Fit & Time Predictors (Exp A1 & Exp A2) ---
                t0 = time.perf_counter()
                tree_splits = extract_tree_splits(est, feature_names=feature_names, X_train=X_tr)
                fit_splits_ms = (time.perf_counter() - t0) * 1000.0

                # Precompute training F_jump references per scale (one-time offline cost)
                ref_fjump_map: dict[float, np.ndarray] = {}
                ref_gpec_map: np.ndarray | None = None
                ref_epi_map: np.ndarray | None = None

                # Timed GPEC uncertainty
                tshap_fn = lambda X_sub: explain_tree_shap(est, bg, X_sub, model_output="raw")[0]
                t0 = time.perf_counter()
                gpec_scores = gpec_boundary_uncertainty(
                    est,
                    X_tr,
                    X_samp,
                    explainer_fn=tshap_fn,
                    n_train_gp=200 if args.smoke else 300,
                    n_boundary=80 if args.smoke else 150,
                    threshold=threshold,
                    seed=seed + ds_idx * 100 + m_idx,
                )
                gpec_ms_per_inst = (time.perf_counter() - t0) * 1000.0 / float(n_samp)
                # Reference GPEC values on same ci_scale for leak-free LODO quantile calibration
                ref_gpec_map = gpec_boundary_uncertainty(
                    est,
                    X_tr,
                    X_ref[:500],
                    explainer_fn=tshap_fn,
                    n_train_gp=200 if args.smoke else 300,
                    n_boundary=80 if args.smoke else 150,
                    threshold=threshold,
                    seed=seed + ds_idx * 100 + m_idx,
                )

                # Timed Epistemic UQ (Mikriukov et al., 2026) & B_var
                t0 = time.perf_counter()
                epistemic_scores = epistemic_uq_score(
                    est,
                    X_tr,
                    y_tr,
                    X_samp,
                    threshold=threshold,
                    mode="auto",
                    seed=seed + ds_idx * 100 + m_idx,
                )
                epi_ms_per_inst = (time.perf_counter() - t0) * 1000.0 / float(n_samp)
                ref_epi_map = epistemic_uq_score(
                    est,
                    X_tr,
                    y_tr,
                    X_ref[:500],
                    threshold=threshold,
                    mode="auto",
                    seed=seed + ds_idx * 100 + m_idx,
                )

                b_var_scores = (
                    baseline_tree_variance(est, X_samp)
                    if canon == "forest"
                    else epistemic_scores.copy()
                )

                t0 = time.perf_counter()
                b_margin_scores = baseline_margin(p_samp, threshold=threshold)
                margin_ms_per_inst = (time.perf_counter() - t0) * 1000.0 / float(n_samp)

                # --- Generate Perturbations & Compute TreeSHAP (all 6 datasets, both scales) ---
                x_primes_map: dict[float, np.ndarray] = {}
                pres_map: dict[float, np.ndarray] = {}
                abs_dp_map: dict[float, np.ndarray] = {}
                ct_map: dict[float, np.ndarray] = {}
                cb_map: dict[float, np.ndarray] = {}
                for s_val in scales:
                    xp_all = np.zeros((n_samp, R, X_samp.shape[1]), dtype=float)
                    pres_all = np.zeros((n_samp, R), dtype=bool)
                    dp_all = np.zeros((n_samp, R), dtype=float)
                    ct_all = np.zeros((n_samp, R), dtype=float)
                    cb_all = np.zeros((n_samp, R), dtype=float)
                    for li, inst_id in enumerate(samp_idx):
                        p_seed = seed + ds_idx * 1000000 + m_idx * 100000 + int(round(s_val * 1000)) * 1000 + int(inst_id)
                        xp = generate_perturbations(
                            X_samp[li],
                            s=s_val,
                            feature_sds=prep.feature_sds_,
                            perturbable_mask=prep.perturbable_mask_,
                            clip_low=prep.clip_low_,
                            clip_high=prep.clip_high_,
                            n_draws=R,
                            seed=p_seed,
                            clip=True,
                        )
                        xp_all[li] = xp
                        f_info = filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau)
                        pres_all[li] = f_info["preserved"]
                        dp_all[li] = f_info["abs_dp"]
                        ct_all[li] = count_tree_crossings(X_samp[li], xp, tree_splits, per_tree=True)
                        cb_all[li] = count_bin_crossings(X_samp[li], xp, bin_edges)
                    x_primes_map[s_val] = xp_all
                    pres_map[s_val] = pres_all
                    abs_dp_map[s_val] = dp_all
                    ct_map[s_val] = ct_all
                    cb_map[s_val] = cb_all

                # Batched TreeSHAP on base + all perturbed draws across both scales
                t0 = time.perf_counter()
                phi_base_tshap = _parallel_treeshap(pool, est, bg, X_samp, n_chunks=workers)
                tshap_pert_map: dict[float, np.ndarray] = {}
                for s_val in scales:
                    flat_xp = x_primes_map[s_val].reshape(n_samp * R, -1)
                    flat_sv = _parallel_treeshap(pool, est, bg, flat_xp, n_chunks=workers)
                    tshap_pert_map[s_val] = flat_sv.reshape(n_samp, R, -1)
                tshap_posthoc_ms = (time.perf_counter() - t0) * 1000.0 / float(n_samp)

                # Evaluate F_jump, F_jump_ens, F_jump_range, F_unw, F_cal, and populate Exp A1/A2 rows
                for s_val in scales:
                    t0 = time.perf_counter()
                    fj_tree = _compute_f_jump_chunked(
                        X_samp, tree_splits, s=s_val, feature_sds=prep.feature_sds_,
                        perturbable_mask=prep.perturbable_mask_, jump_mode="mean", sigma_mode="tree",
                    )
                    fj_ms_per_inst = (time.perf_counter() - t0) * 1000.0 / float(n_samp)

                    fj_ens = _compute_f_jump_chunked(
                        X_samp, tree_splits, s=s_val, feature_sds=prep.feature_sds_,
                        perturbable_mask=prep.perturbable_mask_, jump_mode="mean", sigma_mode="ensemble",
                    )
                    fj_range = _compute_f_jump_chunked(
                        X_samp, tree_splits, s=s_val, feature_sds=prep.feature_sds_,
                        perturbable_mask=prep.perturbable_mask_, jump_mode="range", sigma_mode="tree",
                    )

                    t0 = time.perf_counter()
                    f_unw = np.array([
                        compute_f_tree(
                            X_samp[li], tree_splits, s=s_val, feature_sds=prep.feature_sds_,
                            perturbable_mask=prep.perturbable_mask_, weighted=False,
                        )
                        for li in range(n_samp)
                    ], dtype=float)
                    funw_ms_per_inst = (time.perf_counter() - t0) * 1000.0 / float(n_samp)

                    f_bin = np.array([
                        compute_f_bin(X_samp[li], bin_edges, s=s_val, feature_sds=prep.feature_sds_, perturbable_mask=prep.perturbable_mask_)
                        for li in range(n_samp)
                    ], dtype=float)

                    # Calibration on strictly leak-free X_ref
                    ref_fj = _compute_f_jump_chunked(
                        X_ref, tree_splits, s=s_val, feature_sds=prep.feature_sds_,
                        perturbable_mask=prep.perturbable_mask_, jump_mode="mean", sigma_mode="tree",
                    )
                    ref_fjump_map[s_val] = ref_fj
                    f_cal_q = calibrate_fragility_scores(fj_tree, ref_fj, method="quantile")
                    f_cal_rz = calibrate_fragility_scores(fj_tree, ref_fj, method="rz")
                    gpec_cal_q = calibrate_fragility_scores(gpec_scores, ref_gpec_map, method="quantile")
                    epi_cal_q = calibrate_fragility_scores(epistemic_scores, ref_epi_map, method="quantile")

                    a1_runtime_records.append({
                        "dataset": ds,
                        "model": m_name,
                        "s": s_val,
                        "fit_splits_ms": fit_splits_ms,
                        "f_jump_ms": fj_ms_per_inst,
                        "f_unw_ms": funw_ms_per_inst,
                        "gpec_ms": gpec_ms_per_inst,
                        "epistemic_uq_ms": epi_ms_per_inst,
                        "b_margin_ms": margin_ms_per_inst,
                        "posthoc_treeshap_ms": tshap_posthoc_ms / float(len(scales)),
                    })

                    for li, inst_id in enumerate(samp_idx):
                        pmask = pres_map[s_val][li]
                        summ = _pairwise_distance_summary(
                            phi_base_tshap[li],
                            tshap_pert_map[s_val][li],
                            pmask,
                            top_k=top_k,
                            y_cutoff=y_cutoff,
                        )
                        idx_p = np.where(pmask)[0]
                        if idx_p.size == 0:
                            idx_p = np.arange(R)
                        mean_dp = float(np.mean(abs_dp_map[s_val][li, idx_p]))
                        mean_ct = float(np.mean(ct_map[s_val][li, idx_p]))
                        mean_cb = float(np.mean(cb_map[s_val][li, idx_p]))

                        a1_instance_rows.append({
                            "dataset": ds,
                            "domain": "non_credit" if ds in non_credit_datasets else "credit",
                            "model": m_name,
                            "s": s_val,
                            "instance_id": int(inst_id),
                            "cluster_id": f"{ds}_{m_name}_{int(inst_id)}",
                            "pop_weight": float(pop_weights[li]),
                            "p_x": float(p_samp[li]),
                            "y_true": int(y_samp[li]),
                            "y_hat": int(yhat_samp[li]),
                            "n_preserved": int(np.sum(pmask)),
                            "pass_rate": float(np.mean(pmask)),
                            "d_k": summ["d_k"],
                            "d_rho": summ["d_rho"],
                            "d_1": summ["d_1"],
                            "u_bar": summ["u_bar"],
                            "y_instab": int(summ["y_instab"]),
                            "mean_abs_dp": mean_dp,
                            "mean_c_tree": mean_ct,
                            "mean_c_bin": mean_cb,
                            "f_jump": float(fj_tree[li]),
                            "f_jump_ens": float(fj_ens[li]),
                            "f_jump_range": float(fj_range[li]),
                            "f_unw": float(f_unw[li]),
                            "f_bin": float(f_bin[li]),
                            "f_cal_quant": float(f_cal_q[li]),
                            "f_cal_rz": float(f_cal_rz[li]),
                            "gpec": float(gpec_scores[li]),
                            "gpec_cal_quant": float(gpec_cal_q[li]),
                            "epistemic_uq": float(epistemic_scores[li]),
                            "epistemic_cal_quant": float(epi_cal_q[li]),
                            "b_var": float(b_var_scores[li]),
                            "b_margin": float(b_margin_scores[li]),
                            "posthoc_abs_dp": mean_dp,
                        })

                        if ds in non_credit_datasets:
                            for r_i in range(R):
                                dk_r = jaccard_distance_topk(phi_base_tshap[li], tshap_pert_map[s_val][li, r_i], k=top_k)
                                rc_r = reason_code_change(phi_base_tshap[li], tshap_pert_map[s_val][li, r_i], k=top_k, noise_floor_q=0.0)
                                a4_draws_rows.append({
                                    "dataset": ds,
                                    "model": m_name,
                                    "explainer": "TreeSHAP",
                                    "s": s_val,
                                    "cluster_id": f"{ds}_{m_name}_{int(inst_id)}",
                                    "pop_weight": float(pop_weights[li]),
                                    "preserved": bool(pmask[r_i]),
                                    "abs_dp": float(abs_dp_map[s_val][li, r_i]),
                                    "c_tree": float(ct_map[s_val][li, r_i]),
                                    "c_bin": float(cb_map[s_val][li, r_i]),
                                    "d_k": float(dk_r),
                                    "rc": int(rc_r),
                                })

                # --- Exp A4: LIME disc ON vs OFF on Non-Credit Datasets (all R=25, both scales) ---
                # Precompute LIME disc ON at s_a3 for Exp A3 on ALL 6 datasets; on non-credit datasets run full R & both scales!
                n_chunks = min(workers, n_samp)
                idx_chunks = np.array_split(np.arange(n_samp), n_chunks)

                if ds in non_credit_datasets:
                    xp_on_scales = {s_v: x_primes_map[s_v] for s_v in scales}
                    res_on_chunks = pool(
                        delayed(_lime_worker_batch)(
                            X_tr, feature_names, est,
                            X_samp[ch], samp_idx[ch],
                            {s_v: xp_on_scales[s_v][ch] for s_v in scales},
                            k_a3, lime_num_samples, True, seed + 2000 + ds_idx * 50 + m_idx * 10,
                        )
                        for ch in idx_chunks if len(ch) > 0
                    )
                    lime_on_map: dict[int, dict[str, Any]] = {}
                    for d_c in res_on_chunks:
                        lime_on_map.update(d_c)

                    res_off_chunks = pool(
                        delayed(_lime_worker_batch)(
                            X_tr, feature_names, est,
                            X_samp[ch], samp_idx[ch],
                            {s_v: xp_on_scales[s_v][ch] for s_v in scales},
                            0, lime_num_samples, False, seed + 3000 + ds_idx * 50 + m_idx * 10,
                        )
                        for ch in idx_chunks if len(ch) > 0
                    )
                    lime_off_map: dict[int, dict[str, Any]] = {}
                    for d_c in res_off_chunks:
                        lime_off_map.update(d_c)

                    for s_val in scales:
                        for li, inst_id in enumerate(samp_idx):
                            pmask = pres_map[s_val][li]
                            lon = lime_on_map[int(inst_id)]
                            loff = lime_off_map[int(inst_id)]
                            for r_i in range(R):
                                p_on = lon["pert_phis"][s_val][r_i]
                                p_off = loff["pert_phis"][s_val][r_i]
                                rc_on = reason_code_change(lon["phi_base"], p_on, k=top_k, noise_floor_q=0.0)
                                rc_off = reason_code_change(loff["phi_base"], p_off, k=top_k, noise_floor_q=0.0)
                                dk_on = jaccard_distance_topk(lon["phi_base"], p_on, k=top_k)
                                a4_draws_rows.append({
                                    "dataset": ds,
                                    "model": m_name,
                                    "explainer": "LIME",
                                    "s": s_val,
                                    "cluster_id": f"{ds}_{m_name}_{int(inst_id)}",
                                    "pop_weight": float(pop_weights[li]),
                                    "preserved": bool(pmask[r_i]),
                                    "abs_dp": float(abs_dp_map[s_val][li, r_i]),
                                    "c_tree": float(ct_map[s_val][li, r_i]),
                                    "c_bin": float(cb_map[s_val][li, r_i]),
                                    "d_k": float(dk_on),
                                    "rc": int(rc_on),
                                })
                                if pmask[r_i] and cb_map[s_val][li, r_i] > 0:
                                    a4_h4c_rows.append({
                                        "dataset": ds,
                                        "model": m_name,
                                        "s": s_val,
                                        "cluster_id": f"{ds}_{m_name}_{int(inst_id)}_{s_val}",
                                        "rc_disc_on": rc_on,
                                        "rc_disc_off": rc_off,
                                    })

                # --- Exp A3: Mitigation & Stabilization Head-to-Head at s = s_a3 ---
                if n_samp > n_a3_target:
                    a3_pos, _ = stratified_sample_indices(
                        yhat_samp, sample_n=n_a3_target, seed=seed + 900 + ds_idx * 10 + m_idx
                    )
                else:
                    a3_pos = np.arange(n_samp)
                n_a3 = len(a3_pos)
                X_a3 = X_samp[a3_pos]
                ids_a3 = samp_idx[a3_pos]
                phi_tshap_a3_base = phi_base_tshap[a3_pos]
                xp_a3 = x_primes_map[s_a3][a3_pos, :r_a3, :]
                pmask_a3 = pres_map[s_a3][a3_pos, :r_a3]
                a3_chunks = np.array_split(np.arange(n_a3), min(workers, n_a3))

                # 1) TreeSHAP (+ F_jump deferral at 80% and 70% coverage)
                t0 = time.perf_counter()
                _ = _parallel_treeshap(pool, est, bg, X_a3, n_chunks=workers)
                tshap_a3_ms = (time.perf_counter() - t0) * 1000.0 / float(n_a3)
                tshap_a3_summ = [
                    _pairwise_distance_summary(
                        phi_tshap_a3_base[li], tshap_pert_map[s_a3][a3_pos[li], :r_a3], pmask_a3[li],
                        top_k=top_k, y_cutoff=y_cutoff, floor_phis=None,
                    )
                    for li in range(n_a3)
                ]
                tshap_ubars = np.array([r["u_bar"] for r in tshap_a3_summ], dtype=float)
                tshap_dks = np.array([r["d_k"] for r in tshap_a3_summ], dtype=float)
                fj_a3 = _compute_f_jump_chunked(
                    X_a3, tree_splits, s=s_a3, feature_sds=prep.feature_sds_,
                    perturbable_mask=prep.perturbable_mask_, jump_mode="mean", sigma_mode="tree",
                )
                tshap_a3_summ_clean = pd.DataFrame(tshap_a3_summ)
                tshap_a3_summ_clean["noise_floor_dk"] = 0.0
                tshap_a3_summ_clean["noise_floor_rc"] = 0.0
                a3_rows.append({
                    "dataset": ds, "model": m_name, "family": "SHAP", "method": "TreeSHAP",
                    "coverage": 1.0,
                    "noise_floor_dk": 0.0,
                    "noise_floor_rc": 0.0,
                    "mean_d_k": float(tshap_a3_summ_clean["d_k"].mean()),
                    "mean_d_rho": float(tshap_a3_summ_clean["d_rho"].mean()),
                    "mean_u_bar": float(tshap_a3_summ_clean["u_bar"].mean()),
                    "instab_rate": float(tshap_a3_summ_clean["y_instab"].mean()),
                    "fidelity_dk_vs_treeshap": 0.0,
                    "mean_n_samples": np.nan,
                    "ms_per_inst": tshap_a3_ms,
                    "n_eval": int(n_a3),
                })
                for cov in (0.80, 0.70):
                    keep_cut = float(np.quantile(fj_a3, cov))
                    keep_m = fj_a3 <= keep_cut
                    if not np.any(keep_m):
                        keep_m[:] = True
                    a3_rows.append({
                        "dataset": ds, "model": m_name, "family": "SHAP",
                        "method": f"TreeSHAP+F_jump@{int(cov*100)}%",
                        "coverage": float(np.mean(keep_m)),
                        "noise_floor_dk": 0.0,
                        "noise_floor_rc": 0.0,
                        "mean_d_k": float(tshap_a3_summ_clean.loc[keep_m, "d_k"].mean()),
                        "mean_d_rho": float(tshap_a3_summ_clean.loc[keep_m, "d_rho"].mean()),
                        "mean_u_bar": float(tshap_a3_summ_clean.loc[keep_m, "u_bar"].mean()),
                        "instab_rate": float(tshap_a3_summ_clean.loc[keep_m, "y_instab"].mean()),
                        "fidelity_dk_vs_treeshap": 0.0,
                        "mean_n_samples": np.nan,
                        "ms_per_inst": tshap_a3_ms,
                        "n_eval": int(np.sum(keep_m)),
                    })

                # 2) AXOM (RF only, faithful to Pala & Garcia-Cuesta 2024) — combined base + pert batch to cut TreeExplainer builds in half
                if canon == "forest":
                    t0 = time.perf_counter()
                    flat_xp_a3 = xp_a3.reshape(n_a3 * r_a3, -1)
                    all_ax_in = np.vstack([X_a3, flat_xp_a3])
                    xp_chunks_ax = np.array_split(all_ax_in, min(workers, all_ax_in.shape[0]))
                    ax_all = np.vstack(pool(
                        delayed(_axom_chunk)(est, bg, ch, threshold)
                        for ch in xp_chunks_ax if len(ch) > 0
                    ))
                    ax_total_sec = time.perf_counter() - t0
                    ax_ms = ax_total_sec * 1000.0 / float(all_ax_in.shape[0])
                    ax_base = ax_all[:n_a3]
                    ax_pert = ax_all[n_a3:].reshape(n_a3, r_a3, -1)

                    ax_summ = [
                        _pairwise_distance_summary(
                            ax_base[li], ax_pert[li], pmask_a3[li], top_k=top_k, y_cutoff=y_cutoff
                        )
                        for li in range(n_a3)
                    ]
                    ax_df = pd.DataFrame(ax_summ)
                    ax_fid = [jaccard_distance_topk(phi_tshap_a3_base[li], ax_base[li], k=top_k) for li in range(n_a3)]
                    a3_rows.append({
                        "dataset": ds, "model": m_name, "family": "SHAP", "method": "AXOM",
                        "coverage": 1.0,
                        "noise_floor_dk": 0.0, "noise_floor_rc": 0.0,
                        "mean_d_k": float(ax_df["d_k"].mean()),
                        "mean_d_rho": float(ax_df["d_rho"].mean()),
                        "mean_u_bar": float(ax_df["u_bar"].mean()),
                        "instab_rate": float(ax_df["y_instab"].mean()),
                        "fidelity_dk_vs_treeshap": float(np.mean(ax_fid)),
                        "mean_n_samples": np.nan,
                        "ms_per_inst": ax_ms,
                        "n_eval": int(n_a3),
                    })
                    a3_pair_inputs.append({
                        "dataset": ds, "model": m_name, "comparison": "AXOM_vs_TreeSHAP",
                        "u_new": ax_df["u_bar"].to_numpy(), "u_ref": tshap_ubars,
                        "dk_new": ax_df["d_k"].to_numpy(), "dk_ref": tshap_dks,
                    })

                # 3) ST-SHAP Layer-1 Deterministic (budget = 2 * d, bg = bg_st)
                st_b_l1 = 2 * X_a3.shape[1]
                t0 = time.perf_counter()
                st_base = np.vstack(pool(
                    delayed(_st_shap_chunk)(est, bg_st, X_a3[ch], st_b_l1, seed + 401)
                    for ch in a3_chunks if len(ch) > 0
                ))
                st_ms = (time.perf_counter() - t0) * 1000.0 / float(n_a3)
                flat_xp_a3 = xp_a3.reshape(n_a3 * r_a3, -1)
                xp_chunks_st = np.array_split(flat_xp_a3, min(workers, flat_xp_a3.shape[0]))
                st_pert = np.vstack(pool(
                    delayed(_st_shap_chunk)(est, bg_st, ch, st_b_l1, seed + 401)
                    for ch in xp_chunks_st if len(ch) > 0
                )).reshape(n_a3, r_a3, -1)
                st_summ = [
                    _pairwise_distance_summary(
                        st_base[li], st_pert[li], pmask_a3[li], top_k=top_k, y_cutoff=y_cutoff
                    )
                    for li in range(n_a3)
                ]
                st_df = pd.DataFrame(st_summ)
                st_fid = [jaccard_distance_topk(phi_tshap_a3_base[li], st_base[li], k=top_k) for li in range(n_a3)]
                a3_rows.append({
                    "dataset": ds, "model": m_name, "family": "SHAP", "method": "ST_SHAP_L1",
                    "coverage": 1.0,
                    "noise_floor_dk": 0.0, "noise_floor_rc": 0.0,
                    "mean_d_k": float(st_df["d_k"].mean()),
                    "mean_d_rho": float(st_df["d_rho"].mean()),
                    "mean_u_bar": float(st_df["u_bar"].mean()),
                    "instab_rate": float(st_df["y_instab"].mean()),
                    "fidelity_dk_vs_treeshap": float(np.mean(st_fid)),
                    "mean_n_samples": float(st_b_l1),
                    "ms_per_inst": st_ms,
                    "n_eval": int(n_a3),
                })
                a3_pair_inputs.append({
                    "dataset": ds, "model": m_name, "comparison": "ST_SHAP_L1_vs_TreeSHAP",
                    "u_new": st_df["u_bar"].to_numpy(), "u_ref": tshap_ubars,
                    "dk_new": st_df["d_k"].to_numpy(), "dk_ref": tshap_dks,
                })

                # 4) LIME family: LIME_std vs LIME_disc_off vs S-LIME at s = s_a3
                if ds in non_credit_datasets:
                    # Reuse already computed base + floor + first r_a3 perturbations
                    lime_on_a3 = {
                        int(inst_id): {
                            "phi_base": lime_on_map[int(inst_id)]["phi_base"],
                            "floor_phis": lime_on_map[int(inst_id)]["floor_phis"][:k_a3],
                            "pert_phis": {s_a3: lime_on_map[int(inst_id)]["pert_phis"][s_a3][:r_a3]},
                        }
                        for inst_id in ids_a3
                    }
                    lime_off_a3 = {
                        int(inst_id): {
                            "phi_base": lime_off_map[int(inst_id)]["phi_base"],
                            "floor_phis": np.empty((0, X_a3.shape[1])),
                            "pert_phis": {s_a3: lime_off_map[int(inst_id)]["pert_phis"][s_a3][:r_a3]},
                        }
                        for inst_id in ids_a3
                    }
                    # Quick timing pass for 20 instances of LIME_std and LIME_disc_off
                    t0 = time.perf_counter()
                    _ = explain_lime(le_on_ref, est, X_a3[:20], num_samples=lime_num_samples, seed=seed)
                    lime_on_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
                    le_off_tmp = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=False, seed=seed)
                    t0 = time.perf_counter()
                    _ = explain_lime(le_off_tmp, est, X_a3[:20], num_samples=lime_num_samples, seed=seed)
                    lime_off_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
                else:
                    t0 = time.perf_counter()
                    res_on_a3 = pool(
                        delayed(_lime_worker_batch)(
                            X_tr, feature_names, est,
                            X_a3[ch], ids_a3[ch], {s_a3: xp_a3[ch]},
                            k_a3, lime_num_samples, True, seed + 2100 + ds_idx * 50 + m_idx * 10,
                        )
                        for ch in a3_chunks if len(ch) > 0
                    )
                    lime_on_ms = (time.perf_counter() - t0) * 1000.0 / float(n_a3 * (1 + k_a3 + r_a3))
                    lime_on_a3 = {}
                    for d_c in res_on_a3:
                        lime_on_a3.update(d_c)

                    t0 = time.perf_counter()
                    res_off_a3 = pool(
                        delayed(_lime_worker_batch)(
                            X_tr, feature_names, est,
                            X_a3[ch], ids_a3[ch], {s_a3: xp_a3[ch]},
                            k_a3, lime_num_samples, False, seed + 3100 + ds_idx * 50 + m_idx * 10,
                        )
                        for ch in a3_chunks if len(ch) > 0
                    )
                    lime_off_ms = (time.perf_counter() - t0) * 1000.0 / float(n_a3 * (1 + k_a3 + r_a3))
                    lime_off_a3 = {}
                    for d_c in res_off_a3:
                        lime_off_a3.update(d_c)

                # Run S-LIME (Zhou et al., KDD 2021)
                t0 = time.perf_counter()
                slime_chunks = pool(
                    delayed(_slime_worker_chunk)(
                        X_tr, feature_names, est,
                        X_a3[ch], ids_a3[ch], xp_a3[ch],
                        k_a3, slime_n_init, slime_n_max, top_k, seed + 4100 + ds_idx * 50 + m_idx * 10,
                    )
                    for ch in a3_chunks if len(ch) > 0
                )
                slime_ms = (time.perf_counter() - t0) * 1000.0 / float(n_a3 * (1 + k_a3 + r_a3))
                slime_a3: dict[int, dict[str, Any]] = {}
                for d_c in slime_chunks:
                    slime_a3.update(d_c)

                # Summarize LIME_std, LIME_disc_off, S_LIME, and LIME_std + F_bin deferral
                lon_summ = [
                    _pairwise_distance_summary(
                        lime_on_a3[int(iid)]["phi_base"],
                        lime_on_a3[int(iid)]["pert_phis"][s_a3],
                        pmask_a3[li],
                        top_k=top_k,
                        y_cutoff=y_cutoff,
                        floor_phis=lime_on_a3[int(iid)]["floor_phis"],
                    )
                    for li, iid in enumerate(ids_a3)
                ]
                loff_summ = [
                    _pairwise_distance_summary(
                        lime_off_a3[int(iid)]["phi_base"],
                        lime_off_a3[int(iid)]["pert_phis"][s_a3],
                        pmask_a3[li],
                        top_k=top_k,
                        y_cutoff=y_cutoff,
                        floor_phis=lime_off_a3[int(iid)]["floor_phis"] if len(lime_off_a3[int(iid)]["floor_phis"]) else None,
                    )
                    for li, iid in enumerate(ids_a3)
                ]
                sl_summ = [
                    _pairwise_distance_summary(
                        slime_a3[int(iid)]["phi_base"],
                        slime_a3[int(iid)]["pert_phis"],
                        pmask_a3[li],
                        top_k=top_k,
                        y_cutoff=y_cutoff,
                        floor_phis=slime_a3[int(iid)]["floor_phis"],
                    )
                    for li, iid in enumerate(ids_a3)
                ]

                lon_df = pd.DataFrame(lon_summ)
                loff_df = pd.DataFrame(loff_summ)
                sl_df = pd.DataFrame(sl_summ)
                sl_n_mean = float(np.mean([slime_a3[int(iid)]["n_used_mean"] for iid in ids_a3]))

                for m_lbl, dframe, n_s_val, ms_val in [
                    ("LIME_std", lon_df, float(lime_num_samples), lime_on_ms),
                    ("S_LIME", sl_df, sl_n_mean, slime_ms),
                    ("LIME_disc_off", loff_df, float(lime_num_samples), lime_off_ms),
                ]:
                    a3_rows.append({
                        "dataset": ds, "model": m_name, "family": "LIME", "method": m_lbl,
                        "coverage": 1.0,
                        "noise_floor_dk": float(dframe["noise_floor_dk"].mean()),
                        "noise_floor_rc": float(dframe["noise_floor_rc"].mean()),
                        "mean_d_k": float(dframe["d_k"].mean()),
                        "mean_d_rho": float(dframe["d_rho"].mean()),
                        "mean_u_bar": float(dframe["u_bar"].mean()),
                        "instab_rate": float(dframe["y_instab"].mean()),
                        "fidelity_dk_vs_treeshap": np.nan,
                        "mean_n_samples": n_s_val,
                        "ms_per_inst": ms_val,
                        "n_eval": int(n_a3),
                    })

                f_bin_a3 = np.array([
                    compute_f_bin(X_a3[li], bin_edges, s=s_a3, feature_sds=prep.feature_sds_, perturbable_mask=prep.perturbable_mask_)
                    for li in range(n_a3)
                ], dtype=float)
                keep_bin = f_bin_a3 <= float(np.quantile(f_bin_a3, 0.70))
                if not np.any(keep_bin):
                    keep_bin[:] = True
                a3_rows.append({
                    "dataset": ds, "model": m_name, "family": "LIME", "method": "LIME_std+F_bin@70%",
                    "coverage": float(np.mean(keep_bin)),
                    "noise_floor_dk": float(lon_df.loc[keep_bin, "noise_floor_dk"].mean()),
                    "noise_floor_rc": float(lon_df.loc[keep_bin, "noise_floor_rc"].mean()),
                    "mean_d_k": float(lon_df.loc[keep_bin, "d_k"].mean()),
                    "mean_d_rho": float(lon_df.loc[keep_bin, "d_rho"].mean()),
                    "mean_u_bar": float(lon_df.loc[keep_bin, "u_bar"].mean()),
                    "instab_rate": float(lon_df.loc[keep_bin, "y_instab"].mean()),
                    "fidelity_dk_vs_treeshap": np.nan,
                    "mean_n_samples": float(lime_num_samples),
                    "ms_per_inst": lime_on_ms,
                    "n_eval": int(np.sum(keep_bin)),
                })

                a3_pair_inputs.append({
                    "dataset": ds, "model": m_name, "comparison": "S_LIME_vs_LIME_std",
                    "u_new": sl_df["u_bar"].to_numpy(), "u_ref": lon_df["u_bar"].to_numpy(),
                    "dk_new": sl_df["d_k"].to_numpy(), "dk_ref": lon_df["d_k"].to_numpy(),
                    "nf_new": sl_df["noise_floor_dk"].to_numpy(), "nf_ref": lon_df["noise_floor_dk"].to_numpy(),
                })
                a3_pair_inputs.append({
                    "dataset": ds, "model": m_name, "comparison": "LIME_disc_off_vs_LIME_std",
                    "u_new": loff_df["u_bar"].to_numpy(), "u_ref": lon_df["u_bar"].to_numpy(),
                    "dk_new": loff_df["d_k"].to_numpy(), "dk_ref": lon_df["d_k"].to_numpy(),
                })

                print(f"  [A*] {ds}/{m_name} finished in {time.time() - t_m:.1f}s")
            print(f"  [A*] Dataset {ds} finished in {time.time() - t_ds:.1f}s")
            tmp_ckpt = ckpt_dir / f"ckpt_{ds}.pkl.tmp"
            with tmp_ckpt.open("wb") as fh:
                pickle.dump({
                    "model_perf_rows": model_perf_rows[_starts[0]:],
                    "a1_instance_rows": a1_instance_rows[_starts[1]:],
                    "a1_runtime_records": a1_runtime_records[_starts[2]:],
                    "a3_rows": a3_rows[_starts[3]:],
                    "a3_pair_inputs": a3_pair_inputs[_starts[4]:],
                    "a4_draws_rows": a4_draws_rows[_starts[5]:],
                    "a4_h4c_rows": a4_h4c_rows[_starts[6]:],
                }, fh)
            tmp_ckpt.replace(ckpt_file)

    # =========================================================================
    # Save raw Exp A1 instance table & model performance
    # =========================================================================
    perf_df = pd.DataFrame(model_perf_rows)
    perf_df.to_csv(out_dir / "astar_model_performance.csv", index=False)

    inst_df = pd.DataFrame(a1_instance_rows)
    inst_df.to_parquet(out_dir / "astar_exp_a1_instance_scores.parquet", index=False)
    rt_df = pd.DataFrame(a1_runtime_records)
    rt_df.to_csv(out_dir / "astar_exp_a1_runtimes.csv", index=False)

    # Build runtime lookup (fit_ms_total, infer_ms_per_inst)
    def _rt_map(sub_rt: pd.DataFrame) -> dict[str, tuple[float, float]]:
        if sub_rt.empty:
            return {}
        fit_sp = float(sub_rt["fit_splits_ms"].mean())
        return {
            "f_jump": (fit_sp, float(sub_rt["f_jump_ms"].mean())),
            "f_jump_ens": (fit_sp, float(sub_rt["f_jump_ms"].mean())),
            "f_jump_range": (fit_sp, float(sub_rt["f_jump_ms"].mean())),
            "f_unw": (fit_sp, float(sub_rt["f_unw_ms"].mean())),
            "f_cal_quant": (fit_sp, float(sub_rt["f_jump_ms"].mean())),
            "gpec": (0.0, float(sub_rt["gpec_ms"].mean())),
            "epistemic_uq": (0.0, float(sub_rt["epistemic_uq_ms"].mean())),
            "b_var": (0.0, float(sub_rt["epistemic_uq_ms"].mean())),
            "b_margin": (0.0, float(sub_rt["b_margin_ms"].mean())),
            "posthoc_abs_dp": (0.0, float(sub_rt["posthoc_treeshap_ms"].mean())),
        }

    # =========================================================================
    # Statistical Aggregation: Exp A1 (AUROC, AURC, Spearman, Paired Tests, Sigma Ablation)
    # =========================================================================
    predictors = [
        "f_jump", "f_jump_ens", "f_jump_range", "f_unw",
        "f_cal_quant", "gpec", "epistemic_uq", "b_var", "b_margin", "posthoc_abs_dp",
    ]
    baselines = ["f_unw", "gpec", "epistemic_uq", "b_var", "b_margin", "posthoc_abs_dp"]

    a1_pred_rows: list[dict[str, Any]] = []
    a1_pair_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []

    slices: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    for ds in datasets:
        for m_name in models:
            slices.append((
                ds, m_name,
                inst_df[(inst_df["dataset"] == ds) & (inst_df["model"] == m_name)],
                rt_df[(rt_df["dataset"] == ds) & (rt_df["model"] == m_name)],
            ))
    for m_name in models + ["POOLED"]:
        for grp_name, cond in [
            ("POOLED_ALL", pd.Series(True, index=inst_df.index)),
            ("POOLED_CREDIT", inst_df["domain"] == "credit"),
            ("POOLED_NONCREDIT", inst_df["domain"] == "non_credit"),
        ]:
            m_cond = pd.Series(True, index=inst_df.index) if m_name == "POOLED" else (inst_df["model"] == m_name)
            m_rt = pd.Series(True, index=rt_df.index) if m_name == "POOLED" else (rt_df["model"] == m_name)
            d_rt = (
                pd.Series(True, index=rt_df.index)
                if grp_name == "POOLED_ALL"
                else rt_df["dataset"].isin(non_credit_datasets if grp_name == "POOLED_NONCREDIT" else set(datasets) - non_credit_datasets)
            )
            slices.append((grp_name, m_name, inst_df[cond & m_cond], rt_df[m_rt & d_rt]))

    valid_slices = [(scope_ds, scope_m, sub_df, _rt_map(sub_rt)) for scope_ds, scope_m, sub_df, sub_rt in slices if not sub_df.empty]
    slice_results = Parallel(n_jobs=workers)(
        delayed(_stats_slice_worker)(
            scope_ds, scope_m, sub_df, rmap, predictors, baselines, n_boot, seed, alpha
        )
        for scope_ds, scope_m, sub_df, rmap in valid_slices
    )
    for p_rows, pair_rows, sig_row in slice_results:
        a1_pred_rows.extend(p_rows)
        a1_pair_rows.extend(pair_rows)
        if sig_row is not None:
            sigma_rows.append(sig_row)

    a1_pred_df = pd.DataFrame(a1_pred_rows)
    a1_pair_df = pd.DataFrame(a1_pair_rows)
    if not a1_pair_df.empty:
        # Apply Holm correction within (scope_dataset, model) across baselines
        a1_pair_df["boot_auroc_p_holm"] = np.nan
        a1_pair_df["boot_aurc_p_holm"] = np.nan
        for _, grp in a1_pair_df.groupby(["scope_dataset", "model"]):
            a1_pair_df.loc[grp.index, "boot_auroc_p_holm"] = holm_bonferroni(grp["boot_auroc_p"].to_numpy(float))["p_adjusted"]
            a1_pair_df.loc[grp.index, "boot_aurc_p_holm"] = holm_bonferroni(grp["boot_aurc_p"].to_numpy(float))["p_adjusted"]

    a1_pred_df.to_csv(out_dir / "astar_exp_a1_predictors.csv", index=False)
    a1_pair_df.to_csv(out_dir / "astar_exp_a1_pairwise_tests.csv", index=False)
    pd.DataFrame(sigma_rows).to_csv(out_dir / "astar_exp_a1_sigma_ablation.csv", index=False)

    # =========================================================================
    # Exp A2: LODO Gating
    # =========================================================================
    lodo_df, lodo_summary = run_lodo_experiment(inst_df, datasets, models)
    lodo_df.to_csv(out_dir / "astar_exp_a2_lodo.csv", index=False)
    with (out_dir / "astar_exp_a2_lodo_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(lodo_summary, fh, indent=2)

    # =========================================================================
    # Exp A3: Mitigation & Stabilization Head-to-Head
    # =========================================================================
    a3_df = pd.DataFrame(a3_rows)
    a3_df.to_csv(out_dir / "astar_exp_a3_mitigation.csv", index=False)

    a3_test_rows: list[dict[str, Any]] = []
    for item in a3_pair_inputs:
        for metric_k, arr_new, arr_ref in [
            ("u_bar", item["u_new"], item["u_ref"]),
            ("d_k", item["dk_new"], item["dk_ref"]),
        ] + ([("noise_floor_dk", item["nf_new"], item["nf_ref"])] if "nf_new" in item else []):
            w_res = paired_wilcoxon_test(arr_new, arr_ref, alternative="two-sided")
            a3_test_rows.append({
                "dataset": item["dataset"],
                "model": item["model"],
                "comparison": item["comparison"],
                "metric": metric_k,
                "mean_new": float(np.nanmean(arr_new)),
                "mean_ref": float(np.nanmean(arr_ref)),
                "mean_diff": float(np.nanmean(arr_new - arr_ref)),
                "wilcoxon_p": w_res["pvalue"],
                "rank_biserial_r": w_res["rank_biserial_r"],
                "n_pairs": w_res["n_pairs"],
            })
    a3_test_df = pd.DataFrame(a3_test_rows)
    if not a3_test_df.empty:
        a3_test_df["wilcoxon_p_holm"] = holm_bonferroni(a3_test_df["wilcoxon_p"].to_numpy(float))["p_adjusted"]
    a3_test_df.to_csv(out_dir / "astar_exp_a3_paired_tests.csv", index=False)

    # =========================================================================
    # Exp A4: Cross-Domain Replication of H4b & H4c on Non-Credit Datasets
    # =========================================================================
    a4_draws_df = pd.DataFrame(a4_draws_rows)
    h4b_rows: list[dict[str, Any]] = []
    if not a4_draws_df.empty:
        pres_draws = a4_draws_df[a4_draws_df["preserved"]].copy()
        for exp_lbl in ["TreeSHAP", "LIME"]:
            for scope_ds in sorted(non_credit_datasets) + ["POOLED_NONCREDIT"]:
                sub = (
                    pres_draws[pres_draws["explainer"] == exp_lbl]
                    if scope_ds == "POOLED_NONCREDIT"
                    else pres_draws[(pres_draws["explainer"] == exp_lbl) & (pres_draws["dataset"] == scope_ds)]
                )
                if len(sub) < 50 or sub["rc"].nunique() < 2:
                    continue
                X_mat = np.column_stack([
                    sub["c_tree"].to_numpy(float),
                    sub["c_bin"].to_numpy(float),
                    sub["abs_dp"].to_numpy(float),
                    sub["s"].to_numpy(float),
                ])
                res_lr = joint_logistic_cluster_robust(
                    sub["rc"].to_numpy(int),
                    X_mat,
                    sub["cluster_id"].to_numpy(),
                    feature_names=["c_tree", "c_bin", "abs_dp", "s"],
                )
                terms = res_lr["terms"]
                h4b_rows.append({
                    "scope_dataset": scope_ds,
                    "explainer": exp_lbl,
                    "c_tree_coef": terms["c_tree"]["coef"],
                    "c_tree_or": terms["c_tree"]["odds_ratio"],
                    "c_tree_p": terms["c_tree"]["pvalue"],
                    "c_bin_coef": terms["c_bin"]["coef"],
                    "c_bin_or": terms["c_bin"]["odds_ratio"],
                    "c_bin_p": terms["c_bin"]["pvalue"],
                    "abs_dp_coef": terms["abs_dp"]["coef"],
                    "abs_dp_p": terms["abs_dp"]["pvalue"],
                    "n_obs": res_lr["n_obs"],
                    "n_clusters": res_lr["n_clusters"],
                })
    h4b_df = pd.DataFrame(h4b_rows)
    if not h4b_df.empty:
        h4b_df["c_tree_p_holm"] = holm_bonferroni(h4b_df["c_tree_p"].to_numpy(float))["p_adjusted"]
        h4b_df["c_bin_p_holm"] = holm_bonferroni(h4b_df["c_bin_p"].to_numpy(float))["p_adjusted"]
    h4b_df.to_csv(out_dir / "astar_exp_a4_h4b_noncredit.csv", index=False)

    h4c_df_raw = pd.DataFrame(a4_h4c_rows)
    h4c_rows: list[dict[str, Any]] = []
    if not h4c_df_raw.empty:
        # Aggregate per (dataset, model, s, instance) before paired Wilcoxon to prevent draw-level pseudo-replication
        h4c_inst = (
            h4c_df_raw.groupby(["dataset", "model", "s", "cluster_id"], as_index=False)[["rc_disc_on", "rc_disc_off"]]
            .mean()
        )
        for ds_lbl in sorted(non_credit_datasets) + ["POOLED_NONCREDIT"]:
            sub = h4c_inst if ds_lbl == "POOLED_NONCREDIT" else h4c_inst[h4c_inst["dataset"] == ds_lbl]
            if sub.empty:
                continue
            w_res = paired_wilcoxon_test(
                sub["rc_disc_off"].to_numpy(float),
                sub["rc_disc_on"].to_numpy(float),
                alternative="less",
            )
            h4c_rows.append({
                "scope_dataset": ds_lbl,
                "rc_rate_disc_on": float(sub["rc_disc_on"].mean()),
                "rc_rate_disc_off": float(sub["rc_disc_off"].mean()),
                "abs_reduction": float(sub["rc_disc_on"].mean() - sub["rc_disc_off"].mean()),
                "wilcoxon_p": w_res["pvalue"],
                "rank_biserial_r": w_res["rank_biserial_r"],
                "n_pairs": w_res["n_pairs"],
            })
    h4c_df = pd.DataFrame(h4c_rows)
    h4c_df.to_csv(out_dir / "astar_exp_a4_h4c_noncredit.csv", index=False)

    # Top-level summary JSON
    elapsed = time.time() - t_start
    pooled_a1 = a1_pred_df[(a1_pred_df["scope_dataset"] == "POOLED_ALL") & (a1_pred_df["model"] == "POOLED")]
    summary_payload = {
        "mode": "smoke" if args.smoke else "full",
        "elapsed_sec": round(elapsed, 2),
        "datasets": datasets,
        "models": models,
        "n_eval_per_dataset_model": sample_n,
        "pooled_a1_predictors": pooled_a1[["predictor", "auroc", "aurc", "e_aurc", "spearman_rho_dk", "infer_ms_per_inst"]].to_dict(orient="records") if not pooled_a1.empty else [],
        "lodo_summary_pooled": lodo_summary,
        "a4_h4b_pooled": h4b_df[h4b_df["scope_dataset"] == "POOLED_NONCREDIT"].to_dict(orient="records") if not h4b_df.empty else [],
        "a4_h4c_pooled": h4c_df[h4c_df["scope_dataset"] == "POOLED_NONCREDIT"].to_dict(orient="records") if not h4c_df.empty else [],
    }
    with (out_dir / "astar_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    print(f"=== Stage 06 Complete in {elapsed:.1f}s -> {out_dir} ===")


if __name__ == "__main__":
    main()
