"""Stage 03: Part 2 Extension Experiment, Ablations (H4c, R6, H6), Case Study, and Statistical Suite.

Runs fixed-model prediction-preserving perturbations on Taiwan Default, HMEQ, and SGCS
for LR, RF, and XGB (n=200 stratified 50/50 instances per dataset, K=10 noise floor,
R=25 draws x s in {0.01, 0.05}), computes crossings & ex-ante fragility F(x),
runs H4c (LIME discretisation OFF), R6 (PermutationExplainer bridge), H6 (IDCS bridge),
selects the HMEQ adverse-action case study, and computes H2-H6 + Measurement Quality stats.

Usage:
    python scripts/03_extend.py --config config.yaml
    python scripts/03_extend.py --config config.yaml --stats-only
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.crossings import (  # noqa: E402
    TreeSplits,
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
    _extract_pos_shap,
    build_lime_explainer,
    explain_lime,
    explain_linear_shap,
    explain_permutation_shap,
    explain_tree_shap,
    sample_background,
)
from src.fragility import (  # noqa: E402
    baseline_margin,
    baseline_tree_variance,
    compute_f_bin,
    compute_f_tree,
)
from src.idcs import compute_cost_vectors  # noqa: E402
from src.metrics import (  # noqa: E402
    jaccard_distance_topk,
    normalized_l1_distance,
    spearman_distance,
    top_k_indices,
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
    clopper_pearson_ci,
    cluster_bootstrap_auroc,
    cluster_bootstrap_mediation,
    delong_paired_auroc_test,
    holm_bonferroni,
    joint_logistic_cluster_robust,
    measurement_quality_suite,
    nested_ols_hc3,
    paired_bootstrap_auroc_diff,
    paired_wilcoxon_test,
    proportion_nonzero_with_ci,
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
        # Top up from whichever class has remaining rows
        rem = n_target - (n_pos_take + n_neg_take)
        if len(pos_idx) - n_pos_take >= rem:
            n_pos_take += rem
        else:
            n_neg_take = min(len(neg_idx), n_neg_take + rem)

    chosen_pos = rng.choice(pos_idx, size=n_pos_take, replace=False) if n_pos_take > 0 else np.empty(0, dtype=int)
    chosen_neg = rng.choice(neg_idx, size=n_neg_take, replace=False) if n_neg_take > 0 else np.empty(0, dtype=int)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    chosen.sort()

    # Inverse-probability population weights normalized to mean 1
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


class _FastLogitProxy:
    """Bit-identical fast predict_proba wrapper for sklearn LogisticRegression."""

    def __init__(self, coef: np.ndarray, intercept: float) -> None:
        self.coef_ = np.asarray(coef, dtype=float).ravel()
        self.intercept_ = float(intercept)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


def _unwrap_est(est_or_params: Any) -> Any:
    if isinstance(est_or_params, tuple) and len(est_or_params) == 3 and est_or_params[0] == "__fast_logit__":
        return _FastLogitProxy(est_or_params[1], est_or_params[2])
    return est_or_params


def _to_worker_est(est: Any) -> Any:
    if type(est).__name__ == "LogisticRegression":
        return (
            "__fast_logit__",
            np.asarray(est.coef_, dtype=float).ravel(),
            float(np.asarray(est.intercept_, dtype=float).ravel()[0]),
        )
    return est


def _lime_worker_chunk(
    X_tr: np.ndarray,
    feature_names: list[str],
    est_or_params: Any,
    X_inst_chunk: np.ndarray,
    inst_ids: np.ndarray,
    x_primes_by_scale: dict[float, np.ndarray],
    k_floor: int,
    num_samples: int,
    discretize: bool,
    base_seed: int,
) -> dict[int, dict[str, Any]]:
    """Run LIME baseline, K noise-floor repeats, and perturbed explanations for a chunk of instances."""
    est = _unwrap_est(est_or_params)
    le = build_lime_explainer(
        X_tr, feature_names=feature_names, discretize_continuous=discretize, seed=base_seed
    )
    out: dict[int, dict[str, Any]] = {}
    for local_i, inst_id in enumerate(inst_ids):
        x_row = X_inst_chunk[local_i : local_i + 1]
        # Independent baseline seed (shared-randomness CMB control)
        phi_base = explain_lime(
            le, est, x_row, num_samples=num_samples, seed=base_seed + 900000 + int(inst_id) * 101
        )[0]
        # K noise-floor repeats
        floor_phis = np.vstack([
            explain_lime(
                le, est, x_row, num_samples=num_samples, seed=base_seed + 100000 + int(inst_id) * 100 + kk
            )
            for kk in range(k_floor)
        ])
        pert_phis: dict[float, np.ndarray] = {}
        for s_val, xp_all in x_primes_by_scale.items():
            xp_inst = xp_all[local_i]  # (R, P)
            pert_phis[s_val] = explain_lime(
                le, est, xp_inst, num_samples=num_samples,
                seed=base_seed + 200000 + int(s_val * 1000) * 10000 + int(inst_id) * 100,
            )
        out[int(inst_id)] = {
            "phi_base": phi_base,
            "floor_phis": floor_phis,
            "pert_phis": pert_phis,
        }
    return out


def _perm_worker_chunk(
    est: Any,
    bg: np.ndarray,
    X_inst_chunk: np.ndarray,
    inst_ids: np.ndarray,
    xp_chunk: np.ndarray,
    pres_masks: list[np.ndarray],
    perm_max_evals: int,
    k_floor: int,
    top_k: int,
    ds: str,
    m_name: str,
    primary_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Run PermutationExplainer bridge on a chunk of instances."""
    rows_out: list[dict[str, Any]] = []
    for local_i, iid_raw in enumerate(inst_ids):
        iid = int(iid_raw)
        pres_mask = pres_masks[local_i]
        if not np.any(pres_mask):
            continue
        x_row = X_inst_chunk[local_i : local_i + 1]
        phi_b = explain_permutation_shap(est, bg, x_row, max_evals=perm_max_evals, seed=seed + 8888 + iid)[0]
        fl_phis = np.vstack([
            explain_permutation_shap(est, bg, x_row, max_evals=perm_max_evals, seed=seed + 9999 + iid * 20 + kk)
            for kk in range(k_floor)
        ])
        q_perm, _ = _noise_floor_stats(fl_phis, k_top=top_k)
        xp = xp_chunk[local_i]
        phi_ps = explain_permutation_shap(est, bg, xp, max_evals=perm_max_evals, seed=seed + 7777 + iid)
        dks = [jaccard_distance_topk(phi_b, phi_ps[r_i], k=top_k) for r_i in range(len(xp))]
        rcs = [int(d > q_perm + 1e-12) for d in dks]
        rows_out.append({
            "dataset": ds,
            "model": m_name,
            "instance_id": iid,
            "s": primary_s,
            "perm_noise_floor": q_perm,
            "perm_d_k": float(np.mean(np.asarray(dks)[pres_mask])),
            "perm_u_bar": float(np.mean(np.asarray(rcs)[pres_mask])),
        })
    return rows_out


def _noise_floor_stats(floor_phis: np.ndarray, k_top: int) -> tuple[float, float]:
    """Return (q_95, mean_floor_dk) over all a < b pairs in K repeats."""
    K = floor_phis.shape[0]
    if K < 2:
        return 0.0, 0.0
    dks = []
    for a in range(K):
        for b in range(a + 1, K):
            dks.append(jaccard_distance_topk(floor_phis[a], floor_phis[b], k=k_top))
    arr = np.asarray(dks, dtype=float)
    if np.all(arr == 0.0):
        return 0.0, 0.0
    return float(np.percentile(arr, 95)), float(np.mean(arr))


def run_part2_experiments(cfg: dict[str, Any], n_workers: int) -> dict[str, Any]:
    """Run Part 2 main experiment, H4c ablation, R6 Permutation bridge, and H6 IDCS."""
    seed = int(cfg.get("seed", 42))
    test_frac = float(cfg["data"].get("part2_test_frac", 0.30))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    p2_cfg = cfg["part2"]
    datasets = list(p2_cfg["datasets"])
    models = list(p2_cfg["models"])
    idcs_models = list(p2_cfg.get("idcs_h6_models", []))
    sample_n = int(p2_cfg["sample_n"])
    k_floor = int(p2_cfg["noise_floor_k"])
    R = int(p2_cfg["perturbation_r"])
    scales = [float(s) for s in p2_cfg["scales_s"]]
    primary_s = max(scales) if 0.05 not in scales else 0.05
    tau = float(p2_cfg["tau"])
    top_k = int(p2_cfg["top_k"])
    y_cutoff = float(p2_cfg["instability_cutoff"])
    bg_size = int(p2_cfg["tree_shap_background_size"])
    lime_samples = int(p2_cfg["lime_num_samples"])
    perm_max_evals = int(p2_cfg.get("permutation_shap_max_evals", 500))
    cv_folds = int(cfg["part1"].get("cv_folds", 5))
    cost_cfg = cfg["cost_model"]

    perf_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    inst_rows: list[dict[str, Any]] = []
    icc_records: list[dict[str, Any]] = []
    h4c_paired_rows: list[dict[str, Any]] = []
    r6_bridge_rows: list[dict[str, Any]] = []
    h6_idcs_rows: list[dict[str, Any]] = []
    case_study_payload: dict[str, Any] = {}

    parallel = Parallel(n_jobs=n_workers)
    parallel.__enter__()
    try:
      for ds_idx, ds in enumerate(datasets):
        df, cb = load_dataset(ds, cfg)
        train_df, test_df = make_part2_split(df, test_frac=test_frac, seed=seed)
        prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_df)

        X_tr = prep.transform(train_df)
        y_tr = train_df["target"].to_numpy(dtype=int)
        X_te = prep.transform(test_df)
        y_te = test_df["target"].to_numpy(dtype=int)
        feature_names = prep.feature_names_

        prior_pi1 = float(np.mean(y_tr))
        threshold = prior_pi1 if p2_cfg["threshold_mode"] == "prior" else float(p2_cfg["threshold_mode"])
        bg = sample_background(X_tr, n=bg_size, seed=seed + ds_idx)

        a_tr = prep.extract_exposure(train_df)
        a_te = prep.extract_exposure(test_df)
        c_fn_tr, c_fp_tr = compute_cost_vectors(
            a_tr, a_tr, y_tr, lgd=float(cost_cfg["lgd"]), prior_pi1=prior_pi1,
            int_r_annual=float(cost_cfg["int_r_annual"]),
            int_cf_annual=float(cost_cfg["int_cf_annual"]),
            n_term_months=int(cost_cfg["n_term_months"]),
        )
        c_fn_te, c_fp_te = compute_cost_vectors(
            a_te, a_tr, y_tr, lgd=float(cost_cfg["lgd"]), prior_pi1=prior_pi1,
            int_r_annual=float(cost_cfg["int_r_annual"]),
            int_cf_annual=float(cost_cfg["int_cf_annual"]),
            n_term_months=int(cost_cfg["n_term_months"]),
        )

        # Build LIME discretiser once per dataset (training data is fixed in Part 2)
        lime_ref_on = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed)
        bin_edges = extract_lime_bin_edges(lime_ref_on)

        # Fit RF model first (or cache for surrogate if desired), plus all primary models
        fitted_models: dict[str, Any] = {}
        rf_for_surrogate = None

        all_model_list = models + [m for m in idcs_models if m not in models]
        for m_idx, m_name in enumerate(all_model_list):
            canon_m = canonical_model_name(m_name)
            m_seed = seed + ds_idx * 100 + m_idx
            best_p = tune_hyperparameters(
                canon_m, X_tr, y_tr, c_fn=c_fn_tr, c_fp=c_fp_tr,
                cv_folds=cv_folds, seed=m_seed, n_jobs=1,
            )
            est = build_estimator(canon_m, params=best_p, seed=m_seed, n_jobs=1)
            fit_estimator(est, canon_m, X_tr, y_tr, c_fn=c_fn_tr, c_fp=c_fp_tr)
            fitted_models[m_name] = est
            if m_name == "rf":
                rf_for_surrogate = est

            p_te = est.predict_proba(X_te)[:, 1]
            ev_prior = evaluate_model(y_te, p_te, c_fn=c_fn_te, c_fp=c_fp_te, prior_pi1=prior_pi1, threshold=threshold)
            ev_half = evaluate_model(y_te, p_te, c_fn=c_fn_te, c_fp=c_fp_te, prior_pi1=prior_pi1, threshold=0.5)
            perf_rows.append({
                "dataset": ds,
                "model": m_name,
                "canon_model": canon_m,
                "is_idcs": m_name in idcs_models,
                "threshold_prior": threshold,
                "auc": ev_prior["auc"],
                "ap": ev_prior["ap"],
                "brier": ev_prior["brier"],
                "rel_aec": ev_prior["rel_aec"],
                "savings_at_prior": ev_prior["savings"],
                "savings_at_05": ev_half["savings"],
                "best_params": json.dumps(best_p),
            })

        # Main Part 2 loop over models ["lr", "rf", "xgb"]
        for m_idx, m_name in enumerate(models):
            t_m = time.time()
            est = fitted_models[m_name]
            canon_m = canonical_model_name(m_name)
            p_te = est.predict_proba(X_te)[:, 1]
            y_hat_te = (p_te >= threshold).astype(int)

            # Stratified 50/50 sample of n=200 test instances
            samp_idx, pop_weights = stratified_sample_indices(y_hat_te, sample_n=sample_n, seed=seed + ds_idx * 10 + m_idx)
            X_samp = X_te[samp_idx]
            y_samp = y_te[samp_idx]
            p_samp = p_te[samp_idx]
            yhat_samp = y_hat_te[samp_idx]
            n_samp = len(samp_idx)

            # Tree splits & tree-variance baseline
            is_tree = canon_m in ("forest", "boost", "csboost")
            tree_splits = extract_tree_splits(est, feature_names=feature_names) if is_tree else None
            if canon_m == "forest":
                tree_vars = baseline_tree_variance(est, X_samp)
            elif rf_for_surrogate is not None:
                # Surrogate RF variance on XGB/LR if available
                tree_vars = baseline_tree_variance(rf_for_surrogate, X_samp)
            else:
                tree_vars = np.full(n_samp, np.nan)
            margins = baseline_margin(p_samp, threshold=threshold)

            # Generate perturbations for all n_samp x scales -> dict[s, (n_samp, R, P)]
            x_primes_map: dict[float, np.ndarray] = {}
            filter_info_map: dict[float, list[dict[str, Any]]] = {}
            c_tree_map: dict[float, np.ndarray] = {}
            c_bin_map: dict[float, np.ndarray] = {}

            for s_val in scales:
                xp_all = np.zeros((n_samp, R, X_samp.shape[1]), dtype=float)
                ct_all = np.zeros((n_samp, R), dtype=float)
                cb_all = np.zeros((n_samp, R), dtype=float)
                f_infos: list[dict[str, Any]] = []
                for li, inst_id in enumerate(samp_idx):
                    p_seed = seed + ds_idx * 1000000 + m_idx * 100000 + int(s_val * 1000) * 1000 + int(inst_id)
                    xp = generate_perturbations(
                        X_samp[li], s=s_val, feature_sds=prep.feature_sds_,
                        perturbable_mask=prep.perturbable_mask_,
                        clip_low=prep.clip_low_, clip_high=prep.clip_high_,
                        n_draws=R, seed=p_seed, clip=True,
                    )
                    xp_all[li] = xp
                    f_infos.append(filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau))
                    if tree_splits is not None:
                        ct_all[li] = count_tree_crossings(X_samp[li], xp, tree_splits, per_tree=True)
                    cb_all[li] = count_bin_crossings(X_samp[li], xp, bin_edges)
                x_primes_map[s_val] = xp_all
                filter_info_map[s_val] = f_infos
                c_tree_map[s_val] = ct_all
                c_bin_map[s_val] = cb_all

            # --- 1) Exact SHAP explainer (LinearSHAP for LR, TreeSHAP for RF/XGB) ---
            shap_name = "LinearSHAP" if canon_m == "logit" else "TreeSHAP"
            if shap_name == "LinearSHAP":
                phi_shap_base, _ = explain_linear_shap(est, bg, X_samp)
                phi_shap_floor = np.repeat(phi_shap_base[:, None, :], k_floor, axis=1)
                phi_shap_pert = {
                    s_val: explain_linear_shap(est, bg, x_primes_map[s_val].reshape(-1, X_samp.shape[1]))[0].reshape(n_samp, R, -1)
                    for s_val in scales
                }
            else:
                phi_shap_base, _ = explain_tree_shap(est, bg, X_samp)
                phi_shap_floor = np.repeat(phi_shap_base[:, None, :], k_floor, axis=1)
                phi_shap_pert = {
                    s_val: explain_tree_shap(est, bg, x_primes_map[s_val].reshape(-1, X_samp.shape[1]))[0].reshape(n_samp, R, -1)
                    for s_val in scales
                }

            # --- 2) LIME explainer (discretize=True) parallelized over instances ---
            eff_workers = min(max(1, n_workers), n_samp)
            splits = np.array_split(np.arange(n_samp), eff_workers)
            lime_base_seed = seed + ds_idx * 50000 + m_idx * 5000
            worker_est = _to_worker_est(est)
            lime_chunks = parallel(
                delayed(_lime_worker_chunk)(
                    X_tr, feature_names, worker_est,
                    X_samp[idx], samp_idx[idx],
                    {s_val: x_primes_map[s_val][idx] for s_val in scales},
                    k_floor, lime_samples, True, lime_base_seed,
                )
                for idx in splits if len(idx) > 0
            )
            lime_res: dict[int, dict[str, Any]] = {}
            for ch in lime_chunks:
                lime_res.update(ch)

            # Collect LIME noise-floor ICC across K repeats (using L2 norm and top-feature stability proxy: flattened or max-|phi|?
            # Use per-instance attribution norm / first-PC / mean over features: icc_one_way on each feature averaged,
            # or L1 norm across K repeats)
            lime_floor_cube = np.stack([lime_res[int(iid)]["floor_phis"] for iid in samp_idx], axis=0)  # (n_samp, K, P)
            shap_floor_cube = phi_shap_floor  # (n_samp, K, P)

            for exp_label, cube in [(shap_name, shap_floor_cube), ("LIME", lime_floor_cube)]:
                icc_records.append({
                    "dataset": ds,
                    "model": m_name,
                    "explainer": exp_label,
                    "floor_cube_shape": list(cube.shape),
                })

            # Assemble draw-level and instance-level metrics for SHAP & LIME
            for exp_label in (shap_name, "LIME"):
                for s_val in scales:
                    f_tree_unw_arr = np.array([
                        compute_f_tree(X_samp[li], tree_splits, s_val, prep.feature_sds_, prep.perturbable_mask_, weighted=False)
                        if tree_splits is not None else 0.0
                        for li in range(n_samp)
                    ])
                    f_tree_cov_arr = np.array([
                        compute_f_tree(X_samp[li], tree_splits, s_val, prep.feature_sds_, prep.perturbable_mask_, weighted=True)
                        if tree_splits is not None else 0.0
                        for li in range(n_samp)
                    ])
                    f_bin_arr = np.array([
                        compute_f_bin(X_samp[li], bin_edges, s_val, prep.feature_sds_, prep.perturbable_mask_)
                        for li in range(n_samp)
                    ])

                    for li, inst_id in enumerate(samp_idx):
                        if exp_label == shap_name:
                            phi_b = phi_shap_base[li]
                            q_i, mean_fl_dk = _noise_floor_stats(phi_shap_floor[li], k_top=top_k)
                            phi_ps = phi_shap_pert[s_val][li]
                        else:
                            lr_i = lime_res[int(inst_id)]
                            phi_b = lr_i["phi_base"]
                            q_i, mean_fl_dk = _noise_floor_stats(lr_i["floor_phis"], k_top=top_k)
                            phi_ps = lr_i["pert_phis"][s_val]

                        f_info = filter_info_map[s_val][li]
                        ct_vec = c_tree_map[s_val][li]
                        cb_vec = c_bin_map[s_val][li]

                        d_k_list, d_rho_list, d_1_list, rc_list = [], [], [], []
                        for r_i in range(R):
                            phi_p = phi_ps[r_i]
                            dk = jaccard_distance_topk(phi_b, phi_p, k=top_k)
                            dr = spearman_distance(phi_b, phi_p)
                            d1 = normalized_l1_distance(phi_b, phi_p)
                            rc = int(dk > q_i + 1e-12)
                            d_k_list.append(dk)
                            d_rho_list.append(dr)
                            d_1_list.append(d1)
                            rc_list.append(rc)

                            draw_rows.append({
                                "dataset": ds,
                                "model": m_name,
                                "explainer": exp_label,
                                "s": s_val,
                                "instance_id": int(inst_id),
                                "draw_id": int(r_i),
                                "pop_weight": float(pop_weights[li]),
                                "p_x": float(p_samp[li]),
                                "y_true": int(y_samp[li]),
                                "y_hat": int(yhat_samp[li]),
                                "p_prime": float(f_info["p_primes"][r_i]),
                                "abs_dp": float(f_info["abs_dp"][r_i]),
                                "preserved": bool(f_info["preserved"][r_i]),
                                "c_tree": float(ct_vec[r_i]),
                                "c_bin": float(cb_vec[r_i]),
                                "d_k": float(dk),
                                "d_rho": float(dr),
                                "d_1": float(d1),
                                "noise_floor": float(q_i),
                                "rc": int(rc),
                            })

                        pres_mask = f_info["preserved"]
                        n_pres = int(np.sum(pres_mask))
                        dk_arr = np.asarray(d_k_list, dtype=float)
                        dr_arr = np.asarray(d_rho_list, dtype=float)
                        d1_arr = np.asarray(d_1_list, dtype=float)
                        rc_arr = np.asarray(rc_list, dtype=int)

                        if n_pres > 0:
                            mean_dk = float(np.mean(dk_arr[pres_mask]))
                            mean_dr = float(np.mean(dr_arr[pres_mask]))
                            mean_d1 = float(np.mean(d1_arr[pres_mask]))
                            u_bar = float(np.mean(rc_arr[pres_mask]))
                            mean_dp_pres = float(np.mean(f_info["abs_dp"][pres_mask]))
                            mean_ct_pres = float(np.mean(ct_vec[pres_mask]))
                            mean_cb_pres = float(np.mean(cb_vec[pres_mask]))
                        else:
                            mean_dk = np.nan
                            mean_dr = np.nan
                            mean_d1 = np.nan
                            u_bar = np.nan
                            mean_dp_pres = np.nan
                            mean_ct_pres = np.nan
                            mean_cb_pres = np.nan

                        score_f = (
                            float(f_tree_unw_arr[li]) if exp_label == "TreeSHAP"
                            else float(f_bin_arr[li]) if exp_label == "LIME"
                            else 0.0
                        )
                        score_f_cov = (
                            float(f_tree_cov_arr[li]) if exp_label == "TreeSHAP"
                            else float(f_bin_arr[li]) if exp_label == "LIME"
                            else 0.0
                        )

                        inst_rows.append({
                            "dataset": ds,
                            "model": m_name,
                            "explainer": exp_label,
                            "s": s_val,
                            "instance_id": int(inst_id),
                            "p_x": float(p_samp[li]),
                            "y_true": int(y_samp[li]),
                            "y_hat": int(yhat_samp[li]),
                            "pop_weight": float(pop_weights[li]),
                            "noise_floor": float(q_i),
                            "noise_floor_mean": float(mean_fl_dk),
                            "d_k": mean_dk,
                            "d_k_all": float(np.mean(dk_arr)),
                            "d_rho": mean_dr,
                            "d_1": mean_d1,
                            "u_bar": u_bar,
                            "u_bar_all": float(np.mean(rc_arr)),
                            "y_instab": int(u_bar >= y_cutoff) if np.isfinite(u_bar) else np.nan,
                            "y_instab_010": int(u_bar >= 0.10) if np.isfinite(u_bar) else np.nan,
                            "y_instab_030": int(u_bar >= 0.30) if np.isfinite(u_bar) else np.nan,
                            "n_preserved": n_pres,
                            "pass_rate": float(f_info["pass_rate"]),
                            "mean_abs_dp": mean_dp_pres,
                            "mean_abs_dp_all": float(np.mean(f_info["abs_dp"])),
                            "mean_c_tree": mean_ct_pres,
                            "mean_c_bin": mean_cb_pres,
                            "mean_c_tree_all": float(np.mean(ct_vec)),
                            "mean_c_bin_all": float(np.mean(cb_vec)),
                            "f_tree_unw": float(f_tree_unw_arr[li]),
                            "f_tree_cov": float(f_tree_cov_arr[li]),
                            "f_bin": float(f_bin_arr[li]),
                            "score_f": score_f,
                            "score_f_cov": score_f_cov,
                            "score_margin": float(margins[li]),
                            "score_treevar": float(tree_vars[li]) if np.isfinite(tree_vars[li]) else np.nan,
                        })

            # --- 3) H4c Ablation: LIME discretisation OFF at primary_s ---
            lime_off_chunks = parallel(
                delayed(_lime_worker_chunk)(
                    X_tr, feature_names, worker_est,
                    X_samp[idx], samp_idx[idx],
                    {primary_s: x_primes_map[primary_s][idx]},
                    k_floor, lime_samples, False, lime_base_seed + 777,
                )
                for idx in splits if len(idx) > 0
            )
            lime_off_res: dict[int, dict[str, Any]] = {}
            for ch in lime_off_chunks:
                lime_off_res.update(ch)

            for li, inst_id in enumerate(samp_idx):
                f_info = filter_info_map[primary_s][li]
                pres_mask = f_info["preserved"]
                if not np.any(pres_mask):
                    continue
                lr_on = lime_res[int(inst_id)]
                q_on, _ = _noise_floor_stats(lr_on["floor_phis"], k_top=top_k)
                rc_on = [int(jaccard_distance_topk(lr_on["phi_base"], lr_on["pert_phis"][primary_s][r_i], k=top_k) > q_on + 1e-12) for r_i in range(R)]
                u_on = float(np.mean(np.asarray(rc_on)[pres_mask]))

                lr_off = lime_off_res[int(inst_id)]
                q_off, _ = _noise_floor_stats(lr_off["floor_phis"], k_top=top_k)
                rc_off = [int(jaccard_distance_topk(lr_off["phi_base"], lr_off["pert_phis"][primary_s][r_i], k=top_k) > q_off + 1e-12) for r_i in range(R)]
                u_off = float(np.mean(np.asarray(rc_off)[pres_mask]))
                mean_cb_pres = float(np.mean(c_bin_map[primary_s][li][pres_mask]))
                h4c_paired_rows.append({
                    "dataset": ds,
                    "model": m_name,
                    "instance_id": int(inst_id),
                    "mean_c_bin": mean_cb_pres,
                    "u_bar_on": u_on,
                    "u_bar_off": u_off,
                    "q_on": q_on,
                    "q_off": q_off,
                })

            # --- 4) R6 Bridge: PermutationExplainer on XGB at primary_s ---
            if canon_m == "boost":
                perm_chunks = parallel(
                    delayed(_perm_worker_chunk)(
                        est,
                        bg,
                        X_samp[idx],
                        samp_idx[idx],
                        x_primes_map[primary_s][idx],
                        [filter_info_map[primary_s][int(li)]["preserved"] for li in idx],
                        perm_max_evals,
                        k_floor,
                        top_k,
                        ds,
                        m_name,
                        primary_s,
                        seed,
                    )
                    for idx in splits if len(idx) > 0
                )
                for ch in perm_chunks:
                    r6_bridge_rows.extend(ch)

            # --- 5) Case Study Selection on HMEQ + XGB at primary_s ---
            if ds == "hmeq" and canon_m == "boost":
                # Identify rejected test applicant (y_hat == 1) with highest F_tree whose decision
                # is preserved across ALL R draws (y_prime == 1 for all R) and RC flips >= 20%
                test_df_reset = test_df.reset_index(drop=True)
                f_tree_all = np.array([
                    compute_f_tree(X_samp[li], tree_splits, primary_s, prep.feature_sds_, prep.perturbable_mask_, weighted=False)
                    for li in range(n_samp)
                ])
                f_bin_all = np.array([
                    compute_f_bin(X_samp[li], bin_edges, primary_s, prep.feature_sds_, prep.perturbable_mask_)
                    for li in range(n_samp)
                ])
                candidates = []
                for li, inst_id in enumerate(samp_idx):
                    if yhat_samp[li] != 1:
                        continue
                    f_info = filter_info_map[primary_s][li]
                    all_dec_preserved = bool(np.all(f_info["y_primes"] == 1))
                    all_tau_preserved = bool(np.all(f_info["preserved"]))
                    phi_b_tree = phi_shap_base[li]
                    phi_ps_tree = phi_shap_pert[primary_s][li]
                    flips_tree = [int(jaccard_distance_topk(phi_b_tree, phi_ps_tree[r_i], k=top_k) > 1e-12) for r_i in range(R)]
                    flip_rate_tree = float(np.mean(flips_tree))

                    lr_on = lime_res[int(inst_id)]
                    q_lime, _ = _noise_floor_stats(lr_on["floor_phis"], k_top=top_k)
                    flips_lime = [
                        int(jaccard_distance_topk(lr_on["phi_base"], lr_on["pert_phis"][primary_s][r_i], k=top_k) > q_lime + 1e-12)
                        for r_i in range(R)
                    ]
                    flip_rate_lime = float(np.mean(flips_lime))
                    candidates.append({
                        "li": li,
                        "instance_id": int(inst_id),
                        "p_x": float(p_samp[li]),
                        "y_true": int(y_samp[li]),
                        "f_tree": float(f_tree_all[li]),
                        "f_bin": float(f_bin_all[li]),
                        "all_dec_preserved": all_dec_preserved,
                        "all_tau_preserved": all_tau_preserved,
                        "flip_rate_tree": flip_rate_tree,
                        "flip_rate_lime": flip_rate_lime,
                    })

                cand_df = pd.DataFrame(candidates)
                # Prefer all_tau_preserved, fallback to all_dec_preserved
                strict_high = cand_df[cand_df["all_tau_preserved"] & (cand_df["flip_rate_tree"] >= 0.20)].sort_values("f_tree", ascending=False)
                if strict_high.empty:
                    strict_high = cand_df[cand_df["all_dec_preserved"] & (cand_df["flip_rate_tree"] >= 0.20)].sort_values("f_tree", ascending=False)
                strict_low = cand_df[cand_df["all_tau_preserved"] & (cand_df["flip_rate_tree"] == 0.0)].sort_values("f_tree", ascending=True)
                if strict_low.empty:
                    strict_low = cand_df[cand_df["all_dec_preserved"]].sort_values(["flip_rate_tree", "f_tree"], ascending=[True, True])

                def _describe_applicant(row_info: pd.Series, role_label: str) -> dict[str, Any]:
                    li = int(row_info["li"])
                    iid = int(row_info["instance_id"])
                    raw_row = test_df_reset.iloc[iid].to_dict()
                    phi_b = phi_shap_base[li]
                    top4_idx = top_k_indices(phi_b, k=top_k)
                    top4_feats = [{"feature": feature_names[j], "phi": float(phi_b[j])} for j in top4_idx]

                    # Collect distinct top-4 sets across perturbed draws
                    phi_ps = phi_shap_pert[primary_s][li]
                    alt_sets = []
                    for r_i in range(R):
                        t4 = [feature_names[j] for j in top_k_indices(phi_ps[r_i], k=top_k)]
                        alt_sets.append(t4)
                    # Find most common flipped top-4 set
                    base_set = set(f["feature"] for f in top4_feats)
                    flipped_example = next((s_list for s_list in alt_sets if set(s_list) != base_set), alt_sets[0])

                    lr_on = lime_res[iid]
                    lime_top4_idx = top_k_indices(lr_on["phi_base"], k=top_k)
                    lime_top4 = [{"feature": feature_names[j], "phi": float(lr_on["phi_base"][j])} for j in lime_top4_idx]

                    return {
                        "role": role_label,
                        "instance_id": iid,
                        "raw_profile": {k: (None if pd.isna(v) else v) for k, v in raw_row.items()},
                        "p_default": float(row_info["p_x"]),
                        "threshold_t": float(threshold),
                        "decision": "REJECT (predicted default)",
                        "f_tree": float(row_info["f_tree"]),
                        "f_bin": float(row_info["f_bin"]),
                        "all_dec_preserved": bool(row_info["all_dec_preserved"]),
                        "all_tau_preserved": bool(row_info["all_tau_preserved"]),
                        "reason_flip_rate_treeshap": float(row_info["flip_rate_tree"]),
                        "reason_flip_rate_lime": float(row_info["flip_rate_lime"]),
                        "treeshap_top4": top4_feats,
                        "treeshap_flipped_example": flipped_example,
                        "lime_top4": lime_top4,
                        "mean_c_tree": float(np.mean(c_tree_map[primary_s][li])),
                        "mean_abs_dp": float(np.mean(filter_info_map[primary_s][li]["abs_dp"])),
                    }

                if not strict_high.empty and not strict_low.empty:
                    high_app = _describe_applicant(strict_high.iloc[0], "High-F Unstable Rejected Applicant")
                    low_app = _describe_applicant(strict_low.iloc[0], "Low-F Stable Contrast Rejected Applicant")
                    case_study_payload = {
                        "dataset": "hmeq",
                        "model": "xgb",
                        "s": primary_s,
                        "tau": tau,
                        "threshold_t": float(threshold),
                        "high_fragility_applicant": high_app,
                        "low_fragility_contrast": low_app,
                        "n_rejected_candidates": int(len(cand_df)),
                    }

            print(
                f"  [Part 2] {ds} {m_name} finished in {time.time() - t_m:.1f}s "
                f"(n_samp={n_samp}, R={R})",
                flush=True,
            )

        # --- 6) Exploratory H6: IDCS models (cslogit, csboost) on Part 2 split ---
        for m_name in idcs_models:
            est = fitted_models[m_name]
            canon_m = canonical_model_name(m_name)
            p_te = est.predict_proba(X_te)[:, 1]
            y_hat_te = (p_te >= threshold).astype(int)
            samp_idx, pop_weights = stratified_sample_indices(y_hat_te, sample_n=sample_n, seed=seed + ds_idx * 10 + 99)
            X_samp = X_te[samp_idx]
            n_samp = len(samp_idx)
            is_tree = canon_m == "csboost"
            ts = extract_tree_splits(est, feature_names=feature_names) if is_tree else None

            if is_tree:
                phi_b_all, _ = explain_tree_shap(est, bg, X_samp)
            else:
                phi_b_all, _ = explain_linear_shap(est, bg, X_samp)

            for li, inst_id in enumerate(samp_idx):
                p_seed = seed + 44444 + ds_idx * 10000 + int(inst_id)
                xp = generate_perturbations(
                    X_samp[li], s=primary_s, feature_sds=prep.feature_sds_,
                    perturbable_mask=prep.perturbable_mask_,
                    clip_low=prep.clip_low_, clip_high=prep.clip_high_,
                    n_draws=R, seed=p_seed, clip=True,
                )
                f_info = filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau)
                pres = f_info["preserved"]
                if not np.any(pres):
                    continue
                phi_ps = (
                    explain_tree_shap(est, bg, xp)[0] if is_tree else explain_linear_shap(est, bg, xp)[0]
                )
                dks = np.array([jaccard_distance_topk(phi_b_all[li], phi_ps[r_i], k=top_k) for r_i in range(R)])
                rcs = (dks > 1e-12).astype(int)
                ct_vec = count_tree_crossings(X_samp[li], xp, ts, per_tree=True) if ts is not None else np.zeros(R)
                f_t = compute_f_tree(X_samp[li], ts, primary_s, prep.feature_sds_, prep.perturbable_mask_, weighted=False) if ts is not None else 0.0
                f_b = compute_f_bin(X_samp[li], bin_edges, primary_s, prep.feature_sds_, prep.perturbable_mask_)
                h6_idcs_rows.append({
                    "dataset": ds,
                    "model": m_name,
                    "explainer": "TreeSHAP" if is_tree else "LinearSHAP",
                    "is_idcs": 1,
                    "s": primary_s,
                    "instance_id": int(inst_id),
                    "u_bar": float(np.mean(rcs[pres])),
                    "d_k": float(np.mean(dks[pres])),
                    "mean_abs_dp": float(np.mean(f_info["abs_dp"][pres])),
                    "mean_c_tree": float(np.mean(ct_vec[pres])),
                    "f_tree_unw": float(f_t),
                    "f_bin": float(f_b),
                    "score_f": float(f_t if is_tree else 0.0),
                })
    finally:
        parallel.__exit__(None, None, None)

    return {
        "perf_df": pd.DataFrame(perf_rows),
        "draws_df": pd.DataFrame(draw_rows),
        "inst_all_s_df": pd.DataFrame(inst_rows),
        "h4c_paired_df": pd.DataFrame(h4c_paired_rows),
        "r6_bridge_df": pd.DataFrame(r6_bridge_rows),
        "h6_idcs_df": pd.DataFrame(h6_idcs_rows),
        "case_study": case_study_payload,
    }


def compute_part2_statistics(
    draws_df: pd.DataFrame,
    inst_all_s_df: pd.DataFrame,
    h4c_paired_df: pd.DataFrame,
    h6_idcs_df: pd.DataFrame,
    cfg: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """Run all Part 2 confirmatory & exploratory hypotheses (H2-H6) and Measurement Quality checks."""
    seed = int(cfg.get("seed", 42))
    alpha = float(cfg["stats"].get("alpha", 0.05))
    n_boot = int(cfg["stats"].get("bootstrap_resamples", 5000))
    scales = sorted(inst_all_s_df["s"].unique())
    primary_s = max(scales) if 0.05 not in scales else 0.05

    # --- H2: Perturbed Delta E > Noise Floor ---
    h2_rows: list[dict[str, Any]] = []
    for (ds, model, exp, s_val), grp in inst_all_s_df.groupby(["dataset", "model", "explainer", "s"]):
        valid = grp.dropna(subset=["d_k", "noise_floor", "u_bar"])
        is_exact = exp in ("LinearSHAP", "TreeSHAP")
        if is_exact:
            # Exact explainer: share of instances with U_bar > 0 (any reason-code flip) with Clopper-Pearson CI
            cp = proportion_nonzero_with_ci(valid["u_bar"].to_numpy(dtype=float), tol=0.0, alpha=alpha)
            pop_rate = float(np.average(valid["u_bar"] > 0.0, weights=valid["pop_weight"])) if len(valid) else np.nan
            h2_rows.append({
                "dataset": ds, "model": model, "explainer": exp, "s": float(s_val),
                "type": "exact",
                "n_instances": cp["n"],
                "k_nonzero": cp["k"],
                "share_gt_floor": cp["proportion"],
                "ci_low": cp["ci_low"],
                "ci_high": cp["ci_high"],
                "pop_weighted_share": pop_rate,
                "median_d_k": float(np.median(valid["d_k"])),
                "median_noise_floor": 0.0,
                "median_u_bar": float(np.median(valid["u_bar"])),
                "mean_u_bar": float(np.mean(valid["u_bar"])),
                "wilcoxon_p": np.nan,
                "rank_biserial_r": np.nan,
                "supported_cell": bool(cp["ci_low"] > 0.0),
            })
        else:
            # Stochastic LIME: one-sided paired Wilcoxon (d_k > noise_floor)
            wx = paired_wilcoxon_test(valid["d_k"].to_numpy(), valid["noise_floor"].to_numpy(), alternative="greater")
            cp = proportion_nonzero_with_ci(valid["u_bar"].to_numpy(), tol=0.0, alpha=alpha)
            pop_rate = float(np.average(valid["u_bar"] > 0.0, weights=valid["pop_weight"])) if len(valid) else np.nan
            h2_rows.append({
                "dataset": ds, "model": model, "explainer": exp, "s": float(s_val),
                "type": "stochastic",
                "n_instances": wx["n_pairs"],
                "k_nonzero": cp["k"],
                "share_gt_floor": cp["proportion"],
                "ci_low": cp["ci_low"],
                "ci_high": cp["ci_high"],
                "pop_weighted_share": pop_rate,
                "median_d_k": float(np.median(valid["d_k"])),
                "median_noise_floor": float(np.median(valid["noise_floor"])),
                "median_u_bar": float(np.median(valid["u_bar"])),
                "mean_u_bar": float(np.mean(valid["u_bar"])),
                "wilcoxon_p": wx["pvalue"],
                "rank_biserial_r": wx["rank_biserial_r"],
                "supported_cell": False,
            })

    h2_df = pd.DataFrame(h2_rows)
    if not h2_df.empty:
        stoch_mask = h2_df["type"] == "stochastic"
        if stoch_mask.any():
            holm = holm_bonferroni(h2_df.loc[stoch_mask, "wilcoxon_p"].tolist(), alpha=alpha)
            h2_df.loc[stoch_mask, "wilcoxon_p_holm"] = holm["p_adjusted"]
            h2_df.loc[stoch_mask, "supported_cell"] = holm["reject"]
        h2_df.to_csv(results_dir / "part2_h2_noise_floor.csv", index=False)

    # --- H3: |Delta p| AUROC for predicting RC (full vs filtered sets) ---
    h3_rows: list[dict[str, Any]] = []
    for (ds, model, exp), grp in draws_df.groupby(["dataset", "model", "explainer"]):
        for subset_name, sub_df in [("full", grp), ("tau_filtered", grp[grp["preserved"]])]:
            auc_res = cluster_bootstrap_auroc(
                sub_df["rc"].to_numpy(dtype=int),
                sub_df["abs_dp"].to_numpy(dtype=float),
                sub_df["instance_id"].to_numpy(),
                n_boot=n_boot, seed=seed, alpha=alpha,
            )
            h3_rows.append({
                "dataset": ds, "model": model, "explainer": exp, "subset": subset_name,
                "rc_rate": float(sub_df["rc"].mean()) if len(sub_df) else np.nan,
                **auc_res,
                "upper_ci_lt_070": bool(np.isfinite(auc_res["ci_high"]) and auc_res["ci_high"] < 0.70),
            })
    h3_df = pd.DataFrame(h3_rows)
    h3_df.to_csv(results_dir / "part2_h3_abs_dp_auroc.csv", index=False)

    # --- H4a: Mediation (s -> C -> Delta E) on tau-filtered draws ---
    h4a_rows: list[dict[str, Any]] = []
    filt_draws = draws_df[draws_df["preserved"]].copy()
    for (ds, model, exp), grp in filt_draws.groupby(["dataset", "model", "explainer"]):
        if exp == "LinearSHAP":
            continue
        med_col = "c_tree" if exp == "TreeSHAP" else "c_bin"
        if np.std(grp[med_col]) < 1e-12:
            continue
        med_res = cluster_bootstrap_mediation(
            treatment=grp["s"].to_numpy(dtype=float),
            mediator=grp[med_col].to_numpy(dtype=float),
            outcome=grp["d_k"].to_numpy(dtype=float),
            cluster_ids=grp["instance_id"].to_numpy(),
            n_boot=n_boot, seed=seed, alpha=alpha,
        )
        h4a_rows.append({
            "dataset": ds, "model": model, "explainer": exp, "mediator": med_col,
            **med_res,
        })
    h4a_df = pd.DataFrame(h4a_rows)
    if not h4a_df.empty:
        holm = holm_bonferroni(h4a_df["indirect_pvalue_boot"].tolist(), alpha=alpha)
        h4a_df["p_holm"] = holm["p_adjusted"]
        h4a_df["supported_holm"] = [bool(r and ex) for r, ex in zip(holm["reject"], h4a_df["ci_excludes_zero"])]
        h4a_df.to_csv(results_dir / "part2_h4a_mediation.csv", index=False)

    # --- H4b: Double Dissociation (Joint Logistic RC ~ C_tree + C_bin on tree models) ---
    h4b_rows: list[dict[str, Any]] = []
    dissoc_plot_rows: list[dict[str, Any]] = []
    tree_filt = filt_draws[filt_draws["model"].isin(["rf", "xgb"])].copy()
    for exp in ("TreeSHAP", "LIME"):
        sub_e = tree_filt[tree_filt["explainer"] == exp]
        if sub_e.empty:
            continue
        # Pooled across datasets/models (with dataset+model dummies) AND per-dataset
        dummies = pd.get_dummies(sub_e[["dataset", "model"]], drop_first=True, dtype=float)
        X_joint = pd.concat([sub_e[["c_tree", "c_bin"]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        clusters = (sub_e["dataset"].astype(str) + "_" + sub_e["model"].astype(str) + "_" + sub_e["instance_id"].astype(str)).to_numpy()
        res_pool = joint_logistic_cluster_robust(sub_e["rc"].to_numpy(dtype=int), X_joint, clusters, alpha=alpha)
        for pred in ("c_tree", "c_bin"):
            t_info = res_pool.get("terms", {}).get(pred, {})
            h4b_rows.append({
                "scope": "pooled_trees", "dataset": "all", "explainer": exp, "predictor": pred,
                "converged": res_pool.get("converged", False),
                "n_obs": res_pool.get("n_obs", 0),
                "n_clusters": res_pool.get("n_clusters", 0),
                **t_info,
            })
            dissoc_plot_rows.append({
                "explainer": exp,
                "predictor": "C_tree" if pred == "c_tree" else "C_bin",
                "beta": t_info.get("coef", np.nan),
                "ci_low": t_info.get("ci_low", np.nan),
                "ci_high": t_info.get("ci_high", np.nan),
                "odds_ratio": t_info.get("odds_ratio", np.nan),
                "pvalue": t_info.get("pvalue", np.nan),
            })

        # Per dataset
        for ds, d_grp in sub_e.groupby("dataset"):
            dm = pd.get_dummies(d_grp[["model"]], drop_first=True, dtype=float)
            Xj = pd.concat([d_grp[["c_tree", "c_bin"]].reset_index(drop=True), dm.reset_index(drop=True)], axis=1)
            cl = (d_grp["model"].astype(str) + "_" + d_grp["instance_id"].astype(str)).to_numpy()
            rj = joint_logistic_cluster_robust(d_grp["rc"].to_numpy(dtype=int), Xj, cl, alpha=alpha)
            for pred in ("c_tree", "c_bin"):
                t_info = rj.get("terms", {}).get(pred, {})
                h4b_rows.append({
                    "scope": "per_dataset", "dataset": ds, "explainer": exp, "predictor": pred,
                    "converged": rj.get("converged", False),
                    "n_obs": rj.get("n_obs", 0),
                    "n_clusters": rj.get("n_clusters", 0),
                    **t_info,
                })

    h4b_df = pd.DataFrame(h4b_rows)
    if not h4b_df.empty:
        # Apply Holm across the 4 pooled target paths (TreeSHAP/LIME x C_tree/C_bin)
        pool_mask = h4b_df["scope"] == "pooled_trees"
        holm = holm_bonferroni(h4b_df.loc[pool_mask, "pvalue"].tolist(), alpha=alpha)
        h4b_df.loc[pool_mask, "p_holm"] = holm["p_adjusted"]
        h4b_df.to_csv(results_dir / "part2_h4b_double_dissociation.csv", index=False)

    dissoc_df = pd.DataFrame(dissoc_plot_rows)
    dissoc_df.to_parquet(results_dir / "h4b_double_dissociation.parquet", index=False)
    dissoc_df.to_csv(results_dir / "h4b_double_dissociation.csv", index=False)

    # --- H4c: LIME discretisation ON vs OFF (conditioned on C_bin > 0) ---
    h4c_rows: list[dict[str, Any]] = []
    h4c_plot_rows: list[dict[str, Any]] = []
    if not h4c_paired_df.empty:
        cond_df = h4c_paired_df[h4c_paired_df["mean_c_bin"] > 0.0].copy()
        for ds, grp in cond_df.groupby("dataset"):
            wx = paired_wilcoxon_test(grp["u_bar_on"].to_numpy(), grp["u_bar_off"].to_numpy(), alternative="greater")
            h4c_rows.append({
                "dataset": ds,
                "n_instances": wx["n_pairs"],
                "mean_u_on": float(grp["u_bar_on"].mean()),
                "mean_u_off": float(grp["u_bar_off"].mean()),
                "median_u_on": float(grp["u_bar_on"].median()),
                "median_u_off": float(grp["u_bar_off"].median()),
                "wilcoxon_p": wx["pvalue"],
                "rank_biserial_r": wx["rank_biserial_r"],
            })
            for state, col in [("On", "u_bar_on"), ("Off", "u_bar_off")]:
                vals = grp[col].to_numpy(dtype=float)
                m_v = float(np.mean(vals))
                se_v = float(np.std(vals, ddof=1) / np.sqrt(max(1, len(vals))))
                h4c_plot_rows.append({
                    "dataset": ds,
                    "discretisation": state,
                    "instability_rate": m_v,
                    "ci_low": max(0.0, m_v - 1.96 * se_v),
                    "ci_high": min(1.0, m_v + 1.96 * se_v),
                })
    h4c_df = pd.DataFrame(h4c_rows)
    h4c_df.to_csv(results_dir / "part2_h4c_discretisation.csv", index=False)
    h4c_plot_df = pd.DataFrame(h4c_plot_rows)
    h4c_plot_df.to_parquet(results_dir / "h4c_discretisation.parquet", index=False)
    h4c_plot_df.to_csv(results_dir / "h4c_discretisation.csv", index=False)

    # --- H5: F(x) vs Baselines (DeLong + paired bootstrap + Spearman) at primary_s ---
    inst_primary = inst_all_s_df[inst_all_s_df["s"] == primary_s].copy()
    h5_rows: list[dict[str, Any]] = []
    for (ds, model, exp), grp in inst_primary.groupby(["dataset", "model", "explainer"]):
        if exp == "LinearSHAP":
            continue  # Linear models have no tree splits or bin edges
        valid = grp.dropna(subset=["y_instab", "score_f", "score_margin", "u_bar"])
        if valid["y_instab"].nunique() < 2:
            continue
        yt = valid["y_instab"].to_numpy(dtype=int)
        sf = valid["score_f"].to_numpy(dtype=float)
        sp_f = spearman_with_ci(sf, valid["u_bar"].to_numpy(dtype=float), alpha=alpha)

        for b_name, b_col in [("B_margin", "score_margin"), ("B_var", "score_treevar"), ("posthoc_abs_dp", "mean_abs_dp")]:
            sub_b = valid.dropna(subset=[b_col])
            if sub_b.empty or sub_b["y_instab"].nunique() < 2:
                continue
            yt_b = sub_b["y_instab"].to_numpy(dtype=int)
            sf_b = sub_b["score_f"].to_numpy(dtype=float)
            sb_b = sub_b[b_col].to_numpy(dtype=float)
            dl = delong_paired_auroc_test(yt_b, sf_b, sb_b, alternative="greater", alpha=alpha)
            pb = paired_bootstrap_auroc_diff(yt_b, sf_b, sb_b, n_boot=n_boot, seed=seed, alternative="greater", alpha=alpha)
            h5_rows.append({
                "dataset": ds, "model": model, "explainer": exp,
                "baseline": b_name,
                "is_perturbation_free": b_name in ("B_margin", "B_var"),
                "n_instances": int(len(sub_b)),
                "pos_rate_y_instab": float(np.mean(yt_b)),
                "auroc_f": dl["auc_1"],
                "auroc_baseline": dl["auc_2"],
                "delta_auroc": dl["diff"],
                "delong_se": dl["se_diff"],
                "delong_z": dl["z"],
                "delong_p": dl["pvalue"],
                "delong_ci_low": dl["ci_low"],
                "delong_ci_high": dl["ci_high"],
                "boot_p": pb["pvalue"],
                "boot_ci_low": pb["ci_low"],
                "boot_ci_high": pb["ci_high"],
                "spearman_f_ubar": sp_f["rho"],
                "spearman_f_ubar_p": sp_f["pvalue"],
            })

    # Also dataset-level pooled H5 across tree models / LIME (matching Fig 5)
    for ds, grp in inst_primary[inst_primary["explainer"] != "LinearSHAP"].groupby("dataset"):
        valid = grp.dropna(subset=["y_instab", "score_f", "score_margin", "u_bar"])
        if valid["y_instab"].nunique() < 2:
            continue
        yt = valid["y_instab"].to_numpy(dtype=int)
        sf = valid["score_f"].to_numpy(dtype=float)
        sp_f = spearman_with_ci(sf, valid["u_bar"].to_numpy(dtype=float), alpha=alpha)
        for b_name, b_col in [("B_margin", "score_margin"), ("B_var", "score_treevar"), ("posthoc_abs_dp", "mean_abs_dp")]:
            sub_b = valid.dropna(subset=[b_col])
            if sub_b.empty or sub_b["y_instab"].nunique() < 2:
                continue
            dl = delong_paired_auroc_test(sub_b["y_instab"].to_numpy(dtype=int), sub_b["score_f"].to_numpy(), sub_b[b_col].to_numpy(), alternative="greater", alpha=alpha)
            pb = paired_bootstrap_auroc_diff(sub_b["y_instab"].to_numpy(dtype=int), sub_b["score_f"].to_numpy(), sub_b[b_col].to_numpy(), n_boot=n_boot, seed=seed, alternative="greater", alpha=alpha)
            h5_rows.append({
                "dataset": ds, "model": "POOLED", "explainer": "TreeSHAP+LIME",
                "baseline": b_name,
                "is_perturbation_free": b_name in ("B_margin", "B_var"),
                "n_instances": int(len(sub_b)),
                "pos_rate_y_instab": float(sub_b["y_instab"].mean()),
                "auroc_f": dl["auc_1"],
                "auroc_baseline": dl["auc_2"],
                "delta_auroc": dl["diff"],
                "delong_se": dl["se_diff"],
                "delong_z": dl["z"],
                "delong_p": dl["pvalue"],
                "delong_ci_low": dl["ci_low"],
                "delong_ci_high": dl["ci_high"],
                "boot_p": pb["pvalue"],
                "boot_ci_low": pb["ci_low"],
                "boot_ci_high": pb["ci_high"],
                "spearman_f_ubar": sp_f["rho"],
                "spearman_f_ubar_p": sp_f["pvalue"],
            })

    h5_df = pd.DataFrame(h5_rows)
    if not h5_df.empty:
        h5_df["delong_p_holm"] = np.nan
        h5_df["supported_holm"] = False
        pf_mask = h5_df["is_perturbation_free"] & (h5_df["model"] != "POOLED")
        if pf_mask.any():
            holm = holm_bonferroni(h5_df.loc[pf_mask, "delong_p"].tolist(), alpha=alpha)
            h5_df.loc[pf_mask, "delong_p_holm"] = holm["p_adjusted"]
            h5_df.loc[pf_mask, "supported_holm"] = np.asarray(holm["reject"], dtype=bool)
        h5_df.to_csv(results_dir / "part2_h5_fragility_auroc.csv", index=False)

    # --- H6: Exploratory Bridge (Nested OLS M0->M3 + IDCS vs Traditional mean F(x)) ---
    h6_results: dict[str, Any] = {}
    comb_h6 = inst_primary.copy()
    comb_h6["is_idcs"] = 0
    if not h6_idcs_df.empty:
        comb_h6 = pd.concat([comb_h6, h6_idcs_df], ignore_index=True)

    comb_h6 = comb_h6.dropna(subset=["u_bar", "mean_abs_dp", "mean_c_tree", "mean_c_bin", "score_f"]).reset_index(drop=True)
    if len(comb_h6) >= 20:
        d_ds = pd.get_dummies(comb_h6["dataset"], prefix="ds", drop_first=True, dtype=float)
        d_m = pd.get_dummies(comb_h6["model"], prefix="m", drop_first=True, dtype=float)
        d_e = pd.get_dummies(comb_h6["explainer"], prefix="e", drop_first=True, dtype=float)
        m0_df = pd.concat([d_ds, d_m, d_e], axis=1)
        m1_df = pd.concat([m0_df, comb_h6[["mean_abs_dp"]]], axis=1)
        m2_df = pd.concat([m1_df, comb_h6[["mean_c_tree", "mean_c_bin"]]], axis=1)
        m3_df = pd.concat([m2_df, comb_h6[["score_f"]]], axis=1)
        h6_results["nested_ols"] = nested_ols_hc3(
            comb_h6["u_bar"].to_numpy(dtype=float),
            [("M0: Base Factors", m0_df), ("M1: +|Delta p|", m1_df), ("M2: +Crossings", m2_df), ("M3: +F(x)", m3_df)],
            n_boot=n_boot, seed=seed, alpha=alpha,
        )

    # IDCS vs Traditional F_tree comparison for boost vs csboost
    if not h6_idcs_df.empty:
        idcs_cmp_rows = []
        for ds in h6_idcs_df["dataset"].unique():
            trad_xgb = inst_primary[(inst_primary["dataset"] == ds) & (inst_primary["model"] == "xgb") & (inst_primary["explainer"] == "TreeSHAP")]
            idcs_csb = h6_idcs_df[(h6_idcs_df["dataset"] == ds) & (h6_idcs_df["model"] == "csboost")]
            if not trad_xgb.empty and not idcs_csb.empty:
                from scipy.stats import mannwhitneyu
                u_stat, p_u = mannwhitneyu(idcs_csb["f_tree_unw"], trad_xgb["f_tree_unw"], alternative="greater")
                idcs_cmp_rows.append({
                    "dataset": ds,
                    "mean_f_csboost": float(idcs_csb["f_tree_unw"].mean()),
                    "mean_f_xgb": float(trad_xgb["f_tree_unw"].mean()),
                    "mean_u_csboost": float(idcs_csb["u_bar"].mean()),
                    "mean_u_xgb": float(trad_xgb["u_bar"].mean()),
                    "mannwhitney_u": float(u_stat),
                    "pvalue_greater": float(p_u),
                })
        h6_results["idcs_vs_trad_f"] = idcs_cmp_rows

    with (results_dir / "part2_h6_exploratory.json").open("w", encoding="utf-8") as fh:
        json.dump(h6_results, fh, indent=2)

    # --- Measurement Quality Analogues (§6.6) ---
    mq_rows: list[dict[str, Any]] = []
    for (ds, model, exp), grp in filt_draws.groupby(["dataset", "model", "explainer"]):
        if len(grp) < 10:
            continue
        mq = measurement_quality_suite(
            noise_floor_matrix=None,
            d_k=grp["d_k"].to_numpy(dtype=float),
            d_rho=grp["d_rho"].to_numpy(dtype=float),
            d_1=grp["d_1"].to_numpy(dtype=float),
            abs_delta_p=grp["abs_dp"].to_numpy(dtype=float),
            cluster_ids=grp["instance_id"].to_numpy(),
            n_boot=min(2000, n_boot), seed=seed, alpha=alpha,
        )
        # Also get noise-floor rate & exact vs stochastic test-retest summary from inst_primary
        sub_inst = inst_primary[(inst_primary["dataset"] == ds) & (inst_primary["model"] == model) & (inst_primary["explainer"] == exp)]
        mq_rows.append({
            "dataset": ds,
            "model": model,
            "explainer": exp,
            "is_deterministic": exp in ("LinearSHAP", "TreeSHAP"),
            "mean_noise_floor_q95": float(sub_inst["noise_floor"].mean()) if not sub_inst.empty else 0.0,
            "icc_1_analog": 1.0 if exp in ("LinearSHAP", "TreeSHAP") else float(max(0.0, 1.0 - sub_inst["noise_floor_mean"].mean())) if not sub_inst.empty else np.nan,
            "rho_dk_drho": mq["monotrait_correlations"]["D_k__D_rho"],
            "rho_dk_d1": mq["monotrait_correlations"]["D_k__D_1"],
            "rho_drho_d1": mq["monotrait_correlations"]["D_rho__D_1"],
            "mean_abs_monotrait": mq["mean_abs_monotrait"],
            "mean_abs_heterotrait": mq["mean_abs_heterotrait"],
            "htmt_ratio": mq["htmt_ratio"],
            "htmt_ci_low": mq["htmt_ci_low"],
            "htmt_ci_high": mq["htmt_ci_high"],
            "discriminant_passes_085": mq["discriminant_passes_085"],
        })
    mq_df = pd.DataFrame(mq_rows)
    mq_df.to_csv(results_dir / "part2_measurement_quality.csv", index=False)

    return {
        "h2_cells": int(len(h2_df)),
        "h2_supported": int(h2_df["supported_cell"].sum()) if not h2_df.empty else 0,
        "h3_tree_full_lt_070": int(h3_df[(h3_df["subset"] == "full") & (h3_df["model"].isin(["rf", "xgb"]))]["upper_ci_lt_070"].sum()) if not h3_df.empty else 0,
        "h3_tree_full_total": int(len(h3_df[(h3_df["subset"] == "full") & (h3_df["model"].isin(["rf", "xgb"]))])) if not h3_df.empty else 0,
        "h4a_supported": int(h4a_df["supported_holm"].sum()) if not h4a_df.empty else 0,
        "h4a_total": int(len(h4a_df)),
        "h5_pf_supported": int(h5_df[h5_df["is_perturbation_free"] & (h5_df["model"] != "POOLED")]["supported_holm"].sum()) if not h5_df.empty else 0,
        "h5_pf_total": int(len(h5_df[h5_df["is_perturbation_free"] & (h5_df["model"] != "POOLED")])) if not h5_df.empty else 0,
        "mq_htmt_pass": int(mq_df["discriminant_passes_085"].sum()) if not mq_df.empty else 0,
        "mq_total": int(len(mq_df)),
    }


def main(config_path: Path, stats_only: bool = False) -> None:
    t0 = time.time()
    cfg = load_config(config_path)
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(cfg.get("n_jobs", -1))
    n_workers = os.cpu_count() or 4 if n_jobs <= 0 else n_jobs

    if stats_only:
        draws_df = pd.read_parquet(results_dir / "part2_draws.parquet")
        inst_all_s_df = pd.read_parquet(results_dir / "part2_instance_by_scale.parquet")
        h4c_paired_df = pd.read_parquet(results_dir / "part2_h4c_paired.parquet")
        h6_p = results_dir / "part2_h6_idcs.parquet"
        h6_idcs_df = pd.read_parquet(h6_p) if h6_p.exists() else pd.DataFrame()
    else:
        print("=== Stage 03: Running Part 2 Main Experiment & Ablations ===")
        out = run_part2_experiments(cfg, n_workers=n_workers)
        out["perf_df"].to_csv(results_dir / "part2_model_performance.csv", index=False)
        out["draws_df"].to_parquet(results_dir / "part2_draws.parquet", index=False)
        out["inst_all_s_df"].to_parquet(results_dir / "part2_instance_by_scale.parquet", index=False)
        out["inst_all_s_df"].to_csv(results_dir / "part2_instance_by_scale.csv", index=False)

        scales = sorted(out["inst_all_s_df"]["s"].unique())
        primary_s = max(scales) if 0.05 not in scales else 0.05
        inst_primary = out["inst_all_s_df"][out["inst_all_s_df"]["s"] == primary_s].copy()
        inst_primary.to_parquet(results_dir / "part2_instance_summary.parquet", index=False)
        inst_primary.to_csv(results_dir / "part2_instance_summary.csv", index=False)

        out["h4c_paired_df"].to_parquet(results_dir / "part2_h4c_paired.parquet", index=False)
        out["r6_bridge_df"].to_parquet(results_dir / "part2_r6_perm_bridge.parquet", index=False)
        out["r6_bridge_df"].to_csv(results_dir / "part2_r6_perm_bridge.csv", index=False)
        out["h6_idcs_df"].to_parquet(results_dir / "part2_h6_idcs.parquet", index=False)
        out["h6_idcs_df"].to_csv(results_dir / "part2_h6_idcs.csv", index=False)

        with (results_dir / "case_study_hmeq.json").open("w", encoding="utf-8") as fh:
            json.dump(out["case_study"], fh, indent=2, default=str)

        draws_df = out["draws_df"]
        inst_all_s_df = out["inst_all_s_df"]
        h4c_paired_df = out["h4c_paired_df"]
        h6_idcs_df = out["h6_idcs_df"]

    print("=== Stage 03: Computing H2-H6 & Measurement Quality Statistics ===")
    summary = compute_part2_statistics(draws_df, inst_all_s_df, h4c_paired_df, h6_idcs_df, cfg, results_dir)
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    with (results_dir / "part2_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"=== Stage 03 Complete in {time.time() - t0:.1f}s ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 03: Part 2 Extension & Statistical Suite.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml")
    parser.add_argument("--stats-only", action="store_true", help="Recompute stats from cached parquet files.")
    args = parser.parse_args()
    main(args.config, stats_only=args.stats_only)
