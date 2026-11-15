"""Stage 02: Base-paper replication (Ballegeer, Bogaert & Benoit, 2025) + E1 variance decomposition.

Replicates Algorithm 2 across default rates pi in {1, 3, 5, 10, 15, 20, 25, 30}% on
HMEQ and SGCS for logit, cslogit, boost, csboost, and forest using SHAP
PermutationExplainer and LIME, then runs H1a, H1b, and E1 statistical tests.

Usage:
    python scripts/02_replicate.py --config config.yaml
    python scripts/02_replicate.py --config config.yaml --pilot
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.data import (  # noqa: E402
    Preprocessor,
    load_config,
    load_dataset,
    make_part1_split,
    resample_to_pi,
    resolve_path,
)
from src.explain import (  # noqa: E402
    build_lime_explainer,
    explain_lime,
    explain_permutation_shap,
    sample_background,
)
from src.idcs import compute_cost_vectors  # noqa: E402
from src.metrics import compute_cov, compute_sra  # noqa: E402
from src.models import (  # noqa: E402
    build_estimator,
    canonical_model_name,
    evaluate_model,
    fit_estimator,
    tune_hyperparameters,
)
from src.stats import (  # noqa: E402
    e1_variance_decomposition,
    holm_bonferroni,
    jonckheere_terpstra_test,
    ks_two_sample_test,
    paired_wilcoxon_test,
    spearman_with_ci,
)


def _align_attributions(
    phi: np.ndarray,
    local_features: list[str],
    canonical_features: list[str],
) -> np.ndarray:
    """Align per-refit attribution columns to canonical feature order (0.0 for absent rare levels)."""
    if local_features == canonical_features:
        return phi
    out = np.zeros((phi.shape[0], len(canonical_features)), dtype=float)
    canon_idx = {name: idx for idx, name in enumerate(canonical_features)}
    for j, fname in enumerate(local_features):
        if fname in canon_idx:
            out[:, canon_idx[fname]] = phi[:, j]
    return out


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


def _explain_chunk_perm_shap(
    est_or_params: Any,
    bg: np.ndarray,
    X_chunk: np.ndarray,
    max_evals: int,
    seed: int,
) -> np.ndarray:
    est = _unwrap_est(est_or_params)
    return explain_permutation_shap(est, bg, X_chunk, max_evals=max_evals, seed=seed)


def _explain_chunk_lime(
    X_tr: np.ndarray,
    feature_names: list[str],
    est_or_params: Any,
    X_chunk: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    est = _unwrap_est(est_or_params)
    le = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed)
    return explain_lime(le, est, X_chunk, num_samples=num_samples, seed=seed)


def _explain_all_parallel(
    parallel: Parallel,
    est: Any,
    X_tr: np.ndarray,
    bg: np.ndarray,
    X_te: np.ndarray,
    feature_names: list[str],
    explainer_name: str,
    max_evals: int,
    lime_num_samples: int,
    seed: int,
    n_workers: int,
) -> np.ndarray:
    """Explain all test instances in parallel chunks across a reused joblib pool."""
    if type(est).__name__ == "LogisticRegression":
        pred_payload: Any = (
            "__fast_logit__",
            np.asarray(est.coef_, dtype=float).ravel(),
            float(np.asarray(est.intercept_, dtype=float).ravel()[0]),
        )
    else:
        pred_payload = est

    n_te = X_te.shape[0]
    eff_workers = min(max(1, n_workers), n_te)
    if eff_workers == 1:
        pred_est = _unwrap_est(pred_payload)
        if explainer_name == "permutation_shap":
            return explain_permutation_shap(pred_est, bg, X_te, max_evals=max_evals, seed=seed)
        le = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed)
        return explain_lime(le, pred_est, X_te, num_samples=lime_num_samples, seed=seed)

    splits = np.array_split(np.arange(n_te), eff_workers)
    if explainer_name == "permutation_shap":
        chunks = parallel(
            delayed(_explain_chunk_perm_shap)(
                pred_payload, bg, X_te[idx], max_evals, seed + int(idx[0])
            )
            for idx in splits if len(idx) > 0
        )
    else:
        chunks = parallel(
            delayed(_explain_chunk_lime)(
                X_tr, feature_names, pred_payload, X_te[idx], lime_num_samples, seed + int(idx[0])
            )
            for idx in splits if len(idx) > 0
        )
    return np.vstack(chunks)


def _tune_single_model(
    m_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    c_fn_tr: np.ndarray,
    c_fp_tr: np.ndarray,
    cv_folds: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    canon_m = canonical_model_name(m_name)
    params = tune_hyperparameters(
        canon_m, X_tr, y_tr, c_fn=c_fn_tr, c_fp=c_fp_tr,
        cv_folds=cv_folds, seed=seed, n_jobs=1,
    )
    return m_name, params


def _run_part1_unit(
    j_idx: int,
    m_name: str,
    tuned_p: dict[str, Any],
    iter_bundle: dict[str, Any],
    canon_features: list[str],
    y_te: np.ndarray,
    c_fn_te: np.ndarray,
    c_fp_te: np.ndarray,
    prior_pi1: float,
    explainers: list[str],
    max_evals: int,
    lime_samples: int,
) -> tuple[int, str, dict[str, Any], dict[str, np.ndarray]]:
    """Fit, evaluate, and explain one (j_iter, model) pair on a single worker."""
    X_tr = iter_bundle["X_tr"]
    y_tr = iter_bundle["y_tr"]
    X_te = iter_bundle["X_te"]
    c_fn_tr = iter_bundle["c_fn_tr"]
    c_fp_tr = iter_bundle["c_fp_tr"]
    bg = iter_bundle["bg"]
    local_features = iter_bundle["local_features"]
    le = iter_bundle["le"]
    iter_seed = iter_bundle["iter_seed"]

    canon_m = canonical_model_name(m_name)
    est = build_estimator(canon_m, params=tuned_p, seed=iter_seed, n_jobs=1)
    fit_estimator(est, canon_m, X_tr, y_tr, c_fn=c_fn_tr, c_fp=c_fp_tr)

    p_te = est.predict_proba(X_te)[:, 1]
    ev = evaluate_model(
        y_te, p_te, c_fn=c_fn_te, c_fp=c_fp_te, prior_pi1=prior_pi1, threshold=0.5
    )

    if type(est).__name__ == "LogisticRegression":
        pred_est: Any = _FastLogitProxy(
            np.asarray(est.coef_, dtype=float).ravel(),
            float(np.asarray(est.intercept_, dtype=float).ravel()[0]),
        )
    else:
        pred_est = est

    aligned: dict[str, np.ndarray] = {}
    for exp_name in explainers:
        if exp_name == "permutation_shap":
            phi_local = explain_permutation_shap(
                pred_est, bg, X_te, max_evals=max_evals, seed=iter_seed + 77
            )
        else:
            phi_local = explain_lime(
                le, pred_est, X_te, num_samples=lime_samples, seed=iter_seed + 77
            )
        aligned[exp_name] = _align_attributions(phi_local, local_features, canon_features)

    return j_idx, m_name, ev, aligned


def run_part1_replication(
    cfg: dict[str, Any],
    datasets: list[str],
    rates: list[float],
    j_iters: int,
    models: list[str],
    explainers: list[str],
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Ballegeer et al. (2025) Algorithm 2 across datasets, rates, models, and explainers."""
    seed = int(cfg.get("seed", 42))
    test_size = int(cfg["data"].get("part1_test_size", 300))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    max_ratio = float(cfg["part1"].get("resample_max_ratio", 0.30))
    cv_folds = int(cfg["part1"].get("cv_folds", 5))
    sra_depth = int(cfg["part1"].get("sra_depth", 10))
    max_evals = int(cfg["part1"].get("permutation_shap_max_evals", 150))
    part1_bg_size = int(cfg["part1"].get("bg_size", 25))
    lime_samples = int(cfg["part1"].get("lime_num_samples", 500))
    cost_cfg = cfg["cost_model"]

    metric_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []

    with Parallel(n_jobs=n_workers) as parallel:
        for ds_idx, ds in enumerate(datasets):
            df, cb = load_dataset(ds, cfg)
            train_pool, test_df = make_part1_split(df, test_size=test_size, seed=seed)

            # Canonical feature order from full training pool
            canon_prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_pool)
            canon_features = canon_prep.feature_names_
            n_canon = len(canon_features)

            y_te = test_df["target"].to_numpy(dtype=int)
            a_pool = canon_prep.extract_exposure(train_pool)
            y_pool = train_pool["target"].to_numpy(dtype=float)
            a_te = canon_prep.extract_exposure(test_df)
            prior_pi1 = float(np.mean(y_pool))

            c_fn_te, c_fp_te = compute_cost_vectors(
                amounts_eval=a_te,
                amounts_train_ref=a_pool,
                y_train_ref=y_pool,
                lgd=float(cost_cfg["lgd"]),
                prior_pi1=prior_pi1,
                int_r_annual=float(cost_cfg["int_r_annual"]),
                int_cf_annual=float(cost_cfg["int_cf_annual"]),
                n_term_months=int(cost_cfg["n_term_months"]),
            )

            n_te = len(test_df)

            for r_idx, pi in enumerate(rates):
                t_rate = time.time()
                # Prepare all J resampled datasets + LIME discretisers once per j_idx
                iter_bundles: list[dict[str, Any]] = []
                for j_idx in range(j_iters):
                    iter_seed = seed + 10000 * ds_idx + 100 * r_idx + j_idx
                    tr_res = resample_to_pi(train_pool, target_pi=float(pi), seed=iter_seed, max_ratio=max_ratio)
                    prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(tr_res)
                    X_tr = prep.transform(tr_res)
                    y_tr = tr_res["target"].to_numpy(dtype=int)
                    X_te = prep.transform(test_df)
                    a_tr = prep.extract_exposure(tr_res)

                    # Fixed original priors pi_0, pi_1 per Ballegeer et al. Sec 3.3
                    c_fn_tr, c_fp_tr = compute_cost_vectors(
                        amounts_eval=a_tr,
                        amounts_train_ref=a_pool,
                        y_train_ref=y_pool,
                        lgd=float(cost_cfg["lgd"]),
                        prior_pi1=prior_pi1,
                        int_r_annual=float(cost_cfg["int_r_annual"]),
                        int_cf_annual=float(cost_cfg["int_cf_annual"]),
                        n_term_months=int(cost_cfg["n_term_months"]),
                    )
                    bg = sample_background(X_tr, n=min(part1_bg_size, len(X_tr)), seed=iter_seed)
                    le = build_lime_explainer(
                        X_tr, feature_names=prep.feature_names_, discretize_continuous=True, seed=iter_seed + 77
                    )
                    iter_bundles.append({
                        "iter_seed": iter_seed,
                        "X_tr": X_tr,
                        "y_tr": y_tr,
                        "X_te": X_te,
                        "c_fn_tr": c_fn_tr,
                        "c_fp_tr": c_fp_tr,
                        "bg": bg,
                        "local_features": prep.feature_names_,
                        "le": le,
                    })

                # Tune hyperparameters on iteration 0 in parallel across all models
                b0 = iter_bundles[0]
                tuned_list = parallel(
                    delayed(_tune_single_model)(
                        m_name, b0["X_tr"], b0["y_tr"], b0["c_fn_tr"], b0["c_fp_tr"],
                        cv_folds, b0["iter_seed"],
                    )
                    for m_name in models
                )
                tuned_params: dict[str, dict[str, Any]] = dict(tuned_list)

                # Store attributions: model -> explainer -> (J, n_te, n_canon)
                attr_store: dict[str, dict[str, np.ndarray]] = {
                    m: {e: np.zeros((j_iters, n_te, n_canon), dtype=float) for e in explainers}
                    for m in models
                }

                # Dispatch all (j_idx, model) pairs across the warm pool (order slow->fast for load balance)
                ordered_models = sorted(
                    models,
                    key=lambda m: 0 if "forest" in m else (1 if "boost" in m else 2),
                )
                unit_results = parallel(
                    delayed(_run_part1_unit)(
                        j_idx,
                        m_name,
                        tuned_params[m_name],
                        iter_bundles[j_idx],
                        canon_features,
                        y_te,
                        c_fn_te,
                        c_fp_te,
                        prior_pi1,
                        explainers,
                        max_evals,
                        lime_samples,
                    )
                    for j_idx in range(j_iters)
                    for m_name in ordered_models
                )

                for j_idx, m_name, ev, aligned in unit_results:
                    perf_rows.append({
                        "dataset": ds,
                        "pi": float(pi),
                        "pi_pct": round(float(pi) * 100.0, 2),
                        "j_iter": int(j_idx),
                        "model": m_name,
                        **ev,
                    })
                    for exp_name, mat in aligned.items():
                        attr_store[m_name][exp_name][j_idx] = mat

                # Compute per-instance CoV and SRA across J refits
                for m_name in models:
                    for exp_name in explainers:
                        cube = attr_store[m_name][exp_name]  # (J, n_te, n_canon)
                        for i in range(n_te):
                            inst_mat = cube[:, i, :]
                            cov_val = compute_cov(inst_mat, use_abs=True)
                            sra_val = compute_sra(inst_mat, depth=min(sra_depth, n_canon), use_abs=True)
                            metric_rows.append({
                                "dataset": ds,
                                "pi": float(pi),
                                "pi_pct": round(float(pi) * 100.0, 2),
                                "model": m_name,
                                "explainer": exp_name,
                                "instance_id": int(i),
                                "cov": float(cov_val),
                                "sra": float(sra_val),
                            })

                print(
                    f"  [Part 1] {ds} pi={pi*100:.0f}% (J={j_iters}) completed in {time.time() - t_rate:.1f}s",
                    flush=True,
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(perf_rows)


def _run_e1_unit(
    j_idx: int,
    k_idx: int,
    est: Any,
    bg: np.ndarray,
    X_te: np.ndarray,
    le: Any,
    local_features: list[str],
    canon_features: list[str],
    explainers: list[str],
    max_evals: int,
    lime_samples: int,
    rep_seed: int,
) -> tuple[int, int, dict[str, np.ndarray]]:
    if type(est).__name__ == "LogisticRegression":
        pred_est: Any = _FastLogitProxy(
            np.asarray(est.coef_, dtype=float).ravel(),
            float(np.asarray(est.intercept_, dtype=float).ravel()[0]),
        )
    else:
        pred_est = est
    aligned: dict[str, np.ndarray] = {}
    for exp_name in explainers:
        if exp_name == "permutation_shap":
            phi_local = explain_permutation_shap(pred_est, bg, X_te, max_evals=max_evals, seed=rep_seed)
        else:
            phi_local = explain_lime(le, pred_est, X_te, num_samples=lime_samples, seed=rep_seed)
        aligned[exp_name] = _align_attributions(phi_local, local_features, canon_features)
    return j_idx, k_idx, aligned


def run_e1_decomposition(
    cfg: dict[str, Any],
    j_iters: int,
    n_workers: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run E1 variance decomposition (J refits x K repeats) to isolate explainer stochasticity."""
    e1_cfg = cfg["part1"]["e1"]
    ds = str(e1_cfg["dataset"])
    rates = [float(r) for r in e1_cfg["rates"]]
    k_repeats = int(e1_cfg["k_repeats"])
    models = list(e1_cfg["models"])
    explainers = list(cfg["part1"]["explainers"])

    seed = int(cfg.get("seed", 42))
    test_size = int(cfg["data"].get("part1_test_size", 300))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    max_ratio = float(cfg["part1"].get("resample_max_ratio", 0.30))
    cv_folds = int(cfg["part1"].get("cv_folds", 5))
    max_evals = int(cfg["part1"].get("permutation_shap_max_evals", 150))
    part1_bg_size = int(cfg["part1"].get("bg_size", 25))
    lime_samples = int(cfg["part1"].get("lime_num_samples", 500))
    n_boot = int(cfg["stats"].get("bootstrap_resamples", 5000))
    alpha = float(cfg["stats"].get("alpha", 0.05))

    df, cb = load_dataset(ds, cfg)
    train_pool, test_df = make_part1_split(df, test_size=test_size, seed=seed)
    canon_prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_pool)
    canon_features = canon_prep.feature_names_
    n_canon = len(canon_features)
    n_te = len(test_df)

    e1_results: list[dict[str, Any]] = []
    with Parallel(n_jobs=n_workers) as parallel:
        for r_idx, pi in enumerate(rates):
            for m_name in models:
                canon_m = canonical_model_name(m_name)
                tuned = None
                cubes = {e: np.zeros((n_te, j_iters, k_repeats, n_canon), dtype=float) for e in explainers}

                fitted_iters: list[dict[str, Any]] = []
                for j_idx in range(j_iters):
                    iter_seed = seed + 50000 + 100 * r_idx + j_idx
                    tr_res = resample_to_pi(train_pool, target_pi=pi, seed=iter_seed, max_ratio=max_ratio)
                    prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(tr_res)
                    X_tr = prep.transform(tr_res)
                    y_tr = tr_res["target"].to_numpy(dtype=int)
                    X_te = prep.transform(test_df)
                    if tuned is None:
                        tuned = tune_hyperparameters(canon_m, X_tr, y_tr, cv_folds=cv_folds, seed=iter_seed, n_jobs=min(4, n_workers))
                    est = build_estimator(canon_m, params=tuned, seed=iter_seed, n_jobs=1)
                    fit_estimator(est, canon_m, X_tr, y_tr)
                    bg = sample_background(X_tr, n=min(part1_bg_size, len(X_tr)), seed=iter_seed)
                    le = build_lime_explainer(X_tr, feature_names=prep.feature_names_, discretize_continuous=True, seed=iter_seed)
                    fitted_iters.append({
                        "iter_seed": iter_seed,
                        "est": est,
                        "bg": bg,
                        "X_te": X_te,
                        "le": le,
                        "local_features": prep.feature_names_,
                    })

                unit_out = parallel(
                    delayed(_run_e1_unit)(
                        j_idx,
                        k_idx,
                        fitted_iters[j_idx]["est"],
                        fitted_iters[j_idx]["bg"],
                        fitted_iters[j_idx]["X_te"],
                        fitted_iters[j_idx]["le"],
                        fitted_iters[j_idx]["local_features"],
                        canon_features,
                        explainers,
                        max_evals,
                        lime_samples,
                        fitted_iters[j_idx]["iter_seed"] * 31 + k_idx * 997,
                    )
                    for j_idx in range(j_iters)
                    for k_idx in range(k_repeats)
                )
                for j_idx, k_idx, aligned in unit_out:
                    for exp_name, mat in aligned.items():
                        cubes[exp_name][:, j_idx, k_idx, :] = mat

                for exp_name in explainers:
                    dec = e1_variance_decomposition(cubes[exp_name], n_boot=n_boot, seed=seed, alpha=alpha)
                    e1_results.append({
                        "dataset": ds,
                        "pi": pi,
                        "pi_pct": round(pi * 100.0, 2),
                        "model": m_name,
                        "explainer": exp_name,
                        **dec,
                    })
                    print(
                        f"  [E1] {ds} pi={pi*100:.0f}% {m_name} {exp_name}: "
                        f"explainer_share={dec['pooled_explainer_share']:.3f} "
                        f"[{dec['pooled_explainer_share_ci_low']:.3f}, {dec['pooled_explainer_share_ci_high']:.3f}]",
                        flush=True,
                    )

    return e1_results, pd.DataFrame(e1_results)


def compute_h1_statistics(rep_df: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Run H1a (JT trend + Spearman + Holm) and H1b (paired Wilcoxon + KS + Holm)."""
    h1a_rows: list[dict[str, Any]] = []
    for (ds, model, exp), grp in rep_df.groupby(["dataset", "model", "explainer"]):
        ordered_rates = sorted(grp["pi"].unique())
        if len(ordered_rates) < 2:
            continue
        for metric in ("cov", "sra"):
            groups = [grp.loc[grp["pi"] == r, metric].to_numpy(dtype=float) for r in ordered_rates]
            jt = jonckheere_terpstra_test(groups, alternative="decreasing")
            sp = spearman_with_ci(grp["pi"].to_numpy(dtype=float), grp[metric].to_numpy(dtype=float), alpha=alpha)
            med_lo = float(np.median(groups[0]))
            med_hi = float(np.median(groups[-1]))
            ratio = float(med_lo / med_hi) if med_hi > 1e-12 else np.nan
            h1a_rows.append({
                "dataset": ds,
                "model": model,
                "explainer": exp,
                "metric": metric,
                "jt_z": jt["z"],
                "jt_tau": jt["tau_j"],
                "jt_pvalue": jt["pvalue"],
                "spearman_rho": sp["rho"],
                "spearman_ci_low": sp["ci_low"],
                "spearman_ci_high": sp["ci_high"],
                "spearman_pvalue": sp["pvalue"],
                "median_min_pi": med_lo,
                "median_max_pi": med_hi,
                "median_ratio_min_over_max": ratio,
            })

    h1a_df = pd.DataFrame(h1a_rows)
    if not h1a_df.empty:
        h1a_df["jt_p_holm"] = np.nan
        h1a_df["supported_holm"] = False
        for metric in ("cov", "sra"):
            mask = h1a_df["metric"] == metric
            holm = holm_bonferroni(h1a_df.loc[mask, "jt_pvalue"].tolist(), alpha=alpha)
            h1a_df.loc[mask, "jt_p_holm"] = holm["p_adjusted"]
            h1a_df.loc[mask, "supported_holm"] = np.asarray(holm["reject"], dtype=bool)

    # H1b: IDCS vs Traditional (cslogit vs logit, csboost vs boost) paired per instance at each pi
    h1b_rows: list[dict[str, Any]] = []
    pairs = [("cslogit", "logit"), ("csboost", "boost")]
    for (ds, pi, exp), grp in rep_df.groupby(["dataset", "pi", "explainer"]):
        for idcs_m, trad_m in pairs:
            g_idcs = grp[grp["model"] == idcs_m].sort_values("instance_id")
            g_trad = grp[grp["model"] == trad_m].sort_values("instance_id")
            if g_idcs.empty or g_trad.empty or len(g_idcs) != len(g_trad):
                continue
            for metric in ("cov", "sra"):
                x = g_idcs[metric].to_numpy(dtype=float)
                y = g_trad[metric].to_numpy(dtype=float)
                wx = paired_wilcoxon_test(x, y, alternative="greater")
                ks = ks_two_sample_test(x, y, alternative="two-sided")
                h1b_rows.append({
                    "dataset": ds,
                    "pi": float(pi),
                    "pi_pct": round(float(pi) * 100.0, 2),
                    "pair": f"{idcs_m}_vs_{trad_m}",
                    "explainer": exp,
                    "metric": metric,
                    "median_idcs": float(np.median(x)),
                    "median_trad": float(np.median(y)),
                    "wilcoxon_p": wx["pvalue"],
                    "rank_biserial_r": wx["rank_biserial_r"],
                    "ks_stat": ks["statistic"],
                    "ks_p": ks["pvalue"],
                })

    h1b_df = pd.DataFrame(h1b_rows)
    if not h1b_df.empty:
        h1b_df["wilcoxon_p_holm"] = np.nan
        h1b_df["supported_holm"] = False
        for metric in ("cov", "sra"):
            mask = h1b_df["metric"] == metric
            holm = holm_bonferroni(h1b_df.loc[mask, "wilcoxon_p"].tolist(), alpha=alpha)
            h1b_df.loc[mask, "wilcoxon_p_holm"] = holm["p_adjusted"]
            h1b_df.loc[mask, "supported_holm"] = np.asarray(holm["reject"], dtype=bool)

    return {
        "h1a_df": h1a_df,
        "h1b_df": h1b_df,
        "h1a_summary": {
            m: {
                "n_cells": int((h1a_df["metric"] == m).sum()),
                "n_supported_holm": int(h1a_df.loc[h1a_df["metric"] == m, "supported_holm"].sum()),
            }
            for m in ("cov", "sra")
        } if not h1a_df.empty else {},
        "h1b_summary": {
            m: {
                "n_cells": int((h1b_df["metric"] == m).sum()),
                "n_supported_holm": int(h1b_df.loc[h1b_df["metric"] == m, "supported_holm"].sum()),
            }
            for m in ("cov", "sra")
        } if not h1b_df.empty else {},
    }


def main(config_path: Path, pilot: bool = False, j_override: int | None = None) -> None:
    t_start = time.time()
    cfg = load_config(config_path)
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = int(cfg.get("n_jobs", -1))
    n_workers = os.cpu_count() or 4 if n_jobs <= 0 else n_jobs
    alpha = float(cfg["stats"].get("alpha", 0.05))

    if pilot:
        datasets = ["sgcs"]
        rates = [0.05, 0.30]
        j_iters = j_override or 3
        models = list(cfg["part1"]["models"])
        explainers = list(cfg["part1"]["explainers"])
        print(f"=== Stage 02 PILOT Timing Run (ds={datasets}, rates={rates}, J={j_iters}, workers={n_workers}) ===")
        rep_df, perf_df = run_part1_replication(cfg, datasets, rates, j_iters, models, explainers, n_workers)
        elapsed = time.time() - t_start
        full_j = int(cfg["part1"]["j_iterations"])
        full_cells = len(cfg["part1"]["datasets"]) * len(cfg["part1"]["default_rates"]) * full_j
        pilot_cells = len(datasets) * len(rates) * j_iters
        est_full_sec = elapsed * (full_cells / max(1, pilot_cells))
        print(f"Pilot elapsed: {elapsed:.1f}s for {pilot_cells} dataset*rate*J cells.")
        print(f"Projected full J={full_j} runtime: {est_full_sec/60.0:.1f} min (HMEQ has fewer one-hot features than SGCS so likely faster).")
        return

    datasets = list(cfg["part1"]["datasets"])
    rates = [float(r) for r in cfg["part1"]["default_rates"]]
    j_iters = int(j_override if j_override is not None else cfg["part1"]["j_iterations"])
    cut3_applied = False
    if len(rates) > 5 and j_iters >= 10:
        # Runtime governance (plan.md Sec 12.2 Cut #3): 5 ordered rates spanning 1%..30%
        rates = [0.01, 0.05, 0.10, 0.20, 0.30]
        cut3_applied = True
    models = list(cfg["part1"]["models"])
    explainers = list(cfg["part1"]["explainers"])

    print(f"=== Stage 02: Part 1 Replication (datasets={datasets}, rates={rates}, J={j_iters}, workers={n_workers}) ===")
    rep_df, perf_df = run_part1_replication(cfg, datasets, rates, j_iters, models, explainers, n_workers)

    rep_df.to_parquet(results_dir / "part1_replication_metrics.parquet", index=False)
    rep_df.to_csv(results_dir / "part1_replication_metrics.csv", index=False)
    perf_df.to_csv(results_dir / "part1_model_performance.csv", index=False)

    summary_df = (
        rep_df.groupby(["dataset", "pi", "pi_pct", "model", "explainer"], as_index=False)
        .agg(
            cov_median=("cov", "median"),
            cov_mean=("cov", "mean"),
            cov_q25=("cov", lambda s: float(np.percentile(s, 25))),
            cov_q75=("cov", lambda s: float(np.percentile(s, 75))),
            sra_median=("sra", "median"),
            sra_mean=("sra", "mean"),
            sra_q25=("sra", lambda s: float(np.percentile(s, 25))),
            sra_q75=("sra", lambda s: float(np.percentile(s, 75))),
        )
    )
    summary_df.to_csv(results_dir / "part1_replication_summary.csv", index=False)

    perf_summary = (
        perf_df.groupby(["dataset", "pi", "pi_pct", "model"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            brier_mean=("brier", "mean"),
            rel_aec_mean=("rel_aec", "mean"),
            savings_mean=("savings", "mean"),
        )
    )
    perf_summary.to_csv(results_dir / "part1_model_performance_summary.csv", index=False)

    h1_out = compute_h1_statistics(rep_df, alpha=alpha)
    if not h1_out["h1a_df"].empty:
        h1_out["h1a_df"].to_csv(results_dir / "part1_h1a_trend.csv", index=False)
    if not h1_out["h1b_df"].empty:
        h1_out["h1b_df"].to_csv(results_dir / "part1_h1b_idcs_vs_trad.csv", index=False)

    print("=== Stage 02: E1 Variance Decomposition ===")
    e1_list, e1_df = run_e1_decomposition(cfg, j_iters=j_iters, n_workers=n_workers)
    e1_df.to_csv(results_dir / "part1_e1_variance_decomp.csv", index=False)
    with (results_dir / "part1_e1_variance_decomp.json").open("w", encoding="utf-8") as fh:
        json.dump(e1_list, fh, indent=2)

    h1_summary = {
        "h1a": h1_out["h1a_summary"],
        "h1b": h1_out["h1b_summary"],
        "j_iterations": j_iters,
        "runtime_governance": {
            "cut3_5_rates_applied": cut3_applied,
            "rates": rates,
            "part1_bg_size": int(cfg["part1"].get("bg_size", 25)),
            "part1_lime_num_samples": int(cfg["part1"].get("lime_num_samples", 500)),
            "part1_permutation_shap_max_evals": int(cfg["part1"].get("permutation_shap_max_evals", 150)),
        },
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    with (results_dir / "part1_h1_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(h1_summary, fh, indent=2)

    print(f"=== Stage 02 Complete in {time.time() - t_start:.1f}s ===")
    print("H1 summary:", json.dumps(h1_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 02: Part 1 Base Paper Replication + E1.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml")
    parser.add_argument("--pilot", action="store_true", help="Run fast pilot timing benchmark.")
    parser.add_argument("--j", type=int, default=None, help="Override J iterations.")
    args = parser.parse_args()
    main(args.config, pilot=args.pilot, j_override=args.j)
