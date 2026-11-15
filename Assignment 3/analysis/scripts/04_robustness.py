"""Stage 04: Comprehensive Robustness Analysis (R1-R9).

Runs and aggregates all pre-registered robustness checks:
  R1: Nested OLS M0->M3 & F(x) component ablation (table + plot data)
  R2: Prediction-change tolerance tau in {0.02, 0.05, 0.10}
  R3: Perturbation scale s in {0.01, 0.05, 0.10} (runs s=0.10 and combines with s=0.01, 0.05)
  R4: Top-k depth k in {3, 4, 5} & distance metrics D_k, D_rho, D_1
  R5: LIME num_samples 1000 vs 5000 (+ discretisation on/off summary)
  R6: Exact TreeSHAP vs PermutationExplainer summary
  R7: Subsamples (defaulters vs non-defaulters, per-dataset) & Leave-One-Dataset-Out (LODO) F-gate
  R8: Operating threshold t=prior vs t=0.50
  R9: Part 1 IDCS r_i margin assumption (m=0.02 vs Bahnsen PV formula)

Usage:
    python scripts/04_robustness.py --config config.yaml
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
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from src.crossings import count_bin_crossings, count_tree_crossings, extract_lime_bin_edges, extract_tree_splits  # noqa: E402
from src.data import Preprocessor, load_config, load_dataset, make_part2_split, resolve_path  # noqa: E402
from src.explain import build_lime_explainer, explain_lime, explain_linear_shap, explain_tree_shap, sample_background  # noqa: E402
from src.fragility import baseline_margin, compute_f_bin, compute_f_tree  # noqa: E402
from src.idcs import compute_cost_vectors  # noqa: E402
from src.metrics import jaccard_distance_topk, normalized_l1_distance, spearman_distance  # noqa: E402
from src.models import build_estimator, canonical_model_name, evaluate_model, fit_estimator, tune_hyperparameters  # noqa: E402
from src.perturb import filter_prediction_preserving, generate_perturbations  # noqa: E402
from src.stats import (  # noqa: E402
    cluster_bootstrap_auroc,
    delong_paired_auroc_test,
    nested_ols_hc3,
    spearman_with_ci,
)


def _stratified_sample_indices(y_hat: np.ndarray, sample_n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_total = len(y_hat)
    pos_idx = np.flatnonzero(y_hat == 1)
    neg_idx = np.flatnonzero(y_hat == 0)
    n_target = min(int(sample_n), n_total)
    n_half = n_target // 2
    n_pos = min(len(pos_idx), n_half)
    n_neg = min(len(neg_idx), n_target - n_pos)
    if n_pos + n_neg < n_target:
        rem = n_target - (n_pos + n_neg)
        if len(pos_idx) - n_pos >= rem:
            n_pos += rem
        else:
            n_neg = min(len(neg_idx), n_neg + rem)
    cp = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.empty(0, dtype=int)
    cn = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.empty(0, dtype=int)
    chosen = np.concatenate([cp, cn])
    chosen.sort()
    return chosen


def _noise_floor_q95(floor_phis: np.ndarray, k_top: int) -> float:
    K = floor_phis.shape[0]
    if K < 2:
        return 0.0
    dks = [
        jaccard_distance_topk(floor_phis[a], floor_phis[b], k=k_top)
        for a in range(K) for b in range(a + 1, K)
    ]
    arr = np.asarray(dks, dtype=float)
    return 0.0 if np.all(arr == 0.0) else float(np.percentile(arr, 95))


def run_r1_nested_and_ablation(inst_primary: pd.DataFrame, cfg: dict[str, Any], results_dir: Path) -> pd.DataFrame:
    """R1: Nested models M0->M3 and one-at-a-time F(x) component ablation."""
    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg["stats"].get("bootstrap_resamples", 5000))
    alpha = float(cfg["stats"].get("alpha", 0.05))

    # Filter to TreeSHAP + LIME (where F(x) is defined; also keep LinearSHAP in full M0->M3)
    df_all = inst_primary.dropna(subset=["u_bar", "y_instab", "mean_abs_dp", "mean_c_tree", "mean_c_bin", "score_f"]).reset_index(drop=True)
    d_ds = pd.get_dummies(df_all["dataset"], prefix="ds", drop_first=True, dtype=float)
    d_m = pd.get_dummies(df_all["model"], prefix="m", drop_first=True, dtype=float)
    d_e = pd.get_dummies(df_all["explainer"], prefix="e", drop_first=True, dtype=float)

    m0 = pd.concat([d_ds, d_m, d_e], axis=1)
    m1 = pd.concat([m0, df_all[["mean_abs_dp"]]], axis=1)
    m2 = pd.concat([m1, df_all[["mean_c_tree", "mean_c_bin"]]], axis=1)
    m3 = pd.concat([m2, df_all[["score_f"]]], axis=1)

    nested_res = nested_ols_hc3(
        df_all["u_bar"].to_numpy(dtype=float),
        [("M0: Base", m0), ("M1: +|Delta p|", m1), ("M2: +Crossings", m2), ("M3: +F(x)", m3)],
        n_boot=n_boot, seed=seed, alpha=alpha,
    )

    rows: list[dict[str, Any]] = []
    # For M0..M3, record adj_r2 and delta_r2
    for idx, m_entry in enumerate(nested_res["models"]):
        rows.append({
            "variant": m_entry["model"],
            "category": "nested_ols",
            "n_obs": m_entry["n_obs"],
            "r2": m_entry["r2"],
            "adj_r2": m_entry["adj_r2"],
            "delta_r2": m_entry["delta_r2"] if idx > 0 else m_entry["r2"],
            "ci_low": m_entry["delta_r2_ci_low"] if idx > 0 else m_entry.get("r2_ci_low", np.nan),
            "ci_high": m_entry["delta_r2_ci_high"] if idx > 0 else m_entry.get("r2_ci_high", np.nan),
            "wald_f_hc3": m_entry["wald_f_hc3"],
            "wald_p_hc3": m_entry["wald_pvalue_hc3"],
            "auroc": np.nan,
            "delta_auroc": np.nan,
        })

    # Component ablation of F(x) on TreeSHAP + LIME instances
    df_f = df_all[df_all["explainer"] != "LinearSHAP"].copy().reset_index(drop=True)
    if not df_f.empty and df_f["y_instab"].nunique() >= 2:
        yt = df_f["y_instab"].to_numpy(dtype=int)
        u_vals = df_f["u_bar"].to_numpy(dtype=float)
        f_full_unw = df_f["score_f"].to_numpy(dtype=float)
        f_full_cov = df_f["score_f_cov"].to_numpy(dtype=float)
        # Drop F_bin (set LIME F to 0 -> tree-only)
        f_drop_bin = np.where(df_f["explainer"] == "TreeSHAP", df_f["f_tree_unw"].to_numpy(dtype=float), 0.0)
        # Drop F_tree (set TreeSHAP F to 0 -> bin-only)
        f_drop_tree = np.where(df_f["explainer"] == "LIME", df_f["f_bin"].to_numpy(dtype=float), 0.0)

        full_auc = float(roc_auc_score(yt, f_full_unw))
        d_ds_f = pd.get_dummies(df_f["dataset"], prefix="ds", drop_first=True, dtype=float)
        d_m_f = pd.get_dummies(df_f["model"], prefix="m", drop_first=True, dtype=float)
        d_e_f = pd.get_dummies(df_f["explainer"], prefix="e", drop_first=True, dtype=float)
        base_f = pd.concat([d_ds_f, d_m_f, d_e_f], axis=1)

        base_ols = nested_ols_hc3(u_vals, [("Base", base_f)], n_boot=0, seed=seed)["models"][0]
        base_r2 = base_ols["r2"]

        abl_specs = [
            ("Full F(x) (unw)", f_full_unw),
            ("Abl: Cover-Weighted F_tree", f_full_cov),
            ("Abl: Drop F_bin (Tree only)", f_drop_bin),
            ("Abl: Drop F_tree (Bin only)", f_drop_tree),
        ]
        for label, f_vec in abl_specs:
            auc_v = float(roc_auc_score(yt, f_vec))
            dl = delong_paired_auroc_test(yt, f_vec, f_full_unw, alternative="two-sided", alpha=alpha)
            x_aug = base_f.copy()
            x_aug["f_var"] = f_vec
            ols_v = nested_ols_hc3(u_vals, [("Base", base_f), ("Aug", x_aug)], n_boot=min(2000, n_boot), seed=seed)
            aug_m = ols_v["models"][1]
            rows.append({
                "variant": label,
                "category": "f_ablation",
                "n_obs": len(df_f),
                "r2": aug_m["r2"],
                "adj_r2": aug_m["adj_r2"],
                "delta_r2": float(aug_m["r2"] - base_r2),
                "ci_low": aug_m["delta_r2_ci_low"],
                "ci_high": aug_m["delta_r2_ci_high"],
                "wald_f_hc3": aug_m["wald_f_hc3"],
                "wald_p_hc3": aug_m["wald_pvalue_hc3"],
                "auroc": auc_v,
                "delta_auroc": float(auc_v - full_auc),
                "delong_p_vs_full": dl["pvalue"],
            })

    r1_df = pd.DataFrame(rows)
    r1_df.to_parquet(results_dir / "r1_nested_ablation.parquet", index=False)
    r1_df.to_csv(results_dir / "r1_nested_ablation.csv", index=False)
    with (results_dir / "r1_nested_ols_details.json").open("w", encoding="utf-8") as fh:
        json.dump(nested_res, fh, indent=2)
    return r1_df


def run_r2_tau_sensitivity(draws_df: pd.DataFrame, inst_primary: pd.DataFrame, cfg: dict[str, Any], results_dir: Path) -> pd.DataFrame:
    """R2: Sensitivity to filter strictness tau in {0.02, 0.05, 0.10}."""
    seed = int(cfg.get("seed", 42))
    n_boot = min(2000, int(cfg["stats"].get("bootstrap_resamples", 5000)))
    tau_vals = [float(t) for t in cfg["robustness"]["tau_values"]]
    y_cutoff = float(cfg["part2"]["instability_cutoff"])
    scales = sorted(draws_df["s"].unique())
    primary_s = max(scales) if 0.05 not in scales else 0.05

    sub_s = draws_df[draws_df["s"] == primary_s].copy()
    # Map threshold boundary check from existing p_x and p_prime using y_hat == (p_prime >= t)
    # Note: draws_df already has `preserved` at tau=0.05; since preserved = (same_class) & (abs_dp <= tau),
    # for tau=0.02 we can strictly refine preserved; for tau=0.10, same_class is inferred from p_x and p_prime
    # vs threshold, which we can recover from perf_df or simply infer per dataset threshold!
    perf_df = pd.read_csv(results_dir / "part2_model_performance.csv")
    thresh_map = perf_df.groupby("dataset")["threshold_prior"].first().to_dict()

    f_lookup = inst_primary.set_index(["dataset", "model", "explainer", "instance_id"])["score_f"].to_dict()

    rows: list[dict[str, Any]] = []
    for tau_val in tau_vals:
        df_t = sub_s.copy()
        df_t["thresh"] = df_t["dataset"].map(thresh_map)
        df_t["y_prime"] = (df_t["p_prime"] >= df_t["thresh"]).astype(int)
        df_t["preserved_tau"] = (df_t["y_prime"] == df_t["y_hat"]) & (df_t["abs_dp"] <= tau_val)

        filt = df_t[df_t["preserved_tau"]]
        # Compute instance-level u_bar at tau_val
        inst_t = (
            filt.groupby(["dataset", "model", "explainer", "instance_id"], as_index=False)
            .agg(u_bar=("rc", "mean"), d_k=("d_k", "mean"))
        )
        inst_t["y_instab"] = (inst_t["u_bar"] >= y_cutoff).astype(int)
        inst_t["score_f"] = [
            f_lookup.get((r.dataset, r.model, r.explainer, r.instance_id), np.nan)
            for r in inst_t.itertuples(index=False)
        ]

        # H3 AUROC on tree models (filtered) & H5 AUROC of F(x)
        tree_filt = filt[filt["model"].isin(["rf", "xgb"])]
        h3_tree = cluster_bootstrap_auroc(
            tree_filt["rc"].to_numpy(dtype=int),
            tree_filt["abs_dp"].to_numpy(dtype=float),
            (tree_filt["dataset"] + "_" + tree_filt["model"] + "_" + tree_filt["instance_id"].astype(str)).to_numpy(),
            n_boot=min(1000, n_boot), seed=seed,
        )
        inst_f = inst_t[(inst_t["explainer"] != "LinearSHAP")].dropna(subset=["score_f", "y_instab"])
        auc_f = float(roc_auc_score(inst_f["y_instab"], inst_f["score_f"])) if inst_f["y_instab"].nunique() >= 2 else np.nan

        rows.append({
            "tau": tau_val,
            "pass_rate": float(df_t["preserved_tau"].mean()),
            "n_preserved_draws": int(len(filt)),
            "mean_u_bar": float(inst_t["u_bar"].mean()),
            "y_instab_rate": float(inst_t["y_instab"].mean()),
            "h3_tree_auroc": h3_tree["auroc"],
            "h3_tree_ci_high": h3_tree["ci_high"],
            "h5_f_auroc": auc_f,
        })

    r2_df = pd.DataFrame(rows)
    r2_df.to_csv(results_dir / "r2_tau_sensitivity.csv", index=False)
    return r2_df


def run_r3_scale_s010(inst_all_s: pd.DataFrame, cfg: dict[str, Any], n_workers: int, results_dir: Path) -> pd.DataFrame:
    """R3: Run s=0.10 perturbations and combine with s=0.01 and s=0.05."""
    seed = int(cfg.get("seed", 42))
    test_frac = float(cfg["data"].get("part2_test_frac", 0.30))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    p2_cfg = cfg["part2"]
    datasets = list(p2_cfg["datasets"])
    models = list(p2_cfg["models"])
    sample_n = int(p2_cfg["sample_n"])
    k_floor = int(p2_cfg["noise_floor_k"])
    R = int(p2_cfg["perturbation_r"])
    tau = float(p2_cfg["tau"])
    top_k = int(p2_cfg["top_k"])
    y_cutoff = float(p2_cfg["instability_cutoff"])
    bg_size = int(p2_cfg["tree_shap_background_size"])
    lime_samples = int(p2_cfg["lime_num_samples"])
    cv_folds = int(cfg["part1"].get("cv_folds", 5))
    s_new = 0.10

    # Reuse existing noise_floor from inst_all_s to save LIME K repeats, or compute baseline+perturbed at s=0.10
    s05 = inst_all_s[inst_all_s["s"] == max(inst_all_s["s"].unique())].copy()
    nf_map = s05.set_index(["dataset", "model", "explainer", "instance_id"])["noise_floor"].to_dict()

    new_rows: list[dict[str, Any]] = []
    for ds_idx, ds in enumerate(datasets):
        df, cb = load_dataset(ds, cfg)
        train_df, test_df = make_part2_split(df, test_frac=test_frac, seed=seed)
        prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_df)
        X_tr, y_tr = prep.transform(train_df), train_df["target"].to_numpy(dtype=int)
        X_te = prep.transform(test_df)
        feature_names = prep.feature_names_
        prior_pi1 = float(np.mean(y_tr))
        threshold = prior_pi1 if p2_cfg["threshold_mode"] == "prior" else float(p2_cfg["threshold_mode"])
        bg = sample_background(X_tr, n=bg_size, seed=seed + ds_idx)

        lime_ref = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed)
        bin_edges = extract_lime_bin_edges(lime_ref)

        for m_idx, m_name in enumerate(models):
            canon_m = canonical_model_name(m_name)
            m_seed = seed + ds_idx * 100 + m_idx
            best_p = tune_hyperparameters(canon_m, X_tr, y_tr, cv_folds=cv_folds, seed=m_seed, n_jobs=1)
            est = fit_estimator(build_estimator(canon_m, params=best_p, seed=m_seed, n_jobs=1), canon_m, X_tr, y_tr)
            p_te = est.predict_proba(X_te)[:, 1]
            y_hat_te = (p_te >= threshold).astype(int)
            samp_idx = _stratified_sample_indices(y_hat_te, sample_n=sample_n, seed=seed + ds_idx * 10 + m_idx)
            X_samp = X_te[samp_idx]
            n_samp = len(samp_idx)

            is_tree = canon_m in ("forest", "boost")
            tree_splits = extract_tree_splits(est, feature_names=feature_names) if is_tree else None

            xp_all = np.zeros((n_samp, R, X_samp.shape[1]), dtype=float)
            f_infos = []
            for li, iid in enumerate(samp_idx):
                p_seed = seed + ds_idx * 1000000 + m_idx * 100000 + int(s_new * 1000) * 1000 + int(iid)
                xp = generate_perturbations(
                    X_samp[li], s=s_new, feature_sds=prep.feature_sds_,
                    perturbable_mask=prep.perturbable_mask_,
                    clip_low=prep.clip_low_, clip_high=prep.clip_high_,
                    n_draws=R, seed=p_seed, clip=True,
                )
                xp_all[li] = xp
                f_infos.append(filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau))

            shap_name = "LinearSHAP" if canon_m == "logit" else "TreeSHAP"
            if shap_name == "LinearSHAP":
                phi_b_shap, _ = explain_linear_shap(est, bg, X_samp)
                phi_p_shap = explain_linear_shap(est, bg, xp_all.reshape(-1, X_samp.shape[1]))[0].reshape(n_samp, R, -1)
            else:
                phi_b_shap, _ = explain_tree_shap(est, bg, X_samp)
                phi_p_shap = explain_tree_shap(est, bg, xp_all.reshape(-1, X_samp.shape[1]))[0].reshape(n_samp, R, -1)

            def _lime_s010(idx_arr: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
                le = build_lime_explainer(X_tr, feature_names=feature_names, discretize_continuous=True, seed=seed + ds_idx * 50000 + m_idx * 5000)
                out = []
                for li in idx_arr:
                    iid = int(samp_idx[li])
                    pb = explain_lime(le, est, X_samp[li : li + 1], num_samples=lime_samples, seed=seed + 900000 + iid * 101)[0]
                    pp = explain_lime(le, est, xp_all[li], num_samples=lime_samples, seed=seed + 200000 + int(s_new * 1000) * 10000 + iid * 100)
                    out.append((li, pb, pp))
                return out

            eff_w = min(max(1, n_workers), n_samp)
            lime_res_list = Parallel(n_jobs=eff_w)(delayed(_lime_s010)(idx) for idx in np.array_split(np.arange(n_samp), eff_w) if len(idx) > 0)
            lime_map = {li: (pb, pp) for ch in lime_res_list for (li, pb, pp) in ch}

            for li, iid in enumerate(samp_idx):
                pres = f_infos[li]["preserved"]
                if not np.any(pres):
                    continue
                for exp_label in (shap_name, "LIME"):
                    if exp_label == shap_name:
                        pb = phi_b_shap[li]
                        pp = phi_p_shap[li]
                        q_i = 0.0
                        sf = compute_f_tree(X_samp[li], tree_splits, s_new, prep.feature_sds_, prep.perturbable_mask_, weighted=False) if tree_splits else 0.0
                    else:
                        pb, pp = lime_map[li]
                        q_i = float(nf_map.get((ds, m_name, "LIME", int(iid)), 0.0))
                        sf = compute_f_bin(X_samp[li], bin_edges, s_new, prep.feature_sds_, prep.perturbable_mask_)

                    dks = np.array([jaccard_distance_topk(pb, pp[r_i], k=top_k) for r_i in range(R)])
                    rcs = (dks > q_i + 1e-12).astype(int)
                    u_bar = float(np.mean(rcs[pres]))
                    new_rows.append({
                        "dataset": ds, "model": m_name, "explainer": exp_label,
                        "s": s_new, "instance_id": int(iid),
                        "u_bar": u_bar, "d_k": float(np.mean(dks[pres])),
                        "y_instab": int(u_bar >= y_cutoff),
                        "pass_rate": float(f_infos[li]["pass_rate"]),
                        "score_f": float(sf),
                    })

    s010_df = pd.DataFrame(new_rows)
    combined = pd.concat([
        inst_all_s[["dataset", "model", "explainer", "s", "instance_id", "u_bar", "d_k", "y_instab", "pass_rate", "score_f"]],
        s010_df,
    ], ignore_index=True)
    combined["cell"] = combined["model"].str.upper() + " + " + combined["explainer"]

    agg = (
        combined.groupby(["s", "cell", "model", "explainer"], as_index=False)
        .agg(
            instability_rate=("u_bar", "mean"),
            median_d_k=("d_k", "median"),
            y_instab_rate=("y_instab", "mean"),
            pass_rate=("pass_rate", "mean"),
        )
    )
    agg.to_parquet(results_dir / "r3_scale_sensitivity.parquet", index=False)
    agg.to_csv(results_dir / "r3_scale_sensitivity.csv", index=False)
    combined.to_csv(results_dir / "r3_scale_instance_level.csv", index=False)
    return agg


def run_r4_outcome_definitions(draws_df: pd.DataFrame, inst_primary: pd.DataFrame, cfg: dict[str, Any], results_dir: Path) -> pd.DataFrame:
    """R4: Outcome metric sensitivity (D_k, D_rho, D_1) and top-k depth k in {3, 4, 5}."""
    seed = int(cfg.get("seed", 42))
    test_frac = float(cfg["data"].get("part2_test_frac", 0.30))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    p2_cfg = cfg["part2"]
    k_vals = [int(k) for k in cfg["robustness"]["k_values"]]
    scales = sorted(draws_df["s"].unique())
    primary_s = max(scales) if 0.05 not in scales else 0.05

    rows: list[dict[str, Any]] = []
    # Part A: D_k vs D_rho vs D_1 from existing draws_df at primary_s
    filt = draws_df[(draws_df["s"] == primary_s) & (draws_df["preserved"])].copy()
    inst_metrics = (
        filt.groupby(["dataset", "model", "explainer", "instance_id"], as_index=False)
        .agg(d_k=("d_k", "mean"), d_rho=("d_rho", "mean"), d_1=("d_1", "mean"))
    )
    merged = inst_metrics.merge(
        inst_primary[["dataset", "model", "explainer", "instance_id", "score_f", "y_instab"]],
        on=["dataset", "model", "explainer", "instance_id"],
        how="inner",
    )
    sub_f = merged[merged["explainer"] != "LinearSHAP"].dropna(subset=["score_f"])
    for m_col in ("d_k", "d_rho", "d_1"):
        sp = spearman_with_ci(sub_f["score_f"].to_numpy(), sub_f[m_col].to_numpy())
        rows.append({
            "check_type": "metric_variant",
            "variant": m_col,
            "k": 4,
            "n_instances": int(len(sub_f)),
            "mean_value": float(sub_f[m_col].mean()),
            "median_value": float(sub_f[m_col].median()),
            "spearman_with_f": sp["rho"],
            "spearman_ci_low": sp["ci_low"],
            "spearman_ci_high": sp["ci_high"],
            "auroc_f_y": float(roc_auc_score(sub_f["y_instab"], sub_f["score_f"])) if sub_f["y_instab"].nunique() >= 2 else np.nan,
        })

    # Part B: k in {3, 4, 5} on HMEQ + XGB (TreeSHAP) across all sample_n instances
    df, cb = load_dataset("hmeq", cfg)
    train_df, test_df = make_part2_split(df, test_frac=test_frac, seed=seed)
    prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_df)
    X_tr, y_tr = prep.transform(train_df), train_df["target"].to_numpy(dtype=int)
    X_te = prep.transform(test_df)
    prior_pi1 = float(np.mean(y_tr))
    threshold = prior_pi1 if p2_cfg["threshold_mode"] == "prior" else float(p2_cfg["threshold_mode"])
    best_p = tune_hyperparameters("boost", X_tr, y_tr, cv_folds=int(cfg["part1"]["cv_folds"]), seed=seed + 2, n_jobs=1)
    est = fit_estimator(build_estimator("boost", params=best_p, seed=seed + 2, n_jobs=1), "boost", X_tr, y_tr)
    bg = sample_background(X_tr, n=int(p2_cfg["tree_shap_background_size"]), seed=seed)
    p_te = est.predict_proba(X_te)[:, 1]
    samp_idx = _stratified_sample_indices((p_te >= threshold).astype(int), sample_n=int(p2_cfg["sample_n"]), seed=seed + 2)
    X_samp = X_te[samp_idx]
    R = int(p2_cfg["perturbation_r"])
    tau = float(p2_cfg["tau"])
    y_cutoff = float(p2_cfg["instability_cutoff"])
    ts = extract_tree_splits(est, feature_names=prep.feature_names_)
    phi_b, _ = explain_tree_shap(est, bg, X_samp)

    for k_val in k_vals:
        u_list, f_list, dk_list = [], [], []
        for li, iid in enumerate(samp_idx):
            xp = generate_perturbations(
                X_samp[li], s=primary_s, feature_sds=prep.feature_sds_,
                perturbable_mask=prep.perturbable_mask_,
                clip_low=prep.clip_low_, clip_high=prep.clip_high_,
                n_draws=R, seed=seed + 100000 * 2 + int(primary_s * 1000) * 1000 + int(iid), clip=True,
            )
            f_info = filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau)
            pres = f_info["preserved"]
            if not np.any(pres):
                continue
            phi_ps, _ = explain_tree_shap(est, bg, xp)
            dks = np.array([jaccard_distance_topk(phi_b[li], phi_ps[r_i], k=k_val) for r_i in range(R)])
            u_list.append(float(np.mean((dks[pres] > 1e-12).astype(float))))
            dk_list.append(float(np.mean(dks[pres])))
            f_list.append(compute_f_tree(X_samp[li], ts, primary_s, prep.feature_sds_, prep.perturbable_mask_, weighted=False))
        u_arr = np.asarray(u_list)
        f_arr = np.asarray(f_list)
        y_k = (u_arr >= y_cutoff).astype(int)
        sp_k = spearman_with_ci(f_arr, u_arr)
        rows.append({
            "check_type": "topk_depth_hmeq_xgb_treeshap",
            "variant": f"k={k_val}",
            "k": int(k_val),
            "n_instances": int(len(u_arr)),
            "mean_value": float(np.mean(u_arr)),
            "median_value": float(np.median(dk_list)),
            "spearman_with_f": sp_k["rho"],
            "spearman_ci_low": sp_k["ci_low"],
            "spearman_ci_high": sp_k["ci_high"],
            "auroc_f_y": float(roc_auc_score(y_k, f_arr)) if len(np.unique(y_k)) >= 2 else np.nan,
        })

    r4_df = pd.DataFrame(rows)
    r4_df.to_csv(results_dir / "r4_outcome_definitions.csv", index=False)
    return r4_df


def run_r5_lime_samples(cfg: dict[str, Any], n_workers: int, results_dir: Path) -> pd.DataFrame:
    """R5: Compare LIME num_samples 1000 vs 5000 on HMEQ + XGB."""
    seed = int(cfg.get("seed", 42))
    test_frac = float(cfg["data"].get("part2_test_frac", 0.30))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    p2_cfg = cfg["part2"]
    sample_vals = [int(v) for v in cfg["robustness"]["lime_num_samples_values"]]
    primary_s = 0.05
    R = int(p2_cfg["perturbation_r"])
    k_floor = min(5, int(p2_cfg["noise_floor_k"]))
    top_k = int(p2_cfg["top_k"])
    tau = float(p2_cfg["tau"])
    y_cutoff = float(p2_cfg["instability_cutoff"])

    df, cb = load_dataset("hmeq", cfg)
    train_df, test_df = make_part2_split(df, test_frac=test_frac, seed=seed)
    prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(train_df)
    X_tr, y_tr = prep.transform(train_df), train_df["target"].to_numpy(dtype=int)
    X_te = prep.transform(test_df)
    prior_pi1 = float(np.mean(y_tr))
    threshold = prior_pi1 if p2_cfg["threshold_mode"] == "prior" else float(p2_cfg["threshold_mode"])
    best_p = tune_hyperparameters("boost", X_tr, y_tr, cv_folds=int(cfg["part1"]["cv_folds"]), seed=seed + 2, n_jobs=1)
    est = fit_estimator(build_estimator("boost", params=best_p, seed=seed + 2, n_jobs=1), "boost", X_tr, y_tr)
    p_te = est.predict_proba(X_te)[:, 1]
    samp_idx = _stratified_sample_indices((p_te >= threshold).astype(int), sample_n=int(p2_cfg["sample_n"]), seed=seed + 2)
    X_samp = X_te[samp_idx]
    n_samp = len(samp_idx)

    lime_ref = build_lime_explainer(X_tr, feature_names=prep.feature_names_, discretize_continuous=True, seed=seed)
    bin_edges = extract_lime_bin_edges(lime_ref)

    rows: list[dict[str, Any]] = []
    for n_smp in sample_vals:
        def _worker(idx_arr: np.ndarray) -> list[dict[str, Any]]:
            le = build_lime_explainer(X_tr, feature_names=prep.feature_names_, discretize_continuous=True, seed=seed)
            out = []
            for li in idx_arr:
                iid = int(samp_idx[li])
                xp = generate_perturbations(
                    X_samp[li], s=primary_s, feature_sds=prep.feature_sds_,
                    perturbable_mask=prep.perturbable_mask_,
                    clip_low=prep.clip_low_, clip_high=prep.clip_high_,
                    n_draws=R, seed=seed + 100000 * 2 + int(primary_s * 1000) * 1000 + iid, clip=True,
                )
                f_info = filter_prediction_preserving(est, X_samp[li], xp, threshold=threshold, tau=tau)
                pres = f_info["preserved"]
                if not np.any(pres):
                    continue
                pb = explain_lime(le, est, X_samp[li : li + 1], num_samples=n_smp, seed=seed + 900000 + iid)[0]
                fl = np.vstack([explain_lime(le, est, X_samp[li : li + 1], num_samples=n_smp, seed=seed + 100000 + iid * 20 + kk) for kk in range(k_floor)])
                q95 = _noise_floor_q95(fl, k_top=top_k)
                pp = explain_lime(le, est, xp, num_samples=n_smp, seed=seed + 200000 + iid)
                dks = np.array([jaccard_distance_topk(pb, pp[r_i], k=top_k) for r_i in range(R)])
                rcs = (dks > q95 + 1e-12).astype(float)
                fb = compute_f_bin(X_samp[li], bin_edges, primary_s, prep.feature_sds_, prep.perturbable_mask_)
                out.append({
                    "q95": q95,
                    "d_k": float(np.mean(dks[pres])),
                    "u_bar": float(np.mean(rcs[pres])),
                    "f_bin": float(fb),
                })
            return out

        eff_w = min(max(1, n_workers), n_samp)
        res_list = [r for ch in Parallel(n_jobs=eff_w)(delayed(_worker)(idx) for idx in np.array_split(np.arange(n_samp), eff_w) if len(idx) > 0) for r in ch]
        df_r = pd.DataFrame(res_list)
        y_i = (df_r["u_bar"] >= y_cutoff).astype(int)
        sp = spearman_with_ci(df_r["f_bin"].to_numpy(), df_r["u_bar"].to_numpy())
        rows.append({
            "dataset": "hmeq",
            "model": "xgb",
            "lime_num_samples": n_smp,
            "n_instances": int(len(df_r)),
            "mean_noise_floor_q95": float(df_r["q95"].mean()),
            "mean_d_k": float(df_r["d_k"].mean()),
            "mean_u_bar": float(df_r["u_bar"].mean()),
            "y_instab_rate": float(y_i.mean()),
            "auroc_f_bin": float(roc_auc_score(y_i, df_r["f_bin"])) if y_i.nunique() >= 2 else np.nan,
            "spearman_f_bin_ubar": sp["rho"],
        })

    r5_df = pd.DataFrame(rows)
    r5_df.to_csv(results_dir / "r5_lime_samples.csv", index=False)
    return r5_df


def run_r6_r7_r8(
    draws_df: pd.DataFrame,
    inst_primary: pd.DataFrame,
    cfg: dict[str, Any],
    results_dir: Path,
) -> None:
    """Run R6 comparison, R7 subsamples + LODO gate, and R8 threshold t=0.5."""
    # --- R6: Exact TreeSHAP vs PermutationExplainer ---
    r6_p = results_dir / "part2_r6_perm_bridge.parquet"
    if r6_p.exists():
        perm_df = pd.read_parquet(r6_p)
        exact_xgb = inst_primary[(inst_primary["model"] == "xgb") & (inst_primary["explainer"] == "TreeSHAP")]
        merged = exact_xgb.merge(perm_df, on=["dataset", "model", "instance_id"], how="inner")
        r6_rows = []
        for ds, grp in merged.groupby("dataset"):
            sp = spearman_with_ci(grp["u_bar"].to_numpy(), grp["perm_u_bar"].to_numpy())
            r6_rows.append({
                "dataset": ds,
                "n_instances": int(len(grp)),
                "exact_treeshap_mean_dk": float(grp["d_k"].mean()),
                "perm_shap_mean_dk": float(grp["perm_d_k"].mean()),
                "perm_shap_mean_noise_floor": float(grp["perm_noise_floor"].mean()),
                "exact_treeshap_u_bar": float(grp["u_bar"].mean()),
                "perm_shap_u_bar": float(grp["perm_u_bar"].mean()),
                "spearman_ubar_exact_vs_perm": sp["rho"],
            })
        pd.DataFrame(r6_rows).to_csv(results_dir / "r6_shap_estimator_comparison.csv", index=False)

    # --- R7: Subsamples (defaulters vs non-defaulters, per dataset) & LODO F-gate ---
    sub_rows = []
    f_only = inst_primary[inst_primary["explainer"] != "LinearSHAP"].dropna(subset=["y_instab", "score_f", "u_bar"]).copy()
    for (ds, y_cls), grp in f_only.groupby(["dataset", "y_true"]):
        auc_v = float(roc_auc_score(grp["y_instab"], grp["score_f"])) if grp["y_instab"].nunique() >= 2 else np.nan
        sp = spearman_with_ci(grp["score_f"].to_numpy(), grp["u_bar"].to_numpy())
        sub_rows.append({
            "scope": "by_true_class",
            "dataset": ds,
            "group": f"y_true={int(y_cls)}",
            "n_instances": int(len(grp)),
            "mean_u_bar": float(grp["u_bar"].mean()),
            "y_instab_rate": float(grp["y_instab"].mean()),
            "auroc_f": auc_v,
            "spearman_f_ubar": sp["rho"],
        })
    for ds, grp in f_only.groupby("dataset"):
        auc_v = float(roc_auc_score(grp["y_instab"], grp["score_f"])) if grp["y_instab"].nunique() >= 2 else np.nan
        sp = spearman_with_ci(grp["score_f"].to_numpy(), grp["u_bar"].to_numpy())
        sub_rows.append({
            "scope": "per_dataset",
            "dataset": ds,
            "group": "all",
            "n_instances": int(len(grp)),
            "mean_u_bar": float(grp["u_bar"].mean()),
            "y_instab_rate": float(grp["y_instab"].mean()),
            "auroc_f": auc_v,
            "spearman_f_ubar": sp["rho"],
        })
    pd.DataFrame(sub_rows).to_csv(results_dir / "r7_subsamples_and_lodo.csv", index=False)

    # LODO optimal F_tree cutoff (for TreeSHAP on RF/XGB)
    tree_inst = inst_primary[(inst_primary["explainer"] == "TreeSHAP") & (inst_primary["model"].isin(["rf", "xgb"]))].dropna(subset=["y_instab", "f_tree_unw"]).copy()
    lodo_rows = []
    for held_out in sorted(tree_inst["dataset"].unique()):
        dev = tree_inst[tree_inst["dataset"] != held_out]
        test_d = tree_inst[tree_inst["dataset"] == held_out]
        if dev["y_instab"].nunique() < 2 or test_d.empty:
            continue
        fpr, tpr, thresholds = roc_curve(dev["y_instab"].astype(int), dev["f_tree_unw"].astype(float))
        best_idx = int(np.argmax(tpr - fpr))
        cutoff = float(thresholds[best_idx])

        flagged = test_d["f_tree_unw"] >= cutoff
        y_pos = test_d["y_instab"] == 1
        tp = int((flagged & y_pos).sum())
        fp = int((flagged & ~y_pos).sum())
        fn = int((~flagged & y_pos).sum())
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))  # share of unstable notices avoided
        flag_rate = float(flagged.mean())

        # Also evaluate specifically on rejected applicants (y_hat == 1)
        rej = test_d[test_d["y_hat"] == 1]
        rej_flag = rej["f_tree_unw"] >= cutoff
        rej_pos = rej["y_instab"] == 1
        rej_tp = int((rej_flag & rej_pos).sum())
        rej_recall = float(rej_tp / max(1, int(rej_pos.sum())))
        rej_prec = float(rej_tp / max(1, int(rej_flag.sum())))

        lodo_rows.append({
            "held_out_dataset": held_out,
            "dev_cutoff_f_tree": cutoff,
            "n_test": int(len(test_d)),
            "flag_rate": flag_rate,
            "precision": precision,
            "recall_unstable_avoided": recall,
            "rejected_flag_rate": float(rej_flag.mean()) if len(rej) else np.nan,
            "rejected_precision": rej_prec,
            "rejected_recall_unstable_avoided": rej_recall,
        })
    with (results_dir / "r7_lodo_gate.json").open("w", encoding="utf-8") as fh:
        json.dump(lodo_rows, fh, indent=2)

    # --- R8: Threshold t=prior vs t=0.50 ---
    perf_df = pd.read_csv(results_dir / "part2_model_performance.csv")
    thresh_map = perf_df.groupby("dataset")["threshold_prior"].first().to_dict()
    f_lookup = inst_primary.set_index(["dataset", "model", "explainer", "instance_id"])["score_f"].to_dict()
    scales = sorted(draws_df["s"].unique())
    primary_s = max(scales) if 0.05 not in scales else 0.05
    sub_s = draws_df[draws_df["s"] == primary_s].copy()

    r8_rows = []
    for mode_label, t_val in [("t=prior", None), ("t=0.50", 0.50)]:
        df_r = sub_s.copy()
        df_r["t_use"] = df_r["dataset"].map(thresh_map) if t_val is None else float(t_val)
        df_r["yhat_use"] = (df_r["p_x"] >= df_r["t_use"]).astype(int)
        df_r["yprime_use"] = (df_r["p_prime"] >= df_r["t_use"]).astype(int)
        df_r["pres_use"] = (df_r["yprime_use"] == df_r["yhat_use"]) & (df_r["abs_dp"] <= float(cfg["part2"]["tau"]))
        filt_r = df_r[df_r["pres_use"]]
        inst_r = filt_r.groupby(["dataset", "model", "explainer", "instance_id"], as_index=False).agg(
            u_bar=("rc", "mean"), p_x=("p_x", "first"), t_use=("t_use", "first"),
        )
        inst_r["y_instab"] = (inst_r["u_bar"] >= float(cfg["part2"]["instability_cutoff"])).astype(int)
        inst_r["score_f"] = [f_lookup.get((r.dataset, r.model, r.explainer, r.instance_id), np.nan) for r in inst_r.itertuples(index=False)]
        inst_r["b_margin"] = -np.abs(inst_r["p_x"] - inst_r["t_use"])
        inst_f = inst_r[inst_r["explainer"] != "LinearSHAP"].dropna(subset=["score_f", "y_instab"])
        auc_f = float(roc_auc_score(inst_f["y_instab"], inst_f["score_f"])) if inst_f["y_instab"].nunique() >= 2 else np.nan
        auc_m = float(roc_auc_score(inst_f["y_instab"], inst_f["b_margin"])) if inst_f["y_instab"].nunique() >= 2 else np.nan
        r8_rows.append({
            "threshold_mode": mode_label,
            "pass_rate": float(df_r["pres_use"].mean()),
            "mean_u_bar": float(inst_r["u_bar"].mean()),
            "y_instab_rate": float(inst_r["y_instab"].mean()),
            "h5_auroc_f": auc_f,
            "h5_auroc_margin": auc_m,
        })
    pd.DataFrame(r8_rows).to_csv(results_dir / "r8_threshold_sensitivity.csv", index=False)


def run_r9_cost_sensitivity(cfg: dict[str, Any], results_dir: Path) -> pd.DataFrame:
    """R9: Compare Bahnsen PV loan revenue r_i vs fixed-margin r_i = m * A_i (m=0.02)."""
    seed = int(cfg.get("seed", 42))
    clip_pcts = tuple(cfg["data"].get("clip_percentiles", [0.5, 99.5]))
    cost_cfg = cfg["cost_model"]
    margin_m = float(cost_cfg.get("sensitivity_margin_m", 0.02))
    rows = []

    for ds in cfg["part1"]["datasets"]:
        df, cb = load_dataset(ds, cfg)
        tr, te = make_part2_split(df, test_frac=0.30, seed=seed)
        prep = Preprocessor(cb, clip_percentiles=clip_pcts).fit(tr)
        X_tr, y_tr = prep.transform(tr), tr["target"].to_numpy(dtype=int)
        X_te, y_te = prep.transform(te), te["target"].to_numpy(dtype=int)
        a_tr, a_te = prep.extract_exposure(tr), prep.extract_exposure(te)
        prior_pi1 = float(np.mean(y_tr))

        for r_mode, m_arg in [("bahnsen_pv", None), (f"fixed_margin_{margin_m}", margin_m)]:
            c_fn_tr, c_fp_tr = compute_cost_vectors(
                a_tr, a_tr, y_tr, lgd=float(cost_cfg["lgd"]), prior_pi1=prior_pi1,
                int_r_annual=float(cost_cfg["int_r_annual"]),
                int_cf_annual=float(cost_cfg["int_cf_annual"]),
                n_term_months=int(cost_cfg["n_term_months"]),
                margin_m=m_arg,
            )
            c_fn_te, c_fp_te = compute_cost_vectors(
                a_te, a_tr, y_tr, lgd=float(cost_cfg["lgd"]), prior_pi1=prior_pi1,
                int_r_annual=float(cost_cfg["int_r_annual"]),
                int_cf_annual=float(cost_cfg["int_cf_annual"]),
                n_term_months=int(cost_cfg["n_term_months"]),
                margin_m=m_arg,
            )
            for m_name in ("cslogit", "csboost"):
                est = build_estimator(m_name, seed=seed, n_jobs=1)
                fit_estimator(est, m_name, X_tr, y_tr, c_fn=c_fn_tr, c_fp=c_fp_tr)
                p_te = est.predict_proba(X_te)[:, 1]
                ev = evaluate_model(y_te, p_te, c_fn=c_fn_te, c_fp=c_fp_te, prior_pi1=prior_pi1, threshold=0.5)
                rows.append({"dataset": ds, "model": m_name, "r_i_spec": r_mode, **ev})

    r9_df = pd.DataFrame(rows)
    r9_df.to_csv(results_dir / "r9_cost_sensitivity.csv", index=False)
    return r9_df


def main(config_path: Path) -> None:
    t0 = time.time()
    cfg = load_config(config_path)
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    n_jobs = int(cfg.get("n_jobs", -1))
    n_workers = os.cpu_count() or 4 if n_jobs <= 0 else n_jobs

    draws_df = pd.read_parquet(results_dir / "part2_draws.parquet")
    inst_all_s = pd.read_parquet(results_dir / "part2_instance_by_scale.parquet")
    inst_primary = pd.read_parquet(results_dir / "part2_instance_summary.parquet")

    print("=== Stage 04: Running Robustness R1-R9 ===")
    r1_df = run_r1_nested_and_ablation(inst_primary, cfg, results_dir)
    print(f"  [R1] Nested & ablation rows: {len(r1_df)}")
    r2_df = run_r2_tau_sensitivity(draws_df, inst_primary, cfg, results_dir)
    print(f"  [R2] Tau sensitivity rows: {len(r2_df)}")
    r3_df = run_r3_scale_s010(inst_all_s, cfg, n_workers, results_dir)
    print(f"  [R3] Scale sensitivity rows: {len(r3_df)}")
    r4_df = run_r4_outcome_definitions(draws_df, inst_primary, cfg, results_dir)
    print(f"  [R4] Outcome definition rows: {len(r4_df)}")
    r5_df = run_r5_lime_samples(cfg, n_workers, results_dir)
    print(f"  [R5] LIME sample size rows: {len(r5_df)}")
    run_r6_r7_r8(draws_df, inst_primary, cfg, results_dir)
    print("  [R6-R8] Completed.")
    r9_df = run_r9_cost_sensitivity(cfg, results_dir)
    print(f"  [R9] Cost sensitivity rows: {len(r9_df)}")

    print(f"=== Stage 04 Complete in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 04: Robustness R1-R9.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_ROOT / "config.yaml")
    args = parser.parse_args()
    main(args.config)
