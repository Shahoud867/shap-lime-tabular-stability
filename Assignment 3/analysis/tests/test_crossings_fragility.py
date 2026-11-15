"""Tests for crossings, Proposition P1 (0 crossings => TreeSHAP unchanged), and MC match for F(x)."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.crossings import (
    TreeSplits,
    count_bin_crossings,
    count_tree_crossings,
    extract_lime_bin_edges,
    extract_tree_splits,
)
from src.explain import build_lime_explainer, explain_tree_shap
from src.fragility import compute_f_bin, compute_f_tree
from src.perturb import generate_perturbations


def test_handbuilt_single_tree_le_vs_lt_conventions() -> None:
    """Verify sklearn (<=) and xgboost (<) split-crossing conventions including exact boundary."""
    splits_le = TreeSplits(
        feature_idx=np.array([0], dtype=np.int64),
        threshold=np.array([1.0], dtype=float),
        op="<=",
        cover_weight=np.array([1.0], dtype=float),
        n_trees=1,
    )
    splits_lt = TreeSplits(
        feature_idx=np.array([0], dtype=np.int64),
        threshold=np.array([1.0], dtype=float),
        op="<",
        cover_weight=np.array([1.0], dtype=float),
        n_trees=1,
    )
    x_at_boundary = np.array([1.0, 0.0])
    probes = np.array([[0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])

    # For <= : 1.0 is left (<= 1.0). Only 1.1 is right (> 1.0).
    assert np.array_equal(count_tree_crossings(x_at_boundary, probes, splits_le, per_tree=False), [0.0, 0.0, 1.0])
    # For <  : 1.0 is right (>= 1.0). Only 0.9 is left (< 1.0).
    assert np.array_equal(count_tree_crossings(x_at_boundary, probes, splits_lt, per_tree=False), [1.0, 0.0, 0.0])


def test_extract_tree_splits_dt_rf_xgb() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    dt = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, y)
    ts_dt = extract_tree_splits(dt)
    assert ts_dt.op == "<="
    assert ts_dt.n_trees == 1
    assert ts_dt.feature_idx.size >= 1

    xgbm = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=1, eval_metric="logloss").fit(X, y)
    ts_xgb = extract_tree_splits(xgbm)
    assert ts_xgb.op == "<"
    assert ts_xgb.n_trees == 5
    assert ts_xgb.feature_idx.size >= 1


def test_proposition_p1_zero_crossings_treeshap_unchanged() -> None:
    """Proposition P1: 0 split crossings => exact TreeSHAP unchanged over 1,000 random draws."""
    rng = np.random.default_rng(77)
    X = rng.normal(0.0, 1.0, size=(250, 5))
    y = (X[:, 0] - 0.8 * X[:, 1] + 0.4 * X[:, 2] > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=77).fit(X, y)
    xgbm = xgb.XGBClassifier(n_estimators=15, max_depth=3, learning_rate=0.2, random_state=77, eval_metric="logloss").fit(X, y)

    bg = X[:80]
    x0 = X[80]

    for model, out_mode in [(rf, "raw"), (xgbm, "probability")]:
        splits = extract_tree_splits(model)
        # Compute safe open-interval bounds around x0 so 1,000 draws have strictly 0 crossings
        low_bounds = np.full(X.shape[1], -np.inf)
        high_bounds = np.full(X.shape[1], np.inf)
        for j in range(X.shape[1]):
            t_j = splits.threshold[splits.feature_idx == j]
            if t_j.size == 0:
                low_bounds[j] = x0[j] - 0.5
                high_bounds[j] = x0[j] + 0.5
                continue
            below = t_j[t_j < x0[j]]
            above = t_j[t_j > x0[j]]
            lo_t = below.max() if below.size else x0[j] - 1.0
            hi_t = above.min() if above.size else x0[j] + 1.0
            # Shrink inside strict interior away from thresholds
            mid_lo = x0[j] - 0.4 * (x0[j] - lo_t)
            mid_hi = x0[j] + 0.4 * (hi_t - x0[j])
            low_bounds[j] = mid_lo
            high_bounds[j] = mid_hi

        draws = rng.uniform(low_bounds, high_bounds, size=(1000, X.shape[1]))
        crossings = count_tree_crossings(x0, draws, splits, per_tree=False)
        assert np.all(crossings == 0.0)

        phi_0, _ = explain_tree_shap(model, bg, x0.reshape(1, -1), model_output=out_mode)
        phi_draws, _ = explain_tree_shap(model, bg, draws, model_output=out_mode)
        max_diff = float(np.max(np.abs(phi_draws - phi_0)))
        assert max_diff < 1e-10, f"Proposition P1 violated for {type(model).__name__}: max_diff={max_diff}"


def test_monte_carlo_mean_c_matches_f_within_3_se() -> None:
    """Acceptance gate: unfiltered, unclipped MC mean of C matches F(x) within 3 SEs."""
    rng = np.random.default_rng(99)
    X = rng.normal(0.0, 1.0, size=(300, 6))
    y = (X[:, 0] + 0.7 * X[:, 1] - 0.5 * X[:, 2] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=99).fit(X, y)
    xgbm = xgb.XGBClassifier(n_estimators=30, max_depth=4, learning_rate=0.2, random_state=99, eval_metric="logloss").fit(X, y)
    lime_exp = build_lime_explainer(X, discretize_continuous=True, seed=99)

    x0 = X[0]
    sds = np.std(X, axis=0)
    mask = np.array([True, True, True, False, True, False])
    s = 0.25
    n_mc = 15000

    draws = generate_perturbations(x0, s=s, feature_sds=sds, perturbable_mask=mask, n_draws=n_mc, seed=1234, clip=False)

    # Check F_tree on both RF (<=) and XGB (<)
    for model in (rf, xgbm):
        splits = extract_tree_splits(model)
        f_val = compute_f_tree(x0, splits, s=s, feature_sds=sds, perturbable_mask=mask, weighted=False)
        c_mc = count_tree_crossings(x0, draws, splits, per_tree=True)
        mc_mean = float(np.mean(c_mc))
        mc_se = float(np.std(c_mc, ddof=1) / np.sqrt(n_mc))
        assert abs(mc_mean - f_val) <= 3.0 * mc_se + 1e-9, (
            f"F_tree mismatch ({type(model).__name__}): MC={mc_mean:.5f}, F={f_val:.5f}, 3*SE={3*mc_se:.5f}"
        )

    # Check F_bin on LIME discretizer
    bin_edges = extract_lime_bin_edges(lime_exp)
    f_bin = compute_f_bin(x0, bin_edges, s=s, feature_sds=sds, perturbable_mask=mask)
    c_bin_mc = count_bin_crossings(x0, draws, bin_edges)
    mc_mean_b = float(np.mean(c_bin_mc))
    mc_se_b = float(np.std(c_bin_mc, ddof=1) / np.sqrt(n_mc))
    assert abs(mc_mean_b - f_bin) <= 3.0 * mc_se_b + 1e-9, (
        f"F_bin mismatch: MC={mc_mean_b:.5f}, F={f_bin:.5f}, 3*SE={3*mc_se_b:.5f}"
    )
