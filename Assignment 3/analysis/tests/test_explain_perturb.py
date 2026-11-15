"""Tests for explainers (determinism, local accuracy) and perturbation contracts."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy.special import logit as scipy_logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explain import (
    build_lime_explainer,
    explain_lime,
    explain_linear_shap,
    explain_permutation_shap,
    explain_tree_shap,
)
from src.perturb import filter_prediction_preserving, generate_perturbations


def _synthetic_models(seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(300, 6))
    y = (0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(0, 0.4, 300) > 0).astype(int)
    lr = LogisticRegression(max_iter=500, random_state=seed).fit(X, y)
    rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=seed).fit(X, y)
    xgbm = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, random_state=seed, eval_metric="logloss").fit(X, y)
    return X, y, lr, rf, xgbm


def test_perturbations_respect_mask_clip_and_tau_filter() -> None:
    X, _, lr, _, _ = _synthetic_models()
    x = X[0]
    sds = np.std(X, axis=0)
    mask = np.array([True, True, False, True, False, False])
    lo = np.percentile(X, 0.5, axis=0)
    hi = np.percentile(X, 99.5, axis=0)

    xp = generate_perturbations(x, s=0.5, feature_sds=sds, perturbable_mask=mask, clip_low=lo, clip_high=hi, n_draws=100, seed=9)
    # Non-perturbable features stay bit-identical
    assert np.array_equal(xp[:, ~mask], np.broadcast_to(x[~mask], (100, int((~mask).sum()))))
    # Perturbable features stay strictly inside clip bounds
    assert np.all(xp[:, mask] >= lo[mask] - 1e-12)
    assert np.all(xp[:, mask] <= hi[mask] + 1e-12)

    filt = filter_prediction_preserving(lr, x, xp, threshold=0.5, tau=0.05)
    assert np.all(filt["abs_dp"][filt["preserved"]] <= 0.05 + 1e-12)
    assert np.all(filt["y_primes"][filt["preserved"]] == filt["y_x"])


def test_exact_explainers_deterministic_and_local_accuracy() -> None:
    X, _, lr, rf, xgbm = _synthetic_models()
    bg = X[:100]
    ex = X[100:115]

    # LinearSHAP
    phi_lr1, ev_lr = explain_linear_shap(lr, bg, ex)
    phi_lr2, _ = explain_linear_shap(lr, bg, ex)
    assert np.array_equal(phi_lr1, phi_lr2)
    margin_lr = lr.decision_function(ex)
    assert np.allclose(phi_lr1.sum(axis=1) + ev_lr, margin_lr, atol=1e-5)

    # TreeSHAP on RF (interventional)
    phi_rf1, ev_rf = explain_tree_shap(rf, bg, ex, model_output="raw")
    phi_rf2, _ = explain_tree_shap(rf, bg, ex, model_output="raw")
    assert np.array_equal(phi_rf1, phi_rf2)
    p_rf = rf.predict_proba(ex)[:, 1]
    assert np.allclose(phi_rf1.sum(axis=1) + ev_rf, p_rf, atol=1e-5)

    # TreeSHAP on XGB (interventional, probability output)
    phi_xgb1, ev_xgb = explain_tree_shap(xgbm, bg, ex, model_output="probability")
    phi_xgb2, _ = explain_tree_shap(xgbm, bg, ex, model_output="probability")
    assert np.array_equal(phi_xgb1, phi_xgb2)
    p_xgb = xgbm.predict_proba(ex)[:, 1]
    assert np.allclose(phi_xgb1.sum(axis=1) + ev_xgb, p_xgb, atol=1e-4)


def test_stochastic_explainers_per_call_seed() -> None:
    X, _, lr, _, _ = _synthetic_models()
    bg = X[:50]
    ex = X[50:53]

    lime_exp = build_lime_explainer(X, discretize_continuous=True, seed=42)
    l1 = explain_lime(lime_exp, lr, ex, num_samples=500, seed=101)
    l2 = explain_lime(lime_exp, lr, ex, num_samples=500, seed=101)
    l3 = explain_lime(lime_exp, lr, ex, num_samples=500, seed=202)
    assert np.allclose(l1, l2)
    assert not np.allclose(l1, l3)

    p1 = explain_permutation_shap(lr, bg, ex, max_evals=100, seed=101)
    p2 = explain_permutation_shap(lr, bg, ex, max_evals=100, seed=101)
    assert np.allclose(p1, p2)
