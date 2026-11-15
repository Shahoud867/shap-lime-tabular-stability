"""Closed-form expected-crossing fragility scores F_tree and F_bin plus baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from .crossings import TreeSplits


def compute_f_tree(
    x: np.ndarray,
    tree_splits: TreeSplits,
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
    weighted: bool = False,
) -> float:
    """Compute closed-form expected split crossings per tree under N(0, (s*SD_j)^2)."""
    if tree_splits.feature_idx.size == 0 or tree_splits.n_trees == 0 or s <= 0.0:
        return 0.0

    x_vec = np.asarray(x, dtype=float).ravel()
    sds = np.asarray(feature_sds, dtype=float).ravel()
    mask = np.asarray(perturbable_mask, dtype=bool).ravel()

    f_idx = tree_splits.feature_idx
    active = mask[f_idx] & (sds[f_idx] > 0.0)
    if not np.any(active):
        return 0.0

    f_act = f_idx[active]
    thresh_act = tree_splits.threshold[active]
    sigmas = s * sds[f_act]
    z = np.abs(x_vec[f_act] - thresh_act) / sigmas
    probs = norm.sf(z)

    if weighted:
        probs = probs * tree_splits.cover_weight[active]

    return float(np.sum(probs) / float(tree_splits.n_trees))


def compute_f_bin(
    x: np.ndarray,
    bin_edges: dict[int, np.ndarray],
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
) -> float:
    """Compute closed-form expected LIME bin-edge crossings under N(0, (s*SD_j)^2)."""
    if not bin_edges or s <= 0.0:
        return 0.0

    x_vec = np.asarray(x, dtype=float).ravel()
    sds = np.asarray(feature_sds, dtype=float).ravel()
    mask = np.asarray(perturbable_mask, dtype=bool).ravel()

    total = 0.0
    for feat_idx, edges in bin_edges.items():
        if not mask[feat_idx] or sds[feat_idx] <= 0.0 or edges.size == 0:
            continue
        sigma = s * sds[feat_idx]
        z = np.abs(x_vec[feat_idx] - edges) / sigma
        total += float(np.sum(norm.sf(z)))
    return total


def baseline_margin(p_x: float | np.ndarray, threshold: float) -> float | np.ndarray:
    """Decision-margin fragility baseline -|p(x) - t| (higher = closer to boundary)."""
    if np.ndim(p_x) == 0:
        return float(-abs(float(p_x) - float(threshold)))
    arr = np.asarray(p_x, dtype=float)
    return -np.abs(arr - float(threshold))


def baseline_tree_variance(rf_model: Any, X: np.ndarray) -> np.ndarray:
    """Epistemic tree-variance baseline across Random Forest estimators."""
    if not hasattr(rf_model, "estimators_"):
        raise ValueError("baseline_tree_variance requires an ensemble with .estimators_.")
    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    preds = np.array([est.predict_proba(X_arr)[:, 1] for est in rf_model.estimators_], dtype=float)
    return np.var(preds, axis=0)
