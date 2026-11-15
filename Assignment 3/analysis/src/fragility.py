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


def _jump_weights(
    tree_splits: TreeSplits,
    active: np.ndarray,
    cover_weighted: bool,
    jump_mode: str,
    sigma_mode: str,
) -> np.ndarray:
    """Resolve normalized jump weights |Delta_{t,v}| / sigma_pred(t) on active split nodes."""
    j_mode = jump_mode.strip().lower()
    if j_mode == "mean":
        if tree_splits.value_jump is None:
            raise ValueError("TreeSplits must contain value_jump (use extract_tree_splits).")
        jumps = tree_splits.value_jump[active]
    elif j_mode == "range":
        raw_r = tree_splits.range_jump if tree_splits.range_jump is not None else tree_splits.value_jump
        if raw_r is None:
            raise ValueError("TreeSplits must contain range_jump/value_jump (use extract_tree_splits).")
        jumps = raw_r[active]
    else:
        raise ValueError(f"Unknown jump_mode '{jump_mode}'; expected 'mean' or 'range'.")

    s_mode = sigma_mode.strip().lower()
    if s_mode == "tree":
        if tree_splits.tree_pred_std is None:
            raise ValueError("TreeSplits must contain tree_pred_std (use extract_tree_splits).")
        pred_std = np.maximum(tree_splits.tree_pred_std[active], 1e-6)
    elif s_mode == "ensemble":
        if tree_splits.ensemble_pred_std is not None:
            pred_std = np.full(int(np.sum(active)), max(float(tree_splits.ensemble_pred_std), 1e-6), dtype=float)
        elif tree_splits.tree_pred_std is not None:
            mean_sd = max(float(np.mean(tree_splits.tree_pred_std)), 1e-6)
            pred_std = np.full(int(np.sum(active)), mean_sd, dtype=float)
        else:
            raise ValueError("TreeSplits must contain ensemble_pred_std or tree_pred_std.")
    else:
        raise ValueError(f"Unknown sigma_mode '{sigma_mode}'; expected 'tree' or 'ensemble'.")

    weights = jumps / pred_std
    if cover_weighted:
        weights = weights * tree_splits.cover_weight[active]
    return weights


def compute_f_jump(
    x: np.ndarray,
    tree_splits: TreeSplits,
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
    cover_weighted: bool = False,
    jump_mode: str = "mean",
    sigma_mode: str = "tree",
) -> float:
    """Compute Proposition P2 leaf-value-jump fragility score F_jump(x; s).

    F_jump(x; s) = (1 / T) * sum_{t, v} (|Delta_{t,v}| / sigma_pred(t)) * Phi(-|x_j - theta_v| / (s * SD_j))
    """
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
    weights = _jump_weights(tree_splits, active, cover_weighted, jump_mode, sigma_mode)
    return float(np.sum(probs * weights) / float(tree_splits.n_trees))


def compute_f_jump_batch(
    X: np.ndarray,
    tree_splits: TreeSplits,
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
    cover_weighted: bool = False,
    jump_mode: str = "mean",
    sigma_mode: str = "tree",
    chunk_size: int = 64,
) -> np.ndarray:
    """Memory-bounded chunked batch evaluation of F_jump(x; s) over rows of X."""
    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    n = X_arr.shape[0]
    if tree_splits.feature_idx.size == 0 or tree_splits.n_trees == 0 or s <= 0.0:
        return np.zeros(n, dtype=float)

    sds = np.asarray(feature_sds, dtype=float).ravel()
    mask = np.asarray(perturbable_mask, dtype=bool).ravel()
    f_idx = tree_splits.feature_idx
    active = mask[f_idx] & (sds[f_idx] > 0.0)
    if not np.any(active):
        return np.zeros(n, dtype=float)

    f_act = f_idx[active]
    thresh_act = tree_splits.threshold[active]
    sigmas = s * sds[f_act]
    weights = _jump_weights(tree_splits, active, cover_weighted, jump_mode, sigma_mode)
    inv_t = 1.0 / float(tree_splits.n_trees)

    step = max(int(chunk_size), 1)
    out = np.empty(n, dtype=float)
    for start in range(0, n, step):
        end = min(start + step, n)
        z = np.abs(X_arr[start:end, f_act] - thresh_act[None, :]) / sigmas[None, :]
        out[start:end] = (norm.sf(z) @ weights) * inv_t
    return out


def calibrate_fragility_scores(
    scores: np.ndarray | list[float],
    reference_scores: np.ndarray | list[float],
    method: str = "quantile",
) -> np.ndarray:
    """Calibrate raw fragility scores against a training reference cohort.

    Methods:
    - "quantile": Laplace-smoothed empirical CDF rank (1 + count(r_i <= s)) / (n_ref + 1)
      in (0, 1] (invariant to any strictly monotone transformation, including positive rescaling).
    - "zscore": (scores - mean(ref)) / std(ref) (invariant to positive affine rescaling).
    - "rz": robust MAD/IQR z-score (scores - median(ref)) / max(1.4826*MAD, IQR/1.349, 1e-6).
    """
    sc = np.asarray(scores, dtype=float)
    ref = np.asarray(reference_scores, dtype=float).ravel()
    if ref.size == 0:
        raise ValueError("reference_scores must be non-empty.")

    key = method.strip().lower()
    if key == "quantile":
        ref_sorted = np.sort(ref)
        flat = sc.ravel()
        counts = np.searchsorted(ref_sorted, flat, side="right")
        calibrated = (1.0 + counts) / float(ref_sorted.size + 1)
        return calibrated.reshape(sc.shape).astype(float)
    if key == "zscore":
        mu = float(np.mean(ref))
        sd = float(np.std(ref))
        if sd <= 1e-12:
            return np.zeros_like(sc, dtype=float)
        return ((sc - mu) / sd).astype(float)
    if key == "rz":
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        q75, q25 = np.percentile(ref, [75.0, 25.0])
        iqr = float(q75 - q25)
        scale = max(1.4826 * mad, iqr / 1.349, 1e-6)
        return ((sc - med) / scale).astype(float)
    raise ValueError(f"Unknown calibration method '{method}'; expected 'quantile', 'zscore', or 'rz'.")
