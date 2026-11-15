"""Gaussian input perturbations, percentile clipping, and prediction-preserving tau-filter."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_proba_pos(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(Y=1 | X) as a 1D float array for any binary classifier."""
    probs = model.predict_proba(X)
    if probs.ndim == 2:
        return np.asarray(probs[:, 1], dtype=float)
    return np.asarray(probs, dtype=float)


def generate_perturbations(
    x: np.ndarray,
    s: float,
    feature_sds: np.ndarray,
    perturbable_mask: np.ndarray,
    clip_low: np.ndarray | None = None,
    clip_high: np.ndarray | None = None,
    n_draws: int = 25,
    seed: int = 42,
    clip: bool = True,
) -> np.ndarray:
    """Draw Gaussian perturbations around instance x on perturbable features only.

    Non-perturbable features are kept bit-identical to x.
    """
    x_vec = np.asarray(x, dtype=float).ravel()
    sds = np.asarray(feature_sds, dtype=float).ravel()
    mask = np.asarray(perturbable_mask, dtype=bool).ravel()

    rng = np.random.default_rng(seed)
    n_features = x_vec.shape[0]
    noise = np.zeros((n_draws, n_features), dtype=float)

    pert_idx = np.flatnonzero(mask)
    if pert_idx.size > 0:
        sigmas = s * sds[pert_idx]
        noise[:, pert_idx] = rng.normal(loc=0.0, scale=sigmas, size=(n_draws, pert_idx.size))

    x_primes = np.broadcast_to(x_vec, (n_draws, n_features)).copy() + noise

    if clip and clip_low is not None and clip_high is not None and pert_idx.size > 0:
        lo = np.asarray(clip_low, dtype=float).ravel()[pert_idx]
        hi = np.asarray(clip_high, dtype=float).ravel()[pert_idx]
        x_primes[:, pert_idx] = np.clip(x_primes[:, pert_idx], lo, hi)

    return x_primes


def filter_prediction_preserving(
    model: Any,
    x: np.ndarray,
    x_primes: np.ndarray,
    threshold: float,
    tau: float = 0.05,
) -> dict[str, Any]:
    """Evaluate prediction changes and apply the prediction-preserving tau-filter."""
    x_row = np.asarray(x, dtype=float).reshape(1, -1)
    p_x = float(predict_proba_pos(model, x_row)[0])
    y_x = int(p_x >= threshold)

    p_primes = predict_proba_pos(model, x_primes)
    y_primes = (p_primes >= threshold).astype(int)
    abs_dp = np.abs(p_primes - p_x)
    preserved = (y_primes == y_x) & (abs_dp <= tau)

    return {
        "p_x": p_x,
        "y_x": y_x,
        "p_primes": p_primes,
        "y_primes": y_primes,
        "abs_dp": abs_dp,
        "preserved": preserved,
        "pass_rate": float(np.mean(preserved)) if len(preserved) else 0.0,
    }
