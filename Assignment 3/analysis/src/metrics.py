"""Explanation change, reason-code, and replication stability metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def top_k_indices(phi: np.ndarray, k: int = 4) -> np.ndarray:
    """Return indices of the top-k features ordered by descending |phi|."""
    arr = np.asarray(phi, dtype=float)
    k_eff = min(max(1, int(k)), arr.shape[0])
    # Stable descending sort by |phi|; ties broken deterministically by index
    order = np.argsort(-np.abs(arr), kind="mergesort")
    return order[:k_eff]


def top_k_set(phi: np.ndarray, k: int = 4) -> set[int]:
    """Return the set of top-k feature indices by |phi|."""
    return set(top_k_indices(phi, k=k).tolist())


def jaccard_distance_topk(phi_a: np.ndarray, phi_b: np.ndarray, k: int = 4) -> float:
    """Compute top-k Jaccard distance D_k = 1 - |T_k(a) & T_k(b)| / |T_k(a) | T_k(b)|."""
    a = np.asarray(phi_a, dtype=float)
    b = np.asarray(phi_b, dtype=float)
    if np.array_equal(a, b):
        return 0.0
    set_a = top_k_set(a, k=k)
    set_b = top_k_set(b, k=k)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(1.0 - len(set_a & set_b) / union)


def spearman_distance(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """Compute D_rho = 1 - Spearman_rho(phi_a, phi_b)."""
    a = np.asarray(phi_a, dtype=float)
    b = np.asarray(phi_b, dtype=float)
    if np.allclose(a, b, atol=1e-15, rtol=0.0):
        return 0.0
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        return 1.0
    return float(1.0 - rho)


def normalized_l1_distance(phi_a: np.ndarray, phi_b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute D_1 = ||phi_b - phi_a||_1 / ||phi_a||_1."""
    a = np.asarray(phi_a, dtype=float)
    b = np.asarray(phi_b, dtype=float)
    diff = float(np.sum(np.abs(b - a)))
    if diff == 0.0:
        return 0.0
    denom = float(np.sum(np.abs(a)))
    if denom <= eps:
        return float(diff / eps)
    return float(diff / denom)


def reason_code_change(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    k: int = 4,
    noise_floor_q: float = 0.0,
    tol: float = 1e-12,
) -> int:
    """Return 1 if top-k Jaccard distance exceeds noise_floor_q, else 0."""
    d_k = jaccard_distance_topk(phi_a, phi_b, k=k)
    return int(d_k > noise_floor_q + tol)


def compute_cov(
    attributions: np.ndarray,
    use_abs: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute instance-level mean Coefficient of Variation across P features (Ballegeer Eq. 6).

    Parameters
    ----------
    attributions:
        Array of shape (J, P) across J refits/runs for one instance.
    use_abs:
        Whether to use |phi| (recommended so signed cancellation does not blow up CoV).
    """
    mat = np.asarray(attributions, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        raise ValueError("attributions must have shape (J, P) with J >= 2.")
    vals = np.abs(mat) if use_abs else mat
    mu = np.mean(vals, axis=0)
    sigma = np.std(vals, axis=0, ddof=1)
    cov_p = np.zeros(vals.shape[1], dtype=float)
    nonzero = (np.abs(mu) > eps) & (sigma > eps)
    cov_p[nonzero] = sigma[nonzero] / np.abs(mu[nonzero])
    return float(np.mean(cov_p))


def compute_sra(
    attributions: np.ndarray,
    depth: int = 10,
    use_abs: bool = True,
) -> float:
    """Compute Sequential Rank Agreement at depth d (Ekstrom et al. 2018; Ballegeer Table 4).

    Parameters
    ----------
    attributions:
        Array of shape (L, P) containing L explanation vectors of length P.
    depth:
        Rank depth cutoff d. S(d) includes all features ranked <= d in at least one list.
    use_abs:
        Rank features by descending |phi| (standard for SHAP/LIME).
    """
    mat = np.asarray(attributions, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        raise ValueError("attributions must have shape (L, P) with L >= 2.")
    vals = np.abs(mat) if use_abs else mat
    L, P = vals.shape
    # Rank 1 = highest importance value
    ranks = np.zeros((L, P), dtype=float)
    for l_idx in range(L):
        ranks[l_idx] = rankdata(-vals[l_idx], method="average")

    # A_hat_L(p) = sample variance of ranks across lists
    a_hat = np.var(ranks, axis=0, ddof=1)

    # S(d): features appearing in top-d of at least one list
    in_sd = np.any(ranks <= depth, axis=0)
    if not np.any(in_sd):
        return 0.0
    return float(np.mean(a_hat[in_sd]))
