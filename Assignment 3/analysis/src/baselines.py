"""Faithful SOTA A* baselines for explanation stability, uncertainty gating, and stabilization.

Implements:
- S-LIME (Zhou et al., KDD 2021): CLT adaptive sample-size LARS LIME.
- GPEC (Hill et al., AISTATS 2024): Weighted Exponential Geodesic (WEG) GP boundary uncertainty.
- ST-SHAP (Kelodjou et al., AAAI 2024): Layer-wise coalition-saturated Kernel SHAP.
- Epistemic UQ (Mikriukov et al., 2026): Native RF inter-tree variance / XGB RF-surrogate variance.
- AXOM (Pala & Garcia-Cuesta, 2024): Majority-agreeing tree TreeSHAP aggregation.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .explain import explain_tree_shap
from .fragility import baseline_tree_variance
from .perturb import predict_proba_pos


def _reset_lime_rng(lime_explainer: Any, seed: int) -> None:
    """Seed both LIME explainer and its discretizer deterministically."""
    lime_explainer.random_state = np.random.RandomState(seed)
    if getattr(lime_explainer, "discretizer", None) is not None:
        lime_explainer.discretizer.random_state = np.random.RandomState(seed)


def _draw_lime_neighborhood(
    lime_explainer: Any,
    x_row: np.ndarray,
    n_samples: int,
    seed: int,
    existing: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (or extend) LIME interpretable/inverse perturbation matrices deterministically."""
    _reset_lime_rng(lime_explainer, seed)
    if existing is None:
        return lime_explainer._LimeTabularExplainer__data_inverse(x_row, n_samples)

    prev_data, prev_inv = existing
    n_cur = prev_data.shape[0]
    if n_samples <= n_cur:
        return prev_data[:n_samples], prev_inv[:n_samples]

    n_add = n_samples - n_cur + 1
    new_data, new_inv = lime_explainer._LimeTabularExplainer__data_inverse(x_row, n_add)
    data = np.vstack([prev_data, new_data[1:]])
    inverse = np.vstack([prev_inv, new_inv[1:]])
    return data, inverse


def _lars_next_step_and_clt(
    X_std: np.ndarray,
    r: np.ndarray,
    active: list[int],
    inactive: list[int],
    z_alpha: float,
) -> tuple[int, float, int]:
    """Perform one LARS correlation check and return (best_feat, t_stat, z_score_clt_n_req_factor)."""
    n = X_std.shape[0]
    corrs = np.abs(X_std[:, inactive].T @ r) / float(n)
    order = np.argsort(-corrs, kind="mergesort")
    best_local = int(order[0])
    best_feat = inactive[best_local]
    c1 = float(corrs[best_local])
    if len(order) == 1:
        return best_feat, 1.0, n

    second_feat = inactive[int(order[1])]
    c2 = float(corrs[int(order[1])])

    s1 = float(np.sign(X_std[:, best_feat] @ r)) or 1.0
    s2 = float(np.sign(X_std[:, second_feat] @ r)) or 1.0
    u1 = s1 * r * X_std[:, best_feat]
    u2 = s2 * r * X_std[:, second_feat]

    cov = np.cov(u1, u2, ddof=1)
    var_diff = max(float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]), 1e-18)
    se_delta = math.sqrt(2.0 * var_diff / float(n))
    t_stat = (c1 - c2) - z_alpha * se_delta
    if t_stat >= 0.0:
        return best_feat, t_stat, n

    z_pn = max((c1 - c2) / se_delta, 1e-3)
    n_prime = max(n + 100, int(math.ceil(n * (z_alpha / z_pn) ** 2)))
    return best_feat, t_stat, n_prime


def s_lime_explain(
    lime_explainer: Any,
    model: Any,
    X_explain: np.ndarray,
    n_init: int = 1000,
    n_max: int = 5000,
    alpha: float = 0.05,
    top_k: int = 4,
    seed: int = 42,
    num_features: int | None = None,
    return_n_samples: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run S-LIME (Zhou et al., KDD 2021, Algorithm 3) with CLT adaptive sample sizing.

    For each instance, starts with `n_init` LIME perturbations and evaluates step-by-step
    LARS residual correlations with a one-sided CLT hypothesis test at level `alpha`.
    Whenever the gap between the top-2 candidate correlations lacks statistical
    confidence, augments the perturbation budget (reusing prior samples) up to `n_max`.
    """
    X_arr = np.atleast_2d(np.asarray(X_explain, dtype=float))
    n_inst, n_feat = X_arr.shape
    k_target = min(max(int(top_k), 1), n_feat)
    z_alpha = float(norm.ppf(1.0 - alpha))

    attributions = np.zeros((n_inst, n_feat), dtype=float)
    n_used_arr = np.zeros(n_inst, dtype=np.int64)

    scale = np.where(lime_explainer.scaler.scale_ == 0.0, 1.0, lime_explainer.scaler.scale_)
    mean = lime_explainer.scaler.mean_

    for i in range(n_inst):
        x_row = X_arr[i]
        n_cur = int(n_init)
        neighborhood: tuple[np.ndarray, np.ndarray] | None = None
        y_cache: np.ndarray | None = None
        draw_step = 0

        selected_final: list[int] = []
        weights_final = np.empty(0, dtype=float)
        data_final = np.empty((0, n_feat), dtype=float)
        y_final = np.empty(0, dtype=float)

        while True:
            prev_n = 0 if neighborhood is None else neighborhood[0].shape[0]
            neighborhood = _draw_lime_neighborhood(
                lime_explainer,
                x_row,
                n_cur,
                seed=seed + 10007 * i + draw_step,
                existing=neighborhood,
            )
            draw_step += 1
            data, inverse = neighborhood
            if y_cache is None:
                y_all = predict_proba_pos(model, inverse)
            else:
                y_new = predict_proba_pos(model, inverse[prev_n:])
                y_all = np.concatenate([y_cache, y_new])
            y_cache = y_all

            scaled = (data - mean) / scale
            dists = np.linalg.norm(scaled - scaled[0], axis=1)
            sqrt_w = np.asarray(lime_explainer.base.kernel_fn(dists), dtype=float)
            w_sq = sqrt_w ** 2
            w_sum = max(float(np.sum(w_sq)), 1e-12)
            w_norm = w_sq / w_sum

            # Weighted standardization for LARS correlation testing
            x_wmean = np.sum(w_norm[:, None] * data, axis=0)
            x_centered = data - x_wmean[None, :]
            x_wsd = np.sqrt(np.sum(w_norm[:, None] * (x_centered ** 2), axis=0))
            valid_cols = np.where(x_wsd > 1e-12)[0].tolist()

            X_std = np.zeros_like(data, dtype=float)
            if valid_cols:
                X_std[:, valid_cols] = (
                    x_centered[:, valid_cols] / x_wsd[None, valid_cols]
                ) * np.sqrt(float(n_cur) * w_norm)[:, None]

            y_wmean = float(np.sum(w_norm * y_all))
            r = (y_all - y_wmean) * np.sqrt(float(n_cur) * w_norm)

            active: list[int] = []
            inactive = list(valid_cols)
            restart = False

            max_steps = min(k_target, len(inactive))
            for _ in range(max_steps):
                if not inactive:
                    break
                best_feat, t_stat, n_prime = _lars_next_step_and_clt(
                    X_std, r, active, inactive, z_alpha
                )
                if t_stat < 0.0 and n_cur < n_max:
                    n_cur = min(int(n_max), int(n_prime))
                    restart = True
                    break

                active.append(best_feat)
                inactive.remove(best_feat)
                # OLS orthogonal projection of y onto active standardized covariates
                X_act = X_std[:, active]
                y_std_vec = (y_all - y_wmean) * np.sqrt(float(n_cur) * w_norm)
                beta_act, *_ = np.linalg.lstsq(X_act, y_std_vec, rcond=None)
                r = y_std_vec - X_act @ beta_act

            if not restart:
                selected_final = active
                weights_final = w_sq
                data_final = data
                y_final = y_all
                break

        n_used_arr[i] = n_cur
        # Fit LIME's standard weighted Ridge surrogate on stabilized neighborhood
        if num_features is not None:
            cols = selected_final[: int(num_features)] if selected_final else list(range(n_feat))
        else:
            cols = list(range(n_feat))

        if cols:
            ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=seed + i)
            ridge.fit(data_final[:, cols], y_final, sample_weight=weights_final)
            attributions[i, cols] = ridge.coef_.astype(float)

    if return_n_samples:
        return attributions, n_used_arr
    return attributions


def _sample_boundary_points(
    model: Any,
    X_std: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    threshold: float,
    n_boundary: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Find decision-boundary points F(m) ~= threshold by bisecting opposite-class pairs."""
    X_orig = X_std * sd[None, :] + mu[None, :]
    probs = predict_proba_pos(model, X_orig)
    pos_idx = np.where(probs >= threshold)[0]
    neg_idx = np.where(probs < threshold)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.empty((0, X_std.shape[1]), dtype=float)

    # Pair each positive point with a nearby negative point (and random cross pairs)
    k_pairs = min(int(n_boundary), pos_idx.size * neg_idx.size)
    p_sel = rng.choice(pos_idx, size=k_pairs, replace=(k_pairs > pos_idx.size))
    n_sel = rng.choice(neg_idx, size=k_pairs, replace=(k_pairs > neg_idx.size))

    lo = X_std[n_sel].copy()  # F < threshold
    hi = X_std[p_sel].copy()  # F >= threshold
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        p_mid = predict_proba_pos(model, mid * sd[None, :] + mu[None, :])
        is_pos = p_mid >= threshold
        hi[is_pos] = mid[is_pos]
        lo[~is_pos] = mid[~is_pos]
    return 0.5 * (lo + hi)


def _compute_geodesic_distances(boundary_pts: np.ndarray, n_neighbors_geo: int) -> np.ndarray:
    """Approximate boundary manifold geodesic distances via symmetric k-NN shortest paths."""
    k_pts = boundary_pts.shape[0]
    if k_pts <= 1:
        return np.zeros((k_pts, k_pts), dtype=float)

    k_nn = min(max(int(n_neighbors_geo), 1), k_pts - 1)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean").fit(boundary_pts)
    dists, idxs = nn.kneighbors(boundary_pts)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for a in range(k_pts):
        for r_idx in range(1, k_nn + 1):
            b = int(idxs[a, r_idx])
            d = float(dists[a, r_idx])
            rows.extend([a, b])
            cols.extend([b, a])
            vals.extend([d, d])

    graph = csr_matrix((vals, (rows, cols)), shape=(k_pts, k_pts))
    d_geo = shortest_path(graph, method="D", directed=False)
    euc = cdist(boundary_pts, boundary_pts, metric="euclidean")
    finite = np.isfinite(d_geo)
    if not np.all(finite):
        max_finite = float(np.max(d_geo[finite])) if np.any(finite) else float(np.max(euc))
        penalty = max(2.0 * max_finite, 1.0)
        d_geo = np.where(finite, d_geo, euc + penalty)
    return 0.5 * (d_geo + d_geo.T)


def _weg_basis(X_std: np.ndarray, boundary_pts: np.ndarray, rho: float) -> np.ndarray:
    """Compute normalized boundary-proximity weights W_tilde(x) in R^{N x K}."""
    sq_d = cdist(X_std, boundary_pts, metric="sqeuclidean")
    # Log-sum-exp stabilization for numerical safety far from the manifold
    logits = -rho * sq_d
    logits -= np.max(logits, axis=1, keepdims=True)
    w = np.exp(logits)
    denom = np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
    return w / denom


def gpec_boundary_uncertainty(
    model: Any,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    explainer_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_train_gp: int = 300,
    n_boundary: int = 150,
    n_neighbors_geo: int = 10,
    threshold: float = 0.5,
    noise_var: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """Compute GPEC local GP explanation uncertainty (Hill et al., AISTATS 2024).

    Samples boundary points on F(z) = threshold via bisection, builds a geodesic k-NN
    WEG kernel over the boundary manifold, computes GP posterior variance sigma_post^2(x_*),
    and returns total 95% CI width across features (scaled by empirical explanation
    variance if `explainer_fn` is provided, or unit scale otherwise).
    """
    rng = np.random.default_rng(seed)
    X_tr = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))
    n_ex = X_ex.shape[0]

    mu = np.mean(X_tr, axis=0)
    sd = np.where(np.std(X_tr, axis=0) > 1e-12, np.std(X_tr, axis=0), 1.0)
    X_tr_std = (X_tr - mu[None, :]) / sd[None, :]
    X_ex_std = (X_ex - mu[None, :]) / sd[None, :]

    m_sub = min(int(n_train_gp), X_tr_std.shape[0])
    idx_sub = rng.choice(X_tr_std.shape[0], size=m_sub, replace=False) if X_tr_std.shape[0] > m_sub else np.arange(m_sub)
    X_gp_std = X_tr_std[idx_sub]

    boundary_pts = _sample_boundary_points(
        model, X_gp_std, mu, sd, threshold=threshold, n_boundary=n_boundary, rng=rng
    )
    if boundary_pts.shape[0] < 2:
        return np.zeros(n_ex, dtype=float)

    d_geo = _compute_geodesic_distances(boundary_pts, n_neighbors_geo=n_neighbors_geo)
    pos_geo = d_geo[d_geo > 0.0]
    lam = 1.0 / max(float(np.median(pos_geo)) if pos_geo.size > 0 else 1.0, 1e-6)

    sq_ref_m = cdist(X_gp_std, boundary_pts, metric="sqeuclidean")
    pos_sq = sq_ref_m[sq_ref_m > 0.0]
    rho = 1.0 / max(float(np.median(pos_sq)) if pos_sq.size > 0 else 1.0, 1e-6)

    G = np.exp(-lam * d_geo)
    # Project G onto PSD cone to ensure numerical Cholesky stability
    eigvals, eigvecs = np.linalg.eigh(0.5 * (G + G.T))
    G_psd = (eigvecs * np.maximum(eigvals, 0.0)[None, :]) @ eigvecs.T

    W_tr = _weg_basis(X_gp_std, boundary_pts, rho=rho)
    W_ex = _weg_basis(X_ex_std, boundary_pts, rho=rho)

    K_tr = (W_tr @ G_psd) @ W_tr.T
    K_tr[np.diag_indices_from(K_tr)] += max(float(noise_var), 1e-6)
    c_fac = cho_factor(K_tr, lower=True)

    K_cross = (W_tr @ G_psd) @ W_ex.T  # (M, n_ex)
    K_ex_diag = np.sum((W_ex @ G_psd) * W_ex, axis=1)

    V = cho_solve(c_fac, K_cross)
    post_var = np.maximum(K_ex_diag - np.sum(K_cross * V, axis=0), 0.0)

    # Combine WEG GP variance with boundary proximity factor so points closer to F=threshold
    # carry strictly higher boundary uncertainty when manifold density is comparable
    p_ex = predict_proba_pos(model, X_ex)
    boundary_proximity = np.exp(-2.0 * ((p_ex - float(threshold)) ** 2))
    post_sd = np.sqrt(post_var) * (1.0 + boundary_proximity)

    if explainer_fn is not None:
        E_tr = np.atleast_2d(np.asarray(explainer_fn(X_tr[idx_sub]), dtype=float))
        feat_sd = np.sqrt(np.maximum(np.var(E_tr, axis=0), 1e-8))
        ci_scale = float(np.sum(2.0 * 1.96 * feat_sd))
    else:
        ci_scale = 2.0 * 1.96 * float(X_ex.shape[1])

    return ci_scale * post_sd


def _shapley_kernel_weight(d: int, k: int) -> float:
    """Exact KernelSHAP coalition weight pi_z for subset size k in 1..d-1."""
    return float(d - 1) / float(math.comb(d, k) * k * (d - k))


def _sample_st_shap_coalitions(d: int, budget: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample symmetric layer-wise coalitions (Kelodjou et al., AAAI 2024)."""
    if d <= 1 or budget <= 0:
        return np.empty((0, d), dtype=float), np.empty(0, dtype=float)

    rows: list[np.ndarray] = []
    weights: list[float] = []
    rem = int(budget)

    for layer in range(1, (d // 2) + 1):
        if rem <= 0:
            break
        n_l = math.comb(d, layer)
        is_mid = (d % 2 == 0) and (layer == d // 2)
        layer_size = n_l if is_mid else 2 * n_l
        w_l = _shapley_kernel_weight(d, layer)

        if rem >= layer_size and layer_size <= 20000:
            for comb in itertools.combinations(range(d), layer):
                z = np.zeros(d, dtype=float)
                z[list(comb)] = 1.0
                rows.append(z)
                weights.append(w_l)
                if not is_mid:
                    rows.append(1.0 - z)
                    weights.append(w_l)
            rem -= layer_size
        else:
            take = min(rem, layer_size)
            if layer_size <= 20000:
                all_z: list[np.ndarray] = []
                for comb in itertools.combinations(range(d), layer):
                    z = np.zeros(d, dtype=float)
                    z[list(comb)] = 1.0
                    all_z.append(z)
                    if not is_mid:
                        all_z.append(1.0 - z)
                chosen_idx = rng.choice(len(all_z), size=take, replace=False)
                for idx in chosen_idx:
                    rows.append(all_z[int(idx)])
                    weights.append(w_l)
            else:
                seen: set[tuple[int, ...]] = set()
                while len(seen) < take:
                    flip = False if is_mid else bool(rng.integers(0, 2))
                    comb = tuple(sorted(int(c) for c in rng.choice(d, size=layer, replace=False)))
                    key = (1 if flip else 0, *comb)
                    if key in seen:
                        continue
                    seen.add(key)
                    z = np.zeros(d, dtype=float)
                    z[list(comb)] = 1.0
                    rows.append(1.0 - z if flip else z)
                    weights.append(w_l)
            rem -= take

    return np.asarray(rows, dtype=float), np.asarray(weights, dtype=float)


def st_shap_values(
    model_predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    budget: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Compute ST-SHAP layer-wise stabilized Kernel SHAP (Kelodjou et al., AAAI 2024).

    Returns (phi, base_value) where phi has shape (n_explain, n_features) and
    satisfies local accuracy sum_j phi[i, j] == f(x_i) - base_value.
    """
    bg = np.atleast_2d(np.asarray(X_background, dtype=float))
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))
    n_ex, d = X_ex.shape
    bg_preds = np.asarray(model_predict_fn(bg), dtype=float).ravel()
    base_val = float(np.mean(bg_preds))

    if d == 1:
        fx = np.asarray(model_predict_fn(X_ex), dtype=float).ravel()
        return (fx - base_val)[:, None], base_val

    total_budget = int(budget) if budget is not None else min(2 * d + 2048, max(2 ** d - 2, 2 * d))
    rng = np.random.default_rng(seed)
    Z, w = _sample_st_shap_coalitions(d, total_budget, rng)
    n_coal = Z.shape[0]

    # Reduced linear system enforcing exact sum(phi) = f(x) - base_val
    Z_red = Z[:, :-1] - Z[:, [-1]]
    ZW = Z_red * w[:, None]
    gram = ZW.T @ Z_red + 1e-9 * np.eye(d - 1)

    phi = np.zeros((n_ex, d), dtype=float)
    n_bg = bg.shape[0]
    fx_all = np.asarray(model_predict_fn(X_ex), dtype=float).ravel()

    for i in range(n_ex):
        x_row = X_ex[i]
        # Construct masked background tensor (n_coal * n_bg, d) cleanly in batches
        Z_bool = Z.astype(bool)
        masked = np.repeat(bg[None, :, :], n_coal, axis=0)
        x_broadcast = np.broadcast_to(x_row[None, None, :], (n_coal, n_bg, d))
        mask_3d = np.broadcast_to(Z_bool[:, None, :], (n_coal, n_bg, d))
        masked = np.where(mask_3d, x_broadcast, masked).reshape(n_coal * n_bg, d)

        preds = np.asarray(model_predict_fn(masked), dtype=float).reshape(n_coal, n_bg)
        fz = np.mean(preds, axis=1)

        delta_f = float(fx_all[i]) - base_val
        rhs = ZW.T @ ((fz - base_val) - Z[:, -1] * delta_f)
        phi_red = np.linalg.solve(gram, rhs)
        phi[i, :-1] = phi_red
        phi[i, -1] = delta_f - float(np.sum(phi_red))

    return phi, base_val


def epistemic_uq_score(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_explain: np.ndarray,
    n_sub: int = 5,
    seed: int = 42,
    surr_n_estimators: int = 100,
    surr_max_depth: int = 15,
    threshold: float = 0.5,
    mode: str = "auto",
) -> np.ndarray:
    """Compute Epistemic UQ gating score (Mikriukov et al., 2026).

    - On bagging ensembles (`RF`) with `mode="auto"`: native inter-tree variance.
    - On `XGB` / non-bagging models with `mode="auto"`: lightweight RF surrogate
      trained on `(X_train, y_hat(X_train))` whose inter-tree variance estimates
      epistemic uncertainty.
    - With `mode="bootstrap"`: variance across `n_sub` bootstrap refitted clones.
    """
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))
    req_mode = mode.strip().lower()

    if req_mode == "auto" and hasattr(model, "estimators_"):
        return baseline_tree_variance(model, X_ex)

    X_tr = np.atleast_2d(np.asarray(X_train, dtype=float))
    if req_mode == "bootstrap":
        rng = np.random.default_rng(seed)
        if y_train is None:
            raise ValueError("y_train is required when mode='bootstrap'.")
        y_arr = np.asarray(y_train, dtype=int).ravel()
        preds = np.zeros((int(n_sub), X_ex.shape[0]), dtype=float)
        for m in range(int(n_sub)):
            idx = rng.integers(0, X_tr.shape[0], size=X_tr.shape[0])
            # Guarantee both classes appear in the bootstrap resample
            if np.unique(y_arr[idx]).size < 2:
                idx = np.arange(X_tr.shape[0])
            est_m = clone(model)
            if hasattr(est_m, "random_state"):
                est_m.set_params(random_state=seed + m)
            est_m.fit(X_tr[idx], y_arr[idx])
            preds[m] = predict_proba_pos(est_m, X_ex)
        return np.var(preds, axis=0)

    # Mikriukov et al. (2026) lightweight RF surrogate for non-RF models
    p_tr = predict_proba_pos(model, X_tr)
    y_surr = (p_tr >= float(threshold)).astype(int)
    if np.unique(y_surr).size < 2:
        if y_train is not None and np.unique(y_train).size >= 2:
            y_surr = np.asarray(y_train, dtype=int).ravel()
        else:
            return np.zeros(X_ex.shape[0], dtype=float)

    rf_surr = RandomForestClassifier(
        n_estimators=int(surr_n_estimators),
        max_depth=int(surr_max_depth),
        random_state=seed,
        n_jobs=1,
    )
    rf_surr.fit(X_tr, y_surr)
    return baseline_tree_variance(rf_surr, X_ex)


def axom_treeshap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    threshold: float = 0.5,
    return_agree_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compute AXOM majority-agreeing TreeSHAP attributions (Pala & Garcia-Cuesta, 2024, Alg 1).

    Restricted to bagging tree ensembles with `.estimators_` (Random Forest). For each
    instance x, averages per-tree interventional TreeSHAP exclusively across trees whose
    predicted binary class matches the ensemble's predicted binary class.
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("axom_treeshap_values requires a RandomForest ensemble with .estimators_.")

    X_bg = np.atleast_2d(np.asarray(X_background, dtype=float))
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))
    estimators = list(model.estimators_)
    n_trees = len(estimators)
    n_ex, n_feat = X_ex.shape

    ens_label = (predict_proba_pos(model, X_ex) >= float(threshold)).astype(int)  # (n_ex,)
    tree_labels = np.zeros((n_trees, n_ex), dtype=int)
    per_tree_phi = np.zeros((n_trees, n_ex, n_feat), dtype=float)

    for t_idx, est in enumerate(estimators):
        p_t = est.predict_proba(X_ex)[:, 1]
        tree_labels[t_idx] = (p_t >= float(threshold)).astype(int)
        sv, _ = explain_tree_shap(est, X_bg, X_ex, model_output="raw")
        per_tree_phi[t_idx] = sv

    agree = tree_labels == ens_label[None, :]  # (n_trees, n_ex)
    counts = np.sum(agree, axis=0)
    # Edge-case guard: fall back to all trees if no single tree matches thresholded label
    zero_mask = counts == 0
    if np.any(zero_mask):
        agree[:, zero_mask] = True
        counts[zero_mask] = n_trees

    axom_phi = np.sum(per_tree_phi * agree[:, :, None], axis=0) / counts[:, None].astype(float)
    if return_agree_counts:
        return axom_phi, counts
    return axom_phi
