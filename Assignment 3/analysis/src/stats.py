"""Statistical inference, power analysis, and psychometric quality suite.

Implements all Part 1 (replication), Part 2 (extension), and Robustness tests
specified in Assignment 3/plan.md (§§6.2–6.8, §7.2):
  - Exact bivariate normal correlation power analysis (G*Power analogue)
  - Jonckheere-Terpstra trend test with tie correction
  - Spearman correlation with Fisher-z CI, paired Wilcoxon with rank-biserial r, KS test
  - Holm-Bonferroni multiplicity correction
  - One-way random-effects ICC(1) and E1 refit-vs-explainer variance decomposition
  - Clopper-Pearson exact binomial CI
  - Fast cluster bootstrap (5,000 resamples) for AUROC and mediation indirect effect (a * b)
  - Joint logistic regression with cluster-robust SEs (H4b)
  - Exact Sun-Xu (2014) DeLong paired AUROC test and paired bootstrap AUROC diff (H5)
  - Nested OLS M0->M3 with HC3 SEs, HC3 Wald F-tests, and bootstrap delta-R^2 CIs (H6/R1)
  - Psychometric measurement-quality analogues (test-retest ICC, convergent, HTMT-style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import integrate, optimize, special, stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Power analysis (G*Power "Exact — Correlation: Bivariate normal model")
# ---------------------------------------------------------------------------

def _exact_corr_pdf(r: np.ndarray | float, rho: float, n: int) -> np.ndarray | float:
    """Exact probability density of Pearson correlation r given population rho and sample size n."""
    r_arr = np.asarray(r, dtype=float)
    one_minus_r2 = np.maximum(1.0 - r_arr * r_arr, 1e-300)
    one_minus_rho2 = 1.0 - rho * rho
    # Use log-gamma to avoid overflow for large n
    log_const = (
        np.log(n - 2)
        + special.gammaln(n - 1)
        - 0.5 * np.log(2.0 * np.pi)
        - special.gammaln(n - 0.5)
    )
    const = np.exp(log_const)
    log_kernel = (
        ((n - 1) / 2.0) * np.log(one_minus_rho2)
        + ((n - 4) / 2.0) * np.log(one_minus_r2)
        - (n - 1.5) * np.log(1.0 - rho * r_arr)
    )
    hyp = special.hyp2f1(0.5, 0.5, n - 0.5, (1.0 + rho * r_arr) / 2.0)
    return const * np.exp(log_kernel) * hyp


def exact_correlation_power(
    n: int,
    rho: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Compute exact power of a Pearson correlation test under bivariate normality."""
    if n <= 3:
        return 0.0
    if rho == 0.0:
        return alpha
    df = n - 2
    tail_alpha = alpha / 2.0 if two_sided else alpha
    t_crit = stats.t.ppf(1.0 - tail_alpha, df=df)
    r_crit = t_crit / np.sqrt(t_crit * t_crit + df)

    if two_sided:
        upper_power, _ = integrate.quad(
            lambda r: float(_exact_corr_pdf(r, rho, n)),
            r_crit,
            1.0,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=100,
        )
        lower_power, _ = integrate.quad(
            lambda r: float(_exact_corr_pdf(r, rho, n)),
            -1.0,
            -r_crit,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=100,
        )
        return float(np.clip(upper_power + lower_power, 0.0, 1.0))

    if rho > 0:
        power, _ = integrate.quad(
            lambda r: float(_exact_corr_pdf(r, rho, n)),
            r_crit,
            1.0,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=100,
        )
    else:
        power, _ = integrate.quad(
            lambda r: float(_exact_corr_pdf(r, rho, n)),
            -1.0,
            -r_crit,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=100,
        )
    return float(np.clip(power, 0.0, 1.0))


def power_analysis_correlation(
    rho: float = 0.20,
    alpha: float = 0.05,
    power_target: float = 0.80,
    two_sided: bool = True,
) -> dict[str, Any]:
    """Compute required sample size n for bivariate normal correlation (G*Power exact match).

    For rho=0.20, alpha=0.05, power=0.80, two-tailed: yields n=193 (exact) / 194 (Fisher-z).
    """
    if not (0.0 < abs(rho) < 1.0):
        raise ValueError(f"rho must be in (-1, 0) U (0, 1); got {rho}")

    z_alpha = stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha))
    z_beta = stats.norm.ppf(power_target)
    fisher_z_rho = np.arctanh(abs(rho))
    n_fisher_float = ((z_alpha + z_beta) / fisher_z_rho) ** 2 + 3.0
    n_fisher = int(np.ceil(n_fisher_float))

    # Binary search around Fisher-z estimate using exact density
    lo = max(4, int(n_fisher * 0.5))
    hi = max(lo + 10, int(n_fisher * 1.5))
    while exact_correlation_power(hi, rho, alpha, two_sided) < power_target:
        hi *= 2

    while lo < hi:
        mid = (lo + hi) // 2
        if exact_correlation_power(mid, rho, alpha, two_sided) >= power_target:
            hi = mid
        else:
            lo = mid + 1
    n_exact = lo
    achieved_power = exact_correlation_power(n_exact, rho, alpha, two_sided)

    return {
        "rho": float(rho),
        "alpha": float(alpha),
        "power_target": float(power_target),
        "two_sided": bool(two_sided),
        "n_exact": int(n_exact),
        "achieved_power_exact": float(achieved_power),
        "n_fisher_z": int(n_fisher),
        "n_fisher_z_continuous": float(n_fisher_float),
    }


# ---------------------------------------------------------------------------
# Multiplicity correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Apply Holm-Bonferroni step-down multiplicity correction.

    NaN p-values are preserved as NaN (with reject=False) and excluded from m.
    """
    p_arr = np.asarray(pvalues, dtype=float)
    n_total = len(p_arr)
    adj = np.full(n_total, np.nan, dtype=float)
    reject = np.zeros(n_total, dtype=bool)

    valid_idx = np.where(np.isfinite(p_arr))[0]
    m = len(valid_idx)
    if m == 0:
        return {"p_adjusted": adj.tolist(), "reject": reject.tolist(), "m": 0, "alpha": alpha}

    order = valid_idx[np.argsort(p_arr[valid_idx])]
    sorted_p = p_arr[order]
    raw_adj = (m - np.arange(m)) * sorted_p
    # Enforce monotonicity
    step_adj = np.maximum.accumulate(np.minimum(raw_adj, 1.0))

    adj[order] = step_adj
    # Step-down rejection stops at first non-rejected ordered hypothesis
    rejected_sorted = np.zeros(m, dtype=bool)
    for idx in range(m):
        if step_adj[idx] <= alpha:
            rejected_sorted[idx] = True
        else:
            break
    reject[order] = rejected_sorted

    return {
        "p_adjusted": adj.tolist(),
        "reject": reject.tolist(),
        "m": int(m),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Part 1 Replication Tests (H1a, H1b, E1)
# ---------------------------------------------------------------------------

def jonckheere_terpstra_test(
    groups: Sequence[Sequence[float]],
    alternative: str = "decreasing",
) -> dict[str, Any]:
    """Tie-corrected Jonckheere-Terpstra ordered trend test across k groups.

    Args:
        groups: Ordered sequence of 1D arrays [G_1, G_2, ..., G_k].
        alternative: "decreasing" (theta_1 >= ... >= theta_k, at least one >),
                     "increasing" (theta_1 <= ... <= theta_k, at least one <),
                     or "two-sided".
    """
    if alternative not in {"decreasing", "increasing", "two-sided"}:
        raise ValueError(f"Unsupported alternative: {alternative}")

    clean_groups = [np.asarray(g, dtype=float) for g in groups]
    clean_groups = [g[np.isfinite(g)] for g in clean_groups]
    k = len(clean_groups)
    if k < 2:
        raise ValueError("Need at least 2 groups for Jonckheere-Terpstra test.")

    ns = np.array([len(g) for g in clean_groups], dtype=np.int64)
    if np.any(ns == 0):
        raise ValueError("All groups must contain at least one finite observation.")

    N = int(np.sum(ns))
    # Compute J = sum_{i < j} U_ij efficiently using sorted group j
    J = 0.0
    max_pairs = 0
    for i in range(k - 1):
        gi = clean_groups[i]
        for j in range(i + 1, k):
            gj = np.sort(clean_groups[j])
            # For each x in gi: count(y > x) + 0.5 * count(y == x)
            left = np.searchsorted(gj, gi, side="left")
            right = np.searchsorted(gj, gi, side="right")
            n_j = len(gj)
            greater = n_j - right
            equal = right - left
            J += float(np.sum(greater + 0.5 * equal))
            max_pairs += len(gi) * n_j

    E_J = (N * N - float(np.sum(ns * ns))) / 4.0

    # Tie-corrected variance (Hollander, Wolfe & Chicken, 2014)
    pooled = np.concatenate(clean_groups)
    _, tie_counts = np.unique(pooled, return_counts=True)
    t_m = tie_counts[tie_counts > 1].astype(np.float64)
    n_f = ns.astype(np.float64)

    term1 = (
        N * (N - 1) * (2 * N + 5)
        - np.sum(n_f * (n_f - 1) * (2 * n_f + 5))
        - np.sum(t_m * (t_m - 1) * (2 * t_m + 5))
    ) / 72.0

    term2 = 0.0
    if N >= 3 and len(t_m) > 0:
        term2 = (
            np.sum(n_f * (n_f - 1) * (n_f - 2))
            * np.sum(t_m * (t_m - 1) * (t_m - 2))
        ) / (36.0 * N * (N - 1) * (N - 2))

    term3 = 0.0
    if N >= 2 and len(t_m) > 0:
        term3 = (
            np.sum(n_f * (n_f - 1)) * np.sum(t_m * (t_m - 1))
        ) / (8.0 * N * (N - 1))

    var_J = float(term1 + term2 + term3)
    if var_J <= 0.0:
        z = 0.0
        p = 1.0
    else:
        z = float((J - E_J) / np.sqrt(var_J))
        if alternative == "increasing":
            p = float(stats.norm.sf(z))
        elif alternative == "decreasing":
            p = float(stats.norm.cdf(z))
        else:
            p = float(2.0 * stats.norm.sf(abs(z)))

    tau_j = float((2.0 * J - max_pairs) / max_pairs) if max_pairs > 0 else np.nan
    return {
        "J": float(J),
        "E_J": float(E_J),
        "var_J": float(var_J),
        "z": float(z),
        "pvalue": float(p),
        "tau_j": float(tau_j),
        "n_total": int(N),
        "k_groups": int(k),
        "alternative": alternative,
    }


def spearman_with_ci(
    x: Sequence[float],
    y: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Compute Spearman rank correlation with Bonett-Wright (2000) Fisher-z CI."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_c, y_c = x_arr[mask], y_arr[mask]
    n = len(x_c)
    if n < 4 or np.std(x_c) == 0.0 or np.std(y_c) == 0.0:
        return {
            "rho": np.nan,
            "pvalue": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(n),
        }

    res = stats.spearmanr(x_c, y_c, alternative=alternative)
    rho = float(res.statistic)
    pval = float(res.pvalue)

    # Bonett & Wright (2000) variance in Fisher-z space
    rho_clip = float(np.clip(rho, -0.999999, 0.999999))
    z_r = np.arctanh(rho_clip)
    se_z = np.sqrt((1.0 + 0.5 * rho_clip * rho_clip) / (n - 3.0))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = float(np.tanh(z_r - z_crit * se_z))
    ci_high = float(np.tanh(z_r + z_crit * se_z))
    return {
        "rho": rho,
        "pvalue": pval,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(n),
    }


def paired_wilcoxon_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test (x vs y) with matched-pairs rank-biserial r.

    Tests difference d = x - y. Under alternative='greater', tests median(x - y) > 0.
    rank_biserial_r = (W+ - W-) / (W+ + W-) in [-1, 1], positive when x > y.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    diff = x_arr[mask] - y_arr[mask]
    n_pairs = len(diff)
    nonzero = diff[diff != 0.0]
    n_nz = len(nonzero)

    if n_nz == 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "rank_biserial_r": 0.0,
            "z_over_sqrt_n": 0.0,
            "median_diff": 0.0,
            "mean_diff": 0.0,
            "n_pairs": int(n_pairs),
            "n_nonzero": 0,
            "alternative": alternative,
        }

    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(np.sum(ranks[nonzero > 0.0]))
    w_minus = float(np.sum(ranks[nonzero < 0.0]))
    total_w = w_plus + w_minus
    rb_r = float((w_plus - w_minus) / total_w) if total_w > 0 else 0.0

    # Scipy wilcoxon with normal approx when n_nz >= 25 or ties present
    res = stats.wilcoxon(nonzero, alternative=alternative)
    pval = float(res.pvalue)
    stat = float(res.statistic)

    # Approximate z / sqrt(n) for standardized comparison
    two_sided_p = float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
    z_abs = stats.norm.isf(np.clip(two_sided_p / 2.0, 1e-300, 0.5))
    z_signed = z_abs * (1.0 if w_plus >= w_minus else -1.0)
    r_z = float(z_signed / np.sqrt(n_nz))

    return {
        "statistic": stat,
        "pvalue": pval,
        "rank_biserial_r": rb_r,
        "z_over_sqrt_n": r_z,
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "n_pairs": int(n_pairs),
        "n_nonzero": int(n_nz),
        "alternative": alternative,
    }


def ks_two_sample_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_c = x_arr[np.isfinite(x_arr)]
    y_c = y_arr[np.isfinite(y_arr)]
    if len(x_c) == 0 or len(y_c) == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "n_x": len(x_c), "n_y": len(y_c)}
    res = stats.ks_2samp(x_c, y_c, alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_x": int(len(x_c)),
        "n_y": int(len(y_c)),
    }


def icc_one_way(data_2d: np.ndarray) -> dict[str, float]:
    """One-way random-effects ICC(1,1) and ICC(1,K) across K repeated measures.

    Args:
        data_2d: Array of shape (n_subjects, K_repeats).
    """
    arr = np.asarray(data_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n, K); got shape {arr.shape}")
    # Drop rows with any NaN
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    n, k = arr.shape
    if n < 2 or k < 2:
        return {
            "icc_1": np.nan,
            "icc_k": np.nan,
            "msb": np.nan,
            "msw": np.nan,
            "sigma2_between": np.nan,
            "sigma2_within": np.nan,
            "n_subjects": int(n),
            "k_repeats": int(k),
        }

    grand_mean = float(np.mean(arr))
    row_means = np.mean(arr, axis=1)
    ss_between = float(k * np.sum((row_means - grand_mean) ** 2))
    ss_within = float(np.sum((arr - row_means[:, None]) ** 2))

    msb = ss_between / (n - 1)
    msw = ss_within / (n * (k - 1))

    sigma2_within = msw
    sigma2_between = max(0.0, (msb - msw) / k)

    denom_1 = msb + (k - 1) * msw
    if denom_1 <= 0.0:
        # Completely constant across subjects and repeats -> perfect deterministic repeatability
        icc_1 = 1.0 if msw == 0.0 else 0.0
        icc_k = 1.0 if msw == 0.0 else 0.0
    else:
        icc_1 = float(np.clip((msb - msw) / denom_1, -1.0 / (k - 1), 1.0))
        icc_k = float(np.clip((msb - msw) / msb, 0.0, 1.0)) if msb > 0 else 0.0

    return {
        "icc_1": float(icc_1),
        "icc_k": float(icc_k),
        "msb": float(msb),
        "msw": float(msw),
        "sigma2_between": float(sigma2_between),
        "sigma2_within": float(sigma2_within),
        "n_subjects": int(n),
        "k_repeats": int(k),
    }


def e1_variance_decomposition(
    values_3d: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Decompose across-refit attribution variance into between-refit and explainer parts (E1).

    Args:
        values_3d: Array of shape (n_instances, J_refits, K_repeats), or
                   (n_instances, J_refits, K_repeats, P_features) — if 4D, variance components
                   are summed across features p=1..P per instance before taking shares.
    """
    arr = np.asarray(values_3d, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., None]
    elif arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D array; got shape {arr.shape}")

    n_inst, J, K, P = arr.shape
    if J < 2 or K < 2 or n_inst < 2:
        raise ValueError("Need n_instances >= 2, J_refits >= 2, and K_repeats >= 2.")

    # Compute per-instance, per-feature variance components vectorized
    # refit_means: (n_inst, J, P)
    refit_means = np.mean(arr, axis=2)
    inst_means = np.mean(refit_means, axis=1)  # (n_inst, P)

    ssb = K * np.sum((refit_means - inst_means[:, None, :]) ** 2, axis=1)  # (n_inst, P)
    ssw = np.sum((arr - refit_means[:, :, None, :]) ** 2, axis=(1, 2))  # (n_inst, P)

    msb = ssb / (J - 1)
    msw = ssw / (J * (K - 1))

    sig2_exp_feat = msw
    sig2_ref_feat = np.maximum(0.0, (msb - msw) / K)

    # Total across features per instance
    sig2_exp_inst = np.sum(sig2_exp_feat, axis=1)
    sig2_ref_inst = np.sum(sig2_ref_feat, axis=1)
    sig2_tot_inst = sig2_exp_inst + sig2_ref_inst

    valid = sig2_tot_inst > 0.0
    inst_shares = np.where(valid, sig2_exp_inst / np.maximum(sig2_tot_inst, 1e-300), np.nan)
    inst_iccs = 1.0 - inst_shares

    # Pooled share = sum(sig2_exp) / sum(sig2_tot)
    total_exp = float(np.sum(sig2_exp_inst))
    total_ref = float(np.sum(sig2_ref_inst))
    total_var = total_exp + total_ref
    pooled_share = float(total_exp / total_var) if total_var > 0 else 0.0
    pooled_icc = float(1.0 - pooled_share)

    valid_shares = inst_shares[np.isfinite(inst_shares)]
    mean_share = float(np.mean(valid_shares)) if len(valid_shares) > 0 else np.nan
    median_share = float(np.median(valid_shares)) if len(valid_shares) > 0 else np.nan

    # Instance-level bootstrap CI
    rng = np.random.default_rng(seed)
    boot_pooled = np.empty(n_boot, dtype=float)
    boot_mean = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_inst, size=n_inst)
        b_exp = np.sum(sig2_exp_inst[idx])
        b_tot = b_exp + np.sum(sig2_ref_inst[idx])
        boot_pooled[b] = b_exp / b_tot if b_tot > 0 else 0.0
        b_sh = inst_shares[idx]
        b_sh = b_sh[np.isfinite(b_sh)]
        boot_mean[b] = np.mean(b_sh) if len(b_sh) > 0 else np.nan

    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        "pooled_explainer_share": pooled_share,
        "pooled_explainer_share_ci_low": float(np.nanpercentile(boot_pooled, q_lo)),
        "pooled_explainer_share_ci_high": float(np.nanpercentile(boot_pooled, q_hi)),
        "pooled_icc": pooled_icc,
        "mean_instance_explainer_share": mean_share,
        "mean_instance_share_ci_low": float(np.nanpercentile(boot_mean, q_lo)),
        "mean_instance_share_ci_high": float(np.nanpercentile(boot_mean, q_hi)),
        "median_instance_explainer_share": median_share,
        "sigma2_explainer_total": total_exp,
        "sigma2_refit_total": total_ref,
        "n_instances": int(n_inst),
        "n_valid_instances": int(len(valid_shares)),
        "J_refits": int(J),
        "K_repeats": int(K),
        "P_features": int(P),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Part 2 Extension Tests (H2, H3, H4a, H4b, H4c, H5, H6)
# ---------------------------------------------------------------------------

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> dict[str, Any]:
    """Exact Clopper-Pearson binomial confidence interval for k successes in n trials."""
    if n <= 0:
        return {"k": int(k), "n": int(n), "proportion": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if not (0 <= k <= n):
        raise ValueError(f"k={k} must be in [0, n={n}]")
    prop = float(k / n)
    ci_low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    ci_high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return {
        "k": int(k),
        "n": int(n),
        "proportion": prop,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": float(alpha),
    }


def proportion_nonzero_with_ci(
    values: Sequence[float],
    tol: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute share of finite values strictly greater than tol with Clopper-Pearson CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    k = int(np.sum(arr > tol))
    res = clopper_pearson_ci(k, n, alpha=alpha)
    res["tol"] = float(tol)
    return res


def _fast_auroc(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Fast vectorized AUROC supporting optional non-negative sample weights."""
    if weights is None:
        pos = y_true == 1
        n_pos = int(np.sum(pos))
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        order = np.argsort(y_score, kind="mergesort")
        s_sorted = y_score[order]
        if np.any(np.diff(s_sorted) == 0):
            ranks = stats.rankdata(y_score, method="average")
            sum_ranks_pos = float(np.sum(ranks[pos]))
        else:
            ranks = np.empty(len(y_true), dtype=float)
            ranks[order] = np.arange(1, len(y_true) + 1, dtype=float)
            sum_ranks_pos = float(np.sum(ranks[pos]))
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    desc = np.argsort(y_score, kind="mergesort")[::-1]
    s_s = y_score[desc]
    y_s = y_true[desc]
    w_s = weights[desc]
    tps = np.cumsum(w_s * (y_s == 1))
    fps = np.cumsum(w_s * (y_s != 1))
    distinct = np.flatnonzero(np.diff(s_s))
    thresh_idx = np.r_[distinct, len(y_s) - 1]
    tps = tps[thresh_idx]
    fps = fps[thresh_idx]
    if tps[-1] <= 0.0 or fps[-1] <= 0.0:
        return np.nan
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(tpr, fpr))


def _group_indices(cluster_ids: Sequence[Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map arbitrary cluster IDs to integer codes and list of row index arrays."""
    codes, uniques = pd.factorize(np.asarray(cluster_ids), sort=False)
    n_clusters = len(uniques)
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    splits = np.flatnonzero(np.diff(sorted_codes)) + 1
    groups = np.split(order, splits) if len(order) > 0 else []
    return uniques, groups


def cluster_bootstrap_auroc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    cluster_ids: Sequence[Any],
    weights: Sequence[float] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute AUROC and cluster-bootstrap CI (resampling clusters/instances)."""
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    c_arr = np.asarray(cluster_ids)
    w_arr = None if weights is None else np.asarray(weights, dtype=float)

    mask = np.isfinite(s_arr) & np.isfinite(y_arr)
    if w_arr is not None:
        mask &= np.isfinite(w_arr)
        w_arr = w_arr[mask]
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    c_arr = c_arr[mask]

    point_auc = _fast_auroc(y_arr, s_arr, w_arr)
    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    if n_clusters < 2 or not np.isfinite(point_auc):
        return {
            "auroc": float(point_auc) if np.isfinite(point_auc) else np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": int(len(y_arr)),
            "n_clusters": int(n_clusters),
            "n_valid_boot": 0,
        }

    rng = np.random.default_rng(seed)
    boot_aucs = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[c] for c in chosen])
        yb = y_arr[idx]
        sb = s_arr[idx]
        wb = None if w_arr is None else w_arr[idx]
        boot_aucs[b] = _fast_auroc(yb, sb, wb)

    valid_boot = boot_aucs[np.isfinite(boot_aucs)]
    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        "auroc": float(point_auc),
        "se": float(np.std(valid_boot, ddof=1)) if len(valid_boot) > 1 else np.nan,
        "ci_low": float(np.percentile(valid_boot, q_lo)) if len(valid_boot) > 0 else np.nan,
        "ci_high": float(np.percentile(valid_boot, q_hi)) if len(valid_boot) > 0 else np.nan,
        "n_obs": int(len(y_arr)),
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "n_valid_boot": int(len(valid_boot)),
        "seed": int(seed),
    }


def cluster_bootstrap_mediation(
    treatment: Sequence[float],
    mediator: Sequence[float],
    outcome: Sequence[float],
    cluster_ids: Sequence[Any],
    controls: np.ndarray | pd.DataFrame | None = None,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Mediation analysis with 5,000-resample cluster bootstrap for indirect effect a * b.

    Equations (plan.md §6.4):
      M = a0 + a * X + gamma_a' Z + u
      Y = b0 + c' * X + b * M + gamma_b' Z + v
    Uses pre-computed per-cluster cross-product matrices for fast exact OLS in bootstrap.
    """
    x_arr = np.asarray(treatment, dtype=float)
    m_arr = np.asarray(mediator, dtype=float)
    y_arr = np.asarray(outcome, dtype=float)
    c_arr = np.asarray(cluster_ids)

    mask = np.isfinite(x_arr) & np.isfinite(m_arr) & np.isfinite(y_arr)
    if controls is not None:
        z_arr = np.asarray(controls, dtype=float)
        if z_arr.ndim == 1:
            z_arr = z_arr[:, None]
        mask &= np.all(np.isfinite(z_arr), axis=1)
        z_arr = z_arr[mask]
    else:
        z_arr = None

    x_arr = x_arr[mask]
    m_arr = m_arr[mask]
    y_arr = y_arr[mask]
    c_arr = c_arr[mask]
    n_obs = len(x_arr)

    # Design matrix for M: [1, X, (Z)]
    # Design matrix for Y: [1, X, M, (Z)]
    ones = np.ones((n_obs, 1), dtype=float)
    if z_arr is not None and z_arr.shape[1] > 0:
        X_med = np.hstack([ones, x_arr[:, None], z_arr])
        X_out = np.hstack([ones, x_arr[:, None], m_arr[:, None], z_arr])
    else:
        X_med = np.hstack([ones, x_arr[:, None]])
        X_out = np.hstack([ones, x_arr[:, None], m_arr[:, None]])

    # Point estimates via lstsq
    beta_med, *_ = np.linalg.lstsq(X_med, m_arr, rcond=None)
    beta_out, *_ = np.linalg.lstsq(X_out, y_arr, rcond=None)
    a_hat = float(beta_med[1])
    c_prime_hat = float(beta_out[1])
    b_hat = float(beta_out[2])
    ab_hat = float(a_hat * b_hat)
    total_hat = float(c_prime_hat + ab_hat)
    prop_med = float(ab_hat / total_hat) if abs(total_hat) > 1e-12 else np.nan

    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    if n_clusters < 2:
        return {
            "a": a_hat,
            "b": b_hat,
            "c_prime": c_prime_hat,
            "indirect_ab": ab_hat,
            "total_c": total_hat,
            "proportion_mediated": prop_med,
            "n_obs": int(n_obs),
            "n_clusters": int(n_clusters),
        }

    # Precompute sufficient statistics per cluster: (XtX_m, Xtm, XtX_y, Xty)
    p_m = X_med.shape[1]
    p_y = X_out.shape[1]
    XtX_m_grp = np.empty((n_clusters, p_m, p_m), dtype=float)
    Xtm_grp = np.empty((n_clusters, p_m), dtype=float)
    XtX_y_grp = np.empty((n_clusters, p_y, p_y), dtype=float)
    Xty_grp = np.empty((n_clusters, p_y), dtype=float)

    for g_idx, rows in enumerate(groups):
        Xm_g = X_med[rows]
        m_g = m_arr[rows]
        Xy_g = X_out[rows]
        y_g = y_arr[rows]
        XtX_m_grp[g_idx] = Xm_g.T @ Xm_g
        Xtm_grp[g_idx] = Xm_g.T @ m_g
        XtX_y_grp[g_idx] = Xy_g.T @ Xy_g
        Xty_grp[g_idx] = Xy_g.T @ y_g

    rng = np.random.default_rng(seed)
    boot_a = np.full(n_boot, np.nan, dtype=float)
    boot_b = np.full(n_boot, np.nan, dtype=float)
    boot_cp = np.full(n_boot, np.nan, dtype=float)
    boot_ab = np.full(n_boot, np.nan, dtype=float)
    boot_tot = np.full(n_boot, np.nan, dtype=float)

    # Process in chunks to vectorize np.linalg.solve over bootstrap draws
    chunk_size = 500
    for start in range(0, n_boot, chunk_size):
        end = min(start + chunk_size, n_boot)
        B = end - start
        # Draw cluster multiplicities using multinomial or integer indexing
        chosen = rng.integers(0, n_clusters, size=(B, n_clusters))
        # Count occurrences per cluster per draw: (B, n_clusters)
        counts = np.zeros((B, n_clusters), dtype=float)
        for bi in range(B):
            counts[bi] = np.bincount(chosen[bi], minlength=n_clusters)

        # Sum sufficient stats: (B, p, p) and (B, p)
        S_Xm = np.einsum("bc,cij->bij", counts, XtX_m_grp)
        S_tm = np.einsum("bc,ci->bi", counts, Xtm_grp)
        S_Xy = np.einsum("bc,cij->bij", counts, XtX_y_grp)
        S_ty = np.einsum("bc,ci->bi", counts, Xty_grp)

        try:
            bm = np.linalg.solve(S_Xm, S_tm[..., None])[..., 0]
            by = np.linalg.solve(S_Xy, S_ty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            bm = np.full((B, p_m), np.nan)
            by = np.full((B, p_y), np.nan)
            for bi in range(B):
                try:
                    bm[bi] = np.linalg.lstsq(S_Xm[bi], S_tm[bi], rcond=None)[0]
                    by[bi] = np.linalg.lstsq(S_Xy[bi], S_ty[bi], rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

        boot_a[start:end] = bm[:, 1]
        boot_cp[start:end] = by[:, 1]
        boot_b[start:end] = by[:, 2]
        boot_ab[start:end] = bm[:, 1] * by[:, 2]
        boot_tot[start:end] = by[:, 1] + bm[:, 1] * by[:, 2]

    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    valid_ab = boot_ab[np.isfinite(boot_ab)]
    ab_ci_low = float(np.percentile(valid_ab, q_lo)) if len(valid_ab) > 0 else np.nan
    ab_ci_high = float(np.percentile(valid_ab, q_hi)) if len(valid_ab) > 0 else np.nan
    # Two-sided bootstrap p-value: 2 * min(P(ab <= 0), P(ab >= 0))
    if len(valid_ab) > 0:
        p_pos = float(np.mean(valid_ab >= 0.0))
        p_neg = float(np.mean(valid_ab <= 0.0))
        ab_p = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    else:
        ab_p = np.nan

    return {
        "a": a_hat,
        "a_se": float(np.nanstd(boot_a, ddof=1)),
        "a_ci_low": float(np.nanpercentile(boot_a, q_lo)),
        "a_ci_high": float(np.nanpercentile(boot_a, q_hi)),
        "b": b_hat,
        "b_se": float(np.nanstd(boot_b, ddof=1)),
        "b_ci_low": float(np.nanpercentile(boot_b, q_lo)),
        "b_ci_high": float(np.nanpercentile(boot_b, q_hi)),
        "c_prime": c_prime_hat,
        "c_prime_se": float(np.nanstd(boot_cp, ddof=1)),
        "c_prime_ci_low": float(np.nanpercentile(boot_cp, q_lo)),
        "c_prime_ci_high": float(np.nanpercentile(boot_cp, q_hi)),
        "indirect_ab": ab_hat,
        "indirect_se": float(np.std(valid_ab, ddof=1)) if len(valid_ab) > 1 else np.nan,
        "indirect_ci_low": ab_ci_low,
        "indirect_ci_high": ab_ci_high,
        "indirect_pvalue_boot": ab_p,
        "ci_excludes_zero": bool(
            np.isfinite(ab_ci_low) and np.isfinite(ab_ci_high) and (ab_ci_low > 0.0 or ab_ci_high < 0.0)
        ),
        "total_c": total_hat,
        "total_ci_low": float(np.nanpercentile(boot_tot, q_lo)),
        "total_ci_high": float(np.nanpercentile(boot_tot, q_hi)),
        "proportion_mediated": prop_med,
        "n_obs": int(n_obs),
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def joint_logistic_cluster_robust(
    y: Sequence[int],
    X: pd.DataFrame | np.ndarray,
    cluster_ids: Sequence[Any],
    feature_names: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit logistic regression with cluster-robust SEs (H4b double dissociation)."""
    y_arr = np.asarray(y, dtype=float)
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if feature_names is None else list(feature_names)
        X_mat = X.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim == 1:
            X_mat = X_mat[:, None]
        names = (
            [f"x{i}" for i in range(X_mat.shape[1])]
            if feature_names is None
            else list(feature_names)
        )

    c_arr = np.asarray(cluster_ids)
    mask = np.isfinite(y_arr) & np.all(np.isfinite(X_mat), axis=1)
    y_arr = y_arr[mask]
    X_mat = X_mat[mask]
    c_arr = c_arr[mask]

    # Identify non-constant columns
    non_const = np.std(X_mat, axis=0) > 1e-12
    kept_names = [n for n, keep in zip(names, non_const) if keep]
    X_kept = X_mat[:, non_const]

    if len(np.unique(y_arr)) < 2 or X_kept.shape[1] == 0:
        return {
            "converged": False,
            "reason": "degenerate outcome or all predictors constant",
            "n_obs": int(len(y_arr)),
            "n_clusters": int(len(np.unique(c_arr))),
            "terms": {},
        }

    X_design = sm.add_constant(X_kept, has_constant="add")
    design_names = ["const"] + kept_names
    cluster_codes, _ = pd.factorize(c_arr)

    try:
        glm = sm.GLM(y_arr, X_design, family=sm.families.Binomial())
        res = glm.fit(cov_type="cluster", cov_kwds={"groups": cluster_codes}, maxiter=200)
        converged = bool(getattr(res, "converged", True))
    except Exception as exc:
        return {
            "converged": False,
            "reason": str(exc),
            "n_obs": int(len(y_arr)),
            "n_clusters": int(len(np.unique(c_arr))),
            "terms": {},
        }

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    terms: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(design_names):
        coef = float(res.params[idx])
        se = float(res.bse[idx])
        z_val = float(coef / se) if se > 0 else np.nan
        pval = float(res.pvalues[idx])
        lo = float(coef - z_crit * se)
        hi = float(coef + z_crit * se)
        # Guard exp overflow
        or_val = float(np.exp(np.clip(coef, -50.0, 50.0)))
        or_lo = float(np.exp(np.clip(lo, -50.0, 50.0)))
        or_hi = float(np.exp(np.clip(hi, -50.0, 50.0)))
        terms[name] = {
            "coef": coef,
            "se": se,
            "z": z_val,
            "pvalue": pval,
            "ci_low": lo,
            "ci_high": hi,
            "odds_ratio": or_val,
            "or_ci_low": or_lo,
            "or_ci_high": or_hi,
        }

    # Include dropped constant features explicitly as NaN
    for name, keep in zip(names, non_const):
        if not keep:
            terms[name] = {
                "coef": np.nan,
                "se": np.nan,
                "z": np.nan,
                "pvalue": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "odds_ratio": np.nan,
                "or_ci_low": np.nan,
                "or_ci_high": np.nan,
            }

    # McFadden pseudo-R2 against null deviance
    null_dev = float(res.null_deviance) if hasattr(res, "null_deviance") else np.nan
    dev = float(res.deviance) if hasattr(res, "deviance") else np.nan
    pseudo_r2 = float(1.0 - dev / null_dev) if (np.isfinite(null_dev) and null_dev > 0) else np.nan

    return {
        "converged": converged,
        "n_obs": int(len(y_arr)),
        "n_clusters": int(len(np.unique(c_arr))),
        "pseudo_r2": pseudo_r2,
        "terms": terms,
    }


def _delong_structural_components(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong structural components V10 and V01."""
    m = len(pos_scores)
    n = len(neg_scores)
    combined = np.concatenate([pos_scores, neg_scores])
    tx = stats.rankdata(pos_scores, method="average")
    ty = stats.rankdata(neg_scores, method="average")
    tz = stats.rankdata(combined, method="average")
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    auc = float(np.mean(v10))
    return auc, v10, v01


def delong_paired_auroc_test(
    y_true: Sequence[int],
    score_1: Sequence[float],
    score_2: Sequence[float],
    alternative: str = "greater",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Exact DeLong test for comparing two correlated AUROCs on the same instances (H5).

    Tests H1: AUC(score_1) > AUC(score_2) when alternative='greater'.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(f"Unsupported alternative: {alternative}")

    y_arr = np.asarray(y_true, dtype=int)
    s1 = np.asarray(score_1, dtype=float)
    s2 = np.asarray(score_2, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(s1) & np.isfinite(s2)
    y_arr, s1, s2 = y_arr[mask], s1[mask], s2[mask]

    pos_idx = y_arr == 1
    neg_idx = y_arr == 0
    m = int(np.sum(pos_idx))
    n = int(np.sum(neg_idx))
    if m < 2 or n < 2:
        return {
            "auc_1": np.nan,
            "auc_2": np.nan,
            "diff": np.nan,
            "se_diff": np.nan,
            "z": np.nan,
            "pvalue": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_pos": m,
            "n_neg": n,
        }

    auc_1, v10_1, v01_1 = _delong_structural_components(s1[pos_idx], s1[neg_idx])
    auc_2, v10_2, v01_2 = _delong_structural_components(s2[pos_idx], s2[neg_idx])

    V10 = np.vstack([v10_1, v10_2])  # (2, m)
    V01 = np.vstack([v01_1, v01_2])  # (2, n)
    S10 = np.cov(V10, ddof=1)
    S01 = np.cov(V01, ddof=1)
    cov = S10 / m + S01 / n

    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    var_diff = max(var_diff, 0.0)
    se_diff = float(np.sqrt(var_diff))
    diff = float(auc_1 - auc_2)

    if se_diff <= 1e-15:
        z = 0.0
        pval = 1.0 if abs(diff) <= 1e-15 else 0.0
    else:
        z = float(diff / se_diff)
        if alternative == "greater":
            pval = float(stats.norm.sf(z))
        elif alternative == "less":
            pval = float(stats.norm.cdf(z))
        else:
            pval = float(2.0 * stats.norm.sf(abs(z)))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "auc_1": float(auc_1),
        "auc_2": float(auc_2),
        "diff": diff,
        "se_diff": se_diff,
        "z": z,
        "pvalue": pval,
        "ci_low": float(diff - z_crit * se_diff),
        "ci_high": float(diff + z_crit * se_diff),
        "n_pos": m,
        "n_neg": n,
        "alternative": alternative,
    }


def paired_bootstrap_auroc_diff(
    y_true: Sequence[int],
    score_1: Sequence[float],
    score_2: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
    alternative: str = "greater",
    alpha: float = 0.05,
    cluster_ids: Sequence[Any] | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Paired bootstrap (optionally clustered & weighted) test for AUROC(score_1) - AUROC(score_2)."""
    y_arr = np.asarray(y_true, dtype=int)
    s1 = np.asarray(score_1, dtype=float)
    s2 = np.asarray(score_2, dtype=float)
    w_arr = None if weights is None else np.asarray(weights, dtype=float)
    c_arr = np.arange(len(y_arr)) if cluster_ids is None else np.asarray(cluster_ids)

    mask = np.isfinite(y_arr) & np.isfinite(s1) & np.isfinite(s2)
    if w_arr is not None:
        mask &= np.isfinite(w_arr)
        w_arr = w_arr[mask]
    y_arr, s1, s2, c_arr = y_arr[mask], s1[mask], s2[mask], c_arr[mask]
    n = len(y_arr)

    auc_1 = _fast_auroc(y_arr, s1, w_arr)
    auc_2 = _fast_auroc(y_arr, s2, w_arr)
    diff = float(auc_1 - auc_2)

    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        if cluster_ids is None:
            idx = rng.integers(0, n, size=n)
        else:
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([groups[c] for c in chosen])
        yb = y_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        wb = None if w_arr is None else w_arr[idx]
        diffs[b] = _fast_auroc(yb, s1[idx], wb) - _fast_auroc(yb, s2[idx], wb)

    valid = diffs[np.isfinite(diffs)]
    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    if len(valid) == 0:
        pval = np.nan
    elif alternative == "greater":
        pval = float(np.mean(valid <= 0.0))
    elif alternative == "less":
        pval = float(np.mean(valid >= 0.0))
    else:
        pval = float(min(1.0, 2.0 * min(np.mean(valid <= 0.0), np.mean(valid >= 0.0))))

    return {
        "auc_1": float(auc_1),
        "auc_2": float(auc_2),
        "diff": diff,
        "se_diff": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
        "pvalue": pval,
        "ci_low": float(np.percentile(valid, q_lo)) if len(valid) > 0 else np.nan,
        "ci_high": float(np.percentile(valid, q_hi)) if len(valid) > 0 else np.nan,
        "n": int(n),
        "n_clusters": int(n_clusters),
        "n_valid_boot": int(len(valid)),
        "alternative": alternative,
        "seed": int(seed),
    }


def nested_ols_hc3(
    y: Sequence[float],
    model_specs: Sequence[tuple[str, pd.DataFrame | np.ndarray]],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit nested OLS models M0->M1->M2->M3 with HC3 SEs, Wald F-tests, and delta-R^2 CIs.

    Args:
        y: Target array (e.g. U_bar_i).
        model_specs: Ordered sequence of (model_name, design_df_or_array) where each
                     subsequent model contains the predictors of the previous plus new ones.
    """
    y_arr = np.asarray(y, dtype=float)
    n_models = len(model_specs)
    if n_models == 0:
        raise ValueError("model_specs cannot be empty.")

    # Align finite rows across y and all models so R^2 comparisons use identical observations
    dfs: list[pd.DataFrame] = []
    names: list[str] = []
    valid_mask = np.isfinite(y_arr)
    for m_name, X_in in model_specs:
        names.append(m_name)
        if isinstance(X_in, pd.DataFrame):
            df_x = X_in.copy()
        else:
            arr_x = np.asarray(X_in, dtype=float)
            if arr_x.ndim == 1:
                arr_x = arr_x[:, None]
            df_x = pd.DataFrame(arr_x, columns=[f"{m_name}_x{i}" for i in range(arr_x.shape[1])])
        valid_mask &= np.all(np.isfinite(df_x.to_numpy(dtype=float)), axis=1)
        dfs.append(df_x)

    y_c = y_arr[valid_mask]
    n_obs = len(y_c)
    if n_obs < 5:
        raise ValueError(f"Insufficient finite observations ({n_obs}) for nested OLS.")

    clean_dfs = [df.loc[valid_mask].reset_index(drop=True) for df in dfs]
    results_list: list[dict[str, Any]] = []
    fitted_models: list[Any] = []
    design_matrices: list[np.ndarray] = []

    for idx, (m_name, df_x) in enumerate(zip(names, clean_dfs)):
        # Drop exact zero-variance columns to keep design well-conditioned
        keep_cols = [c for c in df_x.columns if np.std(df_x[c].to_numpy(dtype=float)) > 1e-12]
        X_mat = sm.add_constant(df_x[keep_cols], has_constant="add")
        fit = sm.OLS(y_c, X_mat).fit(cov_type="HC3")
        fitted_models.append(fit)
        design_matrices.append(X_mat.to_numpy(dtype=float))

        coef_table = {}
        ci_df = fit.conf_int(alpha=alpha)
        for col in X_mat.columns:
            coef_table[str(col)] = {
                "coef": float(fit.params[col]),
                "se_hc3": float(fit.bse[col]),
                "t": float(fit.tvalues[col]),
                "pvalue": float(fit.pvalues[col]),
                "ci_low": float(ci_df.loc[col, 0]),
                "ci_high": float(ci_df.loc[col, 1]),
            }

        entry: dict[str, Any] = {
            "model": m_name,
            "n_obs": int(n_obs),
            "df_model": int(fit.df_model),
            "df_resid": int(fit.df_resid),
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "rmse": float(np.sqrt(fit.mse_resid)),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "f_stat_overall": float(fit.fvalue) if fit.fvalue is not None else np.nan,
            "f_pvalue_overall": float(fit.f_pvalue) if fit.f_pvalue is not None else np.nan,
            "predictors": keep_cols,
            "coefficients": coef_table,
            "delta_r2": np.nan,
            "delta_adj_r2": np.nan,
            "delta_r2_ci_low": np.nan,
            "delta_r2_ci_high": np.nan,
            "wald_f_hc3": np.nan,
            "wald_pvalue_hc3": np.nan,
            "f_stat_homoscedastic": np.nan,
            "f_pvalue_homoscedastic": np.nan,
        }

        if idx > 0:
            prev_fit = fitted_models[idx - 1]
            prev_cols = set(results_list[idx - 1]["predictors"])
            added_cols = [c for c in keep_cols if c not in prev_cols]
            entry["delta_r2"] = float(fit.rsquared - prev_fit.rsquared)
            entry["delta_adj_r2"] = float(fit.rsquared_adj - prev_fit.rsquared_adj)
            entry["added_predictors"] = added_cols

            # HC3-robust Wald F-test for added predictors
            if len(added_cols) > 0:
                all_cols = list(X_mat.columns)
                R = np.zeros((len(added_cols), len(all_cols)), dtype=float)
                for r_i, col in enumerate(added_cols):
                    R[r_i, all_cols.index(col)] = 1.0
                wald = fit.wald_test(R, use_f=True, scalar=True)
                entry["wald_f_hc3"] = float(np.asarray(wald.statistic).squeeze())
                entry["wald_pvalue_hc3"] = float(np.asarray(wald.pvalue).squeeze())

                # Standard homoscedastic F-test if strictly nested (computed from SSR without warning)
                if set(prev_cols).issubset(set(keep_cols)):
                    df_diff = fit.df_model - prev_fit.df_model
                    if df_diff > 0 and fit.df_resid > 0 and fit.ssr > 0:
                        f_val = ((prev_fit.ssr - fit.ssr) / df_diff) / (fit.ssr / fit.df_resid)
                        f_p = float(stats.f.sf(f_val, df_diff, fit.df_resid))
                        entry["f_stat_homoscedastic"] = float(f_val)
                        entry["f_pvalue_homoscedastic"] = f_p

        results_list.append(entry)

    # Bootstrap CIs for R^2 and delta_R^2
    if n_boot > 0 and n_models > 1:
        rng = np.random.default_rng(seed)
        boot_r2 = np.full((n_boot, n_models), np.nan, dtype=float)
        # Precompute y_c variance term inside each bootstrap draw
        for b in range(n_boot):
            idx_b = rng.integers(0, n_obs, size=n_obs)
            yb = y_c[idx_b]
            yb_centered = yb - np.mean(yb)
            tss = float(np.dot(yb_centered, yb_centered))
            if tss <= 1e-15:
                continue
            for m_idx, Xm in enumerate(design_matrices):
                Xb = Xm[idx_b]
                beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                resid = yb - Xb @ beta
                rss = float(np.dot(resid, resid))
                boot_r2[b, m_idx] = 1.0 - rss / tss

        q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
        for m_idx in range(n_models):
            r2_b = boot_r2[:, m_idx]
            r2_b = r2_b[np.isfinite(r2_b)]
            results_list[m_idx]["r2_ci_low"] = float(np.percentile(r2_b, q_lo)) if len(r2_b) > 0 else np.nan
            results_list[m_idx]["r2_ci_high"] = float(np.percentile(r2_b, q_hi)) if len(r2_b) > 0 else np.nan
            if m_idx > 0:
                dr2 = boot_r2[:, m_idx] - boot_r2[:, m_idx - 1]
                dr2 = dr2[np.isfinite(dr2)]
                results_list[m_idx]["delta_r2_ci_low"] = float(np.percentile(dr2, q_lo)) if len(dr2) > 0 else np.nan
                results_list[m_idx]["delta_r2_ci_high"] = float(np.percentile(dr2, q_hi)) if len(dr2) > 0 else np.nan

    return {
        "models": results_list,
        "n_obs": int(n_obs),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Measurement Quality Analogues (§6.6)
# ---------------------------------------------------------------------------

def measurement_quality_suite(
    noise_floor_matrix: np.ndarray | None,
    d_k: Sequence[float],
    d_rho: Sequence[float],
    d_1: Sequence[float],
    abs_delta_p: Sequence[float],
    cluster_ids: Sequence[Any],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute Track-C psychometric analogues for computational stability metrics.

    - Test-retest reliability: ICC(1,1) & ICC(1,K) across K noise-floor repeats.
    - Convergent validity: pairwise Spearman correlations among D_k, D_rho, D_1.
    - Discriminant validity (HTMT-style): heterotrait corr(Delta E, |Delta p|) vs
      monotrait corr(D_k, D_rho, D_1).
    """
    icc_res = icc_one_way(noise_floor_matrix) if noise_floor_matrix is not None else None

    dk = np.asarray(d_k, dtype=float)
    dr = np.asarray(d_rho, dtype=float)
    d1 = np.asarray(d_1, dtype=float)
    dp = np.asarray(abs_delta_p, dtype=float)
    c_arr = np.asarray(cluster_ids)

    mask = np.isfinite(dk) & np.isfinite(dr) & np.isfinite(d1) & np.isfinite(dp)
    dk, dr, d1, dp, c_arr = dk[mask], dr[mask], d1[mask], dp[mask], c_arr[mask]

    def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
            return np.nan
        return float(stats.spearmanr(a, b).statistic)

    mono_pairs = {
        "D_k__D_rho": _safe_spearman(dk, dr),
        "D_k__D_1": _safe_spearman(dk, d1),
        "D_rho__D_1": _safe_spearman(dr, d1),
    }
    hetero_pairs = {
        "D_k__abs_dp": _safe_spearman(dk, dp),
        "D_rho__abs_dp": _safe_spearman(dr, dp),
        "D_1__abs_dp": _safe_spearman(d1, dp),
    }

    mono_vals = [abs(v) for v in mono_pairs.values() if np.isfinite(v)]
    hetero_vals = [abs(v) for v in hetero_pairs.values() if np.isfinite(v)]
    mean_mono = float(np.mean(mono_vals)) if mono_vals else np.nan
    mean_hetero = float(np.mean(hetero_vals)) if hetero_vals else np.nan
    # HTMT-style ratio: heterotrait / sqrt(monotrait_E * 1.0)
    htmt_ratio = (
        float(mean_hetero / np.sqrt(mean_mono))
        if (np.isfinite(mean_hetero) and np.isfinite(mean_mono) and mean_mono > 0)
        else np.nan
    )

    # Cluster bootstrap CIs for convergent/heterotrait correlations and HTMT ratio
    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    boot_htmt = np.full(n_boot, np.nan, dtype=float)
    boot_mono_mean = np.full(n_boot, np.nan, dtype=float)
    boot_hetero_mean = np.full(n_boot, np.nan, dtype=float)

    if n_clusters >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        # Pre-rank not valid per bootstrap, so compute Spearman on bootstrap sample
        for b in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([groups[c] for c in chosen])
            b_dk, b_dr, b_d1, b_dp = dk[idx], dr[idx], d1[idx], dp[idx]
            m_v = [
                abs(_safe_spearman(b_dk, b_dr)),
                abs(_safe_spearman(b_dk, b_d1)),
                abs(_safe_spearman(b_dr, b_d1)),
            ]
            h_v = [
                abs(_safe_spearman(b_dk, b_dp)),
                abs(_safe_spearman(b_dr, b_dp)),
                abs(_safe_spearman(b_d1, b_dp)),
            ]
            m_v = [x for x in m_v if np.isfinite(x)]
            h_v = [x for x in h_v if np.isfinite(x)]
            if m_v and h_v:
                mm = float(np.mean(m_v))
                hh = float(np.mean(h_v))
                boot_mono_mean[b] = mm
                boot_hetero_mean[b] = hh
                if mm > 0:
                    boot_htmt[b] = hh / np.sqrt(mm)

    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    valid_htmt = boot_htmt[np.isfinite(boot_htmt)]
    return {
        "icc": icc_res,
        "monotrait_correlations": mono_pairs,
        "heterotrait_correlations": hetero_pairs,
        "mean_abs_monotrait": mean_mono,
        "mean_abs_monotrait_ci_low": float(np.nanpercentile(boot_mono_mean, q_lo)),
        "mean_abs_monotrait_ci_high": float(np.nanpercentile(boot_mono_mean, q_hi)),
        "mean_abs_heterotrait": mean_hetero,
        "mean_abs_heterotrait_ci_low": float(np.nanpercentile(boot_hetero_mean, q_lo)),
        "mean_abs_heterotrait_ci_high": float(np.nanpercentile(boot_hetero_mean, q_hi)),
        "htmt_ratio": htmt_ratio,
        "htmt_ci_low": float(np.percentile(valid_htmt, q_lo)) if len(valid_htmt) > 0 else np.nan,
        "htmt_ci_high": float(np.percentile(valid_htmt, q_hi)) if len(valid_htmt) > 0 else np.nan,
        "discriminant_passes_085": bool(np.isfinite(htmt_ratio) and htmt_ratio < 0.85),
        "n_obs": int(len(dk)),
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Selective Gating Risk-Coverage Metrics (AURC / E-AURC, Geifman & El-Yaniv)
# ---------------------------------------------------------------------------

def _tie_averaged_sorted_risks(y_true: np.ndarray, risk_score: np.ndarray) -> np.ndarray:
    """Sort observations by ascending risk_score with exact tie averaging inside equal-score blocks."""
    order = np.argsort(risk_score, kind="mergesort")
    s_sorted = risk_score[order]
    y_sorted = y_true[order].astype(float)
    n = len(y_sorted)
    if n == 0:
        return y_sorted
    # Average y inside tied score blocks so AURC is strictly permutation-invariant
    diff_idx = np.flatnonzero(np.diff(s_sorted) != 0)
    if len(diff_idx) == n - 1:
        return y_sorted
    starts = np.empty(len(diff_idx) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = diff_idx + 1
    ends = np.empty(len(starts), dtype=np.intp)
    ends[:-1] = diff_idx + 1
    ends[-1] = n
    counts = ends - starts
    block_means = np.add.reduceat(y_sorted, starts) / counts
    return np.repeat(block_means, counts)


def compute_aurc(
    y_true: Sequence[int] | np.ndarray,
    risk_score: Sequence[float] | np.ndarray,
    coverages: Sequence[float] = (0.9, 0.7, 0.5, 0.3),
) -> dict[str, Any]:
    """Compute Area Under Risk-Coverage Curve (AURC) and Excess-AURC (E-AURC).

    Lower `risk_score` means higher confidence (kept first as coverage increases
    from 1/N to 1.0; highest-risk instances are deferred first).
    """
    y_arr = np.asarray(y_true, dtype=float)
    s_arr = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr, s_arr = y_arr[mask], s_arr[mask]
    n = len(y_arr)
    if n == 0:
        return {"aurc": np.nan, "e_aurc": np.nan, "base_risk": np.nan, "oracle_aurc": np.nan, "n_obs": 0}

    y_sorted = _tie_averaged_sorted_risks(y_arr, s_arr)
    counts = np.arange(1, n + 1, dtype=float)
    cum_risks = np.cumsum(y_sorted) / counts
    aurc = float(np.mean(cum_risks))

    # Empirical oracle sorts ascending by true error y (zeros before ones)
    y_oracle = np.sort(y_arr)
    oracle_aurc = float(np.mean(np.cumsum(y_oracle) / counts))
    e_aurc = float(aurc - oracle_aurc)
    base_risk = float(np.mean(y_arr))

    cov_risks: dict[str, float] = {}
    for c in coverages:
        k = max(1, min(n, int(round(float(c) * n))))
        cov_risks[f"risk_at_cov_{int(round(float(c) * 100))}"] = float(cum_risks[k - 1])

    return {
        "aurc": aurc,
        "e_aurc": e_aurc,
        "base_risk": base_risk,
        "oracle_aurc": oracle_aurc,
        "n_obs": int(n),
        **cov_risks,
    }


def cluster_bootstrap_aurc(
    y_true: Sequence[int],
    risk_score: Sequence[float],
    cluster_ids: Sequence[Any],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
    coverages: Sequence[float] = (0.9, 0.7, 0.5, 0.3),
) -> dict[str, Any]:
    """Compute AURC, E-AURC, selective risks, and cluster-bootstrap 95% CIs."""
    y_arr = np.asarray(y_true, dtype=float)
    s_arr = np.asarray(risk_score, dtype=float)
    c_arr = np.asarray(cluster_ids)
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr, s_arr, c_arr = y_arr[mask], s_arr[mask], c_arr[mask]

    pt = compute_aurc(y_arr, s_arr, coverages=coverages)
    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    boot_aurc = np.full(n_boot, np.nan, dtype=float)
    boot_e_aurc = np.full(n_boot, np.nan, dtype=float)

    if n_clusters >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([groups[c] for c in chosen])
            res_b = compute_aurc(y_arr[idx], s_arr[idx], coverages=())
            boot_aurc[b] = res_b["aurc"]
            boot_e_aurc[b] = res_b["e_aurc"]

    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    valid_a = boot_aurc[np.isfinite(boot_aurc)]
    valid_e = boot_e_aurc[np.isfinite(boot_e_aurc)]
    return {
        **pt,
        "aurc_se": float(np.std(valid_a, ddof=1)) if len(valid_a) > 1 else np.nan,
        "aurc_ci_low": float(np.percentile(valid_a, q_lo)) if len(valid_a) > 0 else np.nan,
        "aurc_ci_high": float(np.percentile(valid_a, q_hi)) if len(valid_a) > 0 else np.nan,
        "e_aurc_ci_low": float(np.percentile(valid_e, q_lo)) if len(valid_e) > 0 else np.nan,
        "e_aurc_ci_high": float(np.percentile(valid_e, q_hi)) if len(valid_e) > 0 else np.nan,
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def paired_aurc_diff_test(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    cluster_ids: Sequence[Any] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
    alternative: str = "less",
) -> dict[str, Any]:
    """Paired cluster-bootstrap test for AURC(score_a) - AURC(score_b).

    Since lower AURC is better selective gating, `alternative='less'` tests
    whether score_a achieves significantly lower AURC than score_b.
    """
    y_arr = np.asarray(y_true, dtype=float)
    a_arr = np.asarray(score_a, dtype=float)
    b_arr = np.asarray(score_b, dtype=float)
    c_arr = np.arange(len(y_arr)) if cluster_ids is None else np.asarray(cluster_ids)

    mask = np.isfinite(y_arr) & np.isfinite(a_arr) & np.isfinite(b_arr)
    y_arr, a_arr, b_arr, c_arr = y_arr[mask], a_arr[mask], b_arr[mask], c_arr[mask]

    aurc_a = compute_aurc(y_arr, a_arr, coverages=())["aurc"]
    aurc_b = compute_aurc(y_arr, b_arr, coverages=())["aurc"]
    diff = float(aurc_a - aurc_b)

    _, groups = _group_indices(c_arr)
    n_clusters = len(groups)
    boot_diffs = np.full(n_boot, np.nan, dtype=float)
    if n_clusters >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([groups[c] for c in chosen])
            da = compute_aurc(y_arr[idx], a_arr[idx], coverages=())["aurc"]
            db = compute_aurc(y_arr[idx], b_arr[idx], coverages=())["aurc"]
            boot_diffs[b] = da - db

    valid = boot_diffs[np.isfinite(boot_diffs)]
    se_diff = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
    z_val = float(diff / se_diff) if (np.isfinite(se_diff) and se_diff > 0) else np.nan

    if len(valid) == 0:
        pval = np.nan
    elif alternative == "less":
        pval = float((np.sum(valid >= 0.0) + 1.0) / (len(valid) + 1.0))
    elif alternative == "greater":
        pval = float((np.sum(valid <= 0.0) + 1.0) / (len(valid) + 1.0))
    else:
        p_pos = (np.sum(valid >= 0.0) + 1.0) / (len(valid) + 1.0)
        p_neg = (np.sum(valid <= 0.0) + 1.0) / (len(valid) + 1.0)
        pval = float(min(1.0, 2.0 * min(p_pos, p_neg)))

    q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        "aurc_a": aurc_a,
        "aurc_b": aurc_b,
        "diff": diff,
        "se_diff": se_diff,
        "z": z_val,
        "pvalue": pval,
        "ci_low": float(np.percentile(valid, q_lo)) if len(valid) > 0 else np.nan,
        "ci_high": float(np.percentile(valid, q_hi)) if len(valid) > 0 else np.nan,
        "n_obs": int(len(y_arr)),
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
