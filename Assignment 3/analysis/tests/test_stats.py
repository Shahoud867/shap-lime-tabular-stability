"""Synthetic ground-truth DGP tests for src/stats.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stats as st  # noqa: E402


def test_power_analysis_correlation_matches_gpower():
    res = st.power_analysis_correlation(rho=0.20, alpha=0.05, power_target=0.80, two_sided=True)
    assert res["n_fisher_z"] == 194
    assert 192 <= res["n_exact"] <= 195
    assert res["achieved_power_exact"] >= 0.80
    assert st.exact_correlation_power(res["n_exact"] - 1, 0.20, 0.05, True) < 0.80


def test_holm_bonferroni_stepdown_and_nan():
    pvals = [0.005, 0.02, 0.04, np.nan, 0.20]
    res = st.holm_bonferroni(pvals, alpha=0.05)
    # m = 4 finite p-values:
    # sorted finite: 0.005 (*4=0.02), 0.02 (*3=0.06), 0.04 (*2=0.08), 0.20 (*1=0.20)
    assert np.isclose(res["p_adjusted"][0], 0.02)
    assert np.isclose(res["p_adjusted"][1], 0.06)
    assert np.isclose(res["p_adjusted"][2], 0.08)
    assert np.isnan(res["p_adjusted"][3])
    assert np.isclose(res["p_adjusted"][4], 0.20)
    assert res["reject"] == [True, False, False, False, False]
    assert res["m"] == 4


def test_jonckheere_terpstra_recovers_ordered_trend():
    rng = np.random.default_rng(123)
    # Decreasing trend across 6 ordered groups (mimicking CoV dropping as pi rises)
    groups_dec = [rng.normal(loc=5.0 - 0.6 * i, scale=0.8, size=40) for i in range(6)]
    res_dec = st.jonckheere_terpstra_test(groups_dec, alternative="decreasing")
    assert res_dec["z"] < -5.0
    assert res_dec["pvalue"] < 1e-6
    assert res_dec["tau_j"] < -0.3

    # Null groups with ties
    groups_null = [rng.integers(0, 5, size=35).astype(float) for _ in range(5)]
    res_null = st.jonckheere_terpstra_test(groups_null, alternative="two-sided")
    assert res_null["pvalue"] > 0.05


def test_spearman_wilcoxon_and_ks():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=200)
    y = 0.6 * x + rng.normal(0, 0.4, size=200)
    sp = st.spearman_with_ci(x, y)
    assert 0.70 < sp["rho"] < 0.90
    assert sp["ci_low"] < sp["rho"] < sp["ci_high"]
    assert sp["pvalue"] < 1e-10

    a = rng.normal(1.2, 0.5, size=100)
    b = rng.normal(0.5, 0.5, size=100)
    wx = st.paired_wilcoxon_test(a, b, alternative="greater")
    assert wx["pvalue"] < 1e-8
    assert wx["rank_biserial_r"] > 0.6

    wx_zero = st.paired_wilcoxon_test(a, a, alternative="two-sided")
    assert wx_zero["pvalue"] == 1.0
    assert wx_zero["rank_biserial_r"] == 0.0

    ks = st.ks_two_sample_test(a, b)
    assert ks["statistic"] > 0.35
    assert ks["pvalue"] < 1e-5


def test_icc_and_e1_variance_decomposition_recovers_shares():
    rng = np.random.default_rng(2026)
    n_inst, J, K, P = 150, 10, 5, 3
    # True variance components: sigma_refit^2 = 0.16 (sd=0.4), sigma_exp^2 = 0.04 (sd=0.2)
    # True explainer share = 0.04 / (0.16 + 0.04) = 0.20
    refit_eff = rng.normal(0.0, 0.4, size=(n_inst, J, 1, P))
    exp_noise = rng.normal(0.0, 0.2, size=(n_inst, J, K, P))
    vals = refit_eff + exp_noise

    res = st.e1_variance_decomposition(vals, n_boot=500, seed=42)
    assert abs(res["pooled_explainer_share"] - 0.20) < 0.03
    assert res["pooled_explainer_share_ci_low"] <= 0.20 <= res["pooled_explainer_share_ci_high"]
    assert abs(res["pooled_icc"] - 0.80) < 0.03

    # Test icc_one_way on K repeats (sigma_between=1.0, sigma_within=0.25 -> ICC=0.80)
    subj = rng.normal(0.0, 1.0, size=(300, 1))
    rep = subj + rng.normal(0.0, 0.5, size=(300, 10))
    icc = st.icc_one_way(rep)
    assert abs(icc["icc_1"] - 0.80) < 0.05


def test_clopper_pearson_ci_bounds():
    c0 = st.clopper_pearson_ci(0, 100)
    assert c0["ci_low"] == 0.0
    assert 0.02 < c0["ci_high"] < 0.05

    cn = st.clopper_pearson_ci(100, 100)
    assert cn["ci_high"] == 1.0
    assert 0.95 < cn["ci_low"] < 0.98

    prop = st.proportion_nonzero_with_ci([0.0, 0.0, 0.1, 0.2, 0.0], tol=0.0)
    assert prop["k"] == 2
    assert prop["n"] == 5
    assert np.isclose(prop["proportion"], 0.4)


def test_cluster_bootstrap_auroc():
    rng = np.random.default_rng(99)
    n_clusters = 120
    m_per = 15
    clusters = np.repeat(np.arange(n_clusters), m_per)
    u_i = rng.normal(0, 0.8, size=n_clusters)
    u_obs = np.repeat(u_i, m_per)
    score = rng.normal(0, 1, size=n_clusters * m_per)
    logit = -0.3 + 1.8 * score + u_obs
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    res = st.cluster_bootstrap_auroc(y, score, clusters, n_boot=500, seed=42)
    assert 0.78 < res["auroc"] < 0.92
    assert res["ci_low"] < res["auroc"] < res["ci_high"]
    assert res["n_clusters"] == n_clusters


def test_cluster_bootstrap_mediation_recovers_ab():
    rng = np.random.default_rng(314)
    n_clusters = 150
    r_per = 20
    n_obs = n_clusters * r_per
    clusters = np.repeat(np.arange(n_clusters), r_per)
    u_m = np.repeat(rng.normal(0, 0.3, size=n_clusters), r_per)
    u_y = np.repeat(rng.normal(0, 0.3, size=n_clusters), r_per)

    s = rng.choice([0.01, 0.05], size=n_obs)
    # True parameters: a = 40.0, b = 0.25 -> a*b = 10.0; c' = 2.5 -> total = 12.5
    m = 0.5 + 40.0 * s + u_m + rng.normal(0, 0.4, size=n_obs)
    y = 0.1 + 2.5 * s + 0.25 * m + u_y + rng.normal(0, 0.2, size=n_obs)

    res = st.cluster_bootstrap_mediation(s, m, y, clusters, n_boot=500, seed=42)
    assert abs(res["a"] - 40.0) < 3.0
    assert abs(res["b"] - 0.25) < 0.03
    assert abs(res["indirect_ab"] - 10.0) < 1.2
    assert res["indirect_ci_low"] <= 10.0 <= res["indirect_ci_high"]
    assert res["ci_excludes_zero"] is True
    assert abs(res["proportion_mediated"] - (10.0 / 12.5)) < 0.10

    # Null mediation check (b = 0)
    y_null = 0.1 + 0.5 * s + u_y + rng.normal(0, 0.3, size=n_obs)
    res_null = st.cluster_bootstrap_mediation(s, m, y_null, clusters, n_boot=400, seed=42)
    assert res_null["indirect_ci_low"] <= 0.0 <= res_null["indirect_ci_high"]
    assert res_null["ci_excludes_zero"] is False


def test_joint_logistic_cluster_robust_double_dissociation():
    rng = np.random.default_rng(555)
    n_clusters = 150
    r_per = 25
    n_obs = n_clusters * r_per
    clusters = np.repeat(np.arange(n_clusters), r_per)
    u_i = np.repeat(rng.normal(0, 0.5, size=n_clusters), r_per)

    c_tree = rng.poisson(2.0, size=n_obs).astype(float)
    c_bin = rng.poisson(2.0, size=n_obs).astype(float)
    # TreeSHAP pattern: depends on c_tree (+1.1), independent of c_bin (0.0)
    logit = -2.2 + 1.1 * c_tree + 0.0 * c_bin + u_i
    rc = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    X = pd.DataFrame({"C_tree": c_tree, "C_bin": c_bin})
    res = st.joint_logistic_cluster_robust(rc, X, clusters)
    assert res["converged"] is True
    assert abs(res["terms"]["C_tree"]["coef"] - 1.1) < 0.25
    assert res["terms"]["C_tree"]["pvalue"] < 1e-6
    assert abs(res["terms"]["C_bin"]["coef"]) < 0.15
    assert res["terms"]["C_bin"]["pvalue"] > 0.05


def test_delong_and_paired_bootstrap_auroc_diff():
    rng = np.random.default_rng(88)
    n = 400
    y = rng.binomial(1, 0.4, size=n)
    # score_1 stronger than score_2
    s1 = y * 1.4 + rng.normal(0, 1.0, size=n)
    s2 = y * 0.6 + rng.normal(0, 1.0, size=n)

    dl = st.delong_paired_auroc_test(y, s1, s2, alternative="greater")
    pb = st.paired_bootstrap_auroc_diff(y, s1, s2, n_boot=1000, seed=42, alternative="greater")

    assert dl["diff"] > 0.10
    assert dl["pvalue"] < 0.01
    assert abs(dl["diff"] - pb["diff"]) < 1e-9
    assert abs(dl["se_diff"] - pb["se_diff"]) < 0.02


def test_nested_ols_hc3_recovers_incremental_r2():
    rng = np.random.default_rng(101)
    n = 300
    x0 = rng.normal(0, 1, size=n)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    # Heteroscedastic noise
    eps = rng.normal(0, 0.3 + 0.2 * np.abs(x2), size=n)
    y = 1.0 + 0.4 * x0 + 0.5 * x1 + 0.8 * x2 + 1.0 * x3 + eps

    m0 = pd.DataFrame({"x0": x0})
    m1 = pd.DataFrame({"x0": x0, "x1": x1})
    m2 = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    m3 = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})

    res = st.nested_ols_hc3(y, [("M0", m0), ("M1", m1), ("M2", m2), ("M3", m3)], n_boot=400, seed=42)
    models = res["models"]
    r2s = [m["r2"] for m in models]
    assert r2s[0] < r2s[1] < r2s[2] < r2s[3]
    for m in models[1:]:
        assert m["delta_r2"] > 0.03
        assert m["delta_r2_ci_low"] > 0.0
        assert m["wald_pvalue_hc3"] < 0.01
    assert abs(models[3]["coefficients"]["x3"]["coef"] - 1.0) < 0.15


def test_measurement_quality_suite():
    rng = np.random.default_rng(404)
    n_inst, r_per = 100, 10
    n_obs = n_inst * r_per
    clusters = np.repeat(np.arange(n_inst), r_per)

    latent_e = rng.uniform(0, 1, size=n_obs)
    dk = latent_e + rng.normal(0, 0.08, size=n_obs)
    dr = latent_e + rng.normal(0, 0.08, size=n_obs)
    d1 = latent_e + rng.normal(0, 0.08, size=n_obs)
    dp = 0.1 * latent_e + rng.normal(0, 0.5, size=n_obs)

    nf = rng.normal(0, 1.0, size=(n_inst, 1)) + rng.normal(0, 0.2, size=(n_inst, 10))
    mq = st.measurement_quality_suite(nf, dk, dr, d1, dp, clusters, n_boot=300, seed=42)
    assert mq["icc"]["icc_1"] > 0.90
    assert mq["mean_abs_monotrait"] > 0.85
    assert mq["mean_abs_heterotrait"] < 0.25
    assert mq["htmt_ratio"] < 0.85
    assert mq["discriminant_passes_085"] is True


def test_aurc_and_paired_aurc_diff_test():
    rng = np.random.default_rng(515)
    n = 300
    y = rng.binomial(1, 0.35, size=n)
    strong_risk = y * 2.0 + rng.normal(0, 0.6, size=n)
    weak_risk = y * 0.4 + rng.normal(0, 1.0, size=n)
    clusters = np.arange(n)

    # Tie invariance: shuffling rows with constant scores gives identical AURC
    tied = np.zeros(n)
    a1 = st.compute_aurc(y, tied)
    perm = rng.permutation(n)
    a2 = st.compute_aurc(y[perm], tied[perm])
    assert abs(a1["aurc"] - a2["aurc"]) < 1e-12

    strong_res = st.cluster_bootstrap_aurc(y, strong_risk, clusters, n_boot=300, seed=42)
    weak_res = st.cluster_bootstrap_aurc(y, weak_risk, clusters, n_boot=300, seed=42)
    assert strong_res["aurc"] < weak_res["aurc"]
    assert strong_res["e_aurc"] >= 0.0
    assert strong_res["risk_at_cov_30"] <= strong_res["risk_at_cov_90"]

    diff_res = st.paired_aurc_diff_test(
        y, strong_risk, weak_risk, clusters, n_boot=400, seed=42, alternative="less"
    )
    assert diff_res["diff"] < -0.05
    assert diff_res["pvalue"] < 0.01
    assert diff_res["ci_high"] < 0.0
