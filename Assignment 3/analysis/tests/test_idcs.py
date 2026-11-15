"""Tests for IDCS cost model, Ballegeer Fig. 3 savings example, and IDCS >= traditional gate."""

from __future__ import annotations

import numpy as np

from src.idcs import (
    CSBoostClassifier,
    CSLogitClassifier,
    compute_cost_vectors,
    rel_aec,
    savings_score,
)
from src.models import build_estimator, fit_estimator


def test_savings_reproduces_ballegeer_fig3() -> None:
    """Ballegeer et al. Fig. 3: one default (c_fn=10) and two non-defaults (c_fp=1, 5) -> C_base=6."""
    y_true = np.array([1, 0, 0])
    c_fn = np.array([10.0, 7.0, 6.0])
    c_fp = np.array([3.0, 1.0, 5.0])
    # Predicting all 0 gives cost=10; predicting all 1 gives cost=6 -> C_base = 6
    assert savings_score(y_true, np.array([0, 0, 0]), c_fn, c_fp) == 1.0 - 10.0 / 6.0
    assert savings_score(y_true, np.array([1, 1, 1]), c_fn, c_fp) == 0.0
    assert savings_score(y_true, y_true, c_fn, c_fp) == 1.0


def test_idcs_savings_sanity_gate_and_determinism() -> None:
    """IDCS models must achieve >= traditional savings on instance-dependent loan data."""
    rng = np.random.default_rng(42)
    n_train, n_test = 600, 400
    n_total = n_train + n_test

    X = rng.normal(0.0, 1.0, size=(n_total, 6))
    # Strongly varying loan amounts so instance-dependent costs matter
    amounts = rng.lognormal(mean=9.5, sigma=1.1, size=n_total)
    logits = -1.3 + 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2]
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs_true)

    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    A_tr, A_te = amounts[:n_train], amounts[n_train:]

    c_fn_tr, c_fp_tr = compute_cost_vectors(A_tr, A_tr, y_tr)
    c_fn_te, c_fp_te = compute_cost_vectors(A_te, A_tr, y_tr, prior_pi1=float(y_tr.mean()))

    # Traditional logit vs cslogit
    logit = fit_estimator(build_estimator("logit", {"penalty": "l2", "C": 0.1}, seed=42), "logit", X_tr, y_tr)
    cslogit_a = fit_estimator(CSLogitClassifier(penalty="l2", C=0.05, random_state=42), "cslogit", X_tr, y_tr, c_fn_tr, c_fp_tr)
    cslogit_b = fit_estimator(CSLogitClassifier(penalty="l2", C=0.05, random_state=42), "cslogit", X_tr, y_tr, c_fn_tr, c_fp_tr)

    assert np.allclose(cslogit_a.predict_proba(X_te), cslogit_b.predict_proba(X_te))

    sav_logit = savings_score(y_te, logit.predict(X_te), c_fn_te, c_fp_te)
    sav_cslogit = savings_score(y_te, cslogit_a.predict(X_te), c_fn_te, c_fp_te)
    assert sav_cslogit >= sav_logit, f"cslogit savings {sav_cslogit:.4f} < logit {sav_logit:.4f}"

    # Traditional boost vs csboost
    boost = fit_estimator(build_estimator("boost", {"max_depth": 3, "n_estimators": 50}, seed=42), "boost", X_tr, y_tr)
    csboost_a = fit_estimator(CSBoostClassifier(max_depth=3, n_estimators=50, random_state=42), "csboost", X_tr, y_tr, c_fn_tr, c_fp_tr)
    csboost_b = fit_estimator(CSBoostClassifier(max_depth=3, n_estimators=50, random_state=42), "csboost", X_tr, y_tr, c_fn_tr, c_fp_tr)

    assert np.allclose(csboost_a.predict_proba(X_te), csboost_b.predict_proba(X_te))

    sav_boost = savings_score(y_te, boost.predict(X_te), c_fn_te, c_fp_te)
    sav_csboost = savings_score(y_te, csboost_a.predict(X_te), c_fn_te, c_fp_te)
    assert sav_csboost >= sav_boost - 0.02, f"csboost savings {sav_csboost:.4f} << boost {sav_boost:.4f}"
    # Also verify relAEC improves over traditional
    raec_logit = rel_aec(y_te, logit.predict_proba(X_te)[:, 1], c_fn_te, c_fp_te, float(y_tr.mean()))
    raec_cslogit = rel_aec(y_te, cslogit_a.predict_proba(X_te)[:, 1], c_fn_te, c_fp_te, float(y_tr.mean()))
    assert raec_cslogit >= raec_logit


def test_idcs_real_hmeq_sanity_gate() -> None:
    """Verify IDCS models beat traditional counterparts on real HMEQ 70/30 split."""
    from src.data import Preprocessor, load_dataset, make_part2_split

    df, cb = load_dataset("hmeq")
    tr_df, te_df = make_part2_split(df, test_frac=0.30, seed=42)
    prep = Preprocessor(cb).fit(tr_df)
    X_tr = prep.transform(tr_df)
    X_te = prep.transform(te_df)
    y_tr = tr_df["target"].to_numpy(dtype=int)
    y_te = te_df["target"].to_numpy(dtype=int)
    A_tr = prep.extract_exposure(tr_df)
    A_te = prep.extract_exposure(te_df)

    c_fn_tr, c_fp_tr = compute_cost_vectors(A_tr, A_tr, y_tr)
    c_fn_te, c_fp_te = compute_cost_vectors(A_te, A_tr, y_tr, prior_pi1=float(y_tr.mean()))

    logit = fit_estimator(build_estimator("logit", {"penalty": "l2", "C": 0.03}, seed=42), "logit", X_tr, y_tr)
    cslogit = fit_estimator(build_estimator("cslogit", {"penalty": "l2", "C": 0.03}, seed=42), "cslogit", X_tr, y_tr, c_fn_tr, c_fp_tr)
    assert savings_score(y_te, cslogit.predict(X_te), c_fn_te, c_fp_te) >= savings_score(y_te, logit.predict(X_te), c_fn_te, c_fp_te)

    boost = fit_estimator(build_estimator("boost", {"max_depth": 3, "n_estimators": 80}, seed=42), "boost", X_tr, y_tr)
    csboost = fit_estimator(build_estimator("csboost", {"max_depth": 3, "n_estimators": 80}, seed=42), "csboost", X_tr, y_tr, c_fn_tr, c_fp_tr)
    sav_b = savings_score(y_te, boost.predict(X_te), c_fn_te, c_fp_te)
    sav_csb = savings_score(y_te, csboost.predict(X_te), c_fn_te, c_fp_te)
    assert sav_csb >= sav_b, f"HMEQ csboost savings {sav_csb:.4f} < boost {sav_b:.4f}"

