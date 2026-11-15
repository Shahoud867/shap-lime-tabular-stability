"""Phase 2 A* extension unit tests: multi-domain data, F_jump, F_cal, and SOTA baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.baselines import (
    axom_treeshap_values,
    epistemic_uq_score,
    gpec_boundary_uncertainty,
    s_lime_explain,
    st_shap_values,
)
from src.crossings import count_tree_crossings, extract_tree_splits
from src.data import Preprocessor, load_dataset, make_part2_split
from src.explain import build_lime_explainer, explain_tree_shap
from src.fragility import calibrate_fragility_scores, compute_f_jump, compute_f_jump_batch

ANALYSIS_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# (a) Multi-domain datasets, codebooks, checksums, and non-credit preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["hmeq", "sgcs", "taiwan", "bank", "adult", "heart"])
def test_multidomain_processed_and_codebook_present(dataset_name: str) -> None:
    clean_path = ANALYSIS_DIR / "data" / "processed" / f"{dataset_name}_clean.csv"
    cb_path = ANALYSIS_DIR / "data" / "codebook" / f"{dataset_name}_codebook.csv"
    assert clean_path.exists(), f"Missing {clean_path}"
    assert cb_path.exists(), f"Missing {cb_path}"

    df, cb = load_dataset(dataset_name)
    assert "target" in df.columns
    assert set(np.unique(df["target"].to_numpy())).issubset({0, 1})
    assert "perturbable" in cb.columns


def test_checksums_cover_all_six_datasets() -> None:
    meta_path = ANALYSIS_DIR / "data" / "raw" / "checksums_and_licenses.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in ("hmeq", "sgcs", "taiwan", "bank", "adult", "heart"):
        assert name in meta
        assert len(meta[name]["sha256"]) == 64


@pytest.mark.parametrize("dataset_name", ["bank", "adult", "heart"])
def test_non_credit_preprocessing_without_exposure(dataset_name: str) -> None:
    df, cb = load_dataset(dataset_name)
    train_df, test_df = make_part2_split(df, test_frac=0.30, seed=42)
    prep = Preprocessor(cb).fit(train_df)
    assert prep.exposure_col is None
    X_train = prep.transform(train_df)
    X_test = prep.transform(test_df)
    assert X_train.shape == (len(train_df), len(prep.feature_names_))
    assert X_test.shape == (len(test_df), len(prep.feature_names_))
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_test))


# ---------------------------------------------------------------------------
# (b) Exact Delta_{t,v} and range_jump extraction on RF and XGB trees
# ---------------------------------------------------------------------------


def test_exact_delta_extraction_sklearn_tree() -> None:
    # Deterministic depth-2 dataset so tree has known internal node jumps
    X = np.array(
        [
            [-2.0, 0.0],
            [-2.0, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    dt = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    splits = extract_tree_splits(dt)

    tree = dt.tree_
    internal = tree.feature >= 0
    vals = tree.value[:, 0, :]
    node_p1 = vals[:, 1] / np.sum(vals, axis=1)
    expected_jumps = np.abs(node_p1[tree.children_left[internal]] - node_p1[tree.children_right[internal]])

    assert splits.value_jump is not None
    assert splits.range_jump is not None
    assert np.allclose(splits.value_jump, expected_jumps, atol=1e-12)
    assert np.all(splits.range_jump >= splits.value_jump - 1e-12)


def test_exact_delta_extraction_xgboost_bottom_up() -> None:
    rng = np.random.RandomState(7)
    X = rng.randn(80, 4)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0.0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, learning_rate=0.3, random_state=7).fit(X, y)
    splits = extract_tree_splits(model)

    booster = model.get_booster()
    df = booster.trees_to_dataframe()

    # Independent bottom-up check on tree 0 root node
    t0 = df[df["Tree"] == 0].set_index("ID")

    def _node_val(nid: str) -> float:
        row = t0.loc[nid]
        if row["Feature"] == "Leaf":
            return float(row["Gain"])
        yl, nr = str(row["Yes"]), str(row["No"])
        cl, cr = float(t0.loc[yl, "Cover"]), float(t0.loc[nr, "Cover"])
        return (cl * _node_val(yl) + cr * _node_val(nr)) / (cl + cr)

    root_row = t0.loc["0-0"]
    expected_root_jump = abs(_node_val(str(root_row["Yes"])) - _node_val(str(root_row["No"])))

    assert splits.value_jump is not None
    assert np.isclose(splits.value_jump[0], expected_root_jump, atol=1e-10)
    assert np.all(splits.range_jump >= splits.value_jump - 1e-12)
    assert np.all(splits.tree_pred_std >= 1e-6)


# ---------------------------------------------------------------------------
# (c) F_jump Monte Carlo consistency within 3 SEs & batch consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_kind", ["rf", "xgb"])
def test_f_jump_monte_carlo_and_batch_consistency(model_kind: str) -> None:
    rng = np.random.RandomState(123)
    X = rng.randn(160, 5)
    y = (X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] > 0.0).astype(int)
    if model_kind == "rf":
        model = RandomForestClassifier(n_estimators=15, max_depth=3, random_state=123).fit(X, y)
    else:
        model = xgb.XGBClassifier(n_estimators=12, max_depth=3, learning_rate=0.2, random_state=123).fit(X, y)

    splits = extract_tree_splits(model, X_train=X)
    sds = np.std(X, axis=0)
    mask = np.ones(5, dtype=bool)
    s = 0.20

    x0 = X[0]
    f_exact = compute_f_jump(x0, splits, s=s, feature_sds=sds, perturbable_mask=mask)
    f_batch = compute_f_jump_batch(X[:10], splits, s=s, feature_sds=sds, perturbable_mask=mask, chunk_size=3)
    f_batch_1 = compute_f_jump_batch(X[:10], splits, s=s, feature_sds=sds, perturbable_mask=mask, chunk_size=1)
    assert np.isclose(f_batch[0], f_exact, atol=1e-12)
    assert np.allclose(f_batch, f_batch_1, atol=1e-12)

    # Unclipped, unfiltered MC expectation of weighted crossings must equal F_jump
    n_mc = 12000
    noise = rng.normal(0.0, s * sds, size=(n_mc, 5))
    x_primes = x0[None, :] + noise
    weights = splits.value_jump / np.maximum(splits.tree_pred_std, 1e-6)
    mc_draws = count_tree_crossings(x0, x_primes, splits, per_tree=True, weights=weights)
    mc_mean = float(np.mean(mc_draws))
    mc_se = float(np.std(mc_draws, ddof=1) / np.sqrt(n_mc))
    assert abs(mc_mean - f_exact) <= 3.0 * mc_se + 1e-6


# ---------------------------------------------------------------------------
# (d) Calibration scale invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quantile", "zscore", "rz"])
def test_calibration_scale_invariance(method: str) -> None:
    rng = np.random.default_rng(9)
    ref = rng.gamma(shape=2.0, scale=0.5, size=250)
    scores = rng.gamma(shape=2.0, scale=0.5, size=40)

    c1 = calibrate_fragility_scores(scores, ref, method=method)
    c2 = calibrate_fragility_scores(17.3 * scores, 17.3 * ref, method=method)
    assert np.allclose(c1, c2, atol=1e-12)

    if method == "quantile":
        assert np.all((c1 > 0.0) & (c1 <= 1.0))
        # Monotone ordering preserved
        order_raw = np.argsort(scores)
        assert np.all(np.diff(c1[order_raw]) >= -1e-12)


# ---------------------------------------------------------------------------
# (e) S-LIME CLT convergence and determinism
# ---------------------------------------------------------------------------


def test_s_lime_clt_convergence_and_determinism() -> None:
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=1, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=42).fit(X, y)
    feat_names = [f"f{j}" for j in range(6)]
    expl = build_lime_explainer(X, feat_names, discretize_continuous=True, seed=42)

    phi_a, n_a = s_lime_explain(
        expl, rf, X[:5], n_init=300, n_max=1500, alpha=0.05, top_k=3, seed=42, return_n_samples=True
    )
    phi_b, n_b = s_lime_explain(
        expl, rf, X[:5], n_init=300, n_max=1500, alpha=0.05, top_k=3, seed=42, return_n_samples=True
    )
    assert np.allclose(phi_a, phi_b, atol=1e-12)
    assert np.array_equal(n_a, n_b)
    assert np.all((n_a >= 300) & (n_a <= 1500))

    # Larger sample budget reduces cross-seed variance (CLT 1/sqrt(N) scaling)
    seeds = [10, 11, 12, 13, 14]
    small_runs = np.stack(
        [s_lime_explain(expl, rf, X[:2], n_init=150, n_max=150, top_k=3, seed=s) for s in seeds],
        axis=0,
    )
    large_runs = np.stack(
        [s_lime_explain(expl, rf, X[:2], n_init=1200, n_max=1200, top_k=3, seed=s) for s in seeds],
        axis=0,
    )
    assert float(np.mean(np.var(large_runs, axis=0))) < float(np.mean(np.var(small_runs, axis=0)))


# ---------------------------------------------------------------------------
# (f) GPEC boundary monotonicity
# ---------------------------------------------------------------------------


def test_gpec_boundary_monotonicity() -> None:
    rng = np.random.default_rng(77)
    n = 300
    X = rng.normal(0.0, 1.2, size=(n, 4))
    y = (X[:, 0] > 0.0).astype(int)
    clf = LogisticRegression(C=5.0, random_state=77).fit(X, y)

    # Probe points walking from far confident region (x0=-2.0) to boundary (x0=0.0)
    probes = np.array(
        [
            [-2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    u = gpec_boundary_uncertainty(clf, X, probes, n_train_gp=200, n_boundary=100, seed=77)
    assert u.shape == (4,)
    assert np.all(np.isfinite(u))
    assert u[3] > u[0], f"Expected boundary point uncertainty {u[3]} > deep region {u[0]}"


# ---------------------------------------------------------------------------
# (g) Epistemic UQ gating
# ---------------------------------------------------------------------------


def test_epistemic_uq_positive_on_uncertain_instances() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(0.0, 1.5, size=(240, 4))
    # Uncertain overlap zone around |x0| < 0.5, clean outside
    logits = 2.0 * X[:, 0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    rf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=99).fit(X, y)
    xgb_m = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=99).fit(X, y)

    boundary_pts = np.zeros((10, 4))
    boundary_pts[:, 1:] = rng.normal(0.0, 0.3, size=(10, 3))
    confident_pts = np.zeros((10, 4))
    confident_pts[:, 0] = 3.0

    uq_rf_b = epistemic_uq_score(rf, X, y, boundary_pts)
    uq_rf_c = epistemic_uq_score(rf, X, y, confident_pts)
    assert np.mean(uq_rf_b) > np.mean(uq_rf_c)

    uq_xgb_b = epistemic_uq_score(xgb_m, X, y, boundary_pts, seed=99)
    uq_xgb_c = epistemic_uq_score(xgb_m, X, y, confident_pts, seed=99)
    assert np.all(uq_xgb_b >= 0.0)
    assert np.mean(uq_xgb_b) > np.mean(uq_xgb_c)


# ---------------------------------------------------------------------------
# (h) AXOM majority-agreeing TreeSHAP filtering
# ---------------------------------------------------------------------------


def test_axom_filters_minority_dissenting_trees() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(0.0, 1.0, size=(120, 4))
    y = (X[:, 0] > 0.0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=11).fit(X, y)

    # Corrupt tree 0 so it predicts class 0 everywhere with extreme leaf values
    t0 = rf.estimators_[0].tree_
    t0.value[:] = 0.0
    t0.value[:, 0, 0] = 1.0  # probability 0 for class 1 across all nodes

    x_pos = np.array([[2.0, 0.0, 0.0, 0.0]])
    bg = X[:40]
    axom_phi, agree_cnt = axom_treeshap_values(rf, bg, x_pos, threshold=0.5, return_agree_counts=True)
    full_phi, _ = explain_tree_shap(rf, bg, x_pos, model_output="raw")

    # Tree 0 predicts class 0 while ensemble predicts class 1 => excluded
    ens_cls = int(rf.predict_proba(x_pos)[0, 1] >= 0.5)
    assert ens_cls == 1
    agreeing = [est for est in rf.estimators_ if int(est.predict_proba(x_pos)[0, 1] >= 0.5) == ens_cls]
    assert rf.estimators_[0] not in agreeing
    assert agree_cnt[0] == len(agreeing)

    agree_phis = []
    for est in agreeing:
        sv, _ = explain_tree_shap(est, bg, x_pos, model_output="raw")
        agree_phis.append(sv)
    expected = np.mean(np.stack(agree_phis, axis=0), axis=0)
    assert np.allclose(axom_phi, expected, atol=1e-10)
    assert not np.allclose(axom_phi, full_phi)


# ---------------------------------------------------------------------------
# (i) ST-SHAP layer-wise saturation determinism & local accuracy
# ---------------------------------------------------------------------------


def test_st_shap_local_accuracy_and_layer1_determinism() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(0.0, 1.0, size=(60, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    clf = LogisticRegression().fit(X, y)
    pred_fn = lambda arr: clf.predict_proba(arr)[:, 1]

    bg = X[:20]
    x_ex = X[20:23]
    # Layer 1 budget = 2 * d = 10 is completely deterministic regardless of seed
    phi_1, base_1 = st_shap_values(pred_fn, bg, x_ex, budget=10, seed=1)
    phi_2, base_2 = st_shap_values(pred_fn, bg, x_ex, budget=10, seed=999)
    assert np.allclose(phi_1, phi_2, atol=1e-12)
    assert np.isclose(base_1, base_2, atol=1e-12)

    fx = pred_fn(x_ex)
    assert np.allclose(np.sum(phi_1, axis=1), fx - base_1, atol=1e-10)
