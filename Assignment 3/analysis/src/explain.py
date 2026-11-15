"""SHAP (Linear, interventional Tree, Permutation) and LIME explainers with per-call seeding."""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.utils import check_random_state


def sample_background(X_train: np.ndarray, n: int = 100, seed: int = 42) -> np.ndarray:
    """Deterministically sample a fixed background matrix from training data."""
    X_arr = np.asarray(X_train, dtype=float)
    n_eff = min(int(n), X_arr.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_arr.shape[0], size=n_eff, replace=False)
    return X_arr[idx].copy()


def _extract_pos_shap(values: Any) -> np.ndarray:
    """Normalize SHAP output across versions to shape (N, P) for positive class."""
    if isinstance(values, list):
        arr = np.asarray(values[-1], dtype=float)
    elif hasattr(values, "values"):
        arr = np.asarray(values.values, dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        arr = arr[:, :, -1]
    return np.atleast_2d(arr)


def explain_linear_shap(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute exact interventional LinearSHAP attributions and expected value."""
    X_bg = np.asarray(X_background, dtype=float)
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))

    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=float).ravel()[0])
    explainer = shap.LinearExplainer((coef, intercept), X_bg)
    phi = _extract_pos_shap(explainer.shap_values(X_ex))
    ev = float(np.asarray(explainer.expected_value, dtype=float).ravel()[-1])
    return phi, ev


def explain_tree_shap(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    model_output: str = "raw",
) -> tuple[np.ndarray, float]:
    """Compute exact interventional TreeSHAP attributions with a fixed background."""
    X_bg = np.asarray(X_background, dtype=float)
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))

    target_model = model.get_booster() if hasattr(model, "get_booster") else model
    explainer = shap.TreeExplainer(
        target_model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output=model_output,
    )
    phi = _extract_pos_shap(explainer.shap_values(X_ex, check_additivity=False))
    ev_arr = np.asarray(explainer.expected_value, dtype=float).ravel()
    ev = float(ev_arr[-1])
    return phi, ev


def explain_permutation_shap(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    max_evals: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Compute stochastic SHAP PermutationExplainer attributions for P(Y=1|x)."""
    X_bg = np.asarray(X_background, dtype=float)
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))

    def predict_pos(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)[:, 1]

    # Ensure max_evals is at least 2*P + 1 as required by shap.PermutationExplainer
    n_features = X_ex.shape[1]
    eff_max_evals = max(int(max_evals), 2 * n_features + 1)
    explainer = shap.PermutationExplainer(predict_pos, X_bg, seed=seed)
    sv = explainer(X_ex, max_evals=eff_max_evals, silent=True)
    return _extract_pos_shap(sv)


def build_lime_explainer(
    X_train: np.ndarray,
    feature_names: list[str] | None = None,
    discretize_continuous: bool = True,
    seed: int = 42,
) -> LimeTabularExplainer:
    """Construct a LIME tabular explainer (ridge + exponential kernel, quartile discretizer)."""
    X_tr = np.asarray(X_train, dtype=float)
    names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
    return LimeTabularExplainer(
        training_data=X_tr,
        feature_names=names,
        mode="classification",
        discretize_continuous=discretize_continuous,
        random_state=seed,
    )


def explain_lime(
    lime_explainer: LimeTabularExplainer,
    model: Any,
    X_explain: np.ndarray,
    num_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Compute LIME attribution vectors of shape (N, P) with deterministic per-call seeding."""
    X_ex = np.atleast_2d(np.asarray(X_explain, dtype=float))
    n_rows, n_features = X_ex.shape
    out = np.zeros((n_rows, n_features), dtype=float)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)

    for idx in range(n_rows):
        rs = check_random_state(int(seed) + idx)
        lime_explainer.random_state = rs
        if getattr(lime_explainer, "discretizer", None) is not None:
            lime_explainer.discretizer.random_state = rs
        exp = lime_explainer.explain_instance(
            data_row=X_ex[idx],
            predict_fn=predict_fn,
            labels=(1,),
            num_features=n_features,
            num_samples=int(num_samples),
        )
        for feat_idx, weight in exp.as_map()[1]:
            out[idx, int(feat_idx)] = float(weight)

    return out
