"""Model factories, Ballegeer et al. Table 5 hyperparameter tuning, and evaluation metrics."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .idcs import CSBoostClassifier, CSLogitClassifier, aec_loss, rel_aec, savings_score

MODEL_ALIASES: dict[str, str] = {
    "lr": "logit",
    "logit": "logit",
    "rf": "forest",
    "forest": "forest",
    "xgb": "boost",
    "boost": "boost",
    "cslogit": "cslogit",
    "csboost": "csboost",
}


def canonical_model_name(name: str) -> str:
    """Normalize Part 1 / Part 2 model aliases to canonical names."""
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_ALIASES[key]


def get_param_grid(model_name: str) -> list[dict[str, Any]]:
    """Return Ballegeer et al. (2025) Table 5 hyperparameter grid."""
    canon = canonical_model_name(model_name)
    if canon in ("logit", "cslogit"):
        grid: list[dict[str, Any]] = []
        c_vals = [0.0, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 1e-1]
        for c_val in c_vals:
            if c_val == 0.0:
                grid.append({"penalty": "l2", "C": 0.0})
            else:
                for pen in ("l1", "l2"):
                    grid.append({"penalty": pen, "C": c_val})
        return grid

    if canon == "forest":
        grid = []
        for md, ne in itertools.product([None, 3, 5, 10], [20, 50, 100]):
            grid.append({"max_depth": md, "n_estimators": ne})
        return grid

    if canon in ("boost", "csboost"):
        grid = []
        for mcw, md, cs, gam in itertools.product(
            [0.0, 50.0], [1, 3, 5, 7], [0.8, 1.0], [0.0, 5.0, 10.0]
        ):
            grid.append(
                {
                    "learning_rate": 0.3,
                    "n_estimators": 100,
                    "min_child_weight": mcw,
                    "max_depth": md,
                    "colsample_bytree": cs,
                    "gamma": gam,
                }
            )
        return grid

    raise ValueError(f"No grid for {model_name}")


def build_estimator(
    model_name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_jobs: int = 1,
) -> Any:
    """Instantiate an unfitted classifier with deterministic seed."""
    canon = canonical_model_name(model_name)
    p = dict(params or {})

    if canon == "logit":
        c_val = float(p.get("C", 0.1))
        pen = str(p.get("penalty", "l2")).lower()
        if c_val == 0.0:
            return LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000, random_state=seed
            )
        solver = "liblinear" if pen == "l1" else "lbfgs"
        return LogisticRegression(
            penalty=pen, C=c_val, solver=solver, max_iter=1000, random_state=seed
        )

    if canon == "cslogit":
        return CSLogitClassifier(
            penalty=str(p.get("penalty", "l2")),
            C=float(p.get("C", 0.01)),
            max_iter=500,
            random_state=seed,
        )

    if canon == "forest":
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=p.get("max_depth", 10),
            random_state=seed,
            n_jobs=n_jobs,
        )

    if canon == "boost":
        return xgb.XGBClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.3)),
            max_depth=int(p.get("max_depth", 3)),
            min_child_weight=float(p.get("min_child_weight", 0.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            gamma=float(p.get("gamma", 0.0)),
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if canon == "csboost":
        return CSBoostClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.3)),
            max_depth=int(p.get("max_depth", 3)),
            min_child_weight=float(p.get("min_child_weight", 0.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            gamma=float(p.get("gamma", 0.0)),
            random_state=seed,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def fit_estimator(
    estimator: Any,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    c_fn: np.ndarray | None = None,
    c_fp: np.ndarray | None = None,
) -> Any:
    """Fit traditional or IDCS estimator cleanly."""
    canon = canonical_model_name(model_name)
    if canon in ("cslogit", "csboost"):
        estimator.fit(X, y, c_fn=c_fn, c_fp=c_fp)
    else:
        estimator.fit(X, y)
    return estimator


def tune_hyperparameters(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    c_fn: np.ndarray | None = None,
    c_fp: np.ndarray | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Tune hyperparameters via Stratified K-fold CV (AUC for traditional, -AEC for IDCS)."""
    canon = canonical_model_name(model_name)
    is_idcs = canon in ("cslogit", "csboost")
    grid = get_param_grid(canon)

    # Guard against tiny minority counts in extreme 1% imbalance folds
    n_pos = int(np.sum(y == 1))
    n_folds = min(cv_folds, max(2, n_pos))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    best_score = -np.inf
    best_params = dict(grid[0])

    for params in grid:
        fold_scores: list[float] = []
        for tr_idx, va_idx in splits:
            est = build_estimator(canon, params=params, seed=seed, n_jobs=n_jobs)
            c_fn_tr = c_fn[tr_idx] if c_fn is not None else None
            c_fp_tr = c_fp[tr_idx] if c_fp is not None else None
            fit_estimator(est, canon, X[tr_idx], y[tr_idx], c_fn=c_fn_tr, c_fp=c_fp_tr)
            p_va = est.predict_proba(X[va_idx])[:, 1]
            if is_idcs and c_fn is not None and c_fp is not None:
                score = -aec_loss(y[va_idx], p_va, c_fn[va_idx], c_fp[va_idx])
            else:
                try:
                    score = float(roc_auc_score(y[va_idx], p_va))
                except ValueError:
                    score = 0.5
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    return best_params


def evaluate_model(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    c_fn: np.ndarray | None = None,
    c_fp: np.ndarray | None = None,
    prior_pi1: float | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discriminative (AUC, AP, Brier) and cost (AEC, relAEC, Savings) metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    yhat = (p >= threshold).astype(int)

    out: dict[str, float] = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
    }
    if c_fn is not None and c_fp is not None:
        pi1 = float(np.mean(y)) if prior_pi1 is None else float(prior_pi1)
        out["aec"] = aec_loss(y, p, c_fn, c_fp)
        out["rel_aec"] = rel_aec(y, p, c_fn, c_fp, pi1)
        out["savings"] = savings_score(y, yhat, c_fn, c_fp)
    return out
