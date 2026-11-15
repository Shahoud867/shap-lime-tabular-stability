"""Instance-dependent cost-sensitive (IDCS) cost matrix, metrics, cslogit, and csboost."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def compute_loan_revenue(
    amounts: np.ndarray,
    int_r_annual: float = 0.0479,
    int_cf_annual: float = 0.0294,
    n_term_months: int = 24,
    margin_m: float | None = None,
) -> np.ndarray:
    """Compute lost revenue r_i per loan following Bahnsen et al. (2014) / Vanderschueren et al. (2022b)."""
    a = np.asarray(amounts, dtype=float)
    if margin_m is not None:
        return margin_m * a
    r_m = int_r_annual / 12.0
    cf_m = int_cf_annual / 12.0
    pmt = a * (r_m * (1.0 + r_m) ** n_term_months) / ((1.0 + r_m) ** n_term_months - 1.0)
    pv = pmt * (1.0 - (1.0 + cf_m) ** (-n_term_months)) / cf_m
    return pv - a


def compute_cost_vectors(
    amounts_eval: np.ndarray,
    amounts_train_ref: np.ndarray,
    y_train_ref: np.ndarray,
    lgd: float = 0.75,
    prior_pi1: float | None = None,
    int_r_annual: float = 0.0479,
    int_cf_annual: float = 0.0294,
    n_term_months: int = 24,
    margin_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute instance-dependent FN cost C(0|1) and FP cost C(1|0) from training references."""
    a_eval = np.asarray(amounts_eval, dtype=float)
    a_tr = np.asarray(amounts_train_ref, dtype=float)
    y_tr = np.asarray(y_train_ref, dtype=float)

    r_eval = compute_loan_revenue(a_eval, int_r_annual, int_cf_annual, n_term_months, margin_m)
    r_tr = compute_loan_revenue(a_tr, int_r_annual, int_cf_annual, n_term_months, margin_m)

    a_bar = float(np.mean(a_tr))
    r_bar = float(np.mean(r_tr))
    pi_1 = float(np.mean(y_tr)) if prior_pi1 is None else float(prior_pi1)
    pi_0 = 1.0 - pi_1

    c_alt = -r_bar * pi_0 + a_bar * lgd * pi_1
    c_fn = a_eval * lgd
    c_fp = np.maximum(0.0, r_eval + c_alt)
    return c_fn, c_fp


def aec_loss(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    c_fn: np.ndarray,
    c_fp: np.ndarray,
) -> float:
    """Average Expected Cost (Ballegeer et al. Eq. 4)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean(y * (1.0 - p) * c_fn + (1.0 - y) * p * c_fp))


def rel_aec(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    c_fn: np.ndarray,
    c_fp: np.ndarray,
    prior_pi1: float,
) -> float:
    """Relative Average Expected Cost normalized by prior-probability baseline (Eq. 5)."""
    model_aec = aec_loss(y_true, p_pred, c_fn, c_fp)
    base_p = np.full(len(y_true), float(prior_pi1), dtype=float)
    base_aec = aec_loss(y_true, base_p, c_fn, c_fp)
    if base_aec <= 0.0:
        return 0.0
    return float(1.0 - model_aec / base_aec)


def savings_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_fn: np.ndarray,
    c_fp: np.ndarray,
) -> float:
    """Cost savings relative to predicting all-0 or all-1 (Ballegeer Eqs. 9-11)."""
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    cost_all_0 = float(np.sum(y * c_fn))
    cost_all_1 = float(np.sum((1.0 - y) * c_fp))
    cost_base = min(cost_all_0, cost_all_1)
    if cost_base <= 0.0:
        return 0.0
    cost_model = float(np.sum(y * (1.0 - yhat) * c_fn + (1.0 - y) * yhat * c_fp))
    return float(1.0 - cost_model / cost_base)


class CSLogitClassifier(BaseEstimator, ClassifierMixin):
    """Instance-dependent cost-sensitive logistic regression minimizing AEC + L1/L2 penalty."""

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1e-2,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        c_fn: np.ndarray | None = None,
        c_fp: np.ndarray | None = None,
    ) -> "CSLogitClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape

        if c_fn is None or c_fp is None:
            c_fn_arr = np.ones(n_samples, dtype=float)
            c_fp_arr = np.ones(n_samples, dtype=float)
        else:
            c_fn_arr = np.asarray(c_fn, dtype=float)
            c_fp_arr = np.asarray(c_fp, dtype=float)

        # Normalize costs to O(1) so Table 5 C values match standard LogisticRegression scale
        sample_costs = np.where(y_arr == 1.0, c_fn_arr, c_fp_arr)
        c_scale = float(np.mean(sample_costs))
        if c_scale <= 0.0:
            c_scale = 1.0
        c_fn_n = c_fn_arr / c_scale
        c_fp_n = c_fp_arr / c_scale

        reg_weight = 0.0 if self.C == 0 else 1.0 / float(self.C)
        pen = self.penalty.lower()

        # Warm-start from cost-weighted standard LR
        warm_lr = LogisticRegression(
            penalty=pen if reg_weight > 0 else None,
            C=float(self.C) if reg_weight > 0 else 1.0,
            solver="liblinear" if pen == "l1" and reg_weight > 0 else "lbfgs",
            max_iter=500,
            random_state=self.random_state,
        )
        try:
            warm_lr.fit(X_arr, y_arr.astype(int), sample_weight=sample_costs / c_scale)
            w_warm = warm_lr.coef_.ravel().copy()
            b_warm = float(warm_lr.intercept_[0])
        except Exception:  # noqa: BLE001
            w_warm = np.zeros(n_features, dtype=float)
            b_warm = 0.0

        starts = [(w_warm, b_warm), (np.zeros(n_features, dtype=float), 0.0)]

        if pen == "l1" and reg_weight > 0:
            best_res = None
            bounds = [(None, None)] + [(0.0, None)] * (2 * n_features)

            def obj_l1(theta: np.ndarray) -> tuple[float, np.ndarray]:
                b = theta[0]
                u = theta[1 : 1 + n_features]
                v = theta[1 + n_features :]
                w = u - v
                z = np.clip(X_arr @ w + b, -30.0, 30.0)
                s = expit(z)
                loss = float(np.sum(y_arr * (1.0 - s) * c_fn_n + (1.0 - y_arr) * s * c_fp_n))
                loss += reg_weight * float(np.sum(u + v))
                dz = s * (1.0 - s) * ((1.0 - y_arr) * c_fp_n - y_arr * c_fn_n)
                grad_b = float(np.sum(dz))
                grad_w = X_arr.T @ dz
                grad = np.concatenate([[grad_b], grad_w + reg_weight, -grad_w + reg_weight])
                return loss, grad

            for w0, b0 in starts:
                u0 = np.maximum(w0, 0.0)
                v0 = np.maximum(-w0, 0.0)
                theta0 = np.concatenate([[b0], u0, v0])
                res = minimize(obj_l1, theta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": self.max_iter})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            self.intercept_ = np.array([float(best_res.x[0])])
            self.coef_ = (best_res.x[1 : 1 + n_features] - best_res.x[1 + n_features :]).reshape(1, -1)
        else:
            best_res = None

            def obj_l2(theta: np.ndarray) -> tuple[float, np.ndarray]:
                b = theta[0]
                w = theta[1:]
                z = np.clip(X_arr @ w + b, -30.0, 30.0)
                s = expit(z)
                loss = float(np.sum(y_arr * (1.0 - s) * c_fn_n + (1.0 - y_arr) * s * c_fp_n))
                if reg_weight > 0:
                    loss += 0.5 * reg_weight * float(np.sum(w * w))
                dz = s * (1.0 - s) * ((1.0 - y_arr) * c_fp_n - y_arr * c_fn_n)
                grad_b = float(np.sum(dz))
                grad_w = X_arr.T @ dz + (reg_weight * w if reg_weight > 0 else 0.0)
                return loss, np.concatenate([[grad_b], grad_w])

            for w0, b0 in starts:
                theta0 = np.concatenate([[b0], w0])
                res = minimize(obj_l2, theta0, method="L-BFGS-B", jac=True, options={"maxiter": self.max_iter})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            self.intercept_ = np.array([float(best_res.x[0])])
            self.coef_ = best_res.x[1:].reshape(1, -1)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_.ravel() + float(self.intercept_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(np.clip(self.decision_function(X), -30.0, 30.0))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CSBoostClassifier(BaseEstimator, ClassifierMixin):
    """Instance-dependent cost-sensitive XGBoost minimizing AEC with clipped Hessian."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 3,
        min_child_weight: float = 0.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        reg_lambda: float = 1.0,
        hessian_clip: float = 1e-4,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.hessian_clip = hessian_clip
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        c_fn: np.ndarray | None = None,
        c_fp: np.ndarray | None = None,
    ) -> "CSBoostClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples = X_arr.shape[0]

        if c_fn is None or c_fp is None:
            c_fn_arr = np.ones(n_samples, dtype=float)
            c_fp_arr = np.ones(n_samples, dtype=float)
        else:
            c_fn_arr = np.asarray(c_fn, dtype=float)
            c_fp_arr = np.asarray(c_fp, dtype=float)

        sample_costs = np.where(y_arr == 1.0, c_fn_arr, c_fp_arr)
        c_scale = float(np.mean(sample_costs))
        if c_scale <= 0.0:
            c_scale = 1.0
        d_norm = ((1.0 - y_arr) * c_fp_arr - y_arr * c_fn_arr) / c_scale
        h_clip = float(self.hessian_clip)

        def aec_obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
            z = np.clip(preds, -30.0, 30.0)
            s = expit(z)
            grad = s * (1.0 - s) * d_norm
            hess = s * (1.0 - s) * (1.0 - 2.0 * s) * d_norm
            # Hessian of AEC can be negative; clip to small positive constant for convexity
            hess = np.maximum(hess, h_clip)
            return grad, hess

        dtrain = xgb.DMatrix(X_arr, label=y_arr)
        params: dict[str, Any] = {
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "lambda": self.reg_lambda,
            "base_score": 0.5,
            "seed": self.random_state,
            "nthread": self.n_jobs,
            "tree_method": "hist",
        }
        self.booster_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            obj=aec_obj,
            verbose_eval=False,
        )
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_arr.shape[1]
        return self

    def get_booster(self) -> xgb.Booster:
        return self.booster_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        dmat = xgb.DMatrix(X_arr)
        return self.booster_.predict(dmat, output_margin=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        margin = np.clip(self.decision_function(X), -30.0, 30.0)
        p1 = expit(margin)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
