"""Extract tree split thresholds and LIME bin edges, and count perturbation crossings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TreeSplits:
    """Flattened representation of all internal split nodes in a tree ensemble."""

    feature_idx: np.ndarray
    threshold: np.ndarray
    op: str  # "<=" for scikit-learn, "<" for xgboost
    cover_weight: np.ndarray
    n_trees: int
    value_jump: np.ndarray | None = None
    range_jump: np.ndarray | None = None
    tree_pred_std: np.ndarray | None = None
    ensemble_pred_std: float | None = None
    tree_idx: np.ndarray | None = None


def _weighted_leaf_std(leaf_vals: np.ndarray, leaf_weights: np.ndarray) -> float:
    """Compute weighted standard deviation of leaf predictions, clamped >= 1e-8."""
    if leaf_vals.size <= 1:
        return 1e-8
    w = np.maximum(np.asarray(leaf_weights, dtype=np.float64), 0.0)
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        w_norm = w / w_sum
        mu = float(np.sum(w_norm * leaf_vals))
        var = float(np.sum(w_norm * (leaf_vals - mu) ** 2))
    else:
        var = float(np.var(leaf_vals))
    return max(float(np.sqrt(max(var, 0.0))), 1e-8)


def extract_tree_splits(
    model: Any,
    feature_names: list[str] | None = None,
    X_train: np.ndarray | None = None,
) -> TreeSplits:
    """Extract split thresholds, operators, cover weights, and value jumps from RF or XGB."""
    # Scikit-learn RandomForestClassifier or DecisionTreeClassifier
    if hasattr(model, "estimators_") or hasattr(model, "tree_"):
        estimators = list(model.estimators_) if hasattr(model, "estimators_") else [model]
        feat_list: list[np.ndarray] = []
        thresh_list: list[np.ndarray] = []
        cov_list: list[np.ndarray] = []
        jump_list: list[np.ndarray] = []
        range_list: list[np.ndarray] = []
        std_list: list[np.ndarray] = []
        tidx_list: list[np.ndarray] = []

        X_tr = np.asarray(X_train, dtype=np.float64) if X_train is not None else None
        ens_pred_std: float | None = None
        if X_tr is not None and X_tr.size > 0 and hasattr(model, "predict_proba"):
            p_ens = model.predict_proba(X_tr)[:, 1]
            ens_pred_std = max(float(np.std(p_ens)), 1e-6)

        for t_idx, est in enumerate(estimators):
            tree = est.tree_
            internal = tree.feature >= 0
            if not np.any(internal):
                continue
            feat_list.append(tree.feature[internal].astype(np.int64))
            thresh_list.append(tree.threshold[internal].astype(np.float64))
            root_cov = float(tree.weighted_n_node_samples[0])
            if root_cov <= 0.0:
                root_cov = 1.0
            cov_list.append((tree.weighted_n_node_samples[internal] / root_cov).astype(np.float64))

            # Node prediction value V(u): P(y=1 | u) for classifiers, scalar value for regressors
            vals = tree.value[:, 0, :]
            if vals.shape[1] >= 2:
                totals = np.maximum(np.sum(vals, axis=1), 1e-12)
                node_val = (vals[:, 1] / totals).astype(np.float64)
            else:
                node_val = vals[:, 0].astype(np.float64)

            left_children = tree.children_left[internal]
            right_children = tree.children_right[internal]
            jumps = np.abs(node_val[left_children] - node_val[right_children]).astype(np.float64)
            jump_list.append(jumps)

            # Bottom-up (M_min, M_max) for Proposition P2a worst-case oscillation range_jump
            n_nodes = tree.node_count
            m_min = np.empty(n_nodes, dtype=np.float64)
            m_max = np.empty(n_nodes, dtype=np.float64)
            for u in range(n_nodes - 1, -1, -1):
                if tree.feature[u] < 0:
                    m_min[u] = node_val[u]
                    m_max[u] = node_val[u]
                else:
                    lc = tree.children_left[u]
                    rc = tree.children_right[u]
                    m_min[u] = min(m_min[lc], m_min[rc])
                    m_max[u] = max(m_max[lc], m_max[rc])
            r_jumps = np.maximum(
                np.abs(m_max[left_children] - m_min[right_children]),
                np.abs(m_max[right_children] - m_min[left_children]),
            ).astype(np.float64)
            range_list.append(r_jumps)

            if X_tr is not None and X_tr.size > 0:
                p_t = est.predict_proba(X_tr)[:, 1] if hasattr(est, "predict_proba") else est.predict(X_tr)
                pred_std = max(float(np.std(p_t)), 1e-6)
            else:
                is_leaf = tree.feature < 0
                pred_std = max(_weighted_leaf_std(node_val[is_leaf], tree.weighted_n_node_samples[is_leaf]), 1e-6)
            n_int = int(np.sum(internal))
            std_list.append(np.full(n_int, pred_std, dtype=np.float64))
            tidx_list.append(np.full(n_int, t_idx, dtype=np.int64))

        if not feat_list:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return TreeSplits(
                feature_idx=empty_i,
                threshold=empty_f,
                op="<=",
                cover_weight=empty_f,
                n_trees=len(estimators),
                value_jump=empty_f,
                range_jump=empty_f,
                tree_pred_std=empty_f,
                ensemble_pred_std=ens_pred_std,
                tree_idx=empty_i,
            )
        return TreeSplits(
            feature_idx=np.concatenate(feat_list),
            threshold=np.concatenate(thresh_list),
            op="<=",
            cover_weight=np.concatenate(cov_list),
            n_trees=len(estimators),
            value_jump=np.concatenate(jump_list),
            range_jump=np.concatenate(range_list),
            tree_pred_std=np.concatenate(std_list),
            ensemble_pred_std=ens_pred_std,
            tree_idx=np.concatenate(tidx_list),
        )

    # XGBoost (XGBClassifier, CSBoostClassifier, or Booster)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    df = booster.trees_to_dataframe()
    n_trees = int(df["Tree"].nunique()) if not df.empty else 0
    internal = df[df["Feature"] != "Leaf"].copy()
    if internal.empty:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return TreeSplits(
            feature_idx=empty_i,
            threshold=empty_f,
            op="<",
            cover_weight=empty_f,
            n_trees=n_trees,
            value_jump=empty_f,
            range_jump=empty_f,
            tree_pred_std=empty_f,
            ensemble_pred_std=None,
            tree_idx=empty_i,
        )

    name_to_idx = {name: i for i, name in enumerate(feature_names)} if feature_names else {}

    def _map_feat(val: Any) -> int:
        s = str(val)
        if s in name_to_idx:
            return name_to_idx[s]
        if s.startswith("f") and s[1:].isdigit():
            return int(s[1:])
        raise ValueError(f"Cannot map XGBoost feature name '{s}' to index.")

    feat_idx = np.array([_map_feat(v) for v in internal["Feature"]], dtype=np.int64)
    thresh = internal["Split"].to_numpy(dtype=np.float64)

    # Normalize each node's Cover by its tree's root Cover (Node == 0)
    roots = df[df["Node"] == 0].set_index("Tree")["Cover"].to_dict()
    tree_roots = np.array([max(float(roots.get(t, 1.0)), 1e-12) for t in internal["Tree"]], dtype=np.float64)
    cover_w = internal["Cover"].to_numpy(dtype=np.float64) / tree_roots

    # Bottom-up cover-weighted node values, (M_min, M_max), and leaf SD per tree
    node_val_map: dict[str, float] = {}
    min_map: dict[str, float] = {}
    max_map: dict[str, float] = {}
    tree_std_map: dict[int, float] = {}
    for t_id, sub in df.groupby("Tree", sort=False):
        cov_map: dict[str, float] = {}
        sub_desc = sub.sort_values("Node", ascending=False)
        for nid, feat, gain, cov, yes_id, no_id in zip(
            sub_desc["ID"],
            sub_desc["Feature"],
            sub_desc["Gain"],
            sub_desc["Cover"],
            sub_desc["Yes"],
            sub_desc["No"],
        ):
            nid_s = str(nid)
            cov_map[nid_s] = float(cov)
            if feat == "Leaf":
                val = float(gain)
                node_val_map[nid_s] = val
                min_map[nid_s] = val
                max_map[nid_s] = val
            else:
                y_s = str(yes_id)
                n_s = str(no_id)
                c_l = cov_map.get(y_s, 0.0)
                c_r = cov_map.get(n_s, 0.0)
                v_l = node_val_map.get(y_s, 0.0)
                v_r = node_val_map.get(n_s, 0.0)
                denom = c_l + c_r
                node_val_map[nid_s] = (c_l * v_l + c_r * v_r) / denom if denom > 0.0 else 0.5 * (v_l + v_r)
                min_map[nid_s] = min(min_map.get(y_s, 0.0), min_map.get(n_s, 0.0))
                max_map[nid_s] = max(max_map.get(y_s, 0.0), max_map.get(n_s, 0.0))
        leaves = sub[sub["Feature"] == "Leaf"]
        tree_std_map[int(t_id)] = max(
            _weighted_leaf_std(
                leaves["Gain"].to_numpy(dtype=np.float64),
                leaves["Cover"].to_numpy(dtype=np.float64),
            ),
            1e-6,
        )

    value_jump = np.array(
        [abs(node_val_map[str(y_id)] - node_val_map[str(n_id)]) for y_id, n_id in zip(internal["Yes"], internal["No"])],
        dtype=np.float64,
    )
    range_jump = np.array(
        [
            max(
                abs(max_map[str(y_id)] - min_map[str(n_id)]),
                abs(max_map[str(n_id)] - min_map[str(y_id)]),
            )
            for y_id, n_id in zip(internal["Yes"], internal["No"])
        ],
        dtype=np.float64,
    )

    ens_pred_std_xgb: float | None = None
    if X_train is not None:
        X_tr = np.asarray(X_train, dtype=np.float64)
        if X_tr.size > 0:
            import xgboost as xgb

            dm = xgb.DMatrix(X_tr, feature_names=feature_names if feature_names else None)
            margins = booster.predict(dm, output_margin=True)
            ens_pred_std_xgb = max(float(np.std(margins)) / max(float(n_trees), 1.0), 1e-6)
            for t_idx in range(n_trees):
                t_margin = booster[t_idx : t_idx + 1].predict(dm, output_margin=True)
                tree_std_map[t_idx] = max(float(np.std(t_margin)), 1e-6)

    tree_pred_std = np.array([tree_std_map[int(t)] for t in internal["Tree"]], dtype=np.float64)
    tree_idx_arr = internal["Tree"].to_numpy(dtype=np.int64)

    return TreeSplits(
        feature_idx=feat_idx,
        threshold=thresh,
        op="<",
        cover_weight=cover_w,
        n_trees=n_trees,
        value_jump=value_jump,
        range_jump=range_jump,
        tree_pred_std=tree_pred_std,
        ensemble_pred_std=ens_pred_std_xgb,
        tree_idx=tree_idx_arr,
    )


def count_tree_crossings(
    x: np.ndarray,
    x_primes: np.ndarray,
    tree_splits: TreeSplits,
    per_tree: bool = True,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Count (optionally weighted) split-threshold crossings between x and each row in x_primes."""
    x_vec = np.asarray(x, dtype=float).ravel()
    xp = np.atleast_2d(np.asarray(x_primes, dtype=float))
    n_draws = xp.shape[0]
    if tree_splits.feature_idx.size == 0 or tree_splits.n_trees == 0:
        return np.zeros(n_draws, dtype=float)

    f_idx = tree_splits.feature_idx
    thresh = tree_splits.threshold
    x_vals = x_vec[f_idx]
    xp_vals = xp[:, f_idx]

    if tree_splits.op == "<=":
        side_x = x_vals <= thresh
        side_xp = xp_vals <= thresh
    elif tree_splits.op == "<":
        side_x = x_vals < thresh
        side_xp = xp_vals < thresh
    else:
        raise ValueError(f"Unsupported split operator: {tree_splits.op}")

    crossed = (side_xp != side_x[None, :]).astype(float)
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        crossed = crossed * w[None, :]
    counts = np.sum(crossed, axis=1, dtype=float)
    if per_tree:
        counts = counts / float(tree_splits.n_trees)
    return counts


def extract_lime_bin_edges(lime_explainer: Any) -> dict[int, np.ndarray]:
    """Extract per-feature discretizer bin edges from a LIME tabular explainer."""
    discretizer = getattr(lime_explainer, "discretizer", None)
    if discretizer is None:
        return {}

    # In lime.discretize.BaseDiscretizer:
    # - self.mins[feat] = [min_val] + qts.tolist()
    # - self.maxs[feat] = qts.tolist() + [max_val]
    # - self.lambdas[feat] = lambda x, qts=qts: np.searchsorted(qts, x) (side='left' => <=)
    mins_map = getattr(discretizer, "mins", None)
    lambdas_map = getattr(discretizer, "lambdas", None)
    edges: dict[int, np.ndarray] = {}

    if mins_map:
        for feat_idx, vals in mins_map.items():
            if len(vals) > 1:
                arr = np.unique(np.asarray(vals[1:], dtype=float))
                if arr.size > 0:
                    edges[int(feat_idx)] = arr
    elif lambdas_map:
        for feat_idx, fn in lambdas_map.items():
            defaults = getattr(fn, "__defaults__", None)
            if defaults:
                arr = np.unique(np.asarray(defaults[0], dtype=float).ravel())
                if arr.size > 0:
                    edges[int(feat_idx)] = arr
    return edges


def count_bin_crossings(
    x: np.ndarray,
    x_primes: np.ndarray,
    bin_edges: dict[int, np.ndarray],
    op: str = "<=",
) -> np.ndarray:
    """Count LIME bin-edge crossings between x and each row in x_primes."""
    x_vec = np.asarray(x, dtype=float).ravel()
    xp = np.atleast_2d(np.asarray(x_primes, dtype=float))
    counts = np.zeros(xp.shape[0], dtype=float)
    if not bin_edges:
        return counts

    for feat_idx, edges in bin_edges.items():
        if edges.size == 0:
            continue
        xv = x_vec[feat_idx]
        xpv = xp[:, feat_idx : feat_idx + 1]
        if op == "<=":
            side_x = xv <= edges
            side_xp = xpv <= edges[None, :]
        else:
            side_x = xv < edges
            side_xp = xpv < edges[None, :]
        counts += np.sum(side_xp != side_x[None, :], axis=1, dtype=float)
    return counts
