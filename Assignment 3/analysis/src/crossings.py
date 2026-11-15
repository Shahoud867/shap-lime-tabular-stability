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


def extract_tree_splits(model: Any, feature_names: list[str] | None = None) -> TreeSplits:
    """Extract split thresholds, operators, and normalized node cover from RF or XGB."""
    # Scikit-learn RandomForestClassifier or DecisionTreeClassifier
    if hasattr(model, "estimators_") or hasattr(model, "tree_"):
        estimators = list(model.estimators_) if hasattr(model, "estimators_") else [model]
        feat_list: list[np.ndarray] = []
        thresh_list: list[np.ndarray] = []
        cov_list: list[np.ndarray] = []

        for est in estimators:
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

        if not feat_list:
            return TreeSplits(
                feature_idx=np.empty(0, dtype=np.int64),
                threshold=np.empty(0, dtype=np.float64),
                op="<=",
                cover_weight=np.empty(0, dtype=np.float64),
                n_trees=len(estimators),
            )
        return TreeSplits(
            feature_idx=np.concatenate(feat_list),
            threshold=np.concatenate(thresh_list),
            op="<=",
            cover_weight=np.concatenate(cov_list),
            n_trees=len(estimators),
        )

    # XGBoost (XGBClassifier, CSBoostClassifier, or Booster)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    df = booster.trees_to_dataframe()
    n_trees = int(df["Tree"].nunique()) if not df.empty else 0
    internal = df[df["Feature"] != "Leaf"].copy()
    if internal.empty:
        return TreeSplits(
            feature_idx=np.empty(0, dtype=np.int64),
            threshold=np.empty(0, dtype=np.float64),
            op="<",
            cover_weight=np.empty(0, dtype=np.float64),
            n_trees=n_trees,
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

    return TreeSplits(
        feature_idx=feat_idx,
        threshold=thresh,
        op="<",
        cover_weight=cover_w,
        n_trees=n_trees,
    )


def count_tree_crossings(
    x: np.ndarray,
    x_primes: np.ndarray,
    tree_splits: TreeSplits,
    per_tree: bool = True,
) -> np.ndarray:
    """Count split-threshold crossings between x and each row in x_primes."""
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

    crossed = side_xp != side_x[None, :]
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
