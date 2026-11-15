"""Data loading, screening, leak-free preprocessing, and Part 1/Part 2 splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ANALYSIS_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration (defaults to Assignment 3/analysis/config.yaml)."""
    path = Path(config_path) if config_path else ANALYSIS_DIR / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_or_abs: str | Path) -> Path:
    """Resolve a path relative to Assignment 3/analysis/ if not already absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else ANALYSIS_DIR / p


def load_dataset(dataset: str, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned dataset DataFrame and its codebook DataFrame."""
    cfg = config or load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    codebook_dir = resolve_path(cfg["paths"]["codebook_dir"])

    data_path = processed_dir / f"{dataset}_clean.csv"
    cb_path = codebook_dir / f"{dataset}_codebook.csv"
    if not data_path.exists() or not cb_path.exists():
        raise FileNotFoundError(
            f"Missing processed data or codebook for '{dataset}' ({data_path}, {cb_path}). "
            "Run data/download_data.py first."
        )
    df = pd.read_csv(data_path)
    cb = pd.read_csv(cb_path)
    return df, cb


def compute_data_screening(
    df: pd.DataFrame,
    codebook: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Compute missingness, distributional stats, skewness, and IQR outlier rates per feature."""
    rows = []
    cb_map = codebook.set_index("variable").to_dict(orient="index")
    for col in df.columns:
        if col == "target":
            continue
        meta = cb_map[col]
        s = df[col]
        n_total = len(s)
        n_miss = int(s.isna().sum())
        miss_pct = float(n_miss / n_total * 100.0)
        if meta["type"] == "categorical":
            n_unique = int(s.dropna().nunique())
            mode_val = s.dropna().mode().iloc[0] if n_unique > 0 else None
            rows.append({
                "variable": col,
                "type": meta["type"],
                "perturbable": bool(meta["perturbable"]),
                "n_valid": int(n_total - n_miss),
                "missing_count": n_miss,
                "missing_pct": round(miss_pct, 3),
                "mean": np.nan,
                "sd": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "skewness": np.nan,
                "n_unique": n_unique,
                "mode": str(mode_val),
            })
        else:
            vals = pd.to_numeric(s, errors="coerce").dropna()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = float(q3 - q1)
            lo = q1 - iqr_multiplier * iqr
            hi = q3 + iqr_multiplier * iqr
            out_cnt = int(((vals < lo) | (vals > hi)).sum())
            rows.append({
                "variable": col,
                "type": meta["type"],
                "perturbable": bool(meta["perturbable"]),
                "n_valid": int(len(vals)),
                "missing_count": n_miss,
                "missing_pct": round(miss_pct, 3),
                "mean": round(float(vals.mean()), 4),
                "sd": round(float(vals.std(ddof=1)), 4),
                "median": round(float(med), 4),
                "iqr": round(iqr, 4),
                "outlier_count": out_cnt,
                "outlier_pct": round(float(out_cnt / len(vals) * 100.0), 3) if len(vals) else 0.0,
                "skewness": round(float(vals.skew()), 4),
                "n_unique": int(vals.nunique()),
                "mode": "",
            })
    return pd.DataFrame(rows)


def get_exposure_column(codebook: pd.DataFrame) -> str:
    """Return the loan/exposure column name A_i from the codebook."""
    matches = codebook.loc[codebook["cost_exposure_role"] == "exposure_A_i", "variable"]
    if matches.empty:
        raise ValueError("No exposure_A_i column found in codebook.")
    return str(matches.iloc[0])


@dataclass
class Preprocessor:
    """Leak-free tabular preprocessor fitted strictly on training data."""

    codebook: pd.DataFrame
    clip_percentiles: tuple[float, float] = (0.5, 99.5)

    def __post_init__(self) -> None:
        feat_cb = self.codebook[self.codebook["variable"] != "target"]
        self.num_cols: list[str] = feat_cb.loc[
            feat_cb["type"].isin(["continuous", "count", "ordinal"]), "variable"
        ].tolist()
        self.cat_cols: list[str] = feat_cb.loc[
            feat_cb["type"] == "categorical", "variable"
        ].tolist()
        self.perturbable_raw_cols: set[str] = set(
            feat_cb.loc[feat_cb["perturbable"].astype(bool), "variable"].tolist()
        )
        self.exposure_col: str = get_exposure_column(self.codebook)

        self.num_medians_: dict[str, float] = {}
        self.cat_modes_: dict[str, Any] = {}
        self.scaler_: StandardScaler | None = None
        self.encoder_: OneHotEncoder | None = None
        self.feature_names_: list[str] = []
        self.perturbable_mask_: np.ndarray = np.array([], dtype=bool)
        self.feature_sds_: np.ndarray = np.array([], dtype=float)
        self.clip_low_: np.ndarray = np.array([], dtype=float)
        self.clip_high_: np.ndarray = np.array([], dtype=float)

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Fit imputers, scaler, one-hot encoder, SDs, and clip bounds on train_df only."""
        num_df = train_df[self.num_cols].copy()
        for col in self.num_cols:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
            med = float(num_df[col].median())
            self.num_medians_[col] = med
            num_df[col] = num_df[col].fillna(med)

        self.scaler_ = StandardScaler()
        x_num = self.scaler_.fit_transform(num_df.to_numpy(dtype=float)) if self.num_cols else np.empty((len(train_df), 0))

        cat_feature_names: list[str] = []
        if self.cat_cols:
            cat_df = train_df[self.cat_cols].copy()
            for col in self.cat_cols:
                mode_series = cat_df[col].dropna().mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else "MISSING"
                self.cat_modes_[col] = mode_val
                cat_df[col] = cat_df[col].fillna(mode_val).astype(str)
            self.encoder_ = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            x_cat = self.encoder_.fit_transform(cat_df)
            cat_feature_names = self.encoder_.get_feature_names_out(self.cat_cols).tolist()
        else:
            self.encoder_ = None
            x_cat = np.empty((len(train_df), 0))

        x_all = np.hstack([x_num, x_cat])
        self.feature_names_ = list(self.num_cols) + cat_feature_names
        self.perturbable_mask_ = np.array(
            [col in self.perturbable_raw_cols for col in self.num_cols] + [False] * len(cat_feature_names),
            dtype=bool,
        )
        self.feature_sds_ = np.std(x_all, axis=0, ddof=0)
        # Guard against zero SD on rare constant binary columns
        self.feature_sds_ = np.where(self.feature_sds_ > 0, self.feature_sds_, 1.0)
        lo_q, hi_q = self.clip_percentiles
        self.clip_low_ = np.percentile(x_all, lo_q, axis=0)
        self.clip_high_ = np.percentile(x_all, hi_q, axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a split using parameters fitted strictly on training data."""
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        num_df = df[self.num_cols].copy()
        for col in self.num_cols:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(self.num_medians_[col])
        x_num = self.scaler_.transform(num_df.to_numpy(dtype=float)) if self.num_cols else np.empty((len(df), 0))

        if self.cat_cols and self.encoder_ is not None:
            cat_df = df[self.cat_cols].copy()
            for col in self.cat_cols:
                cat_df[col] = cat_df[col].fillna(self.cat_modes_[col]).astype(str)
            x_cat = self.encoder_.transform(cat_df)
        else:
            x_cat = np.empty((len(df), 0))
        return np.hstack([x_num, x_cat])

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on train_df and return transformed feature matrix."""
        return self.fit(train_df).transform(train_df)

    def extract_exposure(self, df: pd.DataFrame) -> np.ndarray:
        """Extract positive loan/exposure amount A_i in original units (imputed with train median)."""
        vals = pd.to_numeric(df[self.exposure_col], errors="coerce")
        med = self.num_medians_.get(self.exposure_col, float(vals.median()))
        vals = vals.fillna(med).to_numpy(dtype=float)
        return np.maximum(vals, 1.0)


def make_part1_split(
    df: pd.DataFrame,
    test_size: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed stratified test set of `test_size` rows and remaining training pool."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["target"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def resample_to_pi(
    train_pool: pd.DataFrame,
    target_pi: float,
    seed: int = 42,
    max_ratio: float = 0.30,
) -> pd.DataFrame:
    """Resample training pool to default rate `target_pi` with fixed N = #defaults / 0.30.

    Defaults are undersampled without replacement (or kept in full at pi=0.30);
    non-defaults are sampled without replacement when available, or with replacement
    if n_neg exceeds available non-defaults (e.g., SGCS at pi=0.01).
    """
    rng = np.random.default_rng(seed)
    pos_df = train_pool[train_pool["target"] == 1]
    neg_df = train_pool[train_pool["target"] == 0]

    n_fixed = int(round(len(pos_df) / max_ratio))
    n_pos = max(1, min(len(pos_df), int(round(target_pi * n_fixed))))
    n_neg = max(1, n_fixed - n_pos)

    pos_idx = rng.choice(pos_df.index.to_numpy(), size=n_pos, replace=False)
    replace_neg = n_neg > len(neg_df)
    neg_idx = rng.choice(neg_df.index.to_numpy(), size=n_neg, replace=replace_neg)

    combined = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(combined)
    return train_pool.loc[combined].reset_index(drop=True)


def make_part2_split(
    df: pd.DataFrame,
    test_frac: float = 0.30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/30 train/test split at natural default rate."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["target"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
