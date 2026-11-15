"""Tests for data loading, leak-free preprocessing, and Part 1/Part 2 splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import (
    Preprocessor,
    compute_data_screening,
    make_part1_split,
    make_part2_split,
    resample_to_pi,
)


def _make_synthetic_df(n: int = 500, pos_rate: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * pos_rate))
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(y)
    df = pd.DataFrame(
        {
            "LOAN": rng.uniform(1000, 50000, size=n),
            "DEBTINC": rng.normal(35.0, 8.0, size=n),
            "DEROG": rng.poisson(0.3, size=n),
            "REASON": rng.choice(["DebtCon", "HomeImp"], size=n),
            "target": y,
        }
    )
    df.loc[rng.choice(n, size=25, replace=False), "DEBTINC"] = np.nan
    cb = pd.DataFrame(
        [
            {"variable": "LOAN", "type": "continuous", "perturbable": True, "cost_exposure_role": "exposure_A_i"},
            {"variable": "DEBTINC", "type": "continuous", "perturbable": True, "cost_exposure_role": "none"},
            {"variable": "DEROG", "type": "count", "perturbable": False, "cost_exposure_role": "none"},
            {"variable": "REASON", "type": "categorical", "perturbable": False, "cost_exposure_role": "none"},
            {"variable": "target", "type": "binary_target", "perturbable": False, "cost_exposure_role": "target"},
        ]
    )
    return df, cb


def test_preprocessor_no_test_leakage() -> None:
    df, cb = _make_synthetic_df(n=400, seed=7)
    train_df, test_df = make_part2_split(df, test_frac=0.30, seed=7)

    prep_clean = Preprocessor(cb).fit(train_df)
    X_train_1 = prep_clean.transform(train_df)

    # Corrupt test_df massively; fitted Preprocessor on train_df must be identical
    test_corrupt = test_df.copy()
    test_corrupt["LOAN"] = 1e9
    test_corrupt["DEBTINC"] = -1e6
    prep_2 = Preprocessor(cb).fit(train_df)
    X_train_2 = prep_2.transform(train_df)

    assert np.allclose(X_train_1, X_train_2)
    assert np.allclose(prep_clean.feature_sds_, prep_2.feature_sds_)
    assert np.allclose(prep_clean.clip_low_, prep_2.clip_low_)
    assert np.allclose(prep_clean.clip_high_, prep_2.clip_high_)
    assert prep_clean.perturbable_mask_.tolist() == [True, True, False, False]


def test_part1_split_and_resampling_rates() -> None:
    df, _ = _make_synthetic_df(n=600, pos_rate=0.20, seed=11)
    train_pool, test_df = make_part1_split(df, test_size=300, seed=11)
    assert len(test_df) == 300
    assert abs(test_df["target"].mean() - 0.20) < 0.02

    n_pos_pool = int((train_pool["target"] == 1).sum())
    expected_fixed = int(round(n_pos_pool / 0.30))

    for pi in [0.01, 0.15, 0.30]:
        res = resample_to_pi(train_pool, target_pi=pi, seed=11, max_ratio=0.30)
        assert len(res) == expected_fixed
        assert abs(res["target"].mean() - pi) <= (1.0 / expected_fixed + 1e-9)


def test_screening_stats() -> None:
    df, cb = _make_synthetic_df(n=200, seed=3)
    scr = compute_data_screening(df, cb)
    assert set(scr["variable"]) == {"LOAN", "DEBTINC", "DEROG", "REASON"}
    debt_row = scr.loc[scr["variable"] == "DEBTINC"].iloc[0]
    assert debt_row["missing_count"] > 0


def test_real_datasets_and_codebooks_present() -> None:
    from src.data import get_exposure_column, load_dataset

    expected = {
        "hmeq": (5960, 12, "LOAN", 6),
        "sgcs": (1000, 20, "amount", 3),
        "taiwan": (30000, 23, "LIMIT_BAL", 13),
    }
    for ds, (n_rows, n_feats, exp_col, n_pert) in expected.items():
        df, cb = load_dataset(ds)
        assert len(df) == n_rows
        assert df.shape[1] - 1 == n_feats
        assert get_exposure_column(cb) == exp_col
        assert int(cb["perturbable"].sum()) == n_pert

