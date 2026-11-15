"""Tests for metrics, including Ballegeer et al. (2025) Table 4 SRA worked example."""

from __future__ import annotations

import numpy as np

from src.metrics import (
    compute_cov,
    compute_sra,
    jaccard_distance_topk,
    normalized_l1_distance,
    reason_code_change,
    spearman_distance,
)


def test_identical_vectors_zero_distance() -> None:
    v = np.array([0.4, -0.9, 0.1, 0.5, -0.2, 0.05])
    assert jaccard_distance_topk(v, v, k=4) == 0.0
    assert spearman_distance(v, v) == 0.0
    assert normalized_l1_distance(v, v) == 0.0
    assert reason_code_change(v, v, k=4, noise_floor_q=0.0) == 0

    mat = np.vstack([v, v, v])
    assert compute_cov(mat) == 0.0
    assert compute_sra(mat, depth=5) == 0.0


def test_sra_reproduces_ballegeer_table4() -> None:
    """Verify SRA against Table 4 of Ballegeer, Bogaert & Benoit (2025)."""
    # Rows = lists 1, 2, 3; cols = features A, B, C, D, E
    lists = np.array(
        [
            [0.20, 0.84, 0.55, 0.93, 0.12],
            [0.21, 0.22, 0.33, 0.65, 0.10],
            [0.36, 0.28, 0.89, 0.77, 0.11],
        ]
    )
    expected = {
        1: 2.0 / 3.0,   # 0.67
        2: 7.0 / 9.0,   # 0.777... (printed .77 in Table 4c)
        3: 2.0 / 3.0,   # 0.67
        4: 2.0 / 3.0,   # 0.67
        5: 8.0 / 15.0,  # 0.53
    }
    for depth, exp_val in expected.items():
        got = compute_sra(lists, depth=depth)
        assert np.isclose(got, exp_val, atol=1e-12), f"Depth {depth}: expected {exp_val}, got {got}"
