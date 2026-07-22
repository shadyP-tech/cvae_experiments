from __future__ import annotations

import numpy as np
import pytest

from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.penalty import (
    build_conditional_penalty,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_rectangular_factor_matches_dense_reference_and_unit_trace() -> None:
    x = np.asarray(
        [
            [5.0, 1.0, 0.0],   # center 2, class 1
            [-2.0, 0.0, 1.0],  # center 0, class 0
            [3.0, 2.0, -1.0],  # center 1, class 1
            [0.0, -1.0, 2.0],  # center 1, class 0
            [7.0, 0.0, 1.0],   # center 2, class 0
            [1.0, 3.0, 0.0],   # center 0, class 1
        ],
        dtype=np.float64,
    )
    y = (1, 0, 1, 0, 0, 1)
    centers = ("2", "0", "1", "1", "2", "0")

    operator = build_conditional_penalty(x, y, centers)

    assert operator.centers == ("0", "1", "2")
    assert operator.row_keys == (
        ("0", 0),
        ("0", 1),
        ("1", 0),
        ("1", 1),
        ("2", 0),
        ("2", 1),
    )
    assert np.asarray(operator.factor).shape == (6, 3)
    assert operator.trace == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert 0 < operator.rank <= min(3, 2 * (3 - 1))
    assert operator.maximum_rank == 3
    assert operator.audit_payload()["maximum_factor_rank"] == 3
    assert operator.audit_payload()["all_cells_present"] is True

    weights = np.asarray([0.2, -0.7, 1.3], dtype=np.float64)
    r = np.asarray(operator.factor)
    dense = r.T @ r
    assert operator.value(weights) == pytest.approx(float(weights @ dense @ weights))
    np.testing.assert_allclose(operator.gradient(weights), 2.0 * dense @ weights)


def test_centroids_give_equal_mass_to_each_domain_cell() -> None:
    # Center 0/class 0 has three rows; its cell mean must still receive the same
    # class-centering mass as the one-row center 1/class 0 cell.
    x = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [10.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=float,
    )
    y = (0, 0, 0, 0, 1, 1)
    centers = ("0", "0", "0", "1", "0", "1")
    operator = build_conditional_penalty(x, y, centers)

    # class-0 cell means are 2 and 10, so equal-domain centering produces
    # contrasts -4 and +4 (not a sample-frequency-weighted reference).
    raw_contrasts = np.asarray(operator.factor) * np.sqrt(4.0 * operator.t)
    np.testing.assert_allclose(raw_contrasts[0], [-4.0, 0.0])
    np.testing.assert_allclose(raw_contrasts[2], [4.0, 0.0])


def test_penalty_fails_closed_on_missing_or_degenerate_cells() -> None:
    missing_x = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    with pytest.raises(ProtocolError, match="cell is missing"):
        build_conditional_penalty(
            missing_x,
            (0, 1, 0),
            ("0", "0", "1"),
        )

    degenerate_x = np.asarray([[1.0], [2.0], [1.0], [2.0]], dtype=float)
    with pytest.raises(ProtocolError, match="degenerate"):
        build_conditional_penalty(
            degenerate_x,
            (0, 1, 0, 1),
            ("0", "0", "1", "1"),
        )


def test_factor_hash_is_stable_under_fit_row_permutation() -> None:
    x = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 3.0]], dtype=float
    )
    y = np.asarray([0, 1, 0, 1], dtype=int)
    centers = np.asarray(["0", "0", "1", "1"])
    order = np.asarray([3, 0, 2, 1])

    first = build_conditional_penalty(x, y, centers)
    second = build_conditional_penalty(x[order], y[order], centers[order])

    assert first.factor_hash == second.factor_hash
    assert first.centroid_hash == second.centroid_hash
    np.testing.assert_array_equal(first.factor, second.factor)
