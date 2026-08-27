"""Frozen label-free descriptors for local action-value correction."""

from __future__ import annotations

import math

import numpy as np

from ..action_geometry import BoundaryProjection, HARD_THRESHOLD
from ..protocol import ProtocolError
from .contracts import ActionDescriptor


ACTION_FEATURE_NAMES = (
    "crossing_fraction",
    "mean_abs_endpoint_displacement_crossing",
    "mean_abs_boundary_displacement_crossing",
    "mean_baseline_margin_crossing",
    "mean_endpoint_margin_crossing",
    "mean_eta_crossing",
    "eta_entropy_crossing",
    "eta_variance_crossing",
    "mean_posterior_sd_crossing",
    "mean_seed_sd_crossing",
    "mean_vote_disagreement_crossing",
    "case_log_rows",
    "support_log_rows",
    "support_positive_fraction",
    "support_imbalance",
    "estimated_held_positive_fraction",
    "bank_ess",
    "bank_ess_per_case_row",
    "direction_sign",
    "family_B",
    "family_I",
    "family_R",
)

_ENTROPY_EPSILON = 1.0e-12


def _probability_like(values: object, *, length: int, role: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if (
        array.shape != (length,)
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise ProtocolError(f"SCALE-BP {role} vector drifted.")
    return array


def _nonnegative(values: object, *, length: int, role: str) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ProtocolError(f"SCALE-BP {role} vector drifted.")
    return array


def build_action_descriptor(
    projection: BoundaryProjection,
    *,
    case_id: str,
    posterior_eta: object,
    posterior_sd: object,
    seed_sd: object,
    positive_vote_fraction: object,
    support_positive_count: float,
    support_negative_count: float,
    support_row_count: int,
    bank_ess: float,
) -> ActionDescriptor:
    """Build the fixed descriptor without accepting labels or terminal totals."""

    if projection.is_exact_p:
        raise ProtocolError("SCALE-BP cannot describe a structural no-crossing action.")
    baseline = np.asarray(projection.baseline_probabilities, dtype=np.float64)
    projected = np.asarray(projection.projected_probabilities, dtype=np.float64)
    endpoint = np.asarray(projection.full_endpoint_probabilities, dtype=np.float64)
    n_rows = projection.row_count
    eta = _probability_like(posterior_eta, length=n_rows, role="posterior eta")
    eta_sd = _nonnegative(posterior_sd, length=n_rows, role="posterior uncertainty")
    seed = _nonnegative(seed_sd, length=n_rows, role="seed uncertainty")
    votes = _probability_like(
        positive_vote_fraction, length=n_rows, role="positive-vote fraction"
    )
    support_positive = float(support_positive_count)
    support_negative = float(support_negative_count)
    support_rows = int(support_row_count)
    effective_size = float(bank_ess)
    if (
        not math.isfinite(support_positive)
        or not math.isfinite(support_negative)
        or support_positive < 0.0
        or support_negative < 0.0
        or support_rows <= 0
        or abs(support_positive + support_negative - support_rows) > 1.0e-8
        or not math.isfinite(effective_size)
        or effective_size <= 0.0
    ):
        raise ProtocolError("SCALE-BP descriptor support geometry drifted.")

    crossing = projection.crossing_mask()
    eta_crossing = eta[crossing]
    clipped_eta = np.clip(eta_crossing, _ENTROPY_EPSILON, 1.0 - _ENTROPY_EPSILON)
    entropy = -clipped_eta * np.log(clipped_eta) - (
        1.0 - clipped_eta
    ) * np.log(1.0 - clipped_eta)
    support_fraction = support_positive / support_rows
    estimated_held_fraction = float(np.mean(eta, dtype=np.float64))
    direction_sign = 1.0 if projection.direction == "zero_to_one" else -1.0
    family_indicators = tuple(
        1.0 if projection.family == family else 0.0 for family in ("B", "I", "R")
    )
    values = (
        projection.crossing_count / n_rows,
        float(np.mean(np.abs(endpoint[crossing] - baseline[crossing]), dtype=np.float64)),
        float(np.mean(np.abs(projected[crossing] - baseline[crossing]), dtype=np.float64)),
        float(np.mean(np.abs(baseline[crossing] - HARD_THRESHOLD), dtype=np.float64)),
        float(np.mean(np.abs(endpoint[crossing] - HARD_THRESHOLD), dtype=np.float64)),
        float(np.mean(eta_crossing, dtype=np.float64)),
        float(np.mean(entropy, dtype=np.float64)),
        float(np.var(eta_crossing, dtype=np.float64)),
        float(np.mean(eta_sd[crossing], dtype=np.float64)),
        float(np.mean(seed[crossing], dtype=np.float64)),
        float(np.mean(4.0 * votes[crossing] * (1.0 - votes[crossing]), dtype=np.float64)),
        math.log1p(n_rows),
        math.log1p(support_rows),
        support_fraction,
        abs(2.0 * support_fraction - 1.0),
        estimated_held_fraction,
        effective_size,
        effective_size / n_rows,
        direction_sign,
        *family_indicators,
    )
    if len(values) != len(ACTION_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ProtocolError("SCALE-BP descriptor construction produced invalid values.")
    return ActionDescriptor(
        case_id=str(case_id),
        action_id=projection.action_id,
        family=projection.family,
        direction=projection.direction,
        feature_names=ACTION_FEATURE_NAMES,
        values=values,
        crossing_count=projection.crossing_count,
        row_count=n_rows,
        baseline_probability_hash=projection.baseline_probability_hash,
        action_probability_hash=projection.projected_probability_hash,
        endpoint_probability_hash=projection.source_endpoint_hash,
    )


def descriptor_matrix(descriptors: object) -> np.ndarray:
    rows = tuple(descriptors)  # type: ignore[arg-type]
    if not rows or any(not isinstance(row, ActionDescriptor) for row in rows):
        raise ProtocolError("SCALE-BP descriptor matrix input drifted.")
    names = rows[0].feature_names
    if any(row.feature_names != names for row in rows):
        raise ProtocolError("SCALE-BP descriptor feature identity drifted.")
    matrix = np.ascontiguousarray([row.values for row in rows], dtype=np.float64)
    if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
        raise ProtocolError("SCALE-BP descriptor matrix drifted.")
    matrix.setflags(write=False)
    return matrix


__all__ = ("ACTION_FEATURE_NAMES", "build_action_descriptor", "descriptor_matrix")
