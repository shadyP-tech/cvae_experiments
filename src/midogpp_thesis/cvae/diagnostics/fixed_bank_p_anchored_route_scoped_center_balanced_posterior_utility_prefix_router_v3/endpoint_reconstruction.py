"""Thin facade for independent B/I/R/P endpoint reconstruction."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    DIRECTION_IDS,
    ENDPOINT_METHOD_IDS,
    HARD_THRESHOLD,
    IDENTIFICATION_METHOD_ID,
    PORTFOLIO_IDENTIFICATION_WEIGHT,
    PORTFOLIO_METHOD_ID,
    PORTFOLIO_ROBUST_WEIGHT,
    ROBUST_METHOD_ID,
    a1_action_id,
    candidate_sources,
)
from .contracts import BinaryLabel, EndpointCasePrediction
from .endpoint_fitting import (
    EndpointState,
    _predict_irls,
    fit_endpoint_state_from_outcomes,
    rebind_endpoint_state_priors,
)
from .endpoint_preparation import (
    CenterCaseOutcomes,
    PreparedCenter,
    build_center_case_outcomes,
    compute_donor_priors,
    prepare_center,
)

def fit_endpoint_state(
    prepared: PreparedCenter,
    *,
    support_case_ids: Sequence[str],
    scoped_labels: Sequence[BinaryLabel],
    donor_priors: Mapping[tuple[str, str], float],
    excluded_source_centers: Sequence[str] = (),
    expected_label_scope: str | None = None,
) -> EndpointState:
    """Compatibility facade that keeps raw labels at the capability boundary."""

    cases = tuple(sorted(str(value) for value in support_case_ids))
    excluded_cases = set(prepared.cases).difference(cases)
    if len(excluded_cases) != 1:
        raise ProtocolError("Endpoint state must exclude exactly one whole case.")
    held = next(iter(excluded_cases))
    scope = (
        f"outer_support::H={prepared.surface.center}::excluded_c={held}"
        if expected_label_scope is None
        else str(expected_label_scope)
    )
    outcomes = build_center_case_outcomes(
        prepared, scoped_labels, expected_scope=scope
    )
    if outcomes.case_ids != cases:
        raise ProtocolError("Endpoint raw-label capability contains extra cases.")
    return fit_endpoint_state_from_outcomes(
        prepared,
        support_case_ids=cases,
        outcomes=outcomes,
        donor_priors=donor_priors,
        excluded_source_centers=excluded_source_centers,
    )


def reconstruct_case_endpoints(
    prepared: PreparedCenter,
    state: EndpointState,
    *,
    evaluation_case_id: object,
) -> EndpointCasePrediction:
    case = str(evaluation_case_id)
    if case in state.support_case_ids or state.target_center != prepared.surface.center:
        raise ProtocolError("Endpoint evaluation case was not excluded from its state.")
    positions = prepared.case_positions[case]
    sources = candidate_sources(prepared.surface.center)
    if not state.allowed_sources or not set(state.allowed_sources) <= set(sources):
        raise ProtocolError("Endpoint allowed-source contract drifted.")
    case_index = prepared.cases.index(case)
    selected_i: dict[str, str | None] = {}
    for direction_index, direction in enumerate(DIRECTION_IDS):
        intermediate: list[tuple[str, float, float, bool]] = []
        for source_index, source in enumerate(sources):
            if source not in state.allowed_sources:
                continue
            predicted = _predict_irls(
                state.model_mean[source_index, direction_index],
                state.model_scale[source_index, direction_index],
                state.model_coefficients[source_index, direction_index],
                bool(state.model_valid[source_index, direction_index]),
                prepared.feature_values[case_index, source_index, direction_index],
            )
            flips = int(prepared.directional_flip_counts[case_index, source_index, direction_index])
            if predicted is None:
                proxy = 0.0
            elif direction == "zero_to_one":
                proxy = (
                    flips * predicted / (2 * state.support_n_positive)
                    - flips * (1.0 - predicted) / (2 * state.support_n_negative)
                )
            else:
                proxy = (
                    flips * predicted / (2 * state.support_n_negative)
                    - flips * (1.0 - predicted) / (2 * state.support_n_positive)
                )
            intermediate.append(
                (
                    source,
                    float(proxy),
                    float(state.donor_priors[(source, direction)]),
                    bool(flips > 0 and predicted is not None and proxy > 0.0),
                )
            )
        case_scale = float(np.mean([abs(row[1]) for row in intermediate], dtype=np.float64))
        donor_scale = float(np.mean([abs(row[2]) for row in intermediate], dtype=np.float64))
        scores = [
            (
                source,
                0.8 * (proxy / case_scale)
                + 0.2 * (0.0 if donor_scale == 0.0 else prior / donor_scale),
            )
            for source, proxy, prior, eligible in intermediate
            if eligible and case_scale > 0.0
        ]
        if not scores or max(value for _, value in scores) <= 1.0e-12:
            selected_i[direction] = None
        else:
            maximum = max(value for _, value in scores)
            selected_i[direction] = min(
                (source for source, value in scores if maximum - value <= 1.0e-12),
                key=int,
            )
    baseline = prepared.action_means[B_ACTION_ID][positions].astype(np.float64, copy=True)
    baseline_hard = baseline >= HARD_THRESHOLD
    identification = baseline.copy()
    for branch, direction in ((False, "zero_to_one"), (True, "one_to_zero")):
        source = selected_i[direction]
        mask = baseline_hard == branch
        if source is not None:
            identification[mask] = prepared.action_means[a1_action_id(source)][positions][mask]
    arm_probabilities: list[np.ndarray] = []
    for zero_source, one_source in state.robust_sources:
        values = baseline.copy()
        for branch, source in ((False, zero_source), (True, one_source)):
            mask = baseline_hard == branch
            if source is not None:
                values[mask] = prepared.action_means[a1_action_id(source)][positions][mask]
        arm_probabilities.append(values)
    robust = np.mean(np.stack(arm_probabilities), axis=0, dtype=np.float64)
    portfolio = (
        PORTFOLIO_IDENTIFICATION_WEIGHT * identification
        + PORTFOLIO_ROBUST_WEIGHT * robust
    )
    # This is the single canonicalization boundary for every reconstructed
    # endpoint.  In particular, the P surface consumed by composition must be
    # the same float32 bytes as the P surface persisted and scored at the
    # terminal phase; otherwise an exact-P abstention could look like a routed
    # case after an implicit float64-to-float32 round trip.
    probabilities = MappingProxyType(
        {
            method: tuple(
                float(value)
                for value in np.ascontiguousarray(values, dtype=np.float32)
            )
            for method, values in (
                (ENDPOINT_METHOD_IDS[0], baseline),
                (IDENTIFICATION_METHOD_ID, identification),
                (ROBUST_METHOD_ID, robust),
                (PORTFOLIO_METHOD_ID, portfolio),
            )
        }
    )
    return EndpointCasePrediction(
        prepared.surface.center,
        case,
        tuple(prepared.surface.sample_ids[position] for position in positions),
        probabilities,
        state.state_hash,
    )


__all__ = (
    "CenterCaseOutcomes",
    "EndpointState",
    "PreparedCenter",
    "build_center_case_outcomes",
    "compute_donor_priors",
    "fit_endpoint_state",
    "fit_endpoint_state_from_outcomes",
    "prepare_center",
    "rebind_endpoint_state_priors",
    "reconstruct_case_endpoints",
)
