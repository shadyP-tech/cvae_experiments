"""Nested support nomination and additive donor-regret targets."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ENDPOINT_METHOD_IDS,
    ENDPOINT_ORDER,
    HARD_THRESHOLD,
    LOG_LOSS_CLIP_EPSILON,
    PORTFOLIO_METHOD_ID,
    REGRET_FEATURE_NAMES,
)
from .contracts import (
    BinaryLabel,
    CandidateDescriptor,
    DonorRegretRow,
    EndpointCasePrediction,
)


def _choose_endpoint(scores: Mapping[str, float]) -> str:
    if tuple(scores) != ENDPOINT_METHOD_IDS:
        raise ProtocolError("Endpoint nomination score order drifted.")
    return min(
        ENDPOINT_METHOD_IDS,
        key=lambda method: (
            -float(scores[method]),
            0 if method == PORTFOLIO_METHOD_ID else 1,
            ENDPOINT_ORDER[method],
        ),
    )


def _case_label_map(
    prediction: EndpointCasePrediction,
    labels: Sequence[BinaryLabel],
    *,
    expected_case_id: str,
) -> dict[str, int]:
    rows = tuple(labels)
    result = {row.sample_id: row.value for row in rows}
    if (
        len(result) != len(rows)
        or {row.center for row in rows} != {prediction.center}
        or {row.case_id for row in rows} != {expected_case_id}
        or set(result) != set(prediction.sample_ids)
    ):
        raise ProtocolError("Endpoint labels do not match one exact whole case.")
    return result


def _confusion_contribution(
    prediction: EndpointCasePrediction,
    labels: Mapping[str, int],
    method: str,
    *,
    support_n_positive: int,
    support_n_negative: int,
) -> float:
    truth = np.asarray([labels[sample] for sample in prediction.sample_ids], dtype=np.int8)
    probability = np.asarray(prediction.probabilities[method], dtype=np.float64)
    hard = probability >= HARD_THRESHOLD
    return float(
        0.5
        * (
            np.sum((truth == 1) & hard, dtype=np.int64) / support_n_positive
            + np.sum((truth == 0) & (~hard), dtype=np.int64) / support_n_negative
        )
    )


def build_candidate_descriptor(
    *,
    target_center: object,
    outer_case_id: object,
    outer_prediction: EndpointCasePrediction,
    nested_voter_predictions: Mapping[str, EndpointCasePrediction],
    support_labels: Sequence[BinaryLabel],
) -> CandidateDescriptor:
    """Nominate one endpoint from genuinely double-excluded voter states."""

    target, outer = str(target_center), str(outer_case_id)
    voters = tuple(sorted(str(value) for value in nested_voter_predictions))
    rows = tuple(support_labels)
    if (
        outer_prediction.center != target
        or outer_prediction.case_id != outer
        or not voters
        or any(
            nested_voter_predictions[voter].center != target
            or nested_voter_predictions[voter].case_id != voter
            for voter in voters
        )
        or {row.center for row in rows} != {target}
        or {row.case_id for row in rows} != set(voters)
        or outer in {row.case_id for row in rows}
        or len({row.scope for row in rows}) != 1
    ):
        raise ProtocolError("Nested descriptor support topology drifted.")
    support_n_positive = sum(row.value == 1 for row in rows)
    support_n_negative = sum(row.value == 0 for row in rows)
    if not support_n_positive or not support_n_negative:
        raise ProtocolError("Nested descriptor support lacks both classes.")
    labels_by_case = {
        voter: _case_label_map(
            nested_voter_predictions[voter],
            tuple(row for row in rows if row.case_id == voter),
            expected_case_id=voter,
        )
        for voter in voters
    }
    contributions = {
        (voter, method): _confusion_contribution(
            nested_voter_predictions[voter],
            labels_by_case[voter],
            method,
            support_n_positive=support_n_positive,
            support_n_negative=support_n_negative,
        )
        for voter in voters
        for method in ENDPOINT_METHOD_IDS
    }
    totals = {
        method: float(sum(contributions[(voter, method)] for voter in voters))
        for method in ENDPOINT_METHOD_IDS
    }
    alternative = _choose_endpoint(totals)
    nested_hashes = tuple(
        nested_voter_predictions[voter].prediction_hash for voter in voters
    )
    if alternative == PORTFOLIO_METHOD_ID:
        return CandidateDescriptor(
            target,
            outer,
            alternative,
            REGRET_FEATURE_NAMES,
            (0.0,) * len(REGRET_FEATURE_NAMES),
            nested_hashes,
        )
    portfolio = np.asarray(
        outer_prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    candidate = np.asarray(
        outer_prediction.probabilities[alternative], dtype=np.float64
    )
    portfolio_hard = portfolio >= HARD_THRESHOLD
    candidate_hard = candidate >= HARD_THRESHOLD
    crossing = portfolio_hard != candidate_hard
    up = (~portfolio_hard) & candidate_hard
    down = portfolio_hard & (~candidate_hard)
    shift = candidate - portfolio

    def masked_mean(values: np.ndarray) -> float:
        return float(np.mean(values[crossing], dtype=np.float64)) if np.any(crossing) else 0.0

    voter_regrets = np.asarray(
        [
            contributions[(voter, alternative)]
            - contributions[(voter, PORTFOLIO_METHOD_ID)]
            for voter in voters
        ],
        dtype=np.float64,
    )
    winners = [
        _choose_endpoint(
            {
                method: contributions[(voter, method)]
                for method in ENDPOINT_METHOD_IDS
            }
        )
        for voter in voters
    ]
    dispersion = (
        100.0
        * float(np.std(voter_regrets, ddof=1, dtype=np.float64))
        * math.sqrt(len(voter_regrets))
        if len(voter_regrets) > 1
        else 0.0
    )
    values = (
        100.0 * (totals[alternative] - totals[PORTFOLIO_METHOD_ID]),
        winners.count(alternative) / len(winners),
        dispersion,
        float(np.mean(crossing, dtype=np.float64)),
        float(np.mean(up, dtype=np.float64) - np.mean(down, dtype=np.float64)),
        float(np.mean(np.abs(shift), dtype=np.float64)),
        float(np.mean(shift, dtype=np.float64)),
        masked_mean(np.abs(portfolio - HARD_THRESHOLD)),
        masked_mean(np.abs(candidate - HARD_THRESHOLD)),
        float(alternative == "B"),
        float(alternative == "I_OPPORTUNITY_GATED"),
        float(np.any(crossing)),
    )
    return CandidateDescriptor(
        target,
        outer,
        alternative,
        REGRET_FEATURE_NAMES,
        values,
        nested_hashes,
    )


def build_donor_regret_row(
    descriptor: CandidateDescriptor,
    outer_prediction: EndpointCasePrediction,
    case_labels: Sequence[BinaryLabel],
    *,
    center_case_count: int,
    center_n_positive: int,
    center_n_negative: int,
    center_sample_count: int,
) -> DonorRegretRow:
    """Build paired additive responses whose center mean/sum has fixed meaning."""

    if descriptor.case_id != outer_prediction.case_id:
        raise ProtocolError("Donor descriptor and prediction case drifted.")
    labels = _case_label_map(
        outer_prediction,
        case_labels,
        expected_case_id=outer_prediction.case_id,
    )
    if min(
        center_case_count,
        center_n_positive,
        center_n_negative,
        center_sample_count,
    ) <= 0:
        raise ProtocolError("Donor center denominators must be positive.")
    truth = np.asarray(
        [labels[sample] for sample in outer_prediction.sample_ids], dtype=np.int8
    )
    if not descriptor.is_candidate:
        # Keep every donor case in the center-balanced training surface.  A
        # case without an actionable non-P candidate is the protected-P policy
        # and has exact zero paired regret on both response scales.
        return DonorRegretRow(
            descriptor.target_center,
            descriptor.case_id,
            descriptor.alternative,
            descriptor.values,
            0.0,
            0.0,
            int(center_case_count),
            descriptor.descriptor_hash,
        )
    alternative = np.asarray(
        outer_prediction.probabilities[descriptor.alternative], dtype=np.float64
    )
    portfolio = np.asarray(
        outer_prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    alt_hard = alternative >= HARD_THRESHOLD
    p_hard = portfolio >= HARD_THRESHOLD
    bacc_regret = center_case_count * 0.5 * (
        (
            np.sum((truth == 1) & alt_hard, dtype=np.int64)
            - np.sum((truth == 1) & p_hard, dtype=np.int64)
        )
        / center_n_positive
        + (
            np.sum((truth == 0) & (~alt_hard), dtype=np.int64)
            - np.sum((truth == 0) & (~p_hard), dtype=np.int64)
        )
        / center_n_negative
    )
    epsilon = LOG_LOSS_CLIP_EPSILON
    alt_clipped = np.clip(alternative, epsilon, 1.0 - epsilon)
    p_clipped = np.clip(portfolio, epsilon, 1.0 - epsilon)
    alt_loss = -(truth * np.log(alt_clipped) + (1 - truth) * np.log(1.0 - alt_clipped))
    p_loss = -(truth * np.log(p_clipped) + (1 - truth) * np.log(1.0 - p_clipped))
    log_loss_delta = center_case_count * float(
        np.sum(alt_loss - p_loss, dtype=np.float64) / center_sample_count
    )
    return DonorRegretRow(
        descriptor.target_center,
        descriptor.case_id,
        descriptor.alternative,
        descriptor.values,
        float(bacc_regret),
        log_loss_delta,
        int(center_case_count),
        descriptor.descriptor_hash,
    )


__all__ = ("build_candidate_descriptor", "build_donor_regret_row")
