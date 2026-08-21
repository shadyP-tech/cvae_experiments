"""Exact center-normalized endpoint responses for emitted action vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import LOG_LOSS_CLIP_EPSILON, PORTFOLIO_METHOD_ID
from .contracts import BinaryLabel, EndpointCasePrediction
from .projected_contracts import ProjectedDonorUtilityRow, ProjectedUtilityDescriptor
from .projection import ActionEquivalenceClass
from .projection_lattice import THRESHOLD, as_binary32


def build_projected_donor_rows(
    *,
    outer_target_center: str,
    prediction: EndpointCasePrediction,
    actions: Sequence[ActionEquivalenceClass],
    descriptors: Sequence[ProjectedUtilityDescriptor],
    case_labels: Sequence[BinaryLabel],
    center_n_positive: int,
    center_n_negative: int,
) -> tuple[ProjectedDonorUtilityRow, ...]:
    labels = {row.sample_id: row for row in case_labels}
    if (
        outer_target_center == prediction.center
        or set(labels) != set(prediction.sample_ids)
        or len(labels) != len(tuple(case_labels))
        or center_n_positive <= 0
        or center_n_negative <= 0
        or {row.center for row in labels.values()} != {prediction.center}
        or {row.case_id for row in labels.values()} != {prediction.case_id}
        or any(
            not row.scope.startswith(
                f"utility_donor::outer_H={outer_target_center}::donor_J={prediction.center}"
            )
            for row in labels.values()
        )
    ):
        raise ProtocolError("PCSI-RACR donor-response capability drifted.")
    action_by_hash: Mapping[str, ActionEquivalenceClass] = {
        row.action_hash: row for row in actions
    }
    rows = tuple(descriptors)
    if len(action_by_hash) != len(tuple(actions)) or set(action_by_hash) != {row.action_hash for row in rows}:
        raise ProtocolError("PCSI-RACR response action/descriptor surface drifted.")

    y = np.asarray([labels[sample].value for sample in prediction.sample_ids], dtype=np.int8)
    portfolio = as_binary32(prediction.probabilities[PORTFOLIO_METHOD_ID], name="response P").astype(np.float64)
    p_hard = portfolio >= float(THRESHOLD)
    p_clipped = np.clip(portfolio, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    p_brier = (portfolio - y) ** 2
    p_log = -(y * np.log(p_clipped) + (1 - y) * np.log1p(-p_clipped))
    n_total = center_n_positive + center_n_negative
    output: list[ProjectedDonorUtilityRow] = []
    for descriptor in rows:
        action = action_by_hash[descriptor.action_hash]
        emitted = as_binary32(action.probabilities, name="response action").astype(np.float64)
        hard = emitted >= float(THRESHOLD)
        crossing = hard != p_hard
        if int(np.sum(crossing, dtype=np.int64)) != descriptor.crossing_count:
            raise ProtocolError("PCSI-RACR response crossing count drifted.")
        if descriptor.crossing_count == 0:
            bacc_delta = brier_delta = log_delta = 0.0
        else:
            positive = y == 1
            negative = ~positive
            bacc_delta = 0.5 * (
                float(np.sum(hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8), dtype=np.int64)) / center_n_positive
                + float(np.sum((~hard[negative]).astype(np.int8) - (~p_hard[negative]).astype(np.int8), dtype=np.int64)) / center_n_negative
            )
            clipped = np.clip(emitted, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
            brier_delta = float(np.sum((emitted - y) ** 2 - p_brier, dtype=np.float64) / n_total)
            log_delta = float(
                np.sum(
                    -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)) - p_log,
                    dtype=np.float64,
                )
                / n_total
            )
        output.append(
            ProjectedDonorUtilityRow(
                outer_target_center,
                prediction.center,
                prediction.case_id,
                descriptor.geometry_id,
                descriptor.direction,
                descriptor.representative,
                descriptor.feature_values,
                descriptor.crossing_count,
                float(bacc_delta),
                float(brier_delta),
                float(log_delta),
                descriptor.descriptor_hash,
            )
        )
    result = tuple(sorted(output, key=lambda row: row.key))
    if len(result) != len(rows):
        raise ProtocolError("PCSI-RACR donor response count drifted.")
    return result


__all__ = ("build_projected_donor_rows",)
