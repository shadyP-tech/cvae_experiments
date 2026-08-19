"""Scoped donor responses and a deterministic blocked feature control."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import HARD_THRESHOLD, LOG_LOSS_CLIP_EPSILON, PORTFOLIO_METHOD_ID
from .contracts import BinaryLabel, EndpointCasePrediction
from .crossing_contracts import CrossingDescriptor, DonorCrossingRow
from .hashing import canonical_hash


def build_donor_crossing_rows(
    *,
    outer_target_center: str,
    prediction: EndpointCasePrediction,
    descriptors: Sequence[CrossingDescriptor],
    case_labels: Sequence[BinaryLabel],
    center_n_positive: int,
    center_n_negative: int,
) -> tuple[DonorCrossingRow, ...]:
    """Open labels only for a legal donor center and score actionable rows."""

    label_rows = tuple(case_labels)
    labels = {row.sample_id: row for row in label_rows}
    expected_scope = (
        f"crossing_donor::outer_H={outer_target_center}::donor_J={prediction.center}"
    )
    if (
        outer_target_center == prediction.center
        or not labels
        or len(labels) != len(label_rows)
        or set(labels) != set(prediction.sample_ids)
        or {row.center for row in labels.values()} != {prediction.center}
        or {row.case_id for row in labels.values()} != {prediction.case_id}
        or {row.scope for row in labels.values()} != {expected_scope}
        or center_n_positive <= 0
        or center_n_negative <= 0
    ):
        raise ProtocolError("PDCB donor response label scope drifted.")
    index = {sample_id: ordinal for ordinal, sample_id in enumerate(prediction.sample_ids)}
    portfolio = np.asarray(
        prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    rows: list[DonorCrossingRow] = []
    for descriptor in descriptors:
        if (
            descriptor.target_center != prediction.center
            or descriptor.case_id != prediction.case_id
            or descriptor.endpoint_prediction_hash != prediction.prediction_hash
            or descriptor.sample_id not in labels
        ):
            raise ProtocolError("PDCB donor descriptor/label alignment drifted.")
        ordinal = index[descriptor.sample_id]
        y = labels[descriptor.sample_id].value
        p_value = float(portfolio[ordinal])
        a_value = float(prediction.probabilities[descriptor.alternative][ordinal])
        p_hard = p_value >= HARD_THRESHOLD
        a_hard = a_value >= HARD_THRESHOLD
        if p_hard == a_hard:
            raise ProtocolError("PDCB donor response contains a noncrossing row.")
        helpful = int(a_hard == bool(y))
        if y == 1:
            bacc_delta = 0.5 * (float(a_hard) - float(p_hard)) / center_n_positive
        else:
            bacc_delta = 0.5 * (float(not a_hard) - float(not p_hard)) / center_n_negative
        clipped_p = min(max(p_value, LOG_LOSS_CLIP_EPSILON), 1.0 - LOG_LOSS_CLIP_EPSILON)
        clipped_a = min(max(a_value, LOG_LOSS_CLIP_EPSILON), 1.0 - LOG_LOSS_CLIP_EPSILON)
        p_loss = -(y * np.log(clipped_p) + (1 - y) * np.log1p(-clipped_p))
        a_loss = -(y * np.log(clipped_a) + (1 - y) * np.log1p(-clipped_a))
        rows.append(
            DonorCrossingRow(
                outer_target_center,
                prediction.center,
                prediction.case_id,
                descriptor.sample_id,
                descriptor.alternative,
                descriptor.direction,
                descriptor.feature_values,
                helpful,
                float(bacc_delta),
                float(a_loss - p_loss),
                descriptor.descriptor_hash,
            )
        )
    result = tuple(sorted(rows, key=lambda row: row.key))
    if len({row.key for row in result}) != len(result):
        raise ProtocolError("PDCB donor response rows are duplicated.")
    return result


def blocked_feature_permutation(
    rows: Sequence[DonorCrossingRow],
) -> tuple[DonorCrossingRow, ...]:
    """Shift complete, equal-size case blocks within donor/action/direction strata."""

    source = tuple(rows)
    case_blocks: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source):
        case_blocks[
            (row.donor_center, row.alternative, row.direction, row.case_id)
        ].append(index)
    strata: dict[tuple[str, str, str, int], list[tuple[str, list[int]]]] = defaultdict(list)
    for (donor, alternative, direction, case), indices in case_blocks.items():
        strata[(donor, alternative, direction, len(indices))].append(
            (case, sorted(indices, key=lambda index: source[index].sample_id))
        )
    output = list(source)
    for key in sorted(strata):
        blocks = sorted(strata[key], key=lambda value: value[0])
        if len(blocks) <= 1:
            continue
        shift = 1 + int(canonical_hash({"case_block_stratum": list(key)})[:8], 16) % (len(blocks) - 1)
        for position, (target_case, target_indices) in enumerate(blocks):
            source_case, source_indices = blocks[(position + shift) % len(blocks)]
            for target_index, source_index in zip(target_indices, source_indices, strict=True):
                feature_source = source[source_index]
                original = source[target_index]
                output[target_index] = DonorCrossingRow(
                    original.outer_target_center,
                    original.donor_center,
                    original.case_id,
                    original.sample_id,
                    original.alternative,
                    original.direction,
                    feature_source.feature_values,
                    original.helpful,
                    original.bacc_contribution_delta,
                    original.log_loss_delta,
                    canonical_hash(
                        {
                            "schema_version": "fixed_bank_pdcb_case_blocked_feature_control_v1",
                            "response_descriptor_hash": original.descriptor_hash,
                            "feature_descriptor_hash": feature_source.descriptor_hash,
                            "target_case_id": target_case,
                            "feature_case_id": source_case,
                            "stratum": list(key),
                            "shift": shift,
                        }
                    ),
                )
    return tuple(output)


__all__ = ("blocked_feature_permutation", "build_donor_crossing_rows")
