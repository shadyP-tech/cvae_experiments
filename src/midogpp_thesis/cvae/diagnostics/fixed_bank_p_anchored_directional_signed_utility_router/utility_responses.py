"""Scoped direct utility responses and deterministic case-block control."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import HARD_THRESHOLD, LOG_LOSS_CLIP_EPSILON, PORTFOLIO_METHOD_ID
from .contracts import BinaryLabel, EndpointCasePrediction
from .hashing import canonical_hash
from .utility_contracts import DonorUtilityRow, UtilityDescriptor


def build_donor_utility_rows(
    *,
    outer_target_center: str,
    prediction: EndpointCasePrediction,
    descriptors: Sequence[UtilityDescriptor],
    case_labels: Sequence[BinaryLabel],
    center_n_positive: int,
    center_n_negative: int,
) -> tuple[DonorUtilityRow, ...]:
    """Score the exact branch-local action used later on the target case."""

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
        raise ProtocolError("PDSUR donor response label scope drifted.")
    observed = tuple(descriptors)
    if (
        len(observed) != 6
        or len({row.key for row in observed}) != 6
        or any(
            row.target_center != prediction.center
            or row.case_id != prediction.case_id
            or row.endpoint_prediction_hash != prediction.prediction_hash
            for row in observed
        )
    ):
        raise ProtocolError("PDSUR donor descriptor rectangle drifted.")

    y = np.asarray([labels[sample_id].value for sample_id in prediction.sample_ids], dtype=np.int8)
    portfolio = np.asarray(
        prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    p_hard = portfolio >= HARD_THRESHOLD
    p_clipped = np.clip(
        portfolio, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
    )
    p_brier = (portfolio - y) ** 2
    p_log = -(y * np.log(p_clipped) + (1 - y) * np.log1p(-p_clipped))
    n_total = center_n_positive + center_n_negative
    rows: list[DonorUtilityRow] = []
    for descriptor in observed:
        composed, mask = directional_candidate(
            prediction, descriptor.alternative, descriptor.direction
        )
        if tuple(prediction.sample_ids[index] for index in np.flatnonzero(mask)) != descriptor.crossing_sample_ids:
            raise ProtocolError("PDSUR donor descriptor crossing identity drifted.")
        if not np.any(mask):
            bacc_delta = brier_delta = log_delta = 0.0
        else:
            hard = composed >= HARD_THRESHOLD
            positive = y == 1
            negative = ~positive
            bacc_delta = 0.5 * (
                float(
                    np.sum(
                        hard[positive].astype(np.int8)
                        - p_hard[positive].astype(np.int8),
                        dtype=np.int64,
                    )
                )
                / center_n_positive
                + float(
                    np.sum(
                        (~hard[negative]).astype(np.int8)
                        - (~p_hard[negative]).astype(np.int8),
                        dtype=np.int64,
                    )
                )
                / center_n_negative
            )
            clipped = np.clip(
                composed, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
            )
            brier_delta = float(
                np.sum((composed - y) ** 2 - p_brier, dtype=np.float64) / n_total
            )
            log_delta = float(
                np.sum(
                    -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
                    - p_log,
                    dtype=np.float64,
                )
                / n_total
            )
        rows.append(
            DonorUtilityRow(
                outer_target_center,
                prediction.center,
                prediction.case_id,
                descriptor.alternative,
                descriptor.direction,
                descriptor.feature_values,
                descriptor.crossing_count,
                float(bacc_delta),
                float(brier_delta),
                float(log_delta),
                descriptor.descriptor_hash,
            )
        )
    result = tuple(sorted(rows, key=lambda row: row.key))
    if len(result) != 6 or len({row.key for row in result}) != 6:
        raise ProtocolError("PDSUR donor utility rows are not rectangular.")
    return result


def blocked_feature_permutation(
    rows: Sequence[DonorUtilityRow],
) -> tuple[DonorUtilityRow, ...]:
    """Rotate whole case descriptors within donor/action/direction strata."""

    source = tuple(rows)
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source):
        strata[(row.donor_center, row.alternative, row.direction)].append(index)
    output = list(source)
    for key in sorted(strata):
        indices = sorted(strata[key], key=lambda index: source[index].case_id)
        if len(indices) <= 1:
            continue
        shift = 1 + int(canonical_hash({"case_block_stratum": list(key)})[:8], 16) % (len(indices) - 1)
        for position, target_index in enumerate(indices):
            source_index = indices[(position + shift) % len(indices)]
            feature_source = source[source_index]
            original = source[target_index]
            output[target_index] = DonorUtilityRow(
                original.outer_target_center,
                original.donor_center,
                original.case_id,
                original.alternative,
                original.direction,
                feature_source.feature_values,
                # crossing_count belongs to the response geometry.  The
                # blocked control permutes only model inputs; moving this
                # auxiliary response invariant can pair a structural-zero
                # feature row with a nonzero original response.
                original.crossing_count,
                original.bacc_contribution_delta,
                original.brier_contribution_delta,
                original.log_loss_contribution_delta,
                canonical_hash(
                    {
                        "schema_version": "fixed_bank_pdsur_case_blocked_feature_control_v1",
                        "response_descriptor_hash": original.descriptor_hash,
                        "feature_descriptor_hash": feature_source.descriptor_hash,
                        "stratum": list(key),
                        "shift": shift,
                    }
                ),
            )
    return tuple(output)


__all__ = ("blocked_feature_permutation", "build_donor_utility_rows")
