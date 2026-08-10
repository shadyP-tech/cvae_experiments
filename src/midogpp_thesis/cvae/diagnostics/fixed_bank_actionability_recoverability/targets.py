"""Capability-scoped dense utility targets for donor-only ridge fitting."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    GEOMETRY_IDS,
    PROBABILITY_EPSILON,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from .contracts import BinaryLabelRow, ExactNineProbabilitySurface, UtilityTargetRow


def _negative_log_loss(label: int, probability: float) -> float:
    value = min(max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    return -math.log(value if label == 1 else 1.0 - value)


def build_class_balanced_proper_loss_targets(
    probabilities: ExactNineProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
) -> tuple[UtilityTargetRow, ...]:
    """Build per-case additive proper-loss gain versus U.

    The supplied labels are assumed to come from an explicit donor-label
    capability.  Each response is an additive contribution to its query
    center's class-balanced proper loss, so single-class cases remain legal and
    no per-case BACC is created or stored.
    """

    if not isinstance(probabilities, ExactNineProbabilitySurface):
        raise ProtocolError("Proper-loss targets require a sealed exact-nine surface.")
    label_rows = tuple(labels)
    if not label_rows or any(not isinstance(row, BinaryLabelRow) for row in label_rows):
        raise ProtocolError("Proper-loss targets require typed capability-scoped labels.")
    label_by_key = {row.sample_key: row.label for row in label_rows}
    if len(label_by_key) != len(label_rows):
        raise ProtocolError("Proper-loss label capability contains duplicate rows.")
    included_centers = {key[0] for key in label_by_key}
    probability_rows = tuple(
        row for row in probabilities.rows if row.target_center in included_centers
    )
    expected_sample_keys = {
        (row.target_center, row.case_id, row.sample_id) for row in probability_rows
    }
    if set(label_by_key) != expected_sample_keys:
        raise ProtocolError("Donor-label capability must exactly cover every included center row.")
    probabilities_by_key = {
        (row.target_center, row.case_id, row.sample_id, row.action_id): row.probability_mean
        for row in probability_rows
    }
    samples_by_center_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target, case_id, sample_id in sorted(label_by_key):
        samples_by_center_case[(target, case_id)].append(sample_id)
    class_totals: dict[str, dict[int, int]] = {}
    for center in sorted(included_centers):
        totals = {
            label: sum(value == label for key, value in label_by_key.items() if key[0] == center)
            for label in (0, 1)
        }
        if any(count <= 0 for count in totals.values()):
            raise ProtocolError("Every donor query center needs both pooled classes.")
        class_totals[center] = totals

    output: list[UtilityTargetRow] = []
    for (query, case_id), sample_ids in sorted(samples_by_center_case.items()):
        for geometry in GEOMETRY_IDS:
            for source in candidate_sources(query):
                candidate_action = geometry_action_id(geometry, source)
                gain = 0.0
                for sample_id in sample_ids:
                    label = label_by_key[(query, case_id, sample_id)]
                    weight = 0.5 / class_totals[query][label]
                    uniform = probabilities_by_key[(query, case_id, sample_id, U_ACTION_ID)]
                    candidate = probabilities_by_key[
                        (query, case_id, sample_id, candidate_action)
                    ]
                    gain += weight * (
                        _negative_log_loss(label, uniform)
                        - _negative_log_loss(label, candidate)
                    )
                output.append(
                    UtilityTargetRow(
                        query_center=query,
                        case_id=case_id,
                        geometry_id=geometry,
                        selected_source=source,
                        response=gain,
                        response_kind="class_balanced_proper_loss_gain_vs_u",
                    )
                )
    return tuple(output)


__all__ = ("build_class_balanced_proper_loss_targets",)
