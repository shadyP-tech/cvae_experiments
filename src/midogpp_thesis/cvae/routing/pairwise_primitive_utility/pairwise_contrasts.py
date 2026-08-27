"""Canonical pairwise contrast construction and center/case weights."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import ActionQuery, ActionUtilityObservation


@dataclass(frozen=True, slots=True)
class PairwiseContrast:
    center_id: str
    case_id: str
    left: ActionQuery
    right: ActionQuery
    realized_contrast: float


def canonical_observations(
    observations: Sequence[ActionUtilityObservation],
) -> tuple[ActionUtilityObservation, ...]:
    rows = tuple(
        sorted(tuple(observations), key=lambda row: (row.center_id, row.case_id, row.action_id))
    )
    if not rows:
        raise ProtocolError("Pairwise ranking requires source-only action observations.")
    keys = tuple((row.center_id, row.case_id, row.action_id) for row in rows)
    if len(set(keys)) != len(keys):
        raise ProtocolError("Pairwise source observations contain duplicate case/action rows.")
    return rows


def action_schema(
    *surfaces: Sequence[ActionUtilityObservation],
) -> tuple[tuple[str, str, str], ...]:
    by_action: dict[str, tuple[str, str]] = {}
    for rows in surfaces:
        for row in rows:
            value = (row.family, row.direction)
            if row.action_id in by_action and by_action[row.action_id] != value:
                raise ProtocolError("Pairwise action family/direction schema drifted.")
            by_action[row.action_id] = value
    return tuple((action_id, *by_action[action_id]) for action_id in sorted(by_action))


def build_contrasts(
    observations: Sequence[ActionUtilityObservation],
) -> tuple[PairwiseContrast, ...]:
    grouped: dict[tuple[str, str], list[ActionUtilityObservation]] = defaultdict(list)
    for row in observations:
        grouped[(row.center_id, row.case_id)].append(row)
    output: list[PairwiseContrast] = []
    for (center_id, case_id), rows in sorted(grouped.items()):
        queries_and_values: list[tuple[ActionQuery, float]] = [
            (ActionQuery.p_anchor(rows[0].feature_names), 0.0)
        ]
        queries_and_values.extend(
            (
                ActionQuery(
                    row.action_id,
                    row.family,
                    row.direction,
                    row.feature_names,
                    row.feature_values,
                ),
                row.response.bacc_gain,
            )
            for row in sorted(rows, key=lambda value: value.action_id)
        )
        queries_and_values.sort(key=lambda value: value[0].action_id)
        for (left, left_value), (right, right_value) in combinations(queries_and_values, 2):
            output.append(
                PairwiseContrast(
                    center_id=center_id,
                    case_id=case_id,
                    left=left,
                    right=right,
                    realized_contrast=float(left_value - right_value),
                )
            )
    if not output:
        raise ProtocolError("Pairwise ranking produced no candidate-versus-P contrasts.")
    return tuple(output)


def center_case_balanced_contrast_weights(
    contrasts: Sequence[PairwiseContrast],
) -> np.ndarray:
    """Give each center, then case, equal mass regardless of action count."""

    rows = tuple(contrasts)
    if not rows:
        raise ProtocolError("Center-balanced weights require pairwise contrasts.")
    centers = tuple(sorted({row.center_id for row in rows}))
    cases = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    cases_per_center = Counter(center for center, _ in cases)
    pairs_per_case = Counter((row.center_id, row.case_id) for row in rows)
    raw = np.asarray(
        [
            1.0
            / (
                len(centers)
                * cases_per_center[row.center_id]
                * pairs_per_case[(row.center_id, row.case_id)]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return raw * (len(rows) / float(np.sum(raw, dtype=np.float64)))


def action_row_weights(rows: Sequence[ActionUtilityObservation]) -> np.ndarray:
    centers = tuple(sorted({row.center_id for row in rows}))
    cases = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    cases_per_center = Counter(center for center, _ in cases)
    actions_per_case = Counter((row.center_id, row.case_id) for row in rows)
    raw = np.asarray(
        [
            1.0
            / (
                len(centers)
                * cases_per_center[row.center_id]
                * actions_per_case[(row.center_id, row.case_id)]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return raw * (len(rows) / float(np.sum(raw, dtype=np.float64)))


__all__ = ("PairwiseContrast", "center_case_balanced_contrast_weights")
