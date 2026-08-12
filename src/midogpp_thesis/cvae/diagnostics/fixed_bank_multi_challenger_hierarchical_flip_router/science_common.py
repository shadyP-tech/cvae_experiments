"""Shared indexing and label-to-direction sufficient statistics."""

from __future__ import annotations

from collections.abc import Iterator
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import (
    CaseActionFeatures,
    CaseConfusion,
    ContributionTarget,
    case_confusion,
    contribution_target,
)
from .constants import B_ACTION_ID, FEATURE_NAMES
from .hashing import canonical_hash


class ProbabilityIndex(Mapping[tuple[str, str, str, str], object]):
    """Exact cell map plus O(1) case/action slices for workstation replay."""

    def __init__(self, rows: Sequence[object]) -> None:
        cells = {row.key: row for row in rows}
        if len(cells) != len(rows):
            raise ProtocolError("Multi-challenger probability surface has duplicates.")
        grouped: dict[tuple[str, str, str], list[object]] = {}
        for row in rows:
            grouped.setdefault(
                (str(row.target_center), str(row.case_id), str(row.action_id)), []
            ).append(row)
        self._cells = MappingProxyType(cells)
        self._case_actions = MappingProxyType(
            {
                key: tuple(sorted(value, key=lambda item: str(item.sample_id)))
                for key, value in grouped.items()
            }
        )

    def __getitem__(self, key: tuple[str, str, str, str]) -> object:
        return self._cells[key]

    def __iter__(self) -> Iterator[tuple[str, str, str, str]]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def rows_for_case_action(
        self, target_center: str, case_id: str, action_id: str
    ) -> tuple[object, ...]:
        return self._case_actions.get((target_center, case_id, action_id), ())


def probability_index(surface: object) -> ProbabilityIndex:
    rows = tuple(getattr(surface, "rows"))
    return ProbabilityIndex(rows)


def feature_index(prelabel: object) -> Mapping[tuple[str, str, str], object]:
    rows = tuple(getattr(prelabel, "features"))
    result = {row.key: row for row in rows}
    if len(result) != len(rows):
        raise ProtocolError("Multi-challenger feature surface has duplicates.")
    return result


def label_index(labels: Sequence[object]) -> Mapping[tuple[str, str, str], int]:
    result = {
        (str(row.target_center), str(row.case_id), str(row.sample_id)): int(row.value)
        for row in labels
    }
    if len(result) != len(labels):
        raise ProtocolError("Multi-challenger label capability has duplicates.")
    return result


def label_surface_hash(labels: Sequence[object]) -> str:
    return canonical_hash(
        [
            {
                "target_center": str(row.target_center),
                "case_id": str(row.case_id),
                "sample_id": str(row.sample_id),
                "value": int(row.value),
            }
            for row in labels
        ]
    )


def core_feature(row: object) -> CaseActionFeatures:
    values = tuple(float(value) for value in row.values)
    if len(values) != len(FEATURE_NAMES):
        raise ProtocolError("Multi-challenger feature geometry drifted.")
    return CaseActionFeatures(
        target_center=str(row.target_center),
        case_id=str(row.case_id),
        action_id=str(row.action_id),
        candidate_source=str(row.selected_source),
        feature_names=FEATURE_NAMES,
        values=values,
        flip_0to1_count=int(round(values[0])),
        flip_1to0_count=int(round(values[2])),
    )


def cases_for_center(partition: object, target: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row.case_id)
                for row in partition.identities
                if str(row.target_center) == target
            }
        )
    )


def case_rows(
    probability: Mapping[tuple[str, str, str, str], object],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> tuple[object, ...]:
    if isinstance(probability, ProbabilityIndex):
        rows = probability.rows_for_case_action(target_center, case_id, action_id)
    else:
        rows = tuple(
            sorted(
                (
                    row
                    for key, row in probability.items()
                    if key[0] == target_center
                    and key[1] == case_id
                    and key[3] == action_id
                ),
                key=lambda row: str(row.sample_id),
            )
        )
    if not rows:
        raise ProtocolError("Multi-challenger case/action rows are absent.")
    return rows


def case_contribution(
    probability: Mapping[tuple[str, str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> ContributionTarget:
    baseline = case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=B_ACTION_ID,
    )
    candidate = case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=action_id,
    )
    sample_ids = tuple(str(row.sample_id) for row in baseline)
    if sample_ids != tuple(str(row.sample_id) for row in candidate):
        raise ProtocolError("Multi-challenger B/action sample identity drifted.")
    try:
        truth = tuple(
            labels[(target_center, case_id, sample_id)] for sample_id in sample_ids
        )
    except KeyError as exc:
        raise ProtocolError("Multi-challenger contribution lacks a scoped label.") from exc
    return contribution_target(
        case_id=case_id,
        action_id=action_id,
        baseline_probabilities=tuple(float(row.probability_mean) for row in baseline),
        action_probabilities=tuple(float(row.probability_mean) for row in candidate),
        labels=truth,
    )


def direction_counts(
    probability: Mapping[tuple[str, str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> Mapping[str, tuple[int, int]]:
    """Return (beneficial, trials) for 0->1 and 1->0 hard flips."""

    baseline = case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=B_ACTION_ID,
    )
    candidate = case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=action_id,
    )
    result = {"0to1": [0, 0], "1to0": [0, 0]}
    for left, right in zip(baseline, candidate, strict=True):
        baseline_hard = int(float(left.probability_mean) >= 0.5)
        candidate_hard = int(float(right.probability_mean) >= 0.5)
        if baseline_hard == candidate_hard:
            continue
        direction = "0to1" if baseline_hard == 0 else "1to0"
        truth = labels[(target_center, case_id, str(left.sample_id))]
        result[direction][1] += 1
        result[direction][0] += int(candidate_hard == truth)
    return {key: (value[0], value[1]) for key, value in result.items()}


def case_confusion_for_action(
    probability: Mapping[tuple[str, str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> CaseConfusion:
    rows = case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=action_id,
    )
    truth = tuple(
        labels[(target_center, case_id, str(row.sample_id))] for row in rows
    )
    predictions = tuple(int(float(row.probability_mean) >= 0.5) for row in rows)
    return case_confusion(case_id, truth, predictions)


__all__ = (
    "case_confusion_for_action",
    "case_contribution",
    "case_rows",
    "cases_for_center",
    "core_feature",
    "direction_counts",
    "feature_index",
    "label_index",
    "label_surface_hash",
    "ProbabilityIndex",
    "probability_index",
)
