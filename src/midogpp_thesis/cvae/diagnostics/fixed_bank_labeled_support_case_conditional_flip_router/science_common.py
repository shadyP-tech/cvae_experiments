"""Shared indexing, case-surface, and protocol helpers for science phases."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import (
    CaseActionFeatures,
    CaseConfusion,
    ContributionTarget,
    case_confusion,
    contribution_target,
)
from .constants import B_ACTION_ID, CENTERS, FEATURE_NAMES
from .hashing import canonical_hash


def _probability_index(surface: object) -> Mapping[tuple[str, str, str, str], object]:
    rows = tuple(getattr(surface, "rows"))
    result = {row.key: row for row in rows}
    if len(result) != len(rows):
        raise ProtocolError("Probability surface contains duplicate keys.")
    return result


def _feature_index(prelabel: object) -> Mapping[tuple[str, str, str], object]:
    rows = tuple(getattr(prelabel, "features"))
    result = {row.key: row for row in rows}
    if len(result) != len(rows):
        raise ProtocolError("Feature surface contains duplicate keys.")
    return result


def _label_index(labels: Sequence[object]) -> Mapping[tuple[str, str, str], int]:
    result = {
        (str(row.target_center), str(row.case_id), str(row.sample_id)): int(row.value)
        for row in labels
    }
    if len(result) != len(labels):
        raise ProtocolError("Label capability contains duplicate identities.")
    return result


def _label_surface_hash(labels: Sequence[object]) -> str:
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


def _core_feature(row: object) -> CaseActionFeatures:
    values = tuple(float(value) for value in row.values)
    if len(values) != len(FEATURE_NAMES):
        raise ProtocolError("Flip-router feature geometry drifted at core boundary.")
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


def _cases_for_center(partition: object, target: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row.case_id)
                for row in partition.identities
                if str(row.target_center) == target
            }
        )
    )


def _case_rows(
    probability: Mapping[tuple[str, str, str, str], object],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> tuple[object, ...]:
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
        raise ProtocolError("Case/action probability rows are absent.")
    return rows


def _case_contribution(
    probability: Mapping[tuple[str, str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> ContributionTarget:
    baseline = _case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=B_ACTION_ID,
    )
    candidate = _case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=action_id,
    )
    baseline_ids = tuple(str(row.sample_id) for row in baseline)
    if baseline_ids != tuple(str(row.sample_id) for row in candidate):
        raise ProtocolError("Baseline/candidate sample identities drifted.")
    try:
        truth = tuple(labels[(target_center, case_id, sample)] for sample in baseline_ids)
    except KeyError as exc:
        raise ProtocolError("Case contribution lacks a scoped label.") from exc
    return contribution_target(
        case_id=case_id,
        action_id=action_id,
        baseline_probabilities=tuple(float(row.probability_mean) for row in baseline),
        action_probabilities=tuple(float(row.probability_mean) for row in candidate),
        labels=truth,
    )


def _case_confusion_for_action(
    probability: Mapping[tuple[str, str, str, str], object],
    labels: Mapping[tuple[str, str, str], int],
    *,
    target_center: str,
    case_id: str,
    action_id: str,
) -> CaseConfusion:
    rows = _case_rows(
        probability,
        target_center=target_center,
        case_id=case_id,
        action_id=action_id,
    )
    try:
        truth = tuple(
            labels[(target_center, case_id, str(row.sample_id))] for row in rows
        )
    except KeyError as exc:
        raise ProtocolError("Terminal confusion lacks a scoped label.") from exc
    predictions = tuple(int(float(row.probability_mean) >= 0.5) for row in rows)
    return case_confusion(case_id, truth, predictions)


def _assert_science_config(config: object) -> None:
    routing = getattr(config, "routing")
    evaluation = getattr(config, "evaluation")
    runtime = getattr(config, "runtime")
    if (
        float(routing.get("ridge_alpha", -1.0)) != 1.0
        or float(routing.get("variance_floor", -1.0)) != 1.0e-6
        or float(routing.get("heuristic_score_multiplier", -1.0)) != 1.96
        or routing.get("primary_router") != "F_S"
        or evaluation.get("primary_method") != "F_S"
        or int(runtime.get("model_workers", -1)) != 4
        or int(runtime.get("bootstrap_workers", -1)) != 4
        or int(runtime.get("model_threads_per_worker", -1)) != 3
        or int(runtime.get("bootstrap_threads_per_worker", -1)) != 3
    ):
        raise ProtocolError("Flip-router scientific runtime contract drifted.")

