"""Small shared helpers for sealed orchestration products."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .contracts import (
    ActionScoreRow,
    BinaryLabelRow,
    CaseActionFeatureRow,
    MethodDecision,
    RidgeActionModel,
    UtilityTargetRow,
)


def feature_payload(row: CaseActionFeatureRow) -> dict[str, object]:
    return row.to_payload()


def model_payload(row: RidgeActionModel) -> dict[str, object]:
    return row.to_payload()


def score_payload(row: ActionScoreRow) -> dict[str, object]:
    return {
        "target_center": row.target_center,
        "case_id": row.case_id,
        "geometry_id": row.geometry_id,
        "selected_source": row.selected_source,
        "family": row.family,
        "predicted_gain": row.predicted_gain,
        "model_hash": row.model_hash,
    }


def decision_payload(row: MethodDecision) -> dict[str, object]:
    return {
        "target_center": row.target_center,
        "case_id": row.case_id,
        "method_id": row.method_id,
        "action_id": row.action_id,
        "geometry_id": row.geometry_id,
        "predicted_gain": row.predicted_gain,
        "decision_source": row.decision_source,
        "evaluation_labels_used": row.evaluation_labels_used,
    }


def utility_payload(row: UtilityTargetRow) -> dict[str, object]:
    return {
        "query_center": row.query_center,
        "case_id": row.case_id,
        "geometry_id": row.geometry_id,
        "selected_source": row.selected_source,
        "response": row.response,
        "response_kind": row.response_kind,
    }


def coerce_labels(
    labels: Sequence[object], *, expected_scope: str | None
) -> tuple[BinaryLabelRow, ...]:
    """Validate capability scope, then copy into the pure label contract."""

    raw = tuple(labels)
    if not raw:
        raise ProtocolError("A non-empty scoped label surface is required.")
    output: list[BinaryLabelRow] = []
    for row in raw:
        if expected_scope is not None and hasattr(row, "label_scope"):
            if str(getattr(row, "label_scope")) != expected_scope:
                raise ProtocolError("Label capability scope does not match this phase.")
        try:
            output.append(
                BinaryLabelRow(
                    target_center=str(getattr(row, "target_center")),
                    case_id=str(getattr(row, "case_id")),
                    sample_id=str(getattr(row, "sample_id")),
                    label=getattr(row, "label"),
                )
            )
        except (AttributeError, TypeError) as exc:
            raise ProtocolError("Label rows do not satisfy the binary identity contract.") from exc
    canonical = tuple(sorted(output))
    if len({row.sample_key for row in canonical}) != len(canonical):
        raise ProtocolError("Scoped label surface contains duplicate sample identities.")
    return canonical


def partition_cases(partition: CaseOOFPartition, target_center: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                row.case_id
                for row in partition.identities
                if row.target_center == target_center
            }
        )
    )


__all__ = (
    "coerce_labels",
    "decision_payload",
    "feature_payload",
    "model_payload",
    "partition_cases",
    "score_payload",
    "utility_payload",
)
