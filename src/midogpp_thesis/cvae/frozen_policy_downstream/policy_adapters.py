"""Arm-specific adapters for already-frozen Stage-60 policy artifacts.

This module never recalculates metadata compatibility or utility/regret.  It
validates and normalizes the published assignment tables only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ..generation import GenerationLock, equal_union_replicate_plan
from ..protocol import ProtocolError
from ..routing import policy as equal_policy
from ..routing.contracts import PolicyAssignment as EqualAssignment
from ..routing.metadata_tie_union import policy as metadata_policy
from ..routing.metadata_tie_union.contracts import PolicyAssignment as MetadataAssignment
from ..routing.utility_regret_policy import policy as utility_policy
from ..routing.utility_regret_policy.contracts import PolicyAssignment as UtilityAssignment
from .contracts import (
    CONTROL_ARM,
    METADATA_ARM,
    UTILITY_ARM,
    MaterializationAssignment,
    PolicyReplicate,
)


def load_frozen_policy_replicates(
    *,
    generation_lock: GenerationLock,
    equal_union_root: str | Path,
    metadata_tie_union_root: str | Path,
    utility_regret_root: str | Path,
) -> tuple[PolicyReplicate, ...]:
    """Load 81 replicates for each frozen arm without reopening router inputs."""

    control = _load_equal_union(generation_lock, Path(equal_union_root))
    metadata = _load_metadata(generation_lock, Path(metadata_tie_union_root))
    utility = _load_utility(generation_lock, Path(utility_regret_root))
    rows = control + metadata + utility
    if len(rows) != 243:
        raise ProtocolError("Stage-70 frozen-policy replicate count must be 243.")
    keys = {
        (row.policy_id, row.target_center, row.training_seed, row.generation_seed)
        for row in rows
    }
    if len(keys) != 243:
        raise ProtocolError("Stage-70 frozen-policy replicate keys are duplicated.")
    return rows


def _load_equal_union(
    generation_lock: GenerationLock,
    root: Path,
) -> tuple[PolicyReplicate, ...]:
    lock = equal_policy.read_policy_lock(root / "manifests/policy_lock.json")
    lock_payload = lock.to_payload()
    observed_rows = _csv_rows(root / "tables/policy_assignments.csv")
    assignments = tuple(_equal_assignment(row) for row in observed_rows)
    expected = equal_policy.assignment_rows(generation_lock)
    if [row.to_payload() for row in assignments] != [row.to_payload() for row in expected]:
        raise ProtocolError("Equal-union assignment table drifted from GenerationLock.")
    table_hash = equal_policy.assignment_table_hash(assignments)
    if table_hash != lock_payload.get("assignment_table_hash"):
        raise ProtocolError("Equal-union assignment table hash drifted.")
    return _group_assignments(
        policy_id=CONTROL_ARM,
        lock_payload=lock_payload,
        generation_lock=generation_lock,
        rows=assignments,
        prior_method="class_conditional_source_aggregate_posterior",
        selection_source="fixed_all_eligible_non_target_sources",
    )


def _load_metadata(
    generation_lock: GenerationLock,
    root: Path,
) -> tuple[PolicyReplicate, ...]:
    lock = metadata_policy.read_policy_lock(root / "manifests/policy_lock.json")
    lock_payload = lock.to_payload()
    rows = tuple(
        _metadata_assignment(row)
        for row in _csv_rows(root / "tables/policy_assignments.csv")
    )
    table_hash = metadata_policy.assignment_table_hash(rows)
    if table_hash != lock_payload.get("assignment_table_hash"):
        raise ProtocolError("Metadata tie-union assignment table hash drifted.")
    return _group_assignments(
        policy_id=METADATA_ARM,
        lock_payload=lock_payload,
        generation_lock=generation_lock,
        rows=rows,
        prior_method="class_conditional_source_aggregate_posterior",
        selection_source="frozen_metadata_exact_match_all_maximum_ties",
    )


def _load_utility(
    generation_lock: GenerationLock,
    root: Path,
) -> tuple[PolicyReplicate, ...]:
    lock = utility_policy.read_policy_lock(root / "manifests/policy_lock.json")
    lock_payload = lock.to_payload()
    rows = tuple(
        _utility_assignment(row)
        for row in _csv_rows(root / "tables/policy_assignments.csv")
    )
    table_hash = utility_policy.assignment_table_hash(rows)
    outputs = _mapping(lock_payload, "outputs")
    if table_hash != outputs.get("assignment_table_hash"):
        raise ProtocolError("Utility/regret assignment table hash drifted.")
    return _group_assignments(
        policy_id=UTILITY_ARM,
        lock_payload=lock_payload,
        generation_lock=generation_lock,
        rows=rows,
        prior_method="class_conditional_source_aggregate_posterior",
        selection_source="frozen_source_inner_utility_regret_gate",
    )


def _group_assignments(
    *,
    policy_id: str,
    lock_payload: Mapping[str, object],
    generation_lock: GenerationLock,
    rows: Sequence[object],
    prior_method: str,
    selection_source: str,
) -> tuple[PolicyReplicate, ...]:
    controls = {
        (row.target_center, row.training_seed, row.generation_seed): row
        for row in equal_union_replicate_plan(generation_lock)
    }
    grouped: dict[tuple[str, int, int], list[object]] = {}
    for row in rows:
        key = (
            str(getattr(row, "target_center")),
            int(getattr(row, "training_seed")),
            int(getattr(row, "generation_seed")),
        )
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(controls):
        raise ProtocolError(f"{policy_id} assignment coverage drifted.")
    policy_lock_hash = str(lock_payload.get("policy_lock_hash", ""))
    if policy_id == UTILITY_ARM:
        lock_outputs = _mapping(lock_payload, "outputs")
        policy_plan_hash = str(lock_outputs.get("policy_plan_hash", ""))
        assignment_hash = str(lock_outputs.get("assignment_table_hash", ""))
    else:
        policy_plan_hash = str(lock_payload.get("policy_plan_hash", ""))
        assignment_hash = str(lock_payload.get("assignment_table_hash", ""))
    replicates: list[PolicyReplicate] = []
    for key in sorted(grouped):
        control = controls[key]
        raw_group = sorted(
            grouped[key],
            key=lambda row: int(
                getattr(
                    row,
                    "selected_source_ordinal",
                    getattr(row, "source_ordinal", -1),
                )
            ),
        )
        normalized: list[MaterializationAssignment] = []
        for ordinal, row in enumerate(raw_group):
            exact_fallback = bool(getattr(row, "exact_equal_union_fallback", False))
            equal_assignment = str(getattr(row, "equal_union_assignment_id", ""))
            normalized.append(
                MaterializationAssignment(
                    assignment_id=str(getattr(row, "assignment_id")),
                    policy_id=policy_id,
                    target_center=key[0],
                    training_seed=key[1],
                    generation_seed=key[2],
                    source_center=str(getattr(row, "source_center")),
                    source_stream_id=str(getattr(row, "source_stream_id")),
                    source_ordinal=ordinal,
                    source_budget_per_class=int(
                        getattr(row, "source_budget_per_class")
                    ),
                    prior_method=prior_method,
                    selection_source=selection_source,
                    exact_equal_union_fallback=exact_fallback,
                    equal_union_assignment_id=equal_assignment,
                )
            )
        replicates.append(
            PolicyReplicate(
                policy_id=policy_id,
                policy_lock_hash=policy_lock_hash,
                policy_plan_hash=policy_plan_hash,
                assignment_table_hash=assignment_hash,
                replicate_id=str(getattr(control, "replicate_id")),
                target_center=key[0],
                training_seed=key[1],
                generation_seed=key[2],
                assignments=tuple(normalized),
                class_shuffle_seed_by_label=dict(
                    getattr(control, "class_shuffle_seed_by_label")
                ),
            )
        )
    return tuple(replicates)


def _equal_assignment(row: Mapping[str, str]) -> EqualAssignment:
    return EqualAssignment(
        assignment_id=row["assignment_id"],
        replicate_id=row["replicate_id"],
        target_center=row["target_center"],
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        source_center=row["source_center"],
        source_stream_id=row["source_stream_id"],
        source_ordinal=int(row["source_ordinal"]),
        source_budget_per_class=int(row["source_budget_per_class"]),
    )


def _metadata_assignment(row: Mapping[str, str]) -> MetadataAssignment:
    return MetadataAssignment(
        assignment_id=row["assignment_id"],
        selection_id=row["selection_id"],
        replicate_id=row["replicate_id"],
        target_center=row["target_center"],
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        source_center=row["source_center"],
        source_stream_id=row["source_stream_id"],
        canonical_candidate_ordinal=int(row["canonical_candidate_ordinal"]),
        selected_source_ordinal=int(row["selected_source_ordinal"]),
        maximum_exact_match_score=int(row["maximum_exact_match_score"]),
        tie_count=int(row["tie_count"]),
        source_budget_per_class=int(row["source_budget_per_class"]),
    )


def _utility_assignment(row: Mapping[str, str]) -> UtilityAssignment:
    return UtilityAssignment(
        assignment_id=row["assignment_id"],
        selection_id=row["selection_id"],
        target_center=row["target_center"],
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        replicate_id=row["replicate_id"],
        action=row["action"],
        source_center=row["source_center"],
        source_stream_id=row["source_stream_id"],
        canonical_candidate_ordinal=int(row["canonical_candidate_ordinal"]),
        selected_source_ordinal=int(row["selected_source_ordinal"]),
        source_budget_per_class=int(row["source_budget_per_class"]),
        total_per_class=int(row["total_per_class"]),
        class_shuffle_seed_0=int(row["class_shuffle_seed_0"]),
        class_shuffle_seed_1=int(row["class_shuffle_seed_1"]),
        equal_union_assignment_id=row["equal_union_assignment_id"],
        exact_equal_union_fallback=_bool(row["exact_equal_union_fallback"]),
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read frozen policy assignments: {path}.") from exc


def _bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ProtocolError(f"Invalid frozen boolean value: {value!r}.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Frozen policy lock lacks mapping {key!r}.")
    return value


__all__ = ("load_frozen_policy_replicates",)
