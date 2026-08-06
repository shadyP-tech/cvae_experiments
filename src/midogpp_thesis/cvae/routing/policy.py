"""Pure derivation of the frozen Uniform-B v2 equal-union policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import legal_routing_sources
from ..generation import GenerationLock, equal_union_replicate_plan
from ..protocol import ProtocolError
from .config import UniformBV2EqualUnionPolicyConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_ASSIGNMENT_TABLE_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH,
    EXPECTED_REPLICATE_COUNT,
    EXPERIMENT_ID,
    GENERATION_SEEDS,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    SOURCE_BUDGET_PER_CLASS,
    SOURCES_PER_TARGET,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    EqualUnionPolicyLock,
    PolicyAssignment,
    PolicyReplicate,
)


def build_policy_plan(
    lock: GenerationLock,
    config: UniformBV2EqualUnionPolicyConfig | None = None,
) -> tuple[PolicyReplicate, ...]:
    """Lift the Stage-40 replicate plan into explicit Stage-60 decisions."""

    rows, _ = _derive(lock, config)
    return rows


def assignment_rows(
    lock: GenerationLock,
    config: UniformBV2EqualUnionPolicyConfig | None = None,
) -> tuple[PolicyAssignment, ...]:
    """Return all 648 fixed source-to-replicate assignments."""

    _, rows = _derive(lock, config)
    return rows


def build_policy_plan_payload(
    lock: GenerationLock,
    config: UniformBV2EqualUnionPolicyConfig | None = None,
) -> dict[str, object]:
    records = [row.to_payload() for row in build_policy_plan(lock, config)]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_policy_plan_v1",
        "generation_lock_hash": lock.generation_lock_hash,
        "policy_family": POLICY_FAMILY,
        "target_replicate_count": len(records),
        "replicates_per_target": 9,
        "assignments_per_replicate": SOURCES_PER_TARGET,
        "assignment_count": len(records) * SOURCES_PER_TARGET,
        "records": records,
    }
    payload["plan_hash"] = stable_hash(payload)
    if (
        lock.generation_lock_hash == EXPECTED_GENERATION_LOCK_HASH
        and payload["plan_hash"] != EXPECTED_POLICY_PLAN_HASH
    ):
        raise ProtocolError("Equal-union policy-plan semantic identity drifted.")
    return payload


def assignment_table_hash(rows: tuple[PolicyAssignment, ...]) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_uniform_b_v2_equal_union_assignments_v1",
            "records": [row.to_payload() for row in rows],
        }
    )


def build_policy_lock(
    config: UniformBV2EqualUnionPolicyConfig,
    generation_lock: GenerationLock,
) -> EqualUnionPolicyLock:
    plan = build_policy_plan_payload(generation_lock, config)
    assignments = assignment_rows(generation_lock, config)
    assignments_hash = assignment_table_hash(assignments)
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "upstreams": {
            "bank_artifact_id": config.bank_artifact_id,
            "bank_lock_hash": config.expected_bank_lock_hash,
            "generation_lock_artifact_id": config.generation_lock_artifact_id,
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "generation_content_hash": config.expected_generation_content_hash,
            "source_plan_hash": config.expected_source_plan_hash,
            "replicate_plan_hash": config.expected_replicate_plan_hash,
        },
        "policy": dict(config.policy_contract),
        "composition_execution": dict(config.composition_execution),
        "future_evaluation_contract": dict(config.future_evaluation_contract),
        "policy_plan_hash": plan["plan_hash"],
        "assignment_table_hash": assignments_hash,
        "firewalls": dict(config.claim_boundary),
    }
    payload["policy_lock_hash"] = stable_hash(payload)
    if generation_lock.generation_lock_hash == EXPECTED_GENERATION_LOCK_HASH and (
        plan["plan_hash"] != EXPECTED_POLICY_PLAN_HASH
        or assignments_hash != EXPECTED_ASSIGNMENT_TABLE_HASH
        or payload["policy_lock_hash"] != EXPECTED_POLICY_LOCK_HASH
    ):
        raise ProtocolError("Equal-union policy-lock semantic identity drifted.")
    return EqualUnionPolicyLock(payload)


def read_policy_lock(path: str | Path) -> EqualUnionPolicyLock:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read equal-union policy lock: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Equal-union policy lock must be a JSON object.")
    return EqualUnionPolicyLock(payload)


def _derive(
    lock: GenerationLock,
    config: UniformBV2EqualUnionPolicyConfig | None,
) -> tuple[tuple[PolicyReplicate, ...], tuple[PolicyAssignment, ...]]:
    source_rows = equal_union_replicate_plan(lock)
    if len(source_rows) != EXPECTED_REPLICATE_COUNT:
        raise ProtocolError("Equal-union policy must contain exactly 81 replicates.")
    expected_keys = {
        (target, training_seed, generation_seed)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    observed_keys = {
        (row.target_center, row.training_seed, row.generation_seed) for row in source_rows
    }
    if observed_keys != expected_keys:
        raise ProtocolError("Equal-union policy replicate coverage drifted.")

    plan: list[PolicyReplicate] = []
    assignments: list[PolicyAssignment] = []
    for row in source_rows:
        expected_sources = legal_routing_sources(row.target_center)
        if (
            row.candidate_source_centers != expected_sources
            or len(row.source_stream_ids) != SOURCES_PER_TARGET
            or row.target_center in row.candidate_source_centers
            or row.source_budget_per_class != SOURCE_BUDGET_PER_CLASS
            or row.total_per_class != TOTAL_PER_CLASS
        ):
            raise ProtocolError("Equal-union policy candidate pool or budget drifted.")
        if config is not None:
            configured = config.policy_contract["candidate_sources_by_target"]
            if not isinstance(configured, Mapping) or configured.get(row.target_center) != list(
                expected_sources
            ):
                raise ProtocolError("Equal-union policy config candidate order drifted.")
        assignment_ids: list[str] = []
        for ordinal, (source, stream_id) in enumerate(
            zip(row.candidate_source_centers, row.source_stream_ids, strict=True)
        ):
            identity = {
                "namespace": POLICY_NAMESPACE,
                "generation_lock_hash": lock.generation_lock_hash,
                "replicate_id": row.replicate_id,
                "source_center": source,
                "source_stream_id": stream_id,
                "source_ordinal": ordinal,
                "source_budget_per_class": row.source_budget_per_class,
            }
            assignment = PolicyAssignment(
                assignment_id=stable_hash(identity),
                replicate_id=row.replicate_id,
                target_center=row.target_center,
                training_seed=row.training_seed,
                generation_seed=row.generation_seed,
                source_center=source,
                source_stream_id=stream_id,
                source_ordinal=ordinal,
                source_budget_per_class=row.source_budget_per_class,
            )
            assignments.append(assignment)
            assignment_ids.append(assignment.assignment_id)
        plan.append(
            PolicyReplicate(
                replicate_id=row.replicate_id,
                target_center=row.target_center,
                training_seed=row.training_seed,
                generation_seed=row.generation_seed,
                candidate_source_centers=row.candidate_source_centers,
                source_stream_ids=row.source_stream_ids,
                assignment_ids=tuple(assignment_ids),
                class_shuffle_seed_by_label=row.class_shuffle_seed_by_label,
                source_budget_per_class=row.source_budget_per_class,
                total_per_class=row.total_per_class,
            )
        )
    if len(assignments) != EXPECTED_ASSIGNMENT_COUNT or len(
        {row.assignment_id for row in assignments}
    ) != EXPECTED_ASSIGNMENT_COUNT:
        raise ProtocolError("Equal-union policy assignment coverage drifted.")
    return tuple(plan), tuple(assignments)


__all__ = (
    "assignment_rows",
    "assignment_table_hash",
    "build_policy_lock",
    "build_policy_plan",
    "build_policy_plan_payload",
    "read_policy_lock",
)
