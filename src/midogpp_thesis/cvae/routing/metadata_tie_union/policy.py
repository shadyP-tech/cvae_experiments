"""Pure reconstruction of the metadata exact-match all-ties union policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import legal_routing_sources
from ...generation import GenerationLock, equal_union_replicate_plan
from ...protocol import ProtocolError
from .config import UniformBV2MetadataTieUnionPolicyConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_ASSIGNMENT_TABLE_HASH,
    EXPECTED_COMPATIBILITY_LOCK_HASH,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH,
    EXPECTED_REPLICATE_COUNT,
    EXPECTED_SELECTION_COUNT,
    EXPECTED_SELECTION_TABLE_HASH,
    EXPERIMENT_ID,
    GENERATION_SEEDS,
    MetadataTieUnionPolicyLock,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    SELECTED_SOURCES_BY_TARGET,
    SOURCE_BUDGET_BY_TIE_COUNT,
    STAGE40_MAX_SOURCE_BLOCK_PER_CLASS,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    PolicyAssignment,
    PolicyReplicate,
    PolicySelection,
)


def build_policy_selections(
    compatibility_scores: Sequence[object],
    config: UniformBV2MetadataTieUnionPolicyConfig | None = None,
) -> tuple[PolicySelection, ...]:
    """Retain every candidate tied at each target's maximum proxy score."""

    by_pair: dict[tuple[str, str], int] = {}
    for raw in compatibility_scores:
        target = _string_attr(raw, "target_center")
        source = _string_attr(raw, "source_center")
        score = _score_value(raw)
        pair = (target, source)
        if pair in by_pair:
            raise ProtocolError("Metadata tie-union compatibility pairs are duplicated.")
        by_pair[pair] = score

    expected_pairs = {
        (target, source)
        for target in CENTERS
        for source in legal_routing_sources(target)
    }
    if set(by_pair) != expected_pairs:
        raise ProtocolError("Metadata tie-union requires all 72 target-excluded scores.")

    rows: list[PolicySelection] = []
    for target in CENTERS:
        candidates = legal_routing_sources(target)
        scores = tuple(by_pair[(target, source)] for source in candidates)
        maximum = max(scores)
        selected = tuple(
            source
            for source, score in zip(candidates, scores, strict=True)
            if score == maximum
        )
        expected_selected = SELECTED_SOURCES_BY_TARGET[target]
        if selected != expected_selected:
            raise ProtocolError(
                "Metadata tie-union maximum-tie selection drifted: "
                f"target={target}, observed={selected!r}, expected={expected_selected!r}."
            )
        tie_count = len(selected)
        if tie_count not in SOURCE_BUDGET_BY_TIE_COUNT:
            raise ProtocolError("Metadata tie-union encountered an unauthorized tie count.")
        budget = SOURCE_BUDGET_BY_TIE_COUNT[tie_count]
        if TOTAL_PER_CLASS % tie_count or budget != TOTAL_PER_CLASS // tie_count:
            raise ProtocolError("Metadata tie-union class budget is not exactly divisible.")
        if budget > STAGE40_MAX_SOURCE_BLOCK_PER_CLASS:
            raise ProtocolError("Metadata tie-union prefix exceeds the Stage-40 source block.")
        if config is not None:
            configured_candidates = config.policy_contract.get(
                "candidate_sources_by_target"
            )
            configured_selected = config.policy_contract.get("selected_sources_by_target")
            if (
                not isinstance(configured_candidates, Mapping)
                or configured_candidates.get(target) != list(candidates)
                or not isinstance(configured_selected, Mapping)
                or configured_selected.get(target) != list(selected)
            ):
                raise ProtocolError("Metadata tie-union configured source sets drifted.")
        identity = {
            "namespace": POLICY_NAMESPACE,
            "target_center": target,
            "candidate_source_centers": list(candidates),
            "candidate_exact_match_scores": list(scores),
            "selected_source_centers": list(selected),
            "maximum_exact_match_score": maximum,
            "source_budget_per_class": budget,
            "total_per_class": TOTAL_PER_CLASS,
        }
        rows.append(
            PolicySelection(
                selection_id=stable_hash(identity),
                target_center=target,
                candidate_source_centers=candidates,
                candidate_exact_match_scores=scores,
                selected_source_centers=selected,
                maximum_exact_match_score=maximum,
                tie_count=tie_count,
                source_budget_per_class=budget,
            )
        )
    if len(rows) != EXPECTED_SELECTION_COUNT:
        raise ProtocolError("Metadata tie-union selection coverage drifted.")
    return tuple(rows)


def build_policy_plan(
    generation_lock: GenerationLock,
    compatibility_scores: Sequence[object],
    config: UniformBV2MetadataTieUnionPolicyConfig | None = None,
) -> tuple[PolicyReplicate, ...]:
    """Pair each target selection with all nine frozen Stage-40 seed rows."""

    rows, _ = _derive(generation_lock, compatibility_scores, config)
    return rows


def assignment_rows(
    generation_lock: GenerationLock,
    compatibility_scores: Sequence[object],
    config: UniformBV2MetadataTieUnionPolicyConfig | None = None,
) -> tuple[PolicyAssignment, ...]:
    """Return the exact 153 selected source-prefix assignments."""

    _, rows = _derive(generation_lock, compatibility_scores, config)
    return rows


def build_policy_plan_payload(
    generation_lock: GenerationLock,
    compatibility_scores: Sequence[object],
    config: UniformBV2MetadataTieUnionPolicyConfig,
) -> dict[str, object]:
    records = [
        row.to_payload()
        for row in build_policy_plan(generation_lock, compatibility_scores, config)
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_plan_v1",
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "equal_union_policy_plan_hash": config.expected_equal_union_policy_plan_hash,
        "equal_union_assignment_table_hash": (
            config.expected_equal_union_assignment_table_hash
        ),
        "compatibility_lock_hash": config.expected_compatibility_lock_hash,
        "compatibility_score_table_hash": (
            config.expected_compatibility_score_table_hash
        ),
        "policy_family": POLICY_FAMILY,
        "selection_count": EXPECTED_SELECTION_COUNT,
        "target_replicate_count": len(records),
        "replicates_per_target": 9,
        "assignment_count": EXPECTED_ASSIGNMENT_COUNT,
        "records": records,
    }
    payload["plan_hash"] = stable_hash(payload)
    if (
        generation_lock.generation_lock_hash == EXPECTED_GENERATION_LOCK_HASH
        and config.expected_compatibility_score_table_hash
        == EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH
        and payload["plan_hash"] != EXPECTED_POLICY_PLAN_HASH
    ):
        raise ProtocolError("Metadata tie-union policy-plan semantic identity drifted.")
    return payload


def selection_table_hash(rows: Sequence[PolicySelection]) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_selections_v1",
            "records": [row.to_payload() for row in rows],
        }
    )


def assignment_table_hash(rows: Sequence[PolicyAssignment]) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_assignments_v1",
            "records": [row.to_payload() for row in rows],
        }
    )


def build_policy_lock(
    config: UniformBV2MetadataTieUnionPolicyConfig,
    generation_lock: GenerationLock,
    equal_union_policy_lock: object,
    compatibility_lock: object,
    compatibility_scores: Sequence[object],
) -> MetadataTieUnionPolicyLock:
    selections = build_policy_selections(compatibility_scores, config)
    plan = build_policy_plan_payload(generation_lock, compatibility_scores, config)
    assignments = assignment_rows(generation_lock, compatibility_scores, config)
    selection_hash = selection_table_hash(selections)
    assignments_hash = assignment_table_hash(assignments)
    equal_union_hash = _lock_hash(equal_union_policy_lock, "policy_lock_hash")
    compatibility_hash = _lock_hash(compatibility_lock, "compatibility_lock_hash")
    if (
        equal_union_hash != config.expected_equal_union_policy_lock_hash
        or compatibility_hash != config.expected_compatibility_lock_hash
    ):
        raise ProtocolError("Metadata tie-union upstream lock identity drifted.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_tie_union_policy_lock_v1",
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
            "equal_union_policy_artifact_id": config.equal_union_policy_artifact_id,
            "equal_union_policy_lock_hash": equal_union_hash,
            "equal_union_policy_plan_hash": (
                config.expected_equal_union_policy_plan_hash
            ),
            "equal_union_assignment_table_hash": (
                config.expected_equal_union_assignment_table_hash
            ),
            "metadata_compatibility_artifact_id": (
                config.metadata_compatibility_artifact_id
            ),
            "compatibility_lock_hash": compatibility_hash,
            "compatibility_score_table_hash": (
                config.expected_compatibility_score_table_hash
            ),
        },
        "policy": dict(config.policy_contract),
        "composition_execution": dict(config.composition_execution),
        "future_evaluation_contract": dict(config.future_evaluation_contract),
        "selection_table_hash": selection_hash,
        "policy_plan_hash": plan["plan_hash"],
        "assignment_table_hash": assignments_hash,
        "firewalls": dict(config.claim_boundary),
    }
    payload["policy_lock_hash"] = stable_hash(payload)
    if (
        generation_lock.generation_lock_hash == EXPECTED_GENERATION_LOCK_HASH
        and compatibility_hash == EXPECTED_COMPATIBILITY_LOCK_HASH
        and (
            plan["plan_hash"] != EXPECTED_POLICY_PLAN_HASH
            or selection_hash != EXPECTED_SELECTION_TABLE_HASH
            or assignments_hash != EXPECTED_ASSIGNMENT_TABLE_HASH
            or payload["policy_lock_hash"] != EXPECTED_POLICY_LOCK_HASH
        )
    ):
        raise ProtocolError("Metadata tie-union policy-lock semantic identity drifted.")
    return MetadataTieUnionPolicyLock(payload)


def read_policy_lock(path: str | Path) -> MetadataTieUnionPolicyLock:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read metadata tie-union policy lock: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Metadata tie-union policy lock must be a JSON object.")
    return MetadataTieUnionPolicyLock(payload)


def _derive(
    generation_lock: GenerationLock,
    compatibility_scores: Sequence[object],
    config: UniformBV2MetadataTieUnionPolicyConfig | None,
) -> tuple[tuple[PolicyReplicate, ...], tuple[PolicyAssignment, ...]]:
    selections = {
        row.target_center: row
        for row in build_policy_selections(compatibility_scores, config)
    }
    control_rows = equal_union_replicate_plan(generation_lock)
    expected_keys = {
        (target, training_seed, generation_seed)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    observed_keys = {
        (row.target_center, row.training_seed, row.generation_seed)
        for row in control_rows
    }
    if len(control_rows) != EXPECTED_REPLICATE_COUNT or observed_keys != expected_keys:
        raise ProtocolError("Metadata tie-union Stage-40 replicate coverage drifted.")
    generation_payload = generation_lock.to_payload()
    generation = generation_payload.get("generation")
    if (
        not isinstance(generation, Mapping)
        or int(generation.get("max_source_block_per_class", -1))
        != STAGE40_MAX_SOURCE_BLOCK_PER_CLASS
        or int(generation.get("total_per_class", -1)) != TOTAL_PER_CLASS
    ):
        raise ProtocolError("Metadata tie-union Stage-40 source budget drifted.")

    plan: list[PolicyReplicate] = []
    assignments: list[PolicyAssignment] = []
    for control in control_rows:
        expected_candidates = legal_routing_sources(control.target_center)
        if (
            control.candidate_source_centers != expected_candidates
            or len(control.source_stream_ids) != len(expected_candidates)
            or control.target_center in control.candidate_source_centers
            or control.total_per_class != TOTAL_PER_CLASS
            or set(control.class_shuffle_seed_by_label) != {"0", "1"}
        ):
            raise ProtocolError("Metadata tie-union Stage-40 control pairing drifted.")
        selection = selections[control.target_center]
        stream_by_source = dict(
            zip(
                control.candidate_source_centers,
                control.source_stream_ids,
                strict=True,
            )
        )
        selected_streams = tuple(
            stream_by_source[source] for source in selection.selected_source_centers
        )
        assignment_ids: list[str] = []
        for selected_ordinal, (source, stream_id) in enumerate(
            zip(selection.selected_source_centers, selected_streams, strict=True)
        ):
            canonical_ordinal = control.candidate_source_centers.index(source)
            identity = {
                "namespace": POLICY_NAMESPACE,
                "replicate_id": control.replicate_id,
                "selection_id": selection.selection_id,
                "source_center": source,
                "source_stream_id": stream_id,
                "canonical_candidate_ordinal": canonical_ordinal,
                "selected_source_ordinal": selected_ordinal,
                "source_budget_per_class": selection.source_budget_per_class,
            }
            assignment = PolicyAssignment(
                assignment_id=stable_hash(identity),
                selection_id=selection.selection_id,
                replicate_id=control.replicate_id,
                target_center=control.target_center,
                training_seed=control.training_seed,
                generation_seed=control.generation_seed,
                source_center=source,
                source_stream_id=stream_id,
                canonical_candidate_ordinal=canonical_ordinal,
                selected_source_ordinal=selected_ordinal,
                maximum_exact_match_score=selection.maximum_exact_match_score,
                tie_count=selection.tie_count,
                source_budget_per_class=selection.source_budget_per_class,
            )
            assignments.append(assignment)
            assignment_ids.append(assignment.assignment_id)
        if selection.source_budget_per_class * selection.tie_count != TOTAL_PER_CLASS:
            raise ProtocolError("Metadata tie-union replicate budget does not total 1024.")
        plan.append(
            PolicyReplicate(
                replicate_id=control.replicate_id,
                selection_id=selection.selection_id,
                target_center=control.target_center,
                training_seed=control.training_seed,
                generation_seed=control.generation_seed,
                candidate_source_centers=control.candidate_source_centers,
                selected_source_centers=selection.selected_source_centers,
                selected_source_stream_ids=selected_streams,
                assignment_ids=tuple(assignment_ids),
                class_shuffle_seed_by_label=dict(control.class_shuffle_seed_by_label),
                maximum_exact_match_score=selection.maximum_exact_match_score,
                tie_count=selection.tie_count,
                source_budget_per_class=selection.source_budget_per_class,
                total_per_class=control.total_per_class,
            )
        )
    if (
        len(plan) != EXPECTED_REPLICATE_COUNT
        or len({row.replicate_id for row in plan}) != EXPECTED_REPLICATE_COUNT
        or len(assignments) != EXPECTED_ASSIGNMENT_COUNT
        or len({row.assignment_id for row in assignments}) != EXPECTED_ASSIGNMENT_COUNT
    ):
        raise ProtocolError("Metadata tie-union assignment coverage drifted.")
    return tuple(plan), tuple(assignments)


def _score_value(raw: object) -> int:
    for name in (
        "exact_match_count",
        "exact_match_score",
        "metadata_exact_match_score",
        "score",
    ):
        if hasattr(raw, name):
            value = getattr(raw, name)
            if isinstance(value, bool):
                break
            rendered = int(value)
            if rendered != value or rendered < 0 or rendered > 3:
                break
            return rendered
    raise ProtocolError("Metadata tie-union compatibility score is malformed.")


def _string_attr(raw: object, name: str) -> str:
    if not hasattr(raw, name):
        raise ProtocolError("Metadata tie-union compatibility row is malformed.")
    value = str(getattr(raw, name))
    if not value:
        raise ProtocolError("Metadata tie-union compatibility row is malformed.")
    return value


def _lock_hash(lock: object, name: str) -> str:
    if not hasattr(lock, name):
        raise ProtocolError("Metadata tie-union upstream lock is malformed.")
    value = str(getattr(lock, name))
    if not value:
        raise ProtocolError("Metadata tie-union upstream lock is malformed.")
    return value


__all__ = (
    "assignment_rows",
    "assignment_table_hash",
    "build_policy_lock",
    "build_policy_plan",
    "build_policy_plan_payload",
    "build_policy_selections",
    "read_policy_lock",
    "selection_table_hash",
)
