"""Pure selection, fallback, assignment, and lock construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import legal_routing_sources
from ...generation.contracts import GenerationLock
from ...generation.generation import equal_union_replicate_plan, source_generation_plan
from ...protocol import ProtocolError
from ..policy import assignment_rows as equal_union_assignment_rows
from .bootstrap import bootstrap_outer_policy
from .config import UtilityRegretPolicyConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    CONSUMPTION_RULE_HASH,
    EQUAL_UNION_SOURCE_BUDGET_PER_CLASS,
    EXPERIMENT_ID,
    GENERATION_SEEDS,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    SINGLE_SOURCE_BUDGET_PER_CLASS,
    TRAINING_SEEDS,
    BootstrapResult,
    CandidateSummary,
    PolicyAssignment,
    PolicySelection,
    UtilityRegretPolicyLock,
)


def build_policy_selections(
    summaries: Sequence[CandidateSummary],
    case_rows: Sequence[Mapping[str, object]],
) -> tuple[PolicySelection, ...]:
    selections: list[PolicySelection] = []
    for target in CENTERS:
        bootstrap = bootstrap_outer_policy(
            outer_target_center=target,
            summaries=summaries,
            case_rows=case_rows,
        )
        if bootstrap.gate_passed:
            retained = (bootstrap.observed_best_source,)
            action = "single_source_full_budget"
            selected = bootstrap.observed_best_source
            source_budget = SINGLE_SOURCE_BUDGET_PER_CLASS
        else:
            retained = legal_routing_sources(target)
            action = "fallback_equal_union"
            selected = ""
            source_budget = EQUAL_UNION_SOURCE_BUDGET_PER_CLASS
        if target in retained or len(retained) not in {1, 8}:
            raise ProtocolError("Utility/regret selection violates target exclusion.")
        candidates = legal_routing_sources(target)
        selection_id = stable_hash(
            {
                "namespace": POLICY_NAMESPACE,
                "consumption_rule_hash": CONSUMPTION_RULE_HASH,
                "target_center": target,
                "candidate_sources": list(candidates),
                "action": action,
                "retained_sources": list(retained),
                "bootstrap": bootstrap.to_payload(),
            }
        )
        selections.append(
            PolicySelection(
                selection_id=selection_id,
                target_center=target,
                action=action,
                candidate_sources=candidates,
                selected_source=selected,
                retained_sources=retained,
                source_budget_per_class=source_budget,
                total_per_class=1024,
                gate_reason=bootstrap.gate_reason,
                bootstrap=bootstrap,
            )
        )
    return tuple(selections)


def build_policy_assignments(
    generation_lock: GenerationLock,
    selections: Sequence[PolicySelection],
) -> tuple[PolicyAssignment, ...]:
    by_target = {selection.target_center: selection for selection in selections}
    if set(by_target) != set(CENTERS) or len(by_target) != len(selections):
        raise ProtocolError("Utility/regret selections are incomplete or duplicated.")
    streams = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in source_generation_plan(generation_lock)
    }
    replicates = {
        (replicate.target_center, replicate.training_seed, replicate.generation_seed): replicate
        for replicate in equal_union_replicate_plan(generation_lock)
    }
    control_assignments = {
        (
            assignment.target_center,
            assignment.training_seed,
            assignment.generation_seed,
            assignment.source_center,
        ): assignment
        for assignment in equal_union_assignment_rows(generation_lock)
    }
    assignments: list[PolicyAssignment] = []
    for target in CENTERS:
        selection = by_target[target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                control = replicates[(target, training_seed, generation_seed)]
                if tuple(control.candidate_source_centers) != legal_routing_sources(target):
                    raise ProtocolError("GenerationLock control source order drifted.")
                stream_ids = tuple(
                    streams[(source, training_seed, generation_seed)].stream_id
                    for source in selection.retained_sources
                )
                for selected_ordinal, (source, stream_id) in enumerate(
                    zip(selection.retained_sources, stream_ids, strict=True)
                ):
                    control_assignment = control_assignments[
                        (target, training_seed, generation_seed, source)
                    ]
                    exact_fallback = selection.action == "fallback_equal_union"
                    if exact_fallback:
                        assignment_id = control_assignment.assignment_id
                    else:
                        assignment_id = stable_hash(
                            {
                                "namespace": POLICY_NAMESPACE,
                                "generation_lock_hash": (
                                    generation_lock.generation_lock_hash
                                ),
                                "consumption_rule_hash": CONSUMPTION_RULE_HASH,
                                "selection_id": selection.selection_id,
                                "replicate_id": control.replicate_id,
                                "source_center": source,
                                "source_stream_id": stream_id,
                                "source_budget_per_class": (
                                    selection.source_budget_per_class
                                ),
                            }
                        )
                    assignments.append(
                        PolicyAssignment(
                            assignment_id=assignment_id,
                            selection_id=selection.selection_id,
                            target_center=target,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            replicate_id=control.replicate_id,
                            action=selection.action,
                            source_center=source,
                            source_stream_id=stream_id,
                            canonical_candidate_ordinal=(
                                control.candidate_source_centers.index(source)
                            ),
                            selected_source_ordinal=selected_ordinal,
                            source_budget_per_class=selection.source_budget_per_class,
                            total_per_class=selection.total_per_class,
                            class_shuffle_seed_0=int(
                                control.class_shuffle_seed_by_label["0"]
                            ),
                            class_shuffle_seed_1=int(
                                control.class_shuffle_seed_by_label["1"]
                            ),
                            equal_union_assignment_id=(
                                control_assignment.assignment_id
                            ),
                            exact_equal_union_fallback=exact_fallback,
                        )
                    )
    _validate_assignment_geometry(assignments, selections)
    return tuple(assignments)


def build_policy_plan(
    *,
    config: UtilityRegretPolicyConfig,
    generation_lock: GenerationLock,
    selections: Sequence[PolicySelection],
    assignments: Sequence[PolicyAssignment],
    utility_surface_hash: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_policy_plan_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "policy_family": POLICY_FAMILY,
        "policy_namespace": POLICY_NAMESPACE,
        "consumption_rule_hash": CONSUMPTION_RULE_HASH,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "utility_surface_hash": utility_surface_hash,
        "selection_table_hash": selection_table_hash(selections),
        "assignment_table_hash": assignment_table_hash(assignments),
        "selections": [selection.to_payload() for selection in selections],
        "target_labels_used": False,
        "target_support_used": False,
        "seed_selection_performed": False,
        "policy_frozen_before_stage70": True,
    }
    payload["plan_hash"] = stable_hash(payload)
    return payload


def build_policy_lock(
    *,
    config: UtilityRegretPolicyConfig,
    generation_lock: GenerationLock,
    selections: Sequence[PolicySelection],
    assignments: Sequence[PolicyAssignment],
    utility_surface_hash: str,
    utility_table_hash: str,
    case_confusion_table_hash: str,
    regret_table_hash: str,
    summary_table_hash: str,
    plan_hash: str,
) -> UtilityRegretPolicyLock:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_policy_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "policy_family": POLICY_FAMILY,
        "policy_namespace": POLICY_NAMESPACE,
        "consumption_rule_hash": CONSUMPTION_RULE_HASH,
        "inputs": {
            "bank_lock_hash": config.expected_bank_lock_hash,
            "generation_lock_hash": generation_lock.generation_lock_hash,
            "equal_union_policy_lock_hash": (
                config.expected_equal_union_policy_lock_hash
            ),
            "equal_union_policy_plan_hash": (
                config.expected_equal_union_policy_plan_hash
            ),
            "equal_union_assignment_table_hash": (
                config.expected_equal_union_assignment_table_hash
            ),
            "utility_surface_hash": utility_surface_hash,
            "utility_table_hash": utility_table_hash,
            "case_confusion_table_hash": case_confusion_table_hash,
        },
        "outputs": {
            "regret_table_hash": regret_table_hash,
            "summary_table_hash": summary_table_hash,
            "selection_table_hash": selection_table_hash(selections),
            "assignment_table_hash": assignment_table_hash(assignments),
            "policy_plan_hash": plan_hash,
        },
        "selection_count": len(selections),
        "single_source_selection_count": sum(
            selection.action == "single_source_full_budget" for selection in selections
        ),
        "equal_union_fallback_count": sum(
            selection.action == "fallback_equal_union" for selection in selections
        ),
        "target_labels_used": False,
        "target_support_used": False,
        "seed_selection_performed": False,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "policy_frozen_before_stage70": True,
    }
    payload["policy_lock_hash"] = stable_hash(payload)
    return UtilityRegretPolicyLock(payload)


def selection_table_hash(selections: Sequence[PolicySelection]) -> str:
    return stable_hash([selection.to_payload() for selection in selections])


def bootstrap_table_hash(selections: Sequence[PolicySelection]) -> str:
    return stable_hash([selection.bootstrap.to_payload() for selection in selections])


def assignment_table_hash(assignments: Sequence[PolicyAssignment]) -> str:
    return stable_hash([assignment.to_payload() for assignment in assignments])


def case_confusion_table_hash(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            CENTERS.index(str(row["pseudo_target_center"])),
            CENTERS.index(str(row["candidate_source_center"])),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            str(row["case_id"]),
        ),
    )
    return stable_hash(canonical)


def read_policy_lock(path: str | Path) -> UtilityRegretPolicyLock:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility/regret policy lock: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Utility/regret policy lock must be a JSON object.")
    return UtilityRegretPolicyLock(payload)


def _validate_assignment_geometry(
    assignments: Sequence[PolicyAssignment],
    selections: Sequence[PolicySelection],
) -> None:
    by_target = {selection.target_center: selection for selection in selections}
    groups: dict[tuple[str, int, int], list[PolicyAssignment]] = {}
    for assignment in assignments:
        groups.setdefault(
            (
                assignment.target_center,
                assignment.training_seed,
                assignment.generation_seed,
            ),
            [],
        ).append(assignment)
    expected = {
        (target, training_seed, generation_seed)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if set(groups) != expected:
        raise ProtocolError("Utility/regret assignment replicate coverage drifted.")
    for (target, _training_seed, _generation_seed), group in groups.items():
        selection = by_target[target]
        if tuple(item.source_center for item in group) != selection.retained_sources:
            raise ProtocolError("Utility/regret assignment source order drifted.")
        if any(item.source_center == target for item in group):
            raise ProtocolError("Utility/regret assignment includes the target expert.")
        if sum(item.source_budget_per_class for item in group) != 1024:
            raise ProtocolError("Utility/regret assignment class budget drifted.")
        if len({item.replicate_id for item in group}) != 1:
            raise ProtocolError("Utility/regret assignment replicate identity drifted.")
        if selection.action == "fallback_equal_union":
            if any(
                item.assignment_id != item.equal_union_assignment_id
                or not item.exact_equal_union_fallback
                or item.source_budget_per_class != 128
                for item in group
            ):
                raise ProtocolError("Equal-union fallback is not byte-contract exact.")
        elif any(item.exact_equal_union_fallback for item in group):
            raise ProtocolError("Single-source assignment is mislabeled as fallback.")


__all__ = (
    "assignment_table_hash",
    "bootstrap_table_hash",
    "build_policy_assignments",
    "build_policy_lock",
    "build_policy_plan",
    "build_policy_selections",
    "case_confusion_table_hash",
    "read_policy_lock",
    "selection_table_hash",
)
