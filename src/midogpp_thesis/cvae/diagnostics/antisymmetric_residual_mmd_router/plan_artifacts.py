"""Serialization and integrity-bound artifacts for antisymmetric router plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import (
    atomic_write_csv_rows,
    atomic_write_json,
    read_json,
    sha256_file,
)
from .contracts import (
    ARM_ROLES,
    CENTERS,
    CONTROL_ARM,
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    row_identity_hash,
)
from .partitions import CrossfitFold
from ..mmd_kmm_router.source_products import SourceProducts


ROUTER_STATE_MEMBER = "arrays/router_states.npz"
ROUTER_PLAN_LOCK_MEMBER = "manifests/router_plan_lock.json"
ROUTER_PLAN_TABLE_MEMBER = "tables/case_router_plans.csv"
TARGET_ASSIGNMENT_MEMBER = "tables/case_target_assignments.csv"

ROUTER_PLAN_COLUMNS = (
    "schema_version",
    "fold_ordinal",
    "fold_id",
    "target_center",
    "heldout_case_id",
    "router_support_case_count",
    "candidate_sources_json",
    "delta_json",
    "class_0_weights_json",
    "class_1_weights_json",
    "control_weights_json",
    "routed_allocations_by_class_json",
    "control_allocations_by_class_json",
    "allocation_hash",
    "used_uniform_fallback",
    "fallback_reason",
    "robust_proxy_improvement",
    "proposed_robust_proxy_improvement",
    "support_quality_passed",
    "all_variants_nonworsening",
    "class_0_uniform_l1",
    "class_1_uniform_l1",
    "class_0_effective_source_count",
    "class_1_effective_source_count",
    "maximum_source_weight",
    "preprocessing_hash",
    "candidate_pool_fit_hash",
    "kernel_map_hash",
    "prior_model_hash",
    "prior_fit_pool_hash",
    "plan_hash",
    "heldout_case_embeddings_used_for_own_route",
    "support_labels_used",
    "diagnostic_only",
)
TARGET_ASSIGNMENT_COLUMNS = (
    "schema_version",
    "fold_id",
    "target_center",
    "heldout_case_id",
    "arm_role",
    "training_seed",
    "generation_seed",
    "class_label",
    "source_center",
    "source_stream_id",
    "prefix_count",
    "source_weight",
    "target_expert_excluded",
    "seed_selected",
)


@dataclass(frozen=True)
class AntisymmetricRouterPlans:
    plans_by_fold: Mapping[str, Mapping[str, object]]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["router_plan_lock_hash"])


def load_antisymmetric_router_plans(
    root: Path,
    *,
    expected_config_contract_hash: str | None = None,
    expected_crossfit_partition_lock_hash: str | None = None,
    expected_source_products_hash: str | None = None,
    expected_source_products_lock_hash: str | None = None,
) -> AntisymmetricRouterPlans:
    """Load and independently verify a finalized router-plan artifact surface."""

    lock = read_json(root / ROUTER_PLAN_LOCK_MEMBER)
    unhashed = {key: value for key, value in lock.items() if key != "router_plan_lock_hash"}
    raw = lock.get("plans")
    if (
        lock.get("router_plan_lock_hash") != stable_hash(unhashed)
        or lock.get("status") != "LOCKED_BEFORE_CASE_CROSSFIT_PREDICTIONS"
        or not isinstance(raw, list)
        or len(raw) != EXPECTED_CROSS_FIT_FOLD_COUNT
        or lock.get("router_state_sha256") != sha256_file(root / ROUTER_STATE_MEMBER)
        or lock.get("router_plan_table_sha256")
        != sha256_file(root / ROUTER_PLAN_TABLE_MEMBER)
        or lock.get("target_assignment_table_sha256")
        != sha256_file(root / TARGET_ASSIGNMENT_MEMBER)
        or (
            expected_config_contract_hash is not None
            and lock.get("config_contract_hash") != expected_config_contract_hash
        )
        or (
            expected_crossfit_partition_lock_hash is not None
            and lock.get("crossfit_partition_lock_hash")
            != expected_crossfit_partition_lock_hash
        )
        or (
            expected_source_products_hash is not None
            and lock.get("source_products_hash") != expected_source_products_hash
        )
        or (
            expected_source_products_lock_hash is not None
            and lock.get("source_products_lock_hash")
            != expected_source_products_lock_hash
        )
    ):
        raise ProtocolError("Antisymmetric router-plan lock failed validation.")
    plans: dict[str, Mapping[str, object]] = {}
    for plan in raw:
        if not isinstance(plan, Mapping):
            raise ProtocolError("Antisymmetric router plan is malformed.")
        plan_unhashed = {key: value for key, value in plan.items() if key != "plan_hash"}
        if plan.get("plan_hash") != stable_hash(plan_unhashed):
            raise ProtocolError("Antisymmetric plan hash drifted.")
        fold_id = str(plan.get("fold_id", ""))
        if not fold_id or fold_id in plans:
            raise ProtocolError("Antisymmetric plan fold key duplicated.")
        plans[fold_id] = plan
    return AntisymmetricRouterPlans(plans_by_fold=plans, lock_payload=lock)


def _write_plan_artifacts(
    root: Path,
    *,
    plans_in_order: tuple[Mapping[str, object], ...],
    state_arrays: Mapping[str, np.ndarray],
    source_products: SourceProducts,
    config_contract_hash: str,
    support_partition_lock_hash: str,
    crossfit_partition_lock_hash: str,
    source_products_lock_hash: str,
) -> AntisymmetricRouterPlans:
    """Write the immutable state, tables, and lock for a complete plan surface."""

    _atomic_save_npz(root / ROUTER_STATE_MEMBER, state_arrays)
    atomic_write_csv_rows(
        root / ROUTER_PLAN_TABLE_MEMBER,
        [_plan_table_row(plan) for plan in plans_in_order],
        columns=ROUTER_PLAN_COLUMNS,
    )
    atomic_write_csv_rows(
        root / TARGET_ASSIGNMENT_MEMBER,
        _assignment_rows(plans_in_order, source_products),
        columns=TARGET_ASSIGNMENT_COLUMNS,
    )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_antisymmetric_residual_mmd_router_plan_lock_v1",
        "status": "LOCKED_BEFORE_CASE_CROSSFIT_PREDICTIONS",
        "config_contract_hash": config_contract_hash,
        "support_partition_lock_hash": support_partition_lock_hash,
        "crossfit_partition_lock_hash": crossfit_partition_lock_hash,
        "source_products_hash": source_products.source_products_hash,
        "source_products_lock_hash": source_products_lock_hash,
        "router_state_member": ROUTER_STATE_MEMBER,
        "router_state_sha256": sha256_file(root / ROUTER_STATE_MEMBER),
        "router_plan_table_sha256": sha256_file(root / ROUTER_PLAN_TABLE_MEMBER),
        "target_assignment_table_sha256": sha256_file(
            root / TARGET_ASSIGNMENT_MEMBER
        ),
        "target_workspace_count": len(CENTERS),
        "fold_count": len(plans_in_order),
        "plans": list(plans_in_order),
        "target_labels_used": False,
        "support_labels_used": False,
        "heldout_case_embeddings_used_for_own_route": False,
        "cohort_embeddings_used_for_other_case_routes": True,
        "all_retained_seeds_aggregated": True,
        "claim_role": "label_free_proxy_compatibility_only",
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "promotion_eligible": False,
    }
    lock = {**unhashed, "router_plan_lock_hash": stable_hash(unhashed)}
    atomic_write_json(root / ROUTER_PLAN_LOCK_MEMBER, lock)
    return AntisymmetricRouterPlans(
        plans_by_fold={str(plan["fold_id"]): plan for plan in plans_in_order},
        lock_payload=lock,
    )


def _solution_payload(solution: object) -> dict[str, object]:
    axes = {
        axis: {
            "axis": diagnostic.axis,
            "variant_ids": list(diagnostic.variant_ids),
            "uniform_mean_loss": diagnostic.uniform_mean_loss,
            "proposed_mean_loss": diagnostic.proposed_mean_loss,
            "final_mean_loss": diagnostic.final_mean_loss,
            "uniform_worst_loss": diagnostic.uniform_worst_loss,
            "proposed_worst_loss": diagnostic.proposed_worst_loss,
            "final_worst_loss": diagnostic.final_worst_loss,
            "minimum_proposed_variant_improvement": diagnostic.minimum_proposed_variant_improvement,
            "maximum_proposed_variant_worsening": diagnostic.maximum_proposed_variant_worsening,
            "all_proposed_variants_nonworsening": diagnostic.all_proposed_variants_nonworsening,
        }
        for axis, diagnostic in solution.axis_diagnostics.items()
    }
    variants = [
        {
            "axis": row.axis,
            "variant_id": row.variant_id,
            "uniform_components": dict(row.uniform_components),
            "proposed_components": dict(row.proposed_components),
            "final_components": dict(row.final_components),
            "proposed_improvement": row.proposed_improvement,
            "final_improvement": row.final_improvement,
            "proposed_worsened": row.proposed_worsened,
        }
        for row in solution.variant_diagnostics
    ]
    return {
        "candidate_sources": list(solution.candidate_sources),
        "uniform_weights": dict(solution.uniform_weights),
        "proposed_delta": dict(solution.proposed_delta),
        "proposed_class_0_weights": dict(solution.proposed_class_0_weights),
        "proposed_class_1_weights": dict(solution.proposed_class_1_weights),
        "delta": dict(solution.delta),
        "class_0_weights": dict(solution.class_0_weights),
        "class_1_weights": dict(solution.class_1_weights),
        "robust_objective": solution.robust_objective,
        "uniform_robust_objective": solution.uniform_robust_objective,
        "proposed_robust_improvement": solution.proposed_robust_improvement,
        "robust_improvement": solution.robust_improvement,
        "proposed_mean_conditional_loss": solution.proposed_mean_conditional_loss,
        "final_mean_conditional_loss": solution.final_mean_conditional_loss,
        "uniform_mean_conditional_loss": solution.uniform_mean_conditional_loss,
        "proposed_worst_conditional_loss": solution.proposed_worst_conditional_loss,
        "final_worst_conditional_loss": solution.final_worst_conditional_loss,
        "uniform_worst_conditional_loss": solution.uniform_worst_conditional_loss,
        "l2_penalty_value": solution.l2_penalty_value,
        "class_0_effective_source_count": solution.class_0_effective_source_count,
        "class_1_effective_source_count": solution.class_1_effective_source_count,
        "maximum_source_weight": solution.maximum_source_weight,
        "class_0_uniform_l1": solution.class_0_uniform_l1,
        "class_1_uniform_l1": solution.class_1_uniform_l1,
        "used_uniform_fallback": solution.used_uniform_fallback,
        "fallback_reason": solution.fallback_reason,
        "solver_success": solution.solver_success,
        "solver_message": solution.solver_message,
        "solver_iterations": solution.solver_iterations,
        "solver_version": solution.solver_version,
        "solver_method": solution.solver_method,
        "weight_semantics": solution.weight_semantics,
        "objective_semantics": solution.objective_semantics,
        "claim_role": solution.claim_role,
        "label_use_semantics": solution.label_use_semantics,
        "labels_used": solution.labels_used,
        "support_labels_used": solution.support_labels_used,
        "target_labels_used": solution.target_labels_used,
        "evaluation_labels_used": solution.evaluation_labels_used,
        "downstream_utility_used": solution.downstream_utility_used,
        "downstream_utility_claimed": solution.downstream_utility_claimed,
        "promotion_eligible": solution.promotion_eligible,
        "support_quality_passed": solution.support_quality_passed,
        "all_variants_nonworsening": solution.all_variants_nonworsening,
        "variant_diagnostics": variants,
        "axis_diagnostics": axes,
    }


def _fold_task_payload(fold: CrossfitFold) -> dict[str, object]:
    return {
        "fold_ordinal": fold.fold_ordinal,
        "fold_id": fold.fold_id,
        "target_center": fold.target_center,
        "heldout_case_id": fold.heldout_case_id,
        "router_support_sample_ids": [row.sample_id for row in fold.router_support_rows],
        "router_support_case_ids": list(fold.router_support_case_ids),
        "router_support_row_identity_hash": row_identity_hash(fold.router_support_rows),
        "heldout_sample_ids": [row.sample_id for row in fold.heldout_rows],
        "heldout_row_identity_hash": row_identity_hash(fold.heldout_rows),
        "fold_hash": fold.fold_hash,
    }


def _plan_table_row(plan: Mapping[str, object]) -> dict[str, object]:
    solution = plan["solution"]
    return {
        "schema_version": "midogpp_antisymmetric_residual_mmd_case_plan_row_v1",
        "fold_ordinal": plan["fold_ordinal"],
        "fold_id": plan["fold_id"],
        "target_center": plan["target_center"],
        "heldout_case_id": plan["heldout_case_id"],
        "router_support_case_count": len(plan["router_support_case_ids"]),
        "candidate_sources_json": _compact(plan["candidate_sources"]),
        "delta_json": _compact(plan["delta"]),
        "class_0_weights_json": _compact(plan["class_0_weights"]),
        "class_1_weights_json": _compact(plan["class_1_weights"]),
        "control_weights_json": _compact(plan["control_weights"]),
        "routed_allocations_by_class_json": _compact(
            plan["routed_allocations_by_class"]
        ),
        "control_allocations_by_class_json": _compact(
            plan["control_allocations_by_class"]
        ),
        "allocation_hash": plan["allocation_hash"],
        "used_uniform_fallback": plan["used_uniform_fallback"],
        "fallback_reason": plan["fallback_reason"] or "",
        "robust_proxy_improvement": solution["robust_improvement"],
        "proposed_robust_proxy_improvement": solution[
            "proposed_robust_improvement"
        ],
        "support_quality_passed": solution["support_quality_passed"],
        "all_variants_nonworsening": solution["all_variants_nonworsening"],
        "class_0_uniform_l1": solution["class_0_uniform_l1"],
        "class_1_uniform_l1": solution["class_1_uniform_l1"],
        "class_0_effective_source_count": solution[
            "class_0_effective_source_count"
        ],
        "class_1_effective_source_count": solution[
            "class_1_effective_source_count"
        ],
        "maximum_source_weight": solution["maximum_source_weight"],
        "preprocessing_hash": plan["preprocessing_hash"],
        "candidate_pool_fit_hash": plan["candidate_pool_fit_hash"],
        "kernel_map_hash": plan["kernel_map_hash"],
        "prior_model_hash": plan["prior_model_hash"],
        "prior_fit_pool_hash": plan["prior_fit_pool_hash"],
        "plan_hash": plan["plan_hash"],
        "heldout_case_embeddings_used_for_own_route": False,
        "support_labels_used": False,
        "diagnostic_only": True,
    }


def _assignment_rows(
    plans: tuple[Mapping[str, object], ...],
    source_products: SourceProducts,
) -> list[dict[str, object]]:
    source_index = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in source_products.index_rows
    }
    rows: list[dict[str, object]] = []
    for plan in plans:
        for arm in ARM_ROLES:
            allocations = (
                plan["control_allocations_by_class"]
                if arm == CONTROL_ARM
                else plan["routed_allocations_by_class"]
            )
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for label in (0, 1):
                        weights = (
                            plan["control_weights"]
                            if arm == CONTROL_ARM
                            else plan[f"class_{label}_weights"]
                        )
                        for source in plan["candidate_sources"]:
                            source_row = source_index[
                                (source, training_seed, generation_seed)
                            ]
                            rows.append(
                                {
                                    "schema_version": "midogpp_antisymmetric_residual_mmd_assignment_v1",
                                    "fold_id": plan["fold_id"],
                                    "target_center": plan["target_center"],
                                    "heldout_case_id": plan["heldout_case_id"],
                                    "arm_role": arm,
                                    "training_seed": training_seed,
                                    "generation_seed": generation_seed,
                                    "class_label": label,
                                    "source_center": source,
                                    "source_stream_id": source_row["stream_id"],
                                    "prefix_count": allocations[str(label)][source],
                                    "source_weight": weights[source],
                                    "target_expert_excluded": True,
                                    "seed_selected": False,
                                }
                            )
    return rows


def _write_target_checkpoint(
    path: Path,
    *,
    plans: object,
    state: object,
) -> None:
    if not isinstance(plans, list) or not isinstance(state, Mapping):
        raise ProtocolError("Antisymmetric target checkpoint payload is malformed.")
    arrays = {str(key): np.asarray(value) for key, value in state.items()}
    metadata_unhashed = {
        "plans": plans,
        "state_hashes": {key: _sha256_array(value) for key, value in arrays.items()},
    }
    metadata = {**metadata_unhashed, "checkpoint_hash": stable_hash(metadata_unhashed)}
    _atomic_save_npz(path, {**arrays, "checkpoint_json": np.asarray(_compact(metadata))})


def _load_target_checkpoint(
    path: Path,
    *,
    task: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as raw:
            metadata = json.loads(str(np.asarray(raw["checkpoint_json"]).item()))
            state = {
                key: np.asarray(raw[key])
                for key in raw.files
                if key != "checkpoint_json"
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError("Antisymmetric target checkpoint is unreadable.") from exc
    if not isinstance(metadata, Mapping):
        raise ProtocolError("Antisymmetric target checkpoint metadata is malformed.")
    unhashed = {key: value for key, value in metadata.items() if key != "checkpoint_hash"}
    plans = metadata.get("plans")
    hashes = metadata.get("state_hashes")
    if (
        metadata.get("checkpoint_hash") != stable_hash(unhashed)
        or not isinstance(plans, list)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(state)
        or any(hashes[key] != _sha256_array(value) for key, value in state.items())
        or len(plans) != len(task["folds"])
    ):
        raise ProtocolError("Antisymmetric target checkpoint failed validation.")
    expected_fold_ids = [str(row["fold_id"]) for row in task["folds"]]
    if [str(plan.get("fold_id", "")) for plan in plans] != expected_fold_ids:
        raise ProtocolError("Antisymmetric target checkpoint fold coverage drifted.")
    for plan in plans:
        unhashed_plan = {key: value for key, value in plan.items() if key != "plan_hash"}
        if (
            plan.get("plan_hash") != stable_hash(unhashed_plan)
            or plan.get("target_center") != task["target_center"]
            or plan.get("config_contract_hash") != task["config_contract_hash"]
            or plan.get("crossfit_partition_lock_hash")
            != task["crossfit_partition_lock_hash"]
            or plan.get("source_products_hash") != task["source_products_hash"]
            or plan.get("source_products_lock_hash")
            != task["source_products_lock_hash"]
        ):
            raise ProtocolError("Antisymmetric target checkpoint binding drifted.")
    return tuple(plans), state


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "ROUTER_PLAN_COLUMNS",
    "ROUTER_PLAN_LOCK_MEMBER",
    "ROUTER_PLAN_TABLE_MEMBER",
    "ROUTER_STATE_MEMBER",
    "TARGET_ASSIGNMENT_COLUMNS",
    "TARGET_ASSIGNMENT_MEMBER",
    "AntisymmetricRouterPlans",
    "load_antisymmetric_router_plans",
)
