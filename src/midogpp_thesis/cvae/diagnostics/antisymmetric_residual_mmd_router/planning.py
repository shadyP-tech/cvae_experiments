"""Case-cross-fitted planning for the antisymmetric residual-MMD diagnostic."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.antisymmetric_residual_mmd import (
    build_antisymmetric_allocation,
    solve_antisymmetric_residual_mmd,
)
from ...routing.mmd_kmm_mixture import (
    CROSSFIT_COHORT_SUPPORT_ROLE,
    MMDKMMProtocol,
    build_conditional_contrast_problem,
    build_conditional_prior_sensitivity_problems,
    build_conditional_seed_axis_problems,
    build_conditional_support_case_problems,
)
from .config import AntisymmetricResidualMMDDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    KERNEL_DEVICES,
    TOTAL_PER_CLASS,
    candidate_sources,
)
from .partitions import CrossfitSurface
from .plan_artifacts import (
    ROUTER_PLAN_COLUMNS,
    ROUTER_PLAN_LOCK_MEMBER,
    ROUTER_PLAN_TABLE_MEMBER,
    ROUTER_STATE_MEMBER,
    TARGET_ASSIGNMENT_COLUMNS,
    TARGET_ASSIGNMENT_MEMBER,
    AntisymmetricRouterPlans,
    _fold_task_payload,
    _load_target_checkpoint,
    _solution_payload,
    _write_plan_artifacts,
    _write_target_checkpoint,
    load_antisymmetric_router_plans,
)
from .target_workspace import build_target_kernel_workspace
from ..mmd_kmm_router.inputs import LabelFreeValidationFrame, PartitionSurface
from ..mmd_kmm_router.source_products import SourceProducts


def build_antisymmetric_router_plans(
    config: AntisymmetricResidualMMDDiagnosticConfig,
    source_products: SourceProducts,
    frame: LabelFreeValidationFrame,
    base_partitions: PartitionSurface,
    crossfit: CrossfitSurface,
    *,
    source_products_lock_hash: str,
    root: Path,
) -> AntisymmetricRouterPlans:
    """Build 26 plans in nine resumable, one-workspace-per-target jobs."""

    final_members = (
        root / ROUTER_STATE_MEMBER,
        root / ROUTER_PLAN_LOCK_MEMBER,
        root / ROUTER_PLAN_TABLE_MEMBER,
        root / TARGET_ASSIGNMENT_MEMBER,
    )
    if all(path.is_file() for path in final_members):
        plans = load_antisymmetric_router_plans(
            root,
            expected_config_contract_hash=config.contract_hash,
            expected_crossfit_partition_lock_hash=crossfit.lock_hash,
            expected_source_products_hash=source_products.source_products_hash,
            expected_source_products_lock_hash=source_products_lock_hash,
        )
        shutil.rmtree(root / "checkpoints/routes", ignore_errors=True)
        return plans

    checkpoint_root = root / "checkpoints/routes"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    index_rows = [dict(row) for row in source_products.index_rows]
    tasks: list[dict[str, object]] = []
    for ordinal, target in enumerate(CENTERS):
        target_rows = (
            base_partitions.support_rows_by_center[target]
            + base_partitions.evaluation_rows_by_center[target]
        )
        folds = crossfit.folds_by_target[target]
        tasks.append(
            {
                "target_center": target,
                "device": KERNEL_DEVICES[ordinal % len(KERNEL_DEVICES)],
                "config_contract_hash": config.contract_hash,
                "source_array_path": str(source_products.array_path),
                "source_index_rows": index_rows,
                "source_products_hash": source_products.source_products_hash,
                "source_products_lock_hash": source_products_lock_hash,
                "support_partition_lock_hash": base_partitions.lock_hash,
                "crossfit_partition_lock_hash": crossfit.lock_hash,
                "target_embeddings": frame.embeddings_for(target_rows),
                "target_sample_ids": tuple(row.sample_id for row in target_rows),
                "target_case_ids": tuple(row.case_id for row in target_rows),
                "folds": [_fold_task_payload(fold) for fold in folds],
                "prior_control": config.prior_control,
                "conditional_contrast": config.conditional_contrast,
                "residual_optimization": config.residual_optimization,
                "classifier_threads": int(
                    config.runtime["classifier_threads_per_worker"]
                ),
                "checkpoint_path": str(checkpoint_root / f"target_{target}.npz"),
            }
        )

    completed: dict[
        str, tuple[tuple[Mapping[str, object], ...], Mapping[str, np.ndarray]]
    ] = {}
    pending: list[dict[str, object]] = []
    for task in tasks:
        path = Path(str(task["checkpoint_path"]))
        if path.is_file():
            completed[str(task["target_center"])] = _load_target_checkpoint(
                path, task=task
            )
        else:
            pending.append(task)
    if pending:
        context = mp.get_context("spawn")
        executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=context)
            for _ in KERNEL_DEVICES
        ]
        futures: dict[Future[dict[str, object]], dict[str, object]] = {}
        try:
            for task in pending:
                device_index = KERNEL_DEVICES.index(str(task["device"]))
                futures[executors[device_index].submit(_plan_target_task, task)] = task
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                path = Path(str(task["checkpoint_path"]))
                _write_target_checkpoint(
                    path,
                    plans=result["plans"],
                    state=result["state"],
                )
                completed[str(task["target_center"])] = _load_target_checkpoint(
                    path, task=task
                )
                print(
                    f"[antisymmetric-mmd] target workspaces {len(completed)}/{len(CENTERS)}",
                    flush=True,
                )
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    if tuple(sorted(completed)) != tuple(sorted(CENTERS)):
        raise ProtocolError("Antisymmetric target-workspace coverage is incomplete.")

    plans_in_order = tuple(
        plan for target in CENTERS for plan in completed[target][0]
    )
    if len(plans_in_order) != EXPECTED_CROSS_FIT_FOLD_COUNT:
        raise ProtocolError("Antisymmetric plan coverage drifted.")
    plans_by_fold = {str(plan["fold_id"]): plan for plan in plans_in_order}
    if tuple(plans_by_fold) != tuple(fold.fold_id for fold in crossfit.folds):
        raise ProtocolError("Antisymmetric plan order differs from the fold lock.")
    state_arrays = {
        f"center_{target}_{key}": np.asarray(value)
        for target in CENTERS
        for key, value in completed[target][1].items()
    }
    plans = _write_plan_artifacts(
        root,
        plans_in_order=plans_in_order,
        state_arrays=state_arrays,
        source_products=source_products,
        config_contract_hash=config.contract_hash,
        support_partition_lock_hash=base_partitions.lock_hash,
        crossfit_partition_lock_hash=crossfit.lock_hash,
        source_products_lock_hash=source_products_lock_hash,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return plans


def _plan_target_task(task: Mapping[str, object]) -> dict[str, object]:
    workspace = build_target_kernel_workspace(task)
    plans: list[dict[str, object]] = []
    for fold in task["folds"]:
        if not isinstance(fold, Mapping):
            raise ProtocolError("Antisymmetric target task contains a malformed fold.")
        support_cases = tuple(str(value) for value in fold["router_support_case_ids"])
        heldout_case = str(fold["heldout_case_id"])
        protocol = MMDKMMProtocol(
            target_center=workspace.target_center,
            candidate_sources=candidate_sources(workspace.target_center),
            support_case_ids=support_cases,
            evaluation_case_ids=(heldout_case,),
            common_frame_hash=(
                str(task["common_frame_hash"])
                if "common_frame_hash" in task
                else workspace.source_replicas[0].kernel_features.common_frame_hash
            ),
            target_support_role=CROSSFIT_COHORT_SUPPORT_ROLE,
            evaluation_embeddings_available_to_router=True,
            cross_fitted_transductive_diagnostic=True,
            cohort_evaluation_embeddings_available_for_other_case_routes=True,
            heldout_evaluation_embeddings_available_to_own_route=False,
            previous_stage90_router_or_utility_inputs_used=False,
        )
        target_support = workspace.target_support(
            protocol,
            tuple(str(value) for value in fold["router_support_sample_ids"]),
            prior_control=task["prior_control"],
        )
        conditional = task["conditional_contrast"]
        base = build_conditional_contrast_problem(
            protocol,
            workspace.source_replicas,
            target_support,
            config=conditional,
        )
        solution = solve_antisymmetric_residual_mmd(
            base,
            support_case_problems=build_conditional_support_case_problems(
                protocol,
                workspace.source_replicas,
                target_support,
                config=conditional,
            ),
            training_seed_problems=build_conditional_seed_axis_problems(
                protocol,
                workspace.source_replicas,
                target_support,
                config=conditional,
                axis="training_seed",
            ),
            generation_seed_problems=build_conditional_seed_axis_problems(
                protocol,
                workspace.source_replicas,
                target_support,
                config=conditional,
                axis="generation_seed",
            ),
            prior_sensitivity_problems=build_conditional_prior_sensitivity_problems(
                protocol,
                workspace.source_replicas,
                target_support,
                conditional_config=conditional,
                prior_config=task["prior_control"],
            ),
            config=task["residual_optimization"],
        )
        allocation = build_antisymmetric_allocation(
            solution.delta,
            total_per_class=TOTAL_PER_CLASS,
        )
        control_count = TOTAL_PER_CLASS // len(protocol.candidate_sources)
        control_allocations = {
            str(label): {
                source: control_count for source in protocol.candidate_sources
            }
            for label in (0, 1)
        }
        routed_allocations = {
            "0": dict(allocation.class_0_allocations),
            "1": dict(allocation.class_1_allocations),
        }
        fallback_reason = solution.fallback_reason
        if fallback_reason is None and routed_allocations == control_allocations:
            fallback_reason = "integer_allocation_equals_control_uniform"
        used_fallback = fallback_reason is not None
        if used_fallback:
            routed_allocations = control_allocations
        uniform = {
            source: 1.0 / len(protocol.candidate_sources)
            for source in protocol.candidate_sources
        }
        class_zero = uniform if used_fallback else dict(solution.class_0_weights)
        class_one = uniform if used_fallback else dict(solution.class_1_weights)
        final_delta = (
            {source: 0.0 for source in protocol.candidate_sources}
            if used_fallback
            else dict(solution.delta)
        )
        unhashed: dict[str, object] = {
            "schema_version": "midogpp_antisymmetric_residual_mmd_case_plan_v1",
            "config_contract_hash": str(task["config_contract_hash"]),
            "support_partition_lock_hash": str(task["support_partition_lock_hash"]),
            "crossfit_partition_lock_hash": str(task["crossfit_partition_lock_hash"]),
            "source_products_hash": str(task["source_products_hash"]),
            "source_products_lock_hash": str(task["source_products_lock_hash"]),
            "fold_ordinal": int(fold["fold_ordinal"]),
            "fold_id": str(fold["fold_id"]),
            "fold_hash": str(fold["fold_hash"]),
            "target_center": workspace.target_center,
            "heldout_case_id": heldout_case,
            "heldout_row_identity_hash": str(fold["heldout_row_identity_hash"]),
            "router_support_row_identity_hash": str(
                fold["router_support_row_identity_hash"]
            ),
            "router_support_case_ids": list(support_cases),
            "candidate_sources": list(protocol.candidate_sources),
            "control_weights": uniform,
            "delta": final_delta,
            "class_0_weights": class_zero,
            "class_1_weights": class_one,
            "control_allocations_by_class": control_allocations,
            "routed_allocations_by_class": routed_allocations,
            "allocation_hash": allocation.allocation_hash,
            "used_uniform_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "solution": _solution_payload(solution),
            "preprocessing_hash": workspace.preprocessing_hash,
            "candidate_pool_fit_hash": workspace.candidate_pool_fit_hash,
            "kernel_map_hash": workspace.kernel_map_hash,
            "prior_model_hash": workspace.prior_model_hash,
            "prior_fit_pool_hash": workspace.prior_fit_pool_hash,
            "kernel_gpu_probe_max_abs_error": workspace.gpu_probe_max_abs_error,
            "target_labels_used": False,
            "support_labels_used": False,
            "heldout_case_embeddings_used_for_own_route": False,
            "cohort_embeddings_used_for_other_case_routes": True,
            "all_retained_seeds_aggregated": True,
            "claim_role": "label_free_proxy_compatibility_only",
            "proxy_is_nelbo_compatibility": False,
            "proxy_is_downstream_utility": False,
            "diagnostic_only": True,
            "promotion_eligible": False,
        }
        plans.append({**unhashed, "plan_hash": stable_hash(unhashed)})
    return {"plans": plans, "state": workspace.state_arrays}


__all__ = (
    "ROUTER_PLAN_COLUMNS",
    "ROUTER_PLAN_LOCK_MEMBER",
    "ROUTER_PLAN_TABLE_MEMBER",
    "ROUTER_STATE_MEMBER",
    "TARGET_ASSIGNMENT_COLUMNS",
    "TARGET_ASSIGNMENT_MEMBER",
    "AntisymmetricRouterPlans",
    "build_antisymmetric_router_plans",
    "load_antisymmetric_router_plans",
)
