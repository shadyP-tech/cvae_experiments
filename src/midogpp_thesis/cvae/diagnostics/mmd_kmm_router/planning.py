"""Target-excluded MMD/KMM route planning with shared GPU kernel maps."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import (
    build_hamilton_allocation,
    residual_soft_weights,
)
from ...routing.mmd_kmm_mixture import (
    EnergyDirectionReference,
    FrozenNystroemFeatureMap,
    MMDKMMProtocol,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    TransformedKernelFeatures,
    build_kernel_mean_problem,
    build_prior_sensitivity_problems,
    build_seed_axis_problems,
    build_support_case_problems,
    prepare_source_only_responsibilities,
    route_mmd_kmm,
)
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    COMMON_FRAME_HASH,
    DUPLICATE_DIRECTION_COSINE,
    DUPLICATE_WEIGHT_L1,
    ENERGY_REFERENCE_RHO,
    ENERGY_REFERENCE_TEMPERATURE,
    GENERATION_SEEDS,
    KERNEL_BATCH_ROWS,
    KERNEL_DEVICES,
    MAX_SOURCE_PREFIX_PER_CLASS,
    NYSTROEM_COMPONENTS,
    NYSTROEM_GAMMA,
    NYSTROEM_RANDOM_STATE,
    PRIOR_CLASSIFIER,
    ROUTER_PREFIX_PER_CLASS,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    candidate_sources,
)
from .inputs import LabelFreeValidationFrame, PartitionSurface
from .source_products import SourceProducts


ROUTER_STATE_MEMBER = "arrays/router_states.npz"
ROUTER_PLAN_LOCK_MEMBER = "manifests/router_plan_lock.json"
ROUTER_PLAN_TABLE_MEMBER = "tables/router_plans.csv"
TARGET_ASSIGNMENT_MEMBER = "tables/target_assignments.csv"

ROUTER_PLAN_COLUMNS = (
    "schema_version",
    "target_center",
    "candidate_sources_json",
    "final_weights_json",
    "control_weights_json",
    "mmd_allocations_per_class_json",
    "control_allocations_per_class_json",
    "used_uniform_fallback",
    "fallback_reason",
    "proxy_improvement",
    "mmd_squared",
    "uniform_mmd_squared",
    "effective_source_count",
    "maximum_source_weight",
    "support_stability_passed",
    "training_seed_stability_passed",
    "generation_seed_stability_passed",
    "prior_sensitivity_stability_passed",
    "energy_direction_cosine",
    "energy_weight_l1_distance",
    "duplicate_energy_direction",
    "preprocessing_hash",
    "candidate_pool_fit_hash",
    "kernel_map_hash",
    "prior_model_hash",
    "prior_fit_pool_hash",
    "plan_hash",
    "target_labels_used",
    "support_labels_used",
    "diagnostic_only",
)
TARGET_ASSIGNMENT_COLUMNS = (
    "schema_version",
    "target_center",
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
class RouterPlans:
    plans_by_target: Mapping[str, Mapping[str, object]]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["router_plan_lock_hash"])


def build_router_plans(
    config: MMDKMMRouterDiagnosticConfig,
    source_products: SourceProducts,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    *,
    source_products_lock_hash: str,
    root: Path,
) -> RouterPlans:
    final_members = (
        root / ROUTER_STATE_MEMBER,
        root / ROUTER_PLAN_LOCK_MEMBER,
        root / ROUTER_PLAN_TABLE_MEMBER,
        root / TARGET_ASSIGNMENT_MEMBER,
    )
    if all(path.is_file() for path in final_members):
        plans = load_router_plans(
            root,
            expected_config_contract_hash=config.contract_hash,
            expected_support_partition_lock_hash=partitions.lock_hash,
            expected_source_products_hash=source_products.source_products_hash,
            expected_source_products_lock_hash=source_products_lock_hash,
        )
        shutil.rmtree(root / "checkpoints/routes", ignore_errors=True)
        return plans

    checkpoint_root = root / "checkpoints/routes"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    index_rows = [dict(row) for row in source_products.index_rows]
    for ordinal, target in enumerate(CENTERS):
        support_rows = partitions.support_rows_by_center[target]
        tasks.append(
            {
                "target_center": target,
                "device": KERNEL_DEVICES[ordinal % len(KERNEL_DEVICES)],
                "config_contract_hash": config.contract_hash,
                "source_array_path": str(source_products.array_path),
                "source_index_rows": index_rows,
                "support_embeddings": frame.embeddings_for(support_rows),
                "support_case_ids": tuple(row.case_id for row in support_rows),
                "evaluation_case_ids": tuple(
                    sorted({row.case_id for row in partitions.evaluation_rows_by_center[target]})
                ),
                "calibrated_energy": dict(source_products.calibrated_energy_by_target[target]),
                "checkpoint_path": str(checkpoint_root / f"target_{target}.npz"),
                "support_partition_lock_hash": partitions.lock_hash,
                "source_products_hash": source_products.source_products_hash,
                "source_products_lock_hash": source_products_lock_hash,
                "prior_control": config.prior_control,
                "optimization": config.optimization,
                "gates": config.gates,
                "classifier_threads": int(config.runtime["classifier_threads_per_worker"]),
            }
        )

    completed: dict[str, tuple[Mapping[str, object], Mapping[str, np.ndarray]]] = {}
    pending: list[dict[str, object]] = []
    for task in tasks:
        path = Path(str(task["checkpoint_path"]))
        if not path.is_file():
            pending.append(task)
            continue
        completed[str(task["target_center"])] = _load_route_checkpoint(path, task=task)

    if pending:
        context = mp.get_context("spawn")
        executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=context)
            for _ in KERNEL_DEVICES
        ]
        future_to_task: dict[Future[dict[str, object]], dict[str, object]] = {}
        try:
            for task in pending:
                device_index = KERNEL_DEVICES.index(str(task["device"]))
                future_to_task[executors[device_index].submit(_route_target_task, task)] = task
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                _write_route_checkpoint(
                    Path(str(task["checkpoint_path"])),
                    payload=result["payload"],
                    state=result["state"],
                )
                loaded = _load_route_checkpoint(
                    Path(str(task["checkpoint_path"])), task=task
                )
                completed[str(task["target_center"])] = loaded
                print(f"[mmd-kmm] route plans {len(completed)}/{len(CENTERS)}", flush=True)
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    if tuple(sorted(completed)) != tuple(sorted(CENTERS)):
        raise ProtocolError("MMD/KMM route-plan checkpoint coverage is incomplete.")

    plans = {target: completed[target][0] for target in CENTERS}
    state_arrays: dict[str, np.ndarray] = {}
    for target in CENTERS:
        for key, value in completed[target][1].items():
            state_arrays[f"center_{target}_{key}"] = np.asarray(value)
    state_path = root / ROUTER_STATE_MEMBER
    _atomic_save_npz(state_path, state_arrays)
    from ...reporting import write_csv_rows

    write_csv_rows(
        root / ROUTER_PLAN_TABLE_MEMBER,
        [_plan_table_row(plans[target]) for target in CENTERS],
        columns=ROUTER_PLAN_COLUMNS,
    )
    write_csv_rows(
        root / TARGET_ASSIGNMENT_MEMBER,
        _assignment_rows(plans, source_products),
        columns=TARGET_ASSIGNMENT_COLUMNS,
    )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_router_plan_lock_v1",
        "status": "LOCKED_BEFORE_TARGET_PREDICTIONS",
        "config_contract_hash": config.contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_products_hash": source_products.source_products_hash,
        "source_products_lock_hash": source_products_lock_hash,
        "router_state_member": ROUTER_STATE_MEMBER,
        "router_state_sha256": _sha256_file(state_path),
        "router_plan_table_sha256": _sha256_file(root / ROUTER_PLAN_TABLE_MEMBER),
        "target_assignment_table_sha256": _sha256_file(
            root / TARGET_ASSIGNMENT_MEMBER
        ),
        "target_count": len(CENTERS),
        "plans": [plans[target] for target in CENTERS],
        "target_labels_used": False,
        "evaluation_embeddings_used_for_router": False,
        "support_labels_used": False,
        "previous_stage90_router_or_utility_inputs_used": False,
        "claim_role": "proxy_compatibility_only",
        "promotion_eligible": False,
    }
    lock = {**unhashed, "router_plan_lock_hash": stable_hash(unhashed)}
    _atomic_json(root / ROUTER_PLAN_LOCK_MEMBER, lock)
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return RouterPlans(plans_by_target=plans, lock_payload=lock)


def load_router_plans(
    root: Path,
    *,
    expected_config_contract_hash: str | None = None,
    expected_support_partition_lock_hash: str | None = None,
    expected_source_products_hash: str | None = None,
    expected_source_products_lock_hash: str | None = None,
) -> RouterPlans:
    lock = _json(root / ROUTER_PLAN_LOCK_MEMBER)
    unhashed = {key: value for key, value in lock.items() if key != "router_plan_lock_hash"}
    plans_raw = lock.get("plans")
    if (
        lock.get("router_plan_lock_hash") != stable_hash(unhashed)
        or lock.get("status") != "LOCKED_BEFORE_TARGET_PREDICTIONS"
        or not isinstance(plans_raw, list)
        or len(plans_raw) != len(CENTERS)
        or lock.get("router_state_sha256") != _sha256_file(root / ROUTER_STATE_MEMBER)
        or lock.get("router_plan_table_sha256")
        != _sha256_file(root / ROUTER_PLAN_TABLE_MEMBER)
        or lock.get("target_assignment_table_sha256")
        != _sha256_file(root / TARGET_ASSIGNMENT_MEMBER)
        or (
            expected_config_contract_hash is not None
            and lock.get("config_contract_hash") != expected_config_contract_hash
        )
        or (
            expected_support_partition_lock_hash is not None
            and lock.get("support_partition_lock_hash")
            != expected_support_partition_lock_hash
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
        raise ProtocolError("MMD/KMM router-plan lock failed validation.")
    plans = {
        str(plan["target_center"]): plan
        for plan in plans_raw
        if isinstance(plan, Mapping)
    }
    if tuple(plans) != CENTERS:
        raise ProtocolError("MMD/KMM router-plan target order/coverage drifted.")
    return RouterPlans(plans_by_target=plans, lock_payload=lock)


def _route_target_task(task: Mapping[str, object]) -> dict[str, object]:
    target = str(task["target_center"])
    candidates = candidate_sources(target)
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): int(row["block_ordinal"])
        for row in task["source_index_rows"]
    }
    pool_rows: list[np.ndarray] = []
    pool_labels: list[np.ndarray] = []
    replica_slices: list[tuple[str, int, int, int, int, int]] = []
    cursor = 0
    block_hashes: list[str] = []
    index_by_key = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in task["source_index_rows"]
    }
    for source in candidates:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                block = source_array[index[(source, training_seed, generation_seed)]]
                block_hashes.append(str(index_by_key[(source, training_seed, generation_seed)]["output_sha256"]))
                for label, start in ((0, 0), (1, MAX_SOURCE_PREFIX_PER_CLASS)):
                    values = np.ascontiguousarray(
                        block[start : start + ROUTER_PREFIX_PER_CLASS], dtype=np.float64
                    )
                    pool_rows.append(values)
                    pool_labels.append(np.full(ROUTER_PREFIX_PER_CLASS, label, dtype=np.int64))
                    stop = cursor + len(values)
                    replica_slices.append((source, training_seed, generation_seed, label, cursor, stop))
                    cursor = stop
    pool = np.ascontiguousarray(np.concatenate(pool_rows), dtype=np.float64)
    labels = np.concatenate(pool_labels)
    if pool.shape != (len(candidates) * 3 * 3 * 2 * ROUTER_PREFIX_PER_CLASS, 3840):
        raise ProtocolError("MMD/KMM target-excluded source pool geometry drifted.")
    candidate_pool_fit_hash = stable_hash(
        {
            "target_center": target,
            "candidate_sources": list(candidates),
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "prefix_per_class": ROUTER_PREFIX_PER_CLASS,
            "source_block_hashes": block_hashes,
        }
    )
    support = np.asarray(task["support_embeddings"], dtype=np.float64)
    try:
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependencies
        raise RuntimeError("MMD/KMM route planning requires scikit-learn and threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["classifier_threads"])):
        scaler = StandardScaler().fit(pool)
        scaled_pool = scaler.transform(pool)
        scaled_support = scaler.transform(support)
        prior = LogisticRegression(**PRIOR_CLASSIFIER.to_sklearn_kwargs()).fit(
            scaled_pool, labels
        )
        raw_probabilities = prior.predict_proba(scaled_support)
        nystroem = Nystroem(
            kernel="rbf",
            gamma=NYSTROEM_GAMMA,
            n_components=NYSTROEM_COMPONENTS,
            random_state=NYSTROEM_RANDOM_STATE,
        ).fit(scaled_pool)
    if tuple(int(value) for value in prior.classes_) != (0, 1) or int(np.max(prior.n_iter_)) >= PRIOR_CLASSIFIER.max_iter:
        raise ProtocolError("MMD/KMM source-only prior classifier did not converge.")
    preprocessing_hash = stable_hash(
        {
            "candidate_pool_fit_hash": candidate_pool_fit_hash,
            "mean_sha256": _sha256_array(np.asarray(scaler.mean_)),
            "var_sha256": _sha256_array(np.asarray(scaler.var_)),
            "scale_sha256": _sha256_array(np.asarray(scaler.scale_)),
            "fit_role": "target_excluded_candidate_pool_generated_common_frame",
        }
    )
    feature_map = FrozenNystroemFeatureMap(
        components=np.asarray(nystroem.components_, dtype=np.float64),
        normalization=np.asarray(nystroem.normalization_, dtype=np.float64),
        gamma=NYSTROEM_GAMMA,
        common_frame_hash=COMMON_FRAME_HASH,
        preprocessing_hash=preprocessing_hash,
        candidate_pool_fit_hash=candidate_pool_fit_hash,
        random_state=NYSTROEM_RANDOM_STATE,
    )
    transformed_pool, pool_probe_error = _transform_nystroem_batched(
        scaled_pool,
        feature_map,
        device=str(task["device"]),
        batch_rows=KERNEL_BATCH_ROWS,
    )
    transformed_support, support_probe_error = _transform_nystroem_batched(
        scaled_support,
        feature_map,
        device=str(task["device"]),
        batch_rows=KERNEL_BATCH_ROWS,
    )
    prior_fit_pool_hash = stable_hash(
        {
            "candidate_pool_fit_hash": candidate_pool_fit_hash,
            "balance": "equal_source_seed_class_prefix",
            "row_count": len(pool),
            "labels_sha256": _sha256_array(labels),
        }
    )
    prior_model_hash = stable_hash(
        {
            "classifier": PRIOR_CLASSIFIER.to_payload(),
            "preprocessing_hash": preprocessing_hash,
            "prior_fit_pool_hash": prior_fit_pool_hash,
            "coefficient_sha256": _sha256_array(np.asarray(prior.coef_)),
            "intercept_sha256": _sha256_array(np.asarray(prior.intercept_)),
            "classes": [int(value) for value in prior.classes_],
            "n_iter": [int(value) for value in prior.n_iter_],
        }
    )
    support_cases = tuple(str(value) for value in task["support_case_ids"])
    protocol = MMDKMMProtocol(
        target_center=target,
        candidate_sources=candidates,
        support_case_ids=tuple(sorted(set(support_cases))),
        evaluation_case_ids=tuple(str(value) for value in task["evaluation_case_ids"]),
        common_frame_hash=COMMON_FRAME_HASH,
        previous_stage90_router_or_utility_inputs_used=False,
    )
    prior_prediction = prepare_source_only_responsibilities(
        raw_probabilities,
        protocol=protocol,
        prior_model_hash=prior_model_hash,
        prior_fit_pool_hash=prior_fit_pool_hash,
        config=task["prior_control"],
    )
    support_features = TransformedKernelFeatures(
        values=transformed_support,
        common_frame_hash=COMMON_FRAME_HASH,
        preprocessing_hash=preprocessing_hash,
        candidate_pool_fit_hash=candidate_pool_fit_hash,
        kernel_map_hash=feature_map.kernel_map_hash,
    )
    target_support = TargetSupportKernelFeatures(
        target_center=target,
        case_ids=support_cases,
        kernel_features=support_features,
        prior_prediction=prior_prediction,
    )
    replicas: list[SourceKernelReplica] = []
    for source, training_seed, generation_seed, label, start, stop in replica_slices:
        features = TransformedKernelFeatures(
            values=transformed_pool[start:stop],
            common_frame_hash=COMMON_FRAME_HASH,
            preprocessing_hash=preprocessing_hash,
            candidate_pool_fit_hash=candidate_pool_fit_hash,
            kernel_map_hash=feature_map.kernel_map_hash,
        )
        replicas.append(
            SourceKernelReplica(
                source_center=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
                kernel_features=features,
            )
        )
    base = build_kernel_mean_problem(protocol, replicas, target_support)
    energy_scores = {str(key): float(value) for key, value in task["calibrated_energy"].items()}
    energy_weights = residual_soft_weights(
        energy_scores,
        rho=ENERGY_REFERENCE_RHO,
        temperature=ENERGY_REFERENCE_TEMPERATURE,
        max_source_weight=0.25,
        minimum_effective_sources=6.0,
    )
    energy_reference = EnergyDirectionReference(
        target_center=target,
        candidate_sources=candidates,
        support_partition_hash=protocol.support_partition_hash,
        common_frame_hash=COMMON_FRAME_HASH,
        preprocessing_hash=preprocessing_hash,
        candidate_pool_fit_hash=candidate_pool_fit_hash,
        kernel_map_hash=feature_map.kernel_map_hash,
        training_seeds=TRAINING_SEEDS,
        generation_seeds=GENERATION_SEEDS,
        weights=energy_weights.weights,
        energy_calibration_hash=stable_hash(
            {
                "support_partition_lock_hash": task["support_partition_lock_hash"],
                "target_center": target,
                "calibrated_energy": energy_scores,
                "rho": ENERGY_REFERENCE_RHO,
                "temperature": ENERGY_REFERENCE_TEMPERATURE,
            }
        ),
        action_id="rho_0.50",
    )
    decision = route_mmd_kmm(
        base,
        support_case_problems=build_support_case_problems(protocol, replicas, target_support),
        training_seed_problems=build_seed_axis_problems(protocol, replicas, target_support, axis="training_seed"),
        generation_seed_problems=build_seed_axis_problems(protocol, replicas, target_support, axis="generation_seed"),
        prior_sensitivity_problems=build_prior_sensitivity_problems(protocol, replicas, target_support, config=task["prior_control"]),
        energy_direction_reference=energy_reference,
        prior_control=task["prior_control"],
        optimization=task["optimization"],
        gates=task["gates"],
    )
    control_weights = {source: 1.0 / len(candidates) for source in candidates}
    control_allocation = build_hamilton_allocation(control_weights, total=TOTAL_PER_CLASS)
    final_weights = dict(decision.final_weights)
    mmd_allocation = build_hamilton_allocation(final_weights, total=TOTAL_PER_CLASS)
    energy_allocation = build_hamilton_allocation(
        energy_weights.weights, total=TOTAL_PER_CLASS
    )
    execution_fallback_reason = decision.fallback_reason
    if execution_fallback_reason is None and (
        dict(mmd_allocation.allocations) == dict(energy_allocation.allocations)
    ):
        execution_fallback_reason = "duplicate_energy_integer_allocation_uniform"
    if execution_fallback_reason is None and (
        dict(mmd_allocation.allocations) == dict(control_allocation.allocations)
    ):
        execution_fallback_reason = "integer_allocation_equals_control_uniform"
    if execution_fallback_reason is not None:
        final_weights = control_weights
        mmd_allocation = control_allocation
    audits = {audit.axis: audit for audit in decision.stability_audits}
    solution = decision.base_solution
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_target_plan_v1",
        "config_contract_hash": str(task["config_contract_hash"]),
        "support_partition_lock_hash": str(task["support_partition_lock_hash"]),
        "source_products_hash": str(task["source_products_hash"]),
        "source_products_lock_hash": str(task["source_products_lock_hash"]),
        "target_center": target,
        "candidate_sources": list(candidates),
        "final_weights": final_weights,
        "control_weights": control_weights,
        "mmd_allocations_per_class": dict(mmd_allocation.allocations),
        "control_allocations_per_class": dict(control_allocation.allocations),
        "used_uniform_fallback": execution_fallback_reason is not None,
        "fallback_reason": execution_fallback_reason,
        "base_solution": _solution_payload(solution),
        "stability_audits": [_audit_payload(audit) for audit in decision.stability_audits],
        "direction_identity": {
            "reference_role": decision.direction_identity.reference_role,
            "direction_cosine": decision.direction_identity.direction_cosine,
            "weight_l1_distance": decision.direction_identity.weight_l1_distance,
            "duplicate": decision.direction_identity.duplicate,
            "duplicate_direction_cosine_threshold": DUPLICATE_DIRECTION_COSINE,
            "duplicate_weight_l1_threshold": DUPLICATE_WEIGHT_L1,
        },
        "preprocessing_hash": preprocessing_hash,
        "candidate_pool_fit_hash": candidate_pool_fit_hash,
        "kernel_map_hash": feature_map.kernel_map_hash,
        "prior_model_hash": prior_model_hash,
        "prior_fit_pool_hash": prior_fit_pool_hash,
        "prior_control_hash": prior_prediction.prior_control_hash,
        "support_partition_hash": protocol.support_partition_hash,
        "energy_reference_weights": dict(energy_weights.weights),
        "energy_reference_allocations_per_class": dict(
            energy_allocation.allocations
        ),
        "integer_allocation_identity_gate_applied": True,
        "energy_calibration_hash": energy_reference.energy_calibration_hash,
        "kernel_gpu_probe_max_abs_error": max(pool_probe_error, support_probe_error),
        "target_labels_used": False,
        "support_labels_used": False,
        "evaluation_embeddings_used_for_router": False,
        "all_retained_seeds_aggregated": True,
        "previous_stage90_router_or_utility_inputs_used": False,
        "claim_role": "proxy_compatibility_only",
        "downstream_utility_claimed": False,
        "promotion_eligible": False,
    }
    payload = {**unhashed, "plan_hash": stable_hash(unhashed)}
    state = {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_var": np.asarray(scaler.var_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "kernel_components": np.asarray(feature_map.components, dtype=np.float64),
        "kernel_normalization": np.asarray(feature_map.normalization, dtype=np.float64),
        "prior_coef": np.asarray(prior.coef_, dtype=np.float64),
        "prior_intercept": np.asarray(prior.intercept_, dtype=np.float64),
    }
    return {"payload": payload, "state": state}


def _transform_nystroem_batched(
    values: np.ndarray,
    feature_map: FrozenNystroemFeatureMap,
    *,
    device: str,
    batch_rows: int,
) -> tuple[np.ndarray, float]:
    try:
        import torch
        from sklearn.metrics.pairwise import rbf_kernel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("MMD/KMM GPU kernel transform requires torch and scikit-learn.") from exc
    torch.backends.cuda.matmul.allow_tf32 = False
    components = torch.as_tensor(
        np.array(feature_map.components, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    normalization = torch.as_tensor(
        np.array(feature_map.normalization, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    component_norm = torch.sum(components * components, dim=1)[None, :]
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_rows)):
            batch = torch.as_tensor(values[start : start + int(batch_rows)], dtype=torch.float32, device=device)
            distances = torch.sum(batch * batch, dim=1)[:, None] + component_norm - 2.0 * (batch @ components.T)
            kernel = torch.exp(-float(feature_map.gamma) * torch.clamp(distances, min=0.0))
            transformed = kernel @ normalization.T
            outputs.append(transformed.cpu().numpy().astype(np.float64, copy=False))
    output = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float64)
    probe_count = min(3, len(values))
    expected = rbf_kernel(
        np.asarray(values[:probe_count], dtype=np.float64),
        np.asarray(feature_map.components, dtype=np.float64),
        gamma=float(feature_map.gamma),
    ) @ np.asarray(feature_map.normalization, dtype=np.float64).T
    error = float(np.max(np.abs(output[:probe_count] - expected)))
    if output.shape != (len(values), len(feature_map.components)) or not np.isfinite(output).all() or error > 5.0e-4:
        raise ProtocolError("MMD/KMM GPU Nyström transform failed its CPU probe.")
    return output, error


def _solution_payload(solution: object) -> dict[str, object]:
    return {
        "uniform_weights": dict(solution.uniform_weights),
        "weights": dict(solution.weights),
        "delta": dict(solution.delta),
        "proxy_objective": solution.proxy_objective,
        "uniform_proxy_objective": solution.uniform_proxy_objective,
        "proxy_improvement": solution.proxy_improvement,
        "mmd_squared": solution.mmd_squared,
        "uniform_mmd_squared": solution.uniform_mmd_squared,
        "regularization_value": solution.regularization_value,
        "effective_source_count": solution.effective_source_count,
        "maximum_source_weight": solution.maximum_source_weight,
        "used_uniform_fallback": solution.used_uniform_fallback,
        "fallback_reason": solution.fallback_reason,
        "solver_success": solution.solver_success,
        "solver_message": solution.solver_message,
        "solver_iterations": solution.solver_iterations,
        "solver_method": solution.solver_method,
        "solver_version": solution.solver_version,
        "optimality_residual": solution.optimality_residual,
    }


def _audit_payload(audit: object) -> dict[str, object]:
    return {
        "axis": audit.axis,
        "variant_ids": list(audit.variant_ids),
        "maximum_l1_distance": audit.maximum_l1_distance,
        "minimum_direction_cosine": audit.minimum_direction_cosine,
        "passed": audit.passed,
        "failure_reason": audit.failure_reason,
    }


def _plan_table_row(plan: Mapping[str, object]) -> dict[str, object]:
    audits = {str(row["axis"]): row for row in plan["stability_audits"]}
    base = plan["base_solution"]
    identity = plan["direction_identity"]
    return {
        "schema_version": "midogpp_mmd_kmm_router_plan_row_v1",
        "target_center": plan["target_center"],
        "candidate_sources_json": _compact(plan["candidate_sources"]),
        "final_weights_json": _compact(plan["final_weights"]),
        "control_weights_json": _compact(plan["control_weights"]),
        "mmd_allocations_per_class_json": _compact(plan["mmd_allocations_per_class"]),
        "control_allocations_per_class_json": _compact(plan["control_allocations_per_class"]),
        "used_uniform_fallback": plan["used_uniform_fallback"],
        "fallback_reason": plan["fallback_reason"] or "",
        "proxy_improvement": base["proxy_improvement"],
        "mmd_squared": base["mmd_squared"],
        "uniform_mmd_squared": base["uniform_mmd_squared"],
        "effective_source_count": base["effective_source_count"],
        "maximum_source_weight": base["maximum_source_weight"],
        "support_stability_passed": audits["support_case"]["passed"],
        "training_seed_stability_passed": audits["training_seed"]["passed"],
        "generation_seed_stability_passed": audits["generation_seed"]["passed"],
        "prior_sensitivity_stability_passed": audits["class_prior_sensitivity"]["passed"],
        "energy_direction_cosine": identity["direction_cosine"],
        "energy_weight_l1_distance": identity["weight_l1_distance"],
        "duplicate_energy_direction": identity["duplicate"],
        "preprocessing_hash": plan["preprocessing_hash"],
        "candidate_pool_fit_hash": plan["candidate_pool_fit_hash"],
        "kernel_map_hash": plan["kernel_map_hash"],
        "prior_model_hash": plan["prior_model_hash"],
        "prior_fit_pool_hash": plan["prior_fit_pool_hash"],
        "plan_hash": plan["plan_hash"],
        "target_labels_used": False,
        "support_labels_used": False,
        "diagnostic_only": True,
    }


def _assignment_rows(
    plans: Mapping[str, Mapping[str, object]],
    source_products: SourceProducts,
) -> list[dict[str, object]]:
    index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in source_products.index_rows
    }
    rows: list[dict[str, object]] = []
    for target in CENTERS:
        plan = plans[target]
        for arm, weights_key, allocation_key in (
            ("equal_union_control", "control_weights", "control_allocations_per_class"),
            ("mmd_kmm", "final_weights", "mmd_allocations_per_class"),
        ):
            weights = plan[weights_key]
            allocations = plan[allocation_key]
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for label in (0, 1):
                        for source in candidate_sources(target):
                            source_row = index[(source, training_seed, generation_seed)]
                            rows.append(
                                {
                                    "schema_version": "midogpp_mmd_kmm_target_assignment_v1",
                                    "target_center": target,
                                    "arm_role": arm,
                                    "training_seed": training_seed,
                                    "generation_seed": generation_seed,
                                    "class_label": label,
                                    "source_center": source,
                                    "source_stream_id": source_row["stream_id"],
                                    "prefix_count": allocations[source],
                                    "source_weight": weights[source],
                                    "target_expert_excluded": True,
                                    "seed_selected": False,
                                }
                            )
    return rows


def _write_route_checkpoint(
    path: Path,
    *,
    payload: Mapping[str, object],
    state: Mapping[str, np.ndarray],
) -> None:
    arrays = {key: np.asarray(value) for key, value in state.items()}
    state_hashes = {key: _sha256_array(value) for key, value in arrays.items()}
    checkpoint_unhashed = {
        "payload": dict(payload),
        "state_hashes": state_hashes,
    }
    checkpoint = {
        **checkpoint_unhashed,
        "checkpoint_hash": stable_hash(checkpoint_unhashed),
    }
    _atomic_save_npz(
        path,
        {**arrays, "checkpoint_json": np.asarray(_compact(checkpoint))},
    )


def _load_route_checkpoint(
    path: Path,
    *,
    task: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as raw:
            files = set(raw.files)
            if "checkpoint_json" not in files:
                raise ProtocolError("MMD/KMM route checkpoint lacks metadata.")
            checkpoint = json.loads(str(np.asarray(raw["checkpoint_json"]).item()))
            state = {
                key: np.asarray(raw[key])
                for key in files.difference({"checkpoint_json"})
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("MMD/KMM route checkpoint is unreadable.") from exc
    if not isinstance(checkpoint, Mapping):
        raise ProtocolError("MMD/KMM route checkpoint metadata is malformed.")
    unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_hash"}
    payload = checkpoint.get("payload")
    hashes = checkpoint.get("state_hashes")
    if (
        checkpoint.get("checkpoint_hash") != stable_hash(unhashed)
        or not isinstance(payload, Mapping)
        or not isinstance(hashes, Mapping)
        or payload.get("target_center") != task["target_center"]
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("support_partition_lock_hash")
        != task["support_partition_lock_hash"]
        or payload.get("source_products_hash") != task["source_products_hash"]
        or payload.get("source_products_lock_hash")
        != task["source_products_lock_hash"]
        or any(hashes.get(key) != _sha256_array(value) for key, value in state.items())
        or set(hashes) != set(state)
    ):
        raise ProtocolError("MMD/KMM route checkpoint failed validation.")
    plan_unhashed = {key: value for key, value in payload.items() if key != "plan_hash"}
    if payload.get("plan_hash") != stable_hash(plan_unhashed):
        raise ProtocolError("MMD/KMM route checkpoint plan hash drifted.")
    return payload, state


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM route JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("MMD/KMM route JSON must be an object.")
    return payload


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "ROUTER_PLAN_COLUMNS",
    "ROUTER_PLAN_LOCK_MEMBER",
    "ROUTER_PLAN_TABLE_MEMBER",
    "ROUTER_STATE_MEMBER",
    "TARGET_ASSIGNMENT_COLUMNS",
    "TARGET_ASSIGNMENT_MEMBER",
    "RouterPlans",
    "build_router_plans",
    "load_router_plans",
)
