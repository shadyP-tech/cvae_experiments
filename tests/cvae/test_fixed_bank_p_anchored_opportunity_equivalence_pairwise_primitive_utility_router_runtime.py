from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import multiprocessing as mp
from pathlib import Path
import pickle
import struct

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.dtos import (
    MemmapSliceDTO,
    OuterFoldTaskDTO,
    PredictionTaskDTO,
    WorkerResultDTO,
    assert_pickle_safe_label_free_dto,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.workstation import (
    BLAS_ENVIRONMENT_NAMES,
    CPU_WORKER_MARKER,
    GPU_WORKER_MARKER,
    assert_coordinator_process,
    build_workstation_plan,
    capture_threadpool_evidence,
    initialize_cpu_outer_worker,
    initialize_persistent_gpu_worker,
    validate_worker_environment,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.fold_scope import (
    FoldScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.runner import (
    run_planned_router,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.source_fence import (
    SourceFenceReceipt,
    validate_source_fence,
    validate_source_fence_receipt,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "b" * 64


def _spawn_memmap_probe(
    dto: MemmapSliceDTO,
) -> tuple[bool, bool, float, bool, int, tuple[int, ...]]:
    """Top-level spawn callback accepting only the approved primitive DTO."""

    assert_pickle_safe_label_free_dto(dto)
    import numpy as np

    with open(dto.path, "rb") as handle:
        content_matches = hashlib.sha256(handle.read()).hexdigest() == dto.content_sha256
    rows = np.memmap(
        dto.path,
        dtype=dto.dtype,
        mode=dto.mode,
        offset=dto.byte_offset,
        shape=dto.shape,
    )
    evidence = capture_threadpool_evidence(role="cpu")
    return (
        content_matches,
        bool(rows.flags.writeable),
        float(np.sum(rows, dtype=np.float64)),
        evidence.limiter_scope_entered and evidence.observation_performed,
        evidence.loaded_pool_count,
        tuple(row.num_threads for row in evidence.loaded_pools),
    )


def _spawn_validate_source_receipt(receipt: SourceFenceReceipt) -> str:
    return validate_source_fence_receipt(receipt).combined_source_seal_hash


def test_workstation_plan_is_coarse_non_nested_and_non_authorized() -> None:
    plan = build_workstation_plan()
    assert plan.gpu_devices == ("cuda:0", "cuda:1")
    assert plan.persistent_gpu_prediction_workers == 2
    assert plan.prediction_store_dtype == "float32"
    assert plan.prediction_store_mode == "read_only_memmap"
    assert plan.reduction_dtype == "float64"
    assert plan.cpu_outer_workers == 4
    assert plan.blas_threads_per_worker == 1
    assert plan.native_threads_per_worker == 1
    assert plan.multiprocessing_start_method == "spawn"
    assert plan.nested_pools_allowed is False
    assert plan.process_transport == ("paths", "hashes", "tuples", "scalars")
    assert plan.execution_authorized is False


def test_cpu_and_gpu_worker_environment_is_role_separated_and_one_thread() -> None:
    cpu: dict[str, str] = {}
    initialize_cpu_outer_worker(cpu)
    validate_worker_environment(cpu, role="cpu")
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu[CPU_WORKER_MARKER] == "1"
    assert all(cpu[name] == "1" for name in BLAS_ENVIRONMENT_NAMES)
    with pytest.raises(ProtocolError, match="nested CPU"):
        initialize_cpu_outer_worker(cpu)
    with pytest.raises(ProtocolError, match="nested process pools"):
        assert_coordinator_process(cpu)

    gpu: dict[str, str] = {}
    initialize_persistent_gpu_worker(1, gpu)
    validate_worker_environment(gpu, role="gpu")
    assert gpu["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu[GPU_WORKER_MARKER] == "1"
    with pytest.raises(ProtocolError, match="role-changing GPU"):
        initialize_persistent_gpu_worker(0, gpu)


def test_primitive_label_free_dtos_are_pickle_safe() -> None:
    memmap = MemmapSliceDTO("/tmp/predictions.f32", SHA, SHA, (3, 4), 0, 48)
    prediction = PredictionTaskDTO(
        "prediction-0",
        0,
        ("/tmp/input.bin",),
        (SHA,),
        SHA,
        "/tmp/predictions.f32",
        ("route-0", "route-1"),
    )
    scope = FoldScope("0", "1", "2", "3", "case-d")
    outer = OuterFoldTaskDTO(
        scope.H,
        scope.J,
        scope.K,
        scope.L,
        scope.d,
        scope.scope_hash,
        "/tmp/predictions.f32",
        SHA,
        SHA,
        SHA,
    )
    result = WorkerResultDTO(
        outer.task_hash,
        ("/tmp/result.bin",),
        (SHA,),
        10,
        SHA,
        SHA,
    )
    for row in (memmap, prediction, outer, result):
        assert_pickle_safe_label_free_dto(row)
        assert pickle.loads(pickle.dumps(row)) == row
    with pytest.raises(ProtocolError, match="not an approved DTO"):
        assert_pickle_safe_label_free_dto({"labels": (0, 1)})


def test_spawn_worker_reads_float32_memmap_without_write_or_mappingproxy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.f32"
    payload = struct.pack("=6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    path.write_bytes(payload)
    dto = MemmapSliceDTO(
        str(path.resolve()),
        hashlib.sha256(payload).hexdigest(),
        SHA,
        (2, 3),
        0,
        len(payload),
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=initialize_cpu_outer_worker,
    ) as executor:
        result = executor.submit(_spawn_memmap_probe, dto).result(timeout=30)
    content_matches, writeable, total, observed, pool_count, thread_counts = result
    assert content_matches is True
    assert writeable is False
    assert total == pytest.approx(21.0)
    assert observed is True
    assert pool_count == len(thread_counts)
    assert all(count == 1 for count in thread_counts)


def test_dtos_reject_predecessor_paths_and_mappingproxy_style_payloads() -> None:
    with pytest.raises(ProtocolError, match="forbidden predecessor"):
        MemmapSliceDTO(
            "/tmp/fixed_bank_p_anchored_support_calibrated_local_action_"
            "empirical_bayes_boundary_projected_router_v2/predictions.f32",
            SHA,
            SHA,
            (1,),
            0,
            4,
        )
    with pytest.raises(ProtocolError, match="byte extent"):
        MemmapSliceDTO("/tmp/x.f32", SHA, SHA, (2,), 0, 4)


class _PoisonPath:
    def __str__(self) -> str:
        raise AssertionError("runner coerced a forbidden run path")

    def __fspath__(self) -> str:
        raise AssertionError("runner resolved a forbidden run path")


def test_runner_rejects_before_output_or_scratch_resolution_and_mutation(
    tmp_path: Path,
) -> None:
    config = build_planned_config()
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_planned_router(
            config,
            artifact_root=_PoisonPath(),
            scratch_root=_PoisonPath(),
        )
    assert not output.exists()
    assert not scratch.exists()


def test_runner_rejects_source_pin_drift_before_run_path_coercion() -> None:
    config = build_planned_config()
    provenance = config.source_provenance
    provenance["adapter_tree_sha256"] = "0" * 64
    poisoned = {
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "input_artifact_ids": config.input_artifact_ids,
        "execution_authorized": False,
        "contract_hash": config.contract_hash,
        "protocol": config.protocol,
        "runtime": config.runtime,
        "claim_boundary": config.claim_boundary,
        "inputs": config.inputs,
        "source_provenance": provenance,
    }
    with pytest.raises(ProtocolError, match="adapter tree drifted"):
        run_planned_router(
            poisoned,
            artifact_root=_PoisonPath(),
            scratch_root=_PoisonPath(),
        )


def test_live_package_source_fence_passes_read_only() -> None:
    receipt = validate_source_fence()
    assert receipt.adapter.role == "diagnostic_adapter"
    assert receipt.core.role == "neutral_scientific_core"
    assert receipt.adapter_member_count >= 13
    assert receipt.core_member_count >= 10
    assert receipt.member_count == (
        receipt.adapter_member_count + receipt.core_member_count
    )
    assert receipt.adapter.mutation_call_count == 0
    assert receipt.core.mutation_call_count == 0
    assert len(receipt.adapter_tree_sha256) == 64
    assert len(receipt.core_tree_sha256) == 64
    assert receipt.adapter_tree_sha256 != receipt.core_tree_sha256
    assert len(receipt.combined_source_seal_hash) == 64
    assert len(receipt.receipt_hash) == 64
    assert pickle.loads(pickle.dumps(receipt)) == receipt
    validate_source_fence(
        expected_adapter_tree_sha256=receipt.adapter_tree_sha256,
        expected_core_tree_sha256=receipt.core_tree_sha256,
        expected_combined_source_seal_hash=receipt.combined_source_seal_hash,
    )


def test_combined_source_receipt_round_trips_through_spawn() -> None:
    receipt = validate_source_fence()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        observed = executor.submit(
            _spawn_validate_source_receipt, receipt
        ).result(timeout=30)
    assert observed == receipt.combined_source_seal_hash
    poisoned = replace(receipt, combined_source_seal_hash="0" * 64)
    with pytest.raises(ProtocolError, match="combined source receipt drifted"):
        validate_source_fence_receipt(poisoned)


def test_core_source_scope_rejects_runtime_import_and_mutation_api(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    core = tmp_path / "core"
    adapter.mkdir()
    core.mkdir()
    (adapter / "adapter.py").write_text("VALUE = 1\n", encoding="utf-8")
    core_member = core / "core.py"
    core_member.write_text("import os\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="forbidden runtime capability"):
        validate_source_fence(adapter, core_root=core)
    core_member.write_text("sink.write_text('forbidden')\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="forbidden mutation API"):
        validate_source_fence(adapter, core_root=core)
