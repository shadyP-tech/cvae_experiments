from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path
from types import MappingProxyType

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution import pools as pool_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.callbacks import (
    resolve_sealed_callback,
    seal_callback_descriptor,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.dtos import (
    OuterFoldTaskDTO,
    PredictionTaskDTO,
    WorkerExecutionDTO,
    WorkerResultDTO,
    assert_pickle_safe_label_free_dto,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.pools import (
    admit_worker_execution_transport,
    run_cpu_outer_pool,
    run_persistent_gpu_prediction_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.execution.surfaces import (
    CandidateProbabilitySurfaceReceipt,
    _build_strict_test_candidate_probability_surface_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.fold_scope import (
    FoldScope,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "b" * 64


def _synthetic_descriptor_prediction_callback(
    task: PredictionTaskDTO,
) -> WorkerResultDTO:
    return WorkerResultDTO(
        task_hash=task.task_hash,
        result_paths=(task.output_path,),
        result_hashes=(hashlib.sha256(task.task_hash.encode("ascii")).hexdigest(),),
        row_count=9928,
        row_index_sha256=task.row_index_sha256,
        source_surface_sha256=task.source_surface_sha256,
    )


def _regular_prediction_callback(task: PredictionTaskDTO) -> WorkerResultDTO:
    payload = f"regular-result:{task.task_hash}".encode("ascii")
    return WorkerResultDTO(
        task_hash=task.task_hash,
        result_paths=(task.output_path,),
        result_hashes=(hashlib.sha256(payload).hexdigest(),),
        row_count=9928,
        row_index_sha256=task.row_index_sha256,
        source_surface_sha256=task.source_surface_sha256,
    )


def _synthetic_descriptor_outer_callback(task: OuterFoldTaskDTO) -> WorkerResultDTO:
    return WorkerResultDTO(
        task_hash=task.task_hash,
        result_paths=(f"/tmp/oe-ppur-outer-{task.task_hash}.fixture",),
        result_hashes=(hashlib.sha256(task.task_hash.encode("ascii")).hexdigest(),),
        row_count=9928,
        row_index_sha256=task.row_index_sha256,
        source_surface_sha256=task.candidate_probability_surface_sha256,
    )


def _tasks() -> tuple[PredictionTaskDTO, ...]:
    return tuple(
        PredictionTaskDTO(
            task_id=f"sealed-{index}",
            device_index=index,
            input_paths=(f"/tmp/oe-ppur-sealed-input-{index}.bin",),
            input_hashes=(SHA,),
            row_index_sha256=SHA,
            output_path=f"/tmp/oe-ppur-sealed-output-{index}.bin",
            route_ids=(f"route-{index}",),
        )
        for index in (0, 1)
    )


def _outer_tasks(
    *,
    probability_sha256: str,
    candidate_probability_surface_sha256: str,
    row_index_sha256: str,
) -> tuple[OuterFoldTaskDTO, ...]:
    roles = (
        ("0", "1", "2", "3"),
        ("0", "2", "3", "5"),
        ("1", "0", "2", "3"),
        ("1", "2", "3", "5"),
    )
    rows = []
    for index, (H, J, K, L) in enumerate(roles):
        scope = FoldScope(H, J, K, L, f"transport-case-{index}")
        rows.append(
            OuterFoldTaskDTO(
                H=scope.H,
                J=scope.J,
                K=scope.K,
                L=scope.L,
                d=scope.d,
                scope_hash=scope.scope_hash,
                probability_path="/tmp/oe-ppur-transport-probabilities.f32",
                probability_sha256=probability_sha256,
                candidate_probability_surface_sha256=(
                    candidate_probability_surface_sha256
                ),
                row_index_sha256=row_index_sha256,
            )
        )
    return tuple(rows)


@pytest.fixture(scope="module")
def sealed_gpu_batch():
    descriptor = seal_callback_descriptor(
        _synthetic_descriptor_prediction_callback,
        callback_role="gpu_prediction",
    )
    batch = run_persistent_gpu_prediction_pool(
        _tasks(),
        callback=descriptor,
    )
    return descriptor, batch


def test_spawn_transports_only_sealed_descriptor_and_flat_primitive_dto(
    sealed_gpu_batch,
) -> None:
    descriptor, batch = sealed_gpu_batch
    assert descriptor.member_name == "_synthetic_descriptor_prediction_callback"
    assert descriptor.result_evidence_mode == "strict_test_fixture"
    assert resolve_sealed_callback(
        descriptor,
        expected_role="gpu_prediction",
    ) is _synthetic_descriptor_prediction_callback
    assert tuple(row.physical_device_index for row in batch.receipts) == (0, 1)
    assert tuple(row.cuda_visible_devices for row in batch.receipts) == ("0", "1")
    assert all(
        row.callback_visible_logical_device == "cuda:0" for row in batch.receipts
    )
    assert len(batch.worker_roster) == len(set(batch.worker_roster)) == 2
    for row in batch.receipts:
        assert isinstance(row, WorkerExecutionDTO)
        assert row.callback_descriptor_hash == descriptor.descriptor_hash
        assert_pickle_safe_label_free_dto(row)
        assert all(
            value is None
            or isinstance(value, (str, int, float, bool, tuple))
            for value in (getattr(row, item.name) for item in fields(row))
        )


def test_cpu_pool_seals_exact_four_pid_cuda_hidden_roster(
    sealed_gpu_batch,
) -> None:
    _, gpu_batch = sealed_gpu_batch
    surface = _build_strict_test_candidate_probability_surface_receipt(
        gpu_batch
    )
    descriptor = seal_callback_descriptor(
        _synthetic_descriptor_outer_callback,
        callback_role="cpu_outer",
    )
    batch = run_cpu_outer_pool(
        _outer_tasks(
            probability_sha256=surface.output_file_hashes[0],
            candidate_probability_surface_sha256=(
                surface.candidate_probability_surface_sha256
            ),
            row_index_sha256=surface.row_index_sha256,
        ),
        callback=descriptor,
        candidate_surface=surface,
    )
    assert batch.configured_worker_count == 4
    assert len(batch.worker_roster) == len(set(batch.worker_roster)) == 4
    assert all(row.physical_device_index is None for row in batch.receipts)
    assert all(row.cuda_visible_devices == "" for row in batch.receipts)
    assert all(row.callback_visible_logical_device == "cpu" for row in batch.receipts)


def test_descriptor_revalidates_exact_source_member_and_rejects_code_drift() -> None:
    descriptor = seal_callback_descriptor(
        _synthetic_descriptor_prediction_callback,
        callback_role="gpu_prediction",
    )
    assert_pickle_safe_label_free_dto(descriptor)
    poisoned = replace(descriptor, member_code_sha256="0" * 64)
    with pytest.raises(ProtocolError, match="source or code drifted"):
        resolve_sealed_callback(poisoned, expected_role="gpu_prediction")

    captured = SHA

    def closure(task: PredictionTaskDTO) -> WorkerResultDTO:
        assert captured
        return _synthetic_descriptor_prediction_callback(task)

    with pytest.raises(ProtocolError, match="sealed top-level member"):
        seal_callback_descriptor(closure, callback_role="gpu_prediction")


@pytest.mark.parametrize(
    "forbidden",
    (
        MappingProxyType({"x": 1}),
        {"x": 1},
        Path("/tmp/not-a-transport-value"),
        _synthetic_descriptor_prediction_callback,
    ),
)
def test_transport_rejects_mapping_path_and_callable_values(forbidden: object) -> None:
    with pytest.raises(ProtocolError, match="not an approved DTO"):
        assert_pickle_safe_label_free_dto(forbidden)


def test_coordinator_reopens_and_hashes_regular_result_bytes(
    tmp_path: Path,
    sealed_gpu_batch,
) -> None:
    _, batch = sealed_gpu_batch
    task = _tasks()[0]
    path = tmp_path / "result.bin"
    payload = b"sealed-real-result-bytes"
    path.write_bytes(payload)
    result = WorkerResultDTO(
        task_hash=task.task_hash,
        result_paths=(str(path.resolve()),),
        result_hashes=(hashlib.sha256(payload).hexdigest(),),
        row_count=9928,
        row_index_sha256=task.row_index_sha256,
        source_surface_sha256=task.source_surface_sha256,
    )
    descriptor = seal_callback_descriptor(
        _regular_prediction_callback,
        callback_role="gpu_prediction",
        result_evidence_mode="regular_file",
    )
    transported = replace(
        batch.receipts[0],
        callback_descriptor_hash=descriptor.descriptor_hash,
        result_paths=result.result_paths,
        result_hashes=result.result_hashes,
        worker_result_hash=result.result_hash,
        result_evidence_mode="regular_file",
        input_bytes_revalidated=True,
    )
    assert admit_worker_execution_transport(
        transported,
        descriptor=descriptor,
        task=task,
    ) is transported

    path.write_bytes(b"tampered")
    with pytest.raises(ProtocolError, match="result bytes drifted"):
        admit_worker_execution_transport(
            transported,
            descriptor=descriptor,
            task=task,
        )


def test_regular_launch_requires_a_separate_successor_before_callback_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = seal_callback_descriptor(
        _regular_prediction_callback,
        callback_role="gpu_prediction",
        result_evidence_mode="regular_file",
    )
    resolved = False

    def _poison_resolver(*_args: object, **_kwargs: object) -> object:
        nonlocal resolved
        resolved = True
        raise AssertionError("unauthorized callback was resolved")

    monkeypatch.setattr(pool_module, "resolve_sealed_callback", _poison_resolver)
    with pytest.raises(ProtocolError, match="not authorized"):
        run_persistent_gpu_prediction_pool(
            _tasks(),
            callback=descriptor,
        )
    assert resolved is False


def test_worker_regular_input_bytes_are_reopened_and_rehashed(tmp_path: Path) -> None:
    path = tmp_path / "worker-input.bin"
    payload = b"exact-worker-input"
    path.write_bytes(payload)
    task = PredictionTaskDTO(
        task_id="regular-input",
        device_index=0,
        input_paths=(str(path.resolve()),),
        input_hashes=(hashlib.sha256(payload).hexdigest(),),
        row_index_sha256=SHA,
        output_path=str((tmp_path / "output.bin").resolve()),
        route_ids=("route",),
    )
    observed, revalidated = pool_module._validate_worker_input_bytes(
        task,
        evidence_mode="regular_file",
    )
    assert observed == task.input_hashes
    assert revalidated is True

    path.write_bytes(b"drifted")
    with pytest.raises(ProtocolError, match="input bytes drifted"):
        pool_module._validate_worker_input_bytes(
            task,
            evidence_mode="regular_file",
        )

    target = tmp_path / "worker-input-target.bin"
    target.write_bytes(payload)
    link = tmp_path / "worker-input-link.bin"
    link.symlink_to(target)
    linked = replace(
        task,
        input_paths=(str(link.absolute()),),
        input_hashes=(hashlib.sha256(payload).hexdigest(),),
    )
    with pytest.raises(ProtocolError, match="not a regular file"):
        pool_module._validate_worker_input_bytes(
            linked,
            evidence_mode="regular_file",
        )


def test_cpu_outer_tasks_must_consume_an_admitted_gpu_output_hash(
    sealed_gpu_batch,
) -> None:
    _, gpu_batch = sealed_gpu_batch
    surface = _build_strict_test_candidate_probability_surface_receipt(
        gpu_batch
    )
    descriptor = seal_callback_descriptor(
        _synthetic_descriptor_outer_callback,
        callback_role="cpu_outer",
    )
    tasks = _outer_tasks(
        probability_sha256=surface.output_file_hashes[0],
        candidate_probability_surface_sha256=(
            surface.candidate_probability_surface_sha256
        ),
        row_index_sha256=surface.row_index_sha256,
    )
    poisoned = (replace(tasks[0], probability_sha256="0" * 64), *tasks[1:])
    with pytest.raises(ProtocolError, match="outer-task inventory"):
        run_cpu_outer_pool(
            poisoned,
            callback=descriptor,
            candidate_surface=surface,
        )

    with pytest.raises(ProtocolError, match="guarded factory"):
        CandidateProbabilitySurfaceReceipt(
            gpu_prediction_batch_hash=gpu_batch.batch_hash,
            gpu_result_surface_sha256=gpu_batch.result_surface_sha256,
            row_index_sha256=gpu_batch.row_index_sha256,
            row_alignment_receipt_hash=SHA,
            output_file_hashes=gpu_batch.result_file_hashes,
            worker_result_hashes=tuple(
                row.worker_result_hash for row in gpu_batch.receipts
            ),
            canonical_alignment_bound=False,
        )
