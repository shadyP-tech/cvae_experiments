from __future__ import annotations

import pickle
from pathlib import Path
from queue import Empty
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.exact_tail_utility_surface import (
    prediction_execution,
    source_generation,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface import (
    prediction_orchestration,
    production_adapter,
    runner,
    source_gpu_worker,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.config import CLASSIFIER
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.prediction_checkpoint_store import (
    checkpoint_path,
    load_checkpoint,
    write_checkpoint,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.prediction_contracts import (
    CoarsePredictionRecord,
    PredictionWorkerInput,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.production_inputs import (
    DevelopmentReservation,
    PreparedDevelopmentInputs,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.source_contracts import (
    ExpertTask,
    SourceFeatureInputs,
    SourceGenerationConfig,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.source_orchestration import (
    coerce_source_feature_inputs,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.contracts import CENTERS
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.runtime import (
    coarse_prediction_tasks,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.scoring import (
    array_sha256,
)


def test_execution_facades_expose_only_stable_integration_apis() -> None:
    assert source_generation.__all__ == (
        "FeatureComponentRecord",
        "GeneratedDevelopmentCache",
        "SourceBlockRecord",
        "load_component_arrays",
        "load_validated_generation_lock",
        "materialize_generated_development_cache",
    )
    assert prediction_execution.__all__ == (
        "GLOBAL_SEAL_MEMBER",
        "PREDICTION_ARRAY_MEMBER",
        "PREDICTION_INDEX_MEMBER",
        "CoarsePredictionRecord",
        "PredictionExecutionResult",
        "materialize_exact_tail_predictions",
    )
    assert source_generation.SourceBlockRecord.__module__.endswith(
        ".source_contracts"
    )
    assert source_generation.load_component_arrays.__module__.endswith(
        ".source_checkpoint_store"
    )
    assert source_generation.load_validated_generation_lock.__module__.endswith(
        ".source_planning"
    )
    assert source_generation.materialize_generated_development_cache.__module__.endswith(
        ".source_orchestration"
    )
    assert prediction_execution.CoarsePredictionRecord.__module__.endswith(
        ".prediction_contracts"
    )
    assert prediction_execution.materialize_exact_tail_predictions.__module__.endswith(
        ".prediction_orchestration"
    )
    for facade, private_names in (
        (
            source_generation,
            ("_ExpertTask", "_spawn_expert_tasks", "_load_source_record"),
        ),
        (
            prediction_execution,
            ("_WorkerInput", "_prediction_worker", "_load_checkpoint"),
        ),
    ):
        assert all(not hasattr(facade, name) for name in private_names)


def test_workstation_disk_probe_uses_artifact_filesystem_parent(
    monkeypatch, tmp_path: Path
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(production_adapter, "_ram_gib", lambda: 125.0)
    monkeypatch.setattr(
        production_adapter,
        "_nvidia_snapshot",
        lambda: (("RTX A5000", "RTX A5000"), (24564, 24564), (24000, 24000)),
    )
    monkeypatch.setattr(
        production_adapter.shutil,
        "disk_usage",
        lambda path: observed.append(Path(path))
        or SimpleNamespace(free=400 * 1024**3),
    )
    config = SimpleNamespace(artifact_root=tmp_path / "not-created" / "artifact")

    snapshot = production_adapter.ProductionExactTailAdapter().collect_workstation_snapshot(
        config
    )

    assert observed == [tmp_path.resolve()]
    assert snapshot.artifact_disk_free_gib == pytest.approx(400.0)


def test_spawn_contracts_rebuild_immutable_mappings_after_pickle() -> None:
    task = SimpleNamespace(key=("A", "B", 17, 19), action_ids=("a",))
    source_task = ExpertTask(
        source_center="A",
        training_seed=17,
        generation_keys=(),
        existing_source_path_by_generation_seed={19: "/tmp/source.npy"},
        query_centers=("B",),
        support_array_path_by_center={"B": "/tmp/support.npy"},
        support_case_ids_by_center={"B": ("case",)},
        support_partition_hash_by_center={"B": "a" * 64},
        device="cuda:0",
    )
    worker_input = PredictionWorkerInput(
        task=task,
        cache_root="/tmp/cache",
        source_records=(),
        evaluation_array_path="/tmp/evaluation.npy",
        evaluation_row_identity_hash="b" * 64,
        partition_hash="c" * 64,
        source_cache_hash="d" * 64,
        classifier_payload={"family": "logistic_regression"},
        checkpoint_root="/tmp/checkpoints",
    )
    record = CoarsePredictionRecord(
        task=task,
        checkpoint_relative_path="/tmp/checkpoint.npz",
        checkpoint_file_sha256="e" * 64,
        evaluation_row_count=2,
        action_composition_sha256={"a": "f" * 64},
        action_scaler_state_hash={"a": "0" * 64},
        checkpoint_hash="1" * 64,
    )

    observed_source_task = pickle.loads(pickle.dumps(source_task))
    observed_worker_input = pickle.loads(pickle.dumps(worker_input))
    observed_record = pickle.loads(pickle.dumps(record))
    assert isinstance(
        observed_source_task.existing_source_path_by_generation_seed,
        MappingProxyType,
    )
    assert isinstance(observed_worker_input.classifier_payload, MappingProxyType)
    assert isinstance(observed_record.action_composition_sha256, MappingProxyType)


def test_neutral_source_feature_seam_requires_no_development_partition(
    tmp_path: Path,
) -> None:
    config = SourceGenerationConfig(
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        classifier=CLASSIFIER,
    )
    inputs = SourceFeatureInputs(
        support_array_path_by_center={
            center: tmp_path / f"support_{center}.npy" for center in CENTERS
        },
        support_case_ids_by_center={
            center: (f"case::{center}",) for center in CENTERS
        },
        support_partition_hash_by_center={
            center: f"{ordinal:x}" * 64
            for ordinal, center in enumerate(CENTERS, start=1)
        },
    )

    assert config.classifier is CLASSIFIER
    assert coerce_source_feature_inputs(inputs) is inputs
    assert not hasattr(inputs, "reservation")
    assert isinstance(inputs.support_partition_hash_by_center, MappingProxyType)

    exact_inputs = SourceFeatureInputs(
        support_array_path_by_center=inputs.support_array_path_by_center,
        support_case_ids_by_center=inputs.support_case_ids_by_center,
        support_partition_hash_by_center={
            center: f"{ordinal:x}" * 16
            for ordinal, center in enumerate(CENTERS, start=1)
        },
    )
    assert all(
        len(value) == 16
        for value in exact_inputs.support_partition_hash_by_center.values()
    )

    legacy = PreparedDevelopmentInputs(
        reservation=DevelopmentReservation(
            partitions={
                center: SimpleNamespace(
                    reservation_hash=exact_inputs.support_partition_hash_by_center[
                        center
                    ]
                )
                for center in CENTERS
            },
            metadata_similarity_by_query_source={},
            reservation_hash="legacy",
            raw_payload={},
        ),
        support_array_path_by_center=exact_inputs.support_array_path_by_center,
        support_case_ids_by_center=exact_inputs.support_case_ids_by_center,
        evaluation_array_path_by_center={},
    )
    converted = coerce_source_feature_inputs(legacy)
    assert dict(converted.support_partition_hash_by_center) == dict(
        exact_inputs.support_partition_hash_by_center
    )


def test_gpu_scheduler_detects_child_exit_and_cleans_up(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeQueue:
        def __init__(self, *, result: bool = False) -> None:
            self.result = result
            self.values: list[object] = []

        def put(self, value: object) -> None:
            self.values.append(value)

        def get(self, *, timeout: float) -> object:
            assert timeout == source_gpu_worker.GPU_RESULT_POLL_SECONDS
            if self.result:
                raise Empty
            raise AssertionError("task queues are write-only in the parent")

    class FakeProcess:
        def __init__(self, planned_exitcode: int | None) -> None:
            self.planned_exitcode = planned_exitcode
            self.exitcode: int | None = None
            self.alive = False
            self.terminated = False
            self.join_timeouts: list[float] = []

        def start(self) -> None:
            self.exitcode = self.planned_exitcode
            self.alive = self.planned_exitcode is None

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False
            self.exitcode = -15

        def kill(self) -> None:
            self.alive = False
            self.exitcode = -9

        def join(self, *, timeout: float) -> None:
            self.join_timeouts.append(timeout)

    class FakeContext:
        def __init__(self) -> None:
            self.queues = [FakeQueue(), FakeQueue(), FakeQueue(result=True)]
            self.processes: list[FakeProcess] = []

        def Queue(self) -> FakeQueue:
            return self.queues.pop(0)

        def Process(self, **kwargs) -> FakeProcess:
            process = FakeProcess(1 if not self.processes else None)
            self.processes.append(process)
            return process

    context = FakeContext()
    monkeypatch.setattr(source_gpu_worker.mp, "get_context", lambda method: context)
    task = ExpertTask(
        source_center="0",
        training_seed=17,
        generation_keys=(),
        existing_source_path_by_generation_seed={},
        query_centers=(),
        support_array_path_by_center={},
        support_case_ids_by_center={},
        support_partition_hash_by_center={},
        device="cuda:0",
    )

    with pytest.raises(ProtocolError, match="exited before returning"):
        source_gpu_worker.spawn_expert_tasks(
            (task,), tmp_path / "bank", tmp_path / "output"
        )

    assert context.processes[1].terminated is True
    assert all(process.join_timeouts for process in context.processes)


def test_external_checkpoint_root_does_not_change_canonical_output_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    resume_root = tmp_path / "resume/checkpoints"
    partitions = {"A": object()}
    inputs = SimpleNamespace(
        reservation=SimpleNamespace(partitions=partitions),
    )
    captured: dict[str, object] = {}
    records = (object(),)
    consolidated = object()
    expected = object()

    def fake_build(config, prepared, generated, *, checkpoint_root):
        captured["checkpoint_root"] = checkpoint_root
        return (object(),)

    def fake_consolidate(config, observed_partitions, observed_records, *, root):
        captured["consolidation_root"] = root
        assert observed_partitions is partitions
        assert observed_records is records
        return consolidated

    def fake_seal(
        config,
        attestation,
        observed_partitions,
        observed_records,
        observed_consolidated,
        *,
        root,
    ):
        captured["seal_root"] = root
        assert observed_partitions is partitions
        assert observed_records is records
        assert observed_consolidated is consolidated
        return expected

    monkeypatch.setattr(prediction_orchestration, "EXPECTED_COARSE_TASK_COUNT", 1)
    monkeypatch.setattr(
        prediction_orchestration, "build_prediction_worker_inputs", fake_build
    )
    monkeypatch.setattr(
        prediction_orchestration, "execute_or_resume", lambda worker_inputs: records
    )
    monkeypatch.setattr(
        prediction_orchestration, "consolidate_prediction_records", fake_consolidate
    )
    monkeypatch.setattr(
        prediction_orchestration, "seal_consolidated_predictions", fake_seal
    )

    observed = prediction_orchestration.materialize_exact_tail_predictions(
        object(),
        object(),
        inputs,
        object(),
        root=canonical_root,
        checkpoint_root=resume_root,
    )

    assert observed is expected
    assert captured == {
        "checkpoint_root": resume_root,
        "consolidation_root": canonical_root,
        "seal_root": canonical_root,
    }


def test_complete_prediction_checkpoint_is_hash_validated_on_resume(
    tmp_path: Path,
) -> None:
    task = coarse_prediction_tasks()[0]
    evaluation_path = tmp_path / "evaluation.npy"
    np.save(evaluation_path, np.zeros((3, 2), dtype=np.float32), allow_pickle=False)
    item = PredictionWorkerInput(
        task=task,
        cache_root=str(tmp_path / "cache"),
        source_records=(),
        evaluation_array_path=str(evaluation_path),
        evaluation_row_identity_hash="a" * 64,
        partition_hash="b" * 64,
        source_cache_hash="c" * 64,
        classifier_payload=CLASSIFIER.to_payload(),
        checkpoint_root=str(tmp_path / "external_resume"),
    )
    predictions = np.zeros((8, 3), dtype=np.uint8)
    probabilities = np.full((8, 3), 0.5, dtype=np.float32)
    prediction_hashes = {
        action_id: array_sha256(predictions[index])
        for index, action_id in enumerate(task.action_ids)
    }
    probability_hashes = {
        action_id: array_sha256(probabilities[index])
        for index, action_id in enumerate(task.action_ids)
    }
    compositions = {action_id: "d" * 64 for action_id in task.action_ids}
    scalers = {action_id: "e" * 64 for action_id in task.action_ids}

    written = write_checkpoint(
        item,
        classifier_config_hash=CLASSIFIER.config_hash,
        predictions=predictions,
        probabilities=probabilities,
        action_prediction_sha256=prediction_hashes,
        action_probability_sha256=probability_hashes,
        action_composition_sha256=compositions,
        action_scaler_state_hash=scalers,
        evaluation_row_count=3,
    )
    loaded = load_checkpoint(item)
    assert loaded == written
    assert checkpoint_path(item).is_relative_to(tmp_path / "external_resume")

    member = checkpoint_path(item)
    tampered = bytearray(member.read_bytes())
    tampered[-1] ^= 1
    member.write_bytes(tampered)
    with pytest.raises(ProtocolError, match="COMPLETE checkpoint binding drifted"):
        load_checkpoint(item)


class _NoWorkstationCalls:
    def collect_workstation_snapshot(self, _config):
        raise AssertionError("completed bundle must not probe workstation hardware")

    def materialize_label_free_predictions(self, config, attestation):
        raise AssertionError("completed bundle must not materialize predictions")

    def persist_scored_bundle(self, config, capability, rows):
        raise AssertionError("completed bundle must not be republished")


def test_complete_bundle_fast_path_validates_without_fresh_or_hardware_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(artifact_root=tmp_path)
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    expected = object()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        runner,
        "validate_fresh_inputs_ready",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("completed bundle must not reopen fresh inputs")
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_surface_bundle",
        lambda root, *, config: calls.append(("bundle", root)) or expected,
    )

    observed = runner.run_exact_tail_utility_surface(
        config,
        adapter=_NoWorkstationCalls(),
        workspace_validator=lambda value: calls.append(("workspace", value)),
    )

    assert observed is expected
    assert calls == [("workspace", config), ("bundle", tmp_path)]


def test_complete_run_state_with_missing_members_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(artifact_root=tmp_path)
    run_state = tmp_path / "reports/run_state.json"
    run_state.parent.mkdir(parents=True, exist_ok=True)
    run_state.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "validate_fresh_inputs_ready",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("incomplete COMPLETE run must fail before fresh inputs")
        ),
    )

    with pytest.raises(ProtocolError, match="COMPLETE artifact is incomplete"):
        runner.run_exact_tail_utility_surface(
            config,
            adapter=_NoWorkstationCalls(),
            workspace_validator=lambda _config: None,
        )
