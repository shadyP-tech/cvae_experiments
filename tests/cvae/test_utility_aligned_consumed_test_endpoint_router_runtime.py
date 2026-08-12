from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router import (
    feature_execution,
    runner,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.development_runtime import (
    materialize_development_predictions,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.prediction_contracts import (
    DEVELOPMENT_ROLE,
    PlannedPhysicalAction,
    PredictionCell,
    PredictionStore,
    PredictionTask,
    TARGET_ROLE,
    prediction_store_hash,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.checkpoint_store import (
    PredictionCheckpoint,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.runner_dependencies import (
    ConsumedTestEndpointRouterRunnerDependencies,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.target_runtime import (
    materialize_target_predictions,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.artifact_io import (
    sha256_file,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router import (
    run_lock,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router import (
    initialization_recovery,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.inputs import (
    _validate_cache_identity,
)


def _spawn_echo(value: object) -> object:
    return value


def _spawn_write_checkpoint(task: PredictionTask) -> PredictionCheckpoint:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.checkpoint_store import (
        write_task_checkpoint,
    )

    probabilities = np.linspace(
        0.1,
        0.9,
        num=len(task.actions) * 2,
        dtype=np.float32,
    ).reshape(len(task.actions), 2)
    records = tuple(
        {
            "action_id": action.action_id,
            "action_hash": action.action_hash,
            "converged": True,
        }
        for action in task.actions
    )
    return write_task_checkpoint(
        task,
        probabilities=probabilities,
        action_records=records,
    )


def _prediction_task(tmp_path: Path) -> PredictionTask:
    sha = "a" * 64
    sources = ("2", "3", "5", "6", "7", "8", "9")
    actions = []
    for action_id in ("B", *(f"Hxe::{source}" for source in sources)):
        selected = action_id.removeprefix("Hxe::") if action_id != "B" else None
        counts = {
            source: 270 if source == selected else 144 for source in sources
        }
        action_unhashed = {
            "schema_version": "midogpp_endpoint_router_physical_action_v1",
            "phase": DEVELOPMENT_ROLE,
            "outer_target": "0",
            "query_center": "1",
            "action_id": action_id,
            "sources": list(sources),
            "rows_per_class_by_source": counts,
            "labels_used": False,
            "source_prefix_only": True,
        }
        actions.append(
            PlannedPhysicalAction(
                phase=DEVELOPMENT_ROLE,
                outer_target="0",
                query_center="1",
                action_id=action_id,
                sources=sources,
                rows_per_class_by_source=counts,
                action_hash=canonical_sha256(action_unhashed),
            )
        )
    values = {
        "phase": DEVELOPMENT_ROLE,
        "task_ordinal": 0,
        "outer_target": "0",
        "query_center": "1",
        "training_seed": 17,
        "generation_seed": 17,
        "actions": tuple(actions),
        "source_array_path": str(tmp_path / "source.npy"),
        "target_array_path": str(tmp_path / "target.npy"),
        "target_array_sha256": sha,
        "support_row_ordinals": (0,),
        "evaluation_row_ordinals": (1,),
        "support_row_ids": ("support",),
        "evaluation_row_ids": ("evaluation",),
        "support_case_ids": ("support-case",),
        "evaluation_case_ids": ("evaluation-case",),
        "support_row_identity_hash": sha,
        "evaluation_row_identity_hash": sha,
        "config_contract_hash": "b" * 16,
        "source_stream_lock_hash": "c" * 16,
        "partition_lock_hash": sha,
        "cache_binding_hash": sha,
        "classifier_payload": {
            "family": "sklearn_logistic_regression",
            "C": 0.01,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "random_state": 23,
            "l1_ratio": None,
            "threshold_policy": "predict",
            "scaler_fit": "synthetic_train_only",
        },
        "checkpoint_npz_path": str(tmp_path / "development_predictions" / "development_H0_q1_train17_gen17.npz"),
        "checkpoint_json_path": str(tmp_path / "development_predictions" / "development_H0_q1_train17_gen17.json"),
    }
    unhashed = {
        "schema_version": "midogpp_endpoint_router_prediction_task_v1",
        **{
            key: [item.to_payload() for item in value]
            if key == "actions"
            else list(value)
            if isinstance(value, tuple)
            else value
            for key, value in values.items()
        },
        "labels_available": False,
    }
    return PredictionTask(**values, task_hash=canonical_sha256(unhashed))


def test_embedding_slice_accepts_derived_role_but_rejects_physical_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.input_contracts as input_contracts
    from dataclasses import replace
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.partitions import (
        LabelFreeCaseRow,
    )

    monkeypatch.setattr(input_contracts, "EXPECTED_TEST_ROW_COUNT", 2)
    monkeypatch.setattr(input_contracts, "FEATURE_DIM", 3)
    monkeypatch.setattr(input_contracts, "CENTERS", ("0",))
    rows = (
        LabelFreeCaseRow(0, 10, "eval-a", "case-a", "0"),
        LabelFreeCaseRow(1, 11, "eval-b", "case-b", "0"),
    )
    values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    frame = input_contracts.LabelFreeTestFrame(
        embeddings=values,
        rows=rows,
        rows_by_center={"0": rows},
        cache_binding={},
    )

    support_row = replace(rows[1], partition_role="support")
    np.testing.assert_array_equal(
        frame.embeddings_for((support_row,)), values[[1]]
    )
    with pytest.raises(ProtocolError, match="embedding row identity drifted"):
        frame.embeddings_for((replace(support_row, case_id="wrong-case"),))


def test_spawn_boundary_round_trips_prediction_task_and_checkpoint(
    tmp_path: Path,
) -> None:
    task = _prediction_task(tmp_path)
    restored_task = pickle.loads(pickle.dumps(task))
    assert restored_task.task_hash == task.task_hash
    assert dict(restored_task.classifier_payload) == dict(task.classifier_payload)
    assert dict(restored_task.actions[0].rows_per_class_by_source) == dict(
        task.actions[0].rows_per_class_by_source
    )
    with pytest.raises(TypeError):
        restored_task.classifier_payload["C"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        restored_task.actions[0].rows_per_class_by_source["2"] = 1  # type: ignore[index]

    checkpoint = PredictionCheckpoint(
        task_hash=task.task_hash,
        task_key=task.key,
        probabilities=np.asarray([[0.25, 0.75]], dtype=np.float32),
        action_records=({"action_id": "B", "converged": True},),
        checkpoint_hash="d" * 64,
        npz_path=Path(task.checkpoint_npz_path),
        json_path=Path(task.checkpoint_json_path),
    )
    restored_checkpoint = pickle.loads(pickle.dumps(checkpoint))
    assert restored_checkpoint.checkpoint_hash == checkpoint.checkpoint_hash
    assert dict(restored_checkpoint.action_records[0]) == dict(
        checkpoint.action_records[0]
    )
    with pytest.raises(TypeError):
        restored_checkpoint.action_records[0]["converged"] = False  # type: ignore[index]
    assert ForkingPickler.loads(ForkingPickler.dumps(task)).task_hash == task.task_hash
    assert (
        ForkingPickler.loads(ForkingPickler.dumps(checkpoint)).checkpoint_hash
        == checkpoint.checkpoint_hash
    )

    context = mp.get_context("spawn")
    with context.Pool(processes=1) as pool:
        spawned_task, spawned_checkpoint = pool.map(
            _spawn_echo, (task, checkpoint)
        )
    assert spawned_task.task_hash == task.task_hash
    assert spawned_checkpoint.checkpoint_hash == checkpoint.checkpoint_hash


def test_spawned_checkpoint_write_returns_and_reloads_identical_records(
    tmp_path: Path,
) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.checkpoint_store import (
        load_task_checkpoint,
    )

    task = _prediction_task(tmp_path)
    context = mp.get_context("spawn")
    with context.Pool(processes=1) as pool:
        returned = pool.map(_spawn_write_checkpoint, (task,))[0]

    loaded = load_task_checkpoint(task)
    assert loaded is not None
    assert returned.checkpoint_hash == loaded.checkpoint_hash
    assert len(returned.action_records) == len(task.actions) == 8
    assert tuple(dict(row) for row in returned.action_records) == tuple(
        dict(row) for row in loaded.action_records
    )
    assert tuple(row["action_id"] for row in returned.action_records) == tuple(
        action.action_id for action in task.actions
    )


def test_cache_identity_uses_builder_representation_when_frozen_schema_omits_it() -> None:
    config = SimpleNamespace(
        expected_test_cache_semantic_id="uniform_b_v2_descriptive_test_cache_v1",
        expected_test_cache_representation_id="annotation_jpeg_fixed_center_b_v3",
        expected_manifest_sha256="a" * 64,
        expected_test_cache_content_hash="b" * 64,
    )
    frozen = {
        "cache_name": config.expected_test_cache_semantic_id,
        "cache_extractor_protocol": {
            "representation_id": config.expected_test_cache_representation_id,
            "feature_dim": 3840,
        },
        "scoring_manifest_sha256": config.expected_manifest_sha256,
        "expected_row_count": 9928,
    }
    alignment = {
        "status": "PASS",
        "split": "test",
        "manifest_sha256": config.expected_manifest_sha256,
        "row_count": 9928,
    }
    builder = {"representation_id": config.expected_test_cache_representation_id}
    content = {"content_hash": config.expected_test_cache_content_hash}

    _validate_cache_identity(frozen, alignment, builder, content, config=config)

    with pytest.raises(ProtocolError, match="test-cache identity drifted"):
        _validate_cache_identity(
            {**frozen, "representation_id": "wrong-representation"},
            alignment,
            builder,
            content,
            config=config,
        )
    with pytest.raises(ProtocolError, match="test-cache identity drifted"):
        _validate_cache_identity(
            frozen,
            alignment,
            {"representation_id": "wrong-representation"},
            content,
            config=config,
        )
    with pytest.raises(ProtocolError, match="test-cache identity drifted"):
        _validate_cache_identity(
            {
                **frozen,
                "cache_extractor_protocol": {
                    "representation_id": "wrong-representation",
                    "feature_dim": 3840,
                },
            },
            alignment,
            builder,
            content,
            config=config,
        )
    with pytest.raises(ProtocolError, match="test-cache identity drifted"):
        _validate_cache_identity(
            frozen,
            alignment,
            builder,
            {"content_hash": "wrong-content"},
            config=config,
        )


def test_initialization_recovery_requires_exact_failed_inventory(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "provenance").mkdir()
    (root / "reports").mkdir()
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    state = {
        **initialization_recovery.FAILED_CACHE_IDENTITY_STATE,
        "updated_at_utc": "2026-08-12T09:59:40+00:00",
    }
    (root / "reports/run_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    assert initialization_recovery.detect_initializing_cache_identity_recovery(root)

    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="recovery boundary drifted"):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)


def test_source_feature_recovery_requires_exact_prelabel_inventory(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    for relative in initialization_recovery.SOURCE_FEATURE_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **initialization_recovery.FAILED_EMBEDDING_IDENTITY_STATE,
                "updated_at_utc": "2026-08-12T10:16:36+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")
    assert initialization_recovery.detect_initializing_cache_identity_recovery(root)

    (root / "tables/unexpected.csv").write_text("unsafe\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="source-feature recovery boundary drifted"):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)


def test_feature_task_recovery_requires_exact_staged_support_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    for relative in initialization_recovery.FEATURE_TASK_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **initialization_recovery.FAILED_FEATURE_TASK_STATE,
                "updated_at_utc": "2026-08-12T10:33:56+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")
    assert initialization_recovery.detect_initializing_cache_identity_recovery(root)

    support = root / "checkpoints/feature_runtime/support_q0.npy"
    support.unlink()
    with pytest.raises(ProtocolError, match="feature-task recovery boundary drifted"):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)


def test_prediction_pickle_recovery_requires_complete_feature_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    for relative in initialization_recovery.COMPLETE_FEATURE_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **initialization_recovery.FAILED_PREDICTION_PICKLE_STATE,
                "updated_at_utc": "2026-08-12T12:02:01+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")
    assert initialization_recovery.detect_initializing_cache_identity_recovery(root)

    component = root / "checkpoints/feature_runtime/feature_e0_train17.json"
    component.unlink()
    with pytest.raises(ProtocolError, match="prediction-pickle recovery boundary drifted"):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)


def test_checkpoint_action_record_recovery_requires_all_development_pairs(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    expected = initialization_recovery.COMPLETE_DEVELOPMENT_CHECKPOINT_RECOVERY_FILES
    assert len(expected) == 1_371
    for relative in expected:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            state = {
                **initialization_recovery.FAILED_CHECKPOINT_ACTION_RECORDS_STATE,
                "updated_at_utc": "2026-08-12T12:57:00+00:00",
            }
            path.write_text(json.dumps(state), encoding="utf-8")
        else:
            path.write_bytes(b"sealed")
    assert initialization_recovery.detect_initializing_cache_identity_recovery(root)

    member = root / (
        "checkpoints/development_predictions/"
        "development_H0_q1_train17_gen17.json"
    )
    member.unlink()
    with pytest.raises(
        ProtocolError, match="checkpoint-action-records recovery boundary drifted"
    ):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)

    member.write_bytes(b"sealed")
    unexpected = root / (
        "checkpoints/target_predictions/target_H0_q0_train17_gen17.json"
    )
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b"unsafe")
    with pytest.raises(
        ProtocolError, match="checkpoint-action-records recovery boundary drifted"
    ):
        initialization_recovery.detect_initializing_cache_identity_recovery(root)


def test_run_lock_recovers_only_a_dead_same_host_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lock_path = root / ".run.lock"
    stale = {
        "schema_version": run_lock.LOCK_SCHEMA,
        "pid": 424242,
        "hostname": socket.gethostname(),
        "token": "stale-token",
        "acquired_at_utc": "2026-01-01T00:00:00+00:00",
    }
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(run_lock, "_pid_is_alive", lambda pid: False)

    with run_lock.exclusive_run_lock(root):
        active = json.loads(lock_path.read_text(encoding="utf-8"))
        assert active["schema_version"] == run_lock.LOCK_SCHEMA
        assert active["token"] != "stale-token"
    assert not lock_path.exists()


def test_run_lock_refuses_a_live_or_foreign_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lock_path = root / ".run.lock"
    live = {
        "schema_version": run_lock.LOCK_SCHEMA,
        "pid": 424242,
        "hostname": socket.gethostname(),
        "token": "live-token",
        "acquired_at_utc": "2026-01-01T00:00:00+00:00",
    }
    lock_path.write_text(json.dumps(live), encoding="utf-8")
    monkeypatch.setattr(run_lock, "_pid_is_alive", lambda pid: True)
    with pytest.raises(ProtocolError, match="Another endpoint-router process"):
        with run_lock.exclusive_run_lock(root):
            pytest.fail("live lock was stolen")
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "live-token"


def test_production_default_runner_calls_seal_before_terminal_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "output").resolve()
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    inputs = (tmp_path / "inputs").resolve()
    config = SimpleNamespace(
        artifact_root=root,
        expert_bank_root=inputs / "bank",
        generation_lock_root=inputs / "generation_lock",
        test_cache_root=inputs / "test_cache",
        test_manifest_path=inputs / "manifest.csv",
        domain_mapping_path=inputs / "domain_mapping.json",
        test_consumption_ledger_path=inputs / "ledger.json",
        ledger_amendment_path=inputs / "amendment.json",
        expected_manifest_sha256="a" * 64,
        action_library={},
        runtime={},
        contract_hash="b" * 64,
    )
    events: list[str] = []
    frame = SimpleNamespace(rows=(object(),), cache_binding_hash="c" * 64)
    partitions = object()
    source = object()
    staged = SimpleNamespace(
        cache=object(), scratch_root=(tmp_path / "scratch").resolve(),
        report_payload=lambda: {"status": "STAGED"},
    )
    embeddings = object()
    development = object()
    target_store = object()
    seed_features = object()
    shifts = object()
    target_capability = object()

    class Prelabel:
        development_persistence = {}

        def seal_target(self, store: object, *, root: Path) -> object:
            assert store is target_store
            events.append("seal_target")
            return target_capability

        def model_plan_persistence(self, capability: object) -> dict[str, object]:
            assert capability is target_capability
            events.append("model_plan_persistence")
            return {}

    prelabel = Prelabel()
    terminal = SimpleNamespace(persistence={})

    def mark(name: str, value: object = None):
        def call(*args: object, **kwargs: object) -> object:
            events.append(name)
            return value
        return call

    monkeypatch.setattr(runner, "assert_closed_world", mark("closed_world"))
    monkeypatch.setattr(
        runner, "validate_active_diagnostic_workspace_binding", mark("workspace", {})
    )
    monkeypatch.setattr(runner, "validate_workspace_provenance", mark("provenance", {}))
    monkeypatch.setattr(
        runner, "load_validated_locks", mark("locks", SimpleNamespace(generation=object()))
    )
    monkeypatch.setattr(runner, "load_label_free_test_frame", mark("frame", frame))
    monkeypatch.setattr(
        runner, "admit_manifest_without_labels",
        mark("manifest_admission", {"manifest_admission_hash": "d" * 64}),
    )
    monkeypatch.setattr(runner, "validate_pre_gpu_firewall", mark("firewall", {}))
    monkeypatch.setattr(runner, "build_consumed_test_partitions", mark("partition", partitions))
    monkeypatch.setattr(runner, "load_metadata_compatibility", mark("metadata", object()))
    monkeypatch.setattr(runner, "persist_initial_surfaces", mark("persist_initial"))
    monkeypatch.setattr(
        runner, "run_endpoint_router_workstation_preflight", mark("preflight", {})
    )
    monkeypatch.setattr(runner, "materialize_source_cache", mark("source", source))
    monkeypatch.setattr(runner, "stage_source_cache_for_cpu", mark("stage_source", staged))
    def produce_seed(*args: object, **kwargs: object) -> object:
        assert kwargs["retain_checkpoints"] is True
        events.append("seed_features")
        return seed_features

    monkeypatch.setattr(runner, "materialize_label_free_seed_features", produce_seed)
    monkeypatch.setattr(runner, "enter_cuda_free_cpu_phase", mark("cpu_phase"))
    monkeypatch.setattr(runner, "stage_target_embeddings", mark("target_embeddings", embeddings))
    monkeypatch.setattr(runner, "build_development_prediction_plan", mark("development_plan", object()))
    monkeypatch.setattr(runner, "materialize_development_predictions", mark("development_store", development))
    monkeypatch.setattr(runner, "build_target_prediction_plan", mark("target_plan", object()))
    monkeypatch.setattr(runner, "materialize_target_predictions", mark("target_store", target_store))
    monkeypatch.setattr(runner, "cleanup_staged_target_embeddings", mark("cleanup_embeddings"))
    monkeypatch.setattr(runner, "materialize_label_free_support_shifts", mark("shifts", shifts))
    monkeypatch.setattr(runner, "run_prelabel_science", mark("prelabel", prelabel))
    monkeypatch.setattr(runner, "persist_development_surfaces", mark("persist_development"))
    monkeypatch.setattr(runner, "persist_model_and_plan_surfaces", mark("persist_models"))
    monkeypatch.setattr(runner, "run_terminal_science", mark("terminal", terminal))
    monkeypatch.setattr(runner, "persist_terminal_surfaces", mark("persist_terminal"))
    monkeypatch.setattr(
        runner, "cleanup_feature_runtime_checkpoints", mark("cleanup_features")
    )
    monkeypatch.setattr(runner, "cleanup_staged_source_cache", mark("cleanup_source"))
    monkeypatch.setattr(runner, "write_content_index", mark("content_index"))
    monkeypatch.setattr(runner, "_validate_bundle", mark("validate", {"status": "PASS"}))
    monkeypatch.setattr(runner, "persist_validation_report", mark("persist_validation"))

    assert runner.run_utility_aligned_consumed_test_endpoint_router(config) == root
    assert events.index("preflight") < events.index("source")
    assert events.index("seed_features") < events.index("cpu_phase")
    assert events.index("target_store") < events.index("cleanup_embeddings")
    assert events.index("cleanup_embeddings") < events.index("shifts")
    assert events.index("prelabel") < events.index("seal_target")
    assert events.index("seal_target") < events.index("model_plan_persistence")
    assert events.index("model_plan_persistence") < events.index("terminal")
    assert events.index("persist_terminal") < events.index("cleanup_features")
    assert events.index("cleanup_features") < events.index("content_index")
    assert "run_preflight" not in ConsumedTestEndpointRouterRunnerDependencies.__dataclass_fields__


def test_completed_prediction_surfaces_skip_all_worker_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_plan = SimpleNamespace(phase=DEVELOPMENT_ROLE, tasks=(object(),))
    target_plan = SimpleNamespace(phase=TARGET_ROLE, tasks=(object(),))
    development = object()
    target = object()

    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.development_runtime as development_runtime
    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.target_runtime as target_runtime

    monkeypatch.setattr(
        development_runtime, "load_development_prediction_capability",
        lambda plan, *, root: development,
    )
    monkeypatch.setattr(
        target_runtime, "load_target_prediction_store", lambda plan, *, root: target
    )
    monkeypatch.setattr(
        development_runtime, "execute_prediction_plan",
        lambda *args, **kwargs: pytest.fail("development workers ran on final resume"),
    )
    monkeypatch.setattr(
        target_runtime, "execute_prediction_plan",
        lambda *args, **kwargs: pytest.fail("target workers ran on final resume"),
    )

    assert materialize_development_predictions(development_plan, root=tmp_path) is development
    assert materialize_target_predictions(target_plan, root=tmp_path) is target


def test_prediction_index_and_offsets_are_reconstructively_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.prediction_contracts as contracts
    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.prediction_store as store_io

    key = ("0", "0", "B", 17, 17)
    monkeypatch.setattr(contracts, "canonical_cell_keys", lambda phase: (key,))
    monkeypatch.setattr(contracts, "canonical_scopes", lambda phase: ("0",))
    monkeypatch.setattr(store_io, "canonical_cell_keys", lambda phase: (key,))
    monkeypatch.setattr(store_io, "canonical_scopes", lambda phase: ("0",))
    monkeypatch.setattr(store_io, "TARGET_CELL_COUNT", 1)
    sha = "a" * 64
    cell = PredictionCell(
        phase=TARGET_ROLE, outer_target="0", query_center="0", action_id="B",
        action_hash=sha, training_seed=17, generation_seed=17,
        support_row_identity_hash=sha, evaluation_row_identity_hash=sha,
        support_probabilities=np.asarray([0.25], dtype=np.float32),
        evaluation_probabilities=np.asarray([0.75], dtype=np.float32),
        composition_hash=sha, scaler_state_hash=sha, fit_provenance_hash=sha,
    )
    mappings = {"0": ("row",)}
    store_hash = prediction_store_hash(
        TARGET_ROLE, (cell,), support_row_ids_by_scope=mappings,
        evaluation_row_ids_by_scope=mappings, support_case_ids_by_scope=mappings,
        evaluation_case_ids_by_scope=mappings, source_stream_lock_hash=sha,
        partition_lock_hash=sha, cache_binding_hash=sha, action_library_hash=sha,
    )
    store = PredictionStore(
        phase=TARGET_ROLE, cells=(cell,), support_row_ids_by_scope=mappings,
        evaluation_row_ids_by_scope=mappings, support_case_ids_by_scope=mappings,
        evaluation_case_ids_by_scope=mappings, source_stream_lock_hash=sha,
        partition_lock_hash=sha, cache_binding_hash=sha, action_library_hash=sha,
        store_hash=store_hash,
    )
    plan = SimpleNamespace(
        phase=TARGET_ROLE, tasks=tuple(object() for _ in range(81)),
        plan_hash="b" * 64, action_library_hash=sha,
    )
    (tmp_path / "arrays").mkdir()
    (tmp_path / "manifests").mkdir()
    store_io._persist_store(store, root=tmp_path, plan=plan)
    assert store_io.load_prediction_store(tmp_path, phase=TARGET_ROLE).store_hash == store_hash

    index_path = tmp_path / "manifests/target_prediction_index.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    index["unexpected_index_field"] = "tampered"
    index["prediction_index_hash"] = canonical_sha256(
        {key: value for key, value in index.items() if key != "prediction_index_hash"}
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ProtocolError):
        store_io.load_prediction_store(tmp_path, phase=TARGET_ROLE)

    index_path.write_text(original, encoding="utf-8")
    array_path = tmp_path / "arrays/target_action_probabilities.npz"
    with np.load(array_path, allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    values["support_offsets"] = np.asarray([0, 0], dtype=np.int64)
    np.savez(array_path, **values)
    index = json.loads(original)
    index["array_sha256"] = sha256_file(array_path)
    index["prediction_index_hash"] = canonical_sha256(
        {key: value for key, value in index.items() if key != "prediction_index_hash"}
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ProtocolError):
        store_io.load_prediction_store(tmp_path, phase=TARGET_ROLE)


def test_legacy_y_encoding_is_rejected_inside_shard_string_values(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.inputs import (
        _load_label_free_shard,
    )

    path = tmp_path / "center_0.pt"
    torch.save(
        {
            "embeddings": torch.zeros((1, 1), dtype=torch.float32),
            "metadata": [{
                "evaluation_row_id": "slide_y1_patch",
                "contract_row_index": 0,
                "case_id": "case-0",
                "center": "0",
                "split": "test",
            }],
            "feature_extractor": {"name": "virchow2"},
        },
        path,
    )
    with pytest.raises(ProtocolError, match="metadata firewall"):
        _load_label_free_shard(path, center="0")


def test_preflight_reprobes_hardware_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.runtime_preflight as preflight

    runtime = {
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "array_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "phase_order": "two_A5000_generation_then_four_by_three_CPU",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "source_stream_count": 81,
        "development_prediction_cell_count": 5_184,
        "target_physical_action_identity_count": 90,
        "target_prediction_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 5_994,
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        "scratch_preference": ["/data/local", "artifact_parent"],
    }
    calls: list[Path] = []

    def probe(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append(path)
        return {"schema_version": "probe_v1", "status": "PASS"}

    monkeypatch.setattr(preflight, "_shared_preflight", probe)
    root = tmp_path / "output"
    first = preflight.run_endpoint_router_workstation_preflight(root, runtime=runtime)
    second = preflight.run_endpoint_router_workstation_preflight(root, runtime=runtime)
    assert first == second
    assert len(calls) == 2
    assert first["target_prediction_cell_count"] == 810
    assert first["preflight_reprobed_before_each_compute_session"] is True


def test_partial_source_final_surface_fails_closed(tmp_path: Path) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.source_cache import (
        materialize_source_cache,
    )

    path = tmp_path / "arrays/frozen_source_streams.npy"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"orphan")
    with pytest.raises(ProtocolError, match="final surface is incomplete"):
        materialize_source_cache(object(), object(), root=tmp_path)


def test_single_task_checkpoint_orphan_is_safely_recomputed(tmp_path: Path) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.checkpoint_store import (
        load_task_checkpoint,
    )

    checkpoint_root = (tmp_path / "checkpoints/development_predictions").resolve()
    checkpoint_root.mkdir(parents=True)
    npz_path = checkpoint_root / "development_H0_q1_train17_gen17.npz"
    json_path = checkpoint_root / "development_H0_q1_train17_gen17.json"
    npz_path.write_bytes(b"atomic-orphan")
    task = SimpleNamespace(
        phase="development", outer_target="0", query_center="1",
        training_seed=17, generation_seed=17,
        checkpoint_npz_path=str(npz_path), checkpoint_json_path=str(json_path),
    )
    assert load_task_checkpoint(task) is None
    assert not npz_path.exists()


def test_feature_checkpoints_resume_after_later_phase_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.contracts import (
        CENTERS,
        TRAINING_SEEDS,
        candidate_sources,
    )

    complete = False
    calls = 0
    records_by_task: dict[tuple[str, int], tuple[object, ...]] = {}
    tasks = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            records = tuple(
                SimpleNamespace(key=(query, source, training_seed))
                for query in candidate_sources(source)
            )
            records_by_task[(source, training_seed)] = records
            tasks.append(SimpleNamespace(
                source_center=source, training_seed=training_seed,
                task_hash=f"task-{source}-{training_seed}",
            ))

    def load(task: object, *, required: bool = False):
        if complete:
            return records_by_task[(task.source_center, task.training_seed)]
        if required:
            raise AssertionError("executor did not publish test checkpoint")
        return None

    def execute(pending: object) -> None:
        nonlocal complete, calls
        calls += 1
        complete = True

    monkeypatch.setattr(feature_execution, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        feature_execution, "_stage_support_arrays",
        lambda *args, **kwargs: {
            center: SimpleNamespace(slice_hash=f"slice-{center}") for center in CENTERS
        },
    )
    monkeypatch.setattr(feature_execution, "_build_tasks", lambda *args, **kwargs: tuple(tasks))
    monkeypatch.setattr(feature_execution, "load_feature_checkpoint", load)
    product = object()
    monkeypatch.setattr(
        feature_execution, "assemble_seed_feature_production",
        lambda *args, **kwargs: product,
    )
    config = SimpleNamespace(contract_hash="a" * 64, expected_bank_lock_hash="b" * 64)
    source = SimpleNamespace(lock_hash="c" * 64)
    frame = SimpleNamespace(cache_binding_hash="d" * 64)
    partitions = SimpleNamespace(
        lock_hash="e" * 64,
        support_rows_by_center={center: () for center in CENTERS},
    )
    metadata = SimpleNamespace(grid_hash="f" * 64, by_target={})

    first = feature_execution.materialize_label_free_seed_features(
        config, source, frame, partitions, metadata, root=tmp_path.resolve(),
        task_executor=execute, retain_checkpoints=True,
    )
    second = feature_execution.materialize_label_free_seed_features(
        config, source, frame, partitions, metadata, root=tmp_path.resolve(),
        task_executor=lambda pending: pytest.fail("feature workers reran on resume"),
        retain_checkpoints=True,
    )
    assert first is product and second is product
    assert calls == 1


def test_feature_checkpoint_cleanup_accepts_only_exact_owned_names(
    tmp_path: Path,
) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.contracts import (
        CENTERS,
        TRAINING_SEEDS,
    )

    checkpoint = tmp_path.resolve() / "checkpoints/feature_runtime"
    checkpoint.mkdir(parents=True)
    names = {
        "feature_input_seal.json",
        *(f"support_q{center}.npy" for center in CENTERS),
        *(
            f"feature_e{source}_train{training_seed}.{suffix}"
            for source in CENTERS
            for training_seed in TRAINING_SEEDS
            for suffix in ("json", "npz")
        ),
    }
    for name in names:
        (checkpoint / name).write_bytes(b"owned")
    feature_execution.cleanup_feature_runtime_checkpoints(tmp_path.resolve())
    assert not checkpoint.exists()

    checkpoint.mkdir(parents=True)
    (checkpoint / "unexpected.txt").write_text("unsafe", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unowned inventory"):
        feature_execution.cleanup_feature_runtime_checkpoints(tmp_path.resolve())


def test_closed_world_phase_resume_finalizes_without_restarting_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "output").resolve()
    (root / "provenance").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    (root / "reports/run_state.json").write_text(
        json.dumps({"status": "FAILED", "phase": "CLOSED_WORLD_VALIDATION"}),
        encoding="utf-8",
    )
    inputs = tmp_path.resolve() / "inputs"
    config = SimpleNamespace(
        artifact_root=root, expert_bank_root=inputs / "bank",
        generation_lock_root=inputs / "generation", test_cache_root=inputs / "cache",
        test_manifest_path=inputs / "manifest.csv",
        domain_mapping_path=inputs / "domain_mapping.json",
        test_consumption_ledger_path=inputs / "ledger.json",
        ledger_amendment_path=inputs / "amendment.json",
    )
    events: list[str] = []
    monkeypatch.setattr(runner, "assert_closed_world", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner, "validate_active_diagnostic_workspace_binding",
        lambda config: events.append("workspace"),
    )
    monkeypatch.setattr(
        runner, "validate_workspace_provenance",
        lambda root, config: events.append("provenance"),
    )
    monkeypatch.setattr(
        runner, "_finalize_bundle",
        lambda path, **kwargs: events.append("finalize") or path,
    )
    monkeypatch.setattr(
        runner, "run_endpoint_router_workstation_preflight",
        lambda *args, **kwargs: pytest.fail("preflight reran during finalization resume"),
    )
    assert runner.run_utility_aligned_consumed_test_endpoint_router(config) == root
    assert events == ["workspace", "provenance", "finalize"]
