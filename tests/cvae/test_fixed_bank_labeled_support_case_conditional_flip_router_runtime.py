from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router import execution_adapter
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router import (
    runner,
    runner_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.recovery import (
    RecoveryCapability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.runner_dependencies import (
    FlipRouterDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
)


def _write_completed_neutral_source_tree(root: Path) -> None:
    """Mirror the final tree left by materialize_frozen_source_streams."""

    for member in (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed fixture")
    # The neutral runtime removes checkpoints/frozen_source_streams after it
    # seals the trio, but deliberately leaves this experiment-owned parent.
    (root / "checkpoints").mkdir()


def _write_completed_neutral_prediction_scratch(root: Path) -> None:
    """Mirror scratch left after the neutral A1 runtime seals predictions."""

    # The neutral runtime removes its owned
    # checkpoints/fixed_bank_a1_action_predictions subtree after sealing the
    # canonical prediction trio, leaving this package-owned empty parent.
    (root / execution_adapter.LOCAL_PREDICTION_DIRECTORY / "checkpoints").mkdir(
        parents=True
    )


def _scratch_cleanup_config(root: Path, artifact_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime={"scratch_preference": [str(root), "artifact_parent"]},
        artifact_root=artifact_root,
        contract_hash="c" * 64,
    )


def test_exact_source_inventory_accepts_completed_neutral_runtime_tree(
    tmp_path: Path,
) -> None:
    _write_completed_neutral_source_tree(tmp_path)

    execution_adapter._remove_empty_owned_source_checkpoint_parent(tmp_path)

    assert not (tmp_path / "checkpoints").exists()
    execution_adapter._require_exact_source_inventory(tmp_path)


@pytest.mark.parametrize(
    "drift",
    (
        "foreign_member",
        "expected_file_is_directory",
        "symlinked_member",
        "foreign_empty_directory",
        "checkpoint_file",
        "checkpoint_nested_directory",
    ),
)
def test_exact_source_inventory_rejects_member_type_and_link_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root = tmp_path / "source"
    _write_completed_neutral_source_tree(root)
    if drift == "foreign_member":
        (root / "manifests/foreign.json").write_bytes(b"foreign")
    elif drift == "expected_file_is_directory":
        member = root / SOURCE_ARRAY_MEMBER
        member.unlink()
        member.mkdir()
    elif drift == "symlinked_member":
        member = root / SOURCE_LOCK_MEMBER
        member.unlink()
        external = tmp_path / "external-lock.json"
        external.write_bytes(b"outside")
        member.symlink_to(external)
    elif drift == "foreign_empty_directory":
        (root / "foreign").mkdir()
    elif drift == "checkpoint_nested_directory":
        (root / "checkpoints/frozen_source_streams").mkdir()
    else:
        (root / "checkpoints/foreign.bin").write_bytes(b"foreign")

    if drift in {"checkpoint_file", "checkpoint_nested_directory"}:
        with pytest.raises(ProtocolError, match="checkpoint parent is not empty"):
            execution_adapter._remove_empty_owned_source_checkpoint_parent(root)
        return

    execution_adapter._remove_empty_owned_source_checkpoint_parent(root)
    with pytest.raises(ProtocolError, match="local source inventory drifted") as exc:
        execution_adapter._require_exact_source_inventory(root)

    assert "directory_missing=" in str(exc.value)
    assert "directory_extras=" in str(exc.value)


@pytest.mark.parametrize("drift", ("symlink", "foreign_file"))
def test_source_checkpoint_parent_normalization_rejects_unsafe_types(
    tmp_path: Path,
    drift: str,
) -> None:
    root = tmp_path / "source"
    _write_completed_neutral_source_tree(root)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.rmdir()
    if drift == "symlink":
        external = tmp_path / "external-checkpoints"
        external.mkdir()
        checkpoint_root.symlink_to(external, target_is_directory=True)
    else:
        checkpoint_root.write_bytes(b"foreign")

    with pytest.raises(ProtocolError, match="checkpoint parent is unsafe"):
        execution_adapter._remove_empty_owned_source_checkpoint_parent(root)


def test_cleanup_accepts_completed_neutral_prediction_scratch_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    _write_completed_neutral_prediction_scratch(scratch)
    config = _scratch_cleanup_config(scratch, tmp_path / "artifact")
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))

    execution_adapter.cleanup_validated_local_stage(config)

    assert not scratch.exists()


@pytest.mark.parametrize(
    ("drift", "error"),
    (
        ("checkpoint_parent_symlink", "prediction checkpoint parent is unsafe"),
        ("checkpoint_parent_file", "prediction checkpoint parent is unsafe"),
        (
            "checkpoint_parent_nonempty",
            "completed prediction checkpoint parent is not empty",
        ),
        (
            "checkpoint_parent_nested_directory",
            "completed prediction checkpoint parent is not empty",
        ),
        ("foreign_prediction_member", "nonempty/unsafe prediction scratch root"),
    ),
)
def test_cleanup_rejects_unsafe_completed_prediction_scratch_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    error: str,
) -> None:
    scratch = tmp_path / "scratch"
    prediction = scratch / execution_adapter.LOCAL_PREDICTION_DIRECTORY
    prediction.mkdir(parents=True)
    checkpoints = prediction / "checkpoints"
    external = tmp_path / "external"
    external.mkdir()
    if drift == "checkpoint_parent_symlink":
        checkpoints.symlink_to(external, target_is_directory=True)
    elif drift == "checkpoint_parent_file":
        checkpoints.write_bytes(b"foreign")
    elif drift == "checkpoint_parent_nonempty":
        checkpoints.mkdir()
        (checkpoints / "foreign.bin").write_bytes(b"foreign")
    elif drift == "checkpoint_parent_nested_directory":
        (checkpoints / "foreign").mkdir(parents=True)
    else:
        (prediction / "foreign.bin").write_bytes(b"foreign")
    config = _scratch_cleanup_config(scratch, tmp_path / "artifact")
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))

    with pytest.raises(ProtocolError, match=error):
        execution_adapter.cleanup_validated_local_stage(config)

    assert prediction.exists()
    assert external.exists()


def test_completed_recovery_cleans_neutral_prediction_scratch_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    scratch = tmp_path / "scratch"
    _write_completed_neutral_prediction_scratch(scratch)
    config = _scratch_cleanup_config(scratch, root)
    events: list[str] = []
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(runner, "assert_launch_files", lambda *_args: None)
    monkeypatch.setattr(
        runner, "assert_workspace_resolved_paths", lambda *_args, **_kwargs: None
    )

    @contextmanager
    def lock(_root: Path):
        events.append("lock")
        yield

    monkeypatch.setattr(runner, "exclusive_run_lock", lock)
    monkeypatch.setattr(
        runner,
        "recover_if_possible",
        lambda *_args, **_kwargs: (events.append("recover"), root)[1],
    )

    assert runner._run(
        config, artifact_root=root, deps=FlipRouterDependencies()
    ) == root
    assert events == ["lock", "recover"]
    assert not scratch.exists()


def test_dependency_injected_runner_preserves_phase_and_validation_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    config_path = root / "config.resolved.yaml"
    config_path.write_text("experiment: fixture\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    input_ids = tuple(f"input-{ordinal}" for ordinal in range(6))
    config = SimpleNamespace(
        source_path=config_path.resolve(),
        artifact_root=root.resolve(),
        expert_bank_root=(tmp_path / "bank").resolve(),
        generation_lock_root=(tmp_path / "generation").resolve(),
        test_cache_root=(tmp_path / "cache").resolve(),
        test_manifest_path=(tmp_path / "manifest.csv").resolve(),
        test_consumption_ledger_path=(tmp_path / "ledger.json").resolve(),
        ledger_amendment_path=(tmp_path / "amendment.json").resolve(),
        input_artifact_ids=input_ids,
        contract_hash="a" * 64,
        runtime={},
    )
    events: list[str] = []
    flags = {"decisions_persisted": False, "full_validation": False}
    frame = SimpleNamespace(cache_binding_hash="b" * 64)
    partition = object()
    source = SimpleNamespace(root=root, lock_hash="c" * 16)
    prediction = SimpleNamespace(seal_hash="d" * 16)
    probabilities = object()
    prelabel = SimpleNamespace(feature_surface_hash="e" * 64)
    donor = SimpleNamespace(
        contribution_targets=(object(),),
        models=(object(),),
        seals={str(index): {} for index in range(9)},
        permutation_payload={},
    )
    bundle = SimpleNamespace(
        decisions=tuple(range(315)),
        fold_seal_hashes={
            (str(center), fold): "f" * 64
            for center in (0, 1, 2, 3, 5, 6, 7, 8, 9)
            for fold in range(5)
        },
        decision_bundle_hash="1" * 64,
    )
    decisions = SimpleNamespace(
        static_rows=tuple(range(45)),
        calibration_rows=tuple(range(45)),
        static_seal_payload={},
        calibration_seal_payload={},
        bundle=bundle,
    )

    class Manager:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("manager_after_prelabel")

        def seal_all_fold_plans(self):
            events.append("plans")
            return tuple(range(45))

        def open_terminal_evaluation_labels(self):
            assert flags["decisions_persisted"]
            assert len(bundle.fold_seal_hashes) == 45
            events.append("terminal_labels_after_45_decisions")
            return (object(),)

        def report_payload(self):
            return {"terminal_scoring_opened": True}

    monkeypatch.setattr(runner, "assert_input_fence", lambda _config: events.append("admission"))
    monkeypatch.setattr(
        runner,
        "validate_active_diagnostic_workspace_binding",
        lambda _config: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "validate_workspace_provenance",
        lambda *_args: {name: {"artifact_id": name} for name in input_ids},
    )
    monkeypatch.setattr(
        runner,
        "load_validated_locks",
        lambda _config: SimpleNamespace(generation=object()),
    )
    monkeypatch.setattr(runner, "load_label_free_test_frame", lambda _config: frame)
    monkeypatch.setattr(
        runner, "validate_pre_gpu_firewall", lambda *_args: {"status": "PASS"}
    )
    monkeypatch.setattr(runner, "build_case_partition", lambda *_args, **_kwargs: partition)
    monkeypatch.setattr(
        runner,
        "run_workstation_preflight",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner, "persist_initial_surfaces", lambda *_args, **_kwargs: events.append("initial")
    )
    monkeypatch.setattr(runner, "enter_cuda_free_cpu_phase", lambda: events.append("cpu_phase"))
    monkeypatch.setattr(runner, "seed_probability_rows", lambda _prediction: (object(),))
    monkeypatch.setattr(runner, "aggregate_exact_nine", lambda _rows: probabilities)
    monkeypatch.setattr(
        runner,
        "build_prelabel_surface",
        lambda *_args, **_kwargs: prelabel,
    )
    monkeypatch.setattr(
        runner,
        "persist_prelabel_surfaces",
        lambda *_args, **_kwargs: events.append("prelabel_seal"),
    )
    monkeypatch.setattr(runner, "FlipRouterLabelCapabilityManager", Manager)
    monkeypatch.setattr(
        runner, "persist_fold_plans", lambda *_args: events.append("plans_persisted")
    )
    monkeypatch.setattr(
        runner,
        "persist_donor_models",
        lambda *_args, **_kwargs: events.append("donor_persisted"),
    )
    monkeypatch.setattr(
        runner,
        "persist_static_and_calibration",
        lambda *_args, **_kwargs: events.append("static_calibration_persisted"),
    )

    def persist_decisions(*_args, **_kwargs) -> None:
        flags["decisions_persisted"] = True
        events.append("decisions_persisted")

    monkeypatch.setattr(runner, "persist_decisions", persist_decisions)
    monkeypatch.setattr(
        runner,
        "runtime_summary_payload",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "persist_terminal_checkpoint",
        lambda *_args, **_kwargs: events.append("terminal_persisted"),
    )
    monkeypatch.setattr(runner, "finalize_terminal_checkpoint", lambda _root: None)
    monkeypatch.setattr(runner, "remove_validated_terminal_checkpoint", lambda _root: None)
    monkeypatch.setattr(runner, "write_content_index", lambda *_args, **_kwargs: {})

    def full_validate(_root: Path, **kwargs):
        assert kwargs == {"config": config, "allow_pending_validation": True}
        flags["full_validation"] = True
        events.append("full_reconstructive_validation")
        return {
            "schema_version": "fixed_bank_labeled_support_flip_validation_v1",
            "status": "PASS",
            "scientific_factories_replayed": True,
        }

    monkeypatch.setattr(runner, "validate_bundle", full_validate)
    monkeypatch.setattr(
        runner,
        "persist_validation_report",
        lambda *_args: events.append("validation_report_persisted"),
    )

    def completed_binding(_root: Path, **kwargs) -> None:
        assert flags["full_validation"]
        assert kwargs["expected_checks"]["scientific_factories_replayed"] is True
        events.append("lightweight_complete_binding")

    monkeypatch.setattr(runner, "assert_completed_binding", completed_binding)
    monkeypatch.setattr(
        runner,
        "recover_if_possible",
        lambda *_args, **_kwargs: None,
    )

    deps = FlipRouterDependencies(
        materialize_source=lambda *_args, **_kwargs: (
            events.append("gpu_source"), source
        )[1],
        stage_source=lambda *_args, **_kwargs: source,
        materialize_predictions=lambda *_args, **_kwargs: (
            events.append("cpu_prediction"), prediction
        )[1],
        build_donor_models=lambda **_kwargs: (events.append("donor"), donor)[1],
        build_fold_decisions=lambda **_kwargs: (events.append("decisions"), decisions)[1],
        evaluate_terminal=lambda **_kwargs: (
            events.append("terminal"),
            {"sealed_terminal_evaluation": {"sealed_result_hash": "2" * 64}},
        )[1],
        cleanup_staging=lambda *_args, **_kwargs: events.append("cleanup"),
        phase_observer=lambda phase: events.append(f"phase:{phase}"),
    )

    assert runner._run(config, artifact_root=root, deps=deps) == root
    expected = (
        "gpu_source",
        "cpu_phase",
        "cpu_prediction",
        "prelabel_seal",
        "plans",
        "donor",
        "decisions",
        "decisions_persisted",
        "terminal_labels_after_45_decisions",
        "terminal",
        "full_reconstructive_validation",
        "validation_report_persisted",
        "lightweight_complete_binding",
    )
    assert [events.index(item) for item in expected] == sorted(
        events.index(item) for item in expected
    )


def test_production_protocol_gates_are_not_dependency_overrides() -> None:
    fields = set(FlipRouterDependencies.__dataclass_fields__)
    assert not fields & {
        "assert_input_fence",
        "validate_workspace_provenance",
        "validate_pre_gpu_firewall",
        "validate_bundle",
        "assert_completed_binding",
        "write_state",
    }


def test_compute_resume_freshly_reprobes_workstation_but_validation_only_loads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[int] = []

    def fake_preflight(_root: Path, **_kwargs):
        calls.append(len(calls) + 1)
        ordinal = calls[-1]
        return {
            "schema_version": "midogpp_label_free_workstation_preflight_v1",
            "status": "PASS",
            "generation_devices": ["cuda:0", "cuda:1"],
            "persistent_gpu_workers": 2,
            "classifier_workers": 4,
            "blas_threads_per_classifier_worker": 3,
            "target_action_identity_count": 90,
            "target_probability_cell_count": 810,
            "target_unique_classifier_fit_count": 810,
            "maximum_total_classifier_fit_count": 810,
            "gpu_then_cpu_phase_order": True,
            "phase_disjoint_gpu_and_cpu_pools": True,
            "parent_cuda_initialized": False,
            "tf32_enabled": False,
            "amp_enabled": False,
            "scratch_preference": [execution_adapter.SCRATCH_ROOT, "artifact_parent"],
            "available_cpu_affinity_count": 24,
            "physical_ram_bytes": 128_000_000_000,
            "disk_probe_path": str(_root.resolve()),
            "disk_free_bytes_at_launch": 20_000_000_000 + ordinal,
            "thread_environment": dict(execution_adapter.REQUIRED_THREAD_ENVIRONMENT),
            "cuda_visible_devices": "0,1",
            "package_versions": {
                name: "fixture" for name in execution_adapter.REQUIRED_DISTRIBUTIONS
            },
            "gpus": [
                {
                    "index": index,
                    "name": "NVIDIA RTX A5000",
                    "memory_total_mib": 24_000,
                    "memory_free_mib": 20_000,
                }
                for index in (0, 1)
            ],
        }

    monkeypatch.setattr(execution_adapter, "_preflight", fake_preflight)
    runtime = canonical_runtime_payload()
    first = execution_adapter.run_workstation_preflight(tmp_path, runtime=runtime)
    second = execution_adapter.run_workstation_preflight(tmp_path, runtime=runtime)
    loaded = execution_adapter.load_validated_workstation_preflight(
        tmp_path, runtime=runtime
    )

    assert calls == [1, 2]
    assert first["disk_free_bytes_at_launch"] != second["disk_free_bytes_at_launch"]
    assert loaded == second
    assert calls == [1, 2]


def test_terminal_finalization_recovery_only_revalidates_sealed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical header retry cannot re-enter prediction or policy phases."""

    root = tmp_path / "artifact"
    root.mkdir()
    config = SimpleNamespace(contract_hash="c" * 64)
    protocol = SimpleNamespace(contract_hash="p" * 64)
    capability = RecoveryCapability(
        mode="TERMINAL_FINALIZATION",
        state_phase="FINALIZATION",
        resume_phase="FINALIZATION",
        labels_may_be_reopened_for_validation=True,
        labels_may_be_opened_for_deterministic_policy_construction=False,
        labels_may_update_frozen_policy_contract=False,
        validation_only=True,
    )
    events: list[str] = []
    checks = {"status": "PASS", "scientific_factories_replayed": True}

    monkeypatch.setattr(
        runner_runtime, "recovery_capability", lambda _root: capability
    )
    monkeypatch.setattr(
        runner_runtime,
        "cleanup_owned_atomic_temps",
        lambda _root: events.append("cleanup_temps"),
    )
    monkeypatch.setattr(
        runner_runtime,
        "write_state",
        lambda _root, **kwargs: events.append(
            f"state:{kwargs['status']}:{kwargs['phase']}"
        ),
    )
    monkeypatch.setattr(
        runner_runtime,
        "write_content_index",
        lambda *_args, **_kwargs: events.append("content_index"),
    )
    monkeypatch.setattr(
        runner_runtime,
        "enter_cuda_free_cpu_phase",
        lambda: events.append("cuda_free_validation"),
    )

    def validate(_root: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"config": config, "allow_pending_validation": True}
        events.append("reconstructive_validation")
        return checks

    monkeypatch.setattr(runner_runtime, "validate_bundle", validate)
    monkeypatch.setattr(
        runner_runtime,
        "persist_validation_report",
        lambda _root, observed: events.append(
            "validation_report" if observed == checks else "wrong_report"
        ),
    )
    monkeypatch.setattr(
        runner_runtime,
        "assert_completed_binding",
        lambda _root, **kwargs: events.append(
            "complete_binding" if kwargs["expected_checks"] == checks else "wrong_binding"
        ),
    )

    assert runner_runtime.recover_if_possible(
        root, config=config, protocol=protocol
    ) == root
    assert events == [
        "cleanup_temps",
        "state:RUNNING:FINALIZATION",
        "content_index",
        "cuda_free_validation",
        "reconstructive_validation",
        "validation_report",
        "state:COMPLETE:COMPLETE",
        "complete_binding",
    ]
