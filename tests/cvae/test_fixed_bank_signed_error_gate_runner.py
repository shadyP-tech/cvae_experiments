from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.bundle import (
    cleanup_owned_atomic_temps,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    execution_adapter,
    runner as runner_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.runner import (
    FixedBankSignedErrorGateDependencies,
    _exclusive_run_lock,
    run_fixed_bank_signed_error_gate,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_HASH = "a" * 64


def test_runner_persists_every_capability_boundary_before_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    absolute = tmp_path.resolve()
    config = SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=absolute / "bank",
        generation_lock_root=absolute / "generation",
        test_cache_root=absolute / "cache",
        test_manifest_path=absolute / "manifest.csv",
        test_consumption_ledger_path=absolute / "ledger.json",
        ledger_amendment_path=absolute / "amendment.json",
        runtime={
            "classifier_workers": 4,
            "model_workers": 4,
            "model_threads_per_worker": 3,
            "bootstrap_workers": 4,
            "bootstrap_threads_per_worker": 3,
            "multiprocessing_start_method": "spawn",
            "resume_policy": (
                "hash_validated_source_prediction_task_resume_plus_"
                "deterministic_phase_replay"
            ),
        },
        evaluation={
            "whole_case_cluster_bootstrap_replicates": 10_000,
            "whole_case_cluster_bootstrap_seed": 90_912_028,
        },
        contract_hash=_HASH,
    )
    state = {
        "models_persisted": False,
        "models_recorded": False,
        "folds_persisted": False,
        "folds_recorded": False,
        "evaluation_opened": False,
        "postseal_persisted": False,
        "index_written": False,
    }
    phases: list[str] = []
    frame = SimpleNamespace(cache_binding_hash=_HASH)
    partition = SimpleNamespace(partition_hash=_HASH)
    source = SimpleNamespace(lock_hash=_HASH, records=(1,))
    prediction = SimpleNamespace(
        seal_hash=_HASH,
        store=SimpleNamespace(cells=tuple(range(729)), store_hash="b" * 64),
    )
    prelabel = SimpleNamespace(
        feature_surface_hash="c" * 64,
        context_hashes={"context": "d" * 64},
        protocol_contract_hash="e" * 64,
    )
    models = SimpleNamespace(target_fits=tuple(range(9)))
    folds = SimpleNamespace(decisions=tuple({"fold": value} for value in range(45)))

    class _Manager:
        def open_oof_evaluation_labels(self) -> tuple[object, ...]:
            assert state["folds_recorded"]
            state["evaluation_opened"] = True
            return (object(),)

        def access_report(self) -> dict[str, object]:
            assert state["evaluation_opened"]
            return {"evaluation_labels_opened": True}

    manager = _Manager()

    def _persist_prelabel(path: Path, **_: object) -> None:
        target = path / "manifests/signed_prelabel_feature_seal.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"feature_surface_hash": prelabel.feature_surface_hash}),
            encoding="utf-8",
        )

    def _persist_models(_: Path, *, products: object) -> None:
        assert products is models
        state["models_persisted"] = True

    def _record_models(capability: object, products: object) -> None:
        assert capability is manager and products is models
        assert state["models_persisted"]
        state["models_recorded"] = True

    def _fit_folds(**_: object) -> object:
        assert state["models_recorded"]
        return folds

    def _persist_folds(_: Path, *, products: object) -> None:
        assert products is folds and len(folds.decisions) * 6 == 270
        state["folds_persisted"] = True

    def _record_folds(capability: object, products: object) -> None:
        assert capability is manager and products is folds
        assert state["folds_persisted"]
        state["folds_recorded"] = True

    def _evaluate(**_: object) -> object:
        assert state["evaluation_opened"]
        return object()

    def _persist_postseal(_: Path, **__: object) -> None:
        assert state["evaluation_opened"]
        state["postseal_persisted"] = True

    def _write_index(_: Path, **__: object) -> None:
        assert state["postseal_persisted"]
        state["index_written"] = True

    def _validate(_: Path, **__: object) -> dict[str, object]:
        assert state["index_written"]
        return {"status": "PASS"}

    deps = FixedBankSignedErrorGateDependencies(
        validate_inputs=lambda _: None,
        validate_workspace=lambda _: {"status": "PASS"},
        validate_provenance=lambda *_: {},
        load_locks=lambda _: SimpleNamespace(generation=object()),
        load_frame=lambda _: frame,
        validate_firewall=lambda *_: {"status": "PASS"},
        build_partition=lambda *_args, **_kwargs: partition,
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source=lambda *_args, **_kwargs: source,
        stage_source=lambda *_args, **_kwargs: source,
        materialize_predictions=lambda *_args, **_kwargs: prediction,
        build_seed_rows=lambda _: (object(),),
        aggregate_probabilities=lambda _: ((object(),), _HASH),
        build_prelabel=lambda *_args, **_kwargs: prelabel,
        persist_prelabel=_persist_prelabel,
        build_label_manager=lambda *_args, **_kwargs: manager,
        fit_models=lambda **_: models,
        persist_models=_persist_models,
        record_models=_record_models,
        fit_folds=_fit_folds,
        persist_folds=_persist_folds,
        record_folds=_record_folds,
        evaluate=_evaluate,
        persist_postseal=_persist_postseal,
        write_index=_write_index,
        validate_bundle=_validate,
        persist_validation=lambda *_args, **_kwargs: None,
        phase_observer=phases.append,
    )
    observed = run_fixed_bank_signed_error_gate(
        config, artifact_root=root.resolve(), dependencies=deps
    )

    assert observed == root.resolve()
    assert state == {
        "models_persisted": True,
        "models_recorded": True,
        "folds_persisted": True,
        "folds_recorded": True,
        "evaluation_opened": True,
        "postseal_persisted": True,
        "index_written": True,
    }
    assert phases.index("loco_donor_labels_and_signed_models") < phases.index(
        "support_calibrations_and_six_method_decisions"
    ) < phases.index("terminal_evaluation_labels")


def test_run_lock_survives_process_exit_and_still_excludes_concurrent_run(
    tmp_path: Path,
) -> None:
    with _exclusive_run_lock(tmp_path):
        with pytest.raises(ProtocolError, match="already running"):
            with _exclusive_run_lock(tmp_path):
                raise AssertionError("unreachable")

    assert (tmp_path / ".run.lock").is_file()
    with _exclusive_run_lock(tmp_path):
        pass


def test_atomic_temp_cleanup_is_limited_to_owned_bundle_members(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "tables/oof_predictions.csv.123.tmp"
    checkpoint = (
        tmp_path
        / "checkpoints/frozen_source_streams/source_0_train_17.npy.456.tmp"
    )
    unrelated = tmp_path / "notes.txt.789.tmp"
    for path in (owned, checkpoint, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial", encoding="utf-8")

    cleanup_owned_atomic_temps(tmp_path)

    assert not owned.exists()
    assert not checkpoint.exists()
    assert unrelated.is_file()


def test_signed_preflight_translates_only_the_shared_resume_compatibility_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def shared_preflight(
        root: Path, *, runtime: dict[str, object], expected_scratch_root: str
    ) -> dict[str, object]:
        observed["root"] = root
        observed["runtime"] = runtime
        observed["scratch"] = expected_scratch_root
        return {"schema_version": "test", "status": "PASS"}

    monkeypatch.setattr(execution_adapter, "_preflight", shared_preflight)
    runtime = canonical_runtime_payload()
    payload = execution_adapter.run_label_free_workstation_preflight(
        tmp_path, runtime=runtime
    )

    assert observed["runtime"]["resume_policy"] == (  # type: ignore[index]
        "hash_validated_atomic_phase_and_task_checkpoints"
    )
    assert runtime["resume_policy"] == (
        "hash_validated_source_prediction_task_resume_plus_"
        "deterministic_phase_replay"
    )
    assert payload["probability_store_format"] == "compressed_float32_npz"
    assert payload["cross_target_context_cache_present"] is False
    assert json.loads(
        (tmp_path / "reports/workstation_preflight.json").read_text(
            encoding="utf-8"
        )
    ) == payload
    assert execution_adapter.run_label_free_workstation_preflight(
        tmp_path, runtime=runtime
    ) == payload


def test_complete_terminal_phase_recovers_by_validation_without_reopening_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "reports").mkdir()
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    (root / "manifests/sealed_terminal_evaluation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    config = SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
        contract_hash=_HASH,
    )
    events: list[str] = []
    monkeypatch.setattr(runner_module, "cleanup_owned_atomic_temps", lambda _: None)
    monkeypatch.setattr(
        runner_module, "assert_closed_world", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_module,
        "assert_terminal_phase_complete",
        lambda _: events.append("terminal_inventory"),
    )

    deps = FixedBankSignedErrorGateDependencies(
        write_index=lambda *_args, **_kwargs: events.append("index"),
        validate_bundle=lambda *_args, **_kwargs: (
            events.append("validate") or {"status": "PASS"}
        ),
        persist_validation=lambda *_args, **_kwargs: events.append(
            "persist_validation"
        ),
        write_state=lambda *_args, **kwargs: events.append(
            f"state:{kwargs['status']}"
        ),
        validate_inputs=lambda _: (_ for _ in ()).throw(
            AssertionError("terminal recovery must not restart scientific phases")
        ),
    )

    assert run_fixed_bank_signed_error_gate(
        config, artifact_root=root.resolve(), dependencies=deps
    ) == root.resolve()
    assert events == [
        "terminal_inventory",
        "state:RUNNING",
        "index",
        "validate",
        "persist_validation",
        "state:COMPLETE",
        "validate",
    ]
