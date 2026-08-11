from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_contracts import (
    FAILED_INFERENCE_STATE,
    FrozenModelBankView,
    POST_TEST_SEAL_RECOVERY_FILES,
    RECOVERY_TERMINAL_FILES,
    PostTestSealRecovery,
    detect_post_test_seal_recovery,
    rollback_post_test_seal_recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_provenance import (
    recovery_audit_payload,
    validate_recovery_audit_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.runner import (
    run_fixed_bank_disagreement_regret_prediction_only,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_runtime import (
    resume_post_test_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.runner_dependencies import (
    PredictionOnlyDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


def test_detector_accepts_only_exact_post_test_seal_failure(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    (root / ".run.lock").write_text("operational\n", encoding="utf-8")

    assert detect_post_test_seal_recovery(root) is True

    checkpoint = root / "checkpoints/disagreement_regret_test_predictions/task.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="boundary drifted"):
        detect_post_test_seal_recovery(root)


def test_detector_excludes_only_the_root_operational_lock(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    nested_lock = root / "manifests/.run.lock"
    nested_lock.write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="boundary drifted"):
        detect_post_test_seal_recovery(root)


def test_detector_rejects_failed_state_cause_drift(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    atomic_json(
        root / "reports/run_state.json",
        {**FAILED_INFERENCE_STATE, "error": "ProtocolError: another failure"},
    )

    with pytest.raises(ProtocolError, match="state_matches=False"):
        detect_post_test_seal_recovery(root)


def test_detector_rejects_owned_temporary_member(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    temporary = root / "manifests/test_prediction_seal.json.123.tmp"
    temporary.write_text("interrupted\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="boundary drifted"):
        detect_post_test_seal_recovery(root)


def test_recovery_rollback_restores_exact_sealed_boundary(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    preserved = (root / "arrays/model_bank.npz").read_bytes()
    for relative in RECOVERY_TERMINAL_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("recovery-output\n", encoding="utf-8")
    interrupted = root / "reports/runtime_summary.json.123.tmp"
    interrupted.write_text("interrupted\n", encoding="utf-8")

    rollback_post_test_seal_recovery(root)

    assert detect_post_test_seal_recovery(root) is True
    assert (root / "arrays/model_bank.npz").read_bytes() == preserved
    assert all(not (root / relative).exists() for relative in RECOVERY_TERMINAL_FILES)
    assert not interrupted.exists()


def test_validator_failure_rolls_recovery_back_for_exact_retry(
    tmp_path: Path,
) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    recovery = SimpleNamespace(
        generated_sources=object(),
        source_predictions=SimpleNamespace(seal_hash="a" * 64),
        development=FrozenModelBankView(
            model_banks=(object(),), model_bank_hash="c" * 64
        ),
        source_label_capability_report={
            "source_labels_opened": True,
            "source_labels_opened_after_complete_prediction_seal": True,
            "raw_source_labels_persisted": False,
            "test_labels_opened": False,
            "test_labels_available": False,
        },
        test_predictions=SimpleNamespace(seal_hash="b" * 64),
        workstation_preflight={"status": "PASS"},
        train_test_disjointness={"status": "PASS"},
        audit={"recovery_used": True},
    )
    inference = SimpleNamespace(frozen_prediction_hash="d" * 64)

    def persist_inference(path: Path, _inference: object) -> None:
        member = path / "tables/test_case_features.csv"
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("partial\n", encoding="utf-8")

    def persist_reports(path: Path, **_kwargs: object) -> None:
        member = path / "reports/runtime_summary.json"
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("{}\n", encoding="utf-8")

    def write_index(path: Path, **_kwargs: object) -> None:
        member = path / "manifests/content_index.json"
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("{}\n", encoding="utf-8")

    dependencies = PredictionOnlyDependencies(
        aggregate_test_probabilities=lambda *_args, **_kwargs: (),
        build_inference=lambda *_args, **_kwargs: inference,
        persist_inference=persist_inference,
        build_runtime_summary=lambda **_kwargs: {"status": "PASS"},
        persist_reports=persist_reports,
        write_content_index=write_index,
        validate_bundle=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProtocolError("validator mismatch")
        ),
        validate_recovery_checkout=lambda _audit: None,
        validate_terminal_completion=lambda _root: None,
    )
    config = SimpleNamespace(
        runtime={},
        contract_hash="config-hash",
        expected_test_cache_content_hash="5" * 64,
        expected_test_cache_row_order_hash="6" * 64,
    )

    with pytest.raises(ProtocolError, match="validator mismatch"):
        resume_post_test_seal(
            root,
            config=config,
            protocol=SimpleNamespace(contract_hash="protocol-hash"),
            recovery=recovery,
            dependencies=dependencies,
            default_validator=None,
        )

    assert detect_post_test_seal_recovery(root) is True


def test_recovery_resumes_after_seals_without_reopening_inputs(tmp_path: Path) -> None:
    root = _write_partial_bundle(tmp_path / "bundle")
    events: list[str] = []
    forbidden: list[str] = []
    runtime_summaries: list[dict[str, object]] = []
    validator_calls = 0
    model_seal_hash = "e" * 64
    test_seal_hash = "b" * 64
    audit = recovery_audit_payload(
        original_repository_state={
            "repository_revision": "1" * 40,
            "repository_dirty": False,
            "repository_status_hash": "2" * 64,
        },
        repair_repository_state={
            "repository_revision": "3" * 40,
            "repository_dirty": False,
            "repository_status_hash": "4" * 64,
        },
        model_bank_seal_hash=model_seal_hash,
        test_prediction_seal_hash=test_seal_hash,
    )
    source_predictions = SimpleNamespace(seal_hash="a" * 64)
    test_predictions = SimpleNamespace(seal_hash=test_seal_hash)
    development = FrozenModelBankView(model_banks=(object(),), model_bank_hash="c" * 64)
    recovery = PostTestSealRecovery(
        generated_sources=object(),
        source_predictions=source_predictions,
        development=development,
        source_label_capability_report={
            "source_labels_opened": True,
            "source_labels_opened_after_complete_prediction_seal": True,
            "raw_source_labels_persisted": False,
            "test_labels_opened": False,
            "test_labels_available": False,
        },
        test_predictions=test_predictions,
        workstation_preflight={"status": "PASS"},
        train_test_disjointness={"status": "PASS"},
        audit=audit,
    )
    inference = SimpleNamespace(frozen_prediction_hash="d" * 64)

    def forbid(name: str):
        def callback(*_args: object, **_kwargs: object) -> object:
            forbidden.append(name)
            raise AssertionError(f"recovery called forbidden dependency: {name}")

        return callback

    def load_recovery(path: Path, *, config: object) -> object:
        assert path == root
        assert config is not None
        assert read_json(path / "reports/run_state.json") == FAILED_INFERENCE_STATE
        events.append("failed_state_captured")
        return recovery

    def build_inference(view: object, *_args: object, **_kwargs: object) -> object:
        assert view is development
        events.append("inference_built")
        return inference

    def persist_inference(_root: Path, value: object) -> None:
        assert value is inference
        events.append("inference_persisted")

    def persist_reports(
        _root: Path,
        *,
        leakage: object,
        publication: object,
        runtime_summary: object,
    ) -> None:
        assert leakage is not None and publication is not None
        runtime_summaries.append(dict(runtime_summary))  # type: ignore[arg-type]
        events.append("reports_persisted")

    def validate_bundle(_root: Path, *, config: object) -> dict[str, object]:
        nonlocal validator_calls
        assert config is not None
        assert "inference_persisted" in events
        assert "reports_persisted" in events
        validator_calls += 1
        events.append("validator_refit_audit")
        return {"status": "PASS"}

    config = SimpleNamespace(
        source_path=root / "config.resolved.yaml",
        artifact_root=root,
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        train_cache_root=tmp_path / "train",
        test_cache_root=tmp_path / "test",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
        runtime={},
        contract_hash="config-hash",
        expected_test_cache_content_hash="5" * 64,
        expected_test_cache_row_order_hash="6" * 64,
    )
    dependencies = PredictionOnlyDependencies(
        load_post_test_recovery=load_recovery,
        validate_recovery_checkout=lambda _audit: None,
        validate_input_fence=forbid("input_fence"),
        load_locks=forbid("locks"),
        load_source_frame=forbid("source_frame"),
        build_source_label_capability=forbid("source_labels"),
        fit_development=forbid("model_refit"),
        issue_test_admission=forbid("second_test_admission"),
        load_test_frame=forbid("test_cache"),
        materialize_test_predictions=forbid("test_prediction_rerun"),
        aggregate_test_probabilities=lambda *_args, **_kwargs: (object(),),
        build_inference=build_inference,
        persist_inference=persist_inference,
        build_runtime_summary=lambda **_kwargs: {"status": "PASS"},
        persist_reports=persist_reports,
        write_content_index=lambda *_args, **_kwargs: events.append(
            "content_index_persisted"
        ),
        validate_bundle=validate_bundle,
        persist_validation=lambda *_args, **_kwargs: events.append(
            "validation_persisted"
        ),
        validate_terminal_completion=lambda _root: events.append(
            "terminal_completion_validated"
        ),
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=events.append,
    )

    result = run_fixed_bank_disagreement_regret_prediction_only(
        config, artifact_root=root, dependencies=dependencies
    )

    assert result == root
    assert forbidden == []
    assert validator_calls == 1
    assert events.index("failed_state_captured") < events.index(
        "inference_built"
    )
    assert events.index("inference_persisted") < events.index(
        "validator_refit_audit"
    )
    assert events.index("content_index_persisted") < events.index(
        "validator_refit_audit"
    )
    assert runtime_summaries[0]["post_test_seal_recovery"] == audit


def test_recovery_audit_is_hash_bound_and_requires_clean_repair() -> None:
    kwargs = {
        "original_repository_state": {
            "repository_revision": "1" * 40,
            "repository_dirty": True,
            "repository_status_hash": "2" * 64,
        },
        "repair_repository_state": {
            "repository_revision": "3" * 40,
            "repository_dirty": False,
            "repository_status_hash": "4" * 64,
        },
        "model_bank_seal_hash": "5" * 64,
        "test_prediction_seal_hash": "6" * 64,
    }
    audit = recovery_audit_payload(**kwargs)  # type: ignore[arg-type]
    assert (
        validate_recovery_audit_payload(
            audit,
            model_bank_seal_hash="5" * 64,
            test_prediction_seal_hash="6" * 64,
        )
        == audit
    )
    with pytest.raises(ProtocolError, match="audit drifted"):
        validate_recovery_audit_payload(
            {**audit, "production_test_predictions_recomputed": True},
            model_bank_seal_hash="5" * 64,
            test_prediction_seal_hash="6" * 64,
        )
    with pytest.raises(ProtocolError, match="repair repository state"):
        recovery_audit_payload(
            **{
                **kwargs,
                "repair_repository_state": {
                    "repository_revision": "3" * 40,
                    "repository_dirty": True,
                    "repository_status_hash": "4" * 64,
                },
            }
        )


def _write_partial_bundle(root: Path) -> Path:
    for relative in POST_TEST_SEAL_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            atomic_json(path, FAILED_INFERENCE_STATE)
        else:
            path.write_text("sealed-placeholder\n", encoding="utf-8")
    return root
