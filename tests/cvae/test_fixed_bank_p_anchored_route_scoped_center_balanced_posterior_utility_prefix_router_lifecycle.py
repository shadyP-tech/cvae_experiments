from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    fresh_process_validation,
    runner,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.hashing import (
    canonical_hash,
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.preterminal_gate import (
    persist_preterminal_validation_report,
    persist_preterminal_validation_seal,
    preterminal_validation_checks_payload,
    preterminal_validation_report_payload,
    validate_preterminal_gate_artifacts,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _preterminal(events: list[str]) -> tuple[object, dict[str, str]]:
    digests = {name: name[0] * 64 for name in "abcde"}

    class Firewall:
        def open_target_terminal_labels(self) -> tuple[object, ...]:
            events.append("terminal_loader_called")
            return (object(),)

        def audit_payload(self) -> dict[str, object]:
            events.append("terminal_audit_read")
            return {"terminal_opened": True}

    return (
        SimpleNamespace(
            preterminal_hash=digests["a"],
            candidates=SimpleNamespace(
                firewall=Firewall(),
                target_candidate_seal_hash=digests["b"],
                pre_evaluation_seal_hash=digests["c"],
            ),
            decisions=SimpleNamespace(
                replay_calibration_seal_hash=digests["d"],
                aggregate_seal_hash=digests["e"],
            ),
        ),
        digests,
    )


def _persist_in_memory_barriers(
    root: Path, *, digests: dict[str, str]
) -> None:
    (root / "manifests").mkdir(parents=True)
    decision = {
        "schema_version": "fixed_bank_cbpupr_decision_barrier_v1",
        "candidate_seal_hash": digests["b"],
        "pre_evaluation_seal_hash": digests["c"],
        "replay_calibration_seal_hash": digests["d"],
        "pseudo_evaluation_opened_after_candidate_seal": True,
        "target_evaluation_opened": False,
    }
    (root / "manifests/decision_barrier.json").write_text(
        json.dumps(
            {**decision, "decision_barrier_hash": canonical_hash(decision)}
        )
    )
    (root / "manifests/preterminal_aggregate_seal.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "fixed_bank_cbpupr_preterminal_aggregate_seal_v1"
                ),
                "aggregate_seal_hash": digests["e"],
                "preterminal_hash": digests["a"],
                "target_evaluation_opened": False,
            }
        )
    )


def test_preterminal_validation_failure_never_calls_terminal_loader_or_persists_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    preterminal, digests = _preterminal(events)
    _persist_in_memory_barriers(tmp_path, digests=digests)
    checks = {"preterminal_hash": digests["a"]}

    def fail_validation(*_args: object, **_kwargs: object) -> object:
        events.append("attested_validation_failed")
        raise ProtocolError("poisoned preterminal posterior lineage")

    monkeypatch.setattr(runner, "validate_preterminal_gate_artifacts", fail_validation)
    monkeypatch.setattr(
        runner,
        "persist_label_capability_report",
        lambda *_args, **_kwargs: events.append("terminal_capability_persisted"),
    )

    with pytest.raises(ProtocolError, match="poisoned preterminal"):
        runner._open_terminal_after_durable_preterminal(
            tmp_path, preterminal, expected_checks=checks
        )

    assert events == ["attested_validation_failed"]
    terminal_members = (
        "manifests/terminal_label_access_intent.json",
        "reports/terminal_label_access_opened_receipt.json",
        "reports/label_capability_report.json",
        "manifests/terminal_evaluation_seal.json",
        "tables/terminal_method_metrics.json",
        "tables/terminal_center_contrasts.json",
        "tables/terminal_case_oracles.json",
    )
    assert not any((tmp_path / member).exists() for member in terminal_members)


def test_terminal_phase_is_durable_before_loader_or_capability_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    states: list[tuple[str, str]] = []
    firewall = SimpleNamespace(
        audit_payload=lambda: {
            "schema_version": "fixed_bank_cbpupr_label_access_audit_v1",
            "events": [],
            "aggregate_seal_complete": True,
            "terminal_opened": False,
            "raw_labels_persisted": False,
        }
    )
    preterminal = SimpleNamespace(candidates=SimpleNamespace(firewall=firewall))
    physical = SimpleNamespace(prediction=object(), scratch=tmp_path / "scratch")
    config = SimpleNamespace(
        artifact_root=tmp_path / "bundle",
        runtime={},
        experiment_id="v2",
    )

    no_op_names = (
        "reject_quarantined_v1_execution",
        "assert_workspace_resolved_paths",
        "assert_launch_files",
        "reject_existing_run_state",
        "assert_no_partial_state",
        "assert_input_fence",
        "validate_active_workspace_binding",
        "persist_admission",
        "assert_cuda_free_cpu_phase",
        "persist_physical_surface",
        "persist_preterminal",
        "persist_preterminal_capability_report",
        "write_preterminal_content_index",
        "persist_preterminal_validation_report",
        "persist_preterminal_validation_seal",
        "verify_preterminal_attested_bundle",
    )
    for name in no_op_names:
        monkeypatch.setattr(runner, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "exclusive_run_lock", lambda _root: nullcontext())
    monkeypatch.setattr(
        runner,
        "write_state",
        lambda _root, *, status, phase, **_kwargs: states.append((status, phase)),
    )
    monkeypatch.setattr(runner, "validate_workspace_provenance", lambda *_args: {})
    monkeypatch.setattr(runner, "load_validated_locks", lambda _config: SimpleNamespace(generation=object()))
    monkeypatch.setattr(runner, "load_label_free_test_frame", lambda _config: object())
    monkeypatch.setattr(runner, "validate_pre_gpu_firewall", lambda *_args: {})
    monkeypatch.setattr(runner, "run_workstation_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "materialize_physical_inputs", lambda *_args, **_kwargs: physical)
    monkeypatch.setattr(runner, "build_surface", lambda _physical: object())
    monkeypatch.setattr(runner, "probability_index_rows", lambda _prediction: ())
    monkeypatch.setattr(runner, "build_preterminal_result", lambda *_args, **_kwargs: preterminal)
    checks = {"preterminal_hash": "a" * 64}
    attestation = {"attestation_hash": "b" * 64}
    report = {"validation_report_hash": "c" * 64}
    monkeypatch.setattr(runner, "validate_preterminal_bundle", lambda *_args, **_kwargs: checks)
    monkeypatch.setattr(
        runner,
        "require_two_fresh_preterminal_process_validations",
        lambda *_args, **_kwargs: attestation,
    )
    monkeypatch.setattr(
        runner,
        "preterminal_validation_report_payload",
        lambda *_args, **_kwargs: report,
    )

    def fail_after_loader(*_args: object, **_kwargs: object) -> object:
        events.append("terminal_loader_opened")
        raise ProtocolError("capability persistence failed after label open")

    monkeypatch.setattr(
        runner, "_open_terminal_after_durable_preterminal", fail_after_loader
    )

    with pytest.raises(ProtocolError, match="after label open"):
        runner.run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
            config
        )

    terminal_phase = "TERMINAL_LABELS_METRICS_AND_CONTROLS"
    assert events == ["terminal_loader_opened"]
    assert states[-2:] == [("RUNNING", terminal_phase), ("FAILED", terminal_phase)]


def test_two_preterminal_workers_are_phase_bound_and_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "manifests").mkdir()
    checks = preterminal_validation_checks_payload(
        config_contract_hash="1" * 64,
        protocol_contract_hash="2" * 64,
        content_index_hash="a" * 64,
        outer_route_count=218,
        target_posterior_model_fit_count=436,
        pseudo_posterior_reference_count=3488,
        preterminal_hash="c" * 64,
    )
    calls: list[tuple[list[str], dict[str, str]]] = []
    child_ids = iter((81001, 81002))

    def fake_run(command: list[str], **kwargs: object) -> object:
        process_id = next(child_ids)
        environment = dict(kwargs["env"])  # type: ignore[arg-type]
        calls.append((list(command), environment))
        payload = {
            "process_id": process_id,
            "validation_phase": "preterminal",
            "checks": checks,
        }
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_json(payload),
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation.subprocess, "run", fake_run)
    attestation = (
        fresh_process_validation.require_two_fresh_preterminal_process_validations(
            tmp_path, expected_checks=checks
        )
    )
    assert len(calls) == 2
    assert all(
        command[-2:] == ["--phase", "preterminal"]
        and environment["CUDA_VISIBLE_DEVICES"] == ""
        for command, environment in calls
    )
    assert attestation["child_process_ids"] == [81001, 81002]
    assert attestation["terminal_opened"] is False

    report = preterminal_validation_report_payload(checks, attestation)
    persist_preterminal_validation_report(tmp_path, report)
    persist_preterminal_validation_seal(
        tmp_path,
        checks=checks,
        attestation=attestation,
        report=report,
    )
    validate_preterminal_gate_artifacts(tmp_path, expected_checks=checks)

    poisoned = dict(report)
    poisoned["checks"] = {**checks, "preterminal_hash": "d" * 64}
    (tmp_path / "reports/preterminal_validation_report.json").write_text(
        json.dumps(poisoned)
    )
    with pytest.raises(ProtocolError, match="validation report drifted"):
        validate_preterminal_gate_artifacts(tmp_path, expected_checks=checks)
