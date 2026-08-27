from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.config import (
    CANONICAL_CLASSIFIER,
    ScaleBPV2Config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.hashing import (
    canonical_hash,
)


MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2.runner"
)
PHYSICAL_MODULE = MODULE.rsplit(".", 1)[0] + ".physical_runtime"


def _digest(value: object) -> str:
    return canonical_hash({"value": value})


def _config(tmp_path: Path) -> ScaleBPV2Config:
    root = (tmp_path / "artifact").resolve()
    scratch = (tmp_path / "scratch").resolve()
    return ScaleBPV2Config(
        source_path=(tmp_path / "config.yaml").resolve(),
        artifact_root=root,
        scratch_root=scratch,
        expert_bank_root=(tmp_path / "bank").resolve(),
        generation_lock_root=(tmp_path / "generation").resolve(),
        test_cache_root=(tmp_path / "cache").resolve(),
        test_manifest_path=(tmp_path / "manifest.csv").resolve(),
        test_consumption_ledger_path=(tmp_path / "ledger.json").resolve(),
        ledger_amendment_path=(tmp_path / "amendment.json").resolve(),
        protocol={"protocol_hash": _digest("protocol")},
        scientific_contracts={},
        classifier=CANONICAL_CLASSIFIER,
        runtime={},
        claim_boundary={"terminal": True},
        contract_hash=_digest("config"),
        expected_authorization_amendment_sha256=_digest("amendment"),
        expected_source_snapshot_manifest_sha256=_digest("source-manifest"),
        expected_source_snapshot_tree_sha256=_digest("source-tree"),
        expected_source_snapshot_member_count=80,
    )


def _patch_success_lifecycle(monkeypatch, config: ScaleBPV2Config) -> list[str]:
    runner = importlib.import_module(MODULE)
    physical_module = importlib.import_module(PHYSICAL_MODULE)
    events: list[str] = []

    for key in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    admission = SimpleNamespace(
        receipt_hash=_digest("admission"),
        direct_input_binding_hash=_digest("inputs"),
        source_fence_receipt_hash=_digest("source-fence"),
        authorization_lease_path=str(
            config.artifact_root.parent / ".scale_bp_v2_single_use_authorization_consumed"
        ),
    )
    lease = SimpleNamespace(claim_hash=_digest("lease"))
    frame = SimpleNamespace(
        frame_hash=_digest("frame"), cache_binding_hash=_digest("cache")
    )
    inputs = SimpleNamespace(generation_lock=object())
    workstation = SimpleNamespace(
        to_payload=lambda: {"plan_hash": _digest("workstation")}
    )
    physical = SimpleNamespace(store=object(), physical_receipt={})
    memmaps = SimpleNamespace(
        references=(),
        bundle_hash=_digest("bundle"),
        index_hash=_digest("index"),
        index_path=config.scratch_root / "physical-index.json",
    )
    identity = SimpleNamespace(identity_hash=_digest("identity"))
    journal = SimpleNamespace(
        seal_decisions=lambda **_kwargs: events.append("decision_seal"),
        audit_payload=lambda: {"audit_hash": _digest("journal")},
    )
    outer = SimpleNamespace(
        route_payloads={}, center_manifests=(), outer_results_hash=_digest("outer")
    )
    terminal = SimpleNamespace(elapsed_seconds=0.0)

    monkeypatch.setattr(
        runner, "validate_protocol_payload", lambda _payload: events.append("protocol")
    )
    monkeypatch.setattr(
        runner,
        "admit_single_use_execution",
        lambda *_args, **_kwargs: events.append("admission") or admission,
    )
    monkeypatch.setattr(
        runner,
        "validate_outer_worker_callback",
        lambda **_kwargs: events.append("callback")
        or (lambda _task, _maps: None, {"receipt_hash": _digest("worker")}),
    )
    monkeypatch.setattr(
        runner,
        "load_label_free_test_frame",
        lambda _config: events.append("frame") or frame,
    )
    monkeypatch.setattr(
        runner,
        "load_validated_inputs",
        lambda _config: events.append("inputs") or inputs,
    )
    monkeypatch.setattr(
        runner,
        "validate_pre_gpu_firewall",
        lambda *_args: events.append("firewall") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "preflight_workstation",
        lambda *_args: events.append("host") or workstation,
    )
    monkeypatch.setattr(
        runner,
        "claim_authorization_lease",
        lambda *_args, **_kwargs: events.append("claim") or lease,
    )
    monkeypatch.setattr(
        runner,
        "create_single_use_run",
        lambda *_args, **_kwargs: events.append("create") or {},
    )
    monkeypatch.setattr(
        runner,
        "persist_launch_receipts",
        lambda *_args, **_kwargs: events.append("launch_receipts"),
    )
    monkeypatch.setattr(
        runner,
        "read_json_object",
        lambda _path: {"manifest_hash": _digest("protocol-manifest")},
    )
    monkeypatch.setattr(
        runner,
        "transition_run",
        lambda *_args, **_kwargs: events.append("transition") or {},
    )
    monkeypatch.setattr(
        physical_module,
        "materialize_physical_bank",
        lambda *_args, **_kwargs: events.append("physical") or physical,
    )
    monkeypatch.setattr(
        runner,
        "persist_physical_memmaps",
        lambda *_args, **_kwargs: events.append("memmaps") or memmaps,
    )
    monkeypatch.setattr(
        runner,
        "persist_label_identity_index",
        lambda *_args, **_kwargs: events.append("identity") or identity,
    )
    monkeypatch.setattr(runner, "atomic_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "LabelCapabilityJournal", lambda _run_hash: journal
    )
    monkeypatch.setattr(runner, "ManifestLabelDecoder", lambda *_args: object())
    monkeypatch.setattr(
        runner,
        "build_outer_tasks",
        lambda **_kwargs: events.append("tasks") or ((), {}),
    )
    monkeypatch.setattr(
        runner,
        "run_outer_center_tasks",
        lambda *_args, **_kwargs: events.append("outer_workers") or (),
    )
    monkeypatch.setattr(
        runner,
        "collect_outer_results",
        lambda *_args, **_kwargs: events.append("collect") or outer,
    )
    monkeypatch.setattr(
        runner,
        "persist_preterminal_admission_abort",
        lambda *_args: events.append("admission_gate"),
    )
    monkeypatch.setattr(
        runner, "assemble_method_probabilities", lambda _payloads: ({}, {})
    )
    monkeypatch.setattr(
        runner, "build_decision_seal_hash", lambda *_args: _digest("decision")
    )
    monkeypatch.setattr(
        runner,
        "persist_preterminal_bundle",
        lambda *_args, **_kwargs: events.append("preterminal")
        or {
            "aggregate_seal_hash": _digest("preterminal"),
            "label_capability_journal_hash": _digest("journal"),
        },
    )
    monkeypatch.setattr(
        runner,
        "require_two_fresh_process_attestations",
        lambda *_args, **kwargs: events.append(f"attest_{kwargs['phase']}")
        or {"attestation_hash": _digest(kwargs["phase"])},
    )
    monkeypatch.setattr(
        runner,
        "score_terminal_phase",
        lambda *_args, **_kwargs: events.append("terminal") or terminal,
    )
    monkeypatch.setattr(
        runner,
        "finalize_terminal_run",
        lambda *_args, **_kwargs: events.append("finalize"),
    )
    monkeypatch.setattr(
        runner,
        "read_run_state",
        lambda _root: {"status": "COMPLETE", "state_hash": _digest("complete")},
    )
    monkeypatch.setattr(
        runner,
        "record_authorization_outcome",
        lambda _claim, **kwargs: events.append(f"outcome_{kwargs['status']}") or {},
    )
    return events


def test_runner_orders_read_only_checks_claim_science_and_terminal_phases(
    tmp_path, monkeypatch
) -> None:
    runner = importlib.import_module(MODULE)
    config = _config(tmp_path)
    events = _patch_success_lifecycle(monkeypatch, config)

    assert runner.run_scale_bp_v2(config, use_processes=False) == config.artifact_root
    required_order = (
        "protocol",
        "admission",
        "callback",
        "frame",
        "inputs",
        "firewall",
        "host",
        "claim",
        "create",
        "physical",
        "outer_workers",
        "admission_gate",
        "preterminal",
        "attest_preterminal",
        "terminal",
        "finalize",
        "outcome_COMPLETE",
    )
    positions = [events.index(value) for value in required_order]
    assert positions == sorted(positions)


def test_runner_failure_after_claim_records_external_exhaustion(
    tmp_path, monkeypatch
) -> None:
    runner = importlib.import_module(MODULE)
    config = _config(tmp_path)
    events = _patch_success_lifecycle(monkeypatch, config)

    def fail_creation(*_args, **_kwargs):
        events.append("create_failed")
        raise RuntimeError("synthetic create failure")

    monkeypatch.setattr(runner, "create_single_use_run", fail_creation)
    with pytest.raises(RuntimeError, match="synthetic create failure"):
        runner.run_scale_bp_v2(config, use_processes=False)
    assert events.index("claim") < events.index("create_failed")
    assert events[-1] == "outcome_FAILED_EXHAUSTED"
