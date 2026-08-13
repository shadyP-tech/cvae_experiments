from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import ANY

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (
    fresh_process_validation as fresh,
    persistence,
    runner_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.bundle import (
    CONTENT_INDEX_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _checks(**updates: object) -> dict[str, object]:
    return {
        "schema_version": "fixture_validation_v1",
        "status": "PASS",
        "content_hash": "a" * 64,
        **updates,
    }


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def test_two_sequential_fresh_processes_are_launched_with_bounded_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _checks(reconstructed_count=45)
    launches: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        launches.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_canonical(expected) + "\n",
            stderr="",
        )

    monkeypatch.setattr(fresh.subprocess, "run", fake_run)
    attested = fresh.require_two_fresh_process_validations(
        tmp_path,
        expected_checks=expected,
    )

    assert len(launches) == 2
    for command, kwargs in launches:
        assert command[1:4] == ("-m", fresh.WORKER_MODULE, "--worker")
        assert command[4] == str(tmp_path.resolve())
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            assert environment[name] == "1"
    attestation = attested[fresh.ATTESTATION_KEY]
    assert attestation["fresh_python_process_count"] == 2
    assert attestation["process_launches_sequential"] is True
    assert fresh.verify_attested_validation_checks(
        attested,
        expected_reconstructed_checks=expected,
    ) == attested


def test_fresh_process_replay_disagreement_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _checks(reconstructed_count=45)
    outputs = iter((expected, _checks(reconstructed_count=44)))

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_canonical(next(outputs)) + "\n",
            stderr="",
        )

    monkeypatch.setattr(fresh.subprocess, "run", fake_run)
    with pytest.raises(ProtocolError, match="replays disagreed"):
        fresh.require_two_fresh_process_validations(
            tmp_path,
            expected_checks=expected,
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "match"),
    (
        (17, "", "exit code 17"),
        (0, "not-json\n", "invalid JSON"),
    ),
)
def test_fresh_process_failure_or_invalid_json_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    match: str,
) -> None:
    calls = 0

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr="worker failed" if returncode else "",
        )

    monkeypatch.setattr(fresh.subprocess, "run", fake_run)
    with pytest.raises(ProtocolError, match=match):
        fresh.require_two_fresh_process_validations(
            tmp_path,
            expected_checks=_checks(),
        )
    assert calls == 1


def test_pending_runtime_validation_attests_before_report_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _checks()
    attested = {**base, fresh.ATTESTATION_KEY: {"status": "sentinel"}}
    calls: list[object] = []

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics."
        "fixed_bank_multi_challenger_hierarchical_flip_router.validation."
        "validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle",
        lambda root, **kwargs: calls.append(("parent", kwargs)) or base,
    )
    monkeypatch.setattr(
        fresh,
        "require_two_fresh_process_validations",
        lambda root, *, expected_checks: (
            calls.append(("fresh", dict(expected_checks))) or attested
        ),
    )

    observed = runner_runtime.validate_bundle(
        tmp_path,
        config=SimpleNamespace(),
        allow_pending_validation=True,
    )
    assert observed == attested
    assert calls == [
        ("parent", {"config": ANY, "allow_pending_validation": True}),
        ("fresh", base),
    ]


def test_pending_finalization_audit_is_parent_bound_before_two_fresh_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = {
        "schema_version": "fixture_finalization_audit_v1",
        "finalization_recovery_used": True,
    }
    base = _checks(terminal_finalization_recovery=audit)
    attested = {**base, fresh.ATTESTATION_KEY: {"status": "sentinel"}}
    calls: list[object] = []

    def parent(_root: Path, **kwargs: object) -> Mapping[str, object]:
        calls.append(("parent", kwargs))
        assert kwargs["allow_pending_validation"] is True
        assert kwargs["finalization_recovery_audit"] == audit
        return base

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics."
        "fixed_bank_multi_challenger_hierarchical_flip_router.validation."
        "validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle",
        parent,
    )

    def two_processes(
        _root: Path, *, expected_checks: Mapping[str, object]
    ) -> Mapping[str, object]:
        # `require_two_fresh_process_validations` launches workers without an
        # in-memory audit.  Their complete validator independently derives the
        # same check payload from the exact FAILED state and live clean C.
        calls.append(("two_fresh_processes", dict(expected_checks)))
        assert expected_checks["terminal_finalization_recovery"] == audit
        return attested

    monkeypatch.setattr(
        fresh, "require_two_fresh_process_validations", two_processes
    )

    observed = runner_runtime.validate_bundle(
        tmp_path,
        config=SimpleNamespace(),
        allow_pending_validation=True,
        finalization_recovery_audit=audit,
    )

    assert observed == attested
    assert calls == [
        (
            "parent",
            {
                "config": ANY,
                "allow_pending_validation": True,
                "finalization_recovery_audit": audit,
            },
        ),
        ("two_fresh_processes", base),
    ]


def test_validation_report_persistence_rejects_missing_or_tampered_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _checks()
    payload = _attested_fixture(tmp_path, expected, monkeypatch)

    with pytest.raises(ProtocolError, match="attestation"):
        persistence.persist_validation_report(tmp_path / "missing", expected)
    persistence.persist_validation_report(tmp_path / "valid", payload)
    tampered = json.loads(json.dumps(payload))
    tampered[fresh.ATTESTATION_KEY]["valid_json_payload_count"] = 1
    with pytest.raises(ProtocolError, match="attestation drifted"):
        persistence.persist_validation_report(tmp_path / "tampered", tampered)


def test_completed_attestation_verification_does_not_launch_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _checks()
    payload = _attested_fixture(tmp_path, expected, monkeypatch)
    monkeypatch.setattr(
        fresh,
        "_run_worker",
        lambda *args, **kwargs: pytest.fail(
            "completed validation must not recursively spawn"
        ),
    )

    assert fresh.verify_attested_validation_checks(
        payload,
        expected_reconstructed_checks=expected,
    ) == payload


def test_terminal_checkpoint_recovery_uses_attested_pending_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config = SimpleNamespace(contract_hash="config-hash")
    protocol = SimpleNamespace(contract_hash="protocol-hash")
    checks = {**_checks(), fresh.ATTESTATION_KEY: {"status": "PASS"}}
    events: list[object] = []

    monkeypatch.setattr(
        runner_runtime,
        "write_content_index",
        lambda *args, **kwargs: events.append("content_index"),
    )
    monkeypatch.setattr(
        runner_runtime,
        "enter_cuda_free_cpu_phase",
        lambda: events.append("cpu_phase"),
    )

    def validate(root: Path, **kwargs: object) -> Mapping[str, object]:
        events.append(("validate", kwargs.get("allow_pending_validation")))
        return checks

    monkeypatch.setattr(runner_runtime, "validate_bundle", validate)
    monkeypatch.setattr(
        runner_runtime,
        "persist_validation_report",
        lambda root, value: events.append(("persist", value)),
    )
    monkeypatch.setattr(
        runner_runtime,
        "write_state",
        lambda root, **kwargs: events.append(("state", kwargs)),
    )
    monkeypatch.setattr(
        runner_runtime,
        "assert_completed_binding",
        lambda root, **kwargs: events.append(("binding", kwargs["expected_checks"])),
    )

    assert runner_runtime.recover_if_possible(
        root,
        config=config,
        protocol=protocol,
    ) == root
    assert ("validate", True) in events
    assert ("persist", checks) in events
    assert ("binding", checks) in events


def _attested_fixture(
    root: Path,
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, object]:
    monkeypatch.setattr(
        fresh,
        "_run_worker",
        lambda path: subprocess.CompletedProcess(
            (), 0, stdout=_canonical(expected) + "\n", stderr=""
        ),
    )
    return fresh.require_two_fresh_process_validations(
        root,
        expected_checks=expected,
    )
