from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis import oe_ppur_v4
from midogpp_thesis.cvae.diagnostics import (
    fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4
    as router_v4,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4 import (
    runner,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.admission import (
    LaunchAuthority,
    SealedEnvelopeAdmission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.config import (
    ResolvedV4ConfigBundle,
    build_workspace_sealed_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution import (
    preparation_commit,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.authority import (
    LoadedExecutionLaunchAuthority,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.sealed_replay import (
    SealedExecutionReplay,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_claim import (
    AuthorizationLeaseClaim,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.run_admission import (
    SevenInputRunAdmission,
)
from midogpp_thesis.cvae.protocol import ProtocolError


@pytest.mark.parametrize(
    "argv, message",
    (
        (
            ("run", "--repository-root", "/work/repo"),
            "sealed preflight, separate launch authority",
        ),
        (
            (
                "run",
                "--repository-root",
                "/work/repo",
                "--preflight-receipt",
                "/tmp/preflight.json",
                "--authority",
                "/tmp/authority.json",
                "--confirm",
                "RUN",
            ),
            "RUN_TERMINAL_CONSUMED_TEST",
        ),
    ),
)
def test_real_launch_cli_fails_closed_without_exact_inputs_and_confirmation(
    argv: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        oe_ppur_v4.main(argv)


def test_real_launch_cli_dispatches_only_after_exact_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}
    artifact = tmp_path / "artifact"

    def fake_run(repository_root, **kwargs):
        observed["repository_root"] = repository_root
        observed.update(kwargs)
        return artifact

    monkeypatch.setattr(runner, "run_real_oe_ppur_v4", fake_run)
    receipt = tmp_path / "preflight.json"
    authority = tmp_path / "authority.json"

    assert (
        oe_ppur_v4.main(
            (
                "run",
                "--repository-root",
                str(tmp_path),
                "--preflight-receipt",
                str(receipt),
                "--authority",
                str(authority),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--host-id",
                "workstation",
                "--confirm",
                "RUN_TERMINAL_CONSUMED_TEST",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "COMPLETE"
    assert payload["publication_status"] == "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    assert observed == {
        "repository_root": tmp_path,
        "preflight_receipt_path": receipt,
        "launch_authority_path": authority,
        "scratch_root": tmp_path / "scratch",
        "host_id": "workstation",
    }


def test_legacy_preparation_authority_cannot_open_real_launch_edge(
    tmp_path: Path,
) -> None:
    plan_hash = "1" * 64
    amendment_hash = "2" * 64
    envelope_hash = "3" * 64
    config = build_workspace_sealed_config(
        workspace_plan_sha256=plan_hash,
        authorization_amendment_sha256=amendment_hash,
    )
    admission = SealedEnvelopeAdmission(
        config=config,
        workspace_snapshot_sha256="4" * 64,
        workspace_plan_sha256=plan_hash,
        authorization_amendment_sha256=amendment_hash,
        final_envelope_sha256=envelope_hash,
        direct_input_artifact_ids=DIRECT_INPUT_ARTIFACT_IDS,
        resolved_paths=tuple(tmp_path / f"input-{index}" for index in range(7)),
        topology_contract_sha256="5" * 64,
    )
    legacy = LaunchAuthority(
        experiment_id=EXPERIMENT_ID,
        workspace_plan_sha256=plan_hash,
        authorization_amendment_sha256=amendment_hash,
        final_envelope_sha256=envelope_hash,
        authorization_phrase_sha256="6" * 64,
    )

    with pytest.raises(ProtocolError, match="legacy preparation authority"):
        runner.run_oe_ppur_v4(
            config,
            admission=admission,
            launch_authority=legacy,
        )


def _unchecked_instance(cls, **attributes):
    value = object.__new__(cls)
    for name, item in attributes.items():
        object.__setattr__(value, name, item)
    return value


def test_preparation_commit_validates_lease_before_exclusive_marker_last_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts" / "router" / "v4"
    root.parent.mkdir(parents=True)
    topology = SimpleNamespace(
        output_root=root,
        resolved_config_path=root / "config.resolved.yaml",
        input_manifest_path=root / "provenance" / "input_artifacts.json",
        envelope_path=root / "preparation" / "final_authorization_envelope.json",
        commit_marker_path=root / "COMMITTED",
    )
    realized = SimpleNamespace(
        resolved_config_raw=b"config: sealed\n",
        input_manifest_raw=b'{"inputs":[]}\n',
    )
    candidate = SimpleNamespace(
        plan=SimpleNamespace(topology=topology),
        envelope=SimpleNamespace(realized_templates=realized),
        envelope_raw=b'{"envelope":"sealed"}\n',
        commit_marker_raw=b'{"status":"COMMITTED"}\n',
    )
    replay = _unchecked_instance(
        SealedExecutionReplay,
        context=SimpleNamespace(candidate=candidate),
    )
    monkeypatch.setattr(
        SealedExecutionReplay,
        "to_payload",
        lambda self: {"schema_version": "test_sealed_replay_v1"},
    )
    bundle = _unchecked_instance(
        ResolvedV4ConfigBundle,
        artifact_root=root,
        source_path=topology.resolved_config_path,
        input_manifest_path=topology.input_manifest_path,
        final_envelope_path=topology.envelope_path,
    )
    launch_file_hash = "7" * 64
    loaded = _unchecked_instance(
        LoadedExecutionLaunchAuthority,
        file_sha256=launch_file_hash,
        authority=SimpleNamespace(
            canonical_file_bytes=lambda: b'{"authority":"sealed"}\n'
        ),
    )
    run_admission = _unchecked_instance(
        SevenInputRunAdmission,
        artifact_root=root,
        execution_launch_authority_sha256=launch_file_hash,
        receipt_hash="8" * 64,
    )
    lease = AuthorizationLeaseClaim(tmp_path / "lease", {}, "9" * 64)
    claim = SimpleNamespace(
        payload={
            "seven_input_admission_hash": run_admission.receipt_hash,
            "execution_launch_authority_sha256": launch_file_hash,
        },
        claim_hash="9" * 64,
    )
    events: list[str] = []

    def validate_lease(value):
        assert value is lease
        assert not root.exists()
        events.append("lease-validated")
        return claim

    original_write = preparation_commit._write_exclusive

    def recording_write(path: Path, raw: bytes) -> None:
        events.append(path.relative_to(root).as_posix())
        original_write(path, raw)

    monkeypatch.setattr(preparation_commit, "validate_authorization_lease", validate_lease)
    monkeypatch.setattr(
        preparation_commit,
        "validate_loaded_launch_authority",
        lambda replay_value, authority_value: authority_value,
    )
    monkeypatch.setattr(preparation_commit, "_write_exclusive", recording_write)

    receipt = preparation_commit.commit_prepared_output(
        bundle,
        replay=replay,
        launch_authority=loaded,
        run_admission=run_admission,
        lease=lease,
    )

    assert events[0] == "lease-validated"
    assert events[-1] == "COMMITTED"
    assert topology.commit_marker_path.read_bytes() == candidate.commit_marker_raw
    assert receipt.to_payload()["commit_marker_written_last"] is True
    assert receipt.to_payload()["preparation_commit_is_scientific_complete"] is False
    assert "O_EXCL" in original_write.__code__.co_names
    assert "O_NOFOLLOW" in original_write.__code__.co_consts

    protected = topology.resolved_config_path
    with pytest.raises(FileExistsError):
        original_write(protected, b"overwritten\n")
    assert protected.read_bytes() == realized.resolved_config_raw


def test_v4_package_has_no_recursive_predecessor_runtime_imports() -> None:
    package_root = Path(router_v4.__file__).parent
    forbidden = (
        "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3",
        "oe_ppur_v3_preparation",
    )
    violations: list[str] = []

    for source_path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(alias.name for alias in node.names)
            if any(fragment in name for name in imported for fragment in forbidden):
                violations.append(
                    f"{source_path.relative_to(package_root)}:{node.lineno}"
                )

    assert violations == []
