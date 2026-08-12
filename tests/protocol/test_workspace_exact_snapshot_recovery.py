from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_contracts import (
    FAILED_INFERENCE_STATE,
    POST_TEST_SEAL_RECOVERY_FILES,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace import runtime as workspace_runtime
from midogpp_thesis.workspace.recovery import (
    EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
    RecoveryContractError,
    registered_recovery_state_status,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only.v1"
)
CONFIG_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only_v1.yaml"
)
OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_"
    "fixed_bank_disagreement_regret_prediction_only_v1"
)
STRATEGY = "exact_existing_snapshot_disagreement_regret_prediction_only_v1"
REVISION_A = {
    "repository_revision": "a" * 40,
    "repository_dirty": True,
    "repository_status_hash": "1" * 64,
}
REVISION_B = {
    "repository_revision": "b" * 40,
    "repository_dirty": False,
    "repository_status_hash": "2" * 64,
}


def test_repository_registers_only_the_three_exact_recovery_entries() -> None:
    workspace = MidogppWorkspace.load()
    registered = [
        experiment
        for experiment in workspace.experiments.values()
        if experiment.run_recovery_strategy is not None
    ]

    assert {
        experiment.experiment_id: experiment.run_recovery_strategy
        for experiment in registered
    } == {
        EXPERIMENT_ID: STRATEGY,
        (
            "midogpp.oracle.uniform_b_v2_consumed_test_"
            "utility_aligned_target_static_endpoint_router.v1"
        ): (
            "exact_existing_snapshot_utility_aligned_"
            "consumed_test_endpoint_router_v1"
        ),
        (
            "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
            "labeled_support_case_conditional_flip_router.v1"
        ): (
            "exact_existing_snapshot_fixed_bank_labeled_support_"
            "case_conditional_flip_router_v1"
        ),
    }


def test_workspace_validate_rejects_unknown_or_misbound_recovery_strategy() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    target = _registry_experiment(registry, EXPERIMENT_ID)
    target["runner"]["run_recovery_strategy"] = "unknown_existing_snapshot_strategy"
    unknown = _workspace_from_payloads(source, registry=registry)

    with pytest.raises(WorkspaceError, match="unknown runner.run_recovery_strategy"):
        unknown.validate()

    registry = deepcopy(source.registry_payload)
    other = _registry_experiment(
        registry, "midogpp.real_feature.eligible_tuned_predict_reference.v2"
    )
    other["runner"]["run_recovery_strategy"] = STRATEGY
    misbound = _workspace_from_payloads(source, registry=registry)

    with pytest.raises(WorkspaceError, match="requires exact experiment_id"):
        misbound.validate()


@pytest.mark.parametrize(
    ("field", "expected_label"),
    (
        ("config", "config_path"),
        ("argv", "runner.argv"),
        ("environment", "runner.environment"),
        ("inputs", "input_artifact_ids"),
    ),
)
def test_recovery_registration_pins_current_config_runner_and_inputs(
    field: str,
    expected_label: str,
) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    target = _registry_experiment(registry, EXPERIMENT_ID)
    if field == "config":
        target["config_path"] = "experiments/midogpp/other.yaml"
    elif field == "argv":
        target["runner"]["argv"].append("--drifted-runner")
    elif field == "environment":
        target["runner"]["environment"]["OMP_NUM_THREADS"] = "2"
    elif field == "inputs":
        target["input_artifact_ids"] = list(reversed(target["input_artifact_ids"]))
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(field)
    drifted = _workspace_from_payloads(source, registry=registry)

    with pytest.raises(
        WorkspaceError,
        match=rf"recovery strategy .* requires exact {expected_label}",
    ):
        drifted.validate()


def test_recovery_registration_pins_current_output_root() -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    output = next(
        entry for entry in catalog["artifacts"] if entry["artifact_id"] == OUTPUT_ID
    )
    output["canonical_path"] = (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "drifted_fixed_bank_prediction_only/v1"
    )
    drifted = MidogppWorkspace(
        repo_root=source.repo_root,
        registry=source.registry_payload,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )

    with pytest.raises(
        WorkspaceError,
        match=r"recovery strategy .* requires exact output canonical_path",
    ):
        drifted.validate()


def test_exact_recovery_preserves_revision_a_snapshots_under_current_revision_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    config_before = prepared.resolved_config_path.read_bytes()
    manifest_before = prepared.input_manifest_path.read_bytes()
    state.update(REVISION_B)
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def failed_runner(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> SimpleNamespace:
        assert check is False
        assert prepared.resolved_config_path.read_bytes() == config_before
        assert prepared.input_manifest_path.read_bytes() == manifest_before
        calls.append((argv, cwd, env))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(workspace_runtime.subprocess, "run", failed_runner)

    assert workspace.run(EXPERIMENT_ID) == 17
    assert len(calls) == 1
    argv, cwd, env = calls[0]
    assert tuple(argv) == prepared.argv
    assert cwd == tmp_path
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert prepared.resolved_config_path.read_bytes() == config_before
    assert prepared.input_manifest_path.read_bytes() == manifest_before
    preserved_manifest = json.loads(manifest_before)
    assert {
        key: preserved_manifest[key] for key in REVISION_A
    } == REVISION_A


def test_exact_recovery_rejects_force_before_any_snapshot_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )
    state.update(REVISION_B)

    def forbidden_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(workspace_runtime.subprocess, "run", forbidden_runner)

    with pytest.raises(WorkspaceError, match="rejects --force"):
        workspace.run(EXPERIMENT_ID, force=True)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


def test_exact_recovery_rejects_extra_runner_arguments_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )
    state.update(REVISION_B)

    def forbidden_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(workspace_runtime.subprocess, "run", forbidden_runner)

    with pytest.raises(WorkspaceError, match="rejects extra runner arguments"):
        workspace.run(EXPERIMENT_ID, extra_args=("--unregistered",))

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


@pytest.mark.parametrize("force", [False, True])
def test_recovery_inventory_drift_fails_closed_without_preparing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    unexpected = prepared.artifact_root / "manifests/unregistered-recovery-member.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )
    state.update(REVISION_B)

    with pytest.raises(ProtocolError, match="recovery boundary drifted"):
        workspace.run(EXPERIMENT_ID, force=force)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


def test_non_recovery_state_uses_normal_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    config_before = prepared.resolved_config_path.read_bytes()
    manifest_before = prepared.input_manifest_path.read_bytes()
    state.update(REVISION_B)

    with pytest.raises(WorkspaceError, match="Refusing to overwrite changed run snapshot"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == config_before
    assert prepared.input_manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize(
    ("status", "force"),
    (
        ("FAILED", False),
        ("FAILED", True),
        ("RUNNING", False),
        ("RUNNING", True),
    ),
)
def test_registered_recovery_refuses_unrecognized_active_state_before_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    force: bool,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    state_path = prepared.artifact_root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "midogpp_disagreement_regret_prediction_only_run_state_v1"
                ),
                "status": status,
                "phase": "UNREGISTERED_FAILURE_BOUNDARY",
                "error": "RuntimeError: unregistered boundary",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )
    state.update(REVISION_B)

    with pytest.raises(WorkspaceError, match="recovery state is unrecognized"):
        workspace.run(EXPERIMENT_ID, force=force)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


def test_registered_recovery_refuses_broken_state_symlink_before_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    state_path = prepared.artifact_root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.symlink_to(state_path.parent / "missing-run-state.json")
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )

    with pytest.raises(WorkspaceError, match="recovery state is unsafe"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


def test_registered_recovery_refuses_recognized_state_symlink_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    state_path = prepared.artifact_root / "reports/run_state.json"
    external_state = tmp_path / "external-run-state.json"
    external_state.write_bytes(state_path.read_bytes())
    state_path.unlink()
    state_path.symlink_to(external_state)
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )
    state.update(REVISION_B)

    with pytest.raises(WorkspaceError, match="recovery state is unsafe"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


@pytest.mark.parametrize(
    "state_payload",
    (
        {"status": "COMPLETE"},
        {
            "schema_version": (
                "midogpp_disagreement_regret_prediction_only_run_state_v1"
            ),
            "status": "COMPLETE",
            "phase": "NOT_COMPLETE",
        },
        {
            "schema_version": (
                "midogpp_disagreement_regret_prediction_only_run_state_v1"
            ),
            "status": "COMPLETE",
            "phase": "COMPLETE",
            "error": "RuntimeError: contradictory complete state",
        },
    ),
)
@pytest.mark.parametrize("force", [False, True])
def test_registered_recovery_refuses_malformed_complete_state_before_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_payload: Mapping[str, object],
    force: bool,
) -> None:
    workspace, _state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    state_path = prepared.artifact_root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = (
        prepared.resolved_config_path.read_bytes(),
        prepared.input_manifest_path.read_bytes(),
    )

    with pytest.raises(WorkspaceError, match="recovery state is malformed"):
        workspace.run(EXPERIMENT_ID, force=force)

    assert prepared.resolved_config_path.read_bytes() == before[0]
    assert prepared.input_manifest_path.read_bytes() == before[1]


def test_registered_endpoint_recovery_status_uses_endpoint_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "endpoint-root"
    state_path = root / "reports/run_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
                "status": "FAILED",
                "phase": "UNREGISTERED_ENDPOINT_FAILURE",
                "error": "RuntimeError: endpoint boundary",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        registered_recovery_state_status(
            EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
            root,
        )
        == "FAILED"
    )

    state_path.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
                "status": [],
                "phase": "UNREGISTERED_ENDPOINT_FAILURE",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RecoveryContractError, match="recovery state is malformed"):
        registered_recovery_state_status(
            EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1,
            root,
        )


def test_exact_recovery_rehashes_every_current_input_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    config_before = prepared.resolved_config_path.read_bytes()
    manifest_before = prepared.input_manifest_path.read_bytes()
    state.update(REVISION_B)
    inputs[-1].write_bytes(b"drifted-current-input")

    with pytest.raises(WorkspaceError, match="file hash mismatch"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == config_before
    assert prepared.input_manifest_path.read_bytes() == manifest_before


def test_exact_recovery_rejects_current_resolved_config_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    config_before = prepared.resolved_config_path.read_bytes()
    manifest_before = prepared.input_manifest_path.read_bytes()
    source_config = tmp_path / CONFIG_PATH
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    payload["protocol"] = {"unexpected_current_change": True}
    source_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    state.update(REVISION_B)

    with pytest.raises(WorkspaceError, match="resolved config does not exactly match"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == config_before
    assert prepared.input_manifest_path.read_bytes() == manifest_before


def test_exact_recovery_rejects_manifest_drift_outside_top_level_repository_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    preserved = json.loads(prepared.input_manifest_path.read_text(encoding="utf-8"))
    preserved["unexpected_field"] = "not-an-allowed-revision-difference"
    prepared.input_manifest_path.write_text(
        json.dumps(preserved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_before = prepared.resolved_config_path.read_bytes()
    manifest_before = prepared.input_manifest_path.read_bytes()
    state.update(REVISION_B)

    with pytest.raises(WorkspaceError, match="outside the three allowed top-level"):
        workspace.run(EXPERIMENT_ID)

    assert prepared.resolved_config_path.read_bytes() == config_before
    assert prepared.input_manifest_path.read_bytes() == manifest_before


def test_recovery_detects_snapshot_mutation_even_when_runner_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, state, _inputs = _build_exact_workspace(tmp_path, monkeypatch)
    prepared = workspace.prepare(EXPERIMENT_ID)
    _write_exact_failed_inventory(prepared.artifact_root)
    state.update(REVISION_B)

    def mutating_failed_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        prepared.resolved_config_path.write_text("mutated\n", encoding="utf-8")
        return SimpleNamespace(returncode=19)

    monkeypatch.setattr(
        workspace_runtime.subprocess,
        "run",
        mutating_failed_runner,
    )

    with pytest.raises(WorkspaceError, match="modified an immutable run snapshot"):
        workspace.run(EXPERIMENT_ID)


def _build_exact_workspace(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MidogppWorkspace, dict[str, object], list[Path]]:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    target = deepcopy(_registry_experiment(registry, EXPERIMENT_ID))
    registry["experiments"] = [target]
    input_ids = list(target["input_artifact_ids"])
    retained_ids = {*input_ids, OUTPUT_ID}
    catalog = deepcopy(source.catalog_payload)
    catalog["artifacts"] = [
        entry
        for entry in catalog["artifacts"]
        if entry["artifact_id"] in retained_ids
    ]

    input_paths: list[Path] = []
    for entry in catalog["artifacts"]:
        artifact_id = str(entry["artifact_id"])
        if artifact_id == OUTPUT_ID:
            continue
        if entry.get("canonical_path"):
            artifact_root = repo_root / str(entry["canonical_path"])
        else:
            artifact_root = repo_root / "inputs" / artifact_id
            entry["physical_path"] = str(artifact_root)
        payload_path = artifact_root / "payload.bin"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_bytes = f"{artifact_id}-bytes\n".encode("utf-8")
        payload_path.write_bytes(payload_bytes)
        entry["required_files"] = ["payload.bin"]
        entry["authoritative_files"] = []
        entry["expected_file_hashes"] = {
            "payload.bin": {
                "algorithm": "sha256",
                "digest": hashlib.sha256(payload_bytes).hexdigest(),
            }
        }
        input_paths.append(payload_path)

    config = {
        "experiment": {
            "id": EXPERIMENT_ID,
            "artifact_root": f"output://{OUTPUT_ID}",
            "claim_scope": "diagnostic_only",
        },
        "inputs": {
            f"input_{index}": f"artifact://{artifact_id}/payload.bin"
            for index, artifact_id in enumerate(input_ids)
        },
        "protocol": {"contract": "test-exact-existing-snapshot"},
    }
    config_path = repo_root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    workspace = MidogppWorkspace(
        repo_root=repo_root,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
    state: dict[str, object] = dict(REVISION_A)
    monkeypatch.setattr(workspace_runtime, "_git_state", lambda _root: dict(state))
    return workspace, state, input_paths


def _write_exact_failed_inventory(root: Path) -> None:
    for relative in POST_TEST_SEAL_RECOVERY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "reports/run_state.json":
            path.write_text(
                json.dumps(FAILED_INFERENCE_STATE, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif relative not in {
            "config.resolved.yaml",
            "provenance/input_artifacts.json",
        }:
            path.write_bytes(b"sealed-placeholder\n")


def _registry_experiment(registry: dict[str, object], experiment_id: str) -> dict[str, object]:
    return next(
        entry
        for entry in registry["experiments"]  # type: ignore[index]
        if entry["experiment_id"] == experiment_id
    )


def _workspace_from_payloads(
    source: MidogppWorkspace,
    *,
    registry: dict[str, object],
) -> MidogppWorkspace:
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry,
        catalog=source.catalog_payload,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
