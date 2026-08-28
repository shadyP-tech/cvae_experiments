from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4 import (
    experiment_contracts,
    identity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution import (
    workspace_inputs,
    workspace_preparation_authority as sceptre_authority,
)
from midogpp_thesis.workspace import runtime
from midogpp_thesis.workspace.preparation_authority import (
    SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
    preparation_authority_registration_error,
)
from midogpp_thesis.workspace.runtime import (
    ArtifactEntry,
    ExperimentEntry,
    FileHashExpectation,
    MidogppWorkspace,
    PreparedRun,
    WorkspaceError,
)


def test_registry_field_is_closed_world_and_v4_requires_exact_gate() -> None:
    raw = {
        "experiments": [
            {
                "experiment_id": "example",
                "runner": {
                    "argv": ["python"],
                    "preparation_authority_gate": "package.module:callable",
                },
            }
        ]
    }
    with pytest.raises(
        WorkspaceError,
        match="unknown runner.preparation_authority_gate",
    ):
        MidogppWorkspace._parse_experiments(raw)  # noqa: SLF001

    assert preparation_authority_registration_error(
        None,
        experiment_id=identity.EXPERIMENT_ID,
    ) == (
        f"{identity.EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{SCEPTRE_V4_EXECUTION_AMENDMENT_GATE!r}"
    )


@pytest.mark.parametrize("require_inputs", [True, False])
def test_invalid_amendment_blocks_prepare_before_render_or_protected_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_inputs: bool,
) -> None:
    workspace, experiment, _amendment = _workspace_fixture(
        tmp_path,
        amendment_payload={"wrong": True},
    )
    events: list[str] = []
    _patch_adapter(
        monkeypatch,
        experiment=experiment,
        amendment_payload={"expected": True},
        events=events,
    )
    monkeypatch.setattr(workspace, "validate", lambda: events.append("schema"))
    original_resolver = workspace._resolve_preparation_authority_member  # noqa: SLF001

    def resolve_authority_member(artifact_id: str, relative: str):
        events.append("authority_member")
        return original_resolver(artifact_id, relative)

    monkeypatch.setattr(
        workspace,
        "_resolve_preparation_authority_member",
        resolve_authority_member,
    )
    monkeypatch.setattr(
        workspace,
        "resolve_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal/protected artifact resolution ran before authority")
        ),
    )
    monkeypatch.setattr(
        workspace,
        "_artifact_file_integrity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal/protected artifact hashing ran before authority")
        ),
    )
    monkeypatch.setattr(
        workspace,
        "_render_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("workspace render ran before authority")
        ),
    )

    with pytest.raises(
        WorkspaceError,
        match="consumer-specific execution amendment drifted",
    ):
        workspace.prepare(experiment.experiment_id, require_inputs=require_inputs)
    assert events == ["schema", "config", "authority_member"]


def test_invalid_amendment_blocks_run_before_output_or_protected_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, experiment, _amendment = _workspace_fixture(
        tmp_path,
        amendment_payload={"wrong": True},
    )
    _patch_adapter(
        monkeypatch,
        experiment=experiment,
        amendment_payload={"expected": True},
        events=[],
    )
    monkeypatch.setattr(workspace, "validate", lambda: None)
    normal_resolutions: list[str] = []
    monkeypatch.setattr(
        workspace,
        "resolve_artifact",
        lambda artifact_id, **_kwargs: normal_resolutions.append(artifact_id)
        or tmp_path,
    )

    with pytest.raises(
        WorkspaceError,
        match="consumer-specific execution amendment drifted",
    ):
        workspace.run(experiment.experiment_id)
    assert normal_resolutions == []


def test_valid_amendment_precedes_render_and_ordinary_experiments_are_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_payload = {"schema_version": "exact-test-amendment"}
    workspace, experiment, _amendment = _workspace_fixture(
        tmp_path,
        amendment_payload=expected_payload,
    )
    events: list[str] = []
    _patch_adapter(
        monkeypatch,
        experiment=experiment,
        amendment_payload=expected_payload,
        events=events,
    )
    monkeypatch.setattr(workspace, "validate", lambda: events.append("schema"))
    rendered = _rendered_run(tmp_path, experiment)
    monkeypatch.setattr(
        workspace,
        "_render_run",
        lambda *_args, **_kwargs: events.append("render") or rendered,
    )

    prepared = workspace.prepare(experiment.experiment_id, require_inputs=True)
    assert prepared == rendered.prepared
    assert events == ["schema", "config", "render"]

    ordinary = _experiment(
        experiment_id="ordinary.experiment",
        preparation_authority_gate=None,
        config_path=None,
        input_artifact_ids=(),
    )
    workspace.experiments = {ordinary.experiment_id: ordinary}
    ordinary_rendered = _rendered_run(tmp_path / "ordinary", ordinary)
    events.clear()
    monkeypatch.setattr(
        workspace,
        "_render_run",
        lambda *_args, **_kwargs: events.append("render") or ordinary_rendered,
    )
    monkeypatch.setattr(
        runtime,
        "enforce_preparation_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an ordinary experiment invoked an authority gate")
        ),
    )

    workspace.prepare(ordinary.experiment_id, require_inputs=True)
    assert events == ["schema", "render"]


def test_pre_render_receipt_rejects_config_toctou_before_any_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_payload = {"schema_version": "exact-test-amendment"}
    workspace, experiment, _amendment = _workspace_fixture(
        tmp_path,
        amendment_payload=expected_payload,
    )
    _patch_adapter(
        monkeypatch,
        experiment=experiment,
        amendment_payload=expected_payload,
        events=[],
    )
    receipt = workspace._enforce_preparation_authority(  # noqa: SLF001
        experiment,
    )
    assert receipt is not None
    (tmp_path / "config.yaml").write_text("inputs: {}\n", encoding="utf-8")
    normal_resolutions: list[str] = []
    monkeypatch.setattr(
        workspace,
        "resolve_artifact",
        lambda artifact_id, **_kwargs: normal_resolutions.append(artifact_id)
        or tmp_path,
    )

    with pytest.raises(WorkspaceError, match="authority bytes changed"):
        workspace._render_run(  # noqa: SLF001
            experiment.experiment_id,
            require_inputs=True,
            validate_workspace=False,
            include_all_declared_inputs=False,
            authority_receipt=receipt,
        )
    assert normal_resolutions == []


def test_v4_provenance_replay_carries_the_authority_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = object()
    experiment = object()
    payload = {"input_artifacts": []}
    events: list[str] = []

    class FakeWorkspace:
        def validate(self) -> None:
            events.append("schema")

        def get_experiment(self, experiment_id: str):
            assert experiment_id == identity.EXPERIMENT_ID
            events.append("experiment")
            return experiment

        def _enforce_preparation_authority(self, value):
            assert value is experiment
            events.append("authority")
            return receipt

        def _render_run(self, experiment_id: str, **kwargs):
            assert experiment_id == identity.EXPERIMENT_ID
            assert kwargs == {
                "require_inputs": True,
                "validate_workspace": False,
                "include_all_declared_inputs": True,
                "authority_receipt": receipt,
            }
            events.append("render")
            return SimpleNamespace(input_manifest=payload)

    monkeypatch.setattr(
        workspace_inputs.MidogppWorkspace,
        "load",
        lambda: FakeWorkspace(),
    )
    workspace_inputs._replay_workspace_manifest(payload)  # noqa: SLF001
    assert events == ["schema", "experiment", "authority", "render"]


def test_adapter_rejects_wrong_consumer_before_authority_member_resolution(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("inputs: {}\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    with pytest.raises(
        sceptre_authority.SceptreV4WorkspaceAuthorityError,
        match="binding drifted",
    ):
        sceptre_authority.validate_workspace_preparation_authority(
            repo_root=tmp_path,
            experiment_id="wrong.consumer",
            config_path="config.yaml",
            input_artifact_ids=experiment_contracts.INPUT_ARTIFACT_IDS,
            resolve_authority_member=lambda artifact_id, relative: calls.append(
                (artifact_id, relative)
            ),
        )
    assert calls == []


def _workspace_fixture(
    tmp_path: Path,
    *,
    amendment_payload: dict[str, object],
) -> tuple[MidogppWorkspace, ExperimentEntry, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        "inputs:\n"
        "  execution_amendment_path: "
        f"artifact://{experiment_contracts.EXECUTION_AMENDMENT_ARTIFACT_ID}/"
        f"{experiment_contracts.EXECUTION_AMENDMENT_FILENAME}\n",
        encoding="utf-8",
    )
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    amendment = authority_root / experiment_contracts.EXECUTION_AMENDMENT_FILENAME
    amendment.write_text(
        json.dumps(amendment_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(amendment.read_bytes()).hexdigest()
    artifact = ArtifactEntry(
        artifact_id=experiment_contracts.EXECUTION_AMENDMENT_ARTIFACT_ID,
        stage="dataset_contract",
        physical_path=str(authority_root),
        canonical_path=None,
        migration="hash_verified_move",
        availability="local_and_workstation",
        evidence_label="TEST_AUTHORITY",
        claim_scope="dataset_contract_and_split_provenance",
        semantic_identities={},
        required_files=(experiment_contracts.EXECUTION_AMENDMENT_FILENAME,),
        authoritative_files=(),
        expected_file_hashes={
            experiment_contracts.EXECUTION_AMENDMENT_FILENAME: FileHashExpectation(
                algorithm="sha256",
                digest=digest,
            )
        },
        forbidden_reuse=(),
        may_feed_recipe_selection=False,
        may_feed_deployable_selection=False,
    )
    experiment = _experiment(
        experiment_id=identity.EXPERIMENT_ID,
        preparation_authority_gate=SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
        config_path="config.yaml",
        input_artifact_ids=experiment_contracts.INPUT_ARTIFACT_IDS,
    )
    workspace = object.__new__(MidogppWorkspace)
    workspace.repo_root = tmp_path.resolve()
    workspace.experiments = {experiment.experiment_id: experiment}
    workspace.artifacts = {artifact.artifact_id: artifact}
    return workspace, experiment, amendment


def _patch_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment: ExperimentEntry,
    amendment_payload: dict[str, object],
    events: list[str],
) -> None:
    assert experiment.input_artifact_ids[-1] == (
        experiment_contracts.EXECUTION_AMENDMENT_ARTIFACT_ID
    )

    def load_config(_path: Path) -> SimpleNamespace:
        events.append("config")
        authority_root = _path.parent / "authority"
        digest = hashlib.sha256(
            (authority_root / experiment_contracts.EXECUTION_AMENDMENT_FILENAME).read_bytes()
        ).hexdigest()
        return SimpleNamespace(
            experiment_id=identity.EXPERIMENT_ID,
            input_artifact_ids=experiment_contracts.INPUT_ARTIFACT_IDS,
            execution_authorized=True,
            expected_execution_amendment_sha256=digest,
        )

    monkeypatch.setattr(sceptre_authority, "load_config", load_config)
    monkeypatch.setattr(
        sceptre_authority,
        "canonical_execution_amendment_payload",
        lambda _config: amendment_payload,
    )


def _experiment(
    *,
    experiment_id: str,
    preparation_authority_gate: str | None,
    config_path: str | None,
    input_artifact_ids: tuple[str, ...],
) -> ExperimentEntry:
    return ExperimentEntry(
        experiment_id=experiment_id,
        stage="90_oracles_and_diagnostics",
        status="diagnostic",
        claim_scope="diagnostic_only",
        output_artifact_id="output",
        config_path=config_path,
        runner_argv=("python",),
        runner_env={},
        preparation_authority_gate=preparation_authority_gate,
        run_recovery_strategy=None,
        input_artifact_ids=input_artifact_ids,
        input_claim_scope_exceptions={},
        notes=(),
    )


def _rendered_run(tmp_path: Path, experiment: ExperimentEntry) -> runtime._RenderedRun:
    artifact_root = tmp_path / "output"
    prepared = PreparedRun(
        experiment=experiment,
        artifact_root=artifact_root,
        resolved_config_path=artifact_root / "config.resolved.yaml",
        input_manifest_path=artifact_root / "provenance/input_artifacts.json",
        argv=("python",),
        env={},
    )
    return runtime._RenderedRun(
        prepared=prepared,
        resolved_config_content="schema_version: test\n",
        input_manifest_content="{}\n",
        input_manifest={},
    )
