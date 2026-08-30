from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.source_seal import (
    build_source_snapshot_payload,
    source_members,
    source_snapshot_identity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import (
    workspace_preparation_authority as harp_authority,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V1_EXECUTION_AMENDMENT_GATE,
    PreparationAuthorityError,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
    validate_preparation_authority_gate_id,
)
from midogpp_thesis.workspace.runtime import (
    ExperimentEntry,
    MidogppWorkspace,
    WorkspaceError,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
)


def _bound_config():
    base = load_config(CONFIG)
    return replace(
        base,
        execution_authorized=True,
        expected_hashes={
            **dict(base.expected_hashes),
            "test_cache_content_sha256": "a" * 64,
            "development_manifest_sha256": "b" * 64,
            "evaluation_manifest_sha256": "c" * 64,
            "parent_ledger_sha256": "d" * 64,
            "execution_amendment_sha256": "e" * 64,
        },
    )


def _source_identity() -> dict[str, object]:
    return {
        "source_snapshot_schema": "test_harp_source_snapshot_v1",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 17,
        "source_snapshot_member_pattern": "test/**/*.py;closure=transitive",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }


def test_source_snapshot_is_deterministic_and_closes_local_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    package_root = source / "midogpp_thesis"
    harp = (
        source
        / "midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v1"
    )
    workspace = source / "midogpp_thesis/workspace"
    shared = source / "midogpp_thesis/shared"
    for directory in (harp, workspace, shared):
        directory.mkdir(parents=True, exist_ok=True)
    (package_root / "__main__.py").write_text(
        "from .cli import main\n",
        encoding="utf-8",
    )
    (package_root / "cli.py").write_text(
        "def main(): return 0\n",
        encoding="utf-8",
    )
    (harp.parent / "cli.py").write_text(
        "from .fixed_bank_harp_router_v1 import runner\n",
        encoding="utf-8",
    )
    (harp / "runner.py").write_text(
        "from midogpp_thesis.shared import dependency\n",
        encoding="utf-8",
    )
    (workspace / "preparation_authority.py").write_text(
        "VALUE = 'gate'\n",
        encoding="utf-8",
    )
    (workspace / "runtime.py").write_text(
        "from .preparation_authority import VALUE\n",
        encoding="utf-8",
    )
    (workspace / "cli.py").write_text(
        "from .runtime import VALUE\n",
        encoding="utf-8",
    )
    dependency = shared / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")

    members = source_members(tmp_path)
    relative = {path.relative_to(source).as_posix() for path in members}
    assert "midogpp_thesis/shared/dependency.py" in relative
    first = dict(build_source_snapshot_payload(tmp_path))
    second = dict(build_source_snapshot_payload(tmp_path))
    assert first == second
    assert dict(source_snapshot_identity(tmp_path)) == {
        key: first[key]
        for key in (
            "source_snapshot_schema",
            "source_snapshot_manifest_sha256",
            "source_snapshot_tree_sha256",
            "source_snapshot_member_count",
            "source_snapshot_member_pattern",
            "source_snapshot_excludes_bytecode_and_cache",
        )
    }

    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    changed = dict(build_source_snapshot_payload(tmp_path))
    assert changed["source_snapshot_manifest_sha256"] != first[
        "source_snapshot_manifest_sha256"
    ]
    assert changed["source_snapshot_tree_sha256"] != first[
        "source_snapshot_tree_sha256"
    ]


def test_canonical_amendment_binds_authorization_science_source_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    monkeypatch.setattr(
        authorization,
        "source_snapshot_identity",
        lambda _root=None: _source_identity(),
    )
    payload = authorization.canonical_execution_amendment_payload(config)
    contract = authorization.scientific_contract_payload(config)

    assert payload["schema_version"] == authorization.EXECUTION_AMENDMENT_SCHEMA
    assert payload["authorization_basis"] == (
        "explicit_user_authorization_2026_08_30_for_harp_v1_terminal_"
        "consumed_test_diagnostic"
    )
    assert payload["authorization_date"] == "2026-08-30"
    assert payload["authorized_input_binding"] == (
        authorization.authorization_input_binding(config)
    )
    assert payload["scientific_contract_hash"] == contract[
        "scientific_contract_hash"
    ]
    registration = authorization.workspace_registration_execution_contract()
    assert contract["workspace_registration_execution_contract"] == registration
    assert payload["workspace_registration_execution_contract_hash"] == (
        registration["workspace_registration_execution_contract_hash"]
    )
    assert payload["source_snapshot_identity"] == _source_identity()

    # The contract cannot depend on the digest of the amendment that embeds it.
    alternate = replace(
        config,
        config_hash="f" * 64,
        expected_hashes={
            **dict(config.expected_hashes),
            "execution_amendment_sha256": "0" * 64,
        },
    )
    assert authorization.scientific_contract_payload(alternate) == contract
    assert authorization.canonical_execution_amendment_payload(alternate) == payload

    typed = authorization.validate_execution_amendment_payload(payload, config)
    assert typed.amendment_hash == payload["amendment_hash"]
    assert typed.scientific_contract_hash == payload["scientific_contract_hash"]
    assert typed.source_snapshot_schema == _source_identity()[
        "source_snapshot_schema"
    ]
    assert typed.source_snapshot_member_count == 17


def test_typed_amendment_validator_rejects_schema_and_json_type_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    monkeypatch.setattr(
        authorization,
        "source_snapshot_identity",
        lambda _root=None: _source_identity(),
    )
    payload = authorization.canonical_execution_amendment_payload(config)
    with pytest.raises(ProtocolError, match="failed authentication"):
        authorization.validate_execution_amendment_payload(
            {**payload, "scientific_contract_hash": "9" * 64},
            config,
        )
    with pytest.raises(ProtocolError, match="failed authentication"):
        authorization.validate_execution_amendment_payload(
            {**payload, "execution_authorized": 1},
            config,
        )


def test_authorization_lease_seals_reconstructed_v2_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lease"
    monkeypatch.setattr(authorization, "lease_path", lambda *_args: root)
    value = authorization.HarpAuthorization(
        amendment_path=tmp_path / "amendment.json",
        amendment_sha256="a" * 64,
        amendment_hash="b" * 64,
        input_binding_hash="c" * 64,
        scientific_contract_hash="d" * 64,
        workspace_registration_execution_contract_hash="9" * 64,
        source_snapshot_schema="source-v1",
        source_snapshot_manifest_sha256="e" * 64,
        source_snapshot_tree_sha256="f" * 64,
        source_snapshot_member_count=23,
    )
    lease = authorization.claim_authorization(value, admission_hash="0" * 64)
    payload = json.loads((lease.root / "lease.json").read_text(encoding="utf-8"))
    assert payload["input_binding_hash"] == value.input_binding_hash
    assert payload["scientific_contract_hash"] == value.scientific_contract_hash
    assert payload["workspace_registration_execution_contract_hash"] == (
        value.workspace_registration_execution_contract_hash
    )
    assert payload["source_snapshot_schema"] == value.source_snapshot_schema
    assert payload["source_snapshot_manifest_sha256"] == (
        value.source_snapshot_manifest_sha256
    )
    assert payload["source_snapshot_tree_sha256"] == (
        value.source_snapshot_tree_sha256
    )
    assert payload["source_snapshot_member_count"] == 23


def test_harp_gate_is_closed_world_and_exactly_bound() -> None:
    assert (
        validate_preparation_authority_gate_id(
            HARP_V1_EXECUTION_AMENDMENT_GATE
        )
        == HARP_V1_EXECUTION_AMENDMENT_GATE
    )
    assert preparation_authority_registration_error(
        None,
        experiment_id=EXPERIMENT_ID,
    ) == (
        f"{EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V1_EXECUTION_AMENDMENT_GATE!r}"
    )
    wrong = "wrong.consumer"
    assert preparation_authority_registration_error(
        HARP_V1_EXECUTION_AMENDMENT_GATE,
        experiment_id=wrong,
    ) == (
        f"{wrong}: runner.preparation_authority_gate "
        f"{HARP_V1_EXECUTION_AMENDMENT_GATE!r} is bound only to {EXPERIMENT_ID}"
    )


def test_pre_render_gate_checks_lease_before_authority_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_config(tmp_path)
    monkeypatch.setattr(
        harp_authority,
        "load_config",
        lambda _path: SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            execution_authorized=True,
            expected_execution_amendment_sha256="a" * 64,
        ),
    )
    lease = tmp_path / "already-claimed"
    lease.mkdir()
    monkeypatch.setattr(harp_authority, "lease_path", lambda _root: lease)
    resolutions: list[tuple[str, str]] = []

    with pytest.raises(
        harp_authority.HarpV1WorkspaceAuthorityError,
        match="authorization is exhausted",
    ):
        harp_authority.validate_workspace_preparation_authority(
            repo_root=tmp_path,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            registration_projection=_registration_projection(),
            resolve_authority_member=lambda artifact_id, relative: resolutions.append(
                (artifact_id, relative)
            ),
        )
    assert resolutions == []


def test_valid_pre_render_gate_authenticates_only_authority_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_config(tmp_path)
    amendment = tmp_path / authorization.EXECUTION_AMENDMENT_FILENAME
    amendment_payload = {"schema_version": "typed-test-amendment"}
    amendment.write_text(
        json.dumps(amendment_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(amendment.read_bytes()).hexdigest()
    events: list[str] = []
    fake_config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        execution_authorized=True,
        expected_execution_amendment_sha256=digest,
    )
    monkeypatch.setattr(
        harp_authority,
        "load_config",
        lambda _path: events.append("config") or fake_config,
    )
    monkeypatch.setattr(
        harp_authority,
        "lease_path",
        lambda _root: tmp_path / "absent-lease",
    )
    monkeypatch.setattr(
        harp_authority,
        "validate_execution_amendment_payload",
        lambda value, config, *, repo_root: events.append("typed_amendment")
        if value == amendment_payload and config is fake_config and repo_root == tmp_path
        else (_ for _ in ()).throw(AssertionError("wrong typed validation inputs")),
    )

    def resolve(artifact_id: str, relative: str):
        events.append("authority_member")
        assert artifact_id == authorization.EXECUTION_AMENDMENT_ARTIFACT_ID
        assert relative == authorization.EXECUTION_AMENDMENT_FILENAME
        return SimpleNamespace(path=amendment, expected_sha256=digest)

    receipt = harp_authority.validate_workspace_preparation_authority(
        repo_root=tmp_path,
        experiment_id=EXPERIMENT_ID,
        config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        registration_projection=_registration_projection(),
        resolve_authority_member=resolve,
    )
    assert events == ["config", "authority_member", "typed_amendment"]
    assert receipt.amendment_path == amendment
    assert receipt.amendment_sha256 == digest
    assert receipt.workspace_registration_contract_hash == (
        authorization.workspace_registration_execution_contract()[
            "workspace_registration_execution_contract_hash"
        ]
    )


@pytest.mark.parametrize(
    "surface",
    ("runner_argv", "runner_environment", "output_id", "output_path", "recovery"),
)
def test_registry_command_output_environment_and_recovery_tamper_fail_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    _write_minimal_config(tmp_path)
    registry_path = tmp_path / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    catalog_path = (
        tmp_path / authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    entry = registry["experiments"][0]
    if surface == "runner_argv":
        entry["runner"]["argv"][-1] = "output://tampered"
    elif surface == "runner_environment":
        entry["runner"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif surface == "output_id":
        entry["output_artifact_id"] = "tampered-output"
    elif surface == "output_path":
        catalog["artifacts"][0]["canonical_path"] = "artifacts/tampered"
    else:
        entry["runner"]["run_recovery_strategy"] = "unsafe-recovery"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    _patch_fake_executable_config(monkeypatch)
    resolutions: list[tuple[str, str]] = []

    with pytest.raises(harp_authority.HarpV1WorkspaceAuthorityError):
        harp_authority.validate_workspace_preparation_authority(
            repo_root=tmp_path,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            registration_projection=_registration_projection(),
            resolve_authority_member=lambda artifact_id, relative: resolutions.append(
                (artifact_id, relative)
            ),
        )
    assert resolutions == []


def test_duplicate_harp_registry_entry_fails_before_authority_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_config(tmp_path)
    registry_path = tmp_path / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        json.loads(json.dumps(registry["experiments"][0]))
    )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    _patch_fake_executable_config(monkeypatch)
    resolutions: list[tuple[str, str]] = []
    with pytest.raises(
        harp_authority.HarpV1WorkspaceAuthorityError,
        match="exactly one experiment entry",
    ):
        harp_authority.validate_workspace_preparation_authority(
            repo_root=tmp_path,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            registration_projection=_registration_projection(),
            resolve_authority_member=lambda artifact_id, relative: resolutions.append(
                (artifact_id, relative)
            ),
        )
    assert resolutions == []


@pytest.mark.parametrize("surface", ("runner_argv", "output_artifact_id"))
def test_tampered_in_memory_entry_fails_before_disk_or_authority_resolution(
    tmp_path: Path,
    surface: str,
) -> None:
    projection = _registration_projection()
    if surface == "runner_argv":
        projection["runner_argv"] = ["python", "-c", "raise SystemExit(0)"]
    else:
        projection["output_artifact_id"] = "sceptre-output"
    resolutions: list[tuple[str, str]] = []
    with pytest.raises(
        harp_authority.HarpV1WorkspaceAuthorityError,
        match="in-memory workspace registration execution contract drifted",
    ):
        harp_authority.validate_workspace_preparation_authority(
            repo_root=tmp_path,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            registration_projection=projection,
            resolve_authority_member=lambda artifact_id, relative: resolutions.append(
                (artifact_id, relative)
            ),
        )
    assert resolutions == []


def test_harp_workspace_rejects_force_and_unlisted_extra_args_before_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validate_preparation_authority_extra_args(
        HARP_V1_EXECUTION_AMENDMENT_GATE,
        (),
    ) == ()
    assert validate_preparation_authority_extra_args(
        HARP_V1_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)
    with pytest.raises(PreparationAuthorityError, match="reject --force"):
        validate_preparation_authority_extra_args(
            HARP_V1_EXECUTION_AMENDMENT_GATE,
            (),
            force=True,
        )

    experiment = ExperimentEntry(
        experiment_id=EXPERIMENT_ID,
        stage="90_oracles_and_diagnostics",
        status="diagnostic",
        claim_scope="diagnostic_only",
        output_artifact_id=authorization.OUTPUT_ARTIFACT_ID,
        config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        runner_argv=authorization.WORKSPACE_RUNNER_ARGV,
        runner_env=dict(authorization.WORKSPACE_RUNNER_ENV),
        preparation_authority_gate=HARP_V1_EXECUTION_AMENDMENT_GATE,
        run_recovery_strategy=None,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        input_claim_scope_exceptions={},
        notes=(),
    )
    workspace = object.__new__(MidogppWorkspace)
    monkeypatch.setattr(workspace, "validate", lambda: None)
    monkeypatch.setattr(workspace, "get_experiment", lambda _value: experiment)
    monkeypatch.setattr(
        workspace,
        "_enforce_preparation_authority",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("authority gate ran after invalid workspace options")
        ),
    )
    with pytest.raises(WorkspaceError, match="reject --force"):
        workspace.prepare(EXPERIMENT_ID, force=True)
    with pytest.raises(WorkspaceError, match="reject --force"):
        workspace.run(EXPERIMENT_ID, force=True)
    with pytest.raises(WorkspaceError, match="accepts only"):
        workspace.run(EXPERIMENT_ID, extra_args=("--not-allowed",))


def test_workspace_validate_preserves_planned_harp_registration() -> None:
    """The checked-in planned entry remains valid but stays non-runnable."""

    workspace = MidogppWorkspace.load(ROOT)
    assert workspace.experiments[EXPERIMENT_ID].status == "planned"
    workspace.validate()


@pytest.mark.parametrize("surface", ("runner_argv", "output_canonical_path"))
def test_workspace_validate_rejects_runnable_harp_projection_drift(
    surface: str,
) -> None:
    workspace = _runnable_harp_workspace_for_validation()
    workspace.validate()
    experiment = workspace.experiments[EXPERIMENT_ID]
    if surface == "runner_argv":
        workspace.experiments[EXPERIMENT_ID] = replace(
            experiment,
            runner_argv=("{python}", "-c", "raise SystemExit(0)"),
        )
    else:
        output = workspace.artifacts[experiment.output_artifact_id]
        workspace.artifacts[experiment.output_artifact_id] = replace(
            output,
            canonical_path=(
                "artifacts/midogpp/90_oracles_and_diagnostics/"
                "tampered_harp_output/v1"
            ),
        )
    with pytest.raises(
        WorkspaceError,
        match="in-memory workspace registration execution contract drifted",
    ):
        workspace.validate()


def _patch_fake_executable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harp_authority,
        "load_config",
        lambda _path: SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            input_artifact_ids=INPUT_ARTIFACT_IDS,
            execution_authorized=True,
            expected_execution_amendment_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        harp_authority,
        "lease_path",
        lambda root: root / "absent-lease",
    )


def _runnable_harp_workspace_for_validation() -> MidogppWorkspace:
    """Activate only the reviewed HARP consumer fences in a loaded workspace."""

    workspace = MidogppWorkspace.load(ROOT)
    experiment = workspace.experiments[EXPERIMENT_ID]
    workspace.experiments[EXPERIMENT_ID] = replace(
        experiment,
        status="diagnostic",
    )
    for artifact_id in experiment.input_artifact_ids:
        artifact = workspace.artifacts[artifact_id]
        identities = dict(artifact.semantic_identities)
        if "registered_consumer_experiment_ids" not in identities:
            continue
        identities.pop("registered_consumer_experiment_ids")
        identities.pop("consumer_resolution_fence_only", None)
        identities.update(
            {
                "authorized_consumer_experiment_ids": EXPERIMENT_ID,
                "execution_authorized": "true",
                "consumed_test_reuse_authorized": "true",
            }
        )
        workspace.artifacts[artifact_id] = replace(
            artifact,
            semantic_identities=identities,
        )
    return workspace


def _write_minimal_config(root: Path) -> Path:
    path = root / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "inputs:\n"
        "  execution_amendment_path: "
        f"artifact://{authorization.EXECUTION_AMENDMENT_ARTIFACT_ID}/"
        f"{authorization.EXECUTION_AMENDMENT_FILENAME}\n",
        encoding="utf-8",
    )
    _write_workspace_registration(root)
    return path


def _write_workspace_registration(root: Path) -> tuple[Path, Path]:
    contract = authorization.workspace_registration_execution_contract()
    experiment = {
        "experiment_id": contract["experiment_id"],
        "stage": contract["stage"],
        "status": contract["status"],
        "claim_scope": contract["claim_scope"],
        "config_path": contract["config_path"],
        "output_artifact_id": contract["output_artifact_id"],
        "input_artifact_ids": list(contract["input_artifact_ids"]),
        "runner": {
            "preparation_authority_gate": contract[
                "preparation_authority_gate"
            ],
            "environment": dict(contract["runner_environment"]),
            "argv": list(contract["runner_argv"]),
        },
    }
    registry_path = root / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump({"experiments": [experiment]}, sort_keys=False),
        encoding="utf-8",
    )
    catalog_path = (
        root / authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "artifact_id": contract["output_artifact_id"],
                        "canonical_path": contract["output_canonical_path"],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry_path, catalog_path


def _registration_projection() -> dict[str, object]:
    contract = authorization.workspace_registration_execution_contract()
    return {
        key: value
        for key, value in contract.items()
        if key
        not in {
            "schema_version",
            "workspace_registration_execution_contract_hash",
        }
    }
