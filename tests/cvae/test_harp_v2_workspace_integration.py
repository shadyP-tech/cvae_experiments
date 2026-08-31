from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser, main as diagnostics_main
from midogpp_thesis.workspace import preparation_authority as authority_adapter
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V1_EXECUTION_AMENDMENT_GATE,
    HARP_V1_EXPERIMENT_ID,
    HARP_V2_EXECUTION_AMENDMENT_GATE,
    HARP_V2_EXPERIMENT_ID,
    KNOWN_PREPARATION_AUTHORITY_GATES,
    PreparationAuthorityError,
    PreparationAuthorityReceipt,
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
V2_CONFIG = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v2.yaml"
)
V2_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_harp_router_v2"
)
V2_OUTPUT_PATH = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router/v2"
)
V2_INPUT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_harp_consumed_test_cache_v2",
    "midogpp_stage90_harp_consumed_test_development_manifest_v2",
    "midogpp_stage90_harp_consumed_test_evaluation_manifest_v2",
    "midogpp_uniform_b_test_consumption_ledger_harp_parent_v2",
    "midogpp_uniform_b_test_consumption_ledger_harp_execution_amendment_v2",
)
V2_RUNNER_ARGV = (
    "{python}",
    "-m",
    "midogpp_thesis",
    "cvae-diagnostics",
    "fixed-bank-harp-router-v2",
    "--config",
    "{resolved_config}",
    "--artifact-root",
    "output://midogpp_output_uniform_b_v2_consumed_test_fixed_bank_harp_router_v2",
)
V2_RUNNER_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "3",
    "MKL_NUM_THREADS": "3",
    "OPENBLAS_NUM_THREADS": "3",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def _experiment(*, gate: str = HARP_V2_EXECUTION_AMENDMENT_GATE) -> ExperimentEntry:
    return ExperimentEntry(
        experiment_id=HARP_V2_EXPERIMENT_ID,
        stage="90_oracles_and_diagnostics",
        status="diagnostic",
        claim_scope="diagnostic_only",
        output_artifact_id=V2_OUTPUT_ID,
        config_path=V2_CONFIG,
        runner_argv=V2_RUNNER_ARGV,
        runner_env=V2_RUNNER_ENV,
        run_recovery_strategy=None,
        input_artifact_ids=V2_INPUT_IDS,
        input_claim_scope_exceptions={},
        notes=(),
        preparation_authority_gate=gate,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_planned_v2_config(path: Path) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.identity import (
        claim_boundary_payload,
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["experiment"]["status"] = "planned"
    payload["experiment"]["execution_authorized"] = False
    for role in (
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
        "execution_amendment_sha256",
    ):
        payload["inputs"][role] = None
    payload["claim_boundary"] = claim_boundary_payload(
        execution_authorized=False
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_v2_gate_is_exact_closed_world_and_does_not_weaken_v1() -> None:
    assert HARP_V2_EXECUTION_AMENDMENT_GATE == (
        "harp_v2_consumed_test_execution_amendment_v1"
    )
    assert HARP_V2_EXPERIMENT_ID == (
        "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v2"
    )
    assert HARP_V2_EXECUTION_AMENDMENT_GATE in KNOWN_PREPARATION_AUTHORITY_GATES
    assert (
        validate_preparation_authority_gate_id(HARP_V2_EXECUTION_AMENDMENT_GATE)
        == HARP_V2_EXECUTION_AMENDMENT_GATE
    )
    assert preparation_authority_registration_error(
        None,
        experiment_id=HARP_V2_EXPERIMENT_ID,
    ) == (
        f"{HARP_V2_EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V2_EXECUTION_AMENDMENT_GATE!r}"
    )
    assert preparation_authority_registration_error(
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        experiment_id=HARP_V1_EXPERIMENT_ID,
    ) == (
        f"{HARP_V1_EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V1_EXECUTION_AMENDMENT_GATE!r}"
    )
    assert preparation_authority_registration_error(
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        experiment_id="wrong.consumer",
    ) == (
        "wrong.consumer: runner.preparation_authority_gate "
        f"{HARP_V2_EXECUTION_AMENDMENT_GATE!r} is bound only to "
        f"{HARP_V2_EXPERIMENT_ID}"
    )


def test_checked_in_v2_registration_is_authorized_exact_and_workspace_valid() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.config import (
        load_config as load_v2_config,
    )

    workspace = MidogppWorkspace.load(ROOT)
    experiment = workspace.experiments[HARP_V2_EXPERIMENT_ID]
    assert experiment.status == "diagnostic"
    assert experiment.preparation_authority_gate == (
        HARP_V2_EXECUTION_AMENDMENT_GATE
    )
    assert experiment.config_path == V2_CONFIG
    assert experiment.output_artifact_id == V2_OUTPUT_ID
    assert experiment.input_artifact_ids == V2_INPUT_IDS
    assert "fixed-bank-harp-router-v2" in experiment.runner_argv
    assert "fixed-bank-harp-router-v1" not in experiment.runner_argv
    config = load_v2_config(ROOT / V2_CONFIG)
    assert config.execution_authorized is True
    assert config.expected_execution_amendment_sha256 == (
        "bc6f9cb2bff4cd45cfbfe54c98ece16726254e62b1efeff6ca7b0056661f19e9"
    )
    assert config.claim_boundary["execution_authorized"] is True
    workspace.validate()


def test_v2_authority_module_binding_is_source_closed() -> None:
    assert authority_adapter._AUTHORITY_MODULE_BY_GATE[
        HARP_V2_EXECUTION_AMENDMENT_GATE
    ] == (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
        "workspace_preparation_authority",
        "HarpV2WorkspaceAuthorityError",
    )
    with pytest.raises(PreparationAuthorityError, match="unknown"):
        validate_preparation_authority_gate_id(
            "midogpp_thesis.cvae.diagnostics.attacker.authority"
        )


def test_v2_enforcement_uses_only_closed_module_and_carries_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class HarpV2WorkspaceAuthorityError(ValueError):
        pass

    def validate_workspace_preparation_authority(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            config_path=tmp_path / "config.yaml",
            config_sha256="1" * 64,
            amendment_path=tmp_path / "amendment.json",
            amendment_sha256="2" * 64,
            workspace_registration_contract_hash="3" * 64,
            registry_path=tmp_path / "registry.yaml",
            registry_sha256="4" * 64,
            artifact_catalog_path=tmp_path / "catalog.yaml",
            artifact_catalog_sha256="5" * 64,
        )

    module = SimpleNamespace(
        HarpV2WorkspaceAuthorityError=HarpV2WorkspaceAuthorityError,
        validate_workspace_preparation_authority=(
            validate_workspace_preparation_authority
        ),
    )
    imported: list[str] = []

    def import_only_exact(name: str):
        imported.append(name)
        assert name == (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
            "workspace_preparation_authority"
        )
        return module

    monkeypatch.setattr(authority_adapter, "import_module", import_only_exact)
    projection = {"experiment_id": HARP_V2_EXPERIMENT_ID}
    receipt = authority_adapter.enforce_preparation_authority(
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        repo_root=tmp_path,
        experiment_id=HARP_V2_EXPERIMENT_ID,
        config_path=V2_CONFIG,
        input_artifact_ids=V2_INPUT_IDS,
        registration_projection=projection,
        resolve_authority_member=lambda *_args: (_ for _ in ()).throw(
            AssertionError("fake authority must not resolve inputs")
        ),
    )
    assert imported == [
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
        "workspace_preparation_authority"
    ]
    assert observed["registration_projection"] is projection
    assert receipt is not None
    assert receipt.gate_id == HARP_V2_EXECUTION_AMENDMENT_GATE
    assert receipt.workspace_registration_contract_hash == "3" * 64


@pytest.mark.parametrize(
    "extra_args",
    (
        ("--inspect-plan",),
        ("--dry-run", "--dry-run"),
        ("--dry-run", "--inspect-plan"),
        ("--arbitrary",),
    ),
)
def test_v2_workspace_accepts_only_empty_or_exact_dry_run(
    extra_args: tuple[str, ...],
) -> None:
    assert validate_preparation_authority_extra_args(
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        (),
    ) == ()
    assert validate_preparation_authority_extra_args(
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)
    with pytest.raises(PreparationAuthorityError, match="accepts only"):
        validate_preparation_authority_extra_args(
            HARP_V2_EXECUTION_AMENDMENT_GATE,
            extra_args,
        )
    with pytest.raises(PreparationAuthorityError, match="reject --force"):
        validate_preparation_authority_extra_args(
            HARP_V2_EXECUTION_AMENDMENT_GATE,
            (),
            force=True,
        )


def test_v2_invalid_workspace_options_fail_before_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = object.__new__(MidogppWorkspace)
    experiment = _experiment()
    monkeypatch.setattr(workspace, "validate", lambda: None)
    monkeypatch.setattr(workspace, "get_experiment", lambda _value: experiment)
    monkeypatch.setattr(
        workspace,
        "_enforce_preparation_authority",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("authority ran after invalid workspace options")
        ),
    )
    with pytest.raises(WorkspaceError, match="reject --force"):
        workspace.prepare(HARP_V2_EXPERIMENT_ID, force=True)
    with pytest.raises(WorkspaceError, match="reject --force"):
        workspace.run(HARP_V2_EXPERIMENT_ID, force=True)
    with pytest.raises(WorkspaceError, match="accepts only"):
        workspace.run(
            HARP_V2_EXPERIMENT_ID,
            extra_args=("--inspect-plan",),
        )


def test_v2_registration_projection_is_exact_and_version_local() -> None:
    workspace = object.__new__(MidogppWorkspace)
    experiment = _experiment()
    workspace.artifacts = {
        V2_OUTPUT_ID: SimpleNamespace(canonical_path=V2_OUTPUT_PATH)
    }
    projection = workspace._preparation_authority_registration_projection(
        experiment
    )
    assert projection == {
        "experiment_id": HARP_V2_EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "status": "diagnostic",
        "claim_scope": "diagnostic_only",
        "config_path": V2_CONFIG,
        "output_artifact_id": V2_OUTPUT_ID,
        "output_canonical_path": V2_OUTPUT_PATH,
        "input_artifact_ids": list(V2_INPUT_IDS),
        "preparation_authority_gate": HARP_V2_EXECUTION_AMENDMENT_GATE,
        "run_recovery_strategy": None,
        "runner_argv": list(V2_RUNNER_ARGV),
        "runner_environment": dict(V2_RUNNER_ENV),
    }
    assert workspace._preparation_authority_registration_projection(
        _experiment(gate=HARP_V1_EXECUTION_AMENDMENT_GATE)
    ) is not None
    assert workspace._preparation_authority_registration_projection(
        _experiment(gate="not-a-harp-gate")
    ) is None


def test_v2_receipt_revalidates_registration_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / V2_CONFIG
    amendment = tmp_path / "contracts/harp_router_v2/amendment.json"
    registry = tmp_path / "experiments/midogpp/registry.yaml"
    catalog = tmp_path / "experiments/midogpp/artifact_catalog.yaml"
    for path, content in (
        (config, "version: 2\n"),
        (amendment, "{}\n"),
        (registry, "experiments: []\n"),
        (catalog, "artifacts: []\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    registration_hash = "a" * 64
    monkeypatch.setattr(
        "midogpp_thesis.workspace.runtime."
        "expected_workspace_registration_contract_hash",
        lambda gate_id: registration_hash
        if gate_id == HARP_V2_EXECUTION_AMENDMENT_GATE
        else None,
    )
    workspace = object.__new__(MidogppWorkspace)
    workspace.repo_root = tmp_path.resolve()
    receipt = PreparationAuthorityReceipt(
        gate_id=HARP_V2_EXECUTION_AMENDMENT_GATE,
        experiment_id=HARP_V2_EXPERIMENT_ID,
        config_path=config.resolve(),
        config_sha256=_sha256(config),
        authority_path=amendment.resolve(),
        authority_sha256=_sha256(amendment),
        workspace_registration_contract_hash=registration_hash,
        registry_path=registry.resolve(),
        registry_sha256=_sha256(registry),
        artifact_catalog_path=catalog.resolve(),
        artifact_catalog_sha256=_sha256(catalog),
    )
    experiment = _experiment()
    workspace._verify_preparation_authority_receipt(
        experiment,
        receipt=receipt,
    )
    registry.write_text("experiments: [tampered]\n", encoding="utf-8")
    with pytest.raises(
        WorkspaceError,
        match="HARP workspace registration authority bytes changed",
    ):
        workspace._verify_preparation_authority_receipt(
            experiment,
            receipt=receipt,
        )


def test_v2_cli_surfaces_are_fixed_and_mutually_exclusive() -> None:
    parser = build_parser()
    run = parser.parse_args(
        ["fixed-bank-harp-router-v2", "--config", V2_CONFIG, "--dry-run"]
    )
    assert run.surface == "fixed-bank-harp-router-v2"
    assert run.dry_run is True
    assert run.inspect_plan is False
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fixed-bank-harp-router-v2",
                "--config",
                V2_CONFIG,
                "--dry-run",
                "--inspect-plan",
            ]
        )
    prepared = parser.parse_args(
        [
            "prepare-fixed-bank-harp-router-v2-inputs",
            "--canonical-cache-root",
            "/canonical",
            "--canonical-manifest",
            "/canonical.csv",
            "--parent-ledger",
            "/parent.json",
            "--cache-root",
            "/prepared",
            "--development-manifest",
            "/development.csv",
            "--evaluation-manifest",
            "/evaluation.csv",
        ]
    )
    assert prepared.surface == "prepare-fixed-bank-harp-router-v2-inputs"
    assert not hasattr(prepared, "artifact_root")
    assert not hasattr(prepared, "dry_run")
    published = parser.parse_args(
        [
            "publish-fixed-bank-harp-router-v2-amendment",
            "--config",
            V2_CONFIG,
            "--expert-bank-root",
            "/bank",
            "--generation-lock-root",
            "/generation",
            "--prepared-cache-root",
            "/cache",
            "--development-manifest",
            "/development.csv",
            "--evaluation-manifest",
            "/evaluation.csv",
            "--parent-ledger",
            "/parent.json",
            "--amendment-path",
            "/amendment.json",
            "--authorization-basis",
            "explicit",
            "--authorization-date",
            "2026-08-31",
            "--repository-root",
            "/repository",
        ]
    )
    assert published.surface == "publish-fixed-bank-harp-router-v2-amendment"
    assert not hasattr(published, "artifact_root")


def test_real_v2_cli_inspection_is_path_free_authorized_and_nonmutating(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert diagnostics_main(
        [
            "fixed-bank-harp-router-v2",
            "--config",
            str(ROOT / V2_CONFIG),
            "--inspect-plan",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == HARP_V2_EXPERIMENT_ID
    assert payload["execution_revision"] == (
        "v2_byte_equivalent_performance_optimization"
    )
    assert payload["status"] == "EXECUTABLE_AUTHORIZED_UNPROBED"
    assert payload["execution_authorized"] is True
    assert payload["authorization_probed"] is False
    assert payload["paths_resolved"] is False
    assert payload["filesystem_mutations"] == 0
    assert payload["development_labels_opened"] is False
    assert payload["evaluation_labels_opened"] is False
    assert payload["publication_status"] == "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    assert payload["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )
    assert payload["fresh_evidence"] is False
    assert payload["physical_plan"]["action_count"] == 738
    assert payload["physical_plan"]["exact_nine_cell_count"] == 6642


def test_v2_publisher_activation_and_workspace_authority_resolve_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2 import (
        amendment_publisher as v2_publisher,
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2 import (
        authorization as v2_authorization,
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.config import (
        load_config as load_v2_config,
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.identity import (
        claim_boundary_payload,
    )
    from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash

    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT / "experiments/midogpp",
        repository / "experiments/midogpp",
    )
    config_path = repository / V2_CONFIG
    amendment_path = (
        repository / v2_authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    )
    _write_synthetic_planned_v2_config(config_path)
    if amendment_path.exists():
        amendment_path.unlink()
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    planned_config = load_v2_config(config_path)
    assert planned_config.execution_authorized is False
    assert planned_config.claim_boundary == claim_boundary_payload(
        execution_authorized=False
    )
    assert all(
        planned_config.expected_hashes[role] is None
        for role in (
            "test_cache_content_sha256",
            "development_manifest_sha256",
            "evaluation_manifest_sha256",
            "parent_ledger_sha256",
            "execution_amendment_sha256",
        )
    )

    bank = repository / "synthetic-inputs/bank"
    generation = repository / "synthetic-inputs/generation"
    cache = repository / "synthetic-inputs/cache"
    for directory in (bank, generation, cache):
        directory.mkdir(parents=True, exist_ok=True)
    content_base = {
        "schema_version": "test_harp_v2_content_index",
        "members": {},
    }
    content_hash = canonical_hash(content_base)
    content_path = cache / "manifests/content_index.json"
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(
        json.dumps(
            {**content_base, "content_index_hash": content_hash},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    development = repository / "synthetic-inputs/development.csv"
    evaluation = repository / "synthetic-inputs/evaluation.csv"
    parent = repository / "synthetic-inputs/parent.json"
    development.write_text("sample_id,label\ndev,0\n", encoding="utf-8")
    evaluation.write_text("sample_id,label\neval,1\n", encoding="utf-8")
    parent.write_text("{}\n", encoding="utf-8")

    source_identity = {
        "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v2",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 1,
        "source_snapshot_member_pattern": "synthetic_closed_world_source",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(
        v2_authorization,
        "source_snapshot_identity",
        lambda _root=None: source_identity,
    )
    fake_cache = SimpleNamespace(
        root=cache,
        member_sha256={},
        content_sha256=content_hash,
        cache_hash="3" * 64,
    )
    monkeypatch.setattr(v2_publisher, "load_cache_index", lambda _config: fake_cache)
    monkeypatch.setattr(
        v2_publisher,
        "_validate_preparation_receipt",
        lambda observed_cache, _computed: "4" * 64
        if observed_cache is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )
    monkeypatch.setattr(
        v2_publisher,
        "validate_physical_inputs",
        lambda config, observed_cache: SimpleNamespace(receipt_hash="5" * 64)
        if observed_cache is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )

    publication = v2_publisher.publish_harp_v2_execution_amendment(
        planned_config,
        expert_bank_root=bank,
        generation_lock_root=generation,
        prepared_cache_root=cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        parent_ledger_path=parent,
        amendment_path=amendment_path,
        authorization_basis=v2_publisher.AUTHORIZATION_BASIS,
        authorization_date=v2_publisher.AUTHORIZATION_DATE,
        repository_root=repository,
    )
    assert publication.amendment_path == amendment_path
    assert publication.amendment_sha256 == _sha256(amendment_path)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["experiment"]["status"] = "diagnostic"
    config_payload["experiment"]["execution_authorized"] = True
    config_payload["inputs"].update(
        {
            "test_cache_content_sha256": content_hash,
            "development_manifest_sha256": _sha256(development),
            "evaluation_manifest_sha256": _sha256(evaluation),
            "parent_ledger_sha256": _sha256(parent),
            "execution_amendment_sha256": publication.amendment_sha256,
        }
    )
    config_payload["claim_boundary"] = claim_boundary_payload(
        execution_authorized=True
    )
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    activated_config = load_v2_config(config_path)
    assert activated_config.execution_authorized is True
    assert activated_config.expected_execution_amendment_sha256 == (
        publication.amendment_sha256
    )

    registry_path = repository / "experiments/midogpp/registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    experiment_rows = [
        row
        for row in registry["experiments"]
        if row["experiment_id"] == HARP_V2_EXPERIMENT_ID
    ]
    assert len(experiment_rows) == 1
    experiment_rows[0]["status"] = "diagnostic"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    catalog_path = repository / "experiments/midogpp/artifact_catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    input_rows = {
        row["artifact_id"]: row
        for row in catalog["artifacts"]
        if row["artifact_id"] in V2_INPUT_IDS
    }
    assert set(input_rows) == set(V2_INPUT_IDS)
    for row in input_rows.values():
        identities = row.get("semantic_identities")
        if not isinstance(identities, dict) or (
            "registered_consumer_experiment_ids" not in identities
        ):
            continue
        identities.pop("registered_consumer_experiment_ids", None)
        identities.pop("consumer_resolution_fence_only", None)
        identities.update(
            {
                "authorized_consumer_experiment_ids": HARP_V2_EXPERIMENT_ID,
                "execution_authorized": "true",
                "consumed_test_reuse_authorized": "true",
            }
        )
    amendment_entry = input_rows[
        v2_authorization.EXECUTION_AMENDMENT_ARTIFACT_ID
    ]
    amendment_entry["expected_file_hashes"] = {
        v2_authorization.EXECUTION_AMENDMENT_FILENAME: {
            "algorithm": "sha256",
            "digest": publication.amendment_sha256,
        }
    }
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )

    expected_uri = (
        f"artifact://{v2_authorization.EXECUTION_AMENDMENT_ARTIFACT_ID}/"
        f"{v2_authorization.EXECUTION_AMENDMENT_FILENAME}"
    )
    assert config_payload["inputs"]["execution_amendment_path"] == expected_uri
    assert amendment_entry["physical_path"] == str(
        Path(v2_authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH).parent
    )
    assert (
        repository
        / amendment_entry["physical_path"]
        / amendment_entry["required_files"][0]
    ) == amendment_path

    workspace = MidogppWorkspace.load(repository)
    workspace.validate()
    experiment = workspace.experiments[HARP_V2_EXPERIMENT_ID]
    receipt = workspace._enforce_preparation_authority(experiment)
    assert receipt is not None
    assert receipt.gate_id == HARP_V2_EXECUTION_AMENDMENT_GATE
    assert receipt.authority_path == amendment_path
    assert receipt.authority_sha256 == publication.amendment_sha256
    workspace._verify_preparation_authority_receipt(
        experiment,
        receipt=receipt,
    )


def test_v2_cli_dispatches_only_to_v2_interfaces(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[tuple[str, object]] = []
    config_value = object()

    config_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.config"
    )
    config_module.load_config = lambda path: observed.append(("config", path)) or config_value
    runner_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.runner"
    )
    runner_module.inspect_harp_stage90_v2 = lambda *_args, **_kwargs: {
        "mode": "inspect"
    }
    runner_module.dry_run_harp_stage90_v2 = (
        lambda config, *, artifact_root: observed.append(
            ("dry-run", (config, artifact_root))
        )
        or {"mode": "dry-run"}
    )
    runner_module.run_harp_stage90_v2 = lambda *_args, **_kwargs: "run"
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)

    assert diagnostics_main(
        [
            "fixed-bank-harp-router-v2",
            "--config",
            V2_CONFIG,
            "--artifact-root",
            "/artifact-v2",
            "--dry-run",
        ]
    ) == 0
    assert observed == [
        ("config", V2_CONFIG),
        ("dry-run", (config_value, Path("/artifact-v2"))),
    ]
    assert json.loads(capsys.readouterr().out) == {"mode": "dry-run"}


def test_v2_cli_dispatches_preparation_and_publication_without_output_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[tuple[str, object]] = []
    prepared_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.preparation"
    )
    prepared_module.prepare_harp_consumed_test_inputs_v2 = (
        lambda **kwargs: observed.append(("prepare", kwargs))
        or SimpleNamespace(to_payload=lambda: {"mode": "prepare-v2"})
    )
    config_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.config"
    )
    config_value = object()
    config_module.load_config = lambda path: observed.append(("config", path)) or config_value
    publisher_module = ModuleType(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
        "amendment_publisher"
    )
    publisher_module.publish_harp_v2_execution_amendment = (
        lambda config, **kwargs: observed.append(
            ("publish", (config, kwargs))
        )
        or SimpleNamespace(to_payload=lambda: {"mode": "publish-v2"})
    )
    monkeypatch.setitem(sys.modules, prepared_module.__name__, prepared_module)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, publisher_module.__name__, publisher_module)

    assert diagnostics_main(
        [
            "prepare-fixed-bank-harp-router-v2-inputs",
            "--canonical-cache-root",
            "/canonical",
            "--canonical-manifest",
            "/canonical.csv",
            "--parent-ledger",
            "/parent.json",
            "--cache-root",
            "/prepared",
            "--development-manifest",
            "/development.csv",
            "--evaluation-manifest",
            "/evaluation.csv",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "prepare-v2"}
    assert observed[0] == (
        "prepare",
        {
            "canonical_cache_root": "/canonical",
            "canonical_manifest_path": "/canonical.csv",
            "parent_ledger_path": "/parent.json",
            "cache_root": "/prepared",
            "development_manifest_path": "/development.csv",
            "evaluation_manifest_path": "/evaluation.csv",
        },
    )

    assert diagnostics_main(
        [
            "publish-fixed-bank-harp-router-v2-amendment",
            "--config",
            V2_CONFIG,
            "--expert-bank-root",
            "/bank",
            "--generation-lock-root",
            "/generation",
            "--prepared-cache-root",
            "/cache",
            "--development-manifest",
            "/development.csv",
            "--evaluation-manifest",
            "/evaluation.csv",
            "--parent-ledger",
            "/parent.json",
            "--amendment-path",
            "/amendment.json",
            "--authorization-basis",
            "explicit",
            "--authorization-date",
            "2026-08-31",
            "--repository-root",
            "/repository",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "publish-v2"}
    assert observed[1] == ("config", V2_CONFIG)
    publish_config, publish_kwargs = observed[2][1]
    assert publish_config is config_value
    assert publish_kwargs == {
        "expert_bank_root": "/bank",
        "generation_lock_root": "/generation",
        "prepared_cache_root": "/cache",
        "development_manifest_path": "/development.csv",
        "evaluation_manifest_path": "/evaluation.csv",
        "parent_ledger_path": "/parent.json",
        "amendment_path": "/amendment.json",
        "authorization_basis": "explicit",
        "authorization_date": "2026-08-31",
        "repository_root": "/repository",
    }
