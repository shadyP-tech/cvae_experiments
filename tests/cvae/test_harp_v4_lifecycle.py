from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import (
    authorization as predecessor_authorization,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import (
    INPUT_ARTIFACT_IDS as PREDECESSOR_INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.identity import (
    EXPERIMENT_ID as PREDECESSOR_EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID as PREDECESSOR_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import (
    activation,
    amendment_publisher,
    authorization,
    workstation_preparation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.input_surfaces import (
    CONTENT_INDEX,
    V4_CACHE_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.preparation import (
    PREPARATION_RECEIPT,
    HarpV4PreparedInputs,
    V4_PREPARATION_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.workspace_paths import (
    resolve_harp_v4_workspace_paths,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V4_EXECUTION_AMENDMENT_GATE,
    HARP_V4_RUN_CONFIRMATION_TOKEN,
    PreparationAuthorityError,
    validate_preparation_authority_extra_args,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT / "experiments/midogpp",
        repository / "experiments/midogpp",
    )
    for relative in (
        "artifacts/midogpp/30_expert_bank/"
        "uniform_b_v2_routing_authorized_expert_bank_v1",
        "artifacts/midogpp/40_prior_and_generation/"
        "uniform_b_v2_generation_lock/v1",
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_v2_descriptive_test_cache_v1/seed42",
        "datasets/midogpp/contract/annotation_patch_v1",
        "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/reports",
        "artifacts/midogpp/90_oracles_and_diagnostics",
    ):
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (
        repository
        / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    ).write_text("case_id,center,split,label\ncase,0,test,0\n", encoding="utf-8")
    atomic_json(
        repository
        / "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/"
        "reports/test_consumption_ledger.json",
        {"schema_version": "synthetic_parent"},
    )
    return repository


def _fake_builder(
    plan: workstation_preparation.HarpV4WorkstationPreparationPlan,
    *,
    cache_root: Path,
    development_manifest_path: Path,
    evaluation_manifest_path: Path,
) -> HarpV4PreparedInputs:
    cache_root.mkdir(parents=True, exist_ok=False)
    development_manifest_path.parent.mkdir(parents=True, exist_ok=False)
    evaluation_manifest_path.parent.mkdir(parents=True, exist_ok=False)
    development_manifest_path.write_text("role,development\n", encoding="utf-8")
    evaluation_manifest_path.write_text("role,evaluation\n", encoding="utf-8")
    atomic_json(cache_root / "manifests/cache_index.json", {"synthetic": True})
    receipt_base = {
        "schema_version": V4_PREPARATION_IDENTITY.preparation_receipt_schema,
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "partition_hash": "4" * 64,
        "cache_fsynced_and_independently_validated_before_manifest_open": True,
        "execution_amendment_created": False,
        "execution_authorized": False,
    }
    receipt = {**receipt_base, "receipt_hash": canonical_hash(receipt_base)}
    atomic_json(cache_root / PREPARATION_RECEIPT, receipt)
    content_base = {
        "schema_version": V4_CACHE_IDENTITY.content_schema,
        "members": {
            PREPARATION_RECEIPT.as_posix(): sha256_file(
                cache_root / PREPARATION_RECEIPT
            )
        },
    }
    content = {
        **content_base,
        "content_index_hash": canonical_hash(content_base),
    }
    atomic_json(cache_root / CONTENT_INDEX, content)
    return HarpV4PreparedInputs(
        cache_root=cache_root.resolve(),
        development_manifest_path=development_manifest_path.resolve(),
        evaluation_manifest_path=evaluation_manifest_path.resolve(),
        cache_content_sha256=str(content["content_index_hash"]),
        development_manifest_sha256=sha256_file(development_manifest_path),
        evaluation_manifest_sha256=sha256_file(evaluation_manifest_path),
        parent_ledger_sha256=plan.expected_parent_ledger_sha256,
        partition_hash="4" * 64,
        preparation_receipt_hash=str(receipt["receipt_hash"]),
    )


def _predecessor_state(repository: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    predecessor_files = {
        "config": repository / predecessor_authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        "cache": repository
        / "datasets/midogpp/derived/features/virchow2/"
        "harp_consumed_test_cache_v3/predecessor-sentinel.bin",
        "output": repository
        / predecessor_authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
        / "predecessor-sentinel.bin",
        "amendment": repository
        / predecessor_authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH,
        "lease": predecessor_authorization.lease_path(repository)
        / "predecessor-sentinel.bin",
    }
    for name, path in predecessor_files.items():
        if name == "config":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("immutable-" + name).encode("utf-8"))

    registry = yaml.safe_load(
        (repository / "experiments/midogpp/registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    catalog = yaml.safe_load(
        (repository / "experiments/midogpp/artifact_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    predecessor_ids = {
        *PREDECESSOR_INPUT_ARTIFACT_IDS,
        PREDECESSOR_OUTPUT_ARTIFACT_ID,
    }
    metadata = {
        "registry": copy.deepcopy(
            next(
                row
                for row in registry["experiments"]
                if row["experiment_id"] == PREDECESSOR_EXPERIMENT_ID
            )
        ),
        "catalog": copy.deepcopy(
            [
                row
                for row in catalog["artifacts"]
                if row["artifact_id"] in predecessor_ids
            ]
        ),
    }
    return (
        {name: path.read_bytes() for name, path in predecessor_files.items()},
        metadata,
    )


def _assert_predecessor_state(
    repository: Path,
    expected_files: dict[str, bytes],
    expected_metadata: dict[str, object],
) -> None:
    observed_files = {
        "config": repository / predecessor_authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        "cache": repository
        / "datasets/midogpp/derived/features/virchow2/"
        "harp_consumed_test_cache_v3/predecessor-sentinel.bin",
        "output": repository
        / predecessor_authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
        / "predecessor-sentinel.bin",
        "amendment": repository
        / predecessor_authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH,
        "lease": predecessor_authorization.lease_path(repository)
        / "predecessor-sentinel.bin",
    }
    assert {
        name: path.read_bytes() for name, path in observed_files.items()
    } == expected_files

    registry = yaml.safe_load(
        (repository / "experiments/midogpp/registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    catalog = yaml.safe_load(
        (repository / "experiments/midogpp/artifact_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    predecessor_ids = {
        *PREDECESSOR_INPUT_ARTIFACT_IDS,
        PREDECESSOR_OUTPUT_ARTIFACT_ID,
    }
    assert next(
        row
        for row in registry["experiments"]
        if row["experiment_id"] == PREDECESSOR_EXPERIMENT_ID
    ) == expected_metadata["registry"]
    assert [
        row for row in catalog["artifacts"] if row["artifact_id"] in predecessor_ids
    ] == expected_metadata["catalog"]


def _activation_plan(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> activation.HarpV4ActivationPlan:
    paths = resolve_harp_v4_workspace_paths(repository, require_prepared=True)
    config = load_config(paths.config_path)
    cache_root = paths.prepared_cache_root
    content = json.loads((cache_root / CONTENT_INDEX).read_text(encoding="utf-8"))
    fake_cache = SimpleNamespace(
        root=cache_root,
        content_sha256=str(content["content_index_hash"]),
        cache_hash="3" * 64,
        member_sha256={},
    )
    fake_source = {
        "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v4",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 1,
        "source_snapshot_member_pattern": "synthetic_closed_world_source",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(
        authorization,
        "source_snapshot_identity",
        lambda _root=None: fake_source,
    )
    monkeypatch.setattr(
        activation,
        "source_snapshot_identity",
        lambda _root=None: fake_source,
    )
    monkeypatch.setattr(
        amendment_publisher,
        "load_cache_index",
        lambda _config: fake_cache,
    )
    monkeypatch.setattr(
        amendment_publisher,
        "_validate_preparation_receipt",
        lambda observed, _computed: "5" * 64
        if observed is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )
    monkeypatch.setattr(
        amendment_publisher,
        "validate_physical_inputs",
        lambda _config, observed: SimpleNamespace(receipt_hash="6" * 64)
        if observed is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )
    return activation.plan_harp_v4_activation(
        config,
        **paths.activation_kwargs(),
        repository_root=repository,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date="2026-09-01",
    )


def test_v4_preparation_plan_is_catalog_bound_mutation_free_and_label_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    manifest = (
        repository / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    ).resolve()
    before = {
        path: path.read_bytes()
        for path in (
            repository / "experiments/midogpp/registry.yaml",
            repository / "experiments/midogpp/artifact_catalog.yaml",
            repository / authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        )
    }
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == manifest:
            raise AssertionError("scoring manifest opened during label-free planning")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    plan = workstation_preparation.plan_harp_v4_workstation_preparation(repository)

    assert plan.paths.canonical_manifest_path == manifest
    assert plan.paths.prepared_cache_root.name == "harp_consumed_test_cache_v4"
    assert plan.paths.development_manifest_path.parent.name.endswith("_v4")
    assert plan.paths.evaluation_manifest_path.parent.name.endswith("_v4")
    assert plan.to_payload()["canonical_scoring_manifest_opened"] is False
    assert plan.to_payload()["canonical_scoring_manifest_hashed"] is False
    assert plan.to_payload()["filesystem_mutations"] == 0
    assert plan.to_payload()["execution_authorized"] is False
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_v4_full_preparation_and_activation_are_independent_and_registry_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    predecessor_files, predecessor_metadata = _predecessor_state(repository)
    monkeypatch.setattr(
        workstation_preparation,
        "_prepare_staged_inputs",
        _fake_builder,
    )
    preparation_plan = (
        workstation_preparation.plan_harp_v4_workstation_preparation(repository)
    )
    prepared = workstation_preparation.prepare_harp_v4_workstation_inputs(
        preparation_plan,
        confirmation=workstation_preparation.PREPARATION_CONFIRMATION,
    )

    assert prepared.cache_root == preparation_plan.paths.prepared_cache_root
    assert prepared.cache_root.name == "harp_consumed_test_cache_v4"
    assert prepared.development_manifest_path.parent.name.endswith("_v4")
    assert prepared.evaluation_manifest_path.parent.name.endswith("_v4")
    assert not preparation_plan.paths.amendment_path.exists()
    assert not authorization.lease_path(repository).exists()
    assert not preparation_plan.paths.output_root.exists()
    assert load_config(preparation_plan.paths.config_path).execution_authorized is False
    _assert_predecessor_state(repository, predecessor_files, predecessor_metadata)

    plan = _activation_plan(repository, monkeypatch)
    assert plan.to_payload()["filesystem_mutations"] == 0
    assert not plan.amendment_draft.amendment_path.exists()
    events: list[str] = []

    def observe(point: str) -> None:
        events.append(point)
        if point == "journal_durable":
            assert not plan.amendment_draft.amendment_path.exists()
            assert plan.registry_path.read_bytes() == plan.original_registry_bytes
        elif point == "amendment_committed":
            assert plan.amendment_draft.amendment_path.read_bytes() == (
                plan.amendment_draft.amendment_raw
            )
            assert plan.registry_path.read_bytes() == plan.original_registry_bytes
        elif point in {"config_committed", "catalog_committed"}:
            assert plan.registry_path.read_bytes() == plan.original_registry_bytes
        elif point == "registry_committed":
            assert plan.registry_path.read_bytes() == plan.final_registry_bytes

    receipt = activation.activate_harp_v4(
        plan,
        confirmation=activation.ACTIVATION_CONFIRMATION,
        _fault_injector=observe,
    )

    assert events == [
        "journal_durable",
        "amendment_committed",
        "config_committed",
        "catalog_committed",
        "registry_committed",
    ]
    assert receipt.amendment_sha256 == sha256_file(
        plan.amendment_draft.amendment_path
    )
    assert load_config(plan.config_path).execution_authorized is True
    assert not authorization.lease_path(repository).exists()
    assert not preparation_plan.paths.output_root.exists()
    amendment = json.loads(
        plan.amendment_draft.amendment_path.read_text(encoding="utf-8")
    )
    assert amendment["predecessor_authority_reused"] is False
    assert amendment["predecessor_output_or_policy_used"] is False
    assert amendment["predecessor_amendment_lease_output_cache_or_scratch_reused"] is False
    assert amendment["fresh_evidence"] is False
    _assert_predecessor_state(repository, predecessor_files, predecessor_metadata)


def test_v4_planned_workspace_and_exact_gate_fail_closed() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    experiment = workspace.experiments[EXPERIMENT_ID]
    assert experiment.status == "planned"
    assert experiment.runnable is False
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(EXPERIMENT_ID)

    assert validate_preparation_authority_extra_args(
        HARP_V4_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)
    assert validate_preparation_authority_extra_args(
        HARP_V4_EXECUTION_AMENDMENT_GATE,
        ("--confirm", HARP_V4_RUN_CONFIRMATION_TOKEN),
    ) == ("--confirm", HARP_V4_RUN_CONFIRMATION_TOKEN)
    for drifted in (
        (),
        ("--confirm",),
        ("--confirm", "wrong"),
        ("--dry-run", "--confirm", HARP_V4_RUN_CONFIRMATION_TOKEN),
    ):
        with pytest.raises(PreparationAuthorityError, match="accepts only"):
            validate_preparation_authority_extra_args(
                HARP_V4_EXECUTION_AMENDMENT_GATE,
                drifted,
            )


def test_v4_catalog_rejects_predecessor_cache_destination_before_mutation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    catalog = repository / "experiments/midogpp/artifact_catalog.yaml"
    payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    cache = next(
        row
        for row in payload["artifacts"]
        if row["artifact_id"] == V4_CACHE_IDENTITY.artifact_id
    )
    cache["canonical_path"] = (
        "datasets/midogpp/derived/features/virchow2/harp_consumed_test_cache_v3"
    )
    catalog.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError):
        workstation_preparation.plan_harp_v4_workstation_preparation(repository)
    assert not (
        repository
        / "datasets/midogpp/derived/features/virchow2/harp_consumed_test_cache_v4"
    ).exists()
    assert not authorization.lease_path(repository).exists()
    assert not (
        repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    ).exists()
