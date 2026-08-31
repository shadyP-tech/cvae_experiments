from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import authorization as v1_authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import input_surfaces as shared_inputs
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import preparation as shared_preparation
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CanonicalLabelBlindFrame,
    CanonicalFrameRow,
    V1_PREPARATION_IDENTITY,
    build_case_partition_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2 import amendment_publisher
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.config import (
    HarpStage90V2Config,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    claim_boundary_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.input_surfaces import (
    V2_CACHE_IDENTITY,
    load_cache_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.preparation import (
    PARTITION_NAMESPACE,
    V2_PREPARATION_IDENTITY,
    prepare_harp_consumed_test_inputs_v2,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.runner import (
    V2_RUNNER_SERVICES,
    inspect_harp_stage90_v2,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.source_seal import (
    source_members,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.workspace_preparation_authority import (
    HarpV2WorkspaceAuthorityError,
    validate_workspace_preparation_authority,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v2.yaml"
)

_EXECUTION_HASH_ROLES = (
    "test_cache_content_sha256",
    "development_manifest_sha256",
    "evaluation_manifest_sha256",
    "parent_ledger_sha256",
    "execution_amendment_sha256",
)


def _write_synthetic_planned_config(path: Path) -> HarpStage90V2Config:
    """Create a publisher fixture without misclassifying checked-in v2 state."""

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["status"] = "planned"
    raw["experiment"]["execution_authorized"] = False
    for role in _EXECUTION_HASH_ROLES:
        raw["inputs"][role] = None
    raw["claim_boundary"] = claim_boundary_payload(execution_authorized=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    assert config.execution_authorized is False
    assert all(config.expected_hashes[role] is None for role in _EXECUTION_HASH_ROLES)
    return config


def _synthetic_partition_rows() -> dict[str, tuple[CanonicalFrameRow, ...]]:
    return {
        center: tuple(
            CanonicalFrameRow(
                center=center,
                case_id=f"case-{center}-{case}",
                sample_id=f"sample-{center}-{case}",
                contract_row_index=2 * ordinal + case,
                center_row_index=case,
            )
            for case in range(2)
        )
        for ordinal, center in enumerate(CENTERS)
    }


def _prepare_synthetic_v2_inputs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    raw_rows: list[dict[str, str]] = []
    row_specs: list[tuple[str, str, int, int]] = []
    contract_index = 0
    for center in CENTERS:
        center_index = 0
        for case_ordinal in range(4):
            case = f"case-{center}-{case_ordinal}"
            for label in (0, 1):
                raw_rows.append(
                    {
                        "case_id": case,
                        "center": center,
                        "split": "test",
                        "label": str(label),
                    }
                )
                row_specs.append((center, case, contract_index, center_index))
                contract_index += 1
                center_index += 1
    manifest = repo / "inputs/canonical.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case_id", "center", "split", "label")
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest_sha = sha256_file(manifest)
    rows_by_center: dict[str, tuple[CanonicalFrameRow, ...]] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        rows = tuple(
            CanonicalFrameRow(
                center=center,
                case_id=case,
                sample_id=shared_preparation._evaluation_row_id(
                    manifest_sha, global_index
                ),
                contract_row_index=global_index,
                center_row_index=center_index,
            )
            for center_, case, global_index, center_index in row_specs
            if center_ == center
        )
        rows_by_center[center] = rows
        embeddings_by_center[center] = np.arange(
            len(rows) * 3840, dtype=np.float32
        ).reshape(len(rows), 3840)
    frame = CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash=CANONICAL_CACHE_CONTENT_HASH,
        row_order_hash=CANONICAL_CACHE_ROW_ORDER_HASH,
        source_member_sha256={},
    )
    monkeypatch.setattr(shared_preparation, "EXPECTED_ROW_COUNT", len(row_specs))
    monkeypatch.setattr(
        shared_preparation,
        "load_canonical_label_blind_cache",
        lambda _root: frame,
    )
    parent = repo / "inputs/parent.json"
    atomic_json(parent, {"schema_version": "synthetic_harp_v2_parent"})
    cache = repo / "inputs/cache_v2"
    development = repo / "inputs/development_v2.csv"
    evaluation = repo / "inputs/evaluation_v2.csv"
    prepared = prepare_harp_consumed_test_inputs_v2(
        canonical_cache_root=repo / "inputs/unused-canonical-cache",
        canonical_manifest_path=manifest,
        parent_ledger_path=parent,
        cache_root=cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        expected_manifest_sha256=manifest_sha,
        expected_parent_ledger_sha256=sha256_file(parent),
    )
    return prepared, manifest_sha, parent


def test_v2_partition_payload_and_hash_are_exactly_v1_equivalent() -> None:
    rows = _synthetic_partition_rows()
    v1_partition, v1_payload, v1_hash = build_case_partition_payload(
        rows, identity=V1_PREPARATION_IDENTITY
    )
    v2_partition, v2_payload, v2_hash = build_case_partition_payload(
        rows, identity=V2_PREPARATION_IDENTITY
    )
    assert PARTITION_NAMESPACE == "midogpp_harp_consumed_test_case_partition_v1"
    assert v2_partition == v1_partition
    assert v2_payload == v1_payload
    assert v2_hash == v1_hash == canonical_hash(v1_payload)


def test_v2_loader_rejects_v1_cache_schema_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content_base = {
        "schema_version": shared_inputs.V1_CACHE_IDENTITY.content_schema,
        "members": {},
    }
    content = {**content_base, "content_index_hash": canonical_hash(content_base)}
    monkeypatch.setattr(
        shared_inputs,
        "read_json",
        lambda path: content if Path(path).name == "content_index.json" else {},
    )
    config = SimpleNamespace(
        resolved_path=lambda role: tmp_path,
        expected_hashes={"test_cache_content_sha256": content["content_index_hash"]},
    )
    with pytest.raises(ProtocolError, match="cache content index drifted"):
        load_cache_index(config)  # type: ignore[arg-type]


def test_v2_config_rejects_v1_output_and_cache_aliases(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["artifact_root"] = (
        "output://midogpp_output_uniform_b_v2_consumed_test_fixed_bank_harp_router_v1"
    )
    path = tmp_path / "v1-output.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="output identity drifted"):
        load_config(path)

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["inputs"]["test_cache_root"] = (
        "artifact://midogpp_stage90_harp_consumed_test_cache_v1"
    )
    path = tmp_path / "v1-cache.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="reused another execution identity"):
        load_config(path)


def _authorized_config() -> HarpStage90V2Config:
    base = load_config(CONFIG)
    hashes = {
        **dict(base.expected_hashes),
        "test_cache_content_sha256": "a" * 64,
        "development_manifest_sha256": "b" * 64,
        "evaluation_manifest_sha256": "c" * 64,
        "parent_ledger_sha256": "d" * 64,
        "execution_amendment_sha256": "e" * 64,
    }
    return replace(
        base,
        execution_authorized=True,
        expected_hashes=hashes,
        claim_boundary=claim_boundary_payload(execution_authorized=True),
    )


def test_v2_authority_rejects_v1_amendment_and_lease_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_source = {
        "source_snapshot_schema": "test_source_v2",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 1,
        "source_snapshot_member_pattern": "test",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(authorization, "source_snapshot_identity", lambda *_: fake_source)
    with pytest.raises(ProtocolError, match="failed authentication"):
        authorization.validate_execution_amendment_payload(
            {
                "schema_version": v1_authorization.EXECUTION_AMENDMENT_SCHEMA,
                "experiment_id": "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v1",
            },
            _authorized_config(),
        )
    v1_lease = v1_authorization.HarpAuthorizationLease(Path("/tmp/v1"), "0" * 64, 1)
    with pytest.raises(ProtocolError, match="lease finalization is invalid"):
        authorization.finalize_authorization(  # type: ignore[arg-type]
            v1_lease, status="FAILED_EXHAUSTED"
        )
    assert authorization.lease_path(ROOT) != v1_authorization.lease_path(ROOT)


def test_v2_inspection_is_authorized_unprobed_label_closed_and_optimized() -> None:
    inspection = inspect_harp_stage90_v2(load_config(CONFIG))
    assert inspection["status"] == "EXECUTABLE_AUTHORIZED_UNPROBED"
    assert inspection["experiment_id"] == EXPERIMENT_ID
    assert inspection["execution_revision"] == EXECUTION_REVISION
    assert inspection["execution_authorized"] is True
    assert inspection["authorization_probed"] is False
    assert inspection["paths_resolved"] is False
    assert inspection["filesystem_mutations"] == 0
    assert inspection["development_labels_opened"] is False
    assert inspection["evaluation_labels_opened"] is False
    assert inspection["physical_plan"]["action_count"] == 738
    assert inspection["physical_plan"]["exact_nine_cell_count"] == 6642


def test_v2_runner_has_no_v1_identity_or_authorization_import() -> None:
    source = inspect.getsource(
        __import__(
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2.runner",
            fromlist=["*"],
        )
    )
    assert "fixed_bank_harp_router_v1.identity" not in source
    assert "fixed_bank_harp_router_v1.authorization" not in source


def test_v2_runner_authority_services_never_call_v1_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("v1 authority callback must remain unreachable from v2")

    monkeypatch.setattr(v1_authorization, "load_authorization", forbidden)
    monkeypatch.setattr(v1_authorization, "claim_authorization", forbidden)
    monkeypatch.setattr(v1_authorization, "finalize_authorization", forbidden)
    assert V2_RUNNER_SERVICES.load_authorization.__module__.endswith(
        "fixed_bank_harp_router_v2.authorization"
    )
    assert V2_RUNNER_SERVICES.claim_authorization.__module__.endswith(
        "fixed_bank_harp_router_v2.authorization"
    )
    assert V2_RUNNER_SERVICES.finalize_authorization.__module__.endswith(
        "fixed_bank_harp_router_v2.authorization"
    )
    planned = _write_synthetic_planned_config(tmp_path / "planned-v2.yaml")
    with pytest.raises(ProtocolError, match="new v2 single-use amendment"):
        V2_RUNNER_SERVICES.load_authorization(planned)

    lease_root = tmp_path / "v2-lease"
    monkeypatch.setattr(authorization, "lease_path", lambda *_: lease_root)
    typed = authorization.HarpV2Authorization(
        amendment_path=tmp_path / "v2-amendment.json",
        amendment_sha256="a" * 64,
        amendment_hash="b" * 64,
        input_binding_hash="c" * 64,
        scientific_contract_hash="d" * 64,
        workspace_registration_execution_contract_hash="e" * 64,
        source_snapshot_schema="v2-source",
        source_snapshot_manifest_sha256="f" * 64,
        source_snapshot_tree_sha256="1" * 64,
        source_snapshot_member_count=1,
    )
    lease = V2_RUNNER_SERVICES.claim_authorization(
        typed, admission_hash="2" * 64
    )
    final = V2_RUNNER_SERVICES.finalize_authorization(
        lease, status="FAILED_EXHAUSTED", error="test"
    )
    assert json.loads((final.root / "lease.json").read_text(encoding="utf-8"))[
        "status"
    ] == "FAILED_EXHAUSTED"


def test_v2_source_seal_transitively_includes_shared_optimized_numerics() -> None:
    relative = {
        path.relative_to(ROOT / "src").as_posix() for path in source_members(ROOT)
    }
    assert "midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v2/runner.py" in relative
    assert "midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v1/runner.py" in relative
    assert "midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v1/modeling.py" in relative
    assert "midogpp_thesis/cvae/routing/harp_action_model/ridge.py" in relative
    assert "midogpp_thesis/cvae/runtime/harp_probability_menu/indexed.py" in relative


def test_real_v2_preparation_receipt_and_publisher_bind_pre_and_final_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    config_path = repo / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    planned = _write_synthetic_planned_config(config_path)
    amendment_path = repo / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    amendment_path.parent.mkdir(parents=True)
    (repo / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(parents=True)
    prepared, manifest_sha, parent = _prepare_synthetic_v2_inputs(
        repo, monkeypatch
    )
    receipt_payload = read_json(
        prepared.cache_root
        / "manifests/harp_v2_consumed_test_preparation_receipt.json"
    )
    assert (
        receipt_payload["pre_manifest_cache_content_sha256"]
        != prepared.cache_content_sha256
    )

    bank = repo / "inputs/bank"
    generation = repo / "inputs/generation"
    bank.mkdir()
    generation.mkdir()
    fake_source = {
        "source_snapshot_schema": "test_source_v2",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 5,
        "source_snapshot_member_pattern": "test",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(authorization, "source_snapshot_identity", lambda *_: fake_source)
    monkeypatch.setattr(
        amendment_publisher,
        "validate_physical_inputs",
        lambda *_: SimpleNamespace(receipt_hash="5" * 64),
    )
    # The production constant is the canonical manifest digest.  This
    # synthetic fixture supplies its own authenticated manifest digest while
    # exercising the real cache loader and receipt validator unchanged.
    monkeypatch.setattr(
        amendment_publisher, "CANONICAL_MANIFEST_SHA256", manifest_sha
    )
    publication = amendment_publisher.publish_harp_v2_execution_amendment(
        planned,
        expert_bank_root=bank,
        generation_lock_root=generation,
        prepared_cache_root=prepared.cache_root,
        development_manifest_path=prepared.development_manifest_path,
        evaluation_manifest_path=prepared.evaluation_manifest_path,
        parent_ledger_path=parent,
        amendment_path=amendment_path,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date=authorization.AUTHORIZATION_DATE,
        repository_root=repo,
    )
    assert publication.preparation_receipt_hash == prepared.preparation_receipt_hash
    assert publication.amendment_path == amendment_path
    assert publication.amendment_sha256 == sha256_file(amendment_path)


def test_publisher_activation_and_workspace_authority_resolve_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    config_path = repo / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    planned = _write_synthetic_planned_config(config_path)
    amendment_path = repo / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    amendment_path.parent.mkdir(parents=True)
    (repo / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(parents=True)
    registry_path = repo / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    catalog_path = repo / authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    bank = repo / "inputs/bank"
    generation = repo / "inputs/generation"
    cache = repo / "inputs/cache"
    for path in (bank, generation, cache):
        path.mkdir(parents=True)
    development = repo / "inputs/development.csv"
    evaluation = repo / "inputs/evaluation.csv"
    parent = repo / "inputs/parent.json"
    development.write_text("development\n", encoding="utf-8")
    evaluation.write_text("evaluation\n", encoding="utf-8")
    atomic_json(parent, {"schema_version": "test_parent_v2"})
    content_base = {
        "schema_version": V2_CACHE_IDENTITY.content_schema,
        "members": {},
    }
    atomic_json(
        cache / "manifests/content_index.json",
        {**content_base, "content_index_hash": canonical_hash(content_base)},
    )
    fake_cache = SimpleNamespace(
        root=cache,
        content_sha256=canonical_hash(content_base),
        cache_hash="3" * 64,
        member_sha256={},
    )
    fake_source = {
        "source_snapshot_schema": "test_source_v2",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 5,
        "source_snapshot_member_pattern": "test",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(authorization, "source_snapshot_identity", lambda *_: fake_source)
    monkeypatch.setattr(amendment_publisher, "load_cache_index", lambda *_: fake_cache)
    monkeypatch.setattr(amendment_publisher, "_validate_preparation_receipt", lambda *_: "4" * 64)
    monkeypatch.setattr(
        amendment_publisher,
        "validate_physical_inputs",
        lambda *_: SimpleNamespace(receipt_hash="5" * 64),
    )
    receipt = amendment_publisher.publish_harp_v2_execution_amendment(
        planned,
        expert_bank_root=bank,
        generation_lock_root=generation,
        prepared_cache_root=cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        parent_ledger_path=parent,
        amendment_path=amendment_path,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date=authorization.AUTHORIZATION_DATE,
        repository_root=repo,
    )
    assert receipt.amendment_path == amendment_path

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["experiment"]["status"] = "diagnostic"
    raw["experiment"]["execution_authorized"] = True
    raw["inputs"].update(
        {
            "test_cache_content_sha256": fake_cache.content_sha256,
            "development_manifest_sha256": sha256_file(development),
            "evaluation_manifest_sha256": sha256_file(evaluation),
            "parent_ledger_sha256": sha256_file(parent),
            "execution_amendment_sha256": receipt.amendment_sha256,
        }
    )
    raw["claim_boundary"] = claim_boundary_payload(execution_authorized=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    contract = authorization.workspace_registration_execution_contract()
    registry = {
        "experiments": [
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": contract["stage"],
                "status": contract["status"],
                "claim_scope": contract["claim_scope"],
                "config_path": contract["config_path"],
                "output_artifact_id": OUTPUT_ARTIFACT_ID,
                "input_artifact_ids": list(contract["input_artifact_ids"]),
                "runner": {
                    "preparation_authority_gate": contract["preparation_authority_gate"],
                    "environment": contract["runner_environment"],
                    "argv": contract["runner_argv"],
                },
            }
        ]
    }
    catalog = {
        "artifacts": [
            {
                "artifact_id": OUTPUT_ARTIFACT_ID,
                "canonical_path": contract["output_canonical_path"],
            }
        ]
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    projection = {
        key: contract[key]
        for key in (
            "experiment_id", "stage", "status", "claim_scope", "config_path",
            "output_artifact_id", "output_canonical_path", "input_artifact_ids",
            "preparation_authority_gate", "run_recovery_strategy", "runner_argv",
            "runner_environment",
        )
    }
    member = SimpleNamespace(path=amendment_path, expected_sha256=receipt.amendment_sha256)
    resolved = validate_workspace_preparation_authority(
        repo_root=repo,
        experiment_id=EXPERIMENT_ID,
        config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
        input_artifact_ids=contract["input_artifact_ids"],
        registration_projection=projection,
        resolve_authority_member=lambda *_: member,
    )
    assert resolved.amendment_sha256 == receipt.amendment_sha256

    with pytest.raises(HarpV2WorkspaceAuthorityError):
        validate_workspace_preparation_authority(
            repo_root=repo,
            experiment_id=EXPERIMENT_ID,
            config_path=authorization.WORKSPACE_CONFIG_RELATIVE_PATH,
            input_artifact_ids=contract["input_artifact_ids"],
            registration_projection=projection,
            resolve_authority_member=lambda *_: SimpleNamespace(
                path=amendment_path,
                expected_sha256="0" * 64,
            ),
        )
