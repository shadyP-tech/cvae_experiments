from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser, main as diagnostics_main
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import (
    amendment_publisher as publisher,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import preparation
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.authorization import (
    validate_execution_amendment_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.identity import (
    claim_boundary_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.input_surfaces import (
    CONTENT_INDEX,
    load_cache_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.physical_menu import (
    HarpPhysicalInputReceipt,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
)


def _as_synthetic_planned_config(
    config,
    *,
    clear_prepared_hashes: bool = False,
):
    expected_hashes = {
        **dict(config.expected_hashes),
        "execution_amendment_sha256": None,
    }
    if clear_prepared_hashes:
        expected_hashes.update(
            {
                "test_cache_content_sha256": None,
                "development_manifest_sha256": None,
                "evaluation_manifest_sha256": None,
                "parent_ledger_sha256": None,
            }
        )
    return replace(
        config,
        expected_hashes=expected_hashes,
        execution_authorized=False,
        claim_boundary=claim_boundary_payload(execution_authorized=False),
    )


def _bound_config():
    planned = _as_synthetic_planned_config(load_config(CONFIG))
    return replace(
        planned,
        expected_hashes={
            **dict(planned.expected_hashes),
            "test_cache_content_sha256": "a" * 64,
            "development_manifest_sha256": "b" * 64,
            "evaluation_manifest_sha256": "c" * 64,
            "parent_ledger_sha256": "d" * 64,
            "execution_amendment_sha256": None,
        },
    )


def _synthetic_planned_yaml() -> str:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["status"] = "planned"
    raw["experiment"]["execution_authorized"] = False
    raw["inputs"]["execution_amendment_sha256"] = None
    raw["claim_boundary"] = claim_boundary_payload(execution_authorized=False)
    return yaml.safe_dump(raw, sort_keys=False)


def _place_synthetic_planned_config(repository: Path):
    path = (
        repository
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
        / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_synthetic_planned_yaml(), encoding="utf-8")
    return load_config(path)


def test_publisher_creates_exactly_one_canonical_file_without_activating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _place_synthetic_planned_config(tmp_path)
    bound_config = replace(
        config,
        expected_hashes=dict(_bound_config().expected_hashes),
    )
    contract_root = (
        tmp_path
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v1"
    )
    contract_root.mkdir(parents=True)
    amendment = contract_root / publisher.AMENDMENT_FILENAME
    validated = publisher._ValidatedPublisherInputs(
        config=bound_config,
        cache=SimpleNamespace(),  # not reopened after the validation boundary
        preparation_receipt_hash="e" * 64,
        physical_input_receipt_hash="f" * 64,
    )
    calls = []

    def validate(*args, **kwargs):
        calls.append((args, kwargs))
        return validated

    monkeypatch.setattr(publisher, "_validate_inputs", validate)
    monkeypatch.setattr(
        publisher.authorization,
        "source_snapshot_identity",
        lambda _root: {
            "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v1",
            "source_snapshot_manifest_sha256": "1" * 64,
            "source_snapshot_tree_sha256": "2" * 64,
            "source_snapshot_member_count": 1,
            "source_snapshot_member_pattern": "test_closed_world_source",
            "source_snapshot_excludes_bytecode_and_cache": True,
        },
    )
    before = tuple(contract_root.iterdir())
    receipt = publisher.publish_harp_execution_amendment(
        config,
        expert_bank_root=tmp_path / "unused-bank",
        generation_lock_root=tmp_path / "unused-generation",
        prepared_cache_root=tmp_path / "unused-cache",
        development_manifest_path=tmp_path / "unused-dev.csv",
        evaluation_manifest_path=tmp_path / "unused-eval.csv",
        parent_ledger_path=tmp_path / "unused-parent.json",
        amendment_path=amendment,
        authorization_basis=publisher.AUTHORIZATION_BASIS,
        authorization_date=publisher.AUTHORIZATION_DATE,
        repository_root=tmp_path,
    )
    after = tuple(contract_root.iterdir())
    assert before == ()
    assert after == (amendment,)
    assert len(calls) == 1
    assert receipt.amendment_path == amendment
    assert receipt.amendment_sha256 == sha256_file(amendment)
    assert receipt.to_payload()["configuration_or_registry_activated"] is False
    assert receipt.to_payload()["authorization_lease_claimed"] is False
    assert receipt.to_payload()["output_artifact_created"] is False
    assert receipt.to_payload()["experiment_launched"] is False
    payload = json.loads(amendment.read_text(encoding="utf-8"))
    authorized = replace(
        bound_config,
        execution_authorized=True,
        claim_boundary=claim_boundary_payload(execution_authorized=True),
    )
    reconstructed = validate_execution_amendment_payload(
        payload, authorized, repo_root=tmp_path
    )
    assert reconstructed.amendment_hash == receipt.amendment_hash
    original = amendment.read_bytes()
    with pytest.raises(ProtocolError, match="already exists"):
        publisher.publish_harp_execution_amendment(
            config,
            expert_bank_root=tmp_path / "unused-bank",
            generation_lock_root=tmp_path / "unused-generation",
            prepared_cache_root=tmp_path / "unused-cache",
            development_manifest_path=tmp_path / "unused-dev.csv",
            evaluation_manifest_path=tmp_path / "unused-eval.csv",
            parent_ledger_path=tmp_path / "unused-parent.json",
            amendment_path=amendment,
            authorization_basis=publisher.AUTHORIZATION_BASIS,
            authorization_date=publisher.AUTHORIZATION_DATE,
            repository_root=tmp_path,
        )
    assert amendment.read_bytes() == original


@pytest.mark.parametrize(
    ("basis", "date", "message"),
    (
        ("implementation_request", publisher.AUTHORIZATION_DATE, "basis"),
        (publisher.AUTHORIZATION_BASIS, "2026-08-29", "date"),
    ),
)
def test_publisher_requires_exact_explicit_authorization_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    basis: str,
    date: str,
    message: str,
) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreachable")

    monkeypatch.setattr(publisher, "_validate_inputs", forbidden)
    with pytest.raises(ProtocolError, match=message):
        publisher.publish_harp_execution_amendment(
            _as_synthetic_planned_config(load_config(CONFIG)),
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "dev.csv",
            evaluation_manifest_path=tmp_path / "eval.csv",
            parent_ledger_path=tmp_path / "parent.json",
            amendment_path=tmp_path / publisher.AMENDMENT_FILENAME,
            authorization_basis=basis,
            authorization_date=date,
            repository_root=tmp_path,
        )
    assert called is False
    assert tuple(tmp_path.iterdir()) == ()


def test_publisher_rejects_nonregistered_path_and_existing_lease_before_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreachable")

    monkeypatch.setattr(publisher, "_validate_inputs", forbidden)
    config = _place_synthetic_planned_config(tmp_path)
    outside = tmp_path / publisher.AMENDMENT_FILENAME
    with pytest.raises(ProtocolError, match="registered contract member"):
        publisher.publish_harp_execution_amendment(
            config,
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "dev.csv",
            evaluation_manifest_path=tmp_path / "eval.csv",
            parent_ledger_path=tmp_path / "parent.json",
            amendment_path=outside,
            authorization_basis=publisher.AUTHORIZATION_BASIS,
            authorization_date=publisher.AUTHORIZATION_DATE,
            repository_root=tmp_path,
        )
    assert called is False
    assert not outside.exists()

    registered = (
        tmp_path
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v1"
        / publisher.AMENDMENT_FILENAME
    )
    registered.parent.mkdir(parents=True)
    lease = publisher.authorization.lease_path(tmp_path)
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.mkdir()
    with pytest.raises(ProtocolError, match="lease already exists"):
        publisher.publish_harp_execution_amendment(
            config,
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "dev.csv",
            evaluation_manifest_path=tmp_path / "eval.csv",
            parent_ledger_path=tmp_path / "parent.json",
            amendment_path=registered,
            authorization_basis=publisher.AUTHORIZATION_BASIS,
            authorization_date=publisher.AUTHORIZATION_DATE,
            repository_root=tmp_path,
        )
    assert called is False
    assert not registered.exists()


def test_publisher_rejects_arbitrary_planned_config_before_inputs_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "altered-policy-config.yaml"
    copied.write_text(_synthetic_planned_yaml(), encoding="utf-8")
    config = load_config(copied)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreachable")

    monkeypatch.setattr(publisher, "_validate_inputs", forbidden)
    registered = (
        tmp_path
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v1"
        / publisher.AMENDMENT_FILENAME
    )
    with pytest.raises(ProtocolError, match="registered HARP v1 member"):
        publisher.publish_harp_execution_amendment(
            config,
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "dev.csv",
            evaluation_manifest_path=tmp_path / "eval.csv",
            parent_ledger_path=tmp_path / "parent.json",
            amendment_path=registered,
            authorization_basis=publisher.AUTHORIZATION_BASIS,
            authorization_date=publisher.AUTHORIZATION_DATE,
            repository_root=tmp_path,
        )
    assert called is False
    assert not registered.exists()


def test_publisher_reloads_registered_yaml_and_rejects_forged_typed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _place_synthetic_planned_config(tmp_path)
    forged = replace(
        registered,
        policy=replace(registered.policy, gain_threshold=0.125),
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreachable")

    monkeypatch.setattr(publisher, "_validate_inputs", forbidden)
    amendment = (
        tmp_path
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v1"
        / publisher.AMENDMENT_FILENAME
    )
    with pytest.raises(ProtocolError, match="registered YAML bytes"):
        publisher.publish_harp_execution_amendment(
            forged,
            expert_bank_root=tmp_path / "bank",
            generation_lock_root=tmp_path / "generation",
            prepared_cache_root=tmp_path / "cache",
            development_manifest_path=tmp_path / "dev.csv",
            evaluation_manifest_path=tmp_path / "eval.csv",
            parent_ledger_path=tmp_path / "parent.json",
            amendment_path=amendment,
            authorization_basis=publisher.AUTHORIZATION_BASIS,
            authorization_date=publisher.AUTHORIZATION_DATE,
            repository_root=tmp_path,
        )
    assert called is False
    assert not amendment.exists()


def _tiny_prepared_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    manifest = tmp_path / "canonical.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case_id", "center", "split", "label")
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest_sha = sha256_file(manifest)
    rows_by_center = {}
    embeddings_by_center = {}
    for center in CENTERS:
        rows = tuple(
            preparation.CanonicalFrameRow(
                center=center,
                case_id=case,
                sample_id=preparation._evaluation_row_id(manifest_sha, global_index),
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
    frame = preparation.CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash="1" * 64,
        row_order_hash="2" * 64,
        source_member_sha256={},
    )
    parent = tmp_path / "parent.json"
    parent.write_text("{}\n", encoding="utf-8")
    parent_sha = sha256_file(parent)
    monkeypatch.setattr(preparation, "EXPECTED_ROW_COUNT", len(raw_rows))
    monkeypatch.setattr(
        preparation, "load_canonical_label_blind_cache", lambda _root: frame
    )
    cache = tmp_path / "cache"
    development = tmp_path / "development.csv"
    evaluation = tmp_path / "evaluation.csv"
    prepared = preparation.prepare_harp_consumed_test_inputs(
        canonical_cache_root=tmp_path / "unused",
        canonical_manifest_path=manifest,
        parent_ledger_path=parent,
        cache_root=cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        expected_manifest_sha256=manifest_sha,
        expected_parent_ledger_sha256=parent_sha,
    )
    return frame, manifest_sha, parent, parent_sha, cache, development, evaluation, prepared


def test_publisher_revalidates_preparation_content_roles_and_physical_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        frame,
        manifest_sha,
        parent,
        parent_sha,
        cache_root,
        development,
        evaluation,
        prepared,
    ) = _tiny_prepared_inputs(tmp_path, monkeypatch)
    bank = tmp_path / "bank"
    generation = tmp_path / "generation"
    bank.mkdir()
    generation.mkdir()
    monkeypatch.setattr(publisher, "CANONICAL_CACHE_CONTENT_HASH", frame.cache_content_hash)
    monkeypatch.setattr(publisher, "CANONICAL_CACHE_ROW_ORDER_HASH", frame.row_order_hash)
    monkeypatch.setattr(publisher, "CANONICAL_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(publisher, "CANONICAL_PARENT_LEDGER_SHA256", parent_sha)
    observed = []

    def validate_physical(config, cache):
        observed.append((config, cache))
        return HarpPhysicalInputReceipt(
            bank_semantic_lock_hash=str(config.expected_hashes["expert_bank_lock_hash"]),
            generation_semantic_lock_hash=str(
                config.expected_hashes["generation_lock_hash"]
            ),
            expert_bank_index_sha256="3" * 64,
            generation_lock_file_sha256="4" * 64,
            classifier_config_hash="5" * 64,
            classifier_contract_sha256="6" * 64,
            cache_hash=cache.cache_hash,
            cache_content_sha256=cache.content_sha256,
            receipt_hash="7" * 64,
        )

    monkeypatch.setattr(publisher, "validate_physical_inputs", validate_physical)
    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.input_surfaces._read_label_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("publisher must remain label-blind")
        ),
    )
    config = _as_synthetic_planned_config(
        load_config(CONFIG),
        clear_prepared_hashes=True,
    )
    validated = publisher._validate_inputs(
        config,
        expert_bank_root=bank,
        generation_lock_root=generation,
        prepared_cache_root=cache_root,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        parent_ledger_path=parent,
        amendment_path=tmp_path / publisher.AMENDMENT_FILENAME,
    )
    assert validated.preparation_receipt_hash == json.loads(
        (cache_root / preparation.PREPARATION_RECEIPT).read_text(encoding="utf-8")
    )["receipt_hash"]
    assert validated.physical_input_receipt_hash == "7" * 64
    assert validated.config.expected_hashes["test_cache_content_sha256"] == (
        prepared.cache_content_sha256
    )
    assert observed[0][0].resolved_path("expert_bank_root") == bank
    assert observed[0][0].resolved_path("generation_lock_root") == generation

    development.write_text(
        development.read_text(encoding="utf-8").replace(",0,harp_", ",1,harp_", 1),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="preparation receipt/content"):
        publisher._validate_inputs(
            config,
            expert_bank_root=bank,
            generation_lock_root=generation,
            prepared_cache_root=cache_root,
            development_manifest_path=development,
            evaluation_manifest_path=evaluation,
            parent_ledger_path=parent,
            amendment_path=tmp_path / publisher.AMENDMENT_FILENAME,
        )


def test_cli_exposes_publisher_without_an_artifact_output_argument() -> None:
    parsed = build_parser().parse_args(
        [
            "publish-fixed-bank-harp-router-v1-amendment",
            "--config",
            str(CONFIG),
            "--expert-bank-root",
            "/exact/bank",
            "--generation-lock-root",
            "/exact/generation",
            "--prepared-cache-root",
            "/exact/cache",
            "--development-manifest",
            "/exact/development.csv",
            "--evaluation-manifest",
            "/exact/evaluation.csv",
            "--parent-ledger",
            "/exact/parent.json",
            "--amendment-path",
            f"/exact/{publisher.AMENDMENT_FILENAME}",
            "--authorization-basis",
            publisher.AUTHORIZATION_BASIS,
            "--authorization-date",
            publisher.AUTHORIZATION_DATE,
            "--repository-root",
            str(ROOT),
        ]
    )
    assert parsed.surface == "publish-fixed-bank-harp-router-v1-amendment"
    assert not hasattr(parsed, "artifact_root")


def test_cli_dispatches_only_to_the_amendment_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed = {}

    def publish(config, **kwargs):
        observed["config"] = config
        observed.update(kwargs)
        return SimpleNamespace(
            to_payload=lambda: {
                "schema_version": "test_harp_publication_receipt_v1",
                "only_amendment_file_created": True,
            }
        )

    monkeypatch.setattr(publisher, "publish_harp_execution_amendment", publish)
    arguments = [
        "publish-fixed-bank-harp-router-v1-amendment",
        "--config",
        str(CONFIG),
        "--expert-bank-root",
        "/exact/bank",
        "--generation-lock-root",
        "/exact/generation",
        "--prepared-cache-root",
        "/exact/cache",
        "--development-manifest",
        "/exact/development.csv",
        "--evaluation-manifest",
        "/exact/evaluation.csv",
        "--parent-ledger",
        "/exact/parent.json",
        "--amendment-path",
        f"/exact/{publisher.AMENDMENT_FILENAME}",
        "--authorization-basis",
        publisher.AUTHORIZATION_BASIS,
        "--authorization-date",
        publisher.AUTHORIZATION_DATE,
        "--repository-root",
        str(tmp_path),
    ]
    assert diagnostics_main(arguments) == 0
    assert observed["config"].execution_authorized is True
    assert observed["repository_root"] == str(tmp_path)
    assert json.loads(capsys.readouterr().out)["only_amendment_file_created"] is True


def test_historical_v1_publisher_cli_is_fail_closed_before_mutation(
    tmp_path: Path,
) -> None:
    arguments = [
        "publish-fixed-bank-harp-router-v1-amendment",
        "--config",
        str(CONFIG),
        "--expert-bank-root",
        str(tmp_path / "bank"),
        "--generation-lock-root",
        str(tmp_path / "generation"),
        "--prepared-cache-root",
        str(tmp_path / "cache"),
        "--development-manifest",
        str(tmp_path / "development.csv"),
        "--evaluation-manifest",
        str(tmp_path / "evaluation.csv"),
        "--parent-ledger",
        str(tmp_path / "parent.json"),
        "--amendment-path",
        str(tmp_path / publisher.AMENDMENT_FILENAME),
        "--authorization-basis",
        publisher.AUTHORIZATION_BASIS,
        "--authorization-date",
        publisher.AUTHORIZATION_DATE,
        "--repository-root",
        str(tmp_path),
    ]
    with pytest.raises(ProtocolError, match="requires the planned config"):
        diagnostics_main(arguments)
    assert tuple(tmp_path.iterdir()) == ()
