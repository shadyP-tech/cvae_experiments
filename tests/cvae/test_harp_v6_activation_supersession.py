from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import shutil

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6 import (
    activation,
    authorization,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.activation_attempts import (
    audit as activation_attempt_audit,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.activation_supersession import (
    ARCHIVED_AMENDMENT,
    ARCHIVED_JOURNAL,
    SUPERSESSION_CONFIRMATION,
    SUPERSESSION_RECEIPT,
    plan_harp_v6_activation_supersession,
    supersede_harp_v6_activation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.activation_transaction import (
    ActivationJournal,
    TRANSACTION_RELATIVE_PATH,
    inspect_activation_recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.identity import (
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_bytes, canonical_hash


ROOT = Path(__file__).resolve().parents[2]
OLD_SOURCE = {
    "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v6",
    "source_snapshot_manifest_sha256": "1" * 64,
    "source_snapshot_tree_sha256": "2" * 64,
    "source_snapshot_member_count": 218,
    "source_snapshot_member_pattern": "synthetic-old-source",
    "source_snapshot_excludes_bytecode_and_cache": True,
}
NEW_SOURCE = {
    **OLD_SOURCE,
    "source_snapshot_manifest_sha256": "3" * 64,
    "source_snapshot_tree_sha256": "4" * 64,
    "source_snapshot_member_pattern": "synthetic-repaired-source",
}


def _repository(tmp_path: Path) -> tuple[Path, ActivationJournal]:
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT / "experiments/midogpp",
        repository / "experiments/midogpp",
    )
    (
        repository / "artifacts/midogpp/90_oracles_and_diagnostics"
    ).mkdir(parents=True)

    config_path = repository / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    registry_path = repository / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    catalog_path = repository / authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH
    amendment_path = repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    body = {
        "schema_version": authorization.EXECUTION_AMENDMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "execution_authorized": True,
        "single_use": True,
        "authorization_exhausted": False,
        "consumed_test_reuse": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "output_deletion_restores_authority": False,
        "source_snapshot_identity": OLD_SOURCE,
    }
    amendment = {**body, "amendment_hash": canonical_hash(body)}
    amendment_bytes = canonical_bytes(amendment) + b"\n"
    amendment_path.write_bytes(amendment_bytes)

    values = {
        "repository_root": repository.resolve(),
        "activation_plan_hash": "5" * 64,
        "config_path": config_path.resolve(),
        "registry_path": registry_path.resolve(),
        "catalog_path": catalog_path.resolve(),
        "amendment_path": amendment_path.resolve(),
        "original_config_bytes": config_path.read_bytes(),
        "original_registry_bytes": registry_path.read_bytes(),
        "original_catalog_bytes": catalog_path.read_bytes(),
        "final_config_bytes": b"authorized\n" + config_path.read_bytes(),
        "final_registry_bytes": b"authorized\n" + registry_path.read_bytes(),
        "final_catalog_bytes": b"authorized\n" + catalog_path.read_bytes(),
        "amendment_bytes": amendment_bytes,
        "amendment_sha256": hashlib.sha256(amendment_bytes).hexdigest(),
    }
    provisional = ActivationJournal(**values, journal_hash="")
    journal = replace(
        provisional,
        journal_hash=canonical_hash(provisional.payload_without_hash()),
    )
    journal_path = repository / TRANSACTION_RELATIVE_PATH
    journal_path.write_bytes(journal.to_bytes())
    return repository, journal


def _use_repaired_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        activation_attempt_audit,
        "source_snapshot_identity",
        lambda _root=None: NEW_SOURCE,
    )


def test_supersession_plan_and_wrong_token_are_mutation_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    _use_repaired_source(monkeypatch)
    watched = {
        path: path.read_bytes()
        for path in (
            journal.config_path,
            journal.catalog_path,
            journal.registry_path,
            journal.amendment_path,
            repository / TRANSACTION_RELATIVE_PATH,
        )
    }

    plan = plan_harp_v6_activation_supersession(repository)
    assert plan.to_payload()["filesystem_mutations"] == 0
    assert plan.to_payload()["source_snapshot_changed"] is True
    with pytest.raises(ProtocolError, match="confirmation"):
        supersede_harp_v6_activation(plan, confirmation="wrong")
    assert {path: path.read_bytes() for path in watched} == watched
    assert not plan.archive_root.exists()


def test_normal_recovery_blocks_source_drift_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    _use_repaired_source(monkeypatch)
    watched = {
        path: path.read_bytes()
        for path in (
            journal.config_path,
            journal.catalog_path,
            journal.registry_path,
            journal.amendment_path,
            repository / TRANSACTION_RELATIVE_PATH,
        )
    }

    inspection = activation.inspect_harp_v6_activation_recovery(repository)
    assert inspection is not None
    assert inspection["status"] == "SUPERSESSION_REQUIRED_SOURCE_SNAPSHOT_DRIFT"
    assert inspection["normal_recovery_allowed"] is False
    with pytest.raises(ProtocolError, match="archive the rolled-back activation"):
        activation.recover_harp_v6_activation(
            repository,
            confirmation=activation.ACTIVATION_CONFIRMATION,
        )
    assert {path: path.read_bytes() for path in watched} == watched


def test_supersession_archives_exact_attempt_before_retiring_live_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    _use_repaired_source(monkeypatch)
    originals = {
        journal.config_path: journal.original_config_bytes,
        journal.catalog_path: journal.original_catalog_bytes,
        journal.registry_path: journal.original_registry_bytes,
    }
    plan = plan_harp_v6_activation_supersession(repository)
    receipt = supersede_harp_v6_activation(
        plan,
        confirmation=SUPERSESSION_CONFIRMATION,
    ).to_payload()

    assert receipt["status"] == "ROLLED_BACK_ACTIVATION_ARCHIVED_AND_SUPERSEDED"
    assert (plan.archive_root / ARCHIVED_JOURNAL).read_bytes() == journal.to_bytes()
    assert (plan.archive_root / ARCHIVED_AMENDMENT).read_bytes() == journal.amendment_bytes
    assert (plan.archive_root / SUPERSESSION_RECEIPT).is_file()
    assert not journal.amendment_path.exists()
    assert not (repository / TRANSACTION_RELATIVE_PATH).exists()
    assert inspect_activation_recovery(repository) is None
    assert {path: path.read_bytes() for path in originals} == originals
    assert not authorization.lease_path(repository).exists()
    assert not (repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH).exists()


def test_supersession_resumes_after_amendment_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    _use_repaired_source(monkeypatch)
    plan = plan_harp_v6_activation_supersession(repository)

    def fail(point: str) -> None:
        if point == "active_amendment_retired":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        supersede_harp_v6_activation(
            plan,
            confirmation=SUPERSESSION_CONFIRMATION,
            _fault_injector=fail,
        )
    assert not journal.amendment_path.exists()
    assert (repository / TRANSACTION_RELATIVE_PATH).is_file()

    resumed = plan_harp_v6_activation_supersession(repository)
    supersede_harp_v6_activation(
        resumed,
        confirmation=SUPERSESSION_CONFIRMATION,
    )
    assert not (repository / TRANSACTION_RELATIVE_PATH).exists()


def test_supersession_rejects_same_source_or_consumed_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    monkeypatch.setattr(
        activation_attempt_audit,
        "source_snapshot_identity",
        lambda _root=None: OLD_SOURCE,
    )
    with pytest.raises(ProtocolError, match="source snapshot is unchanged"):
        plan_harp_v6_activation_supersession(repository)

    _use_repaired_source(monkeypatch)
    lease = authorization.lease_path(repository)
    lease.mkdir()
    with pytest.raises(ProtocolError, match="authorization lease"):
        plan_harp_v6_activation_supersession(repository)
    assert journal.amendment_path.is_file()
