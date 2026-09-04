from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.activation_attempts import (
    active_audit as supersession_audit,
    admin_snapshot,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.activation_attempts.contracts import (
    ACTIVE_SUPERSESSION_CONFIRMATION,
    ACTIVE_SUPERSESSION_RECEIPT,
    ARCHIVED_ADMIN_CONTENT,
    ARCHIVED_AMENDMENT,
    ARCHIVED_FINAL_CATALOG,
    ARCHIVED_FINAL_CONFIG,
    ARCHIVED_FINAL_REGISTRY,
    ARCHIVED_JOURNAL,
    ARCHIVED_RETIREMENT_FENCE,
    HarpV14ActiveActivationSupersessionPlan,
    RETIRED_ADMIN_OUTPUT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.activation_supersession import (
    plan_harp_v14_active_activation_supersession,
    supersede_harp_v14_active_activation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.activation_transaction import (
    ActivationJournal,
    TRANSACTION_RELATIVE_PATH,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.config import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.identity import (
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_bytes, canonical_hash


OLD_SOURCE = {
    "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v14",
    "source_snapshot_manifest_sha256": "1" * 64,
    "source_snapshot_tree_sha256": "2" * 64,
    "source_snapshot_member_count": 220,
    "source_snapshot_member_pattern": "sealed-old-source",
    "source_snapshot_excludes_bytecode_and_cache": True,
}
NEW_SOURCE = {
    **OLD_SOURCE,
    "source_snapshot_manifest_sha256": "3" * 64,
    "source_snapshot_tree_sha256": "4" * 64,
    "source_snapshot_member_pattern": "repaired-source",
}


def _repository(tmp_path: Path) -> tuple[Path, ActivationJournal]:
    repository = tmp_path / "repository"
    config = repository / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    registry = repository / authorization.WORKSPACE_REGISTRY_RELATIVE_PATH
    catalog = repository / authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH
    amendment = repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    for path in (config, registry, catalog, amendment):
        path.parent.mkdir(parents=True, exist_ok=True)

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
    amendment_payload = {**body, "amendment_hash": canonical_hash(body)}
    amendment_bytes = canonical_bytes(amendment_payload) + b"\n"
    amendment.write_bytes(amendment_bytes)

    originals = {
        "config": b"planned-config\n",
        "registry": b"planned-registry\n",
        "catalog": b"planned-catalog\n",
    }
    finals = {
        "config": (
            "runtime:\n"
            f"  scratch_root: {(repository / 'absent-scratch').as_posix()}\n"
        ).encode("utf-8"),
        "registry": b"activated-registry\n",
        "catalog": b"activated-catalog\n",
    }
    config.write_bytes(finals["config"])
    registry.write_bytes(finals["registry"])
    catalog.write_bytes(finals["catalog"])
    values = {
        "repository_root": repository.resolve(),
        "activation_plan_hash": "5" * 64,
        "config_path": config.resolve(),
        "registry_path": registry.resolve(),
        "catalog_path": catalog.resolve(),
        "amendment_path": amendment.resolve(),
        "original_config_bytes": originals["config"],
        "original_registry_bytes": originals["registry"],
        "original_catalog_bytes": originals["catalog"],
        "final_config_bytes": finals["config"],
        "final_registry_bytes": finals["registry"],
        "final_catalog_bytes": finals["catalog"],
        "amendment_bytes": amendment_bytes,
        "amendment_sha256": hashlib.sha256(amendment_bytes).hexdigest(),
    }
    provisional = ActivationJournal(**values, journal_hash="")
    journal = replace(
        provisional,
        journal_hash=canonical_hash(provisional.payload_without_hash()),
    )
    journal_path = repository / TRANSACTION_RELATIVE_PATH
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_bytes(journal.to_bytes())
    (repository / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(
        parents=True,
        exist_ok=True,
    )
    return repository, journal


def _fake_active_config(scratch: Path) -> SimpleNamespace:
    return SimpleNamespace(runtime={"scratch_root": str(scratch)})


@pytest.mark.parametrize("malformed_identity", ({"nested": "id"}, ["id"], 7, None))
def test_admin_manifest_rejects_non_string_artifact_ids_as_protocol_errors(
    malformed_identity: object,
) -> None:
    rows = [
        {"artifact_id": artifact_id, "exists": True}
        for artifact_id in INPUT_ARTIFACT_IDS
    ]
    rows[0]["artifact_id"] = malformed_identity
    payload = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": rows,
    }

    with pytest.raises(ProtocolError, match="artifact identity is malformed"):
        admin_snapshot.validate_admin_input_manifest(
            json.dumps(payload).encode("utf-8")
        )


def _plan_absent_output(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> HarpV14ActiveActivationSupersessionPlan:
    monkeypatch.setattr(
        supersession_audit,
        "_require_active_workspace",
        lambda _boundary, _journal: _fake_active_config(
            repository / "absent-scratch"
        ),
    )
    monkeypatch.setattr(
        supersession_audit,
        "source_snapshot_identity",
        lambda _root=None: NEW_SOURCE,
    )
    return plan_harp_v14_active_activation_supersession(repository)


def _admin_plan(repository: Path, journal: ActivationJournal) -> HarpV14ActiveActivationSupersessionPlan:
    output = repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    for relative in ("manifests", "provenance", "reports", "tables"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    files = {
        "config.resolved.yaml": b"resolved-config\n",
        "provenance/input_artifacts.json": b"{}\n",
    }
    for relative, raw in files.items():
        (output / relative).write_bytes(raw)
    manifest_body = {
        "schema_version": "midogpp_harp_v14_workspace_admin_snapshot_v1",
        "experiment_id": EXPERIMENT_ID,
        "state": "WORKSPACE_ADMIN_PRISTINE",
        "output_root": authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
        "directories": ["manifests", "provenance", "reports", "tables"],
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            for relative, raw in sorted(files.items())
        ],
        "scientific_files_present": False,
        "labels_opened": False,
        "routes_sealed": False,
    }
    manifest = {**manifest_body, "snapshot_hash": canonical_hash(manifest_body)}
    provisional = HarpV14ActiveActivationSupersessionPlan(
        repository_root=repository.resolve(),
        journal=journal,
        archive_root=(
            repository
            / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
            "harp_router_v14/superseded_active_activations"
            / journal.journal_hash
        ),
        output_root=output,
        scratch_root=repository / "absent-scratch",
        prior_source_snapshot=OLD_SOURCE,
        replacement_source_snapshot=NEW_SOURCE,
        amendment_hash="6" * 64,
        admin_snapshot_manifest=manifest,
        admin_snapshot_files=files,
        supersession_plan_hash="",
    )
    return replace(
        provisional,
        supersession_plan_hash=canonical_hash(provisional.hash_payload()),
    )


def _plan_admin_output(
    repository: Path,
    journal: ActivationJournal,
    monkeypatch: pytest.MonkeyPatch,
) -> HarpV14ActiveActivationSupersessionPlan:
    """Exercise the real planner over an exact workspace-admin tree."""

    _admin_plan(repository, journal)
    monkeypatch.setattr(
        supersession_audit,
        "_require_active_workspace",
        lambda _boundary, _journal: _fake_active_config(
            repository / "absent-scratch"
        ),
    )
    monkeypatch.setattr(
        supersession_audit,
        "source_snapshot_identity",
        lambda _root=None: NEW_SOURCE,
    )
    # The fixture deliberately carries compact placeholder administrative
    # bytes. The planner's exact inventory, hashing, archival, and recovery
    # checks remain real; only the unrelated production-config semantics are
    # isolated here.
    monkeypatch.setattr(
        admin_snapshot,
        "validate_admin_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        admin_snapshot,
        "validate_admin_input_manifest",
        lambda *_args, **_kwargs: None,
    )
    plan = plan_harp_v14_active_activation_supersession(repository)
    assert plan.admin_snapshot_manifest["state"] == "WORKSPACE_ADMIN_PRISTINE"
    assert plan.recovery_state["output_location"] == "live"
    return plan


def test_active_plan_is_mutation_free_and_requires_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    monkeypatch.setattr(
        supersession_audit,
        "_require_active_workspace",
        lambda _boundary, _journal: _fake_active_config(repository / "scratch"),
    )
    monkeypatch.setattr(
        supersession_audit,
        "source_snapshot_identity",
        lambda _root=None: NEW_SOURCE,
    )
    watched = {
        path: path.read_bytes()
        for path in (
            journal.config_path,
            journal.registry_path,
            journal.catalog_path,
            journal.amendment_path,
            repository / TRANSACTION_RELATIVE_PATH,
        )
    }

    plan = plan_harp_v14_active_activation_supersession(repository)
    assert plan.to_payload()["filesystem_mutations"] == 0
    assert plan.to_payload()["observed_states"]["registry"] == "final"
    assert {path: path.read_bytes() for path in watched} == watched
    assert not plan.archive_root.exists()

    monkeypatch.setattr(
        supersession_audit,
        "source_snapshot_identity",
        lambda _root=None: OLD_SOURCE,
    )
    with pytest.raises(ProtocolError, match="source snapshot is unchanged"):
        plan_harp_v14_active_activation_supersession(repository)


def test_active_supersession_archives_then_restores_registry_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _admin_plan(repository, journal)
    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14."
        "activation_attempts.active_transaction.build_active_supersession_plan",
        lambda _boundary: plan,
    )
    observed: list[tuple[str, bytes]] = []

    def observe(point: str) -> None:
        observed.append((point, journal.registry_path.read_bytes()))

    receipt = supersede_harp_v14_active_activation(
        plan,
        confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
        _fault_injector=observe,
    ).to_payload()

    assert receipt["old_authority_may_not_execute"] is True
    assert observed[0] == ("archive_durable", journal.final_registry_bytes)
    assert observed[1] == (
        "retirement_fence_durable",
        journal.final_registry_bytes,
    )
    assert observed[2] == ("registry_restored", journal.original_registry_bytes)
    assert journal.registry_path.read_bytes() == journal.original_registry_bytes
    assert journal.catalog_path.read_bytes() == journal.original_catalog_bytes
    assert journal.config_path.read_bytes() == journal.original_config_bytes
    for name, raw in (
        (ARCHIVED_JOURNAL, journal.to_bytes()),
        (ARCHIVED_AMENDMENT, journal.amendment_bytes),
        (ARCHIVED_FINAL_CONFIG, journal.final_config_bytes),
        (ARCHIVED_FINAL_REGISTRY, journal.final_registry_bytes),
        (ARCHIVED_FINAL_CATALOG, journal.final_catalog_bytes),
    ):
        assert (plan.archive_root / name).read_bytes() == raw
    assert (plan.archive_root / ARCHIVED_ADMIN_CONTENT / "config.resolved.yaml").is_file()
    assert (plan.archive_root / RETIRED_ADMIN_OUTPUT / "config.resolved.yaml").is_file()
    assert (plan.archive_root / ACTIVE_SUPERSESSION_RECEIPT).is_file()
    assert not plan.output_root.exists()
    assert not journal.amendment_path.exists()
    assert not (repository / TRANSACTION_RELATIVE_PATH).exists()


def test_active_supersession_rejects_wrong_token_without_mutation(
    tmp_path: Path,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _admin_plan(repository, journal)
    with pytest.raises(ProtocolError, match="confirmation"):
        supersede_harp_v14_active_activation(plan, confirmation="wrong")
    assert journal.registry_path.read_bytes() == journal.final_registry_bytes
    assert journal.amendment_path.is_file()
    assert plan.output_root.is_dir()
    assert not plan.archive_root.exists()


def test_active_plan_rejects_lease_and_non_admin_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _journal = _repository(tmp_path)
    monkeypatch.setattr(
        supersession_audit,
        "_require_active_workspace",
        lambda _boundary, _journal: _fake_active_config(repository / "scratch"),
    )
    monkeypatch.setattr(
        supersession_audit,
        "source_snapshot_identity",
        lambda _root=None: NEW_SOURCE,
    )
    lease = authorization.lease_path(repository)
    lease.mkdir()
    with pytest.raises(ProtocolError, match="authorization lease"):
        plan_harp_v14_active_activation_supersession(repository)
    lease.rmdir()

    output = repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    for relative in ("manifests", "provenance", "reports", "tables"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_bytes(b"invalid-but-admin\n")
    (output / "provenance/input_artifacts.json").write_bytes(b"{}\n")
    (output / "manifests/routes.json").write_bytes(b"route-state\n")
    with pytest.raises(ProtocolError, match="workspace-admin pristine"):
        plan_harp_v14_active_activation_supersession(repository)


@pytest.mark.parametrize(
    "fault_point",
    (
        "archive_durable",
        "retirement_fence_durable",
        "registry_restored",
        "catalog_restored",
        "config_restored",
        "admin_output_retired",
        "active_amendment_retired",
        "terminal_receipt_durable",
        "retirement_fence_archived",
    ),
)
def test_every_injected_crash_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _plan_absent_output(repository, monkeypatch)

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected-crash")

    with pytest.raises(RuntimeError, match="injected-crash"):
        supersede_harp_v14_active_activation(
            plan,
            confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
            _fault_injector=fail,
        )

    resumed = plan_harp_v14_active_activation_supersession(repository)
    assert resumed.to_payload()["status"].startswith("READY_TO_RESUME")
    receipt = supersede_harp_v14_active_activation(
        resumed,
        confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
    ).to_payload()
    assert receipt["old_authority_may_not_execute"] is True
    assert journal.registry_path.read_bytes() == journal.original_registry_bytes
    assert journal.catalog_path.read_bytes() == journal.original_catalog_bytes
    assert journal.config_path.read_bytes() == journal.original_config_bytes
    assert not authorization.lease_path(repository).exists()
    assert not (repository / TRANSACTION_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    "fault_point",
    (
        "archive_durable",
        "retirement_fence_durable",
        "registry_restored",
        "catalog_restored",
        "config_restored",
        "admin_output_retired",
        "active_amendment_retired",
        "terminal_receipt_durable",
        "retirement_fence_archived",
    ),
)
def test_workspace_admin_snapshot_every_injected_crash_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _plan_admin_output(repository, journal, monkeypatch)

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected-admin-crash")

    with pytest.raises(RuntimeError, match="injected-admin-crash"):
        supersede_harp_v14_active_activation(
            plan,
            confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
            _fault_injector=fail,
        )

    resumed = plan_harp_v14_active_activation_supersession(repository)
    assert resumed.to_payload()["status"].startswith("READY_TO_RESUME")
    receipt = supersede_harp_v14_active_activation(
        resumed,
        confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
    ).to_payload()
    retired = plan.archive_root / RETIRED_ADMIN_OUTPUT
    assert receipt["old_authority_may_not_execute"] is True
    assert (retired / "config.resolved.yaml").read_bytes() == b"resolved-config\n"
    assert (retired / "provenance/input_artifacts.json").read_bytes() == b"{}\n"
    assert not plan.output_root.exists()
    assert not authorization.lease_path(repository).exists()
    assert not (repository / TRANSACTION_RELATIVE_PATH).exists()


def test_delayed_preloaded_authorization_cannot_claim_after_supersession(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _admin_plan(repository, journal)
    amendment_hash = canonical_hash(
        {
            key: value
            for key, value in json.loads(journal.amendment_bytes.decode("utf-8")).items()
            if key != "amendment_hash"
        }
    )
    delayed = authorization.HarpV14Authorization(
        amendment_path=journal.amendment_path,
        amendment_sha256=journal.amendment_sha256,
        amendment_hash=amendment_hash,
        input_binding_hash="7" * 64,
        scientific_contract_hash="8" * 64,
        workspace_registration_execution_contract_hash="9" * 64,
        source_snapshot_schema=str(OLD_SOURCE["source_snapshot_schema"]),
        source_snapshot_manifest_sha256=str(
            OLD_SOURCE["source_snapshot_manifest_sha256"]
        ),
        source_snapshot_tree_sha256=str(OLD_SOURCE["source_snapshot_tree_sha256"]),
        source_snapshot_member_count=int(OLD_SOURCE["source_snapshot_member_count"]),
    )
    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14."
        "activation_attempts.active_transaction.build_active_supersession_plan",
        lambda _boundary: plan,
    )
    supersede_harp_v14_active_activation(
        plan,
        confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
    )
    assert (plan.archive_root / ARCHIVED_RETIREMENT_FENCE).is_file()
    assert not authorization.lease_path(repository).exists()

    with pytest.raises(ProtocolError):
        authorization.claim_authorization(
            delayed,
            admission_hash="a" * 64,
            repo_root=repository,
        )
    assert not authorization.lease_path(repository).exists()


def test_scientific_lease_wins_race_and_blocks_all_retirement_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, journal = _repository(tmp_path)
    plan = _plan_absent_output(repository, monkeypatch)
    scientific_lease = authorization.lease_path(repository)
    scientific_lease.mkdir()
    (scientific_lease / "lease.json").write_bytes(b"scientific-lease\n")

    with pytest.raises(ProtocolError, match="scientific authorization lease"):
        supersede_harp_v14_active_activation(
            plan,
            confirmation=ACTIVE_SUPERSESSION_CONFIRMATION,
        )
    assert journal.registry_path.read_bytes() == journal.final_registry_bytes
    assert journal.catalog_path.read_bytes() == journal.final_catalog_bytes
    assert journal.config_path.read_bytes() == journal.final_config_bytes
    assert journal.amendment_path.is_file()
