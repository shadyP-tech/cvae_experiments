"""Public, mutation-free planning and explicit HARP v14 activation.

Workspace rendering and durable commit/recovery live in dedicated modules.
This surface keeps the scientific authority transition small and auditable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from . import authorization
from .activation_paths import RepositoryBoundary
from .activation_transaction import (
    HarpV14ActivationReceipt,
    commit_activation,
    inspect_activation_recovery,
    recover_activation,
)
from .activation_supersession import (
    SUPERSESSION_CONFIRMATION,
    recovery_source_snapshot_changed,
    require_harp_v14_recovery_source_current,
)
from .activation_workspace import render_activation_workspace
from .amendment_publisher import (
    HarpV14AmendmentDraft,
    build_harp_v14_execution_amendment_draft,
)
from .config import HarpStage90V14Config, load_config
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .source_seal import source_snapshot_identity


ACTIVATION_CONFIRMATION = "ACTIVATE_HARP_V14_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
ACTIVATION_PLAN_SCHEMA = "midogpp_harp_stage90_activation_plan_v14"


@dataclass(frozen=True, slots=True)
class HarpV14ActivationPlan:
    repository_root: Path
    config_path: Path
    registry_path: Path
    catalog_path: Path
    original_config_bytes: bytes = field(repr=False)
    original_registry_bytes: bytes = field(repr=False)
    original_catalog_bytes: bytes = field(repr=False)
    final_config_bytes: bytes = field(repr=False)
    final_registry_bytes: bytes = field(repr=False)
    final_catalog_bytes: bytes = field(repr=False)
    authorized_config: HarpStage90V14Config
    amendment_draft: HarpV14AmendmentDraft
    amendment_already_issued: bool
    activation_plan_hash: str

    def __post_init__(self) -> None:
        if self.activation_plan_hash and (
            canonical_hash(self._hash_payload()) != self.activation_plan_hash
        ):
            raise ProtocolError("HARP v14 activation plan identity drifted.")
        if self.authorized_config.expected_execution_amendment_sha256 != (
            self.amendment_draft.amendment_sha256
        ):
            raise ProtocolError("HARP v14 activation plan amendment binding drifted.")

    def _hash_payload(self) -> dict[str, object]:
        # Amendment presence is recovery state, not a different scientific plan.
        return {
            "schema_version": ACTIVATION_PLAN_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "authorization_basis": self.amendment_draft.amendment_payload[
                "authorization_basis"
            ],
            "authorization_date": self.amendment_draft.amendment_payload[
                "authorization_date"
            ],
            "original_config_sha256": _sha256(self.original_config_bytes),
            "original_registry_sha256": _sha256(self.original_registry_bytes),
            "original_catalog_sha256": _sha256(self.original_catalog_bytes),
            "final_config_sha256": _sha256(self.final_config_bytes),
            "final_registry_sha256": _sha256(self.final_registry_bytes),
            "final_catalog_sha256": _sha256(self.final_catalog_bytes),
            "amendment_sha256": self.amendment_draft.amendment_sha256,
            "amendment_hash": self.amendment_draft.amendment_payload[
                "amendment_hash"
            ],
            "config_hash": self.authorized_config.config_hash,
            "commit_order": [
                "durable_journal",
                "exclusive_amendment",
                "authorized_config",
                "authorized_catalog",
                "diagnostic_registry_commit",
            ],
            "registry_is_last_commit_point": True,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "activation_plan_hash": self.activation_plan_hash,
            "status": "READY_FOR_EXPLICIT_CONFIRMATION",
            "confirmation_required": ACTIVATION_CONFIRMATION,
            "amendment_already_issued_exact": self.amendment_already_issued,
            "durable_journal_will_precede_protected_mutations": True,
            "filesystem_mutations": 0,
            "checked_in_config_remains_planned": True,
            "amendment_created_by_planning": False,
            "authorization_lease_claimed": False,
            "labels_opened": False,
            "output_created": False,
            "may_feed_stage60_or_stage70": False,
            "may_feed_another_experiment": False,
        }


def plan_harp_v14_activation(
    config: HarpStage90V14Config,
    *,
    expert_bank_root: str | Path,
    generation_lock_root: str | Path,
    prepared_cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    repository_root: str | Path,
    authorization_basis: str,
    authorization_date: str,
) -> HarpV14ActivationPlan:
    """Authenticate inputs and render all final bytes without writing."""

    boundary = RepositoryBoundary.open(repository_root)
    repository = boundary.resolved_root
    if type(config) is not HarpStage90V14Config or config.execution_authorized:
        raise ProtocolError("HARP v14 activation planning requires the planned config.")
    authorization.validate_activation_metadata(authorization_basis, authorization_date)
    config_path = boundary.member(
        authorization.WORKSPACE_CONFIG_RELATIVE_PATH, label="config", kind="file"
    )
    registry_path = boundary.member(
        authorization.WORKSPACE_REGISTRY_RELATIVE_PATH, label="registry", kind="file"
    )
    catalog_path = boundary.member(
        authorization.WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH,
        label="artifact catalog",
        kind="file",
    )
    if config.source_path != config_path or load_config(config_path) != config:
        raise ProtocolError("HARP v14 activation config changed after load.")
    amendment_path = boundary.member(
        authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH,
        label="execution amendment",
        kind="optional",
    )
    boundary.member(
        authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
        label="output identity",
        kind="future",
    )
    boundary.path(
        authorization.lease_path(repository),
        label="authorization lease",
        kind="absent",
    )

    original_config = config_path.read_bytes()
    original_registry = registry_path.read_bytes()
    original_catalog = catalog_path.read_bytes()
    draft = build_harp_v14_execution_amendment_draft(
        config,
        expert_bank_root=expert_bank_root,
        generation_lock_root=generation_lock_root,
        prepared_cache_root=prepared_cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        parent_ledger_path=parent_ledger_path,
        amendment_path=amendment_path,
        authorization_basis=authorization_basis,
        authorization_date=authorization_date,
        repository_root=repository,
    )
    amendment_already_issued = amendment_path.exists()
    if amendment_already_issued and amendment_path.read_bytes() != draft.amendment_raw:
        raise ProtocolError(
            "HARP v14 existing amendment does not match the exact activation plan."
        )
    rendered = render_activation_workspace(
        original_config_bytes=original_config,
        original_registry_bytes=original_registry,
        original_catalog_bytes=original_catalog,
        draft=draft,
    )
    provisional = HarpV14ActivationPlan(
        repository_root=repository,
        config_path=config_path,
        registry_path=registry_path,
        catalog_path=catalog_path,
        original_config_bytes=original_config,
        original_registry_bytes=original_registry,
        original_catalog_bytes=original_catalog,
        final_config_bytes=rendered.final_config_bytes,
        final_registry_bytes=rendered.final_registry_bytes,
        final_catalog_bytes=rendered.final_catalog_bytes,
        authorized_config=rendered.authorized_config,
        amendment_draft=draft,
        amendment_already_issued=amendment_already_issued,
        activation_plan_hash="",
    )
    return HarpV14ActivationPlan(
        repository_root=provisional.repository_root,
        config_path=provisional.config_path,
        registry_path=provisional.registry_path,
        catalog_path=provisional.catalog_path,
        original_config_bytes=provisional.original_config_bytes,
        original_registry_bytes=provisional.original_registry_bytes,
        original_catalog_bytes=provisional.original_catalog_bytes,
        final_config_bytes=provisional.final_config_bytes,
        final_registry_bytes=provisional.final_registry_bytes,
        final_catalog_bytes=provisional.final_catalog_bytes,
        authorized_config=provisional.authorized_config,
        amendment_draft=provisional.amendment_draft,
        amendment_already_issued=provisional.amendment_already_issued,
        activation_plan_hash=canonical_hash(provisional._hash_payload()),
    )


def activate_harp_v14(
    plan: HarpV14ActivationPlan,
    *,
    confirmation: str,
    _fault_injector: Callable[[str], None] | None = None,
) -> HarpV14ActivationReceipt:
    """Commit a prevalidated plan with durable exact-byte recovery."""

    if type(plan) is not HarpV14ActivationPlan:
        raise ProtocolError("HARP v14 activation requires a typed plan.")
    if confirmation != ACTIVATION_CONFIRMATION:
        raise ProtocolError("HARP v14 activation confirmation is absent or drifted.")
    _verify_plan_source_state(plan)
    return commit_activation(
        plan,
        confirmation=confirmation,
        expected_confirmation=ACTIVATION_CONFIRMATION,
        fault_injector=_fault_injector,
    )


def recover_harp_v14_activation(
    repository_root: str | Path,
    *,
    confirmation: str,
    _fault_injector: Callable[[str], None] | None = None,
) -> HarpV14ActivationReceipt:
    """Resume an interrupted transaction from its immutable journal."""

    if confirmation != ACTIVATION_CONFIRMATION:
        raise ProtocolError("HARP v14 activation confirmation is absent or drifted.")
    require_harp_v14_recovery_source_current(repository_root)
    return recover_activation(
        repository_root,
        confirmation=confirmation,
        expected_confirmation=ACTIVATION_CONFIRMATION,
        fault_injector=_fault_injector,
    )


def inspect_harp_v14_activation_recovery(
    repository_root: str | Path,
) -> dict[str, object] | None:
    payload = inspect_activation_recovery(repository_root)
    if payload is None:
        return None
    rendered = dict(payload)
    if recovery_source_snapshot_changed(repository_root):
        rendered.update(
            {
                "status": "SUPERSESSION_REQUIRED_SOURCE_SNAPSHOT_DRIFT",
                "confirmation_required": SUPERSESSION_CONFIRMATION,
                "normal_recovery_allowed": False,
                "filesystem_mutations": 0,
            }
        )
    return rendered


def _verify_plan_source_state(plan: HarpV14ActivationPlan) -> None:
    boundary = RepositoryBoundary.open(plan.repository_root)
    for path, expected, label in (
        (plan.config_path, plan.original_config_bytes, "config"),
        (plan.registry_path, plan.original_registry_bytes, "registry"),
        (plan.catalog_path, plan.original_catalog_bytes, "artifact catalog"),
    ):
        checked = boundary.path(path, label=label, kind="file")
        if checked.read_bytes() != expected:
            raise ProtocolError(
                "HARP v14 activation source metadata changed after planning."
            )
    amendment_path = boundary.path(
        plan.amendment_draft.amendment_path,
        label="execution amendment",
        kind="optional",
    )
    if plan.amendment_already_issued:
        if amendment_path.read_bytes() != plan.amendment_draft.amendment_raw:
            raise ProtocolError("HARP v14 issued amendment changed after planning.")
    elif os.path.lexists(amendment_path):
        raise ProtocolError("HARP v14 amendment exists before activation commit.")
    boundary.path(
        authorization.lease_path(boundary.resolved_root),
        label="authorization lease",
        kind="absent",
    )
    expected_source = plan.amendment_draft.amendment_payload[
        "source_snapshot_identity"
    ]
    if dict(source_snapshot_identity(boundary.resolved_root)) != dict(expected_source):
        raise ProtocolError("HARP v14 source snapshot changed after planning.")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


__all__ = (
    "ACTIVATION_CONFIRMATION",
    "HarpV14ActivationPlan",
    "HarpV14ActivationReceipt",
    "activate_harp_v14",
    "inspect_harp_v14_activation_recovery",
    "plan_harp_v14_activation",
    "recover_harp_v14_activation",
)
