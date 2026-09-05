"""Typed identities for archived HARP v19 activation attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from collections.abc import Mapping

from ....routing.harp_protocol import canonical_bytes, canonical_hash
from ..activation_transaction import ActivationJournal
from ..identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION


SUPERSESSION_CONFIRMATION = (
    "ARCHIVE_HARP_V19_ROLLED_BACK_ACTIVATION_FOR_REPLANNING"
)
SUPERSESSION_PLAN_SCHEMA = "midogpp_harp_stage90_activation_supersession_plan_v19"
SUPERSESSION_RECEIPT_SCHEMA = (
    "midogpp_harp_stage90_activation_supersession_receipt_v19"
)
SUPERSESSION_REASON = "activation_source_snapshot_changed_before_authority_consumption"
SUPERSEDED_ROOT_RELATIVE = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v19/superseded_activations"
)
ARCHIVED_JOURNAL = "activation_transaction.json"
ARCHIVED_AMENDMENT = "execution_amendment.json"
SUPERSESSION_RECEIPT = "supersession_receipt.json"

# A successfully committed but still unclaimed activation is a materially
# different state from the rolled-back attempt handled above.  Keep its
# transaction, confirmation phrase, schemas, and archive namespace distinct so
# a caller can never accidentally turn the older rollback path into an
# authority-revocation primitive.
ACTIVE_SUPERSESSION_CONFIRMATION = (
    "ARCHIVE_HARP_V19_ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION_FOR_REPLANNING"
)
ACTIVE_SUPERSESSION_PLAN_SCHEMA = (
    "midogpp_harp_stage90_active_activation_supersession_plan_v19"
)
ACTIVE_SUPERSESSION_RECEIPT_SCHEMA = (
    "midogpp_harp_stage90_active_activation_supersession_receipt_v19"
)
ACTIVE_SUPERSESSION_REASON = (
    "active_unconsumed_prelease_activation_source_snapshot_changed"
)
SUPERSEDED_ACTIVE_ROOT_RELATIVE = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v19/superseded_active_activations"
)
ARCHIVED_FINAL_CONFIG = "activated_config.yaml"
ARCHIVED_FINAL_REGISTRY = "activated_registry.yaml"
ARCHIVED_FINAL_CATALOG = "activated_artifact_catalog.yaml"
ARCHIVED_ADMIN_MANIFEST = "workspace_admin_snapshot_manifest.json"
ARCHIVED_ADMIN_CONTENT = "workspace_admin_snapshot"
RETIRED_ADMIN_OUTPUT = "retired_workspace_admin_output"
ARCHIVED_RETIREMENT_FENCE = "retirement_fence.json"
RETIREMENT_FENCE_SCHEMA = "midogpp_harp_v19_active_supersession_retirement_fence_v1"
ACTIVE_SUPERSESSION_RECEIPT = "active_supersession_receipt.json"


@dataclass(frozen=True, slots=True)
class HarpV19ActivationSupersessionPlan:
    repository_root: Path
    journal: ActivationJournal = field(repr=False)
    archive_root: Path
    prior_source_snapshot: Mapping[str, object]
    replacement_source_snapshot: Mapping[str, object]
    amendment_hash: str
    observed_states: Mapping[str, str]
    supersession_plan_hash: str

    def hash_payload(self) -> dict[str, object]:
        journal = self.journal
        return {
            "schema_version": SUPERSESSION_PLAN_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "reason": SUPERSESSION_REASON,
            "activation_plan_hash": journal.activation_plan_hash,
            "journal_hash": journal.journal_hash,
            "journal_file_sha256": sha256_bytes(journal.to_bytes()),
            "amendment_sha256": journal.amendment_sha256,
            "amendment_hash": self.amendment_hash,
            "original_config_sha256": sha256_bytes(journal.original_config_bytes),
            "original_registry_sha256": sha256_bytes(journal.original_registry_bytes),
            "original_catalog_sha256": sha256_bytes(journal.original_catalog_bytes),
            "prior_source_snapshot": dict(self.prior_source_snapshot),
            "replacement_source_snapshot": dict(self.replacement_source_snapshot),
            "source_snapshot_changed": True,
            "archive_root": self.archive_root.relative_to(
                self.repository_root
            ).as_posix(),
            "observed_states": dict(self.observed_states),
            "exact_byte_rollback_required_before_archive": any(
                self.observed_states.get(name) == "final"
                for name in ("registry", "catalog", "config")
            ),
            "authorization_lease_claimed": False,
            "output_created": False,
            "label_capability_artifact_created": False,
            "authorization_not_consumed": True,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self.hash_payload(),
            "supersession_plan_hash": self.supersession_plan_hash,
            "status": (
                "READY_TO_ROLL_BACK_AND_ARCHIVE_SOURCE_DRIFTED_ACTIVATION"
                if any(
                    self.observed_states.get(name) == "final"
                    for name in ("registry", "catalog", "config")
                )
                else "READY_TO_ARCHIVE_ROLLED_BACK_ACTIVATION"
            ),
            "confirmation_required": SUPERSESSION_CONFIRMATION,
            "filesystem_mutations": 0,
            "old_amendment_remains_non_executable": True,
            "fresh_activation_must_be_replanned": True,
            "forward_recovery_under_drift_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class HarpV19ActivationSupersessionReceipt:
    payload: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class HarpV19ActiveActivationSupersessionPlan:
    """Authenticated, mutation-free plan for revoking unclaimed authority."""

    repository_root: Path
    journal: ActivationJournal = field(repr=False)
    archive_root: Path
    output_root: Path
    scratch_root: Path
    prior_source_snapshot: Mapping[str, object]
    replacement_source_snapshot: Mapping[str, object]
    amendment_hash: str
    admin_snapshot_manifest: Mapping[str, object]
    admin_snapshot_files: Mapping[str, bytes] = field(repr=False)
    supersession_plan_hash: str
    recovery_state: Mapping[str, object] = field(default_factory=dict)

    def hash_payload(self) -> dict[str, object]:
        journal = self.journal
        return {
            "schema_version": ACTIVE_SUPERSESSION_PLAN_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "reason": ACTIVE_SUPERSESSION_REASON,
            "activation_plan_hash": journal.activation_plan_hash,
            "journal_hash": journal.journal_hash,
            "journal_file_sha256": sha256_bytes(journal.to_bytes()),
            "amendment_sha256": journal.amendment_sha256,
            "amendment_hash": self.amendment_hash,
            "activated_config_sha256": sha256_bytes(journal.final_config_bytes),
            "activated_registry_sha256": sha256_bytes(journal.final_registry_bytes),
            "activated_catalog_sha256": sha256_bytes(journal.final_catalog_bytes),
            "original_config_sha256": sha256_bytes(journal.original_config_bytes),
            "original_registry_sha256": sha256_bytes(journal.original_registry_bytes),
            "original_catalog_sha256": sha256_bytes(journal.original_catalog_bytes),
            "prior_source_snapshot": dict(self.prior_source_snapshot),
            "replacement_source_snapshot": dict(self.replacement_source_snapshot),
            "source_snapshot_changed": True,
            "archive_root": self.archive_root.relative_to(
                self.repository_root
            ).as_posix(),
            "output_root": self.output_root.relative_to(
                self.repository_root
            ).as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "workspace_admin_snapshot": dict(self.admin_snapshot_manifest),
            "observed_states": {
                "amendment": "exact",
                "config": "final",
                "catalog": "final",
                "registry": "final",
            },
            "authorization_lease_claimed": False,
            "scratch_created": False,
            "scientific_output_created": False,
            "labels_opened": False,
            "routes_sealed": False,
            "authorization_not_consumed": True,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        recovering = bool(self.recovery_state.get("retirement_started", False))
        return {
            **self.hash_payload(),
            "supersession_plan_hash": self.supersession_plan_hash,
            "status": (
                "READY_TO_RESUME_ACTIVE_UNCONSUMED_PRELEASE_SUPERSESSION"
                if recovering
                else "READY_TO_ARCHIVE_ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION"
            ),
            "confirmation_required": ACTIVE_SUPERSESSION_CONFIRMATION,
            "filesystem_mutations": 0,
            "registry_will_be_restored_first": True,
            "admin_output_will_be_archived_without_deletion": True,
            "old_authority_remains_live_until_explicit_confirmation": True,
            "fresh_activation_must_be_replanned": True,
            "recovery_state": dict(self.recovery_state),
        }


@dataclass(frozen=True, slots=True)
class HarpV19ActiveActivationSupersessionReceipt:
    payload: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def receipt_payload(plan: HarpV19ActivationSupersessionPlan) -> dict[str, object]:
    body = {
        **plan.hash_payload(),
        "schema_version": SUPERSESSION_RECEIPT_SCHEMA,
        "supersession_plan_hash": plan.supersession_plan_hash,
        "status": "ROLLED_BACK_ACTIVATION_ARCHIVED_AND_SUPERSEDED",
        "archive_members": {
            ARCHIVED_JOURNAL: sha256_bytes(plan.journal.to_bytes()),
            ARCHIVED_AMENDMENT: plan.journal.amendment_sha256,
        },
        "archive_durable_before_live_retirement": True,
        "live_retirement_order": [
            "registry_exact_byte_rollback_if_needed",
            "catalog_exact_byte_rollback_if_needed",
            "config_exact_byte_rollback_if_needed",
            "archive_after_planned_workspace_validation",
            "execution_amendment",
            "activation_transaction",
        ],
        "metadata_rolled_back_before_archive": True,
        "forward_recovery_under_drift_used": False,
        "old_amendment_superseded": True,
        "old_authority_may_not_execute": True,
        "fresh_activation_must_be_replanned": True,
    }
    return {**body, "supersession_receipt_hash": canonical_hash(body)}


def active_receipt_payload(
    plan: HarpV19ActiveActivationSupersessionPlan,
) -> dict[str, object]:
    """Build the terminal receipt only after all live authority is retired."""

    journal = plan.journal
    body = {
        **plan.hash_payload(),
        "schema_version": ACTIVE_SUPERSESSION_RECEIPT_SCHEMA,
        "supersession_plan_hash": plan.supersession_plan_hash,
        "status": "ACTIVE_UNCONSUMED_PRELEASE_ACTIVATION_ARCHIVED_AND_SUPERSEDED",
        "archive_members": {
            ARCHIVED_JOURNAL: sha256_bytes(journal.to_bytes()),
            ARCHIVED_AMENDMENT: journal.amendment_sha256,
            ARCHIVED_FINAL_CONFIG: sha256_bytes(journal.final_config_bytes),
            ARCHIVED_FINAL_REGISTRY: sha256_bytes(journal.final_registry_bytes),
            ARCHIVED_FINAL_CATALOG: sha256_bytes(journal.final_catalog_bytes),
            ARCHIVED_ADMIN_MANIFEST: sha256_bytes(
                canonical_bytes(plan.admin_snapshot_manifest) + b"\n"
            ),
            ARCHIVED_RETIREMENT_FENCE: sha256_bytes(
                canonical_bytes(retirement_fence_payload(plan)) + b"\n"
            ),
        },
        "archive_durable_before_live_retirement": True,
        "live_retirement_order": [
            "registry_restored_to_planned_bytes",
            "catalog_restored_to_planned_bytes",
            "config_restored_to_planned_bytes",
            "workspace_admin_output_archived_without_deletion",
            "execution_amendment_retired",
            "terminal_receipt_committed",
            "retirement_fence_archived",
            "activation_transaction_retired",
        ],
        "workspace_admin_output_state": plan.admin_snapshot_manifest["state"],
        "workspace_admin_output_archived_without_deletion": True,
        "old_amendment_superseded": True,
        "old_authority_may_not_execute": True,
        "authorization_not_consumed": True,
        "fresh_activation_must_be_replanned": True,
    }
    return {**body, "supersession_receipt_hash": canonical_hash(body)}


def retirement_fence_payload(
    plan: HarpV19ActiveActivationSupersessionPlan,
) -> dict[str, object]:
    body = {
        "schema_version": RETIREMENT_FENCE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "ACTIVE_AUTHORITY_RETIREMENT_IN_PROGRESS",
        "supersession_plan_hash": plan.supersession_plan_hash,
        "activation_plan_hash": plan.journal.activation_plan_hash,
        "journal_hash": plan.journal.journal_hash,
        "amendment_sha256": plan.journal.amendment_sha256,
        "authorization_lease_claimed": False,
        "scientific_execution_may_start": False,
        "labels_opened": False,
        "routes_sealed": False,
    }
    return {**body, "retirement_fence_hash": canonical_hash(body)}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = (
    "ACTIVE_SUPERSESSION_CONFIRMATION",
    "ACTIVE_SUPERSESSION_RECEIPT",
    "ARCHIVED_ADMIN_CONTENT",
    "ARCHIVED_ADMIN_MANIFEST",
    "ARCHIVED_AMENDMENT",
    "ARCHIVED_FINAL_CATALOG",
    "ARCHIVED_FINAL_CONFIG",
    "ARCHIVED_FINAL_REGISTRY",
    "ARCHIVED_JOURNAL",
    "ARCHIVED_RETIREMENT_FENCE",
    "HarpV19ActiveActivationSupersessionPlan",
    "HarpV19ActiveActivationSupersessionReceipt",
    "HarpV19ActivationSupersessionPlan",
    "HarpV19ActivationSupersessionReceipt",
    "RETIRED_ADMIN_OUTPUT",
    "RETIREMENT_FENCE_SCHEMA",
    "SUPERSEDED_ACTIVE_ROOT_RELATIVE",
    "SUPERSEDED_ROOT_RELATIVE",
    "SUPERSESSION_CONFIRMATION",
    "SUPERSESSION_RECEIPT",
    "active_receipt_payload",
    "is_sha256",
    "receipt_payload",
    "retirement_fence_payload",
    "sha256_bytes",
)
