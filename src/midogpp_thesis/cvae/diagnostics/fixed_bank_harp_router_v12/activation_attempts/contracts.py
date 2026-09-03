"""Typed identities for archived HARP v12 activation attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from collections.abc import Mapping

from ....routing.harp_protocol import canonical_hash
from ..activation_transaction import ActivationJournal
from ..identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION


SUPERSESSION_CONFIRMATION = (
    "ARCHIVE_HARP_V12_ROLLED_BACK_ACTIVATION_FOR_REPLANNING"
)
SUPERSESSION_PLAN_SCHEMA = "midogpp_harp_stage90_activation_supersession_plan_v12"
SUPERSESSION_RECEIPT_SCHEMA = (
    "midogpp_harp_stage90_activation_supersession_receipt_v12"
)
SUPERSESSION_REASON = "activation_source_snapshot_changed_before_authority_consumption"
SUPERSEDED_ROOT_RELATIVE = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v12/superseded_activations"
)
ARCHIVED_JOURNAL = "activation_transaction.json"
ARCHIVED_AMENDMENT = "execution_amendment.json"
SUPERSESSION_RECEIPT = "supersession_receipt.json"


@dataclass(frozen=True, slots=True)
class HarpV12ActivationSupersessionPlan:
    repository_root: Path
    journal: ActivationJournal = field(repr=False)
    archive_root: Path
    prior_source_snapshot: Mapping[str, object]
    replacement_source_snapshot: Mapping[str, object]
    amendment_hash: str
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
            "observed_states": {
                "amendment": "exact",
                "config": "original",
                "catalog": "original",
                "registry": "original",
            },
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
            "status": "READY_TO_ARCHIVE_ROLLED_BACK_ACTIVATION",
            "confirmation_required": SUPERSESSION_CONFIRMATION,
            "filesystem_mutations": 0,
            "old_amendment_remains_non_executable": True,
            "fresh_activation_must_be_replanned": True,
        }


@dataclass(frozen=True, slots=True)
class HarpV12ActivationSupersessionReceipt:
    payload: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def receipt_payload(plan: HarpV12ActivationSupersessionPlan) -> dict[str, object]:
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
            "execution_amendment",
            "activation_transaction",
        ],
        "old_amendment_superseded": True,
        "old_authority_may_not_execute": True,
        "fresh_activation_must_be_replanned": True,
    }
    return {**body, "supersession_receipt_hash": canonical_hash(body)}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = (
    "ARCHIVED_AMENDMENT",
    "ARCHIVED_JOURNAL",
    "HarpV12ActivationSupersessionPlan",
    "HarpV12ActivationSupersessionReceipt",
    "SUPERSEDED_ROOT_RELATIVE",
    "SUPERSESSION_CONFIRMATION",
    "SUPERSESSION_RECEIPT",
    "is_sha256",
    "receipt_payload",
    "sha256_bytes",
)
