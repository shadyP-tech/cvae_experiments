"""Pure, deterministic decision seals for SCEPTRE phase barriers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .partitions import FOLD_COUNT


DecisionKey = tuple[str, int]
EXPECTED_DECISION_KEYS = tuple(
    (center, fold) for center in CENTERS for fold in range(FOLD_COUNT)
)


@dataclass(frozen=True, slots=True)
class GlobalDecisionSeal:
    phase: str
    decision_count: int
    decision_hashes: tuple[tuple[str, int, str], ...]
    predecessor_seal_hash: str | None
    seal_hash: str


@dataclass(frozen=True, slots=True)
class FoldDecisionReceipt:
    """Attach one target-global label-free G proposal to a phase-barrier fold."""

    phase: str
    target_center: str
    fold_ordinal: int
    partition_hash: str
    router_bundle_hash: str
    g_proposal_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        key = (str(self.target_center), int(self.fold_ordinal))
        if self.phase != "G_LABEL_FREE" or key not in set(EXPECTED_DECISION_KEYS):
            raise ProtocolError("SCEPTRE fold decision receipt scope drifted.")
        partition = require_sha256(self.partition_hash, "decision partition")
        router_bundle = require_sha256(
            self.router_bundle_hash, "frozen router bundle"
        )
        proposal = require_sha256(self.g_proposal_hash, "G proposal")
        body = {
            "schema_version": "sceptre_fold_decision_receipt_v2",
            "phase": self.phase,
            "target_center": key[0],
            "fold_ordinal": key[1],
            "partition_hash": partition,
            "router_bundle_hash": router_bundle,
            "g_proposal_hash": proposal,
            "g_proposal_scope": "TARGET_GLOBAL_LABEL_FREE",
            "fold_attachment_role": "PHASE_BARRIER_ONLY",
            "fold_attachment_consumes_labels": False,
            "labels_consumed": False,
        }
        expected = canonical_hash(body)
        if self.receipt_hash and self.receipt_hash != expected:
            raise ProtocolError("SCEPTRE fold decision receipt hash drifted.")
        object.__setattr__(self, "target_center", key[0])
        object.__setattr__(self, "fold_ordinal", key[1])
        object.__setattr__(self, "partition_hash", partition)
        object.__setattr__(self, "router_bundle_hash", router_bundle)
        object.__setattr__(self, "g_proposal_hash", proposal)
        object.__setattr__(self, "receipt_hash", expected)

    @property
    def payload_hash(self) -> str:
        """Compatibility alias: the sole payload is the typed G proposal."""

        return self.g_proposal_hash


@dataclass(frozen=True, slots=True)
class FreshProcessValidation:
    """One independent reconstruction of the complete preterminal policy."""

    process_id: int
    policy_seal_hash: str
    source_tree_sha256: str
    reconstruction_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.process_id, bool) or self.process_id <= 0:
            raise ProtocolError("SCEPTRE fresh validator process id is invalid.")
        body = {
            "schema_version": "sceptre_fresh_process_validation_v1",
            "process_id": self.process_id,
            "policy_seal_hash": require_sha256(self.policy_seal_hash, "policy seal"),
            "source_tree_sha256": require_sha256(self.source_tree_sha256, "source tree"),
            "reconstruction_hash": require_sha256(
                self.reconstruction_hash, "reconstruction"
            ),
            "fresh_process": True,
            "cuda_hidden": True,
            "thread_count": 1,
        }
        expected = canonical_hash(body)
        if self.receipt_hash and self.receipt_hash != expected:
            raise ProtocolError("SCEPTRE fresh validation receipt hash drifted.")
        object.__setattr__(self, "receipt_hash", expected)


@dataclass(frozen=True, slots=True)
class DurablePreterminalAttestation:
    """Two-PID, byte-identical reconstruction gate for terminal access."""

    policy_seal_hash: str
    validations: tuple[FreshProcessValidation, FreshProcessValidation]
    attestation_hash: str = ""

    def __post_init__(self) -> None:
        policy = require_sha256(self.policy_seal_hash, "policy seal")
        if len(self.validations) != 2 or any(
            not isinstance(row, FreshProcessValidation) for row in self.validations
        ):
            raise ProtocolError("SCEPTRE requires exactly two fresh validations.")
        first, second = self.validations
        if first.process_id == second.process_id:
            raise ProtocolError("SCEPTRE fresh validators must have independent PIDs.")
        if (
            first.policy_seal_hash != policy
            or second.policy_seal_hash != policy
            or first.source_tree_sha256 != second.source_tree_sha256
            or first.reconstruction_hash != second.reconstruction_hash
        ):
            raise ProtocolError("SCEPTRE fresh validator reconstructions differ.")
        body = {
            "schema_version": "sceptre_durable_preterminal_attestation_v1",
            "policy_seal_hash": policy,
            "validation_receipt_hashes": [
                first.receipt_hash,
                second.receipt_hash,
            ],
            "independent_process_ids": [first.process_id, second.process_id],
            "byte_identical_reconstruction": True,
            "source_tree_sha256": first.source_tree_sha256,
            "reconstruction_hash": first.reconstruction_hash,
        }
        expected = canonical_hash(body)
        if self.attestation_hash and self.attestation_hash != expected:
            raise ProtocolError("SCEPTRE durable attestation hash drifted.")
        object.__setattr__(self, "attestation_hash", expected)


def build_global_decision_seal(
    phase: str,
    decisions: Mapping[DecisionKey, str],
    *,
    predecessor_seal_hash: str | None = None,
) -> GlobalDecisionSeal:
    if not phase:
        raise ProtocolError("SCEPTRE global seal phase is empty.")
    if set(decisions) != set(EXPECTED_DECISION_KEYS):
        raise ProtocolError("SCEPTRE global seal requires all 45 decisions.")
    rows = tuple(
        (center, fold, require_sha256(decisions[(center, fold)], "decision"))
        for center, fold in EXPECTED_DECISION_KEYS
    )
    if predecessor_seal_hash is not None:
        predecessor_seal_hash = require_sha256(predecessor_seal_hash, "predecessor seal")
    body = {
        "schema_version": "sceptre_global_decision_seal_v1",
        "phase": phase,
        "decision_count": len(rows),
        "decision_hashes": [list(row) for row in rows],
        "predecessor_seal_hash": predecessor_seal_hash,
        "raw_labels_persisted": False,
    }
    return GlobalDecisionSeal(
        phase=phase,
        decision_count=len(rows),
        decision_hashes=rows,
        predecessor_seal_hash=predecessor_seal_hash,
        seal_hash=canonical_hash(body),
    )


__all__ = (
    "DecisionKey",
    "DurablePreterminalAttestation",
    "EXPECTED_DECISION_KEYS",
    "FoldDecisionReceipt",
    "FreshProcessValidation",
    "GlobalDecisionSeal",
    "build_global_decision_seal",
)
