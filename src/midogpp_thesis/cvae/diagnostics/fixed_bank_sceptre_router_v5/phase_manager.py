"""Fail-closed global phase ordering for candidate-set routing."""

from __future__ import annotations

from dataclasses import dataclass
import secrets

from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ..fixed_bank_sceptre_router.phase_contracts import (
    PhaseCapability,
    TerminalEvaluationCapability,
)
from ..fixed_bank_sceptre_router.seals import (
    DecisionKey,
    DurablePreterminalAttestation,
    EXPECTED_DECISION_KEYS,
    GlobalDecisionSeal,
    build_global_decision_seal,
)
from .confirmation_gate import ConfirmationDecision
from .development import FrozenRoutingContext
from .proposal_set import FrozenCandidateSetProposal
from .route_policy import FrozenRoutePolicy
from .support_posterior import SupportPosteriorDecision


@dataclass(frozen=True, slots=True)
class ProposalSetFoldReceipt:
    target_center: str
    fold_ordinal: int
    partition_hash: str
    routing_context_hash: str
    proposal_set_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        key = (str(self.target_center), int(self.fold_ordinal))
        if key not in set(EXPECTED_DECISION_KEYS):
            raise ProtocolError("SCEPTRE v5 proposal receipt key is invalid.")
        for value, role in (
            (self.partition_hash, "proposal partition"),
            (self.routing_context_hash, "proposal routing context"),
            (self.proposal_set_hash, "proposal set"),
        ):
            require_sha256(value, role)
        body = {
            "schema_version": "sceptre_v5_proposal_set_fold_receipt_v1",
            "phase": "G_RANKED_CANDIDATE_SET_LABEL_FREE",
            "target_center": key[0],
            "fold_ordinal": key[1],
            "partition_hash": self.partition_hash,
            "routing_context_hash": self.routing_context_hash,
            "proposal_set_hash": self.proposal_set_hash,
            "target_global_proposal_set": True,
            "fold_attachment_consumes_labels": False,
        }
        expected = canonical_hash(body)
        if self.receipt_hash and self.receipt_hash != expected:
            raise ProtocolError("SCEPTRE v5 proposal receipt hash drifted.")
        object.__setattr__(self, "target_center", key[0])
        object.__setattr__(self, "fold_ordinal", key[1])
        object.__setattr__(self, "receipt_hash", expected)


class CandidateSetPhaseManager:
    """Enforce all proposal sets -> all support -> all confirmations -> terminal."""

    def __init__(
        self, partition: ThreeRolePartition, routing_context: FrozenRoutingContext
    ) -> None:
        if not isinstance(partition, ThreeRolePartition):
            raise ProtocolError("SCEPTRE v5 manager requires its partition.")
        if not isinstance(routing_context, FrozenRoutingContext):
            raise ProtocolError("SCEPTRE v5 manager requires routing context.")
        if routing_context.partition_hash != partition.partition_hash:
            raise ProtocolError("SCEPTRE v5 manager partition lineage drifted.")
        self._partition = partition
        self._context = routing_context
        self._partition_hash = partition.partition_hash
        self._context_hash = routing_context.context_hash
        self._proposal_receipts: dict[DecisionKey, ProposalSetFoldReceipt] = {}
        self._proposal_by_target: dict[str, FrozenCandidateSetProposal] = {}
        self._support: dict[DecisionKey, SupportPosteriorDecision] = {}
        self._confirmation: dict[DecisionKey, ConfirmationDecision] = {}
        self._issued_selection: dict[DecisionKey, PhaseCapability] = {}
        self._issued_calibration: dict[DecisionKey, PhaseCapability] = {}
        self._proposal_seal: GlobalDecisionSeal | None = None
        self._support_seal: GlobalDecisionSeal | None = None
        self._policy_seal: GlobalDecisionSeal | None = None
        self._policy: FrozenRoutePolicy | None = None
        self._terminal_opened = False
        self._terminal_capability: TerminalEvaluationCapability | None = None
        self._terminal_activated = False

    def record_label_free_proposal_set(
        self, proposal: FrozenCandidateSetProposal, fold_ordinal: int
    ) -> ProposalSetFoldReceipt:
        if not isinstance(proposal, FrozenCandidateSetProposal):
            raise ProtocolError("SCEPTRE v5 G must be a frozen proposal set.")
        key = self._key(proposal.target_center, fold_ordinal)
        if self._proposal_seal is not None or key in self._proposal_receipts:
            raise ProtocolError("SCEPTRE v5 proposal set recorded out of order.")
        model = self._context.model_for_target(key[0])
        if proposal.frozen_model_sha256 != model.model_sha256:
            raise ProtocolError("SCEPTRE v5 proposal/model lineage drifted.")
        prior = self._proposal_by_target.setdefault(key[0], proposal)
        if prior.proposal_set_hash != proposal.proposal_set_hash:
            raise ProtocolError("SCEPTRE v5 target proposal changed across folds.")
        receipt = ProposalSetFoldReceipt(
            target_center=key[0],
            fold_ordinal=key[1],
            partition_hash=self._partition_hash,
            routing_context_hash=self._context_hash,
            proposal_set_hash=proposal.proposal_set_hash,
        )
        self._proposal_receipts[key] = receipt
        return receipt

    def seal_all_proposal_sets(self) -> GlobalDecisionSeal:
        if self._proposal_seal is not None or self._issued_selection:
            raise ProtocolError("SCEPTRE v5 proposal-set seal requested out of order.")
        self._proposal_seal = build_global_decision_seal(
            "G_RANKED_CANDIDATE_SETS_LABEL_FREE",
            {key: row.receipt_hash for key, row in self._proposal_receipts.items()},
        )
        return self._proposal_seal

    def issue_selection_capability(
        self, target_center: str, fold_ordinal: int
    ) -> PhaseCapability:
        key = self._key(target_center, fold_ordinal)
        if (
            self._proposal_seal is None
            or self._support_seal is not None
            or key in self._issued_selection
        ):
            raise ProtocolError("SCEPTRE v5 selection capability opened out of order.")
        capability = self._capability(
            "SELECTION_LABELS", key, self._proposal_seal.seal_hash
        )
        self._issued_selection[key] = capability
        return capability

    def record_selection_decision(
        self, capability: PhaseCapability, decision: SupportPosteriorDecision
    ) -> None:
        key = self._validate_capability(capability, role="SELECTION_LABELS")
        if key in self._support or not isinstance(decision, SupportPosteriorDecision):
            raise ProtocolError("SCEPTRE v5 support decision is absent or duplicated.")
        fold = self._partition.fold(*key)
        proposal = self._proposal_by_target[key[0]]
        if (
            (decision.target_center, decision.fold_ordinal) != key
            or decision.fold_hash != fold.fold_hash
            or decision.partition_hash != self._partition_hash
            or decision.routing_context_hash != self._context_hash
            or decision.proposal_set_hash != proposal.proposal_set_hash
            or decision.candidate_menu_hash != proposal.candidate_menu_hash
            or decision.exact_b_control_receipt_hash
            != proposal.exact_b_control_receipt_hash
            or decision.selection_case_set_hash != fold.case_set_hash("SELECTION")
            or decision.calibration_case_set_hash != fold.case_set_hash("CALIBRATION")
            or decision.evaluation_case_set_hash != fold.case_set_hash("EVALUATION")
            or (
                decision.selected_candidate is not None
                and decision.selected_candidate not in proposal.candidate_sources
            )
        ):
            raise ProtocolError("SCEPTRE v5 support decision lineage drifted.")
        self._support[key] = decision

    def seal_all_selection_decisions(self) -> GlobalDecisionSeal:
        if self._proposal_seal is None or self._support_seal is not None:
            raise ProtocolError("SCEPTRE v5 support seal requested out of order.")
        self._support_seal = build_global_decision_seal(
            "S_Y_SUPPORT_SELECTED_MEMBER_OR_EXACT_B",
            {key: row.decision_hash for key, row in self._support.items()},
            predecessor_seal_hash=self._proposal_seal.seal_hash,
        )
        return self._support_seal

    def issue_calibration_capability(
        self, target_center: str, fold_ordinal: int
    ) -> PhaseCapability:
        key = self._key(target_center, fold_ordinal)
        if (
            self._support_seal is None
            or self._policy_seal is not None
            or key in self._issued_calibration
        ):
            raise ProtocolError("SCEPTRE v5 calibration capability opened out of order.")
        capability = self._capability(
            "CALIBRATION_LABELS", key, self._support_seal.seal_hash
        )
        self._issued_calibration[key] = capability
        return capability

    def record_calibration_decision(
        self, capability: PhaseCapability, decision: ConfirmationDecision
    ) -> None:
        key = self._validate_capability(capability, role="CALIBRATION_LABELS")
        if key in self._confirmation or not isinstance(decision, ConfirmationDecision):
            raise ProtocolError("SCEPTRE v5 confirmation is absent or duplicated.")
        support = self._support[key]
        fold = self._partition.fold(*key)
        proposal = self._proposal_by_target[key[0]]
        if (
            (decision.target_center, decision.fold_ordinal) != key
            or decision.fold_hash != fold.fold_hash
            or decision.partition_hash != self._partition_hash
            or decision.routing_context_hash != self._context_hash
            or decision.proposal_set_hash
            != self._proposal_receipts[key].proposal_set_hash
            or decision.support_decision_hash != support.decision_hash
            or decision.selection_case_set_hash
            != fold.case_set_hash("SELECTION")
            or decision.calibration_case_set_hash
            != fold.case_set_hash("CALIBRATION")
            or decision.evaluation_case_set_hash
            != fold.case_set_hash("EVALUATION")
            or decision.candidate_menu_hash != proposal.candidate_menu_hash
            or decision.exact_b_control_receipt_hash
            != proposal.exact_b_control_receipt_hash
            or decision.support_selected_candidate != support.selected_candidate
            or decision.route
            not in (
                {"B::exact_equal_union"}
                if support.selected_candidate is None
                else {support.selected_candidate, "B::exact_equal_union"}
            )
        ):
            raise ProtocolError("SCEPTRE v5 confirmation lineage drifted.")
        self._confirmation[key] = decision

    def seal_complete_policy(self) -> GlobalDecisionSeal:
        if self._support_seal is None or self._policy_seal is not None:
            raise ProtocolError("SCEPTRE v5 policy seal requested out of order.")
        self._policy_seal = build_global_decision_seal(
            "A_CONFIRM_SAME_SUPPORT_MEMBER_OR_EXACT_B",
            {key: row.decision_hash for key, row in self._confirmation.items()},
            predecessor_seal_hash=self._support_seal.seal_hash,
        )
        self._policy = FrozenRoutePolicy(
            partition_hash=self._partition_hash,
            routing_context_hash=self._context_hash,
            proposal_set_seal_hash=self._proposal_seal.seal_hash,
            support_seal_hash=self._support_seal.seal_hash,
            policy_seal_hash=self._policy_seal.seal_hash,
            route_rows=tuple(
                (
                    target,
                    fold,
                    self._proposal_receipts[(target, fold)].proposal_set_hash,
                    self._support[(target, fold)].decision_hash,
                    self._support[(target, fold)].selected_candidate,
                    self._confirmation[(target, fold)].route,
                    self._confirmation[(target, fold)].decision_hash,
                )
                for target, fold in EXPECTED_DECISION_KEYS
            ),
        )
        return self._policy_seal

    def export_frozen_route_policy(self) -> FrozenRoutePolicy:
        if self._policy is None:
            raise ProtocolError("SCEPTRE v5 route policy requested before sealing.")
        return self._policy

    def begin_terminal_evaluation(
        self, durable_attestation: DurablePreterminalAttestation
    ) -> TerminalEvaluationCapability:
        if (
            self._policy_seal is None
            or self._policy is None
            or self._terminal_opened
            or not isinstance(durable_attestation, DurablePreterminalAttestation)
            or durable_attestation.policy_seal_hash != self._policy_seal.seal_hash
        ):
            raise ProtocolError("SCEPTRE v5 terminal capability opened out of order.")
        body = {
            "schema_version": "sceptre_terminal_evaluation_capability_v1",
            "partition_hash": self._partition_hash,
            "router_bundle_hash": self._context_hash,
            "route_policy_hash": self._policy.policy_artifact_hash,
            "policy_seal_hash": self._policy_seal.seal_hash,
            "durable_attestation_hash": durable_attestation.attestation_hash,
            "one_shot": True,
            "raw_labels_may_be_persisted": False,
        }
        self._terminal_opened = True
        capability = TerminalEvaluationCapability(
            partition_hash=self._partition_hash,
            router_bundle_hash=self._context_hash,
            route_policy_hash=self._policy.policy_artifact_hash,
            policy_seal_hash=self._policy_seal.seal_hash,
            durable_attestation_hash=durable_attestation.attestation_hash,
            capability_hash=canonical_hash(body),
        )
        self._terminal_capability = capability
        return capability

    def activate_terminal_capability(
        self, capability: TerminalEvaluationCapability
    ) -> None:
        """Authorize the broker only for the exact manager-minted one-shot token."""

        expected = self._terminal_capability
        if (
            expected is None
            or capability is not expected
            or self._terminal_activated
            or self._policy is None
            or self._policy_seal is None
            or capability.partition_hash != self._partition_hash
            or capability.router_bundle_hash != self._context_hash
            or capability.route_policy_hash != self._policy.policy_artifact_hash
            or capability.policy_seal_hash != self._policy_seal.seal_hash
        ):
            raise ProtocolError(
                "SCEPTRE v5 terminal capability was not minted by this manager."
            )
        self._terminal_activated = True

    @staticmethod
    def _key(target_center: str, fold_ordinal: int) -> DecisionKey:
        key = (str(target_center), int(fold_ordinal))
        if key not in set(EXPECTED_DECISION_KEYS):
            raise ProtocolError("SCEPTRE v5 decision key is invalid.")
        return key

    def _capability(
        self, role: str, key: DecisionKey, predecessor_seal_hash: str
    ) -> PhaseCapability:
        receipt = self._proposal_receipts[key]
        predecessor_decision = (
            receipt.receipt_hash
            if role == "SELECTION_LABELS"
            else self._support[key].decision_hash
        )
        return PhaseCapability(
            role=role,
            target_center=key[0],
            fold_ordinal=key[1],
            partition_hash=self._partition_hash,
            router_bundle_hash=self._context_hash,
            # Compatibility field now binds the complete proposal-set hash.
            g_proposal_hash=receipt.proposal_set_hash,
            predecessor_decision_hash=predecessor_decision,
            predecessor_seal_hash=predecessor_seal_hash,
            nonce_hash=secrets.token_hex(32),
        )

    def _validate_capability(
        self, capability: PhaseCapability, *, role: str
    ) -> DecisionKey:
        if not isinstance(capability, PhaseCapability) or capability.role != role:
            raise ProtocolError("SCEPTRE v5 capability type or role drifted.")
        key = self._key(capability.target_center, capability.fold_ordinal)
        issued = (
            self._issued_selection
            if role == "SELECTION_LABELS"
            else self._issued_calibration
        )
        predecessor = (
            self._proposal_seal
            if role == "SELECTION_LABELS"
            else self._support_seal
        )
        expected_decision = (
            self._proposal_receipts[key].receipt_hash
            if role == "SELECTION_LABELS"
            else self._support[key].decision_hash
        )
        if (
            issued.get(key) is not capability
            or capability.partition_hash != self._partition_hash
            or capability.router_bundle_hash != self._context_hash
            or capability.g_proposal_hash
            != self._proposal_receipts[key].proposal_set_hash
            or capability.predecessor_decision_hash != expected_decision
            or predecessor is None
            or capability.predecessor_seal_hash != predecessor.seal_hash
        ):
            raise ProtocolError("SCEPTRE v5 capability lineage drifted.")
        return key


__all__ = ("CandidateSetPhaseManager", "ProposalSetFoldReceipt")
