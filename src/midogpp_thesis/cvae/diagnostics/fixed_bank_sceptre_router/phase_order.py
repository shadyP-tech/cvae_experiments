"""Fail-closed global phase ordering for future SCEPTRE execution.

The capabilities in this module bind lineage and one-shot manager ownership;
they are not themselves a test-label reader.  The planned identity remains
non-runnable until a separate execution module implements that reader and
validates manager ownership *before* opening raw labels.
"""

from __future__ import annotations

import secrets

from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .calibration_gate import CalibrationGateDecision
from .seals import (
    DecisionKey,
    DurablePreterminalAttestation,
    EXPECTED_DECISION_KEYS,
    FoldDecisionReceipt,
    GlobalDecisionSeal,
    build_global_decision_seal,
)
from .support_tournament import SupportTournamentDecision
from .partitions import ThreeRolePartition
from .phase_contracts import PhaseCapability, TerminalEvaluationCapability
from .route_policy import FrozenRoutePolicy


class SceptrePhaseManager:
    """Enforce G -> all S_y -> all A -> terminal evaluation globally."""

    def __init__(self, partition: ThreeRolePartition, frozen_router: object) -> None:
        from .router_bundle_freeze import FrozenPrelabelRouter

        if not isinstance(partition, ThreeRolePartition):
            raise ProtocolError("SCEPTRE phase manager requires its typed partition.")
        if not isinstance(frozen_router, FrozenPrelabelRouter):
            raise ProtocolError("SCEPTRE phase manager requires its frozen router.")
        if frozen_router.partition_hash != partition.partition_hash:
            raise ProtocolError("SCEPTRE phase router/partition lineage differs.")
        self._partition = partition
        self._frozen_router = frozen_router
        self._partition_hash = partition.partition_hash
        self._router_bundle_hash = frozen_router.router_bundle_hash
        self._g: dict[DecisionKey, str] = {}
        self._g_receipts: dict[DecisionKey, FoldDecisionReceipt] = {}
        self._g_proposals: dict[DecisionKey, object] = {}
        self._g_proposal_by_target: dict[str, object] = {}
        self._selection: dict[DecisionKey, str] = {}
        self._selection_decisions: dict[DecisionKey, SupportTournamentDecision] = {}
        self._calibration_uncertainty: dict[DecisionKey, object] = {}
        self._calibration: dict[DecisionKey, str] = {}
        self._calibration_decisions: dict[DecisionKey, CalibrationGateDecision] = {}
        self._issued_selection: dict[DecisionKey, PhaseCapability] = {}
        self._issued_calibration: dict[DecisionKey, PhaseCapability] = {}
        self._g_seal: GlobalDecisionSeal | None = None
        self._selection_seal: GlobalDecisionSeal | None = None
        self._policy_seal: GlobalDecisionSeal | None = None
        self._frozen_route_policy: FrozenRoutePolicy | None = None
        self._terminal_opened = False

    def record_label_free_g_decision(
        self, proposal: object, fold_ordinal: int
    ) -> FoldDecisionReceipt:
        from .router_bundle_freeze import FrozenGProposal

        if not isinstance(proposal, FrozenGProposal):
            raise ProtocolError("SCEPTRE G decision must be a frozen G proposal.")
        key = self._key(proposal.target_center, fold_ordinal)
        if self._g_seal is not None or key in self._g:
            raise ProtocolError("SCEPTRE G decision was recorded out of order.")
        target_proposal = self._g_proposal_by_target.get(key[0])
        if (
            target_proposal is not None
            and target_proposal.g_proposal_hash != proposal.g_proposal_hash
        ):
            raise ProtocolError(
                "SCEPTRE target-global G proposal changed across fold attachments."
            )
        model = self._frozen_router.model_for_target(key[0])
        if (
            proposal.partition_hash != self._partition_hash
            or proposal.router_bundle_hash != self._router_bundle_hash
            or proposal.frozen_model_hash != model.model_sha256
            or proposal.decision_policy_sha256
            != self._frozen_router.decision_policy_sha256
            or proposal.candidate_menu_hash != model.candidate_menu_hash
            or proposal.candidate_menu_payload_sha256
            != model.candidate_menu_payload_sha256
            or proposal.exact_b_control_receipt_hash
            != model.exact_b_control_receipt_hash
            or proposal.exact_b_control_payload_sha256
            != model.exact_b_control_payload_sha256
        ):
            raise ProtocolError("SCEPTRE G proposal differs from the frozen router.")
        receipt = proposal.to_fold_receipt(key[1])
        self._g[key] = require_sha256(receipt.receipt_hash, "G decision")
        self._g_receipts[key] = receipt
        self._g_proposals[key] = proposal
        self._g_proposal_by_target.setdefault(key[0], proposal)
        return receipt

    def seal_all_g_decisions(self) -> GlobalDecisionSeal:
        if self._g_seal is not None or self._issued_selection:
            raise ProtocolError("SCEPTRE global G seal was requested out of order.")
        self._g_seal = build_global_decision_seal("G_LABEL_FREE", self._g)
        return self._g_seal

    def issue_selection_capability(
        self, target_center: str, fold_ordinal: int
    ) -> PhaseCapability:
        key = self._key(target_center, fold_ordinal)
        if (
            self._g_seal is None
            or self._selection_seal is not None
            or key in self._issued_selection
            or key in self._selection
        ):
            raise ProtocolError("SCEPTRE selection capability opened out of order.")
        capability = self._capability(
            "SELECTION_LABELS", key, self._g_seal.seal_hash
        )
        self._issued_selection[key] = capability
        return capability

    def record_selection_decision(
        self,
        capability: PhaseCapability,
        decision: SupportTournamentDecision,
    ) -> None:
        key = self._validate_capability(capability, role="SELECTION_LABELS")
        if key in self._selection or not isinstance(decision, SupportTournamentDecision):
            raise ProtocolError("SCEPTRE selection decision lacks its capability.")
        fold = self._partition.fold(*key)
        model = self._frozen_router.model_for_target(key[0])
        if (
            (decision.target_center, decision.fold_ordinal) != key
            or decision.fold_hash != fold.fold_hash
            or decision.partition_hash != self._partition_hash
            or decision.selection_case_set_hash != fold.case_set_hash("SELECTION")
            or decision.calibration_case_set_hash
            != fold.case_set_hash("CALIBRATION")
            or decision.evaluation_case_set_hash != fold.case_set_hash("EVALUATION")
            or decision.router_bundle_hash != self._router_bundle_hash
            or decision.g_proposal_hash != self._g_receipts[key].g_proposal_hash
            or decision.frozen_model_hash != model.model_sha256
            or decision.decision_policy_sha256
            != self._frozen_router.decision_policy_sha256
            or decision.candidate_menu_hash != model.candidate_menu_hash
            or decision.candidate_menu_payload_sha256
            != model.candidate_menu_payload_sha256
            or decision.exact_b_control_receipt_hash
            != model.exact_b_control_receipt_hash
            or decision.exact_b_control_payload_sha256
            != model.exact_b_control_payload_sha256
            or decision.g_proposed_candidate
            != self._g_proposals[key].g_proposed_candidate
        ):
            raise ProtocolError("SCEPTRE selection decision fold lineage drifted.")
        self._selection[key] = require_sha256(
            decision.decision_hash, "selection decision"
        )
        self._selection_decisions[key] = decision

    def seal_all_selection_decisions(self) -> GlobalDecisionSeal:
        if (
            self._g_seal is None
            or self._selection_seal is not None
            or self._issued_calibration
        ):
            raise ProtocolError("SCEPTRE selection seal was requested out of order.")
        self._selection_seal = build_global_decision_seal(
            "S_Y_SELECTION",
            self._selection,
            predecessor_seal_hash=self._g_seal.seal_hash,
        )
        return self._selection_seal

    def issue_calibration_capability(
        self, target_center: str, fold_ordinal: int
    ) -> PhaseCapability:
        key = self._key(target_center, fold_ordinal)
        if (
            self._selection_seal is None
            or self._policy_seal is not None
            or key in self._issued_calibration
            or key in self._calibration
        ):
            raise ProtocolError("SCEPTRE calibration capability opened out of order.")
        capability = self._capability(
            "CALIBRATION_LABELS", key, self._selection_seal.seal_hash
        )
        self._issued_calibration[key] = capability
        return capability

    def record_calibration_decision(
        self,
        capability: PhaseCapability,
        decision: CalibrationGateDecision,
    ) -> None:
        key = self._validate_capability(capability, role="CALIBRATION_LABELS")
        if key in self._calibration or not isinstance(decision, CalibrationGateDecision):
            raise ProtocolError("SCEPTRE calibration decision lacks its capability.")
        fold = self._partition.fold(*key)
        model = self._frozen_router.model_for_target(key[0])
        support = self._selection_decisions[key]
        expected_capability_hash = self._capability_identity_hash(capability)
        uncertainty = self._calibration_uncertainty.get(key)
        if (
            (decision.target_center, decision.fold_ordinal) != key
            or decision.fold_hash != fold.fold_hash
            or decision.partition_hash != self._partition_hash
            or decision.selection_case_set_hash != fold.case_set_hash("SELECTION")
            or decision.calibration_case_set_hash
            != fold.case_set_hash("CALIBRATION")
            or decision.evaluation_case_set_hash != fold.case_set_hash("EVALUATION")
            or decision.router_bundle_hash != self._router_bundle_hash
            or decision.g_proposal_hash != self._g_receipts[key].g_proposal_hash
            or decision.g_proposed_candidate
            != self._g_proposals[key].g_proposed_candidate
            or decision.frozen_model_hash != model.model_sha256
            or decision.decision_policy_sha256
            != self._frozen_router.decision_policy_sha256
            or decision.candidate_menu_hash != model.candidate_menu_hash
            or decision.candidate_menu_payload_sha256
            != model.candidate_menu_payload_sha256
            or decision.exact_b_control_receipt_hash
            != model.exact_b_control_receipt_hash
            or decision.exact_b_control_payload_sha256
            != model.exact_b_control_payload_sha256
            or decision.support_decision_hash
            != support.decision_hash
            or decision.support_selected_candidate != support.selected_candidate
            or (
                support.selected_candidate is not None
                and (
                    uncertainty is None
                    or decision.uncertainty_decision_hash
                    != uncertainty.decision_hash
                    or decision.uncertainty_phase_capability_identity_hash
                    != expected_capability_hash
                    or decision.uncertainty_accepted is not uncertainty.accepted
                    or uncertainty.phase_capability_identity_hash
                    != expected_capability_hash
                    or uncertainty.selected_candidate
                    not in (None, support.selected_candidate)
                )
            )
            or (
                support.selected_candidate is None
                and (
                    uncertainty is not None
                    or decision.uncertainty_decision_hash is not None
                    or decision.uncertainty_phase_capability_identity_hash is not None
                )
            )
        ):
            raise ProtocolError("SCEPTRE calibration decision fold lineage drifted.")
        self._calibration[key] = require_sha256(
            decision.decision_hash, "calibration decision"
        )
        self._calibration_decisions[key] = decision

    def record_calibration_uncertainty(
        self,
        capability: PhaseCapability,
        decision: object,
    ) -> None:
        """Register the exact manager-owned uncertainty decision before A."""

        from .uncertainty import UncertaintyRouteDecision

        key = self._validate_capability(capability, role="CALIBRATION_LABELS")
        support = self._selection_decisions[key]
        if support.selected_candidate is None:
            raise ProtocolError(
                "SCEPTRE support fallback cannot register calibration uncertainty."
            )
        if key in self._calibration_uncertainty or key in self._calibration:
            raise ProtocolError("SCEPTRE calibration uncertainty was recorded twice.")
        fold = self._partition.fold(*key)
        model = self._frozen_router.model_for_target(key[0])
        capability_hash = self._capability_identity_hash(capability)
        if (
            not isinstance(decision, UncertaintyRouteDecision)
            or decision.role != "CALIBRATION"
            or (decision.target_center, decision.fold_ordinal) != key
            or decision.fold_hash != fold.fold_hash
            or decision.partition_hash != self._partition_hash
            or decision.role_case_set_hash != fold.case_set_hash("CALIBRATION")
            or decision.router_bundle_hash != self._router_bundle_hash
            or decision.g_proposal_hash != self._g_receipts[key].g_proposal_hash
            or decision.predecessor_decision_hash != support.decision_hash
            or decision.support_decision_hash != support.decision_hash
            or decision.g_proposed_candidate != support.selected_candidate
            or decision.support_selected_candidate != support.selected_candidate
            or decision.selected_candidate not in (None, support.selected_candidate)
            or decision.candidate_menu_hash != model.candidate_menu_hash
            or decision.exact_b_control_receipt_hash
            != model.exact_b_control_receipt_hash
            or decision.bootstrap_config_hash
            != self._frozen_router.dirichlet_config.config_hash
            or decision.phase_capability_identity_hash != capability_hash
        ):
            raise ProtocolError(
                "SCEPTRE calibration uncertainty fold lineage drifted."
            )
        self._calibration_uncertainty[key] = decision

    def seal_complete_policy(self) -> GlobalDecisionSeal:
        if self._selection_seal is None or self._policy_seal is not None:
            raise ProtocolError("SCEPTRE policy seal was requested out of order.")
        self._policy_seal = build_global_decision_seal(
            "A_CALIBRATED_ROUTE_OR_EXACT_B",
            self._calibration,
            predecessor_seal_hash=self._selection_seal.seal_hash,
        )
        self._frozen_route_policy = FrozenRoutePolicy(
            partition_hash=self._partition_hash,
            router_bundle_hash=self._router_bundle_hash,
            g_seal_hash=self._g_seal.seal_hash,
            selection_seal_hash=self._selection_seal.seal_hash,
            policy_seal_hash=self._policy_seal.seal_hash,
            route_rows=tuple(
                (
                    target,
                    fold,
                    self._g_receipts[(target, fold)].g_proposal_hash,
                    self._g_proposals[(target, fold)].g_proposed_candidate,
                    self._calibration_decisions[(target, fold)].route,
                    self._calibration_decisions[(target, fold)].decision_hash,
                )
                for target, fold in EXPECTED_DECISION_KEYS
            ),
        )
        return self._policy_seal

    def export_frozen_route_policy(self) -> FrozenRoutePolicy:
        """Return the canonical public route artifact after the policy seal."""

        if self._frozen_route_policy is None:
            raise ProtocolError("SCEPTRE route policy was requested before sealing.")
        return self._frozen_route_policy

    def begin_terminal_evaluation(
        self, durable_attestation: DurablePreterminalAttestation
    ) -> TerminalEvaluationCapability:
        if (
            self._policy_seal is None
            or self._frozen_route_policy is None
            or self._terminal_opened
        ):
            raise ProtocolError("SCEPTRE terminal evaluation opened out of order.")
        if (
            not isinstance(durable_attestation, DurablePreterminalAttestation)
            or durable_attestation.policy_seal_hash != self._policy_seal.seal_hash
        ):
            raise ProtocolError(
                "SCEPTRE terminal evaluation requires its durable policy attestation."
            )
        attestation = require_sha256(
            durable_attestation.attestation_hash, "durable attestation"
        )
        body = {
            "schema_version": "sceptre_terminal_evaluation_capability_v1",
            "partition_hash": self._partition_hash,
            "router_bundle_hash": self._router_bundle_hash,
            "route_policy_hash": self._frozen_route_policy.policy_artifact_hash,
            "policy_seal_hash": self._policy_seal.seal_hash,
            "durable_attestation_hash": attestation,
            "one_shot": True,
            "raw_labels_may_be_persisted": False,
        }
        self._terminal_opened = True
        return TerminalEvaluationCapability(
            partition_hash=self._partition_hash,
            router_bundle_hash=self._router_bundle_hash,
            route_policy_hash=self._frozen_route_policy.policy_artifact_hash,
            policy_seal_hash=self._policy_seal.seal_hash,
            durable_attestation_hash=attestation,
            capability_hash=canonical_hash(body),
        )

    @staticmethod
    def _key(target_center: str, fold_ordinal: int) -> DecisionKey:
        key = (str(target_center), int(fold_ordinal))
        if key not in set(EXPECTED_DECISION_KEYS):
            raise ProtocolError("SCEPTRE decision key is outside the 45-fold grid.")
        return key

    def _capability(
        self, role: str, key: DecisionKey, predecessor_seal_hash: str
    ) -> PhaseCapability:
        g_receipt = self._g_receipts.get(key)
        if g_receipt is None:
            raise ProtocolError("SCEPTRE phase capability lacks its G proposal.")
        if role == "SELECTION_LABELS":
            predecessor_decision_hash = g_receipt.receipt_hash
        else:
            selection = self._selection_decisions.get(key)
            if selection is None:
                raise ProtocolError("SCEPTRE calibration capability lacks support.")
            predecessor_decision_hash = selection.decision_hash
        return PhaseCapability(
            role=role,
            target_center=key[0],
            fold_ordinal=key[1],
            partition_hash=self._partition_hash,
            router_bundle_hash=self._router_bundle_hash,
            g_proposal_hash=g_receipt.g_proposal_hash,
            predecessor_decision_hash=predecessor_decision_hash,
            predecessor_seal_hash=predecessor_seal_hash,
            nonce_hash=secrets.token_hex(32),
        )

    def _validate_capability(
        self, capability: PhaseCapability, *, role: str
    ) -> DecisionKey:
        if not isinstance(capability, PhaseCapability) or capability.role != role:
            raise ProtocolError("SCEPTRE phase capability type or role drifted.")
        key = self._key(capability.target_center, capability.fold_ordinal)
        issued = (
            self._issued_selection
            if role == "SELECTION_LABELS"
            else self._issued_calibration
        )
        if issued.get(key) is not capability:
            raise ProtocolError("SCEPTRE phase capability is not manager-issued.")
        if (
            capability.partition_hash != self._partition_hash
            or capability.router_bundle_hash != self._router_bundle_hash
            or capability.g_proposal_hash != self._g_receipts[key].g_proposal_hash
        ):
            raise ProtocolError("SCEPTRE phase capability policy binding drifted.")
        expected_decision = (
            self._g_receipts[key].receipt_hash
            if role == "SELECTION_LABELS"
            else self._selection_decisions[key].decision_hash
        )
        if capability.predecessor_decision_hash != expected_decision:
            raise ProtocolError("SCEPTRE phase capability decision predecessor drifted.")
        if role == "SELECTION_LABELS":
            predecessor = self._g_seal
        else:
            predecessor = self._selection_seal
        if predecessor is None or capability.predecessor_seal_hash != predecessor.seal_hash:
            raise ProtocolError("SCEPTRE phase capability predecessor drifted.")
        return key

    @staticmethod
    def _capability_identity_hash(capability: PhaseCapability) -> str:
        return canonical_hash(
            {
                "schema_version": "sceptre_phase_capability_identity_v1",
                "capability_kind": "LABEL_PHASE",
                "capability_role": capability.role,
                "target_center": capability.target_center,
                "fold_ordinal": capability.fold_ordinal,
                "partition_hash": capability.partition_hash,
                "router_bundle_hash": capability.router_bundle_hash,
                "g_proposal_hash": capability.g_proposal_hash,
                "predecessor_decision_hash": (
                    capability.predecessor_decision_hash
                ),
                "predecessor_seal_hash": capability.predecessor_seal_hash,
                "nonce_hash": capability.nonce_hash,
                "manager_ownership_check_location": "phase_manager_record",
            }
        )


__all__ = (
    "PhaseCapability",
    "SceptrePhaseManager",
    "TerminalEvaluationCapability",
)
