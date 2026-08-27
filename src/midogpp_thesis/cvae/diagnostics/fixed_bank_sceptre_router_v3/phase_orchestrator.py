"""Exact global SCEPTRE G -> selection -> calibration orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.calibration_gate import (
    CalibrationGateDecision,
    apply_calibration_gate,
)
from ..fixed_bank_sceptre_router.g_proposal_persistence import FrozenGProposal
from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import FOLD_COUNT, ThreeRolePartition
from ..fixed_bank_sceptre_router.phase_order import SceptrePhaseManager
from ..fixed_bank_sceptre_router.route_policy import FrozenRoutePolicy
from ..fixed_bank_sceptre_router.seals import (
    GlobalDecisionSeal,
)
from ..fixed_bank_sceptre_router.support_tournament import (
    SupportTournamentDecision,
    select_support_family,
)
from ..fixed_bank_sceptre_router.uncertainty import (
    UncertaintyRouteDecision,
    paired_dirichlet_route_decision,
)
from .development_orchestrator import FrozenDevelopmentReplay, proposal_by_target
from .label_broker import RoleLabelBroker
from .outcome_builder import build_role_evidence


@dataclass(frozen=True, slots=True)
class SealedRoutingPhases:
    prediction_store_hash: str
    g_seal: GlobalDecisionSeal
    selection_seal: GlobalDecisionSeal
    policy_seal: GlobalDecisionSeal
    support_decisions: tuple[SupportTournamentDecision, ...]
    uncertainty_decisions: tuple[UncertaintyRouteDecision, ...]
    calibration_decisions: tuple[CalibrationGateDecision, ...]
    route_policy: FrozenRoutePolicy
    label_journal: Mapping[str, object]
    phase_hash: str

    def __post_init__(self) -> None:
        expected_count = len(CENTERS) * FOLD_COUNT
        if (
            len(self.support_decisions) != expected_count
            or len(self.calibration_decisions) != expected_count
            or self.g_seal.decision_count != expected_count
            or self.selection_seal.decision_count != expected_count
            or self.policy_seal.decision_count != expected_count
            or self.route_policy.policy_seal_hash != self.policy_seal.seal_hash
        ):
            raise ProtocolError("SCEPTRE v3 sealed phase inventory drifted.")
        require_sha256(self.prediction_store_hash, "sealed prediction store")
        if self.phase_hash != canonical_hash(self.receipt_payload()):
            raise ProtocolError("SCEPTRE v3 sealed phase hash drifted.")

    def receipt_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v3_sealed_routing_phases_v1",
            "prediction_store_hash": self.prediction_store_hash,
            "g_seal_hash": self.g_seal.seal_hash,
            "selection_seal_hash": self.selection_seal.seal_hash,
            "policy_seal_hash": self.policy_seal.seal_hash,
            "support_decision_hashes": [
                row.decision_hash for row in self.support_decisions
            ],
            "uncertainty_decision_hashes": [
                row.decision_hash for row in self.uncertainty_decisions
            ],
            "calibration_decision_hashes": [
                row.decision_hash for row in self.calibration_decisions
            ],
            "route_policy_hash": self.route_policy.policy_artifact_hash,
            "label_journal_hash": self.label_journal["journal_hash"],
            "raw_labels_persisted": False,
        }


def run_routing_phases(
    replay: FrozenDevelopmentReplay,
    *,
    partition: ThreeRolePartition,
    manager: SceptrePhaseManager,
    broker: RoleLabelBroker,
    candidate_probabilities: np.ndarray,
    exact_b_probabilities: np.ndarray,
    candidate_source_order: Sequence[str],
    prediction_store_hash: str,
    phase_observer: Callable[[str, str], None] | None = None,
) -> SealedRoutingPhases:
    """Complete all 45 decisions without retaining any raw-label surface."""

    if not isinstance(replay, FrozenDevelopmentReplay):
        raise ProtocolError("SCEPTRE v3 routing phases require development replay.")
    store_hash = require_sha256(prediction_store_hash, "prediction store")
    if (
        not isinstance(partition, ThreeRolePartition)
        or replay.router.partition_hash != partition.partition_hash
        or broker.partition_hash != partition.partition_hash
        or broker.prediction_store_hash != store_hash
    ):
        raise ProtocolError("SCEPTRE v3 routing phase lineage drifted.")
    proposals = proposal_by_target(replay)
    keys = tuple(
        (center, fold) for center in CENTERS for fold in range(FOLD_COUNT)
    )

    for target, fold_ordinal in keys:
        manager.record_label_free_g_decision(proposals[target], fold_ordinal)
    g_seal = manager.seal_all_g_decisions()
    if phase_observer is not None:
        phase_observer("ALL_G_DECISIONS_SEALED", g_seal.seal_hash)

    supports: dict[tuple[str, int], SupportTournamentDecision] = {}
    for target, fold_ordinal in keys:
        fold = partition.fold(target, fold_ordinal)
        capability = broker.issue_selection(target, fold_ordinal)
        scoped = broker.open_selection(capability)
        model = replay.router.model_for_target(target)
        evidence = build_role_evidence(
            scoped,
            fold=fold,
            partition_hash=partition.partition_hash,
            candidate_probabilities=candidate_probabilities,
            exact_b_probabilities=exact_b_probabilities,
            candidate_source_order=candidate_source_order,
            prediction_store_hash=store_hash,
            candidate_menu_hash=model.candidate_menu_hash,
            exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
            phase_capability=capability,
        )
        if evidence.exact_b is None:
            raise ProtocolError("SCEPTRE v3 selection lacks exact B.")
        decision = select_support_family(
            evidence.outcomes,
            target_center=target,
            fold=fold,
            partition_hash=partition.partition_hash,
            exact_b=evidence.exact_b,
            g_proposal=proposals[target],
            frozen_router=replay.router,
        )
        manager.record_selection_decision(capability, decision)
        supports[(target, fold_ordinal)] = decision
        del evidence, scoped
    selection_seal = manager.seal_all_selection_decisions()
    if phase_observer is not None:
        phase_observer(
            "ALL_SELECTION_DECISIONS_SEALED", selection_seal.seal_hash
        )

    calibration: dict[tuple[str, int], CalibrationGateDecision] = {}
    uncertainty: dict[tuple[str, int], UncertaintyRouteDecision] = {}
    for target, fold_ordinal in keys:
        fold = partition.fold(target, fold_ordinal)
        capability = broker.issue_calibration(target, fold_ordinal)
        support = supports[(target, fold_ordinal)]
        if support.fallback_required:
            broker.skip_calibration_without_labels(
                capability, support_decision_hash=support.decision_hash
            )
            decision = apply_calibration_gate(
                support,
                uncertainty=None,
                candidate=None,
                exact_b=None,
                frozen_router=replay.router,
            )
        else:
            scoped = broker.open_calibration(capability)
            model = replay.router.model_for_target(target)
            evidence = build_role_evidence(
                scoped,
                fold=fold,
                partition_hash=partition.partition_hash,
                candidate_probabilities=candidate_probabilities,
                exact_b_probabilities=exact_b_probabilities,
                candidate_source_order=candidate_source_order,
                prediction_store_hash=store_hash,
                candidate_menu_hash=model.candidate_menu_hash,
                exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
                phase_capability=capability,
            )
            selected = support.selected_candidate
            if selected is None or evidence.exact_b is None:
                raise ProtocolError("SCEPTRE v3 accepted support lost its candidate.")
            uncertain = paired_dirichlet_route_decision(
                evidence.surface,
                g_proposed_candidate=selected,
                support_selected_candidate=selected,
                config=replay.router.dirichlet_config,
            )
            manager.record_calibration_uncertainty(capability, uncertain)
            outcomes = {row.candidate_center: row for row in evidence.outcomes}
            decision = apply_calibration_gate(
                support,
                uncertainty=uncertain,
                candidate=outcomes[selected],
                exact_b=evidence.exact_b,
                frozen_router=replay.router,
            )
            uncertainty[(target, fold_ordinal)] = uncertain
            del evidence, scoped
        manager.record_calibration_decision(capability, decision)
        calibration[(target, fold_ordinal)] = decision
    policy_seal = manager.seal_complete_policy()
    if phase_observer is not None:
        phase_observer(
            "ALL_CALIBRATION_DECISIONS_SEALED", policy_seal.seal_hash
        )
    route_policy = manager.export_frozen_route_policy()
    journal = broker.journal_payload()

    support_rows = tuple(supports[key] for key in keys)
    uncertainty_rows = tuple(uncertainty[key] for key in keys if key in uncertainty)
    calibration_rows = tuple(calibration[key] for key in keys)
    body = {
        "schema_version": "sceptre_v3_sealed_routing_phases_v1",
        "prediction_store_hash": store_hash,
        "g_seal_hash": g_seal.seal_hash,
        "selection_seal_hash": selection_seal.seal_hash,
        "policy_seal_hash": policy_seal.seal_hash,
        "support_decision_hashes": [row.decision_hash for row in support_rows],
        "uncertainty_decision_hashes": [
            row.decision_hash for row in uncertainty_rows
        ],
        "calibration_decision_hashes": [
            row.decision_hash for row in calibration_rows
        ],
        "route_policy_hash": route_policy.policy_artifact_hash,
        "label_journal_hash": journal["journal_hash"],
        "raw_labels_persisted": False,
    }
    return SealedRoutingPhases(
        prediction_store_hash=store_hash,
        g_seal=g_seal,
        selection_seal=selection_seal,
        policy_seal=policy_seal,
        support_decisions=support_rows,
        uncertainty_decisions=uncertainty_rows,
        calibration_decisions=calibration_rows,
        route_policy=route_policy,
        label_journal=MappingProxyType(dict(journal)),
        phase_hash=canonical_hash(body),
    )


__all__ = ("SealedRoutingPhases", "run_routing_phases")
