"""End-to-end v4 proposal-set, support and confirmation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import FOLD_COUNT, ThreeRolePartition
from ..fixed_bank_sceptre_router.seals import GlobalDecisionSeal
from .confirmation_gate import ConfirmationDecision, apply_confirmation_gate
from .development import FrozenDevelopmentReplay
from .label_broker import RoleLabelBroker
from .outcome_builder import build_role_evidence
from .phase_manager import CandidateSetPhaseManager
from .posterior import PairedCandidatePosterior, build_calibration_posterior
from .route_policy import FrozenRoutePolicy
from .support_posterior import SupportPosteriorDecision, select_support_candidate


@dataclass(frozen=True, slots=True)
class SealedRoutingPhases:
    prediction_store_hash: str
    proposal_set_seal: GlobalDecisionSeal
    support_seal: GlobalDecisionSeal
    policy_seal: GlobalDecisionSeal
    support_decisions: tuple[SupportPosteriorDecision, ...]
    calibration_posteriors: tuple[PairedCandidatePosterior, ...]
    confirmation_decisions: tuple[ConfirmationDecision, ...]
    route_policy: FrozenRoutePolicy
    label_journal: Mapping[str, object]
    phase_hash: str = ""

    def __post_init__(self) -> None:
        expected = len(CENTERS) * FOLD_COUNT
        support_keys = tuple(
            (row.target_center, row.fold_ordinal)
            for row in self.support_decisions
        )
        confirmation_keys = tuple(
            (row.target_center, row.fold_ordinal)
            for row in self.confirmation_decisions
        )
        posterior_keys = tuple(
            (row.target_center, row.fold_ordinal)
            for row in self.calibration_posteriors
        )
        selected_support = tuple(
            row for row in self.support_decisions if row.selected_candidate is not None
        )
        expected_posterior_keys = tuple(
            (row.target_center, row.fold_ordinal) for row in selected_support
        )
        support_by_key = {key: row for key, row in zip(support_keys, self.support_decisions)}
        posterior_by_key = {
            key: row for key, row in zip(posterior_keys, self.calibration_posteriors)
        }
        confirmation_by_key = {
            key: row for key, row in zip(confirmation_keys, self.confirmation_decisions)
        }
        if (
            len(self.support_decisions) != expected
            or len(self.confirmation_decisions) != expected
            or support_keys != tuple(
                (target, fold)
                for target in CENTERS
                for fold in range(FOLD_COUNT)
            )
            or confirmation_keys != support_keys
            or posterior_keys != expected_posterior_keys
            or any(
                posterior.support_decision_hash
                != support_by_key[key].decision_hash
                or posterior.candidate_center
                != support_by_key[key].selected_candidate
                for key, posterior in posterior_by_key.items()
            )
            or any(
                confirmation.support_decision_hash
                != support_by_key[key].decision_hash
                or confirmation.support_selected_candidate
                != support_by_key[key].selected_candidate
                or confirmation.posterior_hash
                != (
                    None
                    if key not in posterior_by_key
                    else posterior_by_key[key].posterior_hash
                )
                for key, confirmation in confirmation_by_key.items()
            )
            or self.proposal_set_seal.decision_count != expected
            or self.support_seal.decision_count != expected
            or self.policy_seal.decision_count != expected
            or self.route_policy.policy_seal_hash != self.policy_seal.seal_hash
            or any(
                policy_row[3] != support_by_key[policy_row[:2]].decision_hash
                or policy_row[4]
                != support_by_key[policy_row[:2]].selected_candidate
                or policy_row[5]
                != confirmation_by_key[policy_row[:2]].route
                or policy_row[6]
                != confirmation_by_key[policy_row[:2]].decision_hash
                for policy_row in self.route_policy.route_rows
            )
        ):
            raise ProtocolError("SCEPTRE v4 sealed phase inventory drifted.")
        require_sha256(self.prediction_store_hash, "prediction store")
        expected_hash = canonical_hash(self._payload_without_hash())
        if self.phase_hash and self.phase_hash != expected_hash:
            raise ProtocolError("SCEPTRE v4 sealed phase hash drifted.")
        object.__setattr__(self, "phase_hash", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v4_sealed_routing_phases_v1",
            "prediction_store_hash": self.prediction_store_hash,
            "proposal_set_seal_hash": self.proposal_set_seal.seal_hash,
            "support_seal_hash": self.support_seal.seal_hash,
            "policy_seal_hash": self.policy_seal.seal_hash,
            "support_decision_hashes": [
                row.decision_hash for row in self.support_decisions
            ],
            "calibration_posterior_hashes": [
                row.posterior_hash for row in self.calibration_posteriors
            ],
            "confirmation_decision_hashes": [
                row.decision_hash for row in self.confirmation_decisions
            ],
            "route_policy_hash": self.route_policy.policy_artifact_hash,
            "label_journal_hash": self.label_journal["journal_hash"],
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "phase_hash": self.phase_hash}


def run_routing_phases(
    replay: FrozenDevelopmentReplay,
    *,
    partition: ThreeRolePartition,
    manager: CandidateSetPhaseManager,
    broker: RoleLabelBroker,
    candidate_probabilities: np.ndarray,
    exact_b_probabilities: np.ndarray,
    candidate_source_order: Sequence[str],
    prediction_store_hash: str,
    phase_observer: Callable[[str, str], None] | None = None,
) -> SealedRoutingPhases:
    if not isinstance(replay, FrozenDevelopmentReplay):
        raise ProtocolError("SCEPTRE v4 routing requires development replay.")
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE v4 routing requires partition.")
    if not isinstance(manager, CandidateSetPhaseManager):
        raise ProtocolError("SCEPTRE v4 routing requires manager.")
    if not isinstance(broker, RoleLabelBroker):
        raise ProtocolError("SCEPTRE v4 routing requires label broker.")
    store_hash = require_sha256(prediction_store_hash, "prediction store")
    if (
        replay.context.partition_hash != partition.partition_hash
        or broker.partition_hash != partition.partition_hash
        or broker.prediction_store_hash != store_hash
    ):
        raise ProtocolError("SCEPTRE v4 routing lineage drifted.")
    keys = tuple(
        (target, fold) for target in CENTERS for fold in range(FOLD_COUNT)
    )

    for target, fold in keys:
        manager.record_label_free_proposal_set(
            replay.proposal_for_target(target), fold
        )
    proposal_seal = manager.seal_all_proposal_sets()
    if phase_observer:
        phase_observer("ALL_PROPOSAL_SETS_SEALED", proposal_seal.seal_hash)

    support_by_key = {}
    for target, fold_ordinal in keys:
        fold = partition.fold(target, fold_ordinal)
        capability = broker.issue_selection(target, fold_ordinal)
        scoped = broker.open_selection(capability)
        model = replay.context.model_for_target(target)
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
            raise ProtocolError("SCEPTRE v4 support lacks exact B.")
        support = select_support_candidate(
            evidence.outcomes,
            exact_b=evidence.exact_b,
            fold=fold,
            partition_hash=partition.partition_hash,
            proposal_set=replay.proposal_for_target(target),
            routing_context=replay.context,
        )
        manager.record_selection_decision(capability, support)
        support_by_key[(target, fold_ordinal)] = support
        del evidence, scoped
    support_seal = manager.seal_all_selection_decisions()
    if phase_observer:
        phase_observer("ALL_SUPPORT_DECISIONS_SEALED", support_seal.seal_hash)

    posterior_by_key = {}
    confirmation_by_key = {}
    for target, fold_ordinal in keys:
        fold = partition.fold(target, fold_ordinal)
        support = support_by_key[(target, fold_ordinal)]
        capability = broker.issue_calibration(target, fold_ordinal)
        if support.fallback_required:
            broker.skip_calibration_without_labels(
                capability, support_decision_hash=support.decision_hash
            )
            confirmation = apply_confirmation_gate(
                support,
                posterior=None,
                candidate=None,
                exact_b=None,
                routing_context=replay.context,
            )
        else:
            scoped = broker.open_calibration(capability)
            model = replay.context.model_for_target(target)
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
            posterior = build_calibration_posterior(
                evidence.surface,
                support,
                routing_context_hash=replay.context.context_hash,
                config=replay.context.dirichlet_config,
            )
            outcomes = {row.candidate_center: row for row in evidence.outcomes}
            if evidence.exact_b is None or support.selected_candidate is None:
                raise ProtocolError("SCEPTRE v4 calibration lost its selected member.")
            confirmation = apply_confirmation_gate(
                support,
                posterior=posterior,
                candidate=outcomes[support.selected_candidate],
                exact_b=evidence.exact_b,
                routing_context=replay.context,
            )
            posterior_by_key[(target, fold_ordinal)] = posterior
            del evidence, scoped
        manager.record_calibration_decision(capability, confirmation)
        confirmation_by_key[(target, fold_ordinal)] = confirmation
    policy_seal = manager.seal_complete_policy()
    if phase_observer:
        phase_observer("ALL_CONFIRMATION_DECISIONS_SEALED", policy_seal.seal_hash)
    policy = manager.export_frozen_route_policy()
    journal = broker.journal_payload()
    return SealedRoutingPhases(
        prediction_store_hash=store_hash,
        proposal_set_seal=proposal_seal,
        support_seal=support_seal,
        policy_seal=policy_seal,
        support_decisions=tuple(support_by_key[key] for key in keys),
        calibration_posteriors=tuple(
            posterior_by_key[key] for key in keys if key in posterior_by_key
        ),
        confirmation_decisions=tuple(confirmation_by_key[key] for key in keys),
        route_policy=policy,
        label_journal=MappingProxyType(dict(journal)),
    )


__all__ = ("SealedRoutingPhases", "run_routing_phases")
