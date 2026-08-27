"""Pre-test-label SCEPTRE development replay for executable v3.

This module consumes the immutable, historically label-derived source-inner
utility surface and its pre-label prediction packet.  That surface is
explicitly exhausted and is used only for adaptive, descriptive development.
It never sees the MIDOG++ test manifest or any target-test label.  Execution
authority lives in the v3 admission layer; the scientific objects remain the
frozen inherited SCEPTRE contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.generation.contracts import GenerationLock
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre import build_candidate_menu

from ..fixed_bank_sceptre_router.adaptive_model_freeze import (
    AdaptiveUtilityDecision,
    FrozenAdaptiveUtilityModel,
    freeze_adaptive_utility_model,
    route_frozen_predicted_utility_or_exact_b,
)
from ..fixed_bank_sceptre_router.development_model import (
    NestedLodoFit,
    fit_nested_lodo_pairwise_ranker,
)
from ..fixed_bank_sceptre_router.development_surface import (
    SourceInnerDevelopmentSurface,
)
from ..fixed_bank_sceptre_router.evidence_builder import (
    build_outer_development_evidence,
    build_target_prediction_evidence,
)
from ..fixed_bank_sceptre_router.frozen_router_bundle import (
    FrozenPrelabelRouter,
    freeze_full_prelabel_router,
)
from ..fixed_bank_sceptre_router.g_proposal_persistence import FrozenGProposal
from ..fixed_bank_sceptre_router.hashing import canonical_hash
from ..fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ..fixed_bank_sceptre_router.source_inner_evidence import (
    SourceInnerPredictionSurface,
    build_outer_raw_evidence,
    build_target_raw_evidence,
)
from ..fixed_bank_sceptre_router.uncertainty import DirichletBootstrapConfig


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentReplay:
    """Complete nine-target pre-test-label development replay."""

    fits: tuple[NestedLodoFit, ...]
    models: tuple[FrozenAdaptiveUtilityModel, ...]
    decisions: tuple[AdaptiveUtilityDecision, ...]
    router: FrozenPrelabelRouter
    proposals: tuple[FrozenGProposal, ...]
    replay_hash: str

    def __post_init__(self) -> None:
        if (
            tuple(row.outer_target for row in self.fits) != CENTERS
            or tuple(row.outer_target for row in self.models) != CENTERS
            or tuple(row.outer_target for row in self.decisions) != CENTERS
            or tuple(row.target_center for row in self.proposals) != CENTERS
            or self.router.models != self.models
        ):
            raise ProtocolError("SCEPTRE v3 development target inventory drifted.")
        expected = canonical_hash(self.receipt_payload())
        if self.replay_hash != expected:
            raise ProtocolError("SCEPTRE v3 development replay hash drifted.")

    def receipt_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v3_label_free_development_replay_v1",
            "target_order": list(CENTERS),
            "fit_receipts": [
                {
                    "outer_target": fit.outer_target,
                    "selected_alpha": fit.selected_alpha,
                    "outer_evidence_receipt_hash": fit.outer_evidence_receipt_hash,
                    "final_training_receipt_hash": fit.final_model.training_receipt_hash,
                    "assessment_hash": canonical_hash(
                        [asdict(row) for row in fit.assessments]
                    ),
                }
                for fit in self.fits
            ],
            "model_sha256_by_target": [
                [model.outer_target, model.model_sha256] for model in self.models
            ],
            "adaptive_decision_sha256_by_target": [
                [decision.outer_target, decision.decision_sha256]
                for decision in self.decisions
            ],
            "full_router_sha256": self.router.full_router_sha256,
            "g_proposal_sha256_by_target": [
                [proposal.target_center, proposal.proposal_sha256]
                for proposal in self.proposals
            ],
            "source_inner_only": True,
            "test_labels_consumed": False,
            "fresh_evidence": False,
        }


def fit_and_freeze_development_router(
    development_surface: SourceInnerDevelopmentSurface,
    prediction_surface: SourceInnerPredictionSurface,
    *,
    generation_lock: GenerationLock,
    partition: ThreeRolePartition,
    dirichlet_config: DirichletBootstrapConfig | None = None,
) -> FrozenDevelopmentReplay:
    """Replay nested LODO, freeze all H models, then create target-global G.

    The full router is frozen before any G proposal is attached.  G is one
    target-global decision per H; the five fold records created later are only
    phase-barrier attachments of the same proposal.
    """

    if not isinstance(development_surface, SourceInnerDevelopmentSurface):
        raise ProtocolError("SCEPTRE v3 requires its typed development surface.")
    if not isinstance(prediction_surface, SourceInnerPredictionSurface):
        raise ProtocolError("SCEPTRE v3 requires its typed prediction surface.")
    if not isinstance(generation_lock, GenerationLock):
        raise ProtocolError("SCEPTRE v3 requires a validated GenerationLock.")
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE v3 requires its whole-case partition.")

    fits: list[NestedLodoFit] = []
    models: list[FrozenAdaptiveUtilityModel] = []
    target_evidence: dict[str, object] = {}
    menus: dict[str, object] = {}
    for target in CENTERS:
        outer_raw = build_outer_raw_evidence(
            prediction_surface, outer_target=target
        )
        outer = build_outer_development_evidence(
            outer_raw.rows,
            outer_target=target,
            raw_source_receipt_hash=outer_raw.packet_hash,
        )
        fit = fit_nested_lodo_pairwise_ranker(
            development_surface.for_outer_target(target), outer
        )
        menu = build_candidate_menu(generation_lock, target)
        model = freeze_adaptive_utility_model(
            fit, generation_lock=generation_lock, candidate_menu=menu
        )
        raw_target = build_target_raw_evidence(
            prediction_surface, target_center=target
        )
        target_evidence[target] = build_target_prediction_evidence(
            raw_target.rows,
            target_center=target,
            raw_source_receipt_hash=raw_target.packet_hash,
        )
        menus[target] = menu
        fits.append(fit)
        models.append(model)

    router = freeze_full_prelabel_router(
        tuple(models),
        generation_lock=generation_lock,
        partition=partition,
        dirichlet_config=dirichlet_config,
    )
    decisions: list[AdaptiveUtilityDecision] = []
    proposals: list[FrozenGProposal] = []
    for target in CENTERS:
        decision = route_frozen_predicted_utility_or_exact_b(
            router.model_for_target(target),
            target_evidence[target],
            generation_lock=generation_lock,
            candidate_menu=menus[target],
        )
        decisions.append(decision)
        proposals.append(router.bind_g_proposal(decision))

    # Construct the hash from the same closed receipt body validated by the DTO.
    body = {
        "schema_version": "sceptre_v3_label_free_development_replay_v1",
        "target_order": list(CENTERS),
        "fit_receipts": [
            {
                "outer_target": fit.outer_target,
                "selected_alpha": fit.selected_alpha,
                "outer_evidence_receipt_hash": fit.outer_evidence_receipt_hash,
                "final_training_receipt_hash": fit.final_model.training_receipt_hash,
                "assessment_hash": canonical_hash(
                    [asdict(row) for row in fit.assessments]
                ),
            }
            for fit in fits
        ],
        "model_sha256_by_target": [
            [model.outer_target, model.model_sha256] for model in models
        ],
        "adaptive_decision_sha256_by_target": [
            [decision.outer_target, decision.decision_sha256]
            for decision in decisions
        ],
        "full_router_sha256": router.full_router_sha256,
        "g_proposal_sha256_by_target": [
            [proposal.target_center, proposal.proposal_sha256]
            for proposal in proposals
        ],
        "source_inner_only": True,
        "test_labels_consumed": False,
        "fresh_evidence": False,
    }
    return FrozenDevelopmentReplay(
        fits=tuple(fits),
        models=tuple(models),
        decisions=tuple(decisions),
        router=router,
        proposals=tuple(proposals),
        replay_hash=canonical_hash(body),
    )


def proposal_by_target(
    replay: FrozenDevelopmentReplay,
) -> Mapping[str, FrozenGProposal]:
    if not isinstance(replay, FrozenDevelopmentReplay):
        raise ProtocolError("SCEPTRE v3 proposal lookup requires its replay.")
    return MappingProxyType({row.target_center: row for row in replay.proposals})


__all__ = (
    "FrozenDevelopmentReplay",
    "fit_and_freeze_development_router",
    "proposal_by_target",
)
