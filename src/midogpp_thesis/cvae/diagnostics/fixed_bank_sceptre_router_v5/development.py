"""Strict outer-center development and pre-label SCEPTRE v5 freeze."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.generation.contracts import GenerationLock
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre import build_candidate_menu

from ..fixed_bank_sceptre_router.adaptive_model_freeze import (
    FrozenAdaptiveUtilityModel,
    freeze_adaptive_utility_model,
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
from ..fixed_bank_sceptre_router.hashing import canonical_hash
from ..fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ..fixed_bank_sceptre_router.source_inner_evidence import (
    SourceInnerPredictionSurface,
    build_outer_raw_evidence,
    build_target_raw_evidence,
)
from ..fixed_bank_sceptre_router.uncertainty import DirichletBootstrapConfig
from .identity import POLICY_TRANSITION, PUBLICATION_STATUS, TERMINAL_DECISION
from .proposal_set import FrozenCandidateSetProposal, build_candidate_set_proposal


SUPPORT_PRIOR_EFFECTIVE_CASES = 8.0
SUPPORT_MINIMUM_SHRUNK_BACC_GAIN = 0.0
CALIBRATION_MINIMUM_BACC_GAIN = 0.0
CALIBRATION_MAXIMUM_BRIER_DELTA = 0.0
CALIBRATION_MAXIMUM_LOG_LOSS_DELTA = 0.0


@dataclass(frozen=True, slots=True)
class FrozenRoutingContext:
    """All model, partition, threshold and claim state frozen pre-label."""

    models: tuple[FrozenAdaptiveUtilityModel, ...]
    partition_hash: str
    partition_identity_sha256: str
    partition_fold_inventory_sha256: str
    dirichlet_config: DirichletBootstrapConfig
    context_hash: str = ""

    def __post_init__(self) -> None:
        models = tuple(self.models)
        if (
            len(models) != len(CENTERS)
            or tuple(row.outer_target for row in models) != CENTERS
            or any(not isinstance(row, FrozenAdaptiveUtilityModel) for row in models)
        ):
            raise ProtocolError("SCEPTRE v5 context lacks nine ordered models.")
        if len({row.generation_lock_payload_sha256 for row in models}) != 1:
            raise ProtocolError("SCEPTRE v5 models do not share one GenerationLock.")
        if not isinstance(self.dirichlet_config, DirichletBootstrapConfig):
            raise ProtocolError("SCEPTRE v5 context lacks its bootstrap config.")
        body = self._payload_without_hash(models)
        expected = canonical_hash(body)
        if self.context_hash and self.context_hash != expected:
            raise ProtocolError("SCEPTRE v5 routing-context hash drifted.")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "context_hash", expected)

    def _payload_without_hash(
        self, models: tuple[FrozenAdaptiveUtilityModel, ...] | None = None
    ) -> dict[str, object]:
        rows = self.models if models is None else models
        return {
            "schema_version": "sceptre_v5_frozen_routing_context_v1",
            "model_sha256_by_target": [
                [row.outer_target, row.model_sha256] for row in rows
            ],
            "partition_hash": self.partition_hash,
            "partition_identity_sha256": self.partition_identity_sha256,
            "partition_fold_inventory_sha256": (
                self.partition_fold_inventory_sha256
            ),
            "dirichlet_config": self.dirichlet_config.to_payload(),
            "support_prior_effective_cases": SUPPORT_PRIOR_EFFECTIVE_CASES,
            "support_minimum_shrunk_bacc_gain": (
                SUPPORT_MINIMUM_SHRUNK_BACC_GAIN
            ),
            "calibration_minimum_bacc_gain": CALIBRATION_MINIMUM_BACC_GAIN,
            "calibration_maximum_brier_delta": (
                CALIBRATION_MAXIMUM_BRIER_DELTA
            ),
            "calibration_maximum_log_loss_delta": (
                CALIBRATION_MAXIMUM_LOG_LOSS_DELTA
            ),
            "policy_transition": POLICY_TRANSITION,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "nelbo_compatibility_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "context_hash": self.context_hash}

    def model_for_target(self, target_center: str) -> FrozenAdaptiveUtilityModel:
        target = str(target_center)
        try:
            return next(row for row in self.models if row.outer_target == target)
        except StopIteration as exc:
            raise ProtocolError("SCEPTRE v5 target model is absent.") from exc


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentReplay:
    """Nine nested-LODO fits and full ranked sets frozen before labels."""

    fits: tuple[NestedLodoFit, ...]
    context: FrozenRoutingContext
    proposal_sets: tuple[FrozenCandidateSetProposal, ...]
    replay_hash: str = ""

    def __post_init__(self) -> None:
        fits = tuple(self.fits)
        proposals = tuple(self.proposal_sets)
        if (
            tuple(row.outer_target for row in fits) != CENTERS
            or tuple(row.target_center for row in proposals) != CENTERS
        ):
            raise ProtocolError("SCEPTRE v5 development inventory drifted.")
        for fit, model, proposal in zip(
            fits, self.context.models, proposals, strict=True
        ):
            if (
                fit.outer_target != model.outer_target
                or fit.final_model.training_receipt_hash
                != model.training_receipt_sha256
                or proposal.frozen_model_sha256 != model.model_sha256
            ):
                raise ProtocolError("SCEPTRE v5 fit/model/proposal lineage drifted.")
        body = self._payload_without_hash(fits, proposals)
        expected = canonical_hash(body)
        if self.replay_hash and self.replay_hash != expected:
            raise ProtocolError("SCEPTRE v5 development replay hash drifted.")
        object.__setattr__(self, "fits", fits)
        object.__setattr__(self, "proposal_sets", proposals)
        object.__setattr__(self, "replay_hash", expected)

    def _payload_without_hash(
        self,
        fits: tuple[NestedLodoFit, ...] | None = None,
        proposals: tuple[FrozenCandidateSetProposal, ...] | None = None,
    ) -> dict[str, object]:
        fit_rows = self.fits if fits is None else fits
        proposal_rows = self.proposal_sets if proposals is None else proposals
        return {
            "schema_version": "sceptre_v5_label_free_development_replay_v1",
            "target_order": list(CENTERS),
            "fit_receipts": [
                {
                    "outer_target": fit.outer_target,
                    "selected_alpha": fit.selected_alpha,
                    "outer_evidence_receipt_hash": (
                        fit.outer_evidence_receipt_hash
                    ),
                    "final_training_receipt_hash": (
                        fit.final_model.training_receipt_hash
                    ),
                    "assessment_hash": canonical_hash(
                        [asdict(row) for row in fit.assessments]
                    ),
                }
                for fit in fit_rows
            ],
            "routing_context_hash": self.context.context_hash,
            "proposal_set_sha256_by_target": [
                [row.target_center, row.proposal_set_hash]
                for row in proposal_rows
            ],
            "all_eight_experts_retained": True,
            "exact_b_advantage_model_available": False,
            "source_inner_only": True,
            "target_test_labels_consumed": False,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "replay_hash": self.replay_hash}

    def proposal_for_target(self, target_center: str) -> FrozenCandidateSetProposal:
        target = str(target_center)
        try:
            return next(row for row in self.proposal_sets if row.target_center == target)
        except StopIteration as exc:
            raise ProtocolError("SCEPTRE v5 proposal set is absent.") from exc

    @property
    def proposals_by_target(self) -> Mapping[str, FrozenCandidateSetProposal]:
        return MappingProxyType(
            {row.target_center: row for row in self.proposal_sets}
        )


def fit_and_freeze_development(
    development_surface: SourceInnerDevelopmentSurface,
    prediction_surface: SourceInnerPredictionSurface,
    *,
    generation_lock: GenerationLock,
    partition: ThreeRolePartition,
    dirichlet_config: DirichletBootstrapConfig | None = None,
) -> FrozenDevelopmentReplay:
    """Fit strict outer/nested LODO rankers and retain every expert score."""

    if not isinstance(development_surface, SourceInnerDevelopmentSurface):
        raise ProtocolError("SCEPTRE v5 requires its development surface.")
    if not isinstance(prediction_surface, SourceInnerPredictionSurface):
        raise ProtocolError("SCEPTRE v5 requires its prediction surface.")
    if not isinstance(generation_lock, GenerationLock):
        raise ProtocolError("SCEPTRE v5 requires its GenerationLock.")
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE v5 requires its case partition.")

    fits: list[NestedLodoFit] = []
    models: list[FrozenAdaptiveUtilityModel] = []
    proposals: list[FrozenCandidateSetProposal] = []
    for target in CENTERS:
        outer_raw = build_outer_raw_evidence(
            prediction_surface, outer_target=target
        )
        outer_evidence = build_outer_development_evidence(
            outer_raw.rows,
            outer_target=target,
            raw_source_receipt_hash=outer_raw.packet_hash,
        )
        fit = fit_nested_lodo_pairwise_ranker(
            development_surface.for_outer_target(target), outer_evidence
        )
        menu = build_candidate_menu(generation_lock, target)
        model = freeze_adaptive_utility_model(
            fit, generation_lock=generation_lock, candidate_menu=menu
        )
        target_raw = build_target_raw_evidence(
            prediction_surface, target_center=target
        )
        target_evidence = build_target_prediction_evidence(
            target_raw.rows,
            target_center=target,
            raw_source_receipt_hash=target_raw.packet_hash,
        )
        fits.append(fit)
        models.append(model)
        proposals.append(build_candidate_set_proposal(model, target_evidence))

    settings = DirichletBootstrapConfig() if dirichlet_config is None else dirichlet_config
    context = FrozenRoutingContext(
        models=tuple(models),
        partition_hash=partition.partition_hash,
        partition_identity_sha256=_partition_identity_hash(partition),
        partition_fold_inventory_sha256=_partition_fold_inventory_hash(partition),
        dirichlet_config=settings,
    )
    return FrozenDevelopmentReplay(
        fits=tuple(fits),
        context=context,
        proposal_sets=tuple(proposals),
    )


def _partition_identity_hash(partition: ThreeRolePartition) -> str:
    return canonical_hash(
        {
            "schema_version": "sceptre_v5_case_identity_inventory_v1",
            "rows": [
                {
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                }
                for row in partition.identities
            ],
            "whole_case_partitioning": True,
            "labels_consumed": False,
        }
    )


def _partition_fold_inventory_hash(partition: ThreeRolePartition) -> str:
    return canonical_hash(
        {
            "schema_version": "sceptre_v5_prelabel_fold_inventory_v1",
            "partition_hash": partition.partition_hash,
            "folds": [
                {
                    "target_center": fold.target_center,
                    "fold_ordinal": fold.fold_ordinal,
                    "fold_hash": fold.fold_hash,
                    "selection_case_set_hash": fold.case_set_hash("SELECTION"),
                    "calibration_case_set_hash": fold.case_set_hash("CALIBRATION"),
                    "evaluation_case_set_hash": fold.case_set_hash("EVALUATION"),
                }
                for fold in partition.folds
            ],
            "whole_case_roles_disjoint": True,
            "every_case_evaluated_exactly_once": True,
        }
    )


__all__ = (
    "CALIBRATION_MAXIMUM_BRIER_DELTA",
    "CALIBRATION_MAXIMUM_LOG_LOSS_DELTA",
    "CALIBRATION_MINIMUM_BACC_GAIN",
    "FrozenDevelopmentReplay",
    "FrozenRoutingContext",
    "SUPPORT_MINIMUM_SHRUNK_BACC_GAIN",
    "SUPPORT_PRIOR_EFFECTIVE_CASES",
    "fit_and_freeze_development",
)
