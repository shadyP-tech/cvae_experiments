"""End-to-end source-science orchestration with exact-P failure semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    AdmissionCase,
    ActionUtilityObservation,
    BaccRankingPolicy,
    OOFResidualObservation,
    OpportunityCaseReceipt,
    PairwiseRankerModel,
    RowPosteriorModel,
    RowPosteriorObservation,
    RowPosteriorOOFPrediction,
    UncertaintyCalibration,
    calibrate_clustered_uncertainty,
    canonical_sha256,
    crossfit_source_row_posterior,
    fit_final_source_row_posterior,
)
from ..feature_engineering import FEATURE_DEFINITION_RECEIPT_HASH
from ..folds import OuterFoldPlanV3
from .admission import (
    ExactPFallbackReceipt,
    SourceOrderingAdmissionReceipt,
    evaluate_source_ordering_admission,
    exact_p_fail_closed_reason,
)
from .pool_indexed_pairwise_fit import fit_pool_indexed_pairwise_ranker


@dataclass(frozen=True, slots=True)
class OuterScienceResult:
    outer_target_center: str
    plan_hash: str
    source_surface_lineage_hash: str
    admitted: bool
    row_posterior_model: RowPosteriorModel | None
    row_oof_predictions: tuple[RowPosteriorOOFPrediction, ...]
    pairwise_model: PairwiseRankerModel | None
    uncertainty_calibration: UncertaintyCalibration | None
    admission: SourceOrderingAdmissionReceipt | None
    fallback: ExactPFallbackReceipt | None
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        success = (
            isinstance(self.row_posterior_model, RowPosteriorModel)
            and bool(self.row_oof_predictions)
            and isinstance(self.pairwise_model, PairwiseRankerModel)
            and isinstance(self.uncertainty_calibration, UncertaintyCalibration)
            and isinstance(self.admission, SourceOrderingAdmissionReceipt)
            and self.admission.admitted
            and self.fallback is None
        )
        failure = (
            isinstance(self.fallback, ExactPFallbackReceipt)
            and not self.admitted
        )
        if bool(self.admitted) != success or (not self.admitted and not failure):
            raise ProtocolError("OE-PPUR v3 outer-science result is not fail-closed.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_outer_science_result_v1",
                    "H": self.outer_target_center,
                    "plan_hash": self.plan_hash,
                    "source_surface_lineage_hash": self.source_surface_lineage_hash,
                    "admitted": self.admitted,
                    "row_posterior_model_hash": (
                        None if self.row_posterior_model is None
                        else self.row_posterior_model.model_hash
                    ),
                    "row_oof_prediction_lineage": tuple(
                        (
                            row.center_id,
                            row.case_id,
                            row.row_id,
                            row.model_hash,
                            row.source_scope_receipt_hash,
                        )
                        for row in self.row_oof_predictions
                    ),
                    "pairwise_model_hash": (
                        None if self.pairwise_model is None else self.pairwise_model.model_hash
                    ),
                    "uncertainty_calibration_hash": (
                        None if self.uncertainty_calibration is None
                        else self.uncertainty_calibration.calibration_hash
                    ),
                    "admission_receipt_hash": (
                        None if self.admission is None else self.admission.receipt_hash
                    ),
                    "fallback_receipt_hash": (
                        None if self.fallback is None else self.fallback.receipt_hash
                    ),
                    "target_labels_used": False,
                }
            ),
        )


def _source_centers(plan: OuterFoldPlanV3) -> tuple[str, ...]:
    return plan.final_pool_receipt.candidate_center_ids


def fit_outer_source_science(
    plan: OuterFoldPlanV3,
    *,
    row_observations: Sequence[RowPosteriorObservation],
    action_observations: Sequence[ActionUtilityObservation],
    opportunity_receipts: Sequence[OpportunityCaseReceipt],
    residual_observations: Sequence[OOFResidualObservation],
    admission_cases: Sequence[AdmissionCase],
    source_surface_lineage_hash: object,
    ranking_policy: BaccRankingPolicy | None = None,
) -> OuterScienceResult:
    """Fit posterior, pairwise ranker, L calibration, then source admission."""

    if not isinstance(plan, OuterFoldPlanV3):
        raise ProtocolError("OE-PPUR v3 outer science requires a typed fold plan.")
    source_hash = str(source_surface_lineage_hash).strip()
    if len(source_hash) != 64:
        raise ProtocolError("OE-PPUR v3 source-science lineage hash drifted.")
    row_rows = tuple(row_observations)
    source_centers = _source_centers(plan)
    if (
        not row_rows
        or tuple(sorted({row.center_id for row in row_rows})) != source_centers
        or any(row.center_id == plan.outer_target_center for row in row_rows)
    ):
        raise ProtocolError("OE-PPUR v3 row-posterior surface is not exact C-minus-H.")
    roles = {
        scope.J: (scope.K, scope.L)
        for scope in plan.scopes
    }
    oof = crossfit_source_row_posterior(
        row_rows,
        outer_target_center=plan.outer_target_center,
        role_centers_by_query=roles,
    )
    expected_keys = {
        (row.center_id, row.case_id, row.row_id) for row in row_rows
    }
    if {
        (row.center_id, row.case_id, row.row_id) for row in oof
    } != expected_keys or {
        row.source_scope_receipt_hash for row in oof
    } != {
        scope.receipt_hash for scope in plan.neutral_case_crossfit_scopes
    }:
        raise ProtocolError("OE-PPUR v3 source row crossfit coverage drifted.")
    final_posterior = fit_final_source_row_posterior(
        row_rows,
        outer_target_center=plan.outer_target_center,
        fixed_capacity_receipt_hash=FEATURE_DEFINITION_RECEIPT_HASH,
    )
    policy = BaccRankingPolicy() if ranking_policy is None else ranking_policy
    if not isinstance(policy, BaccRankingPolicy):
        raise ProtocolError("OE-PPUR v3 ranking policy is untyped.")
    pairwise = fit_pool_indexed_pairwise_ranker(
        action_observations,
        delete_center_scopes=plan.neutral_scopes,
        held_pool_receipts=plan.held_pool_receipts,
        final_pool_receipt=plan.final_pool_receipt,
        compiler=plan.compiler,
        opportunity_receipts=opportunity_receipts,
        ranking_policy=policy,
        source_surface_lineage_hash=source_hash,
    )
    uncertainty = calibrate_clustered_uncertainty(
        residual_observations,
        calibration_scopes=plan.neutral_scopes,
    )
    admission = evaluate_source_ordering_admission(
        admission_cases,
        outer_target_center=plan.outer_target_center,
        source_supervision_contract_hash=plan.source_supervision_contract_hash,
    )
    if not admission.admitted:
        fallback = exact_p_fail_closed_reason(
            outer_target_center=plan.outer_target_center,
            reason_code="source_ordering_admission_failed",
            evidence_hash=admission.receipt_hash,
        )
        return OuterScienceResult(
            outer_target_center=plan.outer_target_center,
            plan_hash=plan.plan_hash,
            source_surface_lineage_hash=source_hash,
            admitted=False,
            row_posterior_model=final_posterior,
            row_oof_predictions=oof,
            pairwise_model=pairwise,
            uncertainty_calibration=uncertainty,
            admission=admission,
            fallback=fallback,
        )
    return OuterScienceResult(
        outer_target_center=plan.outer_target_center,
        plan_hash=plan.plan_hash,
        source_surface_lineage_hash=source_hash,
        admitted=True,
        row_posterior_model=final_posterior,
        row_oof_predictions=oof,
        pairwise_model=pairwise,
        uncertainty_calibration=uncertainty,
        admission=admission,
        fallback=None,
    )


def fit_outer_source_science_fail_closed(
    plan: OuterFoldPlanV3,
    **kwargs: object,
) -> OuterScienceResult:
    """Turn any source lineage/model/calibration failure into exact P for H."""

    try:
        return fit_outer_source_science(plan, **kwargs)  # type: ignore[arg-type]
    except ProtocolError as exc:
        message = str(exc).lower()
        reason = (
            "uncertainty_surface_incomplete"
            if "uncertainty" in message or "residual" in message
            else "source_model_unavailable"
            if "fit" in message or "model" in message or "posterior" in message
            else "protocol_lineage_failure"
        )
        evidence_hash = canonical_sha256(
            {
                "schema": "oe_ppur_v3_fail_closed_exception_evidence_v1",
                "plan_hash": plan.plan_hash,
                "reason_code": reason,
                "exception_type": type(exc).__name__,
                "exception_message_sha256": canonical_sha256(str(exc)),
            }
        )
        fallback = exact_p_fail_closed_reason(
            outer_target_center=plan.outer_target_center,
            reason_code=reason,
            evidence_hash=evidence_hash,
        )
        return OuterScienceResult(
            outer_target_center=plan.outer_target_center,
            plan_hash=plan.plan_hash,
            source_surface_lineage_hash=str(
                kwargs.get("source_surface_lineage_hash", evidence_hash)
            ),
            admitted=False,
            row_posterior_model=None,
            row_oof_predictions=(),
            pairwise_model=None,
            uncertainty_calibration=None,
            admission=None,
            fallback=fallback,
        )


__all__ = (
    "OuterScienceResult",
    "fit_outer_source_science",
    "fit_outer_source_science_fail_closed",
)
