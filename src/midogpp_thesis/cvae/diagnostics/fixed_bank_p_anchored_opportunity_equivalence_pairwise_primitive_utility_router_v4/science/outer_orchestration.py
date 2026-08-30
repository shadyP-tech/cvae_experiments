"""Source-surface-only outer science with exact-P failure semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    BaccRankingPolicy,
    PairwiseRankerModel,
    RowPosteriorModel,
    RowPosteriorOOFPrediction,
    UncertaintyCalibration,
    canonical_sha256,
    crossfit_source_row_posterior,
    fit_final_source_row_posterior,
)
from ..feature_engineering import (
    FEATURE_DEFINITION_RECEIPT_HASH,
    build_row_posterior_observations,
)
from ..folds import OuterFoldPlanV4
from ..identity import CENTERS
from ..source_supervision import SourceTrainingSurface
from .admission import (
    ExactPFallbackReceipt,
    SourceOrderingAdmissionReceipt,
    evaluate_genuine_held_l_source_ordering,
    exact_p_fail_closed_reason,
)
from .held_l_calibration import build_genuine_held_l_calibration
from .pool_indexed_pairwise_fit import (
    fit_genuine_held_l_action_predictions,
    fit_pool_indexed_pairwise_ranker,
)
from .source_products import derive_source_science_products


def _sha256(value: object, *, role: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ProtocolError(f"OE-PPUR v4 {role} is not a SHA-256 digest.")
    return result


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
        h = str(self.outer_target_center)
        plan_hash = _sha256(self.plan_hash, role="outer plan hash")
        source_hash = _sha256(self.source_surface_lineage_hash, role="source surface lineage hash")
        success = (
            isinstance(self.row_posterior_model, RowPosteriorModel)
            and bool(self.row_oof_predictions)
            and isinstance(self.pairwise_model, PairwiseRankerModel)
            and isinstance(self.uncertainty_calibration, UncertaintyCalibration)
            and isinstance(self.admission, SourceOrderingAdmissionReceipt)
            and self.admission.admitted
            and self.fallback is None
        )
        failure = isinstance(self.fallback, ExactPFallbackReceipt) and not self.admitted
        if h not in CENTERS or bool(self.admitted) != success or (not self.admitted and not failure):
            raise ProtocolError("OE-PPUR v4 outer-science result is not fail-closed.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "plan_hash", plan_hash)
        object.__setattr__(self, "source_surface_lineage_hash", source_hash)
        object.__setattr__(self, "row_oof_predictions", tuple(self.row_oof_predictions))
        object.__setattr__(self, "result_hash", canonical_sha256({
            "schema": "oe_ppur_v4_outer_science_result_v2",
            "H": h,
            "plan_hash": plan_hash,
            "source_surface_lineage_hash": source_hash,
            "admitted": self.admitted,
            "row_posterior_model_hash": None if self.row_posterior_model is None else self.row_posterior_model.model_hash,
            "row_oof_prediction_lineage": tuple((row.center_id, row.case_id, row.row_id, row.model_hash, row.source_scope_receipt_hash) for row in self.row_oof_predictions),
            "pairwise_model_hash": None if self.pairwise_model is None else self.pairwise_model.model_hash,
            "uncertainty_calibration_hash": None if self.uncertainty_calibration is None else self.uncertainty_calibration.calibration_hash,
            "admission_receipt_hash": None if self.admission is None else self.admission.receipt_hash,
            "fallback_receipt_hash": None if self.fallback is None else self.fallback.receipt_hash,
            "caller_injected_scientific_products": False,
            "target_labels_used": False,
        }))


def fit_outer_source_science(
    source_surface: SourceTrainingSurface,
    plan: OuterFoldPlanV4,
    *,
    ranking_policy: BaccRankingPolicy | None = None,
) -> OuterScienceResult:
    """Canonical nominal fit deriving all observations from direct input #3."""

    if not isinstance(source_surface, SourceTrainingSurface) or not isinstance(plan, OuterFoldPlanV4):
        raise ProtocolError("OE-PPUR v4 nominal science requires typed surface and plan.")
    if (
        source_surface.receipt.contract.contract_hash != plan.source_supervision_contract_hash
        or source_surface.compiler.receipt_hash != plan.compiler.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 nominal surface/plan lineage drifted.")
    row_rows = build_row_posterior_observations(source_surface.rows_for_outer(plan.outer_target_center))
    roles = {scope.J: (scope.K, scope.L) for scope in plan.scopes}
    oof = crossfit_source_row_posterior(
        row_rows,
        outer_target_center=plan.outer_target_center,
        role_centers_by_query=roles,
    )
    if (
        {(row.center_id, row.case_id, row.row_id) for row in oof}
        != {(row.center_id, row.case_id, row.row_id) for row in row_rows}
        or {row.source_scope_receipt_hash for row in oof}
        != {scope.receipt_hash for scope in plan.neutral_case_crossfit_scopes}
    ):
        raise ProtocolError("OE-PPUR v4 nominal row OOF coverage drifted.")
    products = derive_source_science_products(source_surface, plan, row_oof_predictions=oof)
    final_posterior = fit_final_source_row_posterior(
        row_rows,
        outer_target_center=plan.outer_target_center,
        fixed_capacity_receipt_hash=FEATURE_DEFINITION_RECEIPT_HASH,
    )
    policy = BaccRankingPolicy() if ranking_policy is None else ranking_policy
    if not isinstance(policy, BaccRankingPolicy):
        raise ProtocolError("OE-PPUR v4 nominal ranking policy is untyped.")
    pairwise = fit_pool_indexed_pairwise_ranker(
        products.action_observations,
        delete_center_scopes=plan.neutral_scopes,
        held_pool_receipts=plan.held_pool_receipts,
        final_pool_receipt=plan.final_pool_receipt,
        compiler=plan.compiler,
        opportunity_receipts=products.opportunity_receipts,
        ranking_policy=policy,
        source_surface_lineage_hash=source_surface.surface_hash,
    )
    held_l_predictions = fit_genuine_held_l_action_predictions(
        products.action_observations,
        products.held_l_queries,
        calibration_scopes=plan.neutral_scopes,
        selected_alpha=pairwise.selected_alpha,
    )
    calibration_products = build_genuine_held_l_calibration(products, held_l_predictions, plan=plan)
    admission = evaluate_genuine_held_l_source_ordering(
        calibration_products.ordering_cases,
        outer_target_center=plan.outer_target_center,
        source_supervision_contract_hash=plan.source_supervision_contract_hash,
        uncertainty_calibration_hash=calibration_products.uncertainty_calibration.calibration_hash,
        source_case_inventory_hash=plan.source_case_inventory_hash,
    )
    fallback = None if admission.admitted else exact_p_fail_closed_reason(
        outer_target_center=plan.outer_target_center,
        reason_code="source_ordering_admission_failed",
        evidence_hash=admission.receipt_hash,
    )
    return OuterScienceResult(
        outer_target_center=plan.outer_target_center,
        plan_hash=plan.plan_hash,
        source_surface_lineage_hash=source_surface.surface_hash,
        admitted=admission.admitted,
        row_posterior_model=final_posterior,
        row_oof_predictions=oof,
        pairwise_model=pairwise,
        uncertainty_calibration=calibration_products.uncertainty_calibration,
        admission=admission,
        fallback=fallback,
    )


def fit_outer_source_science_fail_closed(
    source_surface: SourceTrainingSurface,
    plan: OuterFoldPlanV4,
    *,
    ranking_policy: BaccRankingPolicy | None = None,
) -> OuterScienceResult:
    """Canonical service entry; any science failure seals outer H to exact P."""

    try:
        return fit_outer_source_science(source_surface, plan, ranking_policy=ranking_policy)
    except ProtocolError as exc:
        message = str(exc).lower()
        reason = (
            "uncertainty_surface_incomplete"
            if "uncertainty" in message or "residual" in message or "held-l" in message
            else "source_model_unavailable"
            if "fit" in message or "model" in message or "posterior" in message
            else "protocol_lineage_failure"
        )
        evidence_hash = canonical_sha256({
            "schema": "oe_ppur_v4_nominal_fail_closed_exception_v1",
            "plan_hash": plan.plan_hash,
            "source_surface_hash": source_surface.surface_hash,
            "reason_code": reason,
            "exception_type": type(exc).__name__,
            "exception_message_sha256": canonical_sha256(str(exc)),
        })
        return OuterScienceResult(
            outer_target_center=plan.outer_target_center,
            plan_hash=plan.plan_hash,
            source_surface_lineage_hash=source_surface.surface_hash,
            admitted=False,
            row_posterior_model=None,
            row_oof_predictions=(),
            pairwise_model=None,
            uncertainty_calibration=None,
            admission=None,
            fallback=exact_p_fail_closed_reason(
                outer_target_center=plan.outer_target_center,
                reason_code=reason,
                evidence_hash=evidence_hash,
            ),
        )


# Explicit descriptive aliases for service code; neither accepts injected products.
fit_outer_source_science_from_surface = fit_outer_source_science
fit_outer_source_science_from_surface_fail_closed = fit_outer_source_science_fail_closed


__all__ = (
    "OuterScienceResult",
    "fit_outer_source_science",
    "fit_outer_source_science_fail_closed",
    "fit_outer_source_science_from_surface",
    "fit_outer_source_science_from_surface_fail_closed",
)
