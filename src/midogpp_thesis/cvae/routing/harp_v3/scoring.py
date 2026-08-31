"""Donor-calibrated conservative scoring for label-free target actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...protocol import ProtocolError
from .calibration import ConservativeBounds, conservative_bounds
from .compatibility import GeometryAssessment, assess_geometry
from .contracts import (
    ActionKind,
    CaseTargetAction,
    Comparison,
    EffectVector,
    PolicyConfig,
    SupportSummary,
)
from .fitting import HarpV3Fit


@dataclass(frozen=True)
class ConservativeScore:
    action_id: str
    comparison: Comparison
    delete_donor_ids: tuple[str, ...]
    donor_predictions: tuple[EffectVector, ...]
    geometry: GeometryAssessment
    support: SupportSummary
    source_only_bounds: ConservativeBounds
    geometry_adjusted_bounds: ConservativeBounds
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.action_id
            or not self.delete_donor_ids
            or len(self.delete_donor_ids) != len(self.donor_predictions)
            or any(not isinstance(value, EffectVector) for value in self.donor_predictions)
            or not isinstance(self.geometry, GeometryAssessment)
            or not isinstance(self.support, SupportSummary)
            or not isinstance(self.source_only_bounds, ConservativeBounds)
            or not isinstance(self.geometry_adjusted_bounds, ConservativeBounds)
            or self.eligible != (not self.rejection_reasons)
        ):
            raise ProtocolError("HARP v3 conservative action score is malformed.")


def _endpoint_reasons(
    bounds: ConservativeBounds,
    config: PolicyConfig,
    *,
    prefix: str,
) -> list[str]:
    reasons: list[str] = []
    if (
        bounds.case_equal_bacc_contribution_gain_lower
        <= config.case_equal_bacc_contribution_gain_threshold
    ):
        reasons.append(
            f"{prefix}case_equal_bacc_contribution_gain_lower_bound_not_positive"
        )
    if bounds.brier_upper > config.brier_noninferiority_margin:
        reasons.append(f"{prefix}brier_noninferiority_failed")
    if bounds.log_loss_upper > config.log_loss_noninferiority_margin:
        reasons.append(f"{prefix}log_loss_noninferiority_failed")
    return reasons


def score_comparison(
    fit: HarpV3Fit,
    action: CaseTargetAction,
    comparison: Comparison,
    *,
    config: PolicyConfig = PolicyConfig(),
) -> ConservativeScore:
    if not isinstance(fit, HarpV3Fit) or not isinstance(action, CaseTargetAction):
        raise ProtocolError("HARP v3 scoring requires typed fit and action contracts.")
    if not math.isclose(
        config.max_calibrated_geometry_ratio, 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ProtocolError(
            "HARP v3 matched geometry must use its predeclared source quantile exactly."
        )
    comparison = Comparison(comparison)
    if action.outer_target_id != fit.outer_target_id or action.feature_names != fit.feature_names:
        raise ProtocolError("Target action escaped its outer fit or feature schema.")
    if comparison is Comparison.U_VS_B:
        if action.action_kind is not ActionKind.U:
            raise ProtocolError("U-vs-B scoring requires the U action.")
    elif action.action_kind is not ActionKind.HXE:
        raise ProtocolError("Expert comparisons require one physical Hxe action.")

    predictions: list[EffectVector] = []
    raw_leverages: list[float] = []
    for deleted in fit.delete_donor_fits:
        mean, leverage = deleted.model.predict(
            [action.feature_values],
            [action.target_query_id],
            [action.candidate_source_id],
            [comparison],
        )
        predictions.append(EffectVector(*mean[0]))
        raw_leverages.append(float(leverage[0]))
    geometry = assess_geometry(fit.geometry(comparison), raw_leverages)
    residuals = fit.residuals(comparison)
    source_bounds = conservative_bounds(
        predictions, residuals, compatibility_shrinkage=1.0
    )
    adjusted_bounds = conservative_bounds(
        predictions,
        residuals,
        compatibility_shrinkage=geometry.compatibility_shrinkage,
    )
    support = fit.support(comparison, action.candidate_source_id)
    reasons: list[str] = []
    if support.donor_count < config.min_donor_count:
        reasons.append("insufficient_donor_coverage")
    if support.paired_case_count < config.min_paired_case_count:
        reasons.append("insufficient_paired_cases")
    if support.class_counts[0] <= 0 or support.class_counts[1] <= 0:
        reasons.append("both_source_truth_classes_not_covered")
    # The predeclared geometry quantile defines the acceptance boundary.  A
    # multiplier above one would be a post-hoc relaxation and is prohibited.
    # The add-one-smoothed tail is retained for audit, but is not described as
    # formal conformal evidence.
    if geometry.maximum_ratio > 1.0:
        reasons.append("calibrated_geometry_extrapolation")
    if geometry.compatibility_shrinkage < config.min_compatibility_shrinkage:
        reasons.append("compatibility_abstention")
    # Source-only evidence is checked separately to document the one-way veto:
    # geometry adjustment can weaken a route, never create one.
    reasons.extend(_endpoint_reasons(source_bounds, config, prefix="source_"))
    adjusted_reasons = _endpoint_reasons(adjusted_bounds, config, prefix="")
    reasons.extend(reason for reason in adjusted_reasons if reason not in reasons)
    return ConservativeScore(
        action_id=action.action_id,
        comparison=comparison,
        delete_donor_ids=fit.donor_ids,
        donor_predictions=tuple(predictions),
        geometry=geometry,
        support=support,
        source_only_bounds=source_bounds,
        geometry_adjusted_bounds=adjusted_bounds,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = ("ConservativeScore", "score_comparison")
