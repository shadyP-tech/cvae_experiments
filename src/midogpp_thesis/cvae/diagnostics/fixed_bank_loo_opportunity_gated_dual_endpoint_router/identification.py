"""Scale-normalized 4/5 case + 1/5 donor identification with a hard OFF gate."""

from __future__ import annotations

from collections.abc import Sequence
import math

from ...protocol import ProtocolError
from .constants import (
    DIRECTION_IDS,
    IDENTIFICATION_CASE_WEIGHT,
    IDENTIFICATION_DONOR_WEIGHT,
    candidate_sources,
)
from .correctness_model import predict_correctness, support_calibrated_case_proxy
from .correctness_products import (
    DirectionalCorrectnessModel,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)
from .donor_prior import DonorPrior
from .identification_products import (
    CaseIdentificationDecision,
    DirectionIdentificationDecision,
    IDENTIFICATION_METHOD_IDS,
    IdentificationCandidateScore,
)
from .split_plans import WholeCaseLooPlan


def _select_direction(
    *,
    method_id: str,
    plan: WholeCaseLooPlan,
    direction: str,
    features: dict[tuple[str, str], LabelFreeDirectionalFeatures],
    models: dict[tuple[str, str], DirectionalCorrectnessModel],
    priors: dict[tuple[str, str], DonorPrior],
    denominators: SupportClassDenominators,
) -> DirectionIdentificationDecision:
    sources = candidate_sources(plan.target_center)
    intermediate: list[tuple[str, DirectionalCorrectnessModel, LabelFreeDirectionalFeatures, float | None, float, float]] = []
    nonfinite = False
    for source in sources:
        model = models[(source, direction)]
        feature = features[(source, direction)]
        prior = priors[(source, direction)]
        predicted = predict_correctness(model, feature)
        proxy = support_calibrated_case_proxy(
            predicted, feature.directional_flip_count, direction, denominators
        )
        case_value = 0.0 if proxy is None else float(proxy)
        donor_value = float(prior.value)
        if not math.isfinite(case_value) or not math.isfinite(donor_value):
            nonfinite = True
        intermediate.append((source, model, feature, predicted, case_value, donor_value))
    case_scale = sum(abs(row[4]) for row in intermediate) / len(sources)
    donor_scale = sum(abs(row[5]) for row in intermediate) / len(sources)
    if not math.isfinite(case_scale) or not math.isfinite(donor_scale):
        nonfinite = True
        case_scale = 0.0
        donor_scale = 0.0
    rows: list[IdentificationCandidateScore] = []
    for source, model, feature, predicted, case_value, donor_value in intermediate:
        normalized_case = 0.0 if case_scale == 0.0 else case_value / case_scale
        normalized_donor = 0.0 if donor_scale == 0.0 else donor_value / donor_scale
        final_score = (
            float(IDENTIFICATION_CASE_WEIGHT) * normalized_case
            + float(IDENTIFICATION_DONOR_WEIGHT) * normalized_donor
        )
        opportunity = bool(
            feature.directional_flip_count > 0
            and model.is_valid
            and predicted is not None
        )
        eligible = bool(opportunity and case_value > 0.0)
        if feature.directional_flip_count <= 0:
            reason = "zero_directional_opportunity"
        elif not model.is_valid or predicted is None:
            reason = "invalid_route_local_model"
        elif case_value <= 0.0:
            reason = "nonpositive_support_calibrated_proxy"
        else:
            reason = "eligible_positive_opportunity"
        rows.append(
            IdentificationCandidateScore(
                plan.target_center,
                plan.case_id,
                direction,
                source,
                predicted,
                feature.directional_flip_count,
                model.is_valid,
                case_value,
                donor_value,
                case_scale,
                donor_scale,
                normalized_case,
                normalized_donor,
                final_score,
                opportunity,
                eligible,
                reason,
                model.model_hash,
            )
        )
    fail_closed = bool(nonfinite or case_scale == 0.0)
    eligible_sources = tuple(row.source for row in rows if row.eligible)
    if nonfinite:
        reason = "OFF_nonfinite_candidate_surface"
    elif case_scale == 0.0:
        reason = "OFF_zero_case_proxy_scale"
    elif not eligible_sources:
        reason = "OFF_no_positive_eligible_opportunity"
    else:
        reason = "OFF_or_active_by_strict_positive_score"
    eligible = tuple(row for row in rows if row.eligible)
    if fail_closed or not eligible or max(row.final_score for row in eligible) <= 1.0e-12:
        selected = None
    else:
        maximum = max(row.final_score for row in eligible)
        selected = min((row.source for row in eligible if maximum - row.final_score <= 1.0e-12), key=int)
    opportunity = tuple(row for row in rows if row.opportunity_eligible)
    if not opportunity:
        source_only = None
    else:
        maximum = max(row.final_score for row in opportunity)
        source_only = min((row.source for row in opportunity if maximum - row.final_score <= 1.0e-12), key=int)
    return DirectionIdentificationDecision(
        method_id,
        plan.target_center,
        plan.case_id,
        direction,
        tuple(rows),
        selected,
        source_only,
        eligible_sources,
        case_scale,
        donor_scale,
        fail_closed,
        reason,
    )


def select_case_identification_decision(
    plan: WholeCaseLooPlan,
    held_features: Sequence[LabelFreeDirectionalFeatures],
    models: Sequence[DirectionalCorrectnessModel],
    denominators: SupportClassDenominators,
    donor_priors: Sequence[DonorPrior],
    *,
    method_id: str = "I_OPPORTUNITY_GATED",
) -> CaseIdentificationDecision:
    if method_id not in IDENTIFICATION_METHOD_IDS:
        raise ProtocolError("OGDE identification method identity drifted.")
    feature_rows = tuple(
        row
        for row in held_features
        if row.target_center == plan.target_center and row.case_id == plan.case_id
    )
    model_rows = tuple(models)
    prior_rows = tuple(donor_priors)
    features = {(row.source, row.direction): row for row in feature_rows}
    model_index = {(row.source, row.direction): row for row in model_rows}
    prior_index = {(row.source, row.direction): row for row in prior_rows}
    expected = tuple((source, direction) for source in candidate_sources(plan.target_center) for direction in DIRECTION_IDS)
    if (
        tuple(features) != expected
        or tuple(model_index) != expected
        or tuple(prior_index) != expected
        or denominators.target_center != plan.target_center
        or denominators.route_case_id != plan.case_id
        or denominators.support_case_ids != plan.support_case_ids
    ):
        raise ProtocolError("OGDE identification route inputs lack exact canonical topology.")
    directional = tuple(
        _select_direction(
            method_id=method_id,
            plan=plan,
            direction=direction,
            features=features,
            models=model_index,
            priors=prior_index,
            denominators=denominators,
        )
        for direction in DIRECTION_IDS
    )
    return CaseIdentificationDecision(
        method_id,
        plan.target_center,
        plan.case_id,
        directional[0],
        directional[1],
    )


__all__ = ("select_case_identification_decision",)
