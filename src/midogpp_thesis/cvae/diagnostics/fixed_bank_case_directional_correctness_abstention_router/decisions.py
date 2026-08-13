"""Route-local model fitting, frozen shrinkage, and OFF-first selection."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import (
    DESCRIPTIVE_METHOD_IDS,
    DIRECTION_IDS,
    PRIMARY_METHOD_ID,
    TIE_TOLERANCE,
    candidate_sources,
)
from .held_case_plans import HeldCasePlan
from .model import (
    fit_directional_correctness_model,
    predict_directional_correctness,
    support_denominator_case_proxy,
)
from .products import (
    CandidateDirectionalScore,
    CaseAbstentionDecision,
    DirectionalAbstentionDecision,
    DirectionalCorrectnessModel,
    DirectionalCorrectnessObservation,
    DonorDirectionalPrior,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)


_DECISION_METHODS = {
    PRIMARY_METHOD_ID,
    "G_directional_matched",
    "CDCA_case_proxy_only",
    *DESCRIPTIVE_METHOD_IDS,
}


def fit_route_directional_models(
    observations: Sequence[DirectionalCorrectnessObservation],
    plan: HeldCasePlan,
) -> tuple[DirectionalCorrectnessModel, ...]:
    rows = tuple(observations)
    expected_support = set(plan.support_case_ids)
    output: list[DirectionalCorrectnessModel] = []
    for source in candidate_sources(plan.target_center):
        for direction in DIRECTION_IDS:
            selected = tuple(
                row
                for row in rows
                if row.source == source and row.direction == direction
            )
            if (
                {row.support_case_id for row in selected} != expected_support
                or len(selected) != len(expected_support)
            ):
                raise ProtocolError(
                    "Abstention-router route observations lack exact H-minus-c topology."
                )
            output.append(
                fit_directional_correctness_model(
                    selected,
                    target_center=plan.target_center,
                    case_id=plan.case_id,
                    source=source,
                    direction=direction,
                )
            )
    if len(rows) != len(expected_support) * len(candidate_sources(plan.target_center)) * 2:
        raise ProtocolError("Abstention-router route observation count drifted.")
    return tuple(output)


def _select_direction(
    *,
    method_id: str,
    target_center: str,
    case_id: str,
    direction: str,
    models: dict[tuple[str, str], DirectionalCorrectnessModel],
    features: dict[tuple[str, str], LabelFreeDirectionalFeatures],
    priors: dict[tuple[str, str], DonorDirectionalPrior],
    denominators: SupportClassDenominators,
) -> DirectionalAbstentionDecision:
    scores: list[CandidateDirectionalScore] = [
        CandidateDirectionalScore(
            target_center,
            case_id,
            direction,
            None,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            None,
        )
    ]
    for source in candidate_sources(target_center):
        model = models[(source, direction)]
        feature = features[(source, direction)]
        prior = priors[(source, direction)]
        if method_id == "G_directional_matched":
            predicted = 0.0
            case_proxy = 0.0
            donor_value = prior.value
            final = donor_value
            model_hash = None
        else:
            predicted = predict_directional_correctness(model, feature)
            case_proxy = support_denominator_case_proxy(
                predicted,
                feature.directional_flip_count,
                direction,
                denominators.n_positive,
                denominators.n_negative,
                valid_model=model.converged and model.training_trial_count > 0,
            )
            donor_value = 0.0 if method_id == "CDCA_case_proxy_only" else prior.value
            if method_id == "CDCA_case_proxy_only":
                final = case_proxy
            else:
                final = 0.5 * case_proxy + 0.5 * donor_value
            model_hash = model.model_hash
        scores.append(
            CandidateDirectionalScore(
                target_center,
                case_id,
                direction,
                source,
                predicted,
                feature.directional_flip_count,
                case_proxy,
                donor_value,
                final,
                model_hash,
            )
        )
    maximum = max(score.final_score for score in scores)
    selected = next(
        score.source
        for score in scores
        if maximum - score.final_score <= TIE_TOLERANCE
    )
    return DirectionalAbstentionDecision(
        method_id,
        target_center,
        case_id,
        direction,
        tuple(scores),
        selected,
    )


def select_case_directional_abstention_decision(
    *,
    method_id: str,
    target_center: object,
    case_id: object,
    models: Sequence[DirectionalCorrectnessModel],
    held_features: Sequence[LabelFreeDirectionalFeatures],
    donor_priors: Sequence[DonorDirectionalPrior],
    denominators: SupportClassDenominators,
) -> CaseAbstentionDecision:
    target = str(target_center)
    held_case = str(case_id)
    if method_id not in _DECISION_METHODS:
        raise ProtocolError("Abstention-router method is not a directional selector.")
    model_index = {(row.source, row.direction): row for row in models}
    feature_index = {(row.source, row.direction): row for row in held_features}
    prior_index = {(row.source, row.direction): row for row in donor_priors}
    expected = {
        (source, direction)
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    }
    if (
        set(model_index) != expected
        or set(feature_index) != expected
        or set(prior_index) != expected
        or len(model_index) != len(models)
        or len(feature_index) != len(held_features)
        or len(prior_index) != len(donor_priors)
        or any(row.key[:2] != (target, held_case) for row in models)
        or any((row.target_center, row.case_id) != (target, held_case) for row in held_features)
        or any(row.heldout_center != target for row in donor_priors)
        or (denominators.target_center, denominators.route_case_id) != (target, held_case)
    ):
        raise ProtocolError("Abstention-router decision inputs drifted.")
    decisions = tuple(
        _select_direction(
            method_id=method_id,
            target_center=target,
            case_id=held_case,
            direction=direction,
            models=model_index,
            features=feature_index,
            priors=prior_index,
            denominators=denominators,
        )
        for direction in DIRECTION_IDS
    )
    return CaseAbstentionDecision(
        method_id,
        target,
        held_case,
        decisions[0],
        decisions[1],
    )


select_case_decision = select_case_directional_abstention_decision


__all__ = (
    "fit_route_directional_models",
    "select_case_decision",
    "select_case_directional_abstention_decision",
)
