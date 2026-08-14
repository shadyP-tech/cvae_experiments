"""H-minus-c label-scoped correctness observations and class denominators."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, DIRECTION_IDS, candidate_sources
from .correctness_products import (
    DirectionalCorrectnessObservation,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)
from .held_case_features import paired_case_rows
from .probability_surfaces import ExactNineProbabilityRow, ExactNineProbabilitySurface, ProbabilityIndex
from .response_scoring import BinaryLabel
from .split_plans import WholeCaseLooPlan


def route_label_index(
    index: ProbabilityIndex, labels: Sequence[BinaryLabel], plan: WholeCaseLooPlan
) -> dict[tuple[str, str, str], BinaryLabel]:
    result = {row.key: row for row in labels}
    expected = {
        (plan.target_center, case, row.sample_id)
        for case in plan.support_case_ids
        for row in index.rows_for_case_action(plan.target_center, case, B_ACTION_ID)
    }
    if (
        not result
        or len(result) != len(labels)
        or set(result) != expected
        or any(row.case_id == plan.case_id for row in labels)
        or len({row.label_scope for row in labels}) != 1
    ):
        raise ProtocolError("OGDE route labels must be exactly H-minus-c.")
    return result


def support_class_denominators(
    scoped_labels: Sequence[BinaryLabel],
    plan: WholeCaseLooPlan,
    *,
    probability_surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
) -> SupportClassDenominators:
    labels = route_label_index(ProbabilityIndex(probability_surface_or_rows), scoped_labels, plan)
    return SupportClassDenominators(
        plan.target_center,
        plan.case_id,
        sum(row.value == 1 for row in labels.values()),
        sum(row.value == 0 for row in labels.values()),
        plan.support_case_ids,
    )


def score_route_correctness_observations(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    scoped_labels: Sequence[BinaryLabel],
    plan: WholeCaseLooPlan,
    *,
    features: Sequence[LabelFreeDirectionalFeatures],
) -> tuple[DirectionalCorrectnessObservation, ...]:
    index = ProbabilityIndex(surface_or_rows)
    labels = route_label_index(index, scoped_labels, plan)
    feature_index = {row.key: row for row in features}
    route_cases = {*plan.support_case_ids, plan.case_id}
    if any(row.target_center != plan.target_center or row.case_id not in route_cases for row in features):
        raise ProtocolError("OGDE route feature surface escaped the route.")
    output: list[DirectionalCorrectnessObservation] = []
    for support_case in plan.support_case_ids:
        for source in candidate_sources(plan.target_center):
            pairs = paired_case_rows(index, plan.target_center, support_case, source)
            for direction in DIRECTION_IDS:
                try:
                    feature = feature_index[(plan.target_center, support_case, source, direction)]
                except KeyError as exc:
                    raise ProtocolError("OGDE route feature surface is incomplete.") from exc
                successes = 0
                trials = 0
                for baseline, candidate in pairs:
                    is_flip = (
                        direction == "zero_to_one"
                        and baseline.hard_prediction == 0
                        and candidate.hard_prediction == 1
                    ) or (
                        direction == "one_to_zero"
                        and baseline.hard_prediction == 1
                        and candidate.hard_prediction == 0
                    )
                    if not is_flip:
                        continue
                    trials += 1
                    label = labels[(plan.target_center, support_case, baseline.sample_id)].value
                    successes += int(
                        (direction == "zero_to_one" and label == 1)
                        or (direction == "one_to_zero" and label == 0)
                    )
                output.append(
                    DirectionalCorrectnessObservation(
                        plan.target_center,
                        plan.case_id,
                        support_case,
                        source,
                        direction,
                        feature.values,
                        successes,
                        trials,
                    )
                )
    expected = len(plan.support_case_ids) * len(candidate_sources(plan.target_center)) * len(DIRECTION_IDS)
    if len(output) != expected:
        raise ProtocolError("OGDE correctness observation topology drifted.")
    return tuple(output)


score_directional_correctness_observations = score_route_correctness_observations


__all__ = (
    "route_label_index",
    "score_directional_correctness_observations",
    "score_route_correctness_observations",
    "support_class_denominators",
)
