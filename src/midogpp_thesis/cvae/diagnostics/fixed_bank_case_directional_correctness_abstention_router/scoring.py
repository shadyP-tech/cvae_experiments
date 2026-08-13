"""Scoped support responses and exact donor directional gains."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, DIRECTION_IDS, a1_action_id, candidate_sources
from .features import (
    _build_route_label_free_candidate_features_from_index,
    build_label_free_case_candidate_features,
    permute_route_candidate_feature_blocks,
)
from .held_case_plans import HeldCasePlan
from .probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    ProbabilityIndex,
)
from .products import (
    BinaryLabel,
    DirectionalCorrectnessObservation,
    DirectionalGain,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)


def _label_index(labels: Sequence[BinaryLabel]) -> dict[tuple[str, str, str], BinaryLabel]:
    indexed = {row.key: row for row in labels}
    if not labels or len(indexed) != len(labels):
        raise ProtocolError("Abstention-router scoped labels are empty or duplicated.")
    if len({row.label_scope for row in labels}) != 1:
        raise ProtocolError("Abstention-router scoped labels mix capabilities.")
    return indexed


def _case_pairs(
    index: ProbabilityIndex,
    target: str,
    case: str,
    source: str,
) -> tuple[tuple[ExactNineProbabilityRow, ExactNineProbabilityRow], ...]:
    baseline = index.rows_for_case_action(target, case, B_ACTION_ID)
    candidate = index.rows_for_case_action(target, case, a1_action_id(source))
    candidate_by_sample = {row.sample_id: row for row in candidate}
    if (
        not baseline
        or len(baseline) != len(candidate)
        or set(candidate_by_sample) != {row.sample_id for row in baseline}
    ):
        raise ProtocolError("Abstention-router scoring B/A1 rows are not aligned.")
    return tuple((row, candidate_by_sample[row.sample_id]) for row in baseline)


def _validate_route_label_scope(
    index: ProbabilityIndex,
    labels: Sequence[BinaryLabel],
    plan: HeldCasePlan,
) -> dict[tuple[str, str, str], BinaryLabel]:
    indexed = _label_index(labels)
    expected = {
        (plan.target_center, case, row.sample_id)
        for case in plan.support_case_ids
        for row in index.rows_for_case_action(plan.target_center, case, B_ACTION_ID)
    }
    if (
        not expected
        or set(indexed) != expected
        or any(row.case_id == plan.case_id for row in labels)
    ):
        raise ProtocolError(
            "Abstention-router route labels must be exactly H-minus-held-case c."
        )
    return indexed


def support_class_denominators(
    scoped_labels: Sequence[BinaryLabel],
    plan: HeldCasePlan,
    *,
    probability_surface_or_rows: ExactNineProbabilitySurface
    | Sequence[ExactNineProbabilityRow]
    | None = None,
) -> SupportClassDenominators:
    if probability_surface_or_rows is None:
        indexed = _label_index(scoped_labels)
        if (
            any(key[0] != plan.target_center or key[1] not in plan.support_case_ids for key in indexed)
            or any(key[1] == plan.case_id for key in indexed)
            or {key[1] for key in indexed} != set(plan.support_case_ids)
        ):
            raise ProtocolError("Abstention-router support denominator scope drifted.")
    else:
        indexed = _validate_route_label_scope(
            ProbabilityIndex(probability_surface_or_rows), scoped_labels, plan
        )
    positive = sum(row.value == 1 for row in indexed.values())
    negative = sum(row.value == 0 for row in indexed.values())
    return SupportClassDenominators(
        plan.target_center,
        plan.case_id,
        positive,
        negative,
        plan.support_case_ids,
    )


def score_directional_correctness_observations(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    scoped_labels: Sequence[BinaryLabel],
    plan: HeldCasePlan,
    *,
    features: Sequence[LabelFreeDirectionalFeatures] | None = None,
    candidate_feature_blocks_permuted: bool = False,
) -> tuple[DirectionalCorrectnessObservation, ...]:
    """Build per-support-case binomial responses for one `(H,c)` route."""

    index = ProbabilityIndex(surface_or_rows)
    labels = _validate_route_label_scope(index, scoped_labels, plan)
    feature_rows = tuple(
        build_label_free_case_candidate_features(surface_or_rows)
        if features is None
        else features
    )
    feature_index = {row.key: row for row in feature_rows}
    if len(feature_index) != len(feature_rows):
        raise ProtocolError("Abstention-router feature surface duplicated before scoring.")
    if candidate_feature_blocks_permuted:
        expected = permute_route_candidate_feature_blocks(
            _build_route_label_free_candidate_features_from_index(index, plan), plan
        )
        route_case_ids = {*plan.support_case_ids, plan.case_id}
        supplied = tuple(
            sorted(
                (
                    row
                    for row in feature_rows
                    if row.target_center == plan.target_center
                    and row.case_id in route_case_ids
                ),
                key=lambda row: row.key,
            )
        )
        if supplied != expected:
            raise ProtocolError(
                "Abstention-router descriptive features are not the frozen permutation."
            )
    output: list[DirectionalCorrectnessObservation] = []
    for support_case in plan.support_case_ids:
        for source in candidate_sources(plan.target_center):
            pairs = _case_pairs(index, plan.target_center, support_case, source)
            for direction in DIRECTION_IDS:
                successes = 0
                trials = 0
                for baseline, candidate in pairs:
                    b_hard = baseline.hard_prediction
                    a_hard = candidate.hard_prediction
                    is_flip = (
                        direction == "zero_to_one" and b_hard == 0 and a_hard == 1
                    ) or (
                        direction == "one_to_zero" and b_hard == 1 and a_hard == 0
                    )
                    if not is_flip:
                        continue
                    trials += 1
                    label = labels[(plan.target_center, support_case, baseline.sample_id)].value
                    successes += int(
                        (direction == "zero_to_one" and label == 1)
                        or (direction == "one_to_zero" and label == 0)
                    )
                try:
                    feature = feature_index[
                        (plan.target_center, support_case, source, direction)
                    ]
                except KeyError as exc:
                    raise ProtocolError(
                        "Abstention-router support feature surface is incomplete."
                    ) from exc
                if (
                    not candidate_feature_blocks_permuted
                    and trials != feature.directional_flip_count
                ):
                    raise ProtocolError(
                        "Abstention-router label-free flip count changed during scoring."
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
    return tuple(output)


def score_permuted_directional_correctness_observations(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    scoped_labels: Sequence[BinaryLabel],
    plan: HeldCasePlan,
    *,
    permuted_features: Sequence[LabelFreeDirectionalFeatures],
) -> tuple[DirectionalCorrectnessObservation, ...]:
    """Descriptive seam: outcomes stay fixed while whole feature blocks move."""

    return score_directional_correctness_observations(
        surface_or_rows,
        scoped_labels,
        plan,
        features=permuted_features,
        candidate_feature_blocks_permuted=True,
    )


def score_directional_gains(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    scoped_labels: Sequence[BinaryLabel],
) -> tuple[DirectionalGain, ...]:
    """Score exact center-level BACC gains for donor-query centers."""

    index = ProbabilityIndex(surface_or_rows)
    labels = _label_index(scoped_labels)
    query_centers = tuple(sorted({row.target_center for row in scoped_labels}))
    output: list[DirectionalGain] = []
    for query in query_centers:
        query_labels = {key: row for key, row in labels.items() if key[0] == query}
        cases = sorted({key[1] for key in query_labels})
        expected = {
            (query, case, row.sample_id)
            for case in cases
            for row in index.rows_for_case_action(query, case, B_ACTION_ID)
        }
        if not expected or set(query_labels) != expected:
            raise ProtocolError("Abstention-router donor label scope is incomplete.")
        n_positive = sum(row.value == 1 for row in query_labels.values())
        n_negative = sum(row.value == 0 for row in query_labels.values())
        if n_positive <= 0 or n_negative <= 0:
            raise ProtocolError("Abstention-router donor query center is single class.")
        for source in candidate_sources(query):
            counts = {
                "zero_to_one": [0, 0],
                "one_to_zero": [0, 0],
            }
            for case in cases:
                for baseline, candidate in _case_pairs(index, query, case, source):
                    label = query_labels[(query, case, baseline.sample_id)].value
                    if baseline.hard_prediction == 0 and candidate.hard_prediction == 1:
                        counts["zero_to_one"][0 if label == 1 else 1] += 1
                    elif baseline.hard_prediction == 1 and candidate.hard_prediction == 0:
                        counts["one_to_zero"][0 if label == 0 else 1] += 1
            for direction in DIRECTION_IDS:
                favorable, adverse = counts[direction]
                if direction == "zero_to_one":
                    exact = Fraction(favorable, 2 * n_positive) - Fraction(
                        adverse, 2 * n_negative
                    )
                else:
                    exact = Fraction(favorable, 2 * n_negative) - Fraction(
                        adverse, 2 * n_positive
                    )
                output.append(
                    DirectionalGain(
                        query,
                        source,
                        direction,
                        favorable,
                        adverse,
                        n_positive,
                        n_negative,
                        exact.numerator,
                        exact.denominator,
                        float(exact),
                    )
                )
    return tuple(output)


__all__ = (
    "score_directional_correctness_observations",
    "score_directional_gains",
    "score_permuted_directional_correctness_observations",
    "support_class_denominators",
)
