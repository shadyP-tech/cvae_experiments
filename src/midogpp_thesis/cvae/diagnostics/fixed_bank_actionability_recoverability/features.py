"""Label-free, case-action features and their matched blocked control."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .actions import actions_for_target
from .constants import (
    B_ACTION_ID,
    GEOMETRY_IDS,
    HARD_THRESHOLD,
    MIDOGPP_CENTERS,
    NEAR_THRESHOLD_HALF_WIDTH,
    PROBABILITY_EPSILON,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from .contracts import (
    AggregatedProbabilityRow,
    CaseActionFeatureRow,
    ExactNineProbabilitySurface,
)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ProtocolError("Case feature aggregation cannot use an empty vector.")
    return math.fsum(values) / len(values)


def _sd(values: Sequence[float]) -> float:
    center = _mean(values)
    return math.sqrt(max(0.0, _mean(tuple((value - center) ** 2 for value in values))))


def _entropy(probability: float) -> float:
    epsilon = PROBABILITY_EPSILON
    value = min(max(probability, epsilon), 1.0 - epsilon)
    return -(value * math.log(value) + (1.0 - value) * math.log(1.0 - value))


def build_label_free_case_action_features(
    probabilities: ExactNineProbabilitySurface | Sequence[AggregatedProbabilityRow],
) -> tuple[CaseActionFeatureRow, ...]:
    """Summarize sealed sample predictions without reading any label surface."""

    rows = probabilities.rows if isinstance(probabilities, ExactNineProbabilitySurface) else tuple(probabilities)
    if not rows or any(not isinstance(row, AggregatedProbabilityRow) for row in rows):
        raise ProtocolError("Case-action features require typed exact-nine probabilities.")
    by_sample: dict[tuple[str, str, str], dict[str, AggregatedProbabilityRow]] = defaultdict(dict)
    for row in rows:
        actions = by_sample[(row.target_center, row.case_id, row.sample_id)]
        if row.action_id in actions:
            raise ProtocolError("Case-action input contains duplicate action/sample rows.")
        actions[row.action_id] = row
    by_case: dict[tuple[str, str], list[dict[str, AggregatedProbabilityRow]]] = defaultdict(list)
    for (target, case_id, _sample_id), actions in sorted(by_sample.items()):
        expected = {action.action_id for action in actions_for_target(target)}
        if set(actions) != expected:
            raise ProtocolError("Every sample needs the complete frozen action library.")
        by_case[(target, case_id)].append(actions)

    output: list[CaseActionFeatureRow] = []
    for (target, case_id), samples in sorted(by_case.items()):
        baseline = tuple(sample[B_ACTION_ID].probability_mean for sample in samples)
        uniform = tuple(sample[U_ACTION_ID].probability_mean for sample in samples)
        uniform_delta = tuple(value - base for value, base in zip(uniform, baseline, strict=True))
        for geometry in GEOMETRY_IDS:
            for source in candidate_sources(target):
                action = geometry_action_id(geometry, source)
                candidate = tuple(sample[action].probability_mean for sample in samples)
                candidate_sd = tuple(sample[action].probability_sd for sample in samples)
                delta = tuple(value - base for value, base in zip(candidate, baseline, strict=True))
                output.append(
                    CaseActionFeatureRow(
                        query_center=target,
                        case_id=case_id,
                        geometry_id=geometry,
                        selected_source=source,
                        values=(
                            1.0,
                            _mean(baseline),
                            _sd(baseline),
                            _mean(uniform_delta),
                            _mean(tuple(abs(value) for value in uniform_delta)),
                            _mean(delta),
                            _mean(tuple(abs(value) for value in delta)),
                            _sd(delta),
                            _mean(
                                tuple(
                                    float((value >= HARD_THRESHOLD) != (base >= HARD_THRESHOLD))
                                    for value, base in zip(candidate, baseline, strict=True)
                                )
                            ),
                            _mean(
                                tuple(
                                    float((value >= HARD_THRESHOLD) != (control >= HARD_THRESHOLD))
                                    for value, control in zip(candidate, uniform, strict=True)
                                )
                            ),
                            _mean(tuple(_entropy(value) for value in candidate)),
                            _mean(
                                tuple(
                                    float(abs(value - HARD_THRESHOLD) <= NEAR_THRESHOLD_HALF_WIDTH)
                                    for value in candidate
                                )
                            ),
                            _mean(candidate_sd),
                        ),
                    )
                )
    if not output:
        raise ProtocolError("No case-action features were produced.")
    return tuple(sorted(output, key=lambda row: row.row_key))


def restrict_feature_context(
    rows: Sequence[CaseActionFeatureRow],
    *,
    excluded_candidate_centers: Sequence[str],
) -> tuple[CaseActionFeatureRow, ...]:
    """Remove forbidden candidate actions and record the exclusion provenance."""

    exclusions = tuple(sorted(set(str(value) for value in excluded_candidate_centers)))
    if any(value not in MIDOGPP_CENTERS for value in exclusions):
        raise ProtocolError("Feature context excludes an unknown MIDOG++ center.")
    output = tuple(
        CaseActionFeatureRow(
            query_center=row.query_center,
            case_id=row.case_id,
            geometry_id=row.geometry_id,
            selected_source=row.selected_source,
            values=row.values,
            feature_origin_source=row.feature_origin_source,
            context_excluded_centers=exclusions,
        )
        for row in rows
        if row.selected_source not in exclusions and row.feature_origin_source not in exclusions
    )
    if not output:
        raise ProtocolError("Feature exclusion removed the complete surface.")
    return tuple(sorted(output, key=lambda row: row.row_key))


def matched_blocked_feature_permutation(
    rows: Sequence[CaseActionFeatureRow],
    *,
    excluded_candidate_centers: Sequence[str] = (),
) -> tuple[CaseActionFeatureRow, ...]:
    """Cyclically derange complete candidate vectors within case/geometry blocks.

    The destination identity and response remain fixed.  The complete feature
    vector comes from the next legal candidate in canonical source order, so P
    is distribution matched while target/query/candidate exclusions are kept.
    """

    restricted = restrict_feature_context(
        rows, excluded_candidate_centers=excluded_candidate_centers
    )
    grouped: dict[tuple[str, str, str], list[CaseActionFeatureRow]] = defaultdict(list)
    for row in restricted:
        grouped[(row.query_center, row.case_id, row.geometry_id)].append(row)
    output: list[CaseActionFeatureRow] = []
    for (query, case_id, geometry), block in sorted(grouped.items()):
        source_order = candidate_sources(query)
        ordered = tuple(sorted(block, key=lambda row: source_order.index(row.selected_source)))
        expected_sources = tuple(
            source for source in source_order if source not in set(excluded_candidate_centers)
        )
        if tuple(row.selected_source for row in ordered) != expected_sources or len(ordered) < 2:
            raise ProtocolError("Blocked permutation requires every legal candidate in each block.")
        donors = ordered[1:] + ordered[:1]
        for destination, donor in zip(ordered, donors, strict=True):
            output.append(
                CaseActionFeatureRow(
                    query_center=query,
                    case_id=case_id,
                    geometry_id=geometry,
                    selected_source=destination.selected_source,
                    values=donor.values,
                    feature_origin_source=donor.selected_source,
                    context_excluded_centers=destination.context_excluded_centers,
                )
            )
    return tuple(sorted(output, key=lambda row: row.row_key))


__all__ = (
    "CaseActionFeatureRow",
    "build_label_free_case_action_features",
    "matched_blocked_feature_permutation",
    "restrict_feature_context",
)
