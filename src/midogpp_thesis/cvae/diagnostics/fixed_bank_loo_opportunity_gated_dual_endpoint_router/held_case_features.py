"""Six frozen label-free summaries for every case/source/direction cell."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, DIRECTION_IDS, FEATURE_NAMES, a1_action_id, candidate_sources
from .correctness_products import LabelFreeDirectionalFeatures
from .probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    ProbabilityIndex,
    hard_prediction,
)


def paired_case_rows(
    index: ProbabilityIndex, target: str, case: str, source: str
) -> tuple[tuple[ExactNineProbabilityRow, ExactNineProbabilityRow], ...]:
    baseline = index.rows_for_case_action(target, case, B_ACTION_ID)
    candidate = index.rows_for_case_action(target, case, a1_action_id(source))
    by_sample = {row.sample_id: row for row in candidate}
    if not baseline or len(baseline) != len(candidate) or set(by_sample) != {row.sample_id for row in baseline}:
        raise ProtocolError("OGDE B/A1 case rows are not sample aligned.")
    return tuple((row, by_sample[row.sample_id]) for row in baseline)


def _case_directional_features(
    index: ProbabilityIndex, target: str, case: str, source: str, direction: str
) -> LabelFreeDirectionalFeatures:
    if source not in candidate_sources(target) or direction not in DIRECTION_IDS:
        raise ProtocolError("OGDE feature identity drifted.")
    pairs = paired_case_rows(index, target, case, source)
    selected = tuple(
        (baseline, candidate)
        for baseline, candidate in pairs
        if (
            direction == "zero_to_one"
            and baseline.hard_prediction == 0
            and candidate.hard_prediction == 1
        )
        or (
            direction == "one_to_zero"
            and baseline.hard_prediction == 1
            and candidate.hard_prediction == 0
        )
    )
    if not selected:
        values = (0.0,) * len(FEATURE_NAMES)
    else:
        b_margin: list[float] = []
        a_margin: list[float] = []
        shift: list[float] = []
        robustness: list[float] = []
        disagreement: list[float] = []
        for baseline, candidate in selected:
            b_margin.append(abs(baseline.probability_mean - 0.5))
            a_margin.append(abs(candidate.probability_mean - 0.5))
            shift.append(candidate.probability_mean - baseline.probability_mean)
            if direction == "zero_to_one":
                seed_flips = tuple(
                    hard_prediction(b) == 0 and hard_prediction(a) == 1
                    for b, a in zip(baseline.seed_probabilities, candidate.seed_probabilities, strict=True)
                )
            else:
                seed_flips = tuple(
                    hard_prediction(b) == 1 and hard_prediction(a) == 0
                    for b, a in zip(baseline.seed_probabilities, candidate.seed_probabilities, strict=True)
                )
            robustness.append(float(np.mean(seed_flips, dtype=np.float64)))
            positive_rate = float(
                np.mean(tuple(hard_prediction(value) for value in candidate.seed_probabilities), dtype=np.float64)
            )
            disagreement.append(2.0 * positive_rate * (1.0 - positive_rate))
        values = (
            len(selected) / len(pairs),
            float(np.mean(b_margin, dtype=np.float64)),
            float(np.mean(a_margin, dtype=np.float64)),
            float(np.mean(shift, dtype=np.float64)),
            float(np.mean(robustness, dtype=np.float64)),
            float(np.mean(disagreement, dtype=np.float64)),
        )
    return LabelFreeDirectionalFeatures(
        target,
        case,
        source,
        direction,
        FEATURE_NAMES,
        values,
        len(selected),
        len(pairs),
    )


def case_directional_features(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    target_center: object,
    case_id: object,
    source: object,
    direction: object,
) -> LabelFreeDirectionalFeatures:
    return _case_directional_features(
        ProbabilityIndex(surface_or_rows),
        str(target_center),
        str(case_id),
        str(source),
        str(direction),
    )


def build_label_free_features(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    index = ProbabilityIndex(surface_or_rows)
    cases = tuple(sorted({(key[0], key[1]) for key in index}))
    output = tuple(
        _case_directional_features(index, target, case, source, direction)
        for target, case in cases
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    )
    if len({row.key for row in output}) != len(output):
        raise ProtocolError("OGDE label-free feature surface duplicated.")
    return output


build_label_free_case_candidate_features = build_label_free_features


__all__ = (
    "build_label_free_case_candidate_features",
    "build_label_free_features",
    "case_directional_features",
    "paired_case_rows",
)
