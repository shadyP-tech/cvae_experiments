"""Label-free held-case directional response features and permutation control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
    CANDIDATE_FEATURE_PERMUTATION_SEED,
    DIRECTION_IDS,
    FEATURE_NAMES,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
    ProbabilityIndex,
    hard_prediction,
)
from .products import LabelFreeDirectionalFeatures

if TYPE_CHECKING:
    from .held_case_plans import HeldCasePlan


_UINT64_MASK = (1 << 64) - 1


def _rows(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
) -> tuple[ExactNineProbabilityRow, ...]:
    return tuple(
        surface_or_rows.rows
        if isinstance(surface_or_rows, ExactNineProbabilitySurface)
        else surface_or_rows
    )


def _paired_case_rows(
    index: ProbabilityIndex,
    target_center: str,
    case_id: str,
    source: str,
) -> tuple[tuple[ExactNineProbabilityRow, ExactNineProbabilityRow], ...]:
    baseline = index.rows_for_case_action(str(target_center), str(case_id), B_ACTION_ID)
    candidate = index.rows_for_case_action(
        str(target_center), str(case_id), a1_action_id(source)
    )
    by_sample = {row.sample_id: row for row in candidate}
    if (
        not baseline
        or len(baseline) != len(candidate)
        or set(by_sample) != {row.sample_id for row in baseline}
    ):
        raise ProtocolError("Abstention-router B/A1 held-case rows are not aligned.")
    return tuple((row, by_sample[row.sample_id]) for row in baseline)


def case_directional_features(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    target_center: object,
    case_id: object,
    source: object,
    direction: object,
) -> LabelFreeDirectionalFeatures:
    """Construct the frozen six-feature vector without accepting labels."""

    return _case_directional_features_from_index(
        ProbabilityIndex(surface_or_rows),
        str(target_center),
        str(case_id),
        str(source),
        str(direction),
    )


def _case_directional_features_from_index(
    index: ProbabilityIndex,
    target: str,
    case: str,
    candidate_source: str,
    direction_id: str,
) -> LabelFreeDirectionalFeatures:

    if direction_id not in DIRECTION_IDS or candidate_source not in candidate_sources(target):
        raise ProtocolError("Abstention-router feature identity drifted.")
    pairs = _paired_case_rows(index, target, case, candidate_source)
    selected: list[tuple[ExactNineProbabilityRow, ExactNineProbabilityRow]] = []
    for baseline, candidate in pairs:
        b_hard = baseline.hard_prediction
        a_hard = candidate.hard_prediction
        if (direction_id == "zero_to_one" and b_hard == 0 and a_hard == 1) or (
            direction_id == "one_to_zero" and b_hard == 1 and a_hard == 0
        ):
            selected.append((baseline, candidate))

    count = len(selected)
    size = len(pairs)
    if count == 0:
        values = (0.0,) * len(FEATURE_NAMES)
    else:
        baseline_margin = []
        candidate_margin = []
        directional_shift = []
        seed_robustness = []
        candidate_disagreement = []
        for baseline, candidate in selected:
            baseline_margin.append(abs(baseline.probability_mean - 0.5))
            candidate_margin.append(abs(candidate.probability_mean - 0.5))
            if direction_id == "zero_to_one":
                directional_shift.append(candidate.probability_mean - baseline.probability_mean)
                seed_flip = tuple(
                    hard_prediction(b) == 0 and hard_prediction(a) == 1
                    for b, a in zip(
                        baseline.seed_probabilities,
                        candidate.seed_probabilities,
                        strict=True,
                    )
                )
            else:
                directional_shift.append(candidate.probability_mean - baseline.probability_mean)
                seed_flip = tuple(
                    hard_prediction(b) == 1 and hard_prediction(a) == 0
                    for b, a in zip(
                        baseline.seed_probabilities,
                        candidate.seed_probabilities,
                        strict=True,
                    )
                )
            seed_robustness.append(float(np.mean(seed_flip, dtype=np.float64)))
            positive_rate = float(
                np.mean(
                    tuple(hard_prediction(value) for value in candidate.seed_probabilities),
                    dtype=np.float64,
                )
            )
            candidate_disagreement.append(2.0 * positive_rate * (1.0 - positive_rate))
        values = (
            float(count / size),
            float(np.mean(baseline_margin, dtype=np.float64)),
            float(np.mean(candidate_margin, dtype=np.float64)),
            float(np.mean(directional_shift, dtype=np.float64)),
            float(np.mean(seed_robustness, dtype=np.float64)),
            float(np.mean(candidate_disagreement, dtype=np.float64)),
        )
    return LabelFreeDirectionalFeatures(
        target,
        case,
        candidate_source,
        direction_id,
        FEATURE_NAMES,
        values,
        count,
        size,
    )


def build_label_free_case_candidate_features(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    rows = _rows(surface_or_rows)
    index = ProbabilityIndex(rows)
    cases = sorted({(row.target_center, row.case_id) for row in rows})
    output = tuple(
        _case_directional_features_from_index(index, target, case, source, direction)
        for target, case in cases
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    )
    if len({row.key for row in output}) != len(output):
        raise ProtocolError("Abstention-router label-free feature surface duplicated.")
    return output


def build_route_label_free_candidate_features(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    plan: "HeldCasePlan",
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    """Build only the support-plus-held label-free surface for one route."""

    index = ProbabilityIndex(_rows(surface_or_rows))
    return _build_route_label_free_candidate_features_from_index(index, plan)


def _build_route_label_free_candidate_features_from_index(
    index: ProbabilityIndex,
    plan: "HeldCasePlan",
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    route_cases = (*plan.support_case_ids, plan.case_id)
    output = tuple(
        _case_directional_features_from_index(
            index, plan.target_center, case, source, direction
        )
        for case in route_cases
        for source in candidate_sources(plan.target_center)
        for direction in DIRECTION_IDS
    )
    if len({row.key for row in output}) != len(output):
        raise ProtocolError("Abstention-router route feature surface duplicated.")
    return tuple(sorted(output, key=lambda row: row.key))


def _splitmix64(value: int) -> tuple[int, int]:
    state = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return state, (mixed ^ (mixed >> 31)) & _UINT64_MASK


def candidate_feature_permutation(
    target_center: object,
    case_id: object,
    direction: object,
    *,
    seed: int = CANDIDATE_FEATURE_PERMUTATION_SEED,
) -> Mapping[str, str]:
    """Return destination->donor mapping for one route-direction block."""

    target = str(target_center)
    case = str(case_id)
    direction_id = str(direction)
    if direction_id not in DIRECTION_IDS or not case:
        raise ProtocolError("Abstention-router feature permutation identity drifted.")
    sources = candidate_sources(target)
    identity = canonical_hash(
        {
            "algorithm": CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
            "seed": int(seed),
            "target_center": target,
            "case_id": case,
            "direction": direction_id,
            "candidate_sources": list(sources),
        }
    )
    state = (int(seed) ^ int(identity[:16], 16)) & _UINT64_MASK
    donors = list(sources)
    for index in range(len(donors) - 1, 0, -1):
        state, draw = _splitmix64(state)
        chosen = int(draw % (index + 1))
        donors[index], donors[chosen] = donors[chosen], donors[index]
    return {destination: donor for destination, donor in zip(sources, donors, strict=True)}


def permute_route_candidate_feature_blocks(
    features: Sequence[LabelFreeDirectionalFeatures],
    plan: "HeldCasePlan",
    *,
    seed: int = CANDIDATE_FEATURE_PERMUTATION_SEED,
) -> tuple[LabelFreeDirectionalFeatures, ...]:
    """Permute whole candidate vectors, never outcomes, identities, G, or labels."""

    route_cases = (*plan.support_case_ids, plan.case_id)
    indexed = {row.key: row for row in features}
    if len(indexed) != len(features):
        raise ProtocolError("Abstention-router feature permutation input duplicated.")
    output: list[LabelFreeDirectionalFeatures] = []
    for direction in DIRECTION_IDS:
        permutation = candidate_feature_permutation(
            plan.target_center, plan.case_id, direction, seed=seed
        )
        for case in route_cases:
            for destination in candidate_sources(plan.target_center):
                donor = permutation[destination]
                key = (plan.target_center, case, donor, direction)
                try:
                    block = indexed[key]
                except KeyError as exc:
                    raise ProtocolError(
                        "Abstention-router feature permutation route surface is incomplete."
                    ) from exc
                output.append(
                    LabelFreeDirectionalFeatures(
                        plan.target_center,
                        case,
                        destination,
                        direction,
                        block.feature_names,
                        block.values,
                        block.directional_flip_count,
                        block.case_size,
                    )
                )
    return tuple(sorted(output, key=lambda row: row.key))


__all__ = (
    "build_label_free_case_candidate_features",
    "build_route_label_free_candidate_features",
    "candidate_feature_permutation",
    "case_directional_features",
    "permute_route_candidate_feature_blocks",
)
