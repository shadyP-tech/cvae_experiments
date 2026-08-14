"""Capability-scoped hard-flip scoring over immutable response products."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import B_ACTION_ID, DIRECTION_IDS, HARD_THRESHOLD, a1_action_id, candidate_sources, physical_action_ids
from .probability_surfaces import ExactNineProbabilityRow, ExactNineProbabilitySurface, ProbabilityIndex
from .response_products import BinaryLabel, CaseActionConfusion, DirectionalGain
from .split_plans import WholeCaseLooPlan


def _label_index(labels: Sequence[BinaryLabel]) -> dict[tuple[str, str, str], BinaryLabel]:
    rows = tuple(labels)
    result = {row.key: row for row in rows}
    if not rows or len(result) != len(rows) or len({row.label_scope for row in rows}) != 1:
        raise ProtocolError("OGDE labels are empty, duplicated, or mix capabilities.")
    return result


def score_case_action_confusions(
    surface_or_rows: ExactNineProbabilitySurface | Sequence[ExactNineProbabilityRow],
    scoped_labels: Sequence[BinaryLabel],
) -> tuple[CaseActionConfusion, ...]:
    index = ProbabilityIndex(surface_or_rows)
    labels = _label_index(scoped_labels)
    scope = next(iter({row.label_scope for row in labels.values()}))
    sample_actions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for target, case, sample, action in index:
        if (target, case, sample) in labels:
            sample_actions[(target, case, sample)].add(action)
    if any(sample_actions.get(key) != set(physical_action_ids(key[0])) for key in labels):
        raise ProtocolError("OGDE scoped labels lack complete B/U/eight-A1 probabilities.")
    samples_by_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target, case, sample in sorted(labels):
        samples_by_case[(target, case)].append(sample)
    output: list[CaseActionConfusion] = []
    for (target, case), samples in sorted(samples_by_case.items()):
        truth = np.asarray([labels[(target, case, sample)].value for sample in samples], dtype=np.int8)
        baseline = np.asarray(
            [index[(target, case, sample, B_ACTION_ID)].probability_mean for sample in samples], dtype=np.float64
        )
        baseline_hard = baseline >= HARD_THRESHOLD
        for action in physical_action_ids(target):
            probabilities = np.asarray(
                [index[(target, case, sample, action)].probability_mean for sample in samples], dtype=np.float64
            )
            predicted = probabilities >= HARD_THRESHOLD
            positive = truth == 1
            negative = ~positive
            zero_to_one = (~baseline_hard) & predicted
            one_to_zero = baseline_hard & (~predicted)
            output.append(
                CaseActionConfusion(
                    target, case, action,
                    int(np.sum(positive, dtype=np.int64)),
                    int(np.sum(positive & predicted, dtype=np.int64)),
                    int(np.sum(negative, dtype=np.int64)),
                    int(np.sum(negative & (~predicted), dtype=np.int64)),
                    int(np.sum(zero_to_one & positive, dtype=np.int64)),
                    int(np.sum(zero_to_one & negative, dtype=np.int64)),
                    int(np.sum(one_to_zero & positive, dtype=np.int64)),
                    int(np.sum(one_to_zero & negative, dtype=np.int64)),
                    scope,
                )
            )
    return tuple(output)


def directional_hard_flip_gain(
    rows: Sequence[CaseActionConfusion],
    *,
    query_center: object,
    source: object,
    direction: object,
    contributing_case_ids: Sequence[str],
    excluded_case_id: str | None = None,
    label_scope: str,
) -> DirectionalGain:
    query, candidate, direction_id = str(query_center), str(source), str(direction)
    cases = tuple(sorted(str(value) for value in contributing_case_ids))
    action = a1_action_id(candidate)
    selected = tuple(
        row for row in rows if row.target_center == query and row.action_id == action and row.case_id in cases
    )
    if (
        tuple(sorted(row.case_id for row in selected)) != cases
        or len(selected) != len(cases)
        or any(row.label_scope != label_scope for row in selected)
    ):
        raise ProtocolError("OGDE directional gain lacks exact scoped case coverage.")
    positive = sum(row.n_positive for row in selected)
    negative = sum(row.n_negative for row in selected)
    if direction_id == "zero_to_one":
        favorable = sum(row.flip_0to1_positive for row in selected)
        adverse = sum(row.flip_0to1_negative for row in selected)
    elif direction_id == "one_to_zero":
        favorable = sum(row.flip_1to0_negative for row in selected)
        adverse = sum(row.flip_1to0_positive for row in selected)
    else:
        raise ProtocolError("OGDE directional gain direction drifted.")
    return DirectionalGain(
        query, excluded_case_id, candidate, direction_id, positive, negative,
        favorable, adverse, cases, label_scope,
    )


def score_loo_directional_gains(
    rows: Sequence[CaseActionConfusion], plan: WholeCaseLooPlan
) -> tuple[DirectionalGain, ...]:
    values = tuple(rows)
    scopes = {row.label_scope for row in values}
    if len(scopes) != 1:
        raise ProtocolError("OGDE LOO gains mix label capabilities.")
    scope = next(iter(scopes))
    if (
        {row.case_id for row in values if row.target_center == plan.target_center} != set(plan.support_case_ids)
        or any(row.case_id == plan.case_id for row in values)
    ):
        raise ProtocolError("OGDE LOO gain input is not exactly H-minus-c.")
    return tuple(
        directional_hard_flip_gain(
            values,
            query_center=plan.target_center,
            source=source,
            direction=direction,
            contributing_case_ids=plan.support_case_ids,
            excluded_case_id=plan.case_id,
            label_scope=scope,
        )
        for source in candidate_sources(plan.target_center)
        for direction in DIRECTION_IDS
    )


__all__ = (
    "BinaryLabel",
    "CaseActionConfusion",
    "DirectionalGain",
    "directional_hard_flip_gain",
    "score_case_action_confusions",
    "score_loo_directional_gains",
)
