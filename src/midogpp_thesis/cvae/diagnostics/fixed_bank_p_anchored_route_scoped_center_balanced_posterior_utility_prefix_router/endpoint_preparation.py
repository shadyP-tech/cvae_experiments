"""Prepared endpoint surfaces and case-level sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    HARD_THRESHOLD,
    HELD_FEATURE_NAMES,
    a1_action_id,
    candidate_sources,
)
from .contracts import BinaryLabel, CenterProbabilitySurface

@dataclass(frozen=True)
class PreparedCenter:
    surface: CenterProbabilitySurface
    cases: tuple[str, ...]
    case_positions: Mapping[str, np.ndarray]
    action_means: Mapping[str, np.ndarray]
    feature_values: np.ndarray
    directional_flip_counts: np.ndarray

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            PreparedCenter,
            (
                self.surface,
                self.cases,
                dict(self.case_positions),
                dict(self.action_means),
                self.feature_values,
                self.directional_flip_counts,
            ),
        )


@dataclass(frozen=True)

class CenterCaseOutcomes:
    """Small additive label-derived statistics, indexed by whole case.

    Raw labels are decoded only by the capability owner.  Worker processes
    receive these immutable int64 sufficient statistics and explicitly subset
    them to each sealed support scope before fitting.
    """

    center: str
    case_ids: tuple[str, ...]
    successes: np.ndarray
    trials: np.ndarray
    n_positive: np.ndarray
    n_negative: np.ndarray

    def __post_init__(self) -> None:
        cases = tuple(str(value) for value in self.case_ids)
        shape = (len(cases), 8, 2)
        successes = np.ascontiguousarray(self.successes, dtype=np.int64)
        trials = np.ascontiguousarray(self.trials, dtype=np.int64)
        positive = np.ascontiguousarray(self.n_positive, dtype=np.int64)
        negative = np.ascontiguousarray(self.n_negative, dtype=np.int64)
        if (
            self.center not in CENTERS
            or not cases
            or len(cases) != len(set(cases))
            or successes.shape != shape
            or trials.shape != shape
            or positive.shape != (len(cases),)
            or negative.shape != (len(cases),)
            or np.any(successes < 0)
            or np.any(trials < successes)
            or np.any(positive < 0)
            or np.any(negative < 0)
        ):
            raise ProtocolError("Center case sufficient statistics drifted.")
        for values in (successes, trials, positive, negative):
            values.setflags(write=False)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "n_positive", positive)
        object.__setattr__(self, "n_negative", negative)

    def subset(self, case_ids: Sequence[str]) -> "CenterCaseOutcomes":
        requested = tuple(sorted(str(value) for value in case_ids))
        if not requested or not set(requested) <= set(self.case_ids):
            raise ProtocolError("Outcome subset escapes the decoded case capability.")
        positions = np.asarray([self.case_ids.index(case) for case in requested])
        return CenterCaseOutcomes(
            self.center,
            requested,
            self.successes[positions],
            self.trials[positions],
            self.n_positive[positions],
            self.n_negative[positions],
        )


def prepare_center(surface: CenterProbabilitySurface) -> PreparedCenter:
    cases = surface.cases
    positions = {case: surface.positions(case) for case in cases}
    actions = tuple(surface.seed_probabilities)
    means = {action: surface.exact_nine_mean(action) for action in actions}
    sources = candidate_sources(surface.center)
    features = np.zeros((len(cases), len(sources), 2, len(HELD_FEATURE_NAMES)), dtype=np.float64)
    flips = np.zeros((len(cases), len(sources), 2), dtype=np.int64)
    baseline_seed = surface.seed_probabilities[B_ACTION_ID].astype(np.float64, copy=False)
    baseline_mean = means[B_ACTION_ID]
    for case_index, case in enumerate(cases):
        selected_positions = positions[case]
        for source_index, source in enumerate(sources):
            action = a1_action_id(source)
            candidate_seed = surface.seed_probabilities[action].astype(np.float64, copy=False)
            candidate_mean = means[action]
            b = baseline_mean[selected_positions]
            a = candidate_mean[selected_positions]
            for direction_index, direction in enumerate(DIRECTION_IDS):
                crossing = (
                    (b < HARD_THRESHOLD) & (a >= HARD_THRESHOLD)
                    if direction == "zero_to_one"
                    else (b >= HARD_THRESHOLD) & (a < HARD_THRESHOLD)
                )
                count = int(np.sum(crossing, dtype=np.int64))
                flips[case_index, source_index, direction_index] = count
                if not count:
                    continue
                b_selected = b[crossing]
                a_selected = a[crossing]
                seed_b = baseline_seed[:, selected_positions][:, crossing]
                seed_a = candidate_seed[:, selected_positions][:, crossing]
                if direction == "zero_to_one":
                    seed_crossing = (seed_b < HARD_THRESHOLD) & (seed_a >= HARD_THRESHOLD)
                else:
                    seed_crossing = (seed_b >= HARD_THRESHOLD) & (seed_a < HARD_THRESHOLD)
                positive_rate = np.mean(seed_a >= HARD_THRESHOLD, axis=0, dtype=np.float64)
                features[case_index, source_index, direction_index] = (
                    count / len(selected_positions),
                    float(np.mean(np.abs(b_selected - HARD_THRESHOLD), dtype=np.float64)),
                    float(np.mean(np.abs(a_selected - HARD_THRESHOLD), dtype=np.float64)),
                    float(np.mean(a_selected - b_selected, dtype=np.float64)),
                    float(np.mean(seed_crossing, dtype=np.float64)),
                    float(np.mean(2.0 * positive_rate * (1.0 - positive_rate), dtype=np.float64)),
                )
    features.setflags(write=False)
    flips.setflags(write=False)
    return PreparedCenter(
        surface,
        cases,
        MappingProxyType(positions),
        MappingProxyType(means),
        features,
        flips,
    )


def _exact_label_vector(
    prepared: PreparedCenter,
    support_case_ids: Sequence[str],
    scoped_labels: Sequence[BinaryLabel],
) -> tuple[np.ndarray, np.ndarray]:
    cases = tuple(sorted(str(value) for value in support_case_ids))
    expected = {
        (prepared.surface.center, case, prepared.surface.sample_ids[position])
        for case in cases
        for position in prepared.case_positions[case]
    }
    rows = tuple(scoped_labels)
    label_map = {row.key: row.value for row in rows}
    if (
        not cases
        or len(label_map) != len(rows)
        or set(label_map) != expected
        or len({row.scope for row in rows}) != 1
    ):
        raise ProtocolError("Endpoint state labels are not exactly the support cases.")
    positions = np.concatenate([prepared.case_positions[case] for case in cases])
    labels = np.asarray(
        [
            label_map[
                (
                    prepared.surface.center,
                    prepared.surface.case_ids[position],
                    prepared.surface.sample_ids[position],
                )
            ]
            for position in positions
        ],
        dtype=np.int8,
    )
    return positions, labels


def _case_directional_outcomes(
    prepared: PreparedCenter,
    case_id: str,
    labels_by_sample: Mapping[str, int],
    source: str,
    direction: str,
) -> tuple[int, int]:
    positions = prepared.case_positions[case_id]
    samples = tuple(prepared.surface.sample_ids[position] for position in positions)
    if set(samples) != set(labels_by_sample):
        raise ProtocolError("Case label scope drifted during directional scoring.")
    truth = np.asarray([labels_by_sample[sample] for sample in samples], dtype=np.int8)
    baseline = prepared.action_means[B_ACTION_ID][positions] >= HARD_THRESHOLD
    candidate = prepared.action_means[a1_action_id(source)][positions] >= HARD_THRESHOLD
    if direction == "zero_to_one":
        selected = (~baseline) & candidate
        favorable = selected & (truth == 1)
    else:
        selected = baseline & (~candidate)
        favorable = selected & (truth == 0)
    return int(np.sum(favorable, dtype=np.int64)), int(np.sum(selected, dtype=np.int64))



def compute_donor_priors(
    prepared_by_center: Mapping[str, PreparedCenter],
    labels_by_source: Mapping[str, Mapping[str, Sequence[BinaryLabel]]],
    *,
    heldout_center: object,
    excluded_query_centers: Sequence[str] = (),
    excluded_source_centers: Sequence[str] = (),
) -> Mapping[tuple[str, str], float]:
    """Compute equal-query-center priors from q outside H and candidate e."""

    heldout = str(heldout_center)
    excluded = {str(value) for value in excluded_query_centers}
    excluded_sources = {str(value) for value in excluded_source_centers}
    if (
        heldout in excluded
        or heldout in excluded_sources
        or any(center not in CENTERS for center in excluded | excluded_sources)
    ):
        raise ProtocolError("Donor-prior extra query exclusion drifted.")
    priors: dict[tuple[str, str], float] = {}
    for source in candidate_sources(heldout):
        if source in excluded_sources:
            for direction in DIRECTION_IDS:
                priors[(source, direction)] = 0.0
            continue
        legal_queries = tuple(
            center
            for center in prepared_by_center
            if center not in {heldout, source, *excluded}
        )
        try:
            labels_by_center = labels_by_source[source]
        except KeyError as exc:
            raise ProtocolError("Donor-prior source capability is absent.") from exc
        if tuple(labels_by_center) != legal_queries:
            raise ProtocolError(
                "Donor-prior labels must be exactly the declared external query centers."
            )
        directional_values = {direction: [] for direction in DIRECTION_IDS}
        expected_scope = (
            f"source_prior::target={heldout}::source={source}"
            if not excluded
            else (
                f"source_prior::outer_H={next(iter(excluded))}::J={heldout}::source={source}"
                if len(excluded) == 1
                else ""
            )
        )
        if not expected_scope:
            raise ProtocolError("Donor-prior supports exactly one outer exclusion.")
        for query in legal_queries:
            prepared = prepared_by_center[query]
            rows = tuple(labels_by_center[query])
            labels = {row.sample_id: row.value for row in rows}
            if (
                len(labels) != len(rows)
                or {row.center for row in rows} != {query}
                or set(labels) != set(prepared.surface.sample_ids)
                or {row.scope for row in rows} != {expected_scope}
            ):
                raise ProtocolError("Donor-prior center label capability drifted.")
            truth = np.asarray([labels[sample] for sample in prepared.surface.sample_ids], dtype=np.int8)
            baseline = prepared.action_means[B_ACTION_ID] >= HARD_THRESHOLD
            candidate = prepared.action_means[a1_action_id(source)] >= HARD_THRESHOLD
            n_positive = int(np.sum(truth == 1, dtype=np.int64))
            n_negative = int(np.sum(truth == 0, dtype=np.int64))
            if not n_positive or not n_negative:
                raise ProtocolError("Donor-prior query center lacks both classes.")
            zero_to_one = (~baseline) & candidate
            one_to_zero = baseline & (~candidate)
            directional_values["zero_to_one"].append(
                0.5
                * (
                    np.sum(zero_to_one & (truth == 1), dtype=np.int64) / n_positive
                    - np.sum(zero_to_one & (truth == 0), dtype=np.int64) / n_negative
                )
            )
            directional_values["one_to_zero"].append(
                0.5
                * (
                    np.sum(one_to_zero & (truth == 0), dtype=np.int64) / n_negative
                    - np.sum(one_to_zero & (truth == 1), dtype=np.int64) / n_positive
                )
            )
        for direction in DIRECTION_IDS:
            priors[(source, direction)] = float(
                np.mean(directional_values[direction], dtype=np.float64)
            )
    if set(labels_by_source) != set(candidate_sources(heldout)).difference(excluded_sources):
        raise ProtocolError("Donor-prior source capability set drifted.")
    return MappingProxyType(priors)


def build_center_case_outcomes(
    prepared: PreparedCenter,
    scoped_labels: Sequence[BinaryLabel],
    *,
    expected_scope: str,
) -> CenterCaseOutcomes:
    """Reduce one exact case capability to deterministic int64 statistics."""

    rows = tuple(scoped_labels)
    if not expected_scope or {row.scope for row in rows} != {expected_scope}:
        raise ProtocolError("Endpoint outcome label capability role drifted.")
    cases = tuple(sorted({row.case_id for row in rows}))
    _positions, _labels = _exact_label_vector(prepared, cases, rows)
    sources = candidate_sources(prepared.surface.center)
    successes = np.zeros((len(cases), len(sources), 2), dtype=np.int64)
    trials = np.zeros_like(successes)
    positive = np.zeros(len(cases), dtype=np.int64)
    negative = np.zeros(len(cases), dtype=np.int64)
    for case_index, case in enumerate(cases):
        labels = {row.sample_id: row.value for row in rows if row.case_id == case}
        truth = np.asarray(tuple(labels.values()), dtype=np.int8)
        positive[case_index] = np.sum(truth == 1, dtype=np.int64)
        negative[case_index] = np.sum(truth == 0, dtype=np.int64)
        for source_index, source in enumerate(sources):
            for direction_index, direction in enumerate(DIRECTION_IDS):
                favorable, total = _case_directional_outcomes(
                    prepared, case, labels, source, direction
                )
                successes[case_index, source_index, direction_index] = favorable
                trials[case_index, source_index, direction_index] = total
    return CenterCaseOutcomes(
        prepared.surface.center,
        cases,
        successes,
        trials,
        positive,
        negative,
    )



__all__ = (
    "CenterCaseOutcomes",
    "PreparedCenter",
    "build_center_case_outcomes",
    "compute_donor_priors",
    "prepare_center",
)
