"""Independent B/I/R/P reconstruction for arbitrary excluded-case states."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    ENDPOINT_METHOD_IDS,
    HARD_THRESHOLD,
    HELD_FEATURE_NAMES,
    IDENTIFICATION_METHOD_ID,
    K_GRID,
    PORTFOLIO_IDENTIFICATION_WEIGHT,
    PORTFOLIO_METHOD_ID,
    PORTFOLIO_ROBUST_WEIGHT,
    RIDGE_ALPHA,
    ROBUST_METHOD_ID,
    W_GRID,
    a1_action_id,
    candidate_sources,
)
from .contracts import BinaryLabel, CenterProbabilitySurface, EndpointCasePrediction
from .hashing import canonical_hash


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
class EndpointState:
    target_center: str
    support_case_ids: tuple[str, ...]
    model_mean: np.ndarray
    model_scale: np.ndarray
    model_coefficients: np.ndarray
    model_valid: np.ndarray
    robust_sources: tuple[tuple[str | None, str | None], ...]
    support_n_positive: int
    support_n_negative: int
    support_gains: Mapping[tuple[str, str], float]
    donor_priors: Mapping[tuple[str, str], float]
    state_hash: str
    model_fit_count: int = 16

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            EndpointState,
            (
                self.target_center,
                self.support_case_ids,
                self.model_mean,
                self.model_scale,
                self.model_coefficients,
                self.model_valid,
                self.robust_sources,
                self.support_n_positive,
                self.support_n_negative,
                dict(self.support_gains),
                dict(self.donor_priors),
                self.state_hash,
                self.model_fit_count,
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


def _fit_irls(raw: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = np.column_stack((np.ones(len(raw), dtype=np.float64), (raw - mean) / scale))
    valid = trials > 0
    beta = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    if np.any(valid):
        x = design[valid]
        y = successes[valid].astype(np.float64)
        n = trials[valid].astype(np.float64)
        penalty = np.diag(np.asarray([0.0, *([RIDGE_ALPHA] * raw.shape[1])], dtype=np.float64))
        for _ in range(50):
            eta = np.clip(x @ beta, -30.0, 30.0)
            probability = np.clip(1.0 / (1.0 + np.exp(-eta)), 1.0e-12, 1.0 - 1.0e-12)
            gradient = x.T @ (y - n * probability) - penalty @ beta
            information = x.T @ ((n * probability * (1.0 - probability))[:, None] * x) + penalty
            try:
                update = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(update).all():
                break
            beta += update
            if float(np.max(np.abs(update))) <= 1.0e-12:
                converged = True
                break
    return mean, scale, beta, bool(converged and np.any(valid))


def _predict_irls(mean: np.ndarray, scale: np.ndarray, beta: np.ndarray, valid: bool, held: np.ndarray) -> float | None:
    if not valid:
        return None
    design = np.concatenate((np.ones(1, dtype=np.float64), (held - mean) / scale))
    value = float(1.0 / (1.0 + np.exp(-float(np.clip(design @ beta, -30.0, 30.0)))))
    return float(np.clip(value, 1.0e-12, 1.0 - 1.0e-12)) if math.isfinite(value) else None


def compute_donor_priors(
    prepared_by_center: Mapping[str, PreparedCenter],
    labels_by_source: Mapping[str, Mapping[str, Sequence[BinaryLabel]]],
    *,
    heldout_center: object,
    excluded_query_centers: Sequence[str] = (),
) -> Mapping[tuple[str, str], float]:
    """Compute equal-query-center priors from q outside H and candidate e."""

    heldout = str(heldout_center)
    excluded = {str(value) for value in excluded_query_centers}
    if heldout in excluded or any(center not in CENTERS for center in excluded):
        raise ProtocolError("Donor-prior extra query exclusion drifted.")
    priors: dict[tuple[str, str], float] = {}
    for source in candidate_sources(heldout):
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
        for query in legal_queries:
            prepared = prepared_by_center[query]
            rows = tuple(labels_by_center[query])
            labels = {row.sample_id: row.value for row in rows}
            if (
                len(labels) != len(rows)
                or {row.center for row in rows} != {query}
                or set(labels) != set(prepared.surface.sample_ids)
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
    return MappingProxyType(priors)


def build_center_case_outcomes(
    prepared: PreparedCenter,
    scoped_labels: Sequence[BinaryLabel],
) -> CenterCaseOutcomes:
    """Reduce one exact case capability to deterministic int64 statistics."""

    rows = tuple(scoped_labels)
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


def fit_endpoint_state_from_outcomes(
    prepared: PreparedCenter,
    *,
    support_case_ids: Sequence[str],
    outcomes: CenterCaseOutcomes,
    donor_priors: Mapping[tuple[str, str], float],
) -> EndpointState:
    """Fit one state from an exact, already scoped sufficient-stat surface."""

    cases = tuple(sorted(str(value) for value in support_case_ids))
    if (
        outcomes.center != prepared.surface.center
        or cases != outcomes.case_ids
        or tuple(donor_priors)
        != tuple(
            (source, direction)
            for source in candidate_sources(prepared.surface.center)
            for direction in DIRECTION_IDS
        )
    ):
        raise ProtocolError("Endpoint sufficient-stat scope or donor priors drifted.")
    sources = candidate_sources(prepared.surface.center)
    mean = np.zeros((len(sources), 2, len(HELD_FEATURE_NAMES)), dtype=np.float64)
    scale = np.ones_like(mean)
    coefficients = np.zeros((len(sources), 2, 1 + len(HELD_FEATURE_NAMES)), dtype=np.float64)
    valid = np.zeros((len(sources), 2), dtype=bool)
    support_gain: dict[tuple[str, str], Fraction] = {}
    n_positive = int(np.sum(outcomes.n_positive, dtype=np.int64))
    n_negative = int(np.sum(outcomes.n_negative, dtype=np.int64))
    if not n_positive or not n_negative:
        raise ProtocolError("Endpoint support state must retain both classes.")
    case_index = {case: prepared.cases.index(case) for case in cases}
    for source_index, source in enumerate(sources):
        for direction_index, direction in enumerate(DIRECTION_IDS):
            successes = outcomes.successes[:, source_index, direction_index]
            trials = outcomes.trials[:, source_index, direction_index]
            fitted = _fit_irls(
                np.asarray(
                    [prepared.feature_values[case_index[case], source_index, direction_index] for case in cases],
                    dtype=np.float64,
                ),
                successes.astype(np.float64, copy=False),
                trials.astype(np.float64, copy=False),
            )
            mean[source_index, direction_index] = fitted[0]
            scale[source_index, direction_index] = fitted[1]
            coefficients[source_index, direction_index] = fitted[2]
            valid[source_index, direction_index] = fitted[3]
            favorable = int(np.sum(successes, dtype=np.int64))
            adverse = int(np.sum(trials, dtype=np.int64)) - favorable
            if direction == "zero_to_one":
                support_gain[(source, direction)] = (
                    Fraction(favorable, 2 * n_positive)
                    - Fraction(adverse, 2 * n_negative)
                )
            else:
                support_gain[(source, direction)] = (
                    Fraction(favorable, 2 * n_negative)
                    - Fraction(adverse, 2 * n_positive)
                )
    robust_sources = _select_robust_sources(
        sources,
        support_gain,
        donor_priors,
    )
    support_gain_values = MappingProxyType(
        {key: float(value) for key, value in support_gain.items()}
    )
    payload = _endpoint_state_payload(
        target_center=prepared.surface.center,
        support_case_ids=cases,
        model_mean=mean,
        model_scale=scale,
        model_coefficients=coefficients,
        model_valid=valid,
        robust_sources=robust_sources,
        support_n_positive=n_positive,
        support_n_negative=n_negative,
        support_gains=support_gain_values,
        donor_priors=donor_priors,
    )
    return EndpointState(
        prepared.surface.center,
        cases,
        mean,
        scale,
        coefficients,
        valid,
        robust_sources,
        n_positive,
        n_negative,
        support_gain_values,
        MappingProxyType(dict(donor_priors)),
        canonical_hash(payload),
    )


def _select_robust_sources(
    sources: Sequence[str],
    support_gain: Mapping[tuple[str, str], object],
    donor_priors: Mapping[tuple[str, str], float],
) -> tuple[tuple[str | None, str | None], ...]:
    robust_sources: list[tuple[str | None, str | None]] = []
    for k in K_GRID:
        for weight in W_GRID:
            selections: list[str | None] = []
            for direction in DIRECTION_IDS:
                ranked = tuple(
                    sorted(
                        sources,
                        key=lambda source: (-donor_priors[(source, direction)], int(source)),
                    )
                )[:k]
                scores: dict[str | None, float] = {None: 0.0}
                for source in ranked:
                    scores[source] = (
                        weight * float(support_gain[(source, direction)])
                        + (1.0 - weight) * float(donor_priors[(source, direction)])
                    )
                maximum = max(scores.values())
                selected = min(
                    (source for source, value in scores.items() if maximum - value <= 1.0e-12),
                    key=lambda source: -1 if source is None else int(source),
                )
                selections.append(selected)
            robust_sources.append((selections[0], selections[1]))
    return tuple(robust_sources)


def _endpoint_state_payload(
    *,
    target_center: str,
    support_case_ids: Sequence[str],
    model_mean: np.ndarray,
    model_scale: np.ndarray,
    model_coefficients: np.ndarray,
    model_valid: np.ndarray,
    robust_sources: Sequence[tuple[str | None, str | None]],
    support_n_positive: int,
    support_n_negative: int,
    support_gains: Mapping[tuple[str, str], float],
    donor_priors: Mapping[tuple[str, str], float],
) -> dict[str, object]:
    sources = candidate_sources(target_center)
    return {
        "schema_version": "fixed_bank_pdcb_endpoint_state_v1",
        "target_center": target_center,
        "support_case_ids": list(support_case_ids),
        "model_mean": model_mean.tolist(),
        "model_scale": model_scale.tolist(),
        "model_coefficients": model_coefficients.tolist(),
        "model_valid": model_valid.astype(np.uint8).tolist(),
        "robust_sources": [list(row) for row in robust_sources],
        "support_n_positive": support_n_positive,
        "support_n_negative": support_n_negative,
        "support_gains": [
            [source, direction, support_gains[(source, direction)]]
            for source in sources
            for direction in DIRECTION_IDS
        ],
        "donor_priors": [
            [source, direction, donor_priors[(source, direction)]]
            for source in sources
            for direction in DIRECTION_IDS
        ],
        "raw_labels_persisted": False,
    }


def rebind_endpoint_state_priors(
    state: EndpointState,
    donor_priors: Mapping[tuple[str, str], float],
) -> EndpointState:
    """Recompose only the cheap prior-dependent part of a fitted state."""

    sources = candidate_sources(state.target_center)
    expected = tuple(
        (source, direction) for source in sources for direction in DIRECTION_IDS
    )
    if tuple(donor_priors) != expected:
        raise ProtocolError("Rebound donor prior surface drifted.")
    robust_sources = _select_robust_sources(
        sources, state.support_gains, donor_priors
    )
    payload = _endpoint_state_payload(
        target_center=state.target_center,
        support_case_ids=state.support_case_ids,
        model_mean=state.model_mean,
        model_scale=state.model_scale,
        model_coefficients=state.model_coefficients,
        model_valid=state.model_valid,
        robust_sources=robust_sources,
        support_n_positive=state.support_n_positive,
        support_n_negative=state.support_n_negative,
        support_gains=state.support_gains,
        donor_priors=donor_priors,
    )
    return EndpointState(
        state.target_center,
        state.support_case_ids,
        state.model_mean,
        state.model_scale,
        state.model_coefficients,
        state.model_valid,
        robust_sources,
        state.support_n_positive,
        state.support_n_negative,
        state.support_gains,
        MappingProxyType(dict(donor_priors)),
        canonical_hash(payload),
        0,
    )


def fit_endpoint_state(
    prepared: PreparedCenter,
    *,
    support_case_ids: Sequence[str],
    scoped_labels: Sequence[BinaryLabel],
    donor_priors: Mapping[tuple[str, str], float],
) -> EndpointState:
    """Compatibility facade that keeps raw labels at the capability boundary."""

    cases = tuple(sorted(str(value) for value in support_case_ids))
    outcomes = build_center_case_outcomes(prepared, scoped_labels)
    if outcomes.case_ids != cases:
        raise ProtocolError("Endpoint raw-label capability contains extra cases.")
    return fit_endpoint_state_from_outcomes(
        prepared,
        support_case_ids=cases,
        outcomes=outcomes,
        donor_priors=donor_priors,
    )


def reconstruct_case_endpoints(
    prepared: PreparedCenter,
    state: EndpointState,
    *,
    evaluation_case_id: object,
) -> EndpointCasePrediction:
    case = str(evaluation_case_id)
    if case in state.support_case_ids or state.target_center != prepared.surface.center:
        raise ProtocolError("Endpoint evaluation case was not excluded from its state.")
    positions = prepared.case_positions[case]
    sources = candidate_sources(prepared.surface.center)
    case_index = prepared.cases.index(case)
    selected_i: dict[str, str | None] = {}
    for direction_index, direction in enumerate(DIRECTION_IDS):
        intermediate: list[tuple[str, float, float, bool]] = []
        for source_index, source in enumerate(sources):
            predicted = _predict_irls(
                state.model_mean[source_index, direction_index],
                state.model_scale[source_index, direction_index],
                state.model_coefficients[source_index, direction_index],
                bool(state.model_valid[source_index, direction_index]),
                prepared.feature_values[case_index, source_index, direction_index],
            )
            flips = int(prepared.directional_flip_counts[case_index, source_index, direction_index])
            if predicted is None:
                proxy = 0.0
            elif direction == "zero_to_one":
                proxy = (
                    flips * predicted / (2 * state.support_n_positive)
                    - flips * (1.0 - predicted) / (2 * state.support_n_negative)
                )
            else:
                proxy = (
                    flips * predicted / (2 * state.support_n_negative)
                    - flips * (1.0 - predicted) / (2 * state.support_n_positive)
                )
            intermediate.append(
                (
                    source,
                    float(proxy),
                    float(state.donor_priors[(source, direction)]),
                    bool(flips > 0 and predicted is not None and proxy > 0.0),
                )
            )
        case_scale = float(np.mean([abs(row[1]) for row in intermediate], dtype=np.float64))
        donor_scale = float(np.mean([abs(row[2]) for row in intermediate], dtype=np.float64))
        scores = [
            (
                source,
                0.8 * (proxy / case_scale)
                + 0.2 * (0.0 if donor_scale == 0.0 else prior / donor_scale),
            )
            for source, proxy, prior, eligible in intermediate
            if eligible and case_scale > 0.0
        ]
        if not scores or max(value for _, value in scores) <= 1.0e-12:
            selected_i[direction] = None
        else:
            maximum = max(value for _, value in scores)
            selected_i[direction] = min(
                (source for source, value in scores if maximum - value <= 1.0e-12),
                key=int,
            )
    baseline = prepared.action_means[B_ACTION_ID][positions].astype(np.float64, copy=True)
    baseline_hard = baseline >= HARD_THRESHOLD
    identification = baseline.copy()
    for branch, direction in ((False, "zero_to_one"), (True, "one_to_zero")):
        source = selected_i[direction]
        mask = baseline_hard == branch
        if source is not None:
            identification[mask] = prepared.action_means[a1_action_id(source)][positions][mask]
    arm_probabilities: list[np.ndarray] = []
    for zero_source, one_source in state.robust_sources:
        values = baseline.copy()
        for branch, source in ((False, zero_source), (True, one_source)):
            mask = baseline_hard == branch
            if source is not None:
                values[mask] = prepared.action_means[a1_action_id(source)][positions][mask]
        arm_probabilities.append(values)
    robust = np.mean(np.stack(arm_probabilities), axis=0, dtype=np.float64)
    portfolio = (
        PORTFOLIO_IDENTIFICATION_WEIGHT * identification
        + PORTFOLIO_ROBUST_WEIGHT * robust
    )
    probabilities = MappingProxyType(
        {
            ENDPOINT_METHOD_IDS[0]: tuple(float(value) for value in baseline),
            IDENTIFICATION_METHOD_ID: tuple(float(value) for value in identification),
            ROBUST_METHOD_ID: tuple(float(value) for value in robust),
            PORTFOLIO_METHOD_ID: tuple(float(value) for value in portfolio),
        }
    )
    return EndpointCasePrediction(
        prepared.surface.center,
        case,
        tuple(prepared.surface.sample_ids[position] for position in positions),
        probabilities,
        state.state_hash,
    )


__all__ = (
    "CenterCaseOutcomes",
    "EndpointState",
    "PreparedCenter",
    "build_center_case_outcomes",
    "compute_donor_priors",
    "fit_endpoint_state",
    "fit_endpoint_state_from_outcomes",
    "prepare_center",
    "rebind_endpoint_state_priors",
    "reconstruct_case_endpoints",
)
