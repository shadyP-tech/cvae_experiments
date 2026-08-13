"""Exact additive whole-case and directional hard-flip scoring."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    HARD_THRESHOLD,
    a1_action_id,
    candidate_sources,
    physical_action_ids,
)
from .loo_plans import WholeCaseLooPlan
from .products import (
    BinaryLabel,
    CaseActionConfusion,
    CaseMethodConfusion,
    DirectionalGain,
    PooledBacc,
)


def _row_probability(row: object) -> float:
    if hasattr(row, "probability_mean"):
        value = float(getattr(row, "probability_mean"))
    elif hasattr(row, "probability"):
        value = float(getattr(row, "probability"))
    else:
        raise ProtocolError("DCSE probability row has no aggregate probability.")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError("DCSE aggregate probability lies outside [0,1].")
    return value


def directional_flip_counts(
    baseline_probabilities: Sequence[float] | np.ndarray,
    action_probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[int, int, int, int]:
    """Return 01-positive, 01-negative, 10-positive, 10-negative counts."""

    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    action = np.asarray(action_probabilities, dtype=np.float64)
    truth = np.asarray(labels)
    if (
        baseline.ndim != 1
        or baseline.shape != action.shape
        or baseline.shape != truth.shape
        or baseline.size <= 0
        or not np.isfinite(baseline).all()
        or not np.isfinite(action).all()
        or np.any((baseline < 0.0) | (baseline > 1.0))
        or np.any((action < 0.0) | (action > 1.0))
        or not np.all(np.isin(truth, (0, 1)))
    ):
        raise ProtocolError("DCSE directional arrays are not aligned finite binary inputs.")
    b = baseline >= HARD_THRESHOLD
    a = action >= HARD_THRESHOLD
    positive = truth == 1
    negative = ~positive
    zero_to_one = (~b) & a
    one_to_zero = b & (~a)
    return (
        int(np.sum(zero_to_one & positive, dtype=np.int64)),
        int(np.sum(zero_to_one & negative, dtype=np.int64)),
        int(np.sum(one_to_zero & positive, dtype=np.int64)),
        int(np.sum(one_to_zero & negative, dtype=np.int64)),
    )


def directional_flip_counts_scalar(
    baseline_probabilities: Sequence[float],
    action_probabilities: Sequence[float],
    labels: Sequence[int],
) -> tuple[int, int, int, int]:
    if not baseline_probabilities or not (
        len(baseline_probabilities) == len(action_probabilities) == len(labels)
    ):
        raise ProtocolError("DCSE scalar directional inputs are empty or unaligned.")
    counts = [0, 0, 0, 0]
    for baseline, action, label in zip(
        baseline_probabilities, action_probabilities, labels, strict=True
    ):
        b = float(baseline)
        a = float(action)
        if not np.isfinite(b) or not np.isfinite(a) or not 0.0 <= b <= 1.0 or not 0.0 <= a <= 1.0:
            raise ProtocolError("DCSE scalar probabilities lie outside [0,1].")
        if isinstance(label, bool) or int(label) not in (0, 1):
            raise ProtocolError("DCSE scalar labels must be binary integers.")
        b_hard = int(b >= HARD_THRESHOLD)
        a_hard = int(a >= HARD_THRESHOLD)
        if b_hard == 0 and a_hard == 1:
            counts[0 if int(label) == 1 else 1] += 1
        elif b_hard == 1 and a_hard == 0:
            counts[2 if int(label) == 1 else 3] += 1
    return tuple(counts)  # type: ignore[return-value]


def score_case_action_confusions(
    predictions: Sequence[object] | object,
    labels: Sequence[BinaryLabel] | Sequence[object],
) -> tuple[CaseActionConfusion, ...]:
    """Score only the cases in a route-scoped capability against all 10 actions."""

    prediction_rows = tuple(getattr(predictions, "rows", predictions))
    label_rows = tuple(labels)
    if not prediction_rows or not label_rows:
        raise ProtocolError("DCSE case/action scoring requires predictions and labels.")
    truth: dict[tuple[str, str, str], int] = {}
    scopes: set[str] = set()
    for row in label_rows:
        key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        value = int(getattr(row, "value", getattr(row, "label", -1)))
        if key in truth or value not in (0, 1):
            raise ProtocolError("DCSE scoped labels are duplicated or non-binary.")
        truth[key] = value
        scopes.add(str(getattr(row, "label_scope", "unspecified_scoped_labels")))
    if len(scopes) != 1:
        raise ProtocolError("DCSE scoring cannot mix label capabilities.")

    probabilities: dict[tuple[str, str, str, str], float] = {}
    observed: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        sample_key = (str(row.target_center), str(row.case_id), str(row.sample_id))
        if sample_key not in truth:
            continue
        action = str(row.action_id)
        if action not in physical_action_ids(sample_key[0]):
            raise ProtocolError("DCSE scoped probability action is target-illegal.")
        key = (*sample_key, action)
        if key in probabilities:
            raise ProtocolError("DCSE scoped probability cells are duplicated.")
        probabilities[key] = _row_probability(row)
        observed[sample_key].add(action)
    for sample_key in truth:
        if observed.get(sample_key, set()) != set(physical_action_ids(sample_key[0])):
            raise ProtocolError("Every scoped sample needs B, U, and eight A1 probabilities.")

    samples_by_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target, case, sample in sorted(truth):
        samples_by_case[(target, case)].append(sample)
    result: list[CaseActionConfusion] = []
    for (target, case), samples in sorted(
        samples_by_case.items(), key=lambda item: (CENTERS.index(item[0][0]), item[0][1])
    ):
        y = np.asarray([truth[(target, case, sample)] for sample in samples], dtype=np.int8)
        baseline = np.asarray(
            [probabilities[(target, case, sample, B_ACTION_ID)] for sample in samples],
            dtype=np.float64,
        )
        for action in physical_action_ids(target):
            values = np.asarray(
                [probabilities[(target, case, sample, action)] for sample in samples],
                dtype=np.float64,
            )
            predicted = values >= HARD_THRESHOLD
            positive = y == 1
            negative = ~positive
            flips = (0, 0, 0, 0) if action == B_ACTION_ID else directional_flip_counts(baseline, values, y)
            result.append(
                CaseActionConfusion(
                    target_center=target,
                    case_id=case,
                    action_id=action,
                    n_positive=int(np.sum(positive, dtype=np.int64)),
                    true_positive=int(np.sum(positive & predicted, dtype=np.int64)),
                    n_negative=int(np.sum(negative, dtype=np.int64)),
                    true_negative=int(np.sum(negative & (~predicted), dtype=np.int64)),
                    flip_0to1_positive=flips[0],
                    flip_0to1_negative=flips[1],
                    flip_1to0_positive=flips[2],
                    flip_1to0_negative=flips[3],
                )
            )
    return tuple(result)


score_case_action_counts = score_case_action_confusions


def directional_hard_flip_gain(
    rows: Sequence[CaseActionConfusion],
    *,
    query_center: str,
    source: str,
    direction: str,
    excluded_case_id: str | None = None,
    contributing_case_ids: Sequence[str] | None = None,
    label_scope: str,
) -> DirectionalGain:
    """Pool additive counts first, then form one exact directional gain."""

    target = str(query_center)
    candidate = str(source)
    if direction not in DIRECTION_IDS or candidate not in candidate_sources(target):
        raise ProtocolError("DCSE directional gain query/source/direction is invalid.")
    action = a1_action_id(candidate)
    requested = None if contributing_case_ids is None else tuple(sorted(str(value) for value in contributing_case_ids))
    selected = tuple(
        row
        for row in rows
        if row.target_center == target
        and row.action_id == action
        and row.case_id != excluded_case_id
        and (requested is None or row.case_id in set(requested))
    )
    if not selected or len({row.case_id for row in selected}) != len(selected):
        raise ProtocolError("DCSE directional gain rows are empty or duplicated by case.")
    observed_cases = tuple(sorted(row.case_id for row in selected))
    if requested is not None and observed_cases != requested:
        raise ProtocolError("DCSE directional gain does not cover its exact case scope.")
    n_positive = sum(row.n_positive for row in selected)
    n_negative = sum(row.n_negative for row in selected)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("DCSE pooled directional gain requires both classes.")
    if direction == "zero_to_one":
        favorable = sum(row.flip_0to1_positive for row in selected)
        adverse = sum(row.flip_0to1_negative for row in selected)
    else:
        favorable = sum(row.flip_1to0_negative for row in selected)
        adverse = sum(row.flip_1to0_positive for row in selected)
    return DirectionalGain(
        query_center=target,
        excluded_case_id=excluded_case_id,
        source=candidate,
        direction=direction,
        n_positive=n_positive,
        n_negative=n_negative,
        favorable_count=favorable,
        adverse_count=adverse,
        contributing_case_ids=observed_cases,
        label_scope=label_scope,
    )


def score_loo_directional_gains(
    rows: Sequence[CaseActionConfusion],
    plan: WholeCaseLooPlan,
    *,
    label_scope: str | None = None,
) -> tuple[DirectionalGain, ...]:
    scope = label_scope or f"route_support::H={plan.target_center}::c={plan.case_id}"
    return tuple(
        directional_hard_flip_gain(
            rows,
            query_center=plan.target_center,
            source=source,
            direction=direction,
            excluded_case_id=plan.case_id,
            contributing_case_ids=plan.support_case_ids,
            label_scope=scope,
        )
        for source in candidate_sources(plan.target_center)
        for direction in DIRECTION_IDS
    )


def pooled_bacc(
    rows: Sequence[CaseActionConfusion] | Sequence[CaseMethodConfusion],
    *,
    scope_id: str,
    method_id: str | None = None,
) -> PooledBacc:
    values = tuple(rows)
    if not values or len({row.case_key for row in values}) != len(values):
        raise ProtocolError("DCSE pooled BACC needs unique non-empty whole cases.")
    n_positive = sum(row.n_positive for row in values)
    n_negative = sum(row.n_negative for row in values)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("DCSE pooled BACC requires both classes.")
    true_positive = sum(row.true_positive for row in values)
    true_negative = sum(row.true_negative for row in values)
    inferred = method_id
    if inferred is None:
        identities = {
            str(getattr(row, "method_id", getattr(row, "action_id", ""))) for row in values
        }
        if len(identities) != 1:
            raise ProtocolError("DCSE mixed methods require an explicit method_id.")
        inferred = next(iter(identities))
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    return PooledBacc(
        scope_id=str(scope_id),
        method_id=str(inferred),
        case_count=len(values),
        n_positive=n_positive,
        true_positive=true_positive,
        n_negative=n_negative,
        true_negative=true_negative,
        sensitivity=sensitivity,
        specificity=specificity,
        bacc=float(Fraction(true_positive, 2 * n_positive) + Fraction(true_negative, 2 * n_negative)),
    )


def exact_pooled_bacc_gain(
    candidate_rows: Sequence[CaseActionConfusion],
    baseline_rows: Sequence[CaseActionConfusion],
) -> Fraction:
    candidate = {row.case_key: row for row in candidate_rows}
    baseline = {row.case_key: row for row in baseline_rows}
    if not candidate or set(candidate) != set(baseline):
        raise ProtocolError("DCSE exact gain rows are not paired by whole case.")
    n_positive = sum(row.n_positive for row in baseline.values())
    n_negative = sum(row.n_negative for row in baseline.values())
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("DCSE exact gain requires both pooled classes.")
    if any(
        (candidate[key].n_positive, candidate[key].n_negative)
        != (baseline[key].n_positive, baseline[key].n_negative)
        for key in candidate
    ):
        raise ProtocolError("DCSE paired action class denominators drifted.")
    delta_tp = sum(candidate[key].true_positive - baseline[key].true_positive for key in candidate)
    delta_tn = sum(candidate[key].true_negative - baseline[key].true_negative for key in candidate)
    return Fraction(delta_tp, 2 * n_positive) + Fraction(delta_tn, 2 * n_negative)


__all__ = (
    "directional_flip_counts",
    "directional_flip_counts_scalar",
    "directional_hard_flip_gain",
    "exact_pooled_bacc_gain",
    "pooled_bacc",
    "score_case_action_confusions",
    "score_case_action_counts",
    "score_loo_directional_gains",
)
