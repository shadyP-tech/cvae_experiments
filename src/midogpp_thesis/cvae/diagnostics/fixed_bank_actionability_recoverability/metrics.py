"""Exact pooled metrics, terminal oracles, and routability diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from .contracts import (
    BinaryLabelRow,
    BinaryPredictionRow,
    CaseConfusionCounts,
    PooledBacc,
    TerminalOracleResult,
)
from .hashing import finite


@dataclass(frozen=True)
class PairwiseComplementarity:
    target_center: str
    action_left: str
    action_right: str
    sample_count: int
    disagreement_rate: float
    left_only_correct_rate: float
    right_only_correct_rate: float
    class_zero_disagreement_rate: float
    class_one_disagreement_rate: float
    class_zero_complementary_correct_rate: float
    class_one_complementary_correct_rate: float

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.action_left >= self.action_right:
            raise ProtocolError("Pairwise complementarity identity is not canonical.")
        if self.sample_count <= 0:
            raise ProtocolError("Pairwise complementarity requires samples.")
        for name in (
            "disagreement_rate",
            "left_only_correct_rate",
            "right_only_correct_rate",
            "class_zero_disagreement_rate",
            "class_one_disagreement_rate",
            "class_zero_complementary_correct_rate",
            "class_one_complementary_correct_rate",
        ):
            value = finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ProtocolError("Complementarity rates must lie in [0, 1].")


@dataclass(frozen=True)
class RankStabilityResult:
    action_ids: tuple[str, ...]
    support_ranks: tuple[float, ...]
    evaluation_ranks: tuple[float, ...]
    spearman: float
    identifiable: bool

    def __post_init__(self) -> None:
        if len(self.action_ids) < 2 or len(set(self.action_ids)) != len(self.action_ids):
            raise ProtocolError("Rank stability needs at least two unique actions.")
        if len(self.support_ranks) != len(self.action_ids) or len(self.evaluation_ranks) != len(self.action_ids):
            raise ProtocolError("Rank stability vectors are misaligned.")
        correlation = finite(self.spearman, "spearman")
        if not -1.0 <= correlation <= 1.0 or type(self.identifiable) is not bool:
            raise ProtocolError("Rank stability result is invalid.")


def score_case_confusions(
    predictions: Sequence[BinaryPredictionRow],
    labels: Sequence[BinaryLabelRow],
) -> tuple[CaseConfusionCounts, ...]:
    """Reduce sample predictions to sufficient statistics; never compute case BACC."""

    predicted = tuple(predictions)
    truth = tuple(labels)
    if not predicted or not truth:
        raise ProtocolError("Exact scoring inputs must be non-empty.")
    label_by_key = {row.sample_key: row.label for row in truth}
    if len(label_by_key) != len(truth):
        raise ProtocolError("Label surface contains duplicate sample identities.")
    prediction_by_action_key: dict[tuple[str, tuple[str, str, str]], int] = {}
    for row in predicted:
        key = (row.action_id, row.sample_key)
        if key in prediction_by_action_key:
            raise ProtocolError("Prediction surface contains duplicate action/sample rows.")
        prediction_by_action_key[key] = row.hard_prediction
    action_ids = tuple(sorted({row.action_id for row in predicted}))
    for action in action_ids:
        keys = {sample for candidate, sample in prediction_by_action_key if candidate == action}
        if keys != set(label_by_key):
            raise ProtocolError("Every scored action must exactly align to the label surface.")
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for action in action_ids:
        for sample_key in sorted(label_by_key):
            target, case_id, _sample_id = sample_key
            grouped[(target, case_id, action)].append(
                (label_by_key[sample_key], prediction_by_action_key[(action, sample_key)])
            )
    return tuple(
        CaseConfusionCounts(
            target_center=target,
            case_id=case_id,
            action_id=action,
            n_positive=sum(label == 1 for label, _prediction in values),
            true_positive=sum(label == prediction == 1 for label, prediction in values),
            n_negative=sum(label == 0 for label, _prediction in values),
            true_negative=sum(label == prediction == 0 for label, prediction in values),
        )
        for (target, case_id, action), values in sorted(grouped.items())
    )


def pooled_exact_bacc(
    rows: Sequence[CaseConfusionCounts],
    *,
    action_or_method_id: str | None = None,
) -> PooledBacc:
    values = tuple(rows)
    if not values:
        raise ProtocolError("Cannot pool an empty confusion-count surface.")
    cases = {row.case_key for row in values}
    if len(cases) != len(values):
        raise ProtocolError("Pooled exact BACC requires one sufficient-statistic row per case.")
    actions = {row.action_id for row in values}
    if action_or_method_id is None and len(actions) != 1:
        raise ProtocolError("Mixed actions require an explicit method identifier.")
    n_positive = sum(row.n_positive for row in values)
    n_negative = sum(row.n_negative for row in values)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled exact BACC requires both classes.")
    true_positive = sum(row.true_positive for row in values)
    true_negative = sum(row.true_negative for row in values)
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    return PooledBacc(
        action_or_method_id=action_or_method_id or next(iter(actions)),
        case_count=len(values),
        n_positive=n_positive,
        true_positive=true_positive,
        n_negative=n_negative,
        true_negative=true_negative,
        sensitivity=sensitivity,
        specificity=specificity,
        exact_bacc=0.5 * (sensitivity + specificity),
    )


def terminal_oracles(
    counts: Sequence[CaseConfusionCounts],
    *,
    target_center: str,
    geometry_id: str,
) -> tuple[TerminalOracleResult, TerminalOracleResult]:
    """Compute terminal static/case oracles from additive pooled-BACC terms."""

    target = str(target_center)
    if target not in MIDOGPP_CENTERS or geometry_id not in GEOMETRY_IDS:
        raise ProtocolError("Terminal oracle has an invalid target/geometry context.")
    action_order = (
        U_ACTION_ID,
        *(geometry_action_id(geometry_id, source) for source in candidate_sources(target)),
    )
    rows = tuple(row for row in counts if row.target_center == target and row.action_id in action_order)
    cases = tuple(sorted({row.case_id for row in rows}))
    by_key = {(row.action_id, row.case_id): row for row in rows}
    expected = {(action, case_id) for action in action_order for case_id in cases}
    if not cases or set(by_key) != expected:
        raise ProtocolError("Terminal oracle surface lacks complete action/case coverage.")
    for case_id in cases:
        class_counts = {
            (by_key[(action, case_id)].n_positive, by_key[(action, case_id)].n_negative)
            for action in action_order
        }
        if len(class_counts) != 1:
            raise ProtocolError("Oracle actions do not share fixed label denominators.")

    pooled_by_action = {
        action: pooled_exact_bacc(tuple(by_key[(action, case_id)] for case_id in cases))
        for action in action_order
    }
    static_action = max(
        action_order,
        key=lambda action: (pooled_by_action[action].exact_bacc, -action_order.index(action)),
    )
    static_rows = tuple(by_key[(static_action, case_id)] for case_id in cases)
    static = TerminalOracleResult(
        target_center=target,
        geometry_id=geometry_id,
        oracle_method="O_static",
        selected_action_by_case=tuple((case_id, static_action) for case_id in cases),
        pooled_bacc=pooled_exact_bacc(static_rows, action_or_method_id="O_static"),
    )

    n_positive = sum(by_key[(U_ACTION_ID, case_id)].n_positive for case_id in cases)
    n_negative = sum(by_key[(U_ACTION_ID, case_id)].n_negative for case_id in cases)
    selections: list[tuple[str, str]] = []
    selected_rows: list[CaseConfusionCounts] = []
    for case_id in cases:
        action = max(
            action_order,
            key=lambda candidate: (
                0.5
                * (
                    by_key[(candidate, case_id)].true_positive / n_positive
                    + by_key[(candidate, case_id)].true_negative / n_negative
                ),
                -action_order.index(candidate),
            ),
        )
        selections.append((case_id, action))
        selected_rows.append(by_key[(action, case_id)])
    case_oracle = TerminalOracleResult(
        target_center=target,
        geometry_id=geometry_id,
        oracle_method="O_case",
        selected_action_by_case=tuple(selections),
        pooled_bacc=pooled_exact_bacc(tuple(selected_rows), action_or_method_id="O_case"),
    )
    return static, case_oracle


def complementarity_metrics(
    predictions: Sequence[BinaryPredictionRow],
    labels: Sequence[BinaryLabelRow],
    *,
    target_center: str,
    action_ids: Sequence[str] | None = None,
) -> tuple[PairwiseComplementarity, ...]:
    """Measure hard-decision and complementary-correctness structure by class."""

    target = str(target_center)
    truth = {row.sample_key: row.label for row in labels if row.target_center == target}
    if not truth:
        raise ProtocolError("Complementarity requires target-local labels.")
    prediction_map = {
        (row.action_id, row.sample_key): row.hard_prediction
        for row in predictions
        if row.target_center == target
    }
    actions = tuple(sorted(set(action_ids) if action_ids is not None else {key[0] for key in prediction_map}))
    if len(actions) < 2:
        raise ProtocolError("Complementarity requires at least two actions.")
    for action in actions:
        if {sample for candidate, sample in prediction_map if candidate == action} != set(truth):
            raise ProtocolError("Complementarity actions must exactly align to target labels.")
    output: list[PairwiseComplementarity] = []
    keys = tuple(sorted(truth))
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1 :]:
            left_values = tuple(prediction_map[(left, key)] for key in keys)
            right_values = tuple(prediction_map[(right, key)] for key in keys)
            labels_values = tuple(truth[key] for key in keys)
            disagreement = tuple(l != r for l, r in zip(left_values, right_values, strict=True))
            left_correct = tuple(l == y for l, y in zip(left_values, labels_values, strict=True))
            right_correct = tuple(r == y for r, y in zip(right_values, labels_values, strict=True))
            class_rates: dict[int, tuple[float, float]] = {}
            for label in (0, 1):
                indices = tuple(index for index, value in enumerate(labels_values) if value == label)
                if not indices:
                    raise ProtocolError("Class-conditional complementarity requires both classes.")
                class_rates[label] = (
                    sum(disagreement[index] for index in indices) / len(indices),
                    sum(
                        left_correct[index] != right_correct[index] for index in indices
                    )
                    / len(indices),
                )
            output.append(
                PairwiseComplementarity(
                    target_center=target,
                    action_left=left,
                    action_right=right,
                    sample_count=len(keys),
                    disagreement_rate=sum(disagreement) / len(keys),
                    left_only_correct_rate=sum(l and not r for l, r in zip(left_correct, right_correct, strict=True)) / len(keys),
                    right_only_correct_rate=sum(r and not l for l, r in zip(left_correct, right_correct, strict=True)) / len(keys),
                    class_zero_disagreement_rate=class_rates[0][0],
                    class_one_disagreement_rate=class_rates[1][0],
                    class_zero_complementary_correct_rate=class_rates[0][1],
                    class_one_complementary_correct_rate=class_rates[1][1],
                )
            )
    return tuple(output)


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = 0.5 * ((position + 1) + end)
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return tuple(ranks)


def rank_stability(
    support_scores: Mapping[str, float],
    evaluation_scores: Mapping[str, float],
) -> RankStabilityResult:
    """Tie-aware Spearman stability over the same fixed action set."""

    if set(support_scores) != set(evaluation_scores) or len(support_scores) < 2:
        raise ProtocolError("Rank-stability scopes must share at least two actions.")
    actions = tuple(sorted(support_scores))
    support_values = tuple(finite(support_scores[action], "support score") for action in actions)
    evaluation_values = tuple(finite(evaluation_scores[action], "evaluation score") for action in actions)
    support_ranks = _average_ranks(support_values)
    evaluation_ranks = _average_ranks(evaluation_values)
    support_mean = math.fsum(support_ranks) / len(actions)
    evaluation_mean = math.fsum(evaluation_ranks) / len(actions)
    covariance = math.fsum(
        (left - support_mean) * (right - evaluation_mean)
        for left, right in zip(support_ranks, evaluation_ranks, strict=True)
    )
    support_norm = math.sqrt(math.fsum((value - support_mean) ** 2 for value in support_ranks))
    evaluation_norm = math.sqrt(math.fsum((value - evaluation_mean) ** 2 for value in evaluation_ranks))
    identifiable = support_norm > 0.0 and evaluation_norm > 0.0
    spearman = covariance / (support_norm * evaluation_norm) if identifiable else 0.0
    return RankStabilityResult(actions, support_ranks, evaluation_ranks, spearman, identifiable)


def normalized_oracle_gap(*, selected: float, baseline: float, oracle: float, tolerance: float = 1.0e-12) -> float:
    """Return the signed U-to-O_static normalized gap.

    A case-varying method can legitimately exceed the best static action, in
    which case the result is negative.  Callers separately report the
    degenerate ``oracle ~= baseline`` flag.
    """

    selected_value = finite(selected, "selected")
    baseline_value = finite(baseline, "baseline")
    oracle_value = finite(oracle, "oracle")
    if baseline_value > oracle_value + tolerance:
        raise ProtocolError("The static oracle cannot be below its included baseline action.")
    headroom = oracle_value - baseline_value
    if headroom <= tolerance:
        return 0.0
    return (oracle_value - selected_value) / headroom


__all__ = (
    "PairwiseComplementarity",
    "RankStabilityResult",
    "complementarity_metrics",
    "normalized_oracle_gap",
    "pooled_exact_bacc",
    "rank_stability",
    "score_case_confusions",
    "terminal_oracles",
)
