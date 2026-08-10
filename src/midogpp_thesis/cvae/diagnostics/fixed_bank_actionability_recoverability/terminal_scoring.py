"""Exact sufficient-statistic scoring and fold rank diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
import math

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .constants import GEOMETRY_IDS, HARD_THRESHOLD, MIDOGPP_CENTERS, U_ACTION_ID, candidate_sources, geometry_action_id
from .contracts import BinaryLabelRow, BinaryPredictionRow, CaseConfusionCounts, ExactNineProbabilitySurface, MethodDecision
from .decision_contracts import FoldActionScore
from .metrics import pooled_exact_bacc, rank_stability, score_case_confusions, terminal_oracles
from .terminal_contracts import CenterMetric, FoldRankStability, TerminalMethodSummary


def coerce_terminal_labels(labels: Sequence[object]) -> tuple[BinaryLabelRow, ...]:
    raw = tuple(labels)
    if not raw:
        raise ProtocolError("Terminal evaluation requires a non-empty label capability.")
    output = []
    for row in raw:
        if hasattr(row, "label_scope") and str(getattr(row, "label_scope")) != "terminal_evaluation":
            raise ProtocolError("Only terminal-evaluation labels may enter scoring.")
        try:
            output.append(
                BinaryLabelRow(
                    str(getattr(row, "target_center")), str(getattr(row, "case_id")),
                    str(getattr(row, "sample_id")), getattr(row, "label"),
                )
            )
        except (AttributeError, TypeError) as exc:
            raise ProtocolError("Terminal labels violate the binary row contract.") from exc
    rows = tuple(sorted(output))
    if len({x.sample_key for x in rows}) != len(rows) or {x.target_center for x in rows} != set(MIDOGPP_CENTERS):
        raise ProtocolError("Terminal labels have duplicate samples or incomplete center coverage.")
    return rows


def all_action_counts(
    probabilities: ExactNineProbabilitySurface, labels: tuple[BinaryLabelRow, ...]
) -> tuple[tuple[BinaryPredictionRow, ...], tuple[CaseConfusionCounts, ...]]:
    probability_samples = {(x.target_center, x.case_id, x.sample_id) for x in probabilities.rows}
    if probability_samples != {x.sample_key for x in labels}:
        raise ProtocolError("Terminal labels do not align to the sealed probability surface.")
    predictions = tuple(
        BinaryPredictionRow(
            x.target_center, x.case_id, x.sample_id, x.action_id,
            int(x.probability_mean >= HARD_THRESHOLD),
        )
        for x in probabilities.rows
    )
    counts = tuple(
        row
        for center in MIDOGPP_CENTERS
        for row in score_case_confusions(
            tuple(x for x in predictions if x.target_center == center),
            tuple(x for x in labels if x.target_center == center),
        )
    )
    return predictions, tuple(sorted(counts))


def method_counts(
    method_id: str,
    geometry_id: str | None,
    decisions: Sequence[MethodDecision],
    counts: Sequence[CaseConfusionCounts],
    partition: CaseOOFPartition,
) -> tuple[CaseConfusionCounts, ...]:
    expected_cases = {(x.target_center, x.case_id) for x in partition.identities}
    selected = tuple(x for x in decisions if x.method_id == method_id and x.geometry_id == geometry_id)
    by_case = {(x.target_center, x.case_id): x for x in selected}
    if set(by_case) != expected_cases or len(by_case) != len(selected):
        raise ProtocolError("Sealed method decisions do not cover each case exactly once.")
    by_action = {(x.target_center, x.case_id, x.action_id): x for x in counts}
    output = []
    for case_key in sorted(expected_cases):
        decision = by_case[case_key]
        source = by_action.get((*case_key, decision.action_id))
        if source is None:
            raise ProtocolError("A sealed decision selects an unavailable target action.")
        output.append(
            CaseConfusionCounts(
                source.target_center, source.case_id, method_id, source.n_positive,
                source.true_positive, source.n_negative, source.true_negative,
            )
        )
    return tuple(output)


def oracle_counts(
    counts: Sequence[CaseConfusionCounts], geometry_id: str, oracle_method: str
) -> tuple[CaseConfusionCounts, ...]:
    by_action = {(x.target_center, x.case_id, x.action_id): x for x in counts}
    output = []
    for center in MIDOGPP_CENTERS:
        static, case = terminal_oracles(counts, target_center=center, geometry_id=geometry_id)
        oracle = static if oracle_method == "O_static" else case
        for case_id, action_id in oracle.selected_action_by_case:
            source = by_action[(center, case_id, action_id)]
            output.append(
                CaseConfusionCounts(
                    center, case_id, oracle_method, source.n_positive, source.true_positive,
                    source.n_negative, source.true_negative,
                )
            )
    return tuple(sorted(output))


def method_summary(
    counts: Sequence[CaseConfusionCounts], geometry_id: str | None, method_id: str
) -> TerminalMethodSummary:
    rows = tuple(sorted(counts))
    centers = tuple(
        CenterMetric(
            center, geometry_id, method_id,
            pooled_exact_bacc(
                tuple(x for x in rows if x.target_center == center),
                action_or_method_id=method_id,
            ),
        )
        for center in MIDOGPP_CENTERS
    )
    return TerminalMethodSummary(
        geometry_id, method_id, rows, centers,
        pooled_exact_bacc(rows, action_or_method_id=method_id),
        math.fsum(x.pooled_bacc.exact_bacc for x in centers) / 9,
        sum(x.n_positive == 0 or x.n_negative == 0 for x in rows),
    )


def validate_common_scope(summaries: Sequence[TerminalMethodSummary]) -> None:
    methods = tuple(summaries)
    reference = tuple(
        (x.target_center, x.case_id, x.n_positive, x.n_negative)
        for x in methods[0].case_confusions
    )
    for method in methods[1:]:
        observed = tuple(
            (x.target_center, x.case_id, x.n_positive, x.n_negative)
            for x in method.case_confusions
        )
        if observed != reference:
            raise ProtocolError("Terminal methods do not share one whole-case label scope.")


def fold_rank_rows(
    geometry: str,
    support_scores: Sequence[FoldActionScore],
    counts: Sequence[CaseConfusionCounts],
    partition: CaseOOFPartition,
) -> tuple[FoldRankStability, ...]:
    support_by_key = {
        (x.target_center, x.fold_ordinal, x.geometry_id, x.action_id): x.support_exact_bacc
        for x in support_scores
    }
    count_by_key = {(x.target_center, x.case_id, x.action_id): x for x in counts}
    output = []
    for fold in partition.folds:
        actions = tuple(sorted((
            U_ACTION_ID,
            *(geometry_action_id(geometry, source) for source in candidate_sources(fold.target_center)),
        )))
        support = {
            action: support_by_key[(fold.target_center, fold.fold_ordinal, geometry, action)]
            for action in actions
        }
        evaluation = {
            action: pooled_exact_bacc(
                tuple(count_by_key[(fold.target_center, case_id, action)] for case_id in fold.evaluation_case_ids)
            ).exact_bacc
            for action in actions
        }
        result = rank_stability(support, evaluation)
        output.append(
            FoldRankStability(
                fold.target_center, fold.fold_ordinal, geometry,
                tuple((x, support[x]) for x in result.action_ids),
                tuple((x, evaluation[x]) for x in result.action_ids), result,
            )
        )
    return tuple(output)


__all__ = (
    "all_action_counts", "coerce_terminal_labels", "fold_rank_rows", "method_counts",
    "method_summary", "oracle_counts", "validate_common_scope",
)
