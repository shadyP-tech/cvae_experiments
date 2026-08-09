"""Conservative lower-bound routing diagnostic with exact-B abstention."""

from __future__ import annotations

import math

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CONFIDENCE_MULTIPLIER,
    EXACT_BACC_DELTA,
    MINIMUM_ROUTE_GAIN,
    OUTER_INFERENCE_UNIT_COUNT,
    PRIMARY_R_FAMILY_ID,
    STUDENT_T_975_DF8,
    candidate_sources,
)
from .metric_contracts import (
    AbstentionDecisionRow,
    AbstentionSummaryRow,
    FamilySummaryRow,
)
from .model_contracts import ExactCrossfitResult


_TOLERANCE = 1.0e-12


def summarize_abstention_diagnostic(
    crossfit: ExactCrossfitResult,
    *,
    family_summaries: tuple[FamilySummaryRow, ...] | None = None,
    confidence_multiplier: float = CONFIDENCE_MULTIPLIER,
    minimum_route_gain: float = MINIMUM_ROUTE_GAIN,
) -> tuple[tuple[AbstentionDecisionRow, ...], tuple[AbstentionSummaryRow, ...]]:
    """Apply a fixed diagnostic LCB rule; this never authorizes target actions."""

    if not isinstance(crossfit, ExactCrossfitResult):
        raise ProtocolError("Abstention requires a typed exact crossfit result.")
    if (
        not np.isclose(confidence_multiplier, CONFIDENCE_MULTIPLIER, atol=0.0)
        or not np.isclose(minimum_route_gain, MINIMUM_ROUTE_GAIN, atol=0.0)
    ):
        raise ProtocolError("Abstention constants are fixed and may not be tuned.")
    if family_summaries is None:
        from .metrics import summarize_exact_crossfit  # noqa: PLC0415

        _, _, family_summaries = summarize_exact_crossfit(crossfit)
    primary = tuple(
        row for row in family_summaries if row.family_id == PRIMARY_R_FAMILY_ID
    )
    if len(primary) != 1:
        raise ProtocolError("Abstention requires one predeclared primary summary.")
    primary_gate_passed = primary[0].exact_gate_passed
    grouped: dict[tuple[str, str, str], list[object]] = {}
    for row in crossfit.predictions:
        if row.response_name != EXACT_BACC_DELTA:
            raise ProtocolError("Smooth predictions entered exact abstention.")
        grouped.setdefault(
            (row.family_id, row.outer_target_id, row.query_id), []
        ).append(row)
    decisions: list[AbstentionDecisionRow] = []
    for family_id in crossfit.family_ids:
        for outer in CENTERS:
            for query in (value for value in CENTERS if value != outer):
                rows = _ordered(
                    grouped[(family_id, outer, query)], outer, query
                )
                predicted = np.asarray(
                    [row.predicted_delta for row in rows], dtype=np.float64
                )
                selected_index = min(_top_indices(predicted))
                selected = rows[selected_index]
                lower = float(
                    selected.predicted_delta
                    - confidence_multiplier * selected.prediction_standard_error
                )
                routed = bool(
                    primary_gate_passed
                    and family_id == PRIMARY_R_FAMILY_ID
                    and lower > minimum_route_gain
                )
                observed = float(selected.observed_delta)
                decisions.append(
                    AbstentionDecisionRow(
                        family_id=family_id,
                        outer_target_id=outer,
                        query_id=query,
                        selected_source=selected.candidate_source,
                        predicted_exact_gain=float(selected.predicted_delta),
                        prediction_standard_error=float(
                            selected.prediction_standard_error
                        ),
                        lower_confidence_bound=lower,
                        minimum_route_gain=minimum_route_gain,
                        routed=routed,
                        observed_selected_exact_gain=observed,
                        deployed_exact_gain=observed if routed else 0.0,
                    )
                )
    summaries: list[AbstentionSummaryRow] = []
    for family_id in crossfit.family_ids:
        rows = tuple(row for row in decisions if row.family_id == family_id)
        if len(rows) != 72:
            raise ProtocolError("Abstention query coverage drifted.")
        outer_means = np.asarray(
            [
                np.mean(
                    [
                        row.deployed_exact_gain
                        for row in rows
                        if row.outer_target_id == outer
                    ],
                    dtype=np.float64,
                )
                for outer in CENTERS
            ],
            dtype=np.float64,
        )
        lower, upper = _student_t_ci95(outer_means)
        routed_values = [
            row.observed_selected_exact_gain for row in rows if row.routed
        ]
        summaries.append(
            AbstentionSummaryRow(
                family_id=family_id,
                query_count=len(rows),
                routed_query_count=len(routed_values),
                route_coverage=len(routed_values) / float(len(rows)),
                mean_deployed_exact_gain=float(np.mean(outer_means)),
                deployed_gain_ci95_lower=lower,
                deployed_gain_ci95_upper=upper,
                mean_routed_exact_gain=(
                    0.0
                    if not routed_values
                    else float(np.mean(routed_values, dtype=np.float64))
                ),
                confidence_multiplier=confidence_multiplier,
                minimum_route_gain=minimum_route_gain,
            )
        )
    return tuple(decisions), tuple(summaries)


def _ordered(rows: list[object], outer: str, query: str) -> tuple[object, ...]:
    by_source = {row.candidate_source: row for row in rows}
    sources = candidate_sources(outer, query)
    if len(by_source) != len(sources) or set(by_source) != set(sources):
        raise ProtocolError("Abstention candidate geometry drifted.")
    return tuple(by_source[source] for source in sources)


def _top_indices(values: np.ndarray) -> tuple[int, ...]:
    maximum = float(np.max(values))
    return tuple(
        index
        for index, value in enumerate(values)
        if abs(float(value) - maximum) <= _TOLERANCE
    )


def _student_t_ci95(values: np.ndarray) -> tuple[float, float]:
    if values.shape != (OUTER_INFERENCE_UNIT_COUNT,) or not np.isfinite(values).all():
        raise ProtocolError("Abstention inference requires nine outer centers.")
    mean = float(np.mean(values, dtype=np.float64))
    margin = (
        STUDENT_T_975_DF8
        * float(np.std(values, ddof=1))
        / math.sqrt(float(OUTER_INFERENCE_UNIT_COUNT))
    )
    return mean - margin, mean + margin


__all__ = ("summarize_abstention_diagnostic",)
