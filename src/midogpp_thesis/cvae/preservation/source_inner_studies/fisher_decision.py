"""Pure hierarchical decision logic for the Task-Fisher shrinkage study."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from .contracts import (
    FISHER_ALPHAS,
    FisherStudyDecisionV2,
    FisherStudyMetricV2,
    metric_is_finite,
)


PANEL_STABLE_FISHER_ALPHA = "PANEL_STABLE_FISHER_ALPHA"
NO_STABLE_FISHER = "NO_STABLE_FISHER"
INVALID_DECISION = "INVALID_DECISION"


def select_fisher_study_decision(
    metrics: Sequence[FisherStudyMetricV2],
    *,
    outer_target_center: str,
    expected_inner_centers: Sequence[str],
    protocol_hash: str,
    decision_contract_hash: str,
    source_metric_table_hash: str | None = None,
    training_seeds: Sequence[int] = (17, 42, 101),
    generation_seeds: Sequence[int] = (17, 42, 101),
    alphas: Sequence[float] = FISHER_ALPHAS,
    fisher_min_mean_delta: float = 0.01,
    min_inner_wins: int = 6,
    tie_margin: float = 0.01,
    safety_max_bacc_regression: float = 0.01,
) -> FisherStudyDecisionV2:
    """Select one nonzero alpha per training seed, then require exact consensus."""

    outer, inners, train_seeds, gen_seeds, resolved_alphas = _validate_axes(
        outer_target_center,
        expected_inner_centers,
        training_seeds,
        generation_seeds,
        alphas,
    )
    if (
        float(fisher_min_mean_delta) != 0.01
        or int(min_inner_wins) != 6
        or float(tie_margin) != 0.01
        or float(safety_max_bacc_regression) != 0.01
    ):
        raise ProtocolError("Fisher-shrinkage v2 decision thresholds drifted.")
    if not protocol_hash or not decision_contract_hash:
        raise ProtocolError("Fisher-study decisions require protocol and contract hashes.")

    rows = [row for row in metrics if row.outer_target_center == outer]
    computed_metric_hash = _metric_hash(rows)
    if (
        source_metric_table_hash is not None
        and source_metric_table_hash != computed_metric_hash
    ):
        raise ProtocolError("Fisher-study source metric hash is not canonical.")
    recorded_metric_hash = source_metric_table_hash or computed_metric_hash
    invalid_reason, by_key = _index_rows(
        rows,
        outer=outer,
        inners=inners,
        training_seeds=train_seeds,
        generation_seeds=gen_seeds,
        alphas=resolved_alphas,
    )
    if invalid_reason:
        return _decision(
            outer=outer,
            status=INVALID_DECISION,
            selected_alpha=None,
            train_seeds=train_seeds,
            gen_seeds=gen_seeds,
            inners=inners,
            summaries={},
            protocol_hash=protocol_hash,
            decision_contract_hash=decision_contract_hash,
            metric_hash=recorded_metric_hash,
            reason=invalid_reason,
        )

    summaries: dict[str, object] = {}
    selected_by_seed: list[float | None] = []
    for training_seed in train_seeds:
        averaged = _average_generation_seeds(
            by_key,
            training_seed=training_seed,
            inners=inners,
            generation_seeds=gen_seeds,
            alphas=resolved_alphas,
        )
        baseline = averaged[0.0]
        candidates: dict[float, dict[str, object]] = {}
        for alpha in resolved_alphas[1:]:
            summary = _comparison_summary(
                averaged[alpha],
                baseline,
                inners=inners,
            )
            safety_pass = (
                _at_least(
                    float(summary["mean_decode_delta"]),
                    -safety_max_bacc_regression,
                )
                and _at_least(
                    float(summary["mean_posterior_delta"]),
                    -safety_max_bacc_regression,
                )
            )
            qualifies = (
                _strictly_greater(
                    float(summary["mean_preservation_ratio_delta"]),
                    fisher_min_mean_delta,
                )
                and int(summary["strict_inner_wins"]) >= min_inner_wins
                and safety_pass
            )
            summary["safety_pass"] = safety_pass
            summary["qualifies"] = qualifies
            candidates[alpha] = summary
        qualifying = [
            alpha for alpha, summary in candidates.items() if bool(summary["qualifies"])
        ]
        selected_alpha: float | None = None
        if qualifying:
            best_mean = max(
                float(candidates[alpha]["mean_preservation_ratio_delta"])
                for alpha in qualifying
            )
            tied = [
                alpha
                for alpha in qualifying
                if best_mean
                - float(candidates[alpha]["mean_preservation_ratio_delta"])
                <= tie_margin + 1e-12
            ]
            selected_alpha = min(tied)
        summaries[str(training_seed)] = {
            "candidate_summaries": {
                _alpha_key(alpha): candidates[alpha]
                for alpha in resolved_alphas[1:]
            },
            "qualifying_alphas": qualifying,
            "selected_alpha": selected_alpha,
        }
        selected_by_seed.append(selected_alpha)

    first = selected_by_seed[0]
    stable = first is not None and all(alpha == first for alpha in selected_by_seed)
    return _decision(
        outer=outer,
        status=PANEL_STABLE_FISHER_ALPHA if stable else NO_STABLE_FISHER,
        selected_alpha=first if stable else None,
        train_seeds=train_seeds,
        gen_seeds=gen_seeds,
        inners=inners,
        summaries=summaries,
        protocol_hash=protocol_hash,
        decision_contract_hash=decision_contract_hash,
        metric_hash=recorded_metric_hash,
        reason=(
            "exact_nonzero_alpha_consensus_across_training_seeds"
            if stable
            else "complete_valid_panel_has_no_exact_nonzero_alpha_consensus"
        ),
    )


def _validate_axes(
    outer_target_center: str,
    expected_inner_centers: Sequence[str],
    training_seeds: Sequence[int],
    generation_seeds: Sequence[int],
    alphas: Sequence[float],
) -> tuple[
    str,
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    outer = str(outer_target_center)
    inners = tuple(str(value) for value in expected_inner_centers)
    train_seeds = tuple(int(value) for value in training_seeds)
    gen_seeds = tuple(int(value) for value in generation_seeds)
    resolved_alphas = tuple(float(value) for value in alphas)
    if outer not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Fisher-study outer center is not MIDOG++ eligible.")
    expected = tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center != outer)
    if inners != expected:
        raise ProtocolError("Fisher-study inner centers must be the exact ordered H-excluded set.")
    if train_seeds != (17, 42, 101) or gen_seeds != (17, 42, 101):
        raise ProtocolError("Fisher-study decisions require the complete 17/42/101 panel.")
    if resolved_alphas != FISHER_ALPHAS:
        raise ProtocolError("Fisher-study decisions require alphas (0, .05, .10, .25).")
    return outer, inners, train_seeds, gen_seeds, resolved_alphas


def _index_rows(
    rows: Sequence[FisherStudyMetricV2],
    *,
    outer: str,
    inners: Sequence[str],
    training_seeds: Sequence[int],
    generation_seeds: Sequence[int],
    alphas: Sequence[float],
) -> tuple[
    str,
    dict[tuple[str, int, int, float], FisherStudyMetricV2],
]:
    expected = {
        (inner, training_seed, generation_seed, alpha)
        for inner in inners
        for training_seed in training_seeds
        for generation_seed in generation_seeds
        for alpha in alphas
    }
    indexed: dict[tuple[str, int, int, float], FisherStudyMetricV2] = {}
    for row in rows:
        key = (
            str(row.inner_pseudo_target_center),
            int(row.training_seed),
            int(row.generation_seed),
            float(row.alpha),
        )
        if key in indexed:
            return "duplicate_metric_cell", {}
        indexed[key] = row
        if row.outer_target_center != outer:
            return "mixed_outer_center_evidence", {}
        if (
            not row.valid
            or not metric_is_finite(
                row.preservation_ratio, row.decode_bacc, row.posterior_bacc
            )
        ):
            return "invalid_or_nonfinite_metric_cell", {}
    observed = set(indexed)
    if observed != expected:
        missing = len(expected.difference(observed))
        extra = len(observed.difference(expected))
        return f"metric_coverage_mismatch:missing={missing}:extra={extra}", {}
    return "", indexed


def _average_generation_seeds(
    indexed: Mapping[tuple[str, int, int, float], FisherStudyMetricV2],
    *,
    training_seed: int,
    inners: Sequence[str],
    generation_seeds: Sequence[int],
    alphas: Sequence[float],
) -> dict[float, dict[str, dict[str, float]]]:
    output: dict[float, dict[str, dict[str, float]]] = {}
    for alpha in alphas:
        output[alpha] = {}
        for inner in inners:
            cells = [
                indexed[(inner, int(training_seed), int(generation_seed), alpha)]
                for generation_seed in generation_seeds
            ]
            output[alpha][inner] = {
                "preservation_ratio": _mean(row.preservation_ratio for row in cells),
                "decode_bacc": _mean(row.decode_bacc for row in cells),
                "posterior_bacc": _mean(row.posterior_bacc for row in cells),
            }
    return output


def _comparison_summary(
    candidate: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    *,
    inners: Sequence[str],
) -> dict[str, object]:
    ratio_deltas = {
        inner: float(candidate[inner]["preservation_ratio"])
        - float(baseline[inner]["preservation_ratio"])
        for inner in inners
    }
    decode_deltas = {
        inner: float(candidate[inner]["decode_bacc"])
        - float(baseline[inner]["decode_bacc"])
        for inner in inners
    }
    posterior_deltas = {
        inner: float(candidate[inner]["posterior_bacc"])
        - float(baseline[inner]["posterior_bacc"])
        for inner in inners
    }
    return {
        "mean_preservation_ratio_delta": _mean(ratio_deltas.values()),
        "strict_inner_wins": sum(delta > 0.0 for delta in ratio_deltas.values()),
        "mean_decode_delta": _mean(decode_deltas.values()),
        "mean_posterior_delta": _mean(posterior_deltas.values()),
        "preservation_ratio_delta_by_inner": ratio_deltas,
        "decode_delta_by_inner": decode_deltas,
        "posterior_delta_by_inner": posterior_deltas,
    }


def _metric_hash(rows: Sequence[FisherStudyMetricV2]) -> str:
    payloads = [row.to_payload() for row in rows]
    payloads.sort(
        key=lambda row: (
            str(row["outer_target_center"]),
            str(row["inner_pseudo_target_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            float(row["alpha"]),
        )
    )
    return stable_hash(payloads)


def _decision(
    *,
    outer: str,
    status: str,
    selected_alpha: float | None,
    train_seeds: tuple[int, ...],
    gen_seeds: tuple[int, ...],
    inners: tuple[str, ...],
    summaries: Mapping[str, object],
    protocol_hash: str,
    decision_contract_hash: str,
    metric_hash: str,
    reason: str,
) -> FisherStudyDecisionV2:
    return FisherStudyDecisionV2(
        outer_target_center=outer,
        status=status,
        selected_alpha=selected_alpha,
        training_seeds=train_seeds,
        generation_seeds=gen_seeds,
        inner_centers=inners,
        per_training_seed=dict(summaries),
        protocol_hash=protocol_hash,
        decision_contract_hash=decision_contract_hash,
        source_metric_table_hash=metric_hash,
        reason=reason,
    )


def _alpha_key(alpha: float) -> str:
    return format(float(alpha), ".2f")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / float(len(items))


def _strictly_greater(value: float, threshold: float) -> bool:
    return float(value) > float(threshold) + 1e-12


def _at_least(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold) - 1e-12
