"""Pure hierarchical decision logic for the learned conditional-prior study."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from .contracts import (
    PRIOR_ARMS,
    PriorStudyDecisionV2,
    PriorStudyMetricV2,
    metric_is_finite,
)


PANEL_STABLE_E_VS_A = "PANEL_STABLE_E_VS_A"
PANEL_STABLE_E_OVER_C = "PANEL_STABLE_E_OVER_C"
E_VS_C_UNAVAILABLE = "E_VS_C_UNAVAILABLE"
NO_STABLE_E = "NO_STABLE_E"
INVALID_DECISION = "INVALID_DECISION"


def select_prior_study_decision(
    metrics: Sequence[PriorStudyMetricV2],
    *,
    outer_target_center: str,
    expected_inner_centers: Sequence[str],
    protocol_hash: str,
    decision_contract_hash: str,
    source_metric_table_hash: str | None = None,
    training_seeds: Sequence[int] = (17, 42, 101),
    generation_seeds: Sequence[int] = (17, 42, 101),
    e_vs_a_min_mean_delta: float = 0.05,
    e_vs_c_min_mean_delta: float = 0.01,
    min_inner_wins: int = 6,
    safety_max_bacc_regression: float = 0.01,
) -> PriorStudyDecisionV2:
    """Recompute one outer-center decision from the complete ``r[a,H,I,t,g]`` panel.

    Generation seeds are averaged *within each inner center* before the eight
    paired center deltas are computed.  A and E are integrity-critical.  An
    unavailable C-diag panel disables only the secondary E-vs-C comparison;
    E-vs-A remains independently decidable.
    """

    outer, inners, train_seeds, gen_seeds = _validate_axes(
        outer_target_center,
        expected_inner_centers,
        training_seeds,
        generation_seeds,
    )
    if (
        float(e_vs_a_min_mean_delta) != 0.05
        or float(e_vs_c_min_mean_delta) != 0.01
        or int(min_inner_wins) != 6
        or float(safety_max_bacc_regression) != 0.01
    ):
        raise ProtocolError("Learned-prior v2 decision thresholds drifted.")
    if not protocol_hash or not decision_contract_hash:
        raise ProtocolError("Prior-study decisions require protocol and contract hashes.")

    rows = [row for row in metrics if row.outer_target_center == outer]
    computed_metric_hash = _metric_hash(rows)
    if (
        source_metric_table_hash is not None
        and source_metric_table_hash != computed_metric_hash
    ):
        raise ProtocolError("Prior-study source metric hash is not canonical.")
    recorded_metric_hash = source_metric_table_hash or computed_metric_hash
    invalid_reason, by_key, c_unavailable_by_seed = _index_rows(
        rows,
        outer=outer,
        inners=inners,
        training_seeds=train_seeds,
        generation_seeds=gen_seeds,
    )
    if invalid_reason:
        return _decision(
            outer=outer,
            status=INVALID_DECISION,
            e_vs_a_status=INVALID_DECISION,
            e_vs_c_status=INVALID_DECISION,
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
    e_vs_a_passes: list[bool] = []
    e_vs_c_passes: list[bool] = []
    for training_seed in train_seeds:
        averaged = _average_generation_seeds(
            by_key,
            training_seed=training_seed,
            inners=inners,
            generation_seeds=gen_seeds,
            arms=("A", "E"),
        )
        e_vs_a = _comparison_summary(
            averaged["E"],
            averaged["A"],
            inners=inners,
        )
        c_unavailable_reason = c_unavailable_by_seed.get(training_seed, "")
        if c_unavailable_reason:
            e_vs_c: dict[str, object] = {
                "available": False,
                "unavailability_reason": c_unavailable_reason,
            }
        else:
            c_averaged = _average_generation_seeds(
                by_key,
                training_seed=training_seed,
                inners=inners,
                generation_seeds=gen_seeds,
                arms=("C-diag",),
            )
            e_vs_c = _comparison_summary(
                averaged["E"],
                c_averaged["C-diag"],
                inners=inners,
            )
            e_vs_c["available"] = True
            e_vs_c["unavailability_reason"] = ""
        e_cells = [
            by_key[(inner, training_seed, generation_seed, "E")]
            for inner in inners
            for generation_seed in gen_seeds
        ]
        e_eligible = all(row.eligible for row in e_cells)
        ineligibility_reasons = sorted(
            {
                row.ineligibility_reason or "unspecified_mechanism_ineligibility"
                for row in e_cells
                if not row.eligible
            }
        )
        safety_pass = (
            _at_least(
                float(e_vs_a["mean_decode_delta"]),
                -safety_max_bacc_regression,
            )
            and _at_least(
                float(e_vs_a["mean_posterior_delta"]),
                -safety_max_bacc_regression,
            )
        )
        pass_a = (
            _at_least(
                float(e_vs_a["mean_preservation_ratio_delta"]),
                e_vs_a_min_mean_delta,
            )
            and int(e_vs_a["strict_inner_wins"]) >= min_inner_wins
            and safety_pass
            and e_eligible
        )
        # This secondary comparison is intentionally independent of the A
        # gate.  Its strict mean threshold must not be weakened to >=.
        pass_c = (
            not c_unavailable_reason
            and _strictly_greater(
                float(e_vs_c["mean_preservation_ratio_delta"]),
                e_vs_c_min_mean_delta,
            )
            and int(e_vs_c["strict_inner_wins"]) >= min_inner_wins
            and e_eligible
        )
        e_vs_a["safety_pass"] = safety_pass
        e_vs_a["pass"] = pass_a
        e_vs_c["pass"] = pass_c
        e_vs_a["e_mechanism_eligible"] = e_eligible
        e_vs_c["e_mechanism_eligible"] = e_eligible
        e_vs_a["e_ineligibility_reasons"] = ineligibility_reasons
        e_vs_c["e_ineligibility_reasons"] = ineligibility_reasons
        summaries[str(training_seed)] = {
            "e_vs_a": e_vs_a,
            "e_vs_c_diag": e_vs_c,
        }
        e_vs_a_passes.append(pass_a)
        e_vs_c_passes.append(pass_c)

    stable_vs_a = all(e_vs_a_passes)
    c_panel_available = not c_unavailable_by_seed
    stable_over_c = c_panel_available and all(e_vs_c_passes)
    e_vs_a_status = PANEL_STABLE_E_VS_A if stable_vs_a else "NO_STABLE_E_VS_A"
    e_vs_c_status = (
        E_VS_C_UNAVAILABLE
        if not c_panel_available
        else (
            PANEL_STABLE_E_OVER_C if stable_over_c else "NO_STABLE_E_OVER_C"
        )
    )
    if stable_vs_a and stable_over_c:
        status = PANEL_STABLE_E_OVER_C
        reason = "all_training_seeds_pass_e_vs_a_and_e_vs_c_diag"
    elif stable_vs_a:
        status = PANEL_STABLE_E_VS_A
        reason = (
            "all_training_seeds_pass_e_vs_a_c_diag_unavailable"
            if not c_panel_available
            else "all_training_seeds_pass_e_vs_a_only"
        )
    else:
        status = NO_STABLE_E
        reason = "complete_valid_panel_does_not_pass_e_vs_a_consensus"
    return _decision(
        outer=outer,
        status=status,
        e_vs_a_status=e_vs_a_status,
        e_vs_c_status=e_vs_c_status,
        train_seeds=train_seeds,
        gen_seeds=gen_seeds,
        inners=inners,
        summaries=summaries,
        protocol_hash=protocol_hash,
        decision_contract_hash=decision_contract_hash,
        metric_hash=recorded_metric_hash,
        reason=reason,
    )


def _validate_axes(
    outer_target_center: str,
    expected_inner_centers: Sequence[str],
    training_seeds: Sequence[int],
    generation_seeds: Sequence[int],
) -> tuple[str, tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    outer = str(outer_target_center)
    inners = tuple(str(value) for value in expected_inner_centers)
    train_seeds = tuple(int(value) for value in training_seeds)
    gen_seeds = tuple(int(value) for value in generation_seeds)
    if outer not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Prior-study outer center is not MIDOG++ eligible.")
    expected = tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center != outer)
    if inners != expected:
        raise ProtocolError("Prior-study inner centers must be the exact ordered H-excluded set.")
    if train_seeds != (17, 42, 101) or gen_seeds != (17, 42, 101):
        raise ProtocolError("Prior-study decisions require the complete 17/42/101 panel.")
    return outer, inners, train_seeds, gen_seeds


def _index_rows(
    rows: Sequence[PriorStudyMetricV2],
    *,
    outer: str,
    inners: Sequence[str],
    training_seeds: Sequence[int],
    generation_seeds: Sequence[int],
) -> tuple[
    str,
    dict[tuple[str, int, int, str], PriorStudyMetricV2],
    dict[int, str],
]:
    expected_core = {
        (inner, training_seed, generation_seed, arm)
        for inner in inners
        for training_seed in training_seeds
        for generation_seed in generation_seeds
        for arm in ("A", "E")
    }
    expected_c_by_seed = {
        int(training_seed): {
            (inner, int(training_seed), generation_seed, "C-diag")
            for inner in inners
            for generation_seed in generation_seeds
        }
        for training_seed in training_seeds
    }
    expected_c = set().union(*expected_c_by_seed.values())
    indexed: dict[tuple[str, int, int, str], PriorStudyMetricV2] = {}
    c_reasons: dict[int, set[str]] = {}
    for row in rows:
        key = (
            str(row.inner_pseudo_target_center),
            int(row.training_seed),
            int(row.generation_seed),
            str(row.arm),
        )
        if row.arm not in PRIOR_ARMS:
            return "unexpected_prior_arm", {}, {}
        if row.arm == "C-diag" and key not in expected_c:
            _record_c_reason(
                c_reasons,
                row.training_seed,
                training_seeds,
                "unexpected_c_diag_metric_axis",
            )
            continue
        if key in indexed:
            if row.arm == "C-diag":
                _record_c_reason(
                    c_reasons,
                    row.training_seed,
                    training_seeds,
                    "duplicate_c_diag_metric_cell",
                )
                continue
            return "duplicate_a_or_e_metric_cell", {}, {}
        indexed[key] = row
        if row.outer_target_center != outer:
            return "mixed_outer_center_evidence", {}, {}
        if (
            not row.valid
            or not metric_is_finite(
                row.preservation_ratio, row.decode_bacc, row.posterior_bacc
            )
        ):
            if row.arm == "C-diag":
                _record_c_reason(
                    c_reasons,
                    row.training_seed,
                    training_seeds,
                    "invalid_or_nonfinite_c_diag_metric_cell",
                )
                continue
            return "invalid_or_nonfinite_a_or_e_metric_cell", {}, {}
    observed_core = {key for key in indexed if key[3] in {"A", "E"}}
    if observed_core != expected_core:
        missing = len(expected_core.difference(observed_core))
        extra = len(observed_core.difference(expected_core))
        return f"a_e_metric_coverage_mismatch:missing={missing}:extra={extra}", {}, {}
    for training_seed, expected_c in expected_c_by_seed.items():
        observed_c = {
            key
            for key in indexed
            if key[1] == training_seed and key[3] == "C-diag"
        }
        if observed_c != expected_c:
            c_reasons.setdefault(training_seed, set()).add(
                "c_diag_metric_coverage_mismatch"
            )
    unavailable = {
        seed: ";".join(sorted(reasons))
        for seed, reasons in c_reasons.items()
        if reasons
    }
    return "", indexed, unavailable


def _record_c_reason(
    reasons: dict[int, set[str]],
    training_seed: int,
    expected_training_seeds: Sequence[int],
    reason: str,
) -> None:
    seed = int(training_seed)
    if seed in expected_training_seeds:
        reasons.setdefault(seed, set()).add(reason)
        return
    for expected_seed in expected_training_seeds:
        reasons.setdefault(int(expected_seed), set()).add(reason)


def _average_generation_seeds(
    indexed: Mapping[tuple[str, int, int, str], PriorStudyMetricV2],
    *,
    training_seed: int,
    inners: Sequence[str],
    generation_seeds: Sequence[int],
    arms: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        output[arm] = {}
        for inner in inners:
            cells = [
                indexed[(inner, int(training_seed), int(generation_seed), arm)]
                for generation_seed in generation_seeds
            ]
            output[arm][inner] = {
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


def _metric_hash(rows: Sequence[PriorStudyMetricV2]) -> str:
    payloads = [row.to_payload() for row in rows]
    payloads.sort(
        key=lambda row: (
            str(row["outer_target_center"]),
            str(row["inner_pseudo_target_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            str(row["arm"]),
        )
    )
    return stable_hash(payloads)


def _decision(
    *,
    outer: str,
    status: str,
    e_vs_a_status: str,
    e_vs_c_status: str,
    train_seeds: tuple[int, ...],
    gen_seeds: tuple[int, ...],
    inners: tuple[str, ...],
    summaries: Mapping[str, object],
    protocol_hash: str,
    decision_contract_hash: str,
    metric_hash: str,
    reason: str,
) -> PriorStudyDecisionV2:
    return PriorStudyDecisionV2(
        outer_target_center=outer,
        status=status,
        e_vs_a_consensus_status=e_vs_a_status,
        e_vs_c_consensus_status=e_vs_c_status,
        training_seeds=train_seeds,
        generation_seeds=gen_seeds,
        inner_centers=inners,
        per_training_seed=dict(summaries),
        protocol_hash=protocol_hash,
        decision_contract_hash=decision_contract_hash,
        source_metric_table_hash=metric_hash,
        reason=reason,
    )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / float(len(items))


def _at_least(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold) - 1e-12


def _strictly_greater(value: float, threshold: float) -> bool:
    return float(value) > float(threshold) + 1e-12
