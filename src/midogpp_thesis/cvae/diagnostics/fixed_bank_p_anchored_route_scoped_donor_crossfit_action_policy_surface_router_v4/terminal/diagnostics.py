"""Terminal mechanism diagnostics for the already-frozen P-DCAPS router."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    TerminalLabelCapability,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    SealedActionSurfaceSet,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.method_controls import (
    ComposedAdmissionControlledPrediction,
)
from ..identity import (
    ACTION_STRATA,
    METHOD_MENU,
    POLICY_ONLY_METHOD_ID,
    PRIMARY_METHOD_ID,
    P_METHOD_ID,
)


def midrank_spearman(
    expected: Sequence[float], realized: Sequence[float]
) -> float | None:
    """Pearson correlation of deterministic average ranks, or None if undefined."""

    x = np.asarray(expected, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        return None
    rx, ry = _midranks(x), _midranks(y)
    if np.ptp(rx) <= 0.0 or np.ptp(ry) <= 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def build_router_diagnostics(
    *,
    identity_results: Sequence[OuterActionPolicyResult],
    surface_set: SealedActionSurfaceSet,
    compositions: Sequence[ComposedAdmissionControlledPrediction],
    capabilities: Sequence[TerminalLabelCapability],
    center_rows: Sequence[Mapping[str, object]],
    case_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate action and policy signal without changing a frozen decision."""

    results = {row.outer_center: row for row in identity_results}
    capability_by_center = {row.center: row for row in capabilities}
    composition_by_key = {
        (row.decision.outer_center, row.decision.method_id): row
        for row in compositions
    }
    center_by_key = {
        (str(row["target_center"]), str(row["method_id"])): row
        for row in center_rows
    }
    if (
        tuple(results) != CENTERS
        or tuple(capability_by_center) != CENTERS
        or len(composition_by_key) != len(CENTERS) * len(METHOD_MENU)
    ):
        raise ProtocolError("P-DCAPS terminal diagnostic inventory drifted.")

    action_pairs: dict[tuple[str, str], list[tuple[float, float]]] = {
        stratum: [] for stratum in ACTION_STRATA
    }
    frequency: dict[tuple[str, str], int] = defaultdict(int)
    oracle_gaps: list[float] = []
    for center in CENTERS:
        truth = {
            row.sample_id: int(row.value)
            for row in capability_by_center[center].rows
        }
        positive = sum(value == 1 for value in truth.values())
        negative = sum(value == 0 for value in truth.values())
        result = results[center]
        route_by_key = {
            row.route_key: row
            for row in surface_set.identity.routes
            if row.route_key.outer_center == center
            and row.route_key.surface_role == "target"
        }
        primary = composition_by_key[(center, PRIMARY_METHOD_ID)].prediction
        primary_probability = dict(
            zip(
                primary.sample_ids,
                (float(value) for value in primary.probabilities),
                strict=True,
            )
        )
        for decision in result.target_action_decisions:
            selected = decision.selection.selected_action_key
            route = route_by_key[decision.route_key]
            y = np.asarray([truth[value] for value in route.sample_ids], dtype=np.int8)
            baseline = route.baseline_probabilities.astype(np.float64)
            if selected is not None:
                action = route.probability_for_action_key(
                    selected.action_key_hash
                ).astype(np.float64)
                realized = _case_bacc_contribution(
                    action, y, positive=positive, negative=negative
                ) - _case_bacc_contribution(
                    baseline, y, positive=positive, negative=negative
                )
                action_pairs[selected.stratum].append(
                    (decision.selection.selected_utility.bacc_gain, realized)
                )
            candidates = [baseline] + [
                cell.action_probabilities.astype(np.float64) for cell in route.cells
            ]
            utilities = [
                _case_bacc_contribution(
                    values, y, positive=positive, negative=negative
                )
                for values in candidates
            ]
            chosen = np.asarray(
                [primary_probability[value] for value in route.sample_ids]
            )
            realized = _case_bacc_contribution(
                chosen, y, positive=positive, negative=negative
            )
            best, worst = max(utilities), min(utilities)
            if best - worst > 1.0e-15:
                oracle_gaps.append((best - realized) / (best - worst))

        primary_decision = composition_by_key[(center, PRIMARY_METHOD_ID)].decision
        if primary_decision.composition_selection_enabled:
            action_by_hash = {
                cell.prediction.key.action_key_hash: cell.prediction.key
                for route in route_by_key.values()
                for cell in route.cells
            }
            for action_hash in primary_decision.selected_action_hashes:
                frequency[action_by_hash[action_hash].stratum] += 1

    action_correlations = []
    for stratum in ACTION_STRATA:
        pairs = action_pairs[stratum]
        action_correlations.append(
            {
                "family": stratum[0],
                "direction": stratum[1],
                "pair_count": len(pairs),
                "midrank_spearman": midrank_spearman(
                    [row[0] for row in pairs], [row[1] for row in pairs]
                ),
            }
        )

    policy_expected = [
        results[center]
        .target_policy_selection.selected_cell.corrected_utility.bacc_gain
        for center in CENTERS
    ]
    policy_realized = [
        float(center_by_key[(center, POLICY_ONLY_METHOD_ID)]["center_bacc_delta_vs_P"])
        for center in CENTERS
    ]
    routed_centers = [
        center
        for center in CENTERS
        if composition_by_key[
            (center, PRIMARY_METHOD_ID)
        ].decision.composition_selection_enabled
    ]
    jointly_safe = [
        center
        for center in routed_centers
        if float(center_by_key[(center, PRIMARY_METHOD_ID)]["center_bacc_delta_vs_P"])
        > 0.0
        and float(center_by_key[(center, PRIMARY_METHOD_ID)]["center_brier_delta_vs_P"])
        <= 0.0
        and float(center_by_key[(center, PRIMARY_METHOD_ID)]["center_log_loss_delta_vs_P"])
        <= 0.0
    ]
    primary_cases = [
        row for row in case_rows if row["method_id"] == PRIMARY_METHOD_ID
    ]
    return {
        "schema_version": "pdcaps_v4_terminal_router_diagnostics_v1",
        "action_expected_vs_realized_midrank_spearman_by_stratum": (
            action_correlations
        ),
        "policy_expected_vs_realized_midrank_spearman": midrank_spearman(
            policy_expected, policy_realized
        ),
        "policy_expected_vs_realized_pair_count": len(CENTERS),
        "routed_center_count": len(routed_centers),
        "joint_safe_routed_center_count": len(jointly_safe),
        "joint_safe_routed_policy_rate": (
            None if not routed_centers else len(jointly_safe) / len(routed_centers)
        ),
        "normalized_endpoint_oracle_gap_definition": (
            "best_sealed_case_action_minus_primary_over_best_minus_worst_"
            "sealed_case_action"
        ),
        "normalized_endpoint_oracle_gap_defined_case_count": len(oracle_gaps),
        "mean_normalized_endpoint_oracle_gap": (
            None
            if not oracle_gaps
            else float(np.mean(oracle_gaps, dtype=np.float64))
        ),
        "primary_case_harm_count": sum(
            int(bool(row["case_harmed_vs_P"])) for row in primary_cases
        ),
        "primary_case_harm_rate": (
            sum(int(bool(row["case_harmed_vs_P"])) for row in primary_cases)
            / len(primary_cases)
        ),
        "center_action_frequencies": [
            {
                "family": family,
                "direction": direction,
                "selected_count": frequency[(family, direction)],
            }
            for family, direction in ACTION_STRATA
        ],
        "terminal_labels_changed_preterminal_decisions": False,
        "nonzero_route_count_is_not_success": True,
        "descriptive_only": True,
        "formal_claim_authorized": False,
    }


def _case_bacc_contribution(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    positive: int,
    negative: int,
) -> float:
    hard = np.asarray(probability) >= 0.5
    return 0.5 * (
        float(np.sum((truth == 1) & hard, dtype=np.int64)) / positive
        + float(np.sum((truth == 0) & (~hard), dtype=np.int64)) / negative
    )


__all__ = ("build_router_diagnostics", "midrank_spearman")
