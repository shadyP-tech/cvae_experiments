"""Terminal-only HARP v6 evaluation after the frozen-route seal."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.case_equal_metrics import (
    CASE_CONTRIBUTION_METRIC_NAME,
    PRIMARY_ESTIMAND,
    PRIMARY_METRIC_NAME,
    SINGLE_CLASS_CASE_RULE,
    aggregate_case_equal_metrics,
    case_class_support_counts,
    case_metrics,
)
from ...routing.harp_protocol import canonical_hash
from .contracts import (
    ActionKind,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    TerminalEvaluation,
)


def _case_indices(case_ids: Sequence[str]) -> tuple[tuple[str, np.ndarray], ...]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for ordinal, case in enumerate(case_ids):
        grouped[str(case)].append(ordinal)
    return tuple(
        (case, np.asarray(indices, dtype=np.int64))
        for case, indices in sorted(grouped.items())
    )


def _directional_surface(
    baseline: np.ndarray, action: np.ndarray, direction: str
) -> np.ndarray:
    b = np.ascontiguousarray(baseline, dtype=np.float32)
    a = np.ascontiguousarray(action, dtype=np.float32)
    if b.shape != a.shape or b.ndim != 1:
        raise ProtocolError("HARP v6 terminal action geometry drifted.")
    base_positive = b >= np.float32(0.5)
    action_positive = a >= np.float32(0.5)
    if direction == "D01":
        active = (~base_positive) & action_positive
    elif direction == "D10":
        active = base_positive & (~action_positive)
    elif direction == "ALL_MARGINS":
        active = base_positive == action_positive
    else:
        raise ProtocolError("HARP v6 terminal direction is unknown.")
    output = b.copy()
    output[active] = a[active]
    return output


def _center_metrics(
    cases: Sequence[Mapping[str, object]], role: str
) -> dict[str, float]:
    labels = tuple(np.asarray(row["labels"], dtype=np.int64) for row in cases)
    support = case_class_support_counts(labels)
    return aggregate_case_equal_metrics(
        tuple(
            case_metrics(
                np.asarray(row[role], dtype=np.float64),
                truth,
                total_case_count=len(cases),
                class_support_case_counts=support,
            )
            for row, truth in zip(cases, labels, strict=True)
        )
    )


def evaluate_terminal_routes(
    routes: PrelabelRouteSet,
    evaluation_truth: Mapping[tuple[str, str, str], int],
    *,
    menus: Sequence[LabelFreeOuterMenu],
) -> TerminalEvaluation:
    """Evaluate frozen bytes and construct a non-feeding terminal oracle."""

    if not isinstance(routes, PrelabelRouteSet) or not isinstance(
        evaluation_truth, Mapping
    ):
        raise ProtocolError("HARP v6 terminal evaluation inputs are untyped.")
    truth = {tuple(str(part) for part in key): int(value) for key, value in evaluation_truth.items()}
    expected = {
        (case.outer_target_id, case.case_id, sample)
        for case in routes.cases
        for sample in case.sample_ids
    }
    if set(truth) != expected or any(value not in (0, 1) for value in truth.values()):
        raise ProtocolError("HARP v6 terminal truth does not exactly cover sealed routes.")
    by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    exact_b = True
    for case in routes.cases:
        labels = np.asarray(
            [truth[(case.outer_target_id, case.case_id, sample)] for sample in case.sample_ids],
            dtype=np.int64,
        )
        by_center[case.outer_target_id].append(
            {
                "case_id": case.case_id,
                "labels": labels,
                "baseline": case.baseline_probabilities.astype(np.float64),
                "uniform": case.uniform_probabilities.astype(np.float64),
                "routed": case.routed_probabilities.astype(np.float64),
            }
        )
        reasons[case.reason] += 1
        if case.selected_kind is ActionKind.B:
            exact_b &= (
                case.routed_probabilities.tobytes(order="C")
                == case.baseline_probabilities.tobytes(order="C")
            )
    center_metrics = {
        center: {
            role: _center_metrics(rows, role)
            for role in ("baseline", "uniform", "routed")
        }
        for center, rows in sorted(by_center.items())
    }
    equal_center = {
        role: {
            metric: float(
                np.mean(
                    [center_metrics[center][role][metric] for center in sorted(center_metrics)],
                    dtype=np.float64,
                )
            )
            for metric in (PRIMARY_METRIC_NAME, "brier", "log_loss")
        }
        for role in ("baseline", "uniform", "routed")
    }
    routed_count = sum(case.selected_kind is not ActionKind.B for case in routes.cases)
    metrics = {
        "schema_version": "midogpp_harp_v6_terminal_result_v1",
        "status": "TERMINAL_POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "case_count": len(routes.cases),
        "row_count": sum(len(case.sample_ids) for case in routes.cases),
        "routed_case_count": routed_count,
        "expert_routed_case_count": sum(
            case.selected_kind is ActionKind.HXE for case in routes.cases
        ),
        "case_route_rate": routed_count / len(routes.cases),
        "equal_center_metrics": equal_center,
        "center_metrics": center_metrics,
        "primary_estimand": PRIMARY_ESTIMAND,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "single_class_case_rule": SINGLE_CLASS_CASE_RULE,
        "exact_b_fallback_byte_identity": exact_b,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "compatibility_proxy_is_exact_nelbo": False,
        "compatibility_proxy_is_true_utility": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
    }
    metrics = {**metrics, "result_hash": canonical_hash(metrics)}
    oracle = _terminal_oracle(tuple(menus), truth)
    route_reasons = {
        "schema_version": "midogpp_harp_v6_route_reason_summary_v1",
        "reason_counts": dict(sorted(reasons.items())),
        "selected_action_counts": dict(
            sorted(Counter(case.selected_kind.value for case in routes.cases).items())
        ),
        "exact_b_fallback_byte_identity": exact_b,
    }
    return TerminalEvaluation(metrics, oracle, route_reasons)


def _terminal_oracle(
    menus: tuple[LabelFreeOuterMenu, ...],
    truth: Mapping[tuple[str, str, str], int],
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for menu in menus:
        baseline = menu.target_block(ActionKind.B)
        target_blocks = tuple(
            block for block in menu.blocks if block.surface_role == "target"
        )
        cases = _case_indices(baseline.case_ids)
        labels_by_case = tuple(
            np.asarray(
                [
                    truth[(menu.outer_target_id, case_id, baseline.sample_ids[int(index)])]
                    for index in indices
                ],
                dtype=np.int64,
            )
            for case_id, indices in cases
        )
        support = case_class_support_counts(labels_by_case)
        keywords = {
            "total_case_count": len(cases),
            "class_support_case_counts": support,
        }
        for (case_id, indices), labels in zip(cases, labels_by_case, strict=True):
            b = baseline.probabilities[indices]
            b_score = case_metrics(b, labels, **keywords)
            candidates: list[tuple[str, ActionKind, str | None, object]] = [
                ("B", ActionKind.B, None, b_score)
            ]
            for block in target_blocks:
                if block.action_kind is ActionKind.B:
                    continue
                for direction in ("D01", "D10", "ALL_MARGINS"):
                    values = _directional_surface(
                        b, block.probabilities[indices], direction
                    )
                    candidates.append(
                        (
                            f"{block.action_kind.value}:{block.selected_source_id or ''}:{direction}",
                            block.action_kind,
                            block.selected_source_id,
                            case_metrics(values, labels, **keywords),
                        )
                    )
            best = min(
                candidates,
                key=lambda row: (
                    -row[3].case_equal_bacc_contribution,
                    row[3].brier,
                    row[3].log_loss,
                    row[0],
                ),
            )
            safe = tuple(
                row
                for row in candidates
                if row[3].brier <= b_score.brier and row[3].log_loss <= b_score.log_loss
            )
            safe_best = min(
                safe,
                key=lambda row: (
                    -row[3].case_equal_bacc_contribution,
                    row[3].brier,
                    row[3].log_loss,
                    row[0],
                ),
            )
            rows.append(
                {
                    "outer_target_id": menu.outer_target_id,
                    "case_id": case_id,
                    "raw_oracle_action": best[0],
                    "raw_oracle_kind": best[1].value,
                    "raw_oracle_source": best[2],
                    "raw_oracle_bacc_contribution_gain_vs_B": (
                        best[3].case_equal_bacc_contribution
                        - b_score.case_equal_bacc_contribution
                    ),
                    "proper_loss_safe_oracle_action": safe_best[0],
                    "proper_loss_safe_oracle_bacc_contribution_gain_vs_B": (
                        safe_best[3].case_equal_bacc_contribution
                        - b_score.case_equal_bacc_contribution
                    ),
                }
            )
    centers = tuple(sorted({str(row["outer_target_id"]) for row in rows}))
    center_raw = {
        center: float(
            np.mean(
                [
                    row["raw_oracle_bacc_contribution_gain_vs_B"]
                    for row in rows
                    if row["outer_target_id"] == center
                ],
                dtype=np.float64,
            )
        )
        for center in centers
    }
    center_safe = {
        center: float(
            np.mean(
                [
                    row["proper_loss_safe_oracle_bacc_contribution_gain_vs_B"]
                    for row in rows
                    if row["outer_target_id"] == center
                ],
                dtype=np.float64,
            )
        )
        for center in centers
    }
    body = {
        "schema_version": "midogpp_harp_v6_terminal_directional_oracle_v1",
        "rows": rows,
        "case_count": len(rows),
        "positive_raw_oracle_case_count": sum(
            float(row["raw_oracle_bacc_contribution_gain_vs_B"]) > 0.0
            for row in rows
        ),
        "positive_proper_loss_safe_oracle_case_count": sum(
            float(row["proper_loss_safe_oracle_bacc_contribution_gain_vs_B"]) > 0.0
            for row in rows
        ),
        "equal_center_raw_oracle_gain_vs_B": float(
            np.mean(tuple(center_raw.values()), dtype=np.float64)
        ),
        "equal_center_proper_loss_safe_oracle_gain_vs_B": float(
            np.mean(tuple(center_safe.values()), dtype=np.float64)
        ),
        "center_raw_oracle_gain_vs_B": center_raw,
        "center_proper_loss_safe_oracle_gain_vs_B": center_safe,
        "primary_estimand": PRIMARY_ESTIMAND,
        "case_contribution_metric": CASE_CONTRIBUTION_METRIC_NAME,
        "opened_after_frozen_route_seal": True,
        "may_feed_policy_or_thresholds": False,
        "diagnostic_only": True,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("evaluate_terminal_routes",)
