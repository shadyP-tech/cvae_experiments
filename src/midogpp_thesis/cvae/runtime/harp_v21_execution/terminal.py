"""Terminal-only HARP v21 evaluation after the frozen-route seal."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

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
from ..artifact_io import read_json, sha256_file
from .contracts import (
    ActionKind,
    FrozenRouteReceipt,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    TerminalEvaluation,
)
from .stores import read_prelabel_routes


def _verified_seal(
    value: object, *, field: str, role: str
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"HARP v21 durable {role} is malformed.")
    payload = dict(value)
    observed = payload.pop(field, None)
    if (
        type(observed) is not str
        or len(observed) != 64
        or canonical_hash(payload) != observed
    ):
        raise ProtocolError(f"HARP v21 durable {role} hash drifted.")
    return payload, observed


def _validate_frozen_route_receipt(
    routes: PrelabelRouteSet,
    receipt: FrozenRouteReceipt,
    *,
    menus: Sequence[LabelFreeOuterMenu],
    artifact_root: Path,
    config_hash: object,
) -> None:
    """Reopen and authenticate the registered durable route evidence."""

    if not isinstance(receipt, FrozenRouteReceipt):
        raise ProtocolError("HARP v21 terminal evaluation requires a frozen-route receipt.")
    root = Path(artifact_root).resolve()
    frozen_path = root / "manifests/frozen_route_seal.json"
    validations_path = root / "manifests/fresh_validations.json"
    prelabel_path = root / "manifests/prelabel_route_bundle.json"
    route_root = root / "stores/prelabel_routes"
    try:
        frozen_body, frozen_hash = _verified_seal(
            read_json(frozen_path), field="seal_hash", role="frozen-route seal"
        )
        validation_body, validation_hash = _verified_seal(
            read_json(validations_path),
            field="bundle_hash",
            role="fresh-validation bundle",
        )
        prelabel_body, prelabel_hash = _verified_seal(
            read_json(prelabel_path), field="bundle_hash", role="prelabel bundle"
        )
        disk_routes = read_prelabel_routes(route_root)
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError(
            "HARP v21 terminal evaluation lacks its durable frozen-route store."
        ) from exc
    centers = tuple(sorted({menu.outer_target_id for menu in menus}))
    raw_validations = validation_body.get("validations")
    if not isinstance(raw_validations, list) or len(raw_validations) != 2:
        raise ProtocolError("HARP v21 durable fresh-validation inventory drifted.")
    validation_hashes: list[str] = []
    process_ids: set[object] = set()
    for raw in raw_validations:
        body, observed = _verified_seal(
            raw, field="validation_hash", role="fresh validation"
        )
        validation_hashes.append(observed)
        process_ids.add(body.get("process_id"))
    if (
        type(config_hash) is not str
        or frozen_body.get("schema_version")
        != "midogpp_harp_v21_frozen_route_seal_v1"
        or frozen_body.get("status") != "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        or frozen_body.get("evaluation_labels_opened") is not False
        or frozen_body.get("prelabel_bundle_hash") != prelabel_hash
        or frozen_body.get("validation_bundle_hash") != validation_hash
        or frozen_body.get("independent_validation_hashes") != validation_hashes
        or validation_body.get("distinct_process_ids") is not True
        or validation_body.get("evaluation_labels_opened") is not False
        or len(process_ids) != 2
        or prelabel_body.get("evaluation_labels_opened") is not False
        or prelabel_body.get("route_hash") != disk_routes.route_hash
        or prelabel_body.get("policy_hash") != disk_routes.policy_hash
        or prelabel_body.get("model_hash") != disk_routes.model_hash
        or prelabel_body.get("target_action_hash") != disk_routes.target_action_hash
        or prelabel_body.get("route_store_manifest_sha256")
        != sha256_file(route_root / "manifest.json")
        or prelabel_body.get("route_store_npz_sha256")
        != sha256_file(route_root / "arrays.npz")
        or frozen_body.get("config_hash") != config_hash
        or tuple(frozen_body.get("expected_center_ids", ())) != centers
        or frozen_body.get("route_hash") != disk_routes.route_hash
        or frozen_body.get("policy_hash") != disk_routes.policy_hash
        or frozen_body.get("model_hash") != disk_routes.model_hash
        or frozen_body.get("target_action_hash") != disk_routes.target_action_hash
        or frozen_body.get("case_count") != len(disk_routes.cases)
        or routes.route_hash != disk_routes.route_hash
        or routes.policy_hash != disk_routes.policy_hash
        or routes.model_hash != disk_routes.model_hash
        or routes.target_action_hash != disk_routes.target_action_hash
        or len(routes.cases) != len(disk_routes.cases)
        or receipt.seal_hash != frozen_hash
        or receipt.validation_bundle_hash != validation_hash
        or receipt.independent_validation_hashes != tuple(validation_hashes)
        or receipt.config_hash != config_hash
        or receipt.route_hash != routes.route_hash
        or receipt.policy_hash != routes.policy_hash
        or receipt.model_hash != routes.model_hash
        or receipt.target_action_hash != routes.target_action_hash
        or receipt.expected_center_ids != centers
        or receipt.case_count != len(routes.cases)
    ):
        raise ProtocolError("HARP v21 frozen-route receipt binding drifted.")


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
        raise ProtocolError("HARP v21 terminal action geometry drifted.")
    base_positive = b >= np.float32(0.5)
    action_positive = a >= np.float32(0.5)
    if direction == "D01":
        active = (~base_positive) & action_positive
    elif direction == "D10":
        active = base_positive & (~action_positive)
    else:
        raise ProtocolError("HARP v21 terminal direction is unknown.")
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


def _terminal_case_statuses(
    routes: PrelabelRouteSet,
    truth: Mapping[tuple[str, str, str], int],
) -> tuple[dict[str, object], ...]:
    """Score only the frozen composite and keep four route notions distinct."""

    by_center: dict[str, list[object]] = defaultdict(list)
    for case in routes.cases:
        by_center[case.outer_target_id].append(case)
    output: list[dict[str, object]] = []
    for center, center_cases in sorted(by_center.items()):
        labels_by_case = tuple(
            np.asarray(
                [truth[(center, case.case_id, sample)] for sample in case.sample_ids],
                dtype=np.int64,
            )
            for case in center_cases
        )
        support = case_class_support_counts(labels_by_case)
        for case, labels in zip(center_cases, labels_by_case, strict=True):
            keywords = {
                "total_case_count": len(center_cases),
                "class_support_case_counts": support,
            }
            baseline = case_metrics(
                case.baseline_probabilities.astype(np.float64), labels, **keywords
            )
            routed = case_metrics(
                case.routed_probabilities.astype(np.float64), labels, **keywords
            )
            route_selected = case.selected_kind is not ActionKind.B
            probability_changed = (
                case.routed_probabilities.tobytes(order="C")
                != case.baseline_probabilities.tobytes(order="C")
            )
            prediction_changed = bool(
                np.any(
                    (case.routed_probabilities >= np.float32(0.5))
                    != (case.baseline_probabilities >= np.float32(0.5))
                )
            )
            gain = float(
                routed.case_equal_bacc_contribution
                - baseline.case_equal_bacc_contribution
            )
            brier_delta = float(routed.brier - baseline.brier)
            log_loss_delta = float(routed.log_loss - baseline.log_loss)
            raw_entropy = case.decision_payload.get("donor_entropy", 0.0)
            if (
                isinstance(raw_entropy, bool)
                or type(raw_entropy) not in (int, float)
                or not np.isfinite(float(raw_entropy))
                or float(raw_entropy) < 0.0
            ):
                raise ProtocolError("HARP v21 route donor entropy is malformed.")
            output.append(
                {
                    "center_id": center,
                    "case_id": case.case_id,
                    "route_selected": route_selected,
                    "probability_changed": probability_changed,
                    "prediction_changed": prediction_changed,
                    "utility_success": bool(
                        route_selected
                        and gain > 0.0
                        and brier_delta <= 0.002
                        and log_loss_delta <= 0.005
                    ),
                    "bacc_contribution_gain_vs_B": gain,
                    "brier_delta_vs_B": brier_delta,
                    "log_loss_delta_vs_B": log_loss_delta,
                    "donor_entropy": float(raw_entropy),
                }
            )
    return tuple(output)


def evaluate_terminal_routes(
    routes: PrelabelRouteSet,
    evaluation_truth: Mapping[tuple[str, str, str], int],
    *,
    menus: Sequence[LabelFreeOuterMenu],
    frozen_receipt: FrozenRouteReceipt,
    artifact_root: Path,
    config_hash: str,
) -> TerminalEvaluation:
    """Evaluate frozen bytes and construct a non-feeding terminal oracle."""

    if not isinstance(routes, PrelabelRouteSet) or not isinstance(
        evaluation_truth, Mapping
    ):
        raise ProtocolError("HARP v21 terminal evaluation inputs are untyped.")
    _validate_frozen_route_receipt(
        routes,
        frozen_receipt,
        menus=menus,
        artifact_root=artifact_root,
        config_hash=config_hash,
    )
    truth = {tuple(str(part) for part in key): int(value) for key, value in evaluation_truth.items()}
    expected = {
        (case.outer_target_id, case.case_id, sample)
        for case in routes.cases
        for sample in case.sample_ids
    }
    if set(truth) != expected or any(value not in (0, 1) for value in truth.values()):
        raise ProtocolError("HARP v21 terminal truth does not exactly cover sealed routes.")
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
    status_rows = _terminal_case_statuses(routes, truth)
    routed_count = sum(bool(row["route_selected"]) for row in status_rows)
    probability_changed_count = sum(
        bool(row["probability_changed"]) for row in status_rows
    )
    prediction_changed_count = sum(
        bool(row["prediction_changed"]) for row in status_rows
    )
    utility_success_count = sum(bool(row["utility_success"]) for row in status_rows)
    selected_entropies = tuple(
        float(row["donor_entropy"])
        for row in status_rows
        if bool(row["route_selected"])
    )
    center_status = {
        center: {
            "case_count": len(center_rows),
            "route_selected_count": sum(
                bool(row["route_selected"]) for row in center_rows
            ),
            "probability_changed_count": sum(
                bool(row["probability_changed"]) for row in center_rows
            ),
            "prediction_changed_count": sum(
                bool(row["prediction_changed"]) for row in center_rows
            ),
            "utility_success_count": sum(
                bool(row["utility_success"]) for row in center_rows
            ),
        }
        for center in sorted(by_center)
        for center_rows in (
            tuple(row for row in status_rows if row["center_id"] == center),
        )
    }
    metrics = {
        "schema_version": "midogpp_harp_v21_terminal_result_v1",
        "status": "TERMINAL_POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "case_count": len(routes.cases),
        "row_count": sum(len(case.sample_ids) for case in routes.cases),
        "routed_case_count": routed_count,
        "soft_topk_routed_case_count": sum(
            case.selected_kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND
            for case in routes.cases
        ),
        "exact_u_full_routed_case_count": sum(
            case.selected_kind is ActionKind.U for case in routes.cases
        ),
        "case_route_rate": routed_count / len(routes.cases),
        "selection_status": "TERMINALLY_EVALUATED_FROZEN_SOURCE_POLICY",
        "route_selected_count": routed_count,
        "probability_status": "TERMINALLY_EVALUATED_FROZEN_BYTES",
        "probability_changed_count": probability_changed_count,
        "prediction_status": "TERMINALLY_EVALUATED_FROZEN_THRESHOLD",
        "prediction_changed_count": prediction_changed_count,
        "utility_status": "TERMINALLY_EVALUATED_CONSUMED_TEST_DIAGNOSTIC",
        "utility_success_count": utility_success_count,
        "utility_success_rate": utility_success_count / len(status_rows),
        "mean_selected_donor_entropy": (
            0.0
            if not selected_entropies
            else float(np.mean(selected_entropies, dtype=np.float64))
        ),
        "case_status_rows": list(status_rows),
        "center_status_counts": center_status,
        "equal_center_metrics": equal_center,
        "center_metrics": center_metrics,
        "primary_estimand": PRIMARY_ESTIMAND,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "single_class_case_rule": SINGLE_CLASS_CASE_RULE,
        "exact_b_fallback_byte_identity": exact_b,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "deployed_route_kind": "POOLED_POLICY_SOFT_TOPK_PROBABILITY_BLEND_OR_EXACT_U_FULL_OR_EXACT_B",
        "soft_topk_probability_mixture_used": any(
            case.selected_kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND
            for case in routes.cases
        ),
        "all_k_lambda_probability_matrices_persisted": False,
        "compatibility_proxy_is_exact_nelbo": False,
        "compatibility_proxy_is_true_utility": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
    }
    metrics = {**metrics, "result_hash": canonical_hash(metrics)}
    oracle = _terminal_oracle(tuple(menus), truth)
    route_reasons = {
        "schema_version": "midogpp_harp_v21_route_reason_summary_v1",
        "reason_counts": dict(sorted(reasons.items())),
        "selected_action_counts": dict(
            sorted(Counter(case.selected_kind.value for case in routes.cases).items())
        ),
        "route_selected_count": routed_count,
        "probability_changed_count": probability_changed_count,
        "prediction_changed_count": prediction_changed_count,
        "utility_success_count": utility_success_count,
        "mean_selected_donor_entropy": metrics["mean_selected_donor_entropy"],
        "route_selected_is_probability_changed": False,
        "probability_changed_is_prediction_changed": False,
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
                if block.action_kind is ActionKind.U:
                    candidates.append(
                        (
                            "U:FULL",
                            ActionKind.U,
                            None,
                            case_metrics(block.probabilities[indices], labels, **keywords),
                        )
                    )
                    continue
                for direction in ("D01", "D10"):
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
        "schema_version": "midogpp_harp_v21_terminal_component_oracle_v3",
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
        "component_action_families": ["U:FULL", "HXE:D01", "HXE:D10"],
        "soft_k_lambda_lattice_materialized": False,
        "all_margins_excluded": True,
        "may_feed_policy_or_thresholds": False,
        "diagnostic_only": True,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("evaluate_terminal_routes",)
