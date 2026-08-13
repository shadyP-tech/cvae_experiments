"""Terminal-only categorical scoring, oracles, contrasts, and block-null summary."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_npz, sha256_array, sha256_file
from .artifact_io import persist_json
from .constants import (
    B_ACTION_ID,
    CENTERS,
    METHOD_IDS,
    OOF_FOLD_COUNT,
    PERMUTATION_COUNT,
    U_ACTION_ID,
    decision_action_ids,
)
from .hashing import canonical_hash
from .scoring import pooled_bacc


NULL_ARRAY_MEMBER = "arrays/action_identity_null_selections.npz"
_T8_TWO_SIDED_95 = 2.306004135204166


def evaluate_terminal(
    *,
    root: Path,
    partition: object,
    decision_seal: object,
    null_plans: Sequence[object],
    evaluation_counts: Sequence[object],
) -> Mapping[str, object]:
    """Evaluate only after every route and null plan is already sealed."""

    decisions = tuple(getattr(decision_seal, "decisions"))
    plans = tuple(null_plans)
    counts = tuple(evaluation_counts)
    route_keys = tuple((center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT))
    if (
        tuple(decision.route_key for decision in decisions) != route_keys
        or tuple((plan.target_center, plan.fold_ordinal) for plan in plans) != route_keys
        or any(len(plan.selections) != PERMUTATION_COUNT for plan in plans)
    ):
        raise ProtocolError("S4 terminal evaluation requires 45 sealed routes and null plans.")
    by_count = {(row.target_center, row.case_id, row.action_id): row for row in counts}
    expected_count_keys = {
        (identity.target_center, identity.case_id, action)
        for identity in getattr(partition, "identities")
        for action in (B_ACTION_ID, U_ACTION_ID, *decision_action_ids(identity.target_center)[1:])
    }
    if len(by_count) != len(counts) or set(by_count) != expected_count_keys:
        raise ProtocolError("S4 terminal evaluation count surface is incomplete or duplicated.")

    decision_by_route = {decision.route_key: decision for decision in decisions}
    method_rows: dict[tuple[str, str], list[object]] = {
        (target, method): [] for target in CENTERS for method in METHOD_IDS
    }
    method_decisions: list[dict[str, object]] = []
    case_confusions: list[dict[str, object]] = []
    center_oracle_static: dict[str, str] = {}
    center_oracle_case: dict[tuple[str, str], str] = {}

    for target in CENTERS:
        center_cases = tuple(
            sorted(
                {
                    identity.case_id
                    for identity in getattr(partition, "identities")
                    if identity.target_center == target
                }
            )
        )
        action_order = decision_action_ids(target)
        static_scores = {
            action: pooled_bacc(
                tuple(by_count[(target, case_id, action)] for case_id in center_cases)
            ).exact_bacc
            for action in action_order
        }
        center_oracle_static[target] = _first_max(action_order, static_scores)
        total_positive = sum(
            by_count[(target, case_id, B_ACTION_ID)].n_positive
            for case_id in center_cases
        )
        total_negative = sum(
            by_count[(target, case_id, B_ACTION_ID)].n_negative
            for case_id in center_cases
        )
        if total_positive <= 0 or total_negative <= 0:
            raise ProtocolError("S4 terminal center lacks one pooled class.")
        for case_id in center_cases:
            utilities = {
                action: (
                    0.5
                    * by_count[(target, case_id, action)].true_positive
                    / total_positive
                    + 0.5
                    * by_count[(target, case_id, action)].true_negative
                    / total_negative
                )
                for action in action_order
            }
            center_oracle_case[(target, case_id)] = _first_max(action_order, utilities)

        for case_id in center_cases:
            fold = getattr(partition, "evaluation_fold_for_case")(target, case_id)
            route = decision_by_route[(target, fold.fold_ordinal)]
            actions = {
                "B": B_ACTION_ID,
                "U": U_ACTION_ID,
                "G_static": route.g_static.action_id,
                "S4": route.s4.action_id,
                "O_static": center_oracle_static[target],
                "O_case": center_oracle_case[(target, case_id)],
            }
            for method in METHOD_IDS:
                action = actions[method]
                row = by_count[(target, case_id, action)]
                method_rows[(target, method)].append(row)
                decision_payload = {
                    "target_center": target,
                    "fold_ordinal": fold.fold_ordinal,
                    "case_id": case_id,
                    "method_id": method,
                    "action_id": action,
                    "route_decision_hash": route.route_decision_hash,
                    "evaluation_labels_used_for_decision": method
                    in {"O_static", "O_case"},
                }
                method_decisions.append(
                    {**decision_payload, "row_hash": canonical_hash(decision_payload)}
                )
                confusion_payload = {
                    "target_center": target,
                    "fold_ordinal": fold.fold_ordinal,
                    "case_id": case_id,
                    "method_id": method,
                    "action_id": action,
                    "n_positive": row.n_positive,
                    "true_positive": row.true_positive,
                    "n_negative": row.n_negative,
                    "true_negative": row.true_negative,
                }
                case_confusions.append(
                    {**confusion_payload, "row_hash": canonical_hash(confusion_payload)}
                )

    center_metrics = []
    score_by_center_method: dict[tuple[str, str], float] = {}
    for target in CENTERS:
        for method in METHOD_IDS:
            score = pooled_bacc(
                tuple(method_rows[(target, method)]), action_or_method_id=method
            )
            payload = {
                "target_center": target,
                "method_id": method,
                "case_count": score.case_count,
                "n_positive": score.n_positive,
                "true_positive": score.true_positive,
                "n_negative": score.n_negative,
                "true_negative": score.true_negative,
                "sensitivity": score.sensitivity,
                "specificity": score.specificity,
                "exact_bacc": score.exact_bacc,
            }
            center_metrics.append({**payload, "row_hash": canonical_hash(payload)})
            score_by_center_method[(target, method)] = score.exact_bacc

    contrasts = _descriptive_contrasts(score_by_center_method)
    null_plan_seal, null_matrix = load_null_selection_plan_seal(
        root,
        plans=plans,
        decision_seal_hash=decision_seal.decision_seal_hash,
        partition_hash=partition.partition_hash,
    )
    null_path = root / NULL_ARRAY_MEMBER
    null_counts = _null_count_rows(plans, null_matrix)
    null_statistics = _evaluate_null_statistics(
        partition=partition,
        counts=by_count,
        matrix=null_matrix,
        observed=(
            float(
                np.mean(
                    [score_by_center_method[(target, "S4")] for target in CENTERS],
                    dtype=np.float64,
                )
            )
            - float(
                np.mean(
                    [score_by_center_method[(target, "B")] for target in CENTERS],
                    dtype=np.float64,
                )
            )
        ),
    )
    null_seal_unhashed = {
        "schema_version": "fixed_bank_support_static_router_action_identity_null_seal_v1",
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "partition_hash": partition.partition_hash,
        "null_selection_plan_seal_hash": null_plan_seal[
            "null_selection_plan_seal_hash"
        ],
        "route_plan_hashes": [plan.plan_hash for plan in plans],
        "route_plan_count": len(plans),
        "replicate_count": PERMUTATION_COUNT,
        "selection_matrix_shape": list(null_matrix.shape),
        "selection_matrix_dtype": str(null_matrix.dtype),
        "selection_matrix_sha256": sha256_array(null_matrix),
        "selection_array_file_sha256": sha256_file(null_path),
        "complete_A1_blocks_shifted_within_support_case": True,
        "B_fixed": True,
        "evaluation_labels_used_for_null_selection": False,
        "every_route_null_plan_sealed_before_own_evaluation_labels": True,
        "exchangeability_claimed": False,
        "confirmatory_p_value": False,
        "pass_gate_used": False,
    }
    null_seal = {
        **null_seal_unhashed,
        "null_seal_hash": canonical_hash(null_seal_unhashed),
    }
    null_summary = {
        "schema_version": "fixed_bank_support_static_router_action_identity_null_summary_v1",
        **null_statistics,
        "null_seal_hash": null_seal["null_seal_hash"],
        "statistic": "equal_center_mean_S4_minus_B",
        "descriptive_only": True,
        "exchangeability_claimed": False,
        "confirmatory_p_value": False,
        "pass_gate_used": False,
        "routing_success_claimed": False,
    }
    table_payloads = {
        "method_decisions": tuple(method_decisions),
        "terminal_case_confusions": tuple(case_confusions),
        "terminal_center_metrics": tuple(center_metrics),
        "terminal_contrasts": tuple(contrasts),
        "null_route_selection_counts": tuple(null_counts),
    }
    table_hashes = {
        name: canonical_hash(list(rows)) for name, rows in table_payloads.items()
    }
    terminal_unhashed = {
        "schema_version": "fixed_bank_support_static_router_terminal_seal_v1",
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "null_seal_hash": null_seal["null_seal_hash"],
        "table_hashes": table_hashes,
        "method_decision_row_count": len(method_decisions),
        "case_confusion_row_count": len(case_confusions),
        "center_metric_row_count": len(center_metrics),
        "contrast_row_count": len(contrasts),
        "null_count_row_count": len(null_counts),
        "terminal_oracles_used_for_pre_evaluation_decisions": False,
        "all_route_decisions_and_null_plans_sealed_before_own_evaluation_labels": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
    }
    terminal_seal = {
        **terminal_unhashed,
        "sealed_result_hash": canonical_hash(terminal_unhashed),
    }
    return {
        **table_payloads,
        "action_identity_null_summary": null_summary,
        "action_identity_null_seal": null_seal,
        "sealed_terminal_evaluation": terminal_seal,
    }


def _first_max(order: Sequence[str], values: Mapping[str, float]) -> str:
    maximum = max(values.values())
    return next(action for action in order if maximum - values[action] <= 1.0e-12)


def _descriptive_contrasts(
    scores: Mapping[tuple[str, str], float]
) -> tuple[dict[str, object], ...]:
    result = []
    for contrast_id in (
        "S4-B",
        "S4-U",
        "S4-G_static",
        "O_static-S4",
        "O_case-O_static",
    ):
        method, baseline = contrast_id.split("-", maxsplit=1)
        values = np.asarray(
            [scores[(target, method)] - scores[(target, baseline)] for target in CENTERS],
            dtype=np.float64,
        )
        estimate = float(np.mean(values, dtype=np.float64))
        se = float(np.std(values, ddof=1, dtype=np.float64) / math.sqrt(len(values)))
        payload = {
            "contrast_id": contrast_id,
            "method_id": method,
            "baseline_id": baseline,
            "estimate": estimate,
            "ci_low": estimate - _T8_TWO_SIDED_95 * se,
            "ci_high": estimate + _T8_TWO_SIDED_95 * se,
            "center_estimates": [float(value) for value in values],
            "outer_n": len(values),
            "outer_df": len(values) - 1,
            "descriptive_only": True,
            "confirmatory_p_value": False,
            "pass_gate_used": False,
        }
        result.append({**payload, "row_hash": canonical_hash(payload)})
    return tuple(result)


def _null_action_matrix(plans: Sequence[object]) -> np.ndarray:
    values = np.empty((PERMUTATION_COUNT, len(plans)), dtype=np.uint8)
    for route_ordinal, plan in enumerate(plans):
        order = decision_action_ids(plan.target_center)
        values[:, route_ordinal] = np.asarray(
            [order.index(selection.action_id) for selection in plan.selections],
            dtype=np.uint8,
        )
    return np.ascontiguousarray(values)


def seal_null_selection_plans(
    root: Path,
    *,
    plans: Sequence[object],
    decision_seal_hash: str,
    partition_hash: str,
) -> tuple[Mapping[str, object], np.ndarray]:
    """Durably seal all 450k categorical selections before label evaluation."""

    values = tuple(plans)
    expected = tuple((center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT))
    if (
        tuple((plan.target_center, plan.fold_ordinal) for plan in values) != expected
        or any(plan.permutation_count != PERMUTATION_COUNT for plan in values)
    ):
        raise ProtocolError("S4 null selection plan coverage drifted.")
    matrix = _null_action_matrix(values)
    path = root / NULL_ARRAY_MEMBER
    _persist_or_validate_null_array(path, matrix)
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_null_selection_plan_seal_v1",
        "decision_seal_hash": decision_seal_hash,
        "partition_hash": partition_hash,
        "route_plan_hashes": [plan.plan_hash for plan in values],
        "route_plan_count": len(values),
        "replicate_count": PERMUTATION_COUNT,
        "selection_matrix_shape": list(matrix.shape),
        "selection_matrix_dtype": str(matrix.dtype),
        "selection_matrix_sha256": sha256_array(matrix),
        "selection_array_file_sha256": sha256_file(path),
        "sealed_before_any_route_evaluation_labels": True,
        "evaluation_labels_used": False,
        "complete_candidate_A1_blocks_permuted": True,
        "B_fixed": True,
        "exchangeability_claimed": False,
        "confirmatory_p_value": False,
        "pass_gate_used": False,
    }
    payload = {
        **unhashed,
        "null_selection_plan_seal_hash": canonical_hash(unhashed),
    }
    persist_json(
        root / "manifests/action_identity_null_selection_plan_seal.json", payload
    )
    return payload, matrix


def load_null_selection_plan_seal(
    root: Path,
    *,
    plans: Sequence[object],
    decision_seal_hash: str,
    partition_hash: str,
) -> tuple[Mapping[str, object], np.ndarray]:
    """Load and exactly rebind the required pre-evaluation null seal."""

    from ...runtime.artifact_io import read_json

    path = root / "manifests/action_identity_null_selection_plan_seal.json"
    array_path = root / NULL_ARRAY_MEMBER
    if path.is_symlink() or array_path.is_symlink() or not path.is_file() or not array_path.is_file():
        raise ProtocolError("S4 terminal evaluation requires the pre-evaluation null seal.")
    payload = read_json(path)
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "null_selection_plan_seal_hash"
    }
    with np.load(array_path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("selected_action_index",):
            raise ProtocolError("S4 pre-evaluation null array members drifted.")
        matrix = np.asarray(archive["selected_action_index"])
    values = tuple(plans)
    if (
        payload.get("null_selection_plan_seal_hash") != canonical_hash(unhashed)
        or payload.get("decision_seal_hash") != decision_seal_hash
        or payload.get("partition_hash") != partition_hash
        or payload.get("route_plan_hashes") != [plan.plan_hash for plan in values]
        or payload.get("route_plan_count") != 45
        or payload.get("replicate_count") != PERMUTATION_COUNT
        or payload.get("selection_matrix_shape")
        != [PERMUTATION_COUNT, 45]
        or payload.get("selection_matrix_dtype") != "uint8"
        or matrix.shape != (PERMUTATION_COUNT, 45)
        or matrix.dtype != np.uint8
        or payload.get("selection_matrix_sha256") != sha256_array(matrix)
        or payload.get("selection_array_file_sha256") != sha256_file(array_path)
        or payload.get("sealed_before_any_route_evaluation_labels") is not True
        or payload.get("evaluation_labels_used") is not False
        or payload.get("exchangeability_claimed") is not False
        or payload.get("confirmatory_p_value") is not False
        or payload.get("pass_gate_used") is not False
    ):
        raise ProtocolError("S4 pre-evaluation null selection seal drifted.")
    rebuilt = _null_action_matrix(values)
    if sha256_array(matrix) != sha256_array(rebuilt):
        raise ProtocolError("S4 null selection matrix is not reconstructive.")
    return payload, np.ascontiguousarray(matrix)


def _null_count_rows(
    plans: Sequence[object], matrix: np.ndarray
) -> tuple[dict[str, object], ...]:
    rows = []
    for route_ordinal, plan in enumerate(plans):
        order = decision_action_ids(plan.target_center)
        for action_index, action_id in enumerate(order):
            rows.append(
                {
                    "target_center": plan.target_center,
                    "fold_ordinal": plan.fold_ordinal,
                    "action_id": action_id,
                    "selection_count": int(np.count_nonzero(matrix[:, route_ordinal] == action_index)),
                    "replicate_count": PERMUTATION_COUNT,
                    "route_null_selection_hash": plan.plan_hash,
                }
            )
    return tuple(rows)


def _evaluate_null_statistics(
    *,
    partition: object,
    counts: Mapping[tuple[str, str, str], object],
    matrix: np.ndarray,
    observed: float,
) -> dict[str, object]:
    # [route, action, (tp, tn, npos, nneg)] keeps the 10k replay in compact
    # integer space.  The only floating work is the predeclared float64 BACC
    # reduction after categorical selections have been indexed.
    route_totals = np.zeros((45, 9, 4), dtype=np.int64)
    for route_ordinal, fold in enumerate(partition.folds):
        for action_ordinal, action in enumerate(decision_action_ids(fold.target_center)):
            rows = tuple(
                counts[(fold.target_center, case_id, action)]
                for case_id in fold.evaluation_case_ids
            )
            route_totals[route_ordinal, action_ordinal] = (
                sum(row.true_positive for row in rows),
                sum(row.true_negative for row in rows),
                sum(row.n_positive for row in rows),
                sum(row.n_negative for row in rows),
            )
    selected = route_totals[
        np.arange(45, dtype=np.int64)[None, :], matrix.astype(np.int64, copy=False)
    ]
    per_center = selected.reshape(PERMUTATION_COUNT, len(CENTERS), OOF_FOLD_COUNT, 4).sum(
        axis=2, dtype=np.int64
    )
    if np.any(per_center[:, :, 2:] <= 0):
        raise ProtocolError("S4 null replay produced a single-class center aggregate.")
    center_bacc = 0.5 * (
        per_center[:, :, 0].astype(np.float64) / per_center[:, :, 2]
        + per_center[:, :, 1].astype(np.float64) / per_center[:, :, 3]
    )
    baseline_center = route_totals[:, 0].reshape(len(CENTERS), OOF_FOLD_COUNT, 4).sum(
        axis=1, dtype=np.int64
    )
    baseline_bacc = 0.5 * (
        baseline_center[:, 0].astype(np.float64) / baseline_center[:, 2]
        + baseline_center[:, 1].astype(np.float64) / baseline_center[:, 3]
    )
    null = np.mean(center_bacc, axis=1, dtype=np.float64) - float(
        np.mean(baseline_bacc, dtype=np.float64)
    )
    exceedance = int(np.count_nonzero(null >= observed))
    return {
        "observed_statistic": float(observed),
        "null_replicate_count": PERMUTATION_COUNT,
        "null_mean": float(np.mean(null, dtype=np.float64)),
        "null_quantile_0_025": float(np.quantile(null, 0.025)),
        "null_quantile_0_5": float(np.quantile(null, 0.5)),
        "null_quantile_0_975": float(np.quantile(null, 0.975)),
        "exceedance_rule": "null_statistic_greater_than_or_equal_to_observed",
        "exceedance_count": exceedance,
        "exceedance_fraction": float(exceedance / PERMUTATION_COUNT),
        "null_statistics_sha256": sha256_array(null),
    }


def _persist_or_validate_null_array(path: Path, values: np.ndarray) -> None:
    """Never rewrite a durable NPZ during terminal-checkpoint recovery."""

    if path.is_symlink():
        raise ProtocolError("S4 null selection array path is a symlink.")
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if tuple(archive.files) != ("selected_action_index",):
                    raise ProtocolError("S4 null NPZ members drifted.")
                observed = np.asarray(archive["selected_action_index"])
        except (OSError, ValueError) as exc:
            raise ProtocolError("S4 null selection NPZ is unreadable.") from exc
        if (
            observed.shape != (PERMUTATION_COUNT, 45)
            or observed.dtype != np.uint8
            or sha256_array(observed) != sha256_array(values)
        ):
            raise ProtocolError("Existing S4 null selection array differs; refusing repair.")
        return
    atomic_npz(path, selected_action_index=values)
    with np.load(path, allow_pickle=False) as archive:
        observed = np.asarray(archive["selected_action_index"])
    if sha256_array(observed) != sha256_array(values):
        raise ProtocolError("S4 null selection array changed during atomic write.")


__all__ = (
    "NULL_ARRAY_MEMBER",
    "evaluate_terminal",
    "load_null_selection_plan_seal",
    "seal_null_selection_plans",
)
