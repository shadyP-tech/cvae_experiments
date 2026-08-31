"""Post-route-seal diagnostics for the terminal consumed-test sensitivity.

This module accepts truth only after the durable route barrier.  It has no
callback, model, policy, threshold, or artifact-registration edge and therefore
cannot alter routing.  Its purpose is to separate budget lift (U-B), physical
expert lift (Hxe at lambda=1 versus U), predictive blending lift, and the final
operational policy effect versus exact B.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import LAMBDA_GRID
from ...routing.harp_protocol import canonical_hash
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
    HarpRoutedVectorSeal,
)
from ...runtime.harp_probability_menu.hashing import require_sha256
from ...runtime.harp_probability_menu.indexed import validated_target_menu_view


_EPSILON = 1.0e-7
_TIE_TOLERANCE = 1.0e-12


def _case_equal_mean(values: np.ndarray, cases: np.ndarray) -> float:
    unique = tuple(sorted(set(str(value) for value in cases.tolist())))
    if not unique:
        raise ProtocolError("HARP terminal diagnostics contain no cases.")
    return float(
        np.mean(
            [
                float(np.mean(values[cases == case], dtype=np.float64))
                for case in unique
            ],
            dtype=np.float64,
        )
    )


def _case_equal_bacc(
    truth: np.ndarray, probability: np.ndarray, cases: np.ndarray
) -> float:
    prediction = probability >= 0.5
    if set(int(value) for value in truth.tolist()) != {0, 1}:
        raise ProtocolError("Every HARP target center must contain both truth classes.")
    recalls: list[float] = []
    for label in (0, 1):
        supported = tuple(sorted(set(str(value) for value in cases[truth == label])))
        if not supported:
            raise ProtocolError("HARP terminal class/case support is empty.")
        recalls.append(
            float(
                np.mean(
                    [
                        float(
                            np.mean(
                                prediction[(truth == label) & (cases == case)]
                                == bool(label)
                            )
                        )
                        for case in supported
                    ],
                    dtype=np.float64,
                )
            )
        )
    return 0.5 * (recalls[0] + recalls[1])


def _loss(truth: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, _EPSILON, 1.0 - _EPSILON)
    return -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))


def _metric(
    truth: np.ndarray, probability: np.ndarray, cases: np.ndarray
) -> tuple[float, float, float]:
    return (
        _case_equal_bacc(truth, probability, cases),
        _case_equal_mean((probability - truth) ** 2, cases),
        _case_equal_mean(_loss(truth, probability), cases),
    )


def build_terminal_action_diagnostics(
    menu: HarpPredictionMenuSeal,
    vectors: Sequence[HarpRoutedVectorSeal],
    physical_ablation_vectors: Sequence[HarpRoutedVectorSeal],
    truth_by_key: Mapping[tuple[str, str, str], int],
    *,
    prelabel_bundle_hash: str,
    physical_reference_preserving_surface_hash: str,
) -> dict[str, object]:
    """Score the complete sealed action matrix without emitting policy state."""

    view = validated_target_menu_view(menu)
    prelabel_hash = require_sha256(
        prelabel_bundle_hash, name="HARP prelabel bundle hash"
    )
    physical_reference_hash = require_sha256(
        physical_reference_preserving_surface_hash,
        name="HARP physical reference-preserving surface hash",
    )
    routed = tuple(vectors)
    physical_routed = tuple(physical_ablation_vectors)
    if len(routed) != len(CENTERS) or len(physical_routed) != len(CENTERS):
        raise ProtocolError(
            "HARP terminal diagnostics require both complete center-vector sets."
        )
    action_rows: list[dict[str, object]] = []
    center_rows: list[dict[str, object]] = []
    observed_keys: set[tuple[str, str, str]] = set()
    for center, vector, physical_vector in zip(
        CENTERS, routed, physical_routed, strict=True
    ):
        vector.assert_valid()
        physical_vector.assert_valid()
        if {row.outer_target_id for row in vector.decisions} != {center}:
            raise ProtocolError("HARP terminal diagnostic vector crossed centers.")
        if (
            tuple((row.row_id, row.case_id) for row in vector.decisions)
            != tuple((row.row_id, row.case_id) for row in physical_vector.decisions)
            or any(
                row.eligible and row.lambda_value != 1.0
                for row in physical_vector.decisions
            )
        ):
            raise ProtocolError(
                "HARP terminal physical-ablation vector coverage drifted."
            )
        keys = tuple(
            (center, row.case_id, row.row_id) for row in vector.decisions
        )
        observed_keys.update(keys)
        try:
            truth = np.asarray([truth_by_key[key] for key in keys], dtype=np.int64)
        except KeyError as exc:
            raise ProtocolError("HARP terminal diagnostic truth is incomplete.") from exc
        cases = np.asarray([row.case_id for row in vector.decisions], dtype=str)
        baseline_action = view.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=BASE_ACTION_ID,
        )
        reference_action = view.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=UNIFORM_ACTION_ID,
        )
        baseline = view.exact_nine(baseline_action)
        reference = view.exact_nine(reference_action)
        if (
            view.identities_for(baseline_action)
            != view.identities_for(reference_action)
            or view.identities_for(baseline_action)[0] != vector.row_ids
            or view.identities_for(baseline_action)[1] != vector.case_ids
        ):
            raise ProtocolError("HARP terminal diagnostic B/U/vector rows drifted.")
        baseline_metric = _metric(truth, baseline, cases)
        reference_metric = _metric(truth, reference, cases)
        block_probabilities: list[np.ndarray] = [baseline, reference]
        block_rows: list[dict[str, object]] = []

        def append_metric(
            *,
            action_id: str,
            physical_action_id: str,
            source: str | None,
            lam: float,
            physical: bool,
            matched: bool,
            probability: np.ndarray,
        ) -> None:
            bacc, brier, log_loss = _metric(truth, probability, cases)
            block_rows.append(
                {
                    "center": center,
                    "action_id": action_id,
                    "physical_action_id": physical_action_id,
                    "selected_source_id": source,
                    "lambda_value": lam,
                    "physical_generated_action": physical,
                    "matched_budget_action": matched,
                    "row_count": len(truth),
                    "case_count": len(set(cases.tolist())),
                    "balanced_accuracy": bacc,
                    "brier": brier,
                    "log_loss": log_loss,
                    "balanced_accuracy_delta_vs_b": bacc - baseline_metric[0],
                    "balanced_accuracy_delta_vs_u": bacc - reference_metric[0],
                    "brier_delta_vs_b": brier - baseline_metric[1],
                    "brier_delta_vs_u": brier - reference_metric[1],
                    "log_loss_delta_vs_b": log_loss - baseline_metric[2],
                    "log_loss_delta_vs_u": log_loss - reference_metric[2],
                    "diagnostic_only": True,
                }
            )

        append_metric(
            action_id=BASE_ACTION_ID,
            physical_action_id=BASE_ACTION_ID,
            source=None,
            lam=0.0,
            physical=True,
            matched=False,
            probability=baseline,
        )
        append_metric(
            action_id=UNIFORM_ACTION_ID,
            physical_action_id=UNIFORM_ACTION_ID,
            source=None,
            lam=0.0,
            physical=True,
            matched=True,
            probability=reference,
        )
        for source in CENTERS:
            if source == center:
                continue
            physical_action = view.action_for(
                surface_kind=TARGET_SURFACE,
                outer_target_id=center,
                query_center_id=center,
                selected_source_id=source,
            )
            expert = view.exact_nine(physical_action)
            if view.identities_for(physical_action) != view.identities_for(baseline_action):
                raise ProtocolError("HARP terminal candidate rows drifted from B/U.")
            for lam in LAMBDA_GRID:
                probability = np.ascontiguousarray(
                    (1.0 - lam) * reference + lam * expert, dtype=np.float64
                )
                block_probabilities.append(probability)
                append_metric(
                    action_id=f"Hxe::{source}|lambda={lam:.2f}",
                    physical_action_id=physical_action.action_id,
                    source=source,
                    lam=lam,
                    physical=lam == 1.0,
                    matched=True,
                    probability=probability,
                )
        expected_count = 2 + (len(CENTERS) - 1) * len(LAMBDA_GRID)
        if len(block_rows) != expected_count:
            raise ProtocolError("HARP terminal action matrix is incomplete.")

        stack = np.stack(block_probabilities, axis=0)
        true_probability = np.where(truth[None, :] == 1, stack, 1.0 - stack)
        selected_true_probability = np.where(
            truth == 1, vector.routed_probabilities, 1.0 - vector.routed_probabilities
        )
        best_true_probability = np.max(true_probability, axis=0)
        top1 = selected_true_probability >= best_true_probability - _TIE_TOLERANCE
        better = np.sum(
            true_probability > selected_true_probability[None, :] + _TIE_TOLERANCE,
            axis=0,
        )
        ties = np.sum(
            np.abs(true_probability - selected_true_probability[None, :])
            <= _TIE_TOLERANCE,
            axis=0,
        )
        selected_rank = 1.0 + better + 0.5 * np.maximum(0, ties - 1)
        log_regret = _loss(truth, vector.routed_probabilities) - np.min(
            -np.log(np.clip(true_probability, _EPSILON, 1.0)), axis=0
        )
        brier_regret = (vector.routed_probabilities - truth) ** 2 - np.min(
            (stack - truth[None, :]) ** 2, axis=0
        )
        fixed_best = max(row["balanced_accuracy"] for row in block_rows)
        physical_rows = tuple(
            row
            for row in block_rows
            if row["matched_budget_action"] and row["physical_generated_action"]
        )
        physical_hxe_rows = tuple(
            row for row in physical_rows if row["selected_source_id"] is not None
        )
        if len(physical_hxe_rows) != len(CENTERS) - 1:
            raise ProtocolError("HARP terminal physical Hxe ablation is incomplete.")
        physical_best = max(
            row["balanced_accuracy"] for row in physical_hxe_rows
        )
        eligible = np.asarray(
            [row.eligible for row in vector.decisions], dtype=bool
        )
        selected_or_u = np.where(eligible, vector.routed_probabilities, reference)
        selected_or_u_metric = _metric(truth, selected_or_u, cases)
        physical_eligible = np.asarray(
            [row.eligible for row in physical_vector.decisions], dtype=bool
        )
        physical_or_u = np.where(
            physical_eligible, physical_vector.routed_probabilities, reference
        )
        physical_or_u_metric = _metric(truth, physical_or_u, cases)
        final_metric = _metric(truth, vector.routed_probabilities, cases)
        center_rows.append(
            {
                "center": center,
                "action_count": len(block_rows),
                "physical_matched_action_count": len(physical_rows),
                "budget_effect_u_minus_b": {
                    "balanced_accuracy": reference_metric[0] - baseline_metric[0],
                    "brier": reference_metric[1] - baseline_metric[1],
                    "log_loss": reference_metric[2] - baseline_metric[2],
                },
                "best_physical_hxe_lambda1_effect_vs_u": {
                    "balanced_accuracy": physical_best - reference_metric[0],
                    "best_action_ids": [
                        str(row["action_id"])
                        for row in physical_hxe_rows
                        if math.isclose(
                            float(row["balanced_accuracy"]),
                            physical_best,
                            rel_tol=0.0,
                            abs_tol=_TIE_TOLERANCE,
                        )
                    ],
                    "oracle_descriptive_not_a_policy_estimate": True,
                },
                "frozen_lambda_one_policy_effect_vs_u": {
                    "balanced_accuracy": physical_or_u_metric[0]
                    - reference_metric[0],
                    "brier": physical_or_u_metric[1] - reference_metric[1],
                    "log_loss": physical_or_u_metric[2]
                    - reference_metric[2],
                    "route_rate": _case_equal_mean(
                        physical_eligible.astype(np.float64), cases
                    ),
                    "ineligible_rows_preserve_U_for_estimand": True,
                    "selection_labels_used": False,
                },
                "selected_predictive_effect_vs_u": {
                    "balanced_accuracy": selected_or_u_metric[0] - reference_metric[0],
                    "brier": selected_or_u_metric[1] - reference_metric[1],
                    "log_loss": selected_or_u_metric[2] - reference_metric[2],
                },
                "final_operational_effect_vs_b": {
                    "balanced_accuracy": final_metric[0] - baseline_metric[0],
                    "brier": final_metric[1] - baseline_metric[1],
                    "log_loss": final_metric[2] - baseline_metric[2],
                },
                "selected_top1_oracle_tie_credit": _case_equal_mean(
                    top1.astype(np.float64), cases
                ),
                "selected_mean_true_probability_rank": _case_equal_mean(
                    selected_rank, cases
                ),
                "selected_mean_log_loss_regret": max(
                    0.0, _case_equal_mean(log_regret, cases)
                ),
                "selected_mean_brier_regret": max(
                    0.0, _case_equal_mean(brier_regret, cases)
                ),
                "best_fixed_action_bacc_minus_policy_bacc": fixed_best
                - final_metric[0],
                "case_equal_within_center": True,
                "labels_used_after_route_seal_only": True,
                "may_feed_policy_or_thresholds": False,
            }
        )
        action_rows.extend(block_rows)
    if set(truth_by_key) != observed_keys:
        raise ProtocolError("HARP terminal diagnostic truth contains missing or surplus rows.")
    base: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_action_oracle_diagnostics_v1",
        "prelabel_bundle_hash": prelabel_hash,
        "physical_reference_preserving_surface_hash": physical_reference_hash,
        "prediction_menu_seal_hash": view.seal_hash,
        "action_matrix_row_count": len(action_rows),
        "center_diagnostics": center_rows,
        "action_matrix": action_rows,
        "matched_budget_reference_action": "U",
        "operational_fallback_action": "B",
        "physical_expert_ablation_lambda": 1.0,
        "lambda_semantics": (
            "post_classifier_predictive_probability_ensemble_"
            "not_generated_distribution"
        ),
        "diagnostic_only": True,
        "labels_used_after_route_seal_only": True,
        "labels_available_to_policy": False,
        "policy_or_threshold_update_emitted": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
    }
    output = {**base, "diagnostic_hash": canonical_hash(base)}
    view.assert_fully_valid()
    return output


__all__ = ("build_terminal_action_diagnostics",)
