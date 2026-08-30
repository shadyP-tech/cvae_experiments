"""Terminal descriptive inference and durable prelabel validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_portfolio import HarpPortfolioDecision
from ...routing.harp_protocol import canonical_hash
from ...routing.harp_replay import HarpReplayResult
from ...runtime.artifact_io import read_json


TWO_SIDED_95_T_DF8 = 2.306004135204166
ONE_SIDED_95_T_DF8 = 1.8595480375228424
TIE_TOLERANCE = 1.0e-15


def validate_prelabel_bundle(path: Path, *, validator_id: str) -> str:
    raw = read_json(path)
    base = {key: value for key, value in raw.items() if key != "bundle_hash"}
    decisions = raw.get("decisions")
    physical_decisions = raw.get("physical_ablation_decisions")
    if (
        raw.get("schema_version") != "midogpp_harp_stage90_prelabel_bundle_v2"
        or raw.get("status") != "DURABLE_ALL_ROUTES_SEALED_BEFORE_EVALUATION_LABELS"
        or raw.get("publication_status") != "POST_HOC_CONSUMED_TEST_SENSITIVITY"
        or raw.get("terminal_decision") != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or raw.get("evaluation_labels_opened") is not False
        or raw.get("may_feed_another_experiment") is not False
        or not isinstance(decisions, list)
        or not decisions
        or not isinstance(physical_decisions, list)
        or len(physical_decisions) != len(decisions)
        or raw.get("physical_ablation_action_universe")
        != "Hxe_lambda_one_only"
        or raw.get("physical_ablation_selection_labels_used") is not False
        or raw.get("bundle_hash") != canonical_hash(base)
    ):
        raise ProtocolError("HARP Stage-90 prelabel bundle failed validation.")
    keys: list[tuple[str, str, str]] = []
    centers: set[str] = set()
    for row in decisions:
        if not isinstance(row, Mapping):
            raise ProtocolError("HARP Stage-90 prelabel decision is malformed.")
        key = (str(row.get("outer_target_id")), str(row.get("case_id")), str(row.get("sample_id")))
        keys.append(key)
        centers.add(key[0])
        baseline = row.get("baseline_probability_hex")
        output = row.get("output_probability_hex")
        if (
            not isinstance(baseline, str)
            or not isinstance(output, str)
            or len(baseline) != 16
            or len(output) != 16
            or (row.get("routed") is False and baseline != output)
        ):
            raise ProtocolError("HARP Stage-90 fallback probability bytes drifted.")
    if keys != sorted(keys) or len(set(keys)) != len(keys) or tuple(sorted(centers)) != CENTERS:
        raise ProtocolError("HARP Stage-90 prelabel row coverage drifted.")
    physical_keys: list[tuple[str, str, str]] = []
    for row in physical_decisions:
        if not isinstance(row, Mapping):
            raise ProtocolError(
                "HARP Stage-90 physical-ablation decision is malformed."
            )
        physical_keys.append(
            (
                str(row.get("outer_target_id")),
                str(row.get("case_id")),
                str(row.get("sample_id")),
            )
        )
        if row.get("routed") is True and row.get("selected_lambda") != 1.0:
            raise ProtocolError(
                "HARP Stage-90 physical ablation escaped lambda=1."
            )
        if (
            row.get("routed") is False
            and row.get("baseline_probability_hex")
            != row.get("output_probability_hex")
        ):
            raise ProtocolError(
                "HARP Stage-90 physical-ablation exact-B fallback drifted."
            )
    if physical_keys != keys:
        raise ProtocolError(
            "HARP Stage-90 physical-ablation row coverage drifted."
        )
    if not validator_id:
        raise ProtocolError("HARP Stage-90 validator identity is empty.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_stage90_prelabel_validation_v1",
            "validator_id": validator_id,
            "bundle_hash": raw["bundle_hash"],
            "row_count": len(keys),
            "exact_b_fallback_byte_identity": True,
            "evaluation_labels_opened": False,
        }
    )


def decision_payload(decisions: Sequence[HarpPortfolioDecision]) -> list[dict[str, object]]:
    return [
        {
            "outer_target_id": row.outer_target_id,
            "case_id": row.case_id,
            "sample_id": row.sample_id,
            "baseline_probability_hex": row.baseline_probability_bytes.hex(),
            "output_probability_hex": row.output_probability_bytes.hex(),
            "selected_source_id": row.selected_source_id,
            "selected_lambda": row.selected_lambda,
            "routed": row.routed,
            "reason": row.reason,
            "gain_lower": row.gain_lower,
            "brier_upper": row.brier_upper,
            "log_loss_upper": row.log_loss_upper,
            "prediction_seal_hash": row.prediction_seal_hash,
            "ensemble_receipt_hash": row.ensemble_receipt_hash,
        }
        for row in decisions
    ]


def route_reason_summary(decisions: Sequence[HarpPortfolioDecision]) -> dict[str, object]:
    rows = tuple(decisions)
    by_reason = Counter(row.reason for row in rows)
    by_center = {
        center: {
            "row_count": sum(row.outer_target_id == center for row in rows),
            "routed_count": sum(row.outer_target_id == center and row.routed for row in rows),
            "fallback_count": sum(row.outer_target_id == center and not row.routed for row in rows),
            "reasons": dict(sorted(Counter(row.reason for row in rows if row.outer_target_id == center).items())),
        }
        for center in CENTERS
    }
    return {
        "schema_version": "midogpp_harp_stage90_route_reason_summary_v1",
        "row_count": len(rows),
        "routed_count": sum(row.routed for row in rows),
        "fallback_count": sum(not row.routed for row in rows),
        "reasons": dict(sorted(by_reason.items())),
        "by_center": by_center,
        "exact_b_fallback_byte_identity": all(
            row.routed or row.output_probability_bytes == row.baseline_probability_bytes
            for row in rows
        ),
    }


def _inference(endpoint: str, values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (len(CENTERS),) or not np.isfinite(array).all():
        raise ProtocolError("HARP Stage-90 inference requires exactly nine center deltas.")
    mean = float(np.mean(array, dtype=np.float64))
    sd = float(np.std(array, ddof=1, dtype=np.float64))
    se = sd / math.sqrt(len(CENTERS))
    wins = int(np.sum(array > TIE_TOLERANCE))
    losses = int(np.sum(array < -TIE_TOLERANCE))
    ties = len(CENTERS) - wins - losses
    return {
        "endpoint": endpoint,
        "positive_delta_favors_routed": True,
        "center_deltas": [float(value) for value in array],
        "mean_delta": mean,
        "sample_standard_deviation": sd,
        "standard_error": se,
        "two_sided_95_interval_low": mean - TWO_SIDED_95_T_DF8 * se,
        "two_sided_95_interval_high": mean + TWO_SIDED_95_T_DF8 * se,
        "one_sided_95_lower_bound": mean - ONE_SIDED_95_T_DF8 * se,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "inference_unit": "target_center",
        "inference_unit_count": len(CENTERS),
        "degrees_of_freedom": len(CENTERS) - 1,
        "seed_cells_are_inference_units": False,
        "descriptive_post_hoc_only": True,
    }


def replay_payload(result: HarpReplayResult) -> dict[str, object]:
    if not isinstance(result, HarpReplayResult) or tuple(center for center, _ in result.center_metrics) != CENTERS:
        raise ProtocolError("HARP Stage-90 replay center coverage drifted.")
    centers = [
        {"center": center, **asdict(metrics)} for center, metrics in result.center_metrics
    ]
    inference = [
        _inference(
            "balanced_accuracy_improvement",
            [metrics.balanced_accuracy_delta for _center, metrics in result.center_metrics],
        ),
        _inference(
            "brier_improvement",
            [-metrics.brier_delta for _center, metrics in result.center_metrics],
        ),
        _inference(
            "log_loss_improvement",
            [-metrics.log_loss_delta for _center, metrics in result.center_metrics],
        ),
    ]
    payload = {
        "schema_version": "midogpp_harp_stage90_terminal_result_v1",
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
        "prediction_seal_hash": result.prediction_seal_hash,
        "primary_equal_center_metrics": asdict(result.metrics),
        "center_metrics": centers,
        "center_t95_inference": inference,
        "descriptive_raw_row_metrics": asdict(result.descriptive_row_metrics),
        "case_equal_within_center": True,
        "center_equal_primary": True,
        "seed_cells_are_inference_units": False,
        "confirmatory_inference_claimed": False,
        "may_feed_another_experiment": False,
    }
    return {**payload, "result_hash": canonical_hash(payload)}


__all__ = (
    "decision_payload",
    "replay_payload",
    "route_reason_summary",
    "validate_prelabel_bundle",
)
