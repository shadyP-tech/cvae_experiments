"""Terminal-only evaluation after the global aggregate decision seal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import CONTROL_METHOD_IDS, EXPECTED_TEST_ROW_COUNT, EXPECTED_TOTAL_CASE_COUNT, METHOD_IDS, PRE_TERMINAL_METHOD_IDS
from .delete_center import full_pipeline_delete_one_center
from .hashing import canonical_hash, require_sha256
from .identification_metrics import (
    build_case_directional_oracles,
    build_static_directional_oracles,
    score_identification_metrics,
)
from .response_products import BinaryLabel
from .response_scoring import score_case_action_confusions
from .terminal_tables import (
    center_metric_rows,
    contrast_rows,
    method_and_calibration_rows,
    oracle_prediction_views,
    preterminal_prediction_views,
    terminal_case_confusions,
)


def evaluate_terminal(
    *,
    surface: object,
    plans: Sequence[object],
    directional_support_gains: Sequence[object],
    identification_decisions: Sequence[object],
    robust_arm_decisions: Sequence[object],
    method_predictions: Sequence[object],
    terminal_labels: Sequence[BinaryLabel],
    aggregate_seal_hash: str,
    config: object,
) -> Mapping[str, object]:
    """Build every persisted terminal table without exposing labels upstream."""

    del config  # the executable science is frozen in constants; config is validated elsewhere.
    aggregate = require_sha256(aggregate_seal_hash, "aggregate_plan_decision_seal_hash")
    if (
        len(plans) != EXPECTED_TOTAL_CASE_COUNT
        or len(identification_decisions) != EXPECTED_TOTAL_CASE_COUNT * 2
        or len(robust_arm_decisions) != EXPECTED_TOTAL_CASE_COUNT * 18
        or len(method_predictions) != EXPECTED_TEST_ROW_COUNT * len(PRE_TERMINAL_METHOD_IDS)
        or len(terminal_labels) != EXPECTED_TEST_ROW_COUNT
    ):
        raise ProtocolError("OGDE terminal inputs lack the complete sealed topology.")
    physical_confusions = score_case_action_confusions(surface, terminal_labels)
    case_oracles = build_case_directional_oracles(physical_confusions)
    static_oracles = build_static_directional_oracles(physical_confusions)
    views = (
        *preterminal_prediction_views(method_predictions),
        *oracle_prediction_views(surface, case_oracles, static_oracles),
    )
    if {str(row["method_id"]) for row in views} != set(METHOD_IDS):
        raise ProtocolError("OGDE terminal prediction views lack primary/control/oracle methods.")
    case_confusions = terminal_case_confusions(views, terminal_labels)
    center_metrics = center_metric_rows(case_confusions)
    method_metrics, calibration_metrics = method_and_calibration_rows(
        views, terminal_labels, center_metrics
    )
    contrasts = contrast_rows(center_metrics)
    identification_metrics = tuple(
        score_identification_metrics(
            tuple(row for row in identification_decisions if getattr(row, "method_id") == method),
            case_oracles,
        ).to_payload()
        for method in ("I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED")
    )
    metric_index = {str(row["method_id"]): row for row in method_metrics}
    bacc_b = float(metric_index["B"]["equal_center_bacc"])
    attribution_controls = tuple(
        {
            "method_id": method,
            "equal_center_bacc": float(metric_index[method]["equal_center_bacc"]),
            "gain_over_B": float(metric_index[method]["equal_center_bacc"]) - bacc_b,
            "attribution_success_gate": False,
            "terminal_consumed_test_diagnostic": True,
        }
        for method in CONTROL_METHOD_IDS
    )
    delete_one_center = full_pipeline_delete_one_center(
        surface=surface,
        plans=plans,  # type: ignore[arg-type]
        directional_support_gains=directional_support_gains,  # type: ignore[arg-type]
        identification_decisions=identification_decisions,  # type: ignore[arg-type]
        terminal_confusions=physical_confusions,
        terminal_labels=terminal_labels,
    )
    tables = {
        "case_confusions": case_confusions,
        "method_metrics": method_metrics,
        "center_metrics": center_metrics,
        "contrasts": contrasts,
        "identification_metrics": identification_metrics,
        "calibration_metrics": calibration_metrics,
        "delete_one_center": delete_one_center,
        "attribution_controls": attribution_controls,
    }
    terminal_payload = {
        "schema_version": "fixed_bank_ogde_terminal_evaluation_seal_v1",
        "aggregate_plan_decision_seal_hash": aggregate,
        "probability_surface_hash": str(getattr(surface, "surface_hash")),
        "table_hashes": {name: canonical_hash(rows) for name, rows in tables.items()},
        "method_ids": list(METHOD_IDS),
        "terminal_label_count": len(terminal_labels),
        "raw_labels_persisted": False,
        "terminal_consumed_test_diagnostic": True,
    }
    terminal_seal = {**terminal_payload, "seal_hash": canonical_hash(terminal_payload)}
    summary = {
        "B_equal_center_bacc": bacc_b,
        "OGDE_PORTFOLIO_equal_center_bacc": float(metric_index["OGDE_PORTFOLIO"]["equal_center_bacc"]),
        "OGDE_PORTFOLIO_gain_over_B": float(metric_index["OGDE_PORTFOLIO"]["equal_center_bacc"]) - bacc_b,
        "R_NINE_ARM_ROBUST_equal_center_bacc": float(metric_index["R_NINE_ARM_ROBUST"]["equal_center_bacc"]),
        "source_identification_established": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
    }
    return {**tables, "terminal_seal": terminal_seal, "diagnostic_summary": summary}


__all__ = ("evaluate_terminal",)
