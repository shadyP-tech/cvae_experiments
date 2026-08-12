"""Direct parent and single-use amendment validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    CLAIM_ROLE,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(config: object) -> ValidatedLedgerChain:
    parent_path = Path(getattr(config, "test_consumption_ledger_path"))
    amendment_path = Path(getattr(config, "ledger_amendment_path"))
    parent = _json(parent_path)
    amendment = _json(amendment_path)
    if (
        getattr(config, "experiment_id") != EXPERIMENT_ID
        or sha256_file(parent_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or sha256_file(amendment_path) != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("claim_role") != CLAIM_ROLE
        or amendment.get("fresh_evidence") is not False
        or amendment.get("routing_success_claimed") is not False
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_prediction_surfaces_used") is not False
        or amendment.get("all_action_probabilities_globally_sealed_before_any_label_access") is not True
        or amendment.get("heldout_evaluation_fold_absent_from_source_selection_calibration_fit_thresholding_and_decision") is not True
        or amendment.get("all_nonoracle_fold_decisions_sealed_before_their_held_evaluation_labels_can_score_them") is not True
        or amendment.get("routing_identification_metrics") != [
            "top1_oracle_agreement",
            "spearman_rank_correlation",
            "normalized_oracle_gap",
            "fold_stability",
        ]
        or amendment.get("G_static_selection_objective")
        != "unweighted_least_squares_exact_per_q_e_pooled_bacc_gain"
        or amendment.get("G_static_model")
        != "gain_qe=grand_mean+query_effect_q+source_effect_e"
        or amendment.get("G_static_identifiability_constraints")
        != ["sum_query_effects=0", "sum_source_effects=0"]
        or amendment.get("G_static_selection_score")
        != "grand_mean_plus_source_effect"
        or amendment.get("primary_heuristic_router_id") != "F_S"
        or amendment.get("heuristic_prediction_bound_descriptive_only") is not True
        or amendment.get("calibrated_case_confidence_or_safety_claimed") is not False
        or amendment.get("diagnostic_recoverability_gate") != {
            "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
            "lcb_field": "one_sided_95_lcb",
            "threshold": 0.0,
            "comparison": "strictly_greater_than",
            "required_contrast_count": 5,
            "pass_status": "PASS",
            "fail_status": "FAIL",
            "diagnostic_only": True,
            "routing_success_claimed": False,
            "promotion_eligible": False,
        }
    ):
        raise ProtocolError("Flip-router consumption-ledger chain drifted.")
    return ValidatedLedgerChain(MappingProxyType(parent), MappingProxyType(amendment))


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read flip-router ledger: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Flip-router ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
