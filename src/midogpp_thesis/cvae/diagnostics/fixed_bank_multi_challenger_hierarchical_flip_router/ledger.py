"""Direct-parent, single-consumer ledger admission for this diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    AUTHORIZATION_SCOPE,
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
    """Validate the immutable original ledger and this experiment's amendment."""

    parent_path = Path(getattr(config, "test_consumption_ledger_path"))
    amendment_path = Path(getattr(config, "ledger_amendment_path"))
    parent = _json(parent_path)
    amendment = _json(amendment_path)
    gate = {
        "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
        "lcb_field": "one_sided_95_lcb",
        "threshold": 0.0,
        "comparison": "strictly_greater_than",
        "required_contrast_count": 6,
        "pass_status": "PASS",
        "fail_status": "FAIL",
        "diagnostic_only": True,
        "routing_success_claimed": False,
        "promotion_eligible": False,
    }
    exact = (
        getattr(config, "experiment_id") == EXPERIMENT_ID
        and sha256_file(parent_path) == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        and parent.get("status") == "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        and parent.get("split") == "test"
        and sha256_file(amendment_path) == EXPECTED_LEDGER_AMENDMENT_SHA256
        and amendment.get("amendment_id") == LEDGER_AMENDMENT_ARTIFACT_ID
        and amendment.get("parent_sha256")
        == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        and amendment.get("authorized_consumer_experiment_ids") == [EXPERIMENT_ID]
        and amendment.get("authorization_scope") == AUTHORIZATION_SCOPE
        and amendment.get("claim_role") == CLAIM_ROLE
        and amendment.get("fresh_evidence") is False
        and amendment.get("routing_success_claimed") is False
        and amendment.get("previous_stage90_outputs_used") is False
        and amendment.get("previous_stage90_amendments_used") is False
        and amendment.get("previous_prediction_surfaces_used") is False
        and amendment.get("previous_stage90_scratch_or_checkpoints_used") is False
        and amendment.get("all_action_probabilities_globally_sealed_before_any_label_access")
        is True
        and amendment.get("all_label_free_case_action_features_sealed_before_any_label_access")
        is True
        and amendment.get("every_donor_row_requires_H_q_e_distinct") is True
        and amendment.get("selection_calibration_evaluation_case_disjoint") is True
        and amendment.get("target_support_labels_may_update_shared_model") is False
        and amendment.get("candidate_menu_top_k") == 3
        and amendment.get("candidate_menu_always_includes_B") is True
        and amendment.get("U_is_control_not_candidate") is True
        and amendment.get("primary_router_id") == "R_multi"
        and amendment.get("terminal_oracle_ids")
        == ["O_menu", "O_binary", "O_static", "O_case"]
        and amendment.get("terminal_scoring_occurs_only_after_all_45_decision_seals")
        is True
        and amendment.get("held_evaluation_label_mutation_must_leave_menus_models_calibrations_decisions_and_seals_unchanged")
        is True
        and amendment.get("diagnostic_recoverability_gate") == gate
        and amendment.get("may_feed_another_experiment") is False
    )
    if not exact:
        raise ProtocolError("Multi-challenger consumption-ledger chain drifted.")
    return ValidatedLedgerChain(
        MappingProxyType(parent), MappingProxyType(amendment)
    )


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read multi-challenger ledger: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Multi-challenger ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
