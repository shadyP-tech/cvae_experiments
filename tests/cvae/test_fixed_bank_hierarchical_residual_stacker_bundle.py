from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.reports import (
    publication_decision_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.core_hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.validation import (
    _require_closed_hashed_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_bundle_persists_whole_case_sufficient_statistics_without_case_bacc() -> None:
    assert "tables/oof_case_confusion_sufficient_statistics.csv" in REQUIRED_FILES
    assert "tables/oof_pooled_exact_bacc.csv" in REQUIRED_FILES
    assert "tables/oof_case_bacc.csv" not in REQUIRED_FILES
    assert "arrays/permutation_null_actions.npy" not in REQUIRED_FILES


def test_content_index_is_terminal_and_has_no_deployable_capability(
    tmp_path: Path,
) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    payload = write_content_index(tmp_path, config_contract_hash="contract")
    assert payload["terminal_consumed_test_diagnostic_only"] is True
    assert payload["deployable_policy_or_action_capability_present"] is False
    assert payload["may_feed_another_stage90"] is False


def test_content_index_detects_member_tamper_before_semantic_validation(
    tmp_path: Path,
) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    write_content_index(tmp_path, config_contract_hash="contract")
    assert validate_content_index(
        tmp_path, config_contract_hash="contract"
    )["closed_world"] is True
    (tmp_path / "manifests/terminal_pooled_bacc_evaluation.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="member drifted"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_incomplete_bundle_allows_only_owned_hash_resume_checkpoints(
    tmp_path: Path,
) -> None:
    allowed = (
        "checkpoints/frozen_source_streams/source_0_train_17.json",
        "checkpoints/label_free_action_predictions/target_scratch.json",
        "checkpoints/label_free_action_predictions/tasks/target_0_train_17_generation_17.npz",
    )
    for member in allowed:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")
    assert_closed_world(tmp_path, allow_incomplete=True)
    rogue = tmp_path / "checkpoints/fixed_bank_pooled_bacc_case_oof_ceiling_v2/state.json"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("rogue", encoding="utf-8")
    with pytest.raises(ProtocolError, match="extras"):
        assert_closed_world(tmp_path, allow_incomplete=True)


def test_publication_report_uses_exact_terminal_claim_role() -> None:
    payload = publication_decision_payload({"scientific_result_hash": "a" * 64})
    assert payload["decision"] == "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    assert (
        payload["claim_role"]
        == "known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic"
    )
    assert payload["promotion_eligible"] is False
    assert payload["policy_update_authorized"] is False


def test_coherently_rehashed_extra_authorization_field_is_rejected() -> None:
    payload = {"schema_version": "scientific_v1", "value": 1}
    payload["scientific_result_hash"] = canonical_hash(payload)
    tampered = {**payload, "policy_update_authorized": True}
    tampered["scientific_result_hash"] = canonical_hash(
        {
            key: value
            for key, value in tampered.items()
            if key != "scientific_result_hash"
        }
    )
    with pytest.raises(ProtocolError, match="closed schema or hash drifted"):
        _require_closed_hashed_payload(
            tampered,
            hash_key="scientific_result_hash",
            expected_keys={"schema_version", "value", "scientific_result_hash"},
            role="test scientific payload",
        )
