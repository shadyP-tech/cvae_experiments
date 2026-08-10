from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.artifact_io import (
    persist_or_validate_json,
)
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
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.contracts import (
    PooledExactBacc,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.persistence import (
    _persist_rows,
    persist_postseal_results,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    METHOD_IDS,
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


def test_postseal_resume_projects_real_mixed_metric_rows_deterministically(
    tmp_path: Path,
) -> None:
    metric_rows = _real_metric_rows()
    evaluation = {
        "schema_version": "test_terminal_evaluation_v1",
        "metrics": metric_rows,
        "scientific_result_hash": "a" * 64,
    }
    confusion_rows = (
        {
            "method_id": "B",
            "target_center": "0",
            "case_id": "case-0",
            "n_positive": 1,
            "true_positive": 1,
            "n_negative": 1,
            "true_negative": 1,
            "per_case_bacc_stored": False,
        },
    )
    contrast_rows = (
        {
            "contrast_id": "R-B_cal",
            "equal_center_difference": 0.0,
        },
    )
    metric_rows_before = json.dumps(metric_rows, sort_keys=True)

    # Reproduce the workstation checkpoint: evaluation JSON and confusion rows
    # already exist, while the heterogeneous metric table has not been written.
    persist_or_validate_json(
        tmp_path / "manifests/terminal_pooled_bacc_evaluation.json",
        evaluation,
    )
    _persist_rows(
        tmp_path / "tables/oof_case_confusion_sufficient_statistics.csv",
        confusion_rows,
    )
    kwargs = {
        "evaluation": evaluation,
        "confusion_rows": confusion_rows,
        "metric_rows": metric_rows,
        "contrast_rows": contrast_rows,
        "capability_report": {"status": "PASS"},
        "leakage_report": {"status": "PASS"},
        "runtime_summary": {"status": "PASS"},
    }
    persist_postseal_results(tmp_path, **kwargs)

    metric_path = tmp_path / "tables/oof_pooled_exact_bacc.csv"
    with metric_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        persisted = list(reader)
        assert reader.fieldnames == [
            "scope",
            "target_center",
            "schema_version",
            "method_id",
            "case_count",
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
            "sensitivity",
            "specificity",
            "exact_bacc",
            "per_case_bacc_used",
            "smooth_response_used",
            "metric_hash",
        ]
    center_rows = [row for row in persisted if row["scope"] == "center"]
    equal_center_rows = [row for row in persisted if row["scope"] == "equal_center"]
    assert len(center_rows) == 45
    assert len(equal_center_rows) == 5
    assert {
        row["schema_version"] for row in center_rows
    } == {"fixed_bank_hierarchical_residual_stacker_pooled_exact_bacc_v1"}
    assert {
        row["schema_version"] for row in equal_center_rows
    } == {"fixed_bank_hierarchical_residual_stacker_equal_center_exact_bacc_v1"}
    assert all(row["sensitivity"] == "" for row in equal_center_rows)
    assert all(row["specificity"] == "" for row in equal_center_rows)
    assert json.dumps(metric_rows, sort_keys=True) == metric_rows_before

    first_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    persist_postseal_results(tmp_path, **kwargs)
    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == first_bytes

    changed = {**evaluation, "scientific_result_hash": "b" * 64}
    with pytest.raises(ProtocolError, match="differs and will not be repaired"):
        persist_postseal_results(tmp_path, **{**kwargs, "evaluation": changed})
    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == first_bytes


def _real_metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHOD_IDS:
        center_scores: list[float] = []
        for center in CENTERS:
            metric = PooledExactBacc(
                method_id=method,
                case_count=2,
                n_positive=2,
                true_positive=1,
                n_negative=2,
                true_negative=1,
                sensitivity=0.5,
                specificity=0.5,
                exact_bacc=0.5,
            )
            center_scores.append(metric.exact_bacc)
            rows.append(
                {
                    "scope": "center",
                    "target_center": center,
                    **metric.to_payload(),
                }
            )
        rows.append(
            {
                "scope": "equal_center",
                "target_center": "ALL",
                "method_id": method,
                "case_count": 2 * len(CENTERS),
                "n_positive": 2 * len(CENTERS),
                "true_positive": len(CENTERS),
                "n_negative": 2 * len(CENTERS),
                "true_negative": len(CENTERS),
                "sensitivity": None,
                "specificity": None,
                "exact_bacc": 0.5,
                "per_case_bacc_used": False,
                "smooth_response_used": False,
                "metric_hash": canonical_hash([method, *center_scores]),
            }
        )
    return rows
