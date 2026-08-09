"""Closed-world and reconstructive-validation tests for the fixed-bank audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.artifact_io import (
    atomic_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.config_payloads import (
    EXACT_FAMILY_PREDICTORS as CONFIG_EXACT_FAMILY_PREDICTORS,
    SMOOTH_FAMILY_IDS as CONFIG_SMOOTH_FAMILY_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.constants import (
    EXACT_FAMILY_PREDICTORS,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.experiment_contracts import (
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.reports import (
    publication_decision_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_CATALOG = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"
PACKAGE_ROOT = (
    REPOSITORY_ROOT
    / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_decision_audit"
)


def _materialize_index_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{member}\n", encoding="utf-8")


def _write_inventory(root: Path) -> None:
    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_required_files_match_catalog_counts_and_expose_no_action_surface() -> None:
    catalog = yaml.safe_load(ARTIFACT_CATALOG.read_text(encoding="utf-8"))
    output = next(
        row
        for row in catalog["artifacts"]
        if row["artifact_id"] == OUTPUT_ARTIFACT_ID
    )
    assert tuple(output["required_files"]) == REQUIRED_FILES
    identities = output["semantic_identities"]
    assert identities["exact_family_count"] == "9"
    assert identities["exact_prediction_row_count"] == "4536"
    assert identities["exact_fold_audit_row_count"] == "648"
    assert identities["smooth_prediction_row_count"] == "1512"
    assert identities["smooth_fold_audit_row_count"] == "216"
    forbidden = ("target_action", "policy", "selection_feed", "target_prediction")
    assert not any(
        token in member for member in REQUIRED_FILES for token in forbidden
    )
    assert "tables/exact_crossfit_fold_audits.csv" in REQUIRED_FILES
    assert "tables/smooth_descriptive_crossfit_fold_audits.csv" in REQUIRED_FILES


def test_import_graph_does_not_depend_on_prior_case_aware_or_stage60_science() -> None:
    forbidden = (
        "utility_aligned_case_aware_proxy_information_audit",
        "routing.residual_topup",
    )
    imports: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
    assert not [
        module
        for module in imports
        if any(fragment in module for fragment in forbidden)
    ]


def test_config_family_declarations_equal_the_canonical_scientific_constants() -> None:
    assert tuple(CONFIG_EXACT_FAMILY_PREDICTORS) == tuple(EXACT_FAMILY_PREDICTORS)
    assert {
        family_id: tuple(predictors)
        for family_id, predictors in CONFIG_EXACT_FAMILY_PREDICTORS.items()
    } == dict(EXACT_FAMILY_PREDICTORS)
    assert CONFIG_SMOOTH_FAMILY_IDS == SMOOTH_DESCRIPTIVE_FAMILY_IDS


def test_content_index_detects_member_tamper_without_repair(tmp_path: Path) -> None:
    _materialize_index_members(tmp_path)
    write_content_index(tmp_path, config_contract_hash="a" * 64)
    validate_content_index(tmp_path, config_contract_hash="a" * 64)
    member = tmp_path / "tables/exact_crossfit_predictions.csv"
    member.write_bytes(member.read_bytes() + b"tamper")
    tampered_member = member.read_bytes()
    with pytest.raises(ProtocolError, match="content-index member drifted"):
        validate_content_index(tmp_path, config_contract_hash="a" * 64)
    assert member.read_bytes() == tampered_member

    index = tmp_path / "manifests/content_index.json"
    index.write_bytes(index.read_bytes() + b"tamper")
    tampered_index = index.read_bytes()
    with pytest.raises(ProtocolError):
        write_content_index(tmp_path, config_contract_hash="a" * 64)
    assert index.read_bytes() == tampered_index


def test_closed_world_allows_resume_checkpoints_but_rejects_them_at_completion(
    tmp_path: Path,
) -> None:
    _write_inventory(tmp_path)
    checkpoint = tmp_path / "checkpoints/orphan-task.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    assert_closed_world(tmp_path, allow_incomplete=True)
    with pytest.raises(ProtocolError, match="orphan-task"):
        assert_closed_world(tmp_path, allow_incomplete=False)

    checkpoint.unlink()
    extra = tmp_path / "reports/unauthorized_feed.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unauthorized_feed"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_validator_checks_content_bytes_before_scientific_reconstruction_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit import validation

    config = SimpleNamespace(
        contract_hash="a" * 64,
        artifact_root=tmp_path.resolve(),
        input_artifact_ids=("input",),
    )
    reached_reconstruction: list[bool] = []
    monkeypatch.setattr(
        validation, "assert_closed_world", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        validation,
        "load_fixed_bank_decision_audit_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        validation,
        "validate_content_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProtocolError("content byte tamper")
        ),
    )
    monkeypatch.setattr(
        validation,
        "assert_input_fence",
        lambda _config: reached_reconstruction.append(True),
    )
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    with pytest.raises(ProtocolError, match="content byte tamper"):
        validation.validate_fixed_bank_decision_audit_bundle(
            tmp_path, config=config
        )
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert reached_reconstruction == []
    assert after == before


def test_publication_decision_is_byte_invariant_to_smooth_only_changes(
    tmp_path: Path,
) -> None:
    exact = {
        "primary_exact_gate_passed": True,
        "exact_decision_hash": "a" * 64,
        "audit_result_hash": "b" * 64,
        "smooth_descriptive_crossfit": {"result_hash": "c" * 64},
    }
    poisoned = {
        **exact,
        "audit_result_hash": "d" * 64,
        "smooth_descriptive_crossfit": {
            "result_hash": "e" * 64,
            "arbitrary_poison": [1.0, -1.0],
        },
    }
    first = publication_decision_payload(exact)
    second = publication_decision_payload(poisoned)
    assert first == second
    assert first["exact_decision_hash"] == "a" * 64
    assert "audit_result_hash" not in first
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    atomic_json(first_path, first)
    atomic_json(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_publication_decision_fails_closed_without_exact_decision_binding() -> None:
    with pytest.raises(ProtocolError, match="incomplete for publication"):
        publication_decision_payload(
            {"primary_exact_gate_passed": True, "audit_result_hash": "a" * 64}
        )
    with pytest.raises(ProtocolError, match="must be boolean"):
        publication_decision_payload(
            {"primary_exact_gate_passed": 1, "exact_decision_hash": "a" * 64}
        )
