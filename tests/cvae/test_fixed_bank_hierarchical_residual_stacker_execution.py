from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker import (
    execution_adapter,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker import (
    validation_recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.artifact_io import (
    read_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.bundle import (
    CONTENT_INDEX_MEMBERS,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.execution_phases import (
    aggregate_exact_nine_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.execution_adapter import (
    RuntimeSeedProbabilityRow,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_MANIFEST_SHA256,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.input_contracts import (
    TestRowIdentity as ResidualTestRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.label_capabilities import (
    LabelCapabilityManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.ledger import (
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.persistence import (
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.reports import (
    run_state_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root=tmp_path / "expert_bank",
        generation_lock_root=tmp_path / "generation_lock",
        test_cache_root=tmp_path / "dedicated_residual_stacker_cache_v1",
        test_manifest_path=tmp_path / "dedicated_residual_stacker_manifest_v1" / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger_parent" / "reports" / "test_consumption_ledger.json",
        ledger_amendment_path=tmp_path / "residual_stacker_amendment_v1" / "amendment.json",
    )


def _recovery_bundle(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    root = (tmp_path / "artifact").resolve()
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed:{member}".encode("utf-8"))
    write_content_index(root, config_contract_hash="contract")
    write_run_state(
        root,
        status="FAILED",
        phase=validation_recovery.VALIDATION_PHASE,
        error=validation_recovery.RECOVERABLE_VALIDATOR_ERROR,
    )
    return root, SimpleNamespace(
        source_path=root / "config.resolved.yaml",
        artifact_root=root,
        contract_hash="contract",
    )


def test_exact_six_input_fence_rejects_previous_stage90_and_v2_scratch(
    tmp_path: Path,
) -> None:
    assert_input_fence(_config(tmp_path))
    config = _config(tmp_path)
    config.test_cache_root = Path(
        "/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2"
    )
    with pytest.raises(ProtocolError, match="prior Stage-90"):
        assert_input_fence(config)


def test_amendment_chains_directly_to_original_consumption_ledger(tmp_path: Path) -> None:
    parent = tmp_path / "test_consumption_ledger.json"
    parent.write_text(
        """{
  "consumed_decision": "CONFIRMED_WITHIN_CENTER",
  "external_dataset_uncertainty_covered": false,
  "may_be_reused_as_fresh_representation_selection_evidence": false,
  "may_be_reused_for_descriptive_locked-model_scoring": true,
  "new_center_uncertainty_covered": false,
  "observed_centers": 9,
  "row_count": 9928,
  "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
  "split": "test",
  "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION"
}
""",
        encoding="utf-8",
    )
    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker_ledger_amendment_v1.json"
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_bytes(source.read_bytes())
    chain = load_validated_ledger_chain(
        SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            test_consumption_ledger_path=parent,
            ledger_amendment_path=amendment,
        )
    )
    assert chain.amendment["parent_sha256"] == sha256_file(parent)
    assert chain.amendment["previous_stage90_outputs_used"] is False


def test_preflight_propagates_distinct_scratch_root_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake(root: Path, *, runtime: object, expected_scratch_root: str) -> dict[str, object]:
        observed.update(root=root, runtime=runtime, scratch=expected_scratch_root)
        return {"status": "PASS"}

    monkeypatch.setattr(execution_adapter, "_neutral_preflight", fake)
    runtime = {
        "target_task_count": 81,
        "scratch_preference": [execution_adapter.SCRATCH_ROOT, "artifact_parent"],
    }
    assert execution_adapter.run_label_free_workstation_preflight(
        tmp_path, runtime=runtime
    ) == {"status": "PASS"}
    assert observed == {
        "root": tmp_path,
        "runtime": runtime,
        "scratch": "/data/local/fixed_bank_hierarchical_residual_stacker_v1",
    }


def test_exact_nine_aggregation_is_float64_deterministic() -> None:
    rows = tuple(
        RuntimeSeedProbabilityRow(
            target_center="0",
            case_id="case",
            sample_id="sample",
            action_id="B",
            seed_pair_ordinal=index,
            probability=0.1 * index,
            probability_store_hash="a" * 64,
        )
        for index in range(9)
    )
    probabilities, surface_hash = aggregate_exact_nine_probabilities(rows)
    assert probabilities[0].probability == pytest.approx(0.4)
    assert len(surface_hash) == 64
    with pytest.raises(ProtocolError, match="exact-nine"):
        aggregate_exact_nine_probabilities(rows[:-1])


def test_scoped_labels_align_by_canonical_opaque_row_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,case_id,center,split,label\n"
        "raw-private-id,case-0,0,test,1\n",
        encoding="utf-8",
    )
    manager = object.__new__(LabelCapabilityManager)
    manager._manifest_path = manifest
    manager._manifest_sha256 = sha256_file(manifest)
    manager._events = []
    identity = ResidualTestRowIdentity(
        row_ordinal=0,
        manifest_row_index=0,
        evaluation_row_id=evaluation_row_id(EXPECTED_MANIFEST_SHA256, 0),
        case_id="case-0",
        center="0",
    )
    labels = manager._open_rows(
        (identity,), role="terminal_evaluation", target=None, fold=None
    )
    assert labels[0].sample_id == identity.evaluation_row_id
    assert labels[0].sample_id != "raw-private-id"


def test_label_capability_records_all_five_methods_for_one_open_support_fold() -> None:
    manager = object.__new__(LabelCapabilityManager)
    manager._support_opened = {("0", 0)}
    manager._method_decisions = {}
    manager._all_decisions_seal_hash = None
    manager._evaluation_opened = False
    for ordinal, method in enumerate(("B", "B_cal", "G", "R", "P")):
        manager.record_fold_method_decision("0", 0, method, f"{ordinal + 1:064x}")
    assert set(manager._method_decisions) == {
        ("0", 0, method) for method in ("B", "B_cal", "G", "R", "P")
    }


def test_execution_package_has_no_import_from_previous_stage90_science() -> None:
    package = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_hierarchical_residual_stacker"
    )
    forbidden = {
        "fixed_bank_label_aware_case_oof_ceiling",
        "fixed_bank_pooled_bacc_case_oof_ceiling",
    }
    imported: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(token in module for token in forbidden for module in imported)


def test_validation_only_recovery_changes_only_excluded_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _recovery_bundle(tmp_path)
    indexed = ("manifests/content_index.json", *CONTENT_INDEX_MEMBERS)
    before = {member: (root / member).read_bytes() for member in indexed}
    events: list[str] = []
    real_content_validation = validation_recovery.validate_content_index

    def validate_content(root_: Path, *, config_contract_hash: str) -> object:
        events.append("content")
        return real_content_validation(
            root_, config_contract_hash=config_contract_hash
        )

    checks = {"schema_version": "test_validation_v1", "status": "PASS"}

    def validate_semantics(root_: Path, *, config: object) -> object:
        events.append("semantic")
        if len(events) == 3:
            assert read_json(root_ / "reports/run_state.json") == run_state_payload(
                "COMPLETE", "COMPLETE"
            )
            assert read_json(root_ / "reports/validation_report.json") == checks
        else:
            assert read_json(root_ / "reports/run_state.json") == run_state_payload(
                "RUNNING", validation_recovery.VALIDATION_PHASE
            )
            assert not (root_ / "reports/validation_report.json").exists()
        return checks

    monkeypatch.setattr(validation_recovery, "validate_content_index", validate_content)
    monkeypatch.setattr(validation_recovery, "_validate_bundle", validate_semantics)
    assert validation_recovery.recover_fixed_bank_hierarchical_residual_stacker_validation(
        config, artifact_root=root
    ) == root

    assert events == ["content", "semantic", "semantic"]
    assert read_json(root / "reports/run_state.json") == run_state_payload(
        "COMPLETE", "COMPLETE"
    )
    assert read_json(root / "reports/validation_report.json") == checks
    assert {member: (root / member).read_bytes() for member in indexed} == before
    assert not (root / ".run.lock").exists()


@pytest.mark.parametrize(
    ("status", "phase", "error"),
    (
        (
            "RUNNING",
            validation_recovery.VALIDATION_PHASE,
            validation_recovery.RECOVERABLE_VALIDATOR_ERROR,
        ),
        (
            "FAILED",
            "TERMINAL_POOLED_EXACT_BACC_EVALUATION",
            validation_recovery.RECOVERABLE_VALIDATOR_ERROR,
        ),
        (
            "FAILED",
            validation_recovery.VALIDATION_PHASE,
            "ProtocolError: a different failure",
        ),
    ),
)
def test_validation_only_recovery_requires_exact_failed_phase_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    phase: str,
    error: str,
) -> None:
    root, config = _recovery_bundle(tmp_path)
    write_run_state(root, status=status, phase=phase, error=error)
    semantic_calls: list[object] = []
    monkeypatch.setattr(
        validation_recovery,
        "_validate_bundle",
        lambda *args, **kwargs: semantic_calls.append((args, kwargs)),
    )
    with pytest.raises(ProtocolError, match="exact known FAILED"):
        validation_recovery.recover_fixed_bank_hierarchical_residual_stacker_validation(
            config, artifact_root=root
        )
    assert semantic_calls == []
    assert not (root / "reports/validation_report.json").exists()


def test_validation_only_recovery_rejects_indexed_tamper_before_semantic_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _recovery_bundle(tmp_path)
    original_state = (root / "reports/run_state.json").read_bytes()
    (root / "tables/oof_pooled_exact_bacc.csv").write_bytes(b"tampered")
    semantic_calls: list[object] = []
    monkeypatch.setattr(
        validation_recovery,
        "_validate_bundle",
        lambda *args, **kwargs: semantic_calls.append((args, kwargs)),
    )
    with pytest.raises(ProtocolError, match="content-index member drifted"):
        validation_recovery.recover_fixed_bank_hierarchical_residual_stacker_validation(
            config, artifact_root=root
        )
    assert semantic_calls == []
    assert (root / "reports/run_state.json").read_bytes() == original_state
    assert not (root / "reports/validation_report.json").exists()


def test_validation_only_recovery_rolls_back_report_after_final_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _recovery_bundle(tmp_path)
    indexed = ("manifests/content_index.json", *CONTENT_INDEX_MEMBERS)
    before = {member: (root / member).read_bytes() for member in indexed}
    calls = 0

    def validate(root_: Path, *, config: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProtocolError("injected final validation failure")
        return {"schema_version": "test_validation_v1", "status": "PASS"}

    monkeypatch.setattr(validation_recovery, "_validate_bundle", validate)
    with pytest.raises(ProtocolError, match="injected final validation failure"):
        validation_recovery.recover_fixed_bank_hierarchical_residual_stacker_validation(
            config, artifact_root=root
        )
    assert calls == 2
    assert not (root / "reports/validation_report.json").exists()
    assert read_json(root / "reports/run_state.json") == run_state_payload(
        "FAILED",
        validation_recovery.VALIDATION_PHASE,
        error="ProtocolError: injected final validation failure",
    )
    assert {member: (root / member).read_bytes() for member in indexed} == before


def test_validation_only_recovery_requires_resolved_config_inside_exact_root(
    tmp_path: Path,
) -> None:
    root, config = _recovery_bundle(tmp_path)
    config.source_path = tmp_path / "different-config.resolved.yaml"
    with pytest.raises(ProtocolError, match="config/root binding drifted"):
        validation_recovery.recover_fixed_bank_hierarchical_residual_stacker_validation(
            config, artifact_root=root
        )
