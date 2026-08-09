from __future__ import annotations

import inspect
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling import (
    execution_adapter,
    ledger as ledger_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.experiment_contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.ledger import (
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.pooled_prior import (
    PriorConfig,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.core_contracts import (
    BinaryLabelRow,
    CaseIdentityRow,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.case_partitions import (
    build_case_oof_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.label_capabilities import (
    audit_manifest_case_class_topology,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.experiment_contracts import (
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.cvae.runtime.preflight import run_label_free_workstation_preflight


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root=tmp_path / "expert_bank",
        generation_lock_root=tmp_path / "generation_lock",
        test_cache_root=tmp_path / "dedicated_pooled_cache_v2",
        test_manifest_path=tmp_path / "dedicated_pooled_manifest_v2" / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger_parent" / "reports" / "test_consumption_ledger.json",
        ledger_amendment_path=tmp_path / "pooled_amendment_v2" / "amendment.json",
    )


def test_original_six_input_fence_accepts_only_v2_dedicated_aliases(tmp_path: Path) -> None:
    assert_input_fence(_config(tmp_path))

    wrong = _config(tmp_path)
    wrong.input_artifact_ids = (*INPUT_ARTIFACT_IDS[:-1], "rogue_stage90_output")
    with pytest.raises(ProtocolError, match="exact six"):
        assert_input_fence(wrong)


def test_v1_output_and_scratch_are_quarantined_even_with_six_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.test_cache_root = Path(
        "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1"
    )
    with pytest.raises(ProtocolError, match="cannot consume v1"):
        assert_input_fence(config)


def test_published_v2_amendment_is_hash_chained_to_immutable_parent(tmp_path: Path) -> None:
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
        / "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling_ledger_amendment_v2.json"
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_bytes(source.read_bytes())
    assert sha256_file(parent) == "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
    chain = load_validated_ledger_chain(
        SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            test_consumption_ledger_path=parent,
            ledger_amendment_path=amendment,
        )
    )
    assert chain.amendment["v1_output_used"] is False
    assert chain.amendment["support_utility"] == "pooled_exact_bacc"


def test_ledger_rejects_restricted_null_semantic_drift_even_with_spoofed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        / "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling_ledger_amendment_v2.json"
    )
    canonical = json.loads(source.read_text(encoding="utf-8"))
    amendment = tmp_path / "amendment.json"
    config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        test_consumption_ledger_path=parent,
        ledger_amendment_path=amendment,
    )
    real_sha256_file = ledger_module.sha256_file

    def spoof_amendment_hash(path: Path) -> str:
        if Path(path) == amendment:
            return ledger_module.EXPECTED_LEDGER_AMENDMENT_SHA256
        return real_sha256_file(Path(path))

    monkeypatch.setattr(ledger_module, "sha256_file", spoof_amendment_hash)
    semantic_drifts = {
        "permutation_primary_statistic": "fold_mean_R_minus_G_H",
        "permutation_upper_tail_output_field": "upper_tail_p_value",
        "permutation_lower_tail_output_field": "left_tail_p_value",
        "permutation_two_sided_output_field": "absolute_tail_p_value",
        "permutation_upper_tail_p_value_formula": "legacy_upper_tail",
        "permutation_lower_tail_p_value_formula": "legacy_lower_tail",
        "permutation_two_sided_p_value_formula": "absolute_null_tail",
        "permutation_derangement_family": "uniform_all_derangements",
        "permutation_candidate_order": "shared_candidate_order",
        "permutation_shift_generator": "stateful_rng",
        "permutation_shift_range_inclusive": [0, 7],
        "permutation_zero_shift_allowed": True,
        "uniform_over_all_derangements": True,
    }
    for key, drifted in semantic_drifts.items():
        payload = dict(canonical)
        payload[key] = drifted
        amendment.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ProtocolError, match="amendment chain"):
            load_validated_ledger_chain(config)


def test_v2_preflight_wrapper_propagates_distinct_scratch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake(root: Path, *, runtime: object, expected_scratch_root: str) -> dict[str, object]:
        observed.update(root=root, runtime=runtime, scratch=expected_scratch_root)
        return {"status": "PASS"}

    monkeypatch.setattr(execution_adapter, "_neutral_preflight", fake)
    runtime = {"scratch_preference": [execution_adapter.V2_SCRATCH_ROOT, "artifact_parent"]}
    assert execution_adapter.run_label_free_workstation_preflight(
        tmp_path, runtime=runtime
    ) == {"status": "PASS"}
    assert observed["scratch"] == "/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2"


def test_neutral_preflight_default_remains_v1_backward_compatible() -> None:
    default = inspect.signature(run_label_free_workstation_preflight).parameters[
        "expected_scratch_root"
    ].default
    assert default == "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1"


def test_declared_prior_tie_tolerance_is_explicit_and_fail_closed() -> None:
    assert PriorConfig(tie_tolerance=1.0e-12).to_payload()["tie_tolerance"] == 1.0e-12
    with pytest.raises(ProtocolError, match="tie tolerance"):
        PriorConfig(tie_tolerance=1.0e-6)


def test_manifest_topology_retains_213_mixed_four_negative_only_and_one_positive_only() -> None:
    negative_only = {("0", "case-000"), ("1", "case-000"), ("2", "case-000"), ("3", "case-000")}
    positive_only = {("5", "case-000")}
    identities: list[CaseIdentityRow] = []
    labels: list[BinaryLabelRow] = []
    for center, count in EXPECTED_CASE_COUNTS_BY_CENTER.items():
        for ordinal in range(count):
            case_id = f"case-{ordinal:03d}"
            key = (center, case_id)
            case_labels = (0,) if key in negative_only else (1,) if key in positive_only else (0, 1)
            for sample_ordinal, label in enumerate(case_labels):
                sample_id = f"H{center}-{case_id}-sample-{sample_ordinal}"
                identities.append(CaseIdentityRow(center, case_id, sample_id))
                labels.append(BinaryLabelRow(center, case_id, sample_id, label))
    partition = build_case_oof_partition(identities, partition_seed=90_902_026)
    report = audit_manifest_case_class_topology(tuple(labels), partition=partition)
    assert report["total_case_count"] == 218
    assert report["mixed_class_case_count"] == 213
    assert report["negative_only_case_count"] == 4
    assert report["positive_only_case_count"] == 1
    assert report["every_support_scope_has_both_classes"] is True
    assert report["every_evaluation_scope_has_both_classes"] is True


def test_cpu_staging_rejects_v1_or_noncanonical_scratch(tmp_path: Path) -> None:
    config = SimpleNamespace(
        runtime={
            "scratch_preference": [
                "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1",
                "artifact_parent",
            ]
        }
    )
    with pytest.raises(ProtocolError, match="scratch preference"):
        execution_adapter.stage_sources_for_cpu(object(), config=config, root=tmp_path)


def test_v2_and_neutral_runtime_import_fences_exclude_prior_diagnostics_and_routing() -> None:
    root = Path(__file__).resolve().parents[2] / "src/midogpp_thesis/cvae"
    package = root / "diagnostics/fixed_bank_pooled_bacc_case_oof_ceiling"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert node.level != 2, f"{path.name} imports a diagnostics sibling"
                assert ".routing" not in module and not module.startswith("routing")
                if "midogpp_thesis.cvae.diagnostics." in module:
                    assert module.startswith(
                        "midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert ".routing" not in alias.name
                    if "midogpp_thesis.cvae.diagnostics." in alias.name:
                        assert alias.name.startswith(
                            "midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling"
                        )
    for path in sorted((root / "runtime").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [module, *(alias.name for alias in node.names)]
                assert all("diagnostics" not in name and ".routing" not in name for name in names)
