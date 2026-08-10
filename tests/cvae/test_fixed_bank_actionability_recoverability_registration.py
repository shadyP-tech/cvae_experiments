from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli as cli_module
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.config import (
    load_fixed_bank_actionability_recoverability_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.experiment_contracts import (
    AUTHORIZATION_SCOPE,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    PRE_EVALUATION_METHOD_IDS,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    ledger as ledger_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.ledger import (
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.protocol import (
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability_v1.yaml"
)
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability_"
    "ledger_amendment_v1.json"
)
CATALOG_PATH = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"
REGISTRY_PATH = REPOSITORY_ROOT / "experiments/midogpp/registry.yaml"

EXPECTED_REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/actionability_action_probabilities.npz",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/case_oof_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/actionability_prediction_index.json",
    "manifests/actionability_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/prelabel_feature_seal.json",
    "manifests/loco_utility_seals.json",
    "manifests/model_seals.json",
    "manifests/pre_support_decisions_seal.json",
    "manifests/all_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/case_oof_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/case_action_features.csv",
    "tables/loco_utility_targets.csv",
    "tables/model_fits.csv",
    "tables/model_predictions.csv",
    "tables/method_decisions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_method_summary.csv",
    "tables/terminal_contrasts.csv",
    "tables/oracle_rank_metrics.csv",
    "tables/complementarity.csv",
    "tables/rank_stability.csv",
    "tables/permutation_metrics.csv",
    "reports/workstation_preflight.json",
    "reports/phase_01_prelabel_seal_complete.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)


def test_canonical_config_freezes_actions_protocol_and_exact_six_inputs() -> None:
    config = load_fixed_bank_actionability_recoverability_config(CONFIG_PATH)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == "8ca6c33e73144719"
    assert config.protocol["contract_hash"] == (
        canonical_consumed_test_protocol().contract_hash
    )
    assert config.protocol["strict_outer_H_exclusion"] is True
    assert config.protocol["strict_nested_query_q_exclusion"] is True
    assert config.action_library["geometry_ids"] == ["A0", "A1"]
    assert config.action_library["baseline_physical_fit_required"] is True
    assert config.action_library["uniform_physical_fit_required"] is True
    assert config.action_library["A0"]["selected_rows_per_class"] == 256
    assert config.action_library["A0"]["other_rows_per_class"] == 128
    assert config.action_library["A1"]["reuses_exact_A0_row_ids"] is True
    assert config.action_library["A1"]["selected_row_weight_fraction"] == "23/16"
    assert config.action_library["A1"]["other_row_weight_fraction"] == "7/8"
    assert config.action_library["target_probability_cell_count"] == (
        EXPECTED_TARGET_PROBABILITY_CELL_COUNT
    )
    assert config.action_library["action_strength_sweep_used"] is False
    assert config.action_library["geometry_selection_used"] is False
    assert config.recoverability["response"] == (
        "class_balanced_proper_loss_gain_vs_u"
    )
    assert config.recoverability["ridge_alpha"] == 1.0
    assert config.recoverability["G_R_P_fallback_action"] == "U"
    assert config.recoverability["S_y_candidate_set"] == (
        "U_plus_eight_frozen_source_actions_per_geometry"
    )
    assert config.controls["pre_evaluation_method_ids"] == list(
        PRE_EVALUATION_METHOD_IDS
    )
    assert config.controls["terminal_oracles_are_pre_evaluation_methods"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["generation_workers_per_device"] == 1
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["multiprocessing_start_method"] == "spawn"
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["source_storage_dtype"] == "float32"
    assert config.runtime["scientific_reductions_dtype"] == "float64"
    assert config.evaluation["whole_case_cluster_bootstrap_replicates"] == 10_000
    for key in (
        "fresh_evidence",
        "routing_success_claimed",
        "action_selection_authorized",
        "action_geometry_update_authorized",
        "geometry_selection_authorized",
        "policy_update_authorized",
        "model_update_authorized",
        "expert_update_authorized",
        "promotion_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "may_feed_deployable_selection",
    ):
        assert config.claim_boundary[key] is False


def test_amendment_is_byte_bound_direct_single_consumer_and_terminal() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))

    assert sha256_file(AMENDMENT_PATH) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert amendment["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["authorization_scope"] == AUTHORIZATION_SCOPE
    assert amendment["previous_stage90_outputs_used"] is False
    assert amendment["signed_error_output_or_amendment_used"] is False
    assert amendment["terminal_oracles_admitted_as_pre_evaluation_methods"] is False
    assert amendment["geometry_selection_used"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_ledger_chain_accepts_only_its_direct_original_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "reports/test_consumption_ledger.json"
    parent.parent.mkdir(parents=True)
    parent.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
                "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
                "split": "test",
                "may_be_reused_as_fresh_representation_selection_evidence": False,
                "may_be_reused_for_descriptive_locked-model_scoring": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parent_sha = sha256_file(parent)
    amendment_payload = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    amendment_payload["parent_sha256"] = parent_sha
    amendment = tmp_path / "actionability_recoverability_amendment.json"
    amendment.write_text(
        json.dumps(amendment_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ledger_module, "EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256", parent_sha
    )
    monkeypatch.setattr(
        ledger_module,
        "EXPECTED_LEDGER_AMENDMENT_SHA256",
        sha256_file(amendment),
    )

    chain = load_validated_ledger_chain(
        SimpleNamespace(
            experiment_id=EXPERIMENT_ID,
            test_consumption_ledger_path=parent,
            ledger_amendment_path=amendment,
        )
    )

    assert chain.amendment["parent_sha256"] == parent_sha
    assert chain.amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]


def test_ledger_reuse_permission_matches_immutable_parent_schema() -> None:
    published = "may_be_reused_for_descriptive_locked-model_scoring"
    canonical = "may_be_reused_for_descriptive_locked_model_scoring"

    assert ledger_module._descriptive_reuse_permission({published: True}) is True
    assert ledger_module._descriptive_reuse_permission({canonical: True}) is True
    with pytest.raises(ProtocolError, match="descriptive-reuse field is absent"):
        ledger_module._descriptive_reuse_permission(
            {"may_be_reused_for_descriptive_locked-model-scoring": True}
        )
    with pytest.raises(ProtocolError, match="conflicting reuse aliases"):
        ledger_module._descriptive_reuse_permission(
            {published: True, canonical: False}
        )


def test_config_rejects_action_sweep_geometry_selection_and_wrong_amendment(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["action_library"]["action_strength_sweep_used"] = True
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="action library drifted"):
        load_fixed_bank_actionability_recoverability_config(drifted)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["claim_boundary"]["geometry_selection_authorized"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="claim boundary drifted"):
        load_fixed_bank_actionability_recoverability_config(drifted)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["inputs"]["ledger_amendment_path"] = (
        "artifact://midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
        "signed_error_gate_amendment_v1/signed.json"
    )
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="artifact URI drifted"):
        load_fixed_bank_actionability_recoverability_config(drifted)


def test_cli_registers_and_lazily_dispatches_actionability_surface(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_parser().parse_args(
        [
            "fixed-bank-actionability-recoverability",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            "output://actionability-test",
        ]
    )
    assert args.surface == "fixed-bank-actionability-recoverability"

    import midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability as package

    sentinel_config = object()
    observed: dict[str, object] = {}

    def load(path: str) -> object:
        observed["config_path"] = path
        return sentinel_config

    def run(config: object, *, artifact_root: Path) -> Path:
        observed["config"] = config
        observed["artifact_root"] = artifact_root
        return Path("/tmp/actionability-recoverability-test")

    monkeypatch.setattr(
        package,
        "load_fixed_bank_actionability_recoverability_config",
        load,
        raising=False,
    )
    monkeypatch.setattr(
        package,
        "run_fixed_bank_actionability_recoverability",
        run,
        raising=False,
    )

    assert (
        cli_module.main(
            [
                "fixed-bank-actionability-recoverability",
                "--config",
                "actionability.yaml",
                "--artifact-root",
                "/tmp/actionability-output",
            ]
        )
        == 0
    )
    assert observed == {
        "config_path": "actionability.yaml",
        "config": sentinel_config,
        "artifact_root": Path("/tmp/actionability-output"),
    }
    assert capsys.readouterr().out.strip() == (
        "/tmp/actionability-recoverability-test"
    )


def test_registry_catalog_are_single_consumer_closed_world_surfaces() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    by_id = {row["artifact_id"]: row for row in catalog["artifacts"]}
    aliases = (
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
    )
    for artifact_id in aliases:
        semantics = by_id[artifact_id]["semantic_identities"]
        assert semantics["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert semantics["fresh_evidence"] == "false"

    output = by_id[OUTPUT_ARTIFACT_ID]
    assert output["semantic_identities"]["config_contract_hash"] == (
        "8ca6c33e73144719"
    )
    assert output["semantic_identities"]["geometry_selection_used"] == "false"
    assert output["semantic_identities"][
        "terminal_oracles_are_pre_evaluation_methods"
    ] == "false"
    assert REQUIRED_FILES == EXPECTED_REQUIRED_FILES
    assert tuple(output["required_files"]) == EXPECTED_REQUIRED_FILES
    assert "oracle_and_diagnostic_evidence" in output["forbidden_reuse"]
    assert output["may_feed_recipe_selection"] is False
    assert output["may_feed_deployable_selection"] is False

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    experiment = next(
        row
        for row in registry["experiments"]
        if row["experiment_id"] == EXPERIMENT_ID
    )
    assert tuple(experiment["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert experiment["runner"]["argv"][3:5] == [
        "cvae-diagnostics",
        "fixed-bank-actionability-recoverability",
    ]
    assert all(
        OUTPUT_ARTIFACT_ID not in row.get("input_artifact_ids", ())
        for row in registry["experiments"]
    )
    assert not any(
        "fixed_bank_signed_error_gate" in artifact_id
        for artifact_id in experiment["input_artifact_ids"]
    )
