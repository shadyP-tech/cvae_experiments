from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli as cli_module
from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import (
    bundle,
    ledger as ledger_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.config import (
    load_fixed_bank_disagreement_regret_prediction_only_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.experiment_contracts import (
    AUTHORIZATION_SCOPE,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TRAIN_CACHE_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.ledger import (
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.protocol import (
    canonical_prediction_only_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_"
    "prediction_only_v1.yaml"
)
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_"
    "prediction_only_ledger_amendment_v1.json"
)
CATALOG_PATH = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"
REGISTRY_PATH = REPOSITORY_ROOT / "experiments/midogpp/registry.yaml"


def test_config_freezes_six_inputs_and_prediction_only_boundary() -> None:
    config = load_fixed_bank_disagreement_regret_prediction_only_config(CONFIG_PATH)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert not hasattr(config, "source_label_manifest_path")
    assert config.contract_hash == "4d68570f3be01805"
    assert config.protocol["contract_hash"] == (
        canonical_prediction_only_protocol().contract_hash
    )
    assert config.protocol["source_labels_previously_available"] is True
    assert config.protocol["source_authorization_is_fresh_or_unused"] is False
    assert config.protocol["strict_outer_target_H_exclusion"] is True
    assert (
        config.protocol["source_oof_query_q_excluded_from_all_action_compositions"]
        is True
    )
    assert config.protocol["source_oof_physical_classifier_fit_count"] == 5_184
    assert config.protocol["source_oof_oriented_prediction_cell_count"] == 10_368
    assert config.protocol["target_inference_classifier_fit_count"] == 1_458
    assert config.protocol["total_physical_classifier_fit_count_before_test_admission"] == 6_642
    assert config.protocol["nested_query_q_models_used"] is False
    assert config.protocol["candidate_source_e_response_query_excluded"] is True
    assert config.protocol["target_labels_available"] is False
    assert config.protocol["target_scoring_permitted"] is False
    assert config.protocol["all_9928_target_rows_retained"] is True
    assert config.action_library["donor_B_and_U_may_include_H_source_history"] is False
    assert config.action_library["source_oof_B_and_U_exclude_query_q"] is True
    assert config.action_library["source_oof_mass_normalization"] == {
        "B_global_factor": "8/7",
        "U_global_factor": "8/7",
        "A0_global_factor": "9/8",
        "A1_global_factor": "72/65",
        "A1_selected_effective_weight": "207/130",
        "A1_other_effective_weight": "63/65",
        "effective_mass_per_class": {"B": 1024, "U": 1152, "A0": 1152, "A1": 1152},
        "sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
        "label_tuned": False,
    }
    assert config.action_library["target_expert_used"] is False
    assert config.regret_model["model_family_ids"] == ["G", "R", "P"]
    assert config.regret_model["selection_surfaces"] == ["R_raw", "R_safe"]
    assert config.outputs["terminal_metric_table_exists"] is False
    assert config.outputs["raw_source_label_columns_forbidden"] is True
    assert config.claim_boundary["not_routing_success_evidence"] is True
    assert config.claim_boundary["cannot_feed_another_experiment"] is True


def test_amendment_is_direct_singular_and_prediction_only() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))

    assert sha256_file(AMENDMENT_PATH) == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert amendment["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert amendment["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert amendment["authorization_scope"] == AUTHORIZATION_SCOPE
    assert amendment["source_labels_previously_available"] is True
    assert amendment["source_labels_are_fresh_or_unused"] is False
    assert amendment["source_oof_query_q_excluded_from_all_action_compositions"] is True
    assert amendment["source_oof_physical_classifier_fit_count"] == 5_184
    assert amendment["source_oof_oriented_prediction_cell_count"] == 10_368
    assert amendment["target_inference_classifier_fit_count"] == 1_458
    assert amendment["total_physical_classifier_fit_count_before_test_admission"] == 6_642
    assert amendment["target_labels_available"] is False
    assert amendment["target_bacc_accuracy_regret_utility_or_oracle_computed"] is False
    assert amendment["may_feed_another_experiment"] is False


def test_ledger_accepts_only_direct_original_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    amendment = tmp_path / "prediction_only_amendment.json"
    amendment.write_text(
        json.dumps(amendment_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ledger_module,
        "EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256",
        parent_sha,
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


def test_config_rejects_target_scoring_and_nested_model_drift(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["protocol"]["target_scoring_permitted"] = True
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="protocol drifted"):
        load_fixed_bank_disagreement_regret_prediction_only_config(drifted)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["regret_model"]["nested_query_q_models_used"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="regret model drifted"):
        load_fixed_bank_disagreement_regret_prediction_only_config(drifted)


def test_catalog_inventory_and_aliases_are_closed_world() -> None:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    by_id = {row["artifact_id"]: row for row in catalog["artifacts"]}
    assert (
        "midogpp_stage90_fixed_bank_disagreement_regret_prediction_only_"
        "source_label_manifest_v1"
        not in by_id
    )
    assert by_id[OUTPUT_ARTIFACT_ID]["required_files"] == list(
        bundle.REQUIRED_FILES
    )
    aliases = (
        TRAIN_CACHE_ARTIFACT_ID,
        TEST_CACHE_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
    )
    for artifact_id in aliases:
        assert by_id[artifact_id]["semantic_identities"][
            "authorized_consumer_experiment_ids"
        ] == EXPERIMENT_ID
    output_semantics = by_id[OUTPUT_ARTIFACT_ID]["semantic_identities"]
    assert output_semantics["target_labels_available"] == "false"
    assert output_semantics["target_scoring_permitted"] == "false"
    assert output_semantics["prediction_output_is_policy"] == "false"


def test_registry_and_cli_register_prediction_only_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    experiment = next(
        row for row in registry["experiments"] if row["experiment_id"] == EXPERIMENT_ID
    )
    assert tuple(experiment["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert experiment["runner"]["argv"][4] == (
        "fixed-bank-disagreement-regret-prediction-only"
    )
    assert all(
        OUTPUT_ARTIFACT_ID not in row.get("input_artifact_ids", ())
        for row in registry["experiments"]
    )
    args = build_parser().parse_args(
        [
            "fixed-bank-disagreement-regret-prediction-only",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            "output://prediction-only-test",
        ]
    )
    assert args.surface == "fixed-bank-disagreement-regret-prediction-only"

    import midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only as package

    sentinel = object()
    observed: dict[str, object] = {}

    def load(path: str) -> object:
        observed["config_path"] = path
        return sentinel

    def run(config: object, *, artifact_root: Path) -> Path:
        observed["config"] = config
        observed["artifact_root"] = artifact_root
        return Path("/tmp/prediction-only-test")

    monkeypatch.setitem(
        package.__dict__,
        "load_fixed_bank_disagreement_regret_prediction_only_config",
        load,
    )
    monkeypatch.setitem(
        package.__dict__,
        "run_fixed_bank_disagreement_regret_prediction_only",
        run,
    )
    assert (
        cli_module.main(
            [
                "fixed-bank-disagreement-regret-prediction-only",
                "--config",
                "prediction-only.yaml",
                "--artifact-root",
                "/tmp/prediction-only-output",
            ]
        )
        == 0
    )
    assert observed == {
        "config_path": "prediction-only.yaml",
        "config": sentinel,
        "artifact_root": Path("/tmp/prediction-only-output"),
    }
    assert capsys.readouterr().out.strip() == "/tmp/prediction-only-test"
