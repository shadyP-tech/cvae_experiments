from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import constants
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.config import (
    load_fixed_bank_support_static_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.protocol import (
    SupportStaticRouterProtocol,
    assert_terminal_consumed_test_protocol,
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4_"
    "ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_s4_protocol_controls_and_workstation() -> None:
    config = load_fixed_bank_support_static_router_config(CONFIG)

    assert config.experiment_id == contracts.EXPERIMENT_ID
    assert config.output_artifact_id == contracts.OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == "d0830988c454be0d"
    assert config.protocol["publication_status"] == (
        "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    )
    assert config.protocol["fresh_evidence"] is False
    assert config.protocol["support_scope"] == (
        "other_four_same_H_whole_case_folds"
    )
    assert config.protocol[
        "each_H_f_decision_and_seal_precedes_opening_same_H_f_evaluation_role_labels"
    ] is True
    assert config.action_library["physical_action_count_per_target"] == 10
    assert config.action_library["target_probability_cell_count"] == 810
    assert config.action_library["U_is_internal_control_not_selection_candidate"]
    assert config.support_router["method_id"] == "S4"
    assert config.support_router["support_selection_objective"] == (
        "pooled_exact_bacc_gain_vs_B"
    )
    assert config.support_router["tie_tolerance"] == 1.0e-12
    assert config.support_router["single_class_support_falls_back_to_B"] is True
    assert config.support_router["G_static_definition"] == (
        "equal_center_mean_exact_gain_over_q_not_in_H_or_e"
    )
    assert config.support_router["G_static_donor_query_scope"] == "q_not_in_H_or_e"
    assert config.support_router["G_static_candidate_gain_aggregation"] == (
        "equal_center_mean"
    )
    for key in (
        "case_features_used",
        "donor_model_used",
        "target_local_calibration_used",
        "shared_model_fit_used",
        "hyperparameter_search_used",
    ):
        assert config.support_router[key] is False
    assert config.controls["method_ids"] == [
        "B",
        "U",
        "G_static",
        "S4",
        "O_static",
        "O_case",
    ]
    assert config.evaluation["descriptive_interval_degrees_of_freedom"] == 8
    assert config.evaluation["permutation_null_count"] == 10_000
    assert config.evaluation["null_selection_plan_row_count"] == 450_000
    assert config.evaluation["confirmatory_p_value_computed"] is False
    assert config.evaluation["confirmatory_gate_defined"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["target_probability_cell_count"] == 810
    assert config.runtime["two_fresh_process_validation_required"] is True
    assert config.runtime["scratch_preference"][0] == contracts.SCRATCH_ROOT
    for key in (
        "fresh_evidence",
        "routing_success_claimed",
        "routing_quality_claimed",
        "action_selection_authorized",
        "policy_update_authorized",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "confirmatory_p_value_or_gate_used",
    ):
        assert config.claim_boundary[key] is False


def test_science_constants_and_registration_contract_cannot_drift_independently() -> None:
    config = load_fixed_bank_support_static_router_config(CONFIG)

    assert constants.CENTERS == contracts.CENTERS == tuple(config.protocol["centers"])
    assert constants.OOF_FOLD_COUNT == contracts.OOF_FOLD_COUNT == config.fold_count
    assert constants.PARTITION_SEED == contracts.PARTITION_SEED
    assert constants.PARTITION_NAMESPACE == contracts.PARTITION_NAMESPACE == (
        config.protocol["partition_namespace"]
    )
    assert constants.METHOD_IDS == contracts.METHOD_IDS == tuple(
        config.controls["method_ids"]
    )
    assert constants.PRE_EVALUATION_METHOD_IDS == (
        contracts.PRE_EVALUATION_METHOD_IDS
    ) == tuple(config.controls["pre_evaluation_method_ids"])
    assert constants.TERMINAL_ORACLE_IDS == contracts.TERMINAL_ORACLE_IDS == tuple(
        config.controls["terminal_oracle_ids"]
    )
    assert constants.ACTION_COUNT_PER_TARGET == contracts.ACTION_COUNT_PER_TARGET == (
        config.action_library["physical_action_count_per_target"]
    )
    assert constants.TIE_TOLERANCE == contracts.TIE_TOLERANCE == (
        config.support_router["tie_tolerance"]
    )
    assert constants.PERMUTATION_COUNT == contracts.PERMUTATION_COUNT == (
        config.evaluation["permutation_null_count"]
    )
    assert constants.PERMUTATION_SEED == contracts.PERMUTATION_SEED == (
        config.evaluation["permutation_seed"]
    )
    assert constants.NULL_DERANGEMENT_ALGORITHM == (
        contracts.NULL_DERANGEMENT_ALGORITHM
    ) == config.evaluation["permutation_algorithm_id"]
    assert constants.PUBLICATION_STATUS == contracts.PUBLICATION_STATUS
    assert constants.CLAIM_ROLE == contracts.CLAIM_ROLE
    assert constants.TERMINAL_DECISION == contracts.TERMINAL_DECISION


def test_amendment_is_direct_single_consumer_hash_chain_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        contracts.EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert payload["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert payload["parent_sha256"] == (
        contracts.EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    assert payload["authorized_consumer_experiment_ids"] == [
        contracts.EXPERIMENT_ID
    ]
    assert payload["authorization_scope"] == contracts.AUTHORIZATION_SCOPE
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["previous_stage90_amendments_used"] is False
    assert payload["previous_prediction_surfaces_used"] is False
    assert payload["previous_stage90_scratch_or_checkpoints_used"] is False
    assert payload["single_class_support_falls_back_to_B"] is True
    assert payload[
        "each_H_f_decision_and_seal_precedes_opening_same_H_f_evaluation_role_labels"
    ] is True
    assert payload["G_static_definition"] == (
        "equal_center_mean_exact_gain_over_q_not_in_H_or_e"
    )
    assert payload["permutation_keeps_B_fixed"] is True
    assert payload["permutation_changes_labels"] is False
    assert payload["confirmatory_p_value_computed"] is False
    assert payload["confirmatory_gate_used"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_stage90"] is False
    assert payload["may_feed_another_experiment"] is False


def test_registry_catalog_aliases_output_and_closed_world_are_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert experiment.runner_argv[3:5] == (
        "cvae-diagnostics",
        "fixed-bank-support-static-router",
    )
    for artifact_id in (
        contracts.TEST_CACHE_ARTIFACT_ID,
        contracts.TEST_MANIFEST_ARTIFACT_ID,
        contracts.TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        contracts.LEDGER_AMENDMENT_ARTIFACT_ID,
    ):
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == contracts.EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4/v1"
    )
    assert output.semantic_identities["config_contract_hash"] == (
        "d0830988c454be0d"
    )
    assert output.semantic_identities["method_ids"] == (
        "B|U|G_static|S4|O_static|O_case"
    )
    assert output.semantic_identities["single_class_support_falls_back_to_B"] == (
        "true"
    )
    assert output.semantic_identities["confirmatory_p_value_or_gate_used"] == (
        "false"
    )
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 40
    assert "arrays/action_identity_null_selections.npz" in REQUIRED_FILES
    assert (
        "manifests/action_identity_null_selection_plan_seal.json"
        in REQUIRED_FILES
    )
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        contracts.OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_exact_six_input_fence_and_config_drift_fail_closed(tmp_path: Path) -> None:
    config = load_fixed_bank_support_static_router_config(CONFIG)
    assert_input_fence(config)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["support_router"]["single_class_support_falls_back_to_B"] = False
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: support_router"):
        load_fixed_bank_support_static_router_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["evaluation"]["confirmatory_p_value_computed"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: evaluation"):
        load_fixed_bank_support_static_router_config(drifted)


def test_protocol_manifest_rejects_any_boundary_widening() -> None:
    protocol = canonical_consumed_test_protocol()
    assert_terminal_consumed_test_protocol(protocol)
    assert protocol.contract_hash == "13544418e1a29d1a"

    widened = dict(protocol.payload)
    widened["fresh_evidence"] = True
    with pytest.raises(ProtocolError, match="protocol contract drifted"):
        SupportStaticRouterProtocol(widened, protocol.contract_hash)
    with pytest.raises(ProtocolError, match="protocol contract drifted"):
        assert_terminal_consumed_test_protocol(
            replace(protocol, contract_hash="0" * 16)
        )


def test_cli_registers_and_lazily_dispatches_s4(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-support-static-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-support-static-router-s4-v1",
        )
    )
    assert parsed.surface == "fixed-bank-support-static-router"

    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_support_static_router_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-support-static-router-result-v1")

    monkeypatch.setattr(surface, "run_fixed_bank_support_static_router", _run)
    assert cli.main(
        [
            "fixed-bank-support-static-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-support-static-router-s4-v1",
        ]
    ) == 0
    assert calls == [
        (sentinel, Path("/tmp/fixed-bank-support-static-router-s4-v1"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-support-static-router-result-v1"
    )
