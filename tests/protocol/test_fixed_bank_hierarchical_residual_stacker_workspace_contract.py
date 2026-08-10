from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.config import (
    load_fixed_bank_hierarchical_residual_stacker_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.config_payloads import (
    MODEL_FEATURE_NAMES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    FORBIDDEN_PRIOR_STAGE90_ARTIFACT_IDS,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker_"
    "ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_architecture_math_controls_and_claim_boundary() -> None:
    config = load_fixed_bank_hierarchical_residual_stacker_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert not set(config.input_artifact_ids).intersection(
        FORBIDDEN_PRIOR_STAGE90_ARTIFACT_IDS
    )
    assert config.contract_hash == "cb7050fcdaac86ac"
    assert config.feature_names == MODEL_FEATURE_NAMES
    assert config.stacker["diagnostic_method_ids"] == ["B", "B_cal", "G", "R", "P"]
    assert config.probability_surface["global_source_control_input"] == (
        "sealed_probabilities_only"
    )
    assert config.probability_surface[
        "global_source_control_metadata_or_label_input"
    ] is False
    assert config.probability_surface["training_row_source_s_control_mask"] == (
        "query_u_not_in_H_or_e_or_s"
    )
    assert config.probability_surface[
        "nested_training_row_source_s_control_mask"
    ] == "query_u_not_in_H_or_e_or_q_or_s"
    assert config.hierarchical_model["primary_interaction_rank"] == 1
    assert config.hierarchical_model["rank_two_challenger_included"] is False
    assert config.hierarchical_model["learned_candidate_identity_factor_used"] is False
    assert config.hierarchical_model["alpha_selection_objective"] == (
        "nested_legal_query_class_count_weighted_squared_error_on_smooth_"
        "class_effect_responses"
    )
    assert config.hierarchical_model["permuted_control_is_separately_fit"] is True
    assert config.hierarchical_model["permuted_control_reuses_R_coefficients"] is False
    assert config.features["permutation_applied_before_donor_fit"] is True
    assert config.features["permutation_applied_before_target_inference"] is True
    assert config.features[
        "permutation_phi_donor_source_must_be_legal_under_same_H_e_q_mask"
    ] is True
    assert config.target_support["B_cal_selection_objective"] == (
        "fixed_class_balanced_log_loss"
    )
    assert config.target_support["residual_lambda_selection_objective"] == (
        "fixed_class_balanced_log_loss"
    )
    assert config.target_support["exact_pooled_bacc_used_for_grid_selection"] is False
    assert config.target_support["exact_pooled_bacc_used_for_safety_gate"] is True
    assert config.variance_floor == 1.0e-6
    assert config.stacker["class_gate"] == "soft_B_cal_probability"
    assert config.stacker["composed_residual"] == (
        "(1-p_B_cal)*sum_e_alpha0_r_e_plus_p_B_cal*sum_e_alpha1_r_e"
    )
    assert config.evaluation["primary_contrasts"] == ["R-B_cal", "R-G", "R-P"]
    assert config.claim_boundary["claim_role"] == (
        "known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic"
    )
    for key in (
        "fresh_evidence",
        "routing_quality_claimed",
        "target_performance_claimed",
        "source_expert_updated",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "action_selection_authorized",
        "policy_update_authorized",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
        "metadata_artifact_used",
        "previous_stage90_outputs_used",
        "previous_prediction_surface_used",
        "previous_stage90_scratch_or_checkpoint_used",
    ):
        assert config.claim_boundary[key] is False


def test_amendment_is_direct_single_consumer_hash_chain_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_artifact_id"] == "midogpp_uniform_b_test_consumption_ledger_v1"
    assert payload["parent_sha256"] == (
        "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
    )
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["control_method_ids"] == ["B", "B_cal", "G", "R", "P"]
    assert payload["support_selection_surrogate"] == (
        "fixed_class_balanced_log_loss_only"
    )
    assert payload["feature_permutation_applied_before_donor_fit"] is True
    assert payload["feature_permutation_applied_before_target_inference"] is True
    assert payload["feature_permutation_refits_same_capacity_model"] is True
    assert payload["soft_class_gate_avoids_hard_pseudo_class_sign_reversal"] is True
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_stage90"] is False


def test_registry_catalog_aliases_and_output_are_experiment_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.runner_argv[3:5] == (
        "cvae-diagnostics",
        "fixed-bank-hierarchical-residual-stacker",
    )
    for artifact_id in (
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
    ):
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker/v1"
    )
    assert output.semantic_identities["config_contract_hash"] == "cb7050fcdaac86ac"
    assert output.semantic_identities["diagnostic_method_ids"] == "B|B_cal|G|R|P"
    assert output.semantic_identities["strict_inner_H_q_e_exclusion"] == "true"
    assert output.semantic_identities["metadata_artifact_used"] == "false"
    assert output.semantic_identities[
        "feature_permutation_refits_same_capacity_model"
    ] == "true"
    assert output.semantic_identities["primary_contrasts"] == "R-B_cal|R-G|R-P"
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 39
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_input_fence_rejects_prior_stage90_output_and_non_six_input_surface() -> None:
    config = load_fixed_bank_hierarchical_residual_stacker_config(CONFIG)
    assert_input_fence(config)

    with pytest.raises(ProtocolError, match="cannot consume prior Stage-90"):
        assert_input_fence(
            replace(
                config,
                test_cache_root=Path(
                    "artifacts/midogpp/90_oracles_and_diagnostics/"
                    "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_"
                    "case_oof_ceiling/v2"
                ),
            )
        )
    with pytest.raises(ProtocolError, match="exact six fenced inputs"):
        assert_input_fence(
            SimpleNamespace(
                experiment_id=config.experiment_id,
                output_artifact_id=config.output_artifact_id,
                input_artifact_ids=(*INPUT_ARTIFACT_IDS, "unexpected"),
                expert_bank_root=config.expert_bank_root,
                generation_lock_root=config.generation_lock_root,
                test_consumption_ledger_path=config.test_consumption_ledger_path,
                ledger_amendment_path=config.ledger_amendment_path,
                test_cache_root=config.test_cache_root,
                test_manifest_path=config.test_manifest_path,
            )
        )


def test_config_rejects_feature_provenance_or_proper_loss_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["probability_surface"]["global_source_control_input"] = "metadata"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="probability surface drifted"):
        load_fixed_bank_hierarchical_residual_stacker_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["target_support"]["B_cal_selection_objective"] = "exact_bacc"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="target support drifted"):
        load_fixed_bank_hierarchical_residual_stacker_config(drifted)


def test_cli_parser_exposes_residual_stacker_surface() -> None:
    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-hierarchical-residual-stacker",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-hierarchical-residual-stacker-v1",
        )
    )
    assert parsed.surface == "fixed-bank-hierarchical-residual-stacker"


def test_cli_lazy_dispatches_to_residual_stacker_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_hierarchical_residual_stacker_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-hierarchical-residual-stacker-result-v1")

    monkeypatch.setattr(surface, "run_fixed_bank_hierarchical_residual_stacker", _run)
    assert cli.main(
        [
            "fixed-bank-hierarchical-residual-stacker",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-hierarchical-residual-stacker-root-v1",
        ]
    ) == 0
    assert calls == [
        (sentinel, Path("/tmp/fixed-bank-hierarchical-residual-stacker-root-v1"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-hierarchical-residual-stacker-result-v1"
    )


def test_config_modules_keep_execution_imports_lazy() -> None:
    package = (
        ROOT
        / "src/midogpp_thesis/cvae/diagnostics/"
        "fixed_bank_hierarchical_residual_stacker"
    )
    for member in ("experiment_contracts.py", "config_payloads.py", "config_validation.py", "config.py"):
        source = (package / member).read_text(encoding="utf-8")
        assert "import .runner" not in source
        assert "from .runner import" not in source
        assert "execution_phases" not in source
