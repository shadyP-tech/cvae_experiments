from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    SUPPORT_ROLE,
    TARGET_EVALUATION_ROLE,
    TARGET_TRAIN_SUPPORT_ROLE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.source_label_capability import (
    SUPPORT_CAPABILITY_STATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v16.workspace_paths import (
    TARGET_SUPPORT_REQUIRED_OUTPUT_MEMBERS,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V16_EXECUTION_AMENDMENT_GATE,
    HARP_V16_EXPERIMENT_ID,
    HARP_V16_RUN_CONFIRMATION_TOKEN,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v16.yaml"
)


def test_v16_config_declares_target_support_full_test_contract() -> None:
    config = load_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.execution_authorized is False
    assert SUPPORT_ROLE == TARGET_TRAIN_SUPPORT_ROLE == "target_train_support"
    assert DEVELOPMENT_ROLE == SUPPORT_ROLE
    assert EVALUATION_ROLE == TARGET_EVALUATION_ROLE == "target_test_evaluation"
    assert config.protocol["support_label_capability_state"] == SUPPORT_CAPABILITY_STATE
    assert config.protocol["same_center_support_query_q_equals_H"] is True
    assert config.protocol["candidate_source_pool"] == "C_MINUS_H"
    assert config.protocol["support_context_count"] == 9
    assert config.protocol["target_context_count"] == 9
    assert config.protocol["joint_support_target_classifier_task_count"] == 81
    assert config.protocol["total_classifier_fit_count"] == 810
    assert config.protocol["source_H_q_r_crossfit_used"] is False


def test_v16_model_is_direct_support_calibrated_not_pairwise() -> None:
    model = load_config(CONFIG).model

    assert model["feature_set"] == "fixed_mechanism_feature_priority_v16"
    assert model["maximum_numeric_features"] == 20
    assert model["endpoint_estimator"] == "direct_case_balanced_ridge_heads"
    assert model["ridge_alpha"] == 4.0
    assert model["minimum_support_cases"] == 12
    assert model["calibration_alpha"] == 0.20
    assert model["residual_unit"] == "case_level_max_across_active_actions"
    assert model["per_action_worst_support_fold_certificate_used"] is False
    assert model["selection_rule"] == (
        "highest_gain_lower_bound_action_passing_all_endpoint_certificates_else_exact_B"
    )
    assert not any("pairwise" in key or "ranker" in key for key in model)


def test_v16_workspace_is_planned_and_uses_center_keyed_physical_stores() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.preparation_authority_gate == HARP_V16_EXECUTION_AMENDMENT_GATE
    assert HARP_V16_EXPERIMENT_ID == EXPERIMENT_ID
    assert HARP_V16_RUN_CONFIRMATION_TOKEN == (
        "RUN_HARP_V16_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert {
        "manifests/target_support_menu_seals.json",
        "manifests/target_evaluation_menu_seals.json",
        "manifests/target_bank_independence_attestations.json",
        "manifests/support_label_access_begun.json",
    }.issubset(TARGET_SUPPORT_REQUIRED_OUTPUT_MEMBERS)
    for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9"):
        assert f"stores/physical_menu/outer_{center}/manifest.json" in (
            TARGET_SUPPORT_REQUIRED_OUTPUT_MEMBERS
        )
        assert f"stores/physical_menu/outer_{center}/arrays.npz" in (
            TARGET_SUPPORT_REQUIRED_OUTPUT_MEMBERS
        )
    assert not any(
        member.startswith("stores/target_support_physical_menus/")
        for member in TARGET_SUPPORT_REQUIRED_OUTPUT_MEMBERS
    )


def test_v16_output_registration_requires_every_durable_prelabel_binding() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    required = set(workspace.artifacts[OUTPUT_ARTIFACT_ID].required_files)
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")

    expected = {
        "manifests/action_capacity_certificate.json",
        "manifests/center_menu_root_binding.json",
        "stores/label_free_compatibility/manifest.json",
        "stores/label_free_compatibility/arrays.npz",
        "manifests/support_target_role_seals/"
        "fixed_bank_support_independence_attestation.json",
        "manifests/support_policy_admission_seal.json",
        "manifests/support_label_access_begun.json",
    }
    for center in centers:
        expected.update(
            {
                "manifests/support_target_role_seals/"
                f"H{center}/target_train_support_menu_seal.json",
                "manifests/support_target_role_seals/"
                f"H{center}/target_test_evaluation_menu_seal.json",
            }
        )

    assert expected.issubset(required)
