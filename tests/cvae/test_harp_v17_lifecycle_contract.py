from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.input_surfaces import (
    SOURCE_TRAIN_ROLE,
    TARGET_EVALUATION_ROLE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.source_label_capability import (
    SOURCE_TRAIN_CAPABILITY_STATE,
    SOURCE_TRAIN_SURFACE_ROLE,
    TARGET_EVALUATION_SURFACE_ROLE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.workspace_paths import (
    SOURCE_TRAIN_REQUIRED_OUTPUT_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.activation_supersession import (
    ACTIVE_SUPERSESSION_CONFIRMATION,
    SUPERSESSION_CONFIRMATION,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V17_EXECUTION_AMENDMENT_GATE,
    HARP_V17_EXPERIMENT_ID,
    HARP_V17_RUN_CONFIRMATION_TOKEN,
)
from midogpp_thesis.cvae.runtime.harp_v17_execution.support_model_artifacts import (
    _as_router_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v17.yaml"
)


def test_v17_config_freezes_pooled_known_center_estimand() -> None:
    config = load_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.execution_authorized is False
    assert config.protocol["source_train_candidate_pool"] == "C_minus_q"
    assert config.protocol["target_evaluation_candidate_pool"] == "C_minus_H"
    assert config.protocol["H_q_r_seven_expert_folds_used"] is False
    assert config.protocol["source_train_case_count"] == 216
    assert config.protocol["target_evaluation_case_count"] == 218
    assert config.protocol["pooled_policy_fit_once_from_source_train_only"] is True
    assert config.protocol["source_train_label_capability_state"] == (
        SOURCE_TRAIN_CAPABILITY_STATE
    )
    assert SOURCE_TRAIN_ROLE == "harp_source_train_development"
    assert TARGET_EVALUATION_ROLE == "harp_full_test_evaluation"
    assert SOURCE_TRAIN_SURFACE_ROLE == "source_train"
    assert TARGET_EVALUATION_SURFACE_ROLE == "target"


def test_v17_model_and_workstation_contract_are_exact() -> None:
    config = load_config(CONFIG)
    model = config.model
    runtime = config.runtime

    assert model["policy_family"] == (
        "pooled_pairwise_selected_policy_with_exact_B_abstention"
    )
    assert model["outer_folds"] == 5
    assert model["inner_folds"] == 4
    assert model["k_values"] == [1, 2, 4]
    assert model["lambda_values"] == [0.25, 0.5, 0.75, 1.0]
    assert model["minimum_routed_oof_cases"] == 18
    assert model["minimum_routed_oof_centers"] == 6
    assert model["minimum_routed_oof_cases_per_center"] == 2
    assert model["source_oof_bound_kind"] == "approximate_not_conformal"
    assert model["u_full_opportunity_head"] == (
        "exact_u_positive_bacc_and_nonnegative_gain"
    )
    assert model["u_full_route_score"] == (
        "opportunity_probability_times_nonnegative_predicted_gain"
    )
    assert model["selected_action_family_route_score"] is True
    assert model["u_full_source_outcome_included"] is True
    assert model["no_nonzero_safe_oof_coverage_aborts_before_target_actions"] is True
    assert runtime["persistent_gpu_workers"] == 2
    assert runtime["classifier_workers"] == 4
    assert runtime["classifier_blas_threads_per_worker"] == 3
    assert runtime["probability_transport_dtype"] == "float32"
    assert runtime["scientific_reduction_dtype"] == "float64"
    parsed = _as_router_config(config)
    assert parsed.outer_folds == 5
    assert parsed.inner_folds == 4
    assert parsed.k_values == (1, 2, 4)
    assert parsed.lambda_values == (0.25, 0.5, 0.75, 1.0)
    assert parsed.minimum_routed_oof_cases == 18
    assert parsed.minimum_routed_oof_centers == 6
    assert parsed.minimum_routed_oof_cases_per_center == 2
    assert parsed.bootstrap_replicates == 1024
    assert parsed.bootstrap_seed == 17017


def test_v17_authority_is_planned_and_revision_owned() -> None:
    assert HARP_V17_EXPERIMENT_ID == EXPERIMENT_ID
    assert HARP_V17_EXECUTION_AMENDMENT_GATE == (
        "harp_v17_consumed_test_execution_amendment_v1"
    )
    assert HARP_V17_RUN_CONFIRMATION_TOKEN == (
        "RUN_HARP_V17_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert PUBLICATION_STATUS == "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    assert TERMINAL_DECISION == "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    amendment = (
        ROOT
        / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
        / "harp_router_v17/harp_stage90_execution_amendment_v17.json"
    )
    assert not amendment.exists()
    assert ACTIVE_SUPERSESSION_CONFIRMATION != SUPERSESSION_CONFIRMATION
    assert {
        "manifests/source_train_menu_seals.json",
        "manifests/target_evaluation_menu_seals.json",
        "manifests/bank_independence_attestations.json",
        "manifests/source_train_label_access_begun.json",
    }.issubset(SOURCE_TRAIN_REQUIRED_OUTPUT_MEMBERS)
