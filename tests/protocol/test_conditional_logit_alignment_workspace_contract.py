from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = "midogpp.real_feature.conditional_logit_alignment.v1"
OUTPUT_ID = "midogpp_output_real_feature_conditional_logit_alignment_v1"
INPUT_IDS = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
)
CONFIG_PATH = (
    "experiments/midogpp/stages/10_real_feature_reference/configs/"
    "conditional_logit_alignment_v1.yaml"
)
REQUIRED_BUNDLE = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/content_index.json",
    "reports/leakage_provenance_report.json",
    "reports/decision_summary.json",
    "reports/decision_report.md",
    "reports/runtime_summary.json",
    "tables/source_inner_fold_scores.csv",
    "tables/source_inner_gamma_summary.csv",
    "tables/outer_results.csv",
    "tables/outer_predictions.csv",
    "tables/conditional_frame_audit.csv",
    "tables/solver_audit.csv",
    "tables/outer_comparison.csv",
}


def test_conditional_logit_alignment_is_registered_as_runnable_diagnostic() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    artifact = workspace.artifacts[OUTPUT_ID]
    known_reuse_purposes = {
        purpose
        for stage in workspace.stages.values()
        for purpose in stage["input_reuse_purposes"]
    }

    assert experiment.status == "diagnostic"
    assert experiment.runnable is True
    assert experiment.stage == "10_real_feature_reference"
    assert experiment.claim_scope == "real_feature_transfer_only"
    assert experiment.config_path == CONFIG_PATH
    assert experiment.output_artifact_id == OUTPUT_ID
    assert experiment.input_artifact_ids == INPUT_IDS
    assert experiment.runner_argv[-4:] == (
        "real-feature-classifier",
        "conditional-logit-alignment",
        "--config",
        "{resolved_config}",
    )
    assert experiment.runner_env == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    assert artifact.canonical_path == (
        "artifacts/midogpp/10_real_feature_reference/"
        "conditional_logit_alignment_v1/seed42"
    )
    assert set(artifact.required_files) == REQUIRED_BUNDLE
    assert len(artifact.required_files) == 16
    assert set(artifact.forbidden_reuse) == (
        known_reuse_purposes - {"oracle_and_diagnostic_evidence"}
    )
    assert artifact.may_feed_recipe_selection is False
    assert artifact.may_feed_deployable_selection is False

def test_conditional_logit_alignment_config_freezes_nested_source_only_design() -> None:
    workspace = MidogppWorkspace.load()
    config = yaml.safe_load((workspace.repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    used_inputs: set[str] = set()
    resolved = workspace.resolve_value(config, require_inputs=False, used_inputs=used_inputs)

    assert used_inputs == set(INPUT_IDS)
    assert resolved["experiment"]["artifact_root"].endswith(
        "artifacts/midogpp/10_real_feature_reference/"
        "conditional_logit_alignment_v1/seed42"
    )
    assert config["inputs"]["split"] == "train"
    assert config["classifier"] == {
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 5000,
        "class_weight": "none",
        "sample_weight": "none",
        "random_state": 23,
        "threshold_policy": "predict",
        "fit_intercept": True,
        "intercept_penalized": False,
        "dtype": "float64",
        "scaler_fit": "fit_rows_only",
    }
    assert config["alignment"]["gamma_grid"] == [
        0.0,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    ]
    assert config["selection"]["metric"] == "bacc"
    assert config["selection"]["aggregation"] == "equal_center_arithmetic_mean"
    assert config["selection"]["tie_break"] == "smallest_gamma"
    assert config["selection"]["outer_all_gamma_scoring"] is False
    assert config["selection"]["outer_oracle_gamma_computed"] is False
    assert config["run"]["expected_inner_score_count"] == 9 * 8 * 7 == 504
    assert config["run"]["expected_gamma_summary_count"] == 9 * 7 == 63
    assert config["run"]["expected_outer_result_count"] == 18
    claim = config["claim_boundary"]
    assert claim["claim_scope"] == "real_feature_transfer_only"
    assert claim["diagnostic_only"] is True
    assert claim["non_adoptive"] is True
    assert claim["target_evaluation_labels_used_for_selection"] is False
    assert claim["target_evaluation_labels_used_for_scoring_only"] is True
    assert claim["may_feed_recipe_selection"] is False
    assert claim["may_feed_deployable_selection"] is False
    assert all(
        claim[key] is False
        for key in (
            "uses_generated_embeddings",
            "uses_cvae_checkpoint",
            "uses_prior",
            "uses_nelbo",
            "uses_expert_bank",
            "uses_router",
            "performs_expert_selection",
            "performs_expert_weighting",
            "performs_aggregation",
        )
    )


@pytest.mark.parametrize(
    ("stage", "claim_scope"),
    (
        ("20_cvae_preservation", "cvae_preservation_only"),
        ("30_expert_bank", "expert_bank_construction_only"),
        ("40_prior_and_generation", "generation_diagnostics_only"),
        ("50_all_candidate_utility_matrix", "diagnostic_only"),
        ("60_routing_and_composition", "routing_and_composition"),
        ("70_frozen_policy_downstream", "synthetic_downstream_utility"),
    ),
)
def test_stage20_through_stage70_consumers_are_rejected(
    stage: str,
    claim_scope: str,
) -> None:
    workspace = _workspace_with_consumer(stage=stage, claim_scope=claim_scope)

    with pytest.raises(WorkspaceError):
        workspace.validate()


def test_stage90_diagnostic_consumer_is_allowed() -> None:
    workspace = _workspace_with_consumer(
        stage="90_oracles_and_diagnostics",
        claim_scope="diagnostic_only",
        add_scope_exception=False,
    )

    workspace.validate()


def _workspace_with_consumer(
    *,
    stage: str,
    claim_scope: str,
    add_scope_exception: bool = True,
) -> MidogppWorkspace:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    slug = stage.split("_", maxsplit=1)[0]
    consumer_output_id = f"test_cla_consumer_output_{slug}"
    consumer_experiment_id = f"test.cla.consumer.{slug}"
    catalog["artifacts"].append(
        {
            "artifact_id": consumer_output_id,
            "stage": stage,
            "canonical_path": f"artifacts/midogpp/{stage}/test_cla_consumer/{slug}",
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": claim_scope,
        }
    )
    consumer: dict[str, object] = {
        "experiment_id": consumer_experiment_id,
        "stage": stage,
        "status": "diagnostic" if stage == "90_oracles_and_diagnostics" else "planned",
        "claim_scope": claim_scope,
        "output_artifact_id": consumer_output_id,
        "input_artifact_ids": [OUTPUT_ID],
        "runner": {"argv": ["{python}", "-c", "pass"]},
    }
    if add_scope_exception:
        consumer["input_claim_scope_exceptions"] = {
            OUTPUT_ID: "Test-only exception; catalog reuse prohibitions remain binding."
        }
    registry["experiments"].append(consumer)
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
