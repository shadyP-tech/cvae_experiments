from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = "midogpp.real_feature.fixed_c_risk_diagnostic.v1"
OUTPUT_ID = "midogpp_output_real_feature_fixed_c_risk_diagnostic_v1"
INPUT_IDS = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
)
CONFIG_PATH = (
    "experiments/midogpp/stages/10_real_feature_reference/configs/"
    "fixed_c_risk_diagnostic_v1.yaml"
)
REQUIRED_BUNDLE = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/runtime_summary.json",
    "tables/fixed_c_risk_results.csv",
    "tables/fixed_c_risk_predictions.csv",
    "tables/fixed_c_risk_weight_audit.csv",
    "tables/fixed_c_risk_paired_comparison.csv",
}


def test_fixed_c_risk_diagnostic_is_registered_without_stage10_scope_expansion() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    artifact = workspace.artifacts[OUTPUT_ID]
    known_reuse_purposes = {
        purpose
        for stage in workspace.stages.values()
        for purpose in stage["input_reuse_purposes"]
    }

    assert set(workspace.stages["10_real_feature_reference"]["allowed_claim_scopes"]) == {
        "real_feature_transfer_only",
        "real_feature_signal_controls",
    }
    assert experiment.status == "diagnostic"
    assert experiment.stage == "10_real_feature_reference"
    assert experiment.claim_scope == "real_feature_transfer_only"
    assert experiment.config_path == CONFIG_PATH
    assert experiment.output_artifact_id == OUTPUT_ID
    assert experiment.input_artifact_ids == INPUT_IDS
    assert experiment.runner_argv[-6:] == (
        "real-feature-classifier",
        "fixed-c-risk-diagnostic",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ID}",
    )

    assert artifact.stage == experiment.stage
    assert artifact.claim_scope == experiment.claim_scope
    assert artifact.canonical_path == (
        "artifacts/midogpp/10_real_feature_reference/"
        "fixed_c_risk_diagnostic_v1/seed42"
    )
    assert set(artifact.required_files) == REQUIRED_BUNDLE
    assert len(artifact.required_files) == 12
    assert set(artifact.forbidden_reuse) == (
        known_reuse_purposes - {"oracle_and_diagnostic_evidence"}
    )
    assert artifact.may_feed_recipe_selection is False
    assert artifact.may_feed_deployable_selection is False


def test_fixed_c_risk_config_locks_one_classifier_and_thirty_six_non_adoptive_fits() -> None:
    workspace = MidogppWorkspace.load()
    config = yaml.safe_load((workspace.repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    used_inputs: set[str] = set()
    resolved = workspace.resolve_value(
        config,
        require_inputs=False,
        used_inputs=used_inputs,
    )

    assert config["experiment"] == {
        "name": "fixed_c_risk_diagnostic_v1",
        "mode": "fixed_c_risk_diagnostic",
        "artifact_root": f"output://{OUTPUT_ID}",
        "code_version": "fixed_c_risk_diagnostic_v1",
    }
    assert used_inputs == set(INPUT_IDS)
    assert resolved["experiment"]["artifact_root"].endswith(
        "artifacts/midogpp/10_real_feature_reference/"
        "fixed_c_risk_diagnostic_v1/seed42"
    )
    assert config["run"] == {
        "experiment_seed": 42,
        "heldout_centers": "all",
        "expected_feature_dim": 2560,
        "expected_outer_fold_count": 9,
        "expected_arm_count": 4,
        "expected_fit_count": 36,
    }
    assert config["classifier"] == {
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 5000,
        "class_weight": "none",
        "random_state": 23,
        "threshold_policy": "predict",
        "expected_config_hash": "86378e6ceb12136e",
    }
    assert config["weighting"] == {
        "arms": ["pooled", "global_class", "domain", "domain_class"],
        "formulas": {
            "pooled": "1",
            "global_class": "N/(2*n_y)",
            "domain": "N/(D*n_d)",
            "domain_class": "N/(2*D*n_dy)",
        },
        "normalization": "sum_to_n_fit",
        "zero_cell_policy": "fail_closed",
        "require_finite_positive_weights": True,
    }
    assert (
        config["run"]["expected_outer_fold_count"]
        * config["run"]["expected_arm_count"]
        == config["run"]["expected_fit_count"]
        == 36
    )
    assert config["comparison"] == {
        "primary_contrast": "domain_class_minus_pooled",
        "paired_by": "heldout_center",
        "selection_rule": "none",
        "adoption_enabled": False,
    }
    claim = config["claim_boundary"]
    assert claim["claim_scope"] == "real_feature_transfer_only"
    assert claim["diagnostic_only"] is True
    assert claim["non_adoptive"] is True
    assert claim["target_evaluation_labels_used_for_fit"] is False
    assert claim["target_evaluation_labels_used_for_selection"] is False
    assert claim["target_evaluation_labels_used_for_scoring_only"] is True
    assert claim["may_feed_recipe_selection"] is False
    assert claim["may_feed_deployable_selection"] is False
    assert all(
        claim[key] is False
        for key in (
            "uses_cvae_checkpoint",
            "uses_generated_embeddings",
            "uses_prior",
            "uses_router",
        )
    )


@pytest.mark.parametrize(
    ("stage", "claim_scope", "expected_error"),
    (
        (
            "20_cvae_preservation",
            "cvae_preservation_only",
            "forbids reuse as",
        ),
        (
            "60_routing_and_composition",
            "routing_and_composition",
            "forbids reuse as",
        ),
    ),
)
def test_stage20_and_stage60_consumption_fail_even_with_scope_exception(
    stage: str,
    claim_scope: str,
    expected_error: str,
) -> None:
    workspace = _workspace_with_consumer(
        stage=stage,
        claim_scope=claim_scope,
        add_scope_exception=True,
    )

    with pytest.raises(WorkspaceError, match=expected_error):
        workspace.validate()


def test_stage90_diagnostic_consumption_can_validate() -> None:
    workspace = _workspace_with_consumer(
        stage="90_oracles_and_diagnostics",
        claim_scope="diagnostic_only",
        add_scope_exception=False,
    )

    workspace.validate()
    assert workspace.get_experiment(
        "test.fixed_c_risk.consumer.90"
    ).status == "diagnostic"


def _workspace_with_consumer(
    *,
    stage: str,
    claim_scope: str,
    add_scope_exception: bool,
) -> MidogppWorkspace:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    if stage == "60_routing_and_composition":
        # Isolate the deny-list guard; the canonical flag is asserted false above.
        diagnostic = next(
            artifact
            for artifact in catalog["artifacts"]
            if artifact["artifact_id"] == OUTPUT_ID
        )
        diagnostic["may_feed_deployable_selection"] = True
    slug = stage.split("_", maxsplit=1)[0]
    consumer_output_id = f"test_fixed_c_risk_consumer_output_{slug}"
    consumer_experiment_id = f"test.fixed_c_risk.consumer.{slug}"
    catalog["artifacts"].append(
        {
            "artifact_id": consumer_output_id,
            "stage": stage,
            "canonical_path": (
                f"artifacts/midogpp/{stage}/test_fixed_c_risk_consumer/{slug}"
            ),
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
            OUTPUT_ID: (
                "Test-only reviewed scope exception; catalog reuse prohibitions "
                "must remain binding."
            )
        }
    registry["experiments"].append(consumer)
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
