from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.preservation.source_inner_studies.config import (
    FISHER_SHRINKAGE_MODE,
    LEARNED_PRIOR_MODE,
    load_source_inner_study_config,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


CLAIM_SCOPE = "cvae_source_inner_study_only"
INPUT_ARTIFACTS = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
)
CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
SEEDS = (17, 42, 101)
STUDIES = {
    "midogpp.cvae.learned_conditional_prior_source_inner.v2": {
        "artifact_id": "midogpp_output_cvae_learned_conditional_prior_source_inner_v2",
        "config": "learned_conditional_prior_source_inner_v2.yaml",
        "mode": "learned_conditional_prior_source_inner_study",
        "command": "source-inner-learned-conditional-prior-study",
        "root": (
            "artifacts/midogpp/20_cvae_preservation/"
            "learned_conditional_prior_source_inner_v2/seeds17_42_101"
        ),
        "state_index": "manifests/learned_prior_state_index.json",
    },
    "midogpp.cvae.task_fisher_shrinkage_source_inner.v2": {
        "artifact_id": "midogpp_output_cvae_task_fisher_shrinkage_source_inner_v2",
        "config": "task_fisher_shrinkage_source_inner_v2.yaml",
        "mode": "task_fisher_shrinkage_source_inner_study",
        "command": "source-inner-task-fisher-shrinkage-study",
        "root": (
            "artifacts/midogpp/20_cvae_preservation/"
            "task_fisher_shrinkage_source_inner_v2/seeds17_42_101"
        ),
        "state_index": "manifests/task_fisher_shrinkage_state_index.json",
    },
}
FORBIDDEN_REUSE = {
    "expert_bank_evidence",
    "generation_evidence",
    "routing_evidence",
    "expert_selection_evidence",
    "nelbo_compatibility_evidence",
    "synthetic_downstream_utility_evidence",
}


def test_v2_studies_are_registered_as_non_consumable_stage20_outputs() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()

    stage20 = workspace.stages["20_cvae_preservation"]
    assert CLAIM_SCOPE in stage20["allowed_claim_scopes"]
    assert all(
        CLAIM_SCOPE not in tuple(stage.get("allowed_input_claim_scopes", ()))
        for stage in workspace.stages.values()
    )

    registered_inputs = {
        artifact_id
        for experiment in workspace.experiments.values()
        for artifact_id in experiment.input_artifact_ids
    }
    for experiment_id, expected in STUDIES.items():
        experiment = workspace.get_experiment(experiment_id)
        artifact = workspace.artifacts[str(expected["artifact_id"])]

        assert experiment.status == "active"
        assert experiment.stage == "20_cvae_preservation"
        assert experiment.claim_scope == CLAIM_SCOPE
        assert experiment.output_artifact_id == expected["artifact_id"]
        assert experiment.input_artifact_ids == INPUT_ARTIFACTS
        assert experiment.config_path == (
            "experiments/midogpp/stages/20_cvae_preservation/configs/"
            f"{expected['config']}"
        )
        assert expected["command"] in experiment.runner_argv
        assert artifact.claim_scope == CLAIM_SCOPE
        assert set(artifact.forbidden_reuse) == FORBIDDEN_REUSE
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False
        assert artifact.canonical_path == expected["root"]
        assert experiment.output_artifact_id not in registered_inputs


def test_v2_catalog_contract_requires_complete_non_adoptive_bundles() -> None:
    workspace = MidogppWorkspace.load()
    shared_required = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/protocol_manifest.json",
        "manifests/coverage_manifest.json",
        "manifests/selection_evidence_manifest.json",
        "manifests/embedded_v1_preparation_lineage.json",
        "manifests/checkpoint_index.json",
        "manifests/initialization_index.json",
        "manifests/feature_frame_index.json",
        "manifests/generation_budget_manifest.json",
        "reports/study_decision.json",
        "reports/leakage_report.json",
        "reports/runtime_summary.json",
        "reports/run_state.json",
        "tables/source_inner_metrics.csv",
        "tables/paired_deltas.csv",
        "tables/nested_real_references.csv",
        "tables/nested_classifier_tuning.csv",
        "tables/sampler_realizations.csv",
        "tables/checkpoint_reuse_audit.csv",
        "tables/initialization_pairing_audit.csv",
        "tables/generation_budget_audit.csv",
        "tables/rng_pairing_audit.csv",
        "tables/identity_overlap_audit.csv",
        "tables/runtime_timings.csv",
    }
    child_decisions = {
        f"reports/child_decisions/seed{seed}/{center}.json"
        for seed in SEEDS
        for center in CENTERS
    }
    consensus_decisions = {
        f"reports/consensus_decisions/{center}.json" for center in CENTERS
    }

    for expected in STUDIES.values():
        artifact = workspace.artifacts[str(expected["artifact_id"])]
        required = set(artifact.required_files)
        assert shared_required | child_decisions | consensus_decisions | {
            str(expected["state_index"])
        } == required
        assert not any("recipe_lock" in path for path in required)
        assert "reports/publication_state.json" not in required


def test_v2_configs_lock_source_only_seed_panels_and_distinct_questions(
    tmp_path: Path,
) -> None:
    workspace = MidogppWorkspace.load()
    config_root = workspace.repo_root / (
        "experiments/midogpp/stages/20_cvae_preservation/configs"
    )

    payloads: dict[str, dict[str, object]] = {}
    for experiment_id, expected in STUDIES.items():
        path = config_root / str(expected["config"])
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        payloads[experiment_id] = payload
        experiment = payload["experiment"]
        run = payload["run"]
        model = payload["model"]
        claim = payload["claim_boundary"]
        assert experiment["mode"] == expected["mode"]
        assert experiment["study_version"] == "v2"
        assert experiment["artifact_root"] == f"output://{expected['artifact_id']}"
        assert run["heldout_centers"] == "all"
        assert tuple(run["training_seeds"]) == SEEDS
        assert tuple(run["generation_seeds"]) == SEEDS
        assert run["expected_feature_dim"] == 2560
        assert model["pca_dim"] == 128
        assert model["latent_dim"] == 32
        assert payload["generation"]["budget_policy"] == (
            "source_empirical_class_counts_from_y_fit"
        )
        assert payload["decisions"]["minimum_real_bacc"] == 0.55
        assert claim["target_evaluation_data_used"] is False
        assert claim["may_change_existing_consensus_locks"] is False
        assert claim["may_feed_recipe_selection"] is False
        assert claim["may_feed_deployable_selection"] is False

        used_inputs: set[str] = set()
        resolved = workspace.resolve_value(
            payload,
            require_inputs=False,
            used_inputs=used_inputs,
        )
        assert used_inputs == set(INPUT_ARTIFACTS)
        assert resolved["experiment"]["artifact_root"].endswith(str(expected["root"]))
        resolved_path = tmp_path / str(expected["config"])
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False),
            encoding="utf-8",
        )
        loaded = load_source_inner_study_config(
            resolved_path,
            expected_mode=str(expected["mode"]),
        )
        assert loaded.mode == expected["mode"]
        assert loaded.training_seeds == SEEDS
        assert loaded.generation_seeds == SEEDS
        assert loaded.artifact_root.as_posix().endswith(str(expected["root"]))

    prior = payloads["midogpp.cvae.learned_conditional_prior_source_inner.v2"]
    assert prior["objective"] == {
        "family": "stochastic_isotropic_v1",
        "fixed_across_arms": True,
    }
    assert prior["prior"]["arms"] == ["A", "C-diag", "E"]
    assert prior["prior"]["optimizer_weight_decay"] == 0.0
    assert prior["prior"]["prior_gradient_clip_norm"] == 5.0
    assert LEARNED_PRIOR_MODE == STUDIES[
        "midogpp.cvae.learned_conditional_prior_source_inner.v2"
    ]["mode"]

    fisher = payloads["midogpp.cvae.task_fisher_shrinkage_source_inner.v2"]
    assert fisher["prior"] == {
        "family": "standard_normal",
        "fixed_across_alphas": True,
    }
    assert fisher["objective"]["alphas"] == [0.0, 0.05, 0.10, 0.25]
    assert fisher["objective"]["raw_fisher_fit_scope"] == "shared_per_outer_inner"
    assert fisher["objective"]["alpha_zero_policy"] == (
        "literal_isotropic_metric_none"
    )
    assert FISHER_SHRINKAGE_MODE == STUDIES[
        "midogpp.cvae.task_fisher_shrinkage_source_inner.v2"
    ]["mode"]


def test_v1_outer_and_stage30_edges_remain_unchanged() -> None:
    workspace = MidogppWorkspace.load()
    outer = workspace.get_experiment("midogpp.cvae.prior_recovery_outer.v1")
    expert_bank = workspace.get_experiment("midogpp.expert_bank.provenance_clean.v1")

    assert outer.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
        "midogpp_output_cvae_prior_recovery_source_inner_v1",
    )
    assert expert_bank.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_output_cvae_prior_recovery_source_inner_training_seed_stability_v1",
    )
    assert not {
        str(expected["artifact_id"]) for expected in STUDIES.values()
    }.intersection(expert_bank.input_artifact_ids)


def test_recipe_selection_catalog_flag_is_typed() -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    entry = next(
        item
        for item in catalog["artifacts"]
        if item["artifact_id"]
        == "midogpp_output_cvae_learned_conditional_prior_source_inner_v2"
    )
    entry["may_feed_recipe_selection"] = "false"

    with pytest.raises(WorkspaceError, match="may_feed_recipe_selection"):
        MidogppWorkspace(
            repo_root=source.repo_root,
            registry=source.registry_payload,
            catalog=catalog,
            workspace=source.workspace_payload,
            protocol_defaults=source.protocol_defaults_payload,
        )
