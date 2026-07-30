from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.preservation.aggregate_prior_study.config import (
    load_aggregate_prior_study_config,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.contracts import (
    ARMS,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.validation import (
    STATIC_FILES,
)
from midogpp_thesis.real_features.classifier_reference.protocol import (
    ProtocolError,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ARTIFACT_ID = (
    "midogpp_output_cvae_aggregate_posterior_mixture_geco_source_inner_v3"
)
CONFIG_PATH = (
    "experiments/midogpp/stages/20_cvae_preservation/configs/"
    "aggregate_posterior_mixture_geco_source_inner_v3.yaml"
)


def test_v3_is_registered_as_non_consumable_independent_source_study() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    artifact = workspace.artifacts[ARTIFACT_ID]

    assert experiment.stage == "20_cvae_preservation"
    assert experiment.status == "active"
    assert experiment.claim_scope == CLAIM_SCOPE
    assert experiment.config_path == CONFIG_PATH
    assert experiment.output_artifact_id == ARTIFACT_ID
    assert experiment.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
    )
    assert "source-inner-aggregate-posterior-mixture-geco" in (
        experiment.runner_argv
    )
    assert artifact.claim_scope == CLAIM_SCOPE
    assert artifact.may_feed_recipe_selection is False
    assert artifact.may_feed_deployable_selection is False
    assert {
        "expert_bank_evidence",
        "generation_evidence",
        "routing_evidence",
        "expert_selection_evidence",
        "nelbo_compatibility_evidence",
        "synthetic_downstream_utility_evidence",
    }.issubset(artifact.forbidden_reuse)
    assert artifact.canonical_path == (
        "artifacts/midogpp/20_cvae_preservation/"
        "aggregate_posterior_mixture_geco_source_inner_v3/seeds17_42_101"
    )

    registered_inputs = {
        artifact_id
        for registered in workspace.experiments.values()
        for artifact_id in registered.input_artifact_ids
    }
    assert ARTIFACT_ID not in registered_inputs


def test_v3_config_resolves_and_locks_source_isolation(tmp_path: Path) -> None:
    workspace = MidogppWorkspace.load()
    raw_path = workspace.repo_root / CONFIG_PATH
    payload = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    used_inputs: set[str] = set()
    resolved = workspace.resolve_value(
        payload,
        require_inputs=False,
        used_inputs=used_inputs,
    )
    resolved_path = tmp_path / "resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    config = load_aggregate_prior_study_config(resolved_path)

    assert used_inputs == {
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
    }
    assert config.arms == ARMS
    assert config.n_components == 2
    assert config.mixture_rank == 2
    assert config.refit_interval_epochs == 5
    assert config.final_stabilization_epochs == 5
    assert config.geco_target_policy == "source_warmup_mean_mse_times_slack"
    assert config.generation_per_class == 256
    assert config.claim_scope == CLAIM_SCOPE
    assert config.may_feed_recipe_selection is False
    assert config.may_feed_deployable_selection is False
    assert config.optimizer_updates_prior_parameters is False
    assert config.geco_uses_inner_or_outer_data is False
    assert config.same_budget_and_rng_across_arms is True
    assert config.source_or_target_prevalence_used is False
    assert config.inverse_transform_to_common_frame is True
    assert config.separate_promotion_artifact_required is True

    relocated = replace(
        config,
        artifact_root=tmp_path / "other-output",
        manifest_path=tmp_path / "other-manifest.csv",
        feature_cache_path=tmp_path / "other-cache.pt",
    )
    assert relocated.contract_hash == config.contract_hash


def test_v3_rejects_safety_guardrail_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    payload["prior"]["rate_semantics"] = "exact_nelbo"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="config drifted"):
        load_aggregate_prior_study_config(drifted)


def test_v3_does_not_change_outer_or_stage30_inputs() -> None:
    workspace = MidogppWorkspace.load()
    outer = workspace.get_experiment("midogpp.cvae.prior_recovery_outer.v1")
    stage30 = workspace.get_experiment("midogpp.expert_bank.provenance_clean.v1")
    assert ARTIFACT_ID not in outer.input_artifact_ids
    assert ARTIFACT_ID not in stage30.input_artifact_ids
    assert stage30.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_output_cvae_prior_recovery_source_inner_training_seed_stability_v1",
    )


def test_catalog_requires_fail_closed_study_bundle() -> None:
    artifact = MidogppWorkspace.load().artifacts[ARTIFACT_ID]
    required = set(artifact.required_files)
    assert {
        "manifests/source_expert_checkpoint_index.json",
        "manifests/mixture_prior_state_index.json",
        "manifests/geco_state_index.json",
        "reports/expert_isolation_report.json",
        "reports/publication_state.json",
        "tables/source_expert_metrics.csv",
        "tables/mixture_prior_diagnostics.csv",
        "tables/geco_trajectory.csv",
        "tables/identity_overlap_audit.csv",
    }.issubset(required)
    assert set(STATIC_FILES).issubset(required)
    assert not any("recipe_lock" in relative for relative in required)
