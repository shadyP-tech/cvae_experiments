from __future__ import annotations

from midogpp_thesis.cvae.preservation.prior_recovery_config import (
    SAMPLER_FALLBACK_POLICY,
    SAMPLER_VIABILITY_POLICY,
    load_prior_recovery_config,
    recipe_contract_hash,
)
from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    CANONICAL_GRID_HASH,
    canonical_matched_reference_specs,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_prior_recovery_registry_enforces_selection_and_evaluation_boundary() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()

    source_inner = workspace.get_experiment("midogpp.cvae.prior_recovery_source_inner.v1")
    outer = workspace.get_experiment("midogpp.cvae.prior_recovery_outer.v1")
    expert_bank = workspace.get_experiment("midogpp.expert_bank.provenance_clean.v1")
    outer_artifact = workspace.artifacts[outer.output_artifact_id]

    assert source_inner.claim_scope == "cvae_recipe_lock_only"
    assert source_inner.output_artifact_id in outer.input_artifact_ids
    assert source_inner.output_artifact_id in expert_bank.input_artifact_ids
    assert outer.output_artifact_id not in expert_bank.input_artifact_ids
    assert "expert_bank_evidence" in outer_artifact.forbidden_reuse
    assert outer_artifact.may_feed_deployable_selection is False


def test_production_configs_lock_grid_alpha_seeds_and_modes() -> None:
    workspace = MidogppWorkspace.load()
    root = workspace.repo_root / "experiments/midogpp/stages/20_cvae_preservation/configs"
    source_inner = load_prior_recovery_config(root / "prior_recovery_source_inner_v1.yaml")
    outer = load_prior_recovery_config(root / "prior_recovery_outer_v1.yaml")

    assert source_inner.mode == "source_inner"
    assert outer.mode == "outer"
    assert source_inner.task_fisher_variant.alpha == outer.task_fisher_variant.alpha == 1.0
    assert outer.training_seeds == (17, 42, 101)
    assert outer.generation_seeds == (17, 42, 101)
    assert recipe_contract_hash(source_inner) == recipe_contract_hash(outer)
    assert source_inner.sampler_fallback_policy == SAMPLER_FALLBACK_POLICY
    assert source_inner.sampler_viability_policy == SAMPLER_VIABILITY_POLICY
    assert not hasattr(source_inner, "reference_artifact_root")
    assert not hasattr(source_inner, "training_seeds")
    assert len(canonical_matched_reference_specs(classifier_seed=23)) == 20
    assert CANONICAL_GRID_HASH == "16a7a1183ea3f65b"
