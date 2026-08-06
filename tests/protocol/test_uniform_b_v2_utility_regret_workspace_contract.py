from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cli import COMMANDS
from midogpp_thesis.cvae.routing.source_inner_utility.bundle import (
    REQUIRED_FILES as UTILITY_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.source_inner_utility.contracts import (
    OUTPUT_SEMANTIC_IDENTITIES as UTILITY_SEMANTIC_IDENTITIES,
    POLICY_CONSUMPTION_LOCK_HASH,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.bundle import (
    REQUIRED_FILES as POLICY_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.contracts import (
    CONSUMPTION_RULE_HASH,
    OUTPUT_SEMANTIC_IDENTITIES as POLICY_SEMANTIC_IDENTITIES,
)
from midogpp_thesis.data.features.uniform_b_routing_validation.config import (
    CANONICAL_A_VALIDATION_ARTIFACT_ID,
    CANONICAL_VALIDATION_URI,
    load_routing_validation_cache_config,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CACHE_CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_routing_validation_cache_v1.yaml"
)

UTILITY_EXPERIMENT = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_source_inner_candidate_utility.v1"
)
POLICY_EXPERIMENT = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_regret_policy_lock.v1"
)
UTILITY_ARTIFACT = "midogpp_output_uniform_b_v2_source_inner_candidate_utility_v1"
POLICY_ARTIFACT = "midogpp_output_uniform_b_v2_utility_regret_policy_lock_v1"
CACHE_ARTIFACT = "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
MANIFEST_ALIAS = "midogpp_source_inner_validation_manifest_v1"
BROAD_MANIFEST = "midogpp_dataset_contract_annotation_patch_v1"


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_only_the_narrow_validation_manifest_is_selection_authorized() -> None:
    workspace = _workspace()
    narrow = workspace.artifacts[MANIFEST_ALIAS]
    broad = workspace.artifacts[BROAD_MANIFEST]
    assert narrow.required_files == ("manifest.csv",)
    assert narrow.may_feed_deployable_selection is True
    assert broad.may_feed_deployable_selection is not True
    assert narrow.semantic_identities["policy_consumption_lock_hash"] == (
        POLICY_CONSUMPTION_LOCK_HASH
    )
    assert "synthetic_downstream_utility_evidence" in narrow.forbidden_reuse


def test_validation_cache_is_unlabeled_narrow_and_not_a_stage70_input() -> None:
    workspace = _workspace()
    cache = workspace.artifacts[CACHE_ARTIFACT]
    assert cache.stage == "derived_features"
    assert cache.claim_scope == "feature_cache_provenance"
    assert cache.may_feed_deployable_selection is True
    assert cache.semantic_identities["labels_persisted"] == "false"
    assert cache.semantic_identities["split"] == "val"
    assert "synthetic_downstream_utility_evidence" in cache.forbidden_reuse
    assert all("label" not in member for member in cache.required_files)


def test_canonical_a_validation_comparator_has_a_narrow_alias() -> None:
    workspace = _workspace()
    artifact = workspace.artifacts[CANONICAL_A_VALIDATION_ARTIFACT_ID]
    config = load_routing_validation_cache_config(CACHE_CONFIG)
    assert config.canonical_validation_location == CANONICAL_VALIDATION_URI
    assert artifact.required_files == ("embeddings/val.pt",)
    assert artifact.may_feed_deployable_selection is False
    assert artifact.expected_file_hashes["embeddings/val.pt"].digest == (
        "23b0a76d1fb56e033556b44f7f939f957be0284269c218655ace18857cafa117"
    )


def test_utility_and_policy_have_exact_four_input_graphs() -> None:
    workspace = _workspace()
    utility = workspace.get_experiment(UTILITY_EXPERIMENT)
    policy = workspace.get_experiment(POLICY_EXPERIMENT)
    assert utility.input_artifact_ids == (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        CACHE_ARTIFACT,
        MANIFEST_ALIAS,
    )
    assert policy.input_artifact_ids == (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1",
        UTILITY_ARTIFACT,
    )
    assert BROAD_MANIFEST not in utility.input_artifact_ids


def test_utility_has_one_registered_consumer_and_cannot_feed_stage70_directly() -> None:
    workspace = _workspace()
    consumers = tuple(
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if UTILITY_ARTIFACT in experiment.input_artifact_ids
    )
    assert consumers == (POLICY_EXPERIMENT,)
    utility = workspace.artifacts[UTILITY_ARTIFACT]
    policy = workspace.artifacts[POLICY_ARTIFACT]
    assert "synthetic_downstream_utility_evidence" in utility.forbidden_reuse
    assert "synthetic_downstream_utility_evidence" not in policy.forbidden_reuse


def test_catalog_required_files_and_semantic_contracts_match_packages() -> None:
    workspace = _workspace()
    utility = workspace.artifacts[UTILITY_ARTIFACT]
    policy = workspace.artifacts[POLICY_ARTIFACT]
    assert utility.required_files == UTILITY_REQUIRED_FILES
    assert policy.required_files == POLICY_REQUIRED_FILES
    assert dict(utility.semantic_identities) == UTILITY_SEMANTIC_IDENTITIES
    assert dict(policy.semantic_identities) == POLICY_SEMANTIC_IDENTITIES
    assert CONSUMPTION_RULE_HASH == POLICY_CONSUMPTION_LOCK_HASH


def test_stage60_contract_freezes_outer_exclusion_and_exact_fallback() -> None:
    workspace = _workspace()
    requirements = set(
        workspace.stages["60_routing_and_composition"].get("hard_requirements", ())
    )
    assert (
        "source_inner_validation_labels_may_be_consumed_once_only_for_a_"
        "predeclared_policy_family"
    ) in requirements
    assert (
        "outer_target_rows_must_be_removed_for_both_query_q_and_candidate_e_"
        "before_regret_normalization_or_bootstrap"
    ) in requirements
    assert (
        "uncertain_utility_regret_decisions_must_reuse_the_exact_frozen_"
        "equal_union_control"
    ) in requirements
    assert set(workspace.stages["60_routing_and_composition"]["forbidden_upstream"]) == {
        "50_all_candidate_utility_matrix",
        "90_oracles_and_diagnostics",
    }


def test_cli_surface_remains_under_cvae_routing() -> None:
    assert COMMANDS["cvae-routing"][0] == "midogpp_thesis.cvae.routing.cli:main"
    source = (ROOT / "src/midogpp_thesis/cvae/routing/cli.py").read_text(
        encoding="utf-8"
    )
    assert "uniform-b-v2-routing-validation-cache" in source
    assert "uniform-b-v2-source-inner-candidate-utility" in source
    assert "uniform-b-v2-utility-regret-policy-lock" in source
