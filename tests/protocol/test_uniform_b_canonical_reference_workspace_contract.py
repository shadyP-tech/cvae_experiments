from __future__ import annotations

from midogpp_thesis.real_features.classifier_reference.uniform_b_reference.workspace_binding import (
    CACHE_ID,
    CONFIRMATION_ID,
    EXPERIMENT_ID,
    INPUT_IDS,
    OUTPUT_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_uniform_b_canonical_reference_is_distinct_and_review_gated() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]

    assert experiment.stage == "10_real_feature_reference"
    assert experiment.status == "active"
    assert experiment.claim_scope == "real_feature_transfer_only"
    assert experiment.input_artifact_ids == INPUT_IDS
    assert CONFIRMATION_ID in experiment.input_claim_scope_exceptions
    assert output.canonical_path == (
        "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42"
    )
    assert output.evidence_label == "CANONICAL REAL-FEATURE REFERENCE"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_uniform_b_canonical_cache_may_feed_stage20_but_not_later_stages_directly() -> None:
    workspace = MidogppWorkspace.load()
    cache = workspace.artifacts[CACHE_ID]

    assert cache.evidence_label == "CANONICAL INPUT"
    assert "cvae_preservation_evidence" not in cache.forbidden_reuse
    assert {
        "expert_bank_evidence",
        "routing_evidence",
        "synthetic_downstream_utility_evidence",
    }.issubset(cache.forbidden_reuse)
