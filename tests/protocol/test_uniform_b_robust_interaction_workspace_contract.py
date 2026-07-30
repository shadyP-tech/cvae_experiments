from midogpp_thesis.real_features.classifier_reference.uniform_b_robust_interaction_probe.config import (
    CACHE_ID,
    CANONICAL_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    MULTISCALE_ID,
    NONLINEAR_ID,
    OUTPUT_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_robust_interaction_probe_is_stage90_and_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]
    assert experiment.status == "diagnostic"
    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        DATASET_ID,
        CACHE_ID,
        CANONICAL_ID,
        NONLINEAR_ID,
        MULTISCALE_ID,
    )
    assert output.evidence_label == "DIAGNOSTIC ONLY"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
