from midogpp_thesis.real_features.classifier_reference.uniform_b_nonlinear_probe.config import (
    CACHE_ID,
    CANONICAL_REFERENCE_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    OUTPUT_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_uniform_b_nonlinear_probe_is_stage90_and_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]
    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        DATASET_ID,
        CACHE_ID,
        CANONICAL_REFERENCE_ID,
    )
    assert output.evidence_label == "DIAGNOSTIC ONLY"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert "real_feature_reference_evidence" in output.forbidden_reuse
