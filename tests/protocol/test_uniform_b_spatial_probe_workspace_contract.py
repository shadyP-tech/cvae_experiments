from midogpp_thesis.real_features.classifier_reference.cli import build_parser
from midogpp_thesis.real_features.classifier_reference.uniform_b_spatial_probe.config import (
    CANONICAL_CACHE_ID,
    CANONICAL_REFERENCE_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    NONLINEAR_REFERENCE_ID,
    OUTPUT_ID,
    SPATIAL_CACHE_ID,
)
from midogpp_thesis.workspace.cli import _normalize_run_arguments
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_uniform_b_spatial_probe_is_bounded_and_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == (
        DATASET_ID,
        CANONICAL_CACHE_ID,
        SPATIAL_CACHE_ID,
        CANONICAL_REFERENCE_ID,
        NONLINEAR_REFERENCE_ID,
    )
    assert output.evidence_label == "DIAGNOSTIC ONLY"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert "real_feature_reference_evidence" in output.forbidden_reuse


def test_uniform_b_spatial_cli_surfaces_are_registered() -> None:
    parser = build_parser()
    cache = parser.parse_args(
        ["build-uniform-b-spatial-cache", "--config", "cache.yaml"]
    )
    probe = parser.parse_args(
        ["uniform-b-spatial-probe", "--config", "probe.yaml"]
    )
    assert cache.surface == "build-uniform-b-spatial-cache"
    assert probe.surface == "uniform-b-spatial-probe"


def test_workspace_force_is_accepted_after_experiment_id() -> None:
    assert _normalize_run_arguments(["--force"], force=False) == ((), True)
    assert _normalize_run_arguments(["--force", "--other"], force=False) == (
        ("--other",),
        True,
    )
    assert _normalize_run_arguments(["--", "--force"], force=False) == (
        ("--force",),
        False,
    )
