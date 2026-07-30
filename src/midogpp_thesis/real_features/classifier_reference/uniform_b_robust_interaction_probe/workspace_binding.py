"""Exact workspace binding for the robust-interaction diagnostic."""

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import (
    CACHE_ID,
    CANONICAL_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    MULTISCALE_ID,
    NONLINEAR_ID,
    OUTPUT_ID,
    RobustInteractionConfig,
)


INPUT_IDS = (DATASET_ID, CACHE_ID, CANONICAL_ID, NONLINEAR_ID, MULTISCALE_ID)


def validate_production_workspace_binding(config: RobustInteractionConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != OUTPUT_ID
        or experiment.input_artifact_ids != INPUT_IDS
    ):
        raise ProtocolError("Robust-interaction workspace binding drifted.")
    expected = {
        "root": workspace.resolve_artifact(OUTPUT_ID, for_output=True, require_exists=False),
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True) / "manifest.csv",
        "cache": workspace.resolve_artifact(CACHE_ID, require_exists=True) / "embeddings/train.pt",
        "canonical": workspace.resolve_artifact(CANONICAL_ID, require_exists=True),
        "nonlinear": workspace.resolve_artifact(NONLINEAR_ID, require_exists=True),
        "multiscale": workspace.resolve_artifact(MULTISCALE_ID, require_exists=True),
    }
    actual = {
        "root": config.artifact_root,
        "manifest": config.manifest_path,
        "cache": config.feature_cache_path,
        "canonical": config.canonical_root,
        "nonlinear": config.nonlinear_root,
        "multiscale": config.multiscale_root,
    }
    for key, value in expected.items():
        if actual[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Robust-interaction input binding drifted: {key}.")
