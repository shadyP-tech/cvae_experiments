"""Exact workspace binding for the Stage-90 canonical-B nonlinear probe."""

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import (
    CACHE_ID,
    CANONICAL_REFERENCE_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    OUTPUT_ID,
    NonlinearProbeConfig,
)


INPUT_IDS = (DATASET_ID, CACHE_ID, CANONICAL_REFERENCE_ID)


def validate_production_workspace_binding(config: NonlinearProbeConfig) -> None:
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
        raise ProtocolError("Uniform-B nonlinear workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ID, for_output=True, require_exists=False
        ),
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True)
        / "manifest.csv",
        "cache": workspace.resolve_artifact(CACHE_ID, require_exists=True)
        / "embeddings/train.pt",
        "canonical": workspace.resolve_artifact(
            CANONICAL_REFERENCE_ID, require_exists=True
        ),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "manifest": config.manifest_path,
        "cache": config.feature_cache_path,
        "canonical": config.canonical_reference_root,
    }
    for key, path in expected.items():
        if configured[key].resolve() != Path(path).resolve():
            raise ProtocolError(f"Uniform-B nonlinear input binding drifted: {key}.")
