"""Exact workspace binding for the bounded B-spatial diagnostic."""

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace
from ..protocol import ProtocolError
from .config import (
    CANONICAL_CACHE_ID, CANONICAL_REFERENCE_ID, DATASET_ID, EXPERIMENT_ID,
    NONLINEAR_REFERENCE_ID, OUTPUT_ID, SPATIAL_CACHE_ID, SpatialProbeConfig,
)

INPUT_IDS = (DATASET_ID, CANONICAL_CACHE_ID, SPATIAL_CACHE_ID, CANONICAL_REFERENCE_ID, NONLINEAR_REFERENCE_ID)

def validate_production_workspace_binding(config: SpatialProbeConfig) -> None:
    workspace = MidogppWorkspace.load(); workspace.validate(); experiment = workspace.get_experiment(EXPERIMENT_ID)
    if (experiment.status != "diagnostic" or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only" or experiment.output_artifact_id != OUTPUT_ID
        or experiment.input_artifact_ids != INPUT_IDS):
        raise ProtocolError("Uniform-B spatial workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(OUTPUT_ID, for_output=True, require_exists=False),
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True) / "manifest.csv",
        "canonical_cache": workspace.resolve_artifact(CANONICAL_CACHE_ID, require_exists=True) / "embeddings/train.pt",
        "spatial_root": workspace.resolve_artifact(SPATIAL_CACHE_ID, require_exists=True),
        "spatial_feature": workspace.resolve_artifact(SPATIAL_CACHE_ID, require_exists=True) / "embeddings/train.pt",
        "canonical": workspace.resolve_artifact(CANONICAL_REFERENCE_ID, require_exists=True),
        "nonlinear": workspace.resolve_artifact(NONLINEAR_REFERENCE_ID, require_exists=True),
    }
    configured = {
        "artifact_root": config.artifact_root, "manifest": config.manifest_path,
        "canonical_cache": config.canonical_b_cache_path, "spatial_root": config.spatial_cache_root,
        "spatial_feature": config.spatial_feature_cache_path, "canonical": config.canonical_reference_root,
        "nonlinear": config.nonlinear_reference_root,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Uniform-B spatial input binding drifted: {key}.")
