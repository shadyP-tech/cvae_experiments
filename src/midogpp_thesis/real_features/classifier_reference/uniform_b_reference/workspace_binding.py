"""Exact workspace binding for the reviewed Uniform-B reference promotion."""

from __future__ import annotations

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import UniformBCanonicalReferenceConfig


EXPERIMENT_ID = "midogpp.real_feature.uniform_b_canonical_reference.v1"
OUTPUT_ID = "midogpp_output_uniform_b_canonical_reference_v1"
DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CACHE_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
CONFIRMATION_ID = "midogpp_output_uniform_b_v3_prospective_test_confirmation_v1"
INPUT_IDS = (DATASET_ID, CACHE_ID, CONFIRMATION_ID)


def validate_production_workspace_binding(
    config: UniformBCanonicalReferenceConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if (
        experiment.status != "active"
        or experiment.stage != "10_real_feature_reference"
        or experiment.claim_scope != "real_feature_transfer_only"
        or experiment.output_artifact_id != OUTPUT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or CONFIRMATION_ID not in experiment.input_claim_scope_exceptions
    ):
        raise ProtocolError("Uniform-B canonical workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ID, for_output=True, require_exists=False
        ),
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True)
        / "manifest.csv",
        "cache": workspace.resolve_artifact(CACHE_ID, require_exists=True)
        / "embeddings/train.pt",
        "confirmation": workspace.resolve_artifact(CONFIRMATION_ID, require_exists=True),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "manifest": config.manifest_path,
        "cache": config.feature_cache_path,
        "confirmation": config.confirmation_root,
    }
    for key, path in expected.items():
        if configured[key].resolve() != Path(path).resolve():
            raise ProtocolError(f"Uniform-B canonical input binding drifted: {key}.")
