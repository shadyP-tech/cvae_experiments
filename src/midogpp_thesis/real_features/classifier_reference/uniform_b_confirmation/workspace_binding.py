"""Exact workspace binding for prospective uniform-B confirmation."""

from __future__ import annotations

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import UniformBConfirmationConfig


EXPERIMENT_ID = "midogpp.oracle.uniform_b_v3_prospective_test_confirmation.v1"
OUTPUT_ID = "midogpp_output_uniform_b_v3_prospective_test_confirmation_v1"
DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CANONICAL_ID = "midogpp_virchow2_xyxy_feature_cache_seed42"
SOURCE_B_ID = "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42"
TEST_B_ID = "midogpp_virchow2_uniform_b_v3_prospective_test_cache_seed42"
SOURCE_V3_ID = (
    "midogpp_output_real_feature_physical_multiscale_clipped_bbox_"
    "annotation_local_pooling_pilot_v3"
)
RETROSPECTIVE_ID = "midogpp_output_uniform_b_v3_retrospective_replay_v1"
INPUT_IDS = (
    DATASET_ID,
    CANONICAL_ID,
    SOURCE_B_ID,
    TEST_B_ID,
    SOURCE_V3_ID,
    RETROSPECTIVE_ID,
)


def validate_production_workspace_binding(config: UniformBConfirmationConfig) -> None:
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
        raise ProtocolError("Uniform-B prospective workspace binding drifted.")
    test_cache = workspace.artifacts[TEST_B_ID]
    if test_cache.evidence_label != "AUDIT_ONLY" or set(
        test_cache.required_files
    ).difference(test_cache.expected_file_hashes):
        raise ProtocolError("Uniform-B prospective test cache is not hash-promoted.")
    expected_root = workspace.resolve_artifact(OUTPUT_ID, for_output=True, require_exists=False)
    if config.artifact_root.resolve() != expected_root.resolve():
        raise ProtocolError("Uniform-B prospective output root differs from catalog.")
    expected = {
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True) / "manifest.csv",
        "canonical_train": workspace.resolve_artifact(CANONICAL_ID, require_exists=True) / "embeddings/train.pt",
        "canonical_test": workspace.resolve_artifact(CANONICAL_ID, require_exists=True) / "embeddings/test.pt",
        "source_b": workspace.resolve_artifact(SOURCE_B_ID, require_exists=True),
        "test_b": workspace.resolve_artifact(TEST_B_ID, require_exists=True),
        "source_v3": workspace.resolve_artifact(SOURCE_V3_ID, require_exists=True),
        "retrospective": workspace.resolve_artifact(RETROSPECTIVE_ID, require_exists=True),
    }
    configured = {
        "manifest": config.manifest_path,
        "canonical_train": config.canonical_train_cache_path,
        "canonical_test": config.canonical_test_cache_path,
        "source_b": config.source_train_b_cache_root,
        "test_b": config.test_b_cache_root,
        "source_v3": config.source_v3_root,
        "retrospective": config.retrospective_root,
    }
    for key, path in expected.items():
        if configured[key].resolve() != Path(path).resolve():
            raise ProtocolError(f"Uniform-B prospective input binding drifted: {key}.")
