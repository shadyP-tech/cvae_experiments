"""Exact workspace binding for the Stage-90 uniform-B replay."""

from __future__ import annotations

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import UniformBReplayConfig


EXPERIMENT_ID = "midogpp.oracle.uniform_b_v3_retrospective_replay.v1"
OUTPUT_ID = "midogpp_output_uniform_b_v3_retrospective_replay_v1"
SOURCE_ID = (
    "midogpp_output_real_feature_physical_multiscale_clipped_bbox_"
    "annotation_local_pooling_pilot_v3"
)
B_CACHE_ID = (
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42"
)
CANONICAL_REFERENCE_ID = "midogpp_output_eligible_tuned_real_reference_v2"
INPUT_IDS = (SOURCE_ID, B_CACHE_ID, CANONICAL_REFERENCE_ID)


def validate_production_workspace_binding(config: UniformBReplayConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if (
        experiment.status != "diagnostic"
        or not experiment.runnable
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != OUTPUT_ID
        or experiment.input_artifact_ids != INPUT_IDS
    ):
        raise ProtocolError("Uniform-B workspace experiment binding drifted.")
    source = workspace.artifacts[SOURCE_ID]
    if source.evidence_label != "DIAGNOSTIC ONLY" or set(
        source.required_files
    ).difference(source.expected_file_hashes):
        raise ProtocolError("Uniform-B source v3 artifact is not hash-promoted.")
    expected_root = workspace.resolve_artifact(
        OUTPUT_ID, for_output=True, require_exists=False
    )
    if config.artifact_root.resolve() != expected_root.resolve():
        raise ProtocolError("Uniform-B output root differs from catalog binding.")
    expected = {
        "source": workspace.resolve_artifact(SOURCE_ID, require_exists=True),
        "b": workspace.resolve_artifact(B_CACHE_ID, require_exists=True),
        "reference": workspace.resolve_artifact(
            CANONICAL_REFERENCE_ID, require_exists=True
        ),
    }
    configured = {
        "source": config.source_v3_root,
        "b": config.b_cache_root,
        "reference": config.canonical_reference_root,
    }
    for key, path in expected.items():
        if configured[key].resolve() != Path(path).resolve():
            raise ProtocolError(f"Uniform-B input binding drifted: {key}.")
