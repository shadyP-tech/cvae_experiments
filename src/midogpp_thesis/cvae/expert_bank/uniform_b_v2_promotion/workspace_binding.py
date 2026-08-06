"""Exact workspace binding for the reviewed Uniform-B v2 bank promotion."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import UniformBV2PromotionConfig
from .contracts import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID, SOURCE_ARTIFACT_ID


DATASET_ID = "midogpp_dataset_contract_annotation_patch_v1"
CACHE_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
INPUT_IDS = (DATASET_ID, CACHE_ID, SOURCE_ARTIFACT_ID)


def validate_production_workspace_binding(config: UniformBV2PromotionConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    if (
        experiment.status != "active"
        or experiment.stage != "30_expert_bank"
        or experiment.claim_scope != "expert_bank_construction_only"
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or SOURCE_ARTIFACT_ID not in experiment.input_claim_scope_exceptions
        or output.may_feed_deployable_selection is not True
        or "routing_evidence" in output.forbidden_reuse
        or "expert_selection_evidence" in output.forbidden_reuse
    ):
        raise ProtocolError("Uniform-B v2 promotion workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "source_study_root": workspace.resolve_artifact(SOURCE_ARTIFACT_ID),
        "manifest_path": workspace.resolve_artifact(DATASET_ID) / "manifest.csv",
        "feature_cache_path": workspace.resolve_artifact(CACHE_ID) / "embeddings/train.pt",
    }
    configured = {
        "artifact_root": config.artifact_root,
        "source_study_root": config.source_study_root,
        "manifest_path": config.manifest_path,
        "feature_cache_path": config.feature_cache_path,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Uniform-B v2 promotion input binding drifted: {key}.")


__all__ = ("CACHE_ID", "DATASET_ID", "INPUT_IDS", "validate_production_workspace_binding")
