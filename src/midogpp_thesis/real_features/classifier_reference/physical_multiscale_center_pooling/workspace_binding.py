"""Production registry/catalog binding for the Stage-10 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import PhysicalMultiscalePilotConfig
from .profiles import (
    ANNOTATION_LOCAL_POOLING_PILOT_V2,
    CENTER_POOLING_PILOT_V1,
    CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
)


EXPERIMENT_ID = "midogpp.real_feature.physical_multiscale_center_pooling_pilot.v1"
OUTPUT_ID = "midogpp_output_real_feature_physical_multiscale_center_pooling_pilot_v1"
INPUT_IDS = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_dataset_contract_physical_multiscale_center_pooling_pilot_v1",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
    "midogpp_virchow2_jpeg_center_pooling_3840_seed42",
    "midogpp_virchow2_physical_multiscale_center_pooling_11520_seed42",
    "midogpp_output_eligible_tuned_real_reference_v2",
)
PROMOTION_GATED_INPUT_IDS = (
    "midogpp_dataset_contract_physical_multiscale_center_pooling_pilot_v1",
    "midogpp_virchow2_jpeg_center_pooling_3840_seed42",
    "midogpp_virchow2_physical_multiscale_center_pooling_11520_seed42",
)
EXPERIMENT_ID_V2 = (
    "midogpp.real_feature.physical_multiscale_annotation_local_pooling_pilot.v2"
)
OUTPUT_ID_V2 = (
    "midogpp_output_real_feature_physical_multiscale_annotation_local_pooling_pilot_v2"
)
INPUT_IDS_V2 = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_dataset_contract_physical_multiscale_annotation_local_pooling_pilot_v2",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
    "midogpp_virchow2_annotation_local_pooling_bc_bundle_seed42_v2",
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v2_seed42",
    "midogpp_virchow2_physical_multiscale_annotation_local_pooling_11520_v2_seed42",
    "midogpp_output_eligible_tuned_real_reference_v2",
)
PROMOTION_GATED_INPUT_IDS_V2 = (
    "midogpp_dataset_contract_physical_multiscale_annotation_local_pooling_pilot_v2",
    "midogpp_virchow2_annotation_local_pooling_bc_bundle_seed42_v2",
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v2_seed42",
    "midogpp_virchow2_physical_multiscale_annotation_local_pooling_11520_v2_seed42",
)
EXPERIMENT_ID_V3 = (
    "midogpp.real_feature.physical_multiscale_clipped_bbox_annotation_local_pooling_pilot.v3"
)
OUTPUT_ID_V3 = (
    "midogpp_output_real_feature_physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3"
)
INPUT_IDS_V3 = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_dataset_contract_physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
    "midogpp_virchow2_clipped_bbox_annotation_local_pooling_bc_bundle_seed42_v3",
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42",
    "midogpp_virchow2_physical_multiscale_clipped_bbox_annotation_local_pooling_11520_v3_seed42",
    "midogpp_output_eligible_tuned_real_reference_v2",
)
PROMOTION_GATED_INPUT_IDS_V3 = (
    "midogpp_dataset_contract_physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3",
    "midogpp_virchow2_clipped_bbox_annotation_local_pooling_bc_bundle_seed42_v3",
    "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42",
    "midogpp_virchow2_physical_multiscale_clipped_bbox_annotation_local_pooling_11520_v3_seed42",
)


@dataclass(frozen=True)
class PhysicalMultiscaleWorkspaceBinding:
    """One immutable profile-to-workspace identity record."""

    profile_id: str
    experiment_id: str
    output_id: str
    input_ids: tuple[str, ...]
    promotion_gated_input_ids: tuple[str, ...]
    contract_id: str
    cache_bundle_id: str | None
    b_cache_id: str
    c_cache_id: str


CENTER_POOLING_WORKSPACE_BINDING_V1 = PhysicalMultiscaleWorkspaceBinding(
    profile_id=CENTER_POOLING_PILOT_V1,
    experiment_id=EXPERIMENT_ID,
    output_id=OUTPUT_ID,
    input_ids=INPUT_IDS,
    promotion_gated_input_ids=PROMOTION_GATED_INPUT_IDS,
    contract_id="midogpp_dataset_contract_physical_multiscale_center_pooling_pilot_v1",
    cache_bundle_id=None,
    b_cache_id="midogpp_virchow2_jpeg_center_pooling_3840_seed42",
    c_cache_id="midogpp_virchow2_physical_multiscale_center_pooling_11520_seed42",
)

ANNOTATION_LOCAL_WORKSPACE_BINDING_V2 = PhysicalMultiscaleWorkspaceBinding(
    profile_id=ANNOTATION_LOCAL_POOLING_PILOT_V2,
    experiment_id=EXPERIMENT_ID_V2,
    output_id=OUTPUT_ID_V2,
    input_ids=INPUT_IDS_V2,
    promotion_gated_input_ids=PROMOTION_GATED_INPUT_IDS_V2,
    contract_id=(
        "midogpp_dataset_contract_physical_multiscale_annotation_local_pooling_pilot_v2"
    ),
    cache_bundle_id=None,
    b_cache_id=(
        "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v2_seed42"
    ),
    c_cache_id=(
        "midogpp_virchow2_physical_multiscale_annotation_local_pooling_11520_v2_seed42"
    ),
)

CLIPPED_BBOX_ANNOTATION_LOCAL_WORKSPACE_BINDING_V3 = (
    PhysicalMultiscaleWorkspaceBinding(
        profile_id=CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
        experiment_id=EXPERIMENT_ID_V3,
        output_id=OUTPUT_ID_V3,
        input_ids=INPUT_IDS_V3,
        promotion_gated_input_ids=PROMOTION_GATED_INPUT_IDS_V3,
        contract_id=(
            "midogpp_dataset_contract_physical_multiscale_clipped_bbox_"
            "annotation_local_pooling_pilot_v3"
        ),
        cache_bundle_id=(
            "midogpp_virchow2_clipped_bbox_annotation_local_pooling_bc_bundle_"
            "seed42_v3"
        ),
        b_cache_id=(
            "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42"
        ),
        c_cache_id=(
            "midogpp_virchow2_physical_multiscale_clipped_bbox_"
            "annotation_local_pooling_11520_v3_seed42"
        ),
    )
)

PHYSICAL_MULTISCALE_WORKSPACE_BINDINGS = (
    CENTER_POOLING_WORKSPACE_BINDING_V1,
    ANNOTATION_LOCAL_WORKSPACE_BINDING_V2,
    CLIPPED_BBOX_ANNOTATION_LOCAL_WORKSPACE_BINDING_V3,
)
WORKSPACE_BINDINGS_BY_PROFILE_ID: Mapping[
    str,
    PhysicalMultiscaleWorkspaceBinding,
] = MappingProxyType(
    {
        binding.profile_id: binding
        for binding in PHYSICAL_MULTISCALE_WORKSPACE_BINDINGS
    }
)


def get_physical_multiscale_workspace_binding(
    profile_id: str,
) -> PhysicalMultiscaleWorkspaceBinding:
    try:
        return WORKSPACE_BINDINGS_BY_PROFILE_ID[str(profile_id)]
    except KeyError as exc:
        raise ProtocolError(
            f"Unsupported physical multiscale workspace profile: {profile_id!r}"
        ) from exc


def validate_production_workspace_binding(
    config: PhysicalMultiscalePilotConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    binding = get_physical_multiscale_workspace_binding(config.profile.profile_id)
    experiment = workspace.get_experiment(binding.experiment_id)
    if experiment.status != "diagnostic" or not experiment.runnable:
        raise ProtocolError(
            "Physical multiscale pilot may run only after reviewed diagnostic activation."
        )
    if (
        experiment.stage != "10_real_feature_reference"
        or experiment.claim_scope != "real_feature_transfer_only"
        or experiment.output_artifact_id != binding.output_id
        or experiment.input_artifact_ids != binding.input_ids
    ):
        raise ProtocolError("Physical multiscale workspace binding drifted.")
    for artifact_id in binding.promotion_gated_input_ids:
        artifact = workspace.artifacts[artifact_id]
        unhashed = set(artifact.required_files).difference(
            artifact.expected_file_hashes
        )
        if artifact.evidence_label.startswith("TODO_") or unhashed:
            raise ProtocolError(
                f"Physical multiscale input is not hash-promoted: {artifact_id}; "
                f"unhashed_required_files={sorted(unhashed)}"
            )
    expected_root = workspace.resolve_artifact(
        binding.output_id,
        for_output=True,
        require_exists=False,
    )
    if config.artifact_root.resolve() != expected_root.resolve():
        raise ProtocolError("Physical multiscale output root differs from catalog binding.")
    resolved_inputs = {
        "base_manifest": workspace.resolve_value(
            "artifact://midogpp_dataset_contract_annotation_patch_v1/manifest.csv",
            require_inputs=True,
        ),
        "physical_contract": workspace.resolve_value(
            f"artifact://{binding.contract_id}",
            require_inputs=True,
        ),
        "canonical_a_cache": workspace.resolve_value(
            "artifact://midogpp_virchow2_xyxy_feature_cache_seed42/embeddings/train.pt",
            require_inputs=True,
        ),
        **(
            {
                "cache_bundle": workspace.resolve_value(
                    f"artifact://{binding.cache_bundle_id}",
                    require_inputs=True,
                )
            }
            if binding.cache_bundle_id is not None
            else {}
        ),
        "b_cache": workspace.resolve_value(
            f"artifact://{binding.b_cache_id}",
            require_inputs=True,
        ),
        "c_cache": workspace.resolve_value(
            f"artifact://{binding.c_cache_id}",
            require_inputs=True,
        ),
        "canonical_reference": workspace.resolve_value(
            "artifact://midogpp_output_eligible_tuned_real_reference_v2",
            require_inputs=True,
        ),
    }
    configured = {
        "base_manifest": config.base_manifest_path,
        "physical_contract": config.physical_contract_root,
        "canonical_a_cache": config.canonical_a_cache_path,
        **(
            {"cache_bundle": config.cache_bundle_root}
            if config.cache_bundle_root is not None
            else {}
        ),
        "b_cache": config.b_cache_root,
        "c_cache": config.c_cache_root,
        "canonical_reference": config.canonical_reference_root,
    }
    for key, expected in resolved_inputs.items():
        if configured[key].resolve() != Path(str(expected)).resolve():
            raise ProtocolError(f"Physical multiscale input binding drifted: {key}")
