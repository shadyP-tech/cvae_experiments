"""Closed workspace identities for the three HARP Stage-60 surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError


STAGE_ID = "60_routing_and_composition"
CONFIG_SCHEMA_VERSION = "midogpp_harp_stage60_config_v1"
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EXACT_B_POLICY_ARTIFACT_ID = "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"

DEVELOPMENT_RESERVATION_ARTIFACT_ID = "midogpp_harp_router_development_reservation_v1"
DEVELOPMENT_CACHE_ARTIFACT_ID = "midogpp_harp_router_development_cache_v1"
DEVELOPMENT_MANIFEST_ARTIFACT_ID = "midogpp_harp_router_development_manifest_v1"
TARGET_SUPPORT_RESERVATION_ARTIFACT_ID = "midogpp_harp_target_support_reservation_v1"
TARGET_SUPPORT_CACHE_ARTIFACT_ID = "midogpp_harp_target_support_cache_v1"
FRESH_TARGET_RESERVATION_ARTIFACT_ID = "midogpp_harp_fresh_target_reservation_v1"


@dataclass(frozen=True)
class HarpSurfaceContract:
    surface: str
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    claim_scope: str
    input_path_keys: tuple[str, ...]
    readiness_member: str

    def __post_init__(self) -> None:
        if (
            not self.surface
            or not self.experiment_id
            or not self.output_artifact_id
            or not self.input_artifact_ids
            or len(self.input_artifact_ids) != len(set(self.input_artifact_ids))
            or not self.input_path_keys
            or len(self.input_path_keys) != len(set(self.input_path_keys))
            or self.claim_scope
            not in {"routing_and_composition", "routing_compatibility_only"}
        ):
            raise ProtocolError("HARP Stage-60 surface contract is malformed.")
        forbidden = ("stage90", "sceptre", "oracle")
        values = (self.experiment_id, self.output_artifact_id, *self.input_artifact_ids)
        if any(token in value.lower() for value in values for token in forbidden):
            raise ProtocolError("HARP Stage-60 graph references a forbidden lineage.")


ACTION_SURFACE = HarpSurfaceContract(
    surface="uniform-b-v2-harp-action-surface",
    experiment_id=(
        "midogpp.routing_and_composition.uniform_b_v2_harp_action_surface.v1"
    ),
    output_artifact_id="midogpp_output_uniform_b_v2_harp_action_surface_v1",
    input_artifact_ids=(
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
        DEVELOPMENT_RESERVATION_ARTIFACT_ID,
        DEVELOPMENT_CACHE_ARTIFACT_ID,
        DEVELOPMENT_MANIFEST_ARTIFACT_ID,
    ),
    claim_scope="routing_and_composition",
    input_path_keys=(
        "expert_bank_root",
        "generation_lock_root",
        "development_reservation_root",
        "development_cache_root",
        "development_manifest_path",
        "readiness_attestation_path",
    ),
    readiness_member="manifests/harp_fresh_development_attestation.json",
)

TARGET_SUPPORT_SURFACE = HarpSurfaceContract(
    surface="uniform-b-v2-harp-target-support-surface",
    experiment_id=(
        "midogpp.routing_and_composition.uniform_b_v2_harp_target_support_surface.v1"
    ),
    output_artifact_id="midogpp_output_uniform_b_v2_harp_target_support_surface_v1",
    input_artifact_ids=(
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
        TARGET_SUPPORT_RESERVATION_ARTIFACT_ID,
        TARGET_SUPPORT_CACHE_ARTIFACT_ID,
    ),
    claim_scope="routing_compatibility_only",
    input_path_keys=(
        "expert_bank_root",
        "generation_lock_root",
        "target_support_reservation_root",
        "target_support_cache_root",
        "readiness_attestation_path",
    ),
    readiness_member="manifests/harp_fresh_support_attestation.json",
)

POLICY_LOCK = HarpSurfaceContract(
    surface="uniform-b-v2-harp-policy-lock",
    experiment_id="midogpp.routing_and_composition.uniform_b_v2_harp_policy_lock.v1",
    output_artifact_id="midogpp_output_uniform_b_v2_harp_policy_lock_v1",
    input_artifact_ids=(
        ACTION_SURFACE.output_artifact_id,
        EXACT_B_POLICY_ARTIFACT_ID,
        TARGET_SUPPORT_SURFACE.output_artifact_id,
        TARGET_SUPPORT_RESERVATION_ARTIFACT_ID,
        FRESH_TARGET_RESERVATION_ARTIFACT_ID,
    ),
    claim_scope="routing_and_composition",
    input_path_keys=(
        "action_surface_root",
        "exact_b_policy_root",
        "target_support_surface_root",
        "target_support_reservation_root",
        "fresh_target_reservation_root",
        "readiness_attestation_path",
    ),
    readiness_member="manifests/harp_policy_inputs_attestation.json",
)


SURFACE_CONTRACTS: Mapping[str, HarpSurfaceContract] = MappingProxyType(
    {
        contract.surface: contract
        for contract in (ACTION_SURFACE, TARGET_SUPPORT_SURFACE, POLICY_LOCK)
    }
)


def surface_contract(surface: object) -> HarpSurfaceContract:
    try:
        return SURFACE_CONTRACTS[str(surface)]
    except KeyError as exc:
        raise ProtocolError(f"Unknown HARP Stage-60 surface: {surface!r}.") from exc


__all__ = (
    "ACTION_SURFACE",
    "CENTERS",
    "CONFIG_SCHEMA_VERSION",
    "DEVELOPMENT_CACHE_ARTIFACT_ID",
    "DEVELOPMENT_MANIFEST_ARTIFACT_ID",
    "DEVELOPMENT_RESERVATION_ARTIFACT_ID",
    "EXACT_B_POLICY_ARTIFACT_ID",
    "EXPERT_BANK_ARTIFACT_ID",
    "FRESH_TARGET_RESERVATION_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "HarpSurfaceContract",
    "POLICY_LOCK",
    "STAGE_ID",
    "SURFACE_CONTRACTS",
    "TARGET_SUPPORT_CACHE_ARTIFACT_ID",
    "TARGET_SUPPORT_RESERVATION_ARTIFACT_ID",
    "TARGET_SUPPORT_SURFACE",
    "surface_contract",
)
