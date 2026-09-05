"""Typed digest-width contracts for HARP v20 runtime boundaries."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


SEMANTIC_HASH16_FIELD_PATHS = (
    "config.inputs.expert_bank_lock_hash",
    "config.inputs.generation_lock_hash",
    "classifier_task.source_stream_lock_hash",
    "classifier_task.source_stream_index_hash",
    "classifier_task.source_record_projection_hash",
    "classifier_task.full_source_stream_index_hash",
    "classifier_task.source_records[].expert_lock_hash",
    "classifier_task.frame_receipt_hash",
    "classifier_task.frame_projection_hash",
    "classifier_checkpoint.actions[].scaler_state_hash",
    "resident_compatibility.replicas[].source_frame_hash",
    "resident_compatibility.replicas[].sampler_state_hash",
)
CONTENT_SHA256_FIELD_PATHS = (
    "classifier_task.source_stream_lock_sha256",
    "classifier_task.source_index_sha256",
    "classifier_task.source_array_sha256",
    "classifier_task.source_records[].output_sha256",
    "classifier_task.frame_receipt_sha256",
    "classifier_task.frame_array_sha256",
    "classifier_checkpoint.actions[].probability_sha256",
    "classifier_checkpoint.npz_sha256",
    "resident_compatibility.replicas[].checkpoint_sha256",
    "resident_compatibility.support_binding.frame_array_sha256",
)
CANONICAL_SHA256_FIELD_PATHS = (
    "classifier_task.task_hash",
    "classifier_task.actions[].action_hash",
    "classifier_checkpoint.actions[].composition_hash",
    "physical_input_receipt.receipt_hash",
    "label_free_outer_menu.menu_hash",
    "label_free_outer_menu.blocks[].seed_dispersion_bytes_sha256",
    "prelabel_routes.route_hash",
    "pooled_source_policy.model_hash",
    "source_target_compatibility.compatibility_feature_hash",
    "pooled_source_policy.policy_hash",
    "prelabel_routes.cases[].recipe_hash",
    "prelabel_routes.cases[].component_probability_sha256[]",
    "frozen_route_seal.seal_hash",
)


def require_sha256(value: object, *, name: str) -> str:
    """Require an exact lowercase 64-hex content digest."""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"HARP v20 {name} is malformed.")
    return value


def require_stable_hash(value: object, *, name: str) -> str:
    """Require the repository's exact lowercase 16-hex semantic identity."""

    if (
        type(value) is not str
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"HARP v20 {name} is malformed.")
    return value


def runtime_hash_contract_payload() -> Mapping[str, object]:
    """Return the frozen producer/worker/checkpoint digest-role inventory."""

    body: dict[str, object] = {
        "schema_version": "midogpp_harp_v20_runtime_hash_contract_v2",
        "semantic_hash_algorithm": "stable_hash_sha256_prefix_16_lower_hex",
        "semantic_hash_width": 16,
        "semantic_hash_field_paths": list(SEMANTIC_HASH16_FIELD_PATHS),
        "content_hash_algorithm": "sha256_64_lower_hex",
        "content_sha256_field_paths": list(CONTENT_SHA256_FIELD_PATHS),
        "canonical_hash_algorithm": "canonical_json_sha256_64_lower_hex",
        "canonical_sha256_field_paths": list(CANONICAL_SHA256_FIELD_PATHS),
        "mixed_width_field_names_allowed": False,
        "ambiguous_source_cache_hash_allowed": False,
        "compatibility_energy_exact_nelbo": False,
        "source_q_target_H_action_identity_shared_by_center_key": True,
        "selected_component_routes_reconstructed_from_physical_cache": True,
        "soft_topk_probability_mixtures_allowed": True,
        "all_k_lambda_probability_matrices_persisted": False,
        "soft_arm_gpu_or_classifier_fits": 0,
        "source_H_q_r_crossfit_used": False,
    }
    return MappingProxyType(
        {**body, "runtime_hash_contract_hash": canonical_hash(body)}
    )


__all__ = (
    "CANONICAL_SHA256_FIELD_PATHS",
    "CONTENT_SHA256_FIELD_PATHS",
    "SEMANTIC_HASH16_FIELD_PATHS",
    "require_sha256",
    "require_stable_hash",
    "runtime_hash_contract_payload",
)
