"""Pure protocol and candidate-pool identities for the study."""

from __future__ import annotations

from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import UniformBTaskGeometryConfig
from .contracts import ARMS, CLAIM_ROLE, CLAIM_SCOPE, COMPOSITION_MODES, EXPERIMENT_ID


def protocol_manifest(
    config: UniformBTaskGeometryConfig,
    *,
    manifest_hash: str,
    feature_cache_hash: str,
) -> dict[str, object]:
    if feature_cache_hash != config.expected_feature_cache_hash:
        raise ProtocolError("Canonical Uniform-B cache content hash mismatch.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_task_geometry_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "config_hash": config.contract_hash,
        "manifest_hash": manifest_hash,
        "feature_cache_hash": feature_cache_hash,
        "expected_feature_dim": 3840,
        "block_frame": "b_block_pca96_32",
        "arms": list(ARMS),
        "composition_modes": list(COMPOSITION_MODES),
        "outer_rows_used_for_fit": False,
        "inner_rows_used_for_fit": False,
        "inner_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
        "source_training_keys_are_outer_inner_neutral": True,
        "may_feed_recipe_selection": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream_utility": False,
        "separate_promotion_artifact_required": True,
        "probabilistic_semantics": {
            "prior": "standard_normal",
            "requested_class_law": "balanced_uniform_binary_intervention",
            "geco": "conditional_kl_rate_plus_mse_reconstruction_constraint",
            "task_loss_directly_matches_latent_posterior": False,
            "exact_nelbo_claimed": False,
        },
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def candidate_pool_manifest(
    centers: Sequence[str],
    *,
    outer_center: str,
    inner_center: str,
    base_per_class: int,
) -> dict[str, object]:
    outer = str(outer_center)
    inner = str(inner_center)
    if outer == inner:
        raise ProtocolError("Candidate pool requires H != I.")
    legal = tuple(
        center
        for center in sorted(str(value) for value in centers)
        if center not in {outer, inner}
    )
    if len(legal) != len(tuple(centers)) - 2 or not legal:
        raise ProtocolError("Candidate-pool exclusion is malformed.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_candidate_pool_v1",
        "outer_center": outer,
        "inner_center": inner,
        "legal_sources": list(legal),
        "outer_excluded": True,
        "inner_excluded": True,
        "canonical_source_order": True,
        "base_per_class": int(base_per_class),
        "sealed_before_inner_rows_loaded": True,
        "compatibility_signal": None,
        "routing_or_selection": False,
    }
    payload["candidate_pool_hash"] = stable_hash(payload)
    return payload


def validate_candidate_pool(
    payload: Mapping[str, object],
    centers: Sequence[str],
) -> None:
    expected = candidate_pool_manifest(
        centers,
        outer_center=str(payload.get("outer_center", "")),
        inner_center=str(payload.get("inner_center", "")),
        base_per_class=int(payload.get("base_per_class", 0)),
    )
    if dict(payload) != expected:
        raise ProtocolError("Candidate-pool manifest failed exact recomputation.")


__all__ = (
    "candidate_pool_manifest",
    "protocol_manifest",
    "validate_candidate_pool",
)
