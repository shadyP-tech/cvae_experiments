"""Protocol manifest and source-inner candidate identities."""

from __future__ import annotations

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import UniformBResampledPriorConfig
from .contracts import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    PRIORS,
    TRAINING_ARM,
)


def protocol_manifest(
    config: UniformBResampledPriorConfig,
    *,
    manifest_hash: str,
    feature_cache_hash: str,
) -> dict[str, object]:
    if feature_cache_hash != config.expected_feature_cache_hash:
        raise ProtocolError("Canonical Uniform-B cache content hash mismatch.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_resampled_prior_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "config_hash": config.contract_hash,
        "manifest_hash": manifest_hash,
        "feature_cache_hash": feature_cache_hash,
        "feature_dim": 3840,
        "source_frame": "b_block_pca96_32",
        "training_arm": TRAINING_ARM,
        "generation_priors": list(PRIORS),
        "fresh_bg_training_required": True,
        "existing_checkpoint_input_allowed": False,
        "parent_checkpoint_used": False,
        "source_only_ratio_fit": True,
        "outer_rows_used_for_fit": False,
        "inner_rows_used_for_fit": False,
        "inner_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
        "score_key_excludes_outer_center": True,
        "unique_score_reuse_required": True,
        "score_mapping_multiplicity": len(config.heldout_centers) - 2,
        "routing_or_selection": False,
        "may_feed_recipe_selection": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream_utility": False,
        "separate_promotion_artifact_required": True,
        "probabilistic_semantics": {
            "P0": "standard_normal",
            "Pq": "bounded_source_posterior_ratio_resampling_with_exact_p0_fallback",
            "ratio_is_true_utility": False,
            "exact_nelbo_claimed": False,
        },
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


__all__ = ("protocol_manifest",)
