"""Closed-world file contract for HARP Stage-60 surface bundles."""

from __future__ import annotations


TRANSPORT_MANIFEST = "manifests/probability_menu_transport.json"
TRANSPORT_ARRAYS = "arrays/probability_menu_transport.npz"
CONFIG_MEMBER = "config.resolved.yaml"
PROVENANCE_MEMBER = "provenance/input_artifacts.json"
PROTOCOL_MEMBER = "manifests/protocol_manifest.json"
GLOBAL_SEAL_MEMBER = "manifests/global_prediction_seal.json"
SOURCE_CAPABILITY_SEAL_MEMBER = "manifests/source_label_capability_seal.json"
ACTION_FEATURE_MEMBER = "surfaces/action_features.json"
ACTION_RESPONSE_MEMBER = "surfaces/directional_responses.json"
TARGET_SUPPORT_MEMBER = "surfaces/target_support_features.json"
TRAINING_OBSERVATION_MEMBER = "surfaces/harp_training_observations.json"
ACTION_INFERENCE_BINDING_MEMBER = "manifests/harp_action_inference_binding.json"
ACTION_LOCK_MEMBER = "manifests/action_surface_lock.json"
TARGET_SUPPORT_LOCK_MEMBER = "manifests/target_support_surface_lock.json"
CONTENT_INDEX_MEMBER = "manifests/content_index.json"
PROBABILITY_ARRAY_MEMBER = "arrays/probability_menu.npy"
PROBABILITY_INDEX_MEMBER = "tables/probability_index.csv"
DIRECTIONAL_FEATURES_MEMBER = "tables/directional_features.csv"
DIRECTIONAL_RESPONSES_MEMBER = "tables/directional_responses.csv"
LEAKAGE_MEMBER = "reports/leakage_report.json"
PRODUCT_MEMBER = "manifests/product.json"
VALIDATION_MEMBER = "reports/validation_report.json"
STATE_MEMBER = "reports/run_state.json"
RESERVATION_MEMBER = "manifests/reservation.json"


COMMON_REQUIRED_MEMBERS = frozenset(
    {
        CONFIG_MEMBER,
        PROVENANCE_MEMBER,
        PROTOCOL_MEMBER,
        GLOBAL_SEAL_MEMBER,
        CONTENT_INDEX_MEMBER,
        PROBABILITY_ARRAY_MEMBER,
        PROBABILITY_INDEX_MEMBER,
        DIRECTIONAL_FEATURES_MEMBER,
        LEAKAGE_MEMBER,
        STATE_MEMBER,
        VALIDATION_MEMBER,
    }
)
ACTION_REQUIRED_MEMBERS = frozenset(
    {*COMMON_REQUIRED_MEMBERS, ACTION_LOCK_MEMBER, DIRECTIONAL_RESPONSES_MEMBER}
)
TARGET_REQUIRED_MEMBERS = frozenset(
    {*COMMON_REQUIRED_MEMBERS, TARGET_SUPPORT_LOCK_MEMBER}
)


__all__ = tuple(
    name
    for name in globals()
    if name.endswith("_MEMBER")
    or name.endswith("_MEMBERS")
    or name.startswith("TRANSPORT_")
)
