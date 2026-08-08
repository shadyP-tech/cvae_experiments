"""Narrow frozen identities shared by the utility-aligned artifact family."""

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")

EXACT_TAIL_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_exact_tail_utility_surface_v1"
)
DEVELOPMENT_RESERVATION_ARTIFACT_ID = (
    "midogpp_utility_aligned_router_development_reservation_v1"
)
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)
TARGET_SUPPORT_SURFACE_ARTIFACT_ID = (
    "midogpp_utility_aligned_target_support_surface_v1"
)
TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID = (
    "midogpp_utility_aligned_target_support_reservation_v1"
)
TARGET_SUPPORT_CACHE_ARTIFACT_ID = (
    "midogpp_utility_aligned_fresh_target_support_cache_v1"
)
TARGET_RESERVATION_ARTIFACT_ID = (
    "midogpp_utility_aligned_fresh_target_reservation_v1"
)
METADATA_PROFILE_ARTIFACT_ID = "midogpp_routing_metadata_profiles_v1"
METADATA_PROFILE_MEMBER = "domain_mapping.json"
METADATA_PROFILE_SHA256 = (
    "79d703ccf3085ae3968698c2ac44a3eabc2713b434762cc6b2fd2fa90126a211"
)
EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"

TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_aligned_target_support_surface.v1"
)
POLICY_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_aligned_residual_policy_lock.v1"
)
STAGE70_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_utility_aligned_residual_fresh.v1"
)


__all__ = (
    "CENTERS",
    "DEVELOPMENT_RESERVATION_ARTIFACT_ID",
    "EQUAL_UNION_POLICY_ARTIFACT_ID",
    "EXACT_TAIL_OUTPUT_ARTIFACT_ID",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_LOCK_ARTIFACT_ID",
    "METADATA_PROFILE_ARTIFACT_ID",
    "METADATA_PROFILE_MEMBER",
    "METADATA_PROFILE_SHA256",
    "POLICY_EXPERIMENT_ID",
    "STAGE70_EXPERIMENT_ID",
    "TARGET_RESERVATION_ARTIFACT_ID",
    "TARGET_SUPPORT_CACHE_ARTIFACT_ID",
    "TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID",
    "TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID",
    "TARGET_SUPPORT_SURFACE_ARTIFACT_ID",
)
