"""Frozen experiment profiles sharing the resumable MMD/KMM execution engine."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError


POOLED_ROUTER_MODE = "pooled_mmd_kmm"
CONDITIONAL_ROUTER_MODE = "class_conditional_contrast_mmd_kmm"


@dataclass(frozen=True)
class MMDKMMExperimentProfile:
    experiment_id: str
    experiment_name: str
    output_artifact_id: str
    validation_cache_artifact_id: str
    validation_manifest_artifact_id: str
    router_mode: str
    kmm_regularization: float
    maximum_support_l1: float
    maximum_training_seed_l1: float
    maximum_generation_seed_l1: float
    maximum_prior_sensitivity_l1: float
    minimum_direction_cosine: float

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return (
            "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
            "midogpp_output_uniform_b_v2_generation_lock_v1",
            "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1",
            self.validation_cache_artifact_id,
            self.validation_manifest_artifact_id,
        )


POOLED_PROFILE = MMDKMMExperimentProfile(
    experiment_id="midogpp.oracle.uniform_b_v2_consumed_validation_mmd_kmm_router.v1",
    experiment_name="uniform_b_v2_consumed_validation_mmd_kmm_router_v1",
    output_artifact_id="midogpp_output_uniform_b_v2_consumed_validation_mmd_kmm_router_v1",
    validation_cache_artifact_id="midogpp_stage90_mmd_kmm_router_validation_cache_v1",
    validation_manifest_artifact_id="midogpp_stage90_mmd_kmm_router_validation_manifest_v1",
    router_mode=POOLED_ROUTER_MODE,
    kmm_regularization=0.05,
    maximum_support_l1=0.50,
    maximum_training_seed_l1=0.35,
    maximum_generation_seed_l1=0.35,
    maximum_prior_sensitivity_l1=0.35,
    minimum_direction_cosine=0.0,
)


CONDITIONAL_PROFILE = MMDKMMExperimentProfile(
    experiment_id=(
        "midogpp.oracle.uniform_b_v2_consumed_validation_"
        "conditional_contrast_mmd_router.v1"
    ),
    experiment_name=(
        "uniform_b_v2_consumed_validation_conditional_contrast_mmd_router_v1"
    ),
    output_artifact_id=(
        "midogpp_output_uniform_b_v2_consumed_validation_"
        "conditional_contrast_mmd_router_v1"
    ),
    validation_cache_artifact_id=(
        "midogpp_stage90_conditional_contrast_mmd_router_validation_cache_v1"
    ),
    validation_manifest_artifact_id=(
        "midogpp_stage90_conditional_contrast_mmd_router_validation_manifest_v1"
    ),
    router_mode=CONDITIONAL_ROUTER_MODE,
    # Frozen before target scoring.  The stronger anchor and gates respond to
    # the observed two-case support uncertainty without using target utility.
    kmm_regularization=0.10,
    maximum_support_l1=0.20,
    maximum_training_seed_l1=0.20,
    maximum_generation_seed_l1=0.20,
    maximum_prior_sensitivity_l1=0.15,
    minimum_direction_cosine=0.50,
)


PROFILES_BY_EXPERIMENT_ID = {
    profile.experiment_id: profile
    for profile in (POOLED_PROFILE, CONDITIONAL_PROFILE)
}


def profile_for_experiment(experiment_id: object) -> MMDKMMExperimentProfile:
    try:
        return PROFILES_BY_EXPERIMENT_ID[str(experiment_id)]
    except KeyError as exc:
        raise ProtocolError("MMD/KMM experiment profile is not authorized.") from exc


__all__ = (
    "CONDITIONAL_PROFILE",
    "CONDITIONAL_ROUTER_MODE",
    "MMDKMMExperimentProfile",
    "POOLED_PROFILE",
    "POOLED_ROUTER_MODE",
    "profile_for_experiment",
)
