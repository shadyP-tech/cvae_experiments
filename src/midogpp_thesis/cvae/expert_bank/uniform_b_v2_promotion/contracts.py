"""Immutable identities for the reviewed Uniform-B v2 expert-bank promotion."""

from __future__ import annotations

from ...preservation.uniform_b_optimized_prior.contracts import (
    EXPERIMENT_ID as SOURCE_EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID as SOURCE_ARTIFACT_ID,
)


EXPERIMENT_ID = "midogpp.expert_bank.uniform_b_v2_routing_promotion.v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
EXPERIMENT_NAME = "uniform_b_v2_routing_authorized_expert_bank_v1"
PROMOTION_REVIEW_ID = "uniform_b_v2_expert_bank_routing_promotion_review_2026_08_05"
PROMOTION_DECISION = "PROMOTED_AS_ROUTING_AUTHORIZED_EXPERT_BANK"
PUBLICATION_STATE = "ROUTING_AUTHORIZED"
CLAIM_SCOPE = "expert_bank_construction_only"

SOURCE_PROTOCOL_HASH = "70d54442a031a43e"
SOURCE_CONFIG_HASH = "09d5b264d066633b"
SOURCE_CONTENT_INDEX_SHA256 = (
    "7fae389151618950102905f85c5fce300ef9821f15cf304d8469b5a147110279"
)
SOURCE_CHECKPOINT_INDEX_SHA256 = (
    "15691c0414ed0cad7796c91a08965c652a3513064e3d43de1e8d24277d4fe1ac"
)
SOURCE_FRAME_INDEX_SHA256 = (
    "95bfccd220b9aee61b2c7487ad1da51f6b8b1e575db923f016489c6a3204cd07"
)
SOURCE_DECISION_SHA256 = (
    "15da51d54823c00ed8d070f58d5e1373fd547daadcc8d18f6fc3b2e36c3addad"
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
N_EXPERTS = len(CENTERS) * len(TRAINING_SEEDS)
CONTROL_TOTAL_PER_CLASS = 1024
CONTROL_SAMPLER_FAMILY = "class_conditional_shrinkage_full_total_moment"


def legal_routing_sources(target_center: str) -> tuple[str, ...]:
    """Return the frozen all-source control pool for one held-out target."""

    target = str(target_center)
    if target not in CENTERS:
        raise ValueError(f"Unknown MIDOG++ target center: {target!r}.")
    return tuple(center for center in CENTERS if center != target)


__all__ = (
    "CENTERS",
    "CLAIM_SCOPE",
    "CONTROL_SAMPLER_FAMILY",
    "CONTROL_TOTAL_PER_CLASS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GENERATION_SEEDS",
    "N_EXPERTS",
    "OUTPUT_ARTIFACT_ID",
    "PROMOTION_DECISION",
    "PROMOTION_REVIEW_ID",
    "PUBLICATION_STATE",
    "SOURCE_ARTIFACT_ID",
    "SOURCE_CHECKPOINT_INDEX_SHA256",
    "SOURCE_CONFIG_HASH",
    "SOURCE_CONTENT_INDEX_SHA256",
    "SOURCE_DECISION_SHA256",
    "SOURCE_EXPERIMENT_ID",
    "SOURCE_FRAME_INDEX_SHA256",
    "SOURCE_PROTOCOL_HASH",
    "TRAINING_SEEDS",
    "legal_routing_sources",
)
