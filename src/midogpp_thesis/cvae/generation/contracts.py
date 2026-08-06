"""Scientific identities for the frozen Uniform-B v2 generation contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID as EXPERT_BANK_ARTIFACT_ID,
    TRAINING_SEEDS,
)
from ..protocol import ProtocolError


EXPERIMENT_ID = "midogpp.prior_and_generation.uniform_b_v2_generation_lock.v1"
EXPERIMENT_NAME = "uniform_b_v2_generation_lock_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
CLAIM_SCOPE = "generation_settings_and_frame_lock"

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_CONTROL_LOCK_HASH = "cddbcc3b3343fe38"
EXPECTED_BANK_INDEX_SHA256 = (
    "5bc46728fd66d5c2c8a72d3da58cc6721e6c0b72c7291ed7b3b6931a4bcc41c9"
)
EXPECTED_CONTROL_LOCK_SHA256 = (
    "3cae13d5755f27643e0387b1f106f7745b470fb7aafe8d46be41677a1cd8eedd"
)
EXPECTED_CONTENT_INDEX_SHA256 = (
    "6b74fe794bd30cf6c1e42190427e506d1ff50ecd9280b9dcfee2a7592ec6a318"
)
EXPECTED_CONTENT_HASH = "fb1f1194be44ca41"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"

SAMPLER_FAMILY = "class_conditional_shrinkage_full_total_moment"
COMMON_OUTPUT_DIM = 3840
PROJECTED_DIM = 256
LATENT_DIM = 64
TOTAL_PER_CLASS = 1024
SOURCE_BUDGET_PER_CLASS = 128
SOURCE_STREAM_NAMESPACE = "uniform_b_v2_source_stream_v1"
COMPOSITION_SHUFFLE_NAMESPACE = "uniform_b_v2_composition_shuffle_v1"
REPLICATE_POLICY = "report_each_replication_and_predeclared_mean_no_seed_selection"


@dataclass(frozen=True)
class GenerationLock:
    """Immutable wrapper around the JSON generation-lock payload."""

    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        payload = dict(self._payload)
        observed = payload.get("generation_lock_hash")
        unhashed = {key: value for key, value in payload.items() if key != "generation_lock_hash"}
        if observed != stable_hash(unhashed):
            raise ProtocolError("Uniform-B v2 generation-lock hash drifted.")

    @property
    def generation_lock_hash(self) -> str:
        return str(self._payload["generation_lock_hash"])

    @property
    def bank_lock_hash(self) -> str:
        return str(self._payload["bank"]["bank_lock_hash"])  # type: ignore[index]

    def to_payload(self) -> dict[str, object]:
        return deepcopy(dict(self._payload))


@dataclass(frozen=True)
class SourceGenerationKey:
    source_center: str
    training_seed: int
    generation_seed: int
    expert_lock_hash: str
    stream_id: str
    class_seed_by_label: Mapping[str, int]
    max_samples_per_class: int = TOTAL_PER_CLASS
    equal_union_prefix_per_class: int = SOURCE_BUDGET_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_source_generation_key_v1",
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "expert_lock_hash": self.expert_lock_hash,
            "stream_id": self.stream_id,
            "class_seed_by_label": dict(self.class_seed_by_label),
            "max_samples_per_class": self.max_samples_per_class,
            "equal_union_prefix_per_class": self.equal_union_prefix_per_class,
        }


@dataclass(frozen=True)
class ControlReplicate:
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    candidate_source_centers: tuple[str, ...]
    source_stream_ids: tuple[str, ...]
    class_shuffle_seed_by_label: Mapping[str, int]
    source_budget_per_class: int = SOURCE_BUDGET_PER_CLASS
    total_per_class: int = TOTAL_PER_CLASS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_equal_union_replicate_v1",
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "replicate_id": self.replicate_id,
            "candidate_source_centers": list(self.candidate_source_centers),
            "source_stream_ids": list(self.source_stream_ids),
            "class_shuffle_seed_by_label": dict(self.class_shuffle_seed_by_label),
            "source_budget_per_class": self.source_budget_per_class,
            "total_per_class": self.total_per_class,
            "target_expert_excluded": True,
            "target_conditioned_source_weighting": False,
        }


__all__ = (
    "CENTERS",
    "CLAIM_SCOPE",
    "COMMON_OUTPUT_DIM",
    "COMPOSITION_SHUFFLE_NAMESPACE",
    "ControlReplicate",
    "EXPECTED_BANK_INDEX_SHA256",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_CONTENT_HASH",
    "EXPECTED_CONTENT_INDEX_SHA256",
    "EXPECTED_CONTROL_LOCK_HASH",
    "EXPECTED_CONTROL_LOCK_SHA256",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "GenerationLock",
    "LATENT_DIM",
    "OUTPUT_ARTIFACT_ID",
    "PROJECTED_DIM",
    "REPLICATE_POLICY",
    "SAMPLER_FAMILY",
    "SOURCE_BUDGET_PER_CLASS",
    "SOURCE_STREAM_NAMESPACE",
    "SourceGenerationKey",
    "TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
)
