"""Immutable identities for the capacity-and-composition Uniform-B prior study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ....common.hashing import stable_hash
from ...protocol import ProtocolError

EXPERIMENT_ID = "midogpp.cvae.uniform_b_geco_aggregate_prior_union_source_inner.v2"
STUDY_NAME = "virchow2_cvae_midogpp_uniform_b_geco_aggregate_prior_union_source_inner_v2"
MODE = "uniform_b_geco_aggregate_prior_union_source_inner_study"
STUDY_VERSION = "v2"
OUTPUT_ARTIFACT_ID = "midogpp_output_cvae_uniform_b_geco_aggregate_prior_union_source_inner_v2"
UNIFORM_B_INPUT_ARTIFACT_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
UNIFORM_B_FEATURE_HASH = "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"

P0 = "P0"
PS = "PS"
Q = "Q"
QM = "QM"
R = "R"
ARMS = (P0, PS, Q, QM, R)
FRAME = "b_block_pca192_64"
COMPOSITION_MODE = "union_equal_total"
CLAIM_SCOPE = "cvae_source_inner_study_only"
PUBLICATION_STATE = "NON_CONSUMABLE_STUDY_COMPLETE"


@dataclass(frozen=True)
class OptimizedTrainingKey:
    source_center: str
    training_seed: int
    source_row_hash: str
    source_case_hash: str
    frame_hash: str
    manifest_hash: str
    feature_cache_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        if not self.source_center or any(
            not value
            for value in (
                self.source_row_hash,
                self.source_case_hash,
                self.frame_hash,
                self.manifest_hash,
                self.feature_cache_hash,
                self.config_hash,
            )
        ):
            raise ProtocolError("Malformed optimized-prior training key.")

    @property
    def hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_uniform_b_optimized_training_key_v2",
                **asdict(self),
                "fresh_source_only_training": True,
                "outer_or_inner_identity_present": False,
            }
        )


def legal_sources(
    centers: tuple[str, ...], *, outer_center: str, inner_center: str
) -> tuple[str, ...]:
    if outer_center == inner_center:
        raise ProtocolError("Outer and inner centers must differ.")
    sources = tuple(
        center for center in centers if center not in {outer_center, inner_center}
    )
    if len(sources) != len(centers) - 2:
        raise ProtocolError("Optimized-prior task must have exactly seven sources.")
    return sources


__all__ = (
    "ARMS", "CLAIM_SCOPE", "COMPOSITION_MODE", "EXPERIMENT_ID", "FRAME",
    "MODE", "OptimizedTrainingKey", "OUTPUT_ARTIFACT_ID", "P0",
    "PUBLICATION_STATE", "PS", "Q", "QM", "R", "STUDY_NAME", "STUDY_VERSION",
    "UNIFORM_B_FEATURE_HASH", "UNIFORM_B_INPUT_ARTIFACT_ID", "legal_sources",
)
