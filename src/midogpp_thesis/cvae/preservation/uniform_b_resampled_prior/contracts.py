"""Immutable public identities and score keys for the P0/Pq study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ....common.hashing import stable_hash
from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.cvae.uniform_b_geco_posterior_resampled_prior_source_inner.v1"
)
STUDY_NAME = (
    "virchow2_cvae_midogpp_uniform_b_geco_"
    "posterior_resampled_prior_source_inner_v1"
)
MODE = "uniform_b_geco_posterior_resampled_prior_source_inner_study"
STUDY_VERSION = "v1"
CLAIM_SCOPE = "cvae_source_inner_study_only"
CLAIM_ROLE = "held_out_inner_p0_vs_pq_prior_tstr_diagnostic"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_cvae_uniform_b_geco_"
    "posterior_resampled_prior_source_inner_v1"
)
UNIFORM_B_INPUT_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
)
UNIFORM_B_FEATURE_HASH = (
    "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
)

P0 = "P0"
PQ = "Pq"
PRIORS = (P0, PQ)
TRAINING_ARM = "BG"
COMPOSITION_MODE = "single_base"
PUBLICATION_STATE = "NON_CONSUMABLE_STUDY_COMPLETE"


@dataclass(frozen=True)
class SourceTrainingKey:
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
            raise ProtocolError("Malformed resampled-prior training key.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_resampled_prior_training_key_v1",
            **asdict(self),
            "training_arm": TRAINING_ARM,
            "fresh_training": True,
            "parent_checkpoint_used": False,
            "outer_or_inner_identity_present": False,
        }


@dataclass(frozen=True)
class UniqueScoreKey:
    source_center: str
    inner_center: str
    training_seed: int
    generation_seed: int
    prior: str
    block_hash: str
    inner_row_hash: str
    classifier_spec_hash: str

    def __post_init__(self) -> None:
        if (
            self.source_center == self.inner_center
            or self.prior not in PRIORS
            or not self.block_hash
            or not self.inner_row_hash
            or not self.classifier_spec_hash
        ):
            raise ProtocolError("Malformed unique P0/Pq score key.")

    @property
    def hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_resampled_prior_unique_score_key_v1",
                **asdict(self),
                "outer_center_present": False,
            }
        )


def valid_outer_centers(
    centers: tuple[str, ...],
    *,
    source_center: str,
    inner_center: str,
) -> tuple[str, ...]:
    if source_center == inner_center:
        raise ProtocolError("Source and inner center must differ.")
    outers = tuple(
        center
        for center in centers
        if center not in {str(source_center), str(inner_center)}
    )
    if len(outers) != len(centers) - 2:
        raise ProtocolError("Unique score has an invalid outer mapping.")
    return outers


__all__ = (
    "CLAIM_ROLE",
    "CLAIM_SCOPE",
    "COMPOSITION_MODE",
    "EXPERIMENT_ID",
    "MODE",
    "OUTPUT_ARTIFACT_ID",
    "P0",
    "PQ",
    "PRIORS",
    "PUBLICATION_STATE",
    "STUDY_NAME",
    "STUDY_VERSION",
    "SourceTrainingKey",
    "TRAINING_ARM",
    "UNIFORM_B_FEATURE_HASH",
    "UNIFORM_B_INPUT_ARTIFACT_ID",
    "UniqueScoreKey",
    "valid_outer_centers",
)
