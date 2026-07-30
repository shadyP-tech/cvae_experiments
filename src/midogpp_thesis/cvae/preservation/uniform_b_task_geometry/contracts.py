"""Immutable public identities for the Uniform-B task-geometry study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ....common.hashing import stable_hash
from ...protocol import ProtocolError


EXPERIMENT_ID = "midogpp.cvae.uniform_b_geco_task_geometry_source_inner.v1"
STUDY_NAME = "virchow2_cvae_midogpp_uniform_b_geco_task_geometry_source_inner_v1"
MODE = "uniform_b_geco_task_geometry_source_inner_study"
STUDY_VERSION = "v1"
CLAIM_SCOPE = "cvae_source_inner_study_only"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_cvae_uniform_b_geco_task_geometry_source_inner_v1"
)
UNIFORM_B_INPUT_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
)
UNIFORM_B_FEATURE_HASH = (
    "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
)

BF = "BF"
BG = "BG"
BM = "BM"
BT = "BT"
ARMS = (BF, BG, BM, BT)

SINGLE_BASE = "single_base"
SINGLE_BUDGET_MATCHED = "single_budget_matched"
UNION_EQUAL_TOTAL = "union_equal_total"
UNION_EXPANDED = "union_expanded"
COMPOSITION_MODES = (
    SINGLE_BASE,
    SINGLE_BUDGET_MATCHED,
    UNION_EQUAL_TOTAL,
    UNION_EXPANDED,
)

PUBLICATION_STATE = "NON_CONSUMABLE_STUDY_COMPLETE"
CLAIM_ROLE = "held_out_inner_discriminative_prior_tstr_diagnostic"


def objective_family(arm: str) -> str:
    if arm == BF:
        return "fixed_beta"
    if arm in {BG, BM, BT}:
        return "geco"
    raise ProtocolError(f"Unknown Uniform-B task-geometry arm: {arm!r}")


def task_family(arm: str) -> str:
    mapping = {
        BF: "none",
        BG: "none",
        BM: "class_conditional_multiscale_mmd",
        BT: "mmd_plus_combined_margin_cdf_and_curvature_gradient",
    }
    try:
        return mapping[str(arm)]
    except KeyError as exc:
        raise ProtocolError(f"Unknown task family for arm {arm!r}.") from exc


@dataclass(frozen=True)
class SourceTrainingKey:
    source_center: str
    training_seed: int
    arm: str
    source_row_hash: str
    source_case_hash: str
    frame_hash: str
    task_lock_hash: str
    manifest_hash: str
    feature_cache_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS or not self.source_center:
            raise ProtocolError("Malformed Uniform-B source-training key.")
        for value in (
            self.source_row_hash,
            self.source_case_hash,
            self.frame_hash,
            self.task_lock_hash,
            self.manifest_hash,
            self.feature_cache_hash,
            self.config_hash,
        ):
            if not value:
                raise ProtocolError("Source-training key contains an empty hash.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    @property
    def arm_neutral_hash(self) -> str:
        payload = self.to_payload()
        payload.pop("arm")
        payload.pop("task_family")
        payload.pop("task_lock_hash")
        return stable_hash(payload)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_source_training_key_v1",
            **asdict(self),
            "objective_family": objective_family(self.arm),
            "task_family": task_family(self.arm),
            "outer_or_inner_identity_present": False,
        }


@dataclass(frozen=True)
class EvaluationKey:
    outer_center: str
    inner_center: str
    legal_sources: tuple[str, ...]
    arm: str
    training_seed: int
    generation_seed: int
    composition_mode: str
    candidate_pool_hash: str

    def __post_init__(self) -> None:
        if (
            self.arm not in ARMS
            or self.composition_mode not in COMPOSITION_MODES
            or self.outer_center == self.inner_center
            or not self.legal_sources
            or any(
                source in {self.outer_center, self.inner_center}
                for source in self.legal_sources
            )
        ):
            raise ProtocolError("Malformed Uniform-B evaluation key.")

    @property
    def hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_uniform_b_evaluation_key_v1",
                **asdict(self),
            }
        )


__all__ = (
    "ARMS",
    "BF",
    "BG",
    "BM",
    "BT",
    "CLAIM_ROLE",
    "CLAIM_SCOPE",
    "COMPOSITION_MODES",
    "EXPERIMENT_ID",
    "EvaluationKey",
    "MODE",
    "OUTPUT_ARTIFACT_ID",
    "PUBLICATION_STATE",
    "SINGLE_BASE",
    "SINGLE_BUDGET_MATCHED",
    "STUDY_NAME",
    "STUDY_VERSION",
    "SourceTrainingKey",
    "UNIFORM_B_FEATURE_HASH",
    "UNIFORM_B_INPUT_ARTIFACT_ID",
    "UNION_EQUAL_TOTAL",
    "UNION_EXPANDED",
    "objective_family",
    "task_family",
)
