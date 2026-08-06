"""Closed scientific contracts for the consumed-validation residual router.

This package is deliberately a Stage-90 diagnostic.  It reuses already
consumed MIDOG++ validation bytes to answer a mechanism question and is
structurally incapable of authorizing a Stage-60 policy, Stage-70 evaluation,
promotion, or deployment claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import (
    COMMON_OUTPUT_DIM,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    TOTAL_PER_CLASS,
)
from ...protocol import ProtocolError
from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)


EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_validation_dense_residual_router.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_consumed_validation_dense_residual_router_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_dense_residual_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_dense_residual_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_dense_residual_router_validation_manifest_v1"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)

ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_v2_routing_validation_cache_seed42"
)
ORIGINAL_STAGE60_VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_source_inner_validation_manifest_v1"
)
FORBIDDEN_STAGE60_INPUT_ARTIFACT_IDS = frozenset(
    {
        ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID,
        ORIGINAL_STAGE60_VALIDATION_MANIFEST_ARTIFACT_ID,
    }
)

EXCLUDED_CENTER = "4"
VALIDATION_SPLIT = "val"
SUPPORT_CASE_COUNT = 2
SUPPORT_SPLIT_SEED = 20260806
SUPPORT_PARTITION_NAMESPACE = "midogpp_dense_residual_support_v1"

COMPATIBILITY_SEMANTICS = "aggregate_prior_variational_compatibility_energy_v1"
CLASS_PRIOR = (0.5, 0.5)
RHO_VALUES = (0.0, 0.25, 0.5)
TEMPERATURE = 1.0
MAX_SOURCE_WEIGHT = 0.25
MIN_EFFECTIVE_SOURCE_COUNT = 6.0
MINIMUM_INTEGER_ALLOCATION_PER_SOURCE = 1
DEVELOPMENT_TOTAL_PER_CLASS = 1008
ACTION_IDS = tuple(f"rho_{rho:.2f}" for rho in RHO_VALUES)
CONTROL_ACTION_ID = ACTION_IDS[0]

PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "macro_f1_descriptive_only"
SELECTION_OBJECTIVE = (
    "mean_regret_plus_0.5_upper_quartile_cvar_regret_plus_"
    "0.01_mean_squared_l2_distance_from_uniform"
)
NONUNIFORM_PASS_RULE = "strictly_positive_mean_paired_bacc_delta_vs_rho0"
FALLBACK_ACTION_ID = CONTROL_ACTION_ID

CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


def legal_sources(*, outer_target: str, query_center: str) -> tuple[str, ...]:
    """Return the canonical source pool satisfying ``e != H`` and ``e != q``."""

    outer = str(outer_target)
    query = str(query_center)
    if outer not in CENTERS or query not in CENTERS:
        raise ProtocolError("Dense residual routing contains an unknown center.")
    if outer == query:
        raise ProtocolError("Development query center must differ from outer target H.")
    return tuple(center for center in CENTERS if center not in {outer, query})


def target_sources(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Dense residual target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def development_queries(outer_target: str) -> tuple[str, ...]:
    outer = str(outer_target)
    if outer not in CENTERS:
        raise ProtocolError("Dense residual outer target center is unknown.")
    return tuple(center for center in CENTERS if center != outer)


def seed_cells() -> tuple[tuple[int, int], ...]:
    return tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))


EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT = sum(
    len(development_queries(outer_target)) * len(ACTION_IDS) * len(seed_cells())
    for outer_target in CENTERS
)
EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT = (
    len(CENTERS) * len(ACTION_IDS) * len(seed_cells())
)
EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT = (
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
    + EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
)
EXPECTED_TARGET_PREDICTION_CELL_COUNT = (
    len(CENTERS) * (len(ACTION_IDS) + 1) * len(seed_cells())
)
MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS = len(CENTERS)
MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES = (
    MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS
    * 2
    * TOTAL_PER_CLASS
    * COMMON_OUTPUT_DIM
    * 4
)


@dataclass(frozen=True)
class ActionSpec:
    """One predeclared dense residual action."""

    action_id: str
    rho: float
    temperature: float = TEMPERATURE
    max_source_weight: float = MAX_SOURCE_WEIGHT
    min_effective_source_count: float = MIN_EFFECTIVE_SOURCE_COUNT
    minimum_integer_allocation_per_source: int = MINIMUM_INTEGER_ALLOCATION_PER_SOURCE

    def __post_init__(self) -> None:
        if isinstance(self.rho, bool) or not isinstance(self.rho, (int, float)):
            raise ProtocolError("Dense residual action rho must be numeric.")
        if self.action_id not in ACTION_IDS:
            raise ProtocolError(f"Unknown dense residual action: {self.action_id!r}.")
        expected_rho = RHO_VALUES[ACTION_IDS.index(self.action_id)]
        if float(self.rho) != expected_rho:
            raise ProtocolError("Dense residual action id/rho binding drifted.")
        if self.temperature != TEMPERATURE or self.temperature <= 0.0:
            raise ProtocolError("Dense residual temperature drifted.")
        if self.max_source_weight != MAX_SOURCE_WEIGHT:
            raise ProtocolError("Dense residual maximum source weight drifted.")
        if self.min_effective_source_count != MIN_EFFECTIVE_SOURCE_COUNT:
            raise ProtocolError("Dense residual effective-source constraint drifted.")
        if (
            self.minimum_integer_allocation_per_source
            != MINIMUM_INTEGER_ALLOCATION_PER_SOURCE
        ):
            raise ProtocolError("Dense residual integer source floor drifted.")

    @property
    def exact_equal_union(self) -> bool:
        return self.rho == 0.0

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_dense_residual_action_v1",
            "action_id": self.action_id,
            "rho": self.rho,
            "temperature": self.temperature,
            "max_source_weight": self.max_source_weight,
            "min_effective_source_count": self.min_effective_source_count,
            "minimum_integer_allocation_per_source": (
                self.minimum_integer_allocation_per_source
            ),
            "exact_equal_union": self.exact_equal_union,
            "development_total_generated_samples_per_class": (
                DEVELOPMENT_TOTAL_PER_CLASS
            ),
            "target_total_generated_samples_per_class": TOTAL_PER_CLASS,
        }


def action_library() -> tuple[ActionSpec, ...]:
    return tuple(ActionSpec(action_id=action_id, rho=rho) for action_id, rho in zip(ACTION_IDS, RHO_VALUES, strict=True))


ACTION_LIBRARY_HASH = stable_hash([action.to_payload() for action in action_library()])


@dataclass(frozen=True)
class ValidationRowIdentity:
    """Label-free identity for one consumed validation row."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str = VALIDATION_SPLIT
    partition_role: str = "evaluation"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or isinstance(self.manifest_row_index, bool)
            or not isinstance(self.row_ordinal, int)
            or not isinstance(self.manifest_row_index, int)
            or self.row_ordinal < 0
            or self.manifest_row_index < 0
        ):
            raise ProtocolError("Dense residual row indices must be nonnegative integers.")
        if not self.sample_id or not self.case_id:
            raise ProtocolError("Dense residual rows require sample and case identities.")
        if self.center not in CENTERS or self.split != VALIDATION_SPLIT:
            raise ProtocolError("Dense residual rows must be eligible MIDOG++ val rows.")
        if self.partition_role not in {"support", "evaluation"}:
            raise ProtocolError("Dense residual row partition role is invalid.")

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "partition_role": self.partition_role,
        }


def row_identity_hash(rows: Sequence[ValidationRowIdentity]) -> str:
    return stable_hash([row.identity_payload() for row in rows])


@dataclass(frozen=True)
class OpenedLabelVector:
    """A narrowly opened, ordered label vector bound to a prediction seal."""

    outer_target: str
    phase: str
    rows: tuple[ValidationRowIdentity, ...]
    labels: tuple[int, ...]
    manifest_sha256: str
    prediction_seal_hash: str
    label_vector_hash: str

    def __post_init__(self) -> None:
        if self.outer_target not in CENTERS:
            raise ProtocolError("Opened labels contain an unknown outer target.")
        if self.phase not in {"development", "target"}:
            raise ProtocolError("Opened label phase is invalid.")
        if not self.rows or len(self.rows) != len(self.labels):
            raise ProtocolError("Opened label rows and values do not align.")
        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or label not in (0, 1)
            for label in self.labels
        ):
            raise ProtocolError("Opened labels are not binary.")
        expected = stable_hash(
            {
                "outer_target": self.outer_target,
                "phase": self.phase,
                "row_identity_hash": row_identity_hash(self.rows),
                "labels": list(self.labels),
                "manifest_sha256": self.manifest_sha256,
                "prediction_seal_hash": self.prediction_seal_hash,
            }
        )
        if self.label_vector_hash != expected:
            raise ProtocolError("Opened label-vector hash drifted.")


def classifier_payload() -> Mapping[str, object]:
    return CLASSIFIER.to_payload()


__all__ = (
    "ACTION_IDS",
    "ACTION_LIBRARY_HASH",
    "CLASSIFIER",
    "CLASS_PRIOR",
    "CLAIM_SCOPE",
    "COMPATIBILITY_SEMANTICS",
    "CONTROL_ACTION_ID",
    "CENTERS",
    "DEVELOPMENT_TOTAL_PER_CLASS",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_TARGET_PREDICTION_CELL_COUNT",
    "EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT",
    "EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "FALLBACK_ACTION_ID",
    "FORBIDDEN_STAGE60_INPUT_ARTIFACT_IDS",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "INPUT_ARTIFACT_IDS",
    "MAX_SOURCE_WEIGHT",
    "MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES",
    "MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS",
    "MIN_EFFECTIVE_SOURCE_COUNT",
    "MINIMUM_INTEGER_ALLOCATION_PER_SOURCE",
    "NONUNIFORM_PASS_RULE",
    "ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID",
    "ORIGINAL_STAGE60_VALIDATION_MANIFEST_ARTIFACT_ID",
    "OUTPUT_ARTIFACT_ID",
    "PRIMARY_METRIC",
    "PUBLICATION_STATUS",
    "RHO_VALUES",
    "SECONDARY_METRIC",
    "SELECTION_OBJECTIVE",
    "STAGE_ID",
    "SUPPORT_CASE_COUNT",
    "SUPPORT_PARTITION_NAMESPACE",
    "SUPPORT_SPLIT_SEED",
    "TEMPERATURE",
    "TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
    "VALIDATION_CACHE_ARTIFACT_ID",
    "VALIDATION_CACHE_REPRESENTATION_ID",
    "VALIDATION_CACHE_SEMANTIC_ID",
    "VALIDATION_MANIFEST_ARTIFACT_ID",
    "VALIDATION_SPLIT",
    "ActionSpec",
    "OpenedLabelVector",
    "ValidationRowIdentity",
    "action_library",
    "classifier_payload",
    "development_queries",
    "legal_sources",
    "row_identity_hash",
    "seed_cells",
    "target_sources",
)
