"""Frozen scientific and execution identities for residual top-up routing.

The experiment in this package is deliberately a terminal Stage-90 diagnostic.
It tests whether preserving an immutable equal-union backbone while routing a
small disjoint suffix is safer than replacing the whole composition.  The
consumed validation surface cannot authorize a Stage-60/70 policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

from ....common.hashing import stable_hash

from ....data.features.uniform_b_routing_validation.config import (
    CACHE_NAME as VALIDATION_CACHE_SEMANTIC_ID,
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
    REPRESENTATION_ID as VALIDATION_CACHE_REPRESENTATION_ID,
)
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
)
from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_router.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_consumed_validation_residual_topup_router_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_residual_topup_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_residual_topup_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_residual_topup_router_validation_manifest_v1"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)
EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH = "4b9ea514308b084f"

FORBIDDEN_ROUTER_INPUT_ARTIFACT_IDS = frozenset(
    {
        "midogpp_output_uniform_b_v2_consumed_validation_dense_residual_router_v1",
        "midogpp_output_uniform_b_v2_consumed_validation_local_marginal_utility_router_v1",
        "midogpp_output_uniform_b_v2_consumed_validation_mmd_kmm_router_v1",
        "midogpp_output_uniform_b_v2_consumed_validation_conditional_contrast_mmd_router_v1",
        "midogpp_output_uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router_v1",
        "midogpp_output_uniform_b_v2_source_inner_candidate_utility_v1",
    }
)

EXCLUDED_CENTER = "4"
VALIDATION_SPLIT = "val"
SUPPORT_CASE_COUNT = 2
SUPPORT_SPLIT_SEED = 20260806
SUPPORT_PARTITION_NAMESPACE = "midogpp_residual_topup_support_v1"

BASE_ONLY_ACTION_ID = "base_only"
UNIFORM_TOPUP_ACTION_ID = "uniform_topup"
ENERGY_TOPUP_ACTION_ID = "energy_directed_topup"
TARGET_ACTION_IDS = (
    BASE_ONLY_ACTION_ID,
    UNIFORM_TOPUP_ACTION_ID,
    ENERGY_TOPUP_ACTION_ID,
)
DEVELOPMENT_ACTION_IDS = (UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID)
PRIMARY_CONTROL_ACTION_ID = UNIFORM_TOPUP_ACTION_ID
PRIMARY_ROUTED_ACTION_ID = ENERGY_TOPUP_ACTION_ID

TARGET_SOURCE_COUNT = 8
TARGET_BASE_PER_SOURCE = 128
TARGET_BASE_TOTAL_PER_CLASS = 1024
TARGET_TOPUP_TOTAL_PER_CLASS = 128
TARGET_MATCHED_TOTAL_PER_CLASS = 1152
DEVELOPMENT_SOURCE_COUNT = 7
DEVELOPMENT_BASE_PER_SOURCE = 144
DEVELOPMENT_BASE_TOTAL_PER_CLASS = 1008
DEVELOPMENT_TOPUP_TOTAL_PER_CLASS = 126
DEVELOPMENT_MATCHED_TOTAL_PER_CLASS = 1134
TOPUP_FRACTION_OF_BASE = 0.125
MAX_SOURCE_PREFIX_PER_CLASS = 256
ENERGY_RANK_SEMANTICS = (
    "lower_energy_first_linear_rank_priority_k_minus_rank_plus_one_"
    "canonical_source_ties"
)
PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "macro_f1_descriptive_only"
SELECTION_CONFIDENCE_LEVEL = 0.95
SELECTION_THRESHOLD = 0.0
SELECTION_CLUSTER_UNIT = "inner_query_center"
SELECTION_RULE = (
    "one_sided_student_t_lower_confidence_bound_of_query_center_mean_"
    "paired_bacc_gain_strictly_above_zero_else_exact_uniform_topup"
)

COMMON_FEATURE_DIM = COMMON_OUTPUT_DIM
MAX_SOURCE_WEIGHT = 0.25
MIN_EFFECTIVE_SOURCES = 6.0

DOWNSTREAM_CLASSIFIER = ClassifierSpec(
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

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GENERATION_DEVICES = ("cuda:0", "cuda:1")
CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3
MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT = 12
MINIMUM_WORKSTATION_RAM_BYTES = 107374182400
MINIMUM_WORKSTATION_DISK_FREE_BYTES = 8589934592
MINIMUM_WORKSTATION_GPU_FREE_MIB = 18000

EXPECTED_SOURCE_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_SOURCE_BLOCK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
EXPECTED_SEED_CELL_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_DEVELOPMENT_TASK_COUNT = (
    len(CENTERS) * (len(CENTERS) - 1) * EXPECTED_SEED_CELL_COUNT
)
EXPECTED_DEVELOPMENT_PREDICTION_CELL_COUNT = (
    EXPECTED_DEVELOPMENT_TASK_COUNT * len(DEVELOPMENT_ACTION_IDS)
)
EXPECTED_TARGET_TASK_COUNT = len(CENTERS) * EXPECTED_SEED_CELL_COUNT
EXPECTED_TARGET_PREDICTION_CELL_COUNT = (
    EXPECTED_TARGET_TASK_COUNT * len(TARGET_ACTION_IDS)
)
EXPECTED_PREDICTION_CELL_COUNT = (
    EXPECTED_DEVELOPMENT_PREDICTION_CELL_COUNT
    + EXPECTED_TARGET_PREDICTION_CELL_COUNT
)
MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT = EXPECTED_PREDICTION_CELL_COUNT


def target_sources(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Residual top-up target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def development_queries(outer_target: str) -> tuple[str, ...]:
    outer = str(outer_target)
    if outer not in CENTERS:
        raise ProtocolError("Residual top-up outer target is unknown.")
    return tuple(center for center in CENTERS if center != outer)


def legal_development_sources(
    *, outer_target: str, query_center: str
) -> tuple[str, ...]:
    outer = str(outer_target)
    query = str(query_center)
    if outer not in CENTERS or query not in CENTERS or outer == query:
        raise ProtocolError("Residual top-up inner fold geometry is invalid.")
    return tuple(center for center in CENTERS if center not in {outer, query})


def seed_cells() -> tuple[tuple[int, int], ...]:
    return tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))


@dataclass(frozen=True)
class ValidationRowIdentity:
    """Label-free row identity binding support, predictions, and labels."""

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
            or not self.sample_id
            or not self.case_id
            or self.center not in CENTERS
            or self.split != VALIDATION_SPLIT
            or self.partition_role not in {"support", "evaluation"}
        ):
            raise ProtocolError("Residual top-up validation-row identity is invalid.")

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


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "development_queries",
    "legal_development_sources",
    "row_identity_hash",
    "seed_cells",
    "target_sources",
    "ValidationRowIdentity",
)
