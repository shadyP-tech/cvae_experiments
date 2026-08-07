"""Frozen identities for the consumed-validation MMD/KMM diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import (
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
    "midogpp.oracle.uniform_b_v2_consumed_validation_mmd_kmm_router.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_consumed_validation_mmd_kmm_router_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_mmd_kmm_router_v1"
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
    "midogpp_stage90_mmd_kmm_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_mmd_kmm_router_validation_manifest_v1"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)

EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH = "4b9ea514308b084f"
EXCLUDED_CENTER = "4"
VALIDATION_SPLIT = "val"
SUPPORT_CASE_COUNT = 2
SUPPORT_SPLIT_SEED = 20260806
SUPPORT_PARTITION_NAMESPACE = "midogpp_mmd_kmm_support_v1"
COMMON_FEATURE_DIM = 3840
COMMON_FRAME_HASH = stable_hash(
    {
        "semantics": "common_inverse_virchow2",
        "representation_id": VALIDATION_CACHE_REPRESENTATION_ID,
        "feature_dim": COMMON_FEATURE_DIM,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
    }
)

# The source cap proves that no downstream arm can consume more than 256 rows
# from one source/class stream.  Keeping exactly that prefix cuts the resumable
# cache from about 2.37 GiB to about 608 MiB without changing any allocation.
MAX_SOURCE_WEIGHT = 0.25
MIN_EFFECTIVE_SOURCES = 6.0
MAX_SOURCE_PREFIX_PER_CLASS = 256
ROUTER_PREFIX_PER_CLASS = 32
NYSTROEM_COMPONENTS = 256
NYSTROEM_GAMMA = 1.0 / float(COMMON_FEATURE_DIM)
NYSTROEM_RANDOM_STATE = 20260807
PRIOR_PROBABILITY_CLIP = 1.0e-3
PRIOR_TEMPERATURE = 1.0
PRIOR_SENSITIVITY_POSITIVE_PRIORS = (0.35, 0.65)
KMM_REGULARIZATION = 0.05
KMM_MINIMUM_PROXY_IMPROVEMENT = 1.0e-6
KMM_SOLVER_TOLERANCE = 1.0e-12
KMM_OPTIMALITY_TOLERANCE = 1.0e-6
KMM_MAX_ITERATIONS = 2000
MAXIMUM_SUPPORT_L1 = 0.50
MAXIMUM_TRAINING_SEED_L1 = 0.35
MAXIMUM_GENERATION_SEED_L1 = 0.35
MAXIMUM_PRIOR_SENSITIVITY_L1 = 0.35
MINIMUM_DIRECTION_COSINE = 0.0
DUPLICATE_DIRECTION_COSINE = 0.995
DUPLICATE_WEIGHT_L1 = 0.02
ENERGY_REFERENCE_RHO = 0.50
ENERGY_REFERENCE_TEMPERATURE = 1.0

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
PRIOR_CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=29,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GENERATION_DEVICES = ("cuda:0", "cuda:1")
KERNEL_DEVICES = ("cuda:0", "cuda:1")
CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3
KERNEL_BATCH_ROWS = 1024
EXPECTED_SOURCE_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_SOURCE_BLOCK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
EXPECTED_TARGET_COUNT = len(CENTERS)
EXPECTED_SEED_CELL_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_PREDICTION_CELL_COUNT = EXPECTED_TARGET_COUNT * EXPECTED_SEED_CELL_COUNT * 2
MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT = EXPECTED_PREDICTION_CELL_COUNT


def candidate_sources(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("MMD/KMM target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def seed_cells() -> tuple[tuple[int, int], ...]:
    return tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))


@dataclass(frozen=True)
class ValidationRowIdentity:
    """Label-free identity used to bind support and evaluation rows."""

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
            raise ProtocolError("MMD/KMM validation-row identity is invalid.")

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


__all__ = tuple(
    name
    for name in globals()
    if name.isupper()
) + (
    "ValidationRowIdentity",
    "candidate_sources",
    "row_identity_hash",
    "seed_cells",
)
