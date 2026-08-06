"""Closed contracts for the consumed-validation local-utility diagnostic.

The experiment measures paired finite differences around equal union.  It is
deliberately incapable of claiming target performance: target-center labels
are never opened for the plan whose outer target is that center.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping, Sequence

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
    TOTAL_PER_CLASS,
)
from ...protocol import ProtocolError
from ...routing.local_marginal_utility.perturbations import (
    BOOST_ACTION_PREFIX as CORE_BOOST_ACTION_PREFIX,
    CONTROL_ACTION_ID as CORE_CONTROL_ACTION_ID,
    LOCAL_PERTURBATION_EPSILON,
    boost_action_id as core_boost_action_id,
    build_perturbation_library,
)
from ...routing.local_marginal_utility.optimizer import (
    DEFAULT_KAPPA,
    DEFAULT_L2_PENALTY,
)
from ...routing.local_marginal_utility.ridge import DEFAULT_RIDGE_ALPHAS


EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_validation_local_marginal_utility_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_validation_local_marginal_utility_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_"
    "local_marginal_utility_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_local_marginal_utility_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_local_marginal_utility_router_validation_manifest_v1"
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
SUPPORT_PARTITION_NAMESPACE = "midogpp_local_marginal_utility_support_v1"

COMPATIBILITY_SEMANTICS = "aggregate_prior_variational_compatibility_energy_v1"
CLASS_PRIOR = (0.5, 0.5)
EPSILON = float(LOCAL_PERTURBATION_EPSILON)
DEVELOPMENT_TOTAL_PER_CLASS = 1008
CONTROL_ACTION_ID = CORE_CONTROL_ACTION_ID
BOOST_ACTION_PREFIX = CORE_BOOST_ACTION_PREFIX
CONTROL_ARM_ROLE = "control"
BOOST_ARM_ROLE = "source_perturbation"
MAX_SOURCE_WEIGHT = 0.25
MIN_EFFECTIVE_SOURCE_COUNT = 6.0
MINIMUM_INTEGER_ALLOCATION_PER_SOURCE = 1
PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "macro_f1_descriptive_only"
MARGINAL_RESPONSE = "paired_bacc_delta_divided_by_epsilon"

MODEL_FAMILY = "cluster_weighted_ridge_local_marginal_utility_v1"
MODEL_ALPHA_GRID = tuple(float(value) for value in DEFAULT_RIDGE_ALPHAS)
OPTIMIZER_FAMILY = "robust_local_utility_weights_v1"
OPTIMIZER_KAPPA = DEFAULT_KAPPA
OPTIMIZER_L2_PENALTY = DEFAULT_L2_PENALTY

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


def development_queries(outer_target: str) -> tuple[str, ...]:
    outer = str(outer_target)
    if outer not in CENTERS:
        raise ProtocolError("Local-utility outer target is unknown.")
    return tuple(center for center in CENTERS if center != outer)


def legal_sources(*, outer_target: str, query_center: str) -> tuple[str, ...]:
    outer = str(outer_target)
    query = str(query_center)
    if outer not in CENTERS or query not in CENTERS:
        raise ProtocolError("Local-utility geometry contains an unknown center.")
    if outer == query:
        raise ProtocolError("Local-utility query q must differ from outer target H.")
    return tuple(center for center in CENTERS if center not in {outer, query})


def target_sources(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Local-utility target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def seed_cells() -> tuple[tuple[int, int], ...]:
    return tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))


def boost_action_id(source: str) -> str:
    rendered = str(source)
    if rendered not in CENTERS:
        raise ProtocolError("Local-utility boost source is unknown.")
    return core_boost_action_id(rendered)


def action_ids(*, outer_target: str, query_center: str) -> tuple[str, ...]:
    return (CONTROL_ACTION_ID,) + tuple(
        boost_action_id(source)
        for source in legal_sources(
            outer_target=outer_target,
            query_center=query_center,
        )
    )


@dataclass(frozen=True)
class PerturbationSpec:
    """One exact equal-union or one-source finite-difference action."""

    outer_target: str
    query_center: str
    action_id: str
    arm_role: str
    boosted_source: str | None
    candidate_sources: tuple[str, ...]
    weights: Mapping[str, float]
    allocations: Mapping[str, int]

    def __post_init__(self) -> None:
        expected_sources = legal_sources(
            outer_target=self.outer_target,
            query_center=self.query_center,
        )
        if self.candidate_sources != expected_sources:
            raise ProtocolError("Local-utility perturbation candidate pool drifted.")
        weights = {str(key): float(value) for key, value in self.weights.items()}
        allocations = {str(key): int(value) for key, value in self.allocations.items()}
        if tuple(weights) != expected_sources or tuple(allocations) != expected_sources:
            raise ProtocolError("Local-utility perturbation source order drifted.")
        core_by_id = {
            plan.action_id: plan
            for plan in build_perturbation_library(
                expected_sources,
                total_per_class=DEVELOPMENT_TOTAL_PER_CLASS,
                epsilon=LOCAL_PERTURBATION_EPSILON,
            )
        }
        expected = core_by_id.get(self.action_id)
        expected_role = CONTROL_ARM_ROLE if expected and expected.is_control else BOOST_ARM_ROLE
        if (
            expected is None
            or self.arm_role != expected_role
            or self.boosted_source != expected.boosted_source
        ):
            raise ProtocolError("Local-utility perturbation action identity drifted.")
        if any(weights[source] != expected.weights[source] for source in expected_sources):
            raise ProtocolError("Local-utility perturbation weights drifted.")
        if allocations != expected.allocations_per_class:
            raise ProtocolError("Local-utility perturbation allocations drifted.")
        if sum(allocations.values()) != DEVELOPMENT_TOTAL_PER_CLASS:
            raise ProtocolError("Local-utility perturbation allocation total drifted.")
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "allocations", MappingProxyType(allocations))

    @property
    def effective_source_count(self) -> float:
        return 1.0 / sum(value * value for value in self.weights.values())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_local_marginal_perturbation_v1",
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "arm_role": self.arm_role,
            "boosted_source": self.boosted_source,
            "candidate_sources": list(self.candidate_sources),
            "epsilon": EPSILON,
            "weights": dict(self.weights),
            "allocations": dict(self.allocations),
            "total_generated_samples_per_class": DEVELOPMENT_TOTAL_PER_CLASS,
            "effective_source_count": self.effective_source_count,
            "maximum_source_weight": max(self.weights.values()),
        }


def perturbation_library_for(
    *, outer_target: str, query_center: str
) -> tuple[PerturbationSpec, ...]:
    sources = legal_sources(outer_target=outer_target, query_center=query_center)
    core_plans = build_perturbation_library(
        sources,
        total_per_class=DEVELOPMENT_TOTAL_PER_CLASS,
        epsilon=LOCAL_PERTURBATION_EPSILON,
    )
    return tuple(
        PerturbationSpec(
            outer_target=str(outer_target),
            query_center=str(query_center),
            action_id=plan.action_id,
            arm_role=(CONTROL_ARM_ROLE if plan.is_control else BOOST_ARM_ROLE),
            boosted_source=plan.boosted_source,
            candidate_sources=sources,
            weights=plan.weights,
            allocations=plan.allocations_per_class,
        )
        for plan in core_plans
    )


def perturbation_library_payloads() -> tuple[dict[str, object], ...]:
    return tuple(
        action.to_payload()
        for outer in CENTERS
        for query in development_queries(outer)
        for action in perturbation_library_for(
            outer_target=outer,
            query_center=query,
        )
    )


PERTURBATION_LIBRARY_HASH = stable_hash(perturbation_library_payloads())

EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT = (
    len(CENTERS)
    * (len(CENTERS) - 1)
    * (1 + len(CENTERS) - 2)
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
)
EXPECTED_MARGINAL_UTILITY_ROW_COUNT = (
    len(CENTERS)
    * (len(CENTERS) - 1)
    * (len(CENTERS) - 2)
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
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
class ValidationRowIdentity:
    """Label-free identity of one consumed validation row."""

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
            raise ProtocolError("Local-utility row indices must be nonnegative integers.")
        if not self.sample_id or not self.case_id:
            raise ProtocolError("Local-utility rows require sample and case identities.")
        if self.center not in CENTERS or self.split != VALIDATION_SPLIT:
            raise ProtocolError("Local-utility rows must be eligible MIDOG++ val rows.")
        if self.partition_role not in {"support", "evaluation"}:
            raise ProtocolError("Local-utility row partition role is invalid.")

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
    """One query-center label vector opened through the global seal."""

    query_center: str
    rows: tuple[ValidationRowIdentity, ...]
    labels: tuple[int, ...]
    manifest_sha256: str
    prediction_seal_hash: str
    label_vector_hash: str
    phase: str = "development_utility_surface"

    def __post_init__(self) -> None:
        if self.query_center not in CENTERS or self.phase != "development_utility_surface":
            raise ProtocolError("Local-utility opened-label identity drifted.")
        if not self.rows or len(self.rows) != len(self.labels):
            raise ProtocolError("Local-utility opened-label rows do not align.")
        if any(row.center != self.query_center for row in self.rows):
            raise ProtocolError("Local-utility label vector crosses query centers.")
        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or label not in (0, 1)
            for label in self.labels
        ):
            raise ProtocolError("Local-utility opened labels are not binary.")
        expected = stable_hash(
            {
                "query_center": self.query_center,
                "phase": self.phase,
                "row_identity_hash": row_identity_hash(self.rows),
                "labels": list(self.labels),
                "manifest_sha256": self.manifest_sha256,
                "prediction_seal_hash": self.prediction_seal_hash,
            }
        )
        if self.label_vector_hash != expected:
            raise ProtocolError("Local-utility opened label-vector hash drifted.")


def classifier_payload() -> Mapping[str, object]:
    return CLASSIFIER.to_payload()


__all__ = (
    "BOOST_ACTION_PREFIX",
    "BOOST_ARM_ROLE",
    "CENTERS",
    "CLAIM_SCOPE",
    "CLASSIFIER",
    "CLASS_PRIOR",
    "COMPATIBILITY_SEMANTICS",
    "CONTROL_ACTION_ID",
    "CONTROL_ARM_ROLE",
    "DEVELOPMENT_TOTAL_PER_CLASS",
    "EPSILON",
    "EXPECTED_BANK_LOCK_HASH",
    "EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT",
    "EXPECTED_GENERATION_LOCK_HASH",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MARGINAL_UTILITY_ROW_COUNT",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERT_BANK_ARTIFACT_ID",
    "FORBIDDEN_STAGE60_INPUT_ARTIFACT_IDS",
    "GENERATION_LOCK_ARTIFACT_ID",
    "GENERATION_SEEDS",
    "INPUT_ARTIFACT_IDS",
    "MARGINAL_RESPONSE",
    "MAX_SOURCE_WEIGHT",
    "MAXIMUM_RESIDENT_GENERATED_EMBEDDING_BYTES",
    "MAXIMUM_RESIDENT_GENERATED_SOURCE_BLOCKS",
    "MIN_EFFECTIVE_SOURCE_COUNT",
    "MINIMUM_INTEGER_ALLOCATION_PER_SOURCE",
    "MODEL_ALPHA_GRID",
    "MODEL_FAMILY",
    "OPTIMIZER_FAMILY",
    "OPTIMIZER_KAPPA",
    "OPTIMIZER_L2_PENALTY",
    "OUTPUT_ARTIFACT_ID",
    "PERTURBATION_LIBRARY_HASH",
    "PRIMARY_METRIC",
    "PUBLICATION_STATUS",
    "SECONDARY_METRIC",
    "STAGE_ID",
    "SUPPORT_CASE_COUNT",
    "SUPPORT_PARTITION_NAMESPACE",
    "SUPPORT_SPLIT_SEED",
    "TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
    "VALIDATION_CACHE_ARTIFACT_ID",
    "VALIDATION_CACHE_REPRESENTATION_ID",
    "VALIDATION_CACHE_SEMANTIC_ID",
    "VALIDATION_MANIFEST_ARTIFACT_ID",
    "VALIDATION_SPLIT",
    "OpenedLabelVector",
    "PerturbationSpec",
    "ValidationRowIdentity",
    "action_ids",
    "boost_action_id",
    "classifier_payload",
    "development_queries",
    "legal_sources",
    "perturbation_library_for",
    "perturbation_library_payloads",
    "row_identity_hash",
    "seed_cells",
    "target_sources",
)
