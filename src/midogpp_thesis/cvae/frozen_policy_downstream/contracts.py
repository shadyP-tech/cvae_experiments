"""Contracts for the descriptive frozen-policy Stage-70 comparison.

The contracts deliberately distinguish policy identity from materialized
training content.  Two frozen policies may produce identical bytes (the
current utility/regret fallback and equal-union control), but they remain
separate reported arms.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..generation.contracts import COMMON_OUTPUT_DIM, TOTAL_PER_CLASS
from ..protocol import ProtocolError


CLAIM_SCOPE = "descriptive_frozen_policy_comparison_on_previously_consumed_test"
AUTHORIZATION_CLAIM_SCOPE = "target_evaluation_authorization"
EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_frozen_policy_comparison.v1"
)
EXPERIMENT_NAME = "uniform_b_v2_descriptive_frozen_policy_comparison_v1"
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_frozen_policy_comparison_v1"
)
AUTHORIZED_CONSUMER_EXPERIMENT_ID = EXPERIMENT_ID

CONTROL_ARM = "equal_union_control"
METADATA_ARM = "metadata_max_tie_union"
UTILITY_ARM = "utility_regret_frozen_policy"
POLICY_ARMS = (CONTROL_ARM, METADATA_ARM, UTILITY_ARM)
FEATURE_DIM = COMMON_OUTPUT_DIM
SYNTHETIC_PER_CLASS = TOTAL_PER_CLASS
SYNTHETIC_ROW_COUNT = 2 * SYNTHETIC_PER_CLASS
REPLICATES_PER_ARM = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_METRIC_ROWS = len(POLICY_ARMS) * REPLICATES_PER_ARM
PRIMARY_METRIC = "bacc"
SECONDARY_METRIC = "macro_f1_descriptive_only"
PUBLICATION_DECISIONS = (
    "DESCRIPTIVE_COMPARISON_COMPLETE",
    "DESCRIPTIVE_COMPARISON_FAILED_VALIDATION",
)


def array_sha256(array: np.ndarray) -> str:
    """Hash an ndarray with dtype and shape bound into the digest."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def array_bundle_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MaterializationAssignment:
    """One frozen source prefix contributing to a policy replicate."""

    assignment_id: str
    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    source_center: str
    source_stream_id: str
    source_ordinal: int
    source_budget_per_class: int
    prior_method: str
    selection_source: str
    exact_equal_union_fallback: bool = False
    equal_union_assignment_id: str = ""

    def __post_init__(self) -> None:
        if self.policy_id not in POLICY_ARMS:
            raise ProtocolError(f"Unknown Stage-70 policy arm: {self.policy_id!r}.")
        if self.target_center not in CENTERS or self.source_center not in CENTERS:
            raise ProtocolError("Stage-70 assignment contains an ineligible center.")
        if self.source_center == self.target_center:
            raise ProtocolError("Stage-70 assignment includes the target expert.")
        if self.training_seed not in TRAINING_SEEDS:
            raise ProtocolError("Stage-70 assignment training seed drifted.")
        if self.generation_seed not in GENERATION_SEEDS:
            raise ProtocolError("Stage-70 assignment generation seed drifted.")
        if self.source_budget_per_class <= 0:
            raise ProtocolError("Stage-70 assignment budget must be positive.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_materialization_assignment_v1",
            "assignment_id": self.assignment_id,
            "policy_id": self.policy_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "source_center": self.source_center,
            "source_stream_id": self.source_stream_id,
            "source_ordinal": self.source_ordinal,
            "source_budget_per_class": self.source_budget_per_class,
            "prior_method": self.prior_method,
            "selection_source": self.selection_source,
            "exact_equal_union_fallback": self.exact_equal_union_fallback,
            "equal_union_assignment_id": self.equal_union_assignment_id,
            "target_expert": False,
        }


@dataclass(frozen=True)
class PolicyReplicate:
    """One arm/target/training-seed/generation-seed materialization."""

    policy_id: str
    policy_lock_hash: str
    policy_plan_hash: str
    assignment_table_hash: str
    replicate_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    assignments: tuple[MaterializationAssignment, ...]
    class_shuffle_seed_by_label: Mapping[str, int]
    claim_role: str = CLAIM_SCOPE
    row_role: str = "target_evaluation_metric"

    def __post_init__(self) -> None:
        if self.policy_id not in POLICY_ARMS:
            raise ProtocolError("Stage-70 replicate policy identity drifted.")
        if not self.assignments:
            raise ProtocolError("Stage-70 replicate has no source assignments.")
        if any(
            assignment.policy_id != self.policy_id
            or assignment.target_center != self.target_center
            or assignment.training_seed != self.training_seed
            or assignment.generation_seed != self.generation_seed
            for assignment in self.assignments
        ):
            raise ProtocolError("Stage-70 assignment/replicate identity drifted.")
        if sum(row.source_budget_per_class for row in self.assignments) != TOTAL_PER_CLASS:
            raise ProtocolError("Stage-70 replicate does not total 1,024 rows per class.")
        if set(self.class_shuffle_seed_by_label) != {"0", "1"}:
            raise ProtocolError("Stage-70 replicate lacks class-specific shuffle seeds.")
        ordinals = tuple(row.source_ordinal for row in self.assignments)
        if ordinals != tuple(range(len(self.assignments))):
            raise ProtocolError("Stage-70 source order is not canonical and contiguous.")

    @property
    def cell_key(self) -> tuple[str, int, int]:
        return self.target_center, self.training_seed, self.generation_seed


@dataclass(frozen=True)
class SyntheticComposition:
    replicate: PolicyReplicate
    embeddings: np.ndarray
    labels: np.ndarray
    pre_shuffle_sha256_by_label: Mapping[str, str]
    post_shuffle_sha256_by_label: Mapping[str, str]
    train_content_sha256: str
    composition_manifest_hash: str

    def __post_init__(self) -> None:
        if self.embeddings.shape != (SYNTHETIC_ROW_COUNT, FEATURE_DIM):
            raise ProtocolError("Stage-70 synthetic composition geometry drifted.")
        if self.labels.shape != (SYNTHETIC_ROW_COUNT,):
            raise ProtocolError("Stage-70 synthetic label geometry drifted.")
        if not np.isfinite(self.embeddings).all():
            raise ProtocolError("Stage-70 composition contains non-finite embeddings.")
        counts = {label: int(np.sum(self.labels == label)) for label in (0, 1)}
        if counts != {0: SYNTHETIC_PER_CLASS, 1: SYNTHETIC_PER_CLASS}:
            raise ProtocolError("Stage-70 composition is not class balanced.")


@dataclass(frozen=True)
class TargetFrame:
    """Label-sealed target frame used by prediction only."""

    target_center: str
    evaluation_row_ids: tuple[str, ...]
    contract_row_indices: tuple[int, ...]
    case_ids: tuple[str, ...]
    embeddings: np.ndarray
    row_order_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        n_rows = len(self.evaluation_row_ids)
        if self.target_center not in CENTERS or n_rows == 0:
            raise ProtocolError("Stage-70 target frame center/coverage drifted.")
        if (
            len(self.contract_row_indices) != n_rows
            or len(self.case_ids) != n_rows
            or self.embeddings.shape != (n_rows, FEATURE_DIM)
        ):
            raise ProtocolError("Stage-70 target frame rows do not align.")
        if not np.isfinite(self.embeddings).all():
            raise ProtocolError("Stage-70 target frame contains non-finite embeddings.")
        serialized = "|".join(self.evaluation_row_ids)
        if "__y0" in serialized or "__y1" in serialized:
            raise ProtocolError("Stage-70 target frame exposes a legacy label encoding.")


@dataclass(frozen=True)
class PredictionCell:
    policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    replicate_id: str
    evaluation_row_ids: tuple[str, ...]
    contract_row_indices: tuple[int, ...]
    case_ids: tuple[str, ...]
    predictions: np.ndarray
    probabilities: np.ndarray
    composition_manifest_hash: str
    train_content_sha256: str
    classifier_config_hash: str
    scaler_state_hash: str
    target_row_order_hash: str
    prediction_sha256: str
    probability_sha256: str
    reused_from_policy_id: str = ""

    def __post_init__(self) -> None:
        n_rows = len(self.evaluation_row_ids)
        if self.predictions.shape != (n_rows,):
            raise ProtocolError("Stage-70 prediction row count drifted.")
        if self.probabilities.shape[0] != n_rows:
            raise ProtocolError("Stage-70 probability row count drifted.")
        if len(self.contract_row_indices) != n_rows or len(self.case_ids) != n_rows:
            raise ProtocolError("Stage-70 prediction identity rows drifted.")
        if set(int(value) for value in np.unique(self.predictions)) - {0, 1}:
            raise ProtocolError("Stage-70 predictions are not binary.")
        if not np.isfinite(self.probabilities).all():
            raise ProtocolError("Stage-70 probabilities contain non-finite values.")
        if array_sha256(self.predictions) != self.prediction_sha256:
            raise ProtocolError("Stage-70 prediction hash drifted.")
        if array_sha256(self.probabilities) != self.probability_sha256:
            raise ProtocolError("Stage-70 probability hash drifted.")


@dataclass(frozen=True)
class ScoringLabels:
    evaluation_row_ids: tuple[str, ...]
    labels: np.ndarray
    label_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.labels.shape != (len(self.evaluation_row_ids),):
            raise ProtocolError("Stage-70 scoring labels do not align.")
        if set(int(value) for value in np.unique(self.labels)) != {0, 1}:
            raise ProtocolError("Stage-70 scoring labels must contain both classes.")


__all__ = (
    "AUTHORIZED_CONSUMER_EXPERIMENT_ID",
    "AUTHORIZATION_CLAIM_SCOPE",
    "CLAIM_SCOPE",
    "CONTROL_ARM",
    "EXPECTED_METRIC_ROWS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "FEATURE_DIM",
    "METADATA_ARM",
    "MaterializationAssignment",
    "OUTPUT_ARTIFACT_ID",
    "POLICY_ARMS",
    "PRIMARY_METRIC",
    "PUBLICATION_DECISIONS",
    "PolicyReplicate",
    "PredictionCell",
    "REPLICATES_PER_ARM",
    "SECONDARY_METRIC",
    "SYNTHETIC_PER_CLASS",
    "SYNTHETIC_ROW_COUNT",
    "ScoringLabels",
    "SyntheticComposition",
    "TargetFrame",
    "UTILITY_ARM",
    "array_bundle_sha256",
    "array_sha256",
)
