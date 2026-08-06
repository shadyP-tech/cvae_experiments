"""Fail-closed contracts for the two Stage-70 authorization gates.

The authorization artifacts deliberately do not perform generation, classifier
fitting, prediction, label loading, or metric scoring.  They bind a previously
consumed test split to already-frozen Stage-30/40/60 state and make the allowed
descriptive use explicit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import TOTAL_PER_CLASS
from ...protocol import ProtocolError
from ..contracts import (
    AUTHORIZATION_CLAIM_SCOPE,
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    POLICY_ARMS,
)


RESERVATION_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_test_reservation.v1"
)
RESERVATION_EXPERIMENT_NAME = "uniform_b_v2_descriptive_test_reservation_v1"
RESERVATION_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_test_reservation_v1"
)
FINAL_AUTHORIZATION_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_test_final_authorization.v1"
)
FINAL_AUTHORIZATION_EXPERIMENT_NAME = (
    "uniform_b_v2_descriptive_test_final_authorization_v1"
)
FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_test_final_authorization_v1"
)

CLAIM_SCOPE = AUTHORIZATION_CLAIM_SCOPE
PURPOSE = "descriptive_frozen_policy_comparison_on_previously_consumed_test"
FRESH_CONFIRMATORY_STATUS = "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT"
RESERVATION_DESCRIPTIVE_STATUS = (
    "RESERVED_FOR_FROZEN_LOCKED_POLICY_CACHE_EXTRACTION_ONLY"
)
FINAL_DESCRIPTIVE_STATUS = (
    "AUTHORIZED_FOR_FROZEN_LOCKED_POLICY_PREDICTION_ONLY"
)
RESERVATION_PHASE = "TARGET_EVALUATION_RESERVATION"
FINAL_AUTHORIZATION_PHASE = "FINAL_PREDICTION_AUTHORIZATION"
RUN_COMPLETE = "COMPLETE"
VALIDATION_PASS = "PASS"
TEST_CONSUMPTION_STATUS = "CONSUMED_FOR_REPRESENTATION_ADOPTION"
EXPECTED_SPLIT = "test"
EXPECTED_TEST_ROWS = 9_928
EXPECTED_CENTER_COUNT = 9
EXPECTED_REPLICATES_PER_POLICY = 81
EXPECTED_EVALUATION_PLAN_ROWS = 243
EXPECTED_SYNTHETIC_ROWS_PER_CLASS = TOTAL_PER_CLASS


@dataclass(frozen=True)
class ArtifactBinding:
    """One validated upstream identity without embedding upstream content."""

    artifact_id: str
    content_index_sha256: str
    semantic_hashes: Mapping[str, str]
    validator: str
    validation_status: str = VALIDATION_PASS

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.validator:
            raise ProtocolError("Stage-70 artifact binding identity is empty.")
        if self.validation_status != VALIDATION_PASS:
            raise ProtocolError("Stage-70 upstream validation did not pass.")
        _require_sha256(self.content_index_sha256, "content-index SHA-256")
        hashes = {str(key): str(value) for key, value in self.semantic_hashes.items()}
        if not hashes or any(not _is_lower_hex(value) for value in hashes.values()):
            raise ProtocolError("Stage-70 upstream semantic hashes are malformed.")
        object.__setattr__(self, "semantic_hashes", MappingProxyType(hashes))

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "content_index_sha256": self.content_index_sha256,
            "semantic_hashes": dict(self.semantic_hashes),
            "validator": self.validator,
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class PolicyBinding:
    """The exact policy-lock and assignment identities consumed by Stage 70."""

    policy_id: str
    policy_artifact_id: str
    policy_lock_hash: str
    policy_plan_hash: str
    assignment_table_hash: str
    assignment_table_sha256: str
    assignment_count: int
    replicate_count: int = EXPECTED_REPLICATES_PER_POLICY
    target_expert_excluded: bool = True
    total_per_class: int = EXPECTED_SYNTHETIC_ROWS_PER_CLASS

    def __post_init__(self) -> None:
        if self.policy_id not in POLICY_ARMS:
            raise ProtocolError(f"Unknown Stage-70 policy binding: {self.policy_id!r}.")
        if not self.policy_artifact_id:
            raise ProtocolError("Stage-70 policy artifact identity is empty.")
        for label, value in (
            ("policy lock hash", self.policy_lock_hash),
            ("policy plan hash", self.policy_plan_hash),
            ("assignment-table hash", self.assignment_table_hash),
        ):
            if not _is_lower_hex(value):
                raise ProtocolError(f"Stage-70 {label} is malformed.")
        _require_sha256(self.assignment_table_sha256, "assignment-table SHA-256")
        if self.assignment_count <= 0:
            raise ProtocolError("Stage-70 policy assignment coverage is empty.")
        if self.replicate_count != EXPECTED_REPLICATES_PER_POLICY:
            raise ProtocolError("Stage-70 policy replicate coverage drifted.")
        if self.target_expert_excluded is not True:
            raise ProtocolError("Stage-70 policy includes the held-out target expert.")
        if self.total_per_class != EXPECTED_SYNTHETIC_ROWS_PER_CLASS:
            raise ProtocolError("Stage-70 per-class synthetic budget drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_artifact_id": self.policy_artifact_id,
            "policy_lock_hash": self.policy_lock_hash,
            "policy_plan_hash": self.policy_plan_hash,
            "assignment_table_hash": self.assignment_table_hash,
            "assignment_table_sha256": self.assignment_table_sha256,
            "assignment_count": self.assignment_count,
            "replicate_count": self.replicate_count,
            "target_expert_excluded": self.target_expert_excluded,
            "total_per_class": self.total_per_class,
        }


@dataclass(frozen=True)
class AuthorizationValidationInputs:
    """Validated upstreams shared by reservation and final authorization.

    Production construction lives in :mod:`inputs` and traverses all public
    upstream validators.  Tests may inject an instance explicitly; the loaded
    production configuration never enables that seam.
    """

    consumption_ledger: Mapping[str, object]
    canonical_reference: ArtifactBinding
    bank: ArtifactBinding
    generation: ArtifactBinding
    policies: tuple[PolicyBinding, ...]
    generation_lock: object
    policy_replicates: tuple[object, ...]
    classifier_spec: Mapping[str, object]

    def __post_init__(self) -> None:
        ledger = deepcopy(dict(self.consumption_ledger))
        validate_consumption_ledger(ledger)
        policy_ids = tuple(binding.policy_id for binding in self.policies)
        if policy_ids != tuple(POLICY_ARMS):
            raise ProtocolError("Stage-70 policy binding order or coverage drifted.")
        if len(self.policy_replicates) != EXPECTED_EVALUATION_PLAN_ROWS:
            raise ProtocolError("Stage-70 frozen-policy replicate coverage drifted.")
        _validate_replicate_geometry(self.policy_replicates, self.policies)
        classifier = deepcopy(dict(self.classifier_spec))
        _validate_classifier_spec(classifier)
        object.__setattr__(self, "consumption_ledger", MappingProxyType(ledger))
        object.__setattr__(self, "classifier_spec", MappingProxyType(classifier))

    @property
    def generation_lock_hash(self) -> str:
        value = getattr(self.generation_lock, "generation_lock_hash", "")
        if not _is_lower_hex(str(value)):
            raise ProtocolError("Stage-70 GenerationLock identity is malformed.")
        return str(value)

    def bindings_payload(self) -> dict[str, object]:
        return {
            "canonical_reference": self.canonical_reference.to_payload(),
            "expert_bank": self.bank.to_payload(),
            "generation_lock": self.generation.to_payload(),
            "policies": [binding.to_payload() for binding in self.policies],
        }


@dataclass(frozen=True)
class CacheBinding:
    """Validated, label-blind target-cache identity consumed by final auth."""

    artifact_id: str
    manifest_sha256: str
    target_evaluation_reservation_id: str
    target_evaluation_reservation_protocol_hash: str
    cache_extractor_protocol_hash: str
    row_count: int
    rows_by_center: Mapping[str, int]
    row_order_hash: str
    shard_sha256_by_center: Mapping[str, str]
    content_hash: str
    purpose: str
    fresh_evidence: bool

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ProtocolError("Stage-70 cache artifact identity is empty.")
        _require_sha256(self.manifest_sha256, "cache manifest SHA-256")
        for label, value in (
            ("reservation protocol hash", self.target_evaluation_reservation_protocol_hash),
            ("cache extractor protocol hash", self.cache_extractor_protocol_hash),
            ("cache row-order hash", self.row_order_hash),
            ("cache content hash", self.content_hash),
        ):
            if not _is_lower_hex(value):
                raise ProtocolError(f"Stage-70 {label} is malformed.")
        if not self.target_evaluation_reservation_id:
            raise ProtocolError("Stage-70 cache reservation identity is empty.")
        if self.row_count != EXPECTED_TEST_ROWS:
            raise ProtocolError("Stage-70 target-cache row count drifted.")
        rows = {str(key): int(value) for key, value in self.rows_by_center.items()}
        if set(rows) != set(CENTERS) or sum(rows.values()) != EXPECTED_TEST_ROWS:
            raise ProtocolError("Stage-70 target-cache center coverage drifted.")
        shards = {
            str(key): str(value) for key, value in self.shard_sha256_by_center.items()
        }
        if set(shards) != set(CENTERS):
            raise ProtocolError("Stage-70 target-cache shard coverage drifted.")
        for value in shards.values():
            _require_sha256(value, "cache shard SHA-256")
        if self.purpose != PURPOSE or self.fresh_evidence is not False:
            raise ProtocolError("Stage-70 target cache violates the consumed-test boundary.")
        object.__setattr__(self, "rows_by_center", MappingProxyType(rows))
        object.__setattr__(self, "shard_sha256_by_center", MappingProxyType(shards))

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "manifest_sha256": self.manifest_sha256,
            "target_evaluation_reservation_id": (
                self.target_evaluation_reservation_id
            ),
            "target_evaluation_reservation_protocol_hash": (
                self.target_evaluation_reservation_protocol_hash
            ),
            "cache_extractor_protocol_hash": self.cache_extractor_protocol_hash,
            "row_count": self.row_count,
            "rows_by_center": dict(self.rows_by_center),
            "row_order_hash": self.row_order_hash,
            "shard_sha256_by_center": dict(self.shard_sha256_by_center),
            "content_hash": self.content_hash,
            "purpose": self.purpose,
            "fresh_evidence": self.fresh_evidence,
            "labels_persisted": False,
            "sample_ids_persisted": False,
            "image_paths_persisted": False,
        }


@dataclass(frozen=True)
class FinalAuthorizationToken:
    """Immutable, self-hashing prediction-only authorization token."""

    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        payload = deepcopy(dict(self._payload))
        observed = payload.get("authorization_token_hash")
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "authorization_token_hash"
        }
        if observed != stable_hash(unhashed):
            raise ProtocolError("Stage-70 final authorization token hash drifted.")
        required = {
            "schema_version": "midogpp_stage70_final_prediction_authorization_token_v1",
            "phase": FINAL_AUTHORIZATION_PHASE,
            "status": RUN_COMPLETE,
            "experiment_id": FINAL_AUTHORIZATION_EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "purpose": PURPOSE,
            "fresh_evidence": False,
            "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
            "descriptive_status": FINAL_DESCRIPTIVE_STATUS,
            "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
            "prediction_allowed": True,
            "label_access_allowed": False,
            "metric_scoring_allowed": False,
            "policy_or_seed_selection_allowed": False,
        }
        mismatch = [
            key for key, value in required.items() if payload.get(key) != value
        ]
        if mismatch:
            raise ProtocolError(
                f"Stage-70 final authorization token phase/status drifted: {mismatch}."
            )
        object.__setattr__(self, "_payload", MappingProxyType(payload))

    @property
    def authorization_token_hash(self) -> str:
        return str(self._payload["authorization_token_hash"])

    def to_payload(self) -> dict[str, object]:
        return deepcopy(dict(self._payload))


def validate_consumption_ledger(ledger: Mapping[str, object]) -> None:
    """Require the exact previously-consumed test authorization boundary."""

    canonical_reuse_key = "may_be_reused_for_descriptive_locked_model_scoring"
    published_reuse_key = "may_be_reused_for_descriptive_locked-model_scoring"
    if canonical_reuse_key in ledger and published_reuse_key in ledger:
        if ledger[canonical_reuse_key] != ledger[published_reuse_key]:
            raise ProtocolError(
                "Stage-70 test-consumption ledger has conflicting descriptive-use aliases."
            )
    descriptive_reuse = ledger.get(
        canonical_reuse_key,
        ledger.get(published_reuse_key),
    )
    required = {
        "status": TEST_CONSUMPTION_STATUS,
        "split": EXPECTED_SPLIT,
        "row_count": EXPECTED_TEST_ROWS,
        "observed_centers": EXPECTED_CENTER_COUNT,
        "may_be_reused_as_fresh_representation_selection_evidence": False,
    }
    mismatch = [
        f"{key}={ledger.get(key)!r}" for key, value in required.items()
        if ledger.get(key) != value
    ]
    if descriptive_reuse is not True:
        mismatch.append(
            f"descriptive_locked_model_scoring={descriptive_reuse!r}"
        )
    if mismatch:
        raise ProtocolError(
            "Stage-70 test-consumption ledger is not eligible for descriptive "
            "locked-model use: " + "; ".join(mismatch)
        )


def make_final_authorization_token(
    payload: Mapping[str, object],
) -> FinalAuthorizationToken:
    value = deepcopy(dict(payload))
    if "authorization_token_hash" in value:
        raise ProtocolError("Caller may not supply a final authorization token hash.")
    value["authorization_token_hash"] = stable_hash(value)
    return FinalAuthorizationToken(value)


def _validate_replicate_geometry(
    replicates: Sequence[object],
    policies: Sequence[PolicyBinding],
) -> None:
    expected = {
        (policy_id, target, training_seed, generation_seed)
        for policy_id in POLICY_ARMS
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    observed: set[tuple[str, str, int, int]] = set()
    by_policy = {binding.policy_id: binding for binding in policies}
    for row in replicates:
        key = (
            str(getattr(row, "policy_id", "")),
            str(getattr(row, "target_center", "")),
            int(getattr(row, "training_seed", -1)),
            int(getattr(row, "generation_seed", -1)),
        )
        if key in observed:
            raise ProtocolError("Stage-70 frozen-policy replicate keys are duplicated.")
        observed.add(key)
        assignments = tuple(getattr(row, "assignments", ()))
        if not assignments:
            raise ProtocolError("Stage-70 frozen-policy replicate lacks assignments.")
        if any(
            str(getattr(item, "source_center", "")) == key[1]
            or bool(getattr(item, "target_expert", False))
            for item in assignments
        ):
            raise ProtocolError("Stage-70 frozen-policy replicate includes target expert.")
        if sum(int(getattr(item, "source_budget_per_class", 0)) for item in assignments) != TOTAL_PER_CLASS:
            raise ProtocolError("Stage-70 frozen-policy replicate budget drifted.")
        binding = by_policy.get(key[0])
        if binding is None or (
            str(getattr(row, "policy_lock_hash", "")) != binding.policy_lock_hash
            or str(getattr(row, "policy_plan_hash", "")) != binding.policy_plan_hash
            or str(getattr(row, "assignment_table_hash", ""))
            != binding.assignment_table_hash
        ):
            raise ProtocolError("Stage-70 replicate/policy binding drifted.")
    if observed != expected:
        raise ProtocolError("Stage-70 frozen-policy factorial coverage drifted.")


def _validate_classifier_spec(payload: Mapping[str, object]) -> None:
    required = {
        "family": "sklearn_logistic_regression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 3000,
        "class_weight": None,
        "random_state": 23,
        "l1_ratio": None,
        "threshold_policy": "predict",
        "scaler_fit": "synthetic_train_only",
        "scaler_family": "sklearn.preprocessing.StandardScaler",
        "fit_in_stage_40": False,
    }
    mismatch = [key for key, value in required.items() if payload.get(key) != value]
    if mismatch or not _is_lower_hex(str(payload.get("config_hash", ""))):
        raise ProtocolError(f"Stage-70 frozen classifier spec drifted: {mismatch}.")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or not _is_lower_hex(value):
        raise ProtocolError(f"Stage-70 {label} is malformed.")


def _is_lower_hex(value: str) -> bool:
    return bool(value) and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = (
    "ArtifactBinding",
    "AuthorizationValidationInputs",
    "CacheBinding",
    "CLAIM_SCOPE",
    "EXPECTED_CENTER_COUNT",
    "EXPECTED_EVALUATION_PLAN_ROWS",
    "EXPECTED_REPLICATES_PER_POLICY",
    "EXPECTED_SPLIT",
    "EXPECTED_SYNTHETIC_ROWS_PER_CLASS",
    "EXPECTED_TEST_ROWS",
    "FINAL_AUTHORIZATION_EXPERIMENT_ID",
    "FINAL_AUTHORIZATION_EXPERIMENT_NAME",
    "FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID",
    "FINAL_AUTHORIZATION_PHASE",
    "FINAL_DESCRIPTIVE_STATUS",
    "FRESH_CONFIRMATORY_STATUS",
    "FinalAuthorizationToken",
    "POLICY_ARMS",
    "PURPOSE",
    "PolicyBinding",
    "RESERVATION_DESCRIPTIVE_STATUS",
    "RESERVATION_EXPERIMENT_ID",
    "RESERVATION_EXPERIMENT_NAME",
    "RESERVATION_OUTPUT_ARTIFACT_ID",
    "RESERVATION_PHASE",
    "RUN_COMPLETE",
    "TEST_CONSUMPTION_STATUS",
    "VALIDATION_PASS",
    "make_final_authorization_token",
    "validate_consumption_ledger",
)
