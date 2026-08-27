"""Immutable nine-target SCEPTRE router bundle and canonical replay.

The bundle depends on target-specific adaptive freezes and neutral policy
contracts.  G-proposal persistence depends on this bundle only through values
passed at construction time; the optional convenience binder uses a local
import to keep module initialization acyclic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.generation.contracts import GenerationLock
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre.candidate_menu import build_candidate_menu

from .adaptive_model_freeze import (
    AdaptiveUtilityDecision,
    AdaptiveUtilityExactBFallback,
    AdaptiveUtilityRoute,
    FrozenAdaptiveUtilityModel,
    PREDICTED_UTILITY_SEMANTICS,
    _finite,
    _identifier,
    _tuple_field,
    _validate_frozen_bindings,
)
from .experiment_contracts import claim_boundary_payload
from .hashing import canonical_bytes, canonical_hash, require_sha256
from .identity import CLAIM_SCOPE, PUBLICATION_STATUS, TERMINAL_DECISION
from .outcome_surface import EXACT_B_CANDIDATE
from .partitions import (
    FOLD_COUNT,
    PARTITION_NAMESPACE,
    PARTITION_SEED,
    ThreeRolePartition,
)
from .policy_contracts import (
    CALIBRATION_MAXIMUM_BRIER_DELTA,
    CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
    CALIBRATION_MINIMUM_BACC_GAIN,
    FIXED_ACCEPTANCE_PROBABILITY,
    SUPPORT_MINIMUM_BACC_GAIN,
)
from .uncertainty import DirichletBootstrapConfig


FULL_ROUTER_SCHEMA = "sceptre_full_prelabel_router_freeze_v1"
FULL_ROUTER_ROLE = "COMPLETE_ROUTER_FROZEN_BEFORE_TEST_LABEL_ACCESS"
CASE_IDENTITY_SCHEMA = ("target_center", "case_id", "sample_id")
PARTITION_SCHEMA = "sceptre_three_role_partition_v1"
EXPECTED_TOTAL_CASE_COUNT = 218


def _partition_identity_hash(partition: ThreeRolePartition) -> str:
    return canonical_hash(
        {
            "schema_version": "sceptre_case_identity_inventory_v1",
            "case_identity_fields": list(CASE_IDENTITY_SCHEMA),
            "rows": [
                {
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                }
                for row in partition.identities
            ],
            "whole_case_partitioning": True,
            "labels_consumed": False,
        }
    )


def _partition_fold_inventory_hash(partition: ThreeRolePartition) -> str:
    return canonical_hash(
        {
            "schema_version": "sceptre_prelabel_fold_inventory_v1",
            "partition_hash": partition.partition_hash,
            "folds": [
                {
                    "target_center": fold.target_center,
                    "fold_ordinal": fold.fold_ordinal,
                    "fold_hash": fold.fold_hash,
                    "selection_case_set_hash": fold.case_set_hash("SELECTION"),
                    "calibration_case_set_hash": fold.case_set_hash("CALIBRATION"),
                    "evaluation_case_set_hash": fold.case_set_hash("EVALUATION"),
                }
                for fold in partition.folds
            ],
            "whole_case_roles_disjoint": True,
            "every_case_evaluated_exactly_once": True,
        }
    )


def _validate_prelabel_partition(
    partition: ThreeRolePartition,
) -> tuple[str, str]:
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE full-router freeze requires a typed partition.")
    case_keys = {
        (row.target_center, row.case_id) for row in partition.identities
    }
    if len(case_keys) != EXPECTED_TOTAL_CASE_COUNT:
        raise ProtocolError("SCEPTRE prelabel partition case inventory drifted.")
    expected_folds = tuple(
        (target, fold) for target in CENTERS for fold in range(FOLD_COUNT)
    )
    if tuple(
        (fold.target_center, fold.fold_ordinal) for fold in partition.folds
    ) != expected_folds:
        raise ProtocolError("SCEPTRE prelabel partition fold inventory drifted.")
    return (
        _partition_identity_hash(partition),
        _partition_fold_inventory_hash(partition),
    )


def _dirichlet_config_from_payload(
    payload: object,
) -> DirichletBootstrapConfig:
    if not isinstance(payload, Mapping):
        raise ProtocolError("SCEPTRE frozen Dirichlet config is invalid.")
    expected_keys = set(DirichletBootstrapConfig().to_payload())
    if set(payload) != expected_keys:
        raise ProtocolError("SCEPTRE frozen Dirichlet config schema drifted.")
    try:
        config = DirichletBootstrapConfig(
            draw_count=payload["draw_count"],
            rng_seed=payload["rng_seed"],
            config_hash=str(payload["config_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE frozen Dirichlet config is invalid.") from exc
    if config.to_payload() != dict(payload):
        raise ProtocolError("SCEPTRE frozen Dirichlet config semantics drifted.")
    return config


@dataclass(frozen=True, slots=True)
class FrozenPrelabelRouter:
    """Complete nine-target decision policy frozen before any test label access."""

    models: tuple[FrozenAdaptiveUtilityModel, ...]
    generation_lock_hash: str
    generation_lock_payload_sha256: str
    bank_lock_hash: str
    support_minimum_bacc_gain: float
    calibration_minimum_bacc_gain: float
    calibration_maximum_brier_delta: float
    calibration_maximum_log_loss_delta: float
    dirichlet_config: DirichletBootstrapConfig
    partition_hash: str
    partition_identity_sha256: str
    partition_fold_inventory_sha256: str
    partition_namespace: str
    partition_seed: int
    fold_count: int
    partition_schema: str
    case_identity_schema: tuple[str, ...]
    claim_boundary_sha256: str
    publication_status: str
    terminal_decision: str
    claim_scope: str
    descriptive_only: bool
    fresh_evidence: bool
    execution_authorized: bool
    full_router_sha256: str = ""

    def __post_init__(self) -> None:
        models = tuple(self.models)
        if (
            len(models) != len(CENTERS)
            or any(
                not isinstance(model, FrozenAdaptiveUtilityModel)
                for model in models
            )
            or tuple(model.outer_target for model in models) != CENTERS
        ):
            raise ProtocolError(
                "SCEPTRE full-router freeze requires exactly one model per H in order."
            )
        lock_hash = _identifier(self.generation_lock_hash, "full GenerationLock hash")
        lock_payload = require_sha256(
            self.generation_lock_payload_sha256,
            "full GenerationLock payload",
        )
        bank_hash = _identifier(self.bank_lock_hash, "full bank-lock hash")
        if any(
            model.generation_lock_hash != lock_hash
            or model.generation_lock_payload_sha256 != lock_payload
            or model.bank_lock_hash != bank_hash
            for model in models
        ):
            raise ProtocolError("SCEPTRE per-H models do not share one frozen bank.")

        support_minimum = _finite(
            self.support_minimum_bacc_gain,
            "support minimum BACC gain",
        )
        calibration_minimum = _finite(
            self.calibration_minimum_bacc_gain,
            "calibration minimum BACC gain",
        )
        maximum_brier = _finite(
            self.calibration_maximum_brier_delta,
            "calibration maximum Brier delta",
        )
        maximum_log = _finite(
            self.calibration_maximum_log_loss_delta,
            "calibration maximum log-loss delta",
        )
        if (
            support_minimum != SUPPORT_MINIMUM_BACC_GAIN
            or calibration_minimum != CALIBRATION_MINIMUM_BACC_GAIN
            or maximum_brier != CALIBRATION_MAXIMUM_BRIER_DELTA
            or maximum_log != CALIBRATION_MAXIMUM_LOG_LOSS_DELTA
        ):
            raise ProtocolError("SCEPTRE full-router decision thresholds drifted.")
        if (
            not isinstance(self.dirichlet_config, DirichletBootstrapConfig)
            or self.dirichlet_config.acceptance_probability
            != FIXED_ACCEPTANCE_PROBABILITY
        ):
            raise ProtocolError("SCEPTRE full-router Dirichlet policy drifted.")

        partition_hash = require_sha256(
            self.partition_hash,
            "full-router partition",
        )
        partition_identity = require_sha256(
            self.partition_identity_sha256,
            "partition identity inventory",
        )
        fold_inventory = require_sha256(
            self.partition_fold_inventory_sha256,
            "partition fold inventory",
        )
        case_schema = tuple(
            _identifier(value, "case identity field")
            for value in self.case_identity_schema
        )
        if (
            self.partition_namespace != PARTITION_NAMESPACE
            or self.partition_seed != PARTITION_SEED
            or self.fold_count != FOLD_COUNT
            or self.partition_schema != PARTITION_SCHEMA
            or case_schema != CASE_IDENTITY_SCHEMA
        ):
            raise ProtocolError("SCEPTRE whole-case partition schema drifted.")

        expected_claim = canonical_hash(claim_boundary_payload())
        claim_hash = require_sha256(
            self.claim_boundary_sha256,
            "claim boundary",
        )
        if (
            claim_hash != expected_claim
            or self.publication_status != PUBLICATION_STATUS
            or self.terminal_decision != TERMINAL_DECISION
            or self.claim_scope != CLAIM_SCOPE
            or self.descriptive_only is not True
            or self.fresh_evidence is not False
            or self.execution_authorized is not False
        ):
            raise ProtocolError("SCEPTRE full-router claim boundary drifted.")

        object.__setattr__(self, "models", models)
        object.__setattr__(self, "generation_lock_hash", lock_hash)
        object.__setattr__(self, "generation_lock_payload_sha256", lock_payload)
        object.__setattr__(self, "bank_lock_hash", bank_hash)
        object.__setattr__(self, "support_minimum_bacc_gain", support_minimum)
        object.__setattr__(
            self,
            "calibration_minimum_bacc_gain",
            calibration_minimum,
        )
        object.__setattr__(
            self,
            "calibration_maximum_brier_delta",
            maximum_brier,
        )
        object.__setattr__(
            self,
            "calibration_maximum_log_loss_delta",
            maximum_log,
        )
        object.__setattr__(self, "partition_hash", partition_hash)
        object.__setattr__(
            self,
            "partition_identity_sha256",
            partition_identity,
        )
        object.__setattr__(
            self,
            "partition_fold_inventory_sha256",
            fold_inventory,
        )
        object.__setattr__(self, "case_identity_schema", case_schema)
        object.__setattr__(self, "claim_boundary_sha256", claim_hash)
        expected_hash = canonical_hash(self._payload_without_hash())
        if self.full_router_sha256 and require_sha256(
            self.full_router_sha256,
            "full prelabel router",
        ) != expected_hash:
            raise ProtocolError("SCEPTRE full-router SHA-256 drifted.")
        object.__setattr__(self, "full_router_sha256", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": FULL_ROUTER_SCHEMA,
            "artifact_role": FULL_ROUTER_ROLE,
            "frozen_before_test_label_access": True,
            "models": [model.to_payload() for model in self.models],
            "generation_identity": {
                "generation_lock_hash": self.generation_lock_hash,
                "generation_lock_payload_sha256": (
                    self.generation_lock_payload_sha256
                ),
                "bank_lock_hash": self.bank_lock_hash,
                "all_targets_share_generation_lock": True,
                "all_targets_share_bank_lock": True,
            },
            "decision_policy": self._decision_policy_payload(),
            "decision_policy_sha256": self.decision_policy_sha256,
            "partition_identity": {
                "partition_schema": self.partition_schema,
                "partition_hash": self.partition_hash,
                "partition_identity_sha256": self.partition_identity_sha256,
                "partition_fold_inventory_sha256": (
                    self.partition_fold_inventory_sha256
                ),
                "partition_namespace": self.partition_namespace,
                "partition_seed": self.partition_seed,
                "fold_count": self.fold_count,
                "case_identity_schema": list(self.case_identity_schema),
                "expected_total_case_count": EXPECTED_TOTAL_CASE_COUNT,
                "whole_case_roles_disjoint": True,
                "labels_consumed": False,
            },
            "claim_status": {
                "claim_boundary_sha256": self.claim_boundary_sha256,
                "publication_status": self.publication_status,
                "terminal_decision": self.terminal_decision,
                "claim_scope": self.claim_scope,
                "descriptive_only": self.descriptive_only,
                "fresh_evidence": self.fresh_evidence,
                "execution_authorized": self.execution_authorized,
                "promotion_allowed": False,
            },
        }

    def _decision_policy_payload(self) -> dict[str, object]:
        return {
            "adaptive_score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "adaptive_higher_is_better": True,
            "core_proxy_energy_lower_is_better": True,
            "support_minimum_bacc_gain": self.support_minimum_bacc_gain,
            "calibration_thresholds": {
                "minimum_bacc_gain": self.calibration_minimum_bacc_gain,
                "maximum_brier_delta": self.calibration_maximum_brier_delta,
                "maximum_log_loss_delta": self.calibration_maximum_log_loss_delta,
            },
            "dirichlet_config": self.dirichlet_config.to_payload(),
            "fixed_joint_acceptance_probability": FIXED_ACCEPTANCE_PROBABILITY,
            "ties_fall_back_to_exact_b": True,
            "invalid_or_missing_evidence_falls_back_to_exact_b": True,
            "exact_b_action_id": EXACT_B_CANDIDATE,
        }

    @property
    def decision_policy_sha256(self) -> str:
        return canonical_hash(
            {
                "schema_version": "sceptre_complete_decision_policy_v1",
                **self._decision_policy_payload(),
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "full_router_sha256": self.full_router_sha256,
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    @property
    def router_bundle_hash(self) -> str:
        """Phase-manager name for the complete prelabel freeze identity."""

        return self.full_router_sha256

    def model_for_target(self, target_center: str) -> FrozenAdaptiveUtilityModel:
        target = str(target_center)
        matches = tuple(model for model in self.models if model.outer_target == target)
        if len(matches) != 1:
            raise ProtocolError("SCEPTRE full-router target model is absent.")
        return matches[0]

    def bind_g_proposal(
        self,
        decision: AdaptiveUtilityDecision,
    ) -> "FrozenGProposal":
        # Imported here so the immutable bundle does not depend on proposal
        # persistence at module-import time.
        from .g_proposal_persistence import FrozenGProposal

        if not isinstance(
            decision,
            (AdaptiveUtilityRoute, AdaptiveUtilityExactBFallback),
        ):
            raise ProtocolError(
                "SCEPTRE G proposal requires a typed adaptive decision."
            )
        model = self.model_for_target(decision.outer_target)
        if (
            decision.candidate_sources != model.candidate_sources
            or decision.frozen_model_sha256 != model.model_sha256
            or decision.candidate_menu_payload_sha256
            != model.candidate_menu_payload_sha256
            or decision.exact_b_control_payload_sha256
            != model.exact_b_control_payload_sha256
        ):
            raise ProtocolError("SCEPTRE G proposal decision lineage drifted.")
        if isinstance(decision, AdaptiveUtilityRoute):
            proposed_route = decision.selected_source_center
            winner_sources = (decision.selected_source_center,)
            fallback = False
            reason = "UNIQUE_PREDICTED_UTILITY_ROUTE"
        else:
            proposed_route = EXACT_B_CANDIDATE
            winner_sources = decision.winner_sources
            fallback = True
            reason = decision.reason
        return FrozenGProposal(
            target_center=model.outer_target,
            full_router_sha256=self.full_router_sha256,
            frozen_model_sha256=model.model_sha256,
            partition_hash=self.partition_hash,
            generation_lock_payload_sha256=self.generation_lock_payload_sha256,
            candidate_menu_hash=model.candidate_menu_hash,
            candidate_menu_payload_sha256=model.candidate_menu_payload_sha256,
            exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
            exact_b_control_payload_sha256=model.exact_b_control_payload_sha256,
            decision_policy_sha256=self.decision_policy_sha256,
            adaptive_decision_sha256=decision.decision_sha256,
            evidence_sha256=decision.evidence_sha256,
            ranking_sha256=decision.ranking_sha256,
            winner_sources=winner_sources,
            proposed_route=proposed_route,
            fallback_to_exact_b=fallback,
            reason=reason,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "FrozenPrelabelRouter":
        expected_top = {
            "schema_version",
            "artifact_role",
            "frozen_before_test_label_access",
            "models",
            "generation_identity",
            "decision_policy",
            "decision_policy_sha256",
            "partition_identity",
            "claim_status",
            "full_router_sha256",
        }
        if set(payload) != expected_top or (
            payload.get("schema_version") != FULL_ROUTER_SCHEMA
            or payload.get("artifact_role") != FULL_ROUTER_ROLE
            or payload.get("frozen_before_test_label_access") is not True
        ):
            raise ProtocolError("SCEPTRE full-router top-level schema drifted.")
        models_raw = payload.get("models")
        generation = payload.get("generation_identity")
        policy = payload.get("decision_policy")
        partition = payload.get("partition_identity")
        claim = payload.get("claim_status")
        if (
            not isinstance(models_raw, Sequence)
            or isinstance(models_raw, (str, bytes, bytearray))
            or not isinstance(generation, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(partition, Mapping)
            or not isinstance(claim, Mapping)
        ):
            raise ProtocolError("SCEPTRE full-router sections are invalid.")
        generation_keys = {
            "generation_lock_hash",
            "generation_lock_payload_sha256",
            "bank_lock_hash",
            "all_targets_share_generation_lock",
            "all_targets_share_bank_lock",
        }
        policy_keys = {
            "adaptive_score_semantics",
            "adaptive_higher_is_better",
            "core_proxy_energy_lower_is_better",
            "support_minimum_bacc_gain",
            "calibration_thresholds",
            "dirichlet_config",
            "fixed_joint_acceptance_probability",
            "ties_fall_back_to_exact_b",
            "invalid_or_missing_evidence_falls_back_to_exact_b",
            "exact_b_action_id",
        }
        partition_keys = {
            "partition_schema",
            "partition_hash",
            "partition_identity_sha256",
            "partition_fold_inventory_sha256",
            "partition_namespace",
            "partition_seed",
            "fold_count",
            "case_identity_schema",
            "expected_total_case_count",
            "whole_case_roles_disjoint",
            "labels_consumed",
        }
        claim_keys = {
            "claim_boundary_sha256",
            "publication_status",
            "terminal_decision",
            "claim_scope",
            "descriptive_only",
            "fresh_evidence",
            "execution_authorized",
            "promotion_allowed",
        }
        if (
            set(generation) != generation_keys
            or set(policy) != policy_keys
            or set(partition) != partition_keys
            or set(claim) != claim_keys
        ):
            raise ProtocolError("SCEPTRE full-router nested schema drifted.")
        calibration = policy.get("calibration_thresholds")
        if not isinstance(calibration, Mapping) or set(calibration) != {
            "minimum_bacc_gain",
            "maximum_brier_delta",
            "maximum_log_loss_delta",
        }:
            raise ProtocolError("SCEPTRE frozen calibration schema drifted.")
        if (
            generation.get("all_targets_share_generation_lock") is not True
            or generation.get("all_targets_share_bank_lock") is not True
            or policy.get("adaptive_score_semantics")
            != PREDICTED_UTILITY_SEMANTICS
            or policy.get("adaptive_higher_is_better") is not True
            or policy.get("core_proxy_energy_lower_is_better") is not True
            or policy.get("fixed_joint_acceptance_probability")
            != FIXED_ACCEPTANCE_PROBABILITY
            or policy.get("ties_fall_back_to_exact_b") is not True
            or policy.get("invalid_or_missing_evidence_falls_back_to_exact_b")
            is not True
            or policy.get("exact_b_action_id") != EXACT_B_CANDIDATE
            or partition.get("expected_total_case_count")
            != EXPECTED_TOTAL_CASE_COUNT
            or partition.get("whole_case_roles_disjoint") is not True
            or partition.get("labels_consumed") is not False
            or claim.get("promotion_allowed") is not False
            or require_sha256(
                payload.get("decision_policy_sha256"),
                "complete decision policy",
            )
            != canonical_hash(
                {
                    "schema_version": "sceptre_complete_decision_policy_v1",
                    **dict(policy),
                }
            )
        ):
            raise ProtocolError("SCEPTRE full-router frozen semantics drifted.")
        try:
            models = tuple(
                FrozenAdaptiveUtilityModel.from_payload(model)
                for model in models_raw
                if isinstance(model, Mapping)
            )
            if len(models) != len(models_raw):
                raise ProtocolError("SCEPTRE full-router model payload is invalid.")
            return cls(
                models=models,
                generation_lock_hash=str(generation["generation_lock_hash"]),
                generation_lock_payload_sha256=str(
                    generation["generation_lock_payload_sha256"]
                ),
                bank_lock_hash=str(generation["bank_lock_hash"]),
                support_minimum_bacc_gain=_finite(
                    policy["support_minimum_bacc_gain"],
                    "support minimum BACC gain",
                ),
                calibration_minimum_bacc_gain=_finite(
                    calibration["minimum_bacc_gain"],
                    "calibration minimum BACC gain",
                ),
                calibration_maximum_brier_delta=_finite(
                    calibration["maximum_brier_delta"],
                    "calibration maximum Brier delta",
                ),
                calibration_maximum_log_loss_delta=_finite(
                    calibration["maximum_log_loss_delta"],
                    "calibration maximum log-loss delta",
                ),
                dirichlet_config=_dirichlet_config_from_payload(
                    policy["dirichlet_config"]
                ),
                partition_hash=str(partition["partition_hash"]),
                partition_identity_sha256=str(
                    partition["partition_identity_sha256"]
                ),
                partition_fold_inventory_sha256=str(
                    partition["partition_fold_inventory_sha256"]
                ),
                partition_namespace=str(partition["partition_namespace"]),
                partition_seed=partition["partition_seed"],
                fold_count=partition["fold_count"],
                partition_schema=str(partition["partition_schema"]),
                case_identity_schema=tuple(
                    str(value)
                    for value in _tuple_field(
                        partition["case_identity_schema"],
                        "case identity schema",
                    )
                ),
                claim_boundary_sha256=str(claim["claim_boundary_sha256"]),
                publication_status=str(claim["publication_status"]),
                terminal_decision=str(claim["terminal_decision"]),
                claim_scope=str(claim["claim_scope"]),
                descriptive_only=claim["descriptive_only"],
                fresh_evidence=claim["fresh_evidence"],
                execution_authorized=claim["execution_authorized"],
                full_router_sha256=str(payload["full_router_sha256"]),
            )
        except KeyError as exc:
            raise ProtocolError("SCEPTRE full-router field is absent.") from exc

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "FrozenPrelabelRouter":
        if not isinstance(payload, bytes):
            raise ProtocolError("SCEPTRE full-router serialization must be bytes.")
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Cannot parse SCEPTRE full-router bytes.") from exc
        if not isinstance(raw, Mapping) or payload != canonical_bytes(raw):
            raise ProtocolError("SCEPTRE full-router bytes are not canonical.")
        return cls.from_payload(raw)


FrozenRouterBundle = FrozenPrelabelRouter


@dataclass(frozen=True, slots=True)
class FullRouterReplayReceipt:
    full_router_sha256: str
    generation_lock_payload_sha256: str
    partition_hash: str
    partition_identity_sha256: str
    partition_fold_inventory_sha256: str
    dirichlet_config_hash: str
    claim_boundary_sha256: str
    model_sha256_by_target: tuple[tuple[str, str], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        models = tuple(self.model_sha256_by_target)
        if tuple(target for target, _ in models) != CENTERS:
            raise ProtocolError("SCEPTRE full-router replay target inventory drifted.")
        normalized_models = tuple(
            (target, require_sha256(digest, "replayed H model"))
            for target, digest in models
        )
        body = {
            "schema_version": "sceptre_full_router_replay_receipt_v1",
            "status": "PASS",
            "full_router_sha256": require_sha256(
                self.full_router_sha256,
                "full prelabel router",
            ),
            "generation_lock_payload_sha256": require_sha256(
                self.generation_lock_payload_sha256,
                "GenerationLock payload",
            ),
            "partition_hash": require_sha256(
                self.partition_hash,
                "router partition",
            ),
            "partition_identity_sha256": require_sha256(
                self.partition_identity_sha256,
                "partition identity inventory",
            ),
            "partition_fold_inventory_sha256": require_sha256(
                self.partition_fold_inventory_sha256,
                "partition fold inventory",
            ),
            "dirichlet_config_hash": require_sha256(
                self.dirichlet_config_hash,
                "Dirichlet config",
            ),
            "claim_boundary_sha256": require_sha256(
                self.claim_boundary_sha256,
                "claim boundary",
            ),
            "model_sha256_by_target": [list(row) for row in normalized_models],
            "frozen_before_test_label_access": True,
        }
        expected = canonical_hash(body)
        if self.receipt_sha256 and require_sha256(
            self.receipt_sha256,
            "full-router replay receipt",
        ) != expected:
            raise ProtocolError("SCEPTRE full-router replay receipt drifted.")
        for field_name in (
            "full_router_sha256",
            "generation_lock_payload_sha256",
            "partition_hash",
            "partition_identity_sha256",
            "partition_fold_inventory_sha256",
            "dirichlet_config_hash",
            "claim_boundary_sha256",
        ):
            object.__setattr__(self, field_name, body[field_name])
        object.__setattr__(self, "model_sha256_by_target", normalized_models)
        object.__setattr__(self, "receipt_sha256", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_full_router_replay_receipt_v1",
            "status": "PASS",
            "full_router_sha256": self.full_router_sha256,
            "generation_lock_payload_sha256": (
                self.generation_lock_payload_sha256
            ),
            "partition_hash": self.partition_hash,
            "partition_identity_sha256": self.partition_identity_sha256,
            "partition_fold_inventory_sha256": (
                self.partition_fold_inventory_sha256
            ),
            "dirichlet_config_hash": self.dirichlet_config_hash,
            "claim_boundary_sha256": self.claim_boundary_sha256,
            "model_sha256_by_target": [
                list(row) for row in self.model_sha256_by_target
            ],
            "frozen_before_test_label_access": True,
            "receipt_sha256": self.receipt_sha256,
        }


def freeze_full_prelabel_router(
    models: Sequence[FrozenAdaptiveUtilityModel],
    *,
    generation_lock: GenerationLock,
    partition: ThreeRolePartition,
    dirichlet_config: DirichletBootstrapConfig | None = None,
) -> FrozenPrelabelRouter:
    """Freeze the complete nine-target router and policy before label access."""

    if not isinstance(generation_lock, GenerationLock):
        raise ProtocolError("SCEPTRE full-router freeze requires a GenerationLock.")
    frozen_models = tuple(models)
    if (
        len(frozen_models) != len(CENTERS)
        or any(
            not isinstance(model, FrozenAdaptiveUtilityModel)
            for model in frozen_models
        )
        or tuple(model.outer_target for model in frozen_models) != CENTERS
    ):
        raise ProtocolError(
            "SCEPTRE full-router models must follow exact CENTERS order."
        )
    for model in frozen_models:
        menu = build_candidate_menu(generation_lock, model.outer_target)
        _validate_frozen_bindings(
            model,
            generation_lock=generation_lock,
            candidate_menu=menu,
        )
    partition_identity, fold_inventory = _validate_prelabel_partition(partition)
    bootstrap = (
        DirichletBootstrapConfig()
        if dirichlet_config is None
        else dirichlet_config
    )
    if not isinstance(bootstrap, DirichletBootstrapConfig):
        raise ProtocolError("SCEPTRE full-router bootstrap config type drifted.")
    return FrozenPrelabelRouter(
        models=frozen_models,
        generation_lock_hash=generation_lock.generation_lock_hash,
        generation_lock_payload_sha256=canonical_hash(generation_lock.to_payload()),
        bank_lock_hash=generation_lock.bank_lock_hash,
        support_minimum_bacc_gain=SUPPORT_MINIMUM_BACC_GAIN,
        calibration_minimum_bacc_gain=CALIBRATION_MINIMUM_BACC_GAIN,
        calibration_maximum_brier_delta=CALIBRATION_MAXIMUM_BRIER_DELTA,
        calibration_maximum_log_loss_delta=CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
        dirichlet_config=bootstrap,
        partition_hash=partition.partition_hash,
        partition_identity_sha256=partition_identity,
        partition_fold_inventory_sha256=fold_inventory,
        partition_namespace=PARTITION_NAMESPACE,
        partition_seed=PARTITION_SEED,
        fold_count=FOLD_COUNT,
        partition_schema=PARTITION_SCHEMA,
        case_identity_schema=CASE_IDENTITY_SCHEMA,
        claim_boundary_sha256=canonical_hash(claim_boundary_payload()),
        publication_status=PUBLICATION_STATUS,
        terminal_decision=TERMINAL_DECISION,
        claim_scope=CLAIM_SCOPE,
        descriptive_only=True,
        fresh_evidence=False,
        execution_authorized=False,
    )


def replay_full_prelabel_router(
    frozen: FrozenPrelabelRouter,
    models: Sequence[FrozenAdaptiveUtilityModel],
    *,
    generation_lock: GenerationLock,
    partition: ThreeRolePartition,
    dirichlet_config: DirichletBootstrapConfig | None = None,
) -> FullRouterReplayReceipt:
    """Rebuild the complete freeze and require canonical byte identity."""

    if not isinstance(frozen, FrozenPrelabelRouter):
        raise ProtocolError("SCEPTRE full-router replay requires a frozen router.")
    replayed = freeze_full_prelabel_router(
        models,
        generation_lock=generation_lock,
        partition=partition,
        dirichlet_config=dirichlet_config,
    )
    if replayed.to_canonical_bytes() != frozen.to_canonical_bytes():
        raise ProtocolError("SCEPTRE full prelabel router replay differs.")
    round_trip = FrozenPrelabelRouter.from_canonical_bytes(
        frozen.to_canonical_bytes()
    )
    if round_trip != frozen:
        raise ProtocolError("SCEPTRE full-router canonical round-trip differs.")
    return FullRouterReplayReceipt(
        full_router_sha256=frozen.full_router_sha256,
        generation_lock_payload_sha256=(
            frozen.generation_lock_payload_sha256
        ),
        partition_hash=frozen.partition_hash,
        partition_identity_sha256=frozen.partition_identity_sha256,
        partition_fold_inventory_sha256=(
            frozen.partition_fold_inventory_sha256
        ),
        dirichlet_config_hash=frozen.dirichlet_config.config_hash,
        claim_boundary_sha256=frozen.claim_boundary_sha256,
        model_sha256_by_target=tuple(
            (model.outer_target, model.model_sha256) for model in frozen.models
        ),
    )


__all__ = (
    "FULL_ROUTER_ROLE",
    "FULL_ROUTER_SCHEMA",
    "FrozenPrelabelRouter",
    "FrozenRouterBundle",
    "FullRouterReplayReceipt",
    "freeze_full_prelabel_router",
    "replay_full_prelabel_router",
)
