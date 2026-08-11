"""Fitted-model and development-selection contracts for disagreement regret."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ...protocol import ProtocolError
from ._validation import _canonical_id
from .hashing import canonical_sha256, is_sha256
from .probability_contracts import (
    DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    FEATURE_NAMES,
    LABEL_FREE_INFERENCE_SURFACE_ROLE,
    SOURCE_OOF_TRAINING_SURFACE_ROLE,
)
from .inference_contracts import LABEL_FREE_INFERENCE_CLAIM_ROLE


SCORE_SEMANTICS = "exact_regret_spread_weighted_pairwise_logit_preference_margin"
DEVELOPMENT_CLAIM_ROLE = "non_runnable_development_diagnostic_only"


@dataclass(frozen=True)
class PairwiseRegretModel:
    family: str
    outer_target_id: str
    candidate_action_id: str
    candidate_source_id: str
    heldout_query_id: str | None
    action_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    training_query_ids: tuple[str, ...]
    excluded_query_ids: tuple[str, ...]
    observation_count: int
    converged: bool
    iteration_count: int
    feature_surface_hash: str
    response_surface_hash: str
    prediction_seal_hash: str
    development_context_hash: str
    baseline_action_id: str
    control_action_id: str
    candidate_source_by_action: tuple[tuple[str, str], ...]
    training_feature_hash: str
    training_response_hash: str
    shared_l2_penalty: float
    action_l2_penalty: float
    max_newton_iterations: int
    gradient_tolerance: float
    training_scope: str
    training_surface_role: str
    source_history_mode: str = "known_bank"
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Pairwise model family must be G, R, or P.")
        if self.source_history_mode != "known_bank":
            raise ProtocolError("This core implements known-bank history, not unseen transfer.")
        if self.training_scope not in (
            "SYNTHETIC_TEST",
            "AUTHORIZED_SOURCE_OOF",
            "AUTHORIZED_POSTHOC_SOURCE_OOF",
        ):
            raise ProtocolError("Pairwise model training scope drifted.")
        if self.training_surface_role not in (
            DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
            SOURCE_OOF_TRAINING_SURFACE_ROLE,
        ):
            raise ProtocolError("Pairwise models cannot be fitted from inference surfaces.")
        if (
            self.training_surface_role == SOURCE_OOF_TRAINING_SURFACE_ROLE
            and self.training_scope
            not in ("AUTHORIZED_SOURCE_OOF", "AUTHORIZED_POSTHOC_SOURCE_OOF")
        ):
            raise ProtocolError("Source-only model lineage requires authorized source OOF.")
        if self.training_surface_role == LABEL_FREE_INFERENCE_SURFACE_ROLE:
            raise ProtocolError("Inference surface lineage cannot enter a fitted model.")
        for name in (
            "outer_target_id",
            "candidate_action_id",
            "candidate_source_id",
            "baseline_action_id",
            "control_action_id",
        ):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if self.heldout_query_id is not None:
            object.__setattr__(
                self,
                "heldout_query_id",
                _canonical_id(self.heldout_query_id, name="heldout_query_id"),
            )
        if self.action_ids != tuple(sorted(self.action_ids)):
            raise ProtocolError("Pairwise model actions must use canonical ordering.")
        if self.training_query_ids != tuple(sorted(set(self.training_query_ids))):
            raise ProtocolError("Pairwise model training queries must be canonical and unique.")
        if self.excluded_query_ids != tuple(sorted(set(self.excluded_query_ids))):
            raise ProtocolError("Pairwise model exclusions must be canonical and unique.")
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ProtocolError("Pairwise model observation_count must be positive.")
        if type(self.iteration_count) is not int or self.iteration_count <= 0:
            raise ProtocolError("Pairwise model iteration_count must be positive.")
        if not self.converged:
            raise ProtocolError("Nonconverged pairwise models are not admissible.")
        dimension = len(self.feature_names) + len(self.action_ids) + len(self.action_ids) * len(
            self.feature_names
        )
        if self.feature_names != FEATURE_NAMES:
            raise ProtocolError("Pairwise model feature schema drifted.")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ProtocolError("Pairwise model actions must be unique.")
        if self.feature_mean.shape != (len(self.feature_names),) or self.feature_scale.shape != (
            len(self.feature_names),
        ) or self.coefficients.shape != (dimension,):
            raise ProtocolError("Pairwise model array dimensions drifted.")
        if self.coefficient_covariance.shape != (dimension, dimension):
            raise ProtocolError("Pairwise model covariance dimension drifted.")
        arrays = (
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            self.coefficient_covariance,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ProtocolError("Pairwise model arrays must be finite.")
        if np.any(self.feature_scale <= 0.0):
            raise ProtocolError("Pairwise model feature scales must be positive.")
        if not np.allclose(
            self.coefficient_covariance,
            self.coefficient_covariance.T,
            rtol=1.0e-10,
            atol=1.0e-12,
        ) or float(np.linalg.eigvalsh(self.coefficient_covariance).min()) < -1.0e-9:
            raise ProtocolError("Pairwise model covariance must be symmetric PSD.")
        lineage_hashes = (
            self.feature_surface_hash,
            self.response_surface_hash,
            self.prediction_seal_hash,
            self.development_context_hash,
            self.training_feature_hash,
            self.training_response_hash,
        )
        if not all(is_sha256(value) for value in lineage_hashes):
            raise ProtocolError("Pairwise model lineage requires full SHA-256 identities.")
        mapping = tuple(
            sorted(
                (str(action), str(source))
                for action, source in self.candidate_source_by_action
            )
        )
        if (
            len(mapping) != len(set(mapping))
            or len({action for action, _source in mapping}) != len(mapping)
            or len({source for _action, source in mapping}) != len(mapping)
        ):
            raise ProtocolError("Pairwise model candidate-source mapping is invalid.")
        object.__setattr__(self, "candidate_source_by_action", mapping)
        if set(self.training_query_ids).intersection(self.excluded_query_ids):
            raise ProtocolError("Pairwise model training and exclusion queries overlap.")
        if self.baseline_action_id == self.control_action_id:
            raise ProtocolError("Pairwise model requires distinct B/U identities.")
        if self.baseline_action_id not in self.action_ids or self.control_action_id in self.action_ids:
            raise ProtocolError("Pairwise model design must contain B and fix U at the origin.")
        mapping_dict = dict(mapping)
        if mapping_dict.get(self.candidate_action_id) != self.candidate_source_id:
            raise ProtocolError("Pairwise model candidate identity drifted from its source map.")
        if not math.isfinite(self.shared_l2_penalty) or self.shared_l2_penalty <= 0.0:
            raise ProtocolError("Shared L2 penalty must be finite and positive.")
        if not math.isfinite(self.action_l2_penalty) or self.action_l2_penalty <= 0.0:
            raise ProtocolError("Action L2 penalty must be finite and positive.")
        if type(self.max_newton_iterations) is not int or self.max_newton_iterations <= 0:
            raise ProtocolError("Newton iteration cap must be a positive integer.")
        if not math.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.0:
            raise ProtocolError("Gradient tolerance must be finite and positive.")
        for value in arrays:
            value.setflags(write=False)
        object.__setattr__(self, "model_hash", canonical_sha256(self._unhashed_payload()))

    def _unhashed_payload(self) -> dict[str, object]:
        """Return the canonical, hash-bearing model state without its digest."""

        return {
            "schema_version": "midogpp_pairwise_disagreement_regret_model_v1",
            "family": self.family,
            "source_history_mode": self.source_history_mode,
            "training_scope": self.training_scope,
            "training_surface_role": self.training_surface_role,
            "outer_target_id": self.outer_target_id,
            "candidate_action_id": self.candidate_action_id,
            "candidate_source_id": self.candidate_source_id,
            "heldout_query_id": self.heldout_query_id,
            "action_ids": list(self.action_ids),
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "coefficient_covariance": self.coefficient_covariance.tolist(),
            "training_query_ids": list(self.training_query_ids),
            "excluded_query_ids": list(self.excluded_query_ids),
            "observation_count": self.observation_count,
            "converged": True,
            "iteration_count": self.iteration_count,
            "feature_surface_hash": self.feature_surface_hash,
            "response_surface_hash": self.response_surface_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "development_context_hash": self.development_context_hash,
            "baseline_action_id": self.baseline_action_id,
            "control_action_id": self.control_action_id,
            "candidate_source_by_action": [
                list(row) for row in self.candidate_source_by_action
            ],
            "training_feature_hash": self.training_feature_hash,
            "training_response_hash": self.training_response_hash,
            "hyperparameters": {
                "shared_l2_penalty": self.shared_l2_penalty,
                "action_l2_penalty": self.action_l2_penalty,
                "max_newton_iterations": self.max_newton_iterations,
                "gradient_tolerance": self.gradient_tolerance,
            },
        }

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible payload whose identity is self-verifying."""

        payload = self._unhashed_payload()
        payload["model_hash"] = self.model_hash
        return payload


@dataclass(frozen=True)
class CandidateContrastRow:
    family: str
    target_query_id: str
    case_id: str
    candidate_action_id: str
    candidate_source_id: str
    predicted_preference_margin_vs_control: float
    standard_error_vs_control: float
    predicted_preference_margin_vs_baseline: float
    standard_error_vs_baseline: float
    model_hash: str
    score_semantics: str = SCORE_SEMANTICS

    def __post_init__(self) -> None:
        values = (
            self.predicted_preference_margin_vs_control,
            self.standard_error_vs_control,
            self.predicted_preference_margin_vs_baseline,
            self.standard_error_vs_baseline,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ProtocolError("Candidate contrast scores must be finite.")
        if self.standard_error_vs_control < 0.0 or self.standard_error_vs_baseline < 0.0:
            raise ProtocolError("Candidate contrast standard errors cannot be negative.")
        if self.score_semantics != SCORE_SEMANTICS:
            raise ProtocolError("Candidate contrasts must remain pairwise preference margins.")
        if not is_sha256(self.model_hash):
            raise ProtocolError("Candidate contrast model hash must be SHA-256.")

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return (
            self.family,
            self.target_query_id,
            self.case_id,
            self.candidate_action_id,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "target_query_id": self.target_query_id,
            "case_id": self.case_id,
            "candidate_action_id": self.candidate_action_id,
            "candidate_source_id": self.candidate_source_id,
            "predicted_preference_margin_vs_control": (
                self.predicted_preference_margin_vs_control
            ),
            "standard_error_vs_control": self.standard_error_vs_control,
            "predicted_preference_margin_vs_baseline": (
                self.predicted_preference_margin_vs_baseline
            ),
            "standard_error_vs_baseline": self.standard_error_vs_baseline,
            "model_hash": self.model_hash,
            "score_semantics": self.score_semantics,
        }

    @property
    def row_hash(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class DevelopmentSelectionDiagnostic:
    family: str
    target_query_id: str
    case_id: str
    raw_action_id: str
    safe_action_id: str
    baseline_action_id: str
    control_action_id: str
    simultaneous_z_value: float
    safe_margin: float
    fallback_reason: str
    claim_role: str = DEVELOPMENT_CLAIM_ROLE
    may_authorize_routing: bool = False
    may_authorize_promotion: bool = False

    def __post_init__(self) -> None:
        if self.claim_role != DEVELOPMENT_CLAIM_ROLE:
            raise ProtocolError("Development selections cannot carry a routing claim.")
        if self.may_authorize_routing or self.may_authorize_promotion:
            raise ProtocolError("Development selections cannot authorize routing or promotion.")
        if not math.isfinite(float(self.simultaneous_z_value)) or not math.isfinite(
            float(self.safe_margin)
        ):
            raise ProtocolError("Development selection diagnostics must be finite.")


@dataclass(frozen=True)
class InferenceSelectionDiagnostic:
    """An unscored consumed-target suggestion, never a deployable decision."""

    family: str
    target_query_id: str
    case_id: str
    raw_action_id: str
    safe_action_id: str
    baseline_action_id: str
    control_action_id: str
    simultaneous_z_value: float
    safe_margin: float
    fallback_reason: str
    claim_role: str = LABEL_FREE_INFERENCE_CLAIM_ROLE
    consumed_target_data: bool = True
    target_labels_accessed: bool = False
    fresh_evidence: bool = False
    may_authorize_routing: bool = False
    may_authorize_promotion: bool = False
    may_feed_another_experiment: bool = False

    def __post_init__(self) -> None:
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Inference selection family must be G, R, or P.")
        if self.claim_role != LABEL_FREE_INFERENCE_CLAIM_ROLE:
            raise ProtocolError("Inference selections must keep the terminal claim role.")
        if (
            self.consumed_target_data is not True
            or self.target_labels_accessed is not False
            or self.fresh_evidence is not False
            or self.may_authorize_routing is not False
            or self.may_authorize_promotion is not False
            or self.may_feed_another_experiment is not False
        ):
            raise ProtocolError("Inference selections escaped the consumed-data boundary.")
        if not math.isfinite(float(self.simultaneous_z_value)) or not math.isfinite(
            float(self.safe_margin)
        ):
            raise ProtocolError("Inference selection diagnostics must be finite.")

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "target_query_id": self.target_query_id,
            "case_id": self.case_id,
            "raw_action_id": self.raw_action_id,
            "safe_action_id": self.safe_action_id,
            "baseline_action_id": self.baseline_action_id,
            "control_action_id": self.control_action_id,
            "simultaneous_z_value": self.simultaneous_z_value,
            "safe_margin": self.safe_margin,
            "fallback_reason": self.fallback_reason,
            "claim_role": self.claim_role,
            "consumed_target_data": self.consumed_target_data,
            "target_labels_accessed": self.target_labels_accessed,
            "fresh_evidence": self.fresh_evidence,
            "may_authorize_routing": self.may_authorize_routing,
            "may_authorize_promotion": self.may_authorize_promotion,
            "may_feed_another_experiment": self.may_feed_another_experiment,
        }

    @property
    def row_hash(self) -> str:
        return canonical_sha256(self.to_payload())


__all__ = (
    "DEVELOPMENT_CLAIM_ROLE",
    "SCORE_SEMANTICS",
    "CandidateContrastRow",
    "DevelopmentSelectionDiagnostic",
    "PairwiseRegretModel",
)
