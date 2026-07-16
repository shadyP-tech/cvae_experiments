"""Independent contracts for the additive Stage-20 source-inner v2 studies.

These schemas deliberately do not inherit from the v1 prior-recovery
``RecipeLock`` or checkpoint contracts.  The studies can report mechanism and
stability evidence, but their decisions can never become a model recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from ...objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE


LEARNED_PRIOR_MODE = "learned_conditional_prior_source_inner_study"
FISHER_SHRINKAGE_MODE = "task_fisher_shrinkage_source_inner_study"
SOURCE_INNER_STUDY_VERSION = "v2"

PRIOR_ARMS = ("A", "C-diag", "E")
FISHER_ALPHAS = (0.0, 0.05, 0.10, 0.25)

PRIOR_DECISION_SCHEMA = "midogpp_learned_conditional_prior_study_decision_v2"
FISHER_DECISION_SCHEMA = "midogpp_task_fisher_shrinkage_study_decision_v2"
STUDY_TRAINING_VARIANT_SCHEMA = "midogpp_source_inner_study_training_variant_v2"
STUDY_TRAINING_KEY_SCHEMA = "midogpp_source_inner_study_training_key_v2"

STANDARD_MODEL_FAMILY = "class_conditioned_cvae_v1"
LEARNED_PRIOR_MODEL_FAMILY = (
    "class_conditioned_cvae_learned_conditional_diagonal_prior_v2"
)
STANDARD_NORMAL_PRIOR = "standard_normal"
LEARNED_CONDITIONAL_DIAGONAL_PRIOR = (
    "learned_class_conditional_diagonal_gaussian"
)

_SUPPORTED_MODES = {LEARNED_PRIOR_MODE, FISHER_SHRINKAGE_MODE}


@dataclass(frozen=True)
class StudyTrainingVariant:
    """One exact v2 model/objective variant.

    ``arm_neutral_pairing_payload`` removes only the scientific axis varied by
    the containing study.  In particular, it retains the study identity,
    version, raw Fisher state, optimizer, architecture, and training seed (the
    latter lives in :class:`StudyTrainingKey`).
    """

    study_mode: str
    study_version: str
    model_family: str
    prior_family: str
    objective_id: str
    alpha: float
    raw_fisher_state_hash: str
    objective_context_hash: str
    hidden_dim: int = 512
    latent_dim: int = 32
    num_hidden_layers: int = 2
    train_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta_final: float = 1e-3
    kl_warmup_epochs: int = 25
    network_gradient_clip_norm: float = 5.0
    prior_learning_rate_multiplier: float = 1.0
    prior_weight_decay: float = 0.0
    prior_gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        if self.study_mode not in _SUPPORTED_MODES:
            raise ProtocolError(f"Unsupported source-inner study mode: {self.study_mode!r}")
        if self.study_version != SOURCE_INNER_STUDY_VERSION:
            raise ProtocolError("Stage-20 source-inner study version must be 'v2'.")
        if self.objective_id not in {ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE}:
            raise ProtocolError(f"Unsupported study objective: {self.objective_id!r}")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) < 0.0:
            raise ProtocolError("Study alpha must be finite and nonnegative.")
        if self.study_mode == LEARNED_PRIOR_MODE:
            if float(self.alpha) != 0.0 or self.objective_id != ISOTROPIC_OBJECTIVE:
                raise ProtocolError("The learned-prior study fixes the isotropic objective.")
            if self.raw_fisher_state_hash != "none":
                raise ProtocolError("The learned-prior study cannot carry a raw Fisher state.")
        else:
            if self.prior_family != STANDARD_NORMAL_PRIOR:
                raise ProtocolError("The Fisher-shrinkage study fixes the standard-normal prior.")
            if float(self.alpha) == 0.0:
                if self.objective_id != ISOTROPIC_OBJECTIVE:
                    raise ProtocolError("alpha=0 must use the literal isotropic objective.")
                if self.objective_context_hash != "none":
                    raise ProtocolError("alpha=0 must use objective_context_hash='none'.")
            else:
                if self.objective_id != TASK_FISHER_OBJECTIVE:
                    raise ProtocolError("Nonzero alpha must use the Task-Fisher objective.")
                if self.raw_fisher_state_hash == "none":
                    raise ProtocolError("Nonzero alpha requires a raw Fisher state hash.")
                if self.objective_context_hash == "none":
                    raise ProtocolError("Nonzero alpha requires a derived objective context hash.")
        if not self.model_family or not self.prior_family:
            raise ProtocolError("Study model and prior families must be explicit.")
        if not self.raw_fisher_state_hash or not self.objective_context_hash:
            raise ProtocolError("Study objective identities must be explicit; use 'none' when absent.")
        if min(
            self.hidden_dim,
            self.latent_dim,
            self.num_hidden_layers,
            self.train_epochs,
            self.batch_size,
            self.kl_warmup_epochs,
        ) <= 0:
            raise ProtocolError("Study architecture and training counts must be positive.")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.beta_final < 0.0
            or self.network_gradient_clip_norm <= 0.0
            or self.prior_learning_rate_multiplier <= 0.0
            or self.prior_weight_decay < 0.0
            or self.prior_gradient_clip_norm <= 0.0
        ):
            raise ProtocolError("Study optimizer/objective scalars are invalid.")
        if (
            self.prior_learning_rate_multiplier != 1.0
            or self.prior_weight_decay != 0.0
            or self.prior_gradient_clip_norm != 5.0
        ):
            raise ProtocolError("Study learned-prior optimizer group contract drifted.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": STUDY_TRAINING_VARIANT_SCHEMA,
            "study_mode": self.study_mode,
            "study_version": self.study_version,
            "model_family": self.model_family,
            "prior_family": self.prior_family,
            "objective_id": self.objective_id,
            "alpha": float(self.alpha),
            "raw_fisher_state_hash": self.raw_fisher_state_hash,
            "objective_context_hash": self.objective_context_hash,
            "hidden_dim": int(self.hidden_dim),
            "latent_dim": int(self.latent_dim),
            "num_hidden_layers": int(self.num_hidden_layers),
            "train_epochs": int(self.train_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "beta_final": float(self.beta_final),
            "kl_warmup_epochs": int(self.kl_warmup_epochs),
            "network_gradient_clip_norm": float(self.network_gradient_clip_norm),
            "prior_learning_rate_multiplier": float(
                self.prior_learning_rate_multiplier
            ),
            "prior_weight_decay": float(self.prior_weight_decay),
            "prior_gradient_clip_norm": float(self.prior_gradient_clip_norm),
        }

    def arm_neutral_pairing_payload(self) -> dict[str, object]:
        payload = self.to_payload()
        payload["schema_version"] = (
            "midogpp_source_inner_study_training_pairing_variant_v2"
        )
        if self.study_mode == LEARNED_PRIOR_MODE:
            # A and E share the base-network initialization and stochastic
            # streams; only the model/prior mechanism differs.
            payload.pop("model_family")
            payload.pop("prior_family")
        else:
            # The alpha panel shares the model/prior and all training
            # identities; only its Fisher/objective axis is varied. alpha=0 is
            # the literal metric=None path and therefore has no raw-F identity.
            payload.pop("objective_id")
            payload.pop("alpha")
            payload.pop("raw_fisher_state_hash")
            payload.pop("objective_context_hash")
        return payload


@dataclass(frozen=True)
class StudyTrainingKey:
    """Content-addressed identity for one ``(H, I, t)`` model fit."""

    study_id: str
    study_version: str
    outer_target_center: str
    inner_pseudo_target_center: str
    fit_centers: tuple[str, ...]
    fit_row_hash: str
    frame_hash: str
    feature_cache_hash: str
    manifest_hash: str
    protocol_hash: str
    training_seed: int
    variant: StudyTrainingVariant

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ProtocolError("StudyTrainingKey requires a study_id.")
        if self.study_version != self.variant.study_version:
            raise ProtocolError("Training-key and variant study versions differ.")
        if self.study_version != SOURCE_INNER_STUDY_VERSION:
            raise ProtocolError("StudyTrainingKey version must be 'v2'.")
        if self.outer_target_center == self.inner_pseudo_target_center:
            raise ProtocolError("Outer and inner held-out centers must differ.")
        if (
            self.outer_target_center not in MIDOGPP_ELIGIBLE_CENTERS
            or self.inner_pseudo_target_center not in MIDOGPP_ELIGIBLE_CENTERS
        ):
            raise ProtocolError("Training-key H/I centers must be MIDOG++ eligible.")
        expected_fit_centers = tuple(
            center
            for center in MIDOGPP_ELIGIBLE_CENTERS
            if center not in {self.outer_target_center, self.inner_pseudo_target_center}
        )
        if self.fit_centers != expected_fit_centers:
            raise ProtocolError("Training-key fit centers must be the exact ordered H/I-excluded set.")
        if int(self.training_seed) not in {17, 42, 101}:
            raise ProtocolError("Training-key seed must belong to the frozen 17/42/101 panel.")
        identities = (
            self.fit_row_hash,
            self.frame_hash,
            self.feature_cache_hash,
            self.manifest_hash,
            self.protocol_hash,
        )
        if any(not value for value in identities):
            raise ProtocolError("Training-key provenance hashes must be nonempty.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    @property
    def variant_hash(self) -> str:
        return self.variant.hash

    @property
    def model_family(self) -> str:
        return self.variant.model_family

    @property
    def prior_family(self) -> str:
        return self.variant.prior_family

    @property
    def objective_id(self) -> str:
        return self.variant.objective_id

    @property
    def alpha(self) -> float:
        return self.variant.alpha

    @property
    def raw_fisher_state_hash(self) -> str:
        return self.variant.raw_fisher_state_hash

    @property
    def objective_context_hash(self) -> str:
        return self.variant.objective_context_hash

    @property
    def arm_neutral_pairing_hash(self) -> str:
        return stable_hash(self.arm_neutral_pairing_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": STUDY_TRAINING_KEY_SCHEMA,
            "study_id": self.study_id,
            "study_mode": self.variant.study_mode,
            "study_version": self.study_version,
            "outer_target_center": self.outer_target_center,
            "inner_pseudo_target_center": self.inner_pseudo_target_center,
            "fit_centers": list(self.fit_centers),
            "fit_row_hash": self.fit_row_hash,
            "frame_hash": self.frame_hash,
            "feature_cache_hash": self.feature_cache_hash,
            "manifest_hash": self.manifest_hash,
            "protocol_hash": self.protocol_hash,
            "training_seed": int(self.training_seed),
            "model_family": self.variant.model_family,
            "prior_family": self.variant.prior_family,
            "objective_id": self.variant.objective_id,
            "alpha": float(self.variant.alpha),
            "raw_fisher_state_hash": self.variant.raw_fisher_state_hash,
            "objective_context_hash": self.variant.objective_context_hash,
            "variant": self.variant.to_payload(),
            "variant_hash": self.variant.hash,
        }

    def arm_neutral_pairing_payload(self) -> dict[str, object]:
        payload = self.to_payload()
        payload["schema_version"] = (
            "midogpp_source_inner_study_arm_neutral_pairing_key_v2"
        )
        payload["variant"] = self.variant.arm_neutral_pairing_payload()
        payload["variant_hash"] = stable_hash(payload["variant"])
        if self.variant.study_mode == LEARNED_PRIOR_MODE:
            payload.pop("model_family")
            payload.pop("prior_family")
        else:
            payload.pop("objective_id")
            payload.pop("alpha")
            payload.pop("raw_fisher_state_hash")
            payload.pop("objective_context_hash")
        return payload


@dataclass(frozen=True)
class PriorStudyMetricV2:
    outer_target_center: str
    inner_pseudo_target_center: str
    training_seed: int
    generation_seed: int
    arm: str
    preservation_ratio: float
    decode_bacc: float
    posterior_bacc: float
    valid: bool = True
    eligible: bool = True
    ineligibility_reason: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target_center": self.outer_target_center,
            "inner_pseudo_target_center": self.inner_pseudo_target_center,
            "training_seed": int(self.training_seed),
            "generation_seed": int(self.generation_seed),
            "arm": self.arm,
            "preservation_ratio": float(self.preservation_ratio),
            "decode_bacc": float(self.decode_bacc),
            "posterior_bacc": float(self.posterior_bacc),
            "valid": bool(self.valid),
            "eligible": bool(self.eligible),
            "ineligibility_reason": self.ineligibility_reason,
        }


@dataclass(frozen=True)
class FisherStudyMetricV2:
    outer_target_center: str
    inner_pseudo_target_center: str
    training_seed: int
    generation_seed: int
    alpha: float
    preservation_ratio: float
    decode_bacc: float
    posterior_bacc: float
    valid: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target_center": self.outer_target_center,
            "inner_pseudo_target_center": self.inner_pseudo_target_center,
            "training_seed": int(self.training_seed),
            "generation_seed": int(self.generation_seed),
            "alpha": float(self.alpha),
            "preservation_ratio": float(self.preservation_ratio),
            "decode_bacc": float(self.decode_bacc),
            "posterior_bacc": float(self.posterior_bacc),
            "valid": bool(self.valid),
        }


@dataclass(frozen=True)
class PriorStudyDecisionV2:
    outer_target_center: str
    status: str
    e_vs_a_consensus_status: str
    e_vs_c_consensus_status: str
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    inner_centers: tuple[str, ...]
    per_training_seed: Mapping[str, object]
    protocol_hash: str
    decision_contract_hash: str
    source_metric_table_hash: str
    reason: str = ""

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload(include_hash=False))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": PRIOR_DECISION_SCHEMA,
            "outer_target_center": self.outer_target_center,
            "status": self.status,
            "e_vs_a_consensus_status": self.e_vs_a_consensus_status,
            "e_vs_c_consensus_status": self.e_vs_c_consensus_status,
            "training_seeds": list(self.training_seeds),
            "generation_seeds": list(self.generation_seeds),
            "inner_centers": list(self.inner_centers),
            "per_training_seed": dict(self.per_training_seed),
            "protocol_hash": self.protocol_hash,
            "decision_contract_hash": self.decision_contract_hash,
            "source_metric_table_hash": self.source_metric_table_hash,
            "claim_scope": "cvae_source_inner_study_only",
            "selection_source": "fully_nested_source_inner",
            "target_eval_labels_used_for_selection": False,
            "may_feed_model_recipe": False,
            "may_feed_deployable_selection": False,
            "routing_performed": False,
            "composition_performed": False,
            "reason": self.reason,
        }
        if include_hash:
            payload["study_decision_hash"] = self.hash
        return payload


@dataclass(frozen=True)
class FisherStudyDecisionV2:
    outer_target_center: str
    status: str
    selected_alpha: float | None
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    inner_centers: tuple[str, ...]
    per_training_seed: Mapping[str, object]
    protocol_hash: str
    decision_contract_hash: str
    source_metric_table_hash: str
    reason: str = ""

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload(include_hash=False))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": FISHER_DECISION_SCHEMA,
            "outer_target_center": self.outer_target_center,
            "status": self.status,
            "selected_alpha": self.selected_alpha,
            "training_seeds": list(self.training_seeds),
            "generation_seeds": list(self.generation_seeds),
            "inner_centers": list(self.inner_centers),
            "per_training_seed": dict(self.per_training_seed),
            "protocol_hash": self.protocol_hash,
            "decision_contract_hash": self.decision_contract_hash,
            "source_metric_table_hash": self.source_metric_table_hash,
            "claim_scope": "cvae_source_inner_study_only",
            "selection_source": "fully_nested_source_inner",
            "target_eval_labels_used_for_selection": False,
            "may_feed_model_recipe": False,
            "may_feed_deployable_selection": False,
            "routing_performed": False,
            "composition_performed": False,
            "reason": self.reason,
        }
        if include_hash:
            payload["study_decision_hash"] = self.hash
        return payload


def metric_is_finite(*values: float) -> bool:
    """Return whether every decision metric is a finite real number."""

    return all(math.isfinite(float(value)) for value in values)
