"""Frozen identities for the independent-source v3 study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)


EXPERIMENT_ID = "midogpp.cvae.aggregate_posterior_mixture_geco_source_inner.v3"
STUDY_NAME = (
    "virchow2_cvae_midogpp_aggregate_posterior_mixture_geco_source_inner_v3"
)
MODE = "aggregate_posterior_mixture_geco_source_inner_study"
STUDY_VERSION = "v3"
CLAIM_SCOPE = "cvae_source_inner_study_only"

ARMS = ("SF", "KF", "SG", "KG")
PRIMARY_ARM = "KG"
STANDARD_FIXED = "SF"
MIXTURE_FIXED = "KF"
STANDARD_GECO = "SG"
MIXTURE_GECO = "KG"

STANDARD_PRIOR = "standard_normal"
MIXTURE_PRIOR = "class_conditional_k2_diag_plus_rank2_aggregate_posterior"
FIXED_BETA = "fixed_beta_rate_distortion"
GECO = "geco_reconstruction_constrained_rate"
MIXTURE_RATE = "mixture_kl_upper_bound"
STANDARD_RATE = "analytic_standard_normal_kl"


def prior_family(arm: str) -> str:
    _validate_arm(arm)
    return MIXTURE_PRIOR if arm in {MIXTURE_FIXED, MIXTURE_GECO} else STANDARD_PRIOR


def objective_family(arm: str) -> str:
    _validate_arm(arm)
    return GECO if arm in {STANDARD_GECO, MIXTURE_GECO} else FIXED_BETA


def rate_family(arm: str) -> str:
    return MIXTURE_RATE if prior_family(arm) == MIXTURE_PRIOR else STANDARD_RATE


@dataclass(frozen=True)
class SourceExpertTrainingKey:
    """H/I-neutral identity for one independently trained source expert."""

    source_center: str
    training_seed: int
    arm: str
    source_row_hash: str
    source_case_hash: str
    source_frame_hash: str
    manifest_hash: str
    feature_cache_hash: str
    protocol_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        if str(self.source_center) not in MIDOGPP_ELIGIBLE_CENTERS:
            raise ProtocolError("Source expert center is unknown or quarantined.")
        _validate_arm(self.arm)
        for name in (
            "source_row_hash",
            "source_case_hash",
            "source_frame_hash",
            "manifest_hash",
            "feature_cache_hash",
            "protocol_hash",
            "config_hash",
        ):
            if not str(getattr(self, name)):
                raise ProtocolError(f"Source expert key lacks {name}.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_source_expert_training_key_v3",
            "source_center": str(self.source_center),
            "fit_centers": [str(self.source_center)],
            "training_seed": int(self.training_seed),
            "arm": str(self.arm),
            "prior_family": prior_family(self.arm),
            "objective_family": objective_family(self.arm),
            "rate_family": rate_family(self.arm),
            "source_row_hash": str(self.source_row_hash),
            "source_case_hash": str(self.source_case_hash),
            "source_frame_hash": str(self.source_frame_hash),
            "manifest_hash": str(self.manifest_hash),
            "feature_cache_hash": str(self.feature_cache_hash),
            "protocol_hash": str(self.protocol_hash),
            "config_hash": str(self.config_hash),
            "outer_target_identity_in_key": False,
            "inner_pseudo_target_identity_in_key": False,
        }

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    @property
    def arm_neutral_hash(self) -> str:
        payload = self.to_payload()
        for key in ("arm", "prior_family", "objective_family", "rate_family"):
            payload.pop(key)
        return stable_hash(payload)


@dataclass(frozen=True)
class SourceExpertEvaluationKey:
    """Identity for one legal H/I/source/seed/arm scoring cell."""

    outer_target_center: str
    inner_pseudo_target_center: str
    source_center: str
    training_seed: int
    generation_seed: int
    arm: str
    representation_role: str
    generation_noise_hash: str
    posterior_source_index_hash: str | None
    training_key_hash: str
    inner_eval_row_hash: str
    classifier_spec_hash: str
    protocol_hash: str

    def __post_init__(self) -> None:
        outer = str(self.outer_target_center)
        inner = str(self.inner_pseudo_target_center)
        source = str(self.source_center)
        if any(
            value not in MIDOGPP_ELIGIBLE_CENTERS
            for value in (outer, inner, source)
        ):
            raise ProtocolError("Evaluation key contains an ineligible center.")
        if len({outer, inner, source}) != 3:
            raise ProtocolError("Evaluation requires distinct H, I, and source E.")
        _validate_arm(self.arm)
        if self.representation_role not in {"prior", "posterior"}:
            raise ProtocolError(
                "Evaluation representation_role must be prior or posterior."
            )
        if not str(self.generation_noise_hash):
            raise ProtocolError("Evaluation key lacks generation noise identity.")
        if self.representation_role == "posterior":
            if not str(self.posterior_source_index_hash or ""):
                raise ProtocolError(
                    "Posterior evaluation key lacks source-row selection identity."
                )
        elif self.posterior_source_index_hash is not None:
            raise ProtocolError(
                "Prior evaluation key must not claim posterior source rows."
            )
        if not all(
            str(value)
            for value in (
                self.training_key_hash,
                self.inner_eval_row_hash,
                self.classifier_spec_hash,
                self.protocol_hash,
            )
        ):
            raise ProtocolError("Evaluation key lacks a content identity.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_source_expert_evaluation_key_v3",
            "outer_target_center": str(self.outer_target_center),
            "inner_pseudo_target_center": str(self.inner_pseudo_target_center),
            "source_center": str(self.source_center),
            "training_seed": int(self.training_seed),
            "generation_seed": int(self.generation_seed),
            "arm": str(self.arm),
            "representation_role": str(self.representation_role),
            "generation_noise_hash": str(self.generation_noise_hash),
            "posterior_source_index_hash": self.posterior_source_index_hash,
            "training_key_hash": str(self.training_key_hash),
            "inner_eval_row_hash": str(self.inner_eval_row_hash),
            "classifier_spec_hash": str(self.classifier_spec_hash),
            "protocol_hash": str(self.protocol_hash),
            "outer_rows_used": False,
            "inner_rows_used_for_model_or_classifier_fit": False,
            "inner_labels_used_for_scoring_only": True,
        }

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())


def arm_contract() -> Mapping[str, Mapping[str, str]]:
    return {
        arm: {
            "prior_family": prior_family(arm),
            "objective_family": objective_family(arm),
            "rate_family": rate_family(arm),
        }
        for arm in ARMS
    }


def _validate_arm(arm: str) -> None:
    if str(arm) not in ARMS:
        raise ProtocolError(f"Unknown v3 arm: {arm!r}.")
