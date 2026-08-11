"""Fail-closed contracts for terminal label-free consumed-target inference.

Training authority remains represented exclusively by :mod:`provenance`.  This
module provides a separate type for applying an already frozen model bank to a
separately sealed target cache without making target labels or fresh-evidence
claims representable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_sha256, is_sha256
from .probability_contracts import FEATURE_NAMES


LABEL_FREE_INFERENCE_CLAIM_ROLE = (
    "exploratory_consumed_data_prediction_only_terminal_diagnostic"
)


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a nonempty canonical string.")
    return value


@dataclass(frozen=True, kw_only=True)
class InferenceActionSchema:
    """The complete action/feature identity a frozen bank is allowed to score."""

    family: str
    baseline_action_id: str
    control_action_id: str
    candidate_source_by_action: Mapping[str, str] | tuple[tuple[str, str], ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Inference action family must be G, R, or P.")
        baseline = _canonical_text(self.baseline_action_id, name="baseline_action_id")
        control = _canonical_text(self.control_action_id, name="control_action_id")
        if baseline == control:
            raise ProtocolError("Inference action schema requires distinct B/U identities.")
        raw_mapping = (
            tuple(self.candidate_source_by_action.items())
            if isinstance(self.candidate_source_by_action, Mapping)
            else tuple(self.candidate_source_by_action)
        )
        try:
            mapping = tuple(
                sorted(
                    (
                        _canonical_text(action_id, name="candidate action"),
                        _canonical_text(source_id, name="candidate source"),
                    )
                    for action_id, source_id in raw_mapping
                )
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Inference candidate mapping must contain pairs.") from exc
        if (
            not mapping
            or len(mapping) != len(set(mapping))
            or len({action for action, _source in mapping}) != len(mapping)
            or len({source for _action, source in mapping}) != len(mapping)
            or baseline in {action for action, _source in mapping}
            or control in {action for action, _source in mapping}
        ):
            raise ProtocolError("Inference candidate action/source mapping is invalid.")
        names = tuple(self.feature_names)
        if names != FEATURE_NAMES:
            raise ProtocolError("Inference feature schema drifted.")
        object.__setattr__(self, "baseline_action_id", baseline)
        object.__setattr__(self, "control_action_id", control)
        object.__setattr__(self, "candidate_source_by_action", mapping)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "schema_hash", canonical_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_disagreement_regret_inference_action_schema_v1",
            "family": self.family,
            "baseline_action_id": self.baseline_action_id,
            "control_action_id": self.control_action_id,
            "candidate_source_by_action": [
                list(row) for row in self.candidate_source_by_action
            ],
            "feature_names": list(self.feature_names),
        }

    @property
    def candidate_mapping(self) -> dict[str, str]:
        return dict(self.candidate_source_by_action)


@dataclass(frozen=True, kw_only=True)
class LabelFreeInferenceContext:
    """Authority for terminal predictions on an already consumed target cache.

    Every claim-boundary flag is deliberately explicit and fail-closed.  This
    context cannot authorize evaluation, routing, promotion, or a downstream
    experiment; it only binds a frozen model bank to a label-free cache and
    prediction surface.
    """

    dataset_family: str
    outer_target_id: str
    target_cache_content_hash: str
    target_cache_order_hash: str
    prediction_seal_hash: str
    action_schema: InferenceActionSchema
    model_bank_hash: str
    consumed_target_data: bool = True
    target_labels_accessed: bool = False
    fresh_evidence: bool = False
    terminal_diagnostic_only: bool = True
    may_feed_another_experiment: bool = False
    may_authorize_routing: bool = False
    may_authorize_promotion: bool = False
    claim_role: str = LABEL_FREE_INFERENCE_CLAIM_ROLE
    context_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_family", _canonical_text(self.dataset_family, name="dataset_family")
        )
        object.__setattr__(
            self, "outer_target_id", _canonical_text(self.outer_target_id, name="outer_target_id")
        )
        assert_label_free_inference_context(self)
        object.__setattr__(self, "context_hash", canonical_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        assert_label_free_inference_context(self)
        return {
            "schema_version": "midogpp_disagreement_regret_label_free_inference_context_v1",
            "dataset_family": self.dataset_family,
            "outer_target_id": self.outer_target_id,
            "target_cache_content_hash": self.target_cache_content_hash,
            "target_cache_order_hash": self.target_cache_order_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "action_schema": self.action_schema.to_payload(),
            "action_schema_hash": self.action_schema.schema_hash,
            "model_bank_hash": self.model_bank_hash,
            "consumed_target_data": True,
            "target_labels_accessed": False,
            "fresh_evidence": False,
            "terminal_diagnostic_only": True,
            "may_feed_another_experiment": False,
            "may_authorize_routing": False,
            "may_authorize_promotion": False,
            "claim_role": LABEL_FREE_INFERENCE_CLAIM_ROLE,
        }


def assert_label_free_inference_context(
    context: LabelFreeInferenceContext,
) -> LabelFreeInferenceContext:
    if not isinstance(context, LabelFreeInferenceContext):
        raise ProtocolError("Inference context must use the locked label-free type.")
    if not isinstance(context.action_schema, InferenceActionSchema):
        raise ProtocolError("Inference context requires a typed action schema.")
    for name in (
        "target_cache_content_hash",
        "target_cache_order_hash",
        "prediction_seal_hash",
        "model_bank_hash",
    ):
        if not is_sha256(getattr(context, name)):
            raise ProtocolError(f"{name} must be a lowercase SHA-256 identity.")
    if context.outer_target_id in {
        source_id for _action_id, source_id in context.action_schema.candidate_source_by_action
    }:
        raise ProtocolError("The target expert cannot appear in the inference action schema.")
    if context.consumed_target_data is not True:
        raise ProtocolError("This context is restricted to explicitly consumed target data.")
    if context.target_labels_accessed is not False:
        raise ProtocolError("Target labels must remain unaccessed during inference.")
    if context.fresh_evidence is not False:
        raise ProtocolError("Consumed target predictions cannot be fresh evidence.")
    if context.terminal_diagnostic_only is not True:
        raise ProtocolError("Consumed target predictions must remain terminal diagnostics.")
    if context.may_feed_another_experiment is not False:
        raise ProtocolError("Consumed target predictions cannot feed another experiment.")
    if context.may_authorize_routing is not False or context.may_authorize_promotion is not False:
        raise ProtocolError("Consumed target predictions cannot authorize routing or promotion.")
    if context.claim_role != LABEL_FREE_INFERENCE_CLAIM_ROLE:
        raise ProtocolError("Consumed target inference claim role drifted.")
    return context


__all__ = (
    "LABEL_FREE_INFERENCE_CLAIM_ROLE",
    "InferenceActionSchema",
    "LabelFreeInferenceContext",
    "assert_label_free_inference_context",
)
