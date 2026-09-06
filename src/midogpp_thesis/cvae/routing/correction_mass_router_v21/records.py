"""Pre-truth selection seals and post-selection aggregate score records."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .contracts import SoftTopKComposite, SurfaceRole, canonical_text, finite
from .hashing import canonical_hash, require_sha256
from .decision_evidence import winner_evidence_payload


@dataclass(frozen=True, slots=True)
class SealedOOFSelection:
    outer_fold: int
    composite: SoftTopKComposite
    requested_arm_id: str
    route_score: float
    route_threshold: float
    training_case_keys: tuple[tuple[str, str], ...]
    model_hash: str
    policy_enabled: bool = True
    fallback_reason: str | None = None
    winner_arm_id: str | None = None
    winner_composite_hash: str | None = None
    winner_risk_adjusted_score: float | None = None
    winner_gate_harm_probability: float | None = None
    winner_gate_model_hash: str | None = None
    winner_gate_prediction_hash: str | None = None
    winner_gate_prediction_payload: tuple[tuple[str, object], ...] | None = None
    selection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_fold) is not int
            or self.outer_fold < 0
            or not isinstance(self.composite, SoftTopKComposite)
            or self.composite.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
            or type(self.policy_enabled) is not bool
            or (not self.policy_enabled and self.composite.route_selected)
        ):
            raise ProtocolError("HARP v21 OOF selection seal is malformed.")
        if self.fallback_reason is not None:
            canonical_text(self.fallback_reason, name="OOF fallback reason")
        requested = canonical_text(self.requested_arm_id, name="requested OOF arm id")
        score = finite(self.route_score, name="OOF route score")
        threshold = finite(self.route_threshold, name="OOF route threshold")
        if score < 0.0 or threshold < 0.0:
            raise ProtocolError("HARP v21 OOF route score/threshold must be nonnegative.")
        training = tuple(
            sorted(
                (
                    canonical_text(center, name="training center id"),
                    canonical_text(case, name="training case id"),
                )
                for center, case in self.training_case_keys
            )
        )
        heldout = (self.composite.center_id, self.composite.case_id)
        if not training or len(training) != len(set(training)) or heldout in training:
            raise ProtocolError("HARP v21 OOF held case entered its model fit.")
        model_hash = require_sha256(self.model_hash, name="OOF model hash")
        object.__setattr__(self, "requested_arm_id", requested)
        object.__setattr__(self, "route_score", score)
        object.__setattr__(self, "route_threshold", threshold)
        object.__setattr__(self, "training_case_keys", training)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(
            self,
            "selection_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_selected_oof_seal_v21",
                    **winner_evidence_payload(self),
                    "outer_fold": self.outer_fold,
                    "composite_hash": self.composite.composite_hash,
                    "requested_arm_id": requested,
                    "route_score": score,
                    "route_threshold": threshold,
                    "training_case_keys": training,
                    "model_hash": model_hash,
                    "policy_enabled": self.policy_enabled,
                    "fallback_reason": self.fallback_reason,
                    "selected_before_heldout_truth": True,
                    "heldout_truth_joined": False,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_fold": self.outer_fold,
            "composite": self.composite.public_payload(),
            "requested_arm_id": self.requested_arm_id,
            "route_score": self.route_score,
            "route_threshold": self.route_threshold,
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "model_hash": self.model_hash,
            "selection_hash": self.selection_hash,
            "policy_enabled": self.policy_enabled,
            "fallback_reason": self.fallback_reason,
            **winner_evidence_payload(self),
            "selected_before_heldout_truth": True,
            "heldout_truth_joined": False,
            "target_evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SelectedOOFRecord:
    """The sole post-selection join between one OOF seal and source truth."""

    selection: SealedOOFSelection
    bacc_gain: float
    brier_delta: float
    log_loss_delta: float
    class_0_gain: float | None = None
    class_1_gain: float | None = None
    normalization_hash: str | None = None
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, SealedOOFSelection):
            raise ProtocolError("HARP v21 selected OOF record lacks its pre-truth seal.")
        gain = finite(self.bacc_gain, name="selected OOF BACC gain")
        brier = finite(self.brier_delta, name="selected OOF Brier delta")
        logloss = finite(self.log_loss_delta, name="selected OOF log-loss delta")
        if not -1.0 <= brier <= 1.0:
            raise ProtocolError("HARP v21 selected OOF endpoints are outside metric bounds.")
        for value in (self.class_0_gain, self.class_1_gain):
            if value is not None and not -1.0 <= finite(value,name="class recall gain") <= 1.0:
                raise ProtocolError("HARP v21 selected class recall delta is malformed.")
        if self.normalization_hash is not None:
            require_sha256(self.normalization_hash,name="selected normalization hash")
        object.__setattr__(self, "bacc_gain", gain)
        object.__setattr__(self, "brier_delta", brier)
        object.__setattr__(self, "log_loss_delta", logloss)
        object.__setattr__(
            self,
            "score_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_selected_oof_record_v21",
                    "selection_hash": self.selection.selection_hash,
                    "bacc_gain": gain,
                    "class_0_gain": self.class_0_gain,
                    "class_1_gain": self.class_1_gain,
                    "normalization_hash": self.normalization_hash,
                    "harm": self.harm,
                    "brier_delta": brier,
                    "log_loss_delta": logloss,
                    "utility_success": self.utility_success,
                    "only_selected_composite_scored": True,
                    "raw_labels_persisted": False,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def center_id(self) -> str:
        return self.selection.composite.center_id

    @property
    def case_id(self) -> str:
        return self.selection.composite.case_id

    @property
    def route_selected(self) -> bool:
        return self.selection.composite.route_selected

    @property
    def probability_changed(self) -> bool:
        return self.selection.composite.probability_changed

    @property
    def prediction_changed(self) -> bool:
        return self.selection.composite.prediction_changed

    @property
    def harm(self) -> bool:
        return self.bacc_gain < 0.0

    @property
    def utility_success(self) -> bool:
        return bool(
            self.route_selected
            and self.bacc_gain > 0.0
            and self.brier_delta <= 0.002
            and self.log_loss_delta <= 0.005
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "selection": self.selection.public_payload(),
            "center_id": self.center_id,
            "case_id": self.case_id,
            "route_selected": self.route_selected,
            "probability_changed": self.probability_changed,
            "prediction_changed": self.prediction_changed,
            "utility_success": self.utility_success,
            "bacc_gain": self.bacc_gain,
            "class_0_gain": self.class_0_gain,
            "class_1_gain": self.class_1_gain,
            "normalization_hash": self.normalization_hash,
            "harm": self.harm,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "score_hash": self.score_hash,
            "only_selected_composite_scored": True,
            "raw_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    composite: SoftTopKComposite
    requested_arm_id: str
    route_score: float
    route_threshold: float
    policy_hash: str
    admitted: bool
    fallback_reason: str | None = None
    utility_success: None = None
    winner_arm_id: str | None = None
    winner_composite_hash: str | None = None
    winner_risk_adjusted_score: float | None = None
    winner_gate_harm_probability: float | None = None
    winner_gate_model_hash: str | None = None
    winner_gate_prediction_hash: str | None = None
    winner_gate_prediction_payload: tuple[tuple[str, object], ...] | None = None
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.composite, SoftTopKComposite):
            raise ProtocolError("HARP v21 route decision lacks a composite.")
        requested = canonical_text(self.requested_arm_id, name="requested route arm")
        score = finite(self.route_score, name="route score")
        threshold = finite(self.route_threshold, name="route threshold")
        policy_hash = require_sha256(self.policy_hash, name="route policy hash")
        reason = self.fallback_reason
        if reason is not None:
            reason = canonical_text(reason, name="fallback reason")
        if (
            score < 0.0
            or threshold < 0.0
            or type(self.admitted) is not bool
            or (self.composite.route_selected and not self.admitted)
            or self.utility_success is not None
            or (self.composite.route_selected and reason is not None)
        ):
            raise ProtocolError("HARP v21 route decision is malformed.")
        object.__setattr__(self, "requested_arm_id", requested)
        object.__setattr__(self, "route_score", score)
        object.__setattr__(self, "route_threshold", threshold)
        object.__setattr__(self, "policy_hash", policy_hash)
        object.__setattr__(self, "fallback_reason", reason)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_route_decision_v21",
                    **winner_evidence_payload(self),
                    "composite_hash": self.composite.composite_hash,
                    "requested_arm_id": requested,
                    "route_score": score,
                    "route_threshold": threshold,
                    "policy_hash": policy_hash,
                    "admitted": self.admitted,
                    "fallback_reason": reason,
                    "route_selected": self.route_selected,
                    "probability_changed": self.probability_changed,
                    "prediction_changed": self.prediction_changed,
                    "utility_success": None,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def route_selected(self) -> bool:
        return self.composite.route_selected

    @property
    def probability_changed(self) -> bool:
        return self.composite.probability_changed

    @property
    def prediction_changed(self) -> bool:
        return self.composite.prediction_changed

    @property
    def donor_entropy(self) -> float:
        return self.composite.donor_entropy

    @property
    def probability_hex(self) -> tuple[str, ...]:
        return self.composite.probability_hex

    def public_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.composite.surface_role.value,
            "center_id": self.composite.center_id,
            "case_id": self.composite.case_id,
            "menu_hash": self.composite.menu_hash,
            "requested_arm_id": self.requested_arm_id,
            "selected_arm_id": self.composite.arm_id,
            "route_score": self.route_score,
            "route_threshold": self.route_threshold,
            "admitted": self.admitted,
            "fallback_reason": self.fallback_reason,
            "route_selected": self.route_selected,
            "probability_changed": self.probability_changed,
            "prediction_changed": self.prediction_changed,
            "utility_success": None,
            "donor_entropy": self.donor_entropy,
            "probability_hex": list(self.probability_hex),
            "composite_hash": self.composite.composite_hash,
            "policy_hash": self.policy_hash,
            "decision_hash": self.decision_hash,
            **winner_evidence_payload(self),
            "target_evaluation_labels_consumed": False,
        }


__all__ = ("RouteDecision", "SealedOOFSelection", "SelectedOOFRecord")
