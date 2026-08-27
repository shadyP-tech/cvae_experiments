"""Additive primitive-utility contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .shared import P_ACTION_ID, ProtocolError, _text, canonical_sha256

@dataclass(frozen=True, slots=True)
class PrimitiveUtility:
    """Additive, denominator-free expected candidate-minus-P primitives."""

    delta_tp: float
    delta_tn: float
    delta_brier_sum: float
    delta_log_sum: float
    row_count: int
    action_id: str
    baseline_probability_hash: str
    candidate_probability_hash: str
    scope_id: str
    row_manifest_hash: str
    posterior_model_hash: str
    posterior_scope_receipt_hash: str
    response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            float(self.delta_tp),
            float(self.delta_tn),
            float(self.delta_brier_sum),
            float(self.delta_log_sum),
        )
        if not all(math.isfinite(value) for value in values) or int(self.row_count) <= 0:
            raise ProtocolError("Primitive utility is empty or non-finite.")
        object.__setattr__(self, "delta_tp", values[0])
        object.__setattr__(self, "delta_tn", values[1])
        object.__setattr__(self, "delta_brier_sum", values[2])
        object.__setattr__(self, "delta_log_sum", values[3])
        object.__setattr__(self, "row_count", int(self.row_count))
        action_id = _text(self.action_id, role="primitive action")
        if action_id == P_ACTION_ID:
            raise ProtocolError("Primitive utility is for a challenger, not protected P.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(
            self,
            "baseline_probability_hash",
            _text(self.baseline_probability_hash, role="protected probability surface"),
        )
        object.__setattr__(
            self,
            "candidate_probability_hash",
            _text(self.candidate_probability_hash, role="candidate probability surface"),
        )
        object.__setattr__(self, "scope_id", _text(self.scope_id, role="primitive scope"))
        object.__setattr__(
            self, "row_manifest_hash", _text(self.row_manifest_hash, role="primitive row manifest")
        )
        object.__setattr__(
            self,
            "posterior_model_hash",
            _text(self.posterior_model_hash, role="primitive posterior model"),
        )
        object.__setattr__(
            self,
            "posterior_scope_receipt_hash",
            _text(self.posterior_scope_receipt_hash, role="primitive posterior scope receipt"),
        )
        object.__setattr__(
            self,
            "response_hash",
            canonical_sha256(
                {
                    "schema": "primitive_utility_response_v3",
                    "delta_tp": values[0],
                    "delta_tn": values[1],
                    "delta_brier_sum": values[2],
                    "delta_log_sum": values[3],
                    "row_count": int(self.row_count),
                    "action_id": self.action_id,
                    "baseline_probability_hash": self.baseline_probability_hash,
                    "candidate_probability_hash": self.candidate_probability_hash,
                    "scope_id": self.scope_id,
                    "row_manifest_hash": self.row_manifest_hash,
                    "posterior_model_hash": self.posterior_model_hash,
                    "posterior_scope_receipt_hash": self.posterior_scope_receipt_hash,
                }
            ),
        )

    @classmethod
    def zeros(
        cls,
        row_count: int,
        *,
        action_id: str,
        baseline_probability_hash: str,
        candidate_probability_hash: str,
        scope_id: str,
        row_manifest_hash: str,
        posterior_model_hash: str,
        posterior_scope_receipt_hash: str,
    ) -> "PrimitiveUtility":
        return cls(
            0.0,
            0.0,
            0.0,
            0.0,
            int(row_count),
            action_id,
            baseline_probability_hash,
            candidate_probability_hash,
            scope_id,
            row_manifest_hash,
            posterior_model_hash,
            posterior_scope_receipt_hash,
        )

    def __add__(self, other: object) -> "PrimitiveUtility":
        if not isinstance(other, PrimitiveUtility):
            return NotImplemented
        if (
            self.scope_id != other.scope_id
            or self.action_id != other.action_id
            or self.posterior_model_hash != other.posterior_model_hash
            or self.posterior_scope_receipt_hash != other.posterior_scope_receipt_hash
            or self.row_manifest_hash == other.row_manifest_hash
        ):
            raise ProtocolError(
                "Primitive aggregation requires one scope and distinct case row manifests."
            )
        return PrimitiveUtility(
            self.delta_tp + other.delta_tp,
            self.delta_tn + other.delta_tn,
            self.delta_brier_sum + other.delta_brier_sum,
            self.delta_log_sum + other.delta_log_sum,
            self.row_count + other.row_count,
            self.action_id,
            canonical_sha256(
                {
                    "schema": "aggregated_protected_probability_surface_v1",
                    "member_hashes": tuple(
                        sorted((self.baseline_probability_hash, other.baseline_probability_hash))
                    ),
                }
            ),
            canonical_sha256(
                {
                    "schema": "aggregated_candidate_probability_surface_v1",
                    "member_hashes": tuple(
                        sorted((self.candidate_probability_hash, other.candidate_probability_hash))
                    ),
                }
            ),
            self.scope_id,
            canonical_sha256(
                {
                    "schema": "aggregated_primitive_row_manifest_v1",
                    "member_hashes": tuple(sorted((self.row_manifest_hash, other.row_manifest_hash))),
                }
            ),
            self.posterior_model_hash,
            self.posterior_scope_receipt_hash,
        )
@dataclass(frozen=True, slots=True)
class ExpectedDenominators:
    """One action-invariant normalization surface for an evaluation scope."""

    scope_id: str
    expected_positive: float
    expected_negative: float
    row_count: int
    eta_hash: str
    row_manifest_hash: str
    posterior_model_hash: str
    posterior_scope_receipt_hash: str

    def __post_init__(self) -> None:
        positive = float(self.expected_positive)
        negative = float(self.expected_negative)
        rows = int(self.row_count)
        if (
            not math.isfinite(positive)
            or not math.isfinite(negative)
            or positive <= 0.0
            or negative <= 0.0
            or rows <= 0
            or not math.isclose(positive + negative, rows, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise ProtocolError("Expected denominators must be finite, positive, and exhaustive.")
        object.__setattr__(self, "scope_id", _text(self.scope_id, role="denominator scope"))
        object.__setattr__(self, "expected_positive", positive)
        object.__setattr__(self, "expected_negative", negative)
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "eta_hash", _text(self.eta_hash, role="eta hash"))
        object.__setattr__(
            self,
            "row_manifest_hash",
            _text(self.row_manifest_hash, role="denominator row manifest"),
        )
        object.__setattr__(
            self,
            "posterior_model_hash",
            _text(self.posterior_model_hash, role="denominator posterior model"),
        )
        object.__setattr__(
            self,
            "posterior_scope_receipt_hash",
            _text(self.posterior_scope_receipt_hash, role="denominator posterior scope receipt"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedUtility:
    """Expected metric deltas produced from one frozen denominator surface."""

    bacc_gain: float
    brier_loss_delta: float
    log_loss_delta: float
    action_id: str
    baseline_probability_hash: str
    candidate_probability_hash: str
    denominator_scope_id: str
    denominator_eta_hash: str
    row_manifest_hash: str
    primitive_response_hash: str
    posterior_model_hash: str
    posterior_scope_receipt_hash: str
    response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            float(self.bacc_gain),
            float(self.brier_loss_delta),
            float(self.log_loss_delta),
        )
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("Normalized utility is non-finite.")
        object.__setattr__(self, "bacc_gain", values[0])
        object.__setattr__(self, "brier_loss_delta", values[1])
        object.__setattr__(self, "log_loss_delta", values[2])
        action_id = _text(self.action_id, role="normalized action")
        if action_id == P_ACTION_ID:
            raise ProtocolError("Normalized utility is for a challenger, not protected P.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(
            self,
            "baseline_probability_hash",
            _text(self.baseline_probability_hash, role="normalized protected surface"),
        )
        object.__setattr__(
            self,
            "candidate_probability_hash",
            _text(self.candidate_probability_hash, role="normalized candidate surface"),
        )
        object.__setattr__(
            self, "denominator_scope_id", _text(self.denominator_scope_id, role="denominator scope")
        )
        object.__setattr__(
            self,
            "posterior_model_hash",
            _text(self.posterior_model_hash, role="normalized posterior model"),
        )
        object.__setattr__(
            self,
            "posterior_scope_receipt_hash",
            _text(self.posterior_scope_receipt_hash, role="normalized posterior scope receipt"),
        )
        object.__setattr__(
            self, "denominator_eta_hash", _text(self.denominator_eta_hash, role="denominator eta hash")
        )
        object.__setattr__(
            self, "row_manifest_hash", _text(self.row_manifest_hash, role="normalized row manifest")
        )
        object.__setattr__(
            self,
            "primitive_response_hash",
            _text(self.primitive_response_hash, role="primitive response hash"),
        )
        object.__setattr__(
            self,
            "response_hash",
            canonical_sha256(
                {
                    "schema": "normalized_utility_response_v3",
                    "bacc_gain": values[0],
                    "brier_loss_delta": values[1],
                    "log_loss_delta": values[2],
                    "action_id": self.action_id,
                    "baseline_probability_hash": self.baseline_probability_hash,
                    "candidate_probability_hash": self.candidate_probability_hash,
                    "scope_id": self.denominator_scope_id,
                    "eta_hash": self.denominator_eta_hash,
                    "row_manifest_hash": self.row_manifest_hash,
                    "primitive_response_hash": self.primitive_response_hash,
                    "posterior_model_hash": self.posterior_model_hash,
                    "posterior_scope_receipt_hash": self.posterior_scope_receipt_hash,
                }
            ),
        )
