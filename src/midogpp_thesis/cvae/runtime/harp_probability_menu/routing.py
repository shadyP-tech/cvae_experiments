"""Label-free per-row routing over a fully sealed HARP probability menu."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .actions import BASE_ACTION_ID, UNIFORM_ACTION_ID, HarpActionSpec
from .hashing import canonical_sha256, raw_array_sha256, require_digest, require_sha256
from .predictions import HarpPredictionMenuSeal


LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)
ROUTE_DIRECTIONS = ("D01", "D10", "ALL_MARGINS", "NO_DISAGREEMENT")


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"HARP {name} must be a canonical identity.")
    return value


@dataclass(frozen=True, kw_only=True)
class HarpRouteDecision:
    """One sealed label-free policy decision for one probability row."""

    surface_kind: str
    outer_target_id: str
    query_center_id: str
    row_id: str
    case_id: str
    eligible: bool
    selected_source_id: str | None
    lambda_value: float
    direction: str
    decision_reason: str
    policy_hash: str
    prediction_menu_seal_hash: str
    decision_hash: str = field(init=False)
    labels_consumed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "surface_kind",
            "outer_target_id",
            "query_center_id",
            "row_id",
            "case_id",
            "decision_reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.eligible) is not bool:
            raise ProtocolError("HARP decision eligibility must be boolean.")
        if self.labels_consumed is not False:
            raise ProtocolError("HARP route decisions cannot consume labels.")
        if self.direction not in ROUTE_DIRECTIONS:
            raise ProtocolError("HARP route direction is outside the frozen vocabulary.")
        policy_hash = require_digest(self.policy_hash, name="policy hash")
        menu_hash = require_sha256(
            self.prediction_menu_seal_hash, name="prediction-menu seal hash"
        )
        try:
            lam = float(self.lambda_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("HARP lambda is malformed.") from exc
        if not math.isfinite(lam):
            raise ProtocolError("HARP lambda must be finite.")

        HarpActionSpec(
            surface_kind=self.surface_kind,
            outer_target_id=self.outer_target_id,
            query_center_id=self.query_center_id,
            selected_source_id=None,
        )

        if self.eligible:
            if self.selected_source_id is None or lam not in LAMBDA_GRID:
                raise ProtocolError("Eligible HARP routes require one expert and frozen lambda.")
            selected = _identity(self.selected_source_id, name="selected source e")
            # Constructing the action is the role-fence validation: e cannot be H
            # (nor q on development surfaces).
            HarpActionSpec(
                surface_kind=self.surface_kind,
                outer_target_id=self.outer_target_id,
                query_center_id=self.query_center_id,
                selected_source_id=selected,
            )
        else:
            if self.selected_source_id is not None or lam != 0.0:
                raise ProtocolError("Fallback HARP routes must encode exact B explicitly.")
            selected = None

        object.__setattr__(self, "selected_source_id", selected)
        object.__setattr__(self, "lambda_value", lam)
        object.__setattr__(self, "policy_hash", policy_hash)
        object.__setattr__(self, "prediction_menu_seal_hash", menu_hash)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_sha256(self._payload_without_hash()),
        )

    @property
    def query_key(self) -> tuple[str, str, str]:
        return self.surface_kind, self.outer_target_id, self.query_center_id

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_route_decision_v1",
            "surface_kind": self.surface_kind,
            "outer_target_id": self.outer_target_id,
            "query_center_id": self.query_center_id,
            "row_id": self.row_id,
            "case_id": self.case_id,
            "eligible": self.eligible,
            "selected_source_id": self.selected_source_id,
            "lambda_value": self.lambda_value,
            "direction": self.direction,
            "decision_reason": self.decision_reason,
            "policy_hash": self.policy_hash,
            "prediction_menu_seal_hash": self.prediction_menu_seal_hash,
            "labels_consumed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, kw_only=True)
class HarpRoutedVectorSeal:
    """Byte-audited U-relative predictive routes with exact-B fallback."""

    decisions: tuple[HarpRouteDecision, ...]
    # B remains the lower-budget operational comparator and exact fallback.
    baseline_probabilities: np.ndarray
    # U is the matched-budget predictive reference for every eligible route.
    reference_probabilities: np.ndarray
    selected_action_probabilities: np.ndarray
    routed_probabilities: np.ndarray
    prediction_menu_seal_hash: str
    policy_hash: str
    baseline_bytes_sha256: str = field(init=False)
    reference_bytes_sha256: str = field(init=False)
    selected_action_bytes_sha256: str = field(init=False)
    routed_bytes_sha256: str = field(init=False)
    decision_set_hash: str = field(init=False)
    routed_vector_seal_hash: str = field(init=False)
    fallback_byte_identity: bool = field(init=False, default=True)
    labels_consumed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if not decisions or any(not isinstance(row, HarpRouteDecision) for row in decisions):
            raise ProtocolError("HARP routed vector requires typed decisions.")
        menu_hash = require_sha256(
            self.prediction_menu_seal_hash, name="prediction-menu seal hash"
        )
        policy_hash = require_digest(self.policy_hash, name="policy hash")
        if self.labels_consumed is not False:
            raise ProtocolError("HARP routed vectors cannot consume labels.")
        if any(
            decision.prediction_menu_seal_hash != menu_hash
            or decision.policy_hash != policy_hash
            for decision in decisions
        ):
            raise ProtocolError("HARP routed-vector decision binding drifted.")

        arrays: list[np.ndarray] = []
        for values in (
            self.baseline_probabilities,
            self.reference_probabilities,
            self.selected_action_probabilities,
            self.routed_probabilities,
        ):
            raw = np.asarray(values)
            if (
                raw.dtype != np.dtype("float64")
                or raw.ndim != 1
                or len(raw) != len(decisions)
                or not np.isfinite(raw).all()
                or np.any((raw < 0.0) | (raw > 1.0))
            ):
                raise ProtocolError("HARP routed vectors must be aligned float64 probabilities.")
            arrays.append(np.ascontiguousarray(raw, dtype=np.float64))
        baseline, reference, selected, routed = arrays

        fallback = np.asarray([not row.eligible for row in decisions], dtype=bool)
        eligible = ~fallback
        if np.any(
            routed[fallback].view(np.uint64) != baseline[fallback].view(np.uint64)
        ):
            raise ProtocolError("HARP exact-B fallback changed probability bytes.")
        expected = baseline.copy()
        for ordinal in np.flatnonzero(eligible):
            lam = np.float64(decisions[int(ordinal)].lambda_value)
            expected[ordinal] = (
                (np.float64(1.0) - lam) * reference[ordinal]
                + lam * selected[ordinal]
            )
        if np.any(expected.view(np.uint64) != routed.view(np.uint64)):
            raise ProtocolError("HARP routed probability blend drifted.")

        baseline_hash = raw_array_sha256(baseline)
        reference_hash = raw_array_sha256(reference)
        selected_hash = raw_array_sha256(selected)
        routed_hash = raw_array_sha256(routed)
        decision_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_route_decision_set_v1",
                "decisions": [row.to_payload() for row in decisions],
                "prediction_menu_seal_hash": menu_hash,
                "policy_hash": policy_hash,
                "labels_consumed": False,
            }
        )
        vector_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_routed_vector_seal_v2",
                "prediction_menu_seal_hash": menu_hash,
                "policy_hash": policy_hash,
                "decision_set_hash": decision_hash,
                "baseline_bytes_sha256": baseline_hash,
                "reference_bytes_sha256": reference_hash,
                "selected_action_bytes_sha256": selected_hash,
                "routed_bytes_sha256": routed_hash,
                "fallback_count": int(fallback.sum()),
                "eligible_count": int(eligible.sum()),
                "fallback_byte_identity": True,
                "blend_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
                "predictive_reference_action_id": UNIFORM_ACTION_ID,
                "operational_fallback_action_id": BASE_ACTION_ID,
                "labels_consumed": False,
            }
        )
        for values in arrays:
            values.setflags(write=False)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "baseline_probabilities", baseline)
        object.__setattr__(self, "reference_probabilities", reference)
        object.__setattr__(self, "selected_action_probabilities", selected)
        object.__setattr__(self, "routed_probabilities", routed)
        object.__setattr__(self, "prediction_menu_seal_hash", menu_hash)
        object.__setattr__(self, "policy_hash", policy_hash)
        object.__setattr__(self, "baseline_bytes_sha256", baseline_hash)
        object.__setattr__(self, "reference_bytes_sha256", reference_hash)
        object.__setattr__(self, "selected_action_bytes_sha256", selected_hash)
        object.__setattr__(self, "routed_bytes_sha256", routed_hash)
        object.__setattr__(self, "decision_set_hash", decision_hash)
        object.__setattr__(self, "routed_vector_seal_hash", vector_hash)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.decisions)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(row.case_id for row in self.decisions)

    def assert_valid(self) -> None:
        fallback = np.asarray([not row.eligible for row in self.decisions], dtype=bool)
        expected = self.baseline_probabilities.copy()
        for ordinal, row in enumerate(self.decisions):
            if row.eligible:
                lam = np.float64(row.lambda_value)
                expected[ordinal] = (
                    (np.float64(1.0) - lam) * self.reference_probabilities[ordinal]
                    + lam * self.selected_action_probabilities[ordinal]
                )
        decision_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_route_decision_set_v1",
                "decisions": [row.to_payload() for row in self.decisions],
                "prediction_menu_seal_hash": self.prediction_menu_seal_hash,
                "policy_hash": self.policy_hash,
                "labels_consumed": False,
            }
        )
        expected_vector_hash = canonical_sha256(
            {
                "schema_version": "midogpp_harp_routed_vector_seal_v2",
                "prediction_menu_seal_hash": self.prediction_menu_seal_hash,
                "policy_hash": self.policy_hash,
                "decision_set_hash": decision_hash,
                "baseline_bytes_sha256": self.baseline_bytes_sha256,
                "reference_bytes_sha256": self.reference_bytes_sha256,
                "selected_action_bytes_sha256": self.selected_action_bytes_sha256,
                "routed_bytes_sha256": self.routed_bytes_sha256,
                "fallback_count": int(fallback.sum()),
                "eligible_count": int((~fallback).sum()),
                "fallback_byte_identity": True,
                "blend_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
                "predictive_reference_action_id": UNIFORM_ACTION_ID,
                "operational_fallback_action_id": BASE_ACTION_ID,
                "labels_consumed": False,
            }
        )
        if (
            self.labels_consumed is not False
            or not self.fallback_byte_identity
            or self.baseline_probabilities.flags.writeable
            or self.reference_probabilities.flags.writeable
            or self.selected_action_probabilities.flags.writeable
            or self.routed_probabilities.flags.writeable
            or raw_array_sha256(self.baseline_probabilities)
            != self.baseline_bytes_sha256
            or raw_array_sha256(self.reference_probabilities)
            != self.reference_bytes_sha256
            or raw_array_sha256(self.selected_action_probabilities)
            != self.selected_action_bytes_sha256
            or raw_array_sha256(self.routed_probabilities) != self.routed_bytes_sha256
            or np.any(expected.view(np.uint64) != self.routed_probabilities.view(np.uint64))
            or decision_hash != self.decision_set_hash
            or expected_vector_hash != self.routed_vector_seal_hash
            or np.any(
                self.routed_probabilities[fallback].view(np.uint64)
                != self.baseline_probabilities[fallback].view(np.uint64)
            )
        ):
            raise ProtocolError("HARP routed vector seal drifted.")


def route_harp_probability_vector(
    menu: HarpPredictionMenuSeal,
    decisions: Sequence[HarpRouteDecision],
) -> HarpRoutedVectorSeal:
    """Route one query only after its entire prediction menu is sealed."""

    if not isinstance(menu, HarpPredictionMenuSeal):
        raise ProtocolError("HARP routing requires a complete prediction-menu seal.")
    menu.assert_valid()
    rows = tuple(decisions)
    if not rows or any(not isinstance(row, HarpRouteDecision) for row in rows):
        raise ProtocolError("HARP routing requires typed row decisions.")
    query_keys = {row.query_key for row in rows}
    policy_hashes = {row.policy_hash for row in rows}
    if len(query_keys) != 1 or len(policy_hashes) != 1:
        raise ProtocolError("HARP decisions must describe one query and one policy.")
    surface, outer, query = next(iter(query_keys))
    if any(row.prediction_menu_seal_hash != menu.seal_hash for row in rows):
        raise ProtocolError("HARP decisions were not sealed against this menu.")

    baseline_action = menu.action_for(
        surface_kind=surface,
        outer_target_id=outer,
        query_center_id=query,
        selected_source_id=None,
        action_id=BASE_ACTION_ID,
    )
    reference_action = menu.action_for(
        surface_kind=surface,
        outer_target_id=outer,
        query_center_id=query,
        selected_source_id=None,
        action_id=UNIFORM_ACTION_ID,
    )
    row_ids, case_ids = menu.identities_for(baseline_action)
    if tuple((row.row_id, row.case_id) for row in rows) != tuple(
        zip(row_ids, case_ids, strict=True)
    ):
        raise ProtocolError("HARP route decisions do not cover sealed rows in order.")
    baseline = menu.exact_nine(baseline_action)
    reference = menu.exact_nine(reference_action)
    selected = baseline.copy()
    routed = baseline.copy()
    action_cache: dict[str, np.ndarray] = {}
    for ordinal, decision in enumerate(rows):
        if not decision.eligible:
            # Direct assignment preserves the exact float64 B bit pattern.
            selected[ordinal] = baseline[ordinal]
            routed[ordinal] = baseline[ordinal]
            continue
        assert decision.selected_source_id is not None
        source = decision.selected_source_id
        if source not in action_cache:
            action = menu.action_for(
                surface_kind=surface,
                outer_target_id=outer,
                query_center_id=query,
                selected_source_id=source,
            )
            action_cache[source] = menu.exact_nine(action)
        expert_probability = action_cache[source][ordinal]
        selected[ordinal] = expert_probability
        lam = np.float64(decision.lambda_value)
        routed[ordinal] = (
            (np.float64(1.0) - lam) * reference[ordinal]
            + lam * expert_probability
        )

    return HarpRoutedVectorSeal(
        decisions=rows,
        baseline_probabilities=np.ascontiguousarray(baseline, dtype=np.float64),
        reference_probabilities=np.ascontiguousarray(reference, dtype=np.float64),
        selected_action_probabilities=np.ascontiguousarray(selected, dtype=np.float64),
        routed_probabilities=np.ascontiguousarray(routed, dtype=np.float64),
        prediction_menu_seal_hash=menu.seal_hash,
        policy_hash=next(iter(policy_hashes)),
    )


__all__ = (
    "HarpRouteDecision",
    "HarpRoutedVectorSeal",
    "LAMBDA_GRID",
    "ROUTE_DIRECTIONS",
    "route_harp_probability_vector",
)
