"""Immutable pseudo, target, surface, and reference legacy contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import LEGACY_METHOD_ID, canonical_hash, require_sha256


@dataclass(frozen=True)
class LegacyControlDecision:
    """One actual legacy pseudo-policy decision and its realized response."""

    outer_center: str
    donor_center: str
    policy_surface_hash: str
    pseudo_response_surface_hash: str
    selected_k: int
    selected_cell_hash: str
    selected_response_hash: str
    selected_action_hashes: tuple[str, ...]
    realized_utility: FavorableUtility
    routed: bool
    jointly_safe: bool
    endpoint_oracle_bacc_gain: float
    absolute_oracle_regret: float
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.realized_utility, FavorableUtility):
            raise ProtocolError("P-DCAPS legacy realized utility drifted.")
        surface_hash = require_sha256(
            self.policy_surface_hash, "legacy policy surface hash"
        )
        response_surface_hash = require_sha256(
            self.pseudo_response_surface_hash,
            "legacy pseudo-response surface hash",
        )
        cell_hash = require_sha256(
            self.selected_cell_hash, "legacy selected-cell hash"
        )
        response_hash = require_sha256(
            self.selected_response_hash, "legacy selected-response hash"
        )
        action_hashes = tuple(
            require_sha256(value, "legacy selected-action hash")
            for value in self.selected_action_hashes
        )
        values = np.asarray(
            (self.endpoint_oracle_bacc_gain, self.absolute_oracle_regret),
            dtype=np.float64,
        )
        expected_safe = bool(
            self.selected_k > 0
            and self.realized_utility.bacc_gain > 0.0
            and self.realized_utility.brier_gain >= 0.0
            and self.realized_utility.log_gain >= 0.0
        )
        expected_regret = abs(
            float(self.endpoint_oracle_bacc_gain)
            - self.realized_utility.bacc_gain
        )
        if (
            self.outer_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_center == self.donor_center
            or not isinstance(self.selected_k, int)
            or isinstance(self.selected_k, bool)
            or self.selected_k < 0
            or len(action_hashes) != self.selected_k
            or len(set(action_hashes)) != len(action_hashes)
            or bool(self.routed) != (self.selected_k > 0)
            or bool(self.jointly_safe) != expected_safe
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or abs(float(self.absolute_oracle_regret) - expected_regret) > 1.0e-15
        ):
            raise ProtocolError("P-DCAPS legacy control decision drifted.")
        object.__setattr__(self, "policy_surface_hash", surface_hash)
        object.__setattr__(
            self, "pseudo_response_surface_hash", response_surface_hash
        )
        object.__setattr__(self, "selected_cell_hash", cell_hash)
        object.__setattr__(self, "selected_response_hash", response_hash)
        object.__setattr__(self, "selected_action_hashes", action_hashes)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_control_decision_v1",
                    "outer_center": self.outer_center,
                    "donor_center": self.donor_center,
                    "policy_surface_hash": surface_hash,
                    "pseudo_response_surface_hash": response_surface_hash,
                    "selected_k": self.selected_k,
                    "selected_cell_hash": cell_hash,
                    "selected_response_hash": response_hash,
                    "selected_action_hashes": action_hashes,
                    "realized_utility": self.realized_utility.to_payload(),
                    "routed": self.routed,
                    "jointly_safe": self.jointly_safe,
                    "endpoint_oracle_bacc_gain": self.endpoint_oracle_bacc_gain,
                    "absolute_oracle_regret": self.absolute_oracle_regret,
                    "same_run_pseudo_response": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_legacy_control_decision_v1",
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "policy_surface_hash": self.policy_surface_hash,
            "pseudo_response_surface_hash": self.pseudo_response_surface_hash,
            "selected_k": self.selected_k,
            "selected_cell_hash": self.selected_cell_hash,
            "selected_response_hash": self.selected_response_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "realized_utility": self.realized_utility.to_payload(),
            "routed": self.routed,
            "jointly_safe": self.jointly_safe,
            "endpoint_oracle_bacc_gain": self.endpoint_oracle_bacc_gain,
            "absolute_oracle_regret": self.absolute_oracle_regret,
            "same_run_pseudo_response": True,
            "target_labels_used": False,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class LegacyTargetPolicyDecision:
    """Target-H control choice learned only from center-pooled pseudo responses."""

    outer_center: str
    target_policy_surface_hash: str
    selected_k: int
    selected_cell_hash: str
    selected_action_hashes: tuple[str, ...]
    normalized_depth: float
    pooled_realized_utility: FavorableUtility
    matched_pseudo_decision_hashes: tuple[str, ...]
    authorized: bool
    reason: str
    decision_hash: str = field(init=False)

    @property
    def method_id(self) -> str:
        return LEGACY_METHOD_ID

    @property
    def exact_p_fallback(self) -> bool:
        return not self.authorized

    def __post_init__(self) -> None:
        if not isinstance(self.pooled_realized_utility, FavorableUtility):
            raise ProtocolError("P-DCAPS legacy pooled utility drifted.")
        surface_hash = require_sha256(
            self.target_policy_surface_hash,
            "legacy target policy surface hash",
        )
        cell_hash = require_sha256(
            self.selected_cell_hash, "legacy target selected-cell hash"
        )
        action_hashes = tuple(
            require_sha256(value, "legacy target selected-action hash")
            for value in self.selected_action_hashes
        )
        pseudo_hashes = tuple(
            require_sha256(value, "legacy matched pseudo-decision hash")
            for value in self.matched_pseudo_decision_hashes
        )
        depth = float(self.normalized_depth)
        if (
            self.outer_center not in CENTERS
            or not isinstance(self.selected_k, int)
            or isinstance(self.selected_k, bool)
            or self.selected_k < 0
            or len(action_hashes) != self.selected_k
            or len(set(action_hashes)) != len(action_hashes)
            or len(pseudo_hashes) != len(CENTERS) - 1
            or len(set(pseudo_hashes)) != len(pseudo_hashes)
            or not np.isfinite(depth)
            or not 0.0 <= depth <= 1.0
            or bool(self.authorized) != (self.selected_k > 0)
            or not self.reason
            or (
                self.selected_k == 0
                and self.pooled_realized_utility != FavorableUtility.zeros()
            )
        ):
            raise ProtocolError("P-DCAPS legacy target policy decision drifted.")
        object.__setattr__(self, "target_policy_surface_hash", surface_hash)
        object.__setattr__(self, "selected_cell_hash", cell_hash)
        object.__setattr__(self, "selected_action_hashes", action_hashes)
        object.__setattr__(self, "normalized_depth", depth)
        object.__setattr__(self, "matched_pseudo_decision_hashes", pseudo_hashes)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_target_policy_decision_v1",
                    "method_id": LEGACY_METHOD_ID,
                    "outer_center": self.outer_center,
                    "target_policy_surface_hash": surface_hash,
                    "selected_k": self.selected_k,
                    "selected_cell_hash": cell_hash,
                    "selected_action_hashes": action_hashes,
                    "normalized_depth": depth,
                    "pooled_realized_utility": (
                        self.pooled_realized_utility.to_payload()
                    ),
                    "matched_pseudo_decision_hashes": pseudo_hashes,
                    "authorized": self.authorized,
                    "exact_p_fallback": self.exact_p_fallback,
                    "reason": self.reason,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_legacy_target_policy_decision_v1",
            "method_id": LEGACY_METHOD_ID,
            "outer_center": self.outer_center,
            "target_policy_surface_hash": self.target_policy_surface_hash,
            "selected_k": self.selected_k,
            "selected_cell_hash": self.selected_cell_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "normalized_depth": self.normalized_depth,
            "pooled_realized_utility": self.pooled_realized_utility.to_payload(),
            "matched_pseudo_decision_hashes": list(
                self.matched_pseudo_decision_hashes
            ),
            "authorized": self.authorized,
            "exact_p_fallback": self.exact_p_fallback,
            "reason": self.reason,
            "target_labels_used": False,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class LegacyControlSurface:
    """Complete current-run legacy comparator surface for one outer H."""

    outer_center: str
    outer_result_hash: str
    physical_surface_hash: str
    action_surface_seal_hash: str
    pseudo_response_surface_hashes: tuple[tuple[str, str], ...]
    decisions: tuple[LegacyControlDecision, ...]
    target_decision: LegacyTargetPolicyDecision
    control_surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        donors = tuple(center for center in CENTERS if center != self.outer_center)
        result_hash = require_sha256(
            self.outer_result_hash, "legacy outer-result hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "legacy physical surface hash"
        )
        action_hash = require_sha256(
            self.action_surface_seal_hash, "legacy action-surface seal hash"
        )
        response_hashes = tuple(
            (str(center), require_sha256(value, "legacy pseudo-response hash"))
            for center, value in self.pseudo_response_surface_hashes
        )
        decisions = tuple(self.decisions)
        if (
            self.outer_center not in CENTERS
            or any(
                not isinstance(row, LegacyControlDecision) for row in decisions
            )
            or not isinstance(
                self.target_decision, LegacyTargetPolicyDecision
            )
            or tuple(center for center, _value in response_hashes) != donors
            or tuple(row.donor_center for row in decisions) != donors
            or any(row.outer_center != self.outer_center for row in decisions)
            or tuple(
                (row.donor_center, row.pseudo_response_surface_hash)
                for row in decisions
            )
            != response_hashes
            or self.target_decision.outer_center != self.outer_center
            or self.target_decision.matched_pseudo_decision_hashes
            != tuple(row.decision_hash for row in decisions)
        ):
            raise ProtocolError("P-DCAPS legacy control surface inventory drifted.")
        object.__setattr__(self, "outer_result_hash", result_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "action_surface_seal_hash", action_hash)
        object.__setattr__(self, "pseudo_response_surface_hashes", response_hashes)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "control_surface_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_control_surface_v1",
                    "outer_center": self.outer_center,
                    "outer_result_hash": result_hash,
                    "physical_surface_hash": physical_hash,
                    "action_surface_seal_hash": action_hash,
                    "pseudo_response_surface_hashes": response_hashes,
                    "legacy_decision_hashes": tuple(
                        row.decision_hash for row in decisions
                    ),
                    "legacy_target_decision_hash": (
                        self.target_decision.decision_hash
                    ),
                    "complete_pseudo_surfaces_required": True,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_legacy_control_surface_v1",
            "outer_center": self.outer_center,
            "outer_result_hash": self.outer_result_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "pseudo_response_surface_hashes": [
                [center, value]
                for center, value in self.pseudo_response_surface_hashes
            ],
            "decisions": [row.to_payload() for row in self.decisions],
            "target_decision": self.target_decision.to_payload(),
            "complete_pseudo_surfaces_required": True,
            "target_labels_used": False,
            "control_surface_hash": self.control_surface_hash,
        }


@dataclass(frozen=True)
class LegacyPseudoReference:
    """A reference emitted by, and cryptographically rebound to, a seal."""

    outer_result_hash: str
    physical_surface_hash: str
    action_surface_seal_hash: str
    control_surface_hash: str
    legacy_control_seal_hash: str
    target_decision: LegacyTargetPolicyDecision
    decision: LegacyControlDecision
    reference_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_decision, LegacyTargetPolicyDecision)
            or not isinstance(self.decision, LegacyControlDecision)
        ):
            raise ProtocolError("P-DCAPS legacy reference DTO drifted.")
        result_hash = require_sha256(
            self.outer_result_hash, "legacy reference outer-result hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "legacy reference physical hash"
        )
        action_hash = require_sha256(
            self.action_surface_seal_hash, "legacy reference action-seal hash"
        )
        surface_hash = require_sha256(
            self.control_surface_hash, "legacy reference control-surface hash"
        )
        seal_hash = require_sha256(
            self.legacy_control_seal_hash, "legacy reference control-seal hash"
        )
        object.__setattr__(self, "outer_result_hash", result_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "action_surface_seal_hash", action_hash)
        object.__setattr__(self, "control_surface_hash", surface_hash)
        object.__setattr__(self, "legacy_control_seal_hash", seal_hash)
        object.__setattr__(
            self,
            "reference_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_legacy_pseudo_reference_v3",
                    "outer_result_hash": result_hash,
                    "physical_surface_hash": physical_hash,
                    "action_surface_seal_hash": action_hash,
                    "control_surface_hash": surface_hash,
                    "legacy_control_seal_hash": seal_hash,
                    "legacy_target_decision_hash": (
                        self.target_decision.decision_hash
                    ),
                    "decision_hash": self.decision.decision_hash,
                    "same_run_control": True,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def outer_center(self) -> str:
        return self.decision.outer_center

    @property
    def donor_center(self) -> str:
        return self.decision.donor_center

    @property
    def pseudo_response_surface_hash(self) -> str:
        return self.decision.pseudo_response_surface_hash

    @property
    def endpoint_oracle_bacc_gain(self) -> float:
        return self.decision.endpoint_oracle_bacc_gain

    @property
    def legacy_realized(self) -> FavorableUtility:
        return self.decision.realized_utility

    @property
    def legacy_routed(self) -> bool:
        return self.decision.routed

    @property
    def legacy_jointly_safe(self) -> bool:
        return self.decision.jointly_safe

    @property
    def legacy_absolute_oracle_regret(self) -> float:
        return self.decision.absolute_oracle_regret

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_legacy_pseudo_reference_v3",
            "outer_result_hash": self.outer_result_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "control_surface_hash": self.control_surface_hash,
            "legacy_control_seal_hash": self.legacy_control_seal_hash,
            "target_decision": self.target_decision.to_payload(),
            "decision": self.decision.to_payload(),
            "same_run_control": True,
            "target_labels_used": False,
            "reference_hash": self.reference_hash,
        }


__all__ = (
    "LegacyControlDecision",
    "LegacyControlSurface",
    "LegacyPseudoReference",
    "LegacyTargetPolicyDecision",
)
