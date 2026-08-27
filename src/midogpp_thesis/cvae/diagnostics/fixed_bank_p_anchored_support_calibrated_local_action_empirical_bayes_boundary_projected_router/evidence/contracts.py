"""Opaque evidence rows issued only by the deterministic replay executor."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math

from ..controls import METHOD_IDS, P_PROTECTED
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS, DIRECTIONS
from ..protocol import ProtocolError
from ..replay_scope import PseudoReplayScope


_EVIDENCE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class PseudoRouteActionEvidence:
    """One realized action under one mechanically replayed pseudo policy."""

    scope: PseudoReplayScope
    method_id: str
    action_id: str
    opportunity: bool
    selected: bool
    crossing_indices: tuple[int, ...]
    predicted_bacc_gain: float
    realized_bacc_gain: float
    realized_brier_loss_delta: float
    realized_log_loss_delta: float
    descriptor_hash: str
    candidate_hash: str
    replay_request_hash: str
    terminal_label_hash: str
    method_menu_hash: str
    oracle_hash: str
    _factory_token: InitVar[object] = None
    evidence_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _EVIDENCE_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP pseudo action evidence was not issued by replay."
            )
        method = str(self.method_id)
        action = str(self.action_id)
        crossing = tuple(int(value) for value in self.crossing_indices)
        values = (
            float(self.predicted_bacc_gain),
            float(self.realized_bacc_gain),
            float(self.realized_brier_loss_delta),
            float(self.realized_log_loss_delta),
        )
        descriptor_hash = require_sha256(
            self.descriptor_hash, "pseudo action descriptor hash"
        )
        candidate_hash = require_sha256(
            self.candidate_hash, "pseudo action candidate hash"
        )
        request_hash = require_sha256(
            self.replay_request_hash, "pseudo action replay-request hash"
        )
        label_hash = require_sha256(
            self.terminal_label_hash, "pseudo action terminal-label hash"
        )
        menu_hash = require_sha256(
            self.method_menu_hash, "pseudo action method-menu hash"
        )
        oracle_hash = require_sha256(self.oracle_hash, "pseudo action oracle hash")
        if (
            not isinstance(self.scope, PseudoReplayScope)
            or method not in METHOD_IDS
            or action not in ACTION_IDS
            or type(self.opportunity) is not bool
            or type(self.selected) is not bool
            or (self.selected and not self.opportunity)
            or crossing != tuple(sorted(set(crossing)))
            or any(index < 0 for index in crossing)
            or self.opportunity != bool(crossing)
            or not all(math.isfinite(value) for value in values)
        ):
            raise ProtocolError("SCALE-BP pseudo route-action evidence drifted.")
        payload = {
            "schema_version": "scale_bp_pseudo_route_action_evidence_v2",
            "scope_hash": self.scope.scope_hash,
            "method_id": method,
            "action_id": action,
            "opportunity": self.opportunity,
            "selected": self.selected,
            "crossing_indices": crossing,
            "predicted_bacc_gain": values[0],
            "realized_bacc_gain": values[1],
            "realized_brier_loss_delta": values[2],
            "realized_log_loss_delta": values[3],
            "descriptor_hash": descriptor_hash,
            "candidate_hash": candidate_hash,
            "replay_request_hash": request_hash,
            "terminal_label_hash": label_hash,
            "method_menu_hash": menu_hash,
            "oracle_hash": oracle_hash,
            "terminal_labels_reduced_to_aggregate_only": True,
        }
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "crossing_indices", crossing)
        object.__setattr__(self, "predicted_bacc_gain", values[0])
        object.__setattr__(self, "realized_bacc_gain", values[1])
        object.__setattr__(self, "realized_brier_loss_delta", values[2])
        object.__setattr__(self, "realized_log_loss_delta", values[3])
        object.__setattr__(self, "descriptor_hash", descriptor_hash)
        object.__setattr__(self, "candidate_hash", candidate_hash)
        object.__setattr__(self, "replay_request_hash", request_hash)
        object.__setattr__(self, "terminal_label_hash", label_hash)
        object.__setattr__(self, "method_menu_hash", menu_hash)
        object.__setattr__(self, "oracle_hash", oracle_hash)
        object.__setattr__(self, "evidence_hash", canonical_hash(payload))

    @property
    def outer_center(self) -> str:
        return self.scope.outer_center

    @property
    def pseudo_center(self) -> str:
        return self.scope.pseudo_center

    @property
    def case_id(self) -> str:
        return self.scope.held_case_id


@dataclass(frozen=True, slots=True, kw_only=True)
class PseudoRoutePolicyEvidence:
    """Sealed policy result for one ``(H,J,d,method)`` replay."""

    scope: PseudoReplayScope
    method_id: str
    selected_action_ids: tuple[str, ...]
    realized_bacc_gain: float
    realized_brier_loss_delta: float
    realized_log_loss_delta: float
    oracle_bacc_gain: float
    decision_hash: str
    composition_hash: str
    action_evidence_hashes: tuple[str, ...]
    replay_request_hash: str
    terminal_label_hash: str
    method_menu_hash: str
    oracle_hash: str
    _factory_token: InitVar[object] = None
    policy_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _EVIDENCE_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP pseudo policy evidence was not issued by replay."
            )
        method = str(self.method_id)
        selected = tuple(str(value) for value in self.selected_action_ids)
        values = (
            float(self.realized_bacc_gain),
            float(self.realized_brier_loss_delta),
            float(self.realized_log_loss_delta),
            float(self.oracle_bacc_gain),
        )
        decision_hash = require_sha256(
            self.decision_hash, "pseudo policy decision hash"
        )
        composition_hash = require_sha256(
            self.composition_hash, "pseudo policy composition hash"
        )
        evidence_hashes = tuple(str(value) for value in self.action_evidence_hashes)
        for digest in evidence_hashes:
            require_sha256(digest, "pseudo policy action-evidence hash")
        request_hash = require_sha256(
            self.replay_request_hash, "pseudo policy replay-request hash"
        )
        label_hash = require_sha256(
            self.terminal_label_hash, "pseudo policy terminal-label hash"
        )
        menu_hash = require_sha256(
            self.method_menu_hash, "pseudo policy method-menu hash"
        )
        oracle_hash = require_sha256(self.oracle_hash, "pseudo policy oracle hash")
        invalid_selected = any(action not in ACTION_IDS for action in selected)
        directions = (
            ()
            if invalid_selected
            else tuple(action.split("::", 1)[1] for action in selected)
        )
        if (
            not isinstance(self.scope, PseudoReplayScope)
            or method not in METHOD_IDS
            or selected != tuple(sorted(set(selected)))
            or invalid_selected
            or len(selected) > len(DIRECTIONS)
            or len(set(directions)) != len(directions)
            or not all(math.isfinite(value) for value in values)
            or values[3] < 0.0
            or evidence_hashes != tuple(sorted(set(evidence_hashes)))
            or len(evidence_hashes) != len(ACTION_IDS)
            or (method == P_PROTECTED and selected)
            or (not selected and values[:3] != (0.0, 0.0, 0.0))
        ):
            raise ProtocolError("SCALE-BP pseudo route-policy evidence drifted.")
        payload = {
            "schema_version": "scale_bp_pseudo_route_policy_evidence_v2",
            "scope_hash": self.scope.scope_hash,
            "method_id": method,
            "selected_action_ids": selected,
            "realized_bacc_gain": values[0],
            "realized_brier_loss_delta": values[1],
            "realized_log_loss_delta": values[2],
            "oracle_bacc_gain": values[3],
            "decision_hash": decision_hash,
            "composition_hash": composition_hash,
            "action_evidence_hashes": evidence_hashes,
            "replay_request_hash": request_hash,
            "terminal_label_hash": label_hash,
            "method_menu_hash": menu_hash,
            "oracle_hash": oracle_hash,
            "pair_aware_policy_replay": True,
            "terminal_labels_reduced_to_aggregate_only": True,
        }
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "realized_bacc_gain", values[0])
        object.__setattr__(self, "realized_brier_loss_delta", values[1])
        object.__setattr__(self, "realized_log_loss_delta", values[2])
        object.__setattr__(self, "oracle_bacc_gain", values[3])
        object.__setattr__(self, "decision_hash", decision_hash)
        object.__setattr__(self, "composition_hash", composition_hash)
        object.__setattr__(self, "action_evidence_hashes", evidence_hashes)
        object.__setattr__(self, "replay_request_hash", request_hash)
        object.__setattr__(self, "terminal_label_hash", label_hash)
        object.__setattr__(self, "method_menu_hash", menu_hash)
        object.__setattr__(self, "oracle_hash", oracle_hash)
        object.__setattr__(self, "policy_hash", canonical_hash(payload))

    @property
    def outer_center(self) -> str:
        return self.scope.outer_center

    @property
    def pseudo_center(self) -> str:
        return self.scope.pseudo_center

    @property
    def case_id(self) -> str:
        return self.scope.held_case_id


def _issue_action_evidence(**kwargs: object) -> PseudoRouteActionEvidence:
    return PseudoRouteActionEvidence(
        **kwargs,
        _factory_token=_EVIDENCE_FACTORY_TOKEN,
    )


def _issue_policy_evidence(**kwargs: object) -> PseudoRoutePolicyEvidence:
    return PseudoRoutePolicyEvidence(
        **kwargs,
        _factory_token=_EVIDENCE_FACTORY_TOKEN,
    )


__all__ = ("PseudoRouteActionEvidence", "PseudoRoutePolicyEvidence")
