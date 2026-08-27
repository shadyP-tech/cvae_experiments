"""Transient terminal-label input contract for one pseudo-case replay."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine import CaseRouteRequest
from ..hashing import canonical_hash
from ..identity import ACTION_IDS, METHOD_MENU
from ..protocol import ProtocolError
from ..replay_scope import PseudoReplayScope
from .terminal_labels import TerminalCaseLabelReceipt


def method_menu_hash() -> str:
    return canonical_hash(
        {
            "schema_version": "scale_bp_replay_method_menu_v1",
            "method_ids": METHOD_MENU,
            "action_ids": ACTION_IDS,
            "primary_is_boundary_projected": True,
            "full_endpoint_is_sensitivity_only": True,
        }
    )


@dataclass(frozen=True, slots=True)
class PseudoCaseReplayRequest:
    """One H/J/d route plus ephemeral held-case labels and fixed denominators."""

    route_request: CaseRouteRequest
    terminal_label_receipt: TerminalCaseLabelReceipt
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route = self.route_request
        receipt = self.terminal_label_receipt
        if (
            not isinstance(route, CaseRouteRequest)
            or not isinstance(route.route_scope, PseudoReplayScope)
            or not isinstance(receipt, TerminalCaseLabelReceipt)
            or receipt.scope.scope_hash != route.route_scope.scope_hash
            or len(receipt.terminal_labels) != len(route.portfolio_probabilities)
            or any(row.action_id not in ACTION_IDS for row in route.action_inputs)
        ):
            raise ProtocolError("SCALE-BP pseudo replay request drifted.")
        payload = {
            "schema_version": "scale_bp_pseudo_case_replay_request_v2",
            "route_request_hash": route.request_hash,
            "scope_hash": route.route_scope.scope_hash,
            "terminal_label_receipt_hash": receipt.receipt_hash,
            "terminal_label_hash": receipt.terminal_label_hash,
            "center_population_label_hash": receipt.center_population_label_hash,
            "positive_denominator": receipt.positive_denominator,
            "negative_denominator": receipt.negative_denominator,
            "row_denominator": receipt.row_denominator,
            "method_menu_hash": method_menu_hash(),
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "request_hash", canonical_hash(payload))

    @property
    def scope(self) -> PseudoReplayScope:
        scope = self.route_request.route_scope
        assert isinstance(scope, PseudoReplayScope)
        return scope

    @property
    def terminal_labels(self) -> tuple[int, ...]:
        return self.terminal_label_receipt.terminal_labels

    @property
    def terminal_label_hash(self) -> str:
        return self.terminal_label_receipt.terminal_label_hash

    @property
    def positive_denominator(self) -> int:
        return self.terminal_label_receipt.positive_denominator

    @property
    def negative_denominator(self) -> int:
        return self.terminal_label_receipt.negative_denominator

    @property
    def row_denominator(self) -> int:
        return self.terminal_label_receipt.row_denominator

    def sealed_payload(self) -> dict[str, object]:
        """Return the durable request receipt without exposing raw labels."""

        return {
            "schema_version": "scale_bp_pseudo_case_replay_request_v2",
            "route_request_hash": self.route_request.request_hash,
            "scope_hash": self.scope.scope_hash,
            "terminal_label_receipt_hash": self.terminal_label_receipt.receipt_hash,
            "terminal_label_hash": self.terminal_label_hash,
            "center_population_label_hash": (
                self.terminal_label_receipt.center_population_label_hash
            ),
            "positive_denominator": self.positive_denominator,
            "negative_denominator": self.negative_denominator,
            "row_denominator": self.row_denominator,
            "method_menu_hash": method_menu_hash(),
            "raw_labels_persisted": False,
            "request_hash": self.request_hash,
        }

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise ProtocolError(
            "SCALE-BP transient terminal-label requests may not be serialized."
        )


__all__ = ("PseudoCaseReplayRequest", "method_menu_hash")
