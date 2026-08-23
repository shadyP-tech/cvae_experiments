"""Immutable aggregate-only P-DCAPS terminal result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ....protocol import ProtocolError
from ..identity import (
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    canonical_hash,
    require_sha256,
)


@dataclass(frozen=True)
class TerminalEvaluationResult:
    method_rows: tuple[Mapping[str, object], ...]
    center_rows: tuple[Mapping[str, object], ...]
    case_diagnostic_rows: tuple[Mapping[str, object], ...]
    selection_control: Mapping[str, object]
    router_diagnostics: Mapping[str, object]
    preterminal_seal_hash: str
    label_identity_hash: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        preterminal = require_sha256(
            self.preterminal_seal_hash, "terminal preterminal seal"
        )
        label_identity = require_sha256(
            self.label_identity_hash, "terminal label identity"
        )
        if not self.method_rows or not self.center_rows:
            raise ProtocolError("P-DCAPS terminal result inventory is empty.")
        object.__setattr__(self, "preterminal_seal_hash", preterminal)
        object.__setattr__(self, "label_identity_hash", label_identity)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_terminal_evaluation_result_v1",
                    "method_rows": list(self.method_rows),
                    "center_rows": list(self.center_rows),
                    "case_diagnostic_rows": list(self.case_diagnostic_rows),
                    "selection_control": dict(self.selection_control),
                    "router_diagnostics": dict(self.router_diagnostics),
                    "preterminal_seal_hash": preterminal,
                    "label_identity_hash": label_identity,
                    "publication_status": PUBLICATION_STATUS,
                    "terminal_decision": TERMINAL_DECISION,
                    "raw_labels_persisted": False,
                    "routing_authorized": False,
                    "promotion_allowed": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_terminal_evaluation_result_v1",
            "method_rows": [dict(row) for row in self.method_rows],
            "center_rows": [dict(row) for row in self.center_rows],
            "case_diagnostic_rows": [
                dict(row) for row in self.case_diagnostic_rows
            ],
            "selection_control": dict(self.selection_control),
            "router_diagnostics": dict(self.router_diagnostics),
            "preterminal_seal_hash": self.preterminal_seal_hash,
            "label_identity_hash": self.label_identity_hash,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "raw_labels_persisted": False,
            "routing_authorized": False,
            "promotion_allowed": False,
            "result_hash": self.result_hash,
        }


__all__ = ("TerminalEvaluationResult",)
