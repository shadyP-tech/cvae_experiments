"""Fail-closed HARP v3 label-capability phase state machine."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError


PHASE_ORDER = (
    "AUTHORITY_ADMISSION",
    "LABEL_FREE_PHYSICAL_MENU",
    "DEVELOPMENT_SURFACE_SEALED",
    "DEVELOPMENT_LABELS_OPENED",
    "SOURCE_ONLY_MODEL_FIT",
    "TARGET_ACTIONS_COMPLETE",
    "PRELABEL_ROUTES_DURABLE",
    "FRESH_RECONSTRUCTIONS_COMPLETE",
    "FROZEN_ROUTE_SEAL",
    "EVALUATION_LABELS_OPENED",
    "TERMINAL_DIAGNOSTIC_COMPLETE",
)


@dataclass(slots=True)
class PhaseLedger:
    observed: list[str] = field(default_factory=list)
    development_labels_opened: bool = False
    evaluation_labels_opened: bool = False

    def advance(self, phase: str) -> None:
        if phase not in PHASE_ORDER:
            raise ProtocolError("HARP v3 runner phase is unknown.")
        expected = PHASE_ORDER[len(self.observed)] if len(self.observed) < len(PHASE_ORDER) else None
        if phase != expected:
            raise ProtocolError(
                f"HARP v3 phase order drifted: expected {expected}, observed {phase}."
            )
        if phase == "DEVELOPMENT_LABELS_OPENED":
            self.development_labels_opened = True
        if phase == "EVALUATION_LABELS_OPENED":
            if not self.development_labels_opened:
                raise ProtocolError("HARP v3 evaluation capability preceded development.")
            self.evaluation_labels_opened = True
        if phase in {
            "AUTHORITY_ADMISSION",
            "LABEL_FREE_PHYSICAL_MENU",
            "DEVELOPMENT_SURFACE_SEALED",
        } and (self.development_labels_opened or self.evaluation_labels_opened):
            raise ProtocolError("HARP v3 label-free phase observed an open label capability.")
        if phase in {
            "SOURCE_ONLY_MODEL_FIT",
            "TARGET_ACTIONS_COMPLETE",
            "PRELABEL_ROUTES_DURABLE",
            "FRESH_RECONSTRUCTIONS_COMPLETE",
            "FROZEN_ROUTE_SEAL",
        } and self.evaluation_labels_opened:
            raise ProtocolError("HARP v3 prelabel phase observed evaluation labels.")
        self.observed.append(phase)


__all__ = ("PHASE_ORDER", "PhaseLedger")
