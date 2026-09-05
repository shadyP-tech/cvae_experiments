"""Fail-closed HARP v19 label-capability phase state machine."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError


PHASE_ORDER = (
    "AUTHORITY_ADMISSION",
    "LABEL_FREE_SOURCE_TARGET_PHYSICAL_MENUS",
    "SOURCE_TARGET_MENUS_SEALED",
    "FIXED_BANK_INDEPENDENCE_ATTESTED",
    "SOURCE_TRAIN_LABEL_CAPABILITIES_OPENED",
    "POOLED_SOURCE_ROUTER_FIT",
    "SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION",
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

    @property
    def support_labels_opened(self) -> bool:
        """Scientific name for the retained generic development-label flag."""

        return self.development_labels_opened

    def advance(self, phase: str) -> None:
        if phase not in PHASE_ORDER:
            raise ProtocolError("HARP v19 runner phase is unknown.")
        expected = PHASE_ORDER[len(self.observed)] if len(self.observed) < len(PHASE_ORDER) else None
        if phase != expected:
            raise ProtocolError(
                f"HARP v19 phase order drifted: expected {expected}, observed {phase}."
            )
        if phase == "SOURCE_TRAIN_LABEL_CAPABILITIES_OPENED":
            self.development_labels_opened = True
        if phase == "EVALUATION_LABELS_OPENED":
            if not self.development_labels_opened:
                raise ProtocolError("HARP v19 evaluation capability preceded development.")
            self.evaluation_labels_opened = True
        if phase in {
            "AUTHORITY_ADMISSION",
            "LABEL_FREE_SOURCE_TARGET_PHYSICAL_MENUS",
            "SOURCE_TARGET_MENUS_SEALED",
            "FIXED_BANK_INDEPENDENCE_ATTESTED",
        } and (self.development_labels_opened or self.evaluation_labels_opened):
            raise ProtocolError("HARP v19 label-free phase observed an open label capability.")
        if phase in {
            "POOLED_SOURCE_ROUTER_FIT",
            "SOURCE_ONLY_POLICY_RISK_COVERAGE_ADMISSION",
            "TARGET_ACTIONS_COMPLETE",
            "PRELABEL_ROUTES_DURABLE",
            "FRESH_RECONSTRUCTIONS_COMPLETE",
            "FROZEN_ROUTE_SEAL",
        } and self.evaluation_labels_opened:
            raise ProtocolError("HARP v19 prelabel phase observed evaluation labels.")
        self.observed.append(phase)


__all__ = ("PHASE_ORDER", "PhaseLedger")
