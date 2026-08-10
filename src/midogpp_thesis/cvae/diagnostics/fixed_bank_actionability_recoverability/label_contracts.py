"""Capability-scoped binary labels local to this diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...protocol import ProtocolError
from .constants import MIDOGPP_CENTERS
from .hashing import nonempty_text


LabelScope = Literal["loco_donor", "target_support", "terminal_evaluation"]


@dataclass(frozen=True, order=True)
class BinaryLabel:
    target_center: str
    case_id: str
    sample_id: str
    label: int
    label_scope: LabelScope

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Scoped label uses an unknown MIDOG++ center.")
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ProtocolError("Scoped label must be integer zero or one.")
        if self.label_scope not in ("loco_donor", "target_support", "terminal_evaluation"):
            raise ProtocolError("Unknown actionability label capability scope.")

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


__all__ = ("BinaryLabel", "LabelScope")
