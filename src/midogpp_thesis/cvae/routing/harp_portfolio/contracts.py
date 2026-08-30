"""Frozen, outcome-free contracts for HARP portfolio selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct

from ...protocol import ProtocolError
from ..harp_action_model import HarpActionScore


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_id(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a canonical nonempty string.")
    return value


@dataclass(frozen=True, kw_only=True)
class HarpPolicyConfig:
    kappa_gain: float = 1.0
    kappa_loss: float = 1.0
    gain_threshold: float = 0.0
    brier_noninferiority_margin: float = 0.0
    log_loss_noninferiority_margin: float = 0.0
    min_donor_count: int = 4
    min_paired_case_count: int = 16
    max_leverage: float = 1.0
    min_compatibility_shrinkage: float = 0.0

    def __post_init__(self) -> None:
        for name in ("kappa_gain", "kappa_loss", "gain_threshold", "brier_noninferiority_margin", "log_loss_noninferiority_margin", "max_leverage", "min_compatibility_shrinkage"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.kappa_gain < 0 or self.kappa_loss < 0 or self.max_leverage < 0 or not 0 <= self.min_compatibility_shrinkage <= 1:
            raise ProtocolError("HARP policy uncertainty and gate values are invalid.")
        if type(self.min_donor_count) is not int or self.min_donor_count < 1 or type(self.min_paired_case_count) is not int or self.min_paired_case_count < 1:
            raise ProtocolError("HARP support gates must be positive integers.")


@dataclass(frozen=True)
class HarpConservativeAction:
    score: HarpActionScore
    gain_lower: float
    brier_upper: float
    log_loss_upper: float
    maximum_leverage: float
    eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HarpPortfolioDecision:
    outer_target_id: str
    case_id: str
    sample_id: str
    baseline_probability_bytes: bytes
    output_probability_bytes: bytes
    selected_source_id: str | None
    selected_lambda: float | None
    routed: bool
    reason: str
    gain_lower: float | None
    brier_upper: float | None
    log_loss_upper: float | None
    prediction_seal_hash: str
    ensemble_receipt_hash: str

    def __post_init__(self) -> None:
        for name in ("outer_target_id", "case_id", "sample_id", "reason"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        for name in ("prediction_seal_hash", "ensemble_receipt_hash"):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ProtocolError(f"{name} must be a lowercase SHA-256 identity.")
        if type(self.baseline_probability_bytes) is not bytes or len(self.baseline_probability_bytes) != 8 or type(self.output_probability_bytes) is not bytes or len(self.output_probability_bytes) != 8:
            raise ProtocolError("Portfolio probabilities must be exact float64 byte strings.")
        baseline = struct.unpack("<d", self.baseline_probability_bytes)[0]
        probability = struct.unpack("<d", self.output_probability_bytes)[0]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (baseline, probability)):
            raise ProtocolError("Portfolio probabilities must lie in [0, 1].")
        if not self.routed:
            if self.output_probability_bytes != self.baseline_probability_bytes:
                raise ProtocolError("Exact-B fallback must preserve baseline bytes.")
            if self.selected_source_id is not None or self.selected_lambda is not None:
                raise ProtocolError("Exact-B fallback cannot name an expert action.")
            if any(value is not None for value in (self.gain_lower, self.brier_upper, self.log_loss_upper)):
                raise ProtocolError("Exact-B fallback cannot retain admitted-action bounds.")
        else:
            if (
                type(self.selected_source_id) is not str
                or not self.selected_source_id
                or self.selected_source_id != self.selected_source_id.strip()
                or self.selected_lambda not in (0.25, 0.5, 0.75, 1.0)
                or any(value is None or not math.isfinite(float(value)) for value in (self.gain_lower, self.brier_upper, self.log_loss_upper))
            ):
                raise ProtocolError("A routed HARP decision requires one finite admitted action.")

    @property
    def row_key(self) -> tuple[str, str, str]:
        return (self.outer_target_id, self.case_id, self.sample_id)


__all__ = ("HarpConservativeAction", "HarpPolicyConfig", "HarpPortfolioDecision")
