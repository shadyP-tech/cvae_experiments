"""One-way compatibility shrinkage for an already eligible HARP action."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...protocol import ProtocolError


@dataclass(frozen=True)
class CompatibilityShrinkage:
    action_model_eligible: bool
    compatibility_enabled: bool
    calibrated_z: float | None
    original_weight: float
    multiplier: float
    final_weight: float
    abstained: bool
    compatibility_authorized_action: bool = False


def shrink_eligible_weight(
    *,
    action_model_eligible: bool,
    original_weight: float,
    calibrated_z: float | None,
    enabled: bool,
    soft_limit: float = 2.0,
    hard_limit: float = 4.0,
) -> CompatibilityShrinkage:
    """Shrink toward exact-B; compatibility can never authorize routing."""

    weight = float(original_weight)
    soft = float(soft_limit)
    hard = float(hard_limit)
    if (
        not math.isfinite(weight)
        or weight < 0.0
        or weight > 1.0
        or not math.isfinite(soft)
        or not math.isfinite(hard)
        or soft < 0.0
        or hard <= soft
    ):
        raise ProtocolError("HARP compatibility shrinkage parameters are invalid.")
    if not action_model_eligible:
        multiplier = 0.0
    elif not enabled:
        multiplier = 1.0
    else:
        if calibrated_z is None or not math.isfinite(float(calibrated_z)):
            raise ProtocolError("Enabled HARP compatibility requires finite calibrated z.")
        z_value = float(calibrated_z)
        multiplier = max(0.0, min(1.0, (hard - z_value) / (hard - soft)))
    final = min(weight, weight * multiplier)
    return CompatibilityShrinkage(
        action_model_eligible=bool(action_model_eligible),
        compatibility_enabled=bool(enabled),
        calibrated_z=(None if calibrated_z is None else float(calibrated_z)),
        original_weight=weight,
        multiplier=multiplier,
        final_weight=final,
        abstained=final == 0.0,
    )


__all__ = ("CompatibilityShrinkage", "shrink_eligible_weight")
