"""Strict nullable-statistic schema for P-DCAPS v3 admission.

Known mathematical undefined states are represented as JSON ``null`` plus a
closed reason code. Unexpected nonfinite values remain protocol violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .identity import canonical_hash


NULLABLE_STATISTIC_SCHEMA = "pdcaps_v3_nullable_admission_statistic_v1"

ADMISSION_STATISTIC_NAMES = (
    "routed_policy_count",
    "bacc_spearman",
    "brier_spearman",
    "log_spearman",
    "equal_center_realized_bacc",
    "joint_safe_routed_rate",
    "legacy_joint_safe_routed_rate",
    "absolute_oracle_regret",
    "legacy_absolute_oracle_regret",
    "normalized_oracle_gap",
    "legacy_normalized_oracle_gap",
)

CONSTANT_RANK_UNDEFINED_REASON = "CONSTANT_RANK_INPUT"
DENOMINATOR_UNDEFINED_REASON = "INVALID_OR_ZERO_DENOMINATOR"
UNDEFINED_REASONS = (
    CONSTANT_RANK_UNDEFINED_REASON,
    DENOMINATOR_UNDEFINED_REASON,
)


def finite_float(value: object, role: str) -> float:
    """Return a finite scalar while rejecting booleans and IEEE sentinels."""

    if isinstance(value, (bool, np.bool_)):
        raise ProtocolError(f"P-DCAPS v3 {role} is not a finite float.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(
            f"P-DCAPS v3 {role} is not a finite float."
        ) from exc
    if not math.isfinite(result):
        raise ProtocolError(f"P-DCAPS v3 {role} is not a finite float.")
    return result


def strict_bool(value: object, role: str) -> bool:
    """Reject truthy substitutes so persisted booleans stay canonical."""

    if type(value) is not bool:
        raise ProtocolError(f"P-DCAPS v3 {role} is not boolean.")
    return value


@dataclass(frozen=True)
class NullableStatistic:
    """One JSON-safe admission statistic with an explicit defined-state."""

    name: str
    value: float | None
    defined: bool
    undefined_reason: str | None
    statistic_hash: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.name)
        if name not in ADMISSION_STATISTIC_NAMES:
            raise ProtocolError("P-DCAPS v3 admission statistic name drifted.")
        defined = strict_bool(self.defined, "statistic defined-state")
        if defined:
            if self.value is None or self.undefined_reason is not None:
                raise ProtocolError(
                    "P-DCAPS v3 defined statistic has nullable-state drift."
                )
            value: float | None = finite_float(
                self.value, f"{name} statistic value"
            )
            reason: str | None = None
        else:
            if self.value is not None:
                raise ProtocolError(
                    "P-DCAPS v3 undefined statistic must persist value=null."
                )
            reason = str(self.undefined_reason)
            if reason not in UNDEFINED_REASONS:
                raise ProtocolError(
                    "P-DCAPS v3 statistic undefined reason drifted."
                )
            value = None
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "defined", defined)
        object.__setattr__(self, "undefined_reason", reason)
        object.__setattr__(
            self,
            "statistic_hash",
            canonical_hash(
                {
                    "schema_version": NULLABLE_STATISTIC_SCHEMA,
                    **self.to_payload(),
                }
            ),
        )

    @classmethod
    def finite(cls, name: str, value: object) -> "NullableStatistic":
        return cls(name, finite_float(value, f"{name} statistic value"), True, None)

    @classmethod
    def undefined(cls, name: str, reason: str) -> "NullableStatistic":
        return cls(name, None, False, reason)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NullableStatistic":
        expected = {"name", "value", "defined", "undefined_reason"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ProtocolError("P-DCAPS v3 nullable statistic schema drifted.")
        return cls(
            str(payload["name"]),
            payload["value"],  # type: ignore[arg-type]
            strict_bool(payload["defined"], "statistic defined-state"),
            (
                None
                if payload["undefined_reason"] is None
                else str(payload["undefined_reason"])
            ),
        )

    def to_payload(self) -> dict[str, object]:
        # These four keys are the frozen persisted nullable-statistic schema.
        return {
            "name": self.name,
            "value": self.value,
            "defined": self.defined,
            "undefined_reason": self.undefined_reason,
        }


__all__ = (
    "ADMISSION_STATISTIC_NAMES",
    "CONSTANT_RANK_UNDEFINED_REASON",
    "DENOMINATOR_UNDEFINED_REASON",
    "NULLABLE_STATISTIC_SCHEMA",
    "NullableStatistic",
)
