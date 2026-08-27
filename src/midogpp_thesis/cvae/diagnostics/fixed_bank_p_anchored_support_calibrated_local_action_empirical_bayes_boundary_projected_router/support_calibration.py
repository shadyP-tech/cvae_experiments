"""Route-local OOF uncertainty and label-free support-overlap calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .hashing import canonical_hash
from .influence.contracts import ActionDescriptor, MetricStandardError
from .local_residual.contracts import (
    LocalCrossfitResult,
    LocalResidualModel,
    LocalResidualRecord,
)
from .protocol import ProtocolError


MINIMUM_ACTION_MEMBERS = 4
SUPPORT_DISTANCE_QUANTILE = 0.95
SUPPORT_DISTANCE_INFLATION = 1.25
SUPPORT_DISTANCE_FLOOR = 3.0
MINIMUM_BANK_ESS = 2.0


@dataclass(frozen=True, slots=True)
class LocalActionCalibration:
    action_id: str
    member_count: int
    oof_prediction_hashes: tuple[str, ...]
    local_standard_error: MetricStandardError
    target_standardized_distance: float
    support_distance_limit: float
    within_support: bool
    bank_viable: bool
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple(str(value) for value in self.oof_prediction_hashes)
        values = (
            float(self.target_standardized_distance),
            float(self.support_distance_limit),
        )
        if (
            not self.action_id
            or self.member_count < 0
            or len(hashes) != len(set(hashes))
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
            or type(self.within_support) is not bool
            or type(self.bank_viable) is not bool
            or self.within_support != (
                self.member_count >= MINIMUM_ACTION_MEMBERS
                and values[0] <= values[1]
            )
        ):
            raise ProtocolError("SCALE-BP local action calibration drifted.")
        payload = {
            "schema_version": "scale_bp_local_action_calibration_v1",
            "action_id": self.action_id,
            "member_count": self.member_count,
            "oof_prediction_hashes": hashes,
            "local_standard_error": self.local_standard_error.to_payload(),
            "target_standardized_distance": values[0],
            "support_distance_limit": values[1],
            "within_support": self.within_support,
            "bank_viable": self.bank_viable,
            "minimum_action_members": MINIMUM_ACTION_MEMBERS,
            "support_distance_quantile": SUPPORT_DISTANCE_QUANTILE,
            "support_distance_inflation": SUPPORT_DISTANCE_INFLATION,
            "support_distance_floor": SUPPORT_DISTANCE_FLOOR,
            "minimum_bank_ess": MINIMUM_BANK_ESS,
            "labels_used_for_overlap": False,
        }
        object.__setattr__(self, "oof_prediction_hashes", hashes)
        object.__setattr__(self, "target_standardized_distance", values[0])
        object.__setattr__(self, "support_distance_limit", values[1])
        object.__setattr__(self, "calibration_hash", canonical_hash(payload))


def calibrate_local_action(
    descriptor: ActionDescriptor,
    *,
    records: object,
    crossfit: LocalCrossfitResult,
    final_model: LocalResidualModel,
) -> LocalActionCalibration:
    """Estimate action-specific OOF error and a frozen feature-support gate."""

    rows = tuple(records)  # type: ignore[arg-type]
    selected_rows = tuple(row for row in rows if row.action_id == descriptor.action_id)
    predictions = tuple(
        row for row in crossfit.predictions if row.action_id == descriptor.action_id
    )
    if (
        any(not isinstance(row, LocalResidualRecord) for row in rows)
        or crossfit.route_scope_hash != final_model.route_scope_hash
        or descriptor.feature_names != final_model.feature_names
        or {row.record_hash for row in selected_rows}
        != {row.record_hash for row in predictions}
    ):
        raise ProtocolError("SCALE-BP local action calibration lineage drifted.")
    member_count = len({row.member_id for row in selected_rows})
    if predictions:
        errors = np.asarray(
            [row.residual_error.as_tuple() for row in predictions], dtype=np.float64
        )
        standard_error = MetricStandardError.from_iterable(
            np.sqrt(np.mean(errors * errors, axis=0, dtype=np.float64))
        )
    else:
        standard_error = MetricStandardError.zeros()

    mean = np.asarray(final_model.feature_mean, dtype=np.float64)
    scale = np.asarray(final_model.feature_scale, dtype=np.float64)
    target = (np.asarray(descriptor.values, dtype=np.float64) - mean) / scale
    target_distance = float(np.sqrt(np.mean(target * target, dtype=np.float64)))
    if selected_rows:
        support = np.asarray(
            [
                (np.asarray(row.descriptor.values, dtype=np.float64) - mean) / scale
                for row in selected_rows
            ],
            dtype=np.float64,
        )
        distances = np.sqrt(np.mean(support * support, axis=1, dtype=np.float64))
        limit = max(
            SUPPORT_DISTANCE_FLOOR,
            SUPPORT_DISTANCE_INFLATION
            * float(np.quantile(distances, SUPPORT_DISTANCE_QUANTILE, method="higher")),
        )
    else:
        limit = SUPPORT_DISTANCE_FLOOR
    try:
        bank_index = descriptor.feature_names.index("bank_ess")
    except ValueError as exc:
        raise ProtocolError("SCALE-BP descriptor lacks frozen bank ESS.") from exc
    bank_viable = descriptor.values[bank_index] >= MINIMUM_BANK_ESS
    return LocalActionCalibration(
        action_id=descriptor.action_id,
        member_count=member_count,
        oof_prediction_hashes=tuple(
            sorted(row.prediction_hash for row in predictions)
        ),
        local_standard_error=standard_error,
        target_standardized_distance=target_distance,
        support_distance_limit=limit,
        within_support=(
            member_count >= MINIMUM_ACTION_MEMBERS and target_distance <= limit
        ),
        bank_viable=bank_viable,
    )


__all__ = (
    "LocalActionCalibration",
    "MINIMUM_ACTION_MEMBERS",
    "MINIMUM_BANK_ESS",
    "calibrate_local_action",
)
