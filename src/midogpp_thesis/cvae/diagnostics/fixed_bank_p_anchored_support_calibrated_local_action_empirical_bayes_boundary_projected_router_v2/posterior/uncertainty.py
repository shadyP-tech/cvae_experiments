"""Descriptive pre-argmax envelopes; no confidence or coverage claim."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector
from .contracts import ScaleVector
from .empirical_bayes import ActionEstimate


@dataclass(frozen=True, slots=True)
class DescriptiveBounds:
    action_id: str
    mean: MetricVector
    lower: MetricVector
    upper: MetricVector
    transport_rmse: ScaleVector
    heterogeneity: ScaleVector
    estimator_se: ScaleVector
    local_oof_rmse: ScaleVector
    selection_multiplier: float
    structural_noop: bool
    estimate_hash: str
    bounds_hash: str = field(init=False)

    def __post_init__(self) -> None:
        mean = self.mean.as_array()
        lower = self.lower.as_array()
        upper = self.upper.as_array()
        if (
            self.action_id not in ACTION_IDS
            or not math.isfinite(self.selection_multiplier)
            or self.selection_multiplier <= 0.0
            or not self.estimate_hash
            or np.any(lower > mean)
            or np.any(mean > upper)
        ):
            raise GovernanceError("SCALE-BP v2 descriptive bounds drifted.")
        if self.structural_noop and not (
            self.mean == self.lower == self.upper == MetricVector.zeros()
        ):
            raise GovernanceError("SCALE-BP v2 no-op uncertainty is nonzero.")
        object.__setattr__(
            self,
            "bounds_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_descriptive_bounds_v1",
                    "action_id": self.action_id,
                    "mean": self.mean.to_payload(),
                    "lower": self.lower.to_payload(),
                    "upper": self.upper.to_payload(),
                    "transport_rmse": self.transport_rmse.to_payload(),
                    "heterogeneity": self.heterogeneity.to_payload(),
                    "estimator_se": self.estimator_se.to_payload(),
                    "local_oof_rmse": self.local_oof_rmse.to_payload(),
                    "selection_multiplier": self.selection_multiplier,
                    "structural_noop": self.structural_noop,
                    "estimate_hash": self.estimate_hash,
                    "computed_pre_argmax": True,
                    "confidence_or_conformal_claimed": False,
                }
            ),
        )


def build_preargmax_bounds(
    estimates: Sequence[ActionEstimate],
    *,
    base_multiplier: float = 1.2815515655446004,
) -> tuple[DescriptiveBounds, ...]:
    rows = tuple(estimates)
    if tuple(row.action_id for row in rows) != ACTION_IDS:
        raise GovernanceError("SCALE-BP v2 uncertainty requires the complete action menu.")
    if not math.isfinite(base_multiplier) or base_multiplier <= 0.0:
        raise GovernanceError("SCALE-BP v2 uncertainty multiplier is invalid.")
    multiplier = max(float(base_multiplier), math.sqrt(2.0 * math.log(len(rows))))
    result: list[DescriptiveBounds] = []
    for row in rows:
        if row.structural_noop:
            zero = MetricVector.zeros()
            zero_scale = ScaleVector.zeros()
            result.append(
                DescriptiveBounds(
                    row.action_id,
                    zero,
                    zero,
                    zero,
                    zero_scale,
                    zero_scale,
                    zero_scale,
                    zero_scale,
                    multiplier,
                    True,
                    row.estimate_hash,
                )
            )
            continue
        transport = np.asarray(row.transport_rmse.as_tuple(), dtype=np.float64)
        heterogeneity = np.maximum(
            np.asarray(row.donor_heterogeneity.as_tuple(), dtype=np.float64),
            np.asarray(row.local_fold_heterogeneity.as_tuple(), dtype=np.float64),
        )
        estimator = np.asarray(row.combined_estimator_se.as_tuple(), dtype=np.float64)
        local_oof = np.asarray(row.local_oof_rmse.as_tuple(), dtype=np.float64)
        descriptive_scale = np.maximum.reduce((transport, heterogeneity, local_oof)) + estimator
        half_width = multiplier * descriptive_scale
        mean = row.mean.as_array()
        result.append(
            DescriptiveBounds(
                row.action_id,
                row.mean,
                MetricVector.from_array(mean - half_width),
                MetricVector.from_array(mean + half_width),
                row.transport_rmse,
                ScaleVector.from_values(heterogeneity),
                row.combined_estimator_se,
                row.local_oof_rmse,
                multiplier,
                False,
                row.estimate_hash,
            )
        )
    return tuple(result)


__all__ = ("DescriptiveBounds", "build_preargmax_bounds")
