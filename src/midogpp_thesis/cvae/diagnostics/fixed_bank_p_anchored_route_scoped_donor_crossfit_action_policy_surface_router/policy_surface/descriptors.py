"""Fixed, low-capacity policy-response descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....protocol import ProtocolError
from ..identity import ACTION_STRATA, METRICS, canonical_hash
from .contracts import PrefixCell


POLICY_FEATURE_NAMES = (
    "predicted_favorable_utility",
    "normalized_depth",
    "max_positive_candidate_share",
    *tuple(
        f"stratum_share__{family}__{direction}"
        for family, direction in ACTION_STRATA
    ),
)


@dataclass(frozen=True)
class PolicyDescriptor:
    cell_hash: str
    metric: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    descriptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        metric = str(self.metric)
        names = tuple(str(value) for value in self.feature_names)
        values = tuple(float(value) for value in self.feature_values)
        if (
            metric not in METRICS
            or names != POLICY_FEATURE_NAMES
            or len(values) != len(names)
            or not np.isfinite(np.asarray(values, dtype=np.float64)).all()
        ):
            raise ProtocolError("P-DCAPS policy descriptor drifted.")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(
            self,
            "descriptor_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_descriptor_v1",
                    "cell_hash": self.cell_hash,
                    "metric": metric,
                    "feature_names": names,
                    "feature_values": values,
                }
            ),
        )

    def as_array(self) -> np.ndarray:
        result = np.ascontiguousarray(self.feature_values, dtype=np.float64)
        result.setflags(write=False)
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_descriptor_v1",
            "cell_hash": self.cell_hash,
            "metric": self.metric,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "descriptor_hash": self.descriptor_hash,
        }


def descriptor_for_metric(cell: PrefixCell, metric: str) -> PolicyDescriptor:
    metric_name = str(metric)
    if metric_name not in METRICS:
        raise ProtocolError("P-DCAPS policy descriptor metric drifted.")
    metric_index = METRICS.index(metric_name)
    predicted = cell.predicted_utility.as_tuple()[metric_index]
    return PolicyDescriptor(
        cell.cell_hash,
        metric_name,
        POLICY_FEATURE_NAMES,
        (
            float(predicted),
            cell.normalized_depth,
            cell.max_positive_candidate_share,
            *cell.stratum_proportions,
        ),
    )


__all__ = ("POLICY_FEATURE_NAMES", "PolicyDescriptor", "descriptor_for_metric")
