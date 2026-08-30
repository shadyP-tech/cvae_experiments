"""Own-source, case-equal calibration for optional HARP energy."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


ROBUST_SCALE_FACTOR = 1.4826


@dataclass(frozen=True, order=True)
class ReplicaCalibration:
    source_center: str
    training_seed: int
    own_source_location: float
    own_source_scale: float
    query_case_equal_mean: float
    calibrated_z: float
    own_source_case_count: int
    query_case_count: int


@dataclass(frozen=True)
class CompatibilityCalibration:
    replicas: tuple[ReplicaCalibration, ...]
    mean_z_by_source: Mapping[str, float]
    training_seeds: tuple[int, ...]
    scale_floor: float
    labels_consumed: bool = False

    def __post_init__(self) -> None:
        seeds = tuple(int(value) for value in self.training_seeds)
        rows = tuple(sorted(self.replicas))
        values = {str(key): float(value) for key, value in self.mean_z_by_source.items()}
        if (
            not rows
            or not seeds
            or len(seeds) != len(set(seeds))
            or not values
            or any(not np.isfinite(value) for value in values.values())
            or not np.isfinite(float(self.scale_floor))
            or float(self.scale_floor) <= 0.0
            or bool(self.labels_consumed)
        ):
            raise ProtocolError("HARP compatibility calibration is invalid.")
        expected = {(source, seed) for source in values for seed in seeds}
        observed = {(row.source_center, row.training_seed) for row in rows}
        if observed != expected:
            raise ProtocolError("HARP calibration lacks its source/seed Cartesian product.")
        object.__setattr__(self, "replicas", rows)
        object.__setattr__(self, "mean_z_by_source", MappingProxyType(values))
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(self, "scale_floor", float(self.scale_floor))


def calibrate_own_source_energy(
    query_case_energy_by_replica: Mapping[tuple[str, int], Mapping[str, float]],
    own_source_case_energy_by_replica: Mapping[tuple[str, int], Mapping[str, float]],
    *,
    candidate_sources: Sequence[str],
    training_seeds: Sequence[int],
    scale_floor: float = 1e-6,
) -> CompatibilityCalibration:
    """Calibrate every fixed replica on its own source cases.

    The function accepts only case-level energy maps; labels and row counts are
    absent from its interface, preventing label-weighted calibration.
    """

    sources = tuple(sorted(str(value) for value in candidate_sources))
    seeds = tuple(int(value) for value in training_seeds)
    floor = float(scale_floor)
    expected = {(source, seed) for source in sources for seed in seeds}
    query = _normalize(query_case_energy_by_replica)
    own = _normalize(own_source_case_energy_by_replica)
    if (
        not sources
        or len(sources) != len(set(sources))
        or not seeds
        or len(seeds) != len(set(seeds))
        or set(query) != expected
        or set(own) != expected
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ProtocolError("HARP own-source calibration inventory drifted.")
    rows: list[ReplicaCalibration] = []
    for key in sorted(expected):
        query_values = _case_values(query[key], "query")
        own_values = _case_values(own[key], "own-source")
        location = float(np.median(own_values))
        raw_mad = float(np.median(np.abs(own_values - location)))
        robust = ROBUST_SCALE_FACTOR * raw_mad
        sample_std = float(np.std(own_values, ddof=1)) if len(own_values) > 1 else 0.0
        scale = robust if robust > floor else sample_std if sample_std > floor else floor
        query_mean = float(np.mean(query_values, dtype=np.float64))
        rows.append(
            ReplicaCalibration(
                source_center=key[0],
                training_seed=key[1],
                own_source_location=location,
                own_source_scale=scale,
                query_case_equal_mean=query_mean,
                calibrated_z=(query_mean - location) / scale,
                own_source_case_count=len(own_values),
                query_case_count=len(query_values),
            )
        )
    mean_z = {
        source: float(
            np.mean(
                [row.calibrated_z for row in rows if row.source_center == source],
                dtype=np.float64,
            )
        )
        for source in sources
    }
    return CompatibilityCalibration(
        replicas=tuple(rows),
        mean_z_by_source=mean_z,
        training_seeds=seeds,
        scale_floor=floor,
    )


def _normalize(
    values: Mapping[tuple[str, int], Mapping[str, float]],
) -> dict[tuple[str, int], Mapping[str, float]]:
    result: dict[tuple[str, int], Mapping[str, float]] = {}
    for raw_key, cases in values.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ProtocolError("HARP compatibility replica key is malformed.")
        key = (str(raw_key[0]), int(raw_key[1]))
        if key in result or not key[0] or not isinstance(cases, Mapping):
            raise ProtocolError("HARP compatibility replica inventory is malformed.")
        result[key] = cases
    return result


def _case_values(values: Mapping[str, float], role: str) -> np.ndarray:
    normalized = {str(key): float(value) for key, value in values.items()}
    array = np.asarray([normalized[key] for key in sorted(normalized)], dtype=np.float64)
    if not len(array) or any(not key for key in normalized) or not np.isfinite(array).all():
        raise ProtocolError(f"HARP {role} case energies are invalid.")
    return array


__all__ = (
    "ROBUST_SCALE_FACTOR",
    "CompatibilityCalibration",
    "ReplicaCalibration",
    "calibrate_own_source_energy",
)
