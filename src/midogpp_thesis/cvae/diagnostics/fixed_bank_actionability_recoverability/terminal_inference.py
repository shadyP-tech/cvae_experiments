"""Paired center inference and whole-case-cluster bootstrap."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS
from .contracts import CaseConfusionCounts
from .hashing import canonical_hash, finite


@dataclass(frozen=True)
class PairedWholeCaseBootstrap:
    geometry_id: str
    challenger_method: str
    reference_method: str
    replicate_count: int
    seed: int
    observed_equal_center_difference: float
    bootstrap_mean: float
    ci95_lower: float
    ci95_upper: float
    invalid_draw_count: int
    bootstrap_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS or self.challenger_method == self.reference_method:
            raise ProtocolError("Paired bootstrap identity is invalid.")
        if self.replicate_count <= 0 or self.invalid_draw_count < 0:
            raise ProtocolError("Paired bootstrap counts are invalid.")
        for name in (
            "observed_equal_center_difference", "bootstrap_mean", "ci95_lower", "ci95_upper"
        ):
            finite(getattr(self, name), name)
        if self.ci95_lower > self.ci95_upper:
            raise ProtocolError("Bootstrap confidence interval is reversed.")
        object.__setattr__(self, "bootstrap_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_whole_case_bootstrap_v1",
            "geometry_id": self.geometry_id,
            "challenger_method": self.challenger_method,
            "reference_method": self.reference_method,
            "replicate_count": self.replicate_count,
            "seed": self.seed,
            "observed_equal_center_difference": self.observed_equal_center_difference,
            "bootstrap_mean": self.bootstrap_mean,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "invalid_draw_count": self.invalid_draw_count,
            "resampling_unit": "whole_case_within_target_center",
            "aggregation": "equal_target_center",
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "bootstrap_hash": self.bootstrap_hash}


@dataclass(frozen=True)
class TerminalContrast:
    contrast_family: str
    geometry_id: str
    challenger_method: str
    reference_method: str
    center_differences: tuple[tuple[str, float], ...]
    equal_center_difference: float
    center_t_ci95_lower: float
    center_t_ci95_upper: float
    bootstrap: PairedWholeCaseBootstrap
    contrast_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contrast_family not in ("actionability", "recoverability", "secondary"):
            raise ProtocolError("Terminal contrast family is invalid.")
        if self.geometry_id not in GEOMETRY_IDS or tuple(x[0] for x in self.center_differences) != MIDOGPP_CENTERS:
            raise ProtocolError("Terminal contrast center coverage is invalid.")
        mean = finite(self.equal_center_difference, "equal_center_difference")
        expected = math.fsum(value for _center, value in self.center_differences) / len(MIDOGPP_CENTERS)
        if not math.isclose(mean, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ProtocolError("Terminal contrast mean drifted.")
        if (
            self.bootstrap.geometry_id != self.geometry_id
            or self.bootstrap.challenger_method != self.challenger_method
            or self.bootstrap.reference_method != self.reference_method
            or not math.isclose(self.bootstrap.observed_equal_center_difference, mean, abs_tol=1e-12)
        ):
            raise ProtocolError("Terminal contrast/bootstrap mismatch.")
        object.__setattr__(self, "contrast_hash", canonical_hash(self._unhashed()))

    @property
    def contrast_id(self) -> str:
        return f"{self.geometry_id}:{self.challenger_method}-{self.reference_method}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_terminal_contrast_v1",
            "contrast_id": self.contrast_id,
            "contrast_family": self.contrast_family,
            "geometry_id": self.geometry_id,
            "challenger_method": self.challenger_method,
            "reference_method": self.reference_method,
            "center_differences": [list(row) for row in self.center_differences],
            "equal_center_difference": self.equal_center_difference,
            "center_t_ci95_lower": self.center_t_ci95_lower,
            "center_t_ci95_upper": self.center_t_ci95_upper,
            "bootstrap": self.bootstrap.to_payload(),
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "contrast_hash": self.contrast_hash}


def _group(rows: Sequence[CaseConfusionCounts]) -> dict[str, tuple[CaseConfusionCounts, ...]]:
    grouped: dict[str, list[CaseConfusionCounts]] = defaultdict(list)
    for row in rows:
        grouped[row.target_center].append(row)
    result = {center: tuple(sorted(values, key=lambda x: x.case_id)) for center, values in grouped.items()}
    if set(result) != set(MIDOGPP_CENTERS):
        raise ProtocolError("Whole-case contrast lacks a target center.")
    return result


def _bacc(rows: Sequence[CaseConfusionCounts]) -> float:
    positive, negative = sum(x.n_positive for x in rows), sum(x.n_negative for x in rows)
    if positive <= 0 or negative <= 0:
        raise ProtocolError("Pooled center draw lacks a binary class.")
    return 0.5 * (
        sum(x.true_positive for x in rows) / positive
        + sum(x.true_negative for x in rows) / negative
    )


def bootstrap_task(
    task: tuple[str, str, str, tuple[CaseConfusionCounts, ...], tuple[CaseConfusionCounts, ...], int, int, int]
) -> PairedWholeCaseBootstrap:
    geometry, challenger, reference, left_rows, right_rows, replicates, seed, threads = task
    from threadpoolctl import threadpool_limits

    left, right = _group(left_rows), _group(right_rows)
    for center in MIDOGPP_CENTERS:
        if tuple(x.case_id for x in left[center]) != tuple(x.case_id for x in right[center]):
            raise ProtocolError("Bootstrap methods have different whole-case scopes.")
    observed = math.fsum(_bacc(left[c]) - _bacc(right[c]) for c in MIDOGPP_CENTERS) / 9
    generator, values, invalid = np.random.default_rng(seed), [], 0
    with threadpool_limits(limits=threads):
        for _ in range(max(replicates * 100, replicates + 10_000)):
            if len(values) == replicates:
                break
            differences, legal = [], True
            for center in MIDOGPP_CENTERS:
                size = len(left[center])
                indices = generator.integers(0, size, size=size)
                try:
                    differences.append(
                        _bacc(tuple(left[center][int(i)] for i in indices))
                        - _bacc(tuple(right[center][int(i)] for i in indices))
                    )
                except ProtocolError:
                    legal = False
                    break
            if legal:
                values.append(math.fsum(differences) / 9)
            else:
                invalid += 1
    if len(values) != replicates:
        raise ProtocolError("Whole-case bootstrap could not obtain enough legal draws.")
    array = np.asarray(values, dtype=np.float64)
    return PairedWholeCaseBootstrap(
        geometry, challenger, reference, replicates, seed, observed,
        float(array.mean(dtype=np.float64)),
        float(np.quantile(array, .025, method="linear")),
        float(np.quantile(array, .975, method="linear")), invalid,
    )


def build_contrast(
    *, family: str, geometry: str, challenger: object, reference: object,
    bootstrap: PairedWholeCaseBootstrap,
) -> TerminalContrast:
    left = {x.target_center: x.pooled_bacc.exact_bacc for x in challenger.center_metrics}
    right = {x.target_center: x.pooled_bacc.exact_bacc for x in reference.center_metrics}
    differences = tuple((center, left[center] - right[center]) for center in MIDOGPP_CENTERS)
    values = np.asarray([x[1] for x in differences], dtype=np.float64)
    mean = float(values.mean(dtype=np.float64))
    half = float(student_t.ppf(.975, df=8)) * float(values.std(ddof=1)) / 3.0
    return TerminalContrast(
        family, geometry, challenger.method_id, reference.method_id, differences,
        mean, mean - half, mean + half, bootstrap,
    )


__all__ = ("PairedWholeCaseBootstrap", "TerminalContrast", "bootstrap_task", "build_contrast")
