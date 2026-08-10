"""Deterministic whole-case bootstrap and equal-center terminal summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Sequence
import math

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .contracts import CaseConfusionCounts
from .core_hashing import canonical_hash
from .scientific_constants import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


@dataclass(frozen=True)
class BootstrapContrast:
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
        if self.challenger_method == self.reference_method:
            raise ProtocolError("Bootstrap contrast requires distinct methods.")
        if self.replicate_count <= 0 or self.invalid_draw_count < 0:
            raise ProtocolError("Bootstrap replicate counts are invalid.")
        values = (
            self.observed_equal_center_difference,
            self.bootstrap_mean,
            self.ci95_lower,
            self.ci95_upper,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ProtocolError("Bootstrap summary contains non-finite values.")
        if self.ci95_lower > self.ci95_upper:
            raise ProtocolError("Bootstrap confidence interval is reversed.")
        object.__setattr__(self, "bootstrap_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_cluster_bootstrap_v1",
            "challenger_method": self.challenger_method,
            "reference_method": self.reference_method,
            "replicate_count": self.replicate_count,
            "seed": self.seed,
            "observed_equal_center_difference": self.observed_equal_center_difference,
            "bootstrap_mean": self.bootstrap_mean,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "invalid_draw_count": self.invalid_draw_count,
            "resampling_unit": "whole_case_within_center",
            "aggregation": "equal_center",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "bootstrap_hash": self.bootstrap_hash}


@dataclass(frozen=True)
class EqualCenterContrast:
    challenger_method: str
    reference_method: str
    center_count: int
    center_differences: tuple[tuple[str, float], ...]
    mean_difference: float
    ci95_lower: float
    ci95_upper: float


def equal_center_contrast(
    challenger_rows: Sequence[CaseConfusionCounts],
    reference_rows: Sequence[CaseConfusionCounts],
) -> EqualCenterContrast:
    grouped_left = _group_by_center(challenger_rows)
    grouped_right = _group_by_center(reference_rows)
    if set(grouped_left) != set(grouped_right) or len(grouped_left) < 2:
        raise ProtocolError("Equal-center contrast needs aligned rows from at least two centers.")
    differences = tuple(
        (center, _bacc(grouped_left[center]) - _bacc(grouped_right[center]))
        for center in sorted(grouped_left)
    )
    values = np.asarray([value for _center, value in differences], dtype=np.float64)
    mean = float(values.mean())
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    half_width = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    return EqualCenterContrast(
        challenger_method=next(iter({row.method_id for row in challenger_rows})),
        reference_method=next(iter({row.method_id for row in reference_rows})),
        center_count=len(values),
        center_differences=differences,
        mean_difference=mean,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
    )


def whole_case_bootstrap(
    challenger_rows: Sequence[CaseConfusionCounts],
    reference_rows: Sequence[CaseConfusionCounts],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapContrast:
    """Resample intact cases within center and aggregate centers equally.

    A rare resample missing a pooled class in any center is invalid rather
    than being assigned a fabricated per-case or per-center score.  Sampling
    continues deterministically until the predeclared number of legal draws is
    reached, and the discarded-draw count is reported.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ProtocolError("Bootstrap replicate count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError("Bootstrap seed must be an integer.")
    left = _group_by_center(challenger_rows)
    right = _group_by_center(reference_rows)
    if set(left) != set(right):
        raise ProtocolError("Bootstrap methods have different center scopes.")
    for center in left:
        left_keys = tuple(row.case_key for row in left[center])
        right_keys = tuple(row.case_key for row in right[center])
        if left_keys != right_keys:
            raise ProtocolError("Bootstrap methods have different whole-case scopes.")
    observed = equal_center_contrast(challenger_rows, reference_rows).mean_difference
    generator = np.random.default_rng(seed)
    values: list[float] = []
    invalid = 0
    maximum_attempts = max(replicates * 100, replicates + 10_000)
    attempts = 0
    while len(values) < replicates and attempts < maximum_attempts:
        attempts += 1
        center_differences: list[float] = []
        legal = True
        for center in sorted(left):
            count = len(left[center])
            indices = generator.integers(0, count, size=count)
            left_draw = tuple(left[center][int(index)] for index in indices)
            right_draw = tuple(right[center][int(index)] for index in indices)
            try:
                center_differences.append(_bacc(left_draw) - _bacc(right_draw))
            except ProtocolError:
                legal = False
                break
        if legal:
            values.append(math.fsum(center_differences) / len(center_differences))
        else:
            invalid += 1
    if len(values) != replicates:
        raise ProtocolError("Whole-case bootstrap could not obtain enough legal class-complete draws.")
    array = np.asarray(values, dtype=np.float64)
    return BootstrapContrast(
        challenger_method=next(iter({row.method_id for row in challenger_rows})),
        reference_method=next(iter({row.method_id for row in reference_rows})),
        replicate_count=replicates,
        seed=seed,
        observed_equal_center_difference=observed,
        bootstrap_mean=float(array.mean()),
        ci95_lower=float(np.quantile(array, 0.025, method="linear")),
        ci95_upper=float(np.quantile(array, 0.975, method="linear")),
        invalid_draw_count=invalid,
    )


def _group_by_center(
    rows: Sequence[CaseConfusionCounts],
) -> dict[str, tuple[CaseConfusionCounts, ...]]:
    values = tuple(rows)
    if not values or len({row.method_id for row in values}) != 1:
        raise ProtocolError("Uncertainty input must contain exactly one non-empty method.")
    grouped: dict[str, list[CaseConfusionCounts]] = defaultdict(list)
    for row in values:
        grouped[row.target_center].append(row)
    result = {center: tuple(sorted(group, key=lambda row: row.case_key)) for center, group in grouped.items()}
    if any(len({row.case_key for row in group}) != len(group) for group in result.values()):
        raise ProtocolError("Uncertainty input contains duplicate whole cases.")
    return result


def _bacc(rows: Sequence[CaseConfusionCounts]) -> float:
    positive = sum(row.n_positive for row in rows)
    negative = sum(row.n_negative for row in rows)
    if positive <= 0 or negative <= 0:
        raise ProtocolError("Bootstrap pooled center lacks a binary class.")
    return 0.5 * (
        sum(row.true_positive for row in rows) / positive
        + sum(row.true_negative for row in rows) / negative
    )


__all__ = (
    "BootstrapContrast",
    "EqualCenterContrast",
    "equal_center_contrast",
    "whole_case_bootstrap",
)
