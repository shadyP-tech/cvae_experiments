"""Source-only, ensemble-statistic-matched geometry calibration.

Each calibration observation is the maximum leverage for one held-out source
action across strict leave-{pseudo-target, donor} fits.  This matches the
maximum operator used for a target action; an unadjusted quantile of individual
leverages does not.  A pseudo-target cannot also be an eligible donor, so the
source ensemble has ``D - 1`` members while the target ensemble has ``D``.
Pair deletion is conservative in training support, but the unequal cardinality
and dependent source cases preclude a formal conformal-coverage claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import Comparison


CALIBRATION_METHOD = (
    "source_lodo_pseudo_target_delete_donor_ensemble_max_"
    "d_minus_1_donor_balanced_empirical_tail_v2"
)
ENSEMBLE_CARDINALITY_RULE = (
    "SOURCE_PSEUDO_TARGET_D_MINUS_1_VS_TARGET_D_PAIR_DELETE_LIMITATION"
)


def upper_empirical_quantile(values: Sequence[float], level: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    quantile = float(level)
    if (
        array.ndim != 1
        or len(array) == 0
        or not np.isfinite(array).all()
        or not math.isfinite(quantile)
        or not 0.5 <= quantile < 1.0
    ):
        raise ProtocolError("HARP v4 empirical quantile inputs are invalid.")
    ordered = np.sort(array)
    index = min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1)
    return float(ordered[index])


def _donor_balanced_quantile(
    values: Sequence[float], donor_ids: Sequence[str], level: float
) -> float:
    array = tuple(float(value) for value in values)
    donors = tuple(str(value) for value in donor_ids)
    quantile = float(level)
    if (
        not array
        or len(array) != len(donors)
        or any(not math.isfinite(value) or value < 0 for value in array)
        or any(not donor for donor in donors)
        or not 0.5 <= quantile < 1.0
    ):
        raise ProtocolError("HARP v4 donor-balanced geometry inputs are invalid.")
    levels = tuple(sorted(set(donors)))
    counts = {donor: donors.count(donor) for donor in levels}
    weighted = sorted(
        (
            value,
            1.0 / (len(levels) * counts[donor]),
        )
        for value, donor in zip(array, donors, strict=True)
    )
    cumulative = 0.0
    for value, weight in weighted:
        cumulative += weight
        if cumulative + 8.0 * np.finfo(np.float64).eps >= quantile:
            return value
    return weighted[-1][0]


@dataclass(frozen=True)
class GeometryCalibration:
    comparison: Comparison
    quantile_level: float
    reference_median: float
    reference_quantile: float
    heldout_raw_leverages: tuple[float, ...]
    heldout_donor_ids: tuple[str, ...]
    heldout_raw_block_ids: tuple[str, ...]
    heldout_block_ids: tuple[str, ...]
    heldout_block_donor_ids: tuple[str, ...]
    heldout_block_maxima: tuple[float, ...]
    heldout_block_sizes: tuple[int, ...]
    source_donor_ids: tuple[str, ...]
    calibration_method: str = CALIBRATION_METHOD
    ensemble_cardinality_rule: str = ENSEMBLE_CARDINALITY_RULE
    formal_conformal_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison", Comparison(self.comparison))
        raw = tuple(float(value) for value in self.heldout_raw_leverages)
        raw_donors = tuple(str(value) for value in self.heldout_donor_ids)
        raw_blocks = tuple(str(value) for value in self.heldout_raw_block_ids)
        block_ids = tuple(str(value) for value in self.heldout_block_ids)
        block_donors = tuple(str(value) for value in self.heldout_block_donor_ids)
        maxima = tuple(float(value) for value in self.heldout_block_maxima)
        sizes = tuple(int(value) for value in self.heldout_block_sizes)
        source_donors = tuple(str(value) for value in self.source_donor_ids)
        if (
            self.calibration_method != CALIBRATION_METHOD
            or self.ensemble_cardinality_rule != ENSEMBLE_CARDINALITY_RULE
            or self.formal_conformal_claimed is not False
            or not 0.5 <= float(self.quantile_level) < 1.0
            or not raw
            or len(raw) != len(raw_donors)
            or len(raw) != len(raw_blocks)
            or any(not math.isfinite(value) or value < 0 for value in raw)
            or not block_ids
            or len(set(block_ids)) != len(block_ids)
            or len(block_ids) != len(block_donors)
            or len(block_ids) != len(maxima)
            or len(block_ids) != len(sizes)
            or source_donors != tuple(sorted(set(source_donors)))
            or len(source_donors) < 2
            or set(raw_donors) != set(source_donors)
            or set(block_donors) != set(source_donors)
            or any(size != len(source_donors) - 1 for size in sizes)
        ):
            raise ProtocolError("HARP v4 geometry calibration is malformed.")
        grouped: dict[str, list[float]] = {block: [] for block in block_ids}
        grouped_donors: dict[str, set[str]] = {block: set() for block in block_ids}
        for value, donor, block in zip(raw, raw_donors, raw_blocks, strict=True):
            if block not in grouped:
                raise ProtocolError("HARP v4 geometry raw row names an unknown block.")
            grouped[block].append(value)
            grouped_donors[block].add(donor)
        for block, donor, maximum, size in zip(
            block_ids, block_donors, maxima, sizes, strict=True
        ):
            if (
                len(grouped[block]) != size
                or grouped_donors[block] != {donor}
                or max(grouped[block]) != maximum
            ):
                raise ProtocolError("HARP v4 geometry ensemble block is inconsistent.")
        floor = np.finfo(np.float64).eps
        median = max(_donor_balanced_quantile(maxima, block_donors, 0.5), floor)
        reference = max(
            _donor_balanced_quantile(maxima, block_donors, self.quantile_level),
            floor,
        )
        if (
            float(self.reference_median) != median
            or float(self.reference_quantile) != reference
            or reference < median
        ):
            raise ProtocolError("HARP v4 geometry reference is not block-max calibrated.")
        object.__setattr__(self, "quantile_level", float(self.quantile_level))
        object.__setattr__(self, "reference_median", median)
        object.__setattr__(self, "reference_quantile", reference)
        object.__setattr__(self, "heldout_raw_leverages", raw)
        object.__setattr__(self, "heldout_donor_ids", raw_donors)
        object.__setattr__(self, "heldout_raw_block_ids", raw_blocks)
        object.__setattr__(self, "heldout_block_ids", block_ids)
        object.__setattr__(self, "heldout_block_donor_ids", block_donors)
        object.__setattr__(self, "heldout_block_maxima", maxima)
        object.__setattr__(self, "heldout_block_sizes", sizes)
        object.__setattr__(self, "source_donor_ids", source_donors)

    @property
    def shrinkage_start_ratio(self) -> float:
        return min(1.0, self.reference_median / self.reference_quantile)

    @property
    def finite_sample_tail_floor(self) -> float:
        counts = tuple(
            self.heldout_block_donor_ids.count(donor) for donor in self.source_donor_ids
        )
        return float(np.mean([1.0 / (count + 1.0) for count in counts]))


@dataclass(frozen=True)
class GeometryAssessment:
    raw_leverages: tuple[float, ...]
    calibrated_ratios: tuple[float, ...]
    maximum_ratio: float
    empirical_percentile: float
    empirical_tail_probability: float
    finite_sample_tail_floor: float
    calibration_block_count: int
    compatibility_shrinkage: float
    reference_median: float
    reference_quantile: float
    quantile_level: float
    calibration_method: str
    ensemble_cardinality_rule: str
    formal_conformal_claimed: bool

    def __post_init__(self) -> None:
        values = (
            *self.raw_leverages,
            *self.calibrated_ratios,
            self.maximum_ratio,
            self.empirical_percentile,
            self.empirical_tail_probability,
            self.finite_sample_tail_floor,
            self.compatibility_shrinkage,
            self.reference_median,
            self.reference_quantile,
            self.quantile_level,
        )
        if (
            not self.raw_leverages
            or len(self.raw_leverages) != len(self.calibrated_ratios)
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in values)
            or not 0.0 <= self.empirical_percentile <= 1.0
            or not 0.0 < self.finite_sample_tail_floor <= self.empirical_tail_probability <= 1.0
            or type(self.calibration_block_count) is not int
            or self.calibration_block_count < 1
            or not 0.0 <= self.compatibility_shrinkage <= 1.0
            or not 0.5 <= self.quantile_level < 1.0
            or self.calibration_method != CALIBRATION_METHOD
            or self.ensemble_cardinality_rule != ENSEMBLE_CARDINALITY_RULE
            or self.formal_conformal_claimed is not False
        ):
            raise ProtocolError("HARP v4 geometry assessment is malformed.")


def calibrate_geometry(
    comparison: Comparison,
    raw_leverages: Sequence[float],
    donor_ids: Sequence[str],
    raw_block_ids: Sequence[str],
    *,
    quantile_level: float,
) -> GeometryCalibration:
    raw = tuple(float(value) for value in raw_leverages)
    donors = tuple(str(value) for value in donor_ids)
    raw_blocks = tuple(str(value) for value in raw_block_ids)
    if not raw or len(raw) != len(donors) or len(raw) != len(raw_blocks):
        raise ProtocolError("Geometry calibration rows, donors, and blocks must align.")
    source_donors = tuple(sorted(set(donors)))
    grouped: dict[str, list[float]] = {}
    grouped_donors: dict[str, set[str]] = {}
    for value, donor, block in zip(raw, donors, raw_blocks, strict=True):
        if not block:
            raise ProtocolError("Geometry calibration block identities must be nonempty.")
        grouped.setdefault(block, []).append(value)
        grouped_donors.setdefault(block, set()).add(donor)
    block_ids = tuple(sorted(grouped))
    block_donors: list[str] = []
    maxima: list[float] = []
    sizes: list[int] = []
    for block in block_ids:
        if len(grouped_donors[block]) != 1:
            raise ProtocolError("One geometry ensemble block mixed pseudo-target donors.")
        block_donors.append(next(iter(grouped_donors[block])))
        maxima.append(max(grouped[block]))
        sizes.append(len(grouped[block]))
    floor = np.finfo(np.float64).eps
    median = max(_donor_balanced_quantile(maxima, block_donors, 0.5), floor)
    reference = max(
        _donor_balanced_quantile(maxima, block_donors, quantile_level), floor
    )
    return GeometryCalibration(
        comparison=Comparison(comparison),
        quantile_level=quantile_level,
        reference_median=median,
        reference_quantile=reference,
        heldout_raw_leverages=raw,
        heldout_donor_ids=donors,
        heldout_raw_block_ids=raw_blocks,
        heldout_block_ids=block_ids,
        heldout_block_donor_ids=tuple(block_donors),
        heldout_block_maxima=tuple(maxima),
        heldout_block_sizes=tuple(sizes),
        source_donor_ids=source_donors,
    )


def _smoothed_probability(
    calibration: GeometryCalibration, target: float, *, upper_tail: bool
) -> float:
    probabilities: list[float] = []
    for donor in calibration.source_donor_ids:
        values = tuple(
            value
            for value, block_donor in zip(
                calibration.heldout_block_maxima,
                calibration.heldout_block_donor_ids,
                strict=True,
            )
            if block_donor == donor
        )
        count = sum(value >= target for value in values) if upper_tail else sum(
            value <= target for value in values
        )
        probabilities.append((1.0 + count) / (len(values) + 1.0))
    return float(np.mean(probabilities))


def assess_geometry(
    calibration: GeometryCalibration, raw_leverages: Sequence[float]
) -> GeometryAssessment:
    if not isinstance(calibration, GeometryCalibration):
        raise ProtocolError("Geometry assessment requires source-LODO calibration.")
    raw = tuple(float(value) for value in raw_leverages)
    if not raw or any(not math.isfinite(value) or value < 0 for value in raw):
        raise ProtocolError("Target raw leverages must be finite and nonnegative.")
    ratios = tuple(value / calibration.reference_quantile for value in raw)
    maximum_raw = max(raw)
    maximum_ratio = max(ratios)
    start = calibration.shrinkage_start_ratio
    shrinkage = 1.0 if maximum_ratio <= start else min(1.0, start / maximum_ratio)
    return GeometryAssessment(
        raw_leverages=raw,
        calibrated_ratios=ratios,
        maximum_ratio=maximum_ratio,
        empirical_percentile=_smoothed_probability(
            calibration, maximum_raw, upper_tail=False
        ),
        empirical_tail_probability=_smoothed_probability(
            calibration, maximum_raw, upper_tail=True
        ),
        finite_sample_tail_floor=calibration.finite_sample_tail_floor,
        calibration_block_count=len(calibration.heldout_block_ids),
        compatibility_shrinkage=shrinkage,
        reference_median=calibration.reference_median,
        reference_quantile=calibration.reference_quantile,
        quantile_level=calibration.quantile_level,
        calibration_method=calibration.calibration_method,
        ensemble_cardinality_rule=calibration.ensemble_cardinality_rule,
        formal_conformal_claimed=calibration.formal_conformal_claimed,
    )


__all__ = (
    "CALIBRATION_METHOD",
    "ENSEMBLE_CARDINALITY_RULE",
    "GeometryAssessment",
    "GeometryCalibration",
    "assess_geometry",
    "calibrate_geometry",
    "upper_empirical_quantile",
)
