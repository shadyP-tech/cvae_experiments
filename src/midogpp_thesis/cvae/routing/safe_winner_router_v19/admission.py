"""Equal-center source-OOF admission with approximate bootstrap max-stat bounds."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import AdmissionStatus, RouterFitConfig
from .hashing import canonical_hash
from .records import SelectedOOFRecord


_MOMENT_NAMES = ("g", "h", "b", "l")


@dataclass(frozen=True, slots=True)
class ApproximateSourceOOFBounds:
    observed_g: float
    observed_h: float
    observed_b: float
    observed_l: float
    gain_lower: float
    harm_upper: float
    brier_upper: float
    log_loss_upper: float
    max_stat_critical_value: float
    standard_errors: tuple[float, float, float, float]
    bootstrap_replicates: int
    bootstrap_alpha: float
    seed: int
    missing_class_support_replicates: int = 0
    bounds_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.observed_g,
            self.observed_h,
            self.observed_b,
            self.observed_l,
            self.gain_lower,
            self.harm_upper,
            self.brier_upper,
            self.log_loss_upper,
            self.max_stat_critical_value,
            *self.standard_errors,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.max_stat_critical_value < 0.0
            or len(self.standard_errors) != 4
            or any(value < 0.0 for value in self.standard_errors)
            or self.missing_class_support_replicates < 0
            or self.missing_class_support_replicates > self.bootstrap_replicates
            or self.bootstrap_replicates < 32
            or not 0.0 < self.bootstrap_alpha < 0.5
        ):
            raise ProtocolError("HARP v19 approximate source-OOF bounds are malformed.")
        object.__setattr__(
            self,
            "bounds_hash",
            canonical_hash(
                {
                    "schema_version": "approximate_source_oof_maxstat_bounds_v19",
                    "observed": dict(zip(_MOMENT_NAMES, values[:4], strict=True)),
                    "bounds": {
                        "gain_lower": self.gain_lower,
                        "harm_upper": self.harm_upper,
                        "brier_upper": self.brier_upper,
                        "log_loss_upper": self.log_loss_upper,
                    },
                    "max_stat_critical_value": self.max_stat_critical_value,
                    "standard_errors": self.standard_errors,
                    "bootstrap_replicates": self.bootstrap_replicates,
                    "bootstrap_alpha": self.bootstrap_alpha,
                    "seed": self.seed,
                    "missing_class_support_replicates": self.missing_class_support_replicates,
                    "undefined_class_mean_gain_assigned_minus_one": True,
                    "class_support_denominators_recomputed_per_replicate": True,
                    "center_stratified_case_bootstrap": True,
                    "equal_center_estimand": True,
                    "conformal": False,
                    "approximate": True,
                }
            ),
        )

    @property
    def passes(self) -> bool:
        return bool(
            self.gain_lower >= 0.0
            and self.harm_upper <= 0.0
            and self.brier_upper <= 0.0
            and self.log_loss_upper <= 0.0
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "approximate_source_oof_maxstat_bounds_v19",
            "observed_moments": {
                "g": self.observed_g,
                "h": self.observed_h,
                "b": self.observed_b,
                "l": self.observed_l,
            },
            "bounds": {
                "gain_lower": self.gain_lower,
                "harm_upper": self.harm_upper,
                "brier_upper": self.brier_upper,
                "log_loss_upper": self.log_loss_upper,
            },
            "max_stat_critical_value": self.max_stat_critical_value,
            "standard_errors": list(self.standard_errors),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_alpha": self.bootstrap_alpha,
            "seed": self.seed,
            "passes": self.passes,
            "bounds_hash": self.bounds_hash,
            "missing_class_support_replicates": self.missing_class_support_replicates,
            "undefined_class_mean_gain_assigned_minus_one": True,
            "class_support_denominators_recomputed_per_replicate": True,
            "center_stratified_case_bootstrap": True,
            "equal_center_estimand": True,
            "conformal": False,
            "approximate": True,
        }


@dataclass(frozen=True, slots=True)
class SourceOnlyAdmission:
    status: AdmissionStatus
    admitted: bool
    routed_case_count: int
    routed_center_count: int
    routed_cases_by_center: tuple[tuple[str, int], ...]
    total_case_count: int
    total_center_count: int
    bounds: ApproximateSourceOOFBounds | None
    bootstrap_performed: bool
    routed_risk_moments: tuple[tuple[str, float], ...] = ()
    qualifying_routed_center_count: int = 0
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, AdmissionStatus)
            or type(self.admitted) is not bool
            or self.routed_case_count < 0
            or self.routed_center_count < 0
            or not 0 <= self.qualifying_routed_center_count <= self.routed_center_count
            or self.total_case_count < 1
            or self.total_center_count < 1
            or self.routed_center_count != len(self.routed_cases_by_center)
            or self.routed_case_count != sum(count for _, count in self.routed_cases_by_center)
            or self.bootstrap_performed != (self.bounds is not None)
            or self.admitted != (self.status is AdmissionStatus.ADMITTED)
        ):
            raise ProtocolError("HARP v19 source-only admission is malformed.")
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_source_only_admission_v19",
                    "status": self.status.value,
                    "admitted": self.admitted,
                    "routed_case_count": self.routed_case_count,
                    "routed_center_count": self.routed_center_count,
                    "qualifying_routed_center_count": self.qualifying_routed_center_count,
                    "routed_cases_by_center": self.routed_cases_by_center,
                    "total_case_count": self.total_case_count,
                    "total_center_count": self.total_center_count,
                    "bounds_hash": None if self.bounds is None else self.bounds.bounds_hash,
                    "bootstrap_performed": self.bootstrap_performed,
                    "routed_risk_moments": dict(self.routed_risk_moments),
                    "approximate_bounds_are_not_final_refit_safety": True,
                    "no_forced_routes": True,
                    "nested_oof_evaluates_selection_algorithm": True,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pooled_pairwise_source_only_admission_v19",
            "status": self.status.value,
            "admitted": self.admitted,
            "routed_case_count": self.routed_case_count,
            "routed_center_count": self.routed_center_count,
            "qualifying_routed_center_count": self.qualifying_routed_center_count,
            "routed_cases_by_center": [
                {"center_id": center, "routed_case_count": count}
                for center, count in self.routed_cases_by_center
            ],
            "total_case_count": self.total_case_count,
            "total_center_count": self.total_center_count,
            "bounds": None if self.bounds is None else self.bounds.public_payload(),
            "bootstrap_performed": self.bootstrap_performed,
            "routed_risk_moments": dict(self.routed_risk_moments),
            "approximate_bounds_are_not_final_refit_safety": True,
            "no_forced_routes": True,
            "nested_oof_evaluates_selection_algorithm": True,
            "admission_hash": self.admission_hash,
        }


def _moments(record: SelectedOOFRecord) -> np.ndarray:
    route = float(record.route_selected)
    return np.asarray(
        [
            route * record.bacc_gain,
            route * (float(record.harm) - 0.25),
            route * (record.brier_delta - 0.002),
            route * (record.log_loss_delta - 0.005),
        ],
        dtype=np.float64,
    )


def _equal_center_mean(by_center: Mapping[str, np.ndarray]) -> np.ndarray:
    if not by_center:
        raise ProtocolError("HARP v19 equal-center moment surface is empty.")
    return np.mean(
        np.asarray(
            [np.mean(values, axis=0, dtype=np.float64) for _, values in sorted(by_center.items())],
            dtype=np.float64,
        ),
        axis=0,
        dtype=np.float64,
    )


def approximate_source_oof_bounds(
    records: Sequence[SelectedOOFRecord],
    *,
    config: RouterFitConfig,
) -> ApproximateSourceOOFBounds:
    """Compute deterministic center-stratified approximate max-stat bounds.

    This is an ordinary case bootstrap within each observed center.  It is
    explicitly not conformal calibration and makes no finite-sample coverage
    guarantee.
    """

    rows = tuple(sorted(records, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if not rows or len(keys) != len(set(keys)):
        raise ProtocolError("HARP v19 source OOF records are empty or duplicated.")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[row.center_id].append(_moments(row))
    arrays = {center: np.asarray(values, dtype=np.float64) for center, values in grouped.items()}
    observed = _equal_center_mean(arrays)
    rng = np.random.default_rng(config.bootstrap_seed)
    replicates = np.empty((config.bootstrap_replicates, 4), dtype=np.float64)
    centers = tuple(sorted(arrays))
    # Bounded batches avoid the replicate x center x case Python loop and
    # do not materialize a workstation-sized bootstrap tensor.
    class_values = {}
    for center in centers:
        center_rows = tuple(row for row in rows if row.center_id == center)
        if all(row.class_0_gain is not None or row.class_1_gain is not None for row in center_rows):
            class_values[center] = np.asarray([
                [np.nan if row.class_0_gain is None else row.class_0_gain,
                 np.nan if row.class_1_gain is None else row.class_1_gain]
                for row in center_rows], dtype=np.float64)
    missing_class_support_replicates = 0
    for start in range(0, config.bootstrap_replicates, 256):
        stop = min(config.bootstrap_replicates, start + 256)
        batch = np.zeros((stop - start, 4), dtype=np.float64)
        missing = np.zeros(stop - start, dtype=bool)
        for center in centers:
            values = arrays[center]
            indices = rng.integers(0, len(values), size=(stop - start, len(values)), endpoint=False)
            center_mean = np.mean(values[indices], axis=1, dtype=np.float64)
            if center in class_values:
                sampled = class_values[center][indices]
                counts = np.sum(np.isfinite(sampled), axis=1)
                sums = np.nansum(sampled, axis=1)
                missing |= np.any(counts == 0, axis=1)
                # Recompute the equal-class/supporting-case estimator within
                # each resampled center instead of resampling fixed weights.
                recalls = np.divide(sums, counts, out=np.full_like(sums, -1.0), where=counts > 0)
                center_mean[:, 0] = np.mean(recalls, axis=1)
            batch += center_mean / len(centers)
        replicates[start:stop] = batch
        missing_class_support_replicates += int(np.sum(missing))
    standard_errors = np.std(replicates, axis=0, ddof=1, dtype=np.float64)
    safe = np.where(standard_errors > np.finfo(np.float64).eps, standard_errors, 1.0)
    centered = replicates - observed[None, :]
    # One lower bound for g and simultaneous upper bounds for the three risks.
    statistics = np.column_stack(
        (
            -centered[:, 0] / safe[0],
            centered[:, 1] / safe[1],
            centered[:, 2] / safe[2],
            centered[:, 3] / safe[3],
        )
    )
    max_stat = np.max(statistics, axis=1)
    critical = max(
        0.0,
        float(np.quantile(max_stat, 1.0 - config.bootstrap_alpha, method="higher")),
    )
    effective_se = np.where(standard_errors > np.finfo(np.float64).eps, standard_errors, 0.0)
    return ApproximateSourceOOFBounds(
        observed_g=float(observed[0]),
        observed_h=float(observed[1]),
        observed_b=float(observed[2]),
        observed_l=float(observed[3]),
        gain_lower=float(observed[0] - critical * effective_se[0]),
        harm_upper=float(observed[1] + critical * effective_se[1]),
        brier_upper=float(observed[2] + critical * effective_se[2]),
        log_loss_upper=float(observed[3] + critical * effective_se[3]),
        max_stat_critical_value=critical,
        standard_errors=tuple(float(value) for value in effective_se),
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_alpha=config.bootstrap_alpha,
        seed=config.bootstrap_seed,
        missing_class_support_replicates=missing_class_support_replicates,
    )


def build_source_only_admission(
    records: Sequence[SelectedOOFRecord],
    *,
    config: RouterFitConfig,
) -> SourceOnlyAdmission:
    rows = tuple(sorted(records, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if not rows or len(keys) != len(set(keys)):
        raise ProtocolError("HARP v19 source-only admission needs unique nested OOF cases.")
    from .frontier import policy_moments
    diagnostics = tuple(sorted(policy_moments(rows).items()))
    centers = tuple(sorted({row.center_id for row in rows}))
    routed_counts = Counter(row.center_id for row in rows if row.route_selected)
    routed = sum(routed_counts.values())
    qualifying_centers = sum(count >= config.minimum_routed_oof_cases_per_center for count in routed_counts.values())
    if routed == 0:
        return SourceOnlyAdmission(
            status=AdmissionStatus.NO_NONZERO_SAFE_OOF_COVERAGE,
            admitted=False,
            routed_case_count=0,
            routed_center_count=0,
            routed_cases_by_center=(),
            total_case_count=len(rows),
            total_center_count=len(centers),
            bounds=None,
            bootstrap_performed=False,
            routed_risk_moments=diagnostics,
            qualifying_routed_center_count=qualifying_centers,
        )
    inventory_ok = bool(
        routed >= config.minimum_routed_oof_cases
        and qualifying_centers >= config.minimum_routed_oof_centers
    )
    if not inventory_ok:
        return SourceOnlyAdmission(
            status=AdmissionStatus.INSUFFICIENT_ROUTED_OOF,
            admitted=False,
            routed_case_count=routed,
            routed_center_count=len(routed_counts),
            routed_cases_by_center=tuple(sorted(routed_counts.items())),
            total_case_count=len(rows),
            total_center_count=len(centers),
            bounds=None,
            bootstrap_performed=False,
            routed_risk_moments=diagnostics,
            qualifying_routed_center_count=qualifying_centers,
        )
    bounds = approximate_source_oof_bounds(rows, config=config)
    status = AdmissionStatus.ADMITTED if bounds.passes else AdmissionStatus.APPROXIMATE_BOUNDS_FAILED
    return SourceOnlyAdmission(
        status=status,
        admitted=status is AdmissionStatus.ADMITTED,
        routed_case_count=routed,
        routed_center_count=len(routed_counts),
        routed_cases_by_center=tuple(sorted(routed_counts.items())),
        total_case_count=len(rows),
        total_center_count=len(centers),
        bounds=bounds,
        bootstrap_performed=True,
        routed_risk_moments=diagnostics,
        qualifying_routed_center_count=qualifying_centers,
    )


__all__ = (
    "ApproximateSourceOOFBounds",
    "SourceOnlyAdmission",
    "approximate_source_oof_bounds",
    "build_source_only_admission",
)
