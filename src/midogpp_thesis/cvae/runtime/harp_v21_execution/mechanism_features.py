"""Label-free primitive descriptors of the actually executed masked surface.

These seed-dispersion fields describe a primitive surface, not an inferred
variance of a later K/lambda mixture. Composite features have their own schema.
"""
from __future__ import annotations
import math
import numpy as np
from ...protocol import ProtocolError
from ...routing.correction_mass_router_v21.contracts import Direction


LABEL_FREE_FEATURE_NAMES = (
    "active_mask_fraction",
    "threshold_flip_fraction",
    "direction_aligned_branch_mass",
    "action_delta_mean",
    "action_delta_std",
    "action_delta_abs_mean",
    "baseline_probability_mean",
    "baseline_positive_branch_fraction",
    "baseline_boundary_distance_mean",
    "baseline_boundary_distance_min",
    "surface_boundary_distance_mean",
    "surface_boundary_distance_min",
    "boundary_distance_change_mean",
    "baseline_seed_dispersion_mean",
    "surface_seed_dispersion_mean",
    "surface_seed_dispersion_change_mean",
    "compatibility_mean_z",
    "compatibility_std_z",
    "compatibility_reciprocal_rank",
    "compatibility_rank_margin",
)


def feature_values(
    baseline: np.ndarray,
    challenger: np.ndarray,
    baseline_dispersion: np.ndarray,
    challenger_dispersion: np.ndarray,
    *,
    active: np.ndarray,
    direction: Direction,
    compatibility: tuple[float, float, float, float],
) -> tuple[float, ...]:
    surface = baseline.copy()
    surface[active] = challenger[active]
    surface_dispersion = baseline_dispersion.copy()
    surface_dispersion[active] = challenger_dispersion[active]
    delta = surface.astype(np.float64) - baseline.astype(np.float64)
    baseline64 = baseline.astype(np.float64)
    surface64 = surface.astype(np.float64)
    baseline_boundary = np.abs(baseline64 - 0.5)
    surface_boundary = np.abs(surface64 - 0.5)
    aligned_mass = (
        np.abs(delta)
        if direction is Direction.FULL
        else np.maximum(delta, 0.0)
        if direction is Direction.D01
        else np.maximum(-delta, 0.0)
    )
    values = (
        float(np.mean(active, dtype=np.float64)),
        float(np.mean((baseline >= 0.5) != (surface >= 0.5), dtype=np.float64)),
        float(np.mean(aligned_mass, dtype=np.float64)),
        float(np.mean(delta, dtype=np.float64)),
        float(np.std(delta, dtype=np.float64)),
        float(np.mean(np.abs(delta), dtype=np.float64)),
        float(np.mean(baseline64, dtype=np.float64)),
        float(np.mean(baseline >= 0.5, dtype=np.float64)),
        float(np.mean(baseline_boundary, dtype=np.float64)),
        float(np.min(baseline_boundary)),
        float(np.mean(surface_boundary, dtype=np.float64)),
        float(np.min(surface_boundary)),
        float(np.mean(surface_boundary - baseline_boundary, dtype=np.float64)),
        float(np.mean(baseline_dispersion, dtype=np.float64)),
        float(np.mean(surface_dispersion, dtype=np.float64)),
        float(
            np.mean(surface_dispersion, dtype=np.float64)
            - np.mean(baseline_dispersion, dtype=np.float64)
        ),
        *compatibility,
    )
    if len(values) != len(LABEL_FREE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ProtocolError("HARP v21 label-free feature vector is malformed.")
    return values

