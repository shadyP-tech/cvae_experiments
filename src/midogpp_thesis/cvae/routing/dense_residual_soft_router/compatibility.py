"""Compatibility import shim for the stage-neutral variational primitive.

New routing experiments import :mod:`midogpp_thesis.cvae.routing.variational_compatibility`
directly.  This module preserves the public path used by completed predecessor
implementations without making the neutral primitive depend on them.
"""

from ..variational_compatibility import (
    CALIBRATION_SEMANTICS,
    CLASS_PRIOR,
    DEFAULT_SCALE_FLOOR,
    DEFAULT_TRAINING_SEEDS,
    ENERGY_SEMANTICS,
    CompatibilityEnergy,
    OwnSourceCalibration,
    ReplicaCalibration,
    ReplicaKey,
    calibrate_own_source_energies,
    gaussian_kl_diagonal_to_full,
    score_variational_compatibility,
)


__all__ = (
    "CALIBRATION_SEMANTICS",
    "CLASS_PRIOR",
    "DEFAULT_SCALE_FLOOR",
    "DEFAULT_TRAINING_SEEDS",
    "ENERGY_SEMANTICS",
    "CompatibilityEnergy",
    "OwnSourceCalibration",
    "ReplicaCalibration",
    "ReplicaKey",
    "calibrate_own_source_energies",
    "gaussian_kl_diagonal_to_full",
    "score_variational_compatibility",
)
