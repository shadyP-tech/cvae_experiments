"""Reusable primitives for the exploratory dense residual soft router."""

from .allocation import (
    ALLOCATION_SEMANTICS,
    DEFAULT_TOTAL_PER_CLASS,
    HamiltonAllocation,
    build_hamilton_allocation,
    hamilton_allocate,
)
from .compatibility import (
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
from .composition import (
    COMPOSITION_SEMANTICS,
    PrefixComposition,
    compose_prefix_blocks,
)
from .partitions import (
    CasePartitions,
    assert_outer_query_source_exclusions,
    deterministic_case_partitions,
)
from .weights import (
    DEFAULT_MAX_SOURCE_WEIGHT,
    DEFAULT_MIN_EFFECTIVE_SOURCES,
    DEFAULT_TEMPERATURE,
    WEIGHT_SEMANTICS,
    ResidualSoftWeights,
    residual_soft_weights,
)

__all__ = (
    "ALLOCATION_SEMANTICS",
    "CALIBRATION_SEMANTICS",
    "CLASS_PRIOR",
    "COMPOSITION_SEMANTICS",
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "DEFAULT_SCALE_FLOOR",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOTAL_PER_CLASS",
    "DEFAULT_TRAINING_SEEDS",
    "ENERGY_SEMANTICS",
    "WEIGHT_SEMANTICS",
    "CasePartitions",
    "CompatibilityEnergy",
    "HamiltonAllocation",
    "OwnSourceCalibration",
    "PrefixComposition",
    "ReplicaCalibration",
    "ReplicaKey",
    "ResidualSoftWeights",
    "assert_outer_query_source_exclusions",
    "build_hamilton_allocation",
    "calibrate_own_source_energies",
    "compose_prefix_blocks",
    "deterministic_case_partitions",
    "gaussian_kl_diagonal_to_full",
    "hamilton_allocate",
    "residual_soft_weights",
    "score_variational_compatibility",
)
