"""Pure label-free antisymmetric residual MMD routing core.

This package intentionally exposes only proxy mathematics and deterministic
integer allocation.  It has no artifact runner, downstream-utility input, or
promotion policy.
"""

from .allocation import (
    ALLOCATION_SEMANTICS,
    DEFAULT_TOTAL_PER_CLASS,
    AntisymmetricAllocation,
    allocate_antisymmetric_counts,
    antisymmetric_allocate,
    build_antisymmetric_allocation,
)
from .contracts import (
    DEFAULT_L2_SHRINKAGE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SOURCE_WEIGHT,
    DEFAULT_MAX_UNIFORM_L1,
    DEFAULT_MIN_EFFECTIVE_SOURCES,
    DEFAULT_MIN_ROBUST_IMPROVEMENT,
    DEFAULT_MIN_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE,
    DEFAULT_MIN_SOFT_CLASS_MASS_PER_CASE,
    DEFAULT_SOLVER_TOLERANCE,
    DEFAULT_VARIANT_WORSENING_TOLERANCE,
    DEFAULT_WORST_VARIANT_PENALTY,
    LABEL_USE_SEMANTICS,
    PROXY_CLAIM_ROLE,
    ROBUST_OBJECTIVE_SEMANTICS,
    SOLVER_METHOD,
    WEIGHT_SEMANTICS,
    AntisymmetricAxisDiagnostic,
    AntisymmetricResidualConfig,
    AntisymmetricResidualSolution,
    AntisymmetricVariantDiagnostic,
)
from .solver import (
    solve_antisymmetric_residual_mmd,
    solve_antisymmetric_residual_weights,
)


__all__ = (
    "ALLOCATION_SEMANTICS",
    "DEFAULT_L2_SHRINKAGE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MAX_UNIFORM_L1",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "DEFAULT_MIN_ROBUST_IMPROVEMENT",
    "DEFAULT_MIN_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE",
    "DEFAULT_MIN_SOFT_CLASS_MASS_PER_CASE",
    "DEFAULT_SOLVER_TOLERANCE",
    "DEFAULT_TOTAL_PER_CLASS",
    "DEFAULT_VARIANT_WORSENING_TOLERANCE",
    "DEFAULT_WORST_VARIANT_PENALTY",
    "LABEL_USE_SEMANTICS",
    "PROXY_CLAIM_ROLE",
    "ROBUST_OBJECTIVE_SEMANTICS",
    "SOLVER_METHOD",
    "WEIGHT_SEMANTICS",
    "AntisymmetricAllocation",
    "AntisymmetricAxisDiagnostic",
    "AntisymmetricResidualConfig",
    "AntisymmetricResidualSolution",
    "AntisymmetricVariantDiagnostic",
    "allocate_antisymmetric_counts",
    "antisymmetric_allocate",
    "build_antisymmetric_allocation",
    "solve_antisymmetric_residual_mmd",
    "solve_antisymmetric_residual_weights",
)
