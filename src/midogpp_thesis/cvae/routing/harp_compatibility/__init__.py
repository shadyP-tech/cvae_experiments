"""Optional label-free compatibility shrinkage for HARP.

Compatibility is deliberately subordinate to the action model: it may reduce
an already eligible action's weight or force abstention, but it cannot make an
ineligible action eligible and is never interpreted as downstream utility.
"""

from .calibration import (
    CompatibilityCalibration,
    ReplicaCalibration,
    calibrate_own_source_energy,
)
from .ablation import (
    CompatibilityAblationDecision,
    CompatibilityAblationFold,
    decide_compatibility_ablation,
)
from .energy import (
    ENERGY_SEMANTICS,
    VariationalEnergySurface,
    class_marginal_variational_energy,
)
from .shrinkage import CompatibilityShrinkage, shrink_eligible_weight

__all__ = (
    "ENERGY_SEMANTICS",
    "CompatibilityCalibration",
    "CompatibilityAblationDecision",
    "CompatibilityAblationFold",
    "CompatibilityShrinkage",
    "ReplicaCalibration",
    "VariationalEnergySurface",
    "calibrate_own_source_energy",
    "decide_compatibility_ablation",
    "class_marginal_variational_energy",
    "shrink_eligible_weight",
)
