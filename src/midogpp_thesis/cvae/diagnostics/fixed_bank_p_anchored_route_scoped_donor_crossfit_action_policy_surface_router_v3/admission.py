"""Compatibility façade for the modular P-DCAPS v3 admission layer.

Schemas, evidence serialization, and gate computation live in separate
cohesive modules. This façade preserves the public imports used by the router
and existing focused tests.
"""

from .admission_gate import (
    OUTER_ADMISSION_SCHEMA,
    OuterAdmission,
    build_outer_admission,
)
from .nullable_statistics import (
    ADMISSION_STATISTIC_NAMES,
    CONSTANT_RANK_UNDEFINED_REASON,
    DENOMINATOR_UNDEFINED_REASON,
    NULLABLE_STATISTIC_SCHEMA,
    NullableStatistic,
)
from .pseudo_evidence import PSEUDO_EVIDENCE_SCHEMA, PseudoPolicyEvidence


__all__ = (
    "ADMISSION_STATISTIC_NAMES",
    "CONSTANT_RANK_UNDEFINED_REASON",
    "DENOMINATOR_UNDEFINED_REASON",
    "NULLABLE_STATISTIC_SCHEMA",
    "OUTER_ADMISSION_SCHEMA",
    "PSEUDO_EVIDENCE_SCHEMA",
    "NullableStatistic",
    "OuterAdmission",
    "PseudoPolicyEvidence",
    "build_outer_admission",
)
