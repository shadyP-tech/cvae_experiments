"""Public label-free probability-matrix science surface for OE-PPUR v2."""

from .probability_matrix import (
    parse_probability_matrix_shards,
    validate_parsed_probability_matrix_science_receipt,
)
from .probability_matrix_receipts import (
    EXPECTED_PROBABILITY_COLUMNS,
    ParsedProbabilityMatrixScienceReceipt,
    ProbabilityMatrixShardSpec,
)
from ..row_binding import (
    CanonicalAdmittedRowBindingReceipt,
    derive_admitted_row_binding,
    validate_admitted_row_binding,
)


__all__ = (
    "EXPECTED_PROBABILITY_COLUMNS",
    "CanonicalAdmittedRowBindingReceipt",
    "ParsedProbabilityMatrixScienceReceipt",
    "ProbabilityMatrixShardSpec",
    "derive_admitted_row_binding",
    "parse_probability_matrix_shards",
    "validate_parsed_probability_matrix_science_receipt",
    "validate_admitted_row_binding",
)
