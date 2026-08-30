"""Stable facade for OE-PPUR v4 fresh-process attestation.

Contracts, artifact-only validators, and one-shot spawn execution live in
separate package-private modules.  Importers keep this path as the public edge.
"""

from .attestation_contracts import (
    FinalAggregateAttestationReceipt,
    _FINAL_ATTESTATION_TOKEN,
    _issue_final_aggregate_attestation,
    _reconstruct_final_aggregate_attestation,
)
from .attestation_validation import (
    _validate_preterminal_files,
    _validate_terminal_aggregate_file,
)
from .attestation_workers import (
    _run_two_fresh_spawn_workers,
    _spawn_attestation_worker,
    _spawn_final_attestation_worker,
    _validator_runtime_sha256,
    attest_preterminal_artifact_twice,
    attest_terminal_aggregate_twice,
)


__all__ = (
    "FinalAggregateAttestationReceipt",
    "_reconstruct_final_aggregate_attestation",
    "_validate_preterminal_files",
    "attest_preterminal_artifact_twice",
    "attest_terminal_aggregate_twice",
)
