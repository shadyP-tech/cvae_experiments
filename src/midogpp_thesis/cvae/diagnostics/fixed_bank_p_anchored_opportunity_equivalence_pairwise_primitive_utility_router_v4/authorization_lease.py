"""Compatibility façade for modular OE-PPUR v4 authorization lifecycle."""

from .authorization_outcome import (
    AuthorizationOutcomeReceipt,
    finalize_failed_authorization,
    record_authorization_outcome,
    validate_authorization_outcome,
    validate_complete_run_bundle,
)
from .completion_transaction import (
    COMPLETION_ABORT_MEMBER,
    COMPLETION_COMMIT_MEMBER,
    CompletionCommitReceipt,
    InterruptedCompletionReceipt,
    discover_completion_commit,
    record_completion_abort,
    record_completion_commit,
    validate_completion_commit,
)
from .lease_claim import (
    ACQUISITION_FAILURE_MEMBER,
    AuthorizationAcquisitionFailureReceipt,
    AuthorizationLeaseClaim,
    LEASE_DIRECTORY_NAME,
    assert_authorization_unclaimed,
    canonical_authorization_lease_path,
    claim_authorization_lease,
    discover_authorization_acquisition,
    validate_authorization_lease,
)


__all__ = (
    "ACQUISITION_FAILURE_MEMBER",
    "AuthorizationAcquisitionFailureReceipt",
    "AuthorizationLeaseClaim",
    "AuthorizationOutcomeReceipt",
    "COMPLETION_ABORT_MEMBER",
    "COMPLETION_COMMIT_MEMBER",
    "CompletionCommitReceipt",
    "InterruptedCompletionReceipt",
    "LEASE_DIRECTORY_NAME",
    "assert_authorization_unclaimed",
    "canonical_authorization_lease_path",
    "claim_authorization_lease",
    "discover_authorization_acquisition",
    "discover_completion_commit",
    "finalize_failed_authorization",
    "record_authorization_outcome",
    "record_completion_abort",
    "record_completion_commit",
    "validate_authorization_outcome",
    "validate_authorization_lease",
    "validate_complete_run_bundle",
    "validate_completion_commit",
)
