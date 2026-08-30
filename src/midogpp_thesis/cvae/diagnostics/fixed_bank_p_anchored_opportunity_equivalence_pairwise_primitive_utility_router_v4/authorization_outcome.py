"""Stable public façade for terminal v4 authorization outcomes.

Outcome schemas, persistence, failure exhaustion, and completed-run reopening
are kept in separate package-private modules.  Existing imports continue to
resolve through this façade.
"""

from .authorization_failure import finalize_failed_authorization
from .authorization_outcome_contracts import (
    OUTCOME_MEMBER,
    AuthorizationOutcomeReceipt,
)
from .authorization_outcome_recording import record_authorization_outcome
from .authorization_outcome_store import validate_authorization_outcome
from .complete_run_validation import validate_complete_run_bundle


__all__ = (
    "AuthorizationOutcomeReceipt",
    "OUTCOME_MEMBER",
    "finalize_failed_authorization",
    "record_authorization_outcome",
    "validate_authorization_outcome",
    "validate_complete_run_bundle",
)
