"""Durable no-overwrite store for terminal authorization outcomes."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .authorization_outcome_contracts import (
    OUTCOME_MEMBER,
    AuthorizationOutcomeReceipt,
    outcome_receipt,
)
from .completion_transaction import COMPLETION_ABORT_MEMBER
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import (
    fsync_directory,
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)


def persist_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    payload: Mapping[str, object],
) -> AuthorizationOutcomeReceipt:
    validated = validate_authorization_lease(claim)
    publish_json_no_overwrite(
        validated.path / OUTCOME_MEMBER,
        payload,
        role="authorization outcome",
    )
    fsync_directory(validated.path)
    receipt = outcome_receipt(validated.path, payload)
    return validate_authorization_outcome(validated, expected=receipt)


def validate_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    expected: AuthorizationOutcomeReceipt,
) -> AuthorizationOutcomeReceipt:
    validated = validate_authorization_lease(claim)
    if type(expected) is not AuthorizationOutcomeReceipt:
        raise ProtocolError("OE-PPUR v4 authorization outcome receipt is untyped.")
    if pending_publications(validated.path, OUTCOME_MEMBER):
        raise ProtocolError("OE-PPUR v4 authorization outcome is interrupted.")
    payload = read_json_regular(
        validated.path / OUTCOME_MEMBER,
        role="authorization outcome",
    )
    receipt = outcome_receipt(validated.path, payload)
    abort_path = validated.path / COMPLETION_ABORT_MEMBER
    if receipt.status == "COMPLETE" and (
        abort_path.exists()
        or abort_path.is_symlink()
        or pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
    ):
        raise ProtocolError("OE-PPUR v4 complete authorization outcome was aborted.")
    if receipt != expected or receipt.claim_hash != validated.claim_hash:
        raise ProtocolError("OE-PPUR v4 authorization outcome changed after issuance.")
    return receipt


__all__ = ("validate_authorization_outcome",)
