"""Irreversible terminal-label access journal for CBPUPR v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_io import persist_json
from .constants import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    EXPECTED_TEST_ROW_COUNT,
    OUTPUT_ARTIFACT_ID,
)
from .experiment_contracts import AUTHORIZATION_BASIS
from .hashing import canonical_hash


TERMINAL_ACCESS_INTENT_MEMBER = "manifests/terminal_label_access_intent.json"
TERMINAL_ACCESS_OPENED_MEMBER = (
    "reports/terminal_label_access_opened_receipt.json"
)
TERMINAL_ACCESS_JOURNAL_ORDER = (
    TERMINAL_ACCESS_INTENT_MEMBER,
    TERMINAL_ACCESS_OPENED_MEMBER,
)


def persist_terminal_label_access_intent(
    root: Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    """Persist intent immediately before invoking the terminal label loader."""

    if (
        expected_checks.get("terminal_opened") is not False
        or expected_checks.get("terminal_product_count") != 0
        or expected_checks.get("formal_claim_authorized") is not False
    ):
        raise ProtocolError("CBPUPR terminal access intent gate drifted.")
    payload = {
        "schema_version": "fixed_bank_cbpupr_terminal_label_access_intent_v2",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "execution_revision": EXECUTION_REVISION,
        "execution_authorization_basis": AUTHORIZATION_BASIS,
        "phase": "TERMINAL_LABELS_METRICS_AND_CONTROLS",
        "preterminal_content_index_hash": expected_checks["content_index_hash"],
        "preterminal_validation_checks_hash": expected_checks[
            "validation_checks_hash"
        ],
        "preterminal_hash": expected_checks["preterminal_hash"],
        "target_terminal_label_access_requested": True,
        "consumed_test_data": True,
        "raw_labels_persisted": False,
        "cross_run_recovery_allowed": False,
        "predecessor_outputs_or_authorizations_reused": False,
        "formal_claim_authorized": False,
    }
    intent = {**payload, "terminal_access_intent_hash": canonical_hash(payload)}
    persist_json(root / TERMINAL_ACCESS_INTENT_MEMBER, intent)
    return intent


def persist_terminal_label_access_opened_receipt(
    root: Path,
    *,
    intent: Mapping[str, object],
    labels: Sequence[object],
) -> dict[str, object]:
    """Persist label-free proof immediately after the loader returns."""

    validate_terminal_label_access_intent(intent)
    if len(labels) != EXPECTED_TEST_ROW_COUNT:
        raise ProtocolError("CBPUPR terminal label count drifted after opening.")
    payload = {
        "schema_version": (
            "fixed_bank_cbpupr_terminal_label_access_opened_receipt_v2"
        ),
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "execution_revision": EXECUTION_REVISION,
        "execution_authorization_basis": AUTHORIZATION_BASIS,
        "terminal_access_intent_hash": intent["terminal_access_intent_hash"],
        "terminal_capability_opened": True,
        "terminal_label_count": EXPECTED_TEST_ROW_COUNT,
        "consumed_test_data": True,
        "raw_labels_persisted": False,
        "terminal_access_is_irreversible_for_this_run": True,
        "cross_run_recovery_allowed": False,
        "predecessor_outputs_or_authorizations_reused": False,
        "formal_claim_authorized": False,
    }
    receipt = {
        **payload,
        "terminal_access_opened_receipt_hash": canonical_hash(payload),
    }
    persist_json(root / TERMINAL_ACCESS_OPENED_MEMBER, receipt)
    return receipt


def validate_terminal_label_access_intent(
    intent: Mapping[str, object],
) -> None:
    unhashed = {
        key: value
        for key, value in intent.items()
        if key != "terminal_access_intent_hash"
    }
    if (
        set(intent)
        != {
            "schema_version",
            "experiment_id",
            "output_artifact_id",
            "execution_revision",
            "execution_authorization_basis",
            "phase",
            "preterminal_content_index_hash",
            "preterminal_validation_checks_hash",
            "preterminal_hash",
            "target_terminal_label_access_requested",
            "consumed_test_data",
            "raw_labels_persisted",
            "cross_run_recovery_allowed",
            "predecessor_outputs_or_authorizations_reused",
            "formal_claim_authorized",
            "terminal_access_intent_hash",
        }
        or intent.get("schema_version")
        != "fixed_bank_cbpupr_terminal_label_access_intent_v2"
        or intent.get("experiment_id") != EXPERIMENT_ID
        or intent.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or intent.get("execution_revision") != EXECUTION_REVISION
        or intent.get("execution_authorization_basis") != AUTHORIZATION_BASIS
        or intent.get("phase") != "TERMINAL_LABELS_METRICS_AND_CONTROLS"
        or intent.get("target_terminal_label_access_requested") is not True
        or intent.get("consumed_test_data") is not True
        or intent.get("raw_labels_persisted") is not False
        or intent.get("cross_run_recovery_allowed") is not False
        or intent.get("predecessor_outputs_or_authorizations_reused") is not False
        or intent.get("formal_claim_authorized") is not False
        or intent.get("terminal_access_intent_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR terminal label access intent drifted.")


def validate_terminal_label_access_journal(
    root: Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    intent = read_json(root / TERMINAL_ACCESS_INTENT_MEMBER)
    validate_terminal_label_access_intent(intent)
    if (
        intent.get("preterminal_content_index_hash")
        != expected_checks.get("content_index_hash")
        or intent.get("preterminal_validation_checks_hash")
        != expected_checks.get("validation_checks_hash")
        or intent.get("preterminal_hash") != expected_checks.get("preterminal_hash")
    ):
        raise ProtocolError("CBPUPR terminal access/preterminal binding drifted.")
    receipt = read_json(root / TERMINAL_ACCESS_OPENED_MEMBER)
    unhashed = {
        key: value
        for key, value in receipt.items()
        if key != "terminal_access_opened_receipt_hash"
    }
    if (
        set(receipt)
        != {
            "schema_version",
            "experiment_id",
            "output_artifact_id",
            "execution_revision",
            "execution_authorization_basis",
            "terminal_access_intent_hash",
            "terminal_capability_opened",
            "terminal_label_count",
            "consumed_test_data",
            "raw_labels_persisted",
            "terminal_access_is_irreversible_for_this_run",
            "cross_run_recovery_allowed",
            "predecessor_outputs_or_authorizations_reused",
            "formal_claim_authorized",
            "terminal_access_opened_receipt_hash",
        }
        or receipt.get("schema_version")
        != "fixed_bank_cbpupr_terminal_label_access_opened_receipt_v2"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or receipt.get("execution_revision") != EXECUTION_REVISION
        or receipt.get("execution_authorization_basis") != AUTHORIZATION_BASIS
        or receipt.get("terminal_access_intent_hash")
        != intent.get("terminal_access_intent_hash")
        or receipt.get("terminal_capability_opened") is not True
        or receipt.get("terminal_label_count") != EXPECTED_TEST_ROW_COUNT
        or receipt.get("consumed_test_data") is not True
        or receipt.get("raw_labels_persisted") is not False
        or receipt.get("terminal_access_is_irreversible_for_this_run") is not True
        or receipt.get("cross_run_recovery_allowed") is not False
        or receipt.get("predecessor_outputs_or_authorizations_reused") is not False
        or receipt.get("formal_claim_authorized") is not False
        or receipt.get("terminal_access_opened_receipt_hash")
        != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR terminal label access opened receipt drifted.")
    return receipt


__all__ = (
    "TERMINAL_ACCESS_INTENT_MEMBER",
    "TERMINAL_ACCESS_JOURNAL_ORDER",
    "TERMINAL_ACCESS_OPENED_MEMBER",
    "persist_terminal_label_access_intent",
    "persist_terminal_label_access_opened_receipt",
    "validate_terminal_label_access_intent",
    "validate_terminal_label_access_journal",
)
