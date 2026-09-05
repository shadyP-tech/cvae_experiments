"""Fail-closed HARP v19 lease handling for production exceptions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...runtime.artifact_io import atomic_json
from .identity import PUBLICATION_STATUS, TERMINAL_DECISION
from .source_train_label_access_fence import (
    SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER,
    source_train_label_access_has_begun,
)


def handle_run_failure(
    *,
    root: Path,
    lease: object | None,
    ledger: object,
    error: BaseException,
    finalize_authorization: Callable[..., Any],
    announce: Callable[[str], None],
) -> str:
    """Exhaust any lease whose durable source-train-label fence has committed."""

    if lease is None:
        return "NO_LEASE"
    fence_present = source_train_label_access_has_begun(root)
    labels_opened = bool(getattr(ledger, "development_labels_opened", False))
    if not (labels_opened or fence_present):
        announce("LABEL_FREE_RECOVERY_RETAINED")
        return "LABEL_FREE_RECOVERY_RETAINED"
    atomic_json(
        Path(root) / "reports/failure_report.json",
        {
            "schema_version": "midogpp_harp_v19_failure_report_v1",
            "status": "FAILED_EXHAUSTED",
            "phase_order": list(getattr(ledger, "observed", ())),
            "error_class": error.__class__.__name__,
            "error": str(error)[:2000],
            "source_train_label_access_fence_member": SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER,
            "source_train_label_access_begun": fence_present,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
        },
    )
    finalize_authorization(lease, status="FAILED_EXHAUSTED", error=str(error))
    return "FAILED_EXHAUSTED"


__all__ = ("handle_run_failure",)
