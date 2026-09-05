"""Bounded progress reporting for the HARP v19 workstation runner."""

from __future__ import annotations

from ...protocol import ProtocolError


MAX_CLASSIFIER_PROGRESS_REPORTS = 32


def classifier_progress_due(
    completed: int,
    total: int,
    *,
    maximum_reports: int = MAX_CLASSIFIER_PROGRESS_REPORTS,
) -> bool:
    """Return whether one classifier completion should reach the terminal.

    The 5,184-task source cross-fit remains unchanged.  Only log emission is
    throttled so terminal rendering and ``tee`` do not become part of the
    scientific hot path.
    """

    if (
        type(completed) is not int
        or type(total) is not int
        or type(maximum_reports) is not int
        or total < 1
        or completed < 1
        or completed > total
        or maximum_reports < 1
    ):
        raise ProtocolError("HARP v19 classifier progress state is malformed.")
    stride = max(1, (total + maximum_reports - 1) // maximum_reports)
    return completed in {1, total} or completed % stride == 0


__all__ = ("MAX_CLASSIFIER_PROGRESS_REPORTS", "classifier_progress_due")
