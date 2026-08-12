from __future__ import annotations

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.config_payloads import (
    canonical_evaluation_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.diagnostic_outcome import (
    diagnostic_recoverability_outcome,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _rows(values: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    contrasts = canonical_evaluation_payload()["primary_contrasts"]
    return tuple(
        {
            "row_role": "outer_center_aggregate",
            "contrast_id": contrast,
            "one_sided_95_lcb": value,
        }
        for contrast, value in zip(contrasts, values, strict=True)
    )


def test_diagnostic_recoverability_passes_only_when_all_five_lcbs_are_positive() -> None:
    evaluation = canonical_evaluation_payload()
    passed = diagnostic_recoverability_outcome(
        _rows((0.1, 0.2, 0.3, 0.4, 1e-12)), evaluation=evaluation
    )
    failed_zero = diagnostic_recoverability_outcome(
        _rows((0.1, 0.2, 0.3, 0.4, 0.0)), evaluation=evaluation
    )
    failed_negative = diagnostic_recoverability_outcome(
        _rows((0.1, -1e-12, 0.3, 0.4, 0.5)), evaluation=evaluation
    )

    assert passed["status"] == "PASS"
    assert passed["all_five_lcbs_strictly_greater_than_zero"] is True
    assert failed_zero["status"] == "FAIL"
    assert failed_negative["status"] == "FAIL"
    assert all(result["routing_success_claimed"] is False for result in (
        passed, failed_zero, failed_negative
    ))


def test_diagnostic_recoverability_rejects_missing_or_duplicate_aggregate() -> None:
    evaluation = canonical_evaluation_payload()
    rows = _rows((0.1, 0.2, 0.3, 0.4, 0.5))
    with pytest.raises(ProtocolError, match="topology drifted"):
        diagnostic_recoverability_outcome(rows[:-1], evaluation=evaluation)
    with pytest.raises(ProtocolError, match="topology drifted"):
        diagnostic_recoverability_outcome((*rows, rows[0]), evaluation=evaluation)
