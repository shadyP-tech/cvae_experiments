from __future__ import annotations

from midogpp_thesis.cvae.preservation.prior_recovery_decision import paired_worst_center_guard


def test_worst_center_guard_is_paired_by_center() -> None:
    passed, deltas = paired_worst_center_guard(
        {"0": 0.70, "1": 0.61},
        {"0": 0.60, "1": 0.80},
        tolerance=0.01,
    )
    assert not passed
    assert deltas == {"0": 0.10, "1": -0.19}
