"""Predeclared scientific constants for the terminal Stage-90 ceiling."""

from __future__ import annotations


MIDOGPP_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
BASELINE_ACTION_ID = "B"
EXPECTED_CENTER_COUNT = 9
EXPECTED_CANDIDATE_COUNT = 8
EXPECTED_ACTION_COUNT = 9
EXPECTED_FOLD_COUNT = 5
EXPECTED_SEED_PAIR_COUNT = 9
EXPECTED_TOTAL_CASE_COUNT = 218


def candidate_actions(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ValueError(f"Unknown MIDOG++ target center: {target}")
    return tuple(center for center in MIDOGPP_CENTERS if center != target)


def action_ids(target_center: str) -> tuple[str, ...]:
    return (BASELINE_ACTION_ID, *candidate_actions(target_center))


__all__ = (
    "BASELINE_ACTION_ID",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CENTER_COUNT",
    "EXPECTED_FOLD_COUNT",
    "EXPECTED_SEED_PAIR_COUNT",
    "EXPECTED_TOTAL_CASE_COUNT",
    "MIDOGPP_CENTERS",
    "action_ids",
    "candidate_actions",
)
