"""Frozen scientific constants for the terminal pooled-BACC v2 diagnostic."""

from __future__ import annotations


MIDOGPP_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
BASELINE_ACTION_ID = "B"
EXPECTED_CENTER_COUNT = 9
EXPECTED_CANDIDATE_COUNT = 8
EXPECTED_ACTION_COUNT = 9
EXPECTED_FOLD_COUNT = 5
EXPECTED_FOLD_DECISION_COUNT = EXPECTED_CENTER_COUNT * EXPECTED_FOLD_COUNT
EXPECTED_SEED_PAIR_COUNT = 9
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_MIXED_CASE_COUNT = 213
EXPECTED_NEGATIVE_ONLY_CASE_COUNT = 4
EXPECTED_POSITIVE_ONLY_CASE_COUNT = 1
DEFAULT_PERMUTATION_COUNT = 10_000
DEFAULT_PARTITION_SEED = 90_902_026
DEFAULT_PERMUTATION_SEED = 90_912_026
DEFAULT_VARIANCE_FLOOR = 1.0e-6
DEFAULT_CONFIDENCE_MULTIPLIER = 1.96
DEFAULT_MINIMUM_GAIN = 0.0
DEFAULT_TIE_TOLERANCE = 1.0e-12
UTILITY_ID = "pooled_exact_bacc"
UNCERTAINTY_UNIT = "paired_whole_case_cluster"
TERMINAL_DECISION = "DO_NOT_PROMOTE"


def candidate_actions(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ValueError(f"Unknown MIDOG++ target center: {target}")
    return tuple(center for center in MIDOGPP_CENTERS if center != target)


def action_ids(target_center: str) -> tuple[str, ...]:
    return (BASELINE_ACTION_ID, *candidate_actions(target_center))


def routing_challengers(target_center: str, global_action_id: str) -> tuple[str, ...]:
    """Return source challengers, excluding a source-valued G_H itself."""

    candidates = candidate_actions(target_center)
    global_action = str(global_action_id)
    if global_action not in (BASELINE_ACTION_ID, *candidates):
        raise ValueError("Global action is outside the fixed non-target bank.")
    return tuple(action for action in candidates if action != global_action)


def legal_donor_centers(
    target_center: str,
    challenger_action_id: str,
    reference_action_id: str = BASELINE_ACTION_ID,
) -> tuple[str, ...]:
    """Centers on which both source actions are legal and H labels are excluded."""

    target = str(target_center)
    challenger = str(challenger_action_id)
    reference = str(reference_action_id)
    if target not in MIDOGPP_CENTERS or challenger not in candidate_actions(target):
        raise ValueError("Invalid target/challenger for legal donor construction.")
    excluded = {target, challenger}
    if reference != BASELINE_ACTION_ID:
        if reference not in candidate_actions(target) or reference == challenger:
            raise ValueError("Source reference must be a distinct legal source challenger.")
        excluded.add(reference)
    elif reference != BASELINE_ACTION_ID:
        raise ValueError("Unknown reference action.")
    return tuple(center for center in MIDOGPP_CENTERS if center not in excluded)


__all__ = (
    "BASELINE_ACTION_ID",
    "DEFAULT_CONFIDENCE_MULTIPLIER",
    "DEFAULT_MINIMUM_GAIN",
    "DEFAULT_PARTITION_SEED",
    "DEFAULT_PERMUTATION_COUNT",
    "DEFAULT_PERMUTATION_SEED",
    "DEFAULT_TIE_TOLERANCE",
    "DEFAULT_VARIANCE_FLOOR",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CENTER_COUNT",
    "EXPECTED_FOLD_COUNT",
    "EXPECTED_FOLD_DECISION_COUNT",
    "EXPECTED_MIXED_CASE_COUNT",
    "EXPECTED_NEGATIVE_ONLY_CASE_COUNT",
    "EXPECTED_POSITIVE_ONLY_CASE_COUNT",
    "EXPECTED_SEED_PAIR_COUNT",
    "EXPECTED_TOTAL_CASE_COUNT",
    "MIDOGPP_CENTERS",
    "TERMINAL_DECISION",
    "UNCERTAINTY_UNIT",
    "UTILITY_ID",
    "action_ids",
    "candidate_actions",
    "legal_donor_centers",
    "routing_challengers",
)
