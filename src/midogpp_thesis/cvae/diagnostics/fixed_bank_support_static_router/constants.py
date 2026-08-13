"""Frozen scientific constants for the consumed-test support-static diagnostic."""

from __future__ import annotations

from ...protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
MIDOGPP_CENTERS = CENTERS
BINARY_CLASSES = (0, 1)
SEED_PAIR_COUNT = 9

B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_PREFIX = "A1::source="
METHOD_IDS = ("B", "U", "G_static", "S4", "O_static", "O_case")
PRE_EVALUATION_METHOD_IDS = METHOD_IDS[:-2]
TERMINAL_ORACLE_IDS = METHOD_IDS[-2:]

HARD_THRESHOLD = 0.5
OOF_FOLD_COUNT = 5
PARTITION_SEED = 90_902_026
OOF_FOLD_SEED = PARTITION_SEED
PERMUTATION_COUNT = 10_000
PERMUTATION_SEED = 90_912_026
PARTITION_NAMESPACE = "midogpp_fixed_bank_support_static_router_s4_test_folds_v1"
OOF_PARTITION_NAMESPACE = PARTITION_NAMESPACE
NULL_DERANGEMENT_ALGORITHM = (
    "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_shift_1_to_7_v1"
)

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0

ACTION_COUNT_PER_TARGET = 10
EXPECTED_ROUTE_COUNT = len(CENTERS) * OOF_FOLD_COUNT
EXPECTED_CENTER_FOLD_COUNT = EXPECTED_ROUTE_COUNT
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_CASE_COUNTS_BY_CENTER = {
    "0": 23,
    "1": 20,
    "2": 24,
    "3": 39,
    "5": 23,
    "6": 23,
    "7": 21,
    "8": 22,
    "9": 23,
}

DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
CLAIM_ROLE = "posthoc_known_fixed_bank_support_static_router_s4_diagnostic"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
TIE_TOLERANCE = 1.0e-12


def _center(value: object, role: str) -> str:
    center = str(value)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ {role} center: {center}.")
    return center


def candidate_sources(target: object) -> tuple[str, ...]:
    """Return the exact eight non-target sources in frozen numeric-center order."""

    target_center = _center(target, "target")
    return tuple(center for center in CENTERS if center != target_center)


def a1_action_id(source: object) -> str:
    return f"{A1_PREFIX}{_center(source, 'source')}"


def decision_action_ids(target: object) -> tuple[str, ...]:
    """B plus eight A1 candidates; U is deliberately not a decision candidate."""

    return (B_ACTION_ID, *(a1_action_id(source) for source in candidate_sources(target)))


def physical_action_ids(target: object) -> tuple[str, ...]:
    return (B_ACTION_ID, U_ACTION_ID, *(a1_action_id(source) for source in candidate_sources(target)))


def source_from_action(action_id: object) -> str:
    action = str(action_id)
    if not action.startswith(A1_PREFIX):
        raise ProtocolError("Only A1 actions have a selected source.")
    return _center(action[len(A1_PREFIX) :], "action source")


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "candidate_sources",
    "decision_action_ids",
    "physical_action_ids",
    "source_from_action",
)
