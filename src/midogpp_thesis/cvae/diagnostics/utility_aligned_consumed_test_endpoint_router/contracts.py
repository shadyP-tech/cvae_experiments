"""Scientific identities for the consumed-test target-static endpoint router.

The package is deliberately Stage-90-local but depends only on the neutral
``routing.utility_aligned`` and ``routing.residual_topup`` primitives.  Labels
do not appear in any fitting, target-feature, policy, or action contract.
"""

from __future__ import annotations

from itertools import product

from ...protocol import ProtocolError
from ...routing.utility_aligned import (
    DEFAULT_CASE_BOOTSTRAP_SEED,
    MIN_SUPPORT_BOOTSTRAP_REPLICATES,
)
from .experiment_contracts import (
    CENTERS,
    DEVELOPMENT_RESPONSE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
    TRAINING_SEEDS,
)


BASE_ACTION_ID = "B"
UNIFORM_ACTION_ID = "U"
GLOBAL_ACTION_ID = "G"
ROUTED_ACTION_ID = "R"
PERMUTATION_ACTION_ID = "P"
H_X_E_ACTION_PREFIX = "Hxe::"

SOURCE_INNER_ACTION_ROLE = "source_inner_development_endpoint"
TARGET_ACTION_ROLE = "consumed_test_target_static_endpoint"
ORACLE_ACTION_ROLE = "terminal_Hxe_oracle_candidate"

PRIMARY_TARGET_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    ROUTED_ACTION_ID,
    PERMUTATION_ACTION_ID,
)
PRIMARY_CONTRASTS = (
    ("R-B", ROUTED_ACTION_ID, BASE_ACTION_ID),
    ("R-U", ROUTED_ACTION_ID, UNIFORM_ACTION_ID),
    ("R-G", ROUTED_ACTION_ID, GLOBAL_ACTION_ID),
    ("R-P", ROUTED_ACTION_ID, PERMUTATION_ACTION_ID),
)

SEED_PAIRS = tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))
SEED_PAIR_COUNT = len(SEED_PAIRS)
INNER_CANDIDATE_COUNT = len(CENTERS) - 2
TARGET_CANDIDATE_COUNT = len(CENTERS) - 1
EXPECTED_TARGET_ACTION_COUNT = len(PRIMARY_TARGET_ACTION_IDS) + TARGET_CANDIDATE_COUNT
EXPECTED_TERMINAL_SCORE_COUNT = len(CENTERS) * EXPECTED_TARGET_ACTION_COUNT
EXPECTED_CONTRAST_ROW_COUNT = len(CENTERS) * len(PRIMARY_CONTRASTS)

SUPPORT_BOOTSTRAP_SEED = DEFAULT_CASE_BOOTSTRAP_SEED
SUPPORT_BOOTSTRAP_REPLICATES = MIN_SUPPORT_BOOTSTRAP_REPLICATES
PERMUTATION_SEED = 90_902_026

EXPECTED_SUPPORT_ROW_COUNT = 2_902
EXPECTED_EVALUATION_ROW_COUNT = 7_026
EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER = {
    "0": 874,
    "1": 512,
    "2": 1_938,
    "3": 1_188,
    "5": 458,
    "6": 633,
    "7": 210,
    "8": 633,
    "9": 580,
}
DEVELOPMENT_RESPONSE_NAME = "exact_nine_probability_ensemble_bacc_delta"
M0_PREDICTOR_NAMES = ("global_source_control",)
M1_PREDICTOR_NAMES = (
    "global_source_control",
    "target_local::mean_support_row_absolute_exact_nine_ensemble_probability_shift_v2",
)


def candidate_sources(target_center: object) -> tuple[str, ...]:
    """Return the canonical eight-source pool with target expert ``H`` absent."""

    target = _center(target_center, "target_center")
    return tuple(center for center in CENTERS if center != target)


def inner_candidate_sources(
    outer_target: object,
    query_center: object,
) -> tuple[str, ...]:
    """Return the seven legal sources under strict ``H/q/e`` exclusion."""

    outer = _center(outer_target, "outer_target")
    query = _center(query_center, "query_center")
    if outer == query:
        raise ProtocolError("Development response requires q != H.")
    return tuple(center for center in CENTERS if center not in {outer, query})


def h_x_e_action_id(source_center: object) -> str:
    source = _center(source_center, "source_center")
    return f"{H_X_E_ACTION_PREFIX}{source}"


def h_x_e_source(action_id: object) -> str | None:
    value = str(action_id)
    if not value.startswith(H_X_E_ACTION_PREFIX):
        return None
    source = value.removeprefix(H_X_E_ACTION_PREFIX)
    return _center(source, "Hxe source")


def expected_target_action_ids(target_center: object) -> tuple[str, ...]:
    target = _center(target_center, "target_center")
    return (
        *PRIMARY_TARGET_ACTION_IDS,
        *(h_x_e_action_id(source) for source in candidate_sources(target)),
    )


def expected_development_action_ids(
    outer_target: object,
    query_center: object,
) -> tuple[str, ...]:
    return (
        BASE_ACTION_ID,
        *(
            h_x_e_action_id(source)
            for source in inner_candidate_sources(outer_target, query_center)
        ),
    )


def _center(value: object, name: str) -> str:
    text = str(value)
    if text not in CENTERS:
        raise ProtocolError(f"{name} is outside the frozen MIDOG++ center set.")
    return text


__all__ = (
    "BASE_ACTION_ID",
    "CENTERS",
    "DEVELOPMENT_RESPONSE_COUNT",
    "DEVELOPMENT_RESPONSE_NAME",
    "EXPECTED_CASE_COUNTS_BY_CENTER",
    "EXPECTED_CONTRAST_ROW_COUNT",
    "EXPECTED_EVALUATION_CASE_COUNT",
    "EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER",
    "EXPECTED_EVALUATION_ROW_COUNT",
    "EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER",
    "EXPECTED_SUPPORT_CASE_COUNT",
    "EXPECTED_SUPPORT_ROW_COUNT",
    "EXPECTED_TARGET_ACTION_COUNT",
    "EXPECTED_TERMINAL_SCORE_COUNT",
    "EXPECTED_TEST_ROW_COUNT",
    "EXPECTED_TOTAL_CASE_COUNT",
    "GENERATION_SEEDS",
    "GLOBAL_ACTION_ID",
    "H_X_E_ACTION_PREFIX",
    "INNER_CANDIDATE_COUNT",
    "M0_PREDICTOR_NAMES",
    "M1_PREDICTOR_NAMES",
    "ORACLE_ACTION_ROLE",
    "PERMUTATION_ACTION_ID",
    "PERMUTATION_SEED",
    "PRIMARY_CONTRASTS",
    "PRIMARY_TARGET_ACTION_IDS",
    "ROUTED_ACTION_ID",
    "SEED_PAIRS",
    "SEED_PAIR_COUNT",
    "SOURCE_INNER_ACTION_ROLE",
    "SUPPORT_BOOTSTRAP_REPLICATES",
    "SUPPORT_BOOTSTRAP_SEED",
    "SUPPORT_CASE_COUNT_PER_CENTER",
    "SUPPORT_PARTITION_NAMESPACE",
    "TARGET_ACTION_ROLE",
    "TARGET_CANDIDATE_COUNT",
    "TRAINING_SEEDS",
    "UNIFORM_ACTION_ID",
    "candidate_sources",
    "expected_development_action_ids",
    "expected_target_action_ids",
    "h_x_e_action_id",
    "h_x_e_source",
    "inner_candidate_sources",
)
