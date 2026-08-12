"""Frozen topology for the consumed-test hierarchical multi-challenger router."""

from __future__ import annotations

from ...protocol import ProtocolError
from .experiment_contracts import (
    ACTION_COUNT_PER_TARGET,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CENTERS,
    EXPECTED_MANIFEST_SHA256,
    FEATURE_NAMES,
    GENERATION_SEEDS,
    MARGIN_Z,
    METHOD_IDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PRE_EVALUATION_METHOD_IDS,
    PRIMARY_METHOD_ID,
    SCRATCH_ROOT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_ORACLE_IDS,
    TOP_K,
    TRAINING_SEEDS,
    WORKSTATION_PROFILE,
)


BINARY_CLASSES = (0, 1)
SEED_PAIR_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)

B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_PREFIX = "A1::source="
MODEL_FAMILIES = ("G", "R", "P")

HARD_THRESHOLD = 0.5
SAFE_Z = MARGIN_Z

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0
# B-referenced features are computed before any label capability opens.  The
# hierarchical model compares action gains additively, so no pairwise tensor is
# part of this experiment's input surface.
PERMUTATION_SEED = OOF_FOLD_SEED

EXPECTED_TEST_ROW_COUNT = 9_928
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
TARGET_ACTION_IDENTITY_COUNT = len(CENTERS) * ACTION_COUNT_PER_TARGET
PERSISTENT_GPU_WORKERS = 2
CPU_MODEL_WORKERS = 4
BLAS_THREADS_PER_CPU_WORKER = 3

if (
    TARGET_TASK_COUNT != len(CENTERS) * SEED_PAIR_COUNT
    or TARGET_PROBABILITY_CELL_COUNT
    != TARGET_ACTION_IDENTITY_COUNT * SEED_PAIR_COUNT
):
    raise RuntimeError("Frozen multi-challenger topology constants are inconsistent.")


def candidate_sources(target: object) -> tuple[str, ...]:
    center = str(target)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ target center: {center}.")
    return tuple(value for value in CENTERS if value != center)


def a1_action_id(source: object) -> str:
    center = str(source)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ source center: {center}.")
    return f"{A1_PREFIX}{center}"


def legal_actions(target: object) -> frozenset[str]:
    """Return the closed physical action set for one target center."""

    center = str(target)
    return frozenset(
        (B_ACTION_ID, U_ACTION_ID, *(a1_action_id(source) for source in candidate_sources(center)))
    )


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "candidate_sources",
    "legal_actions",
)
