"""Frozen action, topology, and durable-member identities.

This module contains no experiment orchestration and no label access.  The
source and consumed-test phases deliberately have different seals so the test
cache cannot be admitted while the disagreement-regret model bank is mutable.
"""

from __future__ import annotations

from .experiment_contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS


BINARY_CLASSES = (0, 1)
B_ACTION_ID = "B"
U_ACTION_ID = "U"
GEOMETRY_IDS = ("A0", "A1")

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0
SOURCE_ROWS_PER_CLASS = 270

FEATURE_DIM = 3_840
EXPECTED_SOURCE_ROWS = 9_648
EXPECTED_TEST_ROWS = 9_928
EXPECTED_SOURCE_ROWS_BY_CENTER = {
    "0": 1_786,
    "1": 742,
    "2": 3_404,
    "3": 764,
    "5": 626,
    "6": 366,
    "7": 498,
    "8": 1_116,
    "9": 346,
}
EXPECTED_TEST_ROWS_BY_CENTER = {
    "0": 1_532,
    "1": 866,
    "2": 3_210,
    "3": 1_278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}

PHYSICAL_ACTION_COUNT_PER_TARGET = 18
SEED_PAIR_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_TASK_COUNT = len(CENTERS) * SEED_PAIR_COUNT
EXPECTED_CLASSIFIER_FIT_COUNT = (
    EXPECTED_TASK_COUNT * PHYSICAL_ACTION_COUNT_PER_TARGET
)

ACTION_LIBRARY_MEMBER = "manifests/action_library.json"
CLASSIFIER_MEAN_MEMBER = "arrays/action_classifier_scaler_mean.npy"
CLASSIFIER_SCALE_MEMBER = "arrays/action_classifier_scaler_scale.npy"
CLASSIFIER_COEFFICIENT_MEMBER = "arrays/action_classifier_coefficients.npy"
CLASSIFIER_INTERCEPT_MEMBER = "arrays/action_classifier_intercepts.npy"
CLASSIFIER_INDEX_MEMBER = "manifests/action_classifier_bank_index.json"
CLASSIFIER_SEAL_MEMBER = "manifests/action_classifier_bank_seal.json"
SOURCE_ARRAY_MEMBER = "arrays/source_action_probabilities.npz"
SOURCE_INDEX_MEMBER = "manifests/source_prediction_index.json"
SOURCE_SEAL_MEMBER = "manifests/source_prediction_seal.json"
TEST_ARRAY_MEMBER = "arrays/test_action_probabilities.npz"
TEST_INDEX_MEMBER = "manifests/test_prediction_index.json"
TEST_SEAL_MEMBER = "manifests/test_prediction_seal.json"

SOURCE_CHECKPOINT_DIRECTORY = "checkpoints/disagreement_regret_source_predictions"
TEST_CHECKPOINT_DIRECTORY = "checkpoints/disagreement_regret_test_predictions"
SOURCE_SCRATCH_ARRAY = "source_embeddings.npy"
TEST_SCRATCH_ARRAY = "test_embeddings.npy"
PREDICTION_BATCH_ROWS = 256

OPAQUE_SOURCE_ID_NAMESPACE = (
    "midogpp_fixed_bank_disagreement_regret_prediction_only_source_row_v1"
)


def candidate_sources(target: object) -> tuple[str, ...]:
    center = str(target)
    if center not in CENTERS:
        from ...protocol import ProtocolError

        raise ProtocolError(f"Unknown MIDOG++ outer target: {center}.")
    return tuple(value for value in CENTERS if value != center)


def geometry_action_id(geometry: object, source: object) -> str:
    geometry_id = str(geometry)
    source_id = str(source)
    if geometry_id not in GEOMETRY_IDS or source_id not in CENTERS:
        from ...protocol import ProtocolError

        raise ProtocolError("Prediction-only action identity is invalid.")
    return f"{geometry_id}::source={source_id}"


def source_from_action_id(action_id: object) -> str | None:
    value = str(action_id)
    if value in (B_ACTION_ID, U_ACTION_ID):
        return None
    prefix, separator, source = value.partition("::source=")
    if separator != "::source=" or prefix not in GEOMETRY_IDS or source not in CENTERS:
        from ...protocol import ProtocolError

        raise ProtocolError("Prediction-only action identity is invalid.")
    return source


__all__ = tuple(
    name
    for name in globals()
    if name.isupper()
    or name in {"candidate_sources", "geometry_action_id", "source_from_action_id"}
)
