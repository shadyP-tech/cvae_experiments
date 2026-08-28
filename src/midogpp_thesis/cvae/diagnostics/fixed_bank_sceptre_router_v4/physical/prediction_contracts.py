"""Immutable geometry and public value contracts for SCEPTRE v4 prediction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.generation.contracts import (
    COMMON_OUTPUT_DIM,
    SOURCE_BUDGET_PER_CLASS,
    TOTAL_PER_CLASS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec

from .prediction_io import canonical_sha256
from .source_streams import SourceGeometry


CANDIDATE_ARRAY_MEMBER = "arrays/sceptre_v4_candidate_probabilities.npy"
EXACT_B_ARRAY_MEMBER = "arrays/sceptre_v4_exact_b_probabilities.npy"
PREDICTION_INDEX_MEMBER = "manifests/sceptre_v4_prediction_index.json"
PREDICTION_RECEIPT_MEMBER = "manifests/sceptre_v4_prediction_receipt.json"
CHECKPOINT_DIRECTORY = "checkpoints/sceptre_v4_prediction_surface"
EVALUATION_SCRATCH_MEMBER = f"{CHECKPOINT_DIRECTORY}/evaluation_embeddings.npy"
EVALUATION_FRAME_MEMBER = f"{CHECKPOINT_DIRECTORY}/evaluation_frame.json"
CANDIDATE_EXCLUSION_SENTINEL = np.float32(-1.0)
CPU_PREDICTION_WORKERS = 4

EXPECTED_TEST_ROWS_BY_CENTER = (
    ("0", 1_532),
    ("1", 866),
    ("2", 3_210),
    ("3", 1_278),
    ("5", 628),
    ("6", 742),
    ("7", 282),
    ("8", 726),
    ("9", 664),
)

LOCKED_CLASSIFIER_SPEC = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


class PredictionRuntimeConfig(Protocol):
    runtime: Mapping[str, object]
    classifier: ClassifierSpec
    config_hash: str


@dataclass(frozen=True)
class PredictionGeometry:
    centers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    source_rows_per_class: int
    exact_b_prefix_per_source_class: int
    feature_dim: int
    rows_by_center: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        counts = dict(self.rows_by_center)
        if (
            not self.centers
            or len(set(self.centers)) != len(self.centers)
            or tuple(counts) != self.centers
            or any(value <= 0 for value in counts.values())
            or not self.training_seeds
            or len(set(self.training_seeds)) != len(self.training_seeds)
            or not self.generation_seeds
            or len(set(self.generation_seeds)) != len(self.generation_seeds)
            or self.source_rows_per_class <= 0
            or self.exact_b_prefix_per_source_class <= 0
            or self.exact_b_prefix_per_source_class > self.source_rows_per_class
            or self.feature_dim <= 0
            or (len(self.centers) - 1) * self.exact_b_prefix_per_source_class
            != self.source_rows_per_class
        ):
            raise ProtocolError("SCEPTRE v4 prediction geometry is malformed.")

    @property
    def evaluation_rows(self) -> int:
        return sum(value for _, value in self.rows_by_center)

    @property
    def seed_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(product(self.training_seeds, self.generation_seeds))

    @property
    def candidate_shape(self) -> tuple[int, int, int]:
        return len(self.seed_cells), len(self.centers), self.evaluation_rows

    @property
    def exact_b_shape(self) -> tuple[int, int]:
        return len(self.seed_cells), self.evaluation_rows

    @property
    def fit_count(self) -> int:
        return len(self.seed_cells) * 2 * len(self.centers)

    def row_slice(self, center: str) -> slice:
        target = str(center)
        if target not in self.centers:
            raise ProtocolError("SCEPTRE v4 prediction row center is unknown.")
        start = 0
        for observed, count in self.rows_by_center:
            stop = start + count
            if observed == target:
                return slice(start, stop)
            start = stop
        raise ProtocolError("SCEPTRE v4 prediction row-center inventory drifted.")

    @property
    def source_geometry(self) -> SourceGeometry:
        return SourceGeometry(
            centers=self.centers,
            training_seeds=self.training_seeds,
            generation_seeds=self.generation_seeds,
            rows_per_class=self.source_rows_per_class,
            feature_dim=self.feature_dim,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "centers": list(self.centers),
            "training_seeds": list(self.training_seeds),
            "generation_seeds": list(self.generation_seeds),
            "source_rows_per_class": self.source_rows_per_class,
            "exact_b_prefix_per_source_class": self.exact_b_prefix_per_source_class,
            "feature_dim": self.feature_dim,
            "rows_by_center": dict(self.rows_by_center),
            "evaluation_rows": self.evaluation_rows,
            "seed_cells": [list(value) for value in self.seed_cells],
            "candidate_shape": list(self.candidate_shape),
            "exact_b_shape": list(self.exact_b_shape),
            "fit_count": self.fit_count,
        }


PRODUCTION_PREDICTION_GEOMETRY = PredictionGeometry(
    centers=tuple(CENTERS),
    training_seeds=tuple(TRAINING_SEEDS),
    generation_seeds=tuple(GENERATION_SEEDS),
    source_rows_per_class=TOTAL_PER_CLASS,
    exact_b_prefix_per_source_class=SOURCE_BUDGET_PER_CLASS,
    feature_dim=COMMON_OUTPUT_DIM,
    rows_by_center=EXPECTED_TEST_ROWS_BY_CENTER,
)


@dataclass(frozen=True)
class FitOutcome:
    """Narrow result seam used by the production classifier and test doubles."""

    positive_probabilities: np.ndarray
    predictions: np.ndarray
    converged: bool
    classifier_config_hash: str
    scaler_state_hash: str


FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, ClassifierSpec], FitOutcome]


@dataclass(frozen=True)
class PredictionRuntimeTestMode:
    """Explicit small-geometry, serial execution seam for focused tests only."""

    geometry: PredictionGeometry
    fit_predict: FitPredict | None = None

    def __post_init__(self) -> None:
        if self.geometry == PRODUCTION_PREDICTION_GEOMETRY:
            raise ProtocolError("Production prediction geometry cannot use the test seam.")
        if self.fit_predict is not None and not callable(self.fit_predict):
            raise ProtocolError("SCEPTRE v4 prediction test fit seam is not callable.")


@dataclass(frozen=True)
class PredictionSurface:
    """Validated read-only candidate/exact-B probability surface."""

    root: Path
    candidate_array_path: Path
    exact_b_array_path: Path
    index_path: Path
    receipt_path: Path
    geometry: PredictionGeometry
    row_ids: tuple[str, ...]
    row_centers: tuple[str, ...]
    index: Mapping[str, object]
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            len(self.row_ids) != self.geometry.evaluation_rows
            or len(self.row_centers) != self.geometry.evaluation_rows
            or len(set(self.row_ids)) != len(self.row_ids)
            or set(self.row_centers) != set(self.geometry.centers)
            or any(
                tuple(self.row_centers[self.geometry.row_slice(center)])
                != (center,) * dict(self.geometry.rows_by_center)[center]
                for center in self.geometry.centers
            )
        ):
            raise ProtocolError("SCEPTRE v4 prediction row inventory drifted.")
        object.__setattr__(self, "index", MappingProxyType(dict(self.index)))
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @property
    def receipt_hash(self) -> str:
        return str(self.receipt["receipt_sha256"])

    @property
    def attempt_id(self) -> str:
        return str(self.receipt["attempt_id"])

    @property
    def candidate_probabilities(self) -> np.ndarray:
        values = np.load(self.candidate_array_path, mmap_mode="r", allow_pickle=False)
        if values.flags.writeable:
            raise ProtocolError("SCEPTRE v4 candidate memmap unexpectedly became writable.")
        return values

    @property
    def exact_b_probabilities(self) -> np.ndarray:
        values = np.load(self.exact_b_array_path, mmap_mode="r", allow_pickle=False)
        if values.flags.writeable:
            raise ProtocolError("SCEPTRE v4 exact-B memmap unexpectedly became writable.")
        return values

    def candidate(
        self,
        training_seed: int,
        generation_seed: int,
        source_center: str,
    ) -> np.ndarray:
        try:
            seed_ordinal = self.geometry.seed_cells.index(
                (int(training_seed), int(generation_seed))
            )
            source_ordinal = self.geometry.centers.index(str(source_center))
        except ValueError as exc:
            raise ProtocolError("SCEPTRE v4 candidate prediction key is absent.") from exc
        return self.candidate_probabilities[seed_ordinal, source_ordinal]

    def exact_b(self, training_seed: int, generation_seed: int) -> np.ndarray:
        try:
            seed_ordinal = self.geometry.seed_cells.index(
                (int(training_seed), int(generation_seed))
            )
        except ValueError as exc:
            raise ProtocolError("SCEPTRE v4 exact-B prediction key is absent.") from exc
        return self.exact_b_probabilities[seed_ordinal]


def geometry_for(test_mode: PredictionRuntimeTestMode | None) -> PredictionGeometry:
    return PRODUCTION_PREDICTION_GEOMETRY if test_mode is None else test_mode.geometry


def config_hash(config: object) -> str:
    value = getattr(config, "config_hash", getattr(config, "contract_hash", ""))
    text = str(value)
    if not text:
        raise ProtocolError("SCEPTRE v4 prediction config hash is absent.")
    return text


def attempt_id(
    config: object,
    *,
    explicit: str | None,
    root: Path,
    synthetic: bool,
) -> str:
    raw = explicit if explicit is not None else getattr(config, "attempt_id", None)
    text = "" if raw is None else str(raw).strip()
    if not text and synthetic:
        text = canonical_sha256(
            {
                "schema_version": "sceptre_v4_synthetic_physical_attempt_v1",
                "root": str(root.resolve()),
                "config_hash": config_hash(config),
            }
        )
    if not text or len(text) > 256 or any(character.isspace() for character in text):
        raise ProtocolError("SCEPTRE v4 physical attempt identity is absent or malformed.")
    return text


def locked_classifier(config: object) -> ClassifierSpec:
    classifier = getattr(config, "classifier", None)
    if classifier != LOCKED_CLASSIFIER_SPEC:
        raise ProtocolError("SCEPTRE v4 downstream classifier lock drifted.")
    return classifier


def assert_production_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("cpu_prediction_workers", -1)) != CPU_PREDICTION_WORKERS
        or int(runtime.get("blas_threads_per_worker", -1)) != 1
        or int(runtime.get("native_threads_per_worker", -1)) != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("top_level_spawn_pool_only") is not True
    ):
        raise ProtocolError("SCEPTRE v4 prediction workstation topology drifted.")


def assert_owned_root(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("SCEPTRE v4 prediction root is a symlink.")
    if root.exists() and not root.is_dir():
        raise ProtocolError("SCEPTRE v4 prediction root is not a directory.")
    root.mkdir(parents=True, exist_ok=True)


def final_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / CANDIDATE_ARRAY_MEMBER,
        root / EXACT_B_ARRAY_MEMBER,
        root / PREDICTION_INDEX_MEMBER,
        root / PREDICTION_RECEIPT_MEMBER,
    )


__all__ = (
    "CANDIDATE_ARRAY_MEMBER",
    "CANDIDATE_EXCLUSION_SENTINEL",
    "CHECKPOINT_DIRECTORY",
    "CPU_PREDICTION_WORKERS",
    "EVALUATION_FRAME_MEMBER",
    "EVALUATION_SCRATCH_MEMBER",
    "EXACT_B_ARRAY_MEMBER",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "FitOutcome",
    "FitPredict",
    "LOCKED_CLASSIFIER_SPEC",
    "PREDICTION_INDEX_MEMBER",
    "PREDICTION_RECEIPT_MEMBER",
    "PRODUCTION_PREDICTION_GEOMETRY",
    "PredictionGeometry",
    "PredictionRuntimeConfig",
    "PredictionRuntimeTestMode",
    "PredictionSurface",
    "assert_owned_root",
    "assert_production_runtime",
    "attempt_id",
    "config_hash",
    "final_paths",
    "geometry_for",
    "locked_classifier",
)
