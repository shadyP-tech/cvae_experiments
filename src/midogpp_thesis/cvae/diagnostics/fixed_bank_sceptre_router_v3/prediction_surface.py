"""Label-free SCEPTRE candidate and exact-B prediction materialization.

The GPU source workers have exited before this module runs.  One top-level
four-process spawn pool handles the nine (training seed, generation seed)
cells.  A cell fits all nine full single-source classifiers and all nine exact
target-excluded B classifiers.  Only probabilities, opaque row identities,
and immutable fit receipts are persisted; no manifest, outcome, or raw sample
path is accepted by this API.

Production geometry is fixed at ``(9 seed cells, 9 sources, 9_928 rows)`` for
the candidate tensor and ``(9 seed cells, 9_928 rows)`` for exact B.  A source
expert is never evaluated on rows from that same center: those structurally
forbidden cells contain the sealed sentinel ``-1.0``.  Small geometry is
available only through :class:`PredictionRuntimeTestMode`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
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
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_array,
    sha256_file,
)
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)

from .source_streams import (
    PRODUCTION_SOURCE_GEOMETRY,
    SourceGeometry,
    SourceStreamStore,
)


CANDIDATE_ARRAY_MEMBER = "arrays/sceptre_candidate_probabilities.npy"
EXACT_B_ARRAY_MEMBER = "arrays/sceptre_exact_b_probabilities.npy"
PREDICTION_INDEX_MEMBER = "manifests/sceptre_prediction_index.json"
PREDICTION_RECEIPT_MEMBER = "manifests/sceptre_prediction_receipt.json"
CHECKPOINT_DIRECTORY = "checkpoints/sceptre_prediction_surface"
EVALUATION_SCRATCH_MEMBER = f"{CHECKPOINT_DIRECTORY}/evaluation_embeddings.npy"
EVALUATION_FRAME_MEMBER = f"{CHECKPOINT_DIRECTORY}/evaluation_frame.json"
CANDIDATE_EXCLUSION_SENTINEL = np.float32(-1.0)

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
            raise ProtocolError("SCEPTRE prediction geometry is malformed.")

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
            raise ProtocolError("SCEPTRE prediction row center is unknown.")
        start = 0
        for observed, count in self.rows_by_center:
            stop = start + count
            if observed == target:
                return slice(start, stop)
            start = stop
        raise ProtocolError("SCEPTRE prediction row-center inventory drifted.")

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
            raise ProtocolError("SCEPTRE prediction test fit seam is not callable.")


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
        ):
            raise ProtocolError("SCEPTRE prediction row inventory drifted.")
        object.__setattr__(self, "index", MappingProxyType(dict(self.index)))
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @property
    def receipt_hash(self) -> str:
        return str(self.receipt["receipt_sha256"])

    @property
    def candidate_probabilities(self) -> np.ndarray:
        values = np.load(self.candidate_array_path, mmap_mode="r", allow_pickle=False)
        if values.flags.writeable:
            raise ProtocolError("SCEPTRE candidate memmap unexpectedly became writable.")
        return values

    @property
    def exact_b_probabilities(self) -> np.ndarray:
        values = np.load(self.exact_b_array_path, mmap_mode="r", allow_pickle=False)
        if values.flags.writeable:
            raise ProtocolError("SCEPTRE exact-B memmap unexpectedly became writable.")
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
            raise ProtocolError("SCEPTRE candidate prediction key is absent.") from exc
        return self.candidate_probabilities[seed_ordinal, source_ordinal]

    def exact_b(self, training_seed: int, generation_seed: int) -> np.ndarray:
        try:
            seed_ordinal = self.geometry.seed_cells.index(
                (int(training_seed), int(generation_seed))
            )
        except ValueError as exc:
            raise ProtocolError("SCEPTRE exact-B prediction key is absent.") from exc
        return self.exact_b_probabilities[seed_ordinal]


def materialize_prediction_surface(
    config: PredictionRuntimeConfig,
    source_store: SourceStreamStore,
    frame: object,
    *,
    root: Path,
    test_mode: PredictionRuntimeTestMode | None = None,
) -> PredictionSurface:
    """Fit all predeclared classifiers and seal both label-free tensors."""

    geometry = _geometry(test_mode)
    config_hash = _config_hash(config)
    classifier = _locked_classifier(config)
    if source_store.geometry != geometry.source_geometry:
        raise ProtocolError("SCEPTRE prediction/source geometry binding drifted.")
    if test_mode is None:
        _assert_production_runtime(config.runtime)
    destination = Path(root)
    _assert_owned_root(destination)
    final_paths = _final_paths(destination)
    present = tuple(path.is_file() for path in final_paths)
    if any(path.is_symlink() for path in final_paths):
        raise ProtocolError("SCEPTRE prediction final store contains a symlink.")
    if all(present):
        return load_prediction_surface(
            destination,
            expected_config_hash=config_hash,
            expected_source_receipt_hash=source_store.receipt_hash,
            test_mode=test_mode,
        )
    if any(present):
        raise ProtocolError("SCEPTRE prediction final store is an unsafe partial state.")

    frame_payload = _stage_evaluation_frame(destination, frame, geometry=geometry)
    tasks = _build_tasks(
        config_hash=config_hash,
        classifier=classifier,
        source_store=source_store,
        frame_payload=frame_payload,
        root=destination,
        geometry=geometry,
    )
    completed: dict[tuple[int, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint = _load_checkpoint_if_complete(task, geometry=geometry)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[_task_key(task)] = checkpoint
    if pending:
        if test_mode is None:
            results = _execute_cpu_tasks(pending)
        else:
            fit = test_mode.fit_predict or _fit_locked_logistic
            results = tuple(
                _prediction_worker(task, geometry=geometry, fit_predict=fit)
                for task in pending
            )
        for result in results:
            key = (int(result["training_seed"]), int(result["generation_seed"]))
            task = next(task for task in pending if _task_key(task) == key)
            loaded = _load_checkpoint_if_complete(task, geometry=geometry)
            if loaded is None or loaded.get("checkpoint_sha256") != result.get(
                "checkpoint_sha256"
            ):
                raise ProtocolError("SCEPTRE prediction checkpoint return drifted.")
            completed[key] = loaded
    if len(completed) != len(geometry.seed_cells):
        raise ProtocolError("SCEPTRE prediction checkpoint coverage is incomplete.")

    candidate_path = destination / CANDIDATE_ARRAY_MEMBER
    exact_b_path = destination / EXACT_B_ARRAY_MEMBER
    fit_rows = _publish_probability_arrays(
        candidate_path,
        exact_b_path,
        tasks=tasks,
        completed=completed,
        geometry=geometry,
    )
    row_ids = tuple(str(value) for value in frame_payload["row_ids"])
    row_centers = tuple(str(value) for value in frame_payload["row_centers"])
    row_identity_sha256 = _canonical_sha256(
        [
            {"row_ordinal": ordinal, "row_id": row_id, "center": center}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    fit_index_sha256 = _canonical_sha256(fit_rows)
    index_unhashed = {
        "schema_version": "midogpp_sceptre_v3_prediction_index_v1",
        "config_hash": config_hash,
        "source_receipt_sha256": source_store.receipt_hash,
        "source_array_sha256": source_store.receipt["source_array_sha256"],
        "cache_binding_hash": frame_payload["cache_binding_hash"],
        "geometry": geometry.to_payload(),
        "classifier": classifier.to_payload(),
        "classifier_config_hash": classifier.config_hash,
        "row_ids": list(row_ids),
        "row_centers": list(row_centers),
        "row_identity_sha256": row_identity_sha256,
        "fit_rows": fit_rows,
        "fit_index_sha256": fit_index_sha256,
        "fit_count": len(fit_rows),
        "candidate_source_order": list(geometry.centers),
        "seed_cell_order": [list(value) for value in geometry.seed_cells],
        "exact_b_target_exclusion_verified": True,
        "candidate_target_exclusion_mode": "MASKED_BEFORE_SCORING",
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "all_seed_cells_retained": True,
        "seed_selection_performed": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    index = {**index_unhashed, "index_sha256": _canonical_sha256(index_unhashed)}
    index_path = destination / PREDICTION_INDEX_MEMBER
    _persist_exact_json(index_path, index)
    receipt_unhashed = {
        "schema_version": "midogpp_sceptre_v3_prediction_receipt_v1",
        "status": "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS",
        "config_hash": config_hash,
        "source_receipt_sha256": source_store.receipt_hash,
        "cache_binding_hash": frame_payload["cache_binding_hash"],
        "geometry": geometry.to_payload(),
        "classifier_config_hash": classifier.config_hash,
        "candidate_array_file_sha256": sha256_file(candidate_path),
        "exact_b_array_file_sha256": sha256_file(exact_b_path),
        "prediction_index_file_sha256": sha256_file(index_path),
        "prediction_index_sha256": index["index_sha256"],
        "row_identity_sha256": row_identity_sha256,
        "fit_index_sha256": fit_index_sha256,
        "fit_count": len(fit_rows),
        "candidate_shape": list(geometry.candidate_shape),
        "exact_b_shape": list(geometry.exact_b_shape),
        "dtype": "float32",
        "npy_memmap_mode": "read_only",
        "cpu_worker_count": 4 if test_mode is None else 0,
        "blas_threads_per_worker": 1,
        "native_threads_per_worker": 1,
        "top_level_spawn_pool_only": test_mode is None,
        "target_expert_excluded_from_every_exact_b_fit": True,
        "target_expert_excluded_from_every_candidate_score": True,
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
        "classifier_refit_after_seal": False,
        "seed_selection_performed": False,
        "synthetic_test_mode": test_mode is not None,
    }
    receipt = {
        **receipt_unhashed,
        "receipt_sha256": _canonical_sha256(receipt_unhashed),
    }
    _persist_exact_json(destination / PREDICTION_RECEIPT_MEMBER, receipt)
    candidate_path.chmod(0o444)
    exact_b_path.chmod(0o444)
    surface = load_prediction_surface(
        destination,
        expected_config_hash=config_hash,
        expected_source_receipt_hash=source_store.receipt_hash,
        test_mode=test_mode,
    )
    checkpoint_root = destination / CHECKPOINT_DIRECTORY
    _validate_checkpoint_tree(checkpoint_root, geometry=geometry)
    shutil.rmtree(checkpoint_root)
    return surface


def load_prediction_surface(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_receipt_hash: str | None = None,
    test_mode: PredictionRuntimeTestMode | None = None,
) -> PredictionSurface:
    """Load and fully validate a completed read-only probability surface."""

    geometry = _geometry(test_mode)
    destination = Path(root)
    candidate_path, exact_b_path, index_path, receipt_path = _final_paths(destination)
    if any(path.is_symlink() or not path.is_file() for path in (candidate_path, exact_b_path, index_path, receipt_path)):
        raise ProtocolError("SCEPTRE prediction final store is absent or unsafe.")
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    index_unhashed = {key: value for key, value in index.items() if key != "index_sha256"}
    receipt_unhashed = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    row_ids = index.get("row_ids")
    row_centers = index.get("row_centers")
    fit_rows = index.get("fit_rows")
    if not isinstance(row_ids, list) or not isinstance(row_centers, list) or not isinstance(fit_rows, list):
        raise ProtocolError("SCEPTRE prediction index is malformed.")
    if (
        index.get("schema_version") != "midogpp_sceptre_v3_prediction_index_v1"
        or receipt.get("schema_version")
        != "midogpp_sceptre_v3_prediction_receipt_v1"
        or receipt.get("status")
        != "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS"
        or index.get("geometry") != geometry.to_payload()
        or receipt.get("geometry") != geometry.to_payload()
        or index.get("index_sha256") != _canonical_sha256(index_unhashed)
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_unhashed)
        or receipt.get("candidate_array_file_sha256") != sha256_file(candidate_path)
        or receipt.get("exact_b_array_file_sha256") != sha256_file(exact_b_path)
        or receipt.get("prediction_index_file_sha256") != sha256_file(index_path)
        or receipt.get("prediction_index_sha256") != index.get("index_sha256")
        or receipt.get("cache_binding_hash") != index.get("cache_binding_hash")
        or receipt.get("row_identity_sha256") != index.get("row_identity_sha256")
        or receipt.get("fit_index_sha256") != index.get("fit_index_sha256")
        or index.get("fit_index_sha256") != _canonical_sha256(fit_rows)
        or len(fit_rows) != geometry.fit_count
        or index.get("fit_count") != geometry.fit_count
        or receipt.get("fit_count") != geometry.fit_count
        or index.get("classifier") != LOCKED_CLASSIFIER_SPEC.to_payload()
        or index.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        or receipt.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        or index.get("candidate_source_order") != list(geometry.centers)
        or index.get("seed_cell_order")
        != [list(value) for value in geometry.seed_cells]
        or index.get("exact_b_target_exclusion_verified") is not True
        or receipt.get("target_expert_excluded_from_every_exact_b_fit") is not True
        or index.get("candidate_target_exclusion_mode") != "MASKED_BEFORE_SCORING"
        or index.get("candidate_exclusion_sentinel") != float(CANDIDATE_EXCLUSION_SENTINEL)
        or receipt.get("target_expert_excluded_from_every_candidate_score") is not True
        or receipt.get("candidate_exclusion_sentinel") != float(CANDIDATE_EXCLUSION_SENTINEL)
        or index.get("seed_selection_performed") is not False
        or receipt.get("seed_selection_performed") is not False
        or index.get("manifest_opened") is not False
        or receipt.get("manifest_opened") is not False
        or index.get("outcomes_available") is not False
        or receipt.get("outcomes_available") is not False
        or index.get("raw_sample_paths_available") is not False
        or receipt.get("raw_sample_paths_available") is not False
        or receipt.get("classifier_refit_after_seal") is not False
        or receipt.get("synthetic_test_mode") is not (test_mode is not None)
        or (
            expected_config_hash is not None
            and receipt.get("config_hash") != expected_config_hash
        )
        or (
            expected_source_receipt_hash is not None
            and receipt.get("source_receipt_sha256")
            != expected_source_receipt_hash
        )
    ):
        raise ProtocolError("SCEPTRE prediction receipt failed validation.")
    expected_row_hash = _canonical_sha256(
        [
            {"row_ordinal": ordinal, "row_id": str(row_id), "center": str(center)}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    if index.get("row_identity_sha256") != expected_row_hash:
        raise ProtocolError("SCEPTRE prediction row identity hash drifted.")
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    exact_b = np.load(exact_b_path, mmap_mode="r", allow_pickle=False)
    if (
        candidate.shape != geometry.candidate_shape
        or exact_b.shape != geometry.exact_b_shape
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or candidate.flags.writeable
        or exact_b.flags.writeable
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or not _candidate_exclusion_is_valid(candidate, geometry=geometry)
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
    ):
        raise ProtocolError("SCEPTRE prediction tensor geometry or values drifted.")
    return PredictionSurface(
        root=destination,
        candidate_array_path=candidate_path,
        exact_b_array_path=exact_b_path,
        index_path=index_path,
        receipt_path=receipt_path,
        geometry=geometry,
        row_ids=tuple(str(value) for value in row_ids),
        row_centers=tuple(str(value) for value in row_centers),
        index=index,
        receipt=receipt,
    )


def exact_b_source_centers(
    target_center: str, *, geometry: PredictionGeometry = PRODUCTION_PREDICTION_GEOMETRY
) -> tuple[str, ...]:
    """Return the exact ordered ``C - H`` source inventory for B."""

    target = str(target_center)
    if target not in geometry.centers:
        raise ProtocolError("SCEPTRE exact-B target center is unknown.")
    sources = tuple(center for center in geometry.centers if center != target)
    if target in sources or len(sources) != len(geometry.centers) - 1:
        raise ProtocolError("SCEPTRE exact-B source exclusion failed.")
    return sources


def _candidate_exclusion_is_valid(
    values: np.ndarray,
    *,
    geometry: PredictionGeometry,
) -> bool:
    """Authenticate the physical H-on-H exclusion mask.

    The source classifier never receives its own-center evaluation rows.  The
    corresponding durable cells are the unique out-of-domain sentinel, while
    every legally usable cell remains a finite probability.
    """

    candidate = np.asarray(values)
    if (
        candidate.ndim != 3
        or candidate.shape[0] <= 0
        or candidate.shape[1:] != (
            len(geometry.centers),
            geometry.evaluation_rows,
        )
        or not np.isfinite(candidate).all()
    ):
        return False
    for source_ordinal, source in enumerate(geometry.centers):
        forbidden = geometry.row_slice(source)
        if not np.all(
            candidate[:, source_ordinal, forbidden]
            == CANDIDATE_EXCLUSION_SENTINEL
        ):
            return False
        allowed = np.concatenate(
            (
                candidate[:, source_ordinal, : forbidden.start].reshape(-1),
                candidate[:, source_ordinal, forbidden.stop :].reshape(-1),
            )
        )
        if np.any((allowed < 0.0) | (allowed > 1.0)):
            return False
    return True


def _geometry(test_mode: PredictionRuntimeTestMode | None) -> PredictionGeometry:
    return PRODUCTION_PREDICTION_GEOMETRY if test_mode is None else test_mode.geometry


def _config_hash(config: object) -> str:
    value = getattr(config, "config_hash", getattr(config, "contract_hash", ""))
    text = str(value)
    if not text:
        raise ProtocolError("SCEPTRE prediction config hash is absent.")
    return text


def _locked_classifier(config: object) -> ClassifierSpec:
    classifier = getattr(config, "classifier", None)
    if classifier != LOCKED_CLASSIFIER_SPEC:
        raise ProtocolError("SCEPTRE downstream classifier lock drifted.")
    return classifier


def _assert_production_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("cpu_prediction_workers", -1)) != 4
        or runtime.get("cpu_worker_task_unit")
        != "one_complete_training_generation_seed_cell"
        or int(runtime.get("cpu_worker_task_count", -1)) != 9
        or int(runtime.get("blas_threads_per_worker", -1)) != 1
        or int(runtime.get("native_threads_per_worker", -1)) != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("top_level_spawn_pool_only") is not True
        or runtime.get("nested_pools_allowed") is not False
        or runtime.get("prediction_store_dtype") != "float32"
        or runtime.get("prediction_store_mode") != "read_only_memmap"
    ):
        raise ProtocolError("SCEPTRE prediction workstation topology drifted.")


def _assert_owned_root(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("SCEPTRE prediction root is a symlink.")
    if root.exists() and not root.is_dir():
        raise ProtocolError("SCEPTRE prediction root is not a directory.")
    root.mkdir(parents=True, exist_ok=True)


def _final_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / CANDIDATE_ARRAY_MEMBER,
        root / EXACT_B_ARRAY_MEMBER,
        root / PREDICTION_INDEX_MEMBER,
        root / PREDICTION_RECEIPT_MEMBER,
    )


def _stage_evaluation_frame(
    root: Path, frame: object, *, geometry: PredictionGeometry
) -> Mapping[str, object]:
    rows_by_center = getattr(frame, "rows_by_center", None)
    if not isinstance(rows_by_center, Mapping) or tuple(rows_by_center) != geometry.centers:
        raise ProtocolError("SCEPTRE label-free row inventory is unavailable.")
    expected_counts = dict(geometry.rows_by_center)
    rows: list[object] = []
    row_ids: list[str] = []
    row_centers: list[str] = []
    offsets: dict[str, dict[str, int]] = {}
    cursor = 0
    for center in geometry.centers:
        center_rows = tuple(rows_by_center[center])
        if len(center_rows) != expected_counts[center]:
            raise ProtocolError("SCEPTRE label-free rows-by-center drifted.")
        start = cursor
        for row in center_rows:
            _assert_label_free_row(row, expected_center=center)
            rows.append(row)
            row_ids.append(_opaque_row_id(row))
            row_centers.append(center)
            cursor += 1
        offsets[center] = {"start": start, "stop": cursor}
    if cursor != geometry.evaluation_rows or len(set(row_ids)) != len(row_ids):
        raise ProtocolError("SCEPTRE label-free row coverage drifted.")
    embeddings_for = getattr(frame, "embeddings_for", None)
    if callable(embeddings_for):
        embeddings = np.asarray(embeddings_for(rows))
    else:
        embeddings = np.asarray(getattr(frame, "embeddings", None))
    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    if (
        values.shape != (geometry.evaluation_rows, geometry.feature_dim)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE label-free evaluation embeddings drifted.")
    cache_binding_hash = str(getattr(frame, "cache_binding_hash", ""))
    if not cache_binding_hash:
        raise ProtocolError("SCEPTRE label-free cache binding hash is absent.")
    binding = getattr(frame, "cache_binding", None)
    if isinstance(binding, Mapping):
        if any(
            binding.get(key) is True
            for key in (
                "labels_persisted",
                "manifest_opened",
                "sample_paths_persisted",
                "raw_sample_paths_available",
            )
        ):
            raise ProtocolError("SCEPTRE frame escaped the label-free path-free boundary.")
    array_path = root / EVALUATION_SCRATCH_MEMBER
    _persist_exact_npy(array_path, values, role="evaluation scratch")
    frame_unhashed = {
        "schema_version": "midogpp_sceptre_v3_label_free_evaluation_frame_v1",
        "cache_binding_hash": cache_binding_hash,
        "evaluation_array_file_sha256": sha256_file(array_path),
        "evaluation_array_sha256": sha256_array(values),
        "shape": list(values.shape),
        "dtype": "float32",
        "row_ids": row_ids,
        "row_centers": row_centers,
        "offsets": offsets,
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    payload = {
        **frame_unhashed,
        "frame_sha256": _canonical_sha256(frame_unhashed),
        "evaluation_array_path": str(array_path.resolve()),
    }
    _persist_exact_json(root / EVALUATION_FRAME_MEMBER, payload)
    return payload


def _assert_label_free_row(row: object, *, expected_center: str) -> None:
    if str(getattr(row, "center", "")) != expected_center:
        raise ProtocolError("SCEPTRE label-free row center drifted.")
    forbidden = (
        "label",
        "target_label",
        "truth",
        "image_path",
        "sample_path",
        "raw_path",
        "file_path",
    )
    if any(hasattr(row, name) for name in forbidden):
        raise ProtocolError("SCEPTRE prediction row exposes an outcome or raw path.")


def _opaque_row_id(row: object) -> str:
    value = getattr(row, "evaluation_row_id", getattr(row, "sample_id", None))
    text = str(value) if value is not None else ""
    if not text:
        raise ProtocolError("SCEPTRE prediction row lacks an opaque identity.")
    return text


def _build_tasks(
    *,
    config_hash: str,
    classifier: ClassifierSpec,
    source_store: SourceStreamStore,
    frame_payload: Mapping[str, object],
    root: Path,
    geometry: PredictionGeometry,
) -> tuple[Mapping[str, object], ...]:
    source_rows = [record.to_payload() for record in source_store.records]
    source_index_sha256 = _canonical_sha256(source_rows)
    checkpoint_root = root / CHECKPOINT_DIRECTORY / "tasks"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks: list[Mapping[str, object]] = []
    for ordinal, (training_seed, generation_seed) in enumerate(geometry.seed_cells):
        task_id = f"train_{training_seed}_generation_{generation_seed}"
        identity = {
            "schema_version": "midogpp_sceptre_v3_prediction_task_v1",
            "task_ordinal": ordinal,
            "task_id": task_id,
            "config_hash": config_hash,
            "source_receipt_sha256": source_store.receipt_hash,
            "source_array_file_sha256": source_store.receipt["source_array_sha256"],
            "source_index_sha256": source_index_sha256,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "centers": list(geometry.centers),
            "exact_b_sources_by_center": {
                center: list(exact_b_source_centers(center, geometry=geometry))
                for center in geometry.centers
            },
            "source_array_path": str(source_store.array_path.resolve()),
            "source_records": source_rows,
            "evaluation_array_path": str(frame_payload["evaluation_array_path"]),
            "evaluation_array_file_sha256": frame_payload[
                "evaluation_array_file_sha256"
            ],
            "evaluation_array_sha256": frame_payload["evaluation_array_sha256"],
            "evaluation_shape": frame_payload["shape"],
            "evaluation_frame_sha256": frame_payload["frame_sha256"],
            "row_offsets": frame_payload["offsets"],
            "classifier": classifier.to_payload(),
            "classifier_config_hash": classifier.config_hash,
            "geometry": geometry.to_payload(),
            "blas_threads": 1,
            "native_threads": 1,
            "manifest_available": False,
            "outcomes_available": False,
            "raw_sample_paths_available": False,
            "seed_selection_permitted": False,
        }
        tasks.append(
            {
                **identity,
                "task_sha256": _canonical_sha256(identity),
                "checkpoint_candidate_path": str(
                    checkpoint_root / f"{task_id}.candidate.npy"
                ),
                "checkpoint_exact_b_path": str(
                    checkpoint_root / f"{task_id}.exact_b.npy"
                ),
                "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
            }
        )
    return tuple(tasks)


def _task_identity(task: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in task.items()
        if key
        not in {
            "task_sha256",
            "checkpoint_candidate_path",
            "checkpoint_exact_b_path",
            "checkpoint_json_path",
        }
    }


def _execute_cpu_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_production_prediction_worker, task): task for task in tasks
        }
        return tuple(future.result() for future in as_completed(futures))


def _production_prediction_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    return _prediction_worker(
        task,
        geometry=PRODUCTION_PREDICTION_GEOMETRY,
        fit_predict=_fit_locked_logistic,
    )


def _prediction_worker(
    task: Mapping[str, object],
    *,
    geometry: PredictionGeometry,
    fit_predict: FitPredict,
) -> Mapping[str, object]:
    _assert_prediction_task(task, geometry=geometry)
    blocks, evaluation = _load_task_arrays(task, geometry=geometry)
    classifier = _classifier_from_payload(task["classifier"])
    candidate_values: list[np.ndarray] = []
    exact_b_values = np.empty(geometry.evaluation_rows, dtype=np.float32)
    fit_rows: list[dict[str, object]] = []
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("SCEPTRE prediction workers require threadpoolctl.") from exc
    with threadpool_limits(limits=1):
        for source in geometry.centers:
            train_x, train_y, composition_hash = _compose_single_source(
                blocks[source], source=source, geometry=geometry
            )
            forbidden = geometry.row_slice(source)
            allowed_evaluation = np.ascontiguousarray(
                np.concatenate(
                    (evaluation[: forbidden.start], evaluation[forbidden.stop :]),
                    axis=0,
                ),
                dtype=np.float32,
            )
            outcome = fit_predict(
                train_x, train_y, allowed_evaluation, classifier
            )
            probabilities, predictions = _validate_fit_outcome(
                outcome,
                expected_rows=len(allowed_evaluation),
                classifier=classifier,
            )
            masked_probabilities = np.full(
                geometry.evaluation_rows,
                CANDIDATE_EXCLUSION_SENTINEL,
                dtype=np.float32,
            )
            masked_probabilities[: forbidden.start] = probabilities[
                : forbidden.start
            ]
            masked_probabilities[forbidden.stop :] = probabilities[
                forbidden.start :
            ]
            candidate_values.append(masked_probabilities)
            fit_rows.append(
                _fit_receipt(
                    family="single_source",
                    source_center=source,
                    target_center=None,
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    source_centers=(source,),
                    composition_hash=composition_hash,
                    outcome=outcome,
                    probabilities=probabilities,
                    predictions=predictions,
                    excluded_evaluation_center=source,
                    masked_row_count=forbidden.stop - forbidden.start,
                )
            )
        offsets = task["row_offsets"]
        for target in geometry.centers:
            source_order = tuple(task["exact_b_sources_by_center"][target])
            train_x, train_y, composition_hash = _compose_exact_b(
                blocks,
                target_center=target,
                source_order=source_order,
                geometry=geometry,
            )
            offset = offsets[target]
            start, stop = int(offset["start"]), int(offset["stop"])
            outcome = fit_predict(
                train_x,
                train_y,
                np.ascontiguousarray(evaluation[start:stop], dtype=np.float32),
                classifier,
            )
            probabilities, predictions = _validate_fit_outcome(
                outcome,
                expected_rows=stop - start,
                classifier=classifier,
            )
            exact_b_values[start:stop] = probabilities
            fit_rows.append(
                _fit_receipt(
                    family="exact_B",
                    source_center=None,
                    target_center=target,
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    source_centers=source_order,
                    composition_hash=composition_hash,
                    outcome=outcome,
                    probabilities=probabilities,
                    predictions=predictions,
                    excluded_evaluation_center=None,
                    masked_row_count=0,
                )
            )
    candidate = np.ascontiguousarray(np.stack(candidate_values), dtype=np.float32)
    exact_b = np.ascontiguousarray(exact_b_values, dtype=np.float32)
    if (
        candidate.shape != (len(geometry.centers), geometry.evaluation_rows)
        or exact_b.shape != (geometry.evaluation_rows,)
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or not _candidate_exclusion_is_valid(
            candidate[np.newaxis, ...], geometry=geometry
        )
        or len(fit_rows) != 2 * len(geometry.centers)
    ):
        raise ProtocolError("SCEPTRE prediction task output drifted.")
    candidate_path = Path(str(task["checkpoint_candidate_path"]))
    exact_b_path = Path(str(task["checkpoint_exact_b_path"]))
    _persist_exact_npy(candidate_path, candidate, role="candidate checkpoint")
    _persist_exact_npy(exact_b_path, exact_b, role="exact-B checkpoint")
    checkpoint_unhashed = {
        "schema_version": "midogpp_sceptre_v3_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "task_sha256": task["task_sha256"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "candidate_array_file_sha256": sha256_file(candidate_path),
        "exact_b_array_file_sha256": sha256_file(exact_b_path),
        "candidate_array_sha256": sha256_array(candidate),
        "exact_b_array_sha256": sha256_array(exact_b),
        "fit_rows": fit_rows,
        "fit_index_sha256": _canonical_sha256(fit_rows),
        "fit_count": len(fit_rows),
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
        "target_expert_excluded_from_every_exact_b_fit": True,
        "target_expert_excluded_from_every_candidate_score": True,
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "seed_selection_performed": False,
    }
    checkpoint = {
        **checkpoint_unhashed,
        "checkpoint_sha256": _canonical_sha256(checkpoint_unhashed),
    }
    _persist_exact_json(Path(str(task["checkpoint_json_path"])), checkpoint)
    return checkpoint


def _assert_prediction_task(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> None:
    expected_b = {
        center: list(exact_b_source_centers(center, geometry=geometry))
        for center in geometry.centers
    }
    if (
        task.get("task_sha256") != _canonical_sha256(_task_identity(task))
        or task.get("geometry") != geometry.to_payload()
        or tuple(task.get("centers", ())) != geometry.centers
        or task.get("exact_b_sources_by_center") != expected_b
        or int(task.get("training_seed", -1)) not in geometry.training_seeds
        or int(task.get("generation_seed", -1)) not in geometry.generation_seeds
        or int(task.get("blas_threads", -1)) != 1
        or int(task.get("native_threads", -1)) != 1
        or task.get("classifier") != LOCKED_CLASSIFIER_SPEC.to_payload()
        or task.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        or task.get("manifest_available") is not False
        or task.get("outcomes_available") is not False
        or task.get("raw_sample_paths_available") is not False
        or task.get("seed_selection_permitted") is not False
    ):
        raise ProtocolError("SCEPTRE prediction task boundary drifted.")


def _load_task_arrays(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source_records = task.get("source_records")
    if (
        not isinstance(source_records, list)
        or task.get("source_index_sha256") != _canonical_sha256(source_records)
    ):
        raise ProtocolError("SCEPTRE prediction source index drifted.")
    by_key: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for raw in source_records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE prediction source record is malformed.")
        key = (
            str(raw.get("source_center", "")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        if key in by_key:
            raise ProtocolError("SCEPTRE prediction source record is duplicated.")
        by_key[key] = raw
    expected_keys = set(
        product(geometry.centers, geometry.training_seeds, geometry.generation_seeds)
    )
    if set(by_key) != expected_keys:
        raise ProtocolError("SCEPTRE prediction source inventory drifted.")
    source_path = Path(str(task["source_array_path"]))
    if source_path.is_symlink() or not source_path.is_file():
        raise ProtocolError("SCEPTRE prediction source array is unsafe.")
    source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        source_values.shape != geometry.source_geometry.array_shape
        or source_values.dtype != np.float32
        or source_values.flags.writeable
    ):
        raise ProtocolError("SCEPTRE prediction source array geometry drifted.")
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    blocks: dict[str, np.ndarray] = {}
    for source in geometry.centers:
        record = by_key[(source, training_seed, generation_seed)]
        ordinal = int(record.get("block_ordinal", -1))
        block = source_values[ordinal]
        if (
            block.shape
            != (2 * geometry.source_rows_per_class, geometry.feature_dim)
            or not np.isfinite(block).all()
            or record.get("array_sha256") != sha256_array(block)
        ):
            raise ProtocolError("SCEPTRE prediction source block bytes drifted.")
        blocks[source] = block
    evaluation_path = Path(str(task["evaluation_array_path"]))
    if evaluation_path.is_symlink() or not evaluation_path.is_file():
        raise ProtocolError("SCEPTRE prediction evaluation scratch is unsafe.")
    evaluation = np.load(evaluation_path, mmap_mode="r", allow_pickle=False)
    if (
        evaluation.shape != (geometry.evaluation_rows, geometry.feature_dim)
        or evaluation.dtype != np.float32
        or not np.isfinite(evaluation).all()
        or task.get("evaluation_array_file_sha256") != sha256_file(evaluation_path)
        or task.get("evaluation_array_sha256") != sha256_array(evaluation)
    ):
        raise ProtocolError("SCEPTRE prediction evaluation bytes drifted.")
    return blocks, evaluation


def _compose_single_source(
    block: np.ndarray,
    *,
    source: str,
    geometry: PredictionGeometry,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.ascontiguousarray(block, dtype=np.float32)
    if (
        source not in geometry.centers
        or values.shape
        != (2 * geometry.source_rows_per_class, geometry.feature_dim)
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE single-source training block drifted.")
    truth = _synthetic_truth(geometry.source_rows_per_class)
    composition_hash = _canonical_sha256(
        {
            "family": "single_source",
            "source_center": source,
            "rows_per_class": geometry.source_rows_per_class,
            "embedding_sha256": sha256_array(values),
        }
    )
    return values, truth, composition_hash


def _compose_exact_b(
    blocks: Mapping[str, np.ndarray],
    *,
    target_center: str,
    source_order: Sequence[str],
    geometry: PredictionGeometry,
) -> tuple[np.ndarray, np.ndarray, str]:
    target = str(target_center)
    sources = tuple(str(value) for value in source_order)
    expected = exact_b_source_centers(target, geometry=geometry)
    if sources != expected or target in sources:
        raise ProtocolError("SCEPTRE exact-B composition included H or changed C-H order.")
    chunks: list[np.ndarray] = []
    for class_index in (0, 1):
        class_start = class_index * geometry.source_rows_per_class
        for source in sources:
            block = np.asarray(blocks[source])
            if block.shape != (
                2 * geometry.source_rows_per_class,
                geometry.feature_dim,
            ):
                raise ProtocolError("SCEPTRE exact-B source block geometry drifted.")
            chunks.append(
                np.asarray(
                    block[
                        class_start : class_start
                        + geometry.exact_b_prefix_per_source_class
                    ],
                    dtype=np.float32,
                )
            )
    values = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    expected_rows = 2 * geometry.source_rows_per_class
    if values.shape != (expected_rows, geometry.feature_dim) or not np.isfinite(values).all():
        raise ProtocolError("SCEPTRE exact-B composition geometry drifted.")
    truth = _synthetic_truth(geometry.source_rows_per_class)
    composition_hash = _canonical_sha256(
        {
            "family": "exact_B",
            "target_center": target,
            "source_centers": list(sources),
            "prefix_per_source_class": geometry.exact_b_prefix_per_source_class,
            "embedding_sha256": sha256_array(values),
            "target_expert_excluded": True,
        }
    )
    return values, truth, composition_hash


def _synthetic_truth(rows_per_class: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            (
                np.zeros(rows_per_class, dtype=np.uint8),
                np.ones(rows_per_class, dtype=np.uint8),
            )
        ),
        dtype=np.uint8,
    )


def _fit_locked_logistic(
    train_embeddings: np.ndarray,
    train_truth: np.ndarray,
    evaluation_embeddings: np.ndarray,
    classifier: ClassifierSpec,
) -> FitOutcome:
    fitted = fit_logistic_classifier(
        train_embeddings,
        train_truth,
        evaluation_embeddings,
        spec=classifier,
    )
    matrix = np.asarray(fitted.probabilities, dtype=np.float64)
    predictions = np.asarray(fitted.predictions, dtype=np.uint8)
    if fitted.classes != (0, 1) or matrix.shape != (len(evaluation_embeddings), 2):
        raise ProtocolError("SCEPTRE logistic probability matrix drifted.")
    return FitOutcome(
        positive_probabilities=np.ascontiguousarray(matrix[:, 1], dtype=np.float32),
        predictions=np.ascontiguousarray(predictions, dtype=np.uint8),
        converged=bool(fitted.converged),
        classifier_config_hash=str(fitted.classifier_config_hash),
        scaler_state_hash=str(fitted.scaler_state_hash),
    )


def _validate_fit_outcome(
    outcome: FitOutcome,
    *,
    expected_rows: int,
    classifier: ClassifierSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(outcome, FitOutcome):
        raise ProtocolError("SCEPTRE classifier result type drifted.")
    probabilities = np.ascontiguousarray(outcome.positive_probabilities, dtype=np.float32)
    predictions = np.ascontiguousarray(outcome.predictions, dtype=np.uint8)
    derived = np.ascontiguousarray(
        probabilities >= np.float32(0.5), dtype=np.uint8
    )
    if (
        probabilities.shape != (expected_rows,)
        or predictions.shape != (expected_rows,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or not np.array_equal(predictions, derived)
        or outcome.converged is not True
        or outcome.classifier_config_hash != classifier.config_hash
        or not str(outcome.scaler_state_hash)
    ):
        raise ProtocolError("SCEPTRE classifier fit failed convergence or value checks.")
    return probabilities, predictions


def _fit_receipt(
    *,
    family: str,
    source_center: str | None,
    target_center: str | None,
    training_seed: int,
    generation_seed: int,
    source_centers: Sequence[str],
    composition_hash: str,
    outcome: FitOutcome,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    excluded_evaluation_center: str | None,
    masked_row_count: int,
) -> dict[str, object]:
    sources = tuple(str(value) for value in source_centers)
    if family == "exact_B" and (
        target_center is None or target_center in sources
    ):
        raise ProtocolError("SCEPTRE exact-B fit receipt failed target exclusion.")
    if (
        family == "single_source"
        and (
            source_center is None
            or excluded_evaluation_center != source_center
            or masked_row_count <= 0
        )
    ) or (
        family == "exact_B"
        and (excluded_evaluation_center is not None or masked_row_count != 0)
    ):
        raise ProtocolError("SCEPTRE candidate evaluation exclusion drifted.")
    unhashed = {
        "family": family,
        "source_center": source_center,
        "target_center": target_center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "source_centers": list(sources),
        "composition_hash": composition_hash,
        "classifier_config_hash": outcome.classifier_config_hash,
        "scaler_state_hash": outcome.scaler_state_hash,
        "probability_sha256": sha256_array(probabilities),
        "prediction_sha256": sha256_array(predictions),
        "evaluated_row_count": len(probabilities),
        "excluded_evaluation_center": excluded_evaluation_center,
        "masked_row_count": masked_row_count,
        "converged": True,
        "target_expert_excluded": family != "exact_B" or target_center not in sources,
    }
    return {**unhashed, "fit_sha256": _canonical_sha256(unhashed)}


def _classifier_from_payload(raw: object) -> ClassifierSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("SCEPTRE classifier payload is malformed.")
    try:
        classifier = ClassifierSpec(
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None
            if raw["class_weight"] is None
            else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
            family=str(raw["family"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE classifier payload is malformed.") from exc
    if classifier != LOCKED_CLASSIFIER_SPEC:
        raise ProtocolError("SCEPTRE classifier payload drifted.")
    return classifier


def _load_checkpoint_if_complete(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> Mapping[str, object] | None:
    candidate_path = Path(str(task["checkpoint_candidate_path"]))
    exact_b_path = Path(str(task["checkpoint_exact_b_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    paths = (candidate_path, exact_b_path, json_path)
    if any(path.is_symlink() for path in paths):
        raise ProtocolError("SCEPTRE prediction checkpoint contains a symlink.")
    present = tuple(path.is_file() for path in paths)
    if present == (False, False, False):
        return None
    if present != (True, True, True):
        raise ProtocolError("SCEPTRE prediction checkpoint is partial; refusing refit.")
    _assert_prediction_task(task, geometry=geometry)
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    exact_b = np.load(exact_b_path, mmap_mode="r", allow_pickle=False)
    fit_rows = payload.get("fit_rows")
    if (
        payload.get("checkpoint_sha256") != _canonical_sha256(unhashed)
        or payload.get("schema_version")
        != "midogpp_sceptre_v3_prediction_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("task_sha256") != task["task_sha256"]
        or int(payload.get("training_seed", -1)) != task["training_seed"]
        or int(payload.get("generation_seed", -1)) != task["generation_seed"]
        or payload.get("candidate_array_file_sha256") != sha256_file(candidate_path)
        or payload.get("exact_b_array_file_sha256") != sha256_file(exact_b_path)
        or payload.get("candidate_array_sha256") != sha256_array(candidate)
        or payload.get("exact_b_array_sha256") != sha256_array(exact_b)
        or candidate.shape != (len(geometry.centers), geometry.evaluation_rows)
        or exact_b.shape != (geometry.evaluation_rows,)
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or not _candidate_exclusion_is_valid(
            candidate[np.newaxis, ...], geometry=geometry
        )
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
        or not isinstance(fit_rows, list)
        or len(fit_rows) != 2 * len(geometry.centers)
        or payload.get("fit_index_sha256") != _canonical_sha256(fit_rows)
        or payload.get("fit_count") != 2 * len(geometry.centers)
        or payload.get("manifest_opened") is not False
        or payload.get("outcomes_available") is not False
        or payload.get("raw_sample_paths_available") is not False
        or payload.get("target_expert_excluded_from_every_exact_b_fit") is not True
        or payload.get("target_expert_excluded_from_every_candidate_score") is not True
        or payload.get("candidate_exclusion_sentinel") != float(CANDIDATE_EXCLUSION_SENTINEL)
        or payload.get("seed_selection_performed") is not False
    ):
        raise ProtocolError("SCEPTRE prediction checkpoint failed validation.")
    for raw in fit_rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE prediction fit receipt is malformed.")
        fit_unhashed = {key: value for key, value in raw.items() if key != "fit_sha256"}
        if (
            raw.get("fit_sha256") != _canonical_sha256(fit_unhashed)
            or raw.get("converged") is not True
            or raw.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
            or (
                raw.get("family") == "single_source"
                and (
                    raw.get("excluded_evaluation_center")
                    != raw.get("source_center")
                    or int(raw.get("masked_row_count", 0)) <= 0
                )
            )
            or (
                raw.get("family") == "exact_B"
                and raw.get("target_center") in tuple(raw.get("source_centers", ()))
            )
        ):
            raise ProtocolError("SCEPTRE prediction fit receipt drifted.")
    return payload


def _publish_probability_arrays(
    candidate_path: Path,
    exact_b_path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[int, int], Mapping[str, object]],
    geometry: PredictionGeometry,
) -> list[dict[str, object]]:
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temp = candidate_path.with_suffix(
        candidate_path.suffix + f".{os.getpid()}.tmp"
    )
    exact_b_temp = exact_b_path.with_suffix(
        exact_b_path.suffix + f".{os.getpid()}.tmp"
    )
    fit_rows: list[dict[str, object]] = []
    try:
        candidate = np.lib.format.open_memmap(
            candidate_temp,
            mode="w+",
            dtype=np.float32,
            shape=geometry.candidate_shape,
        )
        exact_b = np.lib.format.open_memmap(
            exact_b_temp,
            mode="w+",
            dtype=np.float32,
            shape=geometry.exact_b_shape,
        )
        for seed_ordinal, task in enumerate(tasks):
            checkpoint = completed[_task_key(task)]
            candidate[seed_ordinal] = np.load(
                Path(str(task["checkpoint_candidate_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            exact_b[seed_ordinal] = np.load(
                Path(str(task["checkpoint_exact_b_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            for fit_ordinal, raw in enumerate(checkpoint["fit_rows"]):
                fit_rows.append(
                    {
                        "global_fit_ordinal": len(fit_rows),
                        "seed_cell_ordinal": seed_ordinal,
                        "within_cell_fit_ordinal": fit_ordinal,
                        **dict(raw),
                    }
                )
        candidate.flush()
        exact_b.flush()
        del candidate, exact_b
        if candidate_path.exists() or exact_b_path.exists():
            raise ProtocolError("SCEPTRE prediction final array appeared during publication.")
        os.replace(candidate_temp, candidate_path)
        os.replace(exact_b_temp, exact_b_path)
    except BaseException:
        candidate_temp.unlink(missing_ok=True)
        exact_b_temp.unlink(missing_ok=True)
        raise
    if len(fit_rows) != geometry.fit_count:
        raise ProtocolError("SCEPTRE prediction fit coverage drifted.")
    return fit_rows


def _persist_exact_npy(path: Path, values: np.ndarray, *, role: str) -> None:
    if path.is_symlink():
        raise ProtocolError(f"SCEPTRE {role} is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError(f"SCEPTRE {role} is unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.shape != values.shape
            or observed.dtype != values.dtype
            or sha256_array(observed) != sha256_array(values)
        ):
            raise ProtocolError(f"SCEPTRE {role} differs; refusing regeneration.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _persist_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("SCEPTRE prediction JSON member is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("SCEPTRE prediction JSON differs; refusing overwrite.")
        return
    atomic_json(path, payload)


def _validate_checkpoint_tree(
    directory: Path, *, geometry: PredictionGeometry
) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("SCEPTRE prediction checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("SCEPTRE prediction checkpoint root is unsafe.")
    allowed_top = {"evaluation_embeddings.npy", "evaluation_frame.json", "tasks"}
    for path in directory.iterdir():
        if path.is_symlink() or path.name not in allowed_top:
            raise ProtocolError("SCEPTRE prediction checkpoint tree has an unknown member.")
    tasks_root = directory / "tasks"
    if not tasks_root.exists():
        return
    if tasks_root.is_symlink() or not tasks_root.is_dir():
        raise ProtocolError("SCEPTRE prediction task checkpoint root is unsafe.")
    expected = {
        f"train_{train}_generation_{generation}.{suffix}"
        for train, generation in geometry.seed_cells
        for suffix in ("candidate.npy", "exact_b.npy", "json")
    }
    for path in tasks_root.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ProtocolError("SCEPTRE prediction task tree has an unknown member.")


def _task_key(task: Mapping[str, object]) -> tuple[int, int]:
    return int(task["training_seed"]), int(task["generation_seed"])


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "CANDIDATE_ARRAY_MEMBER",
    "CANDIDATE_EXCLUSION_SENTINEL",
    "CHECKPOINT_DIRECTORY",
    "EXACT_B_ARRAY_MEMBER",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "FitOutcome",
    "LOCKED_CLASSIFIER_SPEC",
    "PREDICTION_INDEX_MEMBER",
    "PREDICTION_RECEIPT_MEMBER",
    "PRODUCTION_PREDICTION_GEOMETRY",
    "PredictionGeometry",
    "PredictionRuntimeConfig",
    "PredictionRuntimeTestMode",
    "PredictionSurface",
    "exact_b_source_centers",
    "load_prediction_surface",
    "materialize_prediction_surface",
)
