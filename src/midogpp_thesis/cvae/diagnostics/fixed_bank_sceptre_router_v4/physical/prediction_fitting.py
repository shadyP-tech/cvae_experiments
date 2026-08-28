"""Spawn-bounded classifier task construction and fitting for SCEPTRE v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    sha256_array,
    sha256_file,
)
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)

from .prediction_composition import (
    candidate_exclusion_is_valid,
    compose_exact_b,
    compose_single_source,
    exact_b_source_centers,
)
from .prediction_contracts import (
    CANDIDATE_EXCLUSION_SENTINEL,
    CHECKPOINT_DIRECTORY,
    CPU_PREDICTION_WORKERS,
    FitOutcome,
    FitPredict,
    LOCKED_CLASSIFIER_SPEC,
    PRODUCTION_PREDICTION_GEOMETRY,
    PredictionGeometry,
)
from .prediction_io import canonical_sha256, persist_exact_json, persist_exact_npy
from .source_streams import SourceStreamStore


_CPU_THREAD_LIMITER: object | None = None


def build_tasks(
    *,
    config_hash: str,
    classifier: ClassifierSpec,
    source_store: SourceStreamStore,
    frame_payload: Mapping[str, object],
    root: Path,
    geometry: PredictionGeometry,
    attempt_id: str,
) -> tuple[Mapping[str, object], ...]:
    source_rows = [record.to_payload() for record in source_store.records]
    source_index_sha256 = canonical_sha256(source_rows)
    checkpoint_root = root / CHECKPOINT_DIRECTORY / "tasks"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks: list[Mapping[str, object]] = []
    for ordinal, (training_seed, generation_seed) in enumerate(geometry.seed_cells):
        task_id = f"train_{training_seed}_generation_{generation_seed}"
        identity = {
            "schema_version": "midogpp_sceptre_v4_physical_prediction_task_v1",
            "attempt_id": attempt_id,
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
                "task_sha256": canonical_sha256(identity),
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


def task_identity(task: Mapping[str, object]) -> dict[str, object]:
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


def execute_cpu_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        return ()
    with ProcessPoolExecutor(
        max_workers=CPU_PREDICTION_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_cpu_worker,
    ) as executor:
        futures = {
            executor.submit(_production_prediction_worker, task): task for task in tasks
        }
        return tuple(future.result() for future in as_completed(futures))


def _initialize_cpu_worker() -> None:
    """Apply the one-thread BLAS/native budget once in each spawned worker."""

    global _CPU_THREAD_LIMITER
    if _CPU_THREAD_LIMITER is not None:
        raise ProtocolError("SCEPTRE v4 CPU worker was initialized twice.")
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
        raise RuntimeError(
            "SCEPTRE v4 prediction workers require threadpoolctl."
        ) from exc
    _CPU_THREAD_LIMITER = threadpool_limits(limits=1)


def _production_prediction_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    if _CPU_THREAD_LIMITER is None:
        raise ProtocolError("SCEPTRE v4 CPU worker initializer was not run.")
    return prediction_worker(
        task,
        geometry=PRODUCTION_PREDICTION_GEOMETRY,
        fit_predict=fit_locked_logistic,
    )


def prediction_worker(
    task: Mapping[str, object],
    *,
    geometry: PredictionGeometry,
    fit_predict: FitPredict,
) -> Mapping[str, object]:
    assert_prediction_task(task, geometry=geometry)
    blocks, evaluation = load_task_arrays(task, geometry=geometry)
    classifier = classifier_from_payload(task["classifier"])
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
        raise RuntimeError("SCEPTRE v4 prediction workers require threadpoolctl.") from exc
    with threadpool_limits(limits=1):
        for source in geometry.centers:
            train_x, train_y, composition_hash = compose_single_source(
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
            outcome = fit_predict(train_x, train_y, allowed_evaluation, classifier)
            probabilities, predictions = validate_fit_outcome(
                outcome,
                expected_rows=len(allowed_evaluation),
                classifier=classifier,
            )
            masked_probabilities = np.full(
                geometry.evaluation_rows,
                CANDIDATE_EXCLUSION_SENTINEL,
                dtype=np.float32,
            )
            masked_probabilities[: forbidden.start] = probabilities[: forbidden.start]
            masked_probabilities[forbidden.stop :] = probabilities[forbidden.start :]
            candidate_values.append(masked_probabilities)
            fit_rows.append(
                fit_receipt(
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
            train_x, train_y, composition_hash = compose_exact_b(
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
            probabilities, predictions = validate_fit_outcome(
                outcome,
                expected_rows=stop - start,
                classifier=classifier,
            )
            exact_b_values[start:stop] = probabilities
            fit_rows.append(
                fit_receipt(
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
        or not candidate_exclusion_is_valid(
            candidate[np.newaxis, ...], geometry=geometry
        )
        or len(fit_rows) != 2 * len(geometry.centers)
    ):
        raise ProtocolError("SCEPTRE v4 prediction task output drifted.")
    candidate_path = Path(str(task["checkpoint_candidate_path"]))
    exact_b_path = Path(str(task["checkpoint_exact_b_path"]))
    persist_exact_npy(candidate_path, candidate, role="candidate checkpoint")
    persist_exact_npy(exact_b_path, exact_b, role="exact-B checkpoint")
    checkpoint_unhashed = {
        "schema_version": "midogpp_sceptre_v4_physical_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "attempt_id": task["attempt_id"],
        "task_sha256": task["task_sha256"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "candidate_array_file_sha256": sha256_file(candidate_path),
        "exact_b_array_file_sha256": sha256_file(exact_b_path),
        "candidate_array_sha256": sha256_array(candidate),
        "exact_b_array_sha256": sha256_array(exact_b),
        "fit_rows": fit_rows,
        "fit_index_sha256": canonical_sha256(fit_rows),
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
        "checkpoint_sha256": canonical_sha256(checkpoint_unhashed),
    }
    persist_exact_json(Path(str(task["checkpoint_json_path"])), checkpoint)
    return checkpoint


def assert_prediction_task(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> None:
    expected_b = {
        center: list(exact_b_source_centers(center, geometry=geometry))
        for center in geometry.centers
    }
    if (
        task.get("task_sha256") != canonical_sha256(task_identity(task))
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
        raise ProtocolError("SCEPTRE v4 prediction task boundary drifted.")


def load_task_arrays(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source_records = task.get("source_records")
    if (
        not isinstance(source_records, list)
        or task.get("source_index_sha256") != canonical_sha256(source_records)
    ):
        raise ProtocolError("SCEPTRE v4 prediction source index drifted.")
    by_key: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for raw in source_records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE v4 prediction source record is malformed.")
        key = (
            str(raw.get("source_center", "")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        if key in by_key:
            raise ProtocolError("SCEPTRE v4 prediction source record is duplicated.")
        by_key[key] = raw
    expected_keys = set(
        product(geometry.centers, geometry.training_seeds, geometry.generation_seeds)
    )
    if set(by_key) != expected_keys:
        raise ProtocolError("SCEPTRE v4 prediction source inventory drifted.")
    source_path = Path(str(task["source_array_path"]))
    if source_path.is_symlink() or not source_path.is_file():
        raise ProtocolError("SCEPTRE v4 prediction source array is unsafe.")
    source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        source_values.shape != geometry.source_geometry.array_shape
        or source_values.dtype != np.float32
        or source_values.flags.writeable
    ):
        raise ProtocolError("SCEPTRE v4 prediction source array geometry drifted.")
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    blocks: dict[str, np.ndarray] = {}
    for source in geometry.centers:
        record = by_key[(source, training_seed, generation_seed)]
        ordinal = int(record.get("block_ordinal", -1))
        block = source_values[ordinal]
        if (
            block.shape != (2 * geometry.source_rows_per_class, geometry.feature_dim)
            or not np.isfinite(block).all()
            or record.get("array_sha256") != sha256_array(block)
        ):
            raise ProtocolError("SCEPTRE v4 prediction source block bytes drifted.")
        blocks[source] = block
    evaluation_path = Path(str(task["evaluation_array_path"]))
    if evaluation_path.is_symlink() or not evaluation_path.is_file():
        raise ProtocolError("SCEPTRE v4 prediction evaluation scratch is unsafe.")
    evaluation = np.load(evaluation_path, mmap_mode="r", allow_pickle=False)
    if (
        evaluation.shape != (geometry.evaluation_rows, geometry.feature_dim)
        or evaluation.dtype != np.float32
        or not np.isfinite(evaluation).all()
        or task.get("evaluation_array_file_sha256") != sha256_file(evaluation_path)
        or task.get("evaluation_array_sha256") != sha256_array(evaluation)
    ):
        raise ProtocolError("SCEPTRE v4 prediction evaluation bytes drifted.")
    return blocks, evaluation


def fit_locked_logistic(
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
        raise ProtocolError("SCEPTRE v4 logistic probability matrix drifted.")
    return FitOutcome(
        positive_probabilities=np.ascontiguousarray(matrix[:, 1], dtype=np.float32),
        predictions=np.ascontiguousarray(predictions, dtype=np.uint8),
        converged=bool(fitted.converged),
        classifier_config_hash=str(fitted.classifier_config_hash),
        scaler_state_hash=str(fitted.scaler_state_hash),
    )


def validate_fit_outcome(
    outcome: FitOutcome,
    *,
    expected_rows: int,
    classifier: ClassifierSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(outcome, FitOutcome):
        raise ProtocolError("SCEPTRE v4 classifier result type drifted.")
    probabilities = np.ascontiguousarray(outcome.positive_probabilities, dtype=np.float32)
    predictions = np.ascontiguousarray(outcome.predictions, dtype=np.uint8)
    derived = np.ascontiguousarray(probabilities >= np.float32(0.5), dtype=np.uint8)
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
        raise ProtocolError("SCEPTRE v4 classifier fit failed convergence or value checks.")
    return probabilities, predictions


def fit_receipt(
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
    if family == "exact_B" and (target_center is None or target_center in sources):
        raise ProtocolError("SCEPTRE v4 exact-B fit receipt failed target exclusion.")
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
        raise ProtocolError("SCEPTRE v4 candidate evaluation exclusion drifted.")
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
    return {**unhashed, "fit_sha256": canonical_sha256(unhashed)}


def classifier_from_payload(raw: object) -> ClassifierSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("SCEPTRE v4 classifier payload is malformed.")
    try:
        classifier = ClassifierSpec(
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
            family=str(raw["family"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v4 classifier payload is malformed.") from exc
    if classifier != LOCKED_CLASSIFIER_SPEC:
        raise ProtocolError("SCEPTRE v4 classifier payload drifted.")
    return classifier


def task_key(task: Mapping[str, object]) -> tuple[int, int]:
    return int(task["training_seed"]), int(task["generation_seed"])


__all__ = (
    "assert_prediction_task",
    "build_tasks",
    "classifier_from_payload",
    "execute_cpu_tasks",
    "fit_locked_logistic",
    "fit_receipt",
    "load_task_arrays",
    "prediction_worker",
    "task_identity",
    "task_key",
    "validate_fit_outcome",
)
