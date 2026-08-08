"""CUDA-free classifier worker for one target/train/generation task."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_single_source_tail_action,
    build_uniform_topup_action,
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
    target_topup_geometry,
)
from .contracts import (
    BASE_ACTION_ID,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    UNIFORM_ACTION_ID,
    legal_sources,
)
from .config import CLASSIFIER_WORKERS
from .prediction_contracts import PREDICTION_TASK_SCHEMA, PredictionTaskSpec
from .prediction_io import (
    array_sha256,
    atomic_json,
    atomic_save_npy,
    sha256_file,
)
from .source_cache import load_source_cache
from .workstation import publish_validated_scratch_file


def execute_prediction_task(
    task: PredictionTaskSpec,
    *,
    fitter: Callable[..., Mapping[str, object]] | None = None,
) -> None:
    """Fit each unique composition once and fan out logical probabilities."""

    payload = dict(task.payload)
    if payload.get("labels_available_to_fit_or_predict") is not False:
        raise ProtocolError("Utility-aligned prediction task crossed the label boundary.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    target = str(payload["target_center"])
    training_seed = int(payload["training_seed"])
    generation_seed = int(payload["generation_seed"])
    source_cache = load_source_cache(Path(str(payload["source_cache_root"])))
    if (
        source_cache.cache_hash != payload["source_cache_hash"]
        or source_cache.generation_lock_hash != payload["generation_lock_hash"]
        or source_cache.bank_lock_hash != payload["bank_lock_hash"]
    ):
        raise ProtocolError("Utility-aligned prediction/source binding drifted.")
    evaluation_path = Path(str(payload["evaluation_array_path"]))
    if sha256_file(evaluation_path) != payload["target_frame_sha256"]:
        raise ProtocolError("Utility-aligned target frame hash drifted.")
    evaluation = np.load(evaluation_path, mmap_mode="r", allow_pickle=False)
    row_ids = tuple(str(row) for row in payload["evaluation_row_ids"])
    if (
        evaluation.dtype != np.float32
        or evaluation.shape != (len(row_ids), COMMON_OUTPUT_DIM)
        or not np.isfinite(evaluation).all()
    ):
        raise ProtocolError("Utility-aligned target frame geometry drifted.")
    sources = legal_sources(target)
    blocks = {
        source: source_cache.block(source, training_seed, generation_seed)
        for source in sources
    }
    geometry = target_topup_geometry(sources)
    shuffle_seeds = {
        label: derived_composition_seed(
            generation_lock_hash=str(payload["generation_lock_hash"]),
            target_center=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) != EXPECTED_ACTION_COUNT_PER_TARGET:
        raise ProtocolError("Utility-aligned prediction action menu drifted.")
    by_composition: dict[str, list[Mapping[str, object]]] = {}
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Utility-aligned prediction action is malformed.")
        by_composition.setdefault(str(raw["composition_hash"]), []).append(raw)

    active_fitter = fitter or _fit_classifier
    probability_by_composition: dict[str, np.ndarray] = {}
    fit_rows: list[dict[str, object]] = []
    for composition_hash, aliases in by_composition.items():
        representative = aliases[0]
        action_id = str(representative["action_id"])
        selected = representative.get("selected_source")
        if action_id == BASE_ACTION_ID or representative.get("abstained_to_base") is True:
            composition = compose_equal_union_base_blocks(
                blocks, geometry, shuffle_seed_by_class=shuffle_seeds
            )
        elif action_id == UNIFORM_ACTION_ID:
            composition = compose_residual_topup_blocks(
                blocks,
                build_uniform_topup_action(geometry),
                shuffle_seed_by_class=shuffle_seeds,
            )
        else:
            if selected not in sources:
                raise ProtocolError("Utility-aligned selected source drifted at execution.")
            composition = compose_residual_topup_blocks(
                blocks,
                build_single_source_tail_action(str(selected), geometry=geometry),
                shuffle_seed_by_class=shuffle_seeds,
            )
        fitted = active_fitter(
            composition.embeddings,
            composition.labels,
            np.asarray(evaluation),
            _classifier_from_payload(payload["classifier"]),
            int(payload["threads_per_worker"]),
        )
        if fitted.get("converged") is not True:
            raise ProtocolError("Utility-aligned classifier did not converge.")
        probability = np.ascontiguousarray(fitted["probabilities"], dtype=np.float32)
        if (
            probability.shape != (len(row_ids),)
            or not np.isfinite(probability).all()
            or bool(np.any(probability < 0.0))
            or bool(np.any(probability > 1.0))
        ):
            raise ProtocolError("Utility-aligned classifier output drifted.")
        probability_by_composition[composition_hash] = probability
        fit_rows.append(
            {
                "composition_hash": composition_hash,
                "representative_action_id": action_id,
                "logical_alias_action_ids": [str(raw["action_id"]) for raw in aliases],
                "classifier_config_hash": str(fitted["classifier_config_hash"]),
                "scaler_state_hash": str(fitted["scaler_state_hash"]),
                "converged": True,
            }
        )
    probabilities = np.stack(
        [probability_by_composition[str(raw["composition_hash"])] for raw in raw_actions],
        axis=0,
    ).astype(np.float32, copy=False)
    work_probability = Path(str(payload["work_probability_path"]))
    atomic_save_npy(work_probability, probabilities)
    logical_rows = [
        {
            "action_id": str(raw["action_id"]),
            "action_hash": str(raw["action_hash"]),
            "composition_hash": str(raw["composition_hash"]),
            "probability_row": index,
            "probability_sha256": array_sha256(probabilities[index]),
            "labels_available_to_fit_or_predict": False,
        }
        for index, raw in enumerate(raw_actions)
    ]
    metadata_unhashed = {
        "schema_version": PREDICTION_TASK_SCHEMA,
        "status": "COMPLETE",
        "task_id": payload["task_id"],
        "task_hash": payload["task_hash"],
        "plan_hash": payload["plan_hash"],
        "target_center": target,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "row_ids": list(row_ids),
        "probability_file_sha256": sha256_file(work_probability),
        "logical_prediction_count": len(logical_rows),
        "unique_composition_fit_count": len(fit_rows),
        "logical_rows": logical_rows,
        "fit_rows": fit_rows,
        "labels_available_to_fit_or_predict": False,
    }
    work_metadata = Path(str(payload["work_metadata_path"]))
    atomic_json(
        work_metadata,
        {**metadata_unhashed, "checkpoint_hash": stable_hash(metadata_unhashed)},
    )
    scratch_root = payload.get("scratch_root")
    if scratch_root is not None:
        publish_validated_scratch_file(
            work_probability,
            Path(str(payload["probability_path"])),
            expected_sha256=sha256_file(work_probability),
            scratch_root=str(scratch_root),
        )
        publish_validated_scratch_file(
            work_metadata,
            Path(str(payload["metadata_path"])),
            expected_sha256=sha256_file(work_metadata),
            scratch_root=str(scratch_root),
        )


def spawn_prediction_tasks(tasks: Sequence[PredictionTaskSpec]) -> None:
    context = mp.get_context("spawn")
    with context.Pool(processes=CLASSIFIER_WORKERS) as pool:
        pool.map(_prediction_worker_entry, [dict(task.payload) for task in tasks], chunksize=1)


def _prediction_worker_entry(payload: Mapping[str, object]) -> None:
    execute_prediction_task(PredictionTaskSpec(MappingProxyType(dict(payload))))


def _fit_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    evaluation_embeddings: np.ndarray,
    classifier: ClassifierSpec,
    threads: int,
) -> Mapping[str, object]:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:
        raise RuntimeError("Utility-aligned prediction requires threadpoolctl.") from exc
    with threadpool_limits(limits=threads):
        fitted = fit_logistic_classifier(
            train_embeddings, train_labels, evaluation_embeddings, spec=classifier
        )
    probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
    if fitted.classes != (0, 1) or probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ProtocolError("Utility-aligned classifier class geometry drifted.")
    return {
        "probabilities": probabilities[:, 1],
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "converged": fitted.converged,
    }


def _classifier_from_payload(raw: object) -> ClassifierSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Utility-aligned classifier payload is malformed.")
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Utility-aligned classifier payload drifted.") from exc


__all__ = ("execute_prediction_task", "spawn_prediction_tasks")
