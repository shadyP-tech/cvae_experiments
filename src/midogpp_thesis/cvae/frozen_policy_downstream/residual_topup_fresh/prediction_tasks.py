"""Prediction-task planning and spawned label-blind classifier execution."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

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
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
    target_topup_geometry,
)
from .config import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    ResidualTopupFreshConfig,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EvaluationPlan,
    FrozenActionPayload,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_action_ids,
    legal_sources,
)
from .policy_loading import FrozenPolicySurface, rebuild_and_validate_core_action
from .prediction_contracts import (
    ClassifierFitter,
    PREDICTION_CELL_SCHEMA,
    PREDICTION_TASK_SCHEMA,
    PredictionTaskSpec,
)
from .prediction_io import array_sha256, sha256_file, write_task_checkpoint
from .source_cache import FreshSourceCache, load_source_cache
from .target_cache import FreshTargetSurface


def execute_prediction_task(
    task: PredictionTaskSpec,
    *,
    fitter: ClassifierFitter | None = None,
) -> None:
    """Execute one label-blind target/seed task; injectable for tiny tests."""

    payload = dict(task.payload)
    source_cache = load_source_cache(Path(str(payload["source_cache_root"])))
    target = str(payload["target_center"])
    training_seed = int(payload["training_seed"])
    generation_seed = int(payload["generation_seed"])
    sources = legal_sources(target)
    if (
        source_cache.cache_hash != payload.get("source_cache_hash")
        or source_cache.generation_lock_hash
        != payload.get("generation_lock_hash")
        or source_cache.bank_lock_hash != payload.get("bank_lock_hash")
    ):
        raise ProtocolError("Fresh prediction task source-cache binding drifted.")
    blocks = {
        source: source_cache.block(source, training_seed, generation_seed)
        for source in sources
    }
    evaluation_path = Path(str(payload["evaluation_array_path"]))
    if sha256_file(evaluation_path) != payload.get("target_frame_sha256"):
        raise ProtocolError("Fresh prediction target frame hash drifted.")
    evaluation = np.load(evaluation_path, mmap_mode="r", allow_pickle=False)
    expected_rows = tuple(str(value) for value in payload["evaluation_row_ids"])
    if (
        evaluation.dtype != np.float32
        or evaluation.ndim != 2
        or evaluation.shape != (len(expected_rows), COMMON_OUTPUT_DIM)
        or not np.isfinite(evaluation).all()
    ):
        raise ProtocolError("Fresh prediction target frame drifted.")
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
    classifier = _classifier_from_payload(payload["classifier"])
    active_fitter = fitter or _fit_classifier
    raw_actions = payload.get("actions")
    if (
        not isinstance(raw_actions, list)
        or len(raw_actions) != EXPECTED_ACTION_COUNT_PER_TARGET
    ):
        raise ProtocolError("Fresh prediction task action menu drifted.")

    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    geometry = target_topup_geometry(sources)
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError(
                "Fresh prediction task action payload is malformed."
            )
        action_id = str(raw["action_id"])
        frozen = FrozenActionPayload(
            target_center=target,
            action_id=action_id,
            source_counts_by_class=raw["final_counts_by_class"],  # type: ignore[arg-type]
            action_hash=str(raw["action_hash"]),
            mean_normalized_midrank_by_source=raw[
                "mean_normalized_midrank_by_source"
            ],  # type: ignore[arg-type]
            source_identity_permutation=raw[
                "source_identity_permutation"
            ],  # type: ignore[arg-type]
        )
        core = rebuild_and_validate_core_action(frozen, raw)
        if action_id == BASE_ACTION_ID:
            composition = compose_equal_union_base_blocks(
                blocks,
                geometry,
                shuffle_seed_by_class=shuffle_seeds,
            )
        else:
            assert core is not None
            composition = compose_residual_topup_blocks(
                blocks,
                core,
                shuffle_seed_by_class=shuffle_seeds,
            )
        fitted = active_fitter(
            composition.embeddings,
            composition.labels,
            np.asarray(evaluation),
            classifier,
            int(payload["threads_per_worker"]),
        )
        if fitted.get("converged") is not True:
            raise ProtocolError(
                "Fresh Stage-70 classifier did not converge; publication is "
                "blocked."
            )
        probability = np.ascontiguousarray(
            fitted["probabilities"],
            dtype=np.float32,
        )
        prediction = np.ascontiguousarray(
            fitted["predictions"],
            dtype=np.uint8,
        )
        if (
            probability.shape != (len(expected_rows),)
            or prediction.shape != (len(expected_rows),)
            or not np.isfinite(probability).all()
            or bool(np.any(probability < 0.0))
            or bool(np.any(probability > 1.0))
            or bool(np.any((prediction != 0) & (prediction != 1)))
        ):
            raise ProtocolError("Fresh classifier output geometry drifted.")
        probabilities.append(probability)
        predictions.append(prediction)
        rows.append(
            {
                "schema_version": PREDICTION_CELL_SCHEMA,
                "target_center": target,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "action_id": action_id,
                "action_hash": frozen.action_hash,
                "task_id": str(payload["task_id"]),
                "task_hash": str(payload["task_hash"]),
                "row_count": len(expected_rows),
                "probability_sha256": array_sha256(probability),
                "prediction_sha256": array_sha256(prediction),
                "composition_hash": composition.composition_hash,
                "classifier_config_hash": str(
                    fitted["classifier_config_hash"]
                ),
                "scaler_state_hash": str(fitted["scaler_state_hash"]),
                "classifier_converged": bool(fitted["converged"]),
                "labels_available_to_fit_or_predict": False,
            }
        )
    write_task_checkpoint(
        task,
        rows=rows,
        probabilities=np.stack(probabilities, axis=0),
        predictions=np.stack(predictions, axis=0),
    )


def build_prediction_tasks(
    config: ResidualTopupFreshConfig,
    *,
    plan: EvaluationPlan,
    policy: FrozenPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
    root: Path,
) -> tuple[PredictionTaskSpec, ...]:
    """Build the canonical 81-task plan with every mutable input hash-bound."""

    tasks: list[PredictionTaskSpec] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = (
                    f"target_{target}__train_{training_seed}__gen_{generation_seed}"
                )
                base = {
                    "schema_version": PREDICTION_TASK_SCHEMA,
                    "task_id": task_id,
                    "target_center": target,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "plan_hash": plan.plan_hash,
                    "policy_lock_hash": policy.policy_lock_hash,
                    "action_library_hash": policy.action_library_hash,
                    "source_cache_hash": source_cache.cache_hash,
                    "bank_lock_hash": source_cache.bank_lock_hash,
                    "generation_lock_hash": generation_lock_hash,
                    "target_cache_content_hash": target_surface.cache_content_hash,
                    "target_cache_protocol_hash": target_surface.cache_protocol_hash,
                    "reservation_hash": target_surface.reservation.reservation_hash,
                    "target_frame_sha256": target_surface.frames_by_center[
                        target
                    ].file_sha256,
                    "source_cache_root": str(source_cache.root.resolve()),
                    "evaluation_array_path": str(
                        (
                            config.fresh_target_cache_root
                            / f"embeddings/by_center/center_{target}.npy"
                        ).resolve()
                    ),
                    "evaluation_row_ids": list(
                        target_surface.frames_by_center[
                            target
                        ].evaluation_row_ids
                    ),
                    "classifier": config.classifier.to_payload(),
                    "threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
                    "actions": [
                        dict(policy.raw_actions_by_key[(target, action_id)])
                        for action_id in expected_action_ids(target)
                    ],
                    "labels_available_to_fit_or_predict": False,
                }
                task_hash = stable_hash(base)
                tasks.append(
                    PredictionTaskSpec(
                        MappingProxyType(
                            {
                                **base,
                                "task_hash": task_hash,
                                "root": str(root.resolve()),
                                "metadata_path": str(
                                    (root / f"tasks/{task_id}.json").resolve()
                                ),
                                "probability_path": str(
                                    (
                                        root
                                        / f"arrays/{task_id}.probabilities.npy"
                                    ).resolve()
                                ),
                                "prediction_path": str(
                                    (
                                        root
                                        / f"arrays/{task_id}.predictions.npy"
                                    ).resolve()
                                ),
                            }
                        )
                    )
                )
    return tuple(tasks)


def spawn_prediction_tasks(tasks: Sequence[PredictionTaskSpec]) -> None:
    """Execute task checkpoints with the frozen spawned-worker geometry."""

    context = mp.get_context("spawn")
    with context.Pool(processes=CLASSIFIER_WORKERS) as pool:
        pool.map(
            _prediction_worker_entry,
            [dict(task.payload) for task in tasks],
            chunksize=1,
        )


def _prediction_worker_entry(payload: Mapping[str, object]) -> None:
    # Keep the entrypoint and its argument picklable under the required spawn
    # start method; MappingProxyType itself cannot cross a process boundary.
    execute_prediction_task(
        PredictionTaskSpec(MappingProxyType(dict(payload)))
    )


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
        raise RuntimeError(
            "Fresh Stage-70 fitting requires threadpoolctl."
        ) from exc
    with threadpool_limits(limits=threads):
        fitted = fit_logistic_classifier(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            spec=classifier,
        )
    probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
    if (
        fitted.classes != (0, 1)
        or probabilities.ndim != 2
        or probabilities.shape[1] != 2
    ):
        raise ProtocolError("Fresh Stage-70 classifier class geometry drifted.")
    return {
        "predictions": np.asarray(fitted.predictions, dtype=np.uint8),
        "probabilities": probabilities[:, 1],
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "converged": fitted.converged,
    }


def _classifier_from_payload(raw: object) -> ClassifierSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Fresh prediction classifier payload is malformed.")
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None
                if raw["class_weight"] is None
                else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(
                None if raw["l1_ratio"] is None else float(raw["l1_ratio"])
            ),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Fresh prediction classifier payload drifted.") from exc


__all__ = (
    "build_prediction_tasks",
    "execute_prediction_task",
    "spawn_prediction_tasks",
)
