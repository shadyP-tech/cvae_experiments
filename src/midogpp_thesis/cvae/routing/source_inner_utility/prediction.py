"""Label-free classifier fitting and prediction for source-inner utility."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...generation.generation import (
    generate_source_block,
    source_generation_plan,
)
from ...protocol import ProtocolError
from .cache_inputs import UnlabeledValidationFrame
from .contracts import (
    EXPECTED_FIT_COUNT,
    FULL_SOURCE_BUDGET_PER_CLASS,
    SourceIdentity,
    evaluation_order_hash,
    expected_fit_keys,
    source_identities_from_generation_lock,
)


FIT_COLUMNS = (
    "schema_version",
    "fit_ordinal",
    "fit_id",
    "source_center",
    "training_seed",
    "generation_seed",
    "source_stream_id",
    "expert_lock_hash",
    "checkpoint_hash",
    "checkpoint_file_sha256",
    "frame_hash",
    "frame_file_sha256",
    "sampler_state_hash",
    "sampler_file_sha256",
    "generated_block_sha256",
    "generated_row_count",
    "generated_rows_per_class",
    "generated_class_0_count",
    "generated_class_1_count",
    "classifier_family",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_classes",
    "classifier_n_iter",
    "classifier_converged",
    "prediction_array_row",
    "all_eval_row_count",
    "all_eval_row_hash",
    "all_eval_prediction_sha256",
    "all_eval_probability_sha256",
    "eval_labels_available_to_fit_or_predict",
    "seed_selection_performed",
)


@dataclass(frozen=True)
class PredictionPass:
    """Complete label-free prediction surface for all 81 classifiers."""

    evaluation_rows: tuple[object, ...]
    fit_rows: tuple[Mapping[str, object], ...]
    y_pred: np.ndarray
    prob_pos: np.ndarray

    def __post_init__(self) -> None:
        predictions = np.asarray(self.y_pred)
        probabilities = np.asarray(self.prob_pos)
        expected_shape = (len(self.fit_rows), len(self.evaluation_rows))
        if (
            predictions.shape != expected_shape
            or probabilities.shape != expected_shape
            or predictions.dtype != np.uint8
            or probabilities.dtype != np.float32
        ):
            raise ProtocolError("Prediction arrays do not match fit/evaluation indices.")
        if not np.isin(predictions, [0, 1]).all():
            raise ProtocolError("Candidate predictions must be binary.")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(
            probabilities > 1.0
        ):
            raise ProtocolError("Candidate probabilities must be finite values in [0,1].")
        if tuple(int(row["fit_ordinal"]) for row in self.fit_rows) != tuple(
            range(len(self.fit_rows))
        ):
            raise ProtocolError("Classifier-fit ordinals drifted from prediction row order.")

    @property
    def evaluation_order_hash(self) -> str:
        return evaluation_order_hash(self.evaluation_rows)  # type: ignore[arg-type]


def run_label_free_prediction_pass(
    frame: UnlabeledValidationFrame,
    *,
    bank_root: str | Path,
    classifier_spec: ClassifierSpec,
    generation_lock: GenerationLock | None = None,
    generation_keys: Sequence[SourceGenerationKey] | None = None,
    source_identities: Mapping[tuple[str, int], SourceIdentity] | None = None,
    per_class: int = FULL_SOURCE_BUDGET_PER_CLASS,
    device: str = "cpu",
    threads_per_fit: int = 1,
    expected_fit_count: int = EXPECTED_FIT_COUNT,
    expert_loader: Callable[..., object] = load_routing_authorized_expert,
    block_generator: Callable[..., object] = generate_source_block,
    classifier_fitter: Callable[..., object] = fit_logistic_classifier,
) -> PredictionPass:
    """Fit exactly one classifier per source/seed cell without eval labels.

    The signature intentionally has no evaluation-label argument. Each fitted
    classifier predicts the full ordered validation frame once; pseudo-target
    slicing happens only in ``metric_scoring.score_prediction_pass``.
    """

    if threads_per_fit <= 0:
        raise ProtocolError("threads_per_fit must be positive.")
    if per_class <= 0:
        raise ProtocolError("Source-inner generation budget must be positive.")
    if generation_keys is None:
        if generation_lock is None:
            raise ProtocolError("A validated GenerationLock is required.")
        generation_keys = source_generation_plan(generation_lock)
    keys = tuple(
        sorted(
            generation_keys,
            key=lambda key: (key.source_center, key.training_seed, key.generation_seed),
        )
    )
    observed_fit_keys = {
        (key.source_center, key.training_seed, key.generation_seed) for key in keys
    }
    if (
        len(keys) != int(expected_fit_count)
        or len(observed_fit_keys) != len(keys)
        or (
            int(expected_fit_count) == EXPECTED_FIT_COUNT
            and observed_fit_keys != set(expected_fit_keys())
        )
    ):
        raise ProtocolError("Source-inner classifier-fit geometry drifted.")
    if source_identities is None:
        if generation_lock is None:
            raise ProtocolError("GenerationLock expert identities are required.")
        source_identities = source_identities_from_generation_lock(
            generation_lock.to_payload()
        )

    matrix = np.asarray(frame.embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame.rows):
        raise ProtocolError("Label-free prediction frame is malformed.")
    prediction_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    fit_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[SourceGenerationKey]] = defaultdict(list)
    for key in keys:
        grouped[(key.source_center, key.training_seed)].append(key)
    if len(grouped) * len({key.generation_seed for key in keys}) != len(keys):
        raise ProtocolError("Source-inner generation keys are not a complete seed product.")

    for source_center, training_seed in sorted(grouped):
        identity = source_identities.get((source_center, training_seed))
        if identity is None:
            raise ProtocolError("GenerationLock lacks a classifier source identity.")
        expert = expert_loader(
            bank_root,
            source_center=source_center,
            training_seed=training_seed,
            device=device,
        )
        if (
            str(getattr(expert, "source_center", "")) != source_center
            or int(getattr(expert, "training_seed", -1)) != training_seed
            or str(getattr(expert, "expert_lock_hash", ""))
            != identity.expert_lock_hash
            or str(getattr(expert, "checkpoint_hash", "")) != identity.checkpoint_hash
        ):
            raise ProtocolError("Loaded routing expert drifted from GenerationLock identity.")
        try:
            source_keys = sorted(
                grouped[(source_center, training_seed)],
                key=lambda item: item.generation_seed,
            )
            for key in source_keys:
                block = block_generator(
                    expert,
                    key,
                    per_class=per_class,
                    device=device,
                )
                generated = np.asarray(getattr(block, "embeddings", None))
                generated_labels_raw = np.asarray(getattr(block, "labels", None))
                generated_labels = generated_labels_raw.astype(np.uint8, copy=False)
                if (
                    getattr(block, "key", None) != key
                    or generated.shape != (2 * per_class, matrix.shape[1])
                    or generated.dtype != np.float32
                    or generated_labels_raw.shape != (2 * per_class,)
                    or generated_labels_raw.dtype != np.int64
                    or not np.isin(generated_labels_raw, [0, 1]).all()
                    or int(np.sum(generated_labels == 0)) != per_class
                    or int(np.sum(generated_labels == 1)) != per_class
                    or not np.isfinite(generated).all()
                ):
                    raise ProtocolError(
                        "Fresh generated source block failed its full-budget contract."
                    )
                generated_sha = str(getattr(block, "output_sha256", ""))
                if not _is_sha256(generated_sha):
                    raise ProtocolError("Generated source block lacks a SHA-256 identity.")
                if generated_sha != generated_block_sha256(
                    generated,
                    generated_labels_raw,
                ):
                    raise ProtocolError("Generated source block content hash drifted.")
                fitted = _bounded_fit(
                    classifier_fitter,
                    generated,
                    generated_labels,
                    matrix,
                    classifier_spec,
                    threads_per_fit=threads_per_fit,
                )
                classes = tuple(int(value) for value in getattr(fitted, "classes", ()))
                predictions_raw = np.asarray(getattr(fitted, "predictions", None))
                predictions = predictions_raw.astype(np.uint8, copy=False)
                probabilities = np.asarray(
                    getattr(fitted, "probabilities", None), dtype=float
                )
                if (
                    classes != (0, 1)
                    or predictions_raw.shape != (len(frame.rows),)
                    or not np.isin(predictions_raw, [0, 1]).all()
                    or probabilities.shape != (len(frame.rows), 2)
                    or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0.0)
                    or np.any(probabilities > 1.0)
                    or not np.allclose(
                        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-7
                    )
                ):
                    raise ProtocolError("Candidate classifier prediction output drifted.")
                converged = bool(getattr(fitted, "converged", False))
                if not converged:
                    raise ProtocolError("Candidate classifier did not converge.")
                fit_ordinal = len(fit_rows)
                fit_id = stable_hash(
                    {
                        "source_center": source_center,
                        "training_seed": training_seed,
                        "generation_seed": key.generation_seed,
                        "source_stream_id": key.stream_id,
                        "generated_block_sha256": generated_sha,
                        "classifier_config_hash": str(
                            getattr(fitted, "classifier_config_hash", "")
                        ),
                        "evaluation_order_hash": frame.row_order_hash,
                    }
                )
                positive = probabilities[:, 1].astype(np.float32, copy=False)
                predictions_u8 = predictions.astype(np.uint8, copy=False)
                row = {
                    "schema_version": (
                        "midogpp_uniform_b_v2_candidate_classifier_fit_v1"
                    ),
                    "fit_ordinal": fit_ordinal,
                    "fit_id": fit_id,
                    "source_center": source_center,
                    "training_seed": training_seed,
                    "generation_seed": key.generation_seed,
                    "source_stream_id": key.stream_id,
                    **identity.to_payload(),
                    "generated_block_sha256": generated_sha,
                    "generated_row_count": int(generated.shape[0]),
                    "generated_rows_per_class": per_class,
                    "generated_class_0_count": int(np.sum(generated_labels == 0)),
                    "generated_class_1_count": int(np.sum(generated_labels == 1)),
                    "classifier_family": classifier_spec.family,
                    "classifier_config_hash": str(
                        getattr(fitted, "classifier_config_hash", "")
                    ),
                    "scaler_state_hash": str(getattr(fitted, "scaler_state_hash", "")),
                    "classifier_classes": _json_compact(list(classes)),
                    "classifier_n_iter": _json_compact(
                        [int(value) for value in getattr(fitted, "n_iter", ())]
                    ),
                    "classifier_converged": True,
                    "prediction_array_row": fit_ordinal,
                    "all_eval_row_count": len(frame.rows),
                    "all_eval_row_hash": frame.row_order_hash,
                    "all_eval_prediction_sha256": array_sha256(predictions_u8),
                    "all_eval_probability_sha256": array_sha256(positive),
                    "eval_labels_available_to_fit_or_predict": False,
                    "seed_selection_performed": False,
                }
                if row["classifier_config_hash"] != classifier_spec.config_hash:
                    raise ProtocolError("Candidate classifier config hash drifted.")
                if not row["scaler_state_hash"]:
                    raise ProtocolError(
                        "Candidate classifier lacks its fit-only scaler hash."
                    )
                fit_rows.append(row)
                prediction_rows.append(predictions_u8)
                probability_rows.append(positive)
        finally:
            del expert
            gc.collect()
            _empty_device_cache(device)

    if len(fit_rows) != int(expected_fit_count):
        raise ProtocolError("Source-inner prediction pass did not fit every declared cell.")
    return PredictionPass(
        evaluation_rows=tuple(frame.rows),
        fit_rows=tuple(fit_rows),
        y_pred=np.stack(prediction_rows, axis=0).astype(np.uint8, copy=False),
        prob_pos=np.stack(probability_rows, axis=0).astype(np.float32, copy=False),
    )


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def generated_block_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    """Reconstruct the canonical hash emitted by ``generate_source_block``."""

    digest = hashlib.sha256()
    for array in (embeddings, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _bounded_fit(
    fitter: Callable[..., object],
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    spec: ClassifierSpec,
    *,
    threads_per_fit: int,
) -> object:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:  # pragma: no cover - sklearn normally provides it
        return fitter(train_embeddings, train_labels, eval_embeddings, spec=spec)
    with threadpool_limits(limits=threads_per_fit):
        return fitter(train_embeddings, train_labels, eval_embeddings, spec=spec)


def _empty_device_cache(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - generation requires torch
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_sha256(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


__all__ = (
    "FIT_COLUMNS",
    "PredictionPass",
    "array_sha256",
    "generated_block_sha256",
    "run_label_free_prediction_pass",
)
