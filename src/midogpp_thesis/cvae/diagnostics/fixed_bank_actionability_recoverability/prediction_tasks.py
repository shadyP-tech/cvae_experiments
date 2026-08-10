"""Scratch, checkpoint, and worker mechanics for action predictions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    atomic_npz,
    read_json,
    sha256_array,
    sha256_file,
)
from ...runtime.frozen_source_streams import (
    EXPECTED_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
    FrozenSourceStreamCache,
    source_block_sha256,
)
from .actions import actions_for_target
from .hashing import canonical_hash as stable_hash
from .prediction_contracts import (
    ActionPredictionConfig,
    CHECKPOINT_DIRECTORY,
    EXPECTED_TASK_COUNT,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PredictionCell,
    canonical_cell_keys,
    hash_like,
    package_scaler_state_hash,
)


def target_cache_binding_hash(frame: object) -> str:
    value = getattr(frame, "cache_binding_hash", None)
    if value is None or not hash_like(value):
        raise ProtocolError("Actionability target cache binding is absent.")
    return str(value)


def row_id(row: object) -> str:
    value = getattr(row, "evaluation_row_id", getattr(row, "sample_id", None))
    if value is None or not str(value):
        raise ProtocolError("Actionability target row lacks an opaque identity.")
    return str(value)


def write_target_scratch(
    root: Path,
    *,
    frame: object,
    partition_hash: str,
    target_cache_binding_hash_value: str,
    output_dim: int = COMMON_OUTPUT_DIM,
) -> Mapping[str, object]:
    checkpoint_root = root / CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_root / "target_scratch.json"
    array_path = checkpoint_root / "target_embeddings.npy"
    if manifest_path.is_file() and array_path.is_file():
        payload = read_json(manifest_path)
        validate_target_scratch(
            payload,
            expected_partition_hash=partition_hash,
            expected_target_cache_binding_hash=target_cache_binding_hash_value,
            output_dim=output_dim,
        )
        return payload
    rows: list[object] = []
    row_ids: dict[str, list[str]] = {}
    case_ids: dict[str, list[str]] = {}
    offsets: dict[str, dict[str, object]] = {}
    cursor = 0
    rows_by_center = getattr(frame, "rows_by_center")
    for target in CENTERS:
        target_rows = tuple(rows_by_center[target])
        identifiers = tuple(row_id(row) for row in target_rows)
        cases = tuple(str(getattr(row, "case_id")) for row in target_rows)
        offsets[target] = {
            "start": cursor,
            "stop": cursor + len(target_rows),
            "row_count": len(target_rows),
            "row_identity_hash": stable_hash(list(identifiers)),
        }
        rows.extend(target_rows)
        row_ids[target] = list(identifiers)
        case_ids[target] = list(cases)
        cursor += len(target_rows)
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    if embeddings.shape != (cursor, output_dim) or not np.isfinite(embeddings).all():
        raise ProtocolError("Actionability target scratch geometry drifted.")
    for target in CENTERS:
        offset = offsets[target]
        offset["embedding_slice_sha256"] = sha256_array(
            embeddings[int(offset["start"]) : int(offset["stop"])]
        )
    atomic_npy(array_path, embeddings)
    unhashed = {
        "schema_version": "midogpp_actionability_target_scratch_v1",
        "array_path": str(array_path.resolve()),
        "array_sha256": sha256_array(embeddings),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "partition_hash": partition_hash,
        "target_cache_binding_hash": target_cache_binding_hash_value,
        "offsets": offsets,
        "row_ids_by_center": row_ids,
        "case_ids_by_center": case_ids,
        "labels_stored": False,
        "manifest_opened": False,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(manifest_path, payload)
    return payload


def validate_target_scratch(
    payload: Mapping[str, object],
    *,
    expected_partition_hash: str,
    expected_target_cache_binding_hash: str,
    output_dim: int = COMMON_OUTPUT_DIM,
) -> None:
    path = Path(str(payload.get("array_path", "")))
    if not path.is_file():
        raise ProtocolError("Actionability target scratch array is absent.")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    unhashed = {key: value for key, value in payload.items() if key != "scratch_hash"}
    offsets = payload.get("offsets")
    if (
        payload.get("scratch_hash") != stable_hash(unhashed)
        or payload.get("partition_hash") != expected_partition_hash
        or payload.get("target_cache_binding_hash")
        != expected_target_cache_binding_hash
        or payload.get("shape") != list(values.shape)
        or payload.get("dtype") != str(values.dtype)
        or payload.get("array_sha256") != sha256_array(values)
        or values.ndim != 2
        or values.shape[1] != output_dim
        or values.dtype != np.float32
        or not isinstance(offsets, Mapping)
        or tuple(offsets) != CENTERS
        or payload.get("labels_stored") is not False
        or payload.get("manifest_opened") is not False
    ):
        raise ProtocolError("Actionability target scratch failed validation.")
    row_ids = payload.get("row_ids_by_center")
    case_ids = payload.get("case_ids_by_center")
    if (
        not isinstance(row_ids, Mapping)
        or not isinstance(case_ids, Mapping)
        or tuple(row_ids) != CENTERS
        or tuple(case_ids) != CENTERS
    ):
        raise ProtocolError("Actionability target scratch identity maps drifted.")
    cursor = 0
    for center in CENTERS:
        raw = offsets[center]
        if not isinstance(raw, Mapping):
            raise ProtocolError("Actionability target scratch offset is malformed.")
        start, stop = int(raw.get("start", -1)), int(raw.get("stop", -1))
        identities = tuple(str(value) for value in row_ids[center])
        cases = tuple(str(value) for value in case_ids[center])
        if (
            start != cursor
            or stop <= start
            or stop - start != len(identities)
            or len(cases) != len(identities)
            or raw.get("row_count") != len(identities)
            or raw.get("row_identity_hash") != stable_hash(list(identities))
            or raw.get("embedding_slice_sha256") != sha256_array(values[start:stop])
        ):
            raise ProtocolError("Actionability target scratch offset drifted.")
        cursor = stop
    if cursor != len(values):
        raise ProtocolError("Actionability target scratch coverage drifted.")


def build_tasks(
    config: ActionPredictionConfig,
    source_cache: FrozenSourceStreamCache,
    *,
    scratch: Mapping[str, object],
    library_payload: Mapping[str, Sequence[Mapping[str, object]]],
    action_library_hash: str,
    partition_hash: str,
    root: Path,
) -> tuple[Mapping[str, object], ...]:
    offsets = scratch.get("offsets")
    if not isinstance(offsets, Mapping):
        raise ProtocolError("Actionability target offsets are absent.")
    records = [record.to_payload() for record in source_cache.records]
    classifier = getattr(config, "classifier")
    classifier_payload = (
        classifier.to_payload() if hasattr(classifier, "to_payload") else dict(classifier)
    )
    task_root = root / CHECKPOINT_DIRECTORY / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    tasks: list[Mapping[str, object]] = []
    for target, training, generation in product(
        CENTERS, TRAINING_SEEDS, GENERATION_SEEDS
    ):
        offset = offsets[target]
        if not isinstance(offset, Mapping):
            raise ProtocolError("Actionability target offset is malformed.")
        task_id = f"target_{target}_train_{training}_generation_{generation}"
        unhashed = {
            "schema_version": "midogpp_actionability_prediction_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "source_stream_lock_hash": source_cache.lock_hash,
            "partition_hash": partition_hash,
            "action_library_hash": action_library_hash,
            "target_center": target,
            "training_seed": training,
            "generation_seed": generation,
            "candidate_sources": [center for center in CENTERS if center != target],
            "source_array_path": str(source_cache.source_array_path.resolve()),
            "source_array_sha256": str(
                source_cache.lock_payload["source_array_sha256"]
            ),
            "source_index_rows": records,
            "source_index_rows_hash": stable_hash(records),
            "target_array_path": str(scratch["array_path"]),
            "target_array_sha256": str(scratch["array_sha256"]),
            "target_array_shape": list(scratch["shape"]),
            "target_array_dtype": str(scratch["dtype"]),
            "target_cache_binding_hash": str(scratch["target_cache_binding_hash"]),
            "target_start": int(offset["start"]),
            "target_stop": int(offset["stop"]),
            "target_row_identity_hash": str(offset["row_identity_hash"]),
            "target_slice_sha256": str(offset["embedding_slice_sha256"]),
            "actions": [dict(value) for value in library_payload[target]],
            "classifier": classifier_payload,
            "threads_per_fit": int(config.runtime["classifier_threads_per_worker"]),
            "labels_available": False,
            "target_expert_available": False,
        }
        task_hash = stable_hash(unhashed)
        tasks.append(
            {
                **unhashed,
                "task_hash": task_hash,
                "checkpoint_json_path": str(task_root / f"{task_id}.json"),
                "checkpoint_npz_path": str(task_root / f"{task_id}.npz"),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ProtocolError("Actionability prediction task coverage drifted.")
    return tuple(tasks)


def execute_or_resume(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Actionability predictions require exactly four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(prediction_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = load_checkpoint(task)
                if loaded is None:
                    raise ProtocolError(
                        "Actionability worker returned without a valid checkpoint."
                    )
                completed[str(task["task_id"])] = loaded
                print(
                    f"[actionability] prediction tasks {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Actionability prediction checkpoint coverage is incomplete.")
    return MappingProxyType(completed)


def prediction_task(
    task: Mapping[str, object], *, output_dim: int = COMMON_OUTPUT_DIM
) -> None:
    actions = task.get("actions")
    candidates = tuple(str(value) for value in task.get("candidate_sources", ()))
    target = str(task.get("target_center", ""))
    expected_actions = [action.to_payload() for action in actions_for_target(target)]
    expected_candidates = tuple(center for center in CENTERS if center != target)
    task_unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    if (
        task.get("task_hash") != stable_hash(task_unhashed)
        or task.get("labels_available") is not False
        or task.get("target_expert_available") is not False
        or candidates != expected_candidates
        or not isinstance(actions, list)
        or len(actions) != PHYSICAL_ACTION_COUNT_PER_TARGET
        or actions != expected_actions
        or task.get("action_library_hash")
        != stable_hash(
            {
                center: [action.to_payload() for action in actions_for_target(center)]
                for center in CENTERS
            }
        )
    ):
        raise ProtocolError("Actionability prediction task escaped its boundary.")
    blocks, evaluation = load_task_arrays(
        task, candidates=candidates, output_dim=output_dim
    )
    classifier = classifier_from_payload(task["classifier"])
    probability_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Actionability fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for raw in actions:
            if not isinstance(raw, Mapping):
                raise ProtocolError("Actionability task action is malformed.")
            train_x, train_y, weights, composition_hash = compose_action(
                blocks, raw, candidates, output_dim=output_dim
            )
            fitted = fit_logistic_classifier(
                train_x,
                train_y,
                evaluation,
                spec=classifier,
                sample_weight=(None if np.all(weights == 1.0) else weights),
            )
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or matrix.shape != (len(evaluation), 2)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != classifier.config_hash
            ):
                raise ProtocolError("Actionability classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
            predictions = np.ascontiguousarray(
                positive >= np.float32(0.5), dtype=np.uint8
            )
            probability_hash = sha256_array(positive)
            prediction_hash = sha256_array(predictions)
            neutral_scaler_state_hash = str(fitted.scaler_state_hash)
            scaler_state_hash = package_scaler_state_hash(
                neutral_scaler_state_hash
            )
            fit_unhashed = {
                "schema_version": "midogpp_actionability_classifier_fit_v1",
                "task_hash": task["task_hash"],
                "action_id": raw["action_id"],
                "action_hash": raw["action_hash"],
                "composition_hash": composition_hash,
                "classifier_config_hash": fitted.classifier_config_hash,
                "scaler_state_hash": scaler_state_hash,
                "neutral_scaler_state_hash": neutral_scaler_state_hash,
                "probability_sha256": probability_hash,
                "predictions_sha256": prediction_hash,
                "sample_weight_scope": "logistic_regression_fit_only",
                "scaler_fit_used_sample_weight": False,
                "labels_available": False,
            }
            probability_rows.append(positive)
            metadata_rows.append(
                {
                    "action_id": str(raw["action_id"]),
                    "action_hash": str(raw["action_hash"]),
                    "probability_sha256": probability_hash,
                    "predictions_sha256": prediction_hash,
                    "composition_hash": composition_hash,
                    "scaler_state_hash": scaler_state_hash,
                    "neutral_scaler_state_hash": neutral_scaler_state_hash,
                    "fit_provenance_hash": stable_hash(fit_unhashed),
                }
            )
    matrix = np.ascontiguousarray(np.stack(probability_rows), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(npz_path, probabilities=matrix)
    checkpoint_unhashed = {
        "schema_version": "midogpp_actionability_prediction_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "target_center": task["target_center"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "target_row_identity_hash": task["target_row_identity_hash"],
        "array_sha256": sha256_file(npz_path),
        "array_shape": list(matrix.shape),
        "array_dtype": str(matrix.dtype),
        "actions": metadata_rows,
        "labels_available": False,
        "target_expert_available": False,
    }
    atomic_json(
        json_path,
        {**checkpoint_unhashed, "checkpoint_hash": stable_hash(checkpoint_unhashed)},
    )


def load_checkpoint(task: Mapping[str, object]) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.is_file() and not npz_path.is_file():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("Actionability checkpoint is partially present.")
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    actions = payload.get("actions")
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Actionability checkpoint array is unreadable.") from exc
    with archive:
        if tuple(archive.files) != ("probabilities",):
            raise ProtocolError("Actionability checkpoint members drifted.")
        values = np.ascontiguousarray(archive["probabilities"], dtype=np.float32)
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("task_id") != task.get("task_id")
        or payload.get("array_sha256") != sha256_file(npz_path)
        or payload.get("array_shape") != list(values.shape)
        or payload.get("array_dtype") != str(values.dtype)
        or values.shape[0] != PHYSICAL_ACTION_COUNT_PER_TARGET
        or not isinstance(actions, list)
        or len(actions) != PHYSICAL_ACTION_COUNT_PER_TARGET
        or payload.get("labels_available") is not False
        or payload.get("target_expert_available") is not False
    ):
        raise ProtocolError("Actionability checkpoint validation failed.")
    expected_actions = task.get("actions")
    if not isinstance(expected_actions, list):
        raise ProtocolError("Actionability task action manifest is malformed.")
    for ordinal, row in enumerate(actions):
        if (
            not isinstance(row, Mapping)
            or row.get("action_id") != expected_actions[ordinal].get("action_id")
            or row.get("action_hash") != expected_actions[ordinal].get("action_hash")
            or row.get("probability_sha256") != sha256_array(values[ordinal])
            or row.get("predictions_sha256")
            != sha256_array((values[ordinal] >= np.float32(0.5)).astype(np.uint8))
            or not hash_like(row.get("composition_hash"))
            or row.get("scaler_state_hash")
            != package_scaler_state_hash(row.get("neutral_scaler_state_hash"))
            or not hash_like(row.get("fit_provenance_hash"))
        ):
            raise ProtocolError("Actionability checkpoint probability hash drifted.")
    return payload


def load_task_arrays(
    task: Mapping[str, object],
    *,
    candidates: tuple[str, ...],
    output_dim: int = COMMON_OUTPUT_DIM,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw_index = task.get("source_index_rows")
    if (
        not isinstance(raw_index, list)
        or task.get("source_index_rows_hash") != stable_hash(raw_index)
    ):
        raise ProtocolError("Actionability task source index drifted.")
    index: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for raw in raw_index:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Actionability source-index row is malformed.")
        key = (
            str(raw.get("source_center", "")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        index[key] = raw
    if len(index) != EXPECTED_STREAM_COUNT:
        raise ProtocolError("Actionability source-index coverage drifted.")
    source_path = Path(str(task["source_array_path"]))
    try:
        source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Actionability source array is unreadable.") from exc
    if (
        source_values.shape
        != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, output_dim)
        or source_values.dtype != np.float32
    ):
        raise ProtocolError("Actionability source array geometry drifted.")
    training = int(task["training_seed"])
    generation = int(task["generation_seed"])
    blocks: dict[str, np.ndarray] = {}
    for source in candidates:
        record = index[(source, training, generation)]
        block = source_values[int(record["block_ordinal"])]
        if source_block_sha256(block) != record.get("output_sha256"):
            raise ProtocolError("Actionability source block bytes drifted.")
        blocks[source] = block
    target_path = Path(str(task["target_array_path"]))
    try:
        target_values = np.load(target_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Actionability target array is unreadable.") from exc
    start, stop = int(task["target_start"]), int(task["target_stop"])
    evaluation = np.ascontiguousarray(target_values[start:stop], dtype=np.float32)
    if (
        evaluation.ndim != 2
        or evaluation.shape[1] != output_dim
        or not np.isfinite(evaluation).all()
        or sha256_array(evaluation) != task.get("target_slice_sha256")
    ):
        raise ProtocolError("Actionability target slice drifted.")
    return blocks, evaluation


def compose_action(
    blocks: Mapping[str, np.ndarray],
    action: Mapping[str, object],
    candidates: tuple[str, ...],
    *,
    output_dim: int = COMMON_OUTPUT_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    raw_counts = action.get("counts_by_class")
    raw_weights = action.get("sample_weight_by_source")
    if not isinstance(raw_counts, Mapping) or not isinstance(raw_weights, Mapping):
        raise ProtocolError("Actionability action composition is incomplete.")
    weights_by_source = {
        str(source): float(value) for source, value in raw_weights.items()
    }
    if (
        tuple(weights_by_source) != candidates
        or not all(
            np.isfinite(value) and value > 0.0
            for value in weights_by_source.values()
        )
    ):
        raise ProtocolError("Actionability source weights drifted.")
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    canonical_counts: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        raw = raw_counts.get(str(label), raw_counts.get(label))
        if not isinstance(raw, Mapping):
            raise ProtocolError("Actionability class counts are absent.")
        counts = {str(source): int(value) for source, value in raw.items()}
        if tuple(counts) != candidates:
            raise ProtocolError("Actionability action source order drifted.")
        canonical_counts[str(label)] = counts
        for source, count in counts.items():
            if count <= 0 or count > SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Actionability source prefix exceeds capacity.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(
                np.asarray(blocks[source][start : start + count], dtype=np.float32)
            )
            labels.append(np.full(count, label, dtype=np.uint8))
            weights.append(np.full(count, weights_by_source[source], dtype=np.float64))
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    sample_weight = np.ascontiguousarray(np.concatenate(weights), dtype=np.float64)
    effective_by_class = {
        str(label): sum(
            canonical_counts[str(label)][source] * weights_by_source[source]
            for source in candidates
        )
        for label in (0, 1)
    }
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != output_dim
        or not np.isfinite(embeddings).all()
        or not np.isfinite(sample_weight).all()
        or any(
            abs(value - round(value)) > 1e-12
            for value in effective_by_class.values()
        )
    ):
        raise ProtocolError("Actionability composed training surface drifted.")
    composition = {
        "counts_by_class": canonical_counts,
        "sample_weight_by_source": weights_by_source,
        "effective_weight_by_class": effective_by_class,
        "action_hash": action["action_hash"],
        "sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
    }
    return embeddings, truth, sample_weight, stable_hash(composition)


def cells_from_checkpoints(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
) -> tuple[PredictionCell, ...]:
    cells: list[PredictionCell] = []
    for task in tasks:
        payload = completed[str(task["task_id"])]
        actions = payload["actions"]
        archive = np.load(Path(str(task["checkpoint_npz_path"])), allow_pickle=False)
        with archive:
            values = np.ascontiguousarray(archive["probabilities"], dtype=np.float32)
        for ordinal, raw in enumerate(actions):
            cells.append(
                PredictionCell(
                    target_center=str(task["target_center"]),
                    action_id=str(raw["action_id"]),
                    action_hash=str(raw["action_hash"]),
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    row_identity_hash=str(task["target_row_identity_hash"]),
                    probabilities=values[ordinal],
                    probability_sha256=str(raw["probability_sha256"]),
                    predictions_sha256=str(raw["predictions_sha256"]),
                    composition_hash=str(raw["composition_hash"]),
                    scaler_state_hash=str(raw["scaler_state_hash"]),
                    fit_provenance_hash=str(raw["fit_provenance_hash"]),
                )
            )
    result = tuple(cells)
    if tuple(cell.key for cell in result) != canonical_cell_keys():
        raise ProtocolError("Actionability checkpoint cell order drifted.")
    return result


def classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Actionability classifier payload is malformed.") from exc


__all__ = (
    "build_tasks",
    "cells_from_checkpoints",
    "classifier_from_payload",
    "compose_action",
    "execute_or_resume",
    "load_checkpoint",
    "load_task_arrays",
    "prediction_task",
    "row_id",
    "target_cache_binding_hash",
    "validate_target_scratch",
    "write_target_scratch",
)
