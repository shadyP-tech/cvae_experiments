"""Pre-plan target probe and post-plan final target probability execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import fit_logistic_classifier
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .actions import FrozenEnsembleEndpointActionLibrary
from .artifact_io import atomic_json, atomic_npy, persist_or_validate_csv, read_json, sha256_file
from .combined_prediction_io import (
    load_task_checkpoint,
    read_combined_store,
    write_combined_store,
    write_task_checkpoint,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    H_X_E_ACTION_PREFIX,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
)
from .input_contracts import row_identity_hash
from .prediction_contracts import CombinedPredictionCell, CombinedPredictionStore, array_sha256, build_store
from .source_cache import SOURCE_ROWS_PER_CLASS, SourceCache
from ..utility_aligned_exact_tail_router.development_prediction_worker import classifier_from_payload


TARGET_ARRAY_MEMBER = "arrays/ensemble_endpoint_target_predictions.npz"
TARGET_INDEX_MEMBER = "manifests/ensemble_endpoint_target_prediction_cache.json"
TARGET_INDEX_TABLE_MEMBER = "tables/target_prediction_index.csv"
TARGET_PROBE_SEAL_MEMBER = "manifests/ensemble_endpoint_target_probe_seal.json"
TARGET_CHECKPOINT_DIRECTORY = "checkpoints/ensemble_endpoint_target"
EXPECTED_PROBE_CELL_COUNT = 9 * 9 * 9
EXPECTED_FINAL_CELL_COUNT = 9 * 13 * 9
EXPECTED_TARGET_UNIQUE_FIT_COUNT = 810

TARGET_INDEX_COLUMNS = (
    "schema_version", "cell_ordinal", "target_center", "action_id", "action_hash",
    "training_seed", "generation_seed", "support_row_identity_hash",
    "evaluation_row_identity_hash", "support_prediction_sha256",
    "support_probability_sha256", "evaluation_prediction_sha256",
    "evaluation_probability_sha256", "composition_hash", "scaler_state_hash",
    "fit_provenance_hash", "aliased_from_action_id", "labels_available",
)


def materialize_target_probe_predictions(
    config: object,
    source_cache: SourceCache,
    frame: object,
    partitions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> CombinedPredictionStore:
    """Fit B plus eight Hxe actions before any target plan exists."""

    checkpoint_root = root / TARGET_CHECKPOINT_DIRECTORY
    array_path = checkpoint_root / "probe_predictions.npz"
    index_path = checkpoint_root / "probe_index.json"
    probe_hash = _probe_library_hash()
    final_array = root / TARGET_ARRAY_MEMBER
    final_index = root / TARGET_INDEX_MEMBER
    if final_array.is_file() and final_index.is_file():
        final = read_combined_store(final_array, final_index)
        probe_actions = {
            (target, action["action_id"]): action
            for target in CENTERS for action in _probe_actions(target)
        }
        cells: list[CombinedPredictionCell] = []
        for cell in final.cells:
            action = probe_actions.get((cell.scope_id, cell.action_id))
            if action is None:
                continue
            cells.append(CombinedPredictionCell(
                scope_id=cell.scope_id, action_id=cell.action_id,
                action_hash=str(action["action_hash"]), training_seed=cell.training_seed,
                generation_seed=cell.generation_seed,
                support_row_identity_hash=cell.support_row_identity_hash,
                evaluation_row_identity_hash=cell.evaluation_row_identity_hash,
                support_predictions=cell.support_predictions,
                support_probabilities=cell.support_probabilities,
                evaluation_predictions=cell.evaluation_predictions,
                evaluation_probabilities=cell.evaluation_probabilities,
                composition_hash=cell.composition_hash,
                scaler_state_hash=cell.scaler_state_hash,
                fit_provenance_hash=cell.fit_provenance_hash,
                aliased_from_action_id=None,
            ))
        rebuilt = build_store(
            role="target_probe", cells=cells,
            source_cache_lock_hash=source_cache_lock_hash,
            partition_lock_hash=str(partitions.lock_hash),
            action_library_hash=probe_hash,
            expected_cell_count=EXPECTED_PROBE_CELL_COUNT,
            unique_classifier_fit_count=EXPECTED_PROBE_CELL_COUNT,
        )
        _validate_probe(rebuilt, source_cache_lock_hash, str(partitions.lock_hash), probe_hash)
        _validate_probe_seal(root, probe=rebuilt, partitions=partitions)
        return rebuilt
    if array_path.is_file() and index_path.is_file():
        store = read_combined_store(array_path, index_path)
        _validate_probe(store, source_cache_lock_hash, str(partitions.lock_hash), probe_hash)
        _persist_probe_seal(root, store=store, partitions=partitions)
        return store
    scratch = _write_target_scratch(root, frame=frame, partitions=partitions)
    tasks = _build_probe_tasks(
        config, source_cache, partitions, scratch=scratch,
        source_cache_lock_hash=source_cache_lock_hash, root=root,
    )
    completed = _execute_or_resume(tasks, workers=int(getattr(config, "runtime")["classifier_workers"]))
    cells = _cells_from_tasks(tasks, completed)
    store = build_store(
        role="target_probe", cells=cells, source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(partitions.lock_hash), action_library_hash=probe_hash,
        expected_cell_count=EXPECTED_PROBE_CELL_COUNT,
        unique_classifier_fit_count=EXPECTED_PROBE_CELL_COUNT,
    )
    _validate_probe(store, source_cache_lock_hash, str(partitions.lock_hash), probe_hash)
    write_combined_store(array_path, index_path, store)
    verified = read_combined_store(array_path, index_path)
    _validate_probe(verified, source_cache_lock_hash, str(partitions.lock_hash), probe_hash)
    _persist_probe_seal(root, store=verified, partitions=partitions)
    return verified


def materialize_target_predictions(
    config: object,
    source_cache: SourceCache,
    probe: CombinedPredictionStore,
    library: FrozenEnsembleEndpointActionLibrary,
    frame: object,
    partitions: object,
    case_folds: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> CombinedPredictionStore:
    """Add U fits and freeze B/U/G/R2E/P/Hxe cells without refitting aliases."""

    del frame
    _validate_probe(
        probe, source_cache_lock_hash, str(partitions.lock_hash), _probe_library_hash()
    )
    _validate_probe_seal(root, probe=probe, partitions=partitions)
    array_path = root / TARGET_ARRAY_MEMBER
    index_path = root / TARGET_INDEX_MEMBER
    if array_path.is_file() and index_path.is_file():
        store = read_combined_store(array_path, index_path)
        _validate_final(store, library, source_cache_lock_hash, str(case_folds.lock_hash))
        _write_target_index_table(root, store)
        _remove_target_working_files(root)
        return store
    scratch = read_json(root / TARGET_CHECKPOINT_DIRECTORY / "combined_validation_index.json")
    tasks = _build_uniform_tasks(
        config, source_cache, library, partitions, scratch=scratch,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=str(case_folds.lock_hash), root=root,
    )
    completed = _execute_or_resume(tasks, workers=int(getattr(config, "runtime")["classifier_workers"]))
    uniform_cells = {cell.key: cell for cell in _cells_from_tasks(tasks, completed)}
    probe_by_key = probe.by_key
    cells: list[CombinedPredictionCell] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for action in library.actions_by_target[target]:
                    key = (target, action.action_id, training_seed, generation_seed)
                    if action.action_id == UNIFORM_ACTION_ID:
                        source = uniform_cells[key]
                        cells.append(source)
                        continue
                    source_action_id = (
                        action.action_id
                        if action.action_id == BASE_ACTION_ID or action.action_id.startswith(H_X_E_ACTION_PREFIX)
                        else h_x_e_action_id(action.selected_source)
                    )
                    source = probe_by_key[(target, source_action_id, training_seed, generation_seed)]
                    cells.append(
                        CombinedPredictionCell(
                            scope_id=target, action_id=action.action_id, action_hash=action.action_hash,
                            training_seed=training_seed, generation_seed=generation_seed,
                            support_row_identity_hash=source.support_row_identity_hash,
                            evaluation_row_identity_hash=source.evaluation_row_identity_hash,
                            support_predictions=source.support_predictions,
                            support_probabilities=source.support_probabilities,
                            evaluation_predictions=source.evaluation_predictions,
                            evaluation_probabilities=source.evaluation_probabilities,
                            composition_hash=source.composition_hash,
                            scaler_state_hash=source.scaler_state_hash,
                            fit_provenance_hash=source.fit_provenance_hash,
                            aliased_from_action_id=(None if source_action_id == action.action_id else source_action_id),
                        )
                    )
    store = build_store(
        role="target_final", cells=cells, source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(case_folds.lock_hash),
        action_library_hash=library.action_library_hash,
        expected_cell_count=EXPECTED_FINAL_CELL_COUNT,
        unique_classifier_fit_count=EXPECTED_TARGET_UNIQUE_FIT_COUNT,
    )
    _validate_final(store, library, source_cache_lock_hash, str(case_folds.lock_hash))
    write_combined_store(array_path, index_path, store)
    verified = read_combined_store(array_path, index_path)
    _validate_final(verified, library, source_cache_lock_hash, str(case_folds.lock_hash))
    _write_target_index_table(root, verified)
    _remove_target_working_files(root)
    return verified


def _write_target_scratch(root: Path, *, frame: object, partitions: object) -> Mapping[str, object]:
    checkpoint_root = root / TARGET_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows: list[object] = []
    centers: dict[str, object] = {}
    cursor = 0
    for target in CENTERS:
        support = tuple(partitions.support_rows_by_center[target])
        evaluation = tuple(partitions.evaluation_rows_by_center[target])
        if len({row.case_id for row in support}) != 2 or {row.case_id for row in support} & {row.case_id for row in evaluation}:
            raise ProtocolError("Target probe support/evaluation partition drifted.")
        rows.extend((*support, *evaluation))
        centers[target] = {
            "support_start": cursor,
            "support_stop": cursor + len(support),
            "evaluation_start": cursor + len(support),
            "evaluation_stop": cursor + len(support) + len(evaluation),
            "support_row_count": len(support), "evaluation_row_count": len(evaluation),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
        }
        cursor += len(support) + len(evaluation)
    embeddings = np.ascontiguousarray(frame.embeddings_for(rows), dtype=np.float32)
    array_path = checkpoint_root / "combined_validation_embeddings.npy"
    atomic_npy(array_path, embeddings)
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_target_scratch_v1",
        "array_path": str(array_path.resolve()), "array_sha256": array_sha256(embeddings),
        "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
        "partition_lock_hash": str(partitions.lock_hash), "centers": centers,
        "support_labels_stored": False, "evaluation_labels_stored": False,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(checkpoint_root / "combined_validation_index.json", payload)
    return payload


def _build_probe_tasks(
    config: object, source_cache: SourceCache, partitions: object, *, scratch: Mapping[str, object],
    source_cache_lock_hash: str, root: Path,
) -> tuple[dict[str, object], ...]:
    actions_by_target = {target: _probe_actions(target) for target in CENTERS}
    return _build_fit_tasks(
        config, source_cache, partitions, scratch=scratch,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(partitions.lock_hash), root=root,
        task_role="target_probe", actions_by_target=actions_by_target,
    )


def _build_uniform_tasks(
    config: object, source_cache: SourceCache, library: FrozenEnsembleEndpointActionLibrary,
    partitions: object, *, scratch: Mapping[str, object], source_cache_lock_hash: str,
    case_fold_lock_hash: str, root: Path,
) -> tuple[dict[str, object], ...]:
    actions = {
        target: (_action_execution_payload(library.action(target, UNIFORM_ACTION_ID)),)
        for target in CENTERS
    }
    return _build_fit_tasks(
        config, source_cache, partitions, scratch=scratch,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=case_fold_lock_hash, root=root,
        task_role="target_uniform", actions_by_target=actions,
    )


def _build_fit_tasks(
    config: object, source_cache: SourceCache, partitions: object, *, scratch: Mapping[str, object],
    source_cache_lock_hash: str, partition_lock_hash: str, root: Path,
    task_role: str, actions_by_target: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    centers = scratch.get("centers")
    if not isinstance(centers, Mapping):
        raise ProtocolError("Target scratch center coverage is absent.")
    checkpoint_root = root / TARGET_CHECKPOINT_DIRECTORY / task_role
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    source_rows = [record.to_row() for record in source_cache.source_records]
    tasks: list[dict[str, object]] = []
    for target in CENTERS:
        offset = centers[target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = f"{task_role}_H{target}_train{training_seed}_gen{generation_seed}"
                task: dict[str, object] = {
                    "schema_version": "midogpp_stage90_ensemble_endpoint_target_task_v1",
                    "task_id": task_id, "task_role": task_role,
                    "config_contract_hash": str(getattr(config, "contract_hash")),
                    "source_cache_lock_hash": source_cache_lock_hash,
                    "partition_lock_hash": partition_lock_hash,
                    "scope_id": target, "target_center": target,
                    "training_seed": training_seed, "generation_seed": generation_seed,
                    "candidate_sources": list(candidate_sources(target)),
                    "source_array_path": str(source_cache.source_array_path.resolve()),
                    "source_index_rows": source_rows,
                    "combined_array_path": str(scratch["array_path"]),
                    "combined_array_sha256": str(scratch["array_sha256"]),
                    **dict(offset), "actions": [dict(action) for action in actions_by_target[target]],
                    "classifier": getattr(config, "classifier").to_payload(),
                    "threads_per_fit": int(getattr(config, "runtime")["classifier_threads_per_worker"]),
                    "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
                    "checkpoint_npz_path": str(checkpoint_root / f"{task_id}.npz"),
                    "labels_available": False,
                }
                task["task_hash"] = stable_hash({
                    key: value for key, value in task.items()
                    if key not in {"checkpoint_json_path", "checkpoint_npz_path"}
                })
                tasks.append(task)
    if len(tasks) != 81:
        raise ProtocolError("Target fit task count drifted.")
    return tuple(tasks)


def target_prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    if task.get("labels_available") is not False or task.get("task_role") not in {"target_probe", "target_uniform"}:
        raise ProtocolError("Target fit task escaped its label-free boundary.")
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r", allow_pickle=False)
    if source_array.shape != (81, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or source_array.dtype != np.float32:
        raise ProtocolError("Target source cache geometry drifted.")
    source_index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in task["source_index_rows"]
    }
    training_seed, generation_seed = int(task["training_seed"]), int(task["generation_seed"])
    blocks: dict[str, np.ndarray] = {}
    stream_ids: dict[str, str] = {}
    for source in task["candidate_sources"]:
        record = source_index[(str(source), training_seed, generation_seed)]
        blocks[str(source)] = source_array[int(record["block_ordinal"])]
        stream_ids[str(source)] = str(record["stream_id"])
    combined_all = np.load(Path(str(task["combined_array_path"])), mmap_mode="r", allow_pickle=False)
    s0, s1 = int(task["support_start"]), int(task["support_stop"])
    e0, e1 = int(task["evaluation_start"]), int(task["evaluation_stop"])
    combined = np.ascontiguousarray(np.concatenate((combined_all[s0:s1], combined_all[e0:e1])), dtype=np.float32)
    support_count = s1 - s0
    spec = classifier_from_payload(task["classifier"])
    support_predictions: list[np.ndarray] = []
    support_probabilities: list[np.ndarray] = []
    evaluation_predictions: list[np.ndarray] = []
    evaluation_probabilities: list[np.ndarray] = []
    action_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Target fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for action in task["actions"]:
            counts = _counts(action, tuple(task["candidate_sources"]))
            train_x, train_y = _compose(blocks, counts)
            result = fit_logistic_classifier(train_x, train_y, combined, spec=spec)
            predictions = np.asarray(result.predictions, dtype=np.uint8)
            probabilities = np.asarray(result.probabilities, dtype=np.float64)
            if (
                tuple(int(value) for value in result.classes) != (0, 1)
                or predictions.shape != (len(combined),)
                or probabilities.shape != (len(combined), 2)
                or not result.converged
                or result.classifier_config_hash != spec.config_hash
            ):
                raise ProtocolError("Target classifier fit drifted.")
            positive = probabilities[:, 1].astype(np.float32, copy=False)
            sp = np.ascontiguousarray(predictions[:support_count], dtype=np.uint8)
            sq = np.ascontiguousarray(positive[:support_count], dtype=np.float32)
            ep = np.ascontiguousarray(predictions[support_count:], dtype=np.uint8)
            eq = np.ascontiguousarray(positive[support_count:], dtype=np.float32)
            composition_hash = stable_hash({
                "schema_version": "midogpp_stage90_ensemble_endpoint_target_composition_v1",
                "counts_by_class": counts, "source_stream_ids": stream_ids,
            })
            fit_hash = stable_hash({
                "task_hash": task["task_hash"], "action_id": action["action_id"],
                "composition_hash": composition_hash, "scaler_state_hash": str(result.scaler_state_hash),
                "support_probability_sha256": array_sha256(sq),
                "evaluation_probability_sha256": array_sha256(eq),
            })
            support_predictions.append(sp); support_probabilities.append(sq)
            evaluation_predictions.append(ep); evaluation_probabilities.append(eq)
            action_rows.append({
                "action_id": str(action["action_id"]), "action_hash": str(action["action_hash"]),
                "composition_hash": composition_hash, "scaler_state_hash": str(result.scaler_state_hash),
                "fit_provenance_hash": fit_hash, "aliased_from_action_id": None,
                "support_prediction_sha256": array_sha256(sp), "support_probability_sha256": array_sha256(sq),
                "evaluation_prediction_sha256": array_sha256(ep), "evaluation_probability_sha256": array_sha256(eq),
            })
    return write_task_checkpoint(
        task, support_predictions=np.stack(support_predictions).astype(np.uint8, copy=False),
        support_probabilities=np.stack(support_probabilities).astype(np.float32, copy=False),
        evaluation_predictions=np.stack(evaluation_predictions).astype(np.uint8, copy=False),
        evaluation_probabilities=np.stack(evaluation_probabilities).astype(np.float32, copy=False),
        action_rows=action_rows,
    )


def _probe_actions(target: str) -> tuple[dict[str, object], ...]:
    sources = candidate_sources(target)
    actions: list[dict[str, object]] = []
    for action_id in (BASE_ACTION_ID, *(h_x_e_action_id(source) for source in sources)):
        selected = None if action_id == BASE_ACTION_ID else action_id.removeprefix(H_X_E_ACTION_PREFIX)
        counts = {
            str(label): {source: 128 + (128 if source == selected else 0) for source in sources}
            for label in (0, 1)
        }
        unhashed = {
            "schema_version": "midogpp_stage90_ensemble_endpoint_target_probe_action_v1",
            "target_center": target, "action_id": action_id, "selected_source": selected,
            "final_counts_by_class": counts, "label_free_preplan_probe": True,
        }
        actions.append({**unhashed, "action_hash": stable_hash(unhashed)})
    return tuple(actions)


def _action_execution_payload(action: object) -> dict[str, object]:
    return {
        "action_id": str(action.action_id), "action_hash": str(action.action_hash),
        "final_counts_by_class": {
            str(label): dict(action.final_counts_by_class[label]) for label in (0, 1)
        },
    }


def _counts(action: Mapping[str, object], candidates: tuple[str, ...]) -> dict[str, dict[str, int]]:
    raw = action.get("final_counts_by_class")
    if not isinstance(raw, Mapping):
        raise ProtocolError("Target action counts are absent.")
    output: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        values = raw.get(str(label), raw.get(label))
        if not isinstance(values, Mapping):
            raise ProtocolError("Target action class counts drifted.")
        counts = {str(source): int(count) for source, count in values.items()}
        if tuple(counts) != candidates or sum(counts.values()) not in {1024, 1152}:
            raise ProtocolError("Target action composition geometry drifted.")
        output[str(label)] = counts
    return output


def _compose(blocks: Mapping[str, np.ndarray], counts: Mapping[str, Mapping[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        for source, count in counts[str(label)].items():
            if source not in blocks or not 0 < count <= SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Target composition source capacity drifted.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(np.asarray(blocks[source][start:start + count], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if embeddings.ndim != 2 or embeddings.shape[1] != COMMON_OUTPUT_DIM:
        raise ProtocolError("Target composed embeddings drifted.")
    return embeddings, truth


def _execute_or_resume(tasks: Sequence[Mapping[str, object]], *, workers: int) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Target execution requires four classifier workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_task_checkpoint(task)
        if loaded is None: pending.append(task)
        else: completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(target_prediction_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]; future.result()
                loaded = load_task_checkpoint(task)
                if loaded is None: raise ProtocolError("Target worker returned without checkpoint.")
                completed[str(task["task_id"])] = loaded
                print(f"[utility-ensemble-endpoint] {task['task_role']} tasks {len(completed)}/{len(tasks)}", flush=True)
    if len(completed) != len(tasks):
        raise ProtocolError("Target checkpoint coverage is incomplete.")
    return completed


def _cells_from_tasks(
    tasks: Sequence[Mapping[str, object]], completed: Mapping[str, Mapping[str, object]]
) -> list[CombinedPredictionCell]:
    cells: list[CombinedPredictionCell] = []
    for task in tasks:
        result = completed[str(task["task_id"])]
        for ordinal, action in enumerate(result["actions"]):
            cells.append(CombinedPredictionCell(
                scope_id=str(task["scope_id"]), action_id=str(action["action_id"]), action_hash=str(action["action_hash"]),
                training_seed=int(task["training_seed"]), generation_seed=int(task["generation_seed"]),
                support_row_identity_hash=str(task["support_row_identity_hash"]),
                evaluation_row_identity_hash=str(task["evaluation_row_identity_hash"]),
                support_predictions=np.ascontiguousarray(result["support_predictions"][ordinal], dtype=np.uint8),
                support_probabilities=np.ascontiguousarray(result["support_probabilities"][ordinal], dtype=np.float32),
                evaluation_predictions=np.ascontiguousarray(result["evaluation_predictions"][ordinal], dtype=np.uint8),
                evaluation_probabilities=np.ascontiguousarray(result["evaluation_probabilities"][ordinal], dtype=np.float32),
                composition_hash=str(action["composition_hash"]), scaler_state_hash=str(action["scaler_state_hash"]),
                fit_provenance_hash=str(action["fit_provenance_hash"]), aliased_from_action_id=None,
            ))
    return cells


def _probe_library_hash() -> str:
    return stable_hash({target: _probe_actions(target) for target in CENTERS})


def _probe_seal_unhashed(store: CombinedPredictionStore, partitions: object) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_target_probe_seal_v1",
        "status": "SEALED_B_PLUS_EIGHT_HXE_BEFORE_TARGET_PLAN",
        "probe_store_hash": store.store_hash,
        "source_cache_lock_hash": store.source_cache_lock_hash,
        "support_partition_lock_hash": str(partitions.lock_hash),
        "probe_action_library_hash": store.action_library_hash,
        "cell_count": len(store.cells), "unique_classifier_fit_count": store.unique_classifier_fit_count,
        "ordered_probability_bindings": [
            {
                "key": list(cell.key),
                "support_prediction_sha256": array_sha256(cell.support_predictions),
                "support_probability_sha256": array_sha256(cell.support_probabilities),
                "evaluation_prediction_sha256": array_sha256(cell.evaluation_predictions),
                "evaluation_probability_sha256": array_sha256(cell.evaluation_probabilities),
                "composition_hash": cell.composition_hash,
                "scaler_state_hash": cell.scaler_state_hash,
                "fit_provenance_hash": cell.fit_provenance_hash,
            } for cell in store.cells
        ],
        "support_vectors_may_feed_target_features": True,
        "evaluation_vectors_may_feed_target_features_or_plan": False,
        "support_labels_opened": False, "evaluation_labels_opened": False,
        "target_plan_built": False,
    }


def _persist_probe_seal(root: Path, *, store: CombinedPredictionStore, partitions: object) -> Mapping[str, object]:
    unhashed = _probe_seal_unhashed(store, partitions)
    payload = {**unhashed, "probe_seal_hash": stable_hash(unhashed)}
    path = root / TARGET_PROBE_SEAL_MEMBER
    if path.is_file() and read_json(path) != payload:
        raise ProtocolError("Persisted target probe seal drifted.")
    if not path.is_file(): atomic_json(path, payload)
    return payload


def _validate_probe_seal(root: Path, *, probe: CombinedPredictionStore, partitions: object) -> Mapping[str, object]:
    expected_unhashed = _probe_seal_unhashed(probe, partitions)
    expected = {**expected_unhashed, "probe_seal_hash": stable_hash(expected_unhashed)}
    observed = read_json(root / TARGET_PROBE_SEAL_MEMBER)
    if observed != expected:
        raise ProtocolError("Target plan requires the durable pre-plan probe seal.")
    return observed


def validate_target_probe_seal(
    root: Path, probe: CombinedPredictionStore, partitions: object
) -> Mapping[str, object]:
    """Public reconstruction gate for every pre-plan probe consumer."""

    _validate_probe(
        probe, probe.source_cache_lock_hash, str(partitions.lock_hash), _probe_library_hash()
    )
    return _validate_probe_seal(root, probe=probe, partitions=partitions)


def _validate_probe(store: CombinedPredictionStore, source_lock: str, partition_lock: str, library_hash: str) -> None:
    if (
        store.role != "target_probe" or len(store.cells) != EXPECTED_PROBE_CELL_COUNT
        or store.unique_classifier_fit_count != EXPECTED_PROBE_CELL_COUNT
        or store.source_cache_lock_hash != source_lock or store.partition_lock_hash != partition_lock
        or store.action_library_hash != library_hash
    ):
        raise ProtocolError("Target probe store binding drifted.")
    for target in CENTERS:
        expected = {BASE_ACTION_ID, *(h_x_e_action_id(source) for source in candidate_sources(target))}
        if {cell.action_id for cell in store.cells if cell.scope_id == target} != expected:
            raise ProtocolError("Target probe B/Hxe coverage drifted.")


def _validate_final(
    store: CombinedPredictionStore, library: FrozenEnsembleEndpointActionLibrary,
    source_lock: str, case_fold_lock: str,
) -> None:
    if (
        store.role != "target_final" or len(store.cells) != EXPECTED_FINAL_CELL_COUNT
        or store.unique_classifier_fit_count != EXPECTED_TARGET_UNIQUE_FIT_COUNT
        or store.source_cache_lock_hash != source_lock or store.partition_lock_hash != case_fold_lock
        or store.action_library_hash != library.action_library_hash
    ):
        raise ProtocolError("Final target store binding drifted.")
    expected_keys = tuple(
        (target, action_id, training_seed, generation_seed)
        for target in CENTERS for training_seed in TRAINING_SEEDS for generation_seed in GENERATION_SEEDS
        for action_id in expected_target_action_ids(target)
    )
    if tuple(cell.key for cell in store.cells) != expected_keys:
        raise ProtocolError("Final target action/seed ordering drifted.")
    for cell in store.cells:
        if cell.action_hash != library.action(cell.scope_id, cell.action_id).action_hash:
            raise ProtocolError("Final target cell action hash drifted.")
        action = library.action(cell.scope_id, cell.action_id)
        if cell.action_id in {BASE_ACTION_ID, UNIFORM_ACTION_ID} or cell.action_id.startswith(H_X_E_ACTION_PREFIX):
            if cell.aliased_from_action_id is not None:
                raise ProtocolError("Directly fitted target cell is unexpectedly aliased.")
            continue
        source_action = h_x_e_action_id(action.selected_source)
        if cell.aliased_from_action_id != source_action:
            raise ProtocolError("Routed target cell alias identity drifted.")
        source = store.by_key[(cell.scope_id, source_action, cell.training_seed, cell.generation_seed)]
        if (
            cell.fit_provenance_hash != source.fit_provenance_hash
            or cell.composition_hash != source.composition_hash
            or array_sha256(cell.support_probabilities) != array_sha256(source.support_probabilities)
            or array_sha256(cell.evaluation_probabilities) != array_sha256(source.evaluation_probabilities)
        ):
            raise ProtocolError("Routed target alias differs from its Hxe fit.")


def _write_target_index_table(root: Path, store: CombinedPredictionStore) -> None:
    rows = [{
        "schema_version": "midogpp_stage90_ensemble_endpoint_target_prediction_cell_v1",
        "cell_ordinal": ordinal, "target_center": cell.scope_id, "action_id": cell.action_id,
        "action_hash": cell.action_hash, "training_seed": cell.training_seed,
        "generation_seed": cell.generation_seed,
        "support_row_identity_hash": cell.support_row_identity_hash,
        "evaluation_row_identity_hash": cell.evaluation_row_identity_hash,
        "support_prediction_sha256": array_sha256(cell.support_predictions),
        "support_probability_sha256": array_sha256(cell.support_probabilities),
        "evaluation_prediction_sha256": array_sha256(cell.evaluation_predictions),
        "evaluation_probability_sha256": array_sha256(cell.evaluation_probabilities),
        "composition_hash": cell.composition_hash, "scaler_state_hash": cell.scaler_state_hash,
        "fit_provenance_hash": cell.fit_provenance_hash,
        "aliased_from_action_id": cell.aliased_from_action_id,
        "labels_available": False,
    } for ordinal, cell in enumerate(store.cells)]
    persist_or_validate_csv(root / TARGET_INDEX_TABLE_MEMBER, rows, TARGET_INDEX_COLUMNS)


def _remove_target_working_files(root: Path) -> None:
    checkpoint = root / TARGET_CHECKPOINT_DIRECTORY
    if checkpoint.exists(): shutil.rmtree(checkpoint)


__all__ = (
    "EXPECTED_FINAL_CELL_COUNT", "EXPECTED_PROBE_CELL_COUNT",
    "EXPECTED_TARGET_UNIQUE_FIT_COUNT", "TARGET_ARRAY_MEMBER", "TARGET_INDEX_MEMBER",
    "TARGET_INDEX_TABLE_MEMBER", "TARGET_PROBE_SEAL_MEMBER",
    "materialize_target_predictions", "materialize_target_probe_predictions",
    "target_prediction_task", "validate_target_probe_seal",
)
