"""Neutral direct-target B plus Hxe probability materialization.

All target embeddings are evaluated for every action and all nine frozen seed
pairs before a label capability can exist.  The store contains probabilities,
row identities, and fit provenance only.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ...real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..generation.contracts import COMMON_OUTPUT_DIM
from ..protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npy, atomic_npz, read_json, sha256_array, sha256_file
from .frozen_source_streams import (
    EXPECTED_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
    FrozenSourceStreamCache,
    source_block_sha256,
)


BASE_ACTION_ID = "B"
H_X_E_ACTION_PREFIX = "Hxe::"
EXPECTED_ACTION_COUNT_PER_TARGET = len(CENTERS)
EXPECTED_CELL_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)

PREDICTION_ARRAY_MEMBER = "arrays/label_free_action_probabilities.npz"
PREDICTION_INDEX_MEMBER = "manifests/label_free_prediction_index.json"
GLOBAL_PREDICTION_SEAL_MEMBER = "manifests/label_free_prediction_seal.json"
CHECKPOINT_DIRECTORY = "checkpoints/label_free_action_predictions"


class LabelFreePredictionConfig(Protocol):
    contract_hash: str
    classifier: object
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class FrozenAction:
    target_center: str
    action_id: str
    selected_source: str | None
    counts_by_class: Mapping[str, Mapping[str, int]]
    action_hash: str

    def __post_init__(self) -> None:
        candidates = tuple(center for center in CENTERS if center != self.target_center)
        selected = self.selected_source
        counts = {
            str(label): MappingProxyType(
                {str(source): int(value) for source, value in self.counts_by_class[str(label)].items()}
            )
            for label in (0, 1)
        }
        expected_id = BASE_ACTION_ID if selected is None else h_x_e_action_id(selected)
        if (
            self.target_center not in CENTERS
            or selected == self.target_center
            or (selected is not None and selected not in candidates)
            or self.action_id != expected_id
            or any(tuple(counts[str(label)]) != candidates for label in (0, 1))
            or any(
                sum(counts[str(label)].values()) != (1024 if selected is None else 1152)
                for label in (0, 1)
            )
            or self.action_hash != stable_hash(self.identity_payload(include_hash=False, counts=counts))
        ):
            raise ProtocolError("Frozen direct-target action drifted.")
        object.__setattr__(self, "counts_by_class", MappingProxyType(counts))

    def identity_payload(
        self,
        *,
        include_hash: bool = True,
        counts: Mapping[str, Mapping[str, int]] | None = None,
    ) -> dict[str, object]:
        payload = {
            "schema_version": "midogpp_direct_target_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "counts_by_class": {
                str(label): dict((counts or self.counts_by_class)[str(label)])
                for label in (0, 1)
            },
            "target_expert_excluded": True,
            "labels_consumed": False,
        }
        if include_hash:
            payload["action_hash"] = self.action_hash
        return payload


@dataclass(frozen=True)
class PredictionCell:
    target_center: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    row_identity_hash: str
    probabilities: np.ndarray
    probability_sha256: str
    predictions_sha256: str
    composition_hash: str
    scaler_state_hash: str
    fit_provenance_hash: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        canonical = {
            action.action_id: action
            for action in build_direct_target_actions(str(self.target_center))
        } if self.target_center in CENTERS else {}
        expected_action = canonical.get(str(self.action_id))
        if (
            self.target_center not in CENTERS
            or expected_action is None
            or self.action_hash != expected_action.action_hash
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or sha256_array(values) != self.probability_sha256
            or sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            != self.predictions_sha256
        ):
            raise ProtocolError("Label-free prediction cell drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return self.target_center, self.action_id, self.training_seed, self.generation_seed


@dataclass(frozen=True)
class LabelFreePredictionStore:
    cells: tuple[PredictionCell, ...]
    rows_by_center: Mapping[str, tuple[str, ...]]
    case_ids_by_center: Mapping[str, tuple[str, ...]]
    source_stream_lock_hash: str
    action_library_hash: str
    target_cache_binding_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        rows = {str(center): tuple(value) for center, value in self.rows_by_center.items()}
        cases = {str(center): tuple(value) for center, value in self.case_ids_by_center.items()}
        expected_keys = tuple(
            (target, action.action_id, training_seed, generation_seed)
            for target in CENTERS
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
            for action in build_direct_target_actions(target)
        )
        if (
            tuple(rows) != CENTERS
            or tuple(cases) != CENTERS
            or any(len(rows[center]) != len(cases[center]) or not rows[center] for center in CENTERS)
            or len(self.cells) != EXPECTED_CELL_COUNT
            or tuple(cell.key for cell in self.cells) != expected_keys
            or any(len(cell.probabilities) != len(rows[cell.target_center]) for cell in self.cells)
            or self.action_library_hash != _canonical_action_library_hash()
            or not str(self.target_cache_binding_hash)
        ):
            raise ProtocolError("Label-free prediction store inventory drifted.")
        object.__setattr__(self, "rows_by_center", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_center", MappingProxyType(cases))

    @cached_property
    def by_key(self) -> Mapping[tuple[str, str, int, int], PredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def probabilities(
        self, target: str, action_id: str, training_seed: int, generation_seed: int
    ) -> np.ndarray:
        try:
            return self.by_key[(str(target), str(action_id), int(training_seed), int(generation_seed))].probabilities
        except KeyError as exc:
            raise ProtocolError("Label-free prediction cell is absent.") from exc

    def exact_nine_mean(self, target: str, action_id: str) -> np.ndarray:
        values = np.stack(
            [self.probabilities(target, action_id, train, generation)
             for train in TRAINING_SEEDS for generation in GENERATION_SEEDS]
        ).astype(np.float64, copy=False)
        return np.mean(values, axis=0, dtype=np.float64)


@dataclass(frozen=True)
class GlobalPredictionSeal:
    store: LabelFreePredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        unhashed = {key: value for key, value in self.seal_payload.items() if key != "global_prediction_seal_hash"}
        if (
            self.seal_payload.get("global_prediction_seal_hash") != _canonical_sha256(unhashed)
            or self.seal_payload.get("status") != "SEALED_ALL_729_LABEL_FREE_ACTION_CELLS"
            or self.seal_payload.get("prediction_store_hash") != self.store.store_hash
            or self.seal_payload.get("source_stream_lock_hash")
            != self.store.source_stream_lock_hash
            or self.seal_payload.get("action_library_hash")
            != self.store.action_library_hash
            or self.seal_payload.get("target_cache_binding_hash")
            != self.store.target_cache_binding_hash
            or self.seal_payload.get("cell_count") != EXPECTED_CELL_COUNT
            or self.seal_payload.get("target_labels_opened") is not False
            or self.seal_payload.get("support_labels_opened") is not False
            or self.seal_payload.get("evaluation_labels_opened") is not False
        ):
            raise ProtocolError("Global prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(dict(self.seal_payload)))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["global_prediction_seal_hash"])


def h_x_e_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("Hxe source center is unknown.")
    return f"{H_X_E_ACTION_PREFIX}{source}"


def build_direct_target_actions(target_center: str) -> tuple[FrozenAction, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Direct-target action target is unknown.")
    candidates = tuple(center for center in CENTERS if center != target)
    actions: list[FrozenAction] = []
    for selected in (None, *candidates):
        action_id = BASE_ACTION_ID if selected is None else h_x_e_action_id(selected)
        counts = {
            str(label): {
                source: 128 + (128 if selected == source else 0)
                for source in candidates
            }
            for label in (0, 1)
        }
        unhashed = {
            "schema_version": "midogpp_direct_target_action_v1",
            "target_center": target,
            "action_id": action_id,
            "selected_source": selected,
            "counts_by_class": counts,
            "target_expert_excluded": True,
            "labels_consumed": False,
        }
        actions.append(
            FrozenAction(
                target_center=target,
                action_id=action_id,
                selected_source=selected,
                counts_by_class=counts,
                action_hash=stable_hash(unhashed),
            )
        )
    return tuple(actions)


def materialize_label_free_action_predictions(
    config: LabelFreePredictionConfig,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    *,
    partition_lock_hash: str,
    root: Path,
) -> GlobalPredictionSeal:
    """Fit 729 direct-target classifier cells and atomically seal all rows."""

    if (
        int(config.runtime.get("classifier_workers", -1)) != 4
        or int(config.runtime.get("classifier_threads_per_worker", -1)) != 3
        or config.runtime.get("multiprocessing_start_method") != "spawn"
        or config.runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or config.runtime.get("scientific_reductions_dtype") != "float64"
        or int(config.runtime.get("target_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(config.runtime.get("target_probability_cell_count", -1))
        != EXPECTED_CELL_COUNT
        or int(config.runtime.get("maximum_total_classifier_fit_count", -1))
        != EXPECTED_CELL_COUNT
    ):
        raise ProtocolError("Label-free prediction execution requires four 3-thread CPU workers.")
    final_array = root / PREDICTION_ARRAY_MEMBER
    final_index = root / PREDICTION_INDEX_MEMBER
    seal_path = root / GLOBAL_PREDICTION_SEAL_MEMBER
    target_cache_binding_hash = _target_cache_binding_hash(frame)
    if final_array.is_file() and final_index.is_file() and seal_path.is_file():
        return load_global_prediction_seal(
            root,
            expected_config_hash=config.contract_hash,
            expected_source_lock_hash=source_cache.lock_hash,
            expected_partition_lock_hash=partition_lock_hash,
            expected_target_cache_binding_hash=target_cache_binding_hash,
        )

    scratch = _write_target_scratch(
        root,
        frame=frame,
        partition_lock_hash=partition_lock_hash,
        target_cache_binding_hash=target_cache_binding_hash,
    )
    _validate_target_scratch(scratch, expected_partition_lock_hash=partition_lock_hash)
    library = {target: build_direct_target_actions(target) for target in CENTERS}
    library_payload = {target: [action.identity_payload() for action in actions] for target, actions in library.items()}
    action_library_hash = stable_hash(library_payload)
    if action_library_hash != _canonical_action_library_hash():
        raise ProtocolError("Direct-target action library identity drifted.")
    tasks = _build_prediction_tasks(
        config,
        source_cache,
        scratch=scratch,
        action_library=library,
        action_library_hash=action_library_hash,
        partition_lock_hash=partition_lock_hash,
        root=root,
    )
    completed = _execute_or_resume(tasks, workers=4)
    cells = _cells_from_tasks(tasks, completed)
    store_hash = _store_hash(
        cells,
        rows_by_center=scratch["row_ids_by_center"],
        case_ids_by_center=scratch["case_ids_by_center"],
        source_stream_lock_hash=source_cache.lock_hash,
        action_library_hash=action_library_hash,
        target_cache_binding_hash=target_cache_binding_hash,
    )
    store = LabelFreePredictionStore(
        cells=tuple(cells),
        rows_by_center={str(k): tuple(str(v) for v in values) for k, values in scratch["row_ids_by_center"].items()},
        case_ids_by_center={str(k): tuple(str(v) for v in values) for k, values in scratch["case_ids_by_center"].items()},
        source_stream_lock_hash=source_cache.lock_hash,
        action_library_hash=action_library_hash,
        target_cache_binding_hash=target_cache_binding_hash,
        store_hash=store_hash,
    )
    _write_store(final_array, final_index, store=store, config_contract_hash=config.contract_hash,
                 partition_lock_hash=partition_lock_hash)
    verified = _read_store(final_array, final_index)
    if verified.store_hash != store.store_hash:
        raise ProtocolError("Persisted label-free prediction store changed bytes.")
    seal_unhashed = {
        "schema_version": "midogpp_global_label_free_prediction_seal_v1",
        "status": "SEALED_ALL_729_LABEL_FREE_ACTION_CELLS",
        "config_contract_hash": config.contract_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "partition_lock_hash": partition_lock_hash,
        "action_library_hash": action_library_hash,
        "target_cache_binding_hash": target_cache_binding_hash,
        "prediction_store_hash": verified.store_hash,
        "prediction_array_sha256": sha256_file(final_array),
        "prediction_index_sha256": sha256_file(final_index),
        "cell_count": len(verified.cells),
        "unique_classifier_fit_count": len(verified.cells),
        "target_count": len(CENTERS),
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "seed_pair_count": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "all_target_rows_predicted": True,
        "exact_nine_seed_averaging_required": True,
        "target_expert_excluded": True,
        "target_labels_opened": False,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "expert_bank_updated": False,
        "shared_representation_updated": False,
    }
    seal = {**seal_unhashed, "global_prediction_seal_hash": _canonical_sha256(seal_unhashed)}
    atomic_json(seal_path, seal)
    shutil.rmtree(root / CHECKPOINT_DIRECTORY, ignore_errors=True)
    return load_global_prediction_seal(
        root,
        expected_config_hash=config.contract_hash,
        expected_source_lock_hash=source_cache.lock_hash,
        expected_partition_lock_hash=partition_lock_hash,
        expected_target_cache_binding_hash=target_cache_binding_hash,
    )


def load_global_prediction_seal(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_lock_hash: str | None = None,
    expected_partition_lock_hash: str | None = None,
    expected_target_cache_binding_hash: str | None = None,
) -> GlobalPredictionSeal:
    array_path = root / PREDICTION_ARRAY_MEMBER
    index_path = root / PREDICTION_INDEX_MEMBER
    seal_path = root / GLOBAL_PREDICTION_SEAL_MEMBER
    store = _read_store(array_path, index_path)
    seal = read_json(seal_path)
    capability = GlobalPredictionSeal(store=store, seal_payload=seal, arrays_path=array_path,
                                      index_path=index_path, seal_path=seal_path)
    if (
        seal.get("prediction_array_sha256") != sha256_file(array_path)
        or seal.get("prediction_index_sha256") != sha256_file(index_path)
        or (expected_config_hash is not None and seal.get("config_contract_hash") != expected_config_hash)
        or (expected_source_lock_hash is not None and seal.get("source_stream_lock_hash") != expected_source_lock_hash)
        or (expected_partition_lock_hash is not None and seal.get("partition_lock_hash") != expected_partition_lock_hash)
        or seal.get("target_cache_binding_hash") != store.target_cache_binding_hash
        or (
            expected_target_cache_binding_hash is not None
            and seal.get("target_cache_binding_hash") != expected_target_cache_binding_hash
        )
    ):
        raise ProtocolError("Global label-free prediction seal binding drifted.")
    return capability


def _write_target_scratch(
    root: Path,
    *,
    frame: object,
    partition_lock_hash: str,
    target_cache_binding_hash: str,
) -> Mapping[str, object]:
    checkpoint_root = root / CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows_by_center = getattr(frame, "rows_by_center", None)
    if not isinstance(rows_by_center, Mapping) or tuple(rows_by_center) != CENTERS:
        raise ProtocolError("Label-free target rows are unavailable.")
    rows: list[object] = []
    offsets: dict[str, object] = {}
    row_ids: dict[str, list[str]] = {}
    case_ids: dict[str, list[str]] = {}
    cursor = 0
    for target in CENTERS:
        target_rows = tuple(rows_by_center[target])
        if not target_rows:
            raise ProtocolError("Label-free target center has no rows.")
        target_row_ids = [_row_id(row) for row in target_rows]
        target_case_ids = [str(getattr(row, "case_id")) for row in target_rows]
        if len(set(target_row_ids)) != len(target_row_ids):
            raise ProtocolError("Label-free target row identities are duplicated.")
        rows.extend(target_rows)
        offsets[target] = {
            "start": cursor,
            "stop": cursor + len(target_rows),
            "row_count": len(target_rows),
            "row_identity_hash": stable_hash(
                [{"row_id": row_id, "case_id": case_id, "target_center": target}
                 for row_id, case_id in zip(target_row_ids, target_case_ids, strict=True)]
            ),
            "embedding_slice_sha256": "PENDING",
        }
        row_ids[target] = target_row_ids
        case_ids[target] = target_case_ids
        cursor += len(target_rows)
    embeddings = np.ascontiguousarray(getattr(frame, "embeddings_for")(rows), dtype=np.float32)
    if embeddings.shape != (cursor, COMMON_OUTPUT_DIM) or not np.isfinite(embeddings).all():
        raise ProtocolError("Label-free target embedding geometry drifted.")
    for target in CENTERS:
        offset = offsets[target]
        if not isinstance(offset, dict):
            raise ProtocolError("Label-free target scratch offset is malformed.")
        offset["embedding_slice_sha256"] = sha256_array(
            embeddings[int(offset["start"]) : int(offset["stop"])]
        )
    path = checkpoint_root / "target_embeddings.npy"
    atomic_npy(path, embeddings)
    unhashed = {
        "schema_version": "midogpp_label_free_target_scratch_v1",
        "array_path": str(path.resolve()),
        "array_sha256": sha256_array(embeddings),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "partition_lock_hash": partition_lock_hash,
        "target_cache_binding_hash": target_cache_binding_hash,
        "offsets": offsets,
        "row_ids_by_center": row_ids,
        "case_ids_by_center": case_ids,
        "labels_stored": False,
        "manifest_opened": False,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(checkpoint_root / "target_scratch.json", payload)
    return payload


def _build_prediction_tasks(
    config: LabelFreePredictionConfig,
    source_cache: FrozenSourceStreamCache,
    *,
    scratch: Mapping[str, object],
    action_library: Mapping[str, Sequence[FrozenAction]],
    action_library_hash: str,
    partition_lock_hash: str,
    root: Path,
) -> tuple[Mapping[str, object], ...]:
    _validate_target_scratch(scratch, expected_partition_lock_hash=partition_lock_hash)
    offsets = scratch.get("offsets")
    if not isinstance(offsets, Mapping):
        raise ProtocolError("Label-free target scratch offsets are absent.")
    source_rows = [record.to_payload() for record in source_cache.records]
    checkpoint_root = root / CHECKPOINT_DIRECTORY / "tasks"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    classifier = getattr(config, "classifier")
    classifier_payload = classifier.to_payload() if hasattr(classifier, "to_payload") else dict(classifier)
    tasks: list[Mapping[str, object]] = []
    for target, training_seed, generation_seed in product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS):
        task_id = f"target_{target}_train_{training_seed}_generation_{generation_seed}"
        offset = offsets[target]
        if not isinstance(offset, Mapping):
            raise ProtocolError("Label-free target offset is malformed.")
        task_unhashed = {
            "schema_version": "midogpp_label_free_action_prediction_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "source_stream_lock_hash": source_cache.lock_hash,
            "partition_lock_hash": partition_lock_hash,
            "action_library_hash": action_library_hash,
            "target_center": target,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "candidate_sources": [center for center in CENTERS if center != target],
            "source_array_path": str(source_cache.source_array_path.resolve()),
            "source_array_sha256": str(source_cache.lock_payload["source_array_sha256"]),
            "source_stream_index_hash": str(source_cache.lock_payload["source_stream_index_hash"]),
            "source_index_rows_hash": stable_hash(source_rows),
            "source_index_rows": source_rows,
            "target_array_path": str(scratch["array_path"]),
            "target_array_sha256": str(scratch["array_sha256"]),
            "target_array_shape": list(scratch["shape"]),
            "target_array_dtype": str(scratch["dtype"]),
            "target_scratch_hash": str(scratch["scratch_hash"]),
            "target_cache_binding_hash": str(scratch["target_cache_binding_hash"]),
            "target_start": int(offset["start"]),
            "target_stop": int(offset["stop"]),
            "target_row_identity_hash": str(offset["row_identity_hash"]),
            "target_slice_sha256": str(offset["embedding_slice_sha256"]),
            "actions": [action.identity_payload() for action in action_library[target]],
            "classifier": classifier_payload,
            "threads_per_fit": int(config.runtime["classifier_threads_per_worker"]),
            "labels_available": False,
            "target_expert_available": False,
        }
        task_hash = stable_hash(task_unhashed)
        tasks.append(
            {
                **task_unhashed,
                "task_hash": task_hash,
                "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
                "checkpoint_npz_path": str(checkpoint_root / f"{task_id}.npz"),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ProtocolError("Label-free target task coverage drifted.")
    return tuple(tasks)


def _execute_or_resume(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Label-free target prediction requires four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = _load_prediction_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(_prediction_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = _load_prediction_checkpoint(task)
                if loaded is None:
                    raise ProtocolError("Prediction worker returned without a checkpoint.")
                completed[str(task["task_id"])] = loaded
                print(f"[label-aware-oof] prediction tasks {len(completed)}/{len(tasks)}", flush=True)
    if len(completed) != len(tasks):
        raise ProtocolError("Label-free prediction checkpoint coverage is incomplete.")
    return completed


def _load_validated_prediction_inputs(
    task: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load only task-used bytes and validate them against frozen identities."""

    unhashed = {
        str(key): value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    target = str(task.get("target_center", ""))
    training_seed = int(task.get("training_seed", -1))
    generation_seed = int(task.get("generation_seed", -1))
    candidates = tuple(str(value) for value in task.get("candidate_sources", ()))
    expected_actions = [
        action.identity_payload() for action in build_direct_target_actions(target)
    ] if target in CENTERS else []
    if (
        task.get("task_hash") != stable_hash(unhashed)
        or task.get("task_id")
        != f"target_{target}_train_{training_seed}_generation_{generation_seed}"
        or target not in CENTERS
        or training_seed not in TRAINING_SEEDS
        or generation_seed not in GENERATION_SEEDS
        or candidates != tuple(center for center in CENTERS if center != target)
        or task.get("actions") != expected_actions
        or task.get("action_library_hash") != _canonical_action_library_hash()
        or int(task.get("threads_per_fit", -1)) != 3
        or task.get("labels_available") is not False
        or task.get("target_expert_available") is not False
        or not str(task.get("source_stream_lock_hash", ""))
        or not str(task.get("source_stream_index_hash", ""))
        or not _is_sha256(task.get("source_array_sha256"))
        or not _is_sha256(task.get("target_array_sha256"))
        or not _is_sha256(task.get("target_slice_sha256"))
        or not str(task.get("target_scratch_hash", ""))
        or not str(task.get("target_cache_binding_hash", ""))
    ):
        raise ProtocolError("Prediction task identity/action boundary drifted.")

    raw_index = task.get("source_index_rows")
    if not isinstance(raw_index, list) or task.get("source_index_rows_hash") != stable_hash(raw_index):
        raise ProtocolError("Prediction task source index identity drifted.")
    expected_source_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed_source_keys: list[tuple[str, int, int]] = []
    source_index: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for ordinal, raw in enumerate(raw_index):
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction task source index row is malformed.")
        key = (
            str(raw.get("source_center", "")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        if (
            int(raw.get("block_ordinal", -1)) != ordinal
            or int(raw.get("rows_per_class", -1)) != SOURCE_ROWS_PER_CLASS
            or int(raw.get("row_count", -1)) != 2 * SOURCE_ROWS_PER_CLASS
            or int(raw.get("feature_dim", -1)) != COMMON_OUTPUT_DIM
            or not _is_sha256(raw.get("output_sha256"))
        ):
            raise ProtocolError("Prediction task source index row drifted.")
        observed_source_keys.append(key)
        source_index[key] = raw
    if tuple(observed_source_keys) != expected_source_keys or len(source_index) != EXPECTED_STREAM_COUNT:
        raise ProtocolError("Prediction task source index coverage/order drifted.")

    source_path = Path(str(task.get("source_array_path", "")))
    try:
        source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Prediction task source cache is unreadable.") from exc
    if (
        source_values.shape
        != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or source_values.dtype != np.float32
    ):
        raise ProtocolError("Prediction task source cache geometry drifted.")
    blocks: dict[str, np.ndarray] = {}
    for source in candidates:
        record = source_index[(source, training_seed, generation_seed)]
        block = source_values[int(record["block_ordinal"])]
        if source_block_sha256(block) != record["output_sha256"]:
            raise ProtocolError("Prediction task source block bytes drifted.")
        blocks[source] = block

    target_path = Path(str(task.get("target_array_path", "")))
    try:
        target_values = np.load(target_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Prediction task target scratch is unreadable.") from exc
    start, stop = int(task.get("target_start", -1)), int(task.get("target_stop", -1))
    if (
        list(target_values.shape) != task.get("target_array_shape")
        or str(target_values.dtype) != task.get("target_array_dtype")
        or target_values.ndim != 2
        or target_values.shape[1] != COMMON_OUTPUT_DIM
        or target_values.dtype != np.float32
        or start < 0
        or stop <= start
        or stop > len(target_values)
    ):
        raise ProtocolError("Prediction task target scratch geometry drifted.")
    evaluation = np.ascontiguousarray(target_values[start:stop], dtype=np.float32)
    if (
        not np.isfinite(evaluation).all()
        or sha256_array(evaluation) != task.get("target_slice_sha256")
    ):
        raise ProtocolError("Prediction task target slice bytes drifted.")
    return blocks, evaluation


def _prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    if task.get("labels_available") is not False or task.get("target_expert_available") is not False:
        raise ProtocolError("Prediction task escaped its label-free target-exclusion boundary.")
    blocks, evaluation = _load_validated_prediction_inputs(task)
    candidates = tuple(str(value) for value in task["candidate_sources"])
    classifier = _classifier_from_payload(task["classifier"])
    probabilities: list[np.ndarray] = []
    action_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Label-free classifier fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for action in task["actions"]:
            train_x, train_y, composition_hash = _compose_action(blocks, action, candidates)
            fitted = fit_logistic_classifier(train_x, train_y, evaluation, spec=classifier)
            prediction = np.asarray(fitted.predictions, dtype=np.uint8)
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or prediction.shape != (len(evaluation),)
                or matrix.shape != (len(evaluation), 2)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != classifier.config_hash
            ):
                raise ProtocolError("Label-free classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
            derived_prediction = np.ascontiguousarray(
                positive >= np.float32(0.5), dtype=np.uint8
            )
            probability_hash = sha256_array(positive)
            prediction_hash = sha256_array(derived_prediction)
            fit_hash = stable_hash(
                {
                    "task_hash": task["task_hash"],
                    "action_id": action["action_id"],
                    "action_hash": action["action_hash"],
                    "composition_hash": composition_hash,
                    "scaler_state_hash": fitted.scaler_state_hash,
                    "probability_sha256": probability_hash,
                    "prediction_sha256": prediction_hash,
                }
            )
            probabilities.append(positive)
            action_rows.append(
                {
                    "action_id": str(action["action_id"]),
                    "action_hash": str(action["action_hash"]),
                    "probability_sha256": probability_hash,
                    "prediction_sha256": prediction_hash,
                    "composition_hash": composition_hash,
                    "scaler_state_hash": str(fitted.scaler_state_hash),
                    "fit_provenance_hash": fit_hash,
                }
            )
    arrays = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    atomic_npz(npz_path, probabilities=arrays)
    unhashed = {
        "schema_version": "midogpp_label_free_action_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "shape": list(arrays.shape),
        "dtype": str(arrays.dtype),
        "actions": action_rows,
        "labels_consumed": False,
        "target_expert_used": False,
        "shared_representation_updated": False,
    }
    payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    atomic_json(Path(str(task["checkpoint_json_path"])), payload)
    return payload


def _load_prediction_checkpoint(task: Mapping[str, object]) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.is_file() and not npz_path.is_file():
        return None
    # Rebind every existing resumable output to the current immutable inputs
    # before reading or trusting checkpoint content.
    _load_validated_prediction_inputs(task)
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("Partial label-free prediction checkpoint cannot be repaired silently.")
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    actions = payload.get("actions")
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"probabilities"}:
            raise ProtocolError("Prediction checkpoint archive inventory drifted.")
        probabilities = np.asarray(loaded["probabilities"])
    expected_rows = int(task["target_stop"]) - int(task["target_start"])
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version") != "midogpp_label_free_action_prediction_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("task_id") != task["task_id"]
        or payload.get("task_hash") != task["task_hash"]
        or Path(str(payload.get("npz_path"))) != npz_path
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or probabilities.shape != (EXPECTED_ACTION_COUNT_PER_TARGET, expected_rows)
        or probabilities.dtype != np.float32
        or not isinstance(actions, list)
        or len(actions) != EXPECTED_ACTION_COUNT_PER_TARGET
        or payload.get("labels_consumed") is not False
        or payload.get("target_expert_used") is not False
        or payload.get("shared_representation_updated") is not False
    ):
        raise ProtocolError("Label-free prediction checkpoint failed validation.")
    for ordinal, (record, action) in enumerate(zip(actions, task["actions"], strict=True)):
        if (
            not isinstance(record, Mapping)
            or record.get("action_id") != action["action_id"]
            or record.get("action_hash") != action["action_hash"]
            or record.get("probability_sha256") != sha256_array(probabilities[ordinal])
            or record.get("prediction_sha256")
            != sha256_array((probabilities[ordinal] >= np.float32(0.5)).astype(np.uint8))
        ):
            raise ProtocolError("Label-free prediction checkpoint action drifted.")
    return {**payload, "probabilities": probabilities}


def _cells_from_tasks(
    tasks: Sequence[Mapping[str, object]], completed: Mapping[str, Mapping[str, object]]
) -> list[PredictionCell]:
    cells: list[PredictionCell] = []
    for task in tasks:
        result = completed[str(task["task_id"])]
        probabilities = result["probabilities"]
        for ordinal, action in enumerate(result["actions"]):
            cells.append(
                PredictionCell(
                    target_center=str(task["target_center"]),
                    action_id=str(action["action_id"]),
                    action_hash=str(action["action_hash"]),
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    row_identity_hash=str(task["target_row_identity_hash"]),
                    probabilities=np.ascontiguousarray(probabilities[ordinal], dtype=np.float32),
                    probability_sha256=str(action["probability_sha256"]),
                    predictions_sha256=str(action["prediction_sha256"]),
                    composition_hash=str(action["composition_hash"]),
                    scaler_state_hash=str(action["scaler_state_hash"]),
                    fit_provenance_hash=str(action["fit_provenance_hash"]),
                )
            )
    return cells


def _write_store(
    array_path: Path,
    index_path: Path,
    *,
    store: LabelFreePredictionStore,
    config_contract_hash: str,
    partition_lock_hash: str,
) -> None:
    arrays = {f"cell_{ordinal:04d}": cell.probabilities for ordinal, cell in enumerate(store.cells)}
    atomic_npz(array_path, **arrays)
    cell_rows = []
    for ordinal, cell in enumerate(store.cells):
        cell_rows.append(
            {
                "cell_ordinal": ordinal,
                "array_key": f"cell_{ordinal:04d}",
                "target_center": cell.target_center,
                "action_id": cell.action_id,
                "action_hash": cell.action_hash,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "row_identity_hash": cell.row_identity_hash,
                "row_count": len(cell.probabilities),
                "probability_sha256": cell.probability_sha256,
                "predictions_sha256": cell.predictions_sha256,
                "composition_hash": cell.composition_hash,
                "scaler_state_hash": cell.scaler_state_hash,
                "fit_provenance_hash": cell.fit_provenance_hash,
            }
        )
    payload = {
        "schema_version": "midogpp_label_free_prediction_index_v1",
        "config_contract_hash": config_contract_hash,
        "partition_lock_hash": partition_lock_hash,
        "source_stream_lock_hash": store.source_stream_lock_hash,
        "action_library_hash": store.action_library_hash,
        "target_cache_binding_hash": store.target_cache_binding_hash,
        "prediction_store_hash": store.store_hash,
        "prediction_array_sha256": sha256_file(array_path),
        "rows_by_center": {center: list(rows) for center, rows in store.rows_by_center.items()},
        "case_ids_by_center": {center: list(rows) for center, rows in store.case_ids_by_center.items()},
        "cells": cell_rows,
        "cell_count": len(cell_rows),
        "labels_consumed": False,
        "float32_store": True,
        "float64_reductions_required": True,
    }
    atomic_json(index_path, payload)


def _read_store(array_path: Path, index_path: Path) -> LabelFreePredictionStore:
    index = read_json(index_path)
    raw_cells = index.get("cells")
    rows = index.get("rows_by_center")
    cases = index.get("case_ids_by_center")
    if not isinstance(raw_cells, list) or not isinstance(rows, Mapping) or not isinstance(cases, Mapping):
        raise ProtocolError("Label-free prediction index is malformed.")
    if (
        index.get("schema_version") != "midogpp_label_free_prediction_index_v1"
        or index.get("prediction_array_sha256") != sha256_file(array_path)
        or index.get("action_library_hash") != _canonical_action_library_hash()
        or index.get("labels_consumed") is not False
        or index.get("float32_store") is not True
        or index.get("float64_reductions_required") is not True
        or not str(index.get("target_cache_binding_hash", ""))
    ):
        raise ProtocolError("Label-free prediction array hash drifted.")
    cells: list[PredictionCell] = []
    with np.load(array_path, allow_pickle=False) as loaded:
        if loaded.files != [f"cell_{ordinal:04d}" for ordinal in range(len(raw_cells))]:
            raise ProtocolError("Label-free prediction archive inventory drifted.")
        for ordinal, raw in enumerate(raw_cells):
            if (
                not isinstance(raw, Mapping)
                or int(raw.get("cell_ordinal", -1)) != ordinal
                or raw.get("array_key") != f"cell_{ordinal:04d}"
            ):
                raise ProtocolError("Label-free prediction index cell drifted.")
            values = np.ascontiguousarray(loaded[str(raw["array_key"])], dtype=np.float32)
            if raw.get("predictions_sha256") != sha256_array(
                (values >= np.float32(0.5)).astype(np.uint8)
            ):
                raise ProtocolError("Label-free derived prediction hash drifted.")
            cells.append(
                PredictionCell(
                    target_center=str(raw["target_center"]),
                    action_id=str(raw["action_id"]),
                    action_hash=str(raw["action_hash"]),
                    training_seed=int(raw["training_seed"]),
                    generation_seed=int(raw["generation_seed"]),
                    row_identity_hash=str(raw["row_identity_hash"]),
                    probabilities=values,
                    probability_sha256=str(raw["probability_sha256"]),
                    predictions_sha256=str(raw["predictions_sha256"]),
                    composition_hash=str(raw["composition_hash"]),
                    scaler_state_hash=str(raw["scaler_state_hash"]),
                    fit_provenance_hash=str(raw["fit_provenance_hash"]),
                )
            )
    store = LabelFreePredictionStore(
        cells=tuple(cells),
        rows_by_center={str(center): tuple(str(value) for value in values) for center, values in rows.items()},
        case_ids_by_center={str(center): tuple(str(value) for value in values) for center, values in cases.items()},
        source_stream_lock_hash=str(index["source_stream_lock_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        target_cache_binding_hash=str(index["target_cache_binding_hash"]),
        store_hash=str(index["prediction_store_hash"]),
    )
    expected_hash = _store_hash(
        store.cells,
        rows_by_center=store.rows_by_center,
        case_ids_by_center=store.case_ids_by_center,
        source_stream_lock_hash=store.source_stream_lock_hash,
        action_library_hash=store.action_library_hash,
        target_cache_binding_hash=store.target_cache_binding_hash,
    )
    if store.store_hash != expected_hash or index.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ProtocolError("Label-free prediction store hash drifted.")
    return store


def _store_hash(
    cells: Sequence[PredictionCell],
    *,
    rows_by_center: Mapping[str, Sequence[str]],
    case_ids_by_center: Mapping[str, Sequence[str]],
    source_stream_lock_hash: str,
    action_library_hash: str,
    target_cache_binding_hash: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "midogpp_label_free_prediction_store_v1",
            "source_stream_lock_hash": source_stream_lock_hash,
            "action_library_hash": action_library_hash,
            "target_cache_binding_hash": target_cache_binding_hash,
            "rows_by_center": {center: list(rows_by_center[center]) for center in CENTERS},
            "case_ids_by_center": {center: list(case_ids_by_center[center]) for center in CENTERS},
            "cells": [
                {
                    "key": list(cell.key),
                    "action_hash": cell.action_hash,
                    "row_identity_hash": cell.row_identity_hash,
                    "probability_sha256": cell.probability_sha256,
                    "predictions_sha256": cell.predictions_sha256,
                    "composition_hash": cell.composition_hash,
                    "scaler_state_hash": cell.scaler_state_hash,
                    "fit_provenance_hash": cell.fit_provenance_hash,
                }
                for cell in cells
            ],
            "labels_consumed": False,
        }
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_action_library_hash() -> str:
    return stable_hash(
        {
            target: [action.identity_payload() for action in build_direct_target_actions(target)]
            for target in CENTERS
        }
    )


def _target_cache_binding_hash(frame: object) -> str:
    value = getattr(frame, "cache_binding_hash", None)
    if value is None or not str(value):
        raise ProtocolError("Label-free target cache binding is absent.")
    return str(value)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def _compose_action(
    blocks: Mapping[str, np.ndarray],
    action: Mapping[str, object],
    candidates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_counts = action.get("counts_by_class")
    if not isinstance(raw_counts, Mapping):
        raise ProtocolError("Direct-target action counts are absent.")
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    canonical: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        raw = raw_counts.get(str(label))
        if not isinstance(raw, Mapping):
            raise ProtocolError("Direct-target class counts drifted.")
        counts = {str(source): int(value) for source, value in raw.items()}
        if tuple(counts) != candidates:
            raise ProtocolError("Direct-target action source order drifted.")
        canonical[str(label)] = counts
        for source, count in counts.items():
            if count not in {128, 256} or count > SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Direct-target source prefix capacity drifted.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(np.asarray(blocks[source][start : start + count], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if embeddings.ndim != 2 or embeddings.shape[1] != COMMON_OUTPUT_DIM or not np.isfinite(embeddings).all():
        raise ProtocolError("Direct-target composed embeddings drifted.")
    return embeddings, truth, stable_hash({"counts_by_class": canonical, "action_hash": action["action_hash"]})


def _classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
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
        raise ProtocolError("Classifier payload is malformed.") from exc


def _row_id(row: object) -> str:
    value = getattr(row, "evaluation_row_id", getattr(row, "sample_id", None))
    if value is None or not str(value):
        raise ProtocolError("Label-free target row lacks an opaque identity.")
    return str(value)


def _validate_target_scratch(
    payload: Mapping[str, object], *, expected_partition_lock_hash: str
) -> None:
    unhashed = {key: value for key, value in payload.items() if key != "scratch_hash"}
    path = Path(str(payload.get("array_path", "")))
    if not path.is_file():
        raise ProtocolError("Label-free target scratch array is absent.")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    offsets = payload.get("offsets")
    if (
        payload.get("scratch_hash") != stable_hash(unhashed)
        or payload.get("partition_lock_hash") != expected_partition_lock_hash
        or payload.get("shape") != list(values.shape)
        or payload.get("dtype") != str(values.dtype)
        or payload.get("array_sha256") != sha256_array(values)
        or values.ndim != 2
        or values.shape[1] != COMMON_OUTPUT_DIM
        or values.dtype != np.float32
        or not str(payload.get("target_cache_binding_hash", ""))
        or not isinstance(offsets, Mapping)
        or tuple(offsets) != CENTERS
        or payload.get("labels_stored") is not False
        or payload.get("manifest_opened") is not False
    ):
        raise ProtocolError("Label-free target scratch failed validation.")
    cursor = 0
    for target in CENTERS:
        offset = offsets[target]
        if not isinstance(offset, Mapping):
            raise ProtocolError("Label-free target scratch offset drifted.")
        start, stop = int(offset.get("start", -1)), int(offset.get("stop", -1))
        if (
            start != cursor
            or stop <= start
            or stop > len(values)
            or int(offset.get("row_count", -1)) != stop - start
            or not str(offset.get("row_identity_hash", ""))
            or not _is_sha256(offset.get("embedding_slice_sha256"))
            or offset.get("embedding_slice_sha256") != sha256_array(values[start:stop])
        ):
            raise ProtocolError("Label-free target scratch slice binding drifted.")
        cursor = stop
    if cursor != len(values):
        raise ProtocolError("Label-free target scratch slice coverage drifted.")


__all__ = (
    "BASE_ACTION_ID",
    "EXPECTED_CELL_COUNT",
    "FrozenAction",
    "GLOBAL_PREDICTION_SEAL_MEMBER",
    "GlobalPredictionSeal",
    "H_X_E_ACTION_PREFIX",
    "LabelFreePredictionStore",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "PredictionCell",
    "build_direct_target_actions",
    "h_x_e_action_id",
    "load_global_prediction_seal",
    "materialize_label_free_action_predictions",
)
