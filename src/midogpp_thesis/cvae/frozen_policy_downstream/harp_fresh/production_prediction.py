"""Actual frozen-generation and classifier prediction provider for HARP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...generation.config import load_generation_lock_config
from ...generation.runner import read_generation_lock
from ...generation.validation import validate_generation_bundle
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from ...runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
    HarpActionSpec,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    build_all_target_actions,
    compose_harp_action,
    harp_composition_seed,
    harp_source_stream_content_hash,
    seal_harp_prediction_menu,
)
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    raw_array_sha256,
)
from ...runtime.frozen_source_streams import (
    SOURCE_ROWS_PER_CLASS,
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
)
from ..fresh_runtime_contract import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
)
from .config import HarpFreshStage70Config
from .contracts import (
    HarpFreshPredictionOutput,
    HarpFreshTargetFrame,
)
from .policy import FrozenHarpPolicy
from .workspace_binding import HarpFreshWorkspaceBinding


@dataclass(frozen=True)
class _GenerationConfigAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class HarpProductionPredictionState:
    provider: "HarpProductionPredictionProvider"
    source_cache: FrozenSourceStreamCache
    source_stream_content_hash: str
    generation_lock_hash: str


@dataclass(frozen=True, kw_only=True)
class HarpProductionPredictionTask:
    outer_target_id: str
    selected_source_id: str | None
    action_id: str
    training_seed: int
    generation_seed: int
    action_hash: str
    frame_hash: str
    target_embedding_bytes_sha256: str
    row_count: int
    policy_lock_hash: str
    source_stream_content_hash: str
    target_cache_hash: str
    classifier_config_hash: str
    task_hash: str

    @property
    def task_id(self) -> str:
        source = self.action_id if self.selected_source_id is None else self.selected_source_id
        return (
            f"H_{self.outer_target_id}__e_{source}__"
            f"train_{self.training_seed}__gen_{self.generation_seed}"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_fresh_prediction_task_v2",
            "task_id": self.task_id,
            "outer_target_id": self.outer_target_id,
            "selected_source_id": self.selected_source_id,
            "action_id": self.action_id,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "action_hash": self.action_hash,
            "frame_hash": self.frame_hash,
            "target_embedding_bytes_sha256": self.target_embedding_bytes_sha256,
            "row_count": self.row_count,
            "policy_lock_hash": self.policy_lock_hash,
            "source_stream_content_hash": self.source_stream_content_hash,
            "target_cache_hash": self.target_cache_hash,
            "classifier_config_hash": self.classifier_config_hash,
            "task_hash": self.task_hash,
        }


class HarpProductionPredictionProvider:
    """CPU classifier edge over deterministic, spawned GPU source streams."""

    __slots__ = (
        "_source_cache",
        "_generation_lock_hash",
        "_classifier",
        "_classifier_threads",
    )

    def __init__(
        self,
        *,
        source_cache: FrozenSourceStreamCache,
        generation_lock_hash: str,
        config: HarpFreshStage70Config,
    ) -> None:
        self._source_cache = source_cache
        self._generation_lock_hash = generation_lock_hash
        self._classifier = config.classifier
        self._classifier_threads = int(
            config.runtime["classifier_threads_per_worker"]
        )
        if self._classifier_threads != CLASSIFIER_THREADS_PER_WORKER:
            raise ProtocolError("Fresh HARP classifier thread geometry drifted.")

    def __call__(
        self,
        action: HarpActionSpec,
        training_seed: int,
        generation_seed: int,
        frame: HarpFreshTargetFrame,
    ) -> HarpFreshPredictionOutput:
        if not isinstance(action, HarpActionSpec) or not isinstance(
            frame, HarpFreshTargetFrame
        ):
            raise ProtocolError("Fresh HARP production predictor requires typed inputs.")
        if action.outer_target_id != frame.center:
            raise ProtocolError("Fresh HARP action escaped its target frame.")
        blocks = {
            source: _composition_block(
                self._source_cache, source, training_seed, generation_seed
            )
            for source in action.source_order
        }
        shuffle_seeds = {
            label: harp_composition_seed(
                generation_lock_hash=self._generation_lock_hash,
                outer_target_id=frame.center,
                query_center_id=frame.center,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
            )
            for label in (0, 1)
        }
        composition = compose_harp_action(
            blocks,
            action,
            shuffle_seed_by_class=shuffle_seeds,
        )
        try:
            from threadpoolctl import threadpool_limits
        except ModuleNotFoundError as exc:
            raise RuntimeError("Fresh HARP prediction requires threadpoolctl.") from exc
        with threadpool_limits(limits=self._classifier_threads):
            fitted = fit_logistic_classifier(
                composition.embeddings,
                composition.labels,
                frame.embeddings,
                spec=self._classifier,
            )
        probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
        if (
            fitted.classes != (0, 1)
            or fitted.converged is not True
            or probabilities.shape != (len(frame.row_ids), 2)
            or not np.isfinite(probabilities).all()
        ):
            raise ProtocolError("Fresh HARP classifier fit or class geometry failed.")
        positive = np.ascontiguousarray(probabilities[:, 1], dtype=np.float32)
        return HarpFreshPredictionOutput(
            probabilities=positive,
            composition_hash=composition.composition_hash,
            scaler_state_hash=fitted.scaler_state_hash,
        )


def _composition_block(
    cache: FrozenSourceStreamCache,
    source: str,
    training_seed: int,
    generation_seed: int,
) -> dict[str, np.ndarray]:
    values = cache.block(source, training_seed, generation_seed)
    if (
        values.dtype != np.float32
        or values.shape != (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
    ):
        raise ProtocolError("Fresh HARP frozen source block geometry drifted.")
    labels = np.concatenate(
        (
            np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
            np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
        )
    )
    labels.setflags(write=False)
    return {"embeddings": values, "labels": labels}


def _load_generation_lock(config: HarpFreshStage70Config, binding: HarpFreshWorkspaceBinding):
    generation_config_path = binding.generation_lock_root / "config.resolved.yaml"
    generation_path = binding.generation_lock_root / "manifests/generation_lock.json"
    generation_config = load_generation_lock_config(generation_config_path)
    if generation_config.bank_root.resolve() != binding.expert_bank_root.resolve():
        raise ProtocolError("Fresh HARP bank/GenerationLock roots disagree.")
    validate_generation_bundle(binding.generation_lock_root, config=generation_config)
    lock = read_generation_lock(generation_path)
    if (
        lock.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or lock.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or generation_config.classifier != config.classifier
    ):
        raise ProtocolError("Fresh HARP validated GenerationLock semantics drifted.")
    return lock


def prepare_harp_production_prediction(
    config: HarpFreshStage70Config,
    binding: HarpFreshWorkspaceBinding,
    policy: FrozenHarpPolicy,
    *,
    source_cache_root: Path,
) -> HarpProductionPredictionState:
    """Validate locks, spawn source generation, and return the actual predictor."""

    if (
        not isinstance(config, HarpFreshStage70Config)
        or not isinstance(binding, HarpFreshWorkspaceBinding)
        or not isinstance(policy, FrozenHarpPolicy)
        or policy.production_ready is not True
    ):
        raise ProtocolError("Fresh HARP production prediction requires full admission.")
    if policy.metadata.classifier_hash != config.classifier.config_hash:
        raise ProtocolError("Fresh HARP frozen inference lineage drifted.")
    lineage = policy.execution_lineage
    if lineage is None:
        raise ProtocolError("Fresh HARP production policy lacks executable lineage.")
    bank_path = binding.expert_bank_root / "manifests/expert_bank_index.json"
    generation_path = binding.generation_lock_root / "manifests/generation_lock.json"
    if (
        sha256_file(bank_path) != lineage.expert_bank_index_sha256
        or sha256_file(generation_path) != lineage.generation_lock_file_sha256
    ):
        raise ProtocolError("Fresh HARP authoritative bank/GenerationLock bytes drifted.")
    generation_lock = _load_generation_lock(config, binding)
    generation_payload = generation_lock.to_payload()
    raw_classifier = generation_payload.get("classifier")
    if not isinstance(raw_classifier, Mapping):
        raise ProtocolError("Fresh HARP GenerationLock classifier receipt is absent.")
    classifier_contract_hash = canonical_sha256(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v1",
            "classifier": config.classifier.to_payload(),
            "scaler_family": raw_classifier.get("scaler_family"),
            "fit_in_stage_40": raw_classifier.get("fit_in_stage_40"),
        }
    )
    if (
        policy.metadata.bank_hash != generation_lock.bank_lock_hash
        or policy.metadata.generation_lock_hash != generation_lock.generation_lock_hash
        or lineage.bank_semantic_lock_hash != generation_lock.bank_lock_hash
        or lineage.generation_semantic_lock_hash
        != generation_lock.generation_lock_hash
        or lineage.classifier_config_hash != config.classifier.config_hash
        or lineage.classifier_contract_sha256 != classifier_contract_hash
    ):
        raise ProtocolError("Fresh HARP frozen semantic lock lineage drifted.")
    adapter = _GenerationConfigAdapter(
        contract_hash=canonical_sha256(
            {
                "schema_version": "midogpp_harp_fresh_source_runtime_binding_v1",
                "stage70_config_contract_hash": config.contract_hash,
                "policy_lock_hash": policy.metadata.policy_lock_hash,
                "bank_semantic_lock_hash": generation_lock.bank_lock_hash,
                "generation_semantic_lock_hash": generation_lock.generation_lock_hash,
            }
        ),
        expert_bank_root=binding.expert_bank_root,
        runtime=MappingProxyType(
            {
                "generation_devices": ["cuda:0", "cuda:1"],
                "source_workers_per_device": 1,
                "generation_workers_per_device": 1,
                "persistent_source_workers": True,
                "multiprocessing_start_method": "spawn",
                "parent_cuda_context_forbidden": True,
                "tf32_enabled": False,
                "amp_enabled": False,
                "generated_cache_format": "float32_npy_memmap",
                "source_prefix_rows_per_class": SOURCE_ROWS_PER_CLASS,
            }
        ),
    )
    source_cache = materialize_frozen_source_streams(
        adapter,
        generation_lock,
        root=Path(source_cache_root),
    )
    source_content_hash = harp_source_stream_content_hash(source_cache.records)
    if (
        source_cache.lock_payload.get("generation_lock_hash")
        != generation_lock.generation_lock_hash
        or source_content_hash != policy.metadata.source_cache_hash
    ):
        raise ProtocolError("Fresh HARP regenerated source-cache lineage drifted.")
    provider = HarpProductionPredictionProvider(
        source_cache=source_cache,
        generation_lock_hash=generation_lock.generation_lock_hash,
        config=config,
    )
    return HarpProductionPredictionState(
        provider=provider,
        source_cache=source_cache,
        source_stream_content_hash=source_content_hash,
        generation_lock_hash=generation_lock.generation_lock_hash,
    )


_CPU_WORKER_STATE: dict[str, object] | None = None


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
        raise ProtocolError("Fresh HARP worker classifier payload drifted.") from exc


def _cpu_worker_init(
    source_cache_root: str,
    source_config_hash: str,
    target_cache_root: str,
    generation_lock_hash: str,
    classifier_payload: Mapping[str, object],
    output_root: str,
    threads: int,
) -> None:
    global _CPU_WORKER_STATE
    if threads != CLASSIFIER_THREADS_PER_WORKER:
        raise ProtocolError("Fresh HARP CPU worker thread geometry drifted.")
    _CPU_WORKER_STATE = {
        "source_cache": load_frozen_source_streams(
            Path(source_cache_root),
            expected_config_hash=source_config_hash,
            expected_generation_lock_hash=generation_lock_hash,
        ),
        "target_cache_root": Path(target_cache_root),
        "generation_lock_hash": generation_lock_hash,
        "classifier": _classifier_from_payload(classifier_payload),
        "output_root": Path(output_root),
        "threads": threads,
    }


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_result_paths(root: Path, task_id: str) -> tuple[Path, Path]:
    return root / f"arrays/{task_id}.probabilities.npy", root / f"tasks/{task_id}.json"


def _cpu_worker_task(raw: Mapping[str, object]) -> dict[str, object]:
    state = _CPU_WORKER_STATE
    if state is None:
        raise ProtocolError("Fresh HARP CPU worker was not initialized.")
    task = HarpProductionPredictionTask(
        outer_target_id=str(raw["outer_target_id"]),
        selected_source_id=(
            None if raw["selected_source_id"] is None else str(raw["selected_source_id"])
        ),
        action_id=str(raw["action_id"]),
        training_seed=int(raw["training_seed"]),
        generation_seed=int(raw["generation_seed"]),
        action_hash=str(raw["action_hash"]),
        frame_hash=str(raw["frame_hash"]),
        target_embedding_bytes_sha256=str(raw["target_embedding_bytes_sha256"]),
        row_count=int(raw["row_count"]),
        policy_lock_hash=str(raw["policy_lock_hash"]),
        source_stream_content_hash=str(raw["source_stream_content_hash"]),
        target_cache_hash=str(raw["target_cache_hash"]),
        classifier_config_hash=str(raw["classifier_config_hash"]),
        task_hash=str(raw["task_hash"]),
    )
    task_unhashed = {
        key: value for key, value in task.to_payload().items() if key != "task_hash"
    }
    if task.task_hash != canonical_sha256(task_unhashed):
        raise ProtocolError("Fresh HARP worker task binding drifted.")
    action = HarpActionSpec(
        surface_kind="target",
        outer_target_id=task.outer_target_id,
        query_center_id=task.outer_target_id,
        selected_source_id=task.selected_source_id,
        action_id=task.action_id,
    )
    if action.action_hash != task.action_hash:
        raise ProtocolError("Fresh HARP worker action reconstruction drifted.")
    source_cache = state["source_cache"]
    assert isinstance(source_cache, FrozenSourceStreamCache)
    if harp_source_stream_content_hash(source_cache.records) != task.source_stream_content_hash:
        raise ProtocolError("Fresh HARP worker source content drifted.")
    blocks = {
        source: _composition_block(
            source_cache, source, task.training_seed, task.generation_seed
        )
        for source in action.source_order
    }
    generation_lock_hash = str(state["generation_lock_hash"])
    shuffle_seeds = {
        label: harp_composition_seed(
            generation_lock_hash=generation_lock_hash,
            outer_target_id=task.outer_target_id,
            query_center_id=task.outer_target_id,
            training_seed=task.training_seed,
            generation_seed=task.generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    composition = compose_harp_action(
        blocks, action, shuffle_seed_by_class=shuffle_seeds
    )
    target_path = Path(state["target_cache_root"]) / (
        f"embeddings/by_center/center_{task.outer_target_id}.npy"
    )
    target = np.load(target_path, mmap_mode="r", allow_pickle=False)
    if (
        target.dtype != np.float32
        or target.shape != (task.row_count, COMMON_OUTPUT_DIM)
        or raw_array_sha256(target) != task.target_embedding_bytes_sha256
    ):
        raise ProtocolError("Fresh HARP worker target memmap geometry drifted.")
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:
        raise RuntimeError("Fresh HARP prediction requires threadpoolctl.") from exc
    classifier = state["classifier"]
    assert isinstance(classifier, ClassifierSpec)
    with threadpool_limits(limits=int(state["threads"])):
        fitted = fit_logistic_classifier(
            composition.embeddings,
            composition.labels,
            target,
            spec=classifier,
        )
    values = np.asarray(fitted.probabilities, dtype=np.float64)
    if (
        fitted.classes != (0, 1)
        or fitted.converged is not True
        or fitted.classifier_config_hash != task.classifier_config_hash
        or values.shape != (task.row_count, 2)
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("Fresh HARP spawned classifier failed.")
    probability = np.ascontiguousarray(values[:, 1], dtype=np.float32)
    probability_path, metadata_path = _task_result_paths(
        Path(state["output_root"]), task.task_id
    )
    _atomic_save_npy(probability_path, probability)
    result = {
        **task.to_payload(),
        "schema_version": "midogpp_harp_fresh_prediction_task_result_v2",
        "probability_member": probability_path.relative_to(Path(state["output_root"])).as_posix(),
        "probability_file_sha256": _file_sha256(probability_path),
        "probability_bytes_sha256": raw_array_sha256(probability),
        "composition_hash": composition.composition_hash,
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "classifier_converged": True,
        "labels_available_to_fit_or_predict": False,
    }
    result["result_hash"] = canonical_sha256(result)
    _atomic_json(metadata_path, result)
    return result


def _task_plan(
    cache: object,
    *,
    policy: FrozenHarpPolicy,
    source_stream_content_hash: str,
    classifier_hash: str,
) -> tuple[HarpProductionPredictionTask, ...]:
    from .contracts import HarpFreshTargetCache

    if not isinstance(cache, HarpFreshTargetCache):
        raise ProtocolError("Fresh HARP prediction plan requires typed cache.")
    tasks: list[HarpProductionPredictionTask] = []
    for action in build_all_target_actions():
        frame = cache.frames_by_center[action.outer_target_id]
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            unhashed = {
                "schema_version": "midogpp_harp_fresh_prediction_task_v2",
                "task_id": (
                    f"H_{action.outer_target_id}__e_"
                    f"{action.action_id if action.selected_source_id is None else action.selected_source_id}__"
                    f"train_{training_seed}__gen_{generation_seed}"
                ),
                "policy_lock_hash": policy.metadata.policy_lock_hash,
                "source_stream_content_hash": source_stream_content_hash,
                "target_cache_hash": cache.cache_hash,
                "classifier_config_hash": classifier_hash,
                "action_hash": action.action_hash,
                "frame_hash": frame.frame_hash,
                "target_embedding_bytes_sha256": raw_array_sha256(frame.embeddings),
                "outer_target_id": action.outer_target_id,
                "selected_source_id": action.selected_source_id,
                "action_id": action.action_id,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "row_count": len(frame.row_ids),
            }
            tasks.append(
                HarpProductionPredictionTask(
                    outer_target_id=action.outer_target_id,
                    selected_source_id=action.selected_source_id,
                    action_id=action.action_id,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    action_hash=action.action_hash,
                    frame_hash=frame.frame_hash,
                    target_embedding_bytes_sha256=raw_array_sha256(frame.embeddings),
                    row_count=len(frame.row_ids),
                    policy_lock_hash=policy.metadata.policy_lock_hash,
                    source_stream_content_hash=source_stream_content_hash,
                    target_cache_hash=cache.cache_hash,
                    classifier_config_hash=classifier_hash,
                    task_hash=canonical_sha256(unhashed),
                )
            )
    if len(tasks) != 810:
        raise ProtocolError("Fresh HARP production prediction plan is incomplete.")
    return tuple(tasks)


def _load_task_result(root: Path, task: HarpProductionPredictionTask) -> dict[str, object] | None:
    probability_path, metadata_path = _task_result_paths(root, task.task_id)
    if not probability_path.is_file() or not metadata_path.is_file():
        return None
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    observed_hash = raw.get("result_hash")
    task_payload = task.to_payload()
    task_fields_match = all(
        raw.get(key) == value
        for key, value in task_payload.items()
        if key != "schema_version"
    )
    if (
        not task_fields_match
        or raw.get("schema_version")
        != "midogpp_harp_fresh_prediction_task_result_v2"
        or observed_hash
        != canonical_sha256({key: value for key, value in raw.items() if key != "result_hash"})
        or raw.get("probability_file_sha256") != _file_sha256(probability_path)
    ):
        return None
    values = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if (
        values.dtype != np.float32
        or values.shape != (task.row_count,)
        or raw.get("probability_bytes_sha256") != raw_array_sha256(values)
        or raw.get("labels_available_to_fit_or_predict") is not False
        or raw.get("classifier_converged") is not True
    ):
        return None
    return raw


def materialize_harp_production_probability_menu(
    config: HarpFreshStage70Config,
    binding: HarpFreshWorkspaceBinding,
    policy: FrozenHarpPolicy,
    state: HarpProductionPredictionState,
    cache: object,
    *,
    root: Path,
) -> HarpPredictionMenuSeal:
    """Run/resume 810 B/U/Hxe cells in exactly four spawned CPU workers."""

    from .contracts import HarpFreshTargetCache

    if (
        not isinstance(cache, HarpFreshTargetCache)
        or int(config.runtime["classifier_workers"]) != CLASSIFIER_WORKERS
        or CLASSIFIER_WORKERS != 4
        or int(config.runtime["classifier_threads_per_worker"])
        != CLASSIFIER_THREADS_PER_WORKER
        or CLASSIFIER_THREADS_PER_WORKER != 3
    ):
        raise ProtocolError("Fresh HARP 4x3 classifier topology drifted.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = _task_plan(
        cache,
        policy=policy,
        source_stream_content_hash=state.source_stream_content_hash,
        classifier_hash=config.classifier.config_hash,
    )
    results: dict[str, dict[str, object]] = {}
    pending: list[HarpProductionPredictionTask] = []
    for task in tasks:
        resumed = _load_task_result(root, task)
        if resumed is None:
            pending.append(task)
        else:
            results[task.task_id] = resumed
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=CLASSIFIER_WORKERS,
            initializer=_cpu_worker_init,
            initargs=(
                str(state.source_cache.root.resolve()),
                str(state.source_cache.lock_payload["config_contract_hash"]),
                str(binding.target_cache_root.resolve()),
                state.generation_lock_hash,
                config.classifier.to_payload(),
                str(root.resolve()),
                CLASSIFIER_THREADS_PER_WORKER,
            ),
        ) as pool:
            generated = pool.map(
                _cpu_worker_task,
                [task.to_payload() for task in pending],
                chunksize=1,
            )
        for raw in generated:
            results[str(raw["task_id"])] = raw
    if len(results) != len(tasks):
        raise ProtocolError("Fresh HARP spawned prediction inventory is incomplete.")
    actions = build_all_target_actions()
    action_by_hash = {action.action_hash: action for action in actions}
    cells: list[HarpPredictionCell] = []
    for task in tasks:
        raw = results[task.task_id]
        probability_path = root / str(raw["probability_member"])
        values = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        frame = cache.frames_by_center[task.outer_target_id]
        cells.append(
            HarpPredictionCell(
                action=action_by_hash[task.action_hash],
                training_seed=task.training_seed,
                generation_seed=task.generation_seed,
                row_ids=frame.row_ids,
                case_ids=frame.case_ids,
                probabilities=np.asarray(values),
                bank_hash=policy.metadata.bank_hash,
                generation_lock_hash=policy.metadata.generation_lock_hash,
                source_cache_hash=state.source_stream_content_hash,
                frame_hash=frame.frame_hash,
                classifier_hash=config.classifier.config_hash,
                composition_hash=str(raw["composition_hash"]),
                scaler_state_hash=str(raw["scaler_state_hash"]),
            )
        )
    seal = seal_harp_prediction_menu(actions, cells)
    seal.assert_valid()
    return seal


__all__ = (
    "HarpProductionPredictionProvider",
    "HarpProductionPredictionState",
    "HarpProductionPredictionTask",
    "materialize_harp_production_probability_menu",
    "prepare_harp_production_prediction",
)
