"""V4-local production of the physical lambda=1 B/U/Hxe probability menu.

The adapter uses only frozen expert/generation primitives, the neutral HARP
action algebra, and the immutable v4 label-blind cache.  CUDA is confined to
the neutral two-worker frozen-source materializer.  That pool is closed before
the four-worker classifier pool is created.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation import read_generation_lock
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, atomic_npy, atomic_npz, read_json, sha256_file
from ...runtime.frozen_source_streams import (
    SOURCE_ROWS_PER_CLASS,
    FrozenSourceStreamCache,
    materialize_frozen_source_streams,
)
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpActionSpec,
    build_all_development_actions,
    build_all_target_actions,
    compose_harp_action,
    harp_composition_seed,
)
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ..bounded_futures import execute_bounded
from .classifier_worker_cache import (
    initialize_classifier_worker as _initialize_classifier_worker,
    load_source_blocks as _load_source_blocks,
    load_worker_arrays as _load_worker_arrays,
)
from .contracts import ActionKind, LabelFreeActionBlock, LabelFreeOuterMenu
from .frame_binding import persist_or_validate_frame_binding
from .hash_contracts import (
    require_sha256 as _require_sha256,
    require_stable_hash as _require_stable_hash,
)
from .task_bindings import (
    validate_frame_task_binding,
    validate_source_task_binding,
)


@dataclass(frozen=True, slots=True)
class PhysicalInputReceipt:
    generation_lock: GenerationLock
    classifier: ClassifierSpec
    bank_hash: str
    generation_hash: str
    bank_index_sha256: str
    generation_file_sha256: str
    cache_hash: str
    receipt_hash: str

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v4_physical_input_receipt_v1",
            "bank_hash": self.bank_hash,
            "generation_hash": self.generation_hash,
            "bank_index_sha256": self.bank_index_sha256,
            "generation_file_sha256": self.generation_file_sha256,
            "cache_hash": self.cache_hash,
            "classifier_hash": self.classifier.config_hash,
            "labels_consumed": False,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class _SourceAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Frames:
    path: Path
    receipt_path: Path
    contexts: Mapping[tuple[str, str], tuple[int, int]]
    sample_ids: Mapping[tuple[str, str], tuple[str, ...]]
    case_ids: Mapping[tuple[str, str], tuple[str, ...]]
    sha256: str
    receipt_hash: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _WorkstationProfile:
    """Typed execution constants for the dedicated two-A5000 workstation.

    This adapter gives the physical runner one fail-closed source for process
    topology, queue bounds, BLAS limits, transport precision, and CUDA
    ownership instead of repeating literals in the plan, source adapter, task
    builder, and executor.
    """

    profile_name: str = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
    gpu_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    persistent_gpu_workers: int = 2
    cpu_fit_workers: int = 4
    blas_threads_per_worker: int = 3
    multiprocessing_start_method: str = "spawn"
    parent_cuda_context_created: bool = False
    late_torch_interop_setter_used: bool = False
    probability_transport_dtype: str = "float32"
    scientific_reduction_dtype: str = "float64"
    memory_mapped_surfaces: bool = True
    bounded_inflight_batches_per_gpu: int = 2
    bounded_inflight_tasks_per_cpu_worker: int = 2

    def __post_init__(self) -> None:
        if (
            self.profile_name != "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
            or self.gpu_devices != tuple(f"cuda:{index}" for index in range(2))
            or type(self.persistent_gpu_workers) is not int
            or self.persistent_gpu_workers != len(self.gpu_devices)
            or type(self.cpu_fit_workers) is not int
            or self.cpu_fit_workers != 4
            or type(self.blas_threads_per_worker) is not int
            or self.blas_threads_per_worker != 3
            or self.multiprocessing_start_method != "spawn"
            or self.parent_cuda_context_created is not False
            or self.late_torch_interop_setter_used is not False
            or self.probability_transport_dtype != "float32"
            or self.scientific_reduction_dtype != "float64"
            or self.memory_mapped_surfaces is not True
            or type(self.bounded_inflight_batches_per_gpu) is not int
            or self.bounded_inflight_batches_per_gpu != 2
            or type(self.bounded_inflight_tasks_per_cpu_worker) is not int
            or self.bounded_inflight_tasks_per_cpu_worker != 2
        ):
            raise ProtocolError("HARP v4 workstation execution profile drifted.")

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, object]) -> _WorkstationProfile:
        raw_devices = runtime.get("gpu_devices")
        devices = tuple(raw_devices) if isinstance(raw_devices, (list, tuple)) else ()
        return cls(
            profile_name=runtime.get("profile"),  # type: ignore[arg-type]
            gpu_devices=devices,  # type: ignore[arg-type]
            persistent_gpu_workers=runtime.get("persistent_gpu_workers"),  # type: ignore[arg-type]
            cpu_fit_workers=runtime.get("cpu_fit_workers"),  # type: ignore[arg-type]
            blas_threads_per_worker=runtime.get("blas_threads_per_worker"),  # type: ignore[arg-type]
            multiprocessing_start_method=runtime.get("multiprocessing_start_method"),  # type: ignore[arg-type]
            parent_cuda_context_created=runtime.get("parent_cuda_context_created"),  # type: ignore[arg-type]
            late_torch_interop_setter_used=runtime.get("late_torch_interop_setter_used"),  # type: ignore[arg-type]
            probability_transport_dtype=runtime.get("probability_transport_dtype"),  # type: ignore[arg-type]
            scientific_reduction_dtype=runtime.get("scientific_reduction_dtype"),  # type: ignore[arg-type]
            memory_mapped_surfaces=runtime.get("memory_mapped_surfaces"),  # type: ignore[arg-type]
            bounded_inflight_batches_per_gpu=runtime.get(
                "bounded_inflight_batches_per_gpu"
            ),  # type: ignore[arg-type]
            bounded_inflight_tasks_per_cpu_worker=runtime.get(
                "bounded_inflight_tasks_per_cpu_worker"
            ),  # type: ignore[arg-type]
        )

    @property
    def profile_hash(self) -> str:
        return canonical_hash(self.public_payload())

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v4_workstation_profile_v1",
            "profile": self.profile_name,
            "gpu_devices": list(self.gpu_devices),
            "persistent_gpu_workers": self.persistent_gpu_workers,
            "cpu_fit_workers": self.cpu_fit_workers,
            "blas_threads_per_worker": self.blas_threads_per_worker,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "parent_cuda_context_created": self.parent_cuda_context_created,
            "late_torch_interop_setter_used": self.late_torch_interop_setter_used,
            "probability_transport_dtype": self.probability_transport_dtype,
            "scientific_reduction_dtype": self.scientific_reduction_dtype,
            "memory_mapped_surfaces": self.memory_mapped_surfaces,
            "bounded_inflight_batches_per_gpu": self.bounded_inflight_batches_per_gpu,
            "bounded_inflight_tasks_per_cpu_worker": (
                self.bounded_inflight_tasks_per_cpu_worker
            ),
        }

    def source_runtime(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "generation_devices": list(self.gpu_devices),
                "source_workers_per_device": 1,
                "generation_workers_per_device": 1,
                "persistent_source_workers": True,
                "multiprocessing_start_method": self.multiprocessing_start_method,
                "parent_cuda_context_forbidden": not self.parent_cuda_context_created,
                "tf32_enabled": False,
                "amp_enabled": False,
                "generated_cache_format": (
                    f"{self.probability_transport_dtype}_npy_memmap"
                ),
                "source_prefix_rows_per_class": SOURCE_ROWS_PER_CLASS,
                "bounded_inflight_batches_per_gpu": self.bounded_inflight_batches_per_gpu,
                "workstation_profile_hash": self.profile_hash,
            }
        )


_DEFAULT_WORKSTATION_PROFILE = _WorkstationProfile()


def build_physical_plan() -> dict[str, object]:
    workstation = _DEFAULT_WORKSTATION_PROFILE
    actions = (*build_all_development_actions(), *build_all_target_actions())
    contexts = {
        (row.surface_kind, row.outer_target_id, row.query_center_id) for row in actions
    }
    body = {
        "schema_version": "midogpp_harp_v4_physical_plan_v1",
        "action_count": len(actions),
        "query_context_count": len(contexts),
        "classifier_task_count": len(contexts) * len(EXACT_NINE_SEED_PAIRS),
        "seed_cell_count": len(actions) * len(EXACT_NINE_SEED_PAIRS),
        "physical_expert_weight": 1.0,
        "probability_blends_present": False,
        "persistent_gpu_workers": workstation.persistent_gpu_workers,
        "gpu_devices": list(workstation.gpu_devices),
        "cpu_fit_workers": workstation.cpu_fit_workers,
        "blas_threads_per_worker": workstation.blas_threads_per_worker,
        "multiprocessing_start_method": workstation.multiprocessing_start_method,
        "transport_dtype": workstation.probability_transport_dtype,
        "reduction_dtype": workstation.scientific_reduction_dtype,
        "tf32_enabled": False,
        "amp_enabled": False,
        "shared_validated_menu_index": True,
        "bounded_inflight_batches_per_gpu": workstation.bounded_inflight_batches_per_gpu,
        "max_inflight_source_tasks": (
            workstation.persistent_gpu_workers
            * workstation.bounded_inflight_batches_per_gpu
        ),
        "bounded_inflight_tasks_per_cpu_worker": (
            workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "max_inflight_classifier_tasks": (
            workstation.cpu_fit_workers
            * workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "labels_consumed": False,
        "workstation_profile_hash": workstation.profile_hash,
    }
    if (
        len(actions) != 738
        or len(contexts) != 81
        or body["classifier_task_count"] != 729
        or body["seed_cell_count"] != 6642
    ):
        raise ProtocolError("HARP v4 physical action topology drifted.")
    return {**body, "plan_hash": canonical_hash(body)}


def validate_physical_inputs(config: object, cache: object) -> PhysicalInputReceipt:
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for root, name in ((bank_root, "expert bank"), (generation_root, "generation lock")):
        if not root.is_dir() or root.is_symlink():
            raise ProtocolError(f"HARP v4 authoritative {name} root is unsafe.")
        state = read_json(root / "reports/run_state.json")
        validation = read_json(root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v4 authoritative {name} is not complete and valid.")
    bank_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    generation = read_generation_lock(generation_path)
    bank_payload = read_json(bank_path)
    lock_payload = generation.to_payload()
    bank_lock_hash = _require_stable_hash(
        generation.bank_lock_hash, name="expert-bank lock hash"
    )
    generation_lock_hash = _require_stable_hash(
        generation.generation_lock_hash, name="generation-lock hash"
    )
    bank_binding = lock_payload.get("bank")
    raw_classifier = lock_payload.get("classifier")
    if not isinstance(bank_binding, Mapping) or not isinstance(raw_classifier, Mapping):
        raise ProtocolError("HARP v4 generation lock lacks bank/classifier bindings.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    if (
        bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v4 authoritative physical lineage drifted.")
    try:
        classifier = ClassifierSpec(
            family=str(raw_classifier["family"]),
            C=float(raw_classifier["C"]),
            penalty=str(raw_classifier["penalty"]),
            solver=str(raw_classifier["solver"]),
            max_iter=int(raw_classifier["max_iter"]),
            class_weight=(
                None
                if raw_classifier["class_weight"] is None
                else str(raw_classifier["class_weight"])
            ),
            random_state=int(raw_classifier["random_state"]),
            l1_ratio=(
                None if raw_classifier["l1_ratio"] is None else float(raw_classifier["l1_ratio"])
            ),
            threshold_policy=str(raw_classifier["threshold_policy"]),
            scaler_fit=str(raw_classifier["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v4 classifier contract is malformed.") from exc
    if classifier.config_hash != raw_classifier.get("config_hash"):
        raise ProtocolError("HARP v4 classifier identity drifted.")
    cache_hash = str(getattr(cache, "cache_hash"))
    body = {
        "schema_version": "midogpp_harp_v4_physical_input_receipt_v1",
        "bank_hash": bank_lock_hash,
        "generation_hash": generation_lock_hash,
        "bank_index_sha256": bank_sha,
        "generation_file_sha256": generation_sha,
        "cache_hash": cache_hash,
        "classifier_hash": classifier.config_hash,
        "labels_consumed": False,
    }
    return PhysicalInputReceipt(
        generation_lock=generation,
        classifier=classifier,
        bank_hash=bank_lock_hash,
        generation_hash=generation_lock_hash,
        bank_index_sha256=bank_sha,
        generation_file_sha256=generation_sha,
        cache_hash=cache_hash,
        receipt_hash=canonical_hash(body),
    )


def materialize_physical_outer_menus(
    config: object,
    cache: object,
    *,
    scratch_root: Path,
    development_role: str,
    evaluation_role: str,
) -> tuple[LabelFreeOuterMenu, ...]:
    inputs = validate_physical_inputs(config, cache)
    workstation = _WorkstationProfile.from_runtime(config.runtime)
    scratch_root.mkdir(parents=True, exist_ok=True)
    frames = _stage_frames(
        cache, scratch_root=scratch_root, roles=(development_role, evaluation_role)
    )
    source_adapter = _SourceAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v4_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": inputs.receipt_hash,
                "frame_sha256": frames.sha256,
                "workstation_profile_hash": workstation.profile_hash,
            }
        ),
        expert_bank_root=config.resolved_path("expert_bank_root"),
        runtime=workstation.source_runtime(),
    )
    source_cache = materialize_frozen_source_streams(
        source_adapter,
        inputs.generation_lock,
        root=scratch_root / "source_streams",
    )
    tasks = _build_tasks(
        scratch_root=scratch_root,
        frames=frames,
        source_cache=source_cache,
        inputs=inputs,
        workstation=workstation,
        development_role=development_role,
        evaluation_role=evaluation_role,
    )
    completed = _execute_tasks(tasks, workstation=workstation)
    return _aggregate_outer_menus(tasks, completed, inputs=inputs)


def _stage_frames(cache: object, *, scratch_root: Path, roles: tuple[str, str]) -> _Frames:
    path = (scratch_root / "frames/consumed_rows.npy").resolve()
    receipt_path = (scratch_root / "frames/receipt.json").resolve()
    contexts: dict[tuple[str, str], tuple[int, int]] = {}
    samples: dict[tuple[str, str], tuple[str, ...]] = {}
    cases: dict[tuple[str, str], tuple[str, ...]] = {}
    values: list[np.ndarray] = []
    cursor = 0
    load_embeddings = getattr(cache, "load_embeddings", None)
    if not callable(load_embeddings):
        raise ProtocolError("HARP v4 cache lacks the typed grouped-shard reader.")
    for role in roles:
        for center in CENTERS:
            rows = tuple(cache.rows_for(center, role))
            if not rows or tuple(row.split_row_index for row in rows) != tuple(range(len(rows))):
                raise ProtocolError("HARP v4 cache role geometry drifted.")
            matrix = np.asarray(load_embeddings(rows))
            if (
                matrix.dtype != np.float32
                or matrix.shape != (len(rows), COMMON_OUTPUT_DIM)
                or not np.isfinite(matrix).all()
            ):
                raise ProtocolError("HARP v4 grouped frame geometry drifted.")
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            contexts[(role, center)] = (cursor, cursor + len(rows))
            samples[(role, center)] = tuple(row.sample_id for row in rows)
            cases[(role, center)] = tuple(row.case_id for row in rows)
            cursor += len(rows)
            values.append(matrix)
    combined = np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
    if path.exists() != receipt_path.exists():
        raise ProtocolError("HARP v4 scratch frame store is only partially durable.")
    if path.exists():
        if (
            not path.is_file()
            or path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
        ):
            raise ProtocolError("HARP v4 scratch frame store paths are unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.dtype != np.float32
            or observed.shape != combined.shape
            or hashlib.sha256(observed.tobytes(order="C")).hexdigest()
            != hashlib.sha256(combined.tobytes(order="C")).hexdigest()
        ):
            raise ProtocolError("HARP v4 existing scratch frame store drifted.")
    else:
        atomic_npy(path, combined)
    binding = persist_or_validate_frame_binding(
        array_path=path,
        receipt_path=receipt_path,
        shape=combined.shape,
    )
    return _Frames(
        path=path,
        receipt_path=receipt_path,
        contexts=MappingProxyType(contexts),
        sample_ids=MappingProxyType(samples),
        case_ids=MappingProxyType(cases),
        sha256=binding.array_sha256,
        receipt_hash=binding.receipt_hash,
        receipt_sha256=binding.receipt_sha256,
    )


def _all_actions() -> tuple[HarpActionSpec, ...]:
    return (*build_all_development_actions(), *build_all_target_actions())


def _build_tasks(
    *,
    scratch_root: Path,
    frames: _Frames,
    source_cache: FrozenSourceStreamCache,
    inputs: PhysicalInputReceipt,
    workstation: _WorkstationProfile,
    development_role: str,
    evaluation_role: str,
) -> tuple[dict[str, object], ...]:
    by_context: dict[tuple[str, str, str], list[HarpActionSpec]] = defaultdict(list)
    for action in _all_actions():
        by_context[(action.surface_kind, action.outer_target_id, action.query_center_id)].append(action)
    source_binding = validate_source_task_binding(source_cache)
    frame_binding = validate_frame_task_binding(frames)
    checkpoint_root = scratch_root / "classifier_checkpoints"
    tasks: list[dict[str, object]] = []
    for surface, outer, query in sorted(by_context):
        role = development_role if surface == DEVELOPMENT_SURFACE else evaluation_role
        start, stop = frames.contexts[(role, query)]
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            ordinal = len(tasks)
            stem = f"task_{ordinal:04d}"
            body = {
                "schema_version": "midogpp_harp_v4_label_free_classifier_task_v1",
                "ordinal": ordinal,
                "surface_kind": surface,
                "outer_target_id": outer,
                "query_center_id": query,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "actions": [row.to_payload() for row in by_context[(surface, outer, query)]],
                "source_array_path": str(source_binding.array_path),
                "source_array_sha256": source_binding.array_sha256,
                "source_index_path": str(source_binding.index_path),
                "source_index_sha256": source_binding.index_sha256,
                "source_stream_index_hash": source_binding.index_hash,
                "source_records": list(source_binding.records),
                "frame_array_path": str(frame_binding.array_path),
                "frame_array_sha256": frame_binding.array_sha256,
                "frame_receipt_hash": frame_binding.receipt_hash,
                "frame_receipt_sha256": frame_binding.receipt_sha256,
                "frame_start": start,
                "frame_stop": stop,
                "sample_ids": list(frames.sample_ids[(role, query)]),
                "case_ids": list(frames.case_ids[(role, query)]),
                "generation_lock_hash": inputs.generation_hash,
                "bank_hash": inputs.bank_hash,
                "source_stream_lock_hash": source_binding.lock_hash,
                "source_stream_lock_sha256": source_binding.lock_sha256,
                "classifier": inputs.classifier.to_payload(),
                "threads_per_worker": workstation.blas_threads_per_worker,
                "workstation_profile_hash": workstation.profile_hash,
                "labels_available": False,
            }
            tasks.append(
                {
                    **body,
                    "task_hash": canonical_hash(body),
                    "npz_path": str(checkpoint_root / f"{stem}.npz"),
                    "receipt_path": str(checkpoint_root / f"{stem}.json"),
                }
            )
    if len(tasks) != 729:
        raise ProtocolError("HARP v4 classifier task coverage drifted.")
    return tuple(tasks)


def _execute_tasks(
    tasks: Sequence[Mapping[str, object]], *, workstation: _WorkstationProfile
) -> dict[int, Mapping[str, object]]:
    complete: dict[int, Mapping[str, object]] = {}
    pending = []
    for task in tasks:
        checkpoint = _load_task_checkpoint(task)
        if checkpoint is None:
            pending.append(task)
        else:
            complete[int(task["ordinal"])] = checkpoint
    if pending:
        with ProcessPoolExecutor(
            max_workers=workstation.cpu_fit_workers,
            mp_context=mp.get_context(workstation.multiprocessing_start_method),
            initializer=_initialize_classifier_worker,
            initargs=(workstation.blas_threads_per_worker,),
        ) as pool:
            def accept_checkpoint(
                _position: int, task: Mapping[str, object], _result: None
            ) -> None:
                checkpoint = _load_task_checkpoint(task)
                if checkpoint is None:
                    raise ProtocolError("HARP v4 classifier checkpoint is absent.")
                complete[int(task["ordinal"])] = checkpoint
                print(
                    f"[harp-v4] label-free classifier tasks {len(complete)}/{len(tasks)}",
                    flush=True,
                )
            max_inflight = (
                workstation.cpu_fit_workers
                * workstation.bounded_inflight_tasks_per_cpu_worker
            )
            bounded = execute_bounded(
                (pool,),
                pending,
                _classifier_task,
                executor_index=lambda _task: 0,
                max_inflight_per_executor=max_inflight,
                on_complete=accept_checkpoint,
            )
            if bounded.stats.max_total_inflight > max_inflight:
                raise ProtocolError("HARP v4 classifier submission bound drifted.")
    if set(complete) != set(range(len(tasks))):
        raise ProtocolError("HARP v4 classifier task coverage is incomplete.")
    return complete


def _classifier_task(task: Mapping[str, object]) -> None:
    body = {key: value for key, value in task.items() if key not in {"task_hash", "npz_path", "receipt_path"}}
    if (
        task.get("schema_version") != "midogpp_harp_v4_label_free_classifier_task_v1"
        or task.get("task_hash") != canonical_hash(body)
        or task.get("workstation_profile_hash") != _DEFAULT_WORKSTATION_PROFILE.profile_hash
        or task.get("threads_per_worker")
        != _DEFAULT_WORKSTATION_PROFILE.blas_threads_per_worker
        or task.get("labels_available") is not False
    ):
        raise ProtocolError("HARP v4 classifier task identity drifted.")
    actions = tuple(
        HarpActionSpec(
            surface_kind=str(raw["surface_kind"]),
            outer_target_id=str(raw["outer_target_id"]),
            query_center_id=str(raw["query_center_id"]),
            selected_source_id=(
                None if raw.get("selected_source_id") is None else str(raw["selected_source_id"])
            ),
            action_id=str(raw["action_id"]),
        )
        for raw in task["actions"]
    )
    source_values, frame, source_key = _load_worker_arrays(task)
    source_blocks = _load_source_blocks(
        actions,
        task,
        source_values=source_values,
        source_key=source_key,
    )
    start = int(task["frame_start"])
    stop = int(task["frame_stop"])
    if start < 0 or stop <= start or stop > len(frame):
        raise ProtocolError("HARP v4 frame slice geometry drifted.")
    evaluation = np.ascontiguousarray(
        frame[start:stop], dtype=np.float32
    )
    if not np.isfinite(evaluation).all():
        raise ProtocolError("HARP v4 frame slice contains nonfinite values.")
    classifier = ClassifierSpec(**dict(task["classifier"]))
    probabilities: list[np.ndarray] = []
    records_out: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP v4 classifier workers require threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_worker"])):
        for action in actions:
            composition = compose_harp_action(
                {source: source_blocks[source] for source in action.source_order},
                action,
                shuffle_seed_by_class={
                    label: harp_composition_seed(
                        generation_lock_hash=str(task["generation_lock_hash"]),
                        outer_target_id=str(task["outer_target_id"]),
                        query_center_id=str(task["query_center_id"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(task["generation_seed"]),
                        class_label=label,
                    )
                    for label in (0, 1)
                },
            )
            fitted = fit_logistic_classifier(
                composition.embeddings, composition.labels, evaluation, spec=classifier
            )
            values = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or values.shape != (len(evaluation), 2)
                or not fitted.converged
                or not np.isfinite(values).all()
                or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
            ):
                raise ProtocolError("HARP v4 physical classifier fit drifted.")
            positive = np.ascontiguousarray(values[:, 1], dtype=np.float32)
            probabilities.append(positive)
            records_out.append(
                {
                    "action_hash": _require_sha256(
                        action.action_hash, name="action hash"
                    ),
                    "composition_hash": _require_sha256(
                        composition.composition_hash, name="composition hash"
                    ),
                    "scaler_state_hash": _require_stable_hash(
                        fitted.scaler_state_hash, name="scaler-state hash"
                    ),
                    "probability_sha256": _require_sha256(
                        hashlib.sha256(positive.tobytes(order="C")).hexdigest(),
                        name="probability hash",
                    ),
                }
            )
    matrix = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["npz_path"]))
    atomic_npz(npz_path, probabilities=matrix)
    checkpoint_body = {
        "schema_version": "midogpp_harp_v4_label_free_classifier_checkpoint_v1",
        "status": "COMPLETE_LABEL_FREE",
        "task_hash": task["task_hash"],
        "npz_sha256": sha256_file(npz_path),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "action_count": len(records_out),
        "probability_row_count": int(matrix.shape[1]),
        "actions": records_out,
        "labels_consumed": False,
    }
    atomic_json(
        Path(str(task["receipt_path"])),
        {**checkpoint_body, "checkpoint_hash": canonical_hash(checkpoint_body)},
    )


def _load_task_checkpoint(task: Mapping[str, object]) -> Mapping[str, object] | None:
    task_body = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "npz_path", "receipt_path"}
    }
    if (
        task.get("schema_version") != "midogpp_harp_v4_label_free_classifier_task_v1"
        or task.get("task_hash") != canonical_hash(task_body)
        or task.get("workstation_profile_hash") != _DEFAULT_WORKSTATION_PROFILE.profile_hash
        or task.get("threads_per_worker")
        != _DEFAULT_WORKSTATION_PROFILE.blas_threads_per_worker
        or task.get("labels_available") is not False
    ):
        raise ProtocolError("HARP v4 classifier checkpoint task identity drifted.")
    if not isinstance(task.get("actions"), list) or not isinstance(
        task.get("sample_ids"), list
    ):
        raise ProtocolError("HARP v4 classifier checkpoint task dimensions are malformed.")
    receipt_path = Path(str(task["receipt_path"]))
    npz_path = Path(str(task["npz_path"]))
    if not receipt_path.exists() and not npz_path.exists():
        return None
    if not receipt_path.is_file() or not npz_path.is_file() or receipt_path.is_symlink() or npz_path.is_symlink():
        raise ProtocolError("HARP v4 partial classifier checkpoint is unsafe.")
    payload = read_json(receipt_path)
    body = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"probabilities"}:
                raise ProtocolError("HARP v4 classifier checkpoint inventory drifted.")
            values = np.asarray(archive["probabilities"])
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("HARP v4 classifier checkpoint could not be loaded.") from exc
    if (
        payload.get("schema_version")
        != "midogpp_harp_v4_label_free_classifier_checkpoint_v1"
        or payload.get("status") != "COMPLETE_LABEL_FREE"
        or payload.get("checkpoint_hash") != canonical_hash(body)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or values.dtype != np.float32
        or values.shape != (len(task["actions"]), len(task["sample_ids"]))
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or payload.get("shape") != list(values.shape)
        or payload.get("dtype") != "float32"
        or payload.get("action_count") != len(task["actions"])
        or payload.get("probability_row_count") != len(task["sample_ids"])
        or payload.get("labels_consumed") is not False
    ):
        raise ProtocolError("HARP v4 classifier checkpoint failed validation.")
    expected_actions = task.get("actions")
    observed_actions = payload.get("actions")
    if not isinstance(expected_actions, list) or not isinstance(observed_actions, list):
        raise ProtocolError("HARP v4 classifier checkpoint action inventory is malformed.")
    if len(observed_actions) != len(expected_actions):
        raise ProtocolError("HARP v4 classifier checkpoint action count drifted.")
    expected_hashes: list[str] = []
    for raw in expected_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v4 classifier task action is malformed.")
        action = HarpActionSpec(
            surface_kind=str(raw.get("surface_kind")),
            outer_target_id=str(raw.get("outer_target_id")),
            query_center_id=str(raw.get("query_center_id")),
            selected_source_id=(
                None if raw.get("selected_source_id") is None else str(raw["selected_source_id"])
            ),
            action_id=str(raw.get("action_id")),
        )
        if action.action_hash != raw.get("action_hash"):
            raise ProtocolError("HARP v4 classifier task action identity drifted.")
        expected_hashes.append(action.action_hash)
    observed_hashes: list[str] = []
    for index, raw in enumerate(observed_actions):
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v4 classifier checkpoint action is malformed.")
        if set(raw) != {
            "action_hash",
            "composition_hash",
            "scaler_state_hash",
            "probability_sha256",
        }:
            raise ProtocolError("HARP v4 classifier checkpoint action schema drifted.")
        try:
            _require_sha256(raw.get("action_hash"), name="action hash")
            _require_sha256(raw.get("composition_hash"), name="composition hash")
            _require_stable_hash(
                raw.get("scaler_state_hash"), name="scaler-state hash"
            )
            _require_sha256(raw.get("probability_sha256"), name="probability hash")
        except ProtocolError as exc:
            raise ProtocolError(
                "HARP v4 classifier checkpoint action hash is malformed."
            ) from exc

        observed_hashes.append(str(raw["action_hash"]))
        probability_hash = hashlib.sha256(
            np.ascontiguousarray(values[index], dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        if raw.get("probability_sha256") != probability_hash:
            raise ProtocolError("HARP v4 classifier checkpoint probability row drifted.")
    if observed_hashes != expected_hashes:
        raise ProtocolError("HARP v4 classifier checkpoint action order drifted.")
    return payload


def _aggregate_outer_menus(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    inputs: PhysicalInputReceipt,
) -> tuple[LabelFreeOuterMenu, ...]:
    cells: dict[str, list[np.ndarray]] = defaultdict(list)
    action_by_hash: dict[str, HarpActionSpec] = {}
    identities: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for task in tasks:
        ordinal = int(task["ordinal"])
        prior = completed.get(ordinal)
        checkpoint = _load_task_checkpoint(task)
        if (
            prior is None
            or checkpoint is None
            or checkpoint.get("checkpoint_hash") != prior.get("checkpoint_hash")
        ):
            raise ProtocolError(
                "HARP v4 classifier checkpoint changed before positional aggregation."
            )
        with np.load(Path(str(task["npz_path"])), allow_pickle=False) as archive:
            values = np.asarray(archive["probabilities"], dtype=np.float32)
        for index, raw in enumerate(task["actions"]):
            action = HarpActionSpec(
                surface_kind=str(raw["surface_kind"]),
                outer_target_id=str(raw["outer_target_id"]),
                query_center_id=str(raw["query_center_id"]),
                selected_source_id=(
                    None if raw.get("selected_source_id") is None else str(raw["selected_source_id"])
                ),
                action_id=str(raw["action_id"]),
            )
            action_by_hash[action.action_hash] = action
            cells[action.action_hash].append(np.ascontiguousarray(values[index], dtype=np.float32))
            identity = (
                tuple(str(value) for value in task["sample_ids"]),
                tuple(str(value) for value in task["case_ids"]),
            )
            previous = identities.setdefault(action.action_hash, identity)
            if previous != identity:
                raise ProtocolError("HARP v4 exact-nine action identities drifted.")
    menus: list[LabelFreeOuterMenu] = []
    for outer in CENTERS:
        blocks: list[LabelFreeActionBlock] = []
        for action in (row for row in _all_actions() if row.outer_target_id == outer):
            members = cells[action.action_hash]
            if len(members) != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP v4 action lacks exact-nine seed cells.")
            reduced = np.ascontiguousarray(
                np.mean(np.stack(members).astype(np.float64), axis=0, dtype=np.float64),
                dtype=np.float32,
            )
            sample_ids, case_ids = identities[action.action_hash]
            kind = (
                ActionKind.B
                if action.action_id == BASE_ACTION_ID
                else ActionKind.U
                if action.action_id == UNIFORM_ACTION_ID
                else ActionKind.HXE
            )
            blocks.append(
                LabelFreeActionBlock(
                    surface_role=(
                        "development"
                        if action.surface_kind == DEVELOPMENT_SURFACE
                        else "target"
                    ),
                    outer_target_id=outer,
                    query_center_id=action.query_center_id,
                    action_kind=kind,
                    selected_source_id=action.selected_source_id,
                    sample_ids=sample_ids,
                    case_ids=case_ids,
                    probabilities=reduced,
                )
            )
        blocks.sort(key=lambda block: block.key)
        menus.append(
            LabelFreeOuterMenu(
                outer_target_id=outer,
                blocks=tuple(blocks),
                lineage={
                    "physical_input_receipt_hash": inputs.receipt_hash,
                    "bank_hash": inputs.bank_hash,
                    "generation_hash": inputs.generation_hash,
                    "source_stream_lock_hash": _require_stable_hash(
                        tasks[0].get("source_stream_lock_hash"),
                        name="source-stream lock hash",
                    ),
                    "source_stream_lock_sha256": _require_sha256(
                        tasks[0].get("source_stream_lock_sha256"),
                        name="source-stream lock SHA-256",
                    ),
                    "source_stream_index_hash": _require_stable_hash(
                        tasks[0].get("source_stream_index_hash"),
                        name="source-stream index hash",
                    ),
                    "source_stream_index_sha256": _require_sha256(
                        tasks[0].get("source_index_sha256"),
                        name="source-stream index SHA-256",
                    ),
                    "frame_array_sha256": _require_sha256(
                        tasks[0].get("frame_array_sha256"),
                        name="frame-array SHA-256",
                    ),
                    "frame_receipt_hash": _require_stable_hash(
                        tasks[0].get("frame_receipt_hash"),
                        name="frame-receipt hash",
                    ),
                    "frame_receipt_sha256": _require_sha256(
                        tasks[0].get("frame_receipt_sha256"),
                        name="frame-receipt SHA-256",
                    ),
                    "classifier_hash": inputs.classifier.config_hash,
                    "exact_nine_seed_pairs": [list(value) for value in EXACT_NINE_SEED_PAIRS],
                    "reduction_dtype": "float64",
                    "transport_dtype": "float32",
                },
            )
        )
    return tuple(menus)


__all__ = (
    "PhysicalInputReceipt",
    "build_physical_plan",
    "materialize_physical_outer_menus",
    "validate_physical_inputs",
)
