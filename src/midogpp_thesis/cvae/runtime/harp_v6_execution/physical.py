"""V6-local production of the physical lambda=1 B/U/Hxe probability menu.

The adapter uses only frozen expert/generation primitives, the neutral HARP
action algebra, and the immutable v6 label-blind cache.  CUDA is confined to
the neutral two-worker frozen-source materializer.  That pool is closed before
the four-worker classifier pool is created.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation import read_generation_lock
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .gpu_surface import (
    ResidentExpertStreamCache,
    materialize_resident_expert_streams,
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
)
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
)
from ..bounded_futures import execute_bounded
from .classifier_worker_cache import (
    initialize_classifier_worker as _initialize_classifier_worker,
)
from .classifier_tasks import (
    execute_classifier_task as _classifier_task,
    load_classifier_task_checkpoint as _load_task_checkpoint,
)
from .contracts import ActionKind, LabelFreeActionBlock, LabelFreeOuterMenu
from .execution_profile import (
    DEFAULT_WORKSTATION_PROFILE as _DEFAULT_WORKSTATION_PROFILE,
    WorkstationProfile as _WorkstationProfile,
)
from .frame_binding import persist_or_validate_frame_binding
from .hash_contracts import (
    require_sha256 as _require_sha256,
    require_stable_hash as _require_stable_hash,
)
from .task_bindings import (
    validate_frame_task_binding,
    validate_source_task_binding,
)
from .physical_contracts import (
    PhysicalInputReceipt,
    SourceAdapter as _SourceAdapter,
    StagedFrames as _Frames,
)

def build_physical_plan() -> dict[str, object]:
    workstation = _DEFAULT_WORKSTATION_PROFILE
    actions = (*build_all_development_actions(), *build_all_target_actions())
    contexts = {
        (row.surface_kind, row.outer_target_id, row.query_center_id) for row in actions
    }
    body = {
        "schema_version": "midogpp_harp_v6_physical_plan_v1",
        "action_count": len(actions),
        "query_context_count": len(contexts),
        "classifier_task_count": len(contexts) * len(EXACT_NINE_SEED_PAIRS),
        "seed_cell_count": len(actions) * len(EXACT_NINE_SEED_PAIRS),
        "physical_expert_weight": 1.0,
        "probability_blends_present": False,
        "persistent_gpu_workers": workstation.persistent_gpu_workers,
        "gpu_devices": list(workstation.gpu_devices),
        "classifier_workers": workstation.cpu_fit_workers,
        "classifier_blas_threads_per_worker": workstation.blas_threads_per_worker,
        "science_workers": workstation.science_workers,
        "science_blas_threads_per_worker": (
            workstation.science_blas_threads_per_worker
        ),
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
        "bounded_inflight_classifier_tasks_per_worker": (
            workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "max_inflight_classifier_tasks": (
            workstation.cpu_fit_workers
            * workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "labels_consumed": False,
        "compatibility_computed_while_expert_resident": True,
        "evaluation_rows_consumed_for_compatibility": False,
        "workstation_profile_hash": workstation.profile_hash,
    }
    if (
        len(actions) != 738
        or len(contexts) != 81
        or body["classifier_task_count"] != 729
        or body["seed_cell_count"] != 6642
    ):
        raise ProtocolError("HARP v6 physical action topology drifted.")
    return {**body, "plan_hash": canonical_hash(body)}


def validate_physical_inputs(config: object, cache: object) -> PhysicalInputReceipt:
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for root, name in ((bank_root, "expert bank"), (generation_root, "generation lock")):
        if not root.is_dir() or root.is_symlink():
            raise ProtocolError(f"HARP v6 authoritative {name} root is unsafe.")
        state = read_json(root / "reports/run_state.json")
        validation = read_json(root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v6 authoritative {name} is not complete and valid.")
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
        raise ProtocolError("HARP v6 generation lock lacks bank/classifier bindings.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    if (
        bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v6 authoritative physical lineage drifted.")
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
        raise ProtocolError("HARP v6 classifier contract is malformed.") from exc
    if classifier.config_hash != raw_classifier.get("config_hash"):
        raise ProtocolError("HARP v6 classifier identity drifted.")
    cache_hash = str(getattr(cache, "cache_hash"))
    body = {
        "schema_version": "midogpp_harp_v6_physical_input_receipt_v1",
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
    outer_targets: Sequence[str] | None = None,
    scratch_root: Path,
    development_role: str,
    evaluation_role: str,
) -> tuple[LabelFreeOuterMenu, ...]:
    requested = CENTERS if outer_targets is None else tuple(str(value) for value in outer_targets)
    if (
        tuple(value for value in CENTERS if value in set(requested)) != requested
        or len(set(requested)) != len(requested)
    ):
        raise ProtocolError("HARP v6 pending outer-target subset is noncanonical.")
    if not requested:
        return ()
    inputs = validate_physical_inputs(config, cache)
    workstation = _WorkstationProfile.from_runtime(config.runtime)
    scratch_root.mkdir(parents=True, exist_ok=True)
    frames = _stage_frames(
        cache, scratch_root=scratch_root, roles=(development_role, evaluation_role)
    )
    source_adapter = _SourceAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v6_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": inputs.receipt_hash,
                "frame_sha256": frames.sha256,
                "workstation_profile_hash": workstation.profile_hash,
            }
        ),
        expert_bank_root=config.resolved_path("expert_bank_root"),
        runtime=workstation.source_runtime(),
    )
    support_binding = _support_binding(
        config,
        cache,
        frames=frames,
        development_role=development_role,
        evaluation_role=evaluation_role,
    )
    source_cache = materialize_resident_expert_streams(
        source_adapter,
        inputs.generation_lock,
        root=scratch_root / "source_streams",
        support_binding=support_binding,
    )
    tasks = _build_tasks(
        scratch_root=scratch_root,
        frames=frames,
        source_cache=source_cache,
        inputs=inputs,
        workstation=workstation,
        development_role=development_role,
        evaluation_role=evaluation_role,
        outer_targets=requested,
    )
    completed = _execute_tasks(tasks, workstation=workstation)
    return _aggregate_outer_menus(
        tasks, completed, inputs=inputs, outer_targets=requested
    )


def _support_binding(
    config: object,
    cache: object,
    *,
    frames: _Frames,
    development_role: str,
    evaluation_role: str,
) -> Mapping[str, object]:
    development_cases = {
        (row.center, row.case_id)
        for row in cache.rows
        if row.split_role == development_role
    }
    evaluation_cases = {
        (row.center, row.case_id)
        for row in cache.rows
        if row.split_role == evaluation_role
    }
    if development_cases & evaluation_cases:
        raise ProtocolError("HARP v6 support/evaluation case partition overlaps.")
    contexts = []
    for center in CENTERS:
        start, stop = frames.contexts[(development_role, center)]
        contexts.append(
            {
                "center": center,
                "frame_start": start,
                "frame_stop": stop,
                "case_ids": list(frames.case_ids[(development_role, center)]),
                "sample_ids_hash": canonical_hash(
                    list(frames.sample_ids[(development_role, center)])
                ),
            }
        )
    expected_hashes = getattr(config, "expected_hashes")
    body = {
        "schema_version": "midogpp_harp_v6_label_free_support_binding_v1",
        "frame_array_path": str(frames.path),
        "frame_array_sha256": frames.sha256,
        "frame_receipt_hash": frames.receipt_hash,
        "cache_hash": str(cache.cache_hash),
        "support_manifest_sha256": expected_hashes[
            "development_manifest_sha256"
        ],
        "support_role": development_role,
        "contexts": contexts,
        "support_evaluation_case_disjoint": True,
        "labels_present": False,
        "evaluation_rows_included": False,
    }
    return MappingProxyType(
        {**body, "support_binding_hash": canonical_hash(body)}
    )


def _stage_frames(cache: object, *, scratch_root: Path, roles: tuple[str, str]) -> _Frames:
    path = (scratch_root / "frames/consumed_rows.npy").resolve()
    receipt_path = (scratch_root / "frames/receipt.json").resolve()
    contexts: dict[tuple[str, str], tuple[int, int]] = {}
    samples: dict[tuple[str, str], tuple[str, ...]] = {}
    cases: dict[tuple[str, str], tuple[str, ...]] = {}
    cursor = 0
    load_embeddings = getattr(cache, "load_embeddings", None)
    if not callable(load_embeddings):
        raise ProtocolError("HARP v6 cache lacks the typed grouped-shard reader.")
    staged_rows: list[tuple[str, str, tuple[object, ...], int, int]] = []
    for role in roles:
        for center in CENTERS:
            rows = tuple(cache.rows_for(center, role))
            if not rows or tuple(row.split_row_index for row in rows) != tuple(range(len(rows))):
                raise ProtocolError("HARP v6 cache role geometry drifted.")
            start, stop = cursor, cursor + len(rows)
            contexts[(role, center)] = (start, stop)
            samples[(role, center)] = tuple(row.sample_id for row in rows)
            cases[(role, center)] = tuple(row.case_id for row in rows)
            staged_rows.append((role, center, rows, start, stop))
            cursor = stop
    if path.exists() != receipt_path.exists():
        raise ProtocolError("HARP v6 scratch frame store is only partially durable.")
    if path.exists():
        if (
            not path.is_file()
            or path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
        ):
            raise ProtocolError("HARP v6 scratch frame store paths are unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if observed.dtype != np.float32 or observed.shape != (cursor, COMMON_OUTPUT_DIM):
            raise ProtocolError("HARP v6 existing scratch frame store drifted.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            output = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(cursor, COMMON_OUTPUT_DIM),
            )
            for _role, _center, rows, start, stop in staged_rows:
                matrix = np.asarray(load_embeddings(rows))
                if (
                    matrix.dtype != np.float32
                    or matrix.shape != (stop - start, COMMON_OUTPUT_DIM)
                    or not np.isfinite(matrix).all()
                ):
                    raise ProtocolError("HARP v6 grouped frame geometry drifted.")
                output[start:stop] = matrix
            output.flush()
            del output
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    binding = persist_or_validate_frame_binding(
        array_path=path,
        receipt_path=receipt_path,
        shape=(cursor, COMMON_OUTPUT_DIM),
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


def _all_actions(
    outer_targets: Sequence[str] = CENTERS,
) -> tuple[HarpActionSpec, ...]:
    requested = set(outer_targets)
    return tuple(
        action
        for action in (*build_all_development_actions(), *build_all_target_actions())
        if action.outer_target_id in requested
    )


def _build_tasks(
    *,
    scratch_root: Path,
    frames: _Frames,
    source_cache: ResidentExpertStreamCache,
    inputs: PhysicalInputReceipt,
    workstation: _WorkstationProfile,
    development_role: str,
    evaluation_role: str,
    outer_targets: Sequence[str],
) -> tuple[dict[str, object], ...]:
    by_context: dict[tuple[str, str, str], list[HarpActionSpec]] = defaultdict(list)
    for action in _all_actions():
        by_context[(action.surface_kind, action.outer_target_id, action.query_center_id)].append(action)
    source_binding = validate_source_task_binding(source_cache)
    frame_binding = validate_frame_task_binding(frames)
    checkpoint_root = scratch_root / "classifier_checkpoints"
    tasks: list[dict[str, object]] = []
    requested = set(outer_targets)
    for global_ordinal, (surface, outer, query) in enumerate(sorted(by_context)):
        if outer not in requested:
            continue
        role = development_role if surface == DEVELOPMENT_SURFACE else evaluation_role
        start, stop = frames.contexts[(role, query)]
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            ordinal = global_ordinal * len(EXACT_NINE_SEED_PAIRS) + (
                EXACT_NINE_SEED_PAIRS.index((training_seed, generation_seed))
            )
            stem = f"task_{ordinal:04d}"
            body = {
                "schema_version": "midogpp_harp_v6_label_free_classifier_task_v1",
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
    if len(tasks) != 81 * len(tuple(outer_targets)):
        raise ProtocolError("HARP v6 classifier task coverage drifted.")
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
                    raise ProtocolError("HARP v6 classifier checkpoint is absent.")
                complete[int(task["ordinal"])] = checkpoint
                print(
                    f"[harp-v6] label-free classifier tasks {len(complete)}/{len(tasks)}",
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
                raise ProtocolError("HARP v6 classifier submission bound drifted.")
    if set(complete) != {int(task["ordinal"]) for task in tasks}:
        raise ProtocolError("HARP v6 classifier task coverage is incomplete.")
    return complete


def _aggregate_outer_menus(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    inputs: PhysicalInputReceipt,
    outer_targets: Sequence[str],
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
                "HARP v6 classifier checkpoint changed before positional aggregation."
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
                raise ProtocolError("HARP v6 exact-nine action identities drifted.")
    menus: list[LabelFreeOuterMenu] = []
    for outer in outer_targets:
        blocks: list[LabelFreeActionBlock] = []
        for action in (
            row for row in _all_actions(outer_targets) if row.outer_target_id == outer
        ):
            members = cells[action.action_hash]
            if len(members) != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP v6 action lacks exact-nine seed cells.")
            exact_nine = np.stack(members).astype(np.float64)
            reduced = np.ascontiguousarray(
                np.mean(exact_nine, axis=0, dtype=np.float64), dtype=np.float32
            )
            dispersion = np.ascontiguousarray(
                np.std(exact_nine, axis=0, dtype=np.float64),
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
                    seed_dispersion=dispersion,
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
                    "exact_nine_seed_dispersion_persisted": True,
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
