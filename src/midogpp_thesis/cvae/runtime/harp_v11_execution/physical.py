"""V11-local production of the physical lambda=1 B/U/Hxe probability menu.

The adapter uses only frozen expert/generation primitives, the neutral HARP
action algebra, and the immutable v11 label-blind cache.  CUDA is confined to
the neutral two-worker frozen-source materializer.  That pool is closed before
the four single-process classifier executors are created.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
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
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_file
from .action_capacity import (
    build_action_capacity_certificate,
    validate_action_capacity,
)
from .gpu_surface import (
    ResidentExpertStreamCache,
    materialize_resident_expert_streams,
)
from .physical_actions import (
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
from .contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    LabelFreeTargetMenu,
)
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
from .projection_hashing import projection_semantic_hash

def build_physical_plan() -> dict[str, object]:
    workstation = _DEFAULT_WORKSTATION_PROFILE
    classifier_pool = classifier_pool_plan(workstation)
    capacity = dict(build_action_capacity_certificate())
    actions = (*build_all_development_actions(), *build_all_target_actions())
    contexts = {
        (row.surface_kind, row.outer_target_id, row.query_center_id) for row in actions
    }
    body = {
        "schema_version": "midogpp_harp_v11_physical_plan_v1",
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
        "classifier_executor_count": classifier_pool["executor_count"],
        "classifier_processes_per_executor": classifier_pool[
            "processes_per_executor"
        ],
        "classifier_executor_assignment": classifier_pool["task_assignment"],
        "max_inflight_classifier_tasks": classifier_pool["max_total_inflight"],
        "labels_consumed": False,
        "compatibility_computed_while_expert_resident": True,
        "target_test_embeddings_used_for_case_local_compatibility": True,
        "evaluation_labels_consumed_for_compatibility": False,
        "workstation_profile_hash": workstation.profile_hash,
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "stream_rows_per_class": capacity["stream_rows_per_class"],
        "maximum_required_rows_per_class": capacity[
            "global_maximum_required_rows_per_class"
        ],
        "action_capacity_validated_before_scheduling": True,
    }
    if (
        len(actions) != 738
        or len(contexts) != 81
        or body["classifier_task_count"] != 729
        or body["seed_cell_count"] != 6642
    ):
        raise ProtocolError("HARP v11 physical action topology drifted.")
    return {**body, "plan_hash": canonical_hash(body)}


def build_target_only_physical_plan() -> dict[str, object]:
    """Plan the non-duplicative target C-{H} phase used with source crossfit."""

    capacity = dict(build_action_capacity_certificate())
    actions = build_all_target_actions()
    validate_action_capacity(actions)
    contexts = {
        (row.outer_target_id, row.query_center_id) for row in actions
    }
    body = {
        "schema_version": "midogpp_harp_v11_target_only_physical_plan_v1",
        "outer_target_count": len(CENTERS),
        "query_context_count": len(contexts),
        "action_count": len(actions),
        "classifier_task_count": len(contexts) * len(EXACT_NINE_SEED_PAIRS),
        "seed_cell_count": len(actions) * len(EXACT_NINE_SEED_PAIRS),
        "source_development_classifier_tasks_scheduled": 0,
        "source_crossfit_prediction_blocks_reused": True,
        "target_source_pool": "C_MINUS_H",
        "action_capacity_certificate_hash": capacity[
            "capacity_certificate_hash"
        ],
        "stream_rows_per_class": capacity["stream_rows_per_class"],
        "target_maximum_required_rows_per_class": capacity[
            "required_rows_per_class_by_surface"
        ]["target"],
        "action_capacity_validated_before_scheduling": True,
        "labels_consumed": False,
    }
    if (
        len(contexts) != 9
        or len(actions) != 90
        or body["classifier_task_count"] != 81
        or body["seed_cell_count"] != 810
    ):
        raise ProtocolError("HARP v11 target-only topology drifted.")
    return {**body, "plan_hash": canonical_hash(body)}


def classifier_pool_plan(
    workstation: _WorkstationProfile = _DEFAULT_WORKSTATION_PROFILE,
) -> dict[str, object]:
    """Return the executable classifier-pool topology.

    A one-process executor is the scheduling unit.  This makes the configured
    submission bound genuinely per worker: no executor can hide several
    worker processes behind a single aggregate queue.
    """

    body: dict[str, object] = {
        "schema_version": "midogpp_harp_v11_classifier_pool_plan_v1",
        "executor_count": workstation.cpu_fit_workers,
        "processes_per_executor": 1,
        "total_worker_processes": workstation.cpu_fit_workers,
        "blas_threads_per_process": workstation.blas_threads_per_worker,
        "multiprocessing_start_method": workstation.multiprocessing_start_method,
        "max_inflight_per_executor": (
            workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "max_total_inflight": (
            workstation.cpu_fit_workers
            * workstation.bounded_inflight_tasks_per_cpu_worker
        ),
        "task_assignment": "ordinal_modulo_executor_count",
    }
    if (
        body["executor_count"] != 4
        or body["processes_per_executor"] != 1
        or body["total_worker_processes"] != 4
        or body["blas_threads_per_process"] != 3
        or body["multiprocessing_start_method"] != "spawn"
        or body["max_inflight_per_executor"] != 2
        or body["max_total_inflight"] != 8
    ):
        raise ProtocolError("HARP v11 classifier-pool topology drifted.")
    return {**body, "plan_hash": canonical_hash(body)}


def _classifier_executor_index(
    task: Mapping[str, object], *, executor_count: int
) -> int:
    ordinal = task.get("ordinal")
    if (
        type(ordinal) is not int
        or type(executor_count) is not int
        or executor_count < 1
    ):
        raise ProtocolError("HARP v11 classifier executor assignment is malformed.")
    return ordinal % executor_count


def validate_physical_inputs(config: object, cache: object) -> PhysicalInputReceipt:
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for root, name in ((bank_root, "expert bank"), (generation_root, "generation lock")):
        if not root.is_dir() or root.is_symlink():
            raise ProtocolError(f"HARP v11 authoritative {name} root is unsafe.")
        state = read_json(root / "reports/run_state.json")
        validation = read_json(root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v11 authoritative {name} is not complete and valid.")
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
        raise ProtocolError("HARP v11 generation lock lacks bank/classifier bindings.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    if (
        bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v11 authoritative physical lineage drifted.")
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
        raise ProtocolError("HARP v11 classifier contract is malformed.") from exc
    if classifier.config_hash != raw_classifier.get("config_hash"):
        raise ProtocolError("HARP v11 classifier identity drifted.")
    cache_hash = str(getattr(cache, "cache_hash"))
    body = {
        "schema_version": "midogpp_harp_v11_physical_input_receipt_v1",
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
        raise ProtocolError("HARP v11 pending outer-target subset is noncanonical.")
    if not requested:
        return ()
    # Prove the complete target/source physical graph before creating scratch
    # state or starting either GPU or CPU executors.
    build_action_capacity_certificate(centers=requested)
    inputs = validate_physical_inputs(config, cache)
    workstation = _WorkstationProfile.from_runtime(config.runtime)
    scratch_root.mkdir(parents=True, exist_ok=True)
    frames = _stage_frames(
        config,
        cache,
        inputs=inputs,
        scratch_root=scratch_root,
        roles=(development_role, evaluation_role),
    )
    source_adapter = _SourceAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v11_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": inputs.receipt_hash,
                "frame_sha256": frames.sha256,
                "frame_provenance_hash": frames.provenance_hash,
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


def materialize_physical_target_menus(
    config: object,
    cache: object,
    *,
    outer_targets: Sequence[str] | None = None,
    scratch_root: Path,
    development_role: str,
    evaluation_role: str,
) -> tuple[LabelFreeTargetMenu, ...]:
    """Materialize only nine C-{H} target contexts (81 seed tasks total).

    Source H/q/r blocks are supplied by the separate crossfit surface, so this
    path deliberately never schedules the legacy 729 ordinary source tasks.
    The staged frames and resident expert stream store use the same roots as
    crossfit and are therefore hash-validated/reused on workstation reruns.
    """

    requested = (
        CENTERS if outer_targets is None else tuple(str(value) for value in outer_targets)
    )
    if (
        tuple(value for value in CENTERS if value in set(requested)) != requested
        or len(set(requested)) != len(requested)
    ):
        raise ProtocolError("HARP v11 target-only outer subset is noncanonical.")
    if not requested:
        return ()
    build_action_capacity_certificate(centers=requested)
    inputs = validate_physical_inputs(config, cache)
    workstation = _WorkstationProfile.from_runtime(config.runtime)
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    frames = _stage_frames(
        config,
        cache,
        inputs=inputs,
        scratch_root=scratch_root,
        roles=(development_role, evaluation_role),
    )
    source_adapter = _SourceAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v11_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": inputs.receipt_hash,
                "frame_sha256": frames.sha256,
                "frame_provenance_hash": frames.provenance_hash,
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
    tasks = _build_target_tasks(
        scratch_root=scratch_root,
        frames=frames,
        source_cache=source_cache,
        inputs=inputs,
        workstation=workstation,
        evaluation_role=evaluation_role,
        outer_targets=requested,
    )
    completed = _execute_tasks(tasks, workstation=workstation)
    return _aggregate_target_menus(
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
        raise ProtocolError("HARP v11 support/evaluation case partition overlaps.")
    contexts = []
    for role in (development_role, evaluation_role):
        for center in CENTERS:
            start, stop = frames.contexts[(role, center)]
            contexts.append(
                {
                    "role": role,
                    "center": center,
                    "frame_start": start,
                    "frame_stop": stop,
                    "case_ids": list(frames.case_ids[(role, center)]),
                    "sample_ids_hash": canonical_hash(
                        list(frames.sample_ids[(role, center)])
                    ),
                    "case_ids_hash": canonical_hash(
                        list(frames.case_ids[(role, center)])
                    ),
                }
            )
    expected_hashes = getattr(config, "expected_hashes")
    body = {
        "schema_version": "midogpp_harp_v11_role_qualified_label_free_binding_v2",
        "frame_array_path": str(frames.path),
        "frame_array_sha256": frames.sha256,
        "frame_provenance_hash": frames.provenance_hash,
        "frame_receipt_hash": frames.receipt_hash,
        "frame_receipt_sha256": frames.receipt_sha256,
        "cache_hash": str(cache.cache_hash),
        "cache_content_sha256": str(cache.content_sha256),
        "config_hash": str(config.config_hash),
        "protocol_hash": canonical_hash(dict(config.protocol)),
        "support_manifest_sha256": expected_hashes[
            "development_manifest_sha256"
        ],
        "evaluation_manifest_sha256": expected_hashes[
            "evaluation_manifest_sha256"
        ],
        "source_role": development_role,
        "target_role": evaluation_role,
        "contexts": contexts,
        "source_train_target_test_case_disjoint": True,
        "labels_present": False,
        "source_train_embeddings_included": True,
        "target_test_embeddings_included": True,
        "target_test_embeddings_case_local_only": True,
        "evaluation_labels_included": False,
    }
    return MappingProxyType(
        {**body, "support_binding_hash": canonical_hash(body)}
    )


def _stage_frames(
    config: object,
    cache: object,
    *,
    inputs: PhysicalInputReceipt,
    scratch_root: Path,
    roles: tuple[str, str],
) -> _Frames:
    path = (scratch_root / "frames/consumed_rows.npy").resolve()
    receipt_path = (scratch_root / "frames/receipt.json").resolve()
    contexts: dict[tuple[str, str], tuple[int, int]] = {}
    samples: dict[tuple[str, str], tuple[str, ...]] = {}
    cases: dict[tuple[str, str], tuple[str, ...]] = {}
    cursor = 0
    load_embeddings = getattr(cache, "load_embeddings", None)
    if not callable(load_embeddings):
        raise ProtocolError("HARP v11 cache lacks the typed grouped-shard reader.")
    staged_rows: list[tuple[str, str, tuple[object, ...], int, int]] = []
    provenance_contexts: list[dict[str, object]] = []
    ordered_rows: list[dict[str, object]] = []
    ordered_samples: list[dict[str, str]] = []
    ordered_cases: list[dict[str, str]] = []
    for role in roles:
        for center in CENTERS:
            rows = tuple(cache.rows_for(center, role))
            if not rows or tuple(row.split_row_index for row in rows) != tuple(range(len(rows))):
                raise ProtocolError("HARP v11 cache role geometry drifted.")
            start, stop = cursor, cursor + len(rows)
            contexts[(role, center)] = (start, stop)
            samples[(role, center)] = tuple(row.sample_id for row in rows)
            cases[(role, center)] = tuple(row.case_id for row in rows)
            staged_rows.append((role, center, rows, start, stop))
            row_identities = [
                {
                    "role": role,
                    "center": center,
                    "case_id": str(row.case_id),
                    "sample_id": str(row.sample_id),
                    "split_row_index": int(row.split_row_index),
                    "embedding_file": str(row.embedding_file),
                    "embedding_row_index": int(row.embedding_row_index),
                }
                for row in rows
            ]
            sample_identities = [
                {"role": role, "center": center, "sample_id": str(row.sample_id)}
                for row in rows
            ]
            case_identities = [
                {"role": role, "center": center, "case_id": str(row.case_id)}
                for row in rows
            ]
            provenance_contexts.append(
                {
                    "role": role,
                    "center": center,
                    "frame_start": start,
                    "frame_stop": stop,
                    "row_count": len(rows),
                    "row_identity_hash": canonical_hash(row_identities),
                    "sample_ids_hash": canonical_hash(sample_identities),
                    "case_ids_hash": canonical_hash(case_identities),
                }
            )
            ordered_rows.extend(row_identities)
            ordered_samples.extend(sample_identities)
            ordered_cases.extend(case_identities)
            cursor = stop
    protocol = getattr(config, "protocol", None)
    if not isinstance(protocol, Mapping) or protocol.get("feature_backbone") != "Virchow2_3840":
        raise ProtocolError("HARP v11 frame protocol provenance drifted.")
    provenance = {
        "schema_version": "midogpp_harp_v11_scratch_frame_provenance_v1",
        "cache_index_hash": _require_sha256(
            str(getattr(cache, "cache_hash", "")), name="frame cache-index hash"
        ),
        "cache_content_sha256": _require_sha256(
            str(getattr(cache, "content_sha256", "")),
            name="frame cache-content hash",
        ),
        "config_hash": _require_sha256(
            getattr(config, "config_hash", None), name="frame config hash"
        ),
        "protocol_hash": canonical_hash(dict(protocol)),
        "physical_input_receipt_hash": _require_sha256(
            inputs.receipt_hash, name="frame physical-input receipt hash"
        ),
        "representation_id": "midogpp_virchow2_common_3840_float32_v1",
        "feature_backbone": "Virchow2_3840",
        "roles": list(roles),
        "centers": list(CENTERS),
        "contexts": provenance_contexts,
        "ordered_row_identity_hash": canonical_hash(ordered_rows),
        "ordered_sample_identity_hash": canonical_hash(ordered_samples),
        "ordered_case_identity_hash": canonical_hash(ordered_cases),
        "row_count": cursor,
        "output_dim": COMMON_OUTPUT_DIM,
        "dtype": "float32",
        "labels_stored": False,
    }
    if path.exists() != receipt_path.exists():
        raise ProtocolError("HARP v11 scratch frame store is only partially durable.")
    created = not path.exists()
    if not created:
        if (
            not path.is_file()
            or path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
        ):
            raise ProtocolError("HARP v11 scratch frame store paths are unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if observed.dtype != np.float32 or observed.shape != (cursor, COMMON_OUTPUT_DIM):
            raise ProtocolError("HARP v11 existing scratch frame store drifted.")
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
                    raise ProtocolError("HARP v11 grouped frame geometry drifted.")
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
        provenance=provenance,
        receipt_creation_authorized=created,
    )
    return _Frames(
        path=path,
        receipt_path=receipt_path,
        contexts=MappingProxyType(contexts),
        sample_ids=MappingProxyType(samples),
        case_ids=MappingProxyType(cases),
        sha256=binding.array_sha256,
        provenance_hash=binding.provenance_hash,
        receipt_hash=binding.receipt_hash,
        receipt_sha256=binding.receipt_sha256,
    )


def _persist_query_frame_projection(
    root: Path,
    *,
    frames: _Frames,
    role: str,
    query_center_id: str,
) -> Mapping[str, object]:
    """Persist one role/query frame with no other center rows visible."""

    start, stop = frames.contexts[(role, query_center_id)]
    source = np.load(frames.path, mmap_mode="r", allow_pickle=False)
    values = np.ascontiguousarray(source[start:stop], dtype=np.float32)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{role}_{query_center_id}.npy"
    receipt_path = root / f"{role}_{query_center_id}.json"
    if path.exists():
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.dtype != np.float32
            or observed.shape != values.shape
            or not np.array_equal(observed, values)
        ):
            raise ProtocolError("HARP v11 query frame projection drifted.")
    else:
        atomic_npy(path, values)
    body = {
        "schema_version": "midogpp_harp_v11_query_frame_projection_v1",
        "parent_frame_sha256": frames.sha256,
        "parent_frame_receipt_hash": frames.receipt_hash,
        "role": role,
        "query_center_id": query_center_id,
        "sample_ids": list(frames.sample_ids[(role, query_center_id)]),
        "case_ids": list(frames.case_ids[(role, query_center_id)]),
        "shape": list(values.shape),
        "dtype": "float32",
        "array_sha256": sha256_file(path),
        "other_center_rows_visible": False,
        "labels_consumed": False,
    }
    receipt = {
        **body,
        "frame_projection_hash": projection_semantic_hash(
            body, name="frame projection hash"
        ),
    }
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ProtocolError("HARP v11 query frame receipt drifted.")
    else:
        atomic_json(receipt_path, receipt)
    return MappingProxyType(
        {
            "path": path.resolve(),
            "sha256": body["array_sha256"],
            "receipt_path": receipt_path.resolve(),
            "receipt_hash": receipt["frame_projection_hash"],
            "receipt_sha256": sha256_file(receipt_path),
            "sample_ids": tuple(body["sample_ids"]),
            "case_ids": tuple(body["case_ids"]),
        }
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
    for actions in by_context.values():
        validate_action_capacity(tuple(actions))
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
                "schema_version": "midogpp_harp_v11_label_free_classifier_task_v1",
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
        raise ProtocolError("HARP v11 classifier task coverage drifted.")
    return tuple(tasks)


def _build_target_tasks(
    *,
    scratch_root: Path,
    frames: _Frames,
    source_cache: ResidentExpertStreamCache,
    inputs: PhysicalInputReceipt,
    workstation: _WorkstationProfile,
    evaluation_role: str,
    outer_targets: Sequence[str],
) -> tuple[dict[str, object], ...]:
    source_binding = validate_source_task_binding(source_cache)
    actions_by_outer: dict[str, list[HarpActionSpec]] = defaultdict(list)
    requested = tuple(outer_targets)
    for action in build_all_target_actions():
        if action.outer_target_id in set(requested):
            actions_by_outer[action.outer_target_id].append(action)
    checkpoint_root = scratch_root / "target_classifier_checkpoints"
    tasks: list[dict[str, object]] = []
    ordinal = 0
    for outer in requested:
        actions = tuple(actions_by_outer[outer])
        validate_action_capacity(actions)
        if len(actions) != 10 or any(
            action.surface_kind != TARGET_SURFACE
            or action.query_center_id != outer
            for action in actions
        ):
            raise ProtocolError("HARP v11 target-only action slate drifted.")
        frame_projection = _persist_query_frame_projection(
            scratch_root / "target_frame_projections",
            frames=frames,
            role=evaluation_role,
            query_center_id=outer,
        )
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            stem = f"H{outer}_t{training_seed}_g{generation_seed}"
            body = {
                "schema_version": "midogpp_harp_v11_label_free_classifier_task_v1",
                "ordinal": ordinal,
                "surface_kind": TARGET_SURFACE,
                "outer_target_id": outer,
                "query_center_id": outer,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "actions": [row.to_payload() for row in actions],
                "target_only_execution": True,
                "source_development_tasks_scheduled": False,
                "source_array_path": str(source_binding.array_path),
                "source_array_sha256": source_binding.array_sha256,
                "source_index_path": str(source_binding.index_path),
                "source_index_sha256": source_binding.index_sha256,
                "source_stream_index_hash": source_binding.index_hash,
                "source_records": list(source_binding.records),
                "frame_array_path": str(frame_projection["path"]),
                "frame_array_sha256": frame_projection["sha256"],
                "frame_receipt_path": str(frame_projection["receipt_path"]),
                "frame_receipt_hash": frame_projection["receipt_hash"],
                "frame_receipt_sha256": frame_projection["receipt_sha256"],
                "frame_projection_schema": (
                    "midogpp_harp_v11_query_frame_projection_v1"
                ),
                "frame_projection_hash": frame_projection["receipt_hash"],
                "frame_start": 0,
                "frame_stop": len(frame_projection["sample_ids"]),
                "sample_ids": list(frames.sample_ids[(evaluation_role, outer)]),
                "case_ids": list(frames.case_ids[(evaluation_role, outer)]),
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
            ordinal += 1
    if len(tasks) != len(requested) * len(EXACT_NINE_SEED_PAIRS):
        raise ProtocolError("HARP v11 target-only classifier coverage drifted.")
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
        pool_plan = classifier_pool_plan(workstation)
        executor_count = int(pool_plan["executor_count"])
        max_inflight_per_executor = int(pool_plan["max_inflight_per_executor"])
        with ExitStack() as stack:
            pools = tuple(
                stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=mp.get_context(
                            workstation.multiprocessing_start_method
                        ),
                        initializer=_initialize_classifier_worker,
                        initargs=(workstation.blas_threads_per_worker,),
                    )
                )
                for _ in range(executor_count)
            )

            def accept_checkpoint(
                _position: int, task: Mapping[str, object], _result: None
            ) -> None:
                checkpoint = _load_task_checkpoint(task)
                if checkpoint is None:
                    raise ProtocolError("HARP v11 classifier checkpoint is absent.")
                complete[int(task["ordinal"])] = checkpoint
                print(
                    f"[harp-v11] label-free classifier tasks {len(complete)}/{len(tasks)}",
                    flush=True,
                )

            bounded = execute_bounded(
                pools,
                pending,
                _classifier_task,
                executor_index=lambda task: _classifier_executor_index(
                    task, executor_count=executor_count
                ),
                max_inflight_per_executor=max_inflight_per_executor,
                on_complete=accept_checkpoint,
            )
            if (
                len(bounded.stats.max_inflight_by_executor) != executor_count
                or any(
                    value > max_inflight_per_executor
                    for value in bounded.stats.max_inflight_by_executor
                )
                or bounded.stats.max_total_inflight
                > int(pool_plan["max_total_inflight"])
            ):
                raise ProtocolError("HARP v11 classifier submission bound drifted.")
    if set(complete) != {int(task["ordinal"]) for task in tasks}:
        raise ProtocolError("HARP v11 classifier task coverage is incomplete.")
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
                "HARP v11 classifier checkpoint changed before positional aggregation."
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
                raise ProtocolError("HARP v11 exact-nine action identities drifted.")
    menus: list[LabelFreeOuterMenu] = []
    for outer in outer_targets:
        blocks: list[LabelFreeActionBlock] = []
        for action in (
            row for row in _all_actions(outer_targets) if row.outer_target_id == outer
        ):
            members = cells[action.action_hash]
            if len(members) != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP v11 action lacks exact-nine seed cells.")
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


def _aggregate_target_menus(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    inputs: PhysicalInputReceipt,
    outer_targets: Sequence[str],
) -> tuple[LabelFreeTargetMenu, ...]:
    cells: dict[str, list[np.ndarray]] = defaultdict(list)
    actions: dict[str, HarpActionSpec] = {}
    identities: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for task in tasks:
        prior = completed.get(int(task["ordinal"]))
        checkpoint = _load_task_checkpoint(task)
        if (
            prior is None
            or checkpoint is None
            or checkpoint.get("checkpoint_hash") != prior.get("checkpoint_hash")
        ):
            raise ProtocolError(
                "HARP v11 target checkpoint changed before aggregation."
            )
        with np.load(Path(str(task["npz_path"])), allow_pickle=False) as archive:
            values = np.asarray(archive["probabilities"], dtype=np.float32)
        for index, raw in enumerate(task["actions"]):
            action = HarpActionSpec(
                surface_kind=str(raw["surface_kind"]),
                outer_target_id=str(raw["outer_target_id"]),
                query_center_id=str(raw["query_center_id"]),
                selected_source_id=(
                    None
                    if raw.get("selected_source_id") is None
                    else str(raw["selected_source_id"])
                ),
                action_id=str(raw["action_id"]),
            )
            if (
                action.surface_kind != TARGET_SURFACE
                or action.query_center_id != action.outer_target_id
            ):
                raise ProtocolError("HARP v11 non-target action entered target-only store.")
            actions[action.action_hash] = action
            cells[action.action_hash].append(
                np.ascontiguousarray(values[index], dtype=np.float32)
            )
            identity = (
                tuple(str(value) for value in task["sample_ids"]),
                tuple(str(value) for value in task["case_ids"]),
            )
            if identities.setdefault(action.action_hash, identity) != identity:
                raise ProtocolError("HARP v11 target exact-nine identities drifted.")
    output: list[LabelFreeTargetMenu] = []
    for outer in outer_targets:
        blocks: list[LabelFreeActionBlock] = []
        scoped = tuple(
            action
            for action in build_all_target_actions()
            if action.outer_target_id == outer
        )
        for action in scoped:
            members = cells.get(action.action_hash, ())
            if len(members) != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP v11 target action lacks exact-nine cells.")
            exact_nine = np.stack(members).astype(np.float64)
            sample_ids, case_ids = identities[action.action_hash]
            blocks.append(
                LabelFreeActionBlock(
                    surface_role="target",
                    outer_target_id=outer,
                    query_center_id=outer,
                    action_kind=(
                        ActionKind.B
                        if action.action_id == BASE_ACTION_ID
                        else ActionKind.U
                        if action.action_id == UNIFORM_ACTION_ID
                        else ActionKind.HXE
                    ),
                    selected_source_id=action.selected_source_id,
                    sample_ids=sample_ids,
                    case_ids=case_ids,
                    probabilities=np.ascontiguousarray(
                        np.mean(exact_nine, axis=0, dtype=np.float64),
                        dtype=np.float32,
                    ),
                    seed_dispersion=np.ascontiguousarray(
                        np.std(exact_nine, axis=0, dtype=np.float64),
                        dtype=np.float32,
                    ),
                )
            )
        blocks.sort(key=lambda row: row.key)
        first_task = next(
            task for task in tasks if task.get("outer_target_id") == outer
        )
        output.append(
            LabelFreeTargetMenu(
                outer_target_id=outer,
                blocks=tuple(blocks),
                lineage={
                    "physical_input_receipt_hash": inputs.receipt_hash,
                    "bank_hash": inputs.bank_hash,
                    "generation_hash": inputs.generation_hash,
                    "source_stream_lock_hash": _require_stable_hash(
                        first_task.get("source_stream_lock_hash"),
                        name="source-stream lock hash",
                    ),
                    "source_stream_lock_sha256": _require_sha256(
                        first_task.get("source_stream_lock_sha256"),
                        name="source-stream lock SHA-256",
                    ),
                    "source_stream_index_hash": _require_stable_hash(
                        first_task.get("source_stream_index_hash"),
                        name="source-stream index hash",
                    ),
                    "source_stream_index_sha256": _require_sha256(
                        first_task.get("source_index_sha256"),
                        name="source-stream index SHA-256",
                    ),
                    "frame_array_sha256": _require_sha256(
                        first_task.get("frame_array_sha256"),
                        name="frame-array SHA-256",
                    ),
                    "frame_receipt_hash": _require_stable_hash(
                        first_task.get("frame_receipt_hash"),
                        name="frame-receipt hash",
                    ),
                    "frame_receipt_sha256": _require_sha256(
                        first_task.get("frame_receipt_sha256"),
                        name="frame-receipt SHA-256",
                    ),
                    "classifier_hash": inputs.classifier.config_hash,
                    "exact_nine_seed_pairs": [
                        list(value) for value in EXACT_NINE_SEED_PAIRS
                    ],
                    "target_only_execution": True,
                    "ordinary_source_classifier_tasks_scheduled": 0,
                    "target_only_plan_hash": build_target_only_physical_plan()[
                        "plan_hash"
                    ],
                    "reduction_dtype": "float64",
                    "transport_dtype": "float32",
                },
            )
        )
    return tuple(output)


__all__ = (
    "PhysicalInputReceipt",
    "build_physical_plan",
    "build_target_only_physical_plan",
    "classifier_pool_plan",
    "materialize_physical_outer_menus",
    "materialize_physical_target_menus",
    "validate_physical_inputs",
)
