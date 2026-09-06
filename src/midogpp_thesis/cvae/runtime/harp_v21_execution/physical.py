"""V21-local production of the physical lambda=1 B/U/Hxe probability menu.

The adapter uses only frozen expert/generation primitives, the neutral HARP
action algebra, and the immutable v21 label-blind cache.  CUDA is confined to
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
    SUPPORT_SURFACE,
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
from .progress import classifier_progress_due
from .support_independence import audit_fixed_bank_support_independence


def build_physical_plan() -> dict[str, object]:
    workstation = _DEFAULT_WORKSTATION_PROFILE
    classifier_pool = classifier_pool_plan(workstation)
    capacity = dict(build_action_capacity_certificate())
    actions = (*build_all_development_actions(), *build_all_target_actions())
    role_contexts = {
        (row.surface_kind, row.outer_target_id, row.query_center_id) for row in actions
    }
    fit_contexts = {(row.outer_target_id, row.query_center_id) for row in actions}
    support_contexts = {
        context for context in role_contexts if context[0] == SUPPORT_SURFACE
    }
    target_contexts = {
        context for context in role_contexts if context[0] == TARGET_SURFACE
    }
    body = {
        "schema_version": "midogpp_harp_v21_physical_plan_v1",
        "action_count": len(actions),
        "role_query_context_count": len(role_contexts),
        "source_train_context_count": len(support_contexts),
        "support_context_count": len(support_contexts),
        "target_context_count": len(target_contexts),
        "candidate_count_per_context": len(CENTERS) - 1,
        "stream_job_count": len(CENTERS) * 3,
        "classifier_fit_context_count": len(fit_contexts),
        "classifier_task_count": len(fit_contexts) * len(EXACT_NINE_SEED_PAIRS),
        "classifier_fit_count": len(fit_contexts) * len(EXACT_NINE_SEED_PAIRS) * 10,
        "seed_cell_count": len(actions) * len(EXACT_NINE_SEED_PAIRS),
        "joint_support_target_prediction": True,
        "classifier_fit_reused_across_support_and_target": True,
        "physical_expert_weight": 1.0,
        "probability_blends_present": False,
        "soft_topk_gpu_task_count": 0,
        "soft_arm_gpu_task_count": 0,
        "soft_topk_classifier_task_count": 0,
        "soft_topk_classifier_fit_count": 0,
        "soft_arm_classifier_fit_count": 0,
        "soft_topk_composition_device": "cpu_vectorized",
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
        "patch_feature_backbone": "Virchow2_3840",
        "patch_feature_dimension": 3840,
        "patch_features_dtype": "float32",
        "patch_feature_reduction_fitted": False,
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
        len(actions) != 180
        or len(role_contexts) != 18
        or len(support_contexts) != 9
        or len(target_contexts) != 9
        or len(fit_contexts) != 9
        or body["classifier_task_count"] != 81
        or body["classifier_fit_count"] != 810
        or body["seed_cell_count"] != 1620
    ):
        raise ProtocolError("HARP v21 physical action topology drifted.")
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
        "schema_version": "midogpp_harp_v21_classifier_pool_plan_v1",
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
        raise ProtocolError("HARP v21 classifier-pool topology drifted.")
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
        raise ProtocolError("HARP v21 classifier executor assignment is malformed.")
    return ordinal % executor_count


def validate_physical_inputs(config: object, cache: object) -> PhysicalInputReceipt:
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for root, name in ((bank_root, "expert bank"), (generation_root, "generation lock")):
        if not root.is_dir() or root.is_symlink():
            raise ProtocolError(f"HARP v21 authoritative {name} root is unsafe.")
        state = read_json(root / "reports/run_state.json")
        validation = read_json(root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP v21 authoritative {name} is not complete and valid.")
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
        raise ProtocolError("HARP v21 generation lock lacks bank/classifier bindings.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    if (
        bank_lock_hash != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock_hash != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP v21 authoritative physical lineage drifted.")
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
        raise ProtocolError("HARP v21 classifier contract is malformed.") from exc
    if classifier.config_hash != raw_classifier.get("config_hash"):
        raise ProtocolError("HARP v21 classifier identity drifted.")
    independence = audit_fixed_bank_support_independence(
        bank_root=bank_root,
        bank_payload=bank_payload,
        generation_payload=lock_payload,
        bank_index_sha256=bank_sha,
        generation_lock_sha256=generation_sha,
    )
    cache_hash = str(getattr(cache, "cache_hash"))
    body = {
        "schema_version": "midogpp_harp_v21_physical_input_receipt_v1",
        "bank_hash": bank_lock_hash,
        "generation_hash": generation_lock_hash,
        "bank_index_sha256": bank_sha,
        "generation_file_sha256": generation_sha,
        "cache_hash": cache_hash,
        "classifier_hash": classifier.config_hash,
        "bank_independence_attestation_hash": independence.attestation_hash,
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
        support_independence=independence,
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
        raise ProtocolError("HARP v21 pending outer-target subset is noncanonical.")
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
                "schema_version": "midogpp_harp_v21_source_runtime_binding_v1",
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
        raise ProtocolError("HARP v21 support/evaluation case partition overlaps.")
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
        "schema_version": "midogpp_harp_v21_role_qualified_label_free_binding_v2",
        "frame_array_path": str(frames.path),
        "frame_array_sha256": frames.sha256,
        "frame_provenance_hash": frames.provenance_hash,
        "frame_receipt_hash": frames.receipt_hash,
        "frame_receipt_sha256": frames.receipt_sha256,
        "cache_hash": str(cache.cache_hash),
        "cache_content_sha256": str(cache.content_sha256),
        "config_hash": str(config.config_hash),
        "protocol_hash": canonical_hash(dict(config.protocol)),
        "source_train_manifest_sha256": expected_hashes[
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
        raise ProtocolError("HARP v21 cache lacks the typed grouped-shard reader.")
    staged_rows: list[tuple[str, str, tuple[object, ...], int, int]] = []
    provenance_contexts: list[dict[str, object]] = []
    ordered_rows: list[dict[str, object]] = []
    ordered_samples: list[dict[str, str]] = []
    ordered_cases: list[dict[str, str]] = []
    for role in roles:
        for center in CENTERS:
            rows = tuple(cache.rows_for(center, role))
            if not rows or tuple(row.split_row_index for row in rows) != tuple(range(len(rows))):
                raise ProtocolError("HARP v21 cache role geometry drifted.")
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
        raise ProtocolError("HARP v21 frame protocol provenance drifted.")
    provenance = {
        "schema_version": "midogpp_harp_v21_scratch_frame_provenance_v1",
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
        raise ProtocolError("HARP v21 scratch frame store is only partially durable.")
    created = not path.exists()
    if not created:
        if (
            not path.is_file()
            or path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
        ):
            raise ProtocolError("HARP v21 scratch frame store paths are unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if observed.dtype != np.float32 or observed.shape != (cursor, COMMON_OUTPUT_DIM):
            raise ProtocolError("HARP v21 existing scratch frame store drifted.")
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
                    raise ProtocolError("HARP v21 grouped frame geometry drifted.")
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


def _persist_joint_support_target_projection(
    root: Path,
    *,
    frames: _Frames,
    support_role: str,
    target_role: str,
    outer_target_id: str,
) -> Mapping[str, object]:
    """Persist source Train-q and target Test-H rows for one shared classifier fit.

    The split offset and both role identities are bound into the receipt.  No
    label values enter this projection.  A classifier is fitted once per
    synthetic action and its probabilities are then split back into the two
    independently sealed role surfaces.
    """

    source = np.load(frames.path, mmap_mode="r", allow_pickle=False)
    support_start, support_stop = frames.contexts[(support_role, outer_target_id)]
    target_start, target_stop = frames.contexts[(target_role, outer_target_id)]
    support = np.ascontiguousarray(source[support_start:support_stop], dtype=np.float32)
    target = np.ascontiguousarray(source[target_start:target_stop], dtype=np.float32)
    values = np.ascontiguousarray(np.concatenate((support, target), axis=0), dtype=np.float32)
    split_offset = len(support)
    support_samples = frames.sample_ids[(support_role, outer_target_id)]
    support_cases = frames.case_ids[(support_role, outer_target_id)]
    target_samples = frames.sample_ids[(target_role, outer_target_id)]
    target_cases = frames.case_ids[(target_role, outer_target_id)]
    if (
        not split_offset
        or not len(target)
        or set(support_samples) & set(target_samples)
        or set(support_cases) & set(target_cases)
    ):
        raise ProtocolError("HARP v21 joint support/target projection overlaps.")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"H{outer_target_id}_support_target.npy"
    receipt_path = root / f"H{outer_target_id}_support_target.json"
    if path.exists():
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.dtype != np.float32
            or observed.shape != values.shape
            or not np.array_equal(observed, values)
        ):
            raise ProtocolError("HARP v21 joint support/target projection drifted.")
    else:
        atomic_npy(path, values)
    body = {
        "schema_version": "midogpp_harp_v21_joint_support_target_projection_v1",
        "parent_frame_sha256": frames.sha256,
        "parent_frame_receipt_hash": frames.receipt_hash,
        "outer_target_id": outer_target_id,
        "support_role": support_role,
        "target_role": target_role,
        "support_sample_ids": list(support_samples),
        "support_case_ids": list(support_cases),
        "target_sample_ids": list(target_samples),
        "target_case_ids": list(target_cases),
        "support_row_count": split_offset,
        "target_row_count": len(target),
        "role_split_offset": split_offset,
        "shape": list(values.shape),
        "dtype": "float32",
        "array_sha256": sha256_file(path),
        "other_center_rows_visible": False,
        "labels_consumed": False,
    }
    receipt = {
        **body,
        "frame_projection_hash": projection_semantic_hash(
            body, name="joint support-target projection hash"
        ),
    }
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ProtocolError("HARP v21 joint support/target receipt drifted.")
    else:
        atomic_json(receipt_path, receipt)
    return MappingProxyType(
        {
            "path": path.resolve(),
            "sha256": body["array_sha256"],
            "receipt_path": receipt_path.resolve(),
            "receipt_hash": receipt["frame_projection_hash"],
            "receipt_sha256": sha256_file(receipt_path),
            "sample_ids": (*support_samples, *target_samples),
            "case_ids": (*support_cases, *target_cases),
            "support_sample_ids": support_samples,
            "support_case_ids": support_cases,
            "target_sample_ids": target_samples,
            "target_case_ids": target_cases,
            "role_split_offset": split_offset,
        }
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
    source_binding = validate_source_task_binding(source_cache)
    # Validate the full parent frame before publishing center-local projections.
    validate_frame_task_binding(frames)
    checkpoint_root = scratch_root / "joint_support_target_classifier_checkpoints"
    tasks: list[dict[str, object]] = []
    for outer_ordinal, outer in enumerate(tuple(outer_targets)):
        actions = tuple(build_all_target_actions())
        actions = tuple(row for row in actions if row.outer_target_id == outer)
        validate_action_capacity(actions)
        if len(actions) != 10:
            raise ProtocolError("HARP v21 joint physical action slate drifted.")
        projection = _persist_joint_support_target_projection(
            scratch_root / "joint_support_target_frame_projections",
            frames=frames,
            support_role=development_role,
            target_role=evaluation_role,
            outer_target_id=outer,
        )
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            ordinal = outer_ordinal * len(EXACT_NINE_SEED_PAIRS) + (
                EXACT_NINE_SEED_PAIRS.index((training_seed, generation_seed))
            )
            stem = f"H{outer}_t{training_seed}_g{generation_seed}"
            body = {
                "schema_version": "midogpp_harp_v21_label_free_classifier_task_v1",
                "ordinal": ordinal,
                "surface_kind": "joint_support_target",
                "outer_target_id": outer,
                "query_center_id": outer,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "actions": [row.to_payload() for row in actions],
                "source_array_path": str(source_binding.array_path),
                "source_array_sha256": source_binding.array_sha256,
                "source_index_path": str(source_binding.index_path),
                "source_index_sha256": source_binding.index_sha256,
                "source_stream_index_hash": source_binding.index_hash,
                "source_records": list(source_binding.records),
                "frame_array_path": str(projection["path"]),
                "frame_array_sha256": projection["sha256"],
                "frame_receipt_path": str(projection["receipt_path"]),
                "frame_receipt_hash": projection["receipt_hash"],
                "frame_receipt_sha256": projection["receipt_sha256"],
                "frame_projection_schema": "midogpp_harp_v21_joint_support_target_projection_v1",
                "frame_projection_hash": projection["receipt_hash"],
                "frame_start": 0,
                "frame_stop": len(projection["sample_ids"]),
                "sample_ids": list(projection["sample_ids"]),
                "case_ids": list(projection["case_ids"]),
                "support_sample_ids": list(projection["support_sample_ids"]),
                "support_case_ids": list(projection["support_case_ids"]),
                "target_sample_ids": list(projection["target_sample_ids"]),
                "target_case_ids": list(projection["target_case_ids"]),
                "role_split_offset": int(projection["role_split_offset"]),
                "support_role": development_role,
                "target_role": evaluation_role,
                "classifier_fit_reused_across_roles": True,
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
    if len(tasks) != len(tuple(outer_targets)) * len(EXACT_NINE_SEED_PAIRS):
        raise ProtocolError("HARP v21 classifier task coverage drifted.")
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
                    raise ProtocolError("HARP v21 classifier checkpoint is absent.")
                complete[int(task["ordinal"])] = checkpoint
                if classifier_progress_due(len(complete), len(tasks)):
                    print(
                        "[harp-v21] label-free classifier tasks "
                        f"{len(complete)}/{len(tasks)}",
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
                raise ProtocolError("HARP v21 classifier submission bound drifted.")
    if set(complete) != {int(task["ordinal"]) for task in tasks}:
        raise ProtocolError("HARP v21 classifier task coverage is incomplete.")
    return complete


def _aggregate_outer_menus(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    inputs: PhysicalInputReceipt,
    outer_targets: Sequence[str],
) -> tuple[LabelFreeOuterMenu, ...]:
    cells: dict[str, list[np.ndarray]] = defaultdict(list)
    actions_by_hash: dict[str, HarpActionSpec] = {}
    split_by_outer: dict[str, tuple[int, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
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
                "HARP v21 classifier checkpoint changed before positional aggregation."
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
            if action.surface_kind != TARGET_SURFACE:
                raise ProtocolError("HARP v21 joint task action is not role-neutral target geometry.")
            actions_by_hash[action.action_hash] = action
            cells[action.action_hash].append(np.ascontiguousarray(values[index], dtype=np.float32))
        outer = str(task["outer_target_id"])
        identity = (
            int(task["role_split_offset"]),
            tuple(str(value) for value in task["support_sample_ids"]),
            tuple(str(value) for value in task["support_case_ids"]),
            tuple(str(value) for value in task["target_sample_ids"]),
            tuple(str(value) for value in task["target_case_ids"]),
        )
        previous = split_by_outer.setdefault(outer, identity)
        if previous != identity:
            raise ProtocolError("HARP v21 joint role split identities drifted.")
        if (
            len(task["sample_ids"]) != len(identity[1]) + len(identity[3])
            or identity[0] != len(identity[1])
            or tuple(task["sample_ids"]) != (*identity[1], *identity[3])
            or tuple(task["case_ids"]) != (*identity[2], *identity[4])
        ):
            raise ProtocolError("HARP v21 joint role split offset drifted.")
    menus: list[LabelFreeOuterMenu] = []
    for outer in outer_targets:
        blocks: list[LabelFreeActionBlock] = []
        split = split_by_outer.get(outer)
        if split is None:
            raise ProtocolError("HARP v21 joint role split is absent.")
        split_offset, support_samples, support_cases, target_samples, target_cases = split
        for action in (
            row for row in actions_by_hash.values() if row.outer_target_id == outer
        ):
            members = cells[action.action_hash]
            if len(members) != len(EXACT_NINE_SEED_PAIRS):
                raise ProtocolError("HARP v21 action lacks exact-nine seed cells.")
            exact_nine = np.stack(members).astype(np.float64)
            reduced = np.ascontiguousarray(
                np.mean(exact_nine, axis=0, dtype=np.float64), dtype=np.float32
            )
            dispersion = np.ascontiguousarray(
                np.std(exact_nine, axis=0, dtype=np.float64),
                dtype=np.float32,
            )
            kind = (
                ActionKind.B
                if action.action_id == BASE_ACTION_ID
                else ActionKind.U
                if action.action_id == UNIFORM_ACTION_ID
                else ActionKind.HXE
            )
            for role, sample_ids, case_ids, role_slice in (
                ("source_train", support_samples, support_cases, slice(0, split_offset)),
                ("target", target_samples, target_cases, slice(split_offset, None)),
            ):
                blocks.append(LabelFreeActionBlock(
                    surface_role=role,
                    outer_target_id=outer,
                    query_center_id=outer,
                    action_kind=kind,
                    selected_source_id=action.selected_source_id,
                    sample_ids=sample_ids,
                    case_ids=case_ids,
                    probabilities=np.ascontiguousarray(reduced[role_slice], dtype=np.float32),
                    seed_dispersion=np.ascontiguousarray(dispersion[role_slice], dtype=np.float32),
                ))
        blocks.sort(key=lambda block: block.key)
        scoped_tasks = tuple(task for task in tasks if task.get("outer_target_id") == outer)
        if len(scoped_tasks) != len(EXACT_NINE_SEED_PAIRS):
            raise ProtocolError("HARP v21 joint task coverage drifted during aggregation.")
        first_task = scoped_tasks[0]
        from ...routing.correction_mass_router_v21.patch_evidence import sketch_virchow2
        frame_path = Path(str(first_task['frame_array_path']))
        if sha256_file(frame_path) != first_task['frame_array_sha256']:
            raise ProtocolError('HARP v21 patch frame bytes drifted before feature sealing.')
        patch = sketch_virchow2(np.load(frame_path, mmap_mode='r', allow_pickle=False))
        menus.append(
            LabelFreeOuterMenu(
                outer_target_id=outer,
                blocks=tuple(blocks),
                patch_features={"source_train": patch[:split_offset], "target": patch[split_offset:]},
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
                    "exact_nine_seed_pairs": [list(value) for value in EXACT_NINE_SEED_PAIRS],
                    "reduction_dtype": "float64",
                    "transport_dtype": "float32",
                    "exact_nine_seed_dispersion_persisted": True,
                    "classifier_fit_reused_across_support_and_target": True,
                    "role_split_offset": split_offset,
                },
            )
        )
    return tuple(menus)


__all__ = (
    "PhysicalInputReceipt",
    "build_physical_plan",
    "classifier_pool_plan",
    "materialize_physical_outer_menus",
    "validate_physical_inputs",
)
