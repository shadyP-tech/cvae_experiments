"""Physical, label-free HARP probability-menu production for Stage 90.

The consumed-test cache provides embeddings and row/case identities only.  The
authoritative fixed expert bank and GenerationLock produce the exact frozen
source streams.  Spawned CPU workers then fit the frozen classifier for every
exact-B and legal single-source action.  No source/evaluation label capability
is representable in this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...generation import read_generation_lock
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    atomic_npz,
    read_json,
    sha256_file,
)
from ...runtime.frozen_source_streams import (
    SOURCE_ROWS_PER_CLASS,
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    source_block_sha256,
)
from ...runtime.harp_probability_menu import (
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpActionSpec,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    build_all_development_actions,
    build_all_target_actions,
    compose_harp_action,
    harp_composition_seed,
    harp_source_stream_content_hash,
    seal_harp_prediction_menu,
)
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from .config import HarpStage90Config
from .input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpConsumedCacheIndex,
)


SOURCE_RUNTIME_ROOT = Path("workstation/source_streams")
FRAME_ARRAY_MEMBER = Path("workstation/frame_cache/consumed_test_rows.npy")
FRAME_RECEIPT_MEMBER = Path("workstation/frame_cache/frame_receipt.json")
LINEAGE_RECEIPT_MEMBER = Path("workstation/source_streams/manifests/harp_lineage_receipt.json")
CLASSIFIER_CHECKPOINT_ROOT = Path("checkpoints/harp_classifiers_v2")
MENU_SEAL_MEMBER = Path("manifests/physical_probability_menu.json")
PLAN_MEMBER = Path("manifests/physical_probability_plan.json")

EXPECTED_ACTION_COUNT = 738
EXPECTED_CELL_COUNT = 6642
EXPECTED_CLASSIFIER_TASK_COUNT = 729


@dataclass(frozen=True, slots=True)
class HarpPhysicalInputReceipt:
    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    classifier_config_hash: str
    classifier_contract_sha256: str
    cache_hash: str
    cache_content_sha256: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_stage90_physical_input_receipt_v1",
            "bank_semantic_lock_hash": self.bank_semantic_lock_hash,
            "generation_semantic_lock_hash": self.generation_semantic_lock_hash,
            "expert_bank_index_sha256": self.expert_bank_index_sha256,
            "generation_lock_file_sha256": self.generation_lock_file_sha256,
            "classifier_config_hash": self.classifier_config_hash,
            "classifier_contract_sha256": self.classifier_contract_sha256,
            "cache_hash": self.cache_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "labels_consumed": False,
            "predecessor_policy_used": False,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class _AuthoritativeInputs:
    generation_lock: GenerationLock
    classifier: ClassifierSpec
    receipt: HarpPhysicalInputReceipt


@dataclass(frozen=True, slots=True)
class _FrozenSourceConfigAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _StagedFrames:
    array_path: Path
    rows_by_context: Mapping[tuple[str, str], tuple[str, ...]]
    cases_by_context: Mapping[tuple[str, str], tuple[str, ...]]
    offsets_by_context: Mapping[tuple[str, str], tuple[int, int]]
    frame_hash_by_context: Mapping[tuple[str, str], str]
    array_sha256: str
    receipt_hash: str


@dataclass(frozen=True, slots=True)
class _PhysicalLineage:
    bank_hash: str
    generation_lock_hash: str
    source_cache_hash: str
    source_content_hash: str
    classifier_hash: str
    receipt_hash: str


def build_physical_plan() -> Mapping[str, object]:
    """Return the complete mutation-free Stage-90 action/task topology."""

    actions = _all_actions()
    by_query: dict[tuple[str, str, str], int] = defaultdict(int)
    for action in actions:
        by_query[
            (action.surface_kind, action.outer_target_id, action.query_center_id)
        ] += 1
    task_count = len(by_query) * len(EXACT_NINE_SEED_PAIRS)
    cell_count = len(actions) * len(EXACT_NINE_SEED_PAIRS)
    if (
        len(actions) != EXPECTED_ACTION_COUNT
        or task_count != EXPECTED_CLASSIFIER_TASK_COUNT
        or cell_count != EXPECTED_CELL_COUNT
        or sorted(by_query.values()).count(9) != 72
        or sorted(by_query.values()).count(10) != 9
    ):
        raise ProtocolError("HARP Stage-90 physical plan coverage drifted.")
    base: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_physical_plan_v2",
        "action_count": len(actions),
        "development_action_count": sum(
            action.surface_kind == DEVELOPMENT_SURFACE for action in actions
        ),
        "target_action_count": sum(
            action.surface_kind == TARGET_SURFACE for action in actions
        ),
        "query_context_count": len(by_query),
        "classifier_task_count": task_count,
        "exact_nine_cell_count": cell_count,
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_generation_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "gpu_and_cpu_phases_disjoint": True,
        "parent_cuda_context_created": False,
        "late_torch_interop_setter_used": False,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "matched_budget_reference_action": "U",
        "operational_fallback_action": "B",
        "predictive_lambda_semantics": (
            "post_classifier_predictive_probability_ensemble_"
            "not_generated_distribution"
        ),
        "source_or_target_labels_available": False,
    }
    return {**base, "plan_hash": canonical_hash(base)}


def validate_physical_inputs(
    config: HarpStage90Config, cache: HarpConsumedCacheIndex
) -> HarpPhysicalInputReceipt:
    """Read-only validation for admission and mutation-free dry runs."""

    authoritative = _load_authoritative_inputs(config, cache)
    _validate_cache_role_geometry(cache)
    return authoritative.receipt


def materialize_physical_harp_menu(
    config: HarpStage90Config,
    cache: HarpConsumedCacheIndex,
    *,
    root: Path,
    expected_input_receipt_hash: str,
) -> HarpPredictionMenuSeal:
    """Produce and durably seal the combined development/target menu."""

    authoritative = _load_authoritative_inputs(config, cache)
    if authoritative.receipt.receipt_hash != expected_input_receipt_hash:
        raise ProtocolError("HARP Stage-90 physical inputs changed after admission.")
    frames = _stage_label_blind_frames(cache, root=root)
    source_adapter = _FrozenSourceConfigAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_stage90_source_runtime_binding_v1",
                "config_hash": config.config_hash,
                "physical_input_receipt_hash": authoritative.receipt.receipt_hash,
                "frame_receipt_hash": frames.receipt_hash,
            }
        ),
        expert_bank_root=config.resolved_path("expert_bank_root"),
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
    # The GPU pool is completely closed when this call returns.  Only then is
    # the independent CPU classifier pool created.
    source_cache = materialize_frozen_source_streams(
        source_adapter,
        authoritative.generation_lock,
        root=root / SOURCE_RUNTIME_ROOT,
    )
    lineage = _physical_lineage(authoritative, source_cache)
    _persist_lineage_receipt(root, authoritative.receipt, source_cache, lineage)
    plan = build_physical_plan()
    atomic_json(root / PLAN_MEMBER, plan)
    tasks = _build_classifier_tasks(
        config,
        root=root,
        frames=frames,
        source_cache=source_cache,
        authoritative=authoritative,
        lineage=lineage,
    )
    completed = _execute_classifier_tasks(
        tasks, workers=int(config.runtime["cpu_model_workers"])
    )
    menu = _menu_from_checkpoints(tasks, completed, lineage=lineage)
    _persist_menu_seal(root, menu, tasks, completed, authoritative.receipt)
    return menu


def _all_actions() -> tuple[HarpActionSpec, ...]:
    actions = (*build_all_development_actions(), *build_all_target_actions())
    if tuple(action.key for action in actions) != tuple(
        sorted(action.key for action in actions)
    ):
        raise ProtocolError("HARP Stage-90 action order is noncanonical.")
    return tuple(actions)


def _load_authoritative_inputs(
    config: HarpStage90Config, cache: HarpConsumedCacheIndex
) -> _AuthoritativeInputs:
    bank_root = config.resolved_path("expert_bank_root")
    generation_root = config.resolved_path("generation_lock_root")
    for input_root, name in (
        (bank_root, "expert bank"),
        (generation_root, "GenerationLock"),
    ):
        if not input_root.is_dir() or input_root.is_symlink():
            raise ProtocolError(f"HARP Stage-90 authoritative {name} root is unsafe.")
        state = read_json(input_root / "reports/run_state.json")
        validation = read_json(input_root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(
                f"HARP Stage-90 authoritative {name} is not complete and valid."
            )
    bank_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    if (
        not bank_path.is_file()
        or bank_path.is_symlink()
        or not generation_path.is_file()
        or generation_path.is_symlink()
    ):
        raise ProtocolError("HARP Stage-90 authoritative bank/GenerationLock is absent.")
    bank_sha = sha256_file(bank_path)
    generation_sha = sha256_file(generation_path)
    generation_lock = read_generation_lock(generation_path)
    lock_payload = generation_lock.to_payload()
    bank_payload = read_json(bank_path)
    bank_binding = lock_payload.get("bank")
    raw_classifier = lock_payload.get("classifier")
    if not isinstance(bank_binding, Mapping) or not isinstance(raw_classifier, Mapping):
        raise ProtocolError("HARP Stage-90 GenerationLock lacks bank/classifier bindings.")
    if (
        generation_lock.bank_lock_hash
        != config.expected_hashes["expert_bank_lock_hash"]
        or generation_lock.generation_lock_hash
        != config.expected_hashes["generation_lock_hash"]
        or bank_payload.get("bank_lock_hash") != generation_lock.bank_lock_hash
        or bank_binding.get("bank_index_sha256") != bank_sha
    ):
        raise ProtocolError("HARP Stage-90 authoritative generation lineage drifted.")
    expected_classifier_keys = {
        "family",
        "C",
        "penalty",
        "solver",
        "max_iter",
        "class_weight",
        "random_state",
        "l1_ratio",
        "threshold_policy",
        "scaler_fit",
        "config_hash",
        "scaler_family",
        "fit_in_stage_40",
    }
    if set(raw_classifier) != expected_classifier_keys:
        raise ProtocolError("HARP Stage-90 classifier contract schema drifted.")
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
                None
                if raw_classifier["l1_ratio"] is None
                else float(raw_classifier["l1_ratio"])
            ),
            threshold_policy=str(raw_classifier["threshold_policy"]),
            scaler_fit=str(raw_classifier["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP Stage-90 classifier contract is malformed.") from exc
    if (
        classifier.config_hash != raw_classifier.get("config_hash")
        or raw_classifier.get("scaler_family")
        != "sklearn.preprocessing.StandardScaler"
        or raw_classifier.get("fit_in_stage_40") is not False
    ):
        raise ProtocolError("HARP Stage-90 classifier identity drifted.")
    classifier_contract_sha = canonical_hash(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v1",
            "classifier": classifier.to_payload(),
            "scaler_family": raw_classifier["scaler_family"],
            "fit_in_stage_40": False,
        }
    )
    receipt_base = {
        "schema_version": "midogpp_harp_stage90_physical_input_receipt_v1",
        "bank_semantic_lock_hash": generation_lock.bank_lock_hash,
        "generation_semantic_lock_hash": generation_lock.generation_lock_hash,
        "expert_bank_index_sha256": bank_sha,
        "generation_lock_file_sha256": generation_sha,
        "classifier_config_hash": classifier.config_hash,
        "classifier_contract_sha256": classifier_contract_sha,
        "cache_hash": cache.cache_hash,
        "cache_content_sha256": cache.content_sha256,
        "labels_consumed": False,
        "predecessor_policy_used": False,
    }
    receipt = HarpPhysicalInputReceipt(
        bank_semantic_lock_hash=generation_lock.bank_lock_hash,
        generation_semantic_lock_hash=generation_lock.generation_lock_hash,
        expert_bank_index_sha256=bank_sha,
        generation_lock_file_sha256=generation_sha,
        classifier_config_hash=classifier.config_hash,
        classifier_contract_sha256=classifier_contract_sha,
        cache_hash=cache.cache_hash,
        cache_content_sha256=cache.content_sha256,
        receipt_hash=canonical_hash(receipt_base),
    )
    if receipt.to_payload() != {**receipt_base, "receipt_hash": receipt.receipt_hash}:
        raise ProtocolError("HARP Stage-90 physical input receipt construction drifted.")
    return _AuthoritativeInputs(generation_lock, classifier, receipt)


def _validate_cache_role_geometry(cache: HarpConsumedCacheIndex) -> None:
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in tuple(str(value) for value in cache_centers()):
            rows = cache.rows_for(center, role)
            if not rows or tuple(row.split_row_index for row in rows) != tuple(
                range(len(rows))
            ):
                raise ProtocolError("HARP Stage-90 cache role geometry drifted.")


def cache_centers() -> tuple[str, ...]:
    # Kept as a function so a test can audit that no observed-center discovery
    # changes the frozen universe.
    from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS

    return CENTERS


def _stage_label_blind_frames(
    cache: HarpConsumedCacheIndex, *, root: Path
) -> _StagedFrames:
    _validate_cache_role_geometry(cache)
    staged: list[np.ndarray] = []
    rows_by_context: dict[tuple[str, str], tuple[str, ...]] = {}
    cases_by_context: dict[tuple[str, str], tuple[str, ...]] = {}
    offsets: dict[tuple[str, str], tuple[int, int]] = {}
    frame_hashes: dict[tuple[str, str], str] = {}
    opened: dict[str, np.ndarray] = {}
    cursor = 0
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in cache_centers():
            scoped = cache.rows_for(center, role)
            values: list[np.ndarray] = []
            row_ids: list[str] = []
            case_ids: list[str] = []
            for row in scoped:
                if row.embedding_file not in opened:
                    opened[row.embedding_file] = np.load(
                        cache.root / row.embedding_file,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                value = np.asarray(
                    opened[row.embedding_file][row.embedding_row_index],
                    dtype=np.float32,
                )
                if value.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(value).all():
                    raise ProtocolError("HARP Stage-90 staged frame row drifted.")
                values.append(value)
                row_ids.append(row.sample_id)
                case_ids.append(row.case_id)
            matrix = np.ascontiguousarray(np.stack(values), dtype=np.float32)
            context = (role, center)
            rows_by_context[context] = tuple(row_ids)
            cases_by_context[context] = tuple(case_ids)
            offsets[context] = (cursor, cursor + len(matrix))
            frame_hashes[context] = canonical_hash(
                {
                    "schema_version": "midogpp_harp_stage90_query_frame_slice_v1",
                    "split_role": role,
                    "center": center,
                    "row_ids": row_ids,
                    "case_ids": case_ids,
                    "embedding_bytes_sha256": hashlib.sha256(
                        matrix.tobytes(order="C")
                    ).hexdigest(),
                    "cache_hash": cache.cache_hash,
                    "cache_content_sha256": cache.content_sha256,
                    "labels_consumed": False,
                }
            )
            staged.append(matrix)
            cursor += len(matrix)
    all_values = np.ascontiguousarray(np.concatenate(staged, axis=0), dtype=np.float32)
    array_path = root / FRAME_ARRAY_MEMBER
    atomic_npy(array_path, all_values)
    array_sha = sha256_file(array_path)
    receipt_base: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_staged_frame_receipt_v1",
        "cache_hash": cache.cache_hash,
        "cache_content_sha256": cache.content_sha256,
        "array_member": FRAME_ARRAY_MEMBER.as_posix(),
        "array_sha256": array_sha,
        "shape": list(all_values.shape),
        "dtype": "float32",
        "contexts": [
            {
                "split_role": role,
                "center": center,
                "start": offsets[(role, center)][0],
                "stop": offsets[(role, center)][1],
                "row_ids": list(rows_by_context[(role, center)]),
                "case_ids": list(cases_by_context[(role, center)]),
                "frame_hash": frame_hashes[(role, center)],
            }
            for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE)
            for center in cache_centers()
        ],
        "labels_stored": False,
    }
    receipt = {**receipt_base, "receipt_hash": canonical_hash(receipt_base)}
    atomic_json(root / FRAME_RECEIPT_MEMBER, receipt)
    if read_json(root / FRAME_RECEIPT_MEMBER) != receipt:
        raise ProtocolError("HARP Stage-90 frame receipt failed durable validation.")
    return _StagedFrames(
        array_path=array_path,
        rows_by_context=MappingProxyType(rows_by_context),
        cases_by_context=MappingProxyType(cases_by_context),
        offsets_by_context=MappingProxyType(offsets),
        frame_hash_by_context=MappingProxyType(frame_hashes),
        array_sha256=array_sha,
        receipt_hash=str(receipt["receipt_hash"]),
    )


def _physical_lineage(
    authoritative: _AuthoritativeInputs, source_cache: FrozenSourceStreamCache
) -> _PhysicalLineage:
    content_hash = harp_source_stream_content_hash(source_cache.records)
    base = {
        "schema_version": "midogpp_harp_stage90_physical_lineage_v1",
        "bank_hash": authoritative.receipt.bank_semantic_lock_hash,
        "generation_lock_hash": authoritative.receipt.generation_semantic_lock_hash,
        "source_cache_hash": source_cache.lock_hash,
        "source_content_hash": content_hash,
        "classifier_hash": authoritative.classifier.config_hash,
        "physical_input_receipt_hash": authoritative.receipt.receipt_hash,
    }
    return _PhysicalLineage(
        bank_hash=str(base["bank_hash"]),
        generation_lock_hash=str(base["generation_lock_hash"]),
        source_cache_hash=str(base["source_cache_hash"]),
        source_content_hash=str(base["source_content_hash"]),
        classifier_hash=str(base["classifier_hash"]),
        receipt_hash=canonical_hash(base),
    )


def _persist_lineage_receipt(
    root: Path,
    inputs: HarpPhysicalInputReceipt,
    source_cache: FrozenSourceStreamCache,
    lineage: _PhysicalLineage,
) -> None:
    source_root = root / SOURCE_RUNTIME_ROOT
    source_lock_path = source_root / "manifests/frozen_source_stream_lock.json"
    source_index_path = source_root / "manifests/frozen_source_stream_index.json"
    source_lock_sha256 = sha256_file(source_lock_path)
    base = {
        "schema_version": "midogpp_harp_stage90_authoritative_lineage_receipt_v1",
        "physical_inputs": inputs.to_payload(),
        "source_stream_lock_hash": source_cache.lock_hash,
        "source_stream_lock_sha256": source_lock_sha256,
        "source_stream_content_hash": lineage.source_content_hash,
        "source_array_sha256": source_cache.lock_payload["source_array_sha256"],
        "source_stream_index_sha256": source_cache.lock_payload[
            "source_stream_index_sha256"
        ],
        "lineage_hash": lineage.receipt_hash,
        "labels_consumed": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_used": False,
    }
    payload = {**base, "receipt_hash": canonical_hash(base)}
    path = root / LINEAGE_RECEIPT_MEMBER
    atomic_json(path, payload)
    reconstructed = load_frozen_source_streams(
        source_root,
        expected_config_hash=str(source_cache.lock_payload["config_contract_hash"]),
        expected_generation_lock_hash=str(
            source_cache.lock_payload["generation_lock_hash"]
        ),
    )
    if (
        read_json(path) != payload
        or sha256_file(source_lock_path) != source_lock_sha256
        or dict(reconstructed.lock_payload) != dict(source_cache.lock_payload)
        or reconstructed.records != source_cache.records
        or sha256_file(source_index_path) != str(
            source_cache.lock_payload["source_stream_index_sha256"]
        )
    ):
        raise ProtocolError("HARP Stage-90 lineage receipt failed durable validation.")


def _build_classifier_tasks(
    config: HarpStage90Config,
    *,
    root: Path,
    frames: _StagedFrames,
    source_cache: FrozenSourceStreamCache,
    authoritative: _AuthoritativeInputs,
    lineage: _PhysicalLineage,
) -> tuple[dict[str, object], ...]:
    actions = _all_actions()
    by_query: dict[tuple[str, str, str], list[HarpActionSpec]] = defaultdict(list)
    for action in actions:
        by_query[
            (action.surface_kind, action.outer_target_id, action.query_center_id)
        ].append(action)
    source_records = [record.to_payload() for record in source_cache.records]
    tasks: list[dict[str, object]] = []
    checkpoint_root = root / CLASSIFIER_CHECKPOINT_ROOT
    for surface, outer, query in sorted(by_query):
        role = DEVELOPMENT_ROLE if surface == DEVELOPMENT_SURFACE else EVALUATION_ROLE
        context = (role, query)
        start, stop = frames.offsets_by_context[context]
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            ordinal = len(tasks)
            stem = f"task_{ordinal:04d}"
            base: dict[str, object] = {
                "schema_version": "midogpp_harp_stage90_physical_classifier_task_v2",
                "ordinal": ordinal,
                "surface_kind": surface,
                "outer_target_id": outer,
                "query_center_id": query,
                "split_role": role,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "actions": [action.to_payload() for action in by_query[(surface, outer, query)]],
                "source_array_path": str(source_cache.source_array_path.resolve()),
                "source_array_sha256": source_cache.lock_payload["source_array_sha256"],
                "source_records": source_records,
                "frame_array_path": str(frames.array_path.resolve()),
                "frame_array_sha256": frames.array_sha256,
                "frame_start": start,
                "frame_stop": stop,
                "row_ids": list(frames.rows_by_context[context]),
                "case_ids": list(frames.cases_by_context[context]),
                "frame_hash": frames.frame_hash_by_context[context],
                "generation_lock_hash": authoritative.generation_lock.generation_lock_hash,
                "classifier": authoritative.classifier.to_payload(),
                "threads_per_worker": int(config.runtime["blas_threads_per_worker"]),
                "bank_hash": lineage.bank_hash,
                "source_cache_hash": lineage.source_cache_hash,
                "source_content_hash": lineage.source_content_hash,
                "classifier_hash": lineage.classifier_hash,
                "lineage_hash": lineage.receipt_hash,
                "labels_available": False,
                "nested_process_pools": False,
                "late_torch_interop_setter_used": False,
            }
            tasks.append(
                {
                    **base,
                    "task_hash": canonical_hash(base),
                    "checkpoint_npz_path": str(checkpoint_root / f"{stem}.npz"),
                    "checkpoint_json_path": str(checkpoint_root / f"{stem}.json"),
                }
            )
    if (
        len(tasks) != EXPECTED_CLASSIFIER_TASK_COUNT
        or tuple(int(task["ordinal"]) for task in tasks) != tuple(range(len(tasks)))
        or sum(len(task["actions"]) for task in tasks) != EXPECTED_CELL_COUNT
    ):
        raise ProtocolError("HARP Stage-90 physical classifier task coverage drifted.")
    return tuple(tasks)


def _execute_classifier_tasks(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> dict[int, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("HARP Stage-90 workstation requires four classifier workers.")
    completed: dict[int, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        existing = _load_classifier_checkpoint(task)
        if existing is None:
            pending.append(task)
        else:
            completed[int(task["ordinal"])] = existing
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(_classifier_worker, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                verified = _load_classifier_checkpoint(task)
                if verified is None:
                    raise ProtocolError("HARP Stage-90 classifier checkpoint is absent.")
                completed[int(task["ordinal"])] = verified
                print(
                    "[harp-stage90-v1] physical classifier tasks "
                    f"{len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if set(completed) != set(range(len(tasks))):
        raise ProtocolError("HARP Stage-90 classifier checkpoint coverage is incomplete.")
    return completed


def _classifier_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    source_blocks, evaluation, actions, classifier = _load_worker_inputs(task)
    probabilities: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP Stage-90 classifier workers require threadpoolctl.") from exc
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
                composition.embeddings,
                composition.labels,
                evaluation,
                spec=classifier,
            )
            values = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or values.shape != (len(evaluation), 2)
                or not np.isfinite(values).all()
                or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != classifier.config_hash
            ):
                raise ProtocolError("HARP Stage-90 physical classifier fit drifted.")
            positive = np.ascontiguousarray(values[:, 1], dtype=np.float32)
            probabilities.append(positive)
            records.append(
                {
                    "action_hash": action.action_hash,
                    "composition_hash": composition.composition_hash,
                    "scaler_state_hash": str(fitted.scaler_state_hash),
                    "probability_sha256": hashlib.sha256(
                        positive.tobytes(order="C")
                    ).hexdigest(),
                }
            )
    matrix = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    atomic_npz(npz_path, probabilities=matrix)
    base: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_classifier_checkpoint_v2",
        "status": "COMPLETE_LABEL_FREE",
        "task_hash": task["task_hash"],
        "npz_sha256": sha256_file(npz_path),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "actions": records,
        "labels_consumed": False,
        "nested_process_pools": False,
        "late_torch_interop_setter_used": False,
    }
    payload = {**base, "checkpoint_hash": canonical_hash(base)}
    atomic_json(Path(str(task["checkpoint_json_path"])), payload)
    return payload


def _load_worker_inputs(
    task: Mapping[str, object],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
    tuple[HarpActionSpec, ...],
    ClassifierSpec,
]:
    base = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_npz_path", "checkpoint_json_path"}
    }
    if (
        task.get("schema_version")
        != "midogpp_harp_stage90_physical_classifier_task_v2"
        or task.get("task_hash") != canonical_hash(base)
        or task.get("labels_available") is not False
        or task.get("nested_process_pools") is not False
        or task.get("late_torch_interop_setter_used") is not False
        or int(task.get("threads_per_worker", -1)) != 3
    ):
        raise ProtocolError("HARP Stage-90 physical classifier task drifted.")
    outer = str(task["outer_target_id"])
    query = str(task["query_center_id"])
    raw_actions = task.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ProtocolError("HARP Stage-90 task has no action menu.")
    actions: list[HarpActionSpec] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP Stage-90 task action is malformed.")
        action = HarpActionSpec(
            surface_kind=str(raw["surface_kind"]),
            outer_target_id=outer,
            query_center_id=query,
            selected_source_id=(
                None
                if raw.get("selected_source_id") is None
                else str(raw["selected_source_id"])
            ),
            action_id=str(raw.get("action_id")),
        )
        if action.to_payload() != dict(raw):
            raise ProtocolError("HARP Stage-90 task action payload drifted.")
        actions.append(action)
    raw_records = task.get("source_records")
    if not isinstance(raw_records, list):
        raise ProtocolError("HARP Stage-90 task source index is absent.")
    records = {
        (
            str(raw["source_center"]),
            int(raw["training_seed"]),
            int(raw["generation_seed"]),
        ): raw
        for raw in raw_records
        if isinstance(raw, Mapping)
    }
    source_values = np.load(
        Path(str(task["source_array_path"])), mmap_mode="r", allow_pickle=False
    )
    blocks: dict[str, dict[str, np.ndarray]] = {}
    required_sources = tuple(
        sorted({source for action in actions for source in action.source_order})
    )
    for source in required_sources:
        key = (source, int(task["training_seed"]), int(task["generation_seed"]))
        record = records.get(key)
        if record is None:
            raise ProtocolError("HARP Stage-90 task source stream is absent.")
        block = np.asarray(source_values[int(record["block_ordinal"])], dtype=np.float32)
        if source_block_sha256(block) != record.get("output_sha256"):
            raise ProtocolError("HARP Stage-90 task source stream bytes drifted.")
        blocks[source] = {
            "embeddings": block,
            "labels": np.concatenate(
                (
                    np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                    np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                )
            ),
        }
    frame = np.load(
        Path(str(task["frame_array_path"])), mmap_mode="r", allow_pickle=False
    )
    start, stop = int(task["frame_start"]), int(task["frame_stop"])
    evaluation = np.ascontiguousarray(frame[start:stop], dtype=np.float32)
    if (
        evaluation.shape != (len(task["row_ids"]), COMMON_OUTPUT_DIM)
        or len(task["row_ids"]) != len(task["case_ids"])
        or not np.isfinite(evaluation).all()
    ):
        raise ProtocolError("HARP Stage-90 task frame slice drifted.")
    classifier_raw = task.get("classifier")
    if not isinstance(classifier_raw, Mapping):
        raise ProtocolError("HARP Stage-90 task classifier is absent.")
    classifier = ClassifierSpec(**dict(classifier_raw))
    return blocks, evaluation, tuple(actions), classifier


def _load_classifier_checkpoint(
    task: Mapping[str, object]
) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("HARP Stage-90 partial classifier checkpoint is unsafe.")
    payload = read_json(json_path)
    base = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"probabilities"}:
            raise ProtocolError("HARP Stage-90 checkpoint archive drifted.")
        values = np.asarray(archive["probabilities"])
    actions = payload.get("actions")
    if (
        payload.get("schema_version")
        != "midogpp_harp_stage90_classifier_checkpoint_v2"
        or payload.get("status") != "COMPLETE_LABEL_FREE"
        or payload.get("checkpoint_hash") != canonical_hash(base)
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or values.dtype != np.float32
        or values.shape != (len(task["actions"]), len(task["row_ids"]))
        or not isinstance(actions, list)
        or len(actions) != len(task["actions"])
        or payload.get("labels_consumed") is not False
        or payload.get("nested_process_pools") is not False
        or payload.get("late_torch_interop_setter_used") is not False
    ):
        raise ProtocolError("HARP Stage-90 classifier checkpoint failed validation.")
    for ordinal, (record, action) in enumerate(
        zip(actions, task["actions"], strict=True)
    ):
        if (
            not isinstance(record, Mapping)
            or record.get("action_hash") != action["action_hash"]
            or record.get("probability_sha256")
            != hashlib.sha256(values[ordinal].tobytes(order="C")).hexdigest()
        ):
            raise ProtocolError("HARP Stage-90 checkpoint action bytes drifted.")
    return payload


def _menu_from_checkpoints(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    lineage: _PhysicalLineage,
) -> HarpPredictionMenuSeal:
    action_by_hash: dict[str, HarpActionSpec] = {}
    cells_by_key: dict[tuple[str, int, int], HarpPredictionCell] = {}
    for task in tasks:
        ordinal = int(task["ordinal"])
        payload = completed[ordinal]
        with np.load(Path(str(task["checkpoint_npz_path"])), allow_pickle=False) as archive:
            values = np.asarray(archive["probabilities"], dtype=np.float32)
        for action_ordinal, (raw_action, record) in enumerate(
            zip(task["actions"], payload["actions"], strict=True)
        ):
            action = HarpActionSpec(
                surface_kind=str(raw_action["surface_kind"]),
                outer_target_id=str(raw_action["outer_target_id"]),
                query_center_id=str(raw_action["query_center_id"]),
                selected_source_id=(
                    None
                    if raw_action.get("selected_source_id") is None
                    else str(raw_action["selected_source_id"])
                ),
                action_id=str(raw_action.get("action_id")),
            )
            action_by_hash[action.action_hash] = action
            cell = HarpPredictionCell(
                action=action,
                training_seed=int(task["training_seed"]),
                generation_seed=int(task["generation_seed"]),
                row_ids=tuple(str(value) for value in task["row_ids"]),
                case_ids=tuple(str(value) for value in task["case_ids"]),
                probabilities=np.ascontiguousarray(values[action_ordinal], dtype=np.float32),
                bank_hash=lineage.bank_hash,
                generation_lock_hash=lineage.generation_lock_hash,
                source_cache_hash=lineage.source_cache_hash,
                frame_hash=str(task["frame_hash"]),
                classifier_hash=lineage.classifier_hash,
                composition_hash=str(record["composition_hash"]),
                scaler_state_hash=str(record["scaler_state_hash"]),
            )
            key = (action.action_hash, cell.training_seed, cell.generation_seed)
            if key in cells_by_key:
                raise ProtocolError("HARP Stage-90 physical cell was produced twice.")
            cells_by_key[key] = cell
    actions = _all_actions()
    if set(action_by_hash) != {action.action_hash for action in actions}:
        raise ProtocolError("HARP Stage-90 produced action inventory drifted.")
    cells = tuple(
        cells_by_key[(action.action_hash, training_seed, generation_seed)]
        for action in actions
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
    )
    menu = seal_harp_prediction_menu(actions, cells)
    menu.assert_valid()
    return menu


def _persist_menu_seal(
    root: Path,
    menu: HarpPredictionMenuSeal,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    inputs: HarpPhysicalInputReceipt,
) -> None:
    checkpoint_members = [
        {
            "ordinal": int(task["ordinal"]),
            "task_hash": task["task_hash"],
            "npz_member": Path(str(task["checkpoint_npz_path"])).relative_to(root).as_posix(),
            "npz_sha256": completed[int(task["ordinal"])]["npz_sha256"],
            "receipt_member": Path(str(task["checkpoint_json_path"])).relative_to(root).as_posix(),
            "receipt_sha256": sha256_file(Path(str(task["checkpoint_json_path"]))),
        }
        for task in tasks
    ]
    base: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_physical_probability_menu_v2",
        "status": "DURABLE_COMPLETE_LABEL_FREE_PHYSICAL_MENU",
        "physical_input_receipt_hash": inputs.receipt_hash,
        "action_menu_hash": menu.action_menu_hash,
        "prediction_store_hash": menu.prediction_store_hash,
        "prediction_menu_seal_hash": menu.seal_hash,
        "action_count": len(menu.actions),
        "cell_count": len(menu.cells),
        "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
        "actions": [action.to_payload() for action in menu.actions],
        "cell_hashes": [cell.cell_hash for cell in menu.cells],
        "probability_members": checkpoint_members,
        "workstation": menu.workstation.to_payload(),
        "all_exact_b_uniform_reference_and_legal_candidates_present": True,
        "matched_budget_reference_action": "U",
        "operational_fallback_action": "B",
        "all_action_cells_present_before_label_access": True,
        "labels_consumed": False,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_used": False,
    }
    payload = {**base, "artifact_hash": canonical_hash(base)}
    path = root / MENU_SEAL_MEMBER
    atomic_json(path, payload)
    observed = read_json(path)
    if observed != payload:
        raise ProtocolError("HARP Stage-90 physical menu seal failed durable validation.")
    for member in checkpoint_members:
        if (
            sha256_file(root / str(member["npz_member"])) != member["npz_sha256"]
            or sha256_file(root / str(member["receipt_member"]))
            != member["receipt_sha256"]
        ):
            raise ProtocolError("HARP Stage-90 sealed probability member drifted.")


__all__ = (
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_CLASSIFIER_TASK_COUNT",
    "HarpPhysicalInputReceipt",
    "build_physical_plan",
    "materialize_physical_harp_menu",
    "validate_physical_inputs",
)
