"""Physical HARP menu materialization on the declared dual-A5000 workstation.

This module is the only Stage-60 action-surface module that imports generation
and classifier execution primitives.  GPU source generation is delegated to
the neutral frozen-source runtime (two persistent spawn workers).  Only after
that pool has closed do four spawn CPU workers fit classifiers with three BLAS
threads each.  Worker DTOs contain paths, hashes, identities, and action specs;
no label column or label capability is representable.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation import read_generation_lock
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
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
    materialize_frozen_source_streams,
    source_block_sha256,
)
from ...runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
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
from ..harp_protocol.hashing import canonical_hash, require_sha256
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE
from .workstation_contracts import HarpWorkstationPlan
from .workstation_checkpoint_io import load_classifier_checkpoint
from .workstation_frame_io import (
    content_hash,
    content_members,
    load_float32_shard,
    read_frame_rows,
    safe_member,
)


SOURCE_RUNTIME_ROOT = "workstation/source_streams"
LINEAGE_RECEIPT_MEMBER = "workstation/source_streams/manifests/harp_lineage_receipt.json"
FRAME_ARRAY_MEMBER = "workstation/frame_cache/query_rows.npy"
FRAME_RECEIPT_MEMBER = "workstation/frame_cache/frame_receipt.json"
CLASSIFIER_CHECKPOINT_ROOT = "checkpoints/harp_classifiers_v2"


@dataclass(frozen=True)
class _FrozenSourceConfigAdapter:
    contract_hash: str
    expert_bank_root: Path
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class _FrameCache:
    array_path: Path
    rows_by_center: Mapping[str, tuple[str, ...]]
    cases_by_center: Mapping[str, tuple[str, ...]]
    offsets_by_center: Mapping[str, tuple[int, int]]
    frame_hash_by_center: Mapping[str, str]
    cache_binding_hash: str
    receipt_hash: str


@dataclass(frozen=True)
class _AuthoritativeLineage:
    bank_semantic_lock_hash: str
    generation_semantic_lock_hash: str
    source_stream_lock_hash: str
    source_stream_index_hash: str
    source_stream_content_hash: str
    classifier_config_hash: str
    expert_bank_index_sha256: str
    generation_lock_file_sha256: str
    source_cache_lock_sha256: str
    source_cache_index_sha256: str
    source_stream_artifact_binding_hash: str
    classifier_contract_sha256: str


def materialize_physical_harp_menu(
    config: HarpStage60Config,
    readiness: HarpInputReadiness,
    plan: HarpWorkstationPlan,
) -> HarpPredictionMenuSeal:
    """Generate sources, fit every HARP action cell, and seal the global menu."""

    (
        generation_lock,
        generation_file_sha,
        bank_file_sha,
        bank_semantic_lock_hash,
        generation_semantic_lock_hash,
        classifier,
        classifier_semantic_hash,
    ) = (
        _load_authoritative_generation_inputs(config)
    )
    frame = _load_and_stage_frame_cache(config, readiness)
    source_config = _FrozenSourceConfigAdapter(
        contract_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_source_runtime_binding_v1",
                "harp_config_contract_hash": config.contract_hash,
                "bank_semantic_lock_hash": bank_semantic_lock_hash,
                "generation_semantic_lock_hash": generation_semantic_lock_hash,
                "expert_bank_index_sha256": bank_file_sha,
                "generation_lock_file_sha256": generation_file_sha,
            }
        ),
        expert_bank_root=config.input_paths["expert_bank_root"],
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
    # This call owns the complete GPU phase.  Its ProcessPoolExecutor is closed
    # before classifier task construction below.
    source_cache = materialize_frozen_source_streams(
        source_config,
        generation_lock,
        root=config.artifact_root / SOURCE_RUNTIME_ROOT,
    )
    source_lock_path = config.artifact_root / SOURCE_RUNTIME_ROOT / "manifests/frozen_source_stream_lock.json"
    source_index_path = config.artifact_root / SOURCE_RUNTIME_ROOT / "manifests/frozen_source_stream_index.json"
    source_cache_sha = sha256_file(source_lock_path)
    source_index_sha = sha256_file(source_index_path)
    source_content_hash = harp_source_stream_content_hash(source_cache.records)
    source_artifact_binding_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": source_cache_sha,
            "source_cache_index_sha256": source_index_sha,
            "source_stream_content_hash": source_content_hash,
        }
    )
    lineage = _AuthoritativeLineage(
        bank_semantic_lock_hash=generation_lock.bank_lock_hash,
        generation_semantic_lock_hash=generation_lock.generation_lock_hash,
        source_stream_lock_hash=source_cache.lock_hash,
        source_stream_index_hash=str(source_cache.lock_payload["source_stream_index_hash"]),
        source_stream_content_hash=source_content_hash,
        classifier_config_hash=classifier.config_hash,
        expert_bank_index_sha256=bank_file_sha,
        generation_lock_file_sha256=generation_file_sha,
        source_cache_lock_sha256=source_cache_sha,
        source_cache_index_sha256=source_index_sha,
        source_stream_artifact_binding_hash=source_artifact_binding_hash,
        classifier_contract_sha256=classifier_semantic_hash,
    )
    _persist_lineage_receipt(config.artifact_root, lineage)
    tasks = _physical_classifier_tasks(
        config,
        plan,
        frame=frame,
        source_cache=source_cache,
        generation_lock=generation_lock,
        classifier=classifier,
        lineage=lineage,
    )
    completed = _execute_classifier_tasks(tasks)
    menu = _menu_from_checkpoints(plan, tasks, completed, lineage=lineage)
    menu.assert_valid()
    return menu


def _persist_lineage_receipt(root: Path, lineage: _AuthoritativeLineage) -> Path:
    """Publish and independently re-read the authoritative Stage-60 lineage.

    Executable semantic identities and file-byte provenance deliberately remain
    separate.  In particular, no semantic token is wrapped to manufacture a
    SHA-256-looking artifact identity.
    """

    semantic_fields = (
        "bank_semantic_lock_hash",
        "generation_semantic_lock_hash",
        "source_stream_lock_hash",
        "source_stream_index_hash",
        "source_stream_content_hash",
        "classifier_config_hash",
    )
    sha_fields = (
        "expert_bank_index_sha256",
        "generation_lock_file_sha256",
        "source_cache_lock_sha256",
        "source_cache_index_sha256",
        "source_stream_artifact_binding_hash",
        "classifier_contract_sha256",
    )
    values = {
        field: str(getattr(lineage, field))
        for field in (*semantic_fields, *sha_fields)
    }
    if any(not values[field] for field in semantic_fields):
        raise ProtocolError("HARP executable lineage contains an empty identity.")
    for field in sha_fields:
        require_sha256(values[field], name=f"HARP lineage {field}")
    require_sha256(
        values["source_stream_content_hash"],
        name="HARP source-stream content hash",
    )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_authoritative_lineage_receipt_v1",
        **values,
    }
    payload = {**unhashed, "receipt_hash": canonical_hash(unhashed)}
    path = root / LINEAGE_RECEIPT_MEMBER
    if path.is_file():
        observed = read_json(path)
        if observed != payload:
            raise ProtocolError("HARP authoritative lineage receipt drifted on restart.")
    else:
        atomic_json(path, payload)
    observed = read_json(path)
    if (
        set(observed) != set(payload)
        or observed != payload
        or observed.get("receipt_hash")
        != canonical_hash(
            {key: value for key, value in observed.items() if key != "receipt_hash"}
        )
    ):
        raise ProtocolError("HARP authoritative lineage receipt failed validation.")
    return path


def _load_authoritative_generation_inputs(
    config: HarpStage60Config,
) -> tuple[GenerationLock, str, str, str, str, ClassifierSpec, str]:
    bank_root = config.input_paths["expert_bank_root"]
    generation_root = config.input_paths["generation_lock_root"]
    bank_index_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    for root, label in ((bank_root, "expert bank"), (generation_root, "generation lock")):
        state = read_json(root / "reports/run_state.json")
        validation = read_json(root / "reports/validation_report.json")
        if state.get("status") != "COMPLETE" or validation.get("status") != "PASS":
            raise ProtocolError(f"HARP authoritative {label} is not complete and valid.")
    if not bank_index_path.is_file() or not generation_path.is_file():
        raise ProtocolError("HARP authoritative bank/generation lock member is absent.")
    bank_sha = sha256_file(bank_index_path)
    generation_sha = sha256_file(generation_path)
    lock = read_generation_lock(generation_path)
    payload = lock.to_payload()
    bank_payload = read_json(bank_index_path)
    if bank_payload.get("bank_lock_hash") != lock.bank_lock_hash:
        raise ProtocolError("HARP GenerationLock bank semantic identity drifted.")
    bank = payload.get("bank")
    raw_classifier = payload.get("classifier")
    if not isinstance(bank, Mapping) or not isinstance(raw_classifier, Mapping):
        raise ProtocolError("HARP GenerationLock lacks bank/classifier bindings.")
    if bank.get("bank_index_sha256") != bank_sha:
        raise ProtocolError("HARP GenerationLock escaped the authoritative bank bytes.")
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
        raise ProtocolError("HARP GenerationLock classifier schema drifted.")
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
        raise ProtocolError("HARP GenerationLock classifier values are malformed.") from exc
    if (
        classifier.config_hash != raw_classifier.get("config_hash")
        or raw_classifier.get("scaler_family")
        != "sklearn.preprocessing.StandardScaler"
        or raw_classifier.get("fit_in_stage_40") is not False
    ):
        raise ProtocolError("HARP GenerationLock classifier contract drifted.")
    classifier_semantic_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_classifier_semantic_identity_v1",
            "classifier": classifier.to_payload(),
            "scaler_family": raw_classifier["scaler_family"],
            "fit_in_stage_40": False,
        }
    )
    return (
        lock,
        generation_sha,
        bank_sha,
        lock.bank_lock_hash,
        lock.generation_lock_hash,
        classifier,
        classifier_semantic_hash,
    )


def _load_and_stage_frame_cache(
    config: HarpStage60Config, readiness: HarpInputReadiness
) -> _FrameCache:
    root = config.input_paths[
        "development_cache_root"
        if config.contract == ACTION_SURFACE
        else "target_support_cache_root"
    ].resolve()
    index_path = root / "manifests/cache_index.json"
    content_path = root / "manifests/content_index.json"
    row_path = root / "tables/row_index.csv"
    index = read_json(index_path)
    content = read_json(content_path)
    index_hash_name = "cache_index_hash"
    observed_index_hash = index.get(index_hash_name)
    if observed_index_hash != canonical_hash(
        {key: value for key, value in index.items() if key != index_hash_name}
    ):
        raise ProtocolError("HARP fresh frame-cache index hash drifted.")
    require_sha256(observed_index_hash, name="HARP frame-cache index hash")
    surface_kind = "development" if config.contract == ACTION_SURFACE else "target"
    expected_index_keys = {
        "schema_version",
        "artifact_id",
        "surface_kind",
        "dataset_family",
        "representation_id",
        "feature_dim",
        "dtype",
        "labels_stored",
        "row_index_member",
        "shards",
        "cache_index_hash",
    }
    if (
        set(index) != expected_index_keys
        or index.get("schema_version") != "midogpp_harp_label_blind_frame_cache_v1"
        or index.get("artifact_id")
        != (
            "midogpp_harp_router_development_cache_v1"
            if config.contract == ACTION_SURFACE
            else "midogpp_harp_target_support_cache_v1"
        )
        or index.get("surface_kind") != surface_kind
        or index.get("dataset_family") != "MIDOG++"
        or index.get("representation_id")
        != "midogpp_virchow2_common_3840_float32_v1"
        or index.get("feature_dim") != COMMON_OUTPUT_DIM
        or index.get("dtype") != "float32"
        or index.get("labels_stored") is not False
        or index.get("row_index_member") != "tables/row_index.csv"
    ):
        raise ProtocolError("HARP fresh frame-cache protocol drifted.")
    files = content_members(content)
    cache_content_hash = content_hash(content)
    valid_bindings = {
        cache_content_hash,
        sha256_file(content_path),
        str(content.get("cache_binding_hash", "")),
        str(content.get("content_hash", "")),
    }
    if readiness.cache_binding_sha256 not in valid_bindings:
        raise ProtocolError("HARP readiness cache binding escaped fresh content bytes.")
    if files.get("manifests/cache_index.json") != sha256_file(index_path):
        raise ProtocolError("HARP fresh content index does not bind its cache index.")
    if files.get("tables/row_index.csv") != sha256_file(row_path):
        raise ProtocolError("HARP fresh content index does not bind its row index.")
    for relative, digest in files.items():
        member = safe_member(root, relative)
        if not member.is_file() or member.is_symlink() or sha256_file(member) != digest:
            raise ProtocolError(f"HARP fresh cache member drifted: {relative}.")

    shards_raw = index.get("shards")
    if not isinstance(shards_raw, list) or not shards_raw:
        raise ProtocolError("HARP fresh frame cache has no shards.")
    shards: dict[str, tuple[Path, tuple[int, int]]] = {}
    for raw in shards_raw:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "file_sha256",
            "shape",
            "dtype",
        }:
            raise ProtocolError("HARP frame-cache shard schema drifted.")
        relative = str(raw["relative_path"])
        shape = raw["shape"]
        if (
            relative in shards
            or not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) <= 0
            or int(shape[1]) != COMMON_OUTPUT_DIM
            or raw.get("dtype") != "float32"
            or files.get(relative) != raw.get("file_sha256")
        ):
            raise ProtocolError("HARP frame-cache shard inventory drifted.")
        shard_path = safe_member(root, relative)
        array = load_float32_shard(shard_path)
        if array.shape != (int(shape[0]), int(shape[1])):
            raise ProtocolError("HARP frame-cache shard header drifted.")
        shards[relative] = (shard_path, (int(shape[0]), int(shape[1])))

    row_columns, rows = read_frame_rows(row_path)
    expected_columns = (
        "schema_version",
        "row_id",
        "center",
        "case_id",
        "center_row_index",
        "embedding_file",
        "embedding_row_index",
    )
    if row_columns != expected_columns:
        raise ProtocolError("HARP frame-cache row-index columns drifted.")
    centers = tuple(str(value) for value in config.protocol["center_universe"])
    grouped: dict[str, list[dict[str, str]]] = {center: [] for center in centers}
    seen_rows: set[str] = set()
    for raw in rows:
        center = raw.get("center", "")
        row_id = raw.get("row_id", "")
        case_id = raw.get("case_id", "")
        if (
            raw.get("schema_version") != "midogpp_harp_label_blind_frame_row_v1"
            or center not in grouped
            or not row_id
            or row_id in seen_rows
            or not case_id
            or raw.get("embedding_file") not in shards
        ):
            raise ProtocolError("HARP frame-cache row identity drifted.")
        seen_rows.add(row_id)
        grouped[center].append(raw)
    if any(not grouped[center] for center in centers):
        raise ProtocolError("HARP frame cache lacks exact center coverage.")

    staged_rows: list[np.ndarray] = []
    rows_by_center: dict[str, tuple[str, ...]] = {}
    cases_by_center: dict[str, tuple[str, ...]] = {}
    offsets: dict[str, tuple[int, int]] = {}
    frame_hashes: dict[str, str] = {}
    opened: dict[str, np.ndarray] = {}
    cursor = 0
    for center in centers:
        center_rows = grouped[center]
        row_ids: list[str] = []
        case_ids: list[str] = []
        center_values: list[np.ndarray] = []
        for ordinal, raw in enumerate(center_rows):
            try:
                center_ordinal = int(raw["center_row_index"])
                embedding_ordinal = int(raw["embedding_row_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("HARP frame-cache row offset is malformed.") from exc
            relative = raw["embedding_file"]
            if center_ordinal != ordinal or embedding_ordinal < 0:
                raise ProtocolError("HARP frame-cache center row order drifted.")
            if relative not in opened:
                opened[relative] = load_float32_shard(shards[relative][0])
            shard = opened[relative]
            if embedding_ordinal >= len(shard):
                raise ProtocolError("HARP frame-cache row offset exceeds its shard.")
            value = np.asarray(shard[embedding_ordinal], dtype=np.float32)
            if value.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(value).all():
                raise ProtocolError("HARP frame-cache row values drifted.")
            center_values.append(value)
            row_ids.append(raw["row_id"])
            case_ids.append(raw["case_id"])
        matrix = np.ascontiguousarray(np.stack(center_values), dtype=np.float32)
        staged_rows.append(matrix)
        rows_by_center[center] = tuple(row_ids)
        cases_by_center[center] = tuple(case_ids)
        offsets[center] = (cursor, cursor + len(matrix))
        frame_hashes[center] = canonical_hash(
            {
                "schema_version": "midogpp_harp_query_frame_slice_v1",
                "center": center,
                "row_ids": row_ids,
                "case_ids": case_ids,
                "embedding_bytes_sha256": hashlib.sha256(
                    matrix.tobytes(order="C")
                ).hexdigest(),
                "cache_content_hash": cache_content_hash,
            }
        )
        cursor += len(matrix)
    all_values = np.ascontiguousarray(np.concatenate(staged_rows), dtype=np.float32)
    array_path = config.artifact_root / FRAME_ARRAY_MEMBER
    atomic_npy(array_path, all_values)
    receipt_unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_staged_frame_receipt_v1",
        "source_cache_root": str(root),
        "cache_content_hash": cache_content_hash,
        "cache_binding_hash": readiness.cache_binding_sha256,
        "array_member": FRAME_ARRAY_MEMBER,
        "array_sha256": sha256_file(array_path),
        "shape": list(all_values.shape),
        "dtype": "float32",
        "rows_by_center": {center: list(rows_by_center[center]) for center in centers},
        "cases_by_center": {center: list(cases_by_center[center]) for center in centers},
        "offsets_by_center": {
            center: list(offsets[center]) for center in centers
        },
        "frame_hash_by_center": frame_hashes,
        "labels_stored": False,
    }
    receipt = {**receipt_unhashed, "receipt_hash": canonical_hash(receipt_unhashed)}
    atomic_json(config.artifact_root / FRAME_RECEIPT_MEMBER, receipt)
    return _FrameCache(
        array_path=array_path,
        rows_by_center=MappingProxyType(rows_by_center),
        cases_by_center=MappingProxyType(cases_by_center),
        offsets_by_center=MappingProxyType(offsets),
        frame_hash_by_center=MappingProxyType(frame_hashes),
        cache_binding_hash=readiness.cache_binding_sha256,
        receipt_hash=str(receipt["receipt_hash"]),
    )


def _physical_classifier_tasks(
    config: HarpStage60Config,
    plan: HarpWorkstationPlan,
    *,
    frame: _FrameCache,
    source_cache: FrozenSourceStreamCache,
    generation_lock: GenerationLock,
    classifier: ClassifierSpec,
    lineage: _AuthoritativeLineage,
) -> tuple[dict[str, object], ...]:
    action_lookup = {
        action.action_hash: action
        for action in (
            build_all_development_actions()
            if plan.surface_kind == "development"
            else build_all_target_actions()
        )
    }
    records = [record.to_payload() for record in source_cache.records]
    tasks: list[dict[str, object]] = []
    checkpoint_root = config.artifact_root / CLASSIFIER_CHECKPOINT_ROOT
    for planned in plan.classifier_tasks:
        start, stop = frame.offsets_by_center[planned.query_center_id]
        actions = [action_lookup[value].to_payload() for value in planned.action_hashes]
        stem = f"task_{planned.ordinal:04d}"
        unhashed: dict[str, object] = {
            "schema_version": "midogpp_harp_physical_classifier_task_v2",
            "planned_task_hash": planned.task_hash,
            "surface_kind": planned.surface_kind,
            "outer_target_id": planned.outer_target_id,
            "query_center_id": planned.query_center_id,
            "training_seed": planned.training_seed,
            "generation_seed": planned.generation_seed,
            "actions": actions,
            "source_array_path": str(source_cache.source_array_path.resolve()),
            "source_array_sha256": str(source_cache.lock_payload["source_array_sha256"]),
            "source_records": records,
            "source_stream_index_hash": str(
                source_cache.lock_payload["source_stream_index_hash"]
            ),
            "frame_array_path": str(frame.array_path.resolve()),
            "frame_array_sha256": sha256_file(frame.array_path),
            "frame_start": start,
            "frame_stop": stop,
            "row_ids": list(frame.rows_by_center[planned.query_center_id]),
            "case_ids": list(frame.cases_by_center[planned.query_center_id]),
            "frame_hash": frame.frame_hash_by_center[planned.query_center_id],
            "frame_receipt_hash": frame.receipt_hash,
            "generation_lock_semantic_hash": generation_lock.generation_lock_hash,
            "classifier": classifier.to_payload(),
            "threads_per_worker": 3,
            "bank_semantic_lock_hash": lineage.bank_semantic_lock_hash,
            "generation_semantic_lock_hash": lineage.generation_semantic_lock_hash,
            "source_stream_lock_hash": lineage.source_stream_lock_hash,
            "source_stream_index_hash": lineage.source_stream_index_hash,
            "source_stream_content_hash": lineage.source_stream_content_hash,
            "classifier_config_hash": lineage.classifier_config_hash,
            "expert_bank_index_sha256": lineage.expert_bank_index_sha256,
            "generation_lock_file_sha256": lineage.generation_lock_file_sha256,
            "source_cache_lock_sha256": lineage.source_cache_lock_sha256,
            "source_cache_index_sha256": lineage.source_cache_index_sha256,
            "source_stream_artifact_binding_hash": (
                lineage.source_stream_artifact_binding_hash
            ),
            "classifier_contract_sha256": lineage.classifier_contract_sha256,
            "labels_available": False,
            "nested_process_pools": False,
        }
        task_hash = canonical_hash(unhashed)
        tasks.append(
            {
                **unhashed,
                "task_hash": task_hash,
                "checkpoint_npz_path": str(checkpoint_root / f"{stem}.npz"),
                "checkpoint_json_path": str(checkpoint_root / f"{stem}.json"),
            }
        )
    return tuple(tasks)


def _execute_classifier_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    completed: dict[int, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for ordinal, task in enumerate(tasks):
        existing = load_classifier_checkpoint(
            task, validate_current_inputs=_load_worker_inputs
        )
        if existing is None:
            pending.append(task)
        else:
            completed[ordinal] = existing
    if pending:
        with ProcessPoolExecutor(
            max_workers=4, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(_classifier_worker, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                verified = load_classifier_checkpoint(
                    task, validate_current_inputs=_load_worker_inputs
                )
                if verified is None:
                    raise ProtocolError("HARP classifier worker returned no checkpoint.")
                ordinal = next(
                    index for index, candidate in enumerate(tasks) if candidate is task
                )
                completed[ordinal] = verified
                print(
                    f"[harp-stage60] classifier tasks {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if set(completed) != set(range(len(tasks))):
        raise ProtocolError("HARP classifier checkpoint coverage is incomplete.")
    return completed


def _classifier_worker(task: Mapping[str, object]) -> Mapping[str, object]:
    source_blocks, evaluation, actions, classifier = _load_worker_inputs(task)
    probabilities: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("HARP classifier workers require threadpoolctl.") from exc
    with threadpool_limits(limits=3):
        for action in actions:
            composition = compose_harp_action(
                {source: source_blocks[source] for source in action.source_order},
                action,
                shuffle_seed_by_class={
                    label: harp_composition_seed(
                        generation_lock_hash=str(task["generation_lock_semantic_hash"]),
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
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or matrix.shape != (len(evaluation), 2)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != classifier.config_hash
            ):
                raise ProtocolError("HARP physical classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
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
    values = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    atomic_npz(npz_path, probabilities=values)
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_classifier_checkpoint_v2",
        "status": "COMPLETE",
        "task_hash": task["task_hash"],
        "npz_sha256": sha256_file(npz_path),
        "shape": list(values.shape),
        "dtype": "float32",
        "actions": records,
        "labels_consumed": False,
        "nested_process_pools": False,
        "late_torch_interop_setter_used": False,
    }
    payload = {**unhashed, "checkpoint_hash": canonical_hash(unhashed)}
    atomic_json(Path(str(task["checkpoint_json_path"])), payload)
    return payload


def _load_worker_inputs(
    task: Mapping[str, object],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, tuple[HarpActionSpec, ...], ClassifierSpec]:
    unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_npz_path", "checkpoint_json_path"}
    }
    if (
        task.get("schema_version") != "midogpp_harp_physical_classifier_task_v2"
        or task.get("task_hash") != canonical_hash(unhashed)
        or task.get("labels_available") is not False
        or task.get("nested_process_pools") is not False
        or int(task.get("threads_per_worker", -1)) != 3
    ):
        raise ProtocolError("HARP physical classifier task drifted.")
    outer = str(task["outer_target_id"])
    query = str(task["query_center_id"])
    raw_actions = task.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ProtocolError("HARP physical task has no actions.")
    actions: list[HarpActionSpec] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP physical task action is malformed.")
        action = HarpActionSpec(
            surface_kind=str(raw["surface_kind"]),
            outer_target_id=outer,
            query_center_id=query,
            selected_source_id=(
                None
                if raw.get("selected_source_id") is None
                else str(raw["selected_source_id"])
            ),
            action_id=str(raw["action_id"]),
        )
        if action.to_payload() != dict(raw):
            raise ProtocolError("HARP physical task action payload drifted.")
        actions.append(action)
    raw_records = task.get("source_records")
    if not isinstance(raw_records, list):
        raise ProtocolError("HARP physical task source index is absent.")
    record_by_key = {
        (
            str(raw["source_center"]),
            int(raw["training_seed"]),
            int(raw["generation_seed"]),
        ): raw
        for raw in raw_records
        if isinstance(raw, Mapping)
    }
    source_path = Path(str(task["source_array_path"]))
    if sha256_file(source_path) != task.get("source_array_sha256"):
        raise ProtocolError("HARP physical task source array bytes drifted.")
    source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    blocks: dict[str, dict[str, np.ndarray]] = {}
    required_sources = tuple(sorted({source for action in actions for source in action.source_order}))
    for source in required_sources:
        key = (source, int(task["training_seed"]), int(task["generation_seed"]))
        record = record_by_key.get(key)
        if record is None:
            raise ProtocolError("HARP physical task source block is absent.")
        block = np.asarray(source_values[int(record["block_ordinal"])], dtype=np.float32)
        if source_block_sha256(block) != record.get("output_sha256"):
            raise ProtocolError("HARP physical task source block bytes drifted.")
        blocks[source] = {
            "embeddings": block,
            "labels": np.concatenate(
                (
                    np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                    np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                )
            ),
        }
    frame_path = Path(str(task["frame_array_path"]))
    if sha256_file(frame_path) != task.get("frame_array_sha256"):
        raise ProtocolError("HARP physical task frame array bytes drifted.")
    frame = np.load(frame_path, mmap_mode="r", allow_pickle=False)
    start, stop = int(task["frame_start"]), int(task["frame_stop"])
    evaluation = np.ascontiguousarray(frame[start:stop], dtype=np.float32)
    if (
        evaluation.shape != (len(task["row_ids"]), COMMON_OUTPUT_DIM)
        or len(task["row_ids"]) != len(task["case_ids"])
        or not np.isfinite(evaluation).all()
    ):
        raise ProtocolError("HARP physical task frame slice drifted.")
    classifier_raw = task.get("classifier")
    if not isinstance(classifier_raw, Mapping):
        raise ProtocolError("HARP physical task classifier is absent.")
    classifier = ClassifierSpec(**dict(classifier_raw))
    return blocks, evaluation, tuple(actions), classifier


def _menu_from_checkpoints(
    plan: HarpWorkstationPlan,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[int, Mapping[str, object]],
    *,
    lineage: _AuthoritativeLineage,
) -> HarpPredictionMenuSeal:
    cells_by_key: dict[tuple[str, int, int], HarpPredictionCell] = {}
    action_by_hash: dict[str, HarpActionSpec] = {}
    for ordinal, task in enumerate(tasks):
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
                action_id=str(raw_action["action_id"]),
            )
            action_by_hash[action.action_hash] = action
            cell = HarpPredictionCell(
                action=action,
                training_seed=int(task["training_seed"]),
                generation_seed=int(task["generation_seed"]),
                row_ids=tuple(str(value) for value in task["row_ids"]),
                case_ids=tuple(str(value) for value in task["case_ids"]),
                probabilities=np.ascontiguousarray(values[action_ordinal], dtype=np.float32),
                bank_hash=lineage.bank_semantic_lock_hash,
                generation_lock_hash=lineage.generation_semantic_lock_hash,
                source_cache_hash=lineage.source_stream_lock_hash,
                frame_hash=str(task["frame_hash"]),
                classifier_hash=lineage.classifier_config_hash,
                composition_hash=str(record["composition_hash"]),
                scaler_state_hash=str(record["scaler_state_hash"]),
            )
            cells_by_key[(action.action_hash, cell.training_seed, cell.generation_seed)] = cell
    actions = tuple(action_by_hash[value] for value in plan.action_hashes)
    cells = tuple(
        cells_by_key[(action.action_hash, training_seed, generation_seed)]
        for action in actions
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
    )
    return seal_harp_prediction_menu(actions, cells)


__all__ = ("materialize_physical_harp_menu",)
