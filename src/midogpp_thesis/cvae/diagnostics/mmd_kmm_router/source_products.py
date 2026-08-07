"""Dual-GPU, resumable source generation and compatibility products."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import gc
import hashlib
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...generation.generation import generate_source_block, source_generation_plan
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import (
    DEFAULT_SCALE_FLOOR,
    ReplicaKey,
    calibrate_own_source_energies,
    score_variational_compatibility,
)
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)
from .inputs import LabelFreeValidationFrame, PartitionSurface


SOURCE_BLOCK_ARRAY_MEMBER = "arrays/source_prefix_blocks.npy"
SOURCE_BLOCK_INDEX_MEMBER = "tables/source_block_index.csv"
COMPATIBILITY_CASE_MEMBER = "tables/compatibility_case_energy.csv"
COMPATIBILITY_SCORE_MEMBER = "tables/compatibility_scores.csv"
SOURCE_PRODUCTS_LOCK_MEMBER = "manifests/source_products_lock.json"

SOURCE_BLOCK_INDEX_COLUMNS = (
    "schema_version",
    "block_ordinal",
    "source_center",
    "training_seed",
    "generation_seed",
    "stream_id",
    "expert_lock_hash",
    "samples_per_class",
    "row_count",
    "feature_dim",
    "output_sha256",
)
COMPATIBILITY_CASE_COLUMNS = (
    "schema_version",
    "source_center",
    "training_seed",
    "query_center",
    "case_id",
    "query_partition_role",
    "row_count",
    "marginal_variational_energy",
    "class_0_energy",
    "class_1_energy",
    "class_0_common_reconstruction_mse",
    "class_1_common_reconstruction_mse",
    "class_0_normalized_ps_kl",
    "class_1_normalized_ps_kl",
    "class_prior_json",
    "labels_used",
    "exact_nelbo_claimed",
)
COMPATIBILITY_SCORE_COLUMNS = (
    "schema_version",
    "query_center",
    "source_center",
    "training_seed_17_z",
    "training_seed_42_z",
    "training_seed_101_z",
    "mean_calibrated_energy_z",
    "query_support_case_count",
    "replica_aggregation",
    "legal_target_candidate",
    "query_support_labels_used",
    "exact_nelbo_claimed",
)


@dataclass(frozen=True)
class SourceProducts:
    array_path: Path
    index_rows: tuple[Mapping[str, object], ...]
    compatibility_case_rows: tuple[Mapping[str, object], ...]
    compatibility_score_rows: tuple[Mapping[str, object], ...]
    calibrated_energy_by_target: Mapping[str, Mapping[str, float]]

    @property
    def block_ordinal_by_key(self) -> dict[tuple[str, int, int], int]:
        return {
            (
                str(row["source_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            ): int(row["block_ordinal"])
            for row in self.index_rows
        }

    @property
    def source_products_hash(self) -> str:
        return stable_hash(
            {
                "index_rows": [
                    _canonical_source_index_row(row) for row in self.index_rows
                ],
                "compatibility_scores": [
                    _canonical_compatibility_score_row(row)
                    for row in self.compatibility_score_rows
                ],
            }
        )


def materialize_source_products(
    config: MMDKMMRouterDiagnosticConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    *,
    root: Path,
) -> SourceProducts:
    """Generate all source prefixes once with one persistent process per GPU."""

    final_members = (
        root / SOURCE_BLOCK_ARRAY_MEMBER,
        root / SOURCE_BLOCK_INDEX_MEMBER,
        root / COMPATIBILITY_CASE_MEMBER,
        root / COMPATIBILITY_SCORE_MEMBER,
    )
    if all(path.is_file() for path in final_members):
        lock_path = root / SOURCE_PRODUCTS_LOCK_MEMBER
        if lock_path.is_file():
            products = load_source_products(root)
            validate_source_products_lock(
                root,
                config=config,
                generation_lock=generation_lock,
                frame=frame,
                partitions=partitions,
                source_products=products,
            )
            shutil.rmtree(root / "checkpoints/generation", ignore_errors=True)
            return products

    checkpoint_root = root / "checkpoints/generation"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    support_array_path = checkpoint_root / "support_embeddings.npy"
    support_index_path = checkpoint_root / "support_index.json"
    support_scratch = _write_support_scratch(
        support_array_path,
        support_index_path,
        frame=frame,
        partitions=partitions,
    )
    keys = source_generation_plan(generation_lock)
    key_map = {
        (key.source_center, key.training_seed, key.generation_seed): key for key in keys
    }
    expected_keys = {
        (source, training_seed, generation_seed)
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if len(keys) != EXPECTED_SOURCE_BLOCK_COUNT or set(key_map) != expected_keys:
        raise ProtocolError("MMD/KMM GenerationLock source grid drifted.")

    tasks: list[dict[str, object]] = []
    for task_ordinal, (source, training_seed) in enumerate(
        product(CENTERS, TRAINING_SEEDS)
    ):
        task_keys = tuple(key_map[(source, training_seed, seed)] for seed in GENERATION_SEEDS)
        checkpoint = checkpoint_root / f"source_{source}_train_{training_seed}.json"
        array_path = checkpoint_root / f"source_{source}_train_{training_seed}.npy"
        tasks.append(
            {
                "task_ordinal": task_ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "generation_keys": task_keys,
                "device": GENERATION_DEVICES[task_ordinal % len(GENERATION_DEVICES)],
                "expert_bank_root": str(config.expert_bank_root),
                "support_array_path": str(support_array_path),
                "support_index_path": str(support_index_path),
                "checkpoint_path": str(checkpoint),
                "array_path": str(array_path),
                "config_contract_hash": config.contract_hash,
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "support_scratch_hash": support_scratch["support_scratch_hash"],
            }
        )
    if len(tasks) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("MMD/KMM source-task scheduler drifted.")

    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[dict[str, object]] = []
    for task in tasks:
        checkpoint = Path(str(task["checkpoint_path"]))
        if not checkpoint.is_file():
            pending.append(task)
            continue
        payload = _load_generation_checkpoint(checkpoint, task=task)
        completed[(str(task["source_center"]), int(task["training_seed"]))] = payload

    if pending:
        context = mp.get_context("spawn")
        executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=context)
            for _ in GENERATION_DEVICES
        ]
        future_to_task: dict[Future[dict[str, object]], dict[str, object]] = {}
        try:
            for task in pending:
                device_index = GENERATION_DEVICES.index(str(task["device"]))
                future = executors[device_index].submit(_generate_source_task, task)
                future_to_task[future] = task
            for completed_count, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                payload = future.result()
                verified = _load_generation_checkpoint(
                    Path(str(task["checkpoint_path"])), task=task
                )
                if payload.get("checkpoint_hash") != verified.get("checkpoint_hash"):
                    raise ProtocolError("MMD/KMM worker checkpoint return drifted.")
                completed[(str(task["source_center"]), int(task["training_seed"]))] = verified
                print(
                    f"[mmd-kmm] source jobs {len(completed)}/{EXPECTED_SOURCE_TASK_COUNT} "
                    f"(new {completed_count}/{len(pending)})",
                    flush=True,
                )
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    if len(completed) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("MMD/KMM source-task checkpoint coverage is incomplete.")

    array_path = root / SOURCE_BLOCK_ARRAY_MEMBER
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temp_array = array_path.with_name(array_path.name + ".tmp")
    target = np.lib.format.open_memmap(
        temp_array,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_SOURCE_BLOCK_COUNT, 2 * MAX_SOURCE_PREFIX_PER_CLASS, 3840),
    )
    index_rows: list[dict[str, object]] = []
    ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            payload = completed[(source, training_seed)]
            source_array = np.load(Path(str(payload["array_path"])), mmap_mode="r")
            if source_array.shape != (len(GENERATION_SEEDS), 2 * MAX_SOURCE_PREFIX_PER_CLASS, 3840):
                raise ProtocolError("MMD/KMM source-task array geometry drifted.")
            records = payload.get("blocks")
            if not isinstance(records, list) or len(records) != len(GENERATION_SEEDS):
                raise ProtocolError("MMD/KMM source-task block inventory drifted.")
            for generation_index, generation_seed in enumerate(GENERATION_SEEDS):
                record = records[generation_index]
                if not isinstance(record, Mapping) or int(record.get("generation_seed", -1)) != generation_seed:
                    raise ProtocolError("MMD/KMM source-task block order drifted.")
                target[ordinal] = source_array[generation_index]
                key = key_map[(source, training_seed, generation_seed)]
                index_rows.append(
                    {
                        "schema_version": "midogpp_mmd_kmm_source_block_v1",
                        "block_ordinal": ordinal,
                        "source_center": source,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "stream_id": key.stream_id,
                        "expert_lock_hash": key.expert_lock_hash,
                        "samples_per_class": MAX_SOURCE_PREFIX_PER_CLASS,
                        "row_count": 2 * MAX_SOURCE_PREFIX_PER_CLASS,
                        "feature_dim": 3840,
                        "output_sha256": str(record["output_sha256"]),
                    }
                )
                ordinal += 1
    target.flush()
    del target
    os.replace(temp_array, array_path)

    case_rows, score_rows, calibrated = _build_compatibility_tables(
        completed,
        partitions=partitions,
    )
    from ...reporting import write_csv_rows

    write_csv_rows(root / SOURCE_BLOCK_INDEX_MEMBER, index_rows, columns=SOURCE_BLOCK_INDEX_COLUMNS)
    write_csv_rows(root / COMPATIBILITY_CASE_MEMBER, case_rows, columns=COMPATIBILITY_CASE_COLUMNS)
    write_csv_rows(root / COMPATIBILITY_SCORE_MEMBER, score_rows, columns=COMPATIBILITY_SCORE_COLUMNS)
    products = SourceProducts(
        array_path=array_path,
        index_rows=tuple(index_rows),
        compatibility_case_rows=tuple(case_rows),
        compatibility_score_rows=tuple(score_rows),
        calibrated_energy_by_target=calibrated,
    )
    _atomic_json(
        root / SOURCE_PRODUCTS_LOCK_MEMBER,
        build_source_products_lock(
            root,
            config=config,
            generation_lock=generation_lock,
            frame=frame,
            partitions=partitions,
            source_products=products,
        ),
    )
    # The phase lock is the durable resume capability. Retain the task
    # checkpoints until that input/config binding has been persisted.
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return products


def load_source_products(root: Path) -> SourceProducts:
    array_path = root / SOURCE_BLOCK_ARRAY_MEMBER
    index_rows = tuple(_read_csv(root / SOURCE_BLOCK_INDEX_MEMBER))
    case_rows = tuple(_read_csv(root / COMPATIBILITY_CASE_MEMBER))
    score_rows = tuple(_read_csv(root / COMPATIBILITY_SCORE_MEMBER))
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (EXPECTED_SOURCE_BLOCK_COUNT, 2 * MAX_SOURCE_PREFIX_PER_CLASS, 3840) or array.dtype != np.float32:
        raise ProtocolError("MMD/KMM final source cache geometry drifted.")
    if len(index_rows) != EXPECTED_SOURCE_BLOCK_COUNT:
        raise ProtocolError("MMD/KMM final source cache index coverage drifted.")
    calibrated: dict[str, dict[str, float]] = {center: {} for center in CENTERS}
    for row in score_rows:
        calibrated[str(row["query_center"])][str(row["source_center"])] = float(
            row["mean_calibrated_energy_z"]
        )
    if any(set(calibrated[target]) != set(CENTERS).difference({target}) for target in CENTERS):
        raise ProtocolError("MMD/KMM compatibility target/source coverage drifted.")
    return SourceProducts(
        array_path=array_path,
        index_rows=index_rows,
        compatibility_case_rows=case_rows,
        compatibility_score_rows=score_rows,
        calibrated_energy_by_target=calibrated,
    )


def build_source_products_lock(
    root: Path,
    *,
    config: MMDKMMRouterDiagnosticConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_products: SourceProducts,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_mmd_kmm_source_products_lock_v1",
        "status": "COMPLETE",
        "config_contract_hash": config.contract_hash,
        "bank_lock_hash": generation_lock.bank_lock_hash,
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_products_hash": source_products.source_products_hash,
        "source_array_sha256": _sha256_file(root / SOURCE_BLOCK_ARRAY_MEMBER),
        "source_index_sha256": _sha256_file(root / SOURCE_BLOCK_INDEX_MEMBER),
        "compatibility_case_sha256": _sha256_file(root / COMPATIBILITY_CASE_MEMBER),
        "compatibility_score_sha256": _sha256_file(root / COMPATIBILITY_SCORE_MEMBER),
        "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
        "expert_load_count": EXPECTED_SOURCE_TASK_COUNT,
        "samples_per_source_class": MAX_SOURCE_PREFIX_PER_CLASS,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return {**unhashed, "source_products_lock_hash": stable_hash(unhashed)}


def validate_source_products_lock(
    root: Path,
    *,
    config: MMDKMMRouterDiagnosticConfig,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_products: SourceProducts,
) -> Mapping[str, object]:
    observed = _json(root / SOURCE_PRODUCTS_LOCK_MEMBER)
    expected = build_source_products_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_products=source_products,
    )
    if observed != expected:
        raise ProtocolError(
            "MMD/KMM source products are not bound to the current inputs/config."
        )
    return observed


def _write_support_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
) -> Mapping[str, object]:
    rows = [row for center in CENTERS for row in partitions.support_rows_by_center[center]]
    embeddings = frame.embeddings_for(rows)
    offsets: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        center_rows = partitions.support_rows_by_center[center]
        stop = cursor + len(center_rows)
        offsets[center] = {
            "start": cursor,
            "stop": stop,
            "case_ids": [row.case_id for row in center_rows],
        }
        cursor = stop
    payload: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_support_scratch_v1",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "offsets": offsets,
        "array_sha256": _sha256_array(embeddings),
    }
    payload["support_scratch_hash"] = stable_hash(payload)
    _atomic_save_npy(array_path, embeddings)
    _atomic_json(index_path, payload)
    return payload


def _generate_source_task(task: Mapping[str, object]) -> dict[str, object]:
    source = str(task["source_center"])
    training_seed = int(task["training_seed"])
    device = str(task["device"])
    keys = tuple(task["generation_keys"])
    if not all(isinstance(key, SourceGenerationKey) for key in keys):
        raise ProtocolError("MMD/KMM worker received invalid generation keys.")
    support_index = _json(Path(str(task["support_index_path"])))
    support = np.load(Path(str(task["support_array_path"])), mmap_mode="r")
    scratch_unhashed = {
        key: value for key, value in support_index.items() if key != "support_scratch_hash"
    }
    if (
        support_index.get("support_scratch_hash") != stable_hash(scratch_unhashed)
        or support_index.get("support_scratch_hash") != task["support_scratch_hash"]
        or support_index.get("shape") != list(support.shape)
        or support_index.get("dtype") != str(support.dtype)
        or support_index.get("array_sha256") != _sha256_array(support)
    ):
        raise ProtocolError("MMD/KMM support scratch failed validation.")
    checkpoint_path = Path(str(task["checkpoint_path"]))
    array_path = Path(str(task["array_path"]))
    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=source,
        training_seed=training_seed,
        device=device,
    )
    try:
        energies: list[dict[str, object]] = []
        offsets = support_index.get("offsets")
        if not isinstance(offsets, Mapping):
            raise ProtocolError("MMD/KMM support scratch offsets are malformed.")
        for query in CENTERS:
            raw = offsets.get(query)
            if not isinstance(raw, Mapping):
                raise ProtocolError("MMD/KMM support scratch lacks a query center.")
            start, stop = int(raw["start"]), int(raw["stop"])
            case_ids = tuple(str(value) for value in raw["case_ids"])
            query_embeddings = np.ascontiguousarray(support[start:stop], dtype=np.float32)
            energy = score_variational_compatibility(expert, query_embeddings, case_ids)
            cases = np.asarray(case_ids, dtype=object)
            for case_id in energy.case_order:
                mask = cases == case_id
                energies.append(
                    {
                        "source_center": source,
                        "training_seed": training_seed,
                        "query_center": query,
                        "case_id": case_id,
                        "row_count": int(np.sum(mask)),
                        "marginal_variational_energy": float(energy.per_case[case_id]),
                        "class_0_energy": float(np.mean(energy.per_class_energy[0][mask])),
                        "class_1_energy": float(np.mean(energy.per_class_energy[1][mask])),
                        "class_0_common_reconstruction_mse": float(np.mean(energy.per_class_reconstruction_mse[0][mask])),
                        "class_1_common_reconstruction_mse": float(np.mean(energy.per_class_reconstruction_mse[1][mask])),
                        "class_0_normalized_ps_kl": float(np.mean(energy.per_class_normalized_ps_kl[0][mask])),
                        "class_1_normalized_ps_kl": float(np.mean(energy.per_class_normalized_ps_kl[1][mask])),
                    }
                )
        blocks = []
        arrays = []
        for key in keys:
            block = generate_source_block(
                expert,
                key,
                per_class=MAX_SOURCE_PREFIX_PER_CLASS,
                device=device,
            )
            arrays.append(np.asarray(block.embeddings, dtype=np.float32))
            blocks.append(
                {
                    "generation_seed": key.generation_seed,
                    "stream_id": key.stream_id,
                    "output_sha256": block.output_sha256,
                }
            )
        task_array = np.ascontiguousarray(np.stack(arrays), dtype=np.float32)
        _atomic_save_npy(array_path, task_array)
        unhashed: dict[str, object] = {
            "schema_version": "midogpp_mmd_kmm_generation_checkpoint_v1",
            "status": "COMPLETE",
            "config_contract_hash": str(task["config_contract_hash"]),
            "generation_lock_hash": str(task["generation_lock_hash"]),
            "support_scratch_hash": str(task["support_scratch_hash"]),
            "task_ordinal": int(task["task_ordinal"]),
            "source_center": source,
            "training_seed": training_seed,
            "device": device,
            "array_path": str(array_path),
            "array_file_sha256": _sha256_file(array_path),
            "blocks": blocks,
            "compatibility_case_records": energies,
            "target_labels_used": False,
            "support_labels_used": False,
        }
        payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
        _atomic_json(checkpoint_path, payload)
        return payload
    finally:
        del expert
        gc.collect()
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _load_generation_checkpoint(
    path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    if not path.is_file():
        raise ProtocolError("MMD/KMM generation checkpoint is absent.")
    payload = _json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    array_path = Path(str(payload.get("array_path", "")))
    records = payload.get("blocks")
    expected_keys = tuple(task.get("generation_keys", ()))
    block_binding_valid = (
        isinstance(records, list)
        and len(records) == len(expected_keys)
        and all(
            isinstance(record, Mapping)
            and int(record.get("generation_seed", -1)) == key.generation_seed
            and record.get("stream_id") == key.stream_id
            and isinstance(record.get("output_sha256"), str)
            and len(str(record["output_sha256"])) == 64
            for record, key in zip(records, expected_keys, strict=True)
        )
    )
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("status") != "COMPLETE"
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("generation_lock_hash") != task["generation_lock_hash"]
        or payload.get("support_scratch_hash") != task["support_scratch_hash"]
        or int(payload.get("task_ordinal", -1)) != int(task["task_ordinal"])
        or payload.get("source_center") != task["source_center"]
        or int(payload.get("training_seed", -1)) != int(task["training_seed"])
        or payload.get("device") != task["device"]
        or array_path != Path(str(task["array_path"]))
        or not array_path.is_file()
        or payload.get("array_file_sha256") != _sha256_file(array_path)
        or not block_binding_valid
    ):
        raise ProtocolError("MMD/KMM generation checkpoint failed validation.")
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (3, 2 * MAX_SOURCE_PREFIX_PER_CLASS, 3840) or array.dtype != np.float32:
        raise ProtocolError("MMD/KMM generation checkpoint array drifted.")
    return payload


def _build_compatibility_tables(
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    partitions: PartitionSurface,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, float]]]:
    energy_by_key: dict[tuple[str, int, str], dict[str, float]] = {}
    case_rows: list[dict[str, object]] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            records = completed[(source, training_seed)].get("compatibility_case_records")
            if not isinstance(records, list):
                raise ProtocolError("MMD/KMM compatibility checkpoint records are absent.")
            for raw in records:
                if not isinstance(raw, Mapping):
                    raise ProtocolError("MMD/KMM compatibility checkpoint row is malformed.")
                query = str(raw["query_center"])
                case_id = str(raw["case_id"])
                energy_by_key.setdefault((source, training_seed, query), {})[case_id] = float(
                    raw["marginal_variational_energy"]
                )
                case_rows.append(
                    {
                        "schema_version": "midogpp_mmd_kmm_compatibility_case_energy_v1",
                        **dict(raw),
                        "query_partition_role": "support",
                        "class_prior_json": "[0.5,0.5]",
                        "labels_used": False,
                        "exact_nelbo_claimed": False,
                    }
                )
    score_rows: list[dict[str, object]] = []
    calibrated_by_target: dict[str, dict[str, float]] = {}
    for query in CENTERS:
        candidates = tuple(center for center in CENTERS if center != query)
        query_map = {
            ReplicaKey(source, seed): energy_by_key[(source, seed, query)]
            for source in candidates
            for seed in TRAINING_SEEDS
        }
        own_map = {
            ReplicaKey(source, seed): energy_by_key[(source, seed, source)]
            for source in candidates
            for seed in TRAINING_SEEDS
        }
        calibration = calibrate_own_source_energies(
            query_map,
            own_map,
            candidate_sources=candidates,
            training_seeds=TRAINING_SEEDS,
            scale_floor=DEFAULT_SCALE_FLOOR,
        )
        calibrated_by_target[query] = dict(calibration.mean_z_by_source)
        replica = {(row.source_center, row.training_seed): row for row in calibration.replicas}
        support_cases = {row.case_id for row in partitions.support_rows_by_center[query]}
        for source in candidates:
            score_rows.append(
                {
                    "schema_version": "midogpp_mmd_kmm_compatibility_score_v1",
                    "query_center": query,
                    "source_center": source,
                    "training_seed_17_z": replica[(source, 17)].calibrated_z,
                    "training_seed_42_z": replica[(source, 42)].calibrated_z,
                    "training_seed_101_z": replica[(source, 101)].calibrated_z,
                    "mean_calibrated_energy_z": calibration.mean_z_by_source[source],
                    "query_support_case_count": len(support_cases),
                    "replica_aggregation": "arithmetic_mean_all_three_no_seed_selection",
                    "legal_target_candidate": True,
                    "query_support_labels_used": False,
                    "exact_nelbo_claimed": False,
                }
            )
    case_rows.sort(key=lambda row: (str(row["source_center"]), int(row["training_seed"]), str(row["query_center"]), str(row["case_id"])))
    score_rows.sort(key=lambda row: (str(row["query_center"]), str(row["source_center"])))
    return case_rows, score_rows, calibrated_by_target


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read MMD/KMM table: {path}.") from exc


def _canonical_source_index_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": str(row["schema_version"]),
        "block_ordinal": int(row["block_ordinal"]),
        "source_center": str(row["source_center"]),
        "training_seed": int(row["training_seed"]),
        "generation_seed": int(row["generation_seed"]),
        "stream_id": str(row["stream_id"]),
        "expert_lock_hash": str(row["expert_lock_hash"]),
        "samples_per_class": int(row["samples_per_class"]),
        "row_count": int(row["row_count"]),
        "feature_dim": int(row["feature_dim"]),
        "output_sha256": str(row["output_sha256"]),
    }


def _canonical_compatibility_score_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": str(row["schema_version"]),
        "query_center": str(row["query_center"]),
        "source_center": str(row["source_center"]),
        "training_seed_17_z": float(row["training_seed_17_z"]),
        "training_seed_42_z": float(row["training_seed_42_z"]),
        "training_seed_101_z": float(row["training_seed_101_z"]),
        "mean_calibrated_energy_z": float(row["mean_calibrated_energy_z"]),
        "query_support_case_count": int(row["query_support_case_count"]),
        "replica_aggregation": str(row["replica_aggregation"]),
        "legal_target_candidate": _truthy(row["legal_target_candidate"]),
        "query_support_labels_used": _truthy(row["query_support_labels_used"]),
        "exact_nelbo_claimed": _truthy(row["exact_nelbo_claimed"]),
    }


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM checkpoint JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("MMD/KMM checkpoint JSON must be an object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_CASE_MEMBER",
    "COMPATIBILITY_SCORE_COLUMNS",
    "COMPATIBILITY_SCORE_MEMBER",
    "SOURCE_BLOCK_ARRAY_MEMBER",
    "SOURCE_BLOCK_INDEX_COLUMNS",
    "SOURCE_BLOCK_INDEX_MEMBER",
    "SOURCE_PRODUCTS_LOCK_MEMBER",
    "SourceProducts",
    "build_source_products_lock",
    "load_source_products",
    "materialize_source_products",
    "validate_source_products_lock",
)
