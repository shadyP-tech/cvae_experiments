"""Spawn-safe persistent-GPU worker for the Stage-90 exact-tail cache."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, SourceGenerationKey
from ...generation.generation import generate_source_block
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import score_variational_compatibility
from .source_cache_contracts import SOURCE_ROWS_PER_CLASS
from .source_cache_store import (
    atomic_save_npy,
    atomic_write_json,
    read_json,
    sha256_array,
    sha256_file,
)


SOURCE_CHECKPOINT_SCHEMA = "midogpp_stage90_utility_aligned_source_checkpoint_v1"


def generate_source_task(task: Mapping[str, object]) -> dict[str, object]:
    """Load one frozen expert, generate its streams, and score fixed support."""

    source = str(task["source_center"])
    training_seed = int(task["training_seed"])
    device = str(task["device"])
    keys = tuple(task["generation_keys"])
    if (
        not all(isinstance(key, SourceGenerationKey) for key in keys)
        or task.get("labels_available") is not False
        or task.get("amp_enabled") is not False
    ):
        raise ProtocolError("Stage-90 source worker input drifted.")
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)

    support_index = read_json(Path(str(task["support_index_path"])))
    support = np.load(
        Path(str(task["support_array_path"])), mmap_mode="r", allow_pickle=False
    )
    _validate_support_scratch(support, support_index, task)

    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=source,
        training_seed=training_seed,
        device=device,
    )
    try:
        source_arrays: list[np.ndarray] = []
        source_records: list[dict[str, object]] = []
        for key in keys:
            block = generate_source_block(
                expert, key, per_class=SOURCE_ROWS_PER_CLASS, device=device
            )
            values = np.ascontiguousarray(block.embeddings, dtype=np.float32)
            source_arrays.append(values)
            source_records.append(
                {
                    "generation_seed": key.generation_seed,
                    "stream_id": key.stream_id,
                    "expert_lock_hash": key.expert_lock_hash,
                    "output_sha256": block.output_sha256,
                }
            )
        stacked_sources = np.ascontiguousarray(np.stack(source_arrays), dtype=np.float32)

        offsets = support_index.get("offsets")
        if not isinstance(offsets, Mapping):
            raise ProtocolError("Stage-90 support offsets are malformed.")
        component_array = np.zeros((4, len(support)), dtype=np.float32)
        component_records: list[dict[str, object]] = []
        generated_means = {
            key.generation_seed: np.mean(values, axis=0, dtype=np.float64)
            for key, values in zip(keys, source_arrays, strict=True)
        }
        for query_center in tuple(offsets):
            if str(query_center) == source:
                continue
            raw = offsets[query_center]
            if not isinstance(raw, Mapping):
                raise ProtocolError("Stage-90 support offset row is malformed.")
            start, stop = int(raw["start"]), int(raw["stop"])
            cases = tuple(str(value) for value in raw["case_ids_by_row"])
            values = np.ascontiguousarray(support[start:stop], dtype=np.float32)
            energy = score_variational_compatibility(expert, values, cases)
            for label in (0, 1):
                component_array[label, start:stop] = np.asarray(
                    energy.per_class_reconstruction_mse[label], dtype=np.float32
                )
                component_array[2 + label, start:stop] = np.asarray(
                    energy.per_class_normalized_ps_kl[label], dtype=np.float32
                )
            support_mean = np.mean(values, axis=0, dtype=np.float64)
            mmd = {}
            for generation_seed, generated_mean in generated_means.items():
                difference = support_mean - generated_mean
                mmd[generation_seed] = float(np.dot(difference, difference))
            component_records.append(
                {
                    "query_center": str(query_center),
                    "support_start": start,
                    "support_stop": stop,
                    "support_row_count": stop - start,
                    "support_case_count": len(set(cases)),
                    "support_partition_hash": str(raw["support_partition_hash"]),
                    "case_equal_energy": float(energy.case_equal_mean),
                    "linear_kernel_mmd2_by_generation_seed": mmd,
                }
            )

        source_path = Path(str(task["source_array_path"]))
        component_path = Path(str(task["component_array_path"]))
        checkpoint_path = Path(str(task["checkpoint_path"]))
        atomic_save_npy(source_path, stacked_sources)
        atomic_save_npy(component_path, component_array)
        unhashed: dict[str, object] = {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA,
            "status": "COMPLETE",
            "config_contract_hash": str(task["config_contract_hash"]),
            "generation_lock_hash": str(task["generation_lock_hash"]),
            "support_scratch_hash": str(task["support_scratch_hash"]),
            "task_ordinal": int(task["task_ordinal"]),
            "source_center": source,
            "training_seed": training_seed,
            "device": device,
            "source_array_path": str(source_path),
            "source_array_file_sha256": sha256_file(source_path),
            "component_array_path": str(component_path),
            "component_array_file_sha256": sha256_file(component_path),
            "source_records": source_records,
            "component_records": component_records,
            "labels_consumed": False,
            "evaluation_embeddings_consumed": False,
            "source_experts_updated": False,
            "tf32_disabled": True,
            "amp_disabled": True,
            "float32_outputs": True,
        }
        payload = {
            **unhashed,
            "checkpoint_hash": stable_hash(_checkpoint_identity_payload(unhashed)),
        }
        atomic_write_json(checkpoint_path, payload)
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


def load_generation_checkpoint(
    path: Path, *, task: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(path)
    unhashed = _checkpoint_identity_payload(payload)
    expected_fields = {
        "schema_version", "status", "config_contract_hash", "generation_lock_hash",
        "support_scratch_hash", "task_ordinal", "source_center", "training_seed",
        "device", "source_array_path", "source_array_file_sha256",
        "component_array_path", "component_array_file_sha256", "source_records",
        "component_records", "labels_consumed", "evaluation_embeddings_consumed",
        "source_experts_updated", "tf32_disabled", "amp_disabled", "float32_outputs",
        "checkpoint_hash",
    }
    source_path = Path(str(payload.get("source_array_path", "")))
    component_path = Path(str(payload.get("component_array_path", "")))
    if (
        set(payload) != expected_fields
        or payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("generation_lock_hash") != task["generation_lock_hash"]
        or payload.get("support_scratch_hash") != task["support_scratch_hash"]
        or int(payload.get("task_ordinal", -1)) != int(task["task_ordinal"])
        or payload.get("source_center") != task["source_center"]
        or int(payload.get("training_seed", -1)) != int(task["training_seed"])
        or payload.get("device") != task["device"]
        or source_path != Path(str(task["source_array_path"]))
        or component_path != Path(str(task["component_array_path"]))
        or not source_path.is_file()
        or not component_path.is_file()
        or payload.get("source_array_file_sha256") != sha256_file(source_path)
        or payload.get("component_array_file_sha256") != sha256_file(component_path)
        or payload.get("labels_consumed") is not False
        or payload.get("evaluation_embeddings_consumed") is not False
        or payload.get("source_experts_updated") is not False
        or payload.get("tf32_disabled") is not True
        or payload.get("amp_disabled") is not True
        or payload.get("float32_outputs") is not True
    ):
        raise ProtocolError("Stage-90 source checkpoint failed validation.")
    source_values = np.load(source_path, mmap_mode="r", allow_pickle=False)
    support_index = read_json(Path(str(task["support_index_path"])))
    support_rows = int(support_index["shape"][0])
    component_values = np.load(component_path, mmap_mode="r", allow_pickle=False)
    if source_values.shape != (3, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or source_values.dtype != np.float32:
        raise ProtocolError("Stage-90 source checkpoint geometry drifted.")
    if component_values.shape != (4, support_rows) or component_values.dtype != np.float32 or not np.isfinite(component_values).all():
        raise ProtocolError("Stage-90 component checkpoint geometry drifted.")
    _validate_source_records(payload, task, source_values)
    _validate_component_records(payload, task, support_index)
    return payload


def _checkpoint_identity_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Restore typed generation-seed keys after their JSON object-key round trip."""

    output = {
        str(key): value
        for key, value in payload.items()
        if str(key) != "checkpoint_hash"
    }
    raw_records = output.get("component_records")
    if not isinstance(raw_records, list):
        raise ProtocolError("Stage-90 checkpoint component records are malformed.")
    records: list[dict[str, object]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ProtocolError("Stage-90 checkpoint component record is malformed.")
        record = dict(raw_record)
        raw_mmd = record.get("linear_kernel_mmd2_by_generation_seed")
        if not isinstance(raw_mmd, Mapping):
            raise ProtocolError("Stage-90 checkpoint MMD record is malformed.")
        typed_mmd: dict[int, object] = {}
        for raw_seed, value in raw_mmd.items():
            if type(raw_seed) is int:
                seed = raw_seed
            elif type(raw_seed) is str:
                try:
                    seed = int(raw_seed)
                except ValueError as exc:
                    raise ProtocolError(
                        "Stage-90 checkpoint MMD seed is malformed."
                    ) from exc
                if raw_seed != str(seed):
                    raise ProtocolError(
                        "Stage-90 checkpoint MMD seed is not canonical."
                    )
            else:
                raise ProtocolError("Stage-90 checkpoint MMD seed is malformed.")
            if seed in typed_mmd:
                raise ProtocolError("Stage-90 checkpoint MMD seeds are duplicated.")
            typed_mmd[seed] = value
        record["linear_kernel_mmd2_by_generation_seed"] = typed_mmd
        records.append(record)
    output["component_records"] = records
    return output


def _validate_support_scratch(
    support: np.ndarray, index: Mapping[str, object], task: Mapping[str, object]
) -> None:
    unhashed = {key: value for key, value in index.items() if key != "support_scratch_hash"}
    if (
        index.get("support_scratch_hash") != stable_hash(unhashed)
        or index.get("support_scratch_hash") != task["support_scratch_hash"]
        or index.get("shape") != list(support.shape)
        or index.get("dtype") != str(support.dtype)
        or index.get("array_sha256") != sha256_array(support)
        or index.get("labels_consumed") is not False
        or index.get("evaluation_embeddings_consumed") is not False
    ):
        raise ProtocolError("Stage-90 support scratch failed validation.")


def _validate_source_records(
    payload: Mapping[str, object], task: Mapping[str, object], values: np.ndarray
) -> None:
    records = payload.get("source_records")
    keys = tuple(task["generation_keys"])
    if not isinstance(records, list) or len(records) != len(keys):
        raise ProtocolError("Stage-90 source record coverage drifted.")
    labels = np.concatenate(
        (
            np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
            np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
        )
    )
    for ordinal, (record, key) in enumerate(zip(records, keys, strict=True)):
        if (
            not isinstance(record, Mapping)
            or set(record) != {
                "generation_seed", "stream_id", "expert_lock_hash", "output_sha256"
            }
            or int(record["generation_seed"]) != key.generation_seed
            or record["stream_id"] != key.stream_id
            or record["expert_lock_hash"] != key.expert_lock_hash
            or record["output_sha256"] != _array_bundle_sha256(values[ordinal], labels)
        ):
            raise ProtocolError("Stage-90 source record binding drifted.")


def _validate_component_records(
    payload: Mapping[str, object], task: Mapping[str, object], support_index: Mapping[str, object]
) -> None:
    records = payload.get("component_records")
    offsets = support_index.get("offsets")
    if (
        not isinstance(records, list)
        or not isinstance(offsets, Mapping)
        or tuple(offsets) != CENTERS
    ):
        raise ProtocolError("Stage-90 component records are malformed.")
    if len(records) != len(offsets) - 1:
        raise ProtocolError("Stage-90 component query coverage drifted.")
    observed = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Stage-90 component record is malformed.")
        query = str(record.get("query_center"))
        raw_offset = offsets.get(query)
        mmd = record.get("linear_kernel_mmd2_by_generation_seed")
        if not isinstance(raw_offset, Mapping) or not isinstance(mmd, Mapping):
            raise ProtocolError("Stage-90 component record binding drifted.")
        numeric = np.asarray(
            [float(record.get("case_equal_energy", np.nan)), *[float(value) for value in mmd.values()]],
            dtype=np.float64,
        )
        if (
            query in observed
            or int(record.get("support_start", -1)) != int(raw_offset["start"])
            or int(record.get("support_stop", -1)) != int(raw_offset["stop"])
            or int(record.get("support_row_count", -1)) != int(raw_offset["stop"]) - int(raw_offset["start"])
            or int(record.get("support_case_count", -1)) != 2
            or record.get("support_partition_hash") != raw_offset["support_partition_hash"]
            or {int(key) for key in mmd} != {key.generation_seed for key in task["generation_keys"]}
            or not np.isfinite(numeric).all()
            or np.any(numeric < 0.0)
        ):
            raise ProtocolError("Stage-90 component record drifted.")
        observed.add(query)
    if observed != set(offsets).difference({str(task["source_center"])}):
        raise ProtocolError("Stage-90 component center coverage drifted.")


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "SOURCE_CHECKPOINT_SCHEMA",
    "generate_source_task",
    "load_generation_checkpoint",
)
