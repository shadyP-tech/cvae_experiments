"""Spawn-safe expert worker for the case-OOF source cache."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, SourceGenerationKey
from ...generation.generation import generate_source_block
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import score_variational_compatibility
from .artifact_io import atomic_save_npy, atomic_write_json, read_json, sha256_file
from .contracts import CENTERS, GENERATION_SEEDS


MAX_SOURCE_PREFIX_PER_CLASS = 256


def generate_source_task(task: Mapping[str, object]) -> dict[str, object]:
    """Load one expert once, score fixed support, and generate three streams."""

    source = str(task["source_center"])
    training_seed = int(task["training_seed"])
    device = str(task["device"])
    keys = tuple(task["generation_keys"])
    if not all(isinstance(key, SourceGenerationKey) for key in keys):
        raise ProtocolError("Case-OOF worker received invalid generation keys.")
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    support_index = read_json(Path(str(task["support_index_path"])))
    support = np.load(Path(str(task["support_array_path"])), mmap_mode="r")
    unhashed_scratch = {
        key: value
        for key, value in support_index.items()
        if key != "support_scratch_hash"
    }
    if (
        support_index.get("support_scratch_hash") != stable_hash(unhashed_scratch)
        or support_index.get("support_scratch_hash")
        != task["support_scratch_hash"]
        or support_index.get("shape") != list(support.shape)
        or support_index.get("dtype") != str(support.dtype)
        or support_index.get("array_sha256") != _sha256_array(support)
    ):
        raise ProtocolError("Case-OOF support scratch failed validation.")

    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=source,
        training_seed=training_seed,
        device=device,
    )
    checkpoint_path = Path(str(task["checkpoint_path"]))
    array_path = Path(str(task["array_path"]))
    try:
        compatibility_rows: list[dict[str, object]] = []
        offsets = support_index.get("offsets")
        if not isinstance(offsets, Mapping):
            raise ProtocolError("Case-OOF support offsets are malformed.")
        for query_center in CENTERS:
            raw_offset = offsets.get(query_center)
            if not isinstance(raw_offset, Mapping):
                raise ProtocolError("Case-OOF support scratch lacks a center.")
            start, stop = int(raw_offset["start"]), int(raw_offset["stop"])
            case_ids = tuple(str(value) for value in raw_offset["case_ids"])
            query = np.ascontiguousarray(support[start:stop], dtype=np.float32)
            energy = score_variational_compatibility(expert, query, case_ids)
            cases = np.asarray(case_ids, dtype=object)
            for case_id in energy.case_order:
                mask = cases == case_id
                compatibility_rows.append(
                    {
                        "source_center": source,
                        "training_seed": training_seed,
                        "query_center": query_center,
                        "case_id": case_id,
                        "row_count": int(np.sum(mask)),
                        "marginal_variational_energy": float(
                            energy.per_case[case_id]
                        ),
                        "class_0_energy": float(
                            np.mean(energy.per_class_energy[0][mask])
                        ),
                        "class_1_energy": float(
                            np.mean(energy.per_class_energy[1][mask])
                        ),
                        "class_0_common_reconstruction_mse": float(
                            np.mean(energy.per_class_reconstruction_mse[0][mask])
                        ),
                        "class_1_common_reconstruction_mse": float(
                            np.mean(energy.per_class_reconstruction_mse[1][mask])
                        ),
                        "class_0_normalized_ps_kl": float(
                            np.mean(energy.per_class_normalized_ps_kl[0][mask])
                        ),
                        "class_1_normalized_ps_kl": float(
                            np.mean(energy.per_class_normalized_ps_kl[1][mask])
                        ),
                    }
                )

        blocks: list[dict[str, object]] = []
        arrays: list[np.ndarray] = []
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
        atomic_save_npy(array_path, task_array)
        unhashed: dict[str, object] = {
            "schema_version": "midogpp_residual_topup_case_oof_source_checkpoint_v1",
            "status": "COMPLETE",
            "config_contract_hash": str(task["config_contract_hash"]),
            "generation_lock_hash": str(task["generation_lock_hash"]),
            "support_scratch_hash": str(task["support_scratch_hash"]),
            "task_ordinal": int(task["task_ordinal"]),
            "source_center": source,
            "training_seed": training_seed,
            "device": device,
            "array_path": str(array_path),
            "array_file_sha256": sha256_file(array_path),
            "blocks": blocks,
            "compatibility_case_records": compatibility_rows,
            "target_labels_used": False,
            "support_labels_used": False,
            "evaluation_embeddings_used": False,
            "tf32_disabled": True,
        }
        payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
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
    path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    payload = read_json(path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    array_path = Path(str(payload.get("array_path", "")))
    block_rows = payload.get("blocks")
    expected_keys = tuple(task.get("generation_keys", ()))
    block_binding_valid = (
        isinstance(block_rows, list)
        and len(block_rows) == len(expected_keys)
        and all(
            isinstance(row, Mapping)
            and int(row.get("generation_seed", -1)) == key.generation_seed
            and row.get("stream_id") == key.stream_id
            and _is_sha256(row.get("output_sha256"))
            for row, key in zip(block_rows, expected_keys, strict=True)
        )
    )
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version")
        != "midogpp_residual_topup_case_oof_source_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("config_contract_hash") != task["config_contract_hash"]
        or payload.get("generation_lock_hash") != task["generation_lock_hash"]
        or payload.get("support_scratch_hash") != task["support_scratch_hash"]
        or int(payload.get("task_ordinal", -1)) != int(task["task_ordinal"])
        or payload.get("source_center") != task["source_center"]
        or int(payload.get("training_seed", -1))
        != int(task["training_seed"])
        or payload.get("device") != task["device"]
        or array_path != Path(str(task["array_path"]))
        or not array_path.is_file()
        or payload.get("array_file_sha256") != sha256_file(array_path)
        or payload.get("target_labels_used") is not False
        or payload.get("support_labels_used") is not False
        or payload.get("evaluation_embeddings_used") is not False
        or payload.get("tf32_disabled") is not True
        or not block_binding_valid
    ):
        raise ProtocolError("Case-OOF generation checkpoint failed validation.")
    array = np.load(array_path, mmap_mode="r")
    expected_shape = (
        len(GENERATION_SEEDS),
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_OUTPUT_DIM,
    )
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ProtocolError("Case-OOF generation checkpoint array drifted.")
    labels = _source_labels()
    assert isinstance(block_rows, list)
    for index, row in enumerate(block_rows):
        if _array_bundle_sha256(array[index], labels) != row["output_sha256"]:
            raise ProtocolError("Case-OOF generation block hash drifted.")
    _validate_checkpoint_case_rows(payload, task=task)
    return payload


def _validate_checkpoint_case_rows(
    payload: Mapping[str, object], *, task: Mapping[str, object]
) -> None:
    support_index = read_json(Path(str(task["support_index_path"])))
    offsets = support_index.get("offsets")
    rows = payload.get("compatibility_case_records")
    if not isinstance(offsets, Mapping) or not isinstance(rows, list):
        raise ProtocolError("Case-OOF compatibility checkpoint is malformed.")
    expected = {
        (query, str(case_id))
        for query in CENTERS
        for case_id in _mapping(offsets.get(query), "support offset")["case_ids"]
    }
    observed: set[tuple[str, str]] = set()
    numeric_fields = (
        "marginal_variational_energy",
        "class_0_energy",
        "class_1_energy",
        "class_0_common_reconstruction_mse",
        "class_1_common_reconstruction_mse",
        "class_0_normalized_ps_kl",
        "class_1_normalized_ps_kl",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Case-OOF compatibility row is malformed.")
        identity = (str(row.get("query_center")), str(row.get("case_id")))
        values = np.asarray(
            [float(row.get(field, np.nan)) for field in numeric_fields]
        )
        if (
            row.get("source_center") != task["source_center"]
            or int(row.get("training_seed", -1))
            != int(task["training_seed"])
            or identity in observed
            or not np.isfinite(values).all()
            or int(row.get("row_count", 0)) <= 0
        ):
            raise ProtocolError("Case-OOF compatibility checkpoint row drifted.")
        observed.add(identity)
    if observed != expected:
        raise ProtocolError("Case-OOF compatibility case coverage drifted.")


def _source_labels() -> np.ndarray:
    return np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Case-OOF {role} must be a mapping.")
    return value


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = (
    "MAX_SOURCE_PREFIX_PER_CLASS",
    "generate_source_task",
    "load_generation_checkpoint",
)
