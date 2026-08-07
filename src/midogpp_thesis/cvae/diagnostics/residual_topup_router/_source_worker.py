"""Spawn-safe GPU worker for the residual top-up source cache."""

from __future__ import annotations

import gc
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.serialization import (
    load_routing_authorized_expert,
)
from ...generation.contracts import SourceGenerationKey
from ...generation.generation import generate_source_block
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import score_variational_compatibility
from .contracts import (
    CENTERS,
    COMMON_FEATURE_DIM,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)


def generate_source_task(task: Mapping[str, object]) -> dict[str, object]:
    """Load one expert once, score all support sets, and generate three blocks."""

    source = str(task["source_center"])
    training_seed = int(task["training_seed"])
    device = str(task["device"])
    keys = tuple(task["generation_keys"])
    if not all(isinstance(key, SourceGenerationKey) for key in keys):
        raise ProtocolError("Residual top-up worker received invalid generation keys.")
    if device.startswith("cuda"):
        # This is intentionally inside the spawned child.  Ampere TF32 defaults
        # must not make the persisted source stream workstation-dependent.
        import torch

        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    support_index = _json(Path(str(task["support_index_path"])))
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
        raise ProtocolError("Residual top-up support scratch failed validation.")

    checkpoint_path = Path(str(task["checkpoint_path"]))
    array_path = Path(str(task["array_path"]))
    expert = load_routing_authorized_expert(
        Path(str(task["expert_bank_root"])),
        source_center=source,
        training_seed=training_seed,
        device=device,
    )
    try:
        compatibility_rows: list[dict[str, object]] = []
        offsets = support_index.get("offsets")
        if not isinstance(offsets, Mapping):
            raise ProtocolError("Residual top-up support offsets are malformed.")
        for query_center in CENTERS:
            raw_offset = offsets.get(query_center)
            if not isinstance(raw_offset, Mapping):
                raise ProtocolError("Residual top-up support scratch lacks a center.")
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

        block_rows: list[dict[str, object]] = []
        arrays: list[np.ndarray] = []
        for key in keys:
            block = generate_source_block(
                expert,
                key,
                per_class=MAX_SOURCE_PREFIX_PER_CLASS,
                device=device,
            )
            arrays.append(np.asarray(block.embeddings, dtype=np.float32))
            block_rows.append(
                {
                    "generation_seed": key.generation_seed,
                    "stream_id": key.stream_id,
                    "output_sha256": block.output_sha256,
                }
            )
        task_array = np.ascontiguousarray(np.stack(arrays), dtype=np.float32)
        _atomic_save_npy(array_path, task_array)
        unhashed: dict[str, object] = {
            "schema_version": "midogpp_residual_topup_source_checkpoint_v1",
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
            "blocks": block_rows,
            "compatibility_case_records": compatibility_rows,
            "target_labels_used": False,
            "support_labels_used": False,
            "evaluation_embeddings_used": False,
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


def load_generation_checkpoint(
    path: Path,
    *,
    task: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate a task checkpoint and each generated block byte-for-byte."""

    payload = _json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
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
            and isinstance(row.get("output_sha256"), str)
            and len(str(row["output_sha256"])) == 64
            for row, key in zip(block_rows, expected_keys, strict=True)
        )
    )
    if (
        payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version")
        != "midogpp_residual_topup_source_checkpoint_v1"
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
        or payload.get("target_labels_used") is not False
        or payload.get("support_labels_used") is not False
        or payload.get("evaluation_embeddings_used") is not False
        or not block_binding_valid
    ):
        raise ProtocolError("Residual top-up generation checkpoint failed validation.")
    array = np.load(array_path, mmap_mode="r")
    expected_shape = (
        len(GENERATION_SEEDS),
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_FEATURE_DIM,
    )
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ProtocolError("Residual top-up generation checkpoint array drifted.")
    labels = _source_labels()
    assert isinstance(block_rows, list)
    for index, row in enumerate(block_rows):
        if _array_bundle_sha256(array[index], labels) != row["output_sha256"]:
            raise ProtocolError("Residual top-up generation block hash drifted.")
    _validate_checkpoint_case_rows(payload, task=task)
    return payload


def _validate_checkpoint_case_rows(
    payload: Mapping[str, object],
    *,
    task: Mapping[str, object],
) -> None:
    support_index = _json(Path(str(task["support_index_path"])))
    offsets = support_index.get("offsets")
    rows = payload.get("compatibility_case_records")
    if not isinstance(offsets, Mapping) or not isinstance(rows, list):
        raise ProtocolError("Residual top-up compatibility checkpoint is malformed.")
    expected = {
        (query, str(case_id))
        for query in CENTERS
        for case_id in _mapping(offsets.get(query), role="support offset")["case_ids"]
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
            raise ProtocolError("Residual top-up compatibility row is malformed.")
        identity = (str(row.get("query_center")), str(row.get("case_id")))
        values = np.asarray([float(row.get(field, np.nan)) for field in numeric_fields])
        if (
            row.get("source_center") != task["source_center"]
            or int(row.get("training_seed", -1)) != int(task["training_seed"])
            or identity in observed
            or not np.isfinite(values).all()
            or int(row.get("row_count", 0)) <= 0
        ):
            raise ProtocolError("Residual top-up compatibility checkpoint row drifted.")
        observed.add(identity)
    if observed != expected:
        raise ProtocolError("Residual top-up compatibility case coverage drifted.")


def validate_source_cache_inventory(cache: object) -> None:
    """Validate final memmap geometry plus complete index and energy grids."""

    try:
        array = np.load(Path(str(getattr(cache, "array_path"))), mmap_mode="r")
        index_rows = tuple(getattr(cache, "index_rows"))
        case_rows = tuple(getattr(cache, "compatibility_case_rows"))
    except (OSError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual top-up source array is unreadable.") from exc
    expected_block_count = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
    expected_shape = (
        expected_block_count,
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        COMMON_FEATURE_DIM,
    )
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ProtocolError("Residual top-up source-cache geometry drifted.")
    expected_keys = tuple(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    observed_keys: list[tuple[str, int, int]] = []
    if len(index_rows) != expected_block_count:
        raise ProtocolError("Residual top-up source-cache index coverage drifted.")
    for ordinal, row in enumerate(index_rows):
        canonical = canonical_source_index_row(row)
        observed_keys.append(
            (
                str(canonical["source_center"]),
                int(canonical["training_seed"]),
                int(canonical["generation_seed"]),
            )
        )
        if (
            int(canonical["block_ordinal"]) != ordinal
            or int(canonical["samples_per_class"]) != MAX_SOURCE_PREFIX_PER_CLASS
            or int(canonical["row_count"]) != 2 * MAX_SOURCE_PREFIX_PER_CLASS
            or int(canonical["feature_dim"]) != COMMON_FEATURE_DIM
            or not str(canonical["stream_id"])
            or not str(canonical["expert_lock_hash"])
            or len(str(canonical["output_sha256"])) != 64
        ):
            raise ProtocolError("Residual top-up source-cache index row drifted.")
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError("Residual top-up source-cache key order drifted.")

    expected_replicas = set(product(CENTERS, TRAINING_SEEDS, CENTERS))
    case_ids_by_replica: dict[tuple[str, int, str], set[str]] = {}
    query_case_ids: dict[str, set[str]] = {}
    for row in case_rows:
        canonical = canonical_compatibility_case_row(row)
        key = (
            str(canonical["source_center"]),
            int(canonical["training_seed"]),
            str(canonical["query_center"]),
        )
        case_id = str(canonical["case_id"])
        cases = case_ids_by_replica.setdefault(key, set())
        if case_id in cases:
            raise ProtocolError("Residual top-up compatibility case is duplicated.")
        cases.add(case_id)
        query_case_ids.setdefault(key[2], set()).add(case_id)
        numeric = np.asarray(
            [
                canonical["marginal_variational_energy"],
                canonical["class_0_energy"],
                canonical["class_1_energy"],
                canonical["class_0_common_reconstruction_mse"],
                canonical["class_1_common_reconstruction_mse"],
                canonical["class_0_normalized_ps_kl"],
                canonical["class_1_normalized_ps_kl"],
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or int(canonical["row_count"]) <= 0
            or canonical["query_partition_role"] != "support"
            or canonical["class_prior_json"] != "[0.5,0.5]"
            or canonical["labels_used"] is not False
            or canonical["exact_nelbo_claimed"] is not False
        ):
            raise ProtocolError("Residual top-up compatibility row drifted.")
    if set(case_ids_by_replica) != expected_replicas:
        raise ProtocolError("Residual top-up compatibility replica grid is incomplete.")
    for source, seed, query in expected_replicas:
        cases = case_ids_by_replica[(source, seed, query)]
        if len(cases) != 2 or cases != query_case_ids[query]:
            raise ProtocolError("Residual top-up compatibility support cases drifted.")


def canonical_source_index_row(row: Mapping[str, object]) -> dict[str, object]:
    try:
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
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual top-up source index row is malformed.") from exc


def canonical_compatibility_case_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    try:
        return {
            "schema_version": str(row["schema_version"]),
            "source_center": str(row["source_center"]),
            "training_seed": int(row["training_seed"]),
            "query_center": str(row["query_center"]),
            "case_id": str(row["case_id"]),
            "query_partition_role": str(row["query_partition_role"]),
            "row_count": int(row["row_count"]),
            "marginal_variational_energy": float(row["marginal_variational_energy"]),
            "class_0_energy": float(row["class_0_energy"]),
            "class_1_energy": float(row["class_1_energy"]),
            "class_0_common_reconstruction_mse": float(
                row["class_0_common_reconstruction_mse"]
            ),
            "class_1_common_reconstruction_mse": float(
                row["class_1_common_reconstruction_mse"]
            ),
            "class_0_normalized_ps_kl": float(row["class_0_normalized_ps_kl"]),
            "class_1_normalized_ps_kl": float(row["class_1_normalized_ps_kl"]),
            "class_prior_json": str(row["class_prior_json"]),
            "labels_used": _truthy(row["labels_used"]),
            "exact_nelbo_claimed": _truthy(row["exact_nelbo_claimed"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual top-up compatibility row is malformed.") from exc


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


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _mapping(value: object, *, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Residual top-up {role} must be a mapping.")
    return value


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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read residual top-up worker JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Residual top-up worker JSON must be an object.")
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


__all__ = ("generate_source_task",)
