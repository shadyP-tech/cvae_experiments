"""Publication and validation of the sealed v5 source-stream store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    read_json,
    sha256_array,
    sha256_file,
)

from .source_checkpoints import persist_exact_json, publish_source_array
from .source_contracts import (
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_RECEIPT_MEMBER,
    SourceGeometry,
    SourceRuntimeTestMode,
    SourceStreamRecord,
    SourceStreamStore,
)
from .source_hashing import block_bundle_sha256, canonical_sha256
from .source_planning import final_paths, geometry_for
from .worker_runtime import GPU_DEVICES


def publish_source_store(
    destination: Path,
    *,
    attempt_id: str,
    config_hash: str,
    generation_lock_hash: str,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    geometry: SourceGeometry,
    test_mode: SourceRuntimeTestMode | None,
) -> SourceStreamStore:
    records = publish_source_array(
        destination / SOURCE_ARRAY_MEMBER,
        tasks=tasks,
        completed=completed,
        geometry=geometry,
    )
    index_unhashed = {
        "schema_version": "midogpp_sceptre_v5_physical_source_stream_index_v1",
        "attempt_id": attempt_id,
        "config_hash": config_hash,
        "generation_lock_hash": generation_lock_hash,
        "geometry": geometry.to_payload(),
        "records": [record.to_payload() for record in records],
        "record_count": len(records),
        "source_streams_only": True,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "seed_selection_performed": False,
    }
    index = {**index_unhashed, "index_sha256": canonical_sha256(index_unhashed)}
    persist_exact_json(destination / SOURCE_INDEX_MEMBER, index)
    array_path = destination / SOURCE_ARRAY_MEMBER
    index_path = destination / SOURCE_INDEX_MEMBER
    receipt_unhashed = {
        "schema_version": "midogpp_sceptre_v5_physical_source_stream_receipt_v1",
        "status": "COMPLETE_LABEL_FREE_FULL_SOURCE_STREAMS",
        "attempt_id": attempt_id,
        "config_hash": config_hash,
        "generation_lock_hash": generation_lock_hash,
        "geometry": geometry.to_payload(),
        "source_array_sha256": sha256_file(array_path),
        "source_index_file_sha256": sha256_file(index_path),
        "source_index_sha256": index["index_sha256"],
        "record_count": len(records),
        "dtype": "float32",
        "npy_memmap_mode": "read_only",
        "two_persistent_gpu_workers": test_mode is None,
        "gpu_devices": list(GPU_DEVICES) if test_mode is None else [],
        "parent_cuda_context_created": False,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "expert_bank_updated": False,
        "seed_selection_performed": False,
        "synthetic_test_mode": test_mode is not None,
    }
    receipt = {
        **receipt_unhashed,
        "receipt_sha256": canonical_sha256(receipt_unhashed),
    }
    persist_exact_json(destination / SOURCE_RECEIPT_MEMBER, receipt)
    array_path.chmod(0o444)
    return load_source_streams(
        destination,
        expected_config_hash=config_hash,
        expected_generation_lock_hash=generation_lock_hash,
        expected_attempt_id=attempt_id,
        test_mode=test_mode,
    )


def load_source_streams(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_generation_lock_hash: str | None = None,
    expected_attempt_id: str | None = None,
    test_mode: SourceRuntimeTestMode | None = None,
) -> SourceStreamStore:
    """Load and fully validate a completed source-stream store."""

    geometry = geometry_for(test_mode)
    destination = Path(root)
    array_path, index_path, receipt_path = final_paths(destination)
    if any(
        path.is_symlink() or not path.is_file()
        for path in (array_path, index_path, receipt_path)
    ):
        raise ProtocolError("SCEPTRE v5 source final store is absent or unsafe.")
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    raw_records = index.get("records")
    if not isinstance(raw_records, list):
        raise ProtocolError("SCEPTRE v5 source index records are absent.")
    try:
        records = tuple(
            SourceStreamRecord(
                block_ordinal=int(raw["block_ordinal"]),
                source_center=str(raw["source_center"]),
                training_seed=int(raw["training_seed"]),
                generation_seed=int(raw["generation_seed"]),
                stream_id=str(raw["stream_id"]),
                expert_lock_hash=str(raw["expert_lock_hash"]),
                rows_per_class=int(raw["rows_per_class"]),
                feature_dim=int(raw["feature_dim"]),
                output_sha256=str(raw["output_sha256"]),
                array_sha256=str(raw["array_sha256"]),
            )
            for raw in raw_records
            if isinstance(raw, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v5 source index record is malformed.") from exc
    index_unhashed = {
        key: value for key, value in index.items() if key != "index_sha256"
    }
    receipt_unhashed = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        len(records) != len(raw_records)
        or index.get("schema_version")
        != "midogpp_sceptre_v5_physical_source_stream_index_v1"
        or receipt.get("schema_version")
        != "midogpp_sceptre_v5_physical_source_stream_receipt_v1"
        or receipt.get("status") != "COMPLETE_LABEL_FREE_FULL_SOURCE_STREAMS"
        or not str(index.get("attempt_id", ""))
        or receipt.get("attempt_id") != index.get("attempt_id")
        or index.get("geometry") != geometry.to_payload()
        or receipt.get("geometry") != geometry.to_payload()
        or index.get("index_sha256") != canonical_sha256(index_unhashed)
        or receipt.get("receipt_sha256") != canonical_sha256(receipt_unhashed)
        or receipt.get("source_array_sha256") != sha256_file(array_path)
        or receipt.get("source_index_file_sha256") != sha256_file(index_path)
        or receipt.get("source_index_sha256") != index.get("index_sha256")
        or receipt.get("record_count") != geometry.stream_count
        or index.get("record_count") != geometry.stream_count
        or receipt.get("dtype") != "float32"
        or receipt.get("npy_memmap_mode") != "read_only"
        or index.get("source_streams_only") is not True
        or index.get("target_cache_opened") is not False
        or index.get("manifest_opened") is not False
        or index.get("outcomes_available") is not False
        or index.get("seed_selection_performed") is not False
        or receipt.get("two_persistent_gpu_workers") is not (test_mode is None)
        or receipt.get("gpu_devices")
        != (list(GPU_DEVICES) if test_mode is None else [])
        or receipt.get("parent_cuda_context_created") is not False
        or receipt.get("target_cache_opened") is not False
        or receipt.get("manifest_opened") is not False
        or receipt.get("outcomes_available") is not False
        or receipt.get("expert_bank_updated") is not False
        or receipt.get("seed_selection_performed") is not False
        or receipt.get("synthetic_test_mode") is not (test_mode is not None)
        or (
            expected_config_hash is not None
            and receipt.get("config_hash") != expected_config_hash
        )
        or (
            expected_generation_lock_hash is not None
            and receipt.get("generation_lock_hash")
            != expected_generation_lock_hash
        )
        or (
            expected_attempt_id is not None
            and receipt.get("attempt_id") != expected_attempt_id
        )
    ):
        raise ProtocolError("SCEPTRE v5 source receipt failed validation.")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != geometry.array_shape
        or values.dtype != np.float32
        or values.flags.writeable
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE v5 source array geometry or values drifted.")
    store = SourceStreamStore(
        root=destination,
        array_path=array_path,
        index_path=index_path,
        receipt_path=receipt_path,
        geometry=geometry,
        records=records,
        receipt=receipt,
    )
    for record in records:
        block = values[record.block_ordinal]
        if (
            sha256_array(block) != record.array_sha256
            or block_bundle_sha256(block, geometry.rows_per_class)
            != record.output_sha256
        ):
            raise ProtocolError("SCEPTRE v5 source block bytes drifted.")
    return store


__all__ = ("load_source_streams", "publish_source_store")
