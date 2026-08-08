"""Durable assembly and CSV persistence for case-OOF source products."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .artifact_io import atomic_write_csv_rows, read_csv_rows
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .source_cache_contracts import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_CASE_MEMBER,
    EXPECTED_SOURCE_BLOCK_COUNT,
    SOURCE_BLOCK_ARRAY_MEMBER,
    SOURCE_BLOCK_INDEX_COLUMNS,
    SOURCE_BLOCK_INDEX_MEMBER,
    SourceCache,
)
from .source_cache_worker import MAX_SOURCE_PREFIX_PER_CLASS


def materialize_source_products(
    root: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    key_map: Mapping[tuple[str, int, int], object],
) -> SourceCache:
    index_rows = _materialize_source_array(
        root / SOURCE_BLOCK_ARRAY_MEMBER,
        completed=completed,
        key_map=key_map,
    )
    compatibility_rows = _build_compatibility_rows(completed)
    atomic_write_csv_rows(
        root / SOURCE_BLOCK_INDEX_MEMBER,
        index_rows,
        columns=SOURCE_BLOCK_INDEX_COLUMNS,
    )
    atomic_write_csv_rows(
        root / COMPATIBILITY_CASE_MEMBER,
        compatibility_rows,
        columns=COMPATIBILITY_CASE_COLUMNS,
    )
    return SourceCache(
        array_path=root / SOURCE_BLOCK_ARRAY_MEMBER,
        index_rows=tuple(index_rows),
        compatibility_case_rows=tuple(compatibility_rows),
    )


def load_source_cache(root: Path) -> SourceCache:
    return SourceCache(
        array_path=root / SOURCE_BLOCK_ARRAY_MEMBER,
        index_rows=read_csv_rows(root / SOURCE_BLOCK_INDEX_MEMBER),
        compatibility_case_rows=read_csv_rows(root / COMPATIBILITY_CASE_MEMBER),
    )


def _materialize_source_array(
    array_path: Path,
    *,
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    key_map: Mapping[tuple[str, int, int], object],
) -> list[dict[str, object]]:
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + f".{os.getpid()}.tmp")
    target = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(
            EXPECTED_SOURCE_BLOCK_COUNT,
            2 * MAX_SOURCE_PREFIX_PER_CLASS,
            COMMON_OUTPUT_DIM,
        ),
    )
    rows: list[dict[str, object]] = []
    ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            payload = completed[(source, training_seed)]
            source_array = np.load(Path(str(payload["array_path"])), mmap_mode="r")
            records = payload.get("blocks")
            if not isinstance(records, list):
                raise ProtocolError("Case-OOF source block inventory is absent.")
            for seed_index, generation_seed in enumerate(GENERATION_SEEDS):
                record = records[seed_index]
                key = key_map[(source, training_seed, generation_seed)]
                target[ordinal] = source_array[seed_index]
                rows.append(
                    {
                        "schema_version": "midogpp_residual_topup_case_oof_source_block_v1",
                        "block_ordinal": ordinal,
                        "source_center": source,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "stream_id": str(getattr(key, "stream_id")),
                        "expert_lock_hash": str(getattr(key, "expert_lock_hash")),
                        "samples_per_class": MAX_SOURCE_PREFIX_PER_CLASS,
                        "row_count": 2 * MAX_SOURCE_PREFIX_PER_CLASS,
                        "feature_dim": COMMON_OUTPUT_DIM,
                        "output_sha256": str(record["output_sha256"]),
                    }
                )
                ordinal += 1
    target.flush()
    del target
    os.replace(temporary, array_path)
    return rows


def _build_compatibility_rows(
    completed: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            raw_rows = completed[(source, training_seed)].get(
                "compatibility_case_records"
            )
            if not isinstance(raw_rows, list):
                raise ProtocolError("Case-OOF compatibility records are absent.")
            for raw in raw_rows:
                output.append(
                    {
                        "schema_version": "midogpp_residual_topup_case_oof_compatibility_case_v1",
                        "source_center": source,
                        "training_seed": training_seed,
                        "query_center": str(raw["query_center"]),
                        "case_id": str(raw["case_id"]),
                        "query_partition_role": "support",
                        "row_count": int(raw["row_count"]),
                        "marginal_variational_energy": float(
                            raw["marginal_variational_energy"]
                        ),
                        "class_0_energy": float(raw["class_0_energy"]),
                        "class_1_energy": float(raw["class_1_energy"]),
                        "class_0_common_reconstruction_mse": float(
                            raw["class_0_common_reconstruction_mse"]
                        ),
                        "class_1_common_reconstruction_mse": float(
                            raw["class_1_common_reconstruction_mse"]
                        ),
                        "class_0_normalized_ps_kl": float(
                            raw["class_0_normalized_ps_kl"]
                        ),
                        "class_1_normalized_ps_kl": float(
                            raw["class_1_normalized_ps_kl"]
                        ),
                        "class_prior_json": "[0.5,0.5]",
                        "labels_used": False,
                        "evaluation_embeddings_used": False,
                        "source_experts_updated": False,
                        "exact_nelbo_claimed": False,
                    }
                )
    return output


__all__ = ("load_source_cache", "materialize_source_products")
