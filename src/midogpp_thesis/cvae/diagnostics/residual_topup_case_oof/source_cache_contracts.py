"""Domain model and schemas for the independent case-OOF source cache."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    GLOBAL_QUERY_ROLE,
    ProxyScoreRow,
    SUPPORT_QUERY_ROLE,
    TRAINING_SEEDS,
    candidate_sources,
    global_candidate_sources,
)


SOURCE_BLOCK_ARRAY_MEMBER = "arrays/source_prefix_blocks.npy"
SOURCE_BLOCK_INDEX_MEMBER = "tables/source_block_index.csv"
COMPATIBILITY_CASE_MEMBER = "tables/compatibility_case_energy.csv"
SOURCE_CACHE_LOCK_MEMBER = "manifests/source_cache_lock.json"
GENERATION_DEVICES = ("cuda:0", "cuda:1")
EXPECTED_SOURCE_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
EXPECTED_SOURCE_BLOCK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)

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
    "evaluation_embeddings_used",
    "source_experts_updated",
    "exact_nelbo_claimed",
)


@dataclass(frozen=True)
class SourceCache:
    array_path: Path
    index_rows: tuple[Mapping[str, object], ...]
    compatibility_case_rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        from .source_cache_validation import validate_source_cache_inventory

        validate_source_cache_inventory(self)

    @cached_property
    def block_ordinal_by_key(self) -> Mapping[tuple[str, int, int], int]:
        return {
            (
                str(row["source_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            ): int(row["block_ordinal"])
            for row in self.index_rows
        }

    @cached_property
    def source_cache_hash(self) -> str:
        return stable_hash(
            {
                "index_rows": [_canonical_index(row) for row in self.index_rows],
                "compatibility_case_rows": [
                    _canonical_case(row) for row in self.compatibility_case_rows
                ],
            }
        )

    def proxy_score_rows(self, crossfit: object) -> tuple[ProxyScoreRow, ...]:
        """Expand raw support energies into legal H/q and H-only score rows."""

        support_by_center = getattr(crossfit, "fixed_support_rows_by_center", None)
        if not isinstance(support_by_center, Mapping):
            raise ProtocolError("Case-OOF fixed-support surface is unavailable.")
        raw = {
            (
                str(row["source_center"]),
                int(row["training_seed"]),
                str(row["query_center"]),
                str(row["case_id"]),
            ): row
            for row in self.compatibility_case_rows
        }
        output: list[ProxyScoreRow] = []
        for target in CENTERS:
            support_cases = tuple(
                sorted({str(row.case_id) for row in support_by_center[target]})
            )
            for source in candidate_sources(target):
                for training_seed in TRAINING_SEEDS:
                    for case_id in support_cases:
                        row = _required_raw(raw, source, training_seed, target, case_id)
                        output.append(
                            ProxyScoreRow(
                                outer_target=target,
                                query_role=SUPPORT_QUERY_ROLE,
                                query_center=target,
                                case_id=case_id,
                                candidate_source=source,
                                training_seed=training_seed,
                                row_count=int(row["row_count"]),
                                proxy_energy=float(row["marginal_variational_energy"]),
                            )
                        )
            for query in CENTERS:
                if query == target:
                    continue
                query_cases = tuple(
                    sorted({str(row.case_id) for row in support_by_center[query]})
                )
                for source in global_candidate_sources(target, query):
                    for training_seed in TRAINING_SEEDS:
                        for case_id in query_cases:
                            row = _required_raw(
                                raw, source, training_seed, query, case_id
                            )
                            output.append(
                                ProxyScoreRow(
                                    outer_target=target,
                                    query_role=GLOBAL_QUERY_ROLE,
                                    query_center=query,
                                    case_id=case_id,
                                    candidate_source=source,
                                    training_seed=training_seed,
                                    row_count=int(row["row_count"]),
                                    proxy_energy=float(
                                        row["marginal_variational_energy"]
                                    ),
                                )
                            )
        return tuple(output)


def _canonical_index(row: Mapping[str, object]) -> dict[str, object]:
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


def _canonical_case(row: Mapping[str, object]) -> dict[str, object]:
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
        "evaluation_embeddings_used": _truthy(row["evaluation_embeddings_used"]),
        "source_experts_updated": _truthy(row["source_experts_updated"]),
        "exact_nelbo_claimed": _truthy(row["exact_nelbo_claimed"]),
    }


def _required_raw(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, object]],
    source: str,
    training_seed: int,
    query: str,
    case_id: str,
) -> Mapping[str, object]:
    try:
        return rows[(source, training_seed, query, case_id)]
    except KeyError as exc:
        raise ProtocolError("Case-OOF proxy score grid is incomplete.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_CASE_MEMBER",
    "EXPECTED_SOURCE_BLOCK_COUNT",
    "EXPECTED_SOURCE_TASK_COUNT",
    "GENERATION_DEVICES",
    "SOURCE_BLOCK_ARRAY_MEMBER",
    "SOURCE_BLOCK_INDEX_COLUMNS",
    "SOURCE_BLOCK_INDEX_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SourceCache",
)
