"""Immutable contracts and hash-bound query shards for fresh proxy scoring."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .contracts import (
    FIXED_TRAINING_SEEDS,
    GLOBAL_PSEUDOQUERY_ROLE,
    QUERY_ROLES,
    TARGET_SUPPORT_ROLE,
    FreshProxyScoreRow,
)


DEFAULT_DEVICES = ("cuda:0", "cuda:1")
SCORE_CHUNK_ROWS = 2048
COMMON_FEATURE_DIM = 3840
EXPECTED_QUERY_SHARD_COUNT = len(CENTERS) * len(CENTERS)
EXPECTED_EXPERT_TASK_COUNT = len(CENTERS) * len(FIXED_TRAINING_SEEDS)
SHARD_SCHEMA_VERSION = "midogpp_residual_topup_fresh_query_shard_v1"
TASK_SCHEMA_VERSION = "midogpp_residual_topup_fresh_proxy_task_v1"
CHECKPOINT_SCHEMA_VERSION = (
    "midogpp_residual_topup_fresh_proxy_expert_checkpoint_v1"
)
FRESH_SURFACE_ATTESTATION_SCHEMA_VERSION = (
    "midogpp_residual_topup_fresh_proxy_surface_attestation_v1"
)
PROXY_SCORE_COLUMNS = (
    "outer_target",
    "query_role",
    "query_center",
    "case_id",
    "candidate_source",
    "training_seed",
    "proxy_energy",
    "labels_consumed",
    "evaluation_overlap",
    "source_expert_updated",
    "proxy_energy_semantics",
)


ArrayLoader = Callable[[Path], np.ndarray]


def embedding_array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes of one query embedding matrix."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _fresh_query_shard_payload(
    *,
    outer_target: str,
    query_role: str,
    query_center: str,
    embedding_path: Path,
    embedding_sha256: str,
    case_ids: Sequence[str],
    evaluation_case_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "outer_target": outer_target,
        "query_role": query_role,
        "query_center": query_center,
        "embedding_path": str(embedding_path),
        "embedding_array_sha256": embedding_sha256,
        "case_ids": list(case_ids),
        "evaluation_case_ids": list(evaluation_case_ids),
        "row_count": len(case_ids),
        "case_count": len(set(case_ids)),
        "labels_consumed": False,
        "evaluation_overlap": False,
        "source_experts_updated": False,
    }


@dataclass(frozen=True)
class FreshQueryShard:
    """One hash-bound unlabeled G pseudoquery or S support embedding shard."""

    outer_target: str
    query_role: str
    query_center: str
    embedding_path: Path
    embedding_array_sha256: str
    case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    shard_hash: str
    labels_consumed: bool = False
    evaluation_overlap: bool = False
    source_experts_updated: bool = False

    def __post_init__(self) -> None:
        target = str(self.outer_target)
        role = str(self.query_role)
        query = str(self.query_center)
        path = Path(self.embedding_path).expanduser().resolve()
        cases = tuple(str(value) for value in self.case_ids)
        evaluation = tuple(str(value) for value in self.evaluation_case_ids)
        if (
            target not in CENTERS
            or query not in CENTERS
            or role not in QUERY_ROLES
            or path.suffix != ".npy"
            or len(self.embedding_array_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.embedding_array_sha256
            )
            or not cases
            or not evaluation
            or any(not value or value.strip() != value for value in cases)
            or any(not value or value.strip() != value for value in evaluation)
            or set(cases).intersection(evaluation)
        ):
            raise ProtocolError(
                "Fresh proxy query shard identity/hash geometry is invalid."
            )
        if role == GLOBAL_PSEUDOQUERY_ROLE and query == target:
            raise ProtocolError("Fresh G pseudoquery q must differ from outer H.")
        if role == TARGET_SUPPORT_ROLE and query != target:
            raise ProtocolError("Fresh S support shard must query its own outer H.")
        if (
            type(self.labels_consumed) is not bool
            or self.labels_consumed
            or type(self.evaluation_overlap) is not bool
            or self.evaluation_overlap
            or type(self.source_experts_updated) is not bool
            or self.source_experts_updated
        ):
            raise ProtocolError(
                "Fresh proxy query shards must be label-blind, evaluation-disjoint, "
                "and non-updating."
            )
        payload = _fresh_query_shard_payload(
            outer_target=target,
            query_role=role,
            query_center=query,
            embedding_path=path,
            embedding_sha256=self.embedding_array_sha256,
            case_ids=cases,
            evaluation_case_ids=evaluation,
        )
        if self.shard_hash != stable_hash(payload):
            raise ProtocolError("Fresh proxy query shard hash drifted.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "query_role", role)
        object.__setattr__(self, "query_center", query)
        object.__setattr__(self, "embedding_path", path)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "evaluation_case_ids", evaluation)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.outer_target, self.query_role, self.query_center

    @property
    def unique_case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.case_ids)))

    def to_payload(self) -> dict[str, object]:
        payload = _fresh_query_shard_payload(
            outer_target=self.outer_target,
            query_role=self.query_role,
            query_center=self.query_center,
            embedding_path=self.embedding_path,
            embedding_sha256=self.embedding_array_sha256,
            case_ids=self.case_ids,
            evaluation_case_ids=self.evaluation_case_ids,
        )
        return {**payload, "shard_hash": self.shard_hash}


def make_fresh_query_shard(
    *,
    outer_target: object,
    query_role: object,
    query_center: object,
    embedding_path: str | Path,
    case_ids: Sequence[object],
    evaluation_case_ids: Sequence[object],
    array_loader: ArrayLoader | None = None,
) -> FreshQueryShard:
    """Load once to bind a query shard to its exact float32 array bytes."""

    path = Path(embedding_path).expanduser().resolve()
    loader = array_loader or default_array_loader
    array = validated_embedding_array(
        loader(path), expected_row_count=len(case_ids)
    )
    target = str(outer_target)
    role = str(query_role)
    query = str(query_center)
    cases = tuple(str(value) for value in case_ids)
    evaluation = tuple(str(value) for value in evaluation_case_ids)
    digest = embedding_array_sha256(array)
    payload = _fresh_query_shard_payload(
        outer_target=target,
        query_role=role,
        query_center=query,
        embedding_path=path,
        embedding_sha256=digest,
        case_ids=cases,
        evaluation_case_ids=evaluation,
    )
    return FreshQueryShard(
        outer_target=target,
        query_role=role,
        query_center=query,
        embedding_path=path,
        embedding_array_sha256=digest,
        case_ids=cases,
        evaluation_case_ids=evaluation,
        shard_hash=stable_hash(payload),
    )


@dataclass(frozen=True)
class FreshProxyScoreTask:
    """All legal fresh query shards scored by one immutable expert replica."""

    task_ordinal: int
    source_center: str
    training_seed: int
    device: str
    expert_bank_root: Path
    expert_bank_binding_hash: str
    shards: tuple[FreshQueryShard, ...]
    checkpoint_path: Path
    surface_hash: str
    task_hash: str
    chunk_rows: int = SCORE_CHUNK_ROWS

    def __post_init__(self) -> None:
        if (
            isinstance(self.task_ordinal, bool)
            or not isinstance(self.task_ordinal, int)
            or self.task_ordinal < 0
            or self.source_center not in CENTERS
            or self.training_seed not in FIXED_TRAINING_SEEDS
            or not self.device
            or not self.expert_bank_binding_hash
            or not self.shards
            or self.chunk_rows != SCORE_CHUNK_ROWS
        ):
            raise ProtocolError("Fresh proxy expert task identity/runtime drifted.")
        if tuple(sorted(self.shards, key=shard_sort_key)) != self.shards:
            raise ProtocolError("Fresh proxy expert task shard order drifted.")
        for shard in self.shards:
            if not source_is_legal_for_shard(self.source_center, shard):
                raise ProtocolError(
                    "Fresh proxy task contains an illegal H/q source."
                )
        if self.task_hash != stable_hash(self.identity_payload()):
            raise ProtocolError("Fresh proxy expert task hash drifted.")

    @property
    def key(self) -> tuple[str, int]:
        return self.source_center, self.training_seed

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_ordinal": self.task_ordinal,
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "device": self.device,
            "expert_bank_root": str(self.expert_bank_root),
            "expert_bank_binding_hash": self.expert_bank_binding_hash,
            "shard_hashes": [shard.shard_hash for shard in self.shards],
            "checkpoint_path": str(self.checkpoint_path),
            "surface_hash": self.surface_hash,
            "chunk_rows": self.chunk_rows,
            "tf32_enabled": False,
            "labels_consumed": False,
            "source_expert_updated": False,
        }


@dataclass(frozen=True)
class FreshProxyTaskResult:
    source_center: str
    training_seed: int
    rows: tuple[FreshProxyScoreRow, ...]
    checkpoint_hash: str
    task_hash: str
    expert_lock_hash: str
    expert_checkpoint_hash: str
    resumed: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return self.source_center, self.training_seed


@dataclass(frozen=True)
class FreshProxyScoreSurface(Sequence[FreshProxyScoreRow]):
    """Deterministically ordered score grid plus resume audit metadata."""

    rows: tuple[FreshProxyScoreRow, ...]
    task_results: tuple[FreshProxyTaskResult, ...]
    surface_hash: str
    expert_bank_binding_hash: str
    resumed_task_count: int
    executed_task_count: int
    labels_consumed: bool = False
    source_experts_updated: bool = False

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int | slice
    ) -> FreshProxyScoreRow | tuple[FreshProxyScoreRow, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[FreshProxyScoreRow]:
        return iter(self.rows)


@dataclass(frozen=True)
class MaterializedFreshProxyInputs:
    """The only two registered files emitted by the fresh surface builder."""

    proxy_score_table_path: Path
    proxy_attestation_path: Path
    proxy_score_table_sha256: str
    proxy_attestation_sha256: str
    attestation_hash: str
    row_count: int


def validated_embedding_array(
    values: object,
    *,
    expected_row_count: int,
) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] != expected_row_count
        or array.shape[1] != COMMON_FEATURE_DIM
        or not np.isfinite(array).all()
    ):
        raise ProtocolError(
            "Fresh proxy embeddings must be finite row-aligned float32 matrices "
            f"in the exact {COMMON_FEATURE_DIM}-D common frame."
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def default_array_loader(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot load fresh proxy embedding shard.") from exc


def source_is_legal_for_shard(source: str, shard: FreshQueryShard) -> bool:
    if source == shard.outer_target:
        return False
    return not (
        shard.query_role == GLOBAL_PSEUDOQUERY_ROLE
        and source == shard.query_center
    )


def shard_sort_key(shard: FreshQueryShard) -> tuple[int, int, int]:
    role_order = 0 if shard.query_role == GLOBAL_PSEUDOQUERY_ROLE else 1
    return (
        CENTERS.index(shard.outer_target),
        role_order,
        CENTERS.index(shard.query_center),
    )


__all__ = (
    "ArrayLoader",
    "CHECKPOINT_SCHEMA_VERSION",
    "COMMON_FEATURE_DIM",
    "DEFAULT_DEVICES",
    "EXPECTED_EXPERT_TASK_COUNT",
    "EXPECTED_QUERY_SHARD_COUNT",
    "FRESH_SURFACE_ATTESTATION_SCHEMA_VERSION",
    "FreshProxyScoreSurface",
    "FreshProxyScoreTask",
    "FreshProxyTaskResult",
    "FreshQueryShard",
    "MaterializedFreshProxyInputs",
    "PROXY_SCORE_COLUMNS",
    "SCORE_CHUNK_ROWS",
    "SHARD_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "default_array_loader",
    "embedding_array_sha256",
    "make_fresh_query_shard",
    "shard_sort_key",
    "source_is_legal_for_shard",
    "validated_embedding_array",
)
