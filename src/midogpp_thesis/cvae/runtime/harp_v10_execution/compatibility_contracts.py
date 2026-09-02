"""Identity-neutral, label-free compatibility contracts for HARP v10.

This module deliberately contains no routing model or endpoint type.  It seals
the exact candidate pools and the three-replica variational compatibility
summary consumed as a proxy feature by the source-active router.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from ...routing.policy_calibrated_residual_router_v10.hashing import canonical_hash
from .hash_contracts import require_sha256, require_stable_hash


TRAINING_SEEDS = (17, 42, 101)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"HARP v10 {name} must be a canonical string.")
    return value


def _finite(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"HARP v10 {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"HARP v10 {name} must be finite.")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True)
class CandidatePoolReceipt:
    """Exact ``C-{H,q}`` source or ``C-{H}`` target candidate inventory."""

    outer_target_id: str
    query_center_id: str
    all_center_ids: tuple[str, ...]
    candidate_center_ids: tuple[str, ...]
    bank_lock_hash: str
    pool_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _text(self.outer_target_id, name="outer target")
        query = _text(self.query_center_id, name="query center")
        centers = tuple(sorted(_text(value, name="center") for value in self.all_center_ids))
        candidates = tuple(
            sorted(_text(value, name="candidate") for value in self.candidate_center_ids)
        )
        if (
            len(centers) != len(set(centers))
            or len(candidates) != len(set(candidates))
            or outer not in centers
            or query not in centers
        ):
            raise ProtocolError("HARP v10 candidate-pool inventory is malformed.")
        excluded = {outer} if outer == query else {outer, query}
        expected = tuple(value for value in centers if value not in excluded)
        if candidates != expected or not candidates:
            raise ProtocolError("HARP v10 candidate pool violated H/query exclusion.")
        bank_hash = require_sha256(self.bank_lock_hash, name="expert-bank lock hash")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_center_id", query)
        object.__setattr__(self, "all_center_ids", centers)
        object.__setattr__(self, "candidate_center_ids", candidates)
        object.__setattr__(self, "bank_lock_hash", bank_hash)
        object.__setattr__(
            self,
            "pool_hash",
            canonical_hash(
                {
                    "schema_version": "harp_v10_candidate_pool_v1",
                    "outer_target_id": outer,
                    "query_center_id": query,
                    "all_center_ids": centers,
                    "candidate_center_ids": candidates,
                    "scope": "C_MINUS_H" if outer == query else "C_MINUS_H_MINUS_Q",
                    "bank_lock_hash": bank_hash,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def target_scope(self) -> bool:
        return self.outer_target_id == self.query_center_id


@dataclass(frozen=True, slots=True)
class SupportPartitionReceipt:
    center_id: str
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    support_manifest_hash: str
    evaluation_manifest_hash: str
    partition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = _text(self.center_id, name="partition center")
        support = tuple(sorted(_text(value, name="support case") for value in self.support_case_ids))
        evaluation = tuple(
            sorted(_text(value, name="evaluation case") for value in self.evaluation_case_ids)
        )
        if (
            not support
            or not evaluation
            or len(support) != len(set(support))
            or len(evaluation) != len(set(evaluation))
            or set(support).intersection(evaluation)
        ):
            raise ProtocolError("HARP v10 support/evaluation cases are not disjoint.")
        support_hash = require_sha256(
            self.support_manifest_hash, name="support manifest hash"
        )
        evaluation_hash = require_sha256(
            self.evaluation_manifest_hash, name="evaluation manifest hash"
        )
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "support_manifest_hash", support_hash)
        object.__setattr__(self, "evaluation_manifest_hash", evaluation_hash)
        object.__setattr__(
            self,
            "partition_hash",
            canonical_hash(
                {
                    "schema_version": "harp_v10_support_partition_v1",
                    "center_id": center,
                    "support_case_ids": support,
                    "evaluation_case_ids": evaluation,
                    "support_manifest_hash": support_hash,
                    "evaluation_manifest_hash": evaluation_hash,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def support_hash(self) -> str:
        return canonical_hash(
            {
                "center_id": self.center_id,
                "support_case_ids": self.support_case_ids,
                "partition_hash": self.partition_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ReplicaEnergyInput:
    candidate_source_id: str
    training_seed: int
    query_case_equal_energy: float
    own_source_location: float
    own_source_scale: float
    checkpoint_hash: str
    source_frame_hash: str
    sampler_hash: str

    def __post_init__(self) -> None:
        source = _text(self.candidate_source_id, name="compatibility candidate")
        seed = int(self.training_seed)
        scale = _finite(self.own_source_scale, name="own-source scale")
        if seed not in TRAINING_SEEDS or scale <= 0.0:
            raise ProtocolError("HARP v10 compatibility replica semantics drifted.")
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "training_seed", seed)
        object.__setattr__(
            self,
            "query_case_equal_energy",
            _finite(self.query_case_equal_energy, name="query energy"),
        )
        object.__setattr__(
            self,
            "own_source_location",
            _finite(self.own_source_location, name="own-source location"),
        )
        object.__setattr__(self, "own_source_scale", scale)
        object.__setattr__(
            self,
            "checkpoint_hash",
            require_sha256(self.checkpoint_hash, name="checkpoint hash"),
        )
        object.__setattr__(
            self,
            "source_frame_hash",
            require_stable_hash(self.source_frame_hash, name="source-frame hash"),
        )
        object.__setattr__(
            self,
            "sampler_hash",
            require_stable_hash(self.sampler_hash, name="sampler-state hash"),
        )

    @property
    def calibrated_z(self) -> float:
        return (self.query_case_equal_energy - self.own_source_location) / self.own_source_scale


@dataclass(frozen=True, slots=True)
class CompatibilityReceipt:
    outer_target_id: str
    query_center_id: str
    candidate_source_id: str
    candidate_pool_hash: str
    support_partition_hash: str
    support_hash: str
    support_manifest_hash: str
    replica_scores: tuple[ReplicaEnergyInput, ...]
    mean_z: float
    std_z: float
    rank: int
    rank_margin: float
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _text(self.outer_target_id, name="compatibility outer")
        query = _text(self.query_center_id, name="compatibility query")
        source = _text(self.candidate_source_id, name="compatibility candidate")
        replicas = tuple(sorted(self.replica_scores, key=lambda row: row.training_seed))
        if (
            tuple(row.training_seed for row in replicas) != TRAINING_SEEDS
            or any(row.candidate_source_id != source for row in replicas)
            or int(self.rank) < 1
        ):
            raise ProtocolError("HARP v10 compatibility receipt replica grid drifted.")
        mean = _finite(self.mean_z, name="compatibility mean")
        std = _finite(self.std_z, name="compatibility dispersion")
        margin = _finite(self.rank_margin, name="compatibility margin")
        expected_mean = sum(row.calibrated_z for row in replicas) / len(replicas)
        expected_std = math.sqrt(
            sum((row.calibrated_z - expected_mean) ** 2 for row in replicas) / len(replicas)
        )
        if (
            std < 0.0
            or not math.isclose(mean, expected_mean, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(std, expected_std, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ProtocolError("HARP v10 compatibility summary drifted from replicas.")
        for name in (
            "candidate_pool_hash",
            "support_partition_hash",
            "support_hash",
            "support_manifest_hash",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_center_id", query)
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "replica_scores", replicas)
        object.__setattr__(self, "mean_z", mean)
        object.__setattr__(self, "std_z", std)
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "rank_margin", margin)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "harp_v10_compatibility_receipt_v1",
                    "outer_target_id": outer,
                    "query_center_id": query,
                    "candidate_source_id": source,
                    "candidate_pool_hash": self.candidate_pool_hash,
                    "support_partition_hash": self.support_partition_hash,
                    "support_hash": self.support_hash,
                    "support_manifest_hash": self.support_manifest_hash,
                    "replicas": replicas,
                    "mean_z": mean,
                    "std_z": std,
                    "rank": int(self.rank),
                    "rank_margin": margin,
                    "proxy_semantics": "VARIATIONAL_COMPATIBILITY_NOT_NELBO_OR_UTILITY",
                    "labels_consumed": False,
                }
            ),
        )


def build_source_candidate_pool(
    *,
    outer_target_id: str,
    pseudo_query_id: str,
    all_center_ids: Sequence[str],
    bank_lock_hash: str,
) -> CandidatePoolReceipt:
    centers = tuple(sorted(str(value) for value in all_center_ids))
    return CandidatePoolReceipt(
        outer_target_id=str(outer_target_id),
        query_center_id=str(pseudo_query_id),
        all_center_ids=centers,
        candidate_center_ids=tuple(
            value for value in centers if value not in {str(outer_target_id), str(pseudo_query_id)}
        ),
        bank_lock_hash=bank_lock_hash,
    )


def build_target_candidate_pool(
    *, outer_target_id: str, all_center_ids: Sequence[str], bank_lock_hash: str
) -> CandidatePoolReceipt:
    centers = tuple(sorted(str(value) for value in all_center_ids))
    return CandidatePoolReceipt(
        outer_target_id=str(outer_target_id),
        query_center_id=str(outer_target_id),
        all_center_ids=centers,
        candidate_center_ids=tuple(value for value in centers if value != str(outer_target_id)),
        bank_lock_hash=bank_lock_hash,
    )


def build_compatibility_receipts(
    *,
    candidate_pool: CandidatePoolReceipt,
    support_partition: SupportPartitionReceipt,
    replica_energies: Sequence[ReplicaEnergyInput],
) -> tuple[CompatibilityReceipt, ...]:
    if support_partition.center_id != candidate_pool.query_center_id:
        raise ProtocolError("HARP v10 compatibility support belongs to another query.")
    grouped: dict[str, list[ReplicaEnergyInput]] = defaultdict(list)
    for row in replica_energies:
        if not isinstance(row, ReplicaEnergyInput):
            raise ProtocolError("HARP v10 compatibility replica is untyped.")
        grouped[row.candidate_source_id].append(row)
    if set(grouped) != set(candidate_pool.candidate_center_ids):
        raise ProtocolError("HARP v10 compatibility replicas do not cover the candidate pool.")
    statistics: dict[str, tuple[float, float]] = {}
    for source, replicas in grouped.items():
        ordered = tuple(sorted(replicas, key=lambda row: row.training_seed))
        if tuple(row.training_seed for row in ordered) != TRAINING_SEEDS:
            raise ProtocolError("HARP v10 compatibility requires seeds 17,42,101.")
        values = tuple(row.calibrated_z for row in ordered)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        statistics[source] = (mean, std)
    order = tuple(sorted(statistics, key=lambda source: (statistics[source][0], source)))
    best_mean = statistics[order[0]][0]
    runner_up = statistics[order[1]][0] if len(order) > 1 else best_mean
    receipts = []
    for rank, source in enumerate(order, start=1):
        mean, std = statistics[source]
        receipts.append(
            CompatibilityReceipt(
                outer_target_id=candidate_pool.outer_target_id,
                query_center_id=candidate_pool.query_center_id,
                candidate_source_id=source,
                candidate_pool_hash=candidate_pool.pool_hash,
                support_partition_hash=support_partition.partition_hash,
                support_hash=support_partition.support_hash,
                support_manifest_hash=support_partition.support_manifest_hash,
                replica_scores=tuple(sorted(grouped[source], key=lambda row: row.training_seed)),
                mean_z=mean,
                std_z=std,
                rank=rank,
                rank_margin=(runner_up - mean if rank == 1 else best_mean - mean),
            )
        )
    return tuple(sorted(receipts, key=lambda row: row.candidate_source_id))


def compatibility_by_candidate(
    receipts: Sequence[CompatibilityReceipt],
) -> dict[str, CompatibilityReceipt]:
    rows = tuple(receipts)
    output = {row.candidate_source_id: row for row in rows}
    lineage = {
        (row.outer_target_id, row.query_center_id, row.candidate_pool_hash, row.support_partition_hash)
        for row in rows
    }
    if not rows or len(output) != len(rows) or len(lineage) != 1:
        raise ProtocolError("HARP v10 compatibility receipts crossed pool lineages.")
    return output


__all__ = (
    "TRAINING_SEEDS",
    "CandidatePoolReceipt",
    "CompatibilityReceipt",
    "ReplicaEnergyInput",
    "SupportPartitionReceipt",
    "build_compatibility_receipts",
    "build_source_candidate_pool",
    "build_target_candidate_pool",
    "compatibility_by_candidate",
)
