"""Label-free variational compatibility aggregation and lineage sealing."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    TRAINING_SEEDS,
    CandidatePoolReceipt,
    CompatibilityReceipt,
    ReplicaEnergyInput,
    SupportPartitionReceipt,
)


def build_compatibility_receipts(
    *,
    candidate_pool: CandidatePoolReceipt,
    support_partition: SupportPartitionReceipt,
    replica_energies: Sequence[ReplicaEnergyInput],
) -> tuple[CompatibilityReceipt, ...]:
    """Aggregate every retained 17/42/101 replica without seed selection.

    Lower calibrated energy is more compatible.  ``rank_margin`` is positive
    only for the best candidate (distance to runner-up), zero for an exact tie,
    and negative for every lower-ranked candidate (distance behind the best).
    This sign convention lets a model distinguish a clear winner from merely a
    low absolute energy.
    """

    if not isinstance(candidate_pool, CandidatePoolReceipt) or not isinstance(
        support_partition, SupportPartitionReceipt
    ):
        raise ProtocolError("Compatibility aggregation requires typed pool and support receipts.")
    if support_partition.center_id != candidate_pool.query_center_id:
        raise ProtocolError("Compatibility support belongs to a different query center.")
    rows = tuple(replica_energies)
    grouped: dict[str, list[ReplicaEnergyInput]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, ReplicaEnergyInput):
            raise ProtocolError("Compatibility aggregation received an untyped replica.")
        grouped[row.candidate_source_id].append(row)
    if set(grouped) != set(candidate_pool.candidate_center_ids):
        raise ProtocolError("Compatibility replicas do not cover the exact candidate pool.")
    for source, replicas in grouped.items():
        seeds = tuple(sorted(row.training_seed for row in replicas))
        if seeds != TRAINING_SEEDS or len({row.training_seed for row in replicas}) != len(replicas):
            raise ProtocolError(
                f"Compatibility candidate {source} does not contain exactly seeds 17/42/101."
            )

    statistics: dict[str, tuple[float, float]] = {}
    for source, replicas in grouped.items():
        values = tuple(row.calibrated_z for row in replicas)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        statistics[source] = (mean, std)
    order = tuple(sorted(statistics, key=lambda source: (statistics[source][0], source)))
    best_mean = statistics[order[0]][0]
    runner_up_mean = statistics[order[1]][0] if len(order) > 1 else best_mean

    output: list[CompatibilityReceipt] = []
    for ordinal, source in enumerate(order, start=1):
        mean, std = statistics[source]
        margin = runner_up_mean - mean if ordinal == 1 else best_mean - mean
        output.append(
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
                rank=ordinal,
                rank_margin=margin,
            )
        )
    return tuple(sorted(output, key=lambda row: row.candidate_source_id))


def compatibility_by_candidate(
    receipts: Sequence[CompatibilityReceipt],
) -> dict[str, CompatibilityReceipt]:
    rows = tuple(receipts)
    output = {row.candidate_source_id: row for row in rows}
    if len(output) != len(rows):
        raise ProtocolError("Compatibility receipts contain duplicate candidates.")
    lineage = {
        (
            row.outer_target_id,
            row.query_center_id,
            row.candidate_pool_hash,
            row.support_partition_hash,
        )
        for row in rows
    }
    if not rows or len(lineage) != 1:
        raise ProtocolError("Compatibility receipts crossed pool or support lineages.")
    return output


__all__ = ("build_compatibility_receipts", "compatibility_by_candidate")
