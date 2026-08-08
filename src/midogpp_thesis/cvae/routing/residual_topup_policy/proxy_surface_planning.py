"""Deterministic leave-H/q-out task planning for fresh proxy scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .contracts import FIXED_TRAINING_SEEDS
from .proxy_surface_contracts import (
    DEFAULT_DEVICES,
    EXPECTED_EXPERT_TASK_COUNT,
    SCORE_CHUNK_ROWS,
    TASK_SCHEMA_VERSION,
    FreshProxyScoreTask,
    FreshQueryShard,
    source_is_legal_for_shard,
)
from .proxy_surface_validation import validate_query_shards


def build_fresh_proxy_score_tasks(
    shards: Iterable[FreshQueryShard],
    *,
    expert_bank_root: str | Path,
    expert_bank_binding_hash: str,
    checkpoint_root: str | Path,
    devices: Sequence[str] = DEFAULT_DEVICES,
    chunk_rows: int = SCORE_CHUNK_ROWS,
) -> tuple[FreshProxyScoreTask, ...]:
    """Build the canonical 27 expert jobs with exact G/S source exclusions."""

    ordered_shards = validate_query_shards(tuple(shards))
    device_order = tuple(str(value) for value in devices)
    if (
        len(device_order) != 2
        or len(set(device_order)) != 2
        or any(not value for value in device_order)
        or chunk_rows != SCORE_CHUNK_ROWS
        or not str(expert_bank_binding_hash)
    ):
        raise ProtocolError("Fresh proxy dual-device runtime contract drifted.")
    bank_root = Path(expert_bank_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_root).expanduser().resolve()
    surface_payload = {
        "schema_version": "midogpp_residual_topup_fresh_query_surface_v1",
        "centers": list(CENTERS),
        "training_seeds": list(FIXED_TRAINING_SEEDS),
        "shards": [shard.to_payload() for shard in ordered_shards],
        "expert_bank_root": str(bank_root),
        "expert_bank_binding_hash": str(expert_bank_binding_hash),
        "devices": list(device_order),
        "chunk_rows": chunk_rows,
        "labels_consumed": False,
        "source_experts_updated": False,
    }
    surface_hash = stable_hash(surface_payload)
    tasks: list[FreshProxyScoreTask] = []
    task_ordinal = 0
    for source in CENTERS:
        legal_shards = tuple(
            shard
            for shard in ordered_shards
            if source_is_legal_for_shard(source, shard)
        )
        for training_seed in FIXED_TRAINING_SEEDS:
            base = {
                "schema_version": TASK_SCHEMA_VERSION,
                "task_ordinal": task_ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "device": device_order[task_ordinal % len(device_order)],
                "expert_bank_root": str(bank_root),
                "expert_bank_binding_hash": str(expert_bank_binding_hash),
                "shard_hashes": [shard.shard_hash for shard in legal_shards],
                "checkpoint_path": str(
                    checkpoint_path
                    / f"source_{source}_train_{training_seed}.json"
                ),
                "surface_hash": surface_hash,
                "chunk_rows": chunk_rows,
                "tf32_enabled": False,
                "labels_consumed": False,
                "source_expert_updated": False,
            }
            tasks.append(
                FreshProxyScoreTask(
                    task_ordinal=task_ordinal,
                    source_center=source,
                    training_seed=training_seed,
                    device=str(base["device"]),
                    expert_bank_root=bank_root,
                    expert_bank_binding_hash=str(expert_bank_binding_hash),
                    shards=legal_shards,
                    checkpoint_path=Path(str(base["checkpoint_path"])),
                    surface_hash=surface_hash,
                    task_hash=stable_hash(base),
                    chunk_rows=chunk_rows,
                )
            )
            task_ordinal += 1
    if (
        len(tasks) != EXPECTED_EXPERT_TASK_COUNT
        or len({task.key for task in tasks}) != EXPECTED_EXPERT_TASK_COUNT
        or tuple(task.task_ordinal for task in tasks)
        != tuple(range(EXPECTED_EXPERT_TASK_COUNT))
    ):
        raise ProtocolError("Fresh proxy expert task grid is incomplete.")
    return tuple(tasks)


__all__ = ("build_fresh_proxy_score_tasks",)
