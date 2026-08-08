"""Hash-valid per-expert checkpoint serialization and resume loading."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..dense_residual_soft_router.compatibility import (
    CLASS_PRIOR as COMPATIBILITY_CLASS_PRIOR,
    ENERGY_SEMANTICS as COMPATIBILITY_ENERGY_SEMANTICS,
)
from .contracts import FreshProxyScoreRow
from .proxy_surface_contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    FreshProxyScoreTask,
    FreshProxyTaskResult,
)
from .proxy_surface_validation import (
    deduplicated_task_scoring_groups,
    expected_task_row_keys,
    score_row_key,
    score_row_sort_key,
)


def write_fresh_proxy_score_checkpoint(
    task: FreshProxyScoreTask,
    *,
    rows: Sequence[FreshProxyScoreRow],
    expert_lock_hash: str,
    expert_checkpoint_hash: str,
) -> FreshProxyTaskResult:
    ordered_rows = tuple(sorted(rows, key=score_row_sort_key))
    scoring_groups = deduplicated_task_scoring_groups(task)
    unhashed = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "task_hash": task.task_hash,
        "surface_hash": task.surface_hash,
        "expert_bank_binding_hash": task.expert_bank_binding_hash,
        "task_ordinal": task.task_ordinal,
        "source_center": task.source_center,
        "training_seed": task.training_seed,
        "device": task.device,
        "expert_lock_hash": expert_lock_hash,
        "expert_checkpoint_hash": expert_checkpoint_hash,
        "shard_hashes": [shard.shard_hash for shard in task.shards],
        "score_chunk_rows": task.chunk_rows,
        "unique_scored_query_count": len(scoring_groups),
        "global_query_scores_replicated_across_outer_targets": True,
        "case_level_aggregation": (
            "row_energy_sum_and_count_across_all_chunks_exact_case_mean"
        ),
        "compatibility_energy_semantics": COMPATIBILITY_ENERGY_SEMANTICS,
        "class_hypothesis_prior": list(COMPATIBILITY_CLASS_PRIOR),
        "exact_nelbo": False,
        "expert_load_count": 1,
        "tf32_enabled": False,
        "labels_consumed": False,
        "evaluation_overlap": False,
        "source_expert_updated": False,
        "records": [row.to_payload() for row in ordered_rows],
        "record_count": len(ordered_rows),
    }
    checkpoint_hash = stable_hash(unhashed)
    atomic_write_json(
        task.checkpoint_path,
        {**unhashed, "checkpoint_hash": checkpoint_hash},
    )
    return FreshProxyTaskResult(
        source_center=task.source_center,
        training_seed=task.training_seed,
        rows=ordered_rows,
        checkpoint_hash=checkpoint_hash,
        task_hash=task.task_hash,
        expert_lock_hash=expert_lock_hash,
        expert_checkpoint_hash=expert_checkpoint_hash,
        resumed=False,
    )


def load_fresh_proxy_score_checkpoint(
    path: str | Path,
    *,
    task: FreshProxyScoreTask,
) -> FreshProxyTaskResult:
    """Fail closed unless a persisted per-expert checkpoint exactly rebinds."""

    checkpoint_path = Path(path)
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read fresh proxy expert checkpoint.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Fresh proxy expert checkpoint must be an object.")
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_hash"
    }
    records = payload.get("records")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("status") != "COMPLETE"
        or payload.get("task_hash") != task.task_hash
        or payload.get("surface_hash") != task.surface_hash
        or payload.get("expert_bank_binding_hash")
        != task.expert_bank_binding_hash
        or int(payload.get("task_ordinal", -1)) != task.task_ordinal
        or str(payload.get("source_center")) != task.source_center
        or int(payload.get("training_seed", -1)) != task.training_seed
        or str(payload.get("device")) != task.device
        or payload.get("shard_hashes")
        != [shard.shard_hash for shard in task.shards]
        or int(payload.get("score_chunk_rows", -1)) != task.chunk_rows
        or int(payload.get("unique_scored_query_count", -1))
        != len(deduplicated_task_scoring_groups(task))
        or payload.get("global_query_scores_replicated_across_outer_targets")
        is not True
        or payload.get("compatibility_energy_semantics")
        != COMPATIBILITY_ENERGY_SEMANTICS
        or payload.get("class_hypothesis_prior")
        != list(COMPATIBILITY_CLASS_PRIOR)
        or payload.get("exact_nelbo") is not False
        or payload.get("expert_load_count") != 1
        or payload.get("tf32_enabled") is not False
        or payload.get("labels_consumed") is not False
        or payload.get("evaluation_overlap") is not False
        or payload.get("source_expert_updated") is not False
        or not isinstance(records, list)
        or int(payload.get("record_count", -1)) != len(records)
        or payload.get("checkpoint_hash") != stable_hash(unhashed)
        or not str(payload.get("expert_lock_hash", ""))
        or not str(payload.get("expert_checkpoint_hash", ""))
    ):
        raise ProtocolError("Fresh proxy expert checkpoint binding drifted.")
    rows = tuple(score_row_from_payload(record) for record in records)
    expected_keys = expected_task_row_keys(task)
    observed_keys = {score_row_key(row) for row in rows}
    if (
        len(rows) != len(observed_keys)
        or observed_keys != expected_keys
        or tuple(sorted(rows, key=score_row_sort_key)) != rows
    ):
        raise ProtocolError("Fresh proxy expert checkpoint row coverage drifted.")
    return FreshProxyTaskResult(
        source_center=task.source_center,
        training_seed=task.training_seed,
        rows=rows,
        checkpoint_hash=str(payload["checkpoint_hash"]),
        task_hash=task.task_hash,
        expert_lock_hash=str(payload["expert_lock_hash"]),
        expert_checkpoint_hash=str(payload["expert_checkpoint_hash"]),
        resumed=True,
    )


def score_row_from_payload(raw: object) -> FreshProxyScoreRow:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Fresh proxy checkpoint score row is malformed.")
    forbidden = {"labels", "target_labels", "utility", "nelbo"}.intersection(raw)
    if forbidden:
        raise ProtocolError("Fresh proxy checkpoint contains forbidden values.")
    return FreshProxyScoreRow(
        outer_target=str(raw.get("outer_target", "")),
        query_role=str(raw.get("query_role", "")),
        query_center=str(raw.get("query_center", "")),
        case_id=str(raw.get("case_id", "")),
        candidate_source=str(raw.get("candidate_source", "")),
        training_seed=int(raw.get("training_seed", -1)),
        proxy_energy=float(raw.get("proxy_energy", math.nan)),
        labels_consumed=raw.get("labels_consumed"),  # type: ignore[arg-type]
        evaluation_overlap=raw.get("evaluation_overlap"),  # type: ignore[arg-type]
        source_expert_updated=raw.get(  # type: ignore[arg-type]
            "source_expert_updated"
        ),
        proxy_energy_semantics=str(raw.get("proxy_energy_semantics", "")),
    )


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = (
    "atomic_write_json",
    "load_fresh_proxy_score_checkpoint",
    "write_fresh_proxy_score_checkpoint",
)
