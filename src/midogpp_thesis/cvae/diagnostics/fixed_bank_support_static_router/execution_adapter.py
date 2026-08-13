"""Small execution adapters around neutral runtime and core partition APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .partitions import build_five_fold_partition
from .prediction_adapter import (
    FrozenSourceStreamCache,
    GlobalPredictionSeal,
    cleanup_validated_local_stage,
    enter_cuda_free_cpu_phase,
    load_frozen_source_streams,
    load_global_prediction_seal,
    load_validated_workstation_preflight,
    materialize_probabilities,
    materialize_sources,
    run_workstation_preflight,
    runtime_summary_payload,
    stage_sources_for_cpu,
)
from .products import CaseIdentityRow


def build_case_partition(frame: object, *, config: object) -> object:
    seed = int(getattr(config, "protocol")["partition_seed"])
    identities = tuple(
        CaseIdentityRow(
            str(row.center), str(row.case_id), str(row.evaluation_row_id)
        )
        for row in getattr(frame, "rows")
    )
    partition = build_five_fold_partition(identities, partition_seed=seed)
    if len(getattr(partition, "folds")) != 45:
        raise ProtocolError("S4 partition did not produce the exact 45 routes.")
    return partition


__all__ = (
    "FrozenSourceStreamCache",
    "GlobalPredictionSeal",
    "build_case_partition",
    "cleanup_validated_local_stage",
    "enter_cuda_free_cpu_phase",
    "load_frozen_source_streams",
    "load_global_prediction_seal",
    "load_validated_workstation_preflight",
    "materialize_probabilities",
    "materialize_sources",
    "run_workstation_preflight",
    "runtime_summary_payload",
    "stage_sources_for_cpu",
)
