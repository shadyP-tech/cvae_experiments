"""Small public facade over the neutral workstation runtime adapters."""

from __future__ import annotations

from .scratch_policy import cleanup_validated_scratch
from .source_surface_runtime import (
    GlobalPredictionSeal,
    MaterializedSourceCaches,
    ProbabilityIndexRow,
    build_exact_nine_surface,
    load_frozen_source_streams,
    materialize_probabilities,
    materialize_sources,
    physical_partition_hash,
    probability_index_rows,
    runtime_summary_payload,
)
from .workstation_preflight import (
    load_validated_workstation_preflight,
    run_label_free_workstation_preflight,
)


__all__ = (
    "GlobalPredictionSeal",
    "MaterializedSourceCaches",
    "ProbabilityIndexRow",
    "build_exact_nine_surface",
    "cleanup_validated_scratch",
    "load_frozen_source_streams",
    "load_validated_workstation_preflight",
    "materialize_probabilities",
    "materialize_sources",
    "physical_partition_hash",
    "probability_index_rows",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
)
