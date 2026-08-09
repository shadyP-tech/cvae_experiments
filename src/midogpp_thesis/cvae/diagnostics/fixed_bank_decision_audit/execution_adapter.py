"""Narrow adapters over label-free exact-tail execution primitives.

The combined prediction primitive fits each canonical exact-tail classifier
once and applies it to both support and evaluation rows.  This package owns
the consuming contracts and never imports the prior case-aware audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ..utility_aligned_ensemble_endpoint_router.combined_prediction_io import (
    read_combined_store,
)
from ..utility_aligned_ensemble_endpoint_router.contracts import (
    BASE_ACTION_ID,
    h_x_e_action_id,
)
from ..utility_aligned_ensemble_endpoint_router.development_label_access import (
    OpenedDevelopmentLabels,
    open_globally_sealed_development_labels,
)
from ..utility_aligned_ensemble_endpoint_router.development_prediction_execution import (
    DEVELOPMENT_ARRAY_MEMBER,
    DEVELOPMENT_INDEX_MEMBER,
    EXPECTED_DEVELOPMENT_CELL_COUNT,
    validate_development_prediction_store,
)
from ..utility_aligned_ensemble_endpoint_router.development_seal import (
    GLOBAL_DEVELOPMENT_SEAL_MEMBER,
    DevelopmentPredictionCapability,
    GlobalDevelopmentPredictionSeal,
    materialize_development_predictions,
    validate_global_development_seal,
)
from ..utility_aligned_exact_tail_router.runtime_preflight import (
    run_workstation_preflight as _run_workstation_preflight,
)
from .artifact_io import read_json


def run_workstation_preflight(
    artifact_root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    if (
        int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or runtime.get("persistent_source_workers") is not True
        or runtime.get("phase_order")
        != "two_gpu_source_streams_then_four_by_three_cpu_then_audit"
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or list(runtime.get("scratch_preference", ()))
        != ["/data/local/fixed_bank_decision_audit_v1", "artifact_parent"]
    ):
        raise ProtocolError("Fixed-bank workstation topology drifted.")
    return _run_workstation_preflight(artifact_root, runtime=runtime)


def load_development_prediction_capability(
    root: Path,
) -> DevelopmentPredictionCapability:
    array_path = root / DEVELOPMENT_ARRAY_MEMBER
    index_path = root / DEVELOPMENT_INDEX_MEMBER
    seal_path = root / GLOBAL_DEVELOPMENT_SEAL_MEMBER
    return DevelopmentPredictionCapability(
        store=read_combined_store(array_path, index_path),
        seal=GlobalDevelopmentPredictionSeal(read_json(seal_path)),
        seal_path=seal_path,
        prediction_index_path=index_path,
        prediction_arrays_path=array_path,
    )


__all__ = (
    "BASE_ACTION_ID",
    "DEVELOPMENT_ARRAY_MEMBER",
    "DEVELOPMENT_INDEX_MEMBER",
    "GLOBAL_DEVELOPMENT_SEAL_MEMBER",
    "EXPECTED_DEVELOPMENT_CELL_COUNT",
    "DevelopmentPredictionCapability",
    "GlobalDevelopmentPredictionSeal",
    "OpenedDevelopmentLabels",
    "load_development_prediction_capability",
    "materialize_development_predictions",
    "open_globally_sealed_development_labels",
    "read_combined_store",
    "run_workstation_preflight",
    "validate_development_prediction_store",
    "validate_global_development_seal",
    "h_x_e_action_id",
)
