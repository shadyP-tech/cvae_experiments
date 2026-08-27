"""Private immutable contracts shared by the outer-worker phases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..label_capabilities import WorkerLabelDelegation
from ..physical.contracts import PRIMARY_METHOD_ID, P_METHOD_ID
from ..posterior.donor import DonorActionModel
from ..physical.planning import DonorDirectionalPriorSurface
from ..pseudo.scope import PseudoRouteScope
from ..routing.admission import AdmissionMetrics, AdmissionThresholds
from ..routing.controls import CONTROL_METHOD_IDS
from ..routing.selection import SafetyThresholds
from ..utility.actions import ActionRectangle
from ..utility.metrics import ScoredActionRectangle


TASK_PAYLOAD_SCHEMA = "scale_bp_v2_outer_science_task_payload_v1"
ROUTE_CHUNK_SCHEMA = "scale_bp_v2_outer_route_decisions_v1"
WORKER_CAPABILITY_CHUNK_SCHEMA = "scale_bp_v2_worker_capability_chunk_v1"

SCIENTIFIC_SECTION_NAMES = (
    "action_geometry",
    "support_folds",
    "influence",
    "donor_prior",
    "local_residual",
    "empirical_bayes",
    "uncertainty",
    "selection",
    "admission",
    "controls",
)
METHOD_IDS = (P_METHOD_ID, PRIMARY_METHOD_ID, *CONTROL_METHOD_IDS)


@dataclass(frozen=True, slots=True)
class ScienceSettings:
    donor_ridge_alpha: float
    local_ridge_alpha: float
    maximum_abs_standardized_feature: float
    minimum_independent_centers: int
    uncertainty_base_multiplier: float
    safety_thresholds: SafetyThresholds
    admission_thresholds: AdmissionThresholds
    contracts_hash: str


@dataclass(frozen=True, slots=True)
class ParsedTaskPayload:
    artifact_root: Path
    physical_index_path: Path
    physical_index_hash: str
    label_identity_index_path: Path
    label_identity_hash: str
    manifest_path: Path
    manifest_sha256: str
    delegation: WorkerLabelDelegation
    scientific_contracts: Mapping[str, Mapping[str, object]]
    settings: ScienceSettings


@dataclass(frozen=True, slots=True)
class PseudoRouteData:
    scope: PseudoRouteScope
    rectangle: ActionRectangle
    scored: ScoredActionRectangle


@dataclass(frozen=True, slots=True)
class DonorSurfaceBundle:
    """One exact exclusion-scoped query rectangle, never a filtered table."""

    rectangle: ActionRectangle
    scored: ScoredActionRectangle
    source_centers: tuple[str, ...]
    prior_hash: str
    plan_hash: str


@dataclass(frozen=True, slots=True)
class DonorPhaseOutput:
    final_prior: DonorDirectionalPriorSurface
    final_model: DonorActionModel
    admission: AdmissionMetrics
    pseudo_replay_hash: str
    pseudo_record_count: int
    pseudo_model_manifest_hash: str
    donor_phase_hash: str


@dataclass(frozen=True, slots=True)
class FinalRouteOutput:
    case_id: str
    sample_ids: tuple[str, ...]
    method_probabilities: Mapping[str, np.ndarray]
    record: Mapping[str, object]
    route_hash: str


__all__ = (
    "DonorPhaseOutput",
    "DonorSurfaceBundle",
    "FinalRouteOutput",
    "METHOD_IDS",
    "ParsedTaskPayload",
    "PseudoRouteData",
    "ROUTE_CHUNK_SCHEMA",
    "SCIENTIFIC_SECTION_NAMES",
    "ScienceSettings",
    "TASK_PAYLOAD_SCHEMA",
    "WORKER_CAPABILITY_CHUNK_SCHEMA",
)
