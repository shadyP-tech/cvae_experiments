"""Prediction-store slices to SCEPTRE role surfaces and additive outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    ConfusionCounts,
    FamilyOutcome,
)
from ..fixed_bank_sceptre_router.partitions import ThreeRoleFold
from ..fixed_bank_sceptre_router.phase_contracts import (
    PhaseCapability,
    TerminalEvaluationCapability,
)
from ..fixed_bank_sceptre_router.uncertainty import (
    SEED_CELL_GRID,
    RolePredictionSurface,
    build_role_prediction_surface,
)
from .label_broker import ScopedRoleLabels
from .prediction_surface import CANDIDATE_EXCLUSION_SENTINEL


@dataclass(frozen=True, slots=True)
class RoleEvidenceBundle:
    """Process-local role surface plus additive selection/calibration outcomes."""

    surface: RolePredictionSurface
    outcomes: tuple[FamilyOutcome, ...]
    exact_b: FamilyOutcome | None
    evidence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.surface, RolePredictionSurface):
            raise ProtocolError("SCEPTRE v3 role evidence surface is untyped.")
        if self.surface.role == "EVALUATION":
            if self.outcomes or self.exact_b is not None:
                raise ProtocolError("SCEPTRE v3 terminal surface carries routing outcomes.")
        else:
            if (
                tuple(row.candidate_center for row in self.outcomes)
                != legal_routing_sources(self.surface.target_center)
                or not isinstance(self.exact_b, FamilyOutcome)
            ):
                raise ProtocolError("SCEPTRE v3 role outcome inventory drifted.")
        require_sha256(self.evidence_hash, "role evidence")

    def __reduce__(self):  # pragma: no cover - label-bearing safety seam
        raise TypeError("SCEPTRE v3 role evidence cannot be serialized.")


def build_role_evidence(
    scoped: ScopedRoleLabels,
    *,
    fold: ThreeRoleFold,
    partition_hash: str,
    candidate_probabilities: np.ndarray,
    exact_b_probabilities: np.ndarray,
    candidate_source_order: Sequence[str],
    prediction_store_hash: str,
    candidate_menu_hash: str,
    exact_b_control_receipt_hash: str,
    phase_capability: PhaseCapability | TerminalEvaluationCapability,
) -> RoleEvidenceBundle:
    """Build one fold role from a read-only, label-free prediction store.

    Candidate array order is ``seed_cell, source_center, test_row`` and exact B
    order is ``seed_cell, test_row``.  Test-row slicing happens only after the
    broker has granted the exact whole-case scope.
    """

    if not isinstance(scoped, ScopedRoleLabels) or not isinstance(fold, ThreeRoleFold):
        raise ProtocolError("SCEPTRE v3 role evidence scope is untyped.")
    if (
        fold.target_center != scoped.target_center
        or fold.fold_ordinal != scoped.fold_ordinal
        or scoped.case_set_hash != fold.case_set_hash(scoped.role)
    ):
        raise ProtocolError("SCEPTRE v3 role evidence fold lineage drifted.")
    store_hash = require_sha256(prediction_store_hash, "prediction store")
    source_order = tuple(map(str, candidate_source_order))
    if source_order != CENTERS:
        raise ProtocolError("SCEPTRE v3 candidate source order drifted.")
    candidates = np.asarray(candidate_probabilities)
    baseline = np.asarray(exact_b_probabilities)
    if (
        candidates.dtype != np.float32
        or baseline.dtype != np.float32
        or candidates.ndim != 3
        or baseline.ndim != 2
        or candidates.shape[0] != len(SEED_CELL_GRID)
        or candidates.shape[1] != len(CENTERS)
        or baseline.shape[0] != len(SEED_CELL_GRID)
        or candidates.shape[2] != baseline.shape[1]
        or not all(array.flags.writeable is False for array in (candidates, baseline))
    ):
        raise ProtocolError("SCEPTRE v3 prediction store geometry or mutability drifted.")
    ordinals = np.asarray(scoped.row_ordinals, dtype=np.int64)
    legal_sources = legal_routing_sources(scoped.target_center)
    legal_ordinals = tuple(source_order.index(source) for source in legal_sources)
    target_ordinal = source_order.index(scoped.target_center)
    legal_slice = candidates[:, legal_ordinals, :][:, :, ordinals]
    excluded_slice = candidates[:, target_ordinal, ordinals]
    if (
        np.any(ordinals < 0)
        or np.any(ordinals >= candidates.shape[2])
        or not np.isfinite(legal_slice).all()
        or not np.isfinite(baseline[:, ordinals]).all()
        or np.any((legal_slice < 0.0) | (legal_slice > 1.0))
        or not np.all(excluded_slice == CANDIDATE_EXCLUSION_SENTINEL)
        or np.any((baseline[:, ordinals] < 0.0) | (baseline[:, ordinals] > 1.0))
    ):
        raise ProtocolError("SCEPTRE v3 prediction role slice is invalid.")

    action_probabilities: dict[
        str, dict[tuple[int, int], tuple[float, ...]]
    ] = {}
    for source in legal_sources:
        source_ordinal = source_order.index(source)
        action_probabilities[source] = {
            seed: tuple(
                float(value)
                for value in candidates[seed_ordinal, source_ordinal, ordinals]
            )
            for seed_ordinal, seed in enumerate(SEED_CELL_GRID)
        }
    action_probabilities[EXACT_B_CANDIDATE] = {
        seed: tuple(float(value) for value in baseline[seed_ordinal, ordinals])
        for seed_ordinal, seed in enumerate(SEED_CELL_GRID)
    }
    surface = build_role_prediction_surface(
        target_center=scoped.target_center,
        fold=fold,
        partition_hash=partition_hash,
        role=scoped.role,
        observation_ids=scoped.observation_ids,
        case_ids=scoped.case_ids,
        labels=scoped.labels,
        probabilities_by_action_and_seed=action_probabilities,
        candidate_menu_hash=candidate_menu_hash,
        exact_b_control_receipt_hash=exact_b_control_receipt_hash,
        prediction_bundle_sha256=store_hash,
        phase_capability=phase_capability,
    )
    if scoped.role == "EVALUATION":
        body = _evidence_body(surface, (), None, store_hash)
        return RoleEvidenceBundle(surface, (), None, canonical_hash(body))

    outcomes = tuple(
        _family_outcome(
            surface,
            action_id=source,
            prediction_store_hash=store_hash,
        )
        for source in legal_routing_sources(scoped.target_center)
    )
    exact_b = _family_outcome(
        surface,
        action_id=EXACT_B_CANDIDATE,
        prediction_store_hash=store_hash,
    )
    body = _evidence_body(surface, outcomes, exact_b, store_hash)
    return RoleEvidenceBundle(
        surface, outcomes, exact_b, canonical_hash(body)
    )


def _family_outcome(
    surface: RolePredictionSurface,
    *,
    action_id: str,
    prediction_store_hash: str,
) -> FamilyOutcome:
    action = next(row for row in surface.actions if row.action_id == action_id)
    labels = np.asarray(surface.labels, dtype=np.int8)
    probabilities = np.asarray(
        [cell.probabilities for cell in action.seed_cells], dtype=np.float64
    )
    predicted = probabilities >= 0.5
    truth = labels.astype(bool)[None, :]
    confusion = ConfusionCounts(
        tn=int(np.sum((~truth) & (~predicted))),
        fp=int(np.sum((~truth) & predicted)),
        fn=int(np.sum(truth & (~predicted))),
        tp=int(np.sum(truth & predicted)),
    )
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    target = labels.astype(np.float64)[None, :]
    brier_sum = float(np.sum(np.square(probabilities - target), dtype=np.float64))
    log_loss_sum = float(
        np.sum(
            -(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)),
            dtype=np.float64,
        )
    )
    if not math.isfinite(brier_sum) or not math.isfinite(log_loss_sum):
        raise ProtocolError("SCEPTRE v3 proper-loss outcome is non-finite.")
    receipt = canonical_hash(
        {
            "schema_version": "sceptre_v3_scoped_action_prediction_receipt_v1",
            "prediction_store_hash": prediction_store_hash,
            "role_prediction_surface_hash": surface.surface_hash,
            "action_id": action_id,
            "action_hash": action.action_hash,
            "role_case_set_hash": surface.fold.case_set_hash(surface.role),
            "seed_cell_count": len(SEED_CELL_GRID),
        }
    )
    return FamilyOutcome(
        target_center=surface.target_center,
        fold_ordinal=surface.fold.fold_ordinal,
        role=surface.role,
        candidate_center=action_id,
        partition_hash=surface.partition_hash,
        case_set_hash=surface.fold.case_set_hash(surface.role),
        candidate_menu_hash=surface.candidate_menu_hash,
        prediction_receipt_hash=receipt,
        confusion=confusion,
        brier_sum=brier_sum,
        log_loss_sum=log_loss_sum,
        case_count=len(surface.whole_case_ids),
        exact_b_control_receipt_hash=(
            surface.exact_b_control_receipt_hash
            if action_id == EXACT_B_CANDIDATE
            else None
        ),
    )


def _evidence_body(
    surface: RolePredictionSurface,
    outcomes: Sequence[FamilyOutcome],
    exact_b: FamilyOutcome | None,
    store_hash: str,
) -> Mapping[str, object]:
    return {
        "schema_version": "sceptre_v3_role_evidence_bundle_v1",
        "prediction_store_hash": store_hash,
        "surface_hash": surface.surface_hash,
        "target_center": surface.target_center,
        "fold_ordinal": surface.fold.fold_ordinal,
        "role": surface.role,
        "outcome_hashes": [row.outcome_hash for row in outcomes],
        "exact_b_outcome_hash": (
            None if exact_b is None else exact_b.outcome_hash
        ),
        "raw_labels_persisted": False,
    }


__all__ = ("RoleEvidenceBundle", "build_role_evidence")
