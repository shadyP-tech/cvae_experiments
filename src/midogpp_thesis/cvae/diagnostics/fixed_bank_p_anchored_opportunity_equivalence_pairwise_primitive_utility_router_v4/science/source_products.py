"""Deterministic derivation of every scientific product from typed source input."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    ActionSurface,
    ActionUtilityObservation,
    NormalizedUtility,
    OpportunityCaseReceipt,
    RowPosteriorOOFPrediction,
    RowPosteriorObservation,
    RowPosteriorPrediction,
    build_expected_denominators,
    build_opportunity_set,
    canonical_sha256,
    expected_additive_utility,
    normalize_expected_utility,
)
from ....routing.pairwise_primitive_utility.opportunity import (
    build_opportunity_case_receipt,
)
from ..candidate_pools import ALL_ACTION_IDS, CANDIDATE_ACTION_IDS
from ..feature_engineering import (
    build_action_query,
    build_case_action_feature,
    build_row_posterior_observations,
)
from ..folds import OuterFoldPlanV4
from ..source_supervision import SourceSupervisionRow, SourceTrainingSurface
from .pool_indexed_pairwise_fit import HeldLActionQuery


@dataclass(frozen=True, slots=True)
class RealizedSourceUtility:
    action_id: str
    bacc_gain: float
    brier_loss_delta: float
    log_loss_delta: float
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        numeric = tuple(float(value) for value in (self.bacc_gain, self.brier_loss_delta, self.log_loss_delta))
        if self.action_id not in CANDIDATE_ACTION_IDS or not all(math.isfinite(value) for value in numeric):
            raise ProtocolError("OE-PPUR v4 realized source utility drifted.")
        object.__setattr__(self, "bacc_gain", numeric[0])
        object.__setattr__(self, "brier_loss_delta", numeric[1])
        object.__setattr__(self, "log_loss_delta", numeric[2])
        object.__setattr__(self, "receipt_hash", canonical_sha256({
            "schema": "oe_ppur_v4_ephemeral_realized_source_utility_v1",
            "action_id": self.action_id,
            "metrics": numeric,
            "raw_labels_persisted": False,
            "target_labels_used": False,
        }))


@dataclass(frozen=True, slots=True)
class SourceCaseProducts:
    center_id: str
    case_id: str
    candidate_pool_receipt_hash: str
    opportunity_receipt: OpportunityCaseReceipt
    utilities: tuple[NormalizedUtility, ...]
    realized: tuple[RealizedSourceUtility, ...]
    held_l_queries: tuple[HeldLActionQuery, ...]
    case_hash: str = field(init=False)

    def __post_init__(self) -> None:
        utilities = tuple(sorted(self.utilities, key=lambda row: row.action_id))
        realized = tuple(sorted(self.realized, key=lambda row: row.action_id))
        queries = tuple(sorted(self.held_l_queries, key=lambda row: row.query.action_id))
        if (
            not isinstance(self.opportunity_receipt, OpportunityCaseReceipt)
            or (self.opportunity_receipt.center_id, self.opportunity_receipt.case_id) != (self.center_id, self.case_id)
            or tuple(row.action_id for row in utilities) != CANDIDATE_ACTION_IDS
            or tuple(row.action_id for row in realized) != CANDIDATE_ACTION_IDS
            or tuple(row.query.action_id for row in queries) != self.opportunity_receipt.active_representative_ids
            or any((row.center_id, row.case_id) != (self.center_id, self.case_id) for row in queries)
        ):
            raise ProtocolError("OE-PPUR v4 derived source case inventory drifted.")
        object.__setattr__(self, "utilities", utilities)
        object.__setattr__(self, "realized", realized)
        object.__setattr__(self, "held_l_queries", queries)
        object.__setattr__(self, "case_hash", canonical_sha256({
            "schema": "oe_ppur_v4_derived_source_case_products_v1",
            "center_id": self.center_id,
            "case_id": self.case_id,
            "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
            "opportunity_receipt_hash": self.opportunity_receipt.receipt_hash,
            "utility_hashes": tuple(row.response_hash for row in utilities),
            "realized_receipt_hashes": tuple(row.receipt_hash for row in realized),
            "query_actions": tuple(row.query.action_id for row in queries),
            "target_labels_used": False,
        }))

    def utility(self, action_id: object) -> NormalizedUtility:
        key = str(action_id)
        for row in self.utilities:
            if row.action_id == key:
                return row
        raise ProtocolError(f"OE-PPUR v4 source case lacks utility {key}.")

    def realized_utility(self, action_id: object) -> RealizedSourceUtility:
        key = str(action_id)
        for row in self.realized:
            if row.action_id == key:
                return row
        raise ProtocolError(f"OE-PPUR v4 source case lacks realized utility {key}.")


@dataclass(frozen=True, slots=True)
class DerivedSourceScienceProducts:
    row_observations: tuple[RowPosteriorObservation, ...]
    action_observations: tuple[ActionUtilityObservation, ...]
    opportunity_receipts: tuple[OpportunityCaseReceipt, ...]
    cases: tuple[SourceCaseProducts, ...]
    held_l_queries: tuple[HeldLActionQuery, ...]
    source_surface_lineage_hash: str
    products_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(self.cases, key=lambda row: (row.center_id, row.case_id)))
        if (
            not self.row_observations
            or not self.action_observations
            or not cases
            or tuple(self.opportunity_receipts) != tuple(row.opportunity_receipt for row in cases)
            or tuple(self.held_l_queries) != tuple(query for row in cases for query in row.held_l_queries)
        ):
            raise ProtocolError("OE-PPUR v4 internally derived source products drifted.")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "products_hash", canonical_sha256({
            "schema": "oe_ppur_v4_internally_derived_source_products_v1",
            "source_surface_lineage_hash": self.source_surface_lineage_hash,
            "case_hashes": tuple(row.case_hash for row in cases),
            "row_keys": tuple((row.center_id, row.case_id, row.row_id) for row in self.row_observations),
            "action_keys": tuple((row.center_id, row.case_id, row.action_id) for row in self.action_observations),
            "caller_injected_products": False,
            "target_labels_used": False,
        }))


def _realized_utility(
    labels: np.ndarray,
    protected: np.ndarray,
    candidate: np.ndarray,
    *,
    action_id: str,
) -> RealizedSourceUtility:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives <= 0 or negatives <= 0:
        raise ProtocolError("OE-PPUR v4 source case lacks both classes for BACC ordering.")
    protected_hard, candidate_hard = protected >= 0.5, candidate >= 0.5
    delta_tp = int(np.sum(candidate_hard & (labels == 1))) - int(np.sum(protected_hard & (labels == 1)))
    delta_tn = int(np.sum(~candidate_hard & (labels == 0))) - int(np.sum(~protected_hard & (labels == 0)))
    bacc = 0.5 * (delta_tp / positives + delta_tn / negatives)
    brier = float(np.mean((candidate - labels) ** 2 - (protected - labels) ** 2, dtype=np.float64))
    epsilon = 1.0e-7
    cp, pp = np.clip(candidate, epsilon, 1.0 - epsilon), np.clip(protected, epsilon, 1.0 - epsilon)
    log_delta = float(np.mean(-labels * np.log(cp / pp) - (1.0 - labels) * np.log((1.0 - cp) / (1.0 - pp)), dtype=np.float64))
    return RealizedSourceUtility(action_id, bacc, brier, log_delta)


def derive_source_science_products(
    surface: SourceTrainingSurface,
    plan: OuterFoldPlanV4,
    *,
    row_oof_predictions: Sequence[RowPosteriorOOFPrediction],
) -> DerivedSourceScienceProducts:
    """Derive action values, opportunities and outcomes without injection seams."""

    if not isinstance(surface, SourceTrainingSurface) or not isinstance(plan, OuterFoldPlanV4):
        raise ProtocolError("OE-PPUR v4 source-product derivation requires typed inputs.")
    if plan.source_supervision_contract_hash != surface.receipt.contract.contract_hash:
        raise ProtocolError("OE-PPUR v4 source products mixed plan/surface contracts.")
    outer_rows = surface.rows_for_outer(plan.outer_target_center)
    row_observations = build_row_posterior_observations(outer_rows)
    oof_by_key = {(row.center_id, row.case_id, row.row_id): row for row in row_oof_predictions}
    expected_keys = {(row.query_center, row.case_id, row.source_row_id) for row in outer_rows}
    if set(oof_by_key) != expected_keys:
        raise ProtocolError("OE-PPUR v4 row OOF predictions do not exactly cover source rows.")
    grouped: dict[tuple[str, str], list[SourceSupervisionRow]] = {}
    for row in outer_rows:
        grouped.setdefault((row.query_center, row.case_id), []).append(row)
    if set(grouped) != set(plan.source_case_inventory):
        raise ProtocolError("OE-PPUR v4 source products omitted or invented J/d cases.")
    cases: list[SourceCaseProducts] = []
    observations: list[ActionUtilityObservation] = []
    for (center, case_id), raw_rows in sorted(grouped.items()):
        rows = tuple(sorted(raw_rows, key=lambda row: row.source_cache_row_index))
        pool = plan.held_pool(center)
        if {row.candidate_pool_receipt_hash for row in rows} != {pool.receipt_hash}:
            raise ProtocolError("OE-PPUR v4 source case mixed held-q candidate pools.")
        protected = np.asarray([row.action_probabilities[0] for row in rows], dtype=np.float64)
        actions = tuple(
            ActionSurface(
                action_id=action_id,
                family=action_id.split("::", 1)[0],
                direction=action_id.split("::", 1)[1],
                probabilities=tuple(row.action_probabilities[ALL_ACTION_IDS.index(action_id)] for row in rows),
            )
            for action_id in CANDIDATE_ACTION_IDS
        )
        opportunity = build_opportunity_set(protected, actions, candidate_action_ids=CANDIDATE_ACTION_IDS)
        opportunity_receipt = build_opportunity_case_receipt(center_id=center, case_id=case_id, opportunity=opportunity)
        eta_oof = tuple(oof_by_key[(center, case_id, row.source_row_id)] for row in rows)
        if len({(row.model_hash, row.source_scope_receipt_hash) for row in eta_oof}) != 1:
            raise ProtocolError("OE-PPUR v4 source case mixed row-posterior OOF fits.")
        eta = tuple(RowPosteriorPrediction(row.eta, row.model_hash, row.source_scope_receipt_hash) for row in eta_oof)
        row_manifest_hash = canonical_sha256(tuple(row.source_row_id for row in rows))
        scope_id = canonical_sha256({"schema": "oe_ppur_v4_source_case_scope_v1", "H": plan.outer_target_center, "q": center, "case_id": case_id, "row_manifest_hash": row_manifest_hash})
        denominators = build_expected_denominators(eta, scope_id=scope_id, row_manifest_hash=row_manifest_hash)
        utilities: list[NormalizedUtility] = []
        realized: list[RealizedSourceUtility] = []
        labels = np.asarray([row.outcome for row in rows], dtype=np.float64)
        held_queries: list[HeldLActionQuery] = []
        for action in actions:
            primitive = expected_additive_utility(protected, action.probabilities, eta, action_id=action.action_id, scope_id=scope_id, row_manifest_hash=row_manifest_hash)
            utility = normalize_expected_utility(primitive, denominators)
            utilities.append(utility)
            candidate = np.asarray(action.probabilities, dtype=np.float64)
            realized.append(_realized_utility(labels, protected, candidate, action_id=action.action_id))
            query = build_action_query(rows, action_id=action.action_id)
            if action.action_id in opportunity.active_representative_ids:
                held_queries.append(HeldLActionQuery(center, case_id, query))
                feature = build_case_action_feature(rows, action_id=action.action_id)
                observations.append(ActionUtilityObservation(
                    center_id=center,
                    case_id=case_id,
                    action_id=action.action_id,
                    family=query.family,
                    direction=query.direction,
                    feature_names=feature.names,
                    feature_values=feature.values,
                    response=utility,
                    source_scope_receipt_hash=surface.surface_hash,
                    candidate_pool_receipt_hash=pool.receipt_hash,
                    opportunity_case_receipt_hash=opportunity_receipt.receipt_hash,
                ))
        cases.append(SourceCaseProducts(center, case_id, pool.receipt_hash, opportunity_receipt, tuple(utilities), tuple(realized), tuple(held_queries)))
    case_rows = tuple(sorted(cases, key=lambda row: (row.center_id, row.case_id)))
    return DerivedSourceScienceProducts(
        row_observations=row_observations,
        action_observations=tuple(sorted(observations, key=lambda row: (row.center_id, row.case_id, row.action_id))),
        opportunity_receipts=tuple(row.opportunity_receipt for row in case_rows),
        cases=case_rows,
        held_l_queries=tuple(query for row in case_rows for query in row.held_l_queries),
        source_surface_lineage_hash=surface.surface_hash,
    )


__all__ = (
    "DerivedSourceScienceProducts",
    "RealizedSourceUtility",
    "SourceCaseProducts",
    "derive_source_science_products",
)
