"""Label-free construction and post-seal pseudo response opening.

This module is the only bridge between package-local B/I/R/P probability
endpoints and the fitted action-response layer.  It deliberately separates
three operations:

1. construct route-local descriptors without held-case labels;
2. seal the complete action inventory; and
3. open pseudo-case responses against the already-sealed vectors.

The probability arrays are in-memory runtime values.  Persisted payloads expose
only their canonical SHA-256 identities and aggregate response statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import BankViability, RouteKey
from ..endpoint_runtime import EndpointPrediction
from ..identity import ACTION_FAMILIES, DIRECTIONS, canonical_hash, require_sha256
from ..label_firewall import PseudoResponseLabelCapability
from ..probabilities import directional_action, expected_favorable_utility
from ..target_local_runtime import CasePosteriorPrediction, POSTERIOR_CONTROL_IDS
from .contracts import ActionKey, ActionPrediction, ActionResponse
from .responses import (
    build_action_response,
    canonical_probabilities,
    probability_sha256,
)


def _complete_action_surface_seal_hash(
    *,
    route_draft_hashes: Sequence[str],
    action_count: int,
    outer_centers: Sequence[str],
    physical_surface_hash: str,
    expected_inventory_hash: str | None,
    posterior_control_id: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "pdcaps_complete_action_surface_seal_v4",
            "route_draft_hashes": tuple(route_draft_hashes),
            "route_count": len(route_draft_hashes),
            "action_count": int(action_count),
            "outer_centers": tuple(outer_centers),
            "physical_surface_hash": physical_surface_hash,
            "expected_inventory_hash": expected_inventory_hash,
            "posterior_control_id": posterior_control_id,
            "all_routes_sealed_before_pseudo_response_access": True,
            "labels_used": False,
        }
    )


@dataclass(frozen=True)
class ActionDraft:
    """One non-empty P-to-endpoint crossing before the global action seal."""

    route_key: RouteKey
    family: str
    direction: str
    action_id: str
    action_probabilities: np.ndarray
    predicted_utility: object
    crossing_fraction: float
    bank_viability: BankViability
    endpoint_hash: str
    posterior_prediction_hash: str
    draft_hash: str = field(init=False)

    def __post_init__(self) -> None:
        from ..contracts import FavorableUtility

        probabilities = canonical_probabilities(self.action_probabilities)
        family = str(self.family)
        direction = str(self.direction)
        action_id = str(self.action_id)
        crossing = float(self.crossing_fraction)
        if (
            family not in ACTION_FAMILIES
            or direction not in DIRECTIONS
            or action_id != f"{family}::{direction}"
            or not isinstance(self.predicted_utility, FavorableUtility)
            or not np.isfinite(crossing)
            or crossing <= 0.0
            or crossing > 1.0
            or len(str(self.endpoint_hash)) != 64
            or len(str(self.posterior_prediction_hash)) != 64
        ):
            raise ProtocolError("P-DCAPS action draft identity drifted.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_probabilities", probabilities)
        object.__setattr__(self, "crossing_fraction", crossing)
        object.__setattr__(
            self,
            "draft_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_action_draft_v1",
                    "route_key": self.route_key.to_payload(),
                    "family": family,
                    "direction": direction,
                    "action_id": action_id,
                    "action_probability_hash": probability_sha256(probabilities),
                    "predicted_utility": self.predicted_utility.to_payload(),
                    "crossing_fraction": crossing,
                    "bank_viability": self.bank_viability.to_payload(),
                    "endpoint_hash": self.endpoint_hash,
                    "posterior_prediction_hash": self.posterior_prediction_hash,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class RouteActionDraftSurface:
    """A route remains in the seal even when it has no hard crossings."""

    route_key: RouteKey
    sample_ids: tuple[str, ...]
    baseline_probabilities: np.ndarray
    drafts: tuple[ActionDraft, ...]
    endpoint_hash: str
    posterior_prediction_hash: str
    physical_surface_hash: str
    posterior_control_id: str
    route_draft_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        baseline = canonical_probabilities(
            self.baseline_probabilities, expected_length=len(samples)
        )
        drafts = tuple(
            sorted(self.drafts, key=lambda row: (row.family, row.direction))
        )
        endpoint_hash = require_sha256(self.endpoint_hash, "endpoint hash")
        posterior_hash = require_sha256(
            self.posterior_prediction_hash, "posterior prediction hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "physical surface hash"
        )
        control_id = str(self.posterior_control_id)
        if (
            not samples
            or len(samples) != len(set(samples))
            or len({row.draft_hash for row in drafts}) != len(drafts)
            or len({(row.family, row.direction) for row in drafts}) != len(drafts)
            or any(row.route_key != self.route_key for row in drafts)
            or any(len(row.action_probabilities) != len(samples) for row in drafts)
            or any(row.endpoint_hash != endpoint_hash for row in drafts)
            or any(row.posterior_prediction_hash != posterior_hash for row in drafts)
            or control_id not in POSTERIOR_CONTROL_IDS
        ):
            raise ProtocolError("P-DCAPS route action draft surface drifted.")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "baseline_probabilities", baseline)
        object.__setattr__(self, "drafts", drafts)
        object.__setattr__(self, "endpoint_hash", endpoint_hash)
        object.__setattr__(self, "posterior_prediction_hash", posterior_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "posterior_control_id", control_id)
        object.__setattr__(
            self,
            "route_draft_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_route_action_draft_surface_v3",
                    "route_key": self.route_key.to_payload(),
                    "sample_ids": samples,
                    "baseline_probability_hash": probability_sha256(baseline),
                    "action_draft_hashes": tuple(row.draft_hash for row in drafts),
                    "endpoint_hash": endpoint_hash,
                    "posterior_prediction_hash": posterior_hash,
                    "physical_surface_hash": physical_hash,
                    "posterior_control_id": control_id,
                    "complete_nonempty_crossing_inventory": True,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class SealedActionCell:
    prediction: ActionPrediction
    action_probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = canonical_probabilities(self.action_probabilities)
        if probability_sha256(values) != self.prediction.key.probability_hash:
            raise ProtocolError("P-DCAPS sealed action probability lineage drifted.")
        object.__setattr__(self, "action_probabilities", values)


@dataclass(frozen=True)
class SealedRouteActionSurface:
    route_key: RouteKey
    sample_ids: tuple[str, ...]
    baseline_probabilities: np.ndarray
    cells: tuple[SealedActionCell, ...]
    route_draft_hash: str
    action_surface_seal_hash: str
    physical_surface_hash: str
    posterior_control_id: str
    posterior_prediction_hash: str

    def __post_init__(self) -> None:
        baseline = canonical_probabilities(
            self.baseline_probabilities, expected_length=len(self.sample_ids)
        )
        cells = tuple(self.cells)
        seal_hash = require_sha256(
            self.action_surface_seal_hash, "action-surface seal hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "physical surface hash"
        )
        route_draft_hash = require_sha256(
            self.route_draft_hash, "route draft hash"
        )
        control_id = str(self.posterior_control_id)
        posterior_hash = require_sha256(
            self.posterior_prediction_hash, "posterior prediction hash"
        )
        if (
            len({row.prediction.prediction_hash for row in cells}) != len(cells)
            or any(row.prediction.key.route_key != self.route_key for row in cells)
            or any(
                row.prediction.key.action_surface_seal_hash
                != seal_hash
                for row in cells
            )
            or control_id not in POSTERIOR_CONTROL_IDS
        ):
            raise ProtocolError("P-DCAPS sealed route action surface drifted.")
        object.__setattr__(self, "baseline_probabilities", baseline)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "action_surface_seal_hash", seal_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "route_draft_hash", route_draft_hash)
        object.__setattr__(self, "posterior_control_id", control_id)
        object.__setattr__(self, "posterior_prediction_hash", posterior_hash)

    @property
    def predictions(self) -> tuple[ActionPrediction, ...]:
        return tuple(row.prediction for row in self.cells)

    def probability_for_action_key(self, action_key_hash: str) -> np.ndarray:
        for cell in self.cells:
            if cell.prediction.key.action_key_hash == str(action_key_hash):
                return cell.action_probabilities
        raise ProtocolError("P-DCAPS sealed action probability is absent.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_sealed_route_action_surface_v3",
            "route_key": self.route_key.to_payload(),
            "sample_ids": list(self.sample_ids),
            "baseline_probability_hash": probability_sha256(
                self.baseline_probabilities
            ),
            "predictions": [row.prediction.to_payload() for row in self.cells],
            "route_draft_hash": self.route_draft_hash,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_control_id": self.posterior_control_id,
            "posterior_prediction_hash": self.posterior_prediction_hash,
            "raw_probability_arrays_persisted": False,
            "raw_labels_persisted": False,
        }


@dataclass(frozen=True)
class SealedActionSurface:
    routes: tuple[SealedRouteActionSurface, ...]
    action_surface_seal_hash: str
    physical_surface_hash: str
    posterior_control_id: str
    expected_inventory_hash: str | None = None

    def __post_init__(self) -> None:
        routes = tuple(sorted(self.routes, key=lambda row: row.route_key))
        seal_hash = require_sha256(
            self.action_surface_seal_hash, "action-surface seal hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "physical surface hash"
        )
        inventory_hash = (
            None
            if self.expected_inventory_hash is None
            else require_sha256(
                self.expected_inventory_hash, "expected route inventory hash"
            )
        )
        control_id = str(self.posterior_control_id)
        observed_outer = tuple(
            center
            for center in CENTERS
            if center in {row.route_key.outer_center for row in routes}
        )
        expected_seal_hash = _complete_action_surface_seal_hash(
            route_draft_hashes=tuple(row.route_draft_hash for row in routes),
            action_count=sum(len(row.cells) for row in routes),
            outer_centers=observed_outer,
            physical_surface_hash=physical_hash,
            expected_inventory_hash=inventory_hash,
            posterior_control_id=control_id,
        )
        if (
            not routes
            or len({row.route_key for row in routes}) != len(routes)
            or any(
                row.action_surface_seal_hash != seal_hash
                for row in routes
            )
            or any(row.physical_surface_hash != physical_hash for row in routes)
            or control_id not in POSTERIOR_CONTROL_IDS
            or any(row.posterior_control_id != control_id for row in routes)
            or seal_hash != expected_seal_hash
        ):
            raise ProtocolError("P-DCAPS sealed global action surface drifted.")
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "action_surface_seal_hash", seal_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "posterior_control_id", control_id)
        object.__setattr__(self, "expected_inventory_hash", inventory_hash)

    @property
    def predictions(self) -> tuple[ActionPrediction, ...]:
        return tuple(
            prediction for route in self.routes for prediction in route.predictions
        )

    def route(self, route_key: RouteKey) -> SealedRouteActionSurface:
        for row in self.routes:
            if row.route_key == route_key:
                return row
        raise ProtocolError("P-DCAPS sealed route is absent.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_sealed_action_surface_v4",
            "routes": [row.to_payload() for row in self.routes],
            "route_count": len(self.routes),
            "action_count": len(self.predictions),
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_control_id": self.posterior_control_id,
            "expected_inventory_hash": self.expected_inventory_hash,
            "raw_probability_arrays_persisted": False,
            "raw_labels_persisted": False,
        }


@dataclass(frozen=True)
class ResponseDenominators:
    positive: int
    negative: int

    def __post_init__(self) -> None:
        if int(self.positive) <= 0 or int(self.negative) <= 0:
            raise ProtocolError("P-DCAPS response denominators lack both classes.")
        object.__setattr__(self, "positive", int(self.positive))
        object.__setattr__(self, "negative", int(self.negative))

    @property
    def total(self) -> int:
        return self.positive + self.negative


def build_route_action_draft_surface(
    endpoint: EndpointPrediction,
    posterior: CasePosteriorPrediction,
    route_key: RouteKey,
    *,
    support_n_positive: int,
    support_n_negative: int,
    bank_viability_by_family: Mapping[str, BankViability],
) -> RouteActionDraftSurface:
    """Construct all and only non-empty directional crossings for one case."""

    if (
        endpoint.center != route_key.route_center
        or endpoint.case_id != route_key.held_case_id
        or posterior.target_center != route_key.route_center
        or posterior.held_case_id != route_key.held_case_id
        or endpoint.sample_ids != posterior.sample_ids
        or endpoint.excluded_source_centers
        != (() if route_key.surface_role == "target" else (route_key.outer_center,))
        or set(bank_viability_by_family) != set(ACTION_FAMILIES)
        or int(support_n_positive) <= 0
        or int(support_n_negative) <= 0
    ):
        raise ProtocolError("P-DCAPS route action construction lineage drifted.")
    support_total = int(support_n_positive) + int(support_n_negative)
    baseline = endpoint.probability("P_PROTECTED")
    drafts: list[ActionDraft] = []
    for family in ACTION_FAMILIES:
        endpoint_probability = endpoint.probability(family)
        for direction in DIRECTIONS:
            action, crossing = directional_action(
                baseline, endpoint_probability, direction
            )
            crossing_count = int(np.sum(crossing, dtype=np.int64))
            if crossing_count == 0:
                continue
            drafts.append(
                ActionDraft(
                    route_key,
                    family,
                    direction,
                    f"{family}::{direction}",
                    action,
                    expected_favorable_utility(
                        baseline,
                        action,
                        posterior.natural_probabilities,
                        support_n_positive=int(support_n_positive),
                        support_n_negative=int(support_n_negative),
                        support_row_count=support_total,
                        crossing_mask=crossing,
                    ),
                    crossing_count / len(crossing),
                    bank_viability_by_family[family],
                    endpoint.endpoint_hash,
                    posterior.prediction_hash,
                )
            )
    return RouteActionDraftSurface(
        route_key,
        endpoint.sample_ids,
        baseline,
        tuple(drafts),
        endpoint.endpoint_hash,
        posterior.prediction_hash,
        endpoint.physical_surface_hash,
        posterior.control_id,
    )


def seal_action_surface(
    routes: Sequence[RouteActionDraftSurface],
    *,
    expected_outer_centers: Sequence[str] | None = CENTERS,
    expected_inventory_hash: str | None = None,
) -> SealedActionSurface:
    """Bind one global seal into every action key before labels can open."""

    rows = tuple(sorted(tuple(routes), key=lambda row: row.route_key))
    inventory_hash = (
        None
        if expected_inventory_hash is None
        else require_sha256(
            expected_inventory_hash, "expected route inventory hash"
        )
    )
    expected = (
        None
        if expected_outer_centers is None
        else tuple(str(value) for value in expected_outer_centers)
    )
    observed_outer = tuple(
        center
        for center in CENTERS
        if center in {row.route_key.outer_center for row in rows}
    )
    if (
        not rows
        or len({row.route_key for row in rows}) != len(rows)
        or len({row.route_draft_hash for row in rows}) != len(rows)
        or len({row.physical_surface_hash for row in rows}) != 1
        or len({row.posterior_control_id for row in rows}) != 1
        or (expected is not None and observed_outer != expected)
    ):
        raise ProtocolError("P-DCAPS complete action inventory drifted before sealing.")
    seal_hash = _complete_action_surface_seal_hash(
        route_draft_hashes=tuple(row.route_draft_hash for row in rows),
        action_count=sum(len(row.drafts) for row in rows),
        outer_centers=observed_outer,
        physical_surface_hash=rows[0].physical_surface_hash,
        expected_inventory_hash=inventory_hash,
        posterior_control_id=rows[0].posterior_control_id,
    )
    sealed_routes: list[SealedRouteActionSurface] = []
    for route in rows:
        cells = tuple(
            SealedActionCell(
                ActionPrediction(
                    ActionKey(
                        draft.route_key,
                        draft.family,
                        draft.direction,
                        draft.action_id,
                        probability_sha256(draft.action_probabilities),
                        seal_hash,
                    ),
                    draft.predicted_utility,
                    draft.crossing_fraction,
                    draft.bank_viability,
                ),
                draft.action_probabilities,
            )
            for draft in route.drafts
        )
        sealed_routes.append(
            SealedRouteActionSurface(
                route.route_key,
                route.sample_ids,
                route.baseline_probabilities,
                cells,
                route.route_draft_hash,
                seal_hash,
                route.physical_surface_hash,
                route.posterior_control_id,
                route.posterior_prediction_hash,
            )
        )
    return SealedActionSurface(
        tuple(sealed_routes),
        seal_hash,
        rows[0].physical_surface_hash,
        rows[0].posterior_control_id,
        inventory_hash,
    )


def open_pseudo_route_action_responses(
    route: SealedRouteActionSurface,
    *,
    label_capability: PseudoResponseLabelCapability,
    denominators: ResponseDenominators,
) -> tuple[ActionResponse, ...]:
    """Score one pseudo held case after the global action seal exists."""

    if (
        route.route_key.surface_role != "pseudo"
        or label_capability.route_key != route.route_key
        or tuple(row.sample_id for row in label_capability.rows) != route.sample_ids
    ):
        raise ProtocolError("P-DCAPS target action responses cannot open preterminally.")
    truth = label_capability.values
    if len(truth) != len(route.sample_ids):
        raise ProtocolError("P-DCAPS pseudo response row count drifted.")
    return tuple(
        build_action_response(
            cell.prediction,
            baseline_probabilities=route.baseline_probabilities,
            action_probabilities=cell.action_probabilities,
            label_capability=label_capability,
            positive_denominator=denominators.positive,
            negative_denominator=denominators.negative,
            row_denominator=denominators.total,
        )
        for cell in route.cells
    )


__all__ = (
    "ActionDraft",
    "ResponseDenominators",
    "RouteActionDraftSurface",
    "SealedActionCell",
    "SealedActionSurface",
    "SealedRouteActionSurface",
    "build_route_action_draft_surface",
    "open_pseudo_route_action_responses",
    "seal_action_surface",
)
