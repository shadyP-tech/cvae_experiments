"""Capability-scoped helpers shared by donor and route phases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from ..hashing import canonical_hash
from ..identity import CENTERS, GovernanceError
from ..label_capabilities import LabelCapability
from ..manifest_labels import ScopedCaseLabels
from ..physical.planning import (
    DonorDirectionalPriorSurface,
    compute_donor_directional_priors,
)
from ..physical_memmaps import MappedPhysicalStore
from ..posterior.contracts import DonorFitScope, DonorObservation
from ..posterior.donor import DonorActionModel
from ..posterior.empirical_bayes import ActionEstimate
from ..posterior.local import LocalResidualModel
from ..posterior.pipeline import estimate_action_rectangle
from ..posterior.uncertainty import build_preargmax_bounds
from ..routing.selection import RouteDecision, select_action
from ..utility.actions import ActionRectangle
from ..utility.metrics import ScoredActionRectangle
from .contracts import ScienceSettings


def labels_for_cases(
    labels: ScopedCaseLabels,
    center: str,
    case_ids: Sequence[str],
) -> Mapping[str, np.ndarray]:
    return MappingProxyType(
        {
            str(case_id): labels.labels_for_case(center, case_id)
            for case_id in case_ids
        }
    )


def subset_scoped_labels(
    labels: ScopedCaseLabels,
    *,
    center: str,
    case_ids: Sequence[str],
    identity_role: str,
) -> ScopedCaseLabels:
    """Create an ephemeral, capability-bound case subset without persistence."""

    cases = tuple(str(case_id) for case_id in case_ids)
    selected = labels_for_cases(labels, center, cases)
    return ScopedCaseLabels(
        labels.kind,
        labels.scope_hash,
        MappingProxyType({str(center): selected}),
        sum(len(values) for values in selected.values()),
        len(selected),
        canonical_hash(
            {
                "schema_version": "scale_bp_v2_ephemeral_scoped_label_subset_v1",
                "parent_identity_hash": labels.identity_hash,
                "identity_role": str(identity_role),
                "center": str(center),
                "case_ids": cases,
                "persisted": False,
            }
        ),
    )


def planning_scope_hash(
    role: str,
    capability: LabelCapability,
    *,
    target_center: str,
    case_id: str,
    source_exclusions: Sequence[str],
    support_exclusions: Sequence[str],
    parent_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "scale_bp_v2_worker_planning_scope_v1",
            "role": str(role),
            "capability_scope_hash": capability.scope_hash,
            "target_center": str(target_center),
            "case_id": str(case_id),
            "source_excluded_centers": tuple(source_exclusions),
            "support_excluded_case_ids": tuple(support_exclusions),
            "parent_hash": str(parent_hash),
        }
    )


def canonical_centers(values: set[str]) -> tuple[str, ...]:
    output = tuple(center for center in CENTERS if center in values)
    if set(output) != set(values):
        raise GovernanceError("SCALE-BP v2 source exclusion center is unknown.")
    return output


def cached_prior(
    cache: dict[tuple[str, tuple[str, ...]], DonorDirectionalPriorSurface],
    store: MappedPhysicalStore,
    labels: ScopedCaseLabels,
    *,
    target_center: str,
    source_excluded_centers: tuple[str, ...],
    donor_scope_hash: str,
) -> DonorDirectionalPriorSurface:
    """Cache an exact exclusion-keyed donor prior inside one capability."""

    key = (str(target_center), tuple(source_excluded_centers))
    prior = cache.get(key)
    if prior is None:
        donor_centers = tuple(
            center
            for center in CENTERS
            if center not in set(source_excluded_centers)
        )
        label_surface = {
            center: labels_for_cases(labels, center, store.case_ids(center))
            for center in donor_centers
        }
        prior = compute_donor_directional_priors(
            store,
            target_center=target_center,
            donor_labels_by_center_case=label_surface,
            source_excluded_centers=source_excluded_centers,
            donor_scope_hash=donor_scope_hash,
        )
        cache[key] = prior
    elif prior.donor_scope_hash != donor_scope_hash:
        raise GovernanceError("SCALE-BP v2 donor prior cache escaped capability.")
    return prior


def donor_observations(
    rectangle: ActionRectangle,
    scored: ScoredActionRectangle,
    *,
    scope: DonorFitScope,
    source_centers: Sequence[str],
) -> tuple[DonorObservation, ...]:
    realized = {row.action_id: row for row in scored.values}
    return tuple(
        DonorObservation(
            query_center=rectangle.target_center,
            case_id=rectangle.case_id,
            action_id=cell.action_id,
            descriptor=cell.evidence.descriptor,
            realized=realized[cell.action_id].value,
            source_centers=tuple(str(value) for value in source_centers),
            scope_hash=scope.scope_hash,
        )
        for cell in rectangle.cells
    )


def estimate(
    rectangle: ActionRectangle,
    donor_model: DonorActionModel,
    local_model: LocalResidualModel | None,
    settings: ScienceSettings,
) -> tuple[ActionEstimate, ...]:
    return estimate_action_rectangle(
        rectangle,
        donor_model,
        local_model,
        maximum_abs_standardized_feature=(
            settings.maximum_abs_standardized_feature
        ),
        minimum_independent_centers=settings.minimum_independent_centers,
    )


def select_control(
    rectangle: ActionRectangle,
    estimates: Sequence[ActionEstimate],
    settings: ScienceSettings,
) -> RouteDecision:
    bounds = build_preargmax_bounds(
        estimates,
        base_multiplier=settings.uncertainty_base_multiplier,
    )
    return select_action(
        rectangle,
        estimates,
        bounds,
        thresholds=settings.safety_thresholds,
    )


__all__ = (
    "cached_prior",
    "canonical_centers",
    "donor_observations",
    "estimate",
    "labels_for_cases",
    "planning_scope_hash",
    "select_control",
    "subset_scoped_labels",
)
