"""Capability-separated same-run planning for the protected portfolio P."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import CENTERS, DIRECTIONS, HARD_THRESHOLD, candidate_sources
from .endpoints import RouteEndpointPlan, derive_route_endpoint_plan
from .store import PhysicalStoreAdapter


LabelRows = Mapping[str, int] | Sequence[int] | np.ndarray
CaseLabelSurface = Mapping[str, LabelRows]
DonorLabelSurface = Mapping[str, CaseLabelSurface]


@dataclass(frozen=True, slots=True)
class DonorDirectionalPriorSurface:
    """Label-free output of the donor capability phase."""

    target_center: str
    source_excluded_centers: tuple[str, ...]
    available_sources: tuple[str, ...]
    values: Mapping[str, Mapping[str, float]]
    donor_scope_hash: str
    physical_store_hash: str
    prior_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = _source_exclusions(
            self.target_center, self.source_excluded_centers
        )
        available = tuple(
            source
            for source in candidate_sources(self.target_center)
            if source not in set(excluded)
        )
        values = {
            str(direction): {
                str(source): float(value)
                for source, value in source_values.items()
            }
            for direction, source_values in self.values.items()
        }
        if (
            self.available_sources != available
            or tuple(values) != DIRECTIONS
            or any(tuple(values[direction]) != available for direction in DIRECTIONS)
            or not all(
                math.isfinite(value)
                for direction in DIRECTIONS
                for value in values[direction].values()
            )
            or not self.donor_scope_hash
            or not self.physical_store_hash
        ):
            raise GovernanceError("SCALE-BP v2 donor prior surface drifted.")
        frozen_values = MappingProxyType(
            {
                direction: MappingProxyType(values[direction])
                for direction in DIRECTIONS
            }
        )
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(self, "available_sources", available)
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(
            self,
            "prior_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_directional_prior_v1",
                    "target_center": self.target_center,
                    "source_excluded_centers": excluded,
                    "available_sources": available,
                    "values": values,
                    "donor_scope_hash": self.donor_scope_hash,
                    "physical_store_hash": self.physical_store_hash,
                    "equal_center_weighting": True,
                    "source_e_excluded_from_own_prior": True,
                    "labels_retained": False,
                }
            ),
        )


def compute_donor_directional_priors(
    store: PhysicalStoreAdapter,
    *,
    target_center: object,
    donor_labels_by_center_case: DonorLabelSurface,
    source_excluded_centers: tuple[str, ...],
    donor_scope_hash: object,
) -> DonorDirectionalPriorSurface:
    """Consume donor labels and return only a hashed, label-free prior map."""

    target = str(target_center)
    excluded = _source_exclusions(target, source_excluded_centers)
    available = tuple(
        source for source in candidate_sources(target) if source not in set(excluded)
    )
    donor_centers = tuple(center for center in CENTERS if center not in set(excluded))
    if (
        tuple(donor_labels_by_center_case) != donor_centers
        or not str(donor_scope_hash)
    ):
        raise GovernanceError("SCALE-BP v2 donor label center scope drifted.")
    for center in donor_centers:
        if tuple(donor_labels_by_center_case[center]) != store.case_ids(center):
            raise GovernanceError("SCALE-BP v2 donor label case scope drifted.")
    prior: dict[str, dict[str, float]] = {
        direction: {} for direction in DIRECTIONS
    }
    for source in available:
        center_values = [
            _directional_gain(
                store,
                query_center=query_center,
                source=source,
                labels_by_case=donor_labels_by_center_case[query_center],
            )
            for query_center in donor_centers
            if query_center != source
        ]
        if len(center_values) < 3:
            raise GovernanceError(
                "SCALE-BP v2 source-excluded donor prior is underpowered."
            )
        for direction in DIRECTIONS:
            prior[direction][source] = float(
                np.mean(
                    [row[direction] for row in center_values], dtype=np.float64
                )
            )
    return DonorDirectionalPriorSurface(
        target,
        excluded,
        available,
        prior,
        str(donor_scope_hash),
        store.adapter_hash,
    )


def build_protected_route_plan_from_prior(
    store: PhysicalStoreAdapter,
    *,
    target_center: object,
    case_id: object,
    support_labels_by_case: CaseLabelSurface,
    donor_prior: DonorDirectionalPriorSurface,
    support_scope_hash: object,
    support_excluded_case_ids: tuple[str, ...] | None = None,
    outer_held_case_id: object | None = None,
) -> RouteEndpointPlan:
    """Open only the legal support complement after donor labels are discarded.

    For a final route the default exclusion is ``(case_id,)``.  For a local
    validation route the caller must supply the canonical union of the outer
    held case and that validation fold, and identify the outer held case
    explicitly.  Thus the validation case itself, the outer case, and every
    own-fold peer are absent before any directional support gain is computed.
    """

    target, held_case = str(target_center), str(case_id)
    target_cases = store.case_ids(target)
    outer_held = (
        held_case if outer_held_case_id is None else str(outer_held_case_id)
    )
    supplied_exclusions = (
        (held_case,)
        if support_excluded_case_ids is None
        else tuple(str(value) for value in support_excluded_case_ids)
    )
    exclusion_set = set(supplied_exclusions)
    canonical_exclusions = tuple(
        candidate for candidate in target_cases if candidate in exclusion_set
    )
    if (
        donor_prior.target_center != target
        or donor_prior.physical_store_hash != store.adapter_hash
        or not str(support_scope_hash)
        or held_case not in target_cases
        or outer_held not in target_cases
        or held_case not in exclusion_set
        or outer_held not in exclusion_set
        or supplied_exclusions != canonical_exclusions
        or len(supplied_exclusions) != len(exclusion_set)
    ):
        raise GovernanceError("SCALE-BP v2 prior/support route binding drifted.")
    expected_support_cases = tuple(
        case for case in target_cases if case not in exclusion_set
    )
    if tuple(support_labels_by_case) != expected_support_cases:
        raise GovernanceError("SCALE-BP v2 H-minus-c support case scope drifted.")
    available = donor_prior.available_sources
    support: dict[str, dict[str, float]] = {
        direction: {} for direction in DIRECTIONS
    }
    for source in available:
        values = _directional_gain(
            store,
            query_center=target,
            source=source,
            labels_by_case=support_labels_by_case,
        )
        for direction in DIRECTIONS:
            support[direction][source] = values[direction]
    identification: dict[str, dict[str, float]] = {
        direction: {} for direction in DIRECTIONS
    }
    for direction in DIRECTIONS:
        support_scale = float(
            np.mean(np.abs(tuple(support[direction].values())), dtype=np.float64)
        )
        donor_scale = float(
            np.mean(
                np.abs(tuple(donor_prior.values[direction].values())),
                dtype=np.float64,
            )
        )
        for source in available:
            normalized_support = (
                0.0
                if support_scale == 0.0
                else support[direction][source] / support_scale
            )
            normalized_donor = (
                0.0
                if donor_scale == 0.0
                else donor_prior.values[direction][source] / donor_scale
            )
            identification[direction][source] = (
                0.8 * normalized_support + 0.2 * normalized_donor
            )
    score_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_protected_plan_score_maps_v2",
            "target_center": target,
            "evaluation_case_id": held_case,
            "outer_held_case_id": outer_held,
            "support_excluded_case_ids": canonical_exclusions,
            "available_sources": available,
            "identification_scores": identification,
            "support_directional_gains": support,
            "donor_prior_hash": donor_prior.prior_hash,
            "identification_weights": ("4/5", "1/5"),
            "support_scope_hash": str(support_scope_hash),
            "raw_labels_persisted": False,
        }
    )
    return derive_route_endpoint_plan(
        target_center=target,
        case_id=held_case,
        identification_scores=identification,
        support_directional_gains=support,
        donor_directional_priors=donor_prior.values,
        support_scope_hash=str(support_scope_hash),
        source_excluded_centers=donor_prior.source_excluded_centers,
        support_excluded_case_ids=canonical_exclusions,
        outer_held_case_id=outer_held,
        derivation_hashes=(donor_prior.prior_hash, score_hash),
    )


def build_protected_route_plan(
    store: PhysicalStoreAdapter,
    *,
    target_center: object,
    case_id: object,
    support_labels_by_case: CaseLabelSurface,
    donor_labels_by_center_case: DonorLabelSurface,
    support_scope_hash: object,
    donor_scope_hash: object,
    outer_center: object | None = None,
    source_excluded_centers: tuple[str, ...] | None = None,
    support_excluded_case_ids: tuple[str, ...] | None = None,
    outer_held_case_id: object | None = None,
) -> RouteEndpointPlan:
    """Convenience composition; lifecycle code should call the two phases."""

    target = str(target_center)
    outer = target if outer_center is None else str(outer_center)
    excluded = (
        tuple(center for center in CENTERS if center in {target, outer})
        if source_excluded_centers is None
        else tuple(source_excluded_centers)
    )
    prior = compute_donor_directional_priors(
        store,
        target_center=target,
        donor_labels_by_center_case=donor_labels_by_center_case,
        source_excluded_centers=excluded,
        donor_scope_hash=donor_scope_hash,
    )
    return build_protected_route_plan_from_prior(
        store,
        target_center=target,
        case_id=case_id,
        support_labels_by_case=support_labels_by_case,
        donor_prior=prior,
        support_scope_hash=support_scope_hash,
        support_excluded_case_ids=support_excluded_case_ids,
        outer_held_case_id=outer_held_case_id,
    )


def _source_exclusions(
    target_center: object, source_excluded_centers: Sequence[object]
) -> tuple[str, ...]:
    target = str(target_center)
    supplied = tuple(str(center) for center in source_excluded_centers)
    expected = tuple(center for center in CENTERS if center in set(supplied))
    if (
        target not in CENTERS
        or target not in supplied
        or supplied != expected
        or len(supplied) != len(set(supplied))
        or len(candidate_sources(target)) - (len(supplied) - 1) < 3
    ):
        raise GovernanceError("SCALE-BP v2 source-exclusion scope drifted.")
    return supplied


def _directional_gain(
    store: PhysicalStoreAdapter,
    *,
    query_center: str,
    source: str,
    labels_by_case: CaseLabelSurface,
) -> dict[str, float]:
    if source == query_center:
        raise GovernanceError("SCALE-BP v2 donor query cannot use its own expert.")
    positive = negative = 0
    counts = {"zero_to_one": [0, 0], "one_to_zero": [0, 0]}
    for case_id, label_rows in labels_by_case.items():
        baseline_view = store.exact_nine_view(query_center, "B", case_id=case_id)
        alternative_view = store.exact_nine_view(
            query_center, f"A1::source={source}", case_id=case_id
        )
        if baseline_view.sample_ids != alternative_view.sample_ids:
            raise GovernanceError("SCALE-BP v2 gain row identity drifted.")
        labels = _ordered_labels(label_rows, baseline_view.sample_ids)
        baseline_hard = baseline_view.mean_probability >= HARD_THRESHOLD
        alternative_hard = alternative_view.mean_probability >= HARD_THRESHOLD
        is_positive = labels == 1
        is_negative = ~is_positive
        positive += int(np.sum(is_positive, dtype=np.int64))
        negative += int(np.sum(is_negative, dtype=np.int64))
        zero = (~baseline_hard) & alternative_hard
        one = baseline_hard & (~alternative_hard)
        counts["zero_to_one"][0] += int(np.sum(zero & is_positive))
        counts["zero_to_one"][1] += int(np.sum(zero & is_negative))
        counts["one_to_zero"][0] += int(np.sum(one & is_negative))
        counts["one_to_zero"][1] += int(np.sum(one & is_positive))
    if positive <= 0 or negative <= 0:
        raise GovernanceError("SCALE-BP v2 directional gain lacks both classes.")
    return {
        "zero_to_one": counts["zero_to_one"][0] / (2.0 * positive)
        - counts["zero_to_one"][1] / (2.0 * negative),
        "one_to_zero": counts["one_to_zero"][0] / (2.0 * negative)
        - counts["one_to_zero"][1] / (2.0 * positive),
    }


def _ordered_labels(label_rows: LabelRows, sample_ids: tuple[str, ...]) -> np.ndarray:
    if isinstance(label_rows, Mapping):
        if set(label_rows) != set(sample_ids) or len(label_rows) != len(sample_ids):
            raise GovernanceError("SCALE-BP v2 scoped label identity drifted.")
        raw = tuple(label_rows[sample_id] for sample_id in sample_ids)
    else:
        array = np.asarray(label_rows)
        if array.ndim != 1 or len(array) != len(sample_ids):
            raise GovernanceError("SCALE-BP v2 ordered label row count drifted.")
        raw = tuple(array.tolist())
    if any(isinstance(value, bool) or int(value) not in (0, 1) for value in raw):
        raise GovernanceError("SCALE-BP v2 scoped labels are not binary integers.")
    result = np.asarray(raw, dtype=np.int8)
    result.setflags(write=False)
    return result


__all__ = (
    "DonorDirectionalPriorSurface",
    "build_protected_route_plan",
    "build_protected_route_plan_from_prior",
    "compute_donor_directional_priors",
)
