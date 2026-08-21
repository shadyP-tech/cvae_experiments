"""Route-scoped, support-conditioned transport for PCSI-RACR.

Transport consumes one whole-case-LOO endpoint state at a time.  It never
aggregates the other cases in the candidate center and never reads an
evaluation-label capability.  Reference cases remain distinct semantic rows
although identical numeric leaves may share storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
    EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
    EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT,
    EXPECTED_TRANSPORT_SCREEN_COUNT,
    PORTFOLIO_METHOD_ID,
    TRANSPORT_FEATURE_NAMES,
    TRANSPORT_MAD_SCALE,
    TRANSPORT_MIN_REFERENCE_CENTER_COUNT,
    TRANSPORT_SCALE_FLOOR,
)
from .contracts import EndpointCasePrediction
from .hashing import canonical_hash, require_sha256
from .projection_lattice import THRESHOLD, as_binary32


TRANSPORT_SEMANTICS = "SUPPORT_CONDITIONED_ENDPOINT_RECONSTRUCTED"
TRANSPORT_PROTOCOL_STATUS = "ROUTE_SCOPED_OWN_CASE_NONINTERFERENCE"
TRANSPORT_PROTOCOL_CONTRACT = MappingProxyType(
    {
        "transport_semantics": TRANSPORT_SEMANTICS,
        "transport_endpoint_support_scope": "endpoint_target_T_minus_held_case",
        "transport_actual_source_prior_scope": "q_not_in_outer_H_or_source_e",
        "transport_donor_source_prior_scope": (
            "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
        ),
        "transport_source_prior_labels_used_upstream": True,
        "transport_route_local_support_labels_used_upstream": True,
        "transport_held_case_evaluation_capability_used_directly": False,
        "transport_pseudo_evaluation_capability_used_directly": False,
        "transport_terminal_evaluation_capability_used_directly": False,
        "transport_label_free_claim": False,
        "transport_uses_pre_equivalence_endpoint_crossing_rates": True,
        "transport_screens_sealed_before_pseudo_evaluation_capability_open": True,
        "transport_screens_sealed_before_terminal_evaluation_capability_open": True,
        "transport_identity_level_route_noninterference_required": True,
        "transport_identity_level_route_noninterference_proven": True,
        "transport_authorization_valid": True,
        "transport_protocol_status": TRANSPORT_PROTOCOL_STATUS,
    }
)
LEGACY_TRANSPORT_PROTOCOL_FIELDS = frozenset(
    {
        "transport_uses_labels",
        "transport_uses_pre_equivalence_physical_crossing_rates",
        "transport_label_free",
        "transport_physical_only",
    }
)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or bool(np.any(weights <= 0.0))
    ):
        raise ProtocolError("PCSI-RACR weighted-median inputs drifted.")
    order = np.lexsort((np.arange(len(values), dtype=np.int64), values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights, dtype=np.float64))
    index = int(
        np.searchsorted(
            np.cumsum(sorted_weights, dtype=np.float64),
            cutoff,
            side="left",
        )
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _location_scale(
    descriptors: Sequence["SupportConditionedCaseDescriptor"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = tuple(descriptors)
    centers = tuple(dict.fromkeys(row.endpoint_center for row in rows))
    if (
        len(centers) < TRANSPORT_MIN_REFERENCE_CENTER_COUNT
        or any(row.feature_names != TRANSPORT_FEATURE_NAMES for row in rows)
    ):
        raise ProtocolError("PCSI-RACR transport reference topology drifted.")
    counts = {center: sum(row.endpoint_center == center for row in rows) for center in centers}
    weights = np.asarray(
        [1.0 / (len(centers) * counts[row.endpoint_center]) for row in rows],
        dtype=np.float64,
    )
    matrix = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    location = np.asarray(
        [_weighted_median(matrix[:, j], weights) for j in range(matrix.shape[1])],
        dtype=np.float64,
    )
    mad = np.asarray(
        [
            _weighted_median(np.abs(matrix[:, j] - location[j]), weights)
            for j in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    scale = np.maximum(TRANSPORT_MAD_SCALE * mad, TRANSPORT_SCALE_FLOOR)
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise ProtocolError("PCSI-RACR transport normalization is nonfinite.")
    return location, scale, weights


def _distance(
    descriptor: "SupportConditionedCaseDescriptor",
    location: np.ndarray,
    scale: np.ndarray,
) -> float:
    values = np.asarray(descriptor.feature_values, dtype=np.float64)
    value = float(np.max(np.abs(values - location) / scale))
    if not math.isfinite(value) or value < 0.0:
        raise ProtocolError("PCSI-RACR transport distance is invalid.")
    return value


@dataclass(frozen=True, order=True)
class RouteTransportLineage:
    outer_center: str
    endpoint_center: str
    case_id: str
    role: str
    support_case_ids: tuple[str, ...]
    excluded_centers: tuple[str, ...]
    endpoint_state_hash: str
    lineage_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        roles = {
            "target_candidate",
            "target_reference",
            "pseudo_candidate",
            "pseudo_reference",
        }
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        excluded = tuple(str(value) for value in self.excluded_centers)
        if (
            self.outer_center not in CENTERS
            or self.endpoint_center not in CENTERS
            or not self.case_id
            or self.role not in roles
            or self.case_id in support
            or not support
            or len(support) != len(set(support))
            or len(excluded) != len(set(excluded))
            or any(center not in CENTERS for center in excluded)
        ):
            raise ProtocolError("PCSI-RACR transport lineage drifted.")
        require_sha256(self.endpoint_state_hash, "transport_endpoint_state_hash")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "excluded_centers", excluded)
        object.__setattr__(self, "lineage_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_route_transport_lineage_v1",
            "transport_semantics": TRANSPORT_SEMANTICS,
            "outer_center": self.outer_center,
            "endpoint_center": self.endpoint_center,
            "case_id": self.case_id,
            "role": self.role,
            "support_case_ids": list(self.support_case_ids),
            "excluded_centers": list(self.excluded_centers),
            "endpoint_state_hash": self.endpoint_state_hash,
            "own_case_evaluation_label_used": False,
            "source_prior_labels_used_upstream": True,
            "support_labels_used_upstream": True,
            "label_free_claim": False,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "lineage_hash": self.lineage_hash}


@dataclass(frozen=True, order=True)
class SupportConditionedCaseDescriptor:
    lineage: RouteTransportLineage
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    sample_count: int
    endpoint_prediction_hash: str
    seed_surface_hash: str
    numeric_leaf_hash: str = field(init=False, compare=True)
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.feature_values)
        if (
            self.feature_names != TRANSPORT_FEATURE_NAMES
            or len(values) != len(TRANSPORT_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in values)
            or type(self.sample_count) is not int
            or self.sample_count <= 0
        ):
            raise ProtocolError("PCSI-RACR route descriptor drifted.")
        require_sha256(self.endpoint_prediction_hash, "transport_endpoint_hash")
        require_sha256(self.seed_surface_hash, "transport_seed_surface_hash")
        numeric_role = (
            "pair_rebound_case"
            if self.lineage.role in {"target_reference", "pseudo_candidate"}
            else self.lineage.role
        )
        numeric = canonical_hash(
            {
                "schema_version": "fixed_bank_pcsi_racr_transport_numeric_leaf_v1",
                "numeric_role": numeric_role,
                "outer_center": self.lineage.outer_center,
                "endpoint_center": self.lineage.endpoint_center,
                "case_id": self.lineage.case_id,
                "excluded_centers": list(self.lineage.excluded_centers),
                "endpoint_state_hash": self.lineage.endpoint_state_hash,
                "feature_names": list(self.feature_names),
                "feature_values": list(values),
                "sample_count": self.sample_count,
                "endpoint_prediction_hash": self.endpoint_prediction_hash,
                "seed_surface_hash": self.seed_surface_hash,
            }
        )
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "numeric_leaf_hash", numeric)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def center(self) -> str:
        return self.lineage.endpoint_center

    @property
    def endpoint_center(self) -> str:
        return self.lineage.endpoint_center

    @property
    def case_id(self) -> str:
        return self.lineage.case_id

    @property
    def case_count(self) -> int:
        return 1

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_support_conditioned_case_descriptor_v1",
            "transport_lineage": self.lineage.to_payload(),
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "sample_count": self.sample_count,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "seed_surface_hash": self.seed_surface_hash,
            "numeric_leaf_hash": self.numeric_leaf_hash,
            "label_free_claim": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class TransportReferenceSummary:
    outer_center: str
    candidate_center: str
    candidate_case_id: str
    role: str
    reference_centers: tuple[str, ...]
    reference_descriptor_hashes: tuple[str, ...]
    center_weight_sums: tuple[tuple[str, float], ...]
    location: tuple[float, ...]
    scale: tuple[float, ...]
    leave_one_center_maxima: tuple[tuple[str, float], ...]
    threshold: float
    summary_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        references = tuple(self.reference_centers)
        if (
            self.outer_center not in CENTERS
            or self.candidate_center not in CENTERS
            or not self.candidate_case_id
            or self.role not in {"target", "pseudo_audit"}
            or self.candidate_center in references
            or len(references) < TRANSPORT_MIN_REFERENCE_CENTER_COUNT
            or len(references) != len(set(references))
            or len(self.location) != len(TRANSPORT_FEATURE_NAMES)
            or len(self.scale) != len(TRANSPORT_FEATURE_NAMES)
            or not math.isfinite(float(self.threshold))
            or self.threshold < 0.0
        ):
            raise ProtocolError("PCSI-RACR transport summary drifted.")
        for digest in self.reference_descriptor_hashes:
            require_sha256(digest, "transport_reference_descriptor_hash")
        if any(abs(total - 1.0 / len(references)) > 1.0e-12 for _center, total in self.center_weight_sums):
            raise ProtocolError("PCSI-RACR equal-center weights drifted.")
        object.__setattr__(self, "summary_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_transport_reference_summary_v1",
            "outer_center": self.outer_center,
            "candidate_center": self.candidate_center,
            "candidate_case_id": self.candidate_case_id,
            "role": self.role,
            "reference_centers": list(self.reference_centers),
            "reference_descriptor_hashes": list(self.reference_descriptor_hashes),
            "center_weight_sums": [list(row) for row in self.center_weight_sums],
            "location": list(self.location),
            "scale": list(self.scale),
            "leave_one_center_maxima": [list(row) for row in self.leave_one_center_maxima],
            "threshold": self.threshold,
            "coverage_claimed": False,
            "ood_detection_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "summary_hash": self.summary_hash}


@dataclass(frozen=True, order=True)
class TransportReferenceBlockSummary:
    outer_center: str
    donor_center: str | None
    reference_center: str
    descriptor_hashes: tuple[str, ...]
    case_ids: tuple[str, ...]
    block_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.reference_center not in CENTERS
            or self.reference_center == self.outer_center
            or (
                self.donor_center is not None
                and (
                    self.donor_center not in CENTERS
                    or self.donor_center in {self.outer_center, self.reference_center}
                )
            )
            or not self.case_ids
            or len(self.case_ids) != len(set(self.case_ids))
            or len(self.descriptor_hashes) != len(self.case_ids)
        ):
            raise ProtocolError("PCSI-RACR transport reference block drifted.")
        for digest in self.descriptor_hashes:
            require_sha256(digest, "transport_block_descriptor_hash")
        object.__setattr__(self, "block_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_transport_reference_block_v1",
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "reference_center": self.reference_center,
            "case_ids": list(self.case_ids),
            "descriptor_hashes": list(self.descriptor_hashes),
            "equal_case_weight_within_center": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "block_hash": self.block_hash}


@dataclass(frozen=True, order=True)
class RouteTransportScreen:
    outer_center: str
    candidate_center: str
    candidate_case_id: str
    role: str
    candidate_distance: float
    threshold: float
    passed: bool
    audit_only: bool
    candidate_descriptor_hash: str
    reference_summary_hash: str
    reference_block_hashes: tuple[str, ...] = ()
    reference_centers: tuple[str, ...] = ()
    location: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    leave_one_center_maxima: tuple[tuple[str, float], ...] = ()
    screen_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.candidate_center not in CENTERS
            or not self.candidate_case_id
            or self.role not in {"target", "pseudo_audit"}
            or bool(self.audit_only) != (self.role == "pseudo_audit")
            or not math.isfinite(float(self.candidate_distance))
            or not math.isfinite(float(self.threshold))
            or bool(self.passed) != bool(self.candidate_distance <= self.threshold)
        ):
            raise ProtocolError("PCSI-RACR route transport screen drifted.")
        require_sha256(self.candidate_descriptor_hash, "transport_candidate_descriptor_hash")
        require_sha256(self.reference_summary_hash, "transport_reference_summary_hash")
        if (
            len(self.reference_block_hashes) != len(self.reference_centers)
            or len(self.location) != len(TRANSPORT_FEATURE_NAMES)
            or len(self.scale) != len(TRANSPORT_FEATURE_NAMES)
            or tuple(center for center, _value in self.leave_one_center_maxima)
            != self.reference_centers
        ):
            raise ProtocolError("PCSI-RACR transport screen reference matrix drifted.")
        for digest in self.reference_block_hashes:
            require_sha256(digest, "transport_reference_block_hash")
        object.__setattr__(self, "screen_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_route_transport_screen_v1",
            "outer_center": self.outer_center,
            "candidate_center": self.candidate_center,
            "candidate_case_id": self.candidate_case_id,
            "role": self.role,
            "candidate_distance": self.candidate_distance,
            "threshold": self.threshold,
            "passed": self.passed,
            "audit_only": self.audit_only,
            "candidate_descriptor_hash": self.candidate_descriptor_hash,
            "reference_summary_hash": self.reference_summary_hash,
            "reference_block_hashes": list(self.reference_block_hashes),
            "reference_centers": list(self.reference_centers),
            "location": list(self.location),
            "scale": list(self.scale),
            "leave_one_center_maxima": [
                list(row) for row in self.leave_one_center_maxima
            ],
            "equality_passes": True,
            "coverage_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "screen_hash": self.screen_hash}


@dataclass(frozen=True)
class TransportRuntimeSeal:
    descriptor_count: int
    numeric_leaf_count: int
    reference_summary_count: int
    screen_count: int
    descriptor_hash: str
    reference_summary_hash: str
    screen_hash: str
    strict_canonical_topology: bool
    transport_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.strict_canonical_topology and (
            self.descriptor_count != EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT
            or self.numeric_leaf_count != EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT
            or self.reference_summary_count != EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT
            or self.screen_count != EXPECTED_TRANSPORT_SCREEN_COUNT
        ):
            raise ProtocolError("PCSI-RACR canonical transport workload drifted.")
        for value in (
            self.descriptor_hash,
            self.reference_summary_hash,
            self.screen_hash,
        ):
            require_sha256(value, "transport_runtime_hash")
        object.__setattr__(self, "transport_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_transport_runtime_seal_v1",
            "transport_semantics": TRANSPORT_SEMANTICS,
            "descriptor_count": self.descriptor_count,
            "numeric_leaf_count": self.numeric_leaf_count,
            "reference_summary_count": self.reference_summary_count,
            "screen_count": self.screen_count,
            "descriptor_hash": self.descriptor_hash,
            "reference_summary_hash": self.reference_summary_hash,
            "screen_hash": self.screen_hash,
            "strict_canonical_topology": self.strict_canonical_topology,
            "own_route_noninterference_proven": True,
            "pseudo_transport_is_audit_only": True,
            "authorization_valid": True,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "transport_hash": self.transport_hash}


def build_case_transport_descriptor(
    prediction: EndpointCasePrediction,
    *,
    portfolio_seed_probabilities: np.ndarray,
    lineage: RouteTransportLineage,
) -> SupportConditionedCaseDescriptor:
    if prediction.center != lineage.endpoint_center or prediction.case_id != lineage.case_id:
        raise ProtocolError("PCSI-RACR descriptor crossed a route identity.")
    portfolio = as_binary32(prediction.probabilities[PORTFOLIO_METHOD_ID], name="transport P")
    seed = np.ascontiguousarray(portfolio_seed_probabilities, dtype=np.float64)
    if seed.shape != (9, len(prediction.sample_ids)) or not np.isfinite(seed).all():
        raise ProtocolError("PCSI-RACR transport seed topology drifted.")
    if not np.allclose(
        np.mean(seed, axis=0, dtype=np.float64),
        portfolio.astype(np.float64),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise ProtocolError("PCSI-RACR seed surface does not reconstruct P.")
    p64 = portfolio.astype(np.float64)
    p_hard = p64 >= float(THRESHOLD)
    clipped = np.clip(p64, 1.0e-12, 1.0 - 1.0e-12)
    values = [
        float(np.log1p(len(p64))),
        float(np.mean(p64, dtype=np.float64)),
        float(np.std(p64, ddof=0, dtype=np.float64)),
        float(np.mean(np.abs(p64 - float(THRESHOLD)), dtype=np.float64)),
        float(
            np.mean(
                -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped)),
                dtype=np.float64,
            )
        ),
        float(np.mean(np.std(seed, axis=0, ddof=0, dtype=np.float64), dtype=np.float64)),
    ]
    for alternative in ALTERNATIVE_METHOD_IDS:
        candidate = as_binary32(prediction.probabilities[alternative], name="transport endpoint").astype(np.float64)
        a_hard = candidate >= float(THRESHOLD)
        values.extend(
            (
                float(np.mean(((~p_hard) & a_hard).astype(np.float64), dtype=np.float64)),
                float(np.mean((p_hard & (~a_hard)).astype(np.float64), dtype=np.float64)),
            )
        )
    return SupportConditionedCaseDescriptor(
        lineage,
        TRANSPORT_FEATURE_NAMES,
        tuple(values),
        len(prediction.sample_ids),
        prediction.prediction_hash,
        sha256_array(seed),
    )


def build_reference_summary(
    candidate: SupportConditionedCaseDescriptor,
    references: Sequence[SupportConditionedCaseDescriptor],
    *,
    role: str,
) -> TransportReferenceSummary:
    rows = tuple(sorted(references, key=lambda row: (row.endpoint_center, row.case_id, row.descriptor_hash)))
    centers = tuple(dict.fromkeys(row.endpoint_center for row in rows))
    if (
        candidate.endpoint_center in centers
        or any(row.lineage.outer_center != candidate.lineage.outer_center for row in rows)
    ):
        raise ProtocolError("PCSI-RACR candidate entered its reference pool.")
    location, scale, weights = _location_scale(rows)
    maxima = []
    for center in centers:
        training = tuple(row for row in rows if row.endpoint_center != center)
        held = tuple(row for row in rows if row.endpoint_center == center)
        loo_location, loo_scale, _loo_weights = _location_scale(training)
        maxima.append((center, max(_distance(row, loo_location, loo_scale) for row in held)))
    threshold = max(value for _center, value in maxima)
    center_weight_sums = tuple(
        (
            center,
            float(
                np.sum(
                    weights[
                        np.asarray([row.endpoint_center == center for row in rows], dtype=bool)
                    ],
                    dtype=np.float64,
                )
            ),
        )
        for center in centers
    )
    return TransportReferenceSummary(
        candidate.lineage.outer_center,
        candidate.endpoint_center,
        candidate.case_id,
        role,
        centers,
        tuple(row.descriptor_hash for row in rows),
        center_weight_sums,
        tuple(float(value) for value in location),
        tuple(float(value) for value in scale),
        tuple(maxima),
        float(threshold),
    )


def evaluate_transport_screen(
    candidate: SupportConditionedCaseDescriptor,
    references: Sequence[SupportConditionedCaseDescriptor],
    *,
    role: str,
    reference_blocks: Sequence[TransportReferenceBlockSummary] = (),
) -> tuple[TransportReferenceSummary, RouteTransportScreen]:
    summary = build_reference_summary(candidate, references, role=role)
    distance = _distance(
        candidate,
        np.asarray(summary.location, dtype=np.float64),
        np.asarray(summary.scale, dtype=np.float64),
    )
    screen = RouteTransportScreen(
        candidate.lineage.outer_center,
        candidate.endpoint_center,
        candidate.case_id,
        role,
        distance,
        summary.threshold,
        bool(distance <= summary.threshold),
        role == "pseudo_audit",
        candidate.descriptor_hash,
        summary.summary_hash,
        tuple(row.block_hash for row in reference_blocks),
        summary.reference_centers,
        summary.location,
        summary.scale,
        summary.leave_one_center_maxima,
    )
    return summary, screen


def build_reference_block_summary(
    descriptors: Sequence[SupportConditionedCaseDescriptor],
    *,
    outer_center: str,
    donor_center: str | None,
    reference_center: str,
) -> TransportReferenceBlockSummary:
    rows = tuple(sorted(descriptors, key=lambda row: (row.case_id, row.descriptor_hash)))
    if (
        not rows
        or {row.endpoint_center for row in rows} != {reference_center}
        or any(row.lineage.outer_center != outer_center for row in rows)
    ):
        raise ProtocolError("PCSI-RACR transport block membership drifted.")
    return TransportReferenceBlockSummary(
        outer_center,
        donor_center,
        reference_center,
        tuple(row.descriptor_hash for row in rows),
        tuple(row.case_id for row in rows),
    )


def seal_transport_runtime(
    descriptors: Mapping[object, SupportConditionedCaseDescriptor],
    summaries: Mapping[object, TransportReferenceBlockSummary],
    screens: Mapping[object, RouteTransportScreen],
    *,
    strict_canonical_topology: bool,
) -> TransportRuntimeSeal:
    descriptor_rows = tuple(descriptors[key] for key in sorted(descriptors, key=str))
    summary_rows = tuple(summaries[key] for key in sorted(summaries, key=str))
    screen_rows = tuple(screens[key] for key in sorted(screens, key=str))
    return TransportRuntimeSeal(
        len(descriptor_rows),
        len({row.numeric_leaf_hash for row in descriptor_rows}),
        len(summary_rows),
        len(screen_rows),
        canonical_hash([row.to_payload() for row in descriptor_rows]),
        canonical_hash([row.to_payload() for row in summary_rows]),
        canonical_hash([row.to_payload() for row in screen_rows]),
        strict_canonical_topology,
    )


# Compatibility names are aliases only; their semantics are route-scoped.
TransportEndpointLineage = RouteTransportLineage
CenterTransportDescriptor = SupportConditionedCaseDescriptor
TransportScreen = RouteTransportScreen


__all__ = (
    "CenterTransportDescriptor",
    "LEGACY_TRANSPORT_PROTOCOL_FIELDS",
    "RouteTransportLineage",
    "RouteTransportScreen",
    "SupportConditionedCaseDescriptor",
    "TRANSPORT_PROTOCOL_CONTRACT",
    "TRANSPORT_PROTOCOL_STATUS",
    "TRANSPORT_SEMANTICS",
    "TransportEndpointLineage",
    "TransportReferenceSummary",
    "TransportReferenceBlockSummary",
    "TransportRuntimeSeal",
    "TransportScreen",
    "build_case_transport_descriptor",
    "build_reference_summary",
    "build_reference_block_summary",
    "evaluate_transport_screen",
    "seal_transport_runtime",
)
