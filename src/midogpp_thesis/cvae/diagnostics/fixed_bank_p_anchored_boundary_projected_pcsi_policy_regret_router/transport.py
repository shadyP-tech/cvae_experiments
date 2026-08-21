"""Support-conditioned endpoint transport screen for policy authorization."""

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
    PORTFOLIO_METHOD_ID,
    TRANSPORT_FEATURE_NAMES,
    TRANSPORT_MAD_SCALE,
    TRANSPORT_MIN_REFERENCE_CENTER_COUNT,
    TRANSPORT_SCALE_FLOOR,
)
from .contracts import EndpointCasePrediction
from .hashing import canonical_hash, require_sha256
from .projection_lattice import THRESHOLD, as_binary32


TRANSPORT_SEMANTICS = "support_conditioned_endpoint_reconstructed_P_B_I_R"
TRANSPORT_ENDPOINT_SUPPORT_SCOPE = "endpoint_target_T_minus_held_case_c"
TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE = (
    "q_not_in_endpoint_target_T_or_source_e"
)
TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE = (
    "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
)
TRANSPORT_PROTOCOL_CONTRACT = MappingProxyType(
    {
        "transport_semantics": TRANSPORT_SEMANTICS,
        "transport_endpoint_support_scope": TRANSPORT_ENDPOINT_SUPPORT_SCOPE,
        "transport_actual_source_prior_scope": TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE,
        "transport_donor_source_prior_scope": TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE,
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
        "transport_identity_level_route_noninterference_proven": False,
        "transport_authorization_valid": False,
        "transport_protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
    }
)
LEGACY_TRANSPORT_PROTOCOL_FIELDS = frozenset(
    {
        "transport_uses_labels",
        "transport_uses_pre_equivalence_physical_crossing_rates",
    }
)


@dataclass(frozen=True, order=True)
class TransportEndpointLineage:
    """Exact legal-label lineage for one outer/candidate endpoint surface."""

    outer_target_center: str
    endpoint_target_center: str
    endpoint_state_matrix_hash: str
    lineage_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        outer = str(self.outer_target_center)
        target = str(self.endpoint_target_center)
        if outer not in CENTERS or target not in CENTERS:
            raise ProtocolError("PCSI-PARC transport lineage center drifted.")
        require_sha256(
            self.endpoint_state_matrix_hash,
            "transport_endpoint_state_matrix_hash",
        )
        object.__setattr__(self, "outer_target_center", outer)
        object.__setattr__(self, "endpoint_target_center", target)
        object.__setattr__(
            self,
            "lineage_hash",
            canonical_hash(self._unhashed()),
        )

    @property
    def source_prior_scope(self) -> str:
        return (
            TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE
            if self.endpoint_target_center == self.outer_target_center
            else TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": (
                "fixed_bank_pcsi_parc_transport_endpoint_lineage_v2"
            ),
            "transport_semantics": TRANSPORT_SEMANTICS,
            "outer_target_center": self.outer_target_center,
            "endpoint_target_center": self.endpoint_target_center,
            "endpoint_support_scope": TRANSPORT_ENDPOINT_SUPPORT_SCOPE,
            "source_prior_scope": self.source_prior_scope,
            "endpoint_state_matrix_hash": self.endpoint_state_matrix_hash,
            "source_prior_labels_used_upstream": True,
            "route_local_support_labels_used_upstream": True,
            "held_case_evaluation_capability_used_directly": False,
            "pseudo_evaluation_capability_used_directly": False,
            "terminal_evaluation_capability_used_directly": False,
            "label_free_claim": False,
            "uses_pre_equivalence_endpoint_crossing_rates": True,
            "identity_level_route_noninterference_required": True,
            "identity_level_route_noninterference_proven": False,
            "authorization_valid": False,
            "protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "lineage_hash": self.lineage_hash}


@dataclass(frozen=True, order=True)
class CenterTransportDescriptor:
    lineage: TransportEndpointLineage
    center: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    sample_count: int
    case_count: int
    endpoint_prediction_hash: str
    seed_surface_hash: str
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.feature_values)
        if (
            self.center not in CENTERS
            or self.lineage.endpoint_target_center != self.center
            or self.feature_names != TRANSPORT_FEATURE_NAMES
            or len(values) != len(TRANSPORT_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in values)
            or type(self.sample_count) is not int
            or type(self.case_count) is not int
            or min(self.sample_count, self.case_count) <= 0
        ):
            raise ProtocolError("PCSI-PARC transport descriptor drifted.")
        require_sha256(self.endpoint_prediction_hash, "transport_endpoint_hash")
        require_sha256(self.seed_surface_hash, "transport_seed_surface_hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_transport_descriptor_v2",
            "center": self.center,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "sample_count": self.sample_count,
            "case_count": self.case_count,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "seed_surface_hash": self.seed_surface_hash,
            "transport_lineage": self.lineage.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class TransportScreen:
    outer_target_center: str
    candidate_center: str
    reference_centers: tuple[str, ...]
    candidate_distance: float
    threshold: float
    leave_one_reference_distances: tuple[tuple[str, float], ...]
    passed: bool
    candidate_descriptor_hash: str
    reference_descriptor_hashes: tuple[str, ...]
    candidate_lineage_hash: str
    reference_lineage_hashes: tuple[str, ...]
    screen_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        references = tuple(str(value) for value in self.reference_centers)
        distances = tuple((str(center), float(value)) for center, value in self.leave_one_reference_distances)
        if (
            self.outer_target_center not in CENTERS
            or self.candidate_center not in CENTERS
            or self.candidate_center in references
            or len(references) < TRANSPORT_MIN_REFERENCE_CENTER_COUNT
            or len(references) != len(set(references))
            or tuple(center for center, _value in distances) != references
            or not math.isfinite(float(self.candidate_distance))
            or not math.isfinite(float(self.threshold))
            or self.candidate_distance < 0.0
            or self.threshold < 0.0
            or any(not math.isfinite(value) or value < 0.0 for _center, value in distances)
            or bool(self.passed) != bool(self.candidate_distance <= self.threshold)
        ):
            raise ProtocolError("PCSI-PARC transport screen drifted.")
        require_sha256(self.candidate_descriptor_hash, "transport_candidate_hash")
        for digest in self.reference_descriptor_hashes:
            require_sha256(digest, "transport_reference_hash")
        require_sha256(
            self.candidate_lineage_hash,
            "transport_candidate_lineage_hash",
        )
        if len(self.reference_lineage_hashes) != len(references):
            raise ProtocolError("PCSI-PARC transport lineage matrix drifted.")
        for digest in self.reference_lineage_hashes:
            require_sha256(digest, "transport_reference_lineage_hash")
        object.__setattr__(self, "reference_centers", references)
        object.__setattr__(self, "leave_one_reference_distances", distances)
        object.__setattr__(self, "screen_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_transport_screen_v2",
            "transport_semantics": TRANSPORT_SEMANTICS,
            "outer_target_center": self.outer_target_center,
            "candidate_center": self.candidate_center,
            "screen_role": (
                "target"
                if self.candidate_center == self.outer_target_center
                else "pseudo"
            ),
            "reference_centers": list(self.reference_centers),
            "candidate_distance": self.candidate_distance,
            "threshold": self.threshold,
            "leave_one_reference_distances": [
                {"center": center, "distance": value}
                for center, value in self.leave_one_reference_distances
            ],
            "passed": self.passed,
            "candidate_descriptor_hash": self.candidate_descriptor_hash,
            "reference_descriptor_hashes": list(self.reference_descriptor_hashes),
            "candidate_lineage_hash": self.candidate_lineage_hash,
            "reference_lineage_hashes": list(self.reference_lineage_hashes),
            "source_prior_labels_used_upstream": True,
            "route_local_support_labels_used_upstream": True,
            "held_case_evaluation_capability_used_directly": False,
            "pseudo_evaluation_capability_used_directly": False,
            "terminal_evaluation_capability_used_directly": False,
            "label_free_claim": False,
            "uses_pre_equivalence_endpoint_crossing_rates": True,
            "identity_level_route_noninterference_required": True,
            "identity_level_route_noninterference_proven": False,
            "authorization_valid": False,
            "protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
            "equality_passes": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "screen_hash": self.screen_hash}


@dataclass(frozen=True)
class TransportRuntimeSeal:
    """Immutable seal built before any pseudo or terminal evaluation grant."""

    descriptor_count: int
    screen_count: int
    descriptor_hash: str
    screen_hash: str
    transport_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = len(CENTERS) * len(CENTERS)
        if self.descriptor_count != expected or self.screen_count != expected:
            raise ProtocolError("PCSI-PARC transport seal workload drifted.")
        require_sha256(self.descriptor_hash, "transport_descriptor_matrix_hash")
        require_sha256(self.screen_hash, "transport_screen_matrix_hash")
        object.__setattr__(
            self,
            "transport_hash",
            canonical_hash(self._unhashed()),
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_transport_runtime_seal_v2",
            "transport_semantics": TRANSPORT_SEMANTICS,
            "descriptor_count": self.descriptor_count,
            "screen_count": self.screen_count,
            "descriptor_hash": self.descriptor_hash,
            "screen_hash": self.screen_hash,
            "source_prior_labels_used_upstream": True,
            "route_local_support_labels_used_upstream": True,
            "held_case_evaluation_capability_used_directly": False,
            "pseudo_evaluation_capability_used_directly": False,
            "terminal_evaluation_capability_used_directly": False,
            "label_free_claim": False,
            "uses_pre_equivalence_endpoint_crossing_rates": True,
            "screens_sealed_before_pseudo_evaluation_capability_open": True,
            "screens_sealed_before_terminal_evaluation_capability_open": True,
            "identity_level_route_noninterference_required": True,
            "identity_level_route_noninterference_proven": False,
            "authorization_valid": False,
            "protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "transport_hash": self.transport_hash}


def build_center_transport_descriptor(
    predictions: Sequence[EndpointCasePrediction],
    *,
    portfolio_seed_probabilities_by_case: Mapping[str, np.ndarray],
    lineage: TransportEndpointLineage,
) -> CenterTransportDescriptor:
    rows = tuple(predictions)
    if not rows or len({row.center for row in rows}) != 1:
        raise ProtocolError("PCSI-PARC transport rows span centers.")
    center = rows[0].center
    if set(portfolio_seed_probabilities_by_case) != {row.case_id for row in rows}:
        raise ProtocolError("PCSI-PARC transport seed surface misses a case.")
    portfolio_parts: list[np.ndarray] = []
    endpoint_parts: dict[str, list[np.ndarray]] = {
        alternative: [] for alternative in ALTERNATIVE_METHOD_IDS
    }
    seed_sd_parts: list[np.ndarray] = []
    seed_hash_rows: list[dict[str, object]] = []
    for row in rows:
        portfolio = as_binary32(row.probabilities[PORTFOLIO_METHOD_ID], name="transport P")
        seed = np.ascontiguousarray(
            portfolio_seed_probabilities_by_case[row.case_id], dtype=np.float64
        )
        if seed.shape != (9, len(row.sample_ids)) or not np.isfinite(seed).all():
            raise ProtocolError("PCSI-PARC transport seed topology drifted.")
        if not np.allclose(
            np.mean(seed, axis=0, dtype=np.float64),
            portfolio.astype(np.float64),
            rtol=0.0,
            atol=2.0e-7,
        ):
            raise ProtocolError("PCSI-PARC transport seed means do not reconstruct P.")
        portfolio_parts.append(portfolio)
        seed_sd_parts.append(np.std(seed, axis=0, ddof=0, dtype=np.float64))
        seed_hash_rows.append({"case_id": row.case_id, "array_sha256": sha256_array(seed)})
        for alternative in ALTERNATIVE_METHOD_IDS:
            endpoint_parts[alternative].append(
                as_binary32(row.probabilities[alternative], name="transport alternative")
            )
    portfolio = np.concatenate(portfolio_parts).astype(np.float64, copy=False)
    p_hard = portfolio >= float(THRESHOLD)
    clipped = np.clip(portfolio, 1.0e-12, 1.0 - 1.0e-12)
    values: list[float] = [
        float(np.log1p(len(portfolio))),
        float(np.mean(portfolio, dtype=np.float64)),
        float(np.std(portfolio, ddof=0, dtype=np.float64)),
        float(np.mean(np.abs(portfolio - float(THRESHOLD)), dtype=np.float64)),
        float(
            np.mean(
                -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped)),
                dtype=np.float64,
            )
        ),
        float(np.mean(np.concatenate(seed_sd_parts), dtype=np.float64)),
    ]
    for alternative in ALTERNATIVE_METHOD_IDS:
        candidate = np.concatenate(endpoint_parts[alternative]).astype(np.float64, copy=False)
        a_hard = candidate >= float(THRESHOLD)
        for direction in DIRECTION_IDS:
            crossing = (
                (~p_hard) & a_hard
                if direction == DIRECTION_IDS[0]
                else p_hard & (~a_hard)
            )
            values.append(float(np.mean(crossing.astype(np.float64), dtype=np.float64)))
    endpoint_hash = canonical_hash([row.prediction_hash for row in rows])
    return CenterTransportDescriptor(
        lineage,
        center,
        TRANSPORT_FEATURE_NAMES,
        tuple(values),
        len(portfolio),
        len(rows),
        endpoint_hash,
        canonical_hash(seed_hash_rows),
    )


def _distance(
    candidate: np.ndarray,
    references: np.ndarray,
) -> float:
    if references.ndim != 2 or len(references) < 2 or references.shape[1] != len(candidate):
        raise ProtocolError("PCSI-PARC transport distance reference drifted.")
    median = np.median(references, axis=0)
    mad = np.median(np.abs(references - median), axis=0)
    scale = np.maximum(TRANSPORT_MAD_SCALE * mad, TRANSPORT_SCALE_FLOOR)
    value = float(np.max(np.abs(candidate - median) / scale))
    if not math.isfinite(value):
        raise ProtocolError("PCSI-PARC transport distance is nonfinite.")
    return value


def evaluate_transport_screen(
    candidate: CenterTransportDescriptor,
    references: Sequence[CenterTransportDescriptor],
) -> TransportScreen:
    rows = tuple(references)
    centers = tuple(row.center for row in rows)
    outer = candidate.lineage.outer_target_center
    if (
        len(rows) < TRANSPORT_MIN_REFERENCE_CENTER_COUNT
        or candidate.center in centers
        or len(centers) != len(set(centers))
        or any(row.feature_names != candidate.feature_names for row in rows)
        or any(row.lineage.outer_target_center != outer for row in rows)
    ):
        raise ProtocolError("PCSI-PARC transport reference set drifted.")
    matrix = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    candidate_values = np.asarray(candidate.feature_values, dtype=np.float64)
    distance = _distance(candidate_values, matrix)
    leave_one = tuple(
        (
            row.center,
            _distance(
                np.asarray(row.feature_values, dtype=np.float64),
                np.delete(matrix, index, axis=0),
            ),
        )
        for index, row in enumerate(rows)
    )
    threshold = max(value for _center, value in leave_one)
    return TransportScreen(
        outer,
        candidate.center,
        centers,
        distance,
        threshold,
        leave_one,
        bool(distance <= threshold),
        candidate.descriptor_hash,
        tuple(row.descriptor_hash for row in rows),
        candidate.lineage.lineage_hash,
        tuple(row.lineage.lineage_hash for row in rows),
    )


def build_transport_screen_matrix(
    descriptors: Mapping[str, CenterTransportDescriptor],
) -> Mapping[tuple[str, str | None], TransportScreen]:
    """Return target screens and every H/J double-excluded pseudo screen."""

    if set(descriptors) != set(CENTERS):
        raise ProtocolError("PCSI-PARC transport matrix lacks a center.")
    output: dict[tuple[str, str | None], TransportScreen] = {}
    for outer in CENTERS:
        target_references = tuple(descriptors[center] for center in CENTERS if center != outer)
        output[(outer, None)] = evaluate_transport_screen(
            descriptors[outer], target_references
        )
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            references = tuple(
                descriptors[center]
                for center in CENTERS
                if center not in {outer, pseudo}
            )
            output[(outer, pseudo)] = evaluate_transport_screen(
                descriptors[pseudo], references
            )
    return MappingProxyType(output)


def seal_transport_runtime(
    descriptors: Mapping[tuple[str, str], CenterTransportDescriptor],
    screens: Mapping[tuple[str, str | None], TransportScreen],
) -> TransportRuntimeSeal:
    """Seal the complete 81-descriptor/81-screen matrix before label replay."""

    expected_descriptors = {
        (outer, candidate)
        for outer in CENTERS
        for candidate in CENTERS
    }
    expected_screens = {
        (outer, candidate)
        for outer in CENTERS
        for candidate in (None, *(center for center in CENTERS if center != outer))
    }
    if set(descriptors) != expected_descriptors or set(screens) != expected_screens:
        raise ProtocolError("PCSI-PARC transport seal matrix drifted.")
    if any(
        descriptor.lineage.outer_target_center != outer
        or descriptor.lineage.endpoint_target_center != candidate
        for (outer, candidate), descriptor in descriptors.items()
    ):
        raise ProtocolError("PCSI-PARC transport descriptor lineage escaped its key.")
    if any(
        screen.outer_target_center != outer
        or screen.candidate_center != (outer if candidate is None else candidate)
        for (outer, candidate), screen in screens.items()
    ):
        raise ProtocolError("PCSI-PARC transport screen lineage escaped its key.")
    descriptor_hash = canonical_hash(
        [
            {
                "outer_target_center": outer,
                "endpoint_target_center": candidate,
                "descriptor_hash": descriptors[(outer, candidate)].descriptor_hash,
                "lineage_hash": descriptors[(outer, candidate)].lineage.lineage_hash,
            }
            for outer, candidate in sorted(descriptors)
        ]
    )
    screen_hash = canonical_hash(
        [
            {
                "outer_target_center": outer,
                "pseudo_target_center": candidate,
                "screen_hash": screens[(outer, candidate)].screen_hash,
            }
            for outer, candidate in sorted(
                screens,
                key=lambda key: (
                    CENTERS.index(key[0]),
                    -1 if key[1] is None else CENTERS.index(key[1]),
                ),
            )
        ]
    )
    return TransportRuntimeSeal(
        len(descriptors),
        len(screens),
        descriptor_hash,
        screen_hash,
    )


__all__ = (
    "CenterTransportDescriptor",
    "LEGACY_TRANSPORT_PROTOCOL_FIELDS",
    "TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE",
    "TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE",
    "TRANSPORT_ENDPOINT_SUPPORT_SCOPE",
    "TRANSPORT_PROTOCOL_CONTRACT",
    "TRANSPORT_SEMANTICS",
    "TransportEndpointLineage",
    "TransportRuntimeSeal",
    "TransportScreen",
    "build_center_transport_descriptor",
    "build_transport_screen_matrix",
    "evaluate_transport_screen",
    "seal_transport_runtime",
)
