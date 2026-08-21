"""Workstation-batched donor response and utility-model runtime.

The expensive endpoint states arrive already reconstructed. This module
materializes the projected and raw action geometries once, builds direct donor
responses, and fits each complete outer-center family in a BLAS-limited worker.
Projected and unprojected rows, models, and hashes never share an identity.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    LEGACY_GEOMETRY_ID,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
    UTILITY_BLAS_THREADS_PER_WORKER,
    UTILITY_CPU_WORKERS,
    candidate_sources,
)
from .contracts import BinaryLabel, EndpointCasePrediction
from .hashing import canonical_hash
from .outer_endpoint_runtime import OuterEndpointProducts
from .projected_contracts import (
    ProjectedDonorUtilityRow,
    ProjectedUtilityDescriptor,
    ProjectedUtilityModel,
    ProjectedUtilityPrediction,
)
from .projected_features import build_projected_descriptors
from .projected_model import fit_projected_model_family
from .projected_responses import build_projected_donor_rows
from .projected_uncertainty import predict_projected_surface
from .projection import ActionEquivalenceClass, build_action_equivalence_classes
from .uncertainty import predict_utility_surface
from .utility_contracts import (
    DonorUtilityRow,
    SignedUtilityModel,
    UtilityDescriptor,
    UtilityPrediction,
)
from .utility_features import build_utility_descriptor_surface
from .utility_model import fit_response_model_family
from .utility_responses import build_donor_utility_rows
from .workstation import BLAS_ENVIRONMENT_NAMES


PARC_GEOMETRIES = (PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID)
EXPECTED_PSEUDO_DONOR_SCOPE_COUNT = len(CENTERS) * (len(CENTERS) - 1) * (
    len(CENTERS) - 2
)
_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class DoubleExcludedDonorPriorProvenance:
    """Hash-bound q scopes for one ordered ``(H, J, K)`` prior rebind."""

    outer_target_center: str
    pseudo_target_center: str
    donor_center: str
    query_centers_by_source: tuple[tuple[str, tuple[str, ...]], ...]
    prior_values: Mapping[tuple[str, str], float]
    prior_hash: str

    def __post_init__(self) -> None:
        outer = str(self.outer_target_center)
        pseudo = str(self.pseudo_target_center)
        donor = str(self.donor_center)
        sources = candidate_sources(donor) if donor in CENTERS else ()
        query_rows = tuple(
            (str(source), tuple(str(center) for center in centers))
            for source, centers in self.query_centers_by_source
        )
        values = {
            (str(source), str(direction)): float(value)
            for (source, direction), value in self.prior_values.items()
        }
        expected_values = tuple(
            (source, direction)
            for source in sources
            for direction in DIRECTION_IDS
        )
        if (
            outer not in CENTERS
            or pseudo not in CENTERS
            or donor not in CENTERS
            or len({outer, pseudo, donor}) != 3
            or tuple(source for source, _centers in query_rows) != sources
            or any(
                centers
                != tuple(
                    center
                    for center in CENTERS
                    if center not in {outer, pseudo, donor, source}
                )
                for source, centers in query_rows
            )
            or tuple(values) != expected_values
            or any(not math.isfinite(value) for value in values.values())
        ):
            raise ProtocolError("PCSI-PARC double-excluded prior provenance drifted.")
        payload = {
            "schema_version": "fixed_bank_pcsi_parc_double_excluded_prior_v1",
            "outer_target_center": outer,
            "pseudo_target_center": pseudo,
            "donor_center": donor,
            "query_centers_by_source": [
                {"source_center": source, "query_centers": list(centers)}
                for source, centers in query_rows
            ],
            "prior_values": [
                [source, direction, values[(source, direction)]]
                for source in sources
                for direction in DIRECTION_IDS
            ],
            "outer_H_excluded_from_every_query_role": True,
            "pseudo_J_excluded_from_every_query_role": True,
            "endpoint_target_K_excluded_from_every_query_role": True,
            "candidate_source_e_excluded_from_its_query_role": True,
            "pseudo_donor_feature_source_prior_scope": (
                "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
            ),
            "raw_labels_persisted": False,
        }
        if canonical_hash(payload) != self.prior_hash:
            raise ProtocolError("PCSI-PARC double-excluded prior hash drifted.")
        object.__setattr__(self, "outer_target_center", outer)
        object.__setattr__(self, "pseudo_target_center", pseudo)
        object.__setattr__(self, "donor_center", donor)
        object.__setattr__(self, "query_centers_by_source", query_rows)
        object.__setattr__(self, "prior_values", MappingProxyType(values))

    @classmethod
    def create(
        cls,
        *,
        outer_target_center: str,
        pseudo_target_center: str,
        donor_center: str,
        query_centers_by_source: Sequence[tuple[str, Sequence[str]]],
        prior_values: Mapping[tuple[str, str], float],
    ) -> "DoubleExcludedDonorPriorProvenance":
        query_rows = tuple(
            (str(source), tuple(str(center) for center in centers))
            for source, centers in query_centers_by_source
        )
        values = {
            (str(source), str(direction)): float(value)
            for (source, direction), value in prior_values.items()
        }
        payload = {
            "schema_version": "fixed_bank_pcsi_parc_double_excluded_prior_v1",
            "outer_target_center": str(outer_target_center),
            "pseudo_target_center": str(pseudo_target_center),
            "donor_center": str(donor_center),
            "query_centers_by_source": [
                {"source_center": source, "query_centers": list(centers)}
                for source, centers in query_rows
            ],
            "prior_values": [
                [source, direction, values[(source, direction)]]
                for source in candidate_sources(donor_center)
                for direction in DIRECTION_IDS
            ],
            "outer_H_excluded_from_every_query_role": True,
            "pseudo_J_excluded_from_every_query_role": True,
            "endpoint_target_K_excluded_from_every_query_role": True,
            "candidate_source_e_excluded_from_its_query_role": True,
            "pseudo_donor_feature_source_prior_scope": (
                "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
            ),
            "raw_labels_persisted": False,
        }
        return cls(
            str(outer_target_center),
            str(pseudo_target_center),
            str(donor_center),
            query_rows,
            MappingProxyType(values),
            canonical_hash(payload),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_double_excluded_prior_v1",
            "outer_target_center": self.outer_target_center,
            "pseudo_target_center": self.pseudo_target_center,
            "donor_center": self.donor_center,
            "query_centers_by_source": [
                {"source_center": source, "query_centers": list(centers)}
                for source, centers in self.query_centers_by_source
            ],
            "prior_values": [
                [source, direction, self.prior_values[(source, direction)]]
                for source in candidate_sources(self.donor_center)
                for direction in DIRECTION_IDS
            ],
            "outer_H_excluded_from_every_query_role": True,
            "pseudo_J_excluded_from_every_query_role": True,
            "endpoint_target_K_excluded_from_every_query_role": True,
            "candidate_source_e_excluded_from_its_query_role": True,
            "pseudo_donor_feature_source_prior_scope": (
                "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
            ),
            "raw_labels_persisted": False,
            "prior_hash": self.prior_hash,
        }


@dataclass(frozen=True)
class GeometryOuterFitProducts:
    geometry_id: str
    outer_target_center: str
    target_full_model: ProjectedUtilityModel
    target_delete_models: Mapping[str, ProjectedUtilityModel]
    target_predictions: tuple[ProjectedUtilityPrediction, ...]
    pseudo_full_models: Mapping[str, ProjectedUtilityModel]
    pseudo_delete_models: Mapping[str, Mapping[str, ProjectedUtilityModel]]
    pseudo_predictions: Mapping[str, tuple[ProjectedUtilityPrediction, ...]]
    model_fit_count: int


@dataclass(frozen=True)
class GeometryDonorRuntimeResult:
    geometry_id: str
    target_actions_by_center: Mapping[str, tuple[ActionEquivalenceClass, ...]]
    target_descriptors_by_center: Mapping[
        str, tuple[ProjectedUtilityDescriptor, ...]
    ]
    pseudo_actions_by_pair: Mapping[
        tuple[str, str], tuple[ActionEquivalenceClass, ...]
    ]
    pseudo_descriptors_by_pair: Mapping[
        tuple[str, str], tuple[ProjectedUtilityDescriptor, ...]
    ]
    donor_rows_by_outer: Mapping[str, tuple[ProjectedDonorUtilityRow, ...]]
    pseudo_donor_rows_by_pair: Mapping[
        tuple[str, str], tuple[ProjectedDonorUtilityRow, ...]
    ]
    target_full_models: Mapping[str, ProjectedUtilityModel]
    target_delete_models: Mapping[str, Mapping[str, ProjectedUtilityModel]]
    target_predictions_by_center: Mapping[
        str, tuple[ProjectedUtilityPrediction, ...]
    ]
    pseudo_full_models: Mapping[tuple[str, str], ProjectedUtilityModel]
    pseudo_delete_models: Mapping[
        tuple[str, str], Mapping[str, ProjectedUtilityModel]
    ]
    pseudo_predictions_by_pair: Mapping[
        tuple[str, str], tuple[ProjectedUtilityPrediction, ...]
    ]
    model_fit_count: int
    runtime_hash: str

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_geometry_runtime_v1",
            "geometry_id": self.geometry_id,
            "donor_row_count": sum(len(rows) for rows in self.donor_rows_by_outer.values()),
            "double_excluded_donor_row_count": sum(
                len(rows) for rows in self.pseudo_donor_rows_by_pair.values()
            ),
            "target_descriptor_count": sum(
                len(rows) for rows in self.target_descriptors_by_center.values()
            ),
            "pseudo_descriptor_count": sum(
                len(rows) for rows in self.pseudo_descriptors_by_pair.values()
            ),
            "target_family_count": len(self.target_full_models),
            "double_exclusion_family_count": len(self.pseudo_full_models),
            "model_fit_count": self.model_fit_count,
            "runtime_hash": self.runtime_hash,
        }


@dataclass(frozen=True)
class LegacyDonorRuntimeResult:
    descriptors_by_center: Mapping[str, tuple[UtilityDescriptor, ...]]
    donor_rows_by_outer: Mapping[str, tuple[DonorUtilityRow, ...]]
    full_models_by_outer: Mapping[str, SignedUtilityModel]
    delete_models_by_outer: Mapping[str, Mapping[str, SignedUtilityModel]]
    predictions_by_center: Mapping[str, tuple[UtilityPrediction, ...]]
    model_fit_count: int
    runtime_hash: str

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_fresh_legacy_runtime_v1",
            "geometry_id": LEGACY_GEOMETRY_ID,
            "donor_row_count": sum(len(rows) for rows in self.donor_rows_by_outer.values()),
            "descriptor_count": sum(len(rows) for rows in self.descriptors_by_center.values()),
            "model_fit_count": self.model_fit_count,
            "fresh_recomputation": True,
            "predecessor_artifact_used": False,
            "runtime_hash": self.runtime_hash,
        }


@dataclass(frozen=True)
class DonorRuntimeResult:
    geometry_results: Mapping[str, GeometryDonorRuntimeResult]
    legacy: LegacyDonorRuntimeResult
    pseudo_prior_provenance: Mapping[
        tuple[str, str, str], DoubleExcludedDonorPriorProvenance
    ]
    pseudo_donor_endpoint_products: Mapping[
        tuple[str, str, str], OuterEndpointProducts
    ]
    model_fit_count: int
    runtime_hash: str

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_donor_runtime_v1",
            "geometries": [
                self.geometry_results[geometry].summary_payload()
                for geometry in PARC_GEOMETRIES
            ],
            "legacy": self.legacy.summary_payload(),
            "double_excluded_prior_scope_count": len(self.pseudo_prior_provenance),
            "double_excluded_endpoint_scope_count": len(
                self.pseudo_donor_endpoint_products
            ),
            "model_fit_count": self.model_fit_count,
            "expected_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
            "runtime_hash": self.runtime_hash,
        }


@dataclass(frozen=True)
class _GeometryFitJob:
    geometry_id: str
    outer_target_center: str
    donor_rows: tuple[ProjectedDonorUtilityRow, ...]
    target_descriptors: tuple[ProjectedUtilityDescriptor, ...]
    pseudo_states: tuple[
        tuple[
            str,
            tuple[ProjectedDonorUtilityRow, ...],
            tuple[ProjectedUtilityDescriptor, ...],
        ],
        ...,
    ]


@dataclass(frozen=True)
class _LegacyFitProducts:
    outer_target_center: str
    full_model: SignedUtilityModel
    delete_models: Mapping[str, SignedUtilityModel]
    predictions: tuple[UtilityPrediction, ...]
    model_fit_count: int


@dataclass(frozen=True)
class _LegacyFitJob:
    outer_target_center: str
    donor_rows: tuple[DonorUtilityRow, ...]
    target_descriptors: tuple[UtilityDescriptor, ...]


def build_donor_runtime(
    *,
    predictions_by_center: Mapping[str, Sequence[EndpointCasePrediction]],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    pseudo_prior_provenance: Mapping[
        tuple[str, str, str], DoubleExcludedDonorPriorProvenance
    ],
    pseudo_donor_endpoint_products: Mapping[
        tuple[str, str, str], OuterEndpointProducts
    ],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
    use_processes: bool = True,
    strict_canonical_topology: bool = True,
) -> DonorRuntimeResult:
    """Build both PARC geometries and a fresh 81-fit legacy control."""

    expected_pairs = {
        (outer, donor)
        for outer in CENTERS
        for donor in CENTERS
        if donor != outer
    }
    expected_triples = {
        (outer, pseudo, donor)
        for outer in CENTERS
        for pseudo in CENTERS
        for donor in CENTERS
        if len({outer, pseudo, donor}) == 3
    }
    if (
        set(predictions_by_center) != set(CENTERS)
        or set(donor_endpoint_products) != expected_pairs
        or set(pseudo_prior_provenance) != expected_triples
        or set(pseudo_donor_endpoint_products) != expected_triples
        or set(donor_labels) != expected_pairs
    ):
        raise ProtocolError("PCSI-PARC donor runtime input matrix drifted.")
    _validate_pseudo_endpoint_scopes(
        pseudo_prior_provenance, pseudo_donor_endpoint_products
    )

    geometries = {
        geometry: _build_geometry_runtime(
            geometry_id=geometry,
            predictions_by_center=predictions_by_center,
            donor_endpoint_products=donor_endpoint_products,
            pseudo_donor_endpoint_products=pseudo_donor_endpoint_products,
            donor_labels=donor_labels,
            use_processes=use_processes,
            strict_canonical_topology=strict_canonical_topology,
        )
        for geometry in PARC_GEOMETRIES
    }
    legacy = _build_legacy_runtime(
        predictions_by_center=predictions_by_center,
        donor_endpoint_products=donor_endpoint_products,
        donor_labels=donor_labels,
        use_processes=use_processes,
        strict_canonical_topology=strict_canonical_topology,
    )
    fit_count = legacy.model_fit_count + sum(
        row.model_fit_count for row in geometries.values()
    )
    if strict_canonical_topology and fit_count != EXPECTED_UTILITY_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI-PARC total utility-model workload drifted.")
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_donor_runtime_v1",
        "geometry_runtime_hashes": [
            geometries[geometry].runtime_hash for geometry in PARC_GEOMETRIES
        ],
        "legacy_runtime_hash": legacy.runtime_hash,
        "double_excluded_prior_hash": canonical_hash(
            [
                pseudo_prior_provenance[key].prior_hash
                for key in sorted(pseudo_prior_provenance)
            ]
        ),
        "double_excluded_endpoint_hash": _pseudo_endpoint_products_hash(
            pseudo_prior_provenance, pseudo_donor_endpoint_products
        ),
        "model_fit_count": fit_count,
        "actual_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_training_donor_K_or_source_e"
        ),
        "pseudo_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
        ),
        "projected_and_unprojected_hashes_distinct": (
            geometries[PROJECTION_GEOMETRY_ID].runtime_hash
            != geometries[UNPROJECTED_GEOMETRY_ID].runtime_hash
        ),
        "raw_labels_persisted": False,
    }
    return DonorRuntimeResult(
        MappingProxyType(geometries),
        legacy,
        MappingProxyType(dict(pseudo_prior_provenance)),
        MappingProxyType(dict(pseudo_donor_endpoint_products)),
        fit_count,
        canonical_hash(payload),
    )


def _build_geometry_runtime(
    *,
    geometry_id: str,
    predictions_by_center: Mapping[str, Sequence[EndpointCasePrediction]],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    pseudo_donor_endpoint_products: Mapping[
        tuple[str, str, str], OuterEndpointProducts
    ],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
    use_processes: bool,
    strict_canonical_topology: bool,
) -> GeometryDonorRuntimeResult:
    target_actions: dict[str, tuple[ActionEquivalenceClass, ...]] = {}
    target_descriptors: dict[str, tuple[ProjectedUtilityDescriptor, ...]] = {}
    for center in CENTERS:
        actions, descriptors = _materialize_projected_surface(
            predictions_by_center[center], geometry_id=geometry_id
        )
        target_actions[center] = actions
        target_descriptors[center] = descriptors

    pseudo_actions: dict[tuple[str, str], tuple[ActionEquivalenceClass, ...]] = {}
    pseudo_descriptors: dict[
        tuple[str, str], tuple[ProjectedUtilityDescriptor, ...]
    ] = {}
    donor_rows: dict[str, tuple[ProjectedDonorUtilityRow, ...]] = {}
    for outer in CENTERS:
        outer_rows: list[ProjectedDonorUtilityRow] = []
        for donor in CENTERS:
            if donor == outer:
                continue
            pair = outer, donor
            products = donor_endpoint_products[pair]
            actions, descriptors = _materialize_projected_surface(
                products.predictions, geometry_id=geometry_id
            )
            pseudo_actions[pair] = actions
            pseudo_descriptors[pair] = descriptors
            labels = tuple(donor_labels[pair])
            by_case = _labels_by_case(labels)
            n_positive = sum(row.value == 1 for row in labels)
            n_negative = sum(row.value == 0 for row in labels)
            actions_by_case = _rows_by_case(actions)
            descriptors_by_case = _rows_by_case(descriptors)
            for prediction in products.predictions:
                outer_rows.extend(
                    build_projected_donor_rows(
                        outer_target_center=outer,
                        prediction=prediction,
                        actions=actions_by_case[prediction.case_id],
                        descriptors=descriptors_by_case[prediction.case_id],
                        case_labels=by_case[prediction.case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                )
        donor_rows[outer] = tuple(sorted(outer_rows, key=lambda row: row.key))

    pseudo_donor_rows: dict[
        tuple[str, str], tuple[ProjectedDonorUtilityRow, ...]
    ] = {}
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            pair_rows: list[ProjectedDonorUtilityRow] = []
            for donor in CENTERS:
                if donor in {outer, pseudo}:
                    continue
                products = pseudo_donor_endpoint_products[(outer, pseudo, donor)]
                actions, descriptors = _materialize_projected_surface(
                    products.predictions, geometry_id=geometry_id
                )
                labels = tuple(donor_labels[(outer, donor)])
                labels_by_case = _labels_by_case(labels)
                n_positive = sum(row.value == 1 for row in labels)
                n_negative = sum(row.value == 0 for row in labels)
                actions_by_case = _rows_by_case(actions)
                descriptors_by_case = _rows_by_case(descriptors)
                for prediction in products.predictions:
                    pair_rows.extend(
                        build_projected_donor_rows(
                            outer_target_center=outer,
                            prediction=prediction,
                            actions=actions_by_case[prediction.case_id],
                            descriptors=descriptors_by_case[prediction.case_id],
                            case_labels=labels_by_case[prediction.case_id],
                            center_n_positive=n_positive,
                            center_n_negative=n_negative,
                        )
                    )
            training_centers = {
                center for center in CENTERS if center not in {outer, pseudo}
            }
            if {row.donor_center for row in pair_rows} != training_centers:
                raise ProtocolError(
                    "PCSI-PARC H/J donor rows escaped the double exclusion."
                )
            pseudo_donor_rows[(outer, pseudo)] = tuple(
                sorted(pair_rows, key=lambda row: row.key)
            )

    jobs = tuple(
        _GeometryFitJob(
            geometry_id,
            outer,
            donor_rows[outer],
            target_descriptors[outer],
            tuple(
                (
                    pseudo,
                    pseudo_donor_rows[(outer, pseudo)],
                    pseudo_descriptors[(outer, pseudo)],
                )
                for pseudo in CENTERS
                if pseudo != outer
            ),
        )
        for outer in CENTERS
    )
    products = _execute_geometry_fit_jobs(jobs, use_processes=use_processes)
    by_outer = {row.outer_target_center: row for row in products}
    fit_count = sum(row.model_fit_count for row in products)
    if (
        strict_canonical_topology
        and fit_count != EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY
    ):
        raise ProtocolError("PCSI-PARC geometry model workload drifted.")

    target_full = {outer: by_outer[outer].target_full_model for outer in CENTERS}
    target_deleted = {
        outer: by_outer[outer].target_delete_models for outer in CENTERS
    }
    target_predictions = {
        outer: by_outer[outer].target_predictions for outer in CENTERS
    }
    pseudo_full = {
        (outer, pseudo): by_outer[outer].pseudo_full_models[pseudo]
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
    }
    pseudo_deleted = {
        (outer, pseudo): by_outer[outer].pseudo_delete_models[pseudo]
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
    }
    pseudo_predictions = {
        (outer, pseudo): by_outer[outer].pseudo_predictions[pseudo]
        for outer in CENTERS
        for pseudo in CENTERS
        if pseudo != outer
    }
    for outer in CENTERS:
        for pseudo in CENTERS:
            if pseudo == outer:
                continue
            training = tuple(
                center for center in CENTERS if center not in {outer, pseudo}
            )
            if (
                pseudo_full[(outer, pseudo)].training_centers != training
                or tuple(pseudo_deleted[(outer, pseudo)]) != training
                or any(
                    model.training_centers
                    != tuple(center for center in training if center != deleted)
                    for deleted, model in pseudo_deleted[(outer, pseudo)].items()
                )
            ):
                raise ProtocolError(
                    "PCSI-PARC pseudo family did not preserve exact H/J exclusion."
                )
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_geometry_runtime_v1",
        "geometry_id": geometry_id,
        "target_action_hash": canonical_hash(
            [row.action_hash for center in CENTERS for row in target_actions[center]]
        ),
        "pseudo_action_hash": canonical_hash(
            [
                row.action_hash
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
                for row in pseudo_actions[(outer, pseudo)]
            ]
        ),
        "donor_response_hash": canonical_hash(
            [
                row.to_payload()
                for outer in CENTERS
                for row in donor_rows[outer]
            ]
        ),
        "double_excluded_donor_response_hash": canonical_hash(
            [
                {
                    "outer_target_center": outer,
                    "pseudo_target_center": pseudo,
                    "row_payloads": [
                        row.to_payload()
                        for row in pseudo_donor_rows[(outer, pseudo)]
                    ],
                }
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
            ]
        ),
        "target_model_hash": canonical_hash(
            [
                model.model_hash
                for outer in CENTERS
                for model in (
                    target_full[outer],
                    *target_deleted[outer].values(),
                )
            ]
        ),
        "double_exclusion_model_hash": canonical_hash(
            [
                model.model_hash
                for outer in CENTERS
                for pseudo in CENTERS
                if pseudo != outer
                for model in (
                    pseudo_full[(outer, pseudo)],
                    *pseudo_deleted[(outer, pseudo)].values(),
                )
            ]
        ),
        "model_fit_count": fit_count,
        "raw_labels_persisted": False,
    }
    return GeometryDonorRuntimeResult(
        geometry_id,
        MappingProxyType(target_actions),
        MappingProxyType(target_descriptors),
        MappingProxyType(pseudo_actions),
        MappingProxyType(pseudo_descriptors),
        MappingProxyType(donor_rows),
        MappingProxyType(pseudo_donor_rows),
        MappingProxyType(target_full),
        MappingProxyType(target_deleted),
        MappingProxyType(target_predictions),
        MappingProxyType(pseudo_full),
        MappingProxyType(pseudo_deleted),
        MappingProxyType(pseudo_predictions),
        fit_count,
        canonical_hash(payload),
    )


def _fit_geometry_outer(job: _GeometryFitJob) -> GeometryOuterFitProducts:
    outer = job.outer_target_center
    target_training = tuple(center for center in CENTERS if center != outer)
    target_full, target_deleted = fit_projected_model_family(
        job.donor_rows,
        outer_target_center=outer,
        geometry_id=job.geometry_id,
        training_centers=target_training,
    )
    target_predictions = predict_projected_surface(
        job.target_descriptors,
        donor_rows=job.donor_rows,
        full_model=target_full,
        delete_models=target_deleted,
        candidate_center=outer,
    )
    pseudo_full: dict[str, ProjectedUtilityModel] = {}
    pseudo_deleted: dict[str, Mapping[str, ProjectedUtilityModel]] = {}
    pseudo_predictions: dict[str, tuple[ProjectedUtilityPrediction, ...]] = {}
    for pseudo, pseudo_rows, descriptors in job.pseudo_states:
        training = tuple(
            center for center in CENTERS if center not in {outer, pseudo}
        )
        if {row.donor_center for row in pseudo_rows} != set(training):
            raise ProtocolError("PCSI-PARC pseudo fit received a leaked donor row.")
        full, deleted = fit_projected_model_family(
            pseudo_rows,
            outer_target_center=outer,
            geometry_id=job.geometry_id,
            training_centers=training,
        )
        pseudo_full[pseudo] = full
        pseudo_deleted[pseudo] = deleted
        pseudo_predictions[pseudo] = predict_projected_surface(
            descriptors,
            donor_rows=pseudo_rows,
            full_model=full,
            delete_models=deleted,
            candidate_center=pseudo,
        )
    fit_count = 1 + len(target_deleted) + sum(
        1 + len(rows) for rows in pseudo_deleted.values()
    )
    return GeometryOuterFitProducts(
        job.geometry_id,
        outer,
        target_full,
        target_deleted,
        target_predictions,
        MappingProxyType(pseudo_full),
        MappingProxyType(pseudo_deleted),
        MappingProxyType(pseudo_predictions),
        fit_count,
    )


def _execute_geometry_fit_jobs(
    jobs: Sequence[_GeometryFitJob], *, use_processes: bool
) -> tuple[GeometryOuterFitProducts, ...]:
    rows = tuple(jobs)
    if tuple(row.outer_target_center for row in rows) != CENTERS:
        raise ProtocolError("PCSI-PARC geometry fit job order drifted.")
    if not use_processes:
        results = tuple(_fit_geometry_outer(row) for row in rows)
    else:
        with ProcessPoolExecutor(
            max_workers=UTILITY_CPU_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(UTILITY_BLAS_THREADS_PER_WORKER,),
        ) as executor:
            unordered = tuple(executor.map(_fit_geometry_outer, rows, chunksize=1))
        indexed = {row.outer_target_center: row for row in unordered}
        results = tuple(indexed[center] for center in CENTERS)
    return results


def _build_legacy_runtime(
    *,
    predictions_by_center: Mapping[str, Sequence[EndpointCasePrediction]],
    donor_endpoint_products: Mapping[tuple[str, str], OuterEndpointProducts],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
    use_processes: bool,
    strict_canonical_topology: bool,
) -> LegacyDonorRuntimeResult:
    descriptors = {
        center: build_utility_descriptor_surface(predictions_by_center[center])
        for center in CENTERS
    }
    donor_rows: dict[str, tuple[DonorUtilityRow, ...]] = {}
    for outer in CENTERS:
        rows: list[DonorUtilityRow] = []
        for donor in CENTERS:
            if donor == outer:
                continue
            pair = outer, donor
            products = donor_endpoint_products[pair]
            donor_descriptors = build_utility_descriptor_surface(products.predictions)
            descriptor_by_case = _rows_by_case(donor_descriptors)
            scoped = tuple(
                BinaryLabel(
                    row.center,
                    row.case_id,
                    row.sample_id,
                    row.value,
                    f"crossing_donor::outer_H={outer}::donor_J={donor}",
                )
                for row in donor_labels[pair]
            )
            labels_by_case = _labels_by_case(scoped)
            n_positive = sum(row.value == 1 for row in scoped)
            n_negative = sum(row.value == 0 for row in scoped)
            for prediction in products.predictions:
                rows.extend(
                    build_donor_utility_rows(
                        outer_target_center=outer,
                        prediction=prediction,
                        descriptors=descriptor_by_case[prediction.case_id],
                        case_labels=labels_by_case[prediction.case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                )
        donor_rows[outer] = tuple(sorted(rows, key=lambda row: row.key))
    jobs = tuple(
        _LegacyFitJob(outer, donor_rows[outer], descriptors[outer])
        for outer in CENTERS
    )
    if not use_processes:
        products = tuple(_fit_legacy_outer(job) for job in jobs)
    else:
        with ProcessPoolExecutor(
            max_workers=UTILITY_CPU_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(UTILITY_BLAS_THREADS_PER_WORKER,),
        ) as executor:
            unordered = tuple(executor.map(_fit_legacy_outer, jobs, chunksize=1))
        by_outer = {row.outer_target_center: row for row in unordered}
        products = tuple(by_outer[center] for center in CENTERS)
    by_outer = {row.outer_target_center: row for row in products}
    fit_count = sum(row.model_fit_count for row in products)
    if strict_canonical_topology and fit_count != EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI-PARC fresh legacy workload drifted.")
    full = {outer: by_outer[outer].full_model for outer in CENTERS}
    deleted = {outer: by_outer[outer].delete_models for outer in CENTERS}
    predictions = {outer: by_outer[outer].predictions for outer in CENTERS}
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_fresh_legacy_runtime_v1",
        "geometry_id": LEGACY_GEOMETRY_ID,
        "donor_response_hash": canonical_hash(
            [row.to_payload() for outer in CENTERS for row in donor_rows[outer]]
        ),
        "model_hash": canonical_hash(
            [
                model.model_hash
                for outer in CENTERS
                for model in (full[outer], *deleted[outer].values())
            ]
        ),
        "prediction_hash": canonical_hash(
            [
                row.prediction_hash
                for outer in CENTERS
                for row in predictions[outer]
            ]
        ),
        "model_fit_count": fit_count,
        "fresh_recomputation": True,
        "predecessor_artifact_used": False,
    }
    return LegacyDonorRuntimeResult(
        MappingProxyType(descriptors),
        MappingProxyType(donor_rows),
        MappingProxyType(full),
        MappingProxyType(deleted),
        MappingProxyType(predictions),
        fit_count,
        canonical_hash(payload),
    )


def _fit_legacy_outer(job: _LegacyFitJob) -> _LegacyFitProducts:
    full, deleted = fit_response_model_family(
        job.donor_rows, outer_target_center=job.outer_target_center
    )
    predictions = predict_utility_surface(
        job.target_descriptors,
        donor_rows=job.donor_rows,
        full_model=full,
        delete_models=deleted,
    )
    return _LegacyFitProducts(
        job.outer_target_center,
        full,
        deleted,
        predictions,
        1 + len(deleted),
    )


def _materialize_projected_surface(
    predictions: Sequence[EndpointCasePrediction], *, geometry_id: str
) -> tuple[
    tuple[ActionEquivalenceClass, ...],
    tuple[ProjectedUtilityDescriptor, ...],
]:
    actions: list[ActionEquivalenceClass] = []
    descriptors: list[ProjectedUtilityDescriptor] = []
    for prediction in predictions:
        case_actions = build_action_equivalence_classes(
            prediction, geometry_id=geometry_id
        )
        actions.extend(case_actions)
        descriptors.extend(build_projected_descriptors(prediction, case_actions))
    return tuple(actions), tuple(descriptors)


def _rows_by_case(rows: Sequence[object]) -> Mapping[str, tuple[object, ...]]:
    cases = tuple(dict.fromkeys(str(getattr(row, "case_id")) for row in rows))
    return MappingProxyType(
        {
            case: tuple(row for row in rows if str(getattr(row, "case_id")) == case)
            for case in cases
        }
    )


def _labels_by_case(
    labels: Sequence[BinaryLabel],
) -> Mapping[str, tuple[BinaryLabel, ...]]:
    rows = tuple(labels)
    cases = tuple(dict.fromkeys(row.case_id for row in rows))
    return MappingProxyType(
        {case: tuple(row for row in rows if row.case_id == case) for case in cases}
    )


def _validate_pseudo_endpoint_scopes(
    provenance: Mapping[
        tuple[str, str, str], DoubleExcludedDonorPriorProvenance
    ],
    products: Mapping[tuple[str, str, str], OuterEndpointProducts],
) -> None:
    if (
        len(provenance) != EXPECTED_PSEUDO_DONOR_SCOPE_COUNT
        or set(provenance) != set(products)
    ):
        raise ProtocolError("PCSI-PARC double-excluded endpoint matrix drifted.")
    for key in sorted(provenance):
        outer, pseudo, donor = key
        prior = provenance[key]
        product = products[key]
        states = dict(product.states)
        query_centers = {
            center
            for _source, centers in prior.query_centers_by_source
            for center in centers
        }
        if (
            (
                prior.outer_target_center,
                prior.pseudo_target_center,
                prior.donor_center,
            )
            != key
            or product.target_center != donor
            or product.endpoint_model_fit_count != 0
            or not product.predictions
            or set(states) != {row.case_id for row in product.predictions}
            or tuple(product.state_hashes)
            != tuple((case, state.state_hash) for case, state in product.states)
            or any(
                state.target_center != donor
                or dict(state.donor_priors) != dict(prior.prior_values)
                for state in states.values()
            )
            or any(
                prediction.state_hash != states[prediction.case_id].state_hash
                for prediction in product.predictions
            )
            or outer in query_centers
            or pseudo in query_centers
        ):
            raise ProtocolError(
                "PCSI-PARC rebound endpoint is not bound to its H/J prior scope."
            )


def _pseudo_endpoint_products_hash(
    provenance: Mapping[
        tuple[str, str, str], DoubleExcludedDonorPriorProvenance
    ],
    products: Mapping[tuple[str, str, str], OuterEndpointProducts],
) -> str:
    _validate_pseudo_endpoint_scopes(provenance, products)
    return canonical_hash(
        [
            {
                "outer_target_center": outer,
                "pseudo_target_center": pseudo,
                "donor_center": donor,
                "prior_hash": provenance[(outer, pseudo, donor)].prior_hash,
                "state_hashes": [
                    list(row) for row in products[(outer, pseudo, donor)].state_hashes
                ],
                "prediction_hashes": [
                    row.prediction_hash
                    for row in products[(outer, pseudo, donor)].predictions
                ],
                "endpoint_model_fit_count": products[
                    (outer, pseudo, donor)
                ].endpoint_model_fit_count,
            }
            for outer, pseudo, donor in sorted(products)
        ]
    )


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("PCSI-PARC utility worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


__all__ = (
    "DoubleExcludedDonorPriorProvenance",
    "DonorRuntimeResult",
    "EXPECTED_PSEUDO_DONOR_SCOPE_COUNT",
    "GeometryDonorRuntimeResult",
    "GeometryOuterFitProducts",
    "LegacyDonorRuntimeResult",
    "PARC_GEOMETRIES",
    "build_donor_runtime",
)
