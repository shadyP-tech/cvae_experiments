"""Immutable donor-prior contracts for SCALE-BP."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .hashing import canonical_hash, require_sha256
from .identity import ACTION_IDS, CENTERS, RIDGE_ALPHA
from .influence.contracts import ActionDescriptor, ActionMetricVector, MetricStandardError
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class DonorObservation:
    """One pseudo-held case/action response from a legal donor center."""

    query_center: str
    case_id: str
    source_centers: tuple[str, ...]
    descriptor: ActionDescriptor
    realized: ActionMetricVector
    scope_hash: str
    observation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        sources = tuple(str(value) for value in self.source_centers)
        if (
            self.query_center not in CENTERS
            or not self.case_id
            or self.case_id != self.descriptor.case_id
            or not sources
            or len(sources) != len(set(sources))
            or self.query_center in sources
            or any(source not in CENTERS for source in sources)
        ):
            raise ProtocolError("SCALE-BP donor observation scope drifted.")
        scope_hash = require_sha256(self.scope_hash, "donor observation scope hash")
        object.__setattr__(self, "source_centers", sources)
        object.__setattr__(self, "scope_hash", scope_hash)
        object.__setattr__(
            self,
            "observation_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_donor_observation_v1",
                    "query_center": self.query_center,
                    "case_id": self.case_id,
                    "source_centers": sources,
                    "descriptor_hash": self.descriptor.descriptor_hash,
                    "realized": self.realized.to_payload(),
                    "scope_hash": scope_hash,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @property
    def cell_id(self) -> str:
        return f"{self.descriptor.family}::{self.descriptor.direction}"


@dataclass(frozen=True, slots=True)
class DonorDeleteCenterFold:
    """Fit-specific rows with the deleted center also source-excluded."""

    deleted_center: str
    training_observations: tuple[DonorObservation, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        deleted = str(self.deleted_center)
        rows = tuple(self.training_observations)
        if (
            deleted not in CENTERS
            or not rows
            or any(not isinstance(row, DonorObservation) for row in rows)
            or any(
                row.query_center == deleted or deleted in row.source_centers
                for row in rows
            )
            or len({row.observation_hash for row in rows}) != len(rows)
        ):
            raise ProtocolError("SCALE-BP delete-center donor fold drifted.")
        object.__setattr__(self, "deleted_center", deleted)
        object.__setattr__(self, "training_observations", rows)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_donor_delete_center_fold_v1",
                    "deleted_center": deleted,
                    "training_observation_hashes": tuple(
                        sorted(row.observation_hash for row in rows)
                    ),
                    "deleted_center_excluded_from_queries": True,
                    "deleted_center_excluded_from_candidate_sources": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DonorPriorModel:
    held_center: str
    scope_hash: str
    fit_role: str
    training_centers: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    between_center_standard_error: MetricStandardError
    training_case_count: int
    training_row_count: int
    delete_center_fold_hashes: tuple[str, ...]
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.training_centers)
        cases = tuple(str(value) for value in self.training_case_ids)
        names = tuple(str(value) for value in self.feature_names)
        mean = tuple(float(value) for value in self.feature_mean)
        scale = tuple(float(value) for value in self.feature_scale)
        coefficients = tuple(
            tuple(float(value) for value in row) for row in self.coefficients
        )
        width = len(ACTION_IDS) + len(names)
        scope_hash = require_sha256(self.scope_hash, "donor model scope hash")
        fold_hashes = tuple(str(value) for value in self.delete_center_fold_hashes)
        if (
            self.held_center not in CENTERS
            or self.held_center in centers
            or len(centers) < 3
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or not cases
            or cases != tuple(sorted(set(cases)))
            or self.fit_role not in {"FINAL_H_C", "PSEUDO_H_J_D"}
            or not names
            or len(names) != len(set(names))
            or len(mean) != len(names)
            or len(scale) != len(names)
            or any(value <= 0.0 or not math.isfinite(value) for value in scale)
            or len(coefficients) != 3
            or any(len(row) != width for row in coefficients)
            or any(not math.isfinite(value) for row in coefficients for value in row)
            or self.training_case_count <= 0
            or self.training_row_count < self.training_case_count
            or len(fold_hashes) != len(centers)
            or len(set(fold_hashes)) != len(fold_hashes)
        ):
            raise ProtocolError("SCALE-BP donor prior model drifted.")
        for digest in fold_hashes:
            require_sha256(digest, "donor delete-center fold hash")
        object.__setattr__(self, "scope_hash", scope_hash)
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "delete_center_fold_hashes", fold_hashes)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_donor_prior_model_v1",
                    "held_center": self.held_center,
                    "scope_hash": scope_hash,
                    "fit_role": self.fit_role,
                    "training_centers": list(centers),
                    "training_case_ids": list(cases),
                    "cell_ids": list(ACTION_IDS),
                    "feature_names": list(names),
                    "feature_mean": list(mean),
                    "feature_scale": list(scale),
                    "coefficients": [list(row) for row in coefficients],
                    "between_center_standard_error": (
                        self.between_center_standard_error.to_payload()
                    ),
                    "training_case_count": self.training_case_count,
                    "training_row_count": self.training_row_count,
                    "delete_center_fold_hashes": fold_hashes,
                    "ridge_alpha": RIDGE_ALPHA,
                    "equal_total_weight_per_center": True,
                    "equal_total_weight_per_case": True,
                    "target_support_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DonorPriorPrediction:
    descriptor_hash: str
    mean: ActionMetricVector
    between_center_standard_error: MetricStandardError
    model_hash: str
    scope_hash: str
    fit_role: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.descriptor_hash, "descriptor hash")
        require_sha256(self.model_hash, "donor model hash")
        require_sha256(self.scope_hash, "donor prediction scope hash")
        if self.fit_role not in {"FINAL_H_C", "PSEUDO_H_J_D"}:
            raise ProtocolError("SCALE-BP donor prediction fit role drifted.")
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_donor_prior_prediction_v1",
                    "descriptor_hash": self.descriptor_hash,
                    "mean": self.mean.to_payload(),
                    "between_center_standard_error": (
                        self.between_center_standard_error.to_payload()
                    ),
                    "model_hash": self.model_hash,
                    "scope_hash": self.scope_hash,
                    "fit_role": self.fit_role,
                }
            ),
        )


__all__ = (
    "DonorDeleteCenterFold",
    "DonorObservation",
    "DonorPriorModel",
    "DonorPriorPrediction",
)
