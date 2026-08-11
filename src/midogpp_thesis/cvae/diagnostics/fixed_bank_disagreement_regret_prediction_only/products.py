"""Small immutable phase products for the prediction-only runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    CandidateContrastRow,
    InferenceSelectionDiagnostic,
    DisagreementFeatureSurface,
    ExactRegretSurface,
)
from .experiment_contracts import CENTERS, GEOMETRY_IDS, MODEL_FAMILY_IDS
from .hashing import canonical_hash


@dataclass(frozen=True)
class FeatureSurfaceRecord:
    outer_target_id: str
    geometry_id: str
    family: str
    surface: DisagreementFeatureSurface

    def __post_init__(self) -> None:
        if (
            self.outer_target_id not in CENTERS
            or self.geometry_id not in GEOMETRY_IDS
            or self.family not in MODEL_FAMILY_IDS
            or self.surface.outer_target_id != self.outer_target_id
            or self.surface.family != self.family
        ):
            raise ProtocolError("Feature-surface record identity drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.outer_target_id, self.geometry_id, self.family)


@dataclass(frozen=True)
class ResponseSurfaceRecord:
    outer_target_id: str
    geometry_id: str
    surface: ExactRegretSurface

    def __post_init__(self) -> None:
        if self.outer_target_id not in CENTERS or self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Response-surface record identity drifted.")

    @property
    def key(self) -> tuple[str, str]:
        return (self.outer_target_id, self.geometry_id)


@dataclass(frozen=True)
class ModelBankRecord:
    outer_target_id: str
    geometry_id: str
    family: str
    bank: object

    def __post_init__(self) -> None:
        if (
            self.outer_target_id not in CENTERS
            or self.geometry_id not in GEOMETRY_IDS
            or self.family not in MODEL_FAMILY_IDS
            or getattr(self.bank, "outer_target_id", None) != self.outer_target_id
            or getattr(self.bank, "family", None) != self.family
        ):
            raise ProtocolError("Model-bank record identity drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.outer_target_id, self.geometry_id, self.family)


def model_bank_collection_hash(
    prelabel: "PrelabelProducts", records: tuple[ModelBankRecord, ...]
) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_disagreement_regret_model_bank_collection_v1",
            "prelabel_feature_seal_hash": prelabel.prelabel_feature_seal_hash,
            "source_prediction_seal_hash": prelabel.source_prediction_seal_hash,
            "banks": [
                {
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "model_bank_hash": str(
                        getattr(
                            record.bank,
                            "model_bank_hash",
                            getattr(record.bank, "bank_hash", ""),
                        )
                    ),
                }
                for record in sorted(records, key=lambda value: value.key)
            ],
            "source_labels_used_for_training_only": True,
            "raw_source_labels_persisted": False,
            "test_labels_used": False,
        }
    )


@dataclass(frozen=True)
class ContrastRecord:
    geometry_id: str
    row: CandidateContrastRow

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Contrast geometry drifted.")

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.geometry_id, *self.row.row_key)


@dataclass(frozen=True)
class SelectionRecord:
    geometry_id: str
    row: InferenceSelectionDiagnostic

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Selection geometry drifted.")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.geometry_id,
            self.row.family,
            self.row.target_query_id,
            self.row.case_id,
        )


@dataclass(frozen=True)
class PrelabelProducts:
    feature_surfaces: tuple[FeatureSurfaceRecord, ...]
    development_contexts: Mapping[tuple[str, str], object]
    source_prediction_seal_hash: str
    prelabel_feature_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.feature_surfaces)
        expected = {
            (target, geometry, family)
            for target in CENTERS
            for geometry in GEOMETRY_IDS
            for family in MODEL_FAMILY_IDS
        }
        if (
            {row.key for row in rows} != expected
            or len(rows) != len(expected)
            or set(self.development_contexts)
            != {(target, geometry) for target in CENTERS for geometry in GEOMETRY_IDS}
        ):
            raise ProtocolError("Prelabel feature topology drifted.")
        object.__setattr__(
            self,
            "prelabel_feature_seal_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_disagreement_regret_prelabel_features_v1",
                    "source_prediction_seal_hash": self.source_prediction_seal_hash,
                    "surfaces": [
                        {
                            "outer_target_id": row.outer_target_id,
                            "geometry_id": row.geometry_id,
                            "family": row.family,
                            "surface_hash": row.surface.surface_hash,
                        }
                        for row in sorted(rows, key=lambda value: value.key)
                    ],
                    "labels_opened": False,
                }
            ),
        )

    def surface(self, target: str, geometry: str, family: str) -> DisagreementFeatureSurface:
        for row in self.feature_surfaces:
            if row.key == (target, geometry, family):
                return row.surface
        raise ProtocolError("Requested prelabel surface is absent.")


@dataclass(frozen=True)
class DevelopmentProducts:
    prelabel: PrelabelProducts
    response_surfaces: tuple[ResponseSurfaceRecord, ...]
    model_banks: tuple[ModelBankRecord, ...]
    source_label_capability_report: Mapping[str, object]
    model_bank_hash: str

    def __post_init__(self) -> None:
        records = tuple(self.model_banks)
        expected = {(target, geometry) for target in CENTERS for geometry in GEOMETRY_IDS}
        if (
            {row.key for row in self.response_surfaces} != expected
            or {row.key for row in records}
            != {
                (target, geometry, family)
                for target in CENTERS
                for geometry in GEOMETRY_IDS
                for family in MODEL_FAMILY_IDS
            }
            or self.source_label_capability_report.get("test_labels_opened") is not False
            or self.source_label_capability_report.get("raw_source_labels_persisted") is not False
            or self.model_bank_hash
            != model_bank_collection_hash(self.prelabel, records)
        ):
            raise ProtocolError("Frozen development-product topology drifted.")


@dataclass(frozen=True)
class InferenceProducts:
    feature_surfaces: tuple[FeatureSurfaceRecord, ...]
    contrasts: tuple[ContrastRecord, ...]
    selections: tuple[SelectionRecord, ...]
    test_prediction_seal_hash: str
    model_bank_hash: str
    frozen_prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = {
            (target, geometry, family)
            for target in CENTERS
            for geometry in GEOMETRY_IDS
            for family in MODEL_FAMILY_IDS
        }
        feature_records = tuple(self.feature_surfaces)
        contrasts = tuple(self.contrasts)
        selections = tuple(self.selections)
        expected_contrast_keys: set[tuple[str, str, str, str, str]] = set()
        expected_selection_keys: set[tuple[str, str, str, str]] = set()
        feature_row_topology_valid = True
        for record in feature_records:
            surface = record.surface
            candidate_action_ids = tuple(surface.candidate_source_by_action)
            case_ids = tuple(sorted({row.case_id for row in surface.rows}))
            observed_feature_keys = tuple(
                (row.case_id, row.action_id) for row in surface.rows
            )
            expected_feature_keys = {
                (case_id, action_id)
                for case_id in case_ids
                for action_id in (surface.baseline_action_id, *candidate_action_ids)
            }
            if (
                not case_ids
                or not candidate_action_ids
                or len(observed_feature_keys) != len(set(observed_feature_keys))
                or set(observed_feature_keys) != expected_feature_keys
            ):
                feature_row_topology_valid = False
            expected_contrast_keys.update(
                (
                    record.geometry_id,
                    record.family,
                    record.outer_target_id,
                    case_id,
                    action_id,
                )
                for case_id in case_ids
                for action_id in candidate_action_ids
            )
            expected_selection_keys.update(
                (
                    record.geometry_id,
                    record.family,
                    record.outer_target_id,
                    case_id,
                )
                for case_id in case_ids
            )
        if (
            {row.key for row in feature_records} != expected
            or len(feature_records) != len(expected)
            or not feature_row_topology_valid
            or {row.key for row in contrasts} != expected_contrast_keys
            or len(contrasts) != len(expected_contrast_keys)
            or {row.key for row in selections} != expected_selection_keys
            or len(selections) != len(expected_selection_keys)
        ):
            raise ProtocolError("Inference feature topology drifted.")
        unhashed = {
            "schema_version": "midogpp_disagreement_regret_frozen_test_predictions_v1",
            "model_bank_hash": self.model_bank_hash,
            "test_prediction_seal_hash": self.test_prediction_seal_hash,
            "feature_surfaces": [
                {
                    "outer_target_id": row.outer_target_id,
                    "geometry_id": row.geometry_id,
                    "family": row.family,
                    "surface_hash": row.surface.surface_hash,
                }
                for row in sorted(self.feature_surfaces, key=lambda value: value.key)
            ],
            "contrast_rows": len(contrasts),
            "contrast_row_hashes": [
                record.row.row_hash
                for record in sorted(contrasts, key=lambda value: value.key)
            ],
            "selection_rows": len(selections),
            "selection_row_hashes": [
                record.row.row_hash
                for record in sorted(selections, key=lambda value: value.key)
            ],
            "test_labels_used": False,
            "test_metrics_computed": False,
            "may_authorize_routing": False,
        }
        object.__setattr__(self, "frozen_prediction_hash", canonical_hash(unhashed))


__all__ = (
    "DevelopmentProducts",
    "ContrastRecord",
    "FeatureSurfaceRecord",
    "InferenceProducts",
    "ModelBankRecord",
    "PrelabelProducts",
    "ResponseSurfaceRecord",
    "SelectionRecord",
    "model_bank_collection_hash",
)
