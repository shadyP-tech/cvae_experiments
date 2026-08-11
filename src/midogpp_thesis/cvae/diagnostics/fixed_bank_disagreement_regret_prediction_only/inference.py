"""Frozen-bank inference over the entire label-free consumed test cache."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    LabelFreeInferenceContext,
    ProbabilityRow,
    build_label_free_inference_feature_surface,
    build_label_free_inference_selection_diagnostics,
    score_label_free_inference_candidate_contrasts,
)
from .experiment_contracts import CENTERS, GEOMETRY_IDS, MODEL_FAMILY_IDS
from .products import (
    ContrastRecord,
    DevelopmentProducts,
    FeatureSurfaceRecord,
    InferenceProducts,
    SelectionRecord,
)


def build_test_inference_products(
    development: DevelopmentProducts,
    probability_rows_by_surface: Mapping[
        tuple[str, str], Sequence[ProbabilityRow]
    ],
    *,
    test_prediction_seal_hash: str,
    target_cache_content_hash: str,
    target_cache_order_hash: str,
) -> InferenceProducts:
    """Apply each frozen G/R/P bank; target labels are not an input."""

    bank_by_key = {record.key: record for record in development.model_banks}
    expected = {
        (target, geometry, family)
        for target in CENTERS
        for geometry in GEOMETRY_IDS
        for family in MODEL_FAMILY_IDS
    }
    if set(bank_by_key) != expected:
        raise ProtocolError("Frozen inference model-bank topology drifted.")
    expected_surfaces = {
        (target, geometry) for target in CENTERS for geometry in GEOMETRY_IDS
    }
    if set(probability_rows_by_surface) != expected_surfaces:
        raise ProtocolError("Frozen inference probability topology drifted.")
    feature_records: list[FeatureSurfaceRecord] = []
    contrast_records: list[ContrastRecord] = []
    selection_records: list[SelectionRecord] = []
    for target in CENTERS:
        for geometry in GEOMETRY_IDS:
            # One physical R probability surface is replayed through each
            # family's frozen action schema. G/P are transformed inside core.
            for family in MODEL_FAMILY_IDS:
                bank_record = bank_by_key[(target, geometry, family)]
                bank = bank_record.bank
                context = LabelFreeInferenceContext(
                    dataset_family="MIDOGPP_CONSUMED_TEST_LABEL_FREE",
                    outer_target_id=target,
                    target_cache_content_hash=target_cache_content_hash,
                    target_cache_order_hash=target_cache_order_hash,
                    prediction_seal_hash=test_prediction_seal_hash,
                    action_schema=bank.action_schema,
                    model_bank_hash=bank.model_bank_hash,
                )
                rows = tuple(probability_rows_by_surface[(target, geometry)])
                if (
                    not rows
                    or {row.query_id for row in rows} != {target}
                    or {row.prediction_seal_hash for row in rows}
                    != {test_prediction_seal_hash}
                ):
                    raise ProtocolError("Frozen inference probability lineage drifted.")
                features = build_label_free_inference_feature_surface(
                    rows, context=context
                )
                feature_records.append(
                    FeatureSurfaceRecord(
                        outer_target_id=target,
                        geometry_id=geometry,
                        family=family,
                        surface=features,
                    )
                )
                contrasts = score_label_free_inference_candidate_contrasts(
                    bank, features, context=context
                )
                contrast_records.extend(
                    ContrastRecord(geometry_id=geometry, row=row)
                    for row in contrasts
                )
                selections = build_label_free_inference_selection_diagnostics(
                    contrasts, context=context
                )
                selection_records.extend(
                    SelectionRecord(geometry_id=geometry, row=row)
                    for row in selections
                )
    return InferenceProducts(
        feature_surfaces=tuple(sorted(feature_records, key=lambda row: row.key)),
        contrasts=tuple(sorted(contrast_records, key=lambda row: row.key)),
        selections=tuple(sorted(selection_records, key=lambda row: row.key)),
        test_prediction_seal_hash=test_prediction_seal_hash,
        model_bank_hash=development.model_bank_hash,
    )


__all__ = ("build_test_inference_products",)
