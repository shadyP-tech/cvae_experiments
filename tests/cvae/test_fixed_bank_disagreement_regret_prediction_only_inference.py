from __future__ import annotations

from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.constants import (
    B_ACTION_ID,
    U_ACTION_ID,
    candidate_sources,
    geometry_action_id,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.experiment_contracts import (
    CENTERS,
    GEOMETRY_IDS,
    MODEL_FAMILY_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.products import (
    ContrastRecord,
    FeatureSurfaceRecord,
    InferenceProducts,
    SelectionRecord,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.disagreement_regret_core import (
    CandidateContrastRow,
    InferenceSelectionDiagnostic,
)


CASE_COUNTS = dict(zip(CENTERS, (24, 24, 24, 24, 24, 24, 24, 24, 26), strict=True))


@pytest.fixture(scope="module")
def full_inference_rows() -> tuple[
    tuple[FeatureSurfaceRecord, ...],
    tuple[ContrastRecord, ...],
    tuple[SelectionRecord, ...],
]:
    features: list[FeatureSurfaceRecord] = []
    contrasts: list[ContrastRecord] = []
    selections: list[SelectionRecord] = []
    for target in CENTERS:
        case_ids = tuple(
            f"test-{target}-case-{index:03d}" for index in range(CASE_COUNTS[target])
        )
        sources = candidate_sources(target)
        for geometry in GEOMETRY_IDS:
            candidate_mapping = {
                geometry_action_id(geometry, source): source for source in sources
            }
            for family in MODEL_FAMILY_IDS:
                feature_rows = tuple(
                    SimpleNamespace(case_id=case_id, action_id=action_id)
                    for case_id in case_ids
                    for action_id in (B_ACTION_ID, *candidate_mapping)
                )
                surface = SimpleNamespace(
                    outer_target_id=target,
                    family=family,
                    baseline_action_id=B_ACTION_ID,
                    control_action_id=U_ACTION_ID,
                    candidate_source_by_action=candidate_mapping,
                    rows=feature_rows,
                    surface_hash=canonical_hash(
                        {
                            "target": target,
                            "geometry": geometry,
                            "family": family,
                        }
                    ),
                )
                features.append(
                    FeatureSurfaceRecord(
                        outer_target_id=target,
                        geometry_id=geometry,
                        family=family,
                        surface=surface,
                    )
                )
                for case_id in case_ids:
                    for action_id, source_id in candidate_mapping.items():
                        contrasts.append(
                            ContrastRecord(
                                geometry_id=geometry,
                                row=CandidateContrastRow(
                                    family=family,
                                    target_query_id=target,
                                    case_id=case_id,
                                    candidate_action_id=action_id,
                                    candidate_source_id=source_id,
                                    predicted_preference_margin_vs_control=0.0,
                                    standard_error_vs_control=0.0,
                                    predicted_preference_margin_vs_baseline=0.0,
                                    standard_error_vs_baseline=0.0,
                                    model_hash="a" * 64,
                                ),
                            )
                        )
                    selections.append(
                        SelectionRecord(
                            geometry_id=geometry,
                            row=InferenceSelectionDiagnostic(
                                family=family,
                                target_query_id=target,
                                case_id=case_id,
                                raw_action_id=B_ACTION_ID,
                                safe_action_id=B_ACTION_ID,
                                baseline_action_id=B_ACTION_ID,
                                control_action_id=U_ACTION_ID,
                                simultaneous_z_value=0.0,
                                safe_margin=0.0,
                                fallback_reason=(
                                    "simultaneous_lcb_nonpositive_vs_b_or_u"
                                ),
                            ),
                        )
                    )
    return tuple(features), tuple(contrasts), tuple(selections)


def test_full_218_case_inference_topology_is_candidate_only(
    full_inference_rows: tuple[
        tuple[FeatureSurfaceRecord, ...],
        tuple[ContrastRecord, ...],
        tuple[SelectionRecord, ...],
    ],
) -> None:
    features, contrasts, selections = full_inference_rows

    products = InferenceProducts(
        feature_surfaces=features,
        contrasts=contrasts,
        selections=selections,
        test_prediction_seal_hash="b" * 64,
        model_bank_hash="c" * 64,
    )

    assert len(features) == 54
    assert sum(len(record.surface.rows) for record in features) == 11_772
    assert len(contrasts) == 10_464
    assert len(selections) == 1_308
    assert all(record.row.candidate_action_id != B_ACTION_ID for record in contrasts)
    assert len(products.frozen_prediction_hash) == 64


def test_full_inference_topology_rejects_a_missing_candidate_contrast(
    full_inference_rows: tuple[
        tuple[FeatureSurfaceRecord, ...],
        tuple[ContrastRecord, ...],
        tuple[SelectionRecord, ...],
    ],
) -> None:
    features, contrasts, selections = full_inference_rows

    with pytest.raises(ProtocolError, match="Inference feature topology drifted"):
        InferenceProducts(
            feature_surfaces=features,
            contrasts=contrasts[:-1],
            selections=selections,
            test_prediction_seal_hash="b" * 64,
            model_bank_hash="c" * 64,
        )


def test_full_inference_topology_rejects_a_baseline_contrast(
    full_inference_rows: tuple[
        tuple[FeatureSurfaceRecord, ...],
        tuple[ContrastRecord, ...],
        tuple[SelectionRecord, ...],
    ],
) -> None:
    features, contrasts, selections = full_inference_rows
    first = contrasts[0]
    baseline = ContrastRecord(
        geometry_id=first.geometry_id,
        row=CandidateContrastRow(
            family=first.row.family,
            target_query_id=first.row.target_query_id,
            case_id=first.row.case_id,
            candidate_action_id=B_ACTION_ID,
            candidate_source_id=first.row.candidate_source_id,
            predicted_preference_margin_vs_control=0.0,
            standard_error_vs_control=0.0,
            predicted_preference_margin_vs_baseline=0.0,
            standard_error_vs_baseline=0.0,
            model_hash="a" * 64,
        ),
    )

    with pytest.raises(ProtocolError, match="Inference feature topology drifted"):
        InferenceProducts(
            feature_surfaces=features,
            contrasts=(*contrasts, baseline),
            selections=selections,
            test_prediction_seal_hash="b" * 64,
            model_bank_hash="c" * 64,
        )
