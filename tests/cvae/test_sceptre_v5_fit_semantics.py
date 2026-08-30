from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.physical.fit_semantics import (
    publish_fit_receipt,
    raw_fit_receipt_body,
    seal_fit_receipt,
    validate_fit_receipt,
    validate_published_fit_inventory,
    validate_publication_ordinals,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.physical.prediction_contracts import (
    LOCKED_CLASSIFIER_SPEC,
    PRODUCTION_PREDICTION_GEOMETRY,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _fit_body(
    *,
    family: str,
    center: str,
    centers: Sequence[str],
    rows_by_center: Mapping[str, int],
    training_seed: int = 17,
    generation_seed: int = 17,
) -> dict[str, object]:
    total = sum(rows_by_center.values())
    if family == "single_source":
        source_center: str | None = center
        target_center: str | None = None
        source_centers = [center]
        excluded_center: str | None = center
        masked_rows = rows_by_center[center]
        evaluated_rows = total - masked_rows
    else:
        source_center = None
        target_center = center
        source_centers = [value for value in centers if value != center]
        excluded_center = None
        masked_rows = 0
        evaluated_rows = rows_by_center[center]
    return {
        "family": family,
        "source_center": source_center,
        "target_center": target_center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "source_centers": source_centers,
        "composition_hash": "1" * 64,
        "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
        "scaler_state_hash": "2" * 64,
        "probability_sha256": "3" * 64,
        "prediction_sha256": "4" * 64,
        "evaluated_row_count": evaluated_rows,
        "excluded_evaluation_center": excluded_center,
        "masked_row_count": masked_rows,
        "converged": True,
        "target_expert_excluded": True,
    }


def _seal(
    body: Mapping[str, object],
    *,
    rows_by_center: Mapping[str, int],
) -> dict[str, object]:
    return seal_fit_receipt(
        body,
        evaluation_rows=sum(rows_by_center.values()),
        rows_by_center=rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )


def test_production_shaped_fit_counts_and_ordinal_after_hash_are_valid() -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    raw = _seal(
        _fit_body(
            family="single_source",
            center="0",
            centers=geometry.centers,
            rows_by_center=counts,
        ),
        rows_by_center=counts,
    )
    assert raw["evaluated_row_count"] == 9_928 - 1_532
    fit_hash = raw["fit_sha256"]
    published = publish_fit_receipt(
        raw,
        global_fit_ordinal=0,
        seed_cell_ordinal=0,
        within_cell_fit_ordinal=0,
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    assert published["fit_sha256"] == fit_hash
    assert raw_fit_receipt_body(published) == raw_fit_receipt_body(raw)
    validate_fit_receipt(
        published,
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    validate_publication_ordinals(
        published,
        global_fit_ordinal=0,
        seed_cell_ordinal=0,
        within_cell_fit_ordinal=0,
        training_seed=17,
        generation_seed=17,
    )


def test_exact_b_production_count_is_target_local() -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    row = _seal(
        _fit_body(
            family="exact_B",
            center="1",
            centers=geometry.centers,
            rows_by_center=counts,
        ),
        rows_by_center=counts,
    )
    assert row["evaluated_row_count"] == 866
    validate_fit_receipt(
        row,
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )


def test_tampered_fit_body_fails_hash_authentication() -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    row = _seal(
        _fit_body(
            family="single_source",
            center="2",
            centers=geometry.centers,
            rows_by_center=counts,
        ),
        rows_by_center=counts,
    )
    row["probability_sha256"] = "f" * 64
    with pytest.raises(ProtocolError, match="fit hash drifted"):
        validate_fit_receipt(
            row,
            evaluation_rows=geometry.evaluation_rows,
            rows_by_center=geometry.rows_by_center,
            expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        )


@pytest.mark.parametrize("family,center", [("single_source", "0"), ("exact_B", "1")])
def test_rehashed_wrong_family_specific_row_count_is_rejected(
    family: str, center: str
) -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    body = _fit_body(
        family=family,
        center=center,
        centers=geometry.centers,
        rows_by_center=counts,
    )
    body["evaluated_row_count"] = int(body["evaluated_row_count"]) + 1
    row = {**body, "fit_sha256": canonical_hash(body)}
    with pytest.raises(ProtocolError, match="fit semantics drifted"):
        validate_fit_receipt(
            row,
            evaluation_rows=geometry.evaluation_rows,
            rows_by_center=geometry.rows_by_center,
            expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        )


def test_exact_b_requires_ordered_c_minus_h_sources() -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    body = _fit_body(
        family="exact_B",
        center="0",
        centers=geometry.centers,
        rows_by_center=counts,
    )
    sources = list(body["source_centers"])
    sources[0], sources[1] = sources[1], sources[0]
    body["source_centers"] = sources
    row = {**body, "fit_sha256": canonical_hash(body)}
    with pytest.raises(ProtocolError, match="exact-B fit semantics drifted"):
        validate_fit_receipt(
            row,
            evaluation_rows=geometry.evaluation_rows,
            rows_by_center=geometry.rows_by_center,
            expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        )


def test_small_geometry_producer_rows_replay_through_full_inventory_validator() -> None:
    centers = ("0", "1", "2")
    counts = {"0": 3, "1": 2, "2": 1}
    seed_cells = ((17, 17), (42, 101))
    rows: list[dict[str, object]] = []
    for seed_ordinal, (training_seed, generation_seed) in enumerate(seed_cells):
        bodies = [
            _fit_body(
                family="single_source",
                center=center,
                centers=centers,
                rows_by_center=counts,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )
            for center in centers
        ] + [
            _fit_body(
                family="exact_B",
                center=center,
                centers=centers,
                rows_by_center=counts,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )
            for center in centers
        ]
        for within_ordinal, body in enumerate(bodies):
            raw = _seal(body, rows_by_center=counts)
            rows.append(
                publish_fit_receipt(
                    raw,
                    global_fit_ordinal=len(rows),
                    seed_cell_ordinal=seed_ordinal,
                    within_cell_fit_ordinal=within_ordinal,
                    evaluation_rows=sum(counts.values()),
                    rows_by_center=counts,
                    expected_classifier_config_hash=(
                        LOCKED_CLASSIFIER_SPEC.config_hash
                    ),
                )
            )
    validate_published_fit_inventory(
        rows,
        centers=centers,
        seed_cells=seed_cells,
        evaluation_rows=sum(counts.values()),
        rows_by_center=counts,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )

    ordinal_tamper = [dict(row) for row in rows]
    ordinal_tamper[7]["global_fit_ordinal"] = 8
    validate_fit_receipt(
        ordinal_tamper[7],
        evaluation_rows=sum(counts.values()),
        rows_by_center=counts,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    with pytest.raises(ProtocolError, match="fit ordinal drifted"):
        validate_published_fit_inventory(
            ordinal_tamper,
            centers=centers,
            seed_cells=seed_cells,
            evaluation_rows=sum(counts.values()),
            rows_by_center=counts,
            expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        )


def test_boolean_publication_ordinal_is_rejected() -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts = dict(geometry.rows_by_center)
    raw = _seal(
        _fit_body(
            family="single_source",
            center="0",
            centers=geometry.centers,
            rows_by_center=counts,
        ),
        rows_by_center=counts,
    )
    published = publish_fit_receipt(
        raw,
        global_fit_ordinal=0,
        seed_cell_ordinal=0,
        within_cell_fit_ordinal=0,
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    published["global_fit_ordinal"] = False
    with pytest.raises(ProtocolError, match="fit ordinal drifted"):
        validate_publication_ordinals(
            published,
            global_fit_ordinal=0,
            seed_cell_ordinal=0,
            within_cell_fit_ordinal=0,
            training_seed=17,
            generation_seed=17,
        )


@pytest.mark.parametrize("bad_count", ["1532", 1532.0, True])
def test_non_integral_center_count_is_rejected(bad_count: object) -> None:
    geometry = PRODUCTION_PREDICTION_GEOMETRY
    counts: dict[str, object] = dict(geometry.rows_by_center)
    counts["0"] = bad_count
    body = _fit_body(
        family="single_source",
        center="1",
        centers=geometry.centers,
        rows_by_center=dict(geometry.rows_by_center),
    )
    with pytest.raises(ProtocolError, match="center-row geometry is malformed"):
        seal_fit_receipt(
            body,
            evaluation_rows=geometry.evaluation_rows,
            rows_by_center=counts,
            expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        )
