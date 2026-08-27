from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.config import (
    repository_root,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.development_model import (
    EvidenceFeatureRow,
    fit_nested_lodo_pairwise_ranker,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.development_surface import (
    HistoricalUtilityCell,
    SourceInnerDevelopmentSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    FEATURE_NAMES,
    RawSourceEvidence,
    build_outer_development_evidence,
    build_target_prediction_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.identity import (
    AMENDMENT_RELATIVE_PATH,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.source_inner_authorization import (
    load_reuse_amendment,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


RAW_RECEIPT_HASH = "5" * 64


def _surface(*, poison_center: str | None = None) -> SourceInnerDevelopmentSurface:
    index = {center: ordinal for ordinal, center in enumerate(CENTERS)}
    cells = []
    for query in CENTERS:
        for candidate in CENTERS:
            if query == candidate:
                continue
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    bacc = (
                        0.55
                        + 0.01 * index[candidate]
                        - 0.001 * abs(index[query] - index[candidate])
                        + 0.00001 * (training_seed + generation_seed)
                    )
                    if poison_center in {query, candidate}:
                        bacc = 0.01
                    cells.append(
                        HistoricalUtilityCell(
                            query_center=query,
                            candidate_center=candidate,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            bacc=bacc,
                            macro_f1=bacc,
                        )
                    )
    marker = "a" if poison_center is None else "b"
    return SourceInnerDevelopmentSurface(
        cells=tuple(cells),
        utility_lock_sha256="1" * 64,
        utility_table_sha256=marker * 64,
        case_confusions_sha256="3" * 64,
        amendment_sha256="4" * 64,
    )


def _evidence(*, poison_query: str | None = None) -> tuple[EvidenceFeatureRow, ...]:
    index = {center: ordinal for ordinal, center in enumerate(CENTERS)}
    rows = []
    for query in CENTERS:
        for candidate in CENTERS:
            if query == candidate:
                continue
            values = (
                float(index[candidate]),
                float(abs(index[query] - index[candidate])),
            )
            if query == poison_query or candidate == poison_query:
                values = (1e12, -1e12)
            rows.append(
                EvidenceFeatureRow(
                    query_center=query,
                    candidate_center=candidate,
                    feature_names=("candidate_axis", "query_candidate_gap"),
                    values=values,
                )
            )
    return tuple(rows)


def _raw_evidence(*, poison_center: str | None = None) -> tuple[RawSourceEvidence, ...]:
    index = {center: ordinal for ordinal, center in enumerate(CENTERS)}
    rows = []
    for query in CENTERS:
        for candidate in CENTERS:
            if query == candidate:
                continue
            base = float(index[candidate] + abs(index[query] - index[candidate]))
            replica = {
                seed: base + 0.001 * seed for seed in TRAINING_SEEDS
            }
            entropy = 0.2 + 0.01 * index[candidate]
            disagreement = 0.1 + 0.01 * index[query]
            if poison_center in {query, candidate}:
                replica = {seed: 1e12 + seed for seed in TRAINING_SEEDS}
                entropy = 1e6
                disagreement = 0.99
            rows.append(
                RawSourceEvidence(
                    query_center=query,
                    candidate_center=candidate,
                    training_replica_proxy_energy=replica,
                    predictive_entropy=entropy,
                    vote_disagreement=disagreement,
                )
            )
    return tuple(rows)


def test_outer_view_deletes_query_and_candidate_h_before_aggregation() -> None:
    view = _surface().for_outer_target("2")

    assert len(view.cells) == 504
    assert len(view.aggregate_rows) == 56
    assert all(cell.query_center != "2" for cell in view.cells)
    assert all(cell.candidate_center != "2" for cell in view.cells)
    assert all(row.seed_cell_count == 9 for row in view.aggregate_rows)

    poisoned = _surface(poison_center="2").for_outer_target("2")
    assert poisoned.aggregate_rows == view.aggregate_rows


def test_nested_lodo_repeats_query_and_candidate_center_deletion() -> None:
    view = _surface().for_outer_target("2")
    fit = fit_nested_lodo_pairwise_ranker(
        view,
        build_outer_development_evidence(
            _raw_evidence(),
            outer_target="2",
            raw_source_receipt_hash=RAW_RECEIPT_HASH,
        ),
        alphas=(0.1, 1.0),
    )

    assert fit.descriptive_only is True
    assert fit.adaptive_surface is True
    assert len(fit.assessments) == 2
    assert all(len(assessment.folds) == 8 for assessment in fit.assessments)
    for assessment in fit.assessments:
        for fold in assessment.folds:
            assert fold.held_center not in fold.training_query_centers
            assert fold.held_center not in fold.training_candidate_centers
            assert "2" not in fold.training_query_centers
            assert "2" not in fold.training_candidate_centers
            assert fold.validation_candidate_count == 7
    assert fit.final_model.candidate_centers == tuple(
        center for center in CENTERS if center != "2"
    )


def test_outer_h_evidence_is_removed_before_normalization_and_fit() -> None:
    view = _surface().for_outer_target("2")
    clean_evidence = build_outer_development_evidence(
        _raw_evidence(),
        outer_target="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    poisoned_evidence = build_outer_development_evidence(
        _raw_evidence(poison_center="2"),
        outer_target="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    clean = fit_nested_lodo_pairwise_ranker(view, clean_evidence, alphas=(1.0,))
    poisoned = fit_nested_lodo_pairwise_ranker(
        view, poisoned_evidence, alphas=(1.0,)
    )

    assert clean.final_model.feature_means == poisoned.final_model.feature_means
    assert clean.final_model.feature_scales == poisoned.final_model.feature_scales
    assert clean.final_model.coefficients == pytest.approx(
        poisoned.final_model.coefficients
    )


def test_raw_evidence_deletes_outer_h_before_every_transform() -> None:
    clean = build_outer_development_evidence(
        _raw_evidence(),
        outer_target="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    poisoned = build_outer_development_evidence(
        _raw_evidence(poison_center="2"),
        outer_target="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )

    assert len(clean.rows) == 56
    assert clean == poisoned
    assert clean.receipt.input_row_count == 72
    assert clean.receipt.retained_row_count == 56
    assert clean.receipt.feature_names == FEATURE_NAMES
    assert all(row.query_center != "2" for row in clean.rows)
    assert all(row.candidate_center != "2" for row in clean.rows)
    assert all(row.labels_consumed is False for row in clean.rows)


def test_target_evidence_is_exact_c_minus_h_and_content_bound() -> None:
    bundle = build_target_prediction_evidence(
        _raw_evidence(),
        target_center="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    changed = build_target_prediction_evidence(
        _raw_evidence(poison_center="2"),
        target_center="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )

    assert len(bundle.rows) == 8
    assert {row.candidate_center for row in bundle.rows} == set(CENTERS) - {"2"}
    assert all(row.query_center == "2" for row in bundle.rows)
    assert bundle.receipt.retained_raw_hash != changed.receipt.retained_raw_hash
    assert bundle.receipt.transformed_feature_hash != (
        changed.receipt.transformed_feature_hash
    )
    assert bundle.rows != changed.rows


def test_nested_k_is_deleted_before_contextual_rank_transform() -> None:
    view = _surface().for_outer_target("2")
    clean = fit_nested_lodo_pairwise_ranker(
        view,
        build_outer_development_evidence(
            _raw_evidence(),
            outer_target="2",
            raw_source_receipt_hash=RAW_RECEIPT_HASH,
        ),
        alphas=(1.0,),
    )
    poisoned = fit_nested_lodo_pairwise_ranker(
        view,
        build_outer_development_evidence(
            _raw_evidence(poison_center="3"),
            outer_target="2",
            raw_source_receipt_hash=RAW_RECEIPT_HASH,
        ),
        alphas=(1.0,),
    )
    clean_fold = next(
        fold for fold in clean.assessments[0].folds if fold.held_center == "3"
    )
    poisoned_fold = next(
        fold for fold in poisoned.assessments[0].folds if fold.held_center == "3"
    )

    assert (
        clean_fold.training_transform_receipt_hash
        == poisoned_fold.training_transform_receipt_hash
    )
    assert (
        clean_fold.validation_transform_receipt_hash
        != poisoned_fold.validation_transform_receipt_hash
    )


def test_nested_fit_rejects_unbound_feature_rows() -> None:
    with pytest.raises(ProtocolError, match="receipt-bound"):
        fit_nested_lodo_pairwise_ranker(
            _surface().for_outer_target("2"), _evidence(), alphas=(1.0,)
        )


def test_evidence_bundle_replays_raw_transform_and_receipt_hash() -> None:
    bundle = build_outer_development_evidence(
        _raw_evidence(),
        outer_target="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    changed_row = replace(
        bundle.rows[0],
        values=(bundle.rows[0].values[0] + 1.0, *bundle.rows[0].values[1:]),
    )
    with pytest.raises(ProtocolError, match="does not replay"):
        replace(bundle, rows=(changed_row, *bundle.rows[1:]))
    with pytest.raises(ProtocolError, match="receipt hash drifted"):
        replace(bundle.receipt, receipt_hash="f" * 64)


def test_model_ranks_only_exact_target_excluded_candidate_families() -> None:
    fit = fit_nested_lodo_pairwise_ranker(
        _surface().for_outer_target("2"),
        build_outer_development_evidence(
            _raw_evidence(),
            outer_target="2",
            raw_source_receipt_hash=RAW_RECEIPT_HASH,
        ),
        alphas=(1.0,),
    )
    target = build_target_prediction_evidence(
        _raw_evidence(),
        target_center="2",
        raw_source_receipt_hash=RAW_RECEIPT_HASH,
    )
    ranked = fit.final_model.rank_target(target.rows)

    assert tuple(candidate for candidate, _ in ranked) != ()
    assert {candidate for candidate, _ in ranked} == set(CENTERS) - {"2"}


def test_repository_amendment_is_single_consumer_and_non_authorizing() -> None:
    receipt = load_reuse_amendment(
        repository_root() / Path(AMENDMENT_RELATIVE_PATH)
    )

    assert receipt.consumer_experiment_id == EXPERIMENT_ID
    assert receipt.publication_status == PUBLICATION_STATUS
    assert len(receipt.amendment_sha256) == 64


def test_development_evidence_cannot_smuggle_realized_utility() -> None:
    with pytest.raises(ProtocolError, match="exposes label utility"):
        EvidenceFeatureRow(
            query_center="0",
            candidate_center="1",
            feature_names=("realized_bacc",),
            values=(0.9,),
        )
