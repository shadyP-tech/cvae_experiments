from __future__ import annotations

import math

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.actions import actions_for_target
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.case_partitions import (
    CaseIdentityRow,
    build_case_oof_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.constants import (
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.contracts import (
    ActionScoreRow,
    AggregatedProbabilityRow,
    BinaryLabelRow,
    ExactNineProbabilitySurface,
    RidgeActionModel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.execution import (
    ModelProducts,
    build_loco_utility_product,
    build_pre_support_decision_products,
    build_prelabel_products,
    build_support_fold_product,
    combine_decision_products,
    fit_target_model_product,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.terminal import evaluate_terminal
from midogpp_thesis.cvae.protocol import ProtocolError


PROTOCOL_HASH = "1" * 64
STORE_HASH = "2" * 64
PERMUTATION_HASH = "3" * 64


def _surface_and_labels():
    rows, labels, identities = [], [], []
    for center_index, center in enumerate(MIDOGPP_CENTERS):
        for case_index in range(5):
            case_id = f"H{center}-case-{case_index}"
            for label in (0, 1):
                sample_id = f"{case_id}-y{label}"
                labels.append(BinaryLabelRow(center, case_id, sample_id, label))
                identities.append(CaseIdentityRow(center, case_id, sample_id))
                for action_index, action in enumerate(actions_for_target(center)):
                    correct = not (action_index > 1 and (action_index + case_index + center_index) % 7 == 0)
                    if correct:
                        probability = 0.70 if label else 0.30
                    else:
                        probability = 0.30 if label else 0.70
                    rows.append(
                        AggregatedProbabilityRow(
                            center, case_id, sample_id, action.action_id,
                            probability, 0.01, 9,
                            canonical_hash([center, case_id, sample_id, action.action_id]),
                        )
                    )
    canonical = tuple(sorted(rows, key=lambda row: row.row_key))
    surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_exact_nine_surface_v1",
            "probability_store_hash": STORE_HASH,
            "rows": [row.to_payload() for row in canonical],
            "predictions_sealed_before_labels": True,
        }
    )
    surface = ExactNineProbabilitySurface(canonical, STORE_HASH, surface_hash)
    partition = build_case_oof_partition(
        identities, partition_seed=123, expected_total_case_count=45
    )
    return surface, tuple(sorted(labels)), partition


@pytest.fixture(scope="module")
def synthetic_inputs():
    return _surface_and_labels()


def test_prelabel_and_loco_utility_are_hash_bound_and_exclude_H(synthetic_inputs) -> None:
    surface, labels, _partition = synthetic_inputs
    prelabel = build_prelabel_products(surface, protocol_contract_hash=PROTOCOL_HASH)
    assert prelabel.probability_surface_hash == surface.surface_hash
    assert len(prelabel.features) == 9 * 5 * 2 * 8
    donor_labels = tuple(row for row in labels if row.target_center != "0")
    utility = build_loco_utility_product(
        surface, donor_labels, outer_target_center="0"
    )
    assert {row.query_center for row in utility.rows} == set(candidate_sources("0"))
    assert {row.response_kind for row in utility.rows} == {
        "class_balanced_proper_loss_gain_vs_u"
    }
    assert all(math.isfinite(row.response) for row in utility.rows)
    with pytest.raises(ProtocolError, match="all and only"):
        build_loco_utility_product(surface, labels, outer_target_center="0")


def test_target_model_product_fits_final_and_nested_H_q_e_surfaces(synthetic_inputs) -> None:
    surface, labels, _partition = synthetic_inputs
    prelabel = build_prelabel_products(surface, protocol_contract_hash=PROTOCOL_HASH)
    utility = build_loco_utility_product(
        surface,
        tuple(row for row in labels if row.target_center != "0"),
        outer_target_center="0",
    )
    product = fit_target_model_product(
        prelabel, utility, workers=1, threads_per_worker=1
    )
    assert len(tuple(row for row in product.models if row.heldout_donor_center is None)) == 48
    assert len(tuple(row for row in product.models if row.heldout_donor_center is not None)) == 336
    assert len(product.nested_mse) == 6
    assert all(row.selected_source != row.heldout_query_center for row in product.nested_predictions)
    assert set(product.model_seals) == {
        f"{geometry}:{family}"
        for geometry in GEOMETRY_IDS
        for family in ("G", "R", "P")
    }


def _minimal_models(surface, partition) -> ModelProducts:
    models, scores, seals = [], [], {}
    for target in MIDOGPP_CENTERS:
        source = candidate_sources(target)[0]
        models.append(
            RidgeActionModel(
                target, None, "A0", source, "G", 1.0,
                ("intercept",), (0.0,), (1.0,), (0.0,),
                tuple(center for center in MIDOGPP_CENTERS if center not in (target, source)),
                "class_balanced_proper_loss_gain_vs_u",
            )
        )
        seals[target] = {
            f"{geometry}:{family}": canonical_hash([target, geometry, family])
            for geometry in GEOMETRY_IDS for family in ("G", "R", "P")
        }
        cases = sorted({row.case_id for row in partition.identities if row.target_center == target})
        for case_id in cases:
            for geometry in GEOMETRY_IDS:
                for family in ("G", "R", "P"):
                    for source_index, candidate in enumerate(candidate_sources(target)):
                        scores.append(
                            ActionScoreRow(
                                target, case_id, geometry, candidate, family,
                                -0.01 - 0.001 * source_index, "4" * 64,
                            )
                        )
    return ModelProducts(
        tuple(models), tuple(sorted(scores, key=lambda row: row.row_key)), (), (), seals,
        "5" * 64, PERMUTATION_HASH, PROTOCOL_HASH, surface.surface_hash,
    )


def _decision_products(surface, labels, partition):
    pre = build_pre_support_decision_products(_minimal_models(surface, partition), partition)
    support_products = []
    for fold in partition.folds:
        support_cases = set(fold.support_case_ids)
        scoped = tuple(
            row for row in labels
            if row.target_center == fold.target_center and row.case_id in support_cases
        )
        support_products.append(
            build_support_fold_product(
                surface, partition, scoped,
                target_center=fold.target_center, fold_ordinal=fold.fold_ordinal,
            )
        )
    return combine_decision_products(pre, support_products, partition)


def _capability_report(decisions):
    event_identities = (
        *(("loco_donor", center, None) for center in MIDOGPP_CENTERS),
        *(
            ("target_support", center, fold)
            for center in MIDOGPP_CENTERS
            for fold in range(5)
        ),
        ("terminal_evaluation", None, None),
    )
    payload = {
        "schema_version": "test_capability_v1",
        "status": "PASS",
        "loco_centers_opened": sorted(MIDOGPP_CENTERS),
        "loco_model_seals": {
            center: {
                f"{geometry}:{family}": canonical_hash([center, geometry, family])
                for geometry in GEOMETRY_IDS for family in ("G", "R", "P")
            }
            for center in MIDOGPP_CENTERS
        },
        "pre_support_decision_count": 405,
        "fold_support_capability_count": 45,
        "support_decision_count": 90,
        "all_decisions_seal_hash": decisions.all_decisions_seal_hash,
        "permutation_provenance_hash": decisions.permutation_provenance_hash,
        "evaluation_labels_opened": True,
        "events": [
            {
                "role": role,
                "target_center": target,
                "fold_ordinal": fold,
                "row_count": 1,
                "case_count": 1,
                "row_identity_hash": canonical_hash(
                    [role, target, fold, "rows"]
                ),
                "label_identity_hash": canonical_hash(
                    [role, target, fold, "labels"]
                ),
                "raw_labels_persisted": False,
            }
            for role, target, fold in event_identities
        ],
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "geometry_selected": False,
        "evaluation_labels_used_for_decisions": False,
    }
    return {**payload, "report_hash": canonical_hash(payload)}


def test_support_decisions_and_terminal_metrics_use_whole_case_sufficient_statistics(synthetic_inputs) -> None:
    surface, labels, partition = synthetic_inputs
    decisions = _decision_products(surface, labels, partition)
    assert len(decisions.pre_support_decision_hashes) == 405
    assert len(decisions.all_decision_hashes) == 495
    assert len(decisions.support_product_hashes) == 45
    envelope = evaluate_terminal(
        surface, decisions, labels, partition,
        capability_report=_capability_report(decisions),
        protocol_contract_hash=PROTOCOL_HASH,
        bootstrap_replicates=50,
        bootstrap_workers=1,
        bootstrap_threads_per_worker=1,
    )
    assert len(envelope.method_summaries) == 15
    assert len(envelope.scientific_result.geometries) == 2
    assert all(len(row.fold_rank_stability) == 45 for row in envelope.scientific_result.geometries)
    assert all(len(row.contrasts) == 9 for row in envelope.scientific_result.geometries)
    tables = envelope.table_rows()
    assert set(tables) == {
        "tables/terminal_case_confusions.csv",
        "tables/terminal_center_metrics.csv",
        "tables/terminal_method_summary.csv",
        "tables/terminal_contrasts.csv",
        "tables/oracle_rank_metrics.csv",
        "tables/complementarity.csv",
        "tables/rank_stability.csv",
        "tables/permutation_metrics.csv",
    }
    assert all(rows for rows in tables.values())
    contrast = tables["tables/terminal_contrasts.csv"][0]
    assert {
        "geometry_id",
        "contrast_family",
        "challenger_method",
        "reference_method",
        "equal_center_difference",
        "center_t_ci95_lower",
        "center_t_ci95_upper",
        "bootstrap_mean",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "bootstrap_replicate_count",
        "bootstrap_invalid_draw_count",
        "bootstrap_hash",
        "contrast_hash",
    }.issubset(contrast)
    assert all("per_case_bacc" not in key and "label" not in key for rows in tables.values() for row in rows for key in row)


def test_terminal_capability_hash_is_replayed_not_shape_checked(synthetic_inputs) -> None:
    surface, labels, partition = synthetic_inputs
    decisions = _decision_products(surface, labels, partition)
    report = _capability_report(decisions)
    report["support_decision_count"] = 89
    with pytest.raises(ProtocolError, match="hash does not replay"):
        evaluate_terminal(
            surface, decisions, labels, partition,
            capability_report=report,
            protocol_contract_hash=PROTOCOL_HASH,
            bootstrap_replicates=5,
            bootstrap_workers=1,
            bootstrap_threads_per_worker=1,
        )


def test_terminal_rejects_rehashed_non_sha_model_seal(synthetic_inputs) -> None:
    surface, labels, partition = synthetic_inputs
    decisions = _decision_products(surface, labels, partition)
    report = _capability_report(decisions)
    report.pop("report_hash")
    report["loco_model_seals"]["0"]["A0:G"] = "not-a-sha256"
    report["report_hash"] = canonical_hash(report)

    with pytest.raises(ProtocolError, match="complete fail-closed capability"):
        evaluate_terminal(
            surface,
            decisions,
            labels,
            partition,
            capability_report=report,
            protocol_contract_hash=PROTOCOL_HASH,
            bootstrap_replicates=5,
            bootstrap_workers=1,
            bootstrap_threads_per_worker=1,
        )

    reordered = _capability_report(decisions)
    reordered.pop("report_hash")
    reordered["events"][0], reordered["events"][1] = (
        reordered["events"][1],
        reordered["events"][0],
    )
    reordered["report_hash"] = canonical_hash(reordered)
    with pytest.raises(ProtocolError, match="complete fail-closed capability"):
        evaluate_terminal(
            surface,
            decisions,
            labels,
            partition,
            capability_report=reordered,
            protocol_contract_hash=PROTOCOL_HASH,
            bootstrap_replicates=5,
            bootstrap_workers=1,
            bootstrap_threads_per_worker=1,
        )


def test_package_identity_and_artifact_helpers_do_not_depend_on_sibling_stage90() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
        artifact_serialization,
        bundle,
        inputs,
        label_capabilities,
        runner,
        runner_runtime,
    )

    expected_prefix = (
        "midogpp_thesis.cvae.diagnostics."
        "fixed_bank_actionability_recoverability"
    )
    assert inputs.LabelFreeTestFrame.__module__.startswith(expected_prefix)
    assert inputs.TestRowIdentity.__module__.startswith(expected_prefix)
    assert label_capabilities.BinaryLabel.__module__.startswith(expected_prefix)
    assert artifact_serialization.persist_or_validate_json.__module__.startswith(
        expected_prefix
    )
    assert bundle.persist_or_validate_json.__module__.startswith(expected_prefix)
    assert runner.read_json.__module__ == "midogpp_thesis.cvae.runtime.artifact_io"
    assert runner_runtime.read_json.__module__ == (
        "midogpp_thesis.cvae.runtime.artifact_io"
    )
