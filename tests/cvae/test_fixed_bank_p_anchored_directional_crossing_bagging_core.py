from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.composition import (
    compose_case_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.config import (
    load_p_anchored_directional_crossing_bagging_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.constants import (
    CENTERS,
    CROSSING_FEATURE_NAMES,
    ENDPOINT_METHOD_IDS,
    EXPECTED_CROSSING_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.crossing_contracts import (
    CrossingPrediction,
    DonorCrossingRow,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.crossing_features import (
    build_crossing_descriptors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.crossing_model import (
    fit_crossing_helpfulness_model,
    fit_full_and_delete_donor_models,
    predict_crossing_helpfulness,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.crossing_responses import (
    blocked_feature_permutation,
    build_donor_crossing_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.donor_center_bagging import (
    predict_with_donor_center_bagging,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.information_diagnostics import (
    crossing_information_diagnostics,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.outer_plans import (
    build_outer_plans,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.artifact_io import (
    persist_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.bundle import (
    CONTENT_INDEX_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.validation import (
    RECONSTRUCTIVE_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.workstation import (
    assert_canonical_workload,
    assert_runtime,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_directional_crossing_bagging_v1.yaml"
)


def _endpoint() -> EndpointCasePrediction:
    return EndpointCasePrediction(
        "0",
        "case-0",
        ("s0", "s1", "s2", "s3"),
        MappingProxyType(
            {
                "B": (0.6, 0.4, 0.45, 0.55),
                "I_OPPORTUNITY_GATED": (0.4, 0.6, 0.55, 0.45),
                "R_NINE_ARM_ROBUST": (0.4, 0.6, 0.45, 0.55),
                "P_PROTECTED": (0.4, 0.6, 0.45, 0.55),
            }
        ),
        "1" * 64,
    )


def _training_rows() -> tuple[DonorCrossingRow, ...]:
    rows = []
    donors = tuple(center for center in CENTERS if center != "0")
    for donor_index, donor in enumerate(donors):
        for helpful in (0, 1):
            feature = tuple(
                float((donor_index + 1) * (feature_index + 1) / 100 + helpful * 0.2)
                for feature_index in range(len(CROSSING_FEATURE_NAMES))
            )
            rows.append(
                DonorCrossingRow(
                    "0",
                    donor,
                    f"case-{donor}-{helpful}",
                    f"sample-{donor}-{helpful}",
                    "B",
                    "zero_to_one",
                    feature,
                    helpful,
                    0.01 if helpful else -0.01,
                    -0.1 if helpful else 0.1,
                    f"{len(rows) + 1:064x}",
                )
            )
    return tuple(rows)


def test_registered_config_and_lean_workload() -> None:
    config = load_p_anchored_directional_crossing_bagging_config(CONFIG)
    assert config.experiment_id.endswith("p_anchored_directional_crossing_bagging.v1")
    assert config.runtime["double_exclusion_state_count"] == 0
    assert config.runtime["expected_outer_endpoint_model_fit_count"] == 3_488
    assert EXPECTED_OUTER_PLAN_COUNT == 218
    assert EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT == 3_488
    assert EXPECTED_CROSSING_MODEL_FIT_COUNT == 162
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.constants import (
        EXPECTED_CASE_COUNTS_BY_CENTER,
    )

    assert_canonical_workload(EXPECTED_CASE_COUNTS_BY_CENTER)
    assert_runtime(config.runtime)


def test_reconstruction_surface_is_closed_under_content_index() -> None:
    assert set(RECONSTRUCTIVE_MEMBERS) <= set(CONTENT_INDEX_MEMBERS)
    assert set(CONTENT_INDEX_MEMBERS) - set(RECONSTRUCTIVE_MEMBERS) == {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "arrays/frozen_source_streams.npy",
        "arrays/fixed_bank_a1_action_probabilities.npz",
        "manifests/frozen_source_stream_index.json",
        "manifests/frozen_source_stream_lock.json",
        "manifests/fixed_bank_a1_prediction_index.json",
        "manifests/fixed_bank_a1_prediction_seal.json",
        "reports/workstation_preflight.json",
        "reports/runtime_summary.json",
    }


def test_outer_plan_seal_has_no_double_exclusion_surface() -> None:
    identities = tuple(
        SimpleNamespace(center=center, case_id=f"{center}-{case}", sample_id=f"{center}-{case}-{row}", group_id=f"{center}-{case}")
        for center in CENTERS
        for case in range(3)
        for row in range(2)
    )
    seal = build_outer_plans(
        identities,
        probability_surface_hash="2" * 64,
        strict_canonical_topology=False,
    )
    assert len(seal.outer_plans) == 27
    assert seal.to_payload()["double_exclusion_states_used"] is False
    assert all(plan.case_id not in plan.support_case_ids for plan in seal.outer_plans)


def test_crossing_descriptors_cover_actions_without_early_nomination() -> None:
    endpoint = _endpoint()
    rows = build_crossing_descriptors(endpoint)
    assert len(rows) == 4
    assert {(row.sample_id, row.alternative) for row in rows} == {
        ("s0", "B"),
        ("s1", "B"),
        ("s2", "I_OPPORTUNITY_GATED"),
        ("s3", "I_OPPORTUNITY_GATED"),
    }
    assert all(row.feature_names == CROSSING_FEATURE_NAMES for row in rows)
    assert all(row.endpoint_prediction_hash == endpoint.prediction_hash for row in rows)


def test_donor_label_changes_response_not_descriptor() -> None:
    endpoint = _endpoint()
    descriptors = build_crossing_descriptors(endpoint)

    def build(values: tuple[int, ...]) -> tuple[DonorCrossingRow, ...]:
        labels = tuple(
            BinaryLabel(
                "0",
                "case-0",
                sample,
                value,
                "crossing_donor::outer_H=1::donor_J=0",
            )
            for sample, value in zip(endpoint.sample_ids, values, strict=True)
        )
        return build_donor_crossing_rows(
            outer_target_center="1",
            prediction=endpoint,
            descriptors=descriptors,
            case_labels=labels,
            center_n_positive=2,
            center_n_negative=2,
        )

    first = build((1, 0, 1, 0))
    second = build((0, 1, 0, 1))
    assert [row.descriptor_hash for row in first] == [row.descriptor_hash for row in second]
    assert [row.feature_values for row in first] == [row.feature_values for row in second]
    assert [row.helpful for row in first] == [1 - row.helpful for row in second]


def test_center_balanced_logistic_and_complete_donor_deletions() -> None:
    rows = _training_rows()
    donors = tuple(center for center in CENTERS if center != "0")
    model = fit_crossing_helpfulness_model(
        rows,
        outer_target_center="0",
        training_centers=donors,
    )
    descriptor = build_crossing_descriptors(_endpoint())[0]
    probability = predict_crossing_helpfulness(model, descriptor)
    assert 0.0 <= probability <= 1.0
    full, deleted = fit_full_and_delete_donor_models(rows, outer_target_center="0")
    assert tuple(deleted) == donors
    assert all(len(candidate.training_centers) == 7 for candidate in deleted.values())
    prediction = predict_with_donor_center_bagging(
        descriptor,
        full_model=full,
        delete_models=deleted,
    )
    assert len(prediction.deletion_probabilities) == 8
    assert prediction.raw_weight >= 0.0


def test_blocked_control_preserves_responses_and_moves_features() -> None:
    rows = _training_rows()
    blocked = blocked_feature_permutation(rows)
    assert [row.helpful for row in blocked] == [row.helpful for row in rows]
    assert [row.bacc_contribution_delta for row in blocked] == [
        row.bacc_contribution_delta for row in rows
    ]
    assert any(a.feature_values != b.feature_values for a, b in zip(rows, blocked, strict=True))


def test_convex_composition_has_exact_p_fallback() -> None:
    endpoint = _endpoint()
    descriptors = build_crossing_descriptors(endpoint)
    donors = tuple(center for center in CENTERS if center != "0")
    zero_predictions = tuple(
        CrossingPrediction(
            row.descriptor_hash,
            0.5,
            tuple((center, 0.5) for center in donors),
            0.5,
            0.0,
            0.0,
            tuple(f"{index + 1:064x}" for index in range(9)),
        )
        for row in descriptors
    )
    composed = compose_case_probabilities(
        endpoint,
        descriptors,
        zero_predictions,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert composed.probabilities == endpoint.probabilities[PORTFOLIO_METHOD_ID]
    assert composed.portfolio_weights == (1.0, 1.0, 1.0, 1.0)


def test_crossing_model_rejects_target_center_as_donor() -> None:
    with pytest.raises(Exception):
        fit_crossing_helpfulness_model(
            _training_rows(),
            outer_target_center="0",
            training_centers=("0", "1"),
        )


def test_donor_response_requires_complete_exact_scope() -> None:
    endpoint = _endpoint()
    descriptors = build_crossing_descriptors(endpoint)
    incomplete = tuple(
        BinaryLabel(
            "0",
            "case-0",
            sample,
            value,
            "crossing_donor::outer_H=1::donor_J=0",
        )
        for sample, value in zip(endpoint.sample_ids[:-1], (1, 0, 1), strict=True)
    )
    with pytest.raises(Exception):
        build_donor_crossing_rows(
            outer_target_center="1",
            prediction=endpoint,
            descriptors=descriptors,
            case_labels=incomplete,
            center_n_positive=2,
            center_n_negative=2,
        )

    wrong_scope = tuple(
        BinaryLabel("0", "case-0", sample, value, "terminal_evaluation")
        for sample, value in zip(endpoint.sample_ids, (1, 0, 1, 0), strict=True)
    )
    with pytest.raises(Exception):
        build_donor_crossing_rows(
            outer_target_center="1",
            prediction=endpoint,
            descriptors=descriptors,
            case_labels=wrong_scope,
            center_n_positive=2,
            center_n_negative=2,
        )


def test_helpfulness_fit_rejects_foreign_outer_rows() -> None:
    rows = list(_training_rows())
    template = rows[0]
    rows.append(
        DonorCrossingRow(
            "1",
            "0",
            template.case_id,
            template.sample_id,
            template.alternative,
            template.direction,
            template.feature_values,
            template.helpful,
            template.bacc_contribution_delta,
            template.log_loss_delta,
            template.descriptor_hash,
        )
    )
    with pytest.raises(Exception):
        fit_crossing_helpfulness_model(
            rows,
            outer_target_center="0",
            training_centers=tuple(center for center in CENTERS if center != "0"),
        )


def test_degenerate_and_nonconverged_models_force_exact_p_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    single_class = tuple(row for row in _training_rows() if row.helpful == 1)
    donors = tuple(center for center in CENTERS if center != "0")
    fallback = fit_crossing_helpfulness_model(
        single_class,
        outer_target_center="0",
        training_centers=donors,
    )
    assert fallback.fit_status == "SINGLE_CLASS_DONOR_EVIDENCE_P_FALLBACK"
    assert fallback.converged is False
    assert fallback.iterations == 0
    assert predict_crossing_helpfulness(fallback, build_crossing_descriptors(_endpoint())[0]) == 0.5

    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_crossing_bagging.crossing_model as model_module

    monkeypatch.setattr(model_module, "CROSSING_IRLS_MAX_ITERATIONS", 1)
    monkeypatch.setattr(model_module, "CROSSING_IRLS_TOLERANCE", 0.0)
    nonconverged = model_module.fit_crossing_helpfulness_model(
        _training_rows(),
        outer_target_center="0",
        training_centers=donors,
    )
    assert nonconverged.fit_status == "IRLS_NONCONVERGENCE_P_FALLBACK"
    assert nonconverged.converged is False
    assert nonconverged.iterations == 1
    assert model_module.predict_crossing_helpfulness(
        nonconverged, build_crossing_descriptors(_endpoint())[0]
    ) == 0.5


def test_multi_alternative_composition_persists_per_sample_simplex() -> None:
    endpoint = EndpointCasePrediction(
        "0",
        "case-0",
        ("s0",),
        MappingProxyType(
            {
                "B": (0.8,),
                "I_OPPORTUNITY_GATED": (0.2,),
                "R_NINE_ARM_ROBUST": (0.7,),
                "P_PROTECTED": (0.2,),
            }
        ),
        "2" * 64,
    )
    descriptors = build_crossing_descriptors(endpoint)
    donors = tuple(center for center in CENTERS if center != "0")
    predictions = tuple(
        CrossingPrediction(
            row.descriptor_hash,
            0.75,
            tuple((center, 0.75) for center in donors),
            0.75,
            1.0,
            0.5,
            tuple(f"{index + 1:064x}" for index in range(9)),
        )
        for row in descriptors
    )
    composed = compose_case_probabilities(
        endpoint,
        descriptors,
        predictions,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    per_method = dict(composed.alternative_weights)
    assert len(descriptors) == 2
    assert composed.portfolio_weights[0] + sum(
        values[0] for values in per_method.values()
    ) == pytest.approx(1.0)
    expected = (
        composed.portfolio_weights[0] * 0.2
        + per_method["B"][0] * 0.8
        + per_method["R_NINE_ARM_ROBUST"][0] * 0.7
    )
    assert composed.probabilities[0] == pytest.approx(expected)
    assert composed.composition_residual_max_abs <= 1.0e-12


def test_empty_crossing_surface_fails_closed_and_persists(tmp_path: Path) -> None:
    preterminal = SimpleNamespace(
        crossing_predictions_by_policy={
            MODEL_BASED_METHOD_ID: (),
            "PDCB_BLOCKED_PERMUTATION": (),
        },
        predictions_by_center={center: () for center in CENTERS},
        crossing_descriptors_by_center={center: () for center in CENTERS},
    )
    rows, centers, summary = crossing_information_diagnostics(
        preterminal,
        {},
        primary_mean_center_bacc_delta_vs_p=0.0,
        primary_mean_center_brier_delta_vs_p=0.0,
        primary_mean_center_log_loss_delta_vs_p=0.0,
        primary_helpful_switches=0,
        primary_harmful_switches=0,
    )
    assert rows == ()
    assert len(centers) == len(CENTERS)
    assert summary["status"] == "FAIL"
    assert summary["diagnosed_bottleneck"] == "NO_ACTIONABLE_CROSSINGS_P_FALLBACK"
    assert summary["primary_proper_loss_safety_pass"] is True
    target = tmp_path / "empty.json"
    persist_rows(target, rows, schema_version="empty_test_v1", allow_empty=True)
    assert target.is_file()


def test_small_end_to_end_surface_seals_routes_before_terminal_labels() -> None:
    store_hash = "3" * 64
    centers = {}
    labels: dict[tuple[str, str, str], int] = {}
    seed_offsets = np.linspace(-0.02, 0.02, 9, dtype=np.float32)[:, None]
    for center in CENTERS:
        sample_ids = tuple(f"{center}-case-{case}-sample-{sample}" for case in range(3) for sample in range(2))
        case_ids = tuple(f"{center}-case-{case}" for case in range(3) for _sample in range(2))
        base = np.asarray((0.30, 0.70) * 3, dtype=np.float32)[None, :]
        actions = {}
        for index, action in enumerate(physical_action_ids(center)):
            mean = base if index < 2 or index % 2 == 0 else 1.0 - base
            actions[action] = np.clip(mean + seed_offsets, 0.01, 0.99).astype(np.float32)
        centers[center] = CenterProbabilitySurface(
            center,
            sample_ids,
            case_ids,
            actions,
            store_hash,
        )
        labels.update(
            {
                (center, case_id, sample_id): sample_index % 2
                for sample_index, (case_id, sample_id) in enumerate(
                    zip(case_ids, sample_ids, strict=True)
                )
            }
        )
    surface = PhysicalProbabilitySurface(centers, store_hash, strict_canonical_topology=False)

    def load(granted: frozenset[tuple[str, str, str]], role: str) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                center=center,
                case_id=case_id,
                sample_id=sample_id,
                value=labels[(center, case_id, sample_id)],
                role=role,
            )
            for center, case_id, sample_id in sorted(granted)
        )

    preterminal = build_preterminal_result(surface, load, use_processes=False)
    assert sum(len(rows) for rows in preterminal.predictions_by_center.values()) == 27
    assert preterminal.label_firewall.report_payload()["terminal_opened"] is False
    assert all(
        len(rows) == 27
        for rows in preterminal.composed_predictions_by_policy.values()
    )
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.capability_report["source_prior_grant_count"] == 72
    assert terminal.capability_report["crossing_donor_grant_count"] == 72
    assert terminal.capability_report["outer_support_grant_count"] == 27
    assert terminal.capability_report["route_decision_seal_count"] == 27
    assert terminal.diagnostic_summary["promotion_eligible"] is False
    assert terminal.diagnostic_summary["fresh_evidence"] is False
    primary = next(
        row for row in terminal.method_metrics if row["method_id"] == MODEL_BASED_METHOD_ID
    )
    assert "mean_center_bacc_delta_vs_P" in primary
    assert "mean_center_brier_delta_vs_P" in primary
    assert "mean_center_log_loss_delta_vs_P" in primary
    assert terminal.information_gate["primary_proper_loss_safety_pass"] is (
        primary["mean_center_brier_delta_vs_P"] <= 0.0
        and primary["mean_center_log_loss_delta_vs_P"] <= 0.0
    )
