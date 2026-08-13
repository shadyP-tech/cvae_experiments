"""Regression against the locally sealed 9,928-row physical action surface.

This is intentionally skipped when the workstation probe surface is absent;
the compact synthetic core suite remains self-contained.  When present, this
test exercises the production formulas end-to-end and catches branch masks,
duplicate-arm collapse, donor weighting, and threshold-order regressions.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.constants import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.decisions import (
    select_arm_decisions,
    select_matched_g_decisions,
    select_nested_frequency_committee_control,
    select_raw_directional_loo_control,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.donor_priors import (
    compute_donor_priors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.ensemble import (
    DESCRIPTIVE_METHOD_IDS,
    compose_control_predictions,
    compose_descriptive_control_predictions,
    compose_method_predictions,
    fixed_physical_method_predictions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.nulls import (
    build_candidate_identity_null_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.loo_plans import (
    build_whole_case_loo_plans,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.products import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.scoring import (
    pooled_bacc,
    score_case_action_confusions,
    score_loo_directional_gains,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.terminal import (
    TERMINAL_REPORTED_METHOD_IDS,
    evaluate_terminal,
    score_terminal_predictions,
)
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    evaluation_row_id,
)


ACTUAL_ROOT = Path("/private/tmp/multi-router-analysis")


@pytest.mark.skipif(
    not (ACTUAL_ROOT / "aggregated_probability_rows.csv").is_file()
    or not (ACTUAL_ROOT / "manifest.csv").is_file(),
    reason="local sealed 9,928-row physical action surface is unavailable",
)
def test_actual_surface_dcse_center_vector_and_equal_center_gain() -> None:
    probability_rows = []
    identities = []
    with (ACTUAL_ROOT / "aggregated_probability_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for raw in csv.DictReader(handle):
            # Assert that this really is the exact-nine surface, even though the
            # science path below needs only its already-reduced float64 mean.
            assert int(raw["seed_pair_count"]) == 9
            assert len(json.loads(raw["seed_probabilities"])) == 9
            row = SimpleNamespace(
                target_center=raw["target_center"],
                case_id=raw["case_id"],
                sample_id=raw["sample_id"],
                action_id=raw["action_id"],
                probability_mean=float(raw["probability_mean"]),
            )
            row.key = (
                row.target_center,
                row.case_id,
                row.sample_id,
                row.action_id,
            )
            row.sample_key = row.key[:3]
            probability_rows.append(row)
            if row.action_id == "B":
                identities.append(row)
    assert len(probability_rows) == 99_280
    assert len(identities) == 9_928

    labels = []
    with (ACTUAL_ROOT / "manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for ordinal, raw in enumerate(csv.DictReader(handle)):
            if raw["split"] == "test" and raw["center"] in CENTERS:
                labels.append(
                    BinaryLabel(
                        raw["center"],
                        raw["case_id"],
                        evaluation_row_id(CANONICAL_MANIFEST_SHA256, ordinal),
                        int(raw["label"]),
                        "actual_surface_terminal_regression",
                    )
                )
    assert len(labels) == 9_928
    surface = SimpleNamespace(rows=tuple(probability_rows), surface_hash="actual-surface")
    counts = score_case_action_confusions(surface, labels)
    plans = build_whole_case_loo_plans(
        identities, probability_surface_hash="actual-surface"
    )
    priors = {
        target: compute_donor_priors(counts, heldout_center=target)
        for target in CENTERS
    }
    decisions = tuple(
        decision
        for plan in plans
        for decision in select_arm_decisions(
            method_id="DCSE_LOO",
            target_center=plan.target_center,
            case_id=plan.case_id,
            support_gains=score_loo_directional_gains(counts, plan),
            donor_priors=priors[plan.target_center],
        )
    )
    dcse = compose_method_predictions(surface, decisions, method_id="DCSE_LOO")
    baseline = fixed_physical_method_predictions(surface, method_id="B")
    confusions = score_terminal_predictions((*dcse, *baseline), labels)
    center_differences = []
    for center in CENTERS:
        dcse_metric = pooled_bacc(
            tuple(
                row
                for row in confusions
                if row.target_center == center and row.method_id == "DCSE_LOO"
            ),
            scope_id=center,
            method_id="DCSE_LOO",
        )
        baseline_metric = pooled_bacc(
            tuple(
                row
                for row in confusions
                if row.target_center == center and row.method_id == "B"
            ),
            scope_id=center,
            method_id="B",
        )
        center_differences.append(dcse_metric.exact - baseline_metric.exact)

    assert [round(float(value) * 100.0, 6) for value in center_differences] == [
        0.632735,
        0.925763,
        0.373832,
        0.078247,
        0.796178,
        0.0,
        0.709220,
        -0.137741,
        0.753012,
    ]
    equal_center = sum(center_differences, Fraction(0)) / len(CENTERS)
    assert round(float(equal_center) * 100.0, 6) == 0.459027

    # Complete vertical terminal reconstruction, including every pre-terminal
    # canonical and descriptive decision surface.
    gains_by_route = {
        plan.key: score_loo_directional_gains(counts, plan) for plan in plans
    }
    matched_g_decisions = tuple(
        decision
        for plan in plans
        for decision in select_matched_g_decisions(
            target_center=plan.target_center,
            case_id=plan.case_id,
            donor_priors=priors[plan.target_center],
        )
    )
    raw_decisions = tuple(
        select_raw_directional_loo_control(
            target_center=plan.target_center,
            case_id=plan.case_id,
            support_gains=gains_by_route[plan.key],
        )
        for plan in plans
    )
    frequency_decisions = tuple(
        select_nested_frequency_committee_control(
            plan=plan, support_counts=counts
        )
        for plan in plans
    )
    uniform = fixed_physical_method_predictions(surface, method_id="U")
    matched_g = compose_method_predictions(
        surface, matched_g_decisions, method_id="G_directional_matched"
    )
    raw = compose_control_predictions(
        surface, raw_decisions, method_id="DLOO_raw"
    )
    frequency = compose_control_predictions(
        surface,
        frequency_decisions,
        method_id="LOO_frequency_committee",
    )
    descriptive = compose_descriptive_control_predictions(surface, decisions)
    assert tuple(dict.fromkeys(row.method_id for row in descriptive)) == DESCRIPTIVE_METHOD_IDS
    assert len(descriptive) == len(DESCRIPTIVE_METHOD_IDS) * 9_928
    null_plan = build_candidate_identity_null_plan(tuple(plan.key for plan in plans))
    terminal = evaluate_terminal(
        probability_surface=surface,
        plans=plans,
        donor_counts=counts,
        case_action_confusions=counts,
        donor_priors=tuple(
            row for target in CENTERS for row in priors[target]
        ),
        arm_decisions=(*decisions, *matched_g_decisions),
        method_predictions=(
            *baseline,
            *uniform,
            *dcse,
            *matched_g,
            *raw,
            *frequency,
        ),
        descriptive_predictions=descriptive,
        terminal_labels=labels,
        config=SimpleNamespace(),
        null_plan=null_plan,
        aggregate_plan_decision_seal_hash="a" * 64,
    )
    assert {
        key: len(terminal[key])
        for key in (
            "case_confusions",
            "method_metrics",
            "center_metrics",
            "equal_center_contrasts",
            "delete_one_center",
            "leave_one_arm",
            "null_statistics",
        )
    } == {
        "case_confusions": len(TERMINAL_REPORTED_METHOD_IDS) * 218,
        "method_metrics": len(TERMINAL_REPORTED_METHOD_IDS),
        "center_metrics": len(TERMINAL_REPORTED_METHOD_IDS) * len(CENTERS),
        "equal_center_contrasts": 3,
        "delete_one_center": 18,
        "leave_one_arm": 9,
        "null_statistics": 1,
    }
    terminal_primary = terminal["equal_center_contrasts"][0]
    assert terminal_primary["contrast_id"] == "DCSE_LOO-B"
    assert [
        round(float(Fraction(numerator, denominator)) * 100.0, 6)
        for _center, numerator, denominator in terminal_primary[
            "center_differences"
        ]
    ] == [
        0.632735,
        0.925763,
        0.373832,
        0.078247,
        0.796178,
        0.0,
        0.709220,
        -0.137741,
        0.753012,
    ]
    null_summary = terminal["null_statistics"][0]
    assert null_summary["all_replicates_evaluated"] is True
    assert null_summary["replicates"] == 10_000
    assert null_summary["permutation_sha256"] == (
        "f9b85f048be1469ba46b48d3ba9bd87ee4bcb8066a915c06c29ffdede5c49101"
    )
    assert null_summary["null_mean"] == pytest.approx(
        0.000618912961334161, abs=1.0e-15
    )
    seal = terminal["terminal_seal"]
    assert seal["reported_method_ids"] == list(TERMINAL_REPORTED_METHOD_IDS)
    assert all(seal["descriptive_success_rubric"].values())
    assert len(seal["seal_hash"]) == 64
