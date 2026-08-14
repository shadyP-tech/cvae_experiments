"""Optional regression on the local 9,928-row consumed-test probability surface."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.constants import CENTERS, candidate_sources
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.correctness_model import fit_route_correctness_models
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.correctness_observations import score_route_correctness_observations, support_class_denominators
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.donor_prior import compute_donor_priors
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.held_case_features import build_label_free_features
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.identification import select_case_identification_decision
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.identification_metrics import (
    build_case_directional_oracles,
    build_static_directional_oracles,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.predictions import compose_identification_predictions, compose_physical_action_predictions, compose_robust_predictions
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.composition import compose_portfolio_predictions
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.response_products import BinaryLabel
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.response_scoring import score_case_action_confusions, score_loo_directional_gains
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.robust import select_robust_arm_decisions
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.split_plans import build_whole_case_loo_plans
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.terminal_tables import oracle_prediction_views
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import CANONICAL_MANIFEST_SHA256, evaluation_row_id


ACTUAL_ROOT = Path("/private/tmp/multi-router-analysis")


@dataclass(frozen=True)
class _ProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability_mean: float
    seed_probabilities: tuple[float, ...]

    @property
    def key(self):
        return self.target_center, self.case_id, self.sample_id, self.action_id

    @property
    def sample_key(self):
        return self.target_center, self.case_id, self.sample_id

    @property
    def hard_prediction(self):
        return int(self.probability_mean >= 0.5)


def _equal_center_bacc(predictions, labels) -> float:
    truth = {row.key: row.value for row in labels}
    predicted = {row.key: row.hard_prediction for row in predictions}
    values = []
    for center in CENTERS:
        keys = tuple(key for key in truth if key[0] == center)
        y = np.asarray([truth[key] for key in keys], dtype=np.int8)
        p = np.asarray([predicted[key] for key in keys], dtype=np.int8)
        values.append(
            0.5
            * (
                float(np.mean(p[y == 1] == 1, dtype=np.float64))
                + float(np.mean(p[y == 0] == 0, dtype=np.float64))
            )
        )
    return float(np.mean(values, dtype=np.float64))


@pytest.mark.skipif(
    not (ACTUAL_ROOT / "aggregated_probability_rows.csv").is_file()
    or not (ACTUAL_ROOT / "manifest.csv").is_file(),
    reason="local sealed consumed-test surface is unavailable",
)
def test_actual_surface_reproduces_frozen_dual_endpoint_diagnostic() -> None:
    rows = []
    identities = []
    with (ACTUAL_ROOT / "aggregated_probability_rows.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            seeds = tuple(float(value) for value in json.loads(raw["seed_probabilities"]))
            assert len(seeds) == int(raw["seed_pair_count"]) == 9
            row = _ProbabilityRow(
                raw["target_center"], raw["case_id"], raw["sample_id"], raw["action_id"],
                float(raw["probability_mean"]), seeds,
            )
            rows.append(row)
            if row.action_id == "B":
                identities.append(row)
    labels = []
    with (ACTUAL_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for ordinal, raw in enumerate(csv.DictReader(handle)):
            if raw["split"] == "test" and raw["center"] in CENTERS:
                labels.append(
                    BinaryLabel(
                        raw["center"], raw["case_id"],
                        evaluation_row_id(CANONICAL_MANIFEST_SHA256, ordinal),
                        int(raw["label"]), "actual_surface_terminal_regression",
                    )
                )
    assert len(rows) == 99_280
    assert len(labels) == len(identities) == 9_928
    surface = SimpleNamespace(rows=tuple(rows), surface_hash="actual-surface")
    plans = build_whole_case_loo_plans(identities, probability_surface_hash="actual-surface")
    features = build_label_free_features(surface)
    full_counts = score_case_action_confusions(surface, labels)
    single_class_cases = {
        (row.target_center, row.case_id)
        for row in full_counts
        if row.action_id == "B" and (row.n_positive == 0 or row.n_negative == 0)
    }
    assert single_class_cases == {
        ("5", "129"),
        ("7", "042"),
        ("7", "044"),
        ("8", "507"),
        ("8", "508"),
    }
    case_oracles = build_case_directional_oracles(full_counts)
    static_oracles = build_static_directional_oracles(full_counts)
    assert len(case_oracles) == 436
    assert len(static_oracles) == 18
    assert sum(row.selected_source is None for row in case_oracles) == 268
    assert sum(row.selected_source is None for row in static_oracles) == 2
    case_index = {row.key: row for row in case_oracles}
    static_index = {
        (row.target_center, row.direction): row for row in static_oracles
    }
    for target in CENTERS:
        target_cases = tuple(
            sorted(
                {
                    row.case_id
                    for row in full_counts
                    if row.target_center == target
                }
            )
        )
        for direction in ("zero_to_one", "one_to_zero"):
            for source in candidate_sources(target):
                assert sum(
                    (
                        case_index[(target, case, direction)].utility_for(source)
                        for case in target_cases
                    ),
                    start=Fraction(0),
                ) == static_index[(target, direction)].utility_for(source)
    oracle_views = oracle_prediction_views(surface, case_oracles, static_oracles)
    for method_id, expected in (
        ("O_DIRECTIONAL_STATIC", 0.8112101586571822),
        ("O_CASE_DIRECTIONAL", 0.8330926502994553),
    ):
        predictions = tuple(
            SimpleNamespace(
                key=(row["target_center"], row["case_id"], row["sample_id"]),
                hard_prediction=row["hard_prediction"],
            )
            for row in oracle_views
            if row["method_id"] == method_id
        )
        assert _equal_center_bacc(predictions, labels) == pytest.approx(
            expected, abs=1.0e-15
        )
    priors = {
        target: compute_donor_priors(
            {source: full_counts for source in candidate_sources(target)},
            heldout_center=target,
        )
        for target in CENTERS
    }
    identification = []
    robust = []
    for plan in plans:
        route_cases = {*plan.support_case_ids, plan.case_id}
        route_features = tuple(
            row for row in features
            if row.target_center == plan.target_center and row.case_id in route_cases
        )
        support_labels = tuple(
            row for row in labels
            if row.target_center == plan.target_center and row.case_id in plan.support_case_ids
        )
        observations = score_route_correctness_observations(
            surface, support_labels, plan, features=route_features
        )
        denominators = support_class_denominators(
            support_labels, plan, probability_surface_or_rows=surface
        )
        models = fit_route_correctness_models(observations, plan)
        identification.append(
            select_case_identification_decision(
                plan, route_features, models, denominators, priors[plan.target_center]
            )
        )
        support_counts = tuple(
            row for row in full_counts
            if row.target_center == plan.target_center and row.case_id in plan.support_case_ids
        )
        gains = score_loo_directional_gains(support_counts, plan)
        robust.extend(select_robust_arm_decisions(plan, gains, priors[plan.target_center]))
    baseline = compose_physical_action_predictions(surface, action_id="B")
    i_predictions = compose_identification_predictions(surface, identification)
    r_predictions = compose_robust_predictions(surface, robust)
    portfolio = compose_portfolio_predictions(i_predictions, r_predictions)
    assert _equal_center_bacc(baseline, labels) == pytest.approx(0.8008955128104029, abs=1.0e-15)
    assert _equal_center_bacc(r_predictions, labels) == pytest.approx(0.8054857861278540, abs=1.0e-15)
    assert _equal_center_bacc(portfolio, labels) == pytest.approx(0.8073167788170417, abs=1.0e-15)
