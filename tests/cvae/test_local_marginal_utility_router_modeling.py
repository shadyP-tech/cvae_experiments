from __future__ import annotations

import json

import numpy as np

from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_sources,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.modeling import (
    fit_models_and_build_unscored_target_plans,
)


def _compatibility() -> dict[str, dict[str, float]]:
    center_index = {center: index for index, center in enumerate(CENTERS)}
    return {
        query: {
            source: float(
                0.35 * center_index[source]
                - 0.07 * center_index[query]
                + 0.01 * abs(center_index[source] - center_index[query])
            )
            for source in CENTERS
            if source != query
        }
        for query in CENTERS
    }


def _marginal_rows() -> tuple[dict[str, object], ...]:
    center_index = {center: index for index, center in enumerate(CENTERS)}
    output: list[dict[str, object]] = []
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            for source in legal_sources(
                outer_target=outer,
                query_center=query,
            ):
                for training_seed in TRAINING_SEEDS:
                    for generation_seed in GENERATION_SEEDS:
                        utility = (
                            -0.08 * center_index[source]
                            + 0.015 * center_index[query]
                            + 0.0001 * (training_seed - generation_seed)
                        )
                        output.append(
                            {
                                "outer_target": outer,
                                "query_center": query,
                                "source_center": source,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "marginal_bacc_utility": utility,
                                "support_labels_used": False,
                                "target_H_labels_used": False,
                                "seed_selection_performed": False,
                            }
                        )
    return tuple(output)


def test_models_use_strict_domain_loqdo_and_emit_only_unscored_feasible_plans() -> None:
    result = fit_models_and_build_unscored_target_plans(
        calibrated_energy_by_query=_compatibility(),
        marginal_utility_rows=_marginal_rows(),
        alpha_grid=(0.1,),
        kappa=1.0,
        l2_penalty=0.01,
    )

    assert len(result.learnability_prediction_rows) == 4536
    assert len(result.learnability_summary_rows) == 72
    assert len(result.model_fit_rows) == 9
    assert len(result.target_plan_rows) == 9
    assert all(
        row["heldout_query_excluded_from_fit"] is True
        and row["heldout_query_excluded_from_source_role"] is True
        and row["outer_target_excluded_from_fit"] is True
        and row["seed_selection_performed"] is False
        for row in result.learnability_prediction_rows
    )

    for row in result.model_fit_rows:
        outer = str(row["outer_target"])
        assert row["training_row_count"] == 504
        assert outer not in json.loads(str(row["training_query_centers_json"]))
        assert row["outer_target_query_excluded_from_fit"] is True
        assert row["outer_target_source_excluded_from_fit"] is True

    assert tuple(str(row["target_center"]) for row in result.target_plan_rows) == CENTERS
    for row in result.target_plan_rows:
        weights = json.loads(str(row["weights_json"]))
        allocations = json.loads(str(row["allocations_per_class_json"]))
        values = np.asarray(list(weights.values()), dtype=float)
        assert np.isclose(values.sum(), 1.0, atol=1e-8)
        assert values.min() >= -1e-10
        assert values.max() <= 0.25 + 1e-8
        assert 1.0 / float(values @ values) >= 6.0 - 1e-7
        assert sum(int(value) for value in allocations.values()) == 1024
        assert row["target_labels_used"] is False
        assert row["target_performance_scored"] is False
        assert row["geometry_transfer_status"] == (
            "extrapolative_unscored_diagnostic_only"
        )
        assert row["oracle_eligible"] is False
        assert row["may_feed_stage60"] is False
        assert row["may_feed_stage70"] is False
