from pathlib import Path

from midogpp_thesis.real_features.classifier_reference.uniform_b_sens_spec_constrained_nystroem_probe.config import (
    ALPHAS,
    BOUNDED_SHRINKAGE_ALPHAS,
    OBJECTIVES,
    load_constrained_nystroem_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_sens_spec_constrained_nystroem_probe.estimator import (
    select_constrained_candidates,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_sens_spec_constrained_nystroem_probe_v1.yaml"
)
SHRINKAGE_CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_bounded_shrinkage_probe_v1.yaml"
)


def test_constrained_grid_is_bounded_and_threshold_is_fixed() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    assert config.objectives == OBJECTIVES
    assert config.alphas == ALPHAS
    assert config.threshold == 0.5
    assert config.fallback_alpha == 0.0
    assert config.fallback_role == "exact_linear_b"
    assert config.pair_jobs == 4
    assert config.threads_per_job == 3
    assert config.claim_boundary["validation_scored"] is False



def test_bounded_shrinkage_grid_reuses_base_scores_and_stays_fixed() -> None:
    config = load_constrained_nystroem_config(SHRINKAGE_CONFIG)
    assert config.alphas == BOUNDED_SHRINKAGE_ALPHAS
    assert len(config.alphas) == 20
    assert config.alphas[0] == 0.02
    assert config.alphas[-1] == 0.4
    assert config.source_inner_replay_root is not None
    summaries, selected = select_constrained_candidates(
        _cells(config, feasible=True), config
    )
    assert len(summaries) == 9 * 4 * 20
    assert all(row["alpha"] == 0.02 for row in selected.values())

def test_constrained_selector_uses_feasible_capacity_and_deterministic_tie() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    cells = _cells(config, feasible=True)
    summaries, selected = select_constrained_candidates(cells, config)
    assert len(summaries) == 9 * 4 * 4
    assert set(selected) == set(config.heldout_centers)
    assert all(row["fallback"] is False for row in selected.values())
    assert all(row["objective"] == "canonical_class_weight" for row in selected.values())
    assert all(row["alpha"] == 0.25 for row in selected.values())


def test_constrained_selector_fails_closed_to_exact_linear() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    _, selected = select_constrained_candidates(_cells(config, feasible=False), config)
    assert all(row["fallback"] is True for row in selected.values())
    assert all(row["alpha"] == 0.0 for row in selected.values())
    assert all(
        row["candidate_id"] == "exact_linear_b__alpha_0"
        for row in selected.values()
    )


def _cells(config: object, *, feasible: bool) -> list[dict[str, object]]:
    rows = []
    delta = 0.01 if feasible else -0.03
    for outer in config.heldout_centers:
        for inner in config.heldout_centers:
            if inner == outer:
                continue
            for objective in config.objectives:
                for alpha in config.alphas:
                    rows.append(
                        {
                            "outer_center": outer,
                            "inner_center": inner,
                            "objective": objective,
                            "alpha": alpha,
                            "bacc": 0.8 + delta,
                            "delta_bacc": delta,
                            "delta_recall": delta,
                            "delta_specificity": delta,
                        }
                    )
    return rows
