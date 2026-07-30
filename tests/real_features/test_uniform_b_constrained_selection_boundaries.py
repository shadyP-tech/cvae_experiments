from pathlib import Path

import pytest

from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_sens_spec_constrained_nystroem_probe.config import (
    load_constrained_nystroem_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_sens_spec_constrained_nystroem_probe.estimator import (
    select_constrained_candidates,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_sens_spec_constrained_nystroem_probe_v1.yaml"
)


def test_strict_zero_mean_constraints_fail_closed() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    rows = _uniform_rows(config, delta=0.0)
    _, selected = select_constrained_candidates(rows, config)
    assert all(row["fallback"] is True for row in selected.values())


def test_inclusive_per_inner_bounds_are_accepted_when_means_are_positive() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    rows = _uniform_rows(config, delta=0.01)
    target = next(
        row
        for row in rows
        if row["outer_center"] == "0"
        and row["inner_center"] == "1"
        and row["objective"] == "canonical_class_weight"
        and row["alpha"] == 0.25
    )
    target["delta_recall"] = -0.02
    target["delta_specificity"] = -0.02
    target["delta_bacc"] = -0.01
    _, selected = select_constrained_candidates(rows, config)
    assert selected["0"]["fallback"] is False


def test_duplicate_inner_cell_is_rejected() -> None:
    config = load_constrained_nystroem_config(CONFIG)
    rows = _uniform_rows(config, delta=0.01)
    rows.append(dict(rows[0]))
    with pytest.raises(ProtocolError, match="coverage"):
        select_constrained_candidates(rows, config)


def _uniform_rows(config: object, *, delta: float) -> list[dict[str, object]]:
    return [
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
        for outer in config.heldout_centers
        for inner in config.heldout_centers
        if inner != outer
        for objective in config.objectives
        for alpha in config.alphas
    ]
