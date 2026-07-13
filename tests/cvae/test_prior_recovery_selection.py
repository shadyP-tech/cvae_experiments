from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.generation_samplers import DIAGONAL_SAMPLER, FULL_SAMPLER, STANDARD_SAMPLER
from midogpp_thesis.cvae.objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from midogpp_thesis.cvae.preservation.source_inner_selection import (
    InnerCenterMetric,
    load_recipe_lock,
    select_recipe_lock,
    write_recipe_lock,
)


def test_sampler_gate_selects_diagonal_on_near_tie() -> None:
    metrics = _metrics(diagonal_gain=0.08, full_gain=0.085, task_gain=None)
    lock = _select(metrics)
    assert lock.status == "VALID"
    assert lock.primary_arm == "C"
    assert lock.sampler_family == DIAGONAL_SAMPLER
    assert lock.objective_id == ISOTROPIC_OBJECTIVE


def test_task_fisher_is_selected_only_when_it_adds_value_and_is_safe() -> None:
    metrics = _metrics(diagonal_gain=0.08, full_gain=0.01, task_gain=0.03)
    lock = _select(metrics)
    assert lock.primary_arm == "D"
    assert lock.objective_id == TASK_FISHER_OBJECTIVE


def test_invalid_denominator_fails_complete_lock_closed() -> None:
    metrics = list(_metrics(diagonal_gain=0.08, full_gain=0.01, task_gain=None))
    metrics[0] = InnerCenterMetric(**(metrics[0].__dict__ | {"real_reference_bacc": 0.54}))
    lock = _select(metrics)
    assert lock.status == "INVALID"
    assert not lock.may_feed_model_recipe


def test_required_factorial_fails_closed_without_task_fisher_cells() -> None:
    lock = _select(
        _metrics(diagonal_gain=0.08, full_gain=0.01, task_gain=None),
        require_task_factorial=True,
    )
    assert lock.status == "INVALID"
    assert lock.reason == "incomplete_task_fisher_factorial"


def test_recipe_lock_hash_round_trip(tmp_path: Path) -> None:
    lock = _select(_metrics(diagonal_gain=0.08, full_gain=0.01, task_gain=None))
    path = tmp_path / "0.json"
    write_recipe_lock(path, lock)
    loaded = load_recipe_lock(path)
    assert loaded.hash == lock.hash
    assert loaded.to_payload() == lock.to_payload()


def _select(
    metrics: tuple[InnerCenterMetric, ...],
    *,
    require_task_factorial: bool = False,
):
    return select_recipe_lock(
        metrics,
        outer_target_center="0",
        expected_inner_centers=("1", "2", "3"),
        generation_seeds=(17, 42, 101),
        beta_final=0.001,
        classifier_grid_hash="grid",
        protocol_hash="protocol",
        fit_center_sets_hash="fits",
        recipe_contract_hash="recipe",
        selection_bundle_hash="bundle",
        source_metric_table_hash="metrics",
        gate_min_inner_wins=2,
        require_task_factorial=require_task_factorial,
    )


def _metrics(*, diagonal_gain: float, full_gain: float, task_gain: float | None) -> tuple[InnerCenterMetric, ...]:
    rows = []
    for center in ("1", "2", "3"):
        baseline = 0.65
        common = {
            "outer_target_center": "0",
            "inner_pseudo_target_center": center,
            "decode_bacc": 0.72,
            "posterior_bacc": 0.71,
            "real_reference_bacc": 0.70,
        }
        rows.append(
            InnerCenterMetric(
                **common,
                arm="A",
                sampler_family=STANDARD_SAMPLER,
                objective_id=ISOTROPIC_OBJECTIVE,
                prior_ratio=baseline,
            )
        )
        rows.append(
            InnerCenterMetric(
                **common,
                arm="C",
                sampler_family=DIAGONAL_SAMPLER,
                objective_id=ISOTROPIC_OBJECTIVE,
                prior_ratio=baseline + diagonal_gain,
            )
        )
        rows.append(
            InnerCenterMetric(
                **common,
                arm="C",
                sampler_family=FULL_SAMPLER,
                objective_id=ISOTROPIC_OBJECTIVE,
                prior_ratio=baseline + full_gain,
            )
        )
        if task_gain is not None:
            rows.append(
                InnerCenterMetric(
                    **common,
                    arm="B",
                    sampler_family=STANDARD_SAMPLER,
                    objective_id=TASK_FISHER_OBJECTIVE,
                    prior_ratio=baseline + 0.01,
                )
            )
            rows.append(
                InnerCenterMetric(
                    **common,
                    arm="D",
                    sampler_family=DIAGONAL_SAMPLER,
                    objective_id=TASK_FISHER_OBJECTIVE,
                    prior_ratio=baseline + diagonal_gain + task_gain,
                )
            )
    return tuple(rows)
