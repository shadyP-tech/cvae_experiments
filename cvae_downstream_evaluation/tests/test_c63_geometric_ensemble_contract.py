from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c63_geometric_ensemble import (  # noqa: E402
    GEOMETRIC_SOFTMAX_TEMPERATURE,
    LOG_PROBABILITY_EPSILON,
    POLICY_C62_REPLAY,
    POLICY_GEOM_FIXED_TOTAL,
    POLICY_GEOM_SAFE_MULTI,
    build_c63_ensemble_plans,
    geometric_pool_probabilities,
    normalize_weights,
)
from cvae_downstream_evaluation.c62_late_ensemble import GLOBAL_CLASS_ORDER, assert_c62_prejoin_rows_safe  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_c63_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c63_geometric_late_ensemble.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c62-artifacts-root" in result.stdout
    assert "--limit-generation-seeds" in result.stdout


def test_geometric_pooling_uses_weighted_log_probability_argmax() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    probs = np.asarray(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        ],
        dtype=float,
    )
    scores, geometric = geometric_pool_probabilities(probs, [0.25, 0.75])
    expected_scores = (np.log(np.clip(probs, LOG_PROBABILITY_EPSILON, 1.0)) * np.asarray([0.25, 0.75])[:, None, None]).sum(axis=0)

    assert np.allclose(scores, expected_scores)
    assert np.argmax(geometric, axis=1).tolist() == [0, 1]
    assert np.allclose(geometric.sum(axis=1), 1.0)


def test_geometric_pooling_rejects_temperature_tuning() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    try:
        geometric_pool_probabilities(np.ones((1, 2, 2)) * 0.5, [1.0], temperature=0.5)
    except ProtocolError:
        pass
    else:
        raise AssertionError("C6.3 accepted non-unit geometric softmax temperature")


def test_weight_normalization_is_nonnegative_and_sums_to_one() -> None:
    weights = normalize_weights([-1.0, 0.0, 3.0])

    assert all(value >= 0.0 for value in weights)
    assert abs(sum(weights) - 1.0) < 1.0e-12
    assert weights == (0.0, 0.0, 1.0)


def test_c63_plan_mapping_preserves_safe_multiseed_member_bank() -> None:
    plans = build_c63_ensemble_plans(
        policies=(POLICY_GEOM_SAFE_MULTI,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17, 23, 31),
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.policy == POLICY_GEOM_SAFE_MULTI
    assert len(plan.specs) == 48
    assert {spec.source_expert for spec in plan.specs} == {"1", "2", "3", "4"}
    assert {spec.bank.mode_label for spec in plan.specs} == {"hetero_mean", "gmm_k1", "gmm_k2", "standard_prior"}
    assert {spec.generation_seed for spec in plan.specs} == {17, 23, 31}
    assert round(sum(spec.weight for spec in plan.specs), 12) == 1.0


def test_c63_c62_replay_uses_c62_safe_multiseed_source_policy() -> None:
    plans = build_c63_ensemble_plans(
        policies=(POLICY_C62_REPLAY,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17, 23, 31),
    )

    assert len(plans) == 1
    assert plans[0].policy == POLICY_C62_REPLAY
    assert plans[0].diagnostic_only == 1
    assert len(plans[0].specs) == 48


def test_c63_fixed_total_control_conserves_budget_across_generation_seeds() -> None:
    plan = build_c63_ensemble_plans(
        policies=(POLICY_GEOM_FIXED_TOTAL,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17, 23, 31),
    )[0]

    assert sum(spec.allocated_budget_per_class for spec in plan.specs) == 128
    assert min(spec.allocated_budget_per_class for spec in plan.specs) < 8
    assert all(spec.fixed_total_draw_control == 1 for spec in plan.specs)


def test_c63_prejoin_guard_rejects_target_or_utility_columns() -> None:
    assert_c62_prejoin_rows_safe([{"member_key": "expert_1::hetero_mean::seed_17"}])
    for bad_key in ("selected_bacc", "target_label", "oracle_expert", "current_heldout_utility"):
        try:
            assert_c62_prejoin_rows_safe([{"member_key": "x", bad_key: 1}])
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"C6.3 pre-join guard accepted forbidden column {bad_key}")


def test_c63_global_class_order_remains_binary_locked_v1() -> None:
    assert GLOBAL_CLASS_ORDER == (0, 1)
    assert GEOMETRIC_SOFTMAX_TEMPERATURE == 1.0
