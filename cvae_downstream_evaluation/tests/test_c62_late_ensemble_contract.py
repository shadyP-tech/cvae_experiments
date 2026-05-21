from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c62_late_ensemble import (  # noqa: E402
    GLOBAL_CLASS_ORDER,
    POLICY_FIXED_TOTAL,
    POLICY_METADATA_WEIGHTED,
    POLICY_SAFE_MULTI,
    align_probabilities_to_class_order,
    assert_c62_prejoin_rows_safe,
    build_c62_ensemble_plans,
    ensemble_weight_diagnostics,
    fixed_predictions_from_probabilities,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_c62_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c62_late_probability_ensemble.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c61-artifacts-root" in result.stdout
    assert "--limit-generation-seeds" in result.stdout


def test_safe_multiseed_policy_enumerates_all_safe_members_and_excludes_heldout() -> None:
    plans = build_c62_ensemble_plans(
        policies=(POLICY_SAFE_MULTI,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17, 23, 31),
    )

    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.specs) == 48
    assert {spec.source_expert for spec in plan.specs} == {"1", "2", "3", "4"}
    assert {spec.bank.mode_label for spec in plan.specs} == {"hetero_mean", "gmm_k1", "gmm_k2", "standard_prior"}
    assert {spec.generation_seed for spec in plan.specs} == {17, 23, 31}
    assert min(spec.allocated_budget_per_class for spec in plan.specs) >= 8
    assert round(sum(spec.weight for spec in plan.specs), 12) == 1.0


def test_fixed_total_draw_control_conserves_single_seed_budget_across_generation_seeds() -> None:
    plans = build_c62_ensemble_plans(
        policies=(POLICY_FIXED_TOTAL,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17, 23, 31),
    )

    plan = plans[0]
    assert len(plan.specs) == 48
    assert sum(spec.allocated_budget_per_class for spec in plan.specs) == 128
    assert min(spec.allocated_budget_per_class for spec in plan.specs) < 8
    assert all(spec.fixed_total_draw_control == 1 for spec in plan.specs)


def test_metadata_weighted_policy_applies_prejoin_selected_expert_weight() -> None:
    support_rows = [
        {
            "method": "support_metadata_routing",
            "selected_expert": "2",
            "support_nelbo_by_expert_json": "{}",
        }
    ]
    plan = build_c62_ensemble_plans(
        policies=(POLICY_METADATA_WEIGHTED,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17,),
        support_rows=support_rows,
    )[0]

    expert_weight = {
        expert: sum(spec.weight for spec in plan.specs if spec.source_expert == expert)
        for expert in {"1", "2", "3", "4"}
    }
    assert expert_weight["2"] > expert_weight["1"]
    assert round(sum(expert_weight.values()), 12) == 1.0
    assert all("metadata_selected_expert" in spec.weight_source for spec in plan.specs)


def test_class_probability_alignment_maps_to_global_class_order_and_rejects_mismatch() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    aligned = align_probabilities_to_class_order(np.array([[0.2, 0.8], [0.7, 0.3]]), classes=(1, 0), global_class_order=GLOBAL_CLASS_ORDER)

    assert aligned.tolist() == [[0.8, 0.2], [0.3, 0.7]]
    try:
        align_probabilities_to_class_order(np.array([[1.0]]), classes=(1,), global_class_order=GLOBAL_CLASS_ORDER)
    except ProtocolError:
        pass
    else:
        raise AssertionError("C6.2 accepted a member missing the global class order")


def test_locked_binary_prediction_rule_uses_positive_probability_threshold() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    predictions = fixed_predictions_from_probabilities(np.array([[0.51, 0.49], [0.50, 0.50], [0.2, 0.8]]), GLOBAL_CLASS_ORDER)

    assert predictions == [0, 1, 1]


def test_ensemble_weight_diagnostics_report_effective_member_count() -> None:
    plan = build_c62_ensemble_plans(
        policies=(POLICY_SAFE_MULTI,),
        candidates=("1", "2", "3", "4"),
        total_budget_per_class=128,
        generation_seeds=(17,),
    )[0]
    diagnostics = ensemble_weight_diagnostics(plan.specs)

    assert diagnostics["num_members"] == 16.0
    assert diagnostics["effective_num_members"] > 10.0
    assert diagnostics["member_budget_min"] >= 8


def test_c62_prejoin_guard_rejects_labels_and_utility_columns() -> None:
    assert_c62_prejoin_rows_safe([{"member_key": "expert_1::hetero_mean::seed_17"}])
    for bad_key in ("selected_bacc", "target_label", "current_heldout_utility"):
        try:
            assert_c62_prejoin_rows_safe([{"member_key": "x", bad_key: 1}])
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"C6.2 pre-join guard accepted forbidden column {bad_key}")
