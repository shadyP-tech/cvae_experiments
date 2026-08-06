from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.generation.contracts import (
    SOURCE_STREAM_NAMESPACE,
    GenerationLock,
)
from midogpp_thesis.cvae.routing.policy import (
    assignment_rows as equal_union_assignment_rows,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.bootstrap import (
    bootstrap_outer_policy,
    validate_case_confusions,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.config import (
    load_utility_regret_policy_config,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.contracts import (
    CENTERS,
    CONSUMPTION_RULE_HASH,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    BootstrapResult,
    PolicySelection,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.inputs import (
    _validate_utility_case_integrity,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.policy import (
    build_policy_assignments,
)
from midogpp_thesis.cvae.routing.utility_regret_policy.regret import (
    build_outer_regret_cells,
    summarize_candidates,
    validate_utility_rows,
)
from midogpp_thesis.cvae.routing.source_inner_utility.contracts import (
    POLICY_CONSUMPTION_LOCK_HASH,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_utility_regret_policy_lock_v1.yaml"
)


def _generation_lock() -> GenerationLock:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "claim_scope": "generation_settings_and_frame_lock",
        "bank": {
            "bank_lock_hash": "9972a41dcd4814cd",
            "expert_locks": [
                {
                    "source_center": center,
                    "training_seed": training_seed,
                    "expert_lock_hash": stable_hash(
                        {"center": center, "seed": training_seed}
                    ),
                }
                for center in CENTERS
                for training_seed in TRAINING_SEEDS
            ],
            "candidate_sources_by_target": {
                target: [center for center in CENTERS if center != target]
                for target in CENTERS
            },
        },
        "generation": {
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "source_stream_namespace": SOURCE_STREAM_NAMESPACE,
            "max_source_block_per_class": 1024,
            "equal_union_source_budget_per_class": 128,
            "total_per_class": 1024,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return GenerationLock(payload)


def _surface() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    utility: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for query in CENTERS:
        case_count = 4 if query == CENTERS[-1] else 5
        for candidate in CENTERS:
            if query == candidate:
                continue
            error_count = CENTERS.index(candidate) + 1
            bacc = 1.0 - (error_count / 50.0)
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    utility_row_id = stable_hash(
                        {
                            "query": query,
                            "candidate": candidate,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                        }
                    )
                    utility.append(
                        {
                            "utility_row_id": utility_row_id,
                            "pseudo_target_center": query,
                            "candidate_source_center": candidate,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "eval_row_count": case_count * 100,
                            "eval_class_0_count": case_count * 50,
                            "eval_class_1_count": case_count * 50,
                            "eval_case_count": case_count,
                            "bacc": bacc,
                            "macro_f1": bacc,
                            "source_stream_id": (
                                f"stream-{candidate}-{training_seed}-{generation_seed}"
                            ),
                            "eval_labels_used_for_scoring_only": True,
                            "outer_target_instantiated": False,
                            "candidate_ranking_performed": False,
                            "policy_selection_performed": False,
                            "seed_selection_performed": False,
                        }
                    )
                    for case_ordinal in range(case_count):
                        case_id = f"case-{query}-{case_ordinal}"
                        cases.append(
                            {
                                "case_confusion_row_id": stable_hash(
                                    {
                                        "utility_row_id": utility_row_id,
                                        "case_id": case_id,
                                    }
                                ),
                                "utility_row_id": utility_row_id,
                                "pseudo_target_center": query,
                                "candidate_source_center": candidate,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "case_id": case_id,
                                "case_row_hash": stable_hash(
                                    {"query": query, "case_id": case_id}
                                ),
                                "tn": 50 - error_count,
                                "fp": error_count,
                                "fn": error_count,
                                "tp": 50 - error_count,
                                "n": 100,
                                "true_class_0_count": 50,
                                "true_class_1_count": 50,
                                "eval_labels_used_for_scoring_only": True,
                            }
                        )
    return utility, cases


def _bootstrap(target: str, *, passed: bool) -> BootstrapResult:
    candidates = tuple(center for center in CENTERS if center != target)
    return BootstrapResult(
        outer_target_center=target,
        observed_best_source=candidates[0],
        observed_runner_up_source=candidates[1],
        observed_best_mean_regret=0.01,
        observed_runner_up_mean_regret=0.02,
        observed_margin=0.01,
        unique_observed_winner=True,
        unique_winner_probability=0.9 if passed else 0.5,
        margin_lower_2_5=0.001 if passed else -0.001,
        margin_upper_97_5=0.02,
        valid_replicates=2000,
        attempted_replicates=2000,
        rejected_replicates=0,
        gate_passed=passed,
        gate_reason=(
            "all_uncertainty_gates_passed"
            if passed
            else "unique_winner_probability_below_threshold"
        ),
    )


def _selection(target: str, *, passed: bool) -> PolicySelection:
    candidates = tuple(center for center in CENTERS if center != target)
    retained = (candidates[0],) if passed else candidates
    bootstrap = _bootstrap(target, passed=passed)
    return PolicySelection(
        selection_id=stable_hash({"target": target, "passed": passed}),
        target_center=target,
        action="single_source_full_budget" if passed else "fallback_equal_union",
        candidate_sources=candidates,
        selected_source=candidates[0] if passed else "",
        retained_sources=retained,
        source_budget_per_class=1024 if passed else 128,
        total_per_class=1024,
        gate_reason=bootstrap.gate_reason,
        bootstrap=bootstrap,
    )


def test_policy_config_imports_the_exact_preconsumption_rule() -> None:
    config = load_utility_regret_policy_config(CONFIG)
    assert config.expected_consumption_rule_hash == POLICY_CONSUMPTION_LOCK_HASH
    assert CONSUMPTION_RULE_HASH == POLICY_CONSUMPTION_LOCK_HASH
    assert config.policy_contract["policy_consumption_lock_hash"] == (
        POLICY_CONSUMPTION_LOCK_HASH
    )
    assert config.claim_boundary["target_labels_used"] is False
    assert config.claim_boundary["routing_quality_claimed"] is False


def test_outer_target_is_removed_from_both_q_and_e_before_regret() -> None:
    rows, _ = _surface()
    baseline = tuple(
        row.to_payload()
        for row in build_outer_regret_cells(rows)
        if row.outer_target_center == "0"
    )
    tampered = deepcopy(rows)
    for row in tampered:
        if (
            row["pseudo_target_center"] == "0"
            or row["candidate_source_center"] == "0"
        ):
            row["bacc"] = 0.0
            row["macro_f1"] = 0.0
    observed = tuple(
        row.to_payload()
        for row in build_outer_regret_cells(tampered)
        if row.outer_target_center == "0"
    )
    assert observed == baseline
    assert len(observed) == 8 * 7 * 9
    assert all(row["query_center"] != "0" for row in observed)
    assert all(row["candidate_source"] != "0" for row in observed)


def test_three_level_bootstrap_is_deterministic_and_can_pass() -> None:
    rows, cases = _surface()
    summaries = summarize_candidates(build_outer_regret_cells(rows))
    first = bootstrap_outer_policy(
        outer_target_center="0",
        summaries=summaries,
        case_rows=cases,
        valid_replicates=50,
        max_attempts=500,
        seed=123,
    )
    second = bootstrap_outer_policy(
        outer_target_center="0",
        summaries=summaries,
        case_rows=cases,
        valid_replicates=50,
        max_attempts=500,
        seed=123,
    )
    assert first == second
    assert first.gate_passed is True
    assert first.observed_best_source == "1"
    assert first.unique_winner_probability == 1.0
    assert first.margin_lower_2_5 > 0.0


def test_fallback_assignments_reuse_the_exact_equal_union_control() -> None:
    lock = _generation_lock()
    selections = tuple(_selection(target, passed=False) for target in CENTERS)
    rows = build_policy_assignments(lock, selections)
    control = equal_union_assignment_rows(lock)
    control_by_key = {
        (
            row.target_center,
            row.training_seed,
            row.generation_seed,
            row.source_center,
        ): row
        for row in control
    }
    assert len(rows) == 648
    for row in rows:
        expected = control_by_key[
            (
                row.target_center,
                row.training_seed,
                row.generation_seed,
                row.source_center,
            )
        ]
        assert row.assignment_id == expected.assignment_id
        assert row.equal_union_assignment_id == expected.assignment_id
        assert row.replicate_id == expected.replicate_id
        assert row.source_stream_id == expected.source_stream_id
        assert row.canonical_candidate_ordinal == expected.source_ordinal
        assert row.source_budget_per_class == expected.source_budget_per_class
        assert row.exact_equal_union_fallback is True


def test_single_source_action_uses_full_budget_without_seed_selection() -> None:
    lock = _generation_lock()
    selections = tuple(
        _selection(target, passed=(target == "0")) for target in CENTERS
    )
    rows = build_policy_assignments(lock, selections)
    selected = [row for row in rows if row.target_center == "0"]
    assert len(selected) == 9
    assert {row.source_center for row in selected} == {"1"}
    assert {row.source_budget_per_class for row in selected} == {1024}
    assert all(row.exact_equal_union_fallback is False for row in selected)
    assert all(row.target_center != row.source_center for row in rows)


def test_utility_surface_fails_closed_on_selection_or_q_equal_e() -> None:
    rows, _ = _surface()
    rows[0]["candidate_ranking_performed"] = True
    with pytest.raises(Exception, match="non-selecting"):
        validate_utility_rows(rows)

    rows, _ = _surface()
    rows[0]["candidate_source_center"] = rows[0]["pseudo_target_center"]
    with pytest.raises(Exception, match="Illegal candidate utility key"):
        validate_utility_rows(rows)


def test_consumer_reconstructs_complete_utility_case_surface() -> None:
    rows, cases = _surface()
    validate_utility_rows(rows)
    validate_case_confusions(cases)
    _validate_utility_case_integrity(rows, cases)


@pytest.mark.parametrize("metric", ["bacc", "macro_f1"])
def test_consumer_rejects_utility_metric_tampering(metric: str) -> None:
    rows, cases = _surface()
    rows[0][metric] = float(rows[0][metric]) - 0.01
    with pytest.raises(Exception, match="metrics do not reconstruct"):
        _validate_utility_case_integrity(rows, cases)


@pytest.mark.parametrize("tamper", ["identity", "coverage", "counts"])
def test_consumer_rejects_cross_table_integrity_tampering(tamper: str) -> None:
    rows, cases = _surface()
    if tamper == "identity":
        cases[0]["utility_row_id"] = rows[1]["utility_row_id"]
        message = "identity"
    elif tamper == "coverage":
        cases.pop()
        message = "coverage"
    else:
        cases[0]["true_class_0_count"] = 49
        message = "counts"
    with pytest.raises(Exception, match=message):
        _validate_utility_case_integrity(rows, cases)
