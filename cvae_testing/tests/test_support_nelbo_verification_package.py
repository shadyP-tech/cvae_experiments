from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_support_nelbo_verification_package import (
    CONSERVATIVE_METHOD,
    DIRECT_METHOD,
    METHOD_LABELS,
    build_support_bootstrap_artifacts,
    build_direct_vs_conservative,
    build_expected_count_assertions,
    build_high_regret_distribution,
    build_protocol_gate,
    flatten_candidate_rows,
    annotate_decisions,
)


def _row(
    *,
    method: str = DIRECT_METHOD,
    run_seed: int = 42,
    center: int = 0,
    support_seed: int = 17,
    k: int = 4,
    selected: int = 1,
    oracle: int = 2,
    gap_pct: float = 3.0,
    support: dict[int, float] | None = None,
    eval_scores: dict[int, float] | None = None,
    routing_uses_eval_nelbo: int = 0,
    alpha: float = 0.0,
    stderr: dict[int, float] | None = None,
) -> dict:
    support = support or {1: 1.0, 2: 10.0, 3: 11.0}
    eval_scores = eval_scores or {1: 10.0, 2: 1.0, 3: 2.0}
    stderr = stderr or {expert: 0.0 for expert in support}
    candidates = sorted(support)
    return {
        "protocol_version": "support_response_candidate_specific_v1",
        "method": method,
        "run_seed": run_seed,
        "seed": run_seed,
        "run_id": f"support_utility_v2_seed{run_seed}",
        "query_domain": center,
        "fold_query_domain": center,
        "target_domain": center,
        "support_seed": support_seed,
        "support_size_requested": k,
        "support_eval_split_id": f"target{center}_seed{support_seed}_random_k{k}",
        "candidate_experts": "|".join(str(expert) for expert in candidates),
        "n_candidate_experts": len(candidates),
        "target_expert_excluded": 1,
        "routing_uses_eval_nelbo": routing_uses_eval_nelbo,
        "routing_uses_eval_domain_statistics": 0,
        "selected_expert": selected,
        "candidate_oracle_expert": oracle,
        "selected_nelbo": eval_scores[selected],
        "candidate_oracle_nelbo": eval_scores[oracle],
        "oracle_nelbo": eval_scores[oracle],
        "mean_oracle_gap_pct": gap_pct,
        "oracle_gap_pct": gap_pct,
        "top1_oracle_hit": int(selected == oracle),
        "spearman": 0.5,
        "pairwise_auc": 0.5,
        "selected_rank": 1.0,
        "alpha": alpha,
        "support_nelbo_by_expert_json": json.dumps({str(k): v for k, v in support.items()}),
        "support_stderr_nelbo_by_expert_json": json.dumps({str(k): v for k, v in stderr.items()}),
        "eval_nelbo_by_expert_json": json.dumps({str(k): v for k, v in eval_scores.items()}),
        "predicted_score_by_expert_json": json.dumps({str(k): v for k, v in support.items()}),
    }


def _split_row(*, run_seed: int = 42, center: int = 0, support_seed: int = 17, k: int = 4) -> dict:
    return {
        "run_seed": run_seed,
        "query_domain": center,
        "support_seed": support_seed,
        "support_size_requested": k,
        "split_role": "target",
        "split_status": "ok",
        "support_eval_disjoint": 1,
        "support_labels_used": 0,
    }


def _audit_row() -> dict:
    return {
        "split_row_found": 1,
        "support_eval_disjoint_ok": 1,
        "support_labels_unused_for_routing_ok": 1,
        "target_expert_excluded_ok": 1,
        "candidate_pool_excludes_target_expert_ok": 1,
        "selected_expert_in_candidate_pool_ok": 1,
        "candidate_oracle_in_candidate_pool_ok": 1,
        "routing_uses_eval_nelbo_ok": 1,
        "routing_uses_eval_domain_statistics_ok": 1,
    }


def _protocol_lock() -> dict:
    return {
        "protocol_version": "support_response_candidate_specific_v1",
        "run_id": "support_utility_v2_seed42",
        "support_raw_rows_exported": True,
        "support_raw_rows_contains_eval_nelbo": False,
        "support_raw_rows_contains_identity_fields": False,
        "support_bootstrap_posthoc_only": True,
        "bootstrap_reps": 10000,
        "bootstrap_seed": 1337,
        "conservative_alpha_selection": "source_inner_fixed",
        "support_estimated_utility": {
            "selected_before_target_eval_scoring": 1,
            "support_labels_used_for_routing": 0,
        },
    }


def _raw_rows(
    *,
    run_seed: int = 42,
    center: int = 0,
    support_seed: int = 17,
    k: int = 4,
    values_by_pos: dict[int, dict[int, float]] | None = None,
) -> list[dict]:
    values_by_pos = values_by_pos or {
        0: {1: 1.0, 2: 10.0, 3: 11.0},
        1: {1: 1.0, 2: 10.0, 3: 11.0},
        2: {1: 1.0, 2: 10.0, 3: 11.0},
        3: {1: 1.0, 2: 10.0, 3: 11.0},
    }
    out: list[dict] = []
    for pos, by_expert in values_by_pos.items():
        for expert, value in by_expert.items():
            out.append(
                {
                    "run_id": f"support_utility_v2_seed{run_seed}",
                    "experiment_seed": run_seed,
                    "heldout_center": center,
                    "support_size": k,
                    "support_seed": support_seed,
                    "support_pos_anon": pos,
                    "candidate_expert": expert,
                    "support_nelbo": value,
                    "method_family": "support_nelbo",
                    "split_id": f"target{center}_seed{support_seed}_random_k{k}",
                    "outer_fold_id": f"heldout_center_{center}",
                    "target_expert_excluded": 1,
                    "protocol_version": "support_response_candidate_specific_v1",
                }
            )
    return out


def test_candidate_flattening_uses_lower_nelbo_ranks_and_failure_labels() -> None:
    decisions = annotate_decisions([_row()])
    assert decisions[0]["regret_class"] == "high_regret"
    assert decisions[0]["support_confidence_class"] == "wrong_confident"
    assert decisions[0]["support_rank_of_eval_oracle"] == 2
    assert decisions[0]["eval_rank_of_support_selected"] == 3
    assert decisions[0]["support_margin"] == 9.0
    assert decisions[0]["eval_margin"] == 9.0

    candidates = flatten_candidate_rows(decisions)
    selected = [row for row in candidates if row["is_selected"] == 1][0]
    oracle = [row for row in candidates if row["is_eval_oracle"] == 1][0]
    assert selected["support_rank"] == 1
    assert selected["eval_rank"] == 3
    assert oracle["support_rank"] == 2
    assert oracle["eval_rank"] == 1


def test_high_regret_distribution_reports_all_thresholds() -> None:
    rows = [
        _row(gap_pct=0.5, selected=1, oracle=1),
        _row(gap_pct=2.5, support_seed=23),
        _row(gap_pct=6.0, support_seed=31),
    ]
    dist = build_high_regret_distribution(rows)
    overall = [
        row for row in dist
        if row["scope"] == "overall" and row["method"] == METHOD_LABELS[DIRECT_METHOD]
    ][0]
    assert overall["n_decisions"] == 3
    assert overall["high_regret_rate_gt1"] == 2 / 3
    assert overall["high_regret_rate_gt2"] == 2 / 3
    assert overall["high_regret_rate_gt5"] == 1 / 3


def test_expected_count_assertions_are_computed_from_split_and_candidate_rows() -> None:
    direct_rows = [
        _row(center=0, support_seed=17),
        _row(center=1, support_seed=17),
    ]
    decisions = annotate_decisions(direct_rows)
    candidates = flatten_candidate_rows(decisions)
    split_rows = [
        _split_row(center=0, support_seed=17),
        _split_row(center=1, support_seed=17),
    ]
    support_raw_rows = _raw_rows(center=0, support_seed=17) + _raw_rows(center=1, support_seed=17)
    passed = build_expected_count_assertions(
        direct_rows=direct_rows,
        direct_candidate_rows=candidates,
        split_rows=split_rows,
        support_raw_rows=support_raw_rows,
    )
    assert passed["status"] == "pass"
    assert passed["expected_decisions"] == 2
    assert passed["observed_candidates"] == 6
    assert passed["expected_support_raw_rows"] == 24
    assert passed["observed_support_raw_rows"] == 24

    failed = build_expected_count_assertions(
        direct_rows=direct_rows[:1],
        direct_candidate_rows=candidates[:3],
        split_rows=split_rows,
        support_raw_rows=support_raw_rows[:11],
    )
    assert failed["status"] == "fail"
    assert failed["expected_decisions"] == 2
    assert failed["observed_decisions"] == 1
    assert failed["support_raw_group_failure_count"] > 0


def test_protocol_gate_blocks_eval_nelbo_leakage() -> None:
    sample_rows = [_row(selected=1, oracle=2)]
    split_rows = [_split_row()]
    passed = build_protocol_gate(
        sample_rows=sample_rows,
        split_rows=split_rows,
        protocol_locks=[_protocol_lock()],
        audit_rows=[_audit_row()],
    )
    assert passed["status"] == "pass"

    leaking_rows = [_row(selected=1, oracle=2, routing_uses_eval_nelbo=1)]
    failed = build_protocol_gate(
        sample_rows=leaking_rows,
        split_rows=split_rows,
        protocol_locks=[_protocol_lock()],
        audit_rows=[_audit_row()],
    )
    assert failed["status"] == "fail"
    assert any("uses eval NELBO" in reason for reason in failed["failures"])


def test_direct_vs_conservative_disagreement_records_winner() -> None:
    direct = _row(method=DIRECT_METHOD, selected=1, oracle=2, gap_pct=3.0)
    conservative = _row(
        method="support_set_nelbo_conservative",
        selected=2,
        oracle=2,
        gap_pct=0.0,
        support={1: 5.0, 2: 1.0, 3: 8.0},
        eval_scores={1: 10.0, 2: 1.0, 3: 2.0},
    )
    rows = build_direct_vs_conservative([direct, conservative])
    assert len(rows) == 1
    assert rows[0]["agreement_flag"] == 0
    assert rows[0]["winner"] == "conservative"
    assert rows[0]["direct_minus_conservative_oracle_gap_pct"] == 3.0


def test_support_bootstrap_uses_paired_positions_and_tie_breaking() -> None:
    raw_rows = _raw_rows(
        values_by_pos={
            0: {1: 9.0, 2: 1.0, 3: 1.0},
            1: {1: 9.0, 2: 1.0, 3: 1.0},
            2: {1: 9.0, 2: 1.0, 3: 1.0},
            3: {1: 9.0, 2: 1.0, 3: 1.0},
        }
    )
    direct = _row(
        method=DIRECT_METHOD,
        selected=3,
        oracle=3,
        support={1: 9.0, 2: 1.0, 3: 0.5},
        eval_scores={1: 9.0, 2: 5.0, 3: 1.0},
    )
    conservative = _row(
        method=CONSERVATIVE_METHOD,
        selected=3,
        oracle=3,
        support={1: 9.0, 2: 1.0, 3: 0.5},
        eval_scores={1: 9.0, 2: 5.0, 3: 1.0},
        alpha=2.0,
    )
    stability, summary, margin, status = build_support_bootstrap_artifacts(
        sample_rows=[direct, conservative, _row(method="support_metadata_routing", selected=2, oracle=3)],
        support_raw_rows=raw_rows,
        bootstrap_reps=16,
        bootstrap_seed=7,
    )
    assert status["status"] == "pass"
    direct_row = [row for row in stability if row["source_method"] == DIRECT_METHOD][0]
    assert direct_row["deterministic_selected_expert"] == 3
    assert direct_row["selection_stability"] == 1.0
    assert direct_row["p_oracle_selected"] == 1.0
    assert summary
    assert margin


def test_support_bootstrap_conservative_uses_fixed_alpha_and_stderr() -> None:
    raw_rows = _raw_rows(
        values_by_pos={
            0: {1: 0.0, 2: 1.2, 3: 4.0},
            1: {1: 0.0, 2: 1.2, 3: 4.0},
            2: {1: 2.0, 2: 1.2, 3: 4.0},
            3: {1: 2.0, 2: 1.2, 3: 4.0},
        }
    )
    direct = _row(
        method=DIRECT_METHOD,
        selected=1,
        oracle=2,
        support={1: 1.0, 2: 1.2, 3: 4.0},
        eval_scores={1: 4.0, 2: 1.0, 3: 8.0},
    )
    conservative = _row(
        method=CONSERVATIVE_METHOD,
        selected=2,
        oracle=2,
        support={1: 1.0, 2: 1.2, 3: 4.0},
        stderr={1: 0.577350269, 2: 0.0, 3: 0.0},
        eval_scores={1: 4.0, 2: 1.0, 3: 8.0},
        alpha=10.0,
    )
    stability, summary, _margin, _status = build_support_bootstrap_artifacts(
        sample_rows=[direct, conservative, _row(method="support_metadata_routing", selected=1, oracle=2)],
        support_raw_rows=raw_rows,
        bootstrap_reps=128,
        bootstrap_seed=11,
    )
    conservative_row = [row for row in stability if row["source_method"] == CONSERVATIVE_METHOD][0]
    direct_row = [row for row in stability if row["source_method"] == DIRECT_METHOD][0]
    assert conservative_row["p_oracle_selected"] > direct_row["p_oracle_selected"]
    assert conservative_row["selection_stability"] >= 0.75
    summary_row = [
        row for row in summary
        if row["source_method"] == CONSERVATIVE_METHOD and int(row["support_size"]) == 4
    ][0]
    assert summary_row["spearman_mean"] > 0.0
