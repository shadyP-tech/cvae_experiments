from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_compatibility_decision_table import _aggregate, _read_rows
from scripts.compatibility_stability import (
    LEGACY_STD_POLICY,
    SIGN_CI_POLICY,
    effective_positive_threshold,
)


PROTOCOL_VERSION = "learned_utility_loqdo_candidate_exclusion_v2"


def _metric(
    *,
    method_role: str,
    adoption_eligible: int,
    diagnostic_only: int,
    top1: float,
    spearman: float,
    gap_pct: float,
    routing_uses_eval_nelbo: int = 0,
    routing_uses_eval_domain_statistics: int = 0,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict:
    return {
        "protocol_version": protocol_version,
        "method_role": method_role,
        "adoption_eligible": float(adoption_eligible),
        "diagnostic_only": float(diagnostic_only),
        "routing_uses_eval_nelbo": float(routing_uses_eval_nelbo),
        "routing_uses_eval_domain_statistics": float(routing_uses_eval_domain_statistics),
        "top1_oracle_hit": float(top1),
        "spearman": float(spearman),
        "mean_oracle_gap_pct": float(gap_pct),
    }


def _write_result(
    path: Path,
    *,
    protocol_version: str = PROTOCOL_VERSION,
    methods: dict,
    domain_rows: list[dict] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": protocol_version,
        "protocol_contract": {"protocol_version": protocol_version},
        "metrics_by_method": methods,
        "artifacts": {"domain_breakdown": "learned_utility_domain_breakdown.csv"},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if domain_rows is not None:
        csv_path = path.parent / "learned_utility_domain_breakdown.csv"
        fieldnames = [
            "protocol_version",
            "method",
            "query_domain",
            "top1_oracle_hit",
            "spearman",
            "mean_oracle_gap_pct",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(domain_rows)
    return path


def _decision_rows(
    result_paths: list[Path],
    *,
    decision_policy_version: str = LEGACY_STD_POLICY,
    allow_missing_domain_breakdown_as_diagnostic: bool = False,
) -> list[dict]:
    rows = _read_rows(result_paths, uplift_reference_method="metadata_routing")
    out_rows, _summary = _aggregate(
        rows=rows,
        uplift_reference_method="metadata_routing",
        min_improving_seeds=2,
        strong={
            "spearman_uplift_min": 0.05,
            "top1_uplift_min": 0.10,
            "oracle_gap_pct_reduction_min": 5.0,
        },
        weak={
            "spearman_uplift_min": 0.025,
            "top1_uplift_min": 0.05,
            "oracle_gap_pct_reduction_min": 2.5,
        },
        instability_std_threshold=0.05,
        instability_sign_inconsistency_min_count=2,
        decision_policy_version=decision_policy_version,
        top1_uplift_std_threshold=0.05,
        spearman_uplift_std_threshold=0.05,
        gap_pct_reduction_std_threshold=3.0,
        ci_bootstrap_reps=200,
        allow_missing_domain_breakdown_as_diagnostic=allow_missing_domain_breakdown_as_diagnostic,
    )
    return out_rows


def _domain_rows(
    *,
    query_domain: int,
    baseline_top1: float,
    baseline_spearman: float,
    baseline_gap_pct: float,
    candidate_method: str,
    candidate_top1: float,
    candidate_spearman: float,
    candidate_gap_pct: float,
) -> list[dict]:
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "query_domain": int(query_domain),
    }
    return [
        {
            **base,
            "method": "metadata_routing",
            "top1_oracle_hit": float(baseline_top1),
            "spearman": float(baseline_spearman),
            "mean_oracle_gap_pct": float(baseline_gap_pct),
        },
        {
            **base,
            "method": str(candidate_method),
            "top1_oracle_hit": float(candidate_top1),
            "spearman": float(candidate_spearman),
            "mean_oracle_gap_pct": float(candidate_gap_pct),
        },
    ]


def test_candidate_oracle_is_reference_only_even_with_perfect_metrics(tmp_path: Path) -> None:
    result_paths = [
        _write_result(
            tmp_path / "run_seed42" / "learned_utility_results.json",
            methods={
                "metadata_routing": _metric(
                    method_role="baseline",
                    adoption_eligible=1,
                    diagnostic_only=0,
                    top1=0.2,
                    spearman=0.1,
                    gap_pct=50.0,
                ),
                "candidate_oracle_routing": _metric(
                    method_role="diagnostic",
                    adoption_eligible=0,
                    diagnostic_only=1,
                    routing_uses_eval_nelbo=1,
                    top1=1.0,
                    spearman=1.0,
                    gap_pct=0.0,
                ),
                "linear_regressor": _metric(
                    method_role="learned",
                    adoption_eligible=1,
                    diagnostic_only=0,
                    top1=0.5,
                    spearman=0.4,
                    gap_pct=35.0,
                ),
            },
        ),
        _write_result(
            tmp_path / "run_seed43" / "learned_utility_results.json",
            methods={
                "metadata_routing": _metric(
                    method_role="baseline",
                    adoption_eligible=1,
                    diagnostic_only=0,
                    top1=0.2,
                    spearman=0.1,
                    gap_pct=60.0,
                ),
                "candidate_oracle_routing": _metric(
                    method_role="diagnostic",
                    adoption_eligible=0,
                    diagnostic_only=1,
                    routing_uses_eval_nelbo=1,
                    top1=1.0,
                    spearman=1.0,
                    gap_pct=0.0,
                ),
                "linear_regressor": _metric(
                    method_role="learned",
                    adoption_eligible=1,
                    diagnostic_only=0,
                    top1=0.55,
                    spearman=0.45,
                    gap_pct=45.0,
                ),
            },
        ),
    ]

    by_method = {row["method"]: row for row in _decision_rows(result_paths)}

    oracle = by_method["candidate_oracle_routing"]
    assert oracle["tier"] == "reference_only"
    assert oracle["decision"] == "not_selected"
    assert oracle["selection_eligible"] == 0
    assert oracle["raw_instability_breach"] == 1
    assert oracle["instability_gate_applied"] == 0
    assert oracle["instability_breach"] == 0

    baseline = by_method["metadata_routing"]
    assert baseline["tier"] == "baseline"
    assert baseline["decision"] == "baseline_reference"
    assert baseline["selection_eligible"] == 0

    learned = by_method["linear_regressor"]
    assert learned["selection_eligible"] == 1
    assert learned["decision"] == "selected"
    assert learned["tier"] == "strong_pass"


def test_sign_ci_policy_uses_two_of_three_and_does_not_hard_veto_gap_pct_std(tmp_path: Path) -> None:
    candidate = "pairwise_ranker_combined"
    seed_specs = [
        (42, 0.50, 0.30, 4.5),
        (43, 0.51, 0.30, 1.5),
        (44, 0.29, 0.10, 8.0),
    ]
    result_paths = []
    for seed, top1, spearman, gap in seed_specs:
        result_paths.append(
            _write_result(
                tmp_path / f"run_seed{seed}" / "learned_utility_results.json",
                methods={
                    "metadata_routing": _metric(
                        method_role="baseline",
                        adoption_eligible=1,
                        diagnostic_only=0,
                        top1=0.30,
                        spearman=0.0,
                        gap_pct=10.0,
                    ),
                    candidate: _metric(
                        method_role="learned",
                        adoption_eligible=1,
                        diagnostic_only=0,
                        top1=top1,
                        spearman=spearman,
                        gap_pct=gap,
                    ),
                },
                domain_rows=_domain_rows(
                    query_domain=40,
                    baseline_top1=0.30,
                    baseline_spearman=0.0,
                    baseline_gap_pct=10.0,
                    candidate_method=candidate,
                    candidate_top1=top1,
                    candidate_spearman=spearman,
                    candidate_gap_pct=gap,
                ),
            )
        )

    sign_ci = {row["method"]: row for row in _decision_rows(result_paths, decision_policy_version=SIGN_CI_POLICY)}
    learned = sign_ci[candidate]
    assert learned["positive_observation_threshold"] == 2
    assert learned["top1_positive_count"] == 2
    assert learned["oracle_gap_pct_reduction_vs_metadata_std"] > 2.0
    assert learned["oracle_gap_pct_reduction_vs_metadata_std"] < 3.0
    assert learned["instability_breach"] == 0
    assert learned["tier"] == "strong_pass"
    assert learned["decision"] == "selected"

    legacy = {row["method"]: row for row in _decision_rows(result_paths, decision_policy_version=LEGACY_STD_POLICY)}
    assert legacy[candidate]["tier"] == "fail"
    assert legacy[candidate]["instability_breach"] == 1


def test_positive_threshold_four_or_more_runs_uses_ceiling_fraction() -> None:
    assert effective_positive_threshold(3, min_improving_runs=2, min_positive_fraction=0.67) == 2
    assert effective_positive_threshold(4, min_improving_runs=2, min_positive_fraction=0.67) == 3


def test_direct_pairprob_diagnostic_alias_is_excluded_from_sign_ci_selection(tmp_path: Path) -> None:
    direct_diag = _metric(
        method_role="diagnostic",
        adoption_eligible=0,
        diagnostic_only=1,
        top1=0.7,
        spearman=0.75,
        gap_pct=2.0,
    )
    direct_diag.update({"excluded_from_sign_ci_selection": 1, "sign_ci_candidate": 0})
    direct_adoption = _metric(
        method_role="learned",
        adoption_eligible=1,
        diagnostic_only=0,
        top1=0.7,
        spearman=0.75,
        gap_pct=2.0,
    )
    direct_adoption.update(
        {
            "excluded_from_sign_ci_selection": 0,
            "sign_ci_candidate": 1,
            "direct_adoption_is_alias_of": "pairwise_direct_pairprob_tournament_v1",
            "direct_adoption_same_route_as_direct": 1,
            "direct_adoption_audit_failure_reason": "none",
            "source_only_audit_pass": 1,
            "target_leakage_audit_pass": 1,
            "adoption_feature_family": "pairprob_latent_only_v1",
        }
    )
    result = _write_result(
        tmp_path / "run_seed42" / "learned_utility_results.json",
        methods={
            "metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.2,
                spearman=0.0,
                gap_pct=12.0,
            ),
            "pairwise_direct_pairprob_tournament_v1": direct_diag,
            "pairwise_direct_pairprob_adoption_v1": direct_adoption,
            "pairwise_group_robust_pairprob_tournament_v1": _metric(
                method_role="learned",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.68,
                spearman=0.72,
                gap_pct=2.3,
            ),
        },
        domain_rows=_domain_rows(
            query_domain=0,
            baseline_top1=0.2,
            baseline_spearman=0.0,
            baseline_gap_pct=12.0,
            candidate_method="pairwise_direct_pairprob_adoption_v1",
            candidate_top1=0.7,
            candidate_spearman=0.75,
            candidate_gap_pct=2.0,
        ),
    )

    by_method = {
        row["method"]: row
        for row in _decision_rows(
            [result],
            decision_policy_version=SIGN_CI_POLICY,
            allow_missing_domain_breakdown_as_diagnostic=True,
        )
    }

    assert by_method["pairwise_direct_pairprob_tournament_v1"]["selection_eligible"] == 0
    assert (
        by_method["pairwise_direct_pairprob_tournament_v1"]["selection_ineligible_reason"]
        == "excluded_from_sign_ci_selection"
    )
    assert by_method["pairwise_direct_pairprob_adoption_v1"]["selection_eligible"] == 1


def test_sign_ci_policy_vetoes_paired_domain_regression(tmp_path: Path) -> None:
    candidate = "pairwise_ranker_combined"
    result = _write_result(
        tmp_path / "run_seed42" / "learned_utility_results.json",
        methods={
            "metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.30,
                spearman=0.0,
                gap_pct=10.0,
            ),
            candidate: _metric(
                method_role="learned",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.60,
                spearman=0.40,
                gap_pct=3.0,
            ),
        },
        domain_rows=[
            *_domain_rows(
                query_domain=40,
                baseline_top1=0.30,
                baseline_spearman=0.0,
                baseline_gap_pct=10.0,
                candidate_method=candidate,
                candidate_top1=0.60,
                candidate_spearman=0.40,
                candidate_gap_pct=12.5,
            ),
            *_domain_rows(
                query_domain=100,
                baseline_top1=0.30,
                baseline_spearman=0.0,
                baseline_gap_pct=10.0,
                candidate_method=candidate,
                candidate_top1=0.90,
                candidate_spearman=0.80,
                candidate_gap_pct=1.0,
            ),
        ],
    )

    by_method = {row["method"]: row for row in _decision_rows([result], decision_policy_version=SIGN_CI_POLICY)}
    learned = by_method[candidate]
    assert learned["catastrophic_regression_breach"] == 1
    assert learned["catastrophic_regression_metric"] == "gap_pct"
    assert learned["catastrophic_regression_query_domain"] == "40"
    assert learned["tier"] == "fail"
    assert learned["decision"] == "not_selected"


def test_missing_domain_breakdown_override_marks_needs_evidence(tmp_path: Path) -> None:
    candidate = "pairwise_ranker_combined"
    result = _write_result(
        tmp_path / "run_seed42" / "learned_utility_results.json",
        methods={
            "metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.30,
                spearman=0.0,
                gap_pct=10.0,
            ),
            candidate: _metric(
                method_role="learned",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.60,
                spearman=0.40,
                gap_pct=3.0,
            ),
        },
    )

    closed = {row["method"]: row for row in _decision_rows([result], decision_policy_version=SIGN_CI_POLICY)}
    assert closed[candidate]["regression_check_missing"] == 1
    assert closed[candidate]["selection_eligible"] == 1
    assert closed[candidate]["tier"] == "fail"

    diagnostic = {
        row["method"]: row
        for row in _decision_rows(
            [result],
            decision_policy_version=SIGN_CI_POLICY,
            allow_missing_domain_breakdown_as_diagnostic=True,
        )
    }
    assert diagnostic[candidate]["regression_check_missing"] == 1
    assert diagnostic[candidate]["selection_eligible"] == 0
    assert diagnostic[candidate]["decision"] == "NEEDS_EVIDENCE"
    assert diagnostic[candidate]["tier"] == "needs_evidence"


def test_delta_gate_guard_failure_makes_decision_table_reference_only(tmp_path: Path) -> None:
    candidate = "pairwise_tournament_delta_gated_sparse_mix_v1"
    result_paths = []
    seed_specs = [
        (42, "selected", ""),
        (43, "failed_guards_noop", "activation_rate_too_high"),
    ]
    for seed, status, reason in seed_specs:
        result_paths.append(
            _write_result(
                tmp_path / f"run_seed{seed}" / "learned_utility_results.json",
                methods={
                    "metadata_routing": _metric(
                        method_role="baseline",
                        adoption_eligible=1,
                        diagnostic_only=0,
                        top1=0.30,
                        spearman=0.0,
                        gap_pct=10.0,
                    ),
                    candidate: {
                        **_metric(
                            method_role="learned",
                            adoption_eligible=1,
                            diagnostic_only=0,
                            top1=0.60,
                            spearman=0.50,
                            gap_pct=3.0,
                        ),
                        "delta_gate_selection_status": status,
                        "delta_gate_diagnostic_only_reason": reason,
                    },
                },
            )
        )

    by_method = {row["method"]: row for row in _decision_rows(result_paths)}
    delta = by_method[candidate]

    assert delta["method_role"] == "diagnostic"
    assert delta["adoption_eligible"] == 0
    assert delta["diagnostic_only"] == 1
    assert delta["selection_eligible"] == 0
    assert delta["tier"] == "reference_only"
    assert delta["decision"] == "not_selected"
    assert delta["delta_gate_source_inner_guard_pass"] == 0
    assert delta["selection_ineligible_reason"] == (
        "delta_gate_source_inner_guard_failed:activation_rate_too_high"
    )


def test_v2_protocol_validation_hard_fails_for_mixed_manifest(tmp_path: Path) -> None:
    v2_result = _write_result(
        tmp_path / "run_seed42" / "learned_utility_results.json",
        methods={
            "metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.2,
                spearman=0.1,
                gap_pct=50.0,
            )
        },
    )
    old_result = _write_result(
        tmp_path / "run_seed43" / "learned_utility_results.json",
        protocol_version="pre_v2",
        methods={
            "metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                protocol_version="pre_v2",
                top1=0.2,
                spearman=0.1,
                gap_pct=50.0,
            )
        },
    )

    with pytest.raises(RuntimeError, match="requires learned utility LOQDO v2 artifacts"):
        _read_rows([v2_result, old_result], uplift_reference_method="metadata_routing")


def test_missing_protocol_or_uplift_reference_hard_fails(tmp_path: Path) -> None:
    missing_protocol = tmp_path / "run_seed42" / "learned_utility_results.json"
    missing_protocol.parent.mkdir(parents=True, exist_ok=True)
    missing_protocol.write_text(
        json.dumps(
            {
                "metrics_by_method": {
                    "metadata_routing": _metric(
                        method_role="baseline",
                        adoption_eligible=1,
                        diagnostic_only=0,
                        top1=0.2,
                        spearman=0.1,
                        gap_pct=50.0,
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requires learned utility LOQDO v2 artifacts"):
        _read_rows([missing_protocol], uplift_reference_method="metadata_routing")

    missing_baseline = _write_result(
        tmp_path / "run_seed43" / "learned_utility_results.json",
        methods={
            "linear_regressor": _metric(
                method_role="learned",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.5,
                spearman=0.4,
                gap_pct=35.0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="uplift_reference_method='metadata_routing' is missing"):
        _read_rows([missing_baseline], uplift_reference_method="metadata_routing")


def test_missing_method_policy_fields_hard_fails(tmp_path: Path) -> None:
    result = _write_result(
        tmp_path / "run_seed42" / "learned_utility_results.json",
        methods={
            "metadata_routing": {
                "protocol_version": PROTOCOL_VERSION,
                "top1_oracle_hit": 0.2,
                "spearman": 0.1,
                "mean_oracle_gap_pct": 50.0,
            }
        },
    )

    with pytest.raises(RuntimeError, match="missing required v2 method policy fields"):
        _read_rows([result], uplift_reference_method="metadata_routing")
