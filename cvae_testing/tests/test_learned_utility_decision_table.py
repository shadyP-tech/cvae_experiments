from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_compatibility_decision_table import _aggregate, _read_rows


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


def _write_result(path: Path, *, protocol_version: str = PROTOCOL_VERSION, methods: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": protocol_version,
        "protocol_contract": {"protocol_version": protocol_version},
        "metrics_by_method": methods,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _decision_rows(result_paths: list[Path]) -> list[dict]:
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
    )
    return out_rows


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
