from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_support_response_decision_table import _aggregate, _read_rows


PROTOCOL_VERSION = "support_response_candidate_specific_v1"


def _metric(
    *,
    method_role: str,
    adoption_eligible: int,
    diagnostic_only: int,
    top1: float,
    spearman: float,
    gap: float,
    routing_uses_eval_nelbo: int = 0,
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "method_role": method_role,
        "adoption_eligible": float(adoption_eligible),
        "diagnostic_only": float(diagnostic_only),
        "routing_uses_eval_nelbo": float(routing_uses_eval_nelbo),
        "routing_uses_eval_domain_statistics": 0.0,
        "top1_oracle_hit": float(top1),
        "spearman": float(spearman),
        "mean_oracle_gap_pct": float(gap),
        "n_query_domains_macro": 2.0,
    }


def _domain_rows(methods: dict[str, tuple[float, float, float]]) -> list[dict]:
    rows: list[dict] = []
    for query_domain in [0, 1]:
        for method, (top1, spearman, gap) in methods.items():
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "method": method,
                    "query_domain": query_domain,
                    "top1_oracle_hit": top1,
                    "spearman": spearman,
                    "mean_oracle_gap_pct": gap,
                }
            )
    return rows


def _write_result(path: Path, methods: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "metrics_by_method": methods,
        "artifacts": {"domain_breakdown": "support_response_domain_breakdown.csv"},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (path.parent / "support_response_domain_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "protocol_version",
            "method",
            "query_domain",
            "top1_oracle_hit",
            "spearman",
            "mean_oracle_gap_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            _domain_rows(
                {
                    method: (
                        float(metric["top1_oracle_hit"]),
                        float(metric["spearman"]),
                        float(metric["mean_oracle_gap_pct"]),
                    )
                    for method, metric in methods.items()
                }
            )
        )
    return path


def test_support_response_decision_table_never_selects_controls_or_oracle(tmp_path: Path) -> None:
    result = _write_result(
        tmp_path / "run_seed11" / "support_response_results.json",
        {
            "support_metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.20,
                spearman=0.10,
                gap=30.0,
            ),
            "support_static_embedding_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.25,
                spearman=0.12,
                gap=25.0,
            ),
            "support_set_nelbo_top1": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.60,
                spearman=0.50,
                gap=10.0,
            ),
            "source_global_prior_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.22,
                spearman=0.11,
                gap=28.0,
            ),
            "support_response_pairwise_static_response_indirect": _metric(
                method_role="learned",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.62,
                spearman=0.52,
                gap=8.0,
            ),
            "support_response_pairwise_response_indirect_shuffled": _metric(
                method_role="control",
                adoption_eligible=0,
                diagnostic_only=0,
                top1=1.0,
                spearman=1.0,
                gap=0.0,
            ),
            "expert_id_only_pairwise": _metric(
                method_role="control",
                adoption_eligible=0,
                diagnostic_only=0,
                top1=1.0,
                spearman=1.0,
                gap=0.0,
            ),
            "support_candidate_oracle": _metric(
                method_role="diagnostic",
                adoption_eligible=0,
                diagnostic_only=1,
                routing_uses_eval_nelbo=1,
                top1=1.0,
                spearman=1.0,
                gap=0.0,
            ),
        },
    )

    rows = _read_rows([result])
    decisions, summary = _aggregate(rows)
    by_method = {row["method"]: row for row in decisions}

    learned = by_method["support_response_pairwise_static_response_indirect"]
    assert learned["tier"] == "strong_pass"
    assert learned["decision"] == "selected"
    assert learned["selection_eligible"] == 1
    assert learned["uses_no_direct_support_utility_terms"] == 1

    shuffled = by_method["support_response_pairwise_response_indirect_shuffled"]
    assert shuffled["selection_eligible"] == 0
    assert shuffled["decision"] == "not_selected"
    assert shuffled["tier"] == "diagnostic_only"

    expert_id = by_method["expert_id_only_pairwise"]
    assert expert_id["selection_eligible"] == 0
    assert expert_id["decision"] == "not_selected"

    oracle = by_method["support_candidate_oracle"]
    assert oracle["selection_eligible"] == 0
    assert oracle["tier"] == "reference_only"
    assert oracle["decision"] == "not_selected"

    assert summary["aggregation_unit"] == "seed_x_heldout_center_x_support_seed_x_support_size"
    assert summary["selected_methods"] == ["support_response_pairwise_static_response_indirect"]


def test_support_response_decision_table_reads_nested_learned_utility_results(tmp_path: Path) -> None:
    support_result = _write_result(
        tmp_path / "nested" / "support_response_results.json",
        {
            "support_metadata_routing": _metric(
                method_role="baseline",
                adoption_eligible=1,
                diagnostic_only=0,
                top1=0.2,
                spearman=0.1,
                gap=20.0,
            )
        },
    )
    support_payload = json.loads(support_result.read_text(encoding="utf-8"))
    nested_path = tmp_path / "nested" / "learned_utility_results.json"
    nested_path.write_text(
        json.dumps({"support_response_results": support_payload}, indent=2),
        encoding="utf-8",
    )

    rows = _read_rows([nested_path])
    assert len(rows) == 1
    assert rows[0]["method"] == "support_metadata_routing"
    assert rows[0]["n_domain_level_units"] == 2
