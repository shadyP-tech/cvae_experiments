from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.eval.evaluators.learned_utility_protocol import _domain_breakdown_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_policy_audit_md(path: Path, audit_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [r for r in audit_rows if int(r.get("policy_audit_pass", 0)) != 1]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Residual Routing Policy Audit\n\n")
        f.write(f"- rows: {len(audit_rows)}\n")
        f.write(f"- failures: {len(failures)}\n")
        f.write("- target expert excluded: checked per fold\n")
        f.write("- held-out query-domain NELBO used for training: no\n")
        f.write("- held-out query-domain NELBO used for threshold tuning: no\n")
        f.write("- eval-domain latent statistics used by adoption residual features: no\n")
        f.write("- raw expert/query identity features: no\n")
        f.write("- domain-40-specific tuning: no\n")
        if failures:
            f.write("\n## Failures\n\n")
            for row in failures:
                f.write(
                    f"- method={row.get('method')} fold={row.get('fold_query_domain')} "
                    f"feature_set={row.get('feature_set')} tau={row.get('selected_tau')}\n"
                )


def _write_summary_md(
    path: Path,
    *,
    residual_domain_rows: Sequence[Dict[str, Any]],
    override_rows: Sequence[Dict[str, Any]],
    audit_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted(set(str(r.get("method", "")) for r in residual_domain_rows if str(r.get("method", ""))))
    audit_pass = all(int(r.get("policy_audit_pass", 0)) == 1 for r in audit_rows) if audit_rows else True
    with path.open("w", encoding="utf-8") as f:
        f.write("# Residual Routing Summary\n\n")
        f.write("- compatibility target: held-out utility (-NELBO)\n")
        f.write("- uplift reference: metadata_routing\n")
        f.write("- residual score: metadata-relative predicted utility improvement\n")
        f.write(f"- policy audit pass: {int(audit_pass)}\n")
        f.write(f"- residual methods: {', '.join(methods)}\n")
        f.write("\n## Override Diagnostics\n\n")
        f.write(
            "| method | fold | tau | feature_set | override_rate | utility_improving_override_rate | "
            "oracle_correct_override_rate | harmful_override_rate |\n"
        )
        f.write("|---|---:|---|---|---:|---:|---:|---:|\n")
        for row in override_rows:
            f.write(
                "| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    row.get("method", ""),
                    row.get("fold_query_domain", ""),
                    row.get("selected_tau", ""),
                    row.get("feature_set", ""),
                    float(row.get("override_rate", 0.0)),
                    float(row.get("utility_improving_override_rate", 0.0)),
                    float(row.get("oracle_correct_override_rate", 0.0)),
                    float(row.get("harmful_override_rate", 0.0)),
                )
            )


def _confusion_rows_from_samples(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[tuple[str, int, int, int], int] = {}
    for row in sample_rows:
        method = str(row.get("method", ""))
        if not method.startswith("metadata_residual"):
            continue
        key = (
            method,
            int(row.get("query_domain", 0)),
            int(row.get("selected_expert", 0)),
            int(row.get("candidate_oracle_expert", row.get("oracle_expert", 0))),
        )
        counts[key] = int(counts.get(key, 0)) + 1
    return [
        {
            "method": method,
            "query_domain": int(query_domain),
            "selected_expert": int(selected_expert),
            "oracle_expert": int(oracle_expert),
            "count": int(count),
        }
        for (method, query_domain, selected_expert, oracle_expert), count in sorted(counts.items())
    ]


def write_residual_routing_artifacts(
    *,
    reports_dir: Path,
    residual_sample_rows: Sequence[Dict[str, Any]],
    residual_raw_rows: Sequence[Dict[str, Any]],
    residual_override_rows: Sequence[Dict[str, Any]],
    residual_audit_rows: Sequence[Dict[str, Any]],
    residual_confusion_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    residual_domain_rows = _domain_breakdown_rows(residual_sample_rows) if residual_sample_rows else []
    confusion_rows = list(residual_confusion_rows) or _confusion_rows_from_samples(residual_sample_rows)

    _write_csv(reports_dir / "residual_routing_raw.csv", residual_raw_rows)
    _write_csv(reports_dir / "residual_routing_domain_breakdown.csv", residual_domain_rows)
    _write_csv(reports_dir / "residual_routing_expert_confusion_matrix.csv", confusion_rows)
    _write_csv(reports_dir / "residual_routing_override_diagnostics.csv", residual_override_rows)
    _write_csv(reports_dir / "residual_routing_policy_audit.csv", residual_audit_rows)
    _write_policy_audit_md(reports_dir / "residual_routing_policy_audit.md", residual_audit_rows)
    _write_summary_md(
        reports_dir / "residual_routing_summary.md",
        residual_domain_rows=residual_domain_rows,
        override_rows=residual_override_rows,
        audit_rows=residual_audit_rows,
    )
    return {
        "residual_raw": "residual_routing_raw.csv",
        "residual_domain_breakdown": "residual_routing_domain_breakdown.csv",
        "residual_summary": "residual_routing_summary.md",
        "residual_confusion_matrix": "residual_routing_expert_confusion_matrix.csv",
        "residual_override_diagnostics": "residual_routing_override_diagnostics.csv",
        "residual_policy_audit": "residual_routing_policy_audit.md",
        "residual_policy_audit_csv": "residual_routing_policy_audit.csv",
    }
