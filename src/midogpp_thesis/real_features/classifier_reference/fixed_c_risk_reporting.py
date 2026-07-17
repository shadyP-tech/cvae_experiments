"""Summary construction and Markdown reporting for the fixed-C diagnostic."""

from __future__ import annotations

from typing import Mapping, Sequence

from .protocol import ProtocolError
from .schemas.fixed_c_risk_diagnostic import PRIMARY_CONTRAST, RISK_POLICY_IDS


def build_diagnostic_summary(
    results: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    *,
    protocol_hash: str,
    bundle_hash: str,
    heldout_count: int,
) -> dict[str, object]:
    """Construct the non-adoptive execution summary from persisted table rows."""

    arm_summaries: list[dict[str, object]] = []
    for policy in RISK_POLICY_IDS:
        rows = [row for row in results if row.get("risk_policy_id") == policy]
        if not rows:
            raise ProtocolError(
                f"Fixed-C risk summary has no rows for policy {policy!r}."
            )
        arm_summaries.append(
            {
                "risk_policy_id": policy,
                "n_heldout_centers": len(rows),
                "mean_bacc": sum(float(row["heldout_bacc"]) for row in rows)
                / float(len(rows)),
                "mean_macro_f1": sum(
                    float(row["heldout_macro_f1"]) for row in rows
                )
                / float(len(rows)),
            }
        )
    if not paired:
        raise ProtocolError("Fixed-C risk summary requires paired comparison rows.")
    return {
        "schema_version": "midogpp_fixed_c_risk_summary_v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "protocol_hash": str(protocol_hash),
        "bundle_hash": str(bundle_hash),
        "n_heldout_centers": int(heldout_count),
        "n_fits": len(results),
        "arm_summaries": arm_summaries,
        "primary_contrast": PRIMARY_CONTRAST,
        "mean_primary_delta_bacc": sum(
            float(row["delta_bacc"]) for row in paired
        )
        / float(len(paired)),
        "diagnostic_only": True,
        "adoption_enabled": False,
        "claim_scope": "real_feature_transfer_only",
    }


def render_diagnostic_report(summary: Mapping[str, object]) -> str:
    """Render the deterministic human-readable view of the JSON summary."""

    arm_rows = summary.get("arm_summaries")
    if not isinstance(arm_rows, list):
        raise ProtocolError("Fixed-C risk diagnostic summary lacks arm_summaries.")
    lines = [
        "# Fixed-C Risk-Weighting Diagnostic",
        "",
        "Status: `DIAGNOSTIC_ONLY`; adoption is disabled.",
        "",
        "| arm | mean BACC | mean macro-F1 |",
        "| --- | ---: | ---: |",
    ]
    for row in arm_rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Malformed fixed-C risk arm summary.")
        lines.append(
            f"| {row['risk_policy_id']} | {float(row['mean_bacc']):.12f} | "
            f"{float(row['mean_macro_f1']):.12f} |"
        )
    lines.extend(
        [
            "",
            f"Primary contrast: `{PRIMARY_CONTRAST}`.",
            "",
            "Target-center labels were used for final scoring only. This real-feature "
            "diagnostic cannot select a classifier or establish CVAE, prior, routing, "
            "generation, or synthetic-utility evidence.",
            "",
            f"Protocol hash: `{summary['protocol_hash']}`.",
            "",
            f"Bundle hash: `{summary['bundle_hash']}`.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_diagnostic_summary", "render_diagnostic_report"]
