"""Versioned terminal-table schemas for the multi-challenger bundle."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from .artifact_io import object_payload


# Terminal rows cross a canonical, recursively key-sorted JSON checkpoint before
# CSV finalization.  Freeze that actual producer order explicitly so persistence
# and reconstructive replay never infer different schemas from mapping insertion
# order.  These fields also preserve already-sealed workstation CSV bytes.
TERMINAL_CASE_CONFUSION_FIELDS = (
    "action_id",
    "case_id",
    "fn",
    "fp",
    "method_id",
    "row_hash",
    "target_center",
    "tn",
    "tp",
)
TERMINAL_CENTER_METRIC_FIELDS = (
    "bacc",
    "method_id",
    "n_negative",
    "n_positive",
    "row_hash",
    "target_center",
    "tn",
    "tp",
)
TERMINAL_CONTRAST_FIELDS = (
    "baseline_id",
    "center_estimates",
    "ci_high",
    "ci_low",
    "contrast_id",
    "estimate",
    "method_id",
    "one_sided_95_lcb",
    "outer_df",
    "outer_n",
    "outer_sd",
    "outer_se",
    "replicates",
    "row_hash",
    "row_role",
    "seed",
    "target_center",
)
ROUTER_IDENTIFICATION_METRIC_FIELDS = (
    "anchor_selection_rate",
    "fold_stability",
    "normalized_oracle_gap",
    "oracle_static_action_id",
    "positive_margin_switch_rate",
    "recovered_B_to_case_oracle_headroom",
    "row_hash",
    "spearman",
    "target_center",
    "top1_oracle_agreement",
    "top3_menu_oracle_coverage",
)
PERMUTATION_METRIC_FIELDS = (
    "P_multi_bacc",
    "R_multi_bacc",
    "R_multi_minus_P_multi",
    "action_agreement",
    "row_hash",
    "target_center",
)
MENU_ORACLE_METRIC_FIELDS = (
    "O_binary_action_set",
    "binary_oracle_bacc",
    "case_oracle_bacc",
    "menu_oracle_bacc",
    "menu_oracle_equals_full_case_oracle_rate",
    "row_hash",
    "static_oracle_bacc",
    "target_center",
)
TERMINAL_TABLE_FIELDS = {
    "terminal_case_confusions": TERMINAL_CASE_CONFUSION_FIELDS,
    "terminal_center_metrics": TERMINAL_CENTER_METRIC_FIELDS,
    "terminal_contrasts": TERMINAL_CONTRAST_FIELDS,
    "router_identification_metrics": ROUTER_IDENTIFICATION_METRIC_FIELDS,
    "permutation_metrics": PERMUTATION_METRIC_FIELDS,
    "menu_oracle_metrics": MENU_ORACLE_METRIC_FIELDS,
}
TERMINAL_TABLE_MEMBERS = {
    "terminal_case_confusions": "tables/terminal_case_confusions.csv",
    "terminal_center_metrics": "tables/terminal_center_metrics.csv",
    "terminal_contrasts": "tables/terminal_contrasts.csv",
    "router_identification_metrics": "tables/router_identification_metrics.csv",
    "permutation_metrics": "tables/permutation_metrics.csv",
    "menu_oracle_metrics": "tables/menu_oracle_metrics.csv",
}


def canonical_terminal_rows(
    table_name: str, rows: Sequence[object]
) -> tuple[dict[str, object], ...]:
    """Return terminal rows in the producer's versioned CSV field order.

    JSON object ordering is not a table-schema contract: an atomic checkpoint
    round-trip may reorder mapping keys.  Terminal CSV production and replay
    therefore both pass through this exact field contract.  Missing and extra
    keys remain protocol errors; this helper only restores canonical order.
    """

    fields = TERMINAL_TABLE_FIELDS.get(table_name)
    if fields is None:
        raise ProtocolError(
            f"Unknown multi-challenger terminal table: {table_name}."
        )
    canonical = []
    for row in rows:
        payload = object_payload(row)
        if set(payload) != set(fields):
            raise ProtocolError(
                f"Multi-challenger terminal table schema drifted: {table_name}."
            )
        canonical.append({field: payload[field] for field in fields})
    if not canonical:
        raise ProtocolError(
            f"Multi-challenger terminal table is empty: {table_name}."
        )
    return tuple(canonical)


__all__ = (
    "MENU_ORACLE_METRIC_FIELDS",
    "PERMUTATION_METRIC_FIELDS",
    "ROUTER_IDENTIFICATION_METRIC_FIELDS",
    "TERMINAL_CASE_CONFUSION_FIELDS",
    "TERMINAL_CENTER_METRIC_FIELDS",
    "TERMINAL_CONTRAST_FIELDS",
    "TERMINAL_TABLE_FIELDS",
    "TERMINAL_TABLE_MEMBERS",
    "canonical_terminal_rows",
)
