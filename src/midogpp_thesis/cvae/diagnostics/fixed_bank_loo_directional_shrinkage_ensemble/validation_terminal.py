"""Exact persisted terminal-table and descriptive-rubric validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .constants import ARM_IDS, CENTERS, METHOD_IDS
from .ensemble import DESCRIPTIVE_METHOD_IDS
from .persistence import object_payload, read_rows


TERMINAL_TABLE_MEMBERS = {
    "case_confusions": "tables/terminal_case_confusions.csv",
    "method_metrics": "tables/terminal_method_metrics.csv",
    "center_metrics": "tables/terminal_center_metrics.csv",
    "equal_center_contrasts": "tables/terminal_contrasts.csv",
    "delete_one_center": "tables/whole_pipeline_delete_one_center.csv",
    "leave_one_arm": "tables/leave_one_arm_ablations.csv",
    "null_statistics": "tables/null_statistics.csv",
}
_REPORTED_METHOD_IDS = (*METHOD_IDS, *DESCRIPTIVE_METHOD_IDS)
_EXPECTED_TERMINAL_ROW_COUNTS = {
    "case_confusions": 218 * len(_REPORTED_METHOD_IDS),
    "method_metrics": len(_REPORTED_METHOD_IDS),
    "center_metrics": len(CENTERS) * len(_REPORTED_METHOD_IDS),
    "equal_center_contrasts": 3,
    "delete_one_center": len(CENTERS) * 2,
    "leave_one_arm": len(ARM_IDS),
    "null_statistics": 1,
}


def validate_terminal_products(
    root: Path,
    *,
    reconstructed: Mapping[str, object],
    expected_lineage_bindings: Mapping[str, object],
) -> Mapping[str, object]:
    descriptive = reconstructed.get("descriptive_inference", {})
    if not isinstance(descriptive, Mapping):
        descriptive = {}
    expected_tables: dict[str, tuple[dict[str, object], ...]] = {}
    for key, member in TERMINAL_TABLE_MEMBERS.items():
        values = reconstructed.get(key, descriptive.get(key))
        if not isinstance(values, (list, tuple)) or not values:
            raise ProtocolError(
                f"Directional-shrinkage reconstructed terminal table absent: {key}."
            )
        expected = tuple(object_payload(value) for value in values)
        expected_tables[key] = expected
        if len(expected) != _EXPECTED_TERMINAL_ROW_COUNTS[key]:
            raise ProtocolError(
                f"Directional-shrinkage terminal topology drifted: {key}."
            )
        if read_rows(root / member) != expected:
            raise ProtocolError(
                f"Directional-shrinkage terminal table is not exact: {key}."
            )
    _validate_terminal_topology(expected_tables)
    seal = reconstructed.get("terminal_seal")
    if not isinstance(seal, Mapping) or read_json(
        root / "manifests/terminal_evaluation_seal.json"
    ) != dict(seal):
        raise ProtocolError("Directional-shrinkage terminal seal is not reconstructive.")
    rubric = seal.get("descriptive_success_rubric")
    if not isinstance(rubric, Mapping):
        raise ProtocolError("Directional-shrinkage descriptive rubric is absent.")
    required = (
        "full_DCSE_LOO_minus_B_strictly_positive",
        "full_DCSE_LOO_minus_U_strictly_positive",
        "both_primary_B_and_U_contrasts_positive_in_all_nine_whole_pipeline_center_deletions",
        "at_least_eight_of_nine_center_DCSE_LOO_minus_B_deltas_nonnegative",
        "every_leave_one_arm_DCSE_LOO_minus_B_contrast_strictly_positive",
    )
    if any(key not in rubric or not isinstance(rubric[key], bool) for key in required):
        raise ProtocolError("Directional-shrinkage descriptive rubric drifted.")
    if seal.get("terminal_decision") != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE":
        raise ProtocolError("Directional-shrinkage terminal claim boundary drifted.")
    if (
        seal.get("bindings") != dict(expected_lineage_bindings)
        or seal.get("bound_product_counts")
        != {
            "loo_plans": 218,
            "donor_priors": 9 * 16,
            "arm_decisions": 218 * 18,
            "preterminal_predictions": 9_928 * 6,
            "descriptive_predictions": 9_928 * 5,
        }
        or seal.get("bound_preterminal_method_order") != list(METHOD_IDS[:6])
        or seal.get("bound_descriptive_method_order")
        != list(DESCRIPTIVE_METHOD_IDS)
        or seal.get("canonical_method_ids") != list(METHOD_IDS)
        or seal.get("reported_method_ids") != list(_REPORTED_METHOD_IDS)
        or seal.get("descriptive_control_method_ids")
        != list(DESCRIPTIVE_METHOD_IDS)
    ):
        raise ProtocolError("Directional-shrinkage terminal lineage topology drifted.")
    return {
        "terminal_seal_hash": seal.get("seal_hash", seal.get("terminal_seal_hash")),
        "descriptive_success_rubric": dict(rubric),
        "terminal_tables_exact": True,
    }


def _validate_terminal_topology(
    tables: Mapping[str, tuple[dict[str, object], ...]],
) -> None:
    case_methods = tuple(
        dict.fromkeys(str(row.get("method_id")) for row in tables["case_confusions"])
    )
    center_methods = tuple(
        dict.fromkeys(str(row.get("method_id")) for row in tables["center_metrics"])
    )
    metric_methods = tuple(
        str(row.get("method_id")) for row in tables["method_metrics"]
    )
    if (
        case_methods != _REPORTED_METHOD_IDS
        or center_methods != _REPORTED_METHOD_IDS
        or metric_methods != _REPORTED_METHOD_IDS
        or Counter(
            str(row.get("method_id")) for row in tables["case_confusions"]
        )
        != Counter({method: 218 for method in _REPORTED_METHOD_IDS})
        or Counter(
            str(row.get("method_id")) for row in tables["center_metrics"]
        )
        != Counter({method: len(CENTERS) for method in _REPORTED_METHOD_IDS})
    ):
        raise ProtocolError("Directional-shrinkage terminal method topology drifted.")
    contrasts = tuple(
        (str(row.get("method_id")), str(row.get("reference_id")))
        for row in tables["equal_center_contrasts"]
    )
    deleted = tuple(
        (str(row.get("deleted_center")), str(row.get("reference_id")))
        for row in tables["delete_one_center"]
    )
    leave_arms = tuple(
        str(row.get("deleted_arm_id")) for row in tables["leave_one_arm"]
    )
    if (
        contrasts
        != (
            ("DCSE_LOO", "B"),
            ("DCSE_LOO", "U"),
            ("DCSE_LOO", "G_directional_matched"),
        )
        or deleted
        != tuple(
            (center, reference)
            for center in CENTERS
            for reference in ("B", "U")
        )
        or leave_arms != ARM_IDS
    ):
        raise ProtocolError(
            "Directional-shrinkage terminal contrast/ablation topology drifted."
        )


__all__ = ("TERMINAL_TABLE_MEMBERS", "validate_terminal_products")
