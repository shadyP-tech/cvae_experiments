"""Exact physical-fingerprint rectangle and posterior lineage validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CANONICAL_PHYSICAL_ROW_ORDER,
    CENTERS,
    FINGERPRINT_FEATURE_COUNT,
    PRIMARY_FINGERPRINT_CONTROL_ID,
)
from .hashing import canonical_hash, require_sha256
from .physical_fingerprint import fingerprint_feature_names
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list


def validate_fingerprint_topology(
    *,
    physical: Row,
    rows: Sequence[Row],
    topology: PlanPosteriorTopology,
) -> Mapping[tuple[str, str], Row]:
    controls = (
        PRIMARY_FINGERPRINT_CONTROL_ID,
        BLOCKED_FINGERPRINT_CONTROL_ID,
    )
    indexed = index_rows(rows, ("center", "control_id"), "physical fingerprints")
    expected = {(center, control) for control in controls for center in CENTERS}
    expected_order = [
        (center, control) for control in controls for center in CENTERS
    ]
    if set(indexed) != expected or [
        (str(row.get("center")), str(row.get("control_id"))) for row in rows
    ] != expected_order:
        fail("physical fingerprint rectangle/order")

    center_surface_hashes: dict[str, str] = {}
    for center in CENTERS:
        cases = topology.cases_by_center[center]
        sample_ids = tuple(
            sample
            for case in cases
            for sample in string_list(
                topology.plans[(center, case)], "evaluation_sample_ids"
            )
        )
        case_ids = tuple(
            case
            for case in cases
            for _sample in string_list(
                topology.plans[(center, case)], "evaluation_sample_ids"
            )
        )
        source_hashes: set[str] = set()
        for control in controls:
            row = indexed[(center, control)]
            names = string_list(row, "feature_names")
            source_hash = require_sha256(
                row.get("source_surface_hash"), "fingerprint source surface hash"
            )
            feature_hash = require_sha256(
                row.get("feature_array_sha256"), "fingerprint feature array hash"
            )
            payload = {
                "schema_version": "fixed_bank_cbpupr_fingerprint_v1",
                "center": center,
                "sample_ids": list(sample_ids),
                "case_ids": list(case_ids),
                "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
                "feature_names": list(names),
                "feature_array_sha256": feature_hash,
                "source_surface_hash": source_hash,
                "control_id": control,
                "labels_used": False,
            }
            if (
                row.get("schema_version")
                != "fixed_bank_cbpupr_fingerprint_summary_v1"
                or row.get("sample_count") != len(sample_ids)
                or row.get("case_count") != len(cases)
                or row.get("row_order") != CANONICAL_PHYSICAL_ROW_ORDER
                or names != fingerprint_feature_names(center)
                or len(names) != FINGERPRINT_FEATURE_COUNT
                or row.get("fingerprint_hash") != canonical_hash(payload)
                or row.get("raw_feature_rows_persisted") is not False
                or row.get("labels_used") is not False
            ):
                fail("physical fingerprint summary/hash")
            source_hashes.add(source_hash)
        if len(source_hashes) != 1:
            fail("primary/blocked fingerprint source lineage")
        center_surface_hashes[center] = source_hashes.pop()

    expected_surface_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_physical_surface_v1",
            "probability_store_hash": physical.get("probability_store_hash"),
            "center_surface_hashes": {
                center: center_surface_hashes[center] for center in CENTERS
            },
            "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
            "strict_canonical_topology": True,
            "labels_used": False,
        }
    )
    if physical.get("surface_hash") != expected_surface_hash:
        fail("fingerprint/global physical surface lineage")

    for key, model in topology.models.items():
        center, _case, control = key
        expected_hash = indexed[(center, control)].get("fingerprint_hash")
        if (
            model.get("fingerprint_hash") != expected_hash
            or topology.posteriors[key].get("fingerprint_hash") != expected_hash
        ):
            fail("posterior/fingerprint lineage")
    return indexed


__all__ = ("validate_fingerprint_topology",)
