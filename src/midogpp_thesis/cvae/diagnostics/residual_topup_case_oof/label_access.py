"""Narrow terminal capability for streaming evaluation labels after sealing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import sha256_file
from .config import EXPECTED_MANIFEST_SHA256
from .contracts import CENTERS, EXPECTED_CASE_OOF_FOLD_COUNT
from .prediction_store import (
    EXPECTED_PREDICTION_CELL_COUNT,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
)
from .seals import GLOBAL_PREDICTION_SEAL_STATUS, validate_global_prediction_seal


def open_evaluation_labels_after_global_seal(
    config: object,
    crossfit: object,
    plan: object,
    predictions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> tuple[dict[str, int], Mapping[str, object]]:
    """Open only the 26 evaluation cases after revalidating the durable seal."""

    seal = validate_global_prediction_seal(
        config,
        crossfit,
        plan,
        predictions,
        source_cache_lock_hash=source_cache_lock_hash,
        root=root,
    )
    if (
        seal.get("status") != GLOBAL_PREDICTION_SEAL_STATUS
        or seal.get("config_contract_hash") != getattr(config, "contract_hash", None)
        or seal.get("source_cache_lock_hash") != source_cache_lock_hash
        or seal.get("crossfit_fold_lock_hash") != getattr(crossfit, "lock_hash", None)
        or seal.get("router_plan_lock_hash") != getattr(plan, "lock_hash", None)
        or seal.get("validation_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or int(seal.get("fold_count", -1)) != EXPECTED_CASE_OOF_FOLD_COUNT
        or int(seal.get("cell_count", -1)) != EXPECTED_PREDICTION_CELL_COUNT
        or seal.get("prediction_array_sha256")
        != sha256_file(root / PREDICTION_ARRAY_MEMBER)
        or seal.get("prediction_index_sha256")
        != sha256_file(root / PREDICTION_INDEX_MEMBER)
        or seal.get("support_labels_opened") is not False
        or seal.get("evaluation_labels_opened") is not False
        or seal.get("selector_or_fallback_performed") is not False
    ):
        raise ProtocolError("Case-OOF label capability failed seal validation.")
    evaluation_by_center = getattr(crossfit, "evaluation_rows_by_center", {})
    support_by_center = getattr(crossfit, "fixed_support_rows_by_center", {})
    rows = tuple(
        row for center in CENTERS for row in evaluation_by_center[center]
    )
    support_ids = {
        str(row.sample_id)
        for center in CENTERS
        for row in support_by_center[center]
    }
    evaluation_ids = [str(row.sample_id) for row in rows]
    evaluation_cases = {str(row.case_id) for row in rows}
    if (
        len(evaluation_ids) != len(set(evaluation_ids))
        or support_ids.intersection(evaluation_ids)
        or len(evaluation_cases) != EXPECTED_CASE_OOF_FOLD_COUNT
    ):
        raise ProtocolError("Case-OOF support/evaluation boundary drifted.")
    labels = _stream_labels(
        Path(getattr(config, "validation_manifest_path")),
        rows,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    by_sample = {
        str(row.sample_id): label for row, label in zip(rows, labels, strict=True)
    }
    report: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_label_access_report_v1",
        "status": "OPENED_AFTER_GLOBAL_B_U_G_S_P_HXE_PREDICTION_SEAL",
        "prediction_seal_hash": seal["seal_hash"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_split": "val",
        "opened_row_count": len(rows),
        "opened_case_count": len(evaluation_cases),
        "opened_target_count": len(CENTERS),
        "label_vector_hash": stable_hash(
            {
                "row_identity_hash": _row_hash(rows),
                "labels": list(labels),
                "prediction_seal_hash": seal["seal_hash"],
            }
        ),
        "support_label_count": 0,
        "whole_label_column_loaded": False,
        "labels_used_for_route_or_action_construction": False,
        "labels_used_for_selector_or_fallback": False,
        "labels_used_for_terminal_scoring_only": True,
        "oracle_Hxe_labels_used_after_seal_only": True,
        "oracle_Hxe_may_update_policy": False,
        "fresh_evidence": False,
        "diagnostic_only": True,
    }
    report["label_access_report_hash"] = stable_hash(report)
    return by_sample, report


def _stream_labels(
    manifest_path: Path,
    rows: Sequence[object],
    *,
    expected_sha256: str,
) -> tuple[int, ...]:
    if sha256_file(manifest_path) != expected_sha256:
        raise ProtocolError("Case-OOF validation manifest hash drifted.")
    expected = {int(getattr(row, "manifest_row_index")): row for row in rows}
    if len(expected) != len(rows):
        raise ProtocolError("Case-OOF label requests duplicate manifest rows.")
    found: dict[int, int] = {}
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open case-OOF scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Case-OOF manifest lacks scoring fields.")
        for index, raw in enumerate(reader):
            row = expected.get(index)
            if row is None:
                continue
            if (
                raw["sample_id"] != str(getattr(row, "sample_id"))
                or raw["case_id"] != str(getattr(row, "case_id"))
                or raw["center"] != str(getattr(row, "center"))
                or raw["split"] != "val"
                or str(getattr(row, "split", "val")) != "val"
            ):
                raise ProtocolError("Case-OOF manifest identity/split drifted.")
            try:
                label = int(raw["label"])
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Case-OOF manifest label is invalid.") from exc
            if label not in (0, 1):
                raise ProtocolError("Case-OOF labels must be binary.")
            found[index] = label
    if set(found) != set(expected):
        raise ProtocolError("Case-OOF manifest lacks requested rows.")
    return tuple(found[int(getattr(row, "manifest_row_index"))] for row in rows)


def _row_hash(rows: Sequence[object]) -> str:
    return stable_hash(
        [
            {
                "row_ordinal": int(getattr(row, "row_ordinal")),
                "manifest_row_index": int(getattr(row, "manifest_row_index")),
                "sample_id": str(getattr(row, "sample_id")),
                "case_id": str(getattr(row, "case_id")),
                "center": str(getattr(row, "center")),
                "split": str(getattr(row, "split", "val")),
                "partition_role": str(getattr(row, "partition_role")),
            }
            for row in rows
        ]
    )


__all__ = ("open_evaluation_labels_after_global_seal",)
