"""Global all-action prediction seal and narrow validation-label capability."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, read_json, sha256_file
from .contracts import (
    CENTERS,
    DEVELOPMENT_ACTION_IDS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PREDICTION_CELL_COUNT,
    GENERATION_SEEDS,
    TARGET_ACTION_IDS,
    TRAINING_SEEDS,
)


GLOBAL_PREDICTION_SEAL_MEMBER = "manifests/global_all_action_prediction_seal.json"
GLOBAL_PREDICTION_SEAL_STATUS = (
    "COMPLETE_ALL_DEVELOPMENT_AND_TARGET_ACTION_PREDICTIONS_BEFORE_ANY_LABEL_ACCESS"
)


def build_global_prediction_seal(
    config: object,
    partitions: object,
    plans: object,
    predictions: object,
    *,
    root: Path,
) -> Mapping[str, object]:
    payload = _seal_payload(config, partitions, plans, predictions, root=root)
    atomic_write_json(root / GLOBAL_PREDICTION_SEAL_MEMBER, payload)
    return payload


def validate_global_prediction_seal(
    config: object,
    partitions: object,
    plans: object,
    predictions: object,
    *,
    root: Path,
) -> Mapping[str, object]:
    observed = read_json(root / GLOBAL_PREDICTION_SEAL_MEMBER)
    expected = _seal_payload(config, partitions, plans, predictions, root=root)
    if observed != expected:
        raise ProtocolError("Residual top-up global prediction seal drifted.")
    return observed


def _seal_payload(
    config: object,
    partitions: object,
    plans: object,
    predictions: object,
    *,
    root: Path,
) -> Mapping[str, object]:
    index_rows = tuple(getattr(predictions, "index_rows", ()))
    plan_lock_hash = str(getattr(plans, "lock_hash", ""))
    partition_lock_hash = str(getattr(partitions, "lock_hash", ""))
    config_hash = str(getattr(config, "contract_hash", ""))
    if len(index_rows) != EXPECTED_PREDICTION_CELL_COUNT or not all(
        (config_hash, plan_lock_hash, partition_lock_hash)
    ):
        raise ProtocolError("Residual top-up seal inputs are incomplete.")
    expected_keys = []
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for action in DEVELOPMENT_ACTION_IDS:
                        expected_keys.append(
                            ("development", outer, query, training_seed, generation_seed, action)
                        )
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for action in TARGET_ACTION_IDS:
                    expected_keys.append(
                        ("target", target, target, training_seed, generation_seed, action)
                    )
    observed_keys = []
    cells: list[dict[str, object]] = []
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center", {})
    for ordinal, row in enumerate(index_rows):
        phase = str(row.get("phase"))
        outer = str(row.get("outer_target"))
        query = str(row.get("query_center"))
        key = (
            phase,
            outer,
            query,
            _integer(row.get("training_seed")),
            _integer(row.get("generation_seed")),
            str(row.get("action_id")),
        )
        observed_keys.append(key)
        expected_rows = tuple(evaluation_by_center.get(query, ()))
        expected_ids = [str(getattr(item, "sample_id")) for item in expected_rows]
        if (
            _integer(row.get("cell_ordinal")) != ordinal
            or row.get("config_contract_hash") != config_hash
            or row.get("router_plan_lock_hash") != plan_lock_hash
            or _json_list(row.get("evaluation_row_ids_json")) != expected_ids
            or row.get("evaluation_row_identity_hash") != _row_hash(expected_rows)
            or not _truthy(row.get("classifier_converged"))
            or _truthy(row.get("labels_available_to_fit_or_predict"))
            or _truthy(row.get("support_labels_used"))
            or _truthy(row.get("seed_selection_performed"))
            or not _truthy(row.get("target_expert_excluded"))
            or (phase == "development" and not _truthy(row.get("outer_and_query_experts_excluded")))
        ):
            raise ProtocolError("Residual top-up prediction cell escaped its seal boundary.")
        cells.append(
            {
                "cell_ordinal": ordinal,
                "phase": phase,
                "outer_target": outer,
                "query_center": query,
                "action_id": str(row["action_id"]),
                "training_seed": _integer(row["training_seed"]),
                "generation_seed": _integer(row["generation_seed"]),
                "evaluation_row_identity_hash": str(row["evaluation_row_identity_hash"]),
                "prediction_sha256": str(row["prediction_sha256"]),
                "probability_sha256": str(row["probability_sha256"]),
                "composition_hash": str(row["composition_hash"]),
                "action_hash": str(row["action_hash"]),
            }
        )
    if observed_keys != expected_keys:
        raise ProtocolError("Residual top-up global action/seed coverage drifted.")
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_global_prediction_seal_v1",
        "status": GLOBAL_PREDICTION_SEAL_STATUS,
        "config_contract_hash": config_hash,
        "support_partition_lock_hash": partition_lock_hash,
        "router_plan_lock_hash": plan_lock_hash,
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "prediction_array_member": "arrays/all_action_predictions.npz",
        "prediction_array_sha256": sha256_file(root / "arrays/all_action_predictions.npz"),
        "prediction_index_member": "tables/prediction_index.csv",
        "prediction_index_sha256": sha256_file(root / "tables/prediction_index.csv"),
        "cell_count": len(cells),
        "development_cell_count": sum(cell["phase"] == "development" for cell in cells),
        "target_cell_count": sum(cell["phase"] == "target" for cell in cells),
        "cells": cells,
        "all_actions_materialized": True,
        "all_predictions_hashed": True,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "whole_label_column_loaded": False,
    }
    return {**unhashed, "seal_hash": stable_hash(unhashed)}


def open_evaluation_labels_after_global_seal(
    config: object,
    partitions: object,
    plans: object,
    predictions: object,
    *,
    root: Path,
) -> tuple[dict[str, int], Mapping[str, object]]:
    """Open only non-support validation rows after the complete prediction seal."""

    # The capability is self-defending: callers cannot rely on having run a
    # separate validator earlier in the process.  Current partition, plan, and
    # prediction identities must reproduce the durable seal immediately before
    # the label-bearing manifest is touched.
    seal = validate_global_prediction_seal(
        config,
        partitions,
        plans,
        predictions,
        root=root,
    )
    if (
        seal.get("status") != GLOBAL_PREDICTION_SEAL_STATUS
        or seal.get("config_contract_hash") != getattr(config, "contract_hash", None)
        or seal.get("support_partition_lock_hash")
        != getattr(partitions, "lock_hash", None)
        or seal.get("router_plan_lock_hash") != getattr(plans, "lock_hash", None)
        or seal.get("validation_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or int(seal.get("cell_count", -1)) != EXPECTED_PREDICTION_CELL_COUNT
        or seal.get("prediction_array_sha256") != sha256_file(root / "arrays/all_action_predictions.npz")
        or seal.get("prediction_index_sha256") != sha256_file(root / "tables/prediction_index.csv")
        or seal.get("support_labels_opened") is not False
        or seal.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError("Residual top-up label capability failed seal validation.")
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center", {})
    support_by_center = getattr(partitions, "support_rows_by_center", {})
    rows = tuple(row for center in CENTERS for row in evaluation_by_center[center])
    support_ids = {
        str(getattr(row, "sample_id"))
        for center in CENTERS
        for row in support_by_center[center]
    }
    evaluation_ids = [str(getattr(row, "sample_id")) for row in rows]
    if len(evaluation_ids) != len(set(evaluation_ids)) or support_ids.intersection(evaluation_ids):
        raise ProtocolError("Residual top-up support/evaluation identity boundary drifted.")
    labels = _stream_labels(
        Path(getattr(config, "validation_manifest_path")),
        rows,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    by_sample = {
        str(getattr(row, "sample_id")): label
        for row, label in zip(rows, labels, strict=True)
    }
    report: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_label_access_report_v1",
        "status": "OPENED_AFTER_GLOBAL_ALL_ACTION_PREDICTION_SEAL",
        "prediction_seal_hash": seal["seal_hash"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "opened_row_count": len(rows),
        "opened_case_count": len({str(getattr(row, "case_id")) for row in rows}),
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
        "labels_used_for_q_not_H_diagnostic_action_calibration": True,
        "target_H_labels_used_for_own_action_selection": False,
        "labels_used_for_final_descriptive_scoring": True,
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
        raise ProtocolError("Residual top-up validation manifest hash drifted.")
    expected = {int(getattr(row, "manifest_row_index")): row for row in rows}
    if len(expected) != len(rows):
        raise ProtocolError("Residual top-up label requests duplicate manifest rows.")
    found: dict[int, int] = {}
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open residual top-up scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Residual top-up manifest lacks scoring fields.")
        for index, raw in enumerate(reader):
            row = expected.get(index)
            if row is None:
                continue
            if (
                raw["sample_id"] != str(getattr(row, "sample_id"))
                or raw["case_id"] != str(getattr(row, "case_id"))
                or raw["center"] != str(getattr(row, "center"))
                or raw["split"] != str(getattr(row, "split"))
            ):
                raise ProtocolError("Residual top-up manifest identity drifted.")
            try:
                label = int(raw["label"])
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Residual top-up manifest label is invalid.") from exc
            if label not in (0, 1):
                raise ProtocolError("Residual top-up labels must be binary.")
            found[index] = label
    if set(found) != set(expected):
        raise ProtocolError("Residual top-up manifest lacks requested rows.")
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
                "split": str(getattr(row, "split")),
                "partition_role": str(getattr(row, "partition_role")),
            }
            for row in rows
        ]
    )


def _integer(value: object) -> int:
    if isinstance(value, bool):
        raise ProtocolError("Residual top-up integer field is invalid.")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Residual top-up integer field is invalid.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _json_list(value: object) -> list[object]:
    import json

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Residual top-up JSON list is invalid.") from exc
    if not isinstance(parsed, list):
        raise ProtocolError("Residual top-up JSON field must be a list.")
    return parsed


__all__ = (
    "GLOBAL_PREDICTION_SEAL_MEMBER",
    "build_global_prediction_seal",
    "open_evaluation_labels_after_global_seal",
    "validate_global_prediction_seal",
)
