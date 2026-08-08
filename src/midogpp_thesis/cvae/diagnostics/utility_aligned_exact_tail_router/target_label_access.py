"""Capability-gated terminal target label access."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import (
    MANIFEST_SHA256 as EXPECTED_MANIFEST_SHA256,
)
from ...protocol import ProtocolError
from .artifact_io import read_json, sha256_file
from .input_contracts import FixedPartitionSurface
from .target_prediction_contracts import TARGET_PREDICTION_SEAL_MEMBER


def open_target_labels_after_global_seal(
    config: object,
    partitions: FixedPartitionSurface,
    *,
    root: Path,
) -> tuple[dict[str, int], Mapping[str, object]]:
    seal_path = root / TARGET_PREDICTION_SEAL_MEMBER
    seal = read_json(seal_path)
    unhashed = {key: value for key, value in seal.items() if key != "seal_hash"}
    if (
        seal.get("status")
        != "SEALED_ALL_TARGET_ACTIONS_BEFORE_TERMINAL_TARGET_SCORING"
        or seal.get("seal_hash") != stable_hash(unhashed)
        or seal.get("development_crossfit_labels_previously_opened") is not True
        or seal.get("outer_H_development_rows_excluded_from_plan_H") is not True
        or seal.get("terminal_target_scoring_capability_opened") is not False
        or seal.get("all_predictions_materialized") is not True
    ):
        raise ProtocolError("Target labels require a valid durable global seal.")
    manifest = Path(getattr(config, "validation_manifest_path"))
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ProtocolError("Target scoring manifest hash drifted.")
    requested = {
        row.manifest_row_index: row
        for target in partitions.evaluation_rows_by_center
        for row in partitions.evaluation_rows_by_center[target]
    }
    support_indices = {
        row.manifest_row_index
        for target in partitions.support_rows_by_center
        for row in partitions.support_rows_by_center[target]
    }
    if set(requested).intersection(support_indices):
        raise ProtocolError("Target scoring rows overlap support rows.")
    labels: dict[str, int] = {}
    converted_label_count = 0
    try:
        handle = manifest.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open target scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Target scoring manifest fields drifted.")
        for index, raw in enumerate(reader):
            wanted = requested.get(index)
            if wanted is None:
                # Crucially, no label conversion occurs for fixed-support or
                # otherwise unrequested manifest rows.
                continue
            if (
                str(raw["sample_id"]) != wanted.sample_id
                or str(raw["case_id"]) != wanted.case_id
                or str(raw["center"]) != wanted.center
                or str(raw["split"]) != wanted.split
            ):
                raise ProtocolError("Target scoring row identity drifted.")
            label = _binary(raw["label"])
            if wanted.sample_id in labels:
                raise ProtocolError("Target scoring sample is duplicated.")
            labels[wanted.sample_id] = label
            converted_label_count += 1
    if set(labels) != {row.sample_id for row in requested.values()}:
        raise ProtocolError("Target scoring label coverage drifted.")
    label_hash = stable_hash(
        [[sample_id, labels[sample_id]] for sample_id in sorted(labels)]
    )
    report = {
        "schema_version": "midogpp_utility_aligned_stage90_target_label_access_v1",
        "status": "PASS",
        "global_target_prediction_seal_hash": str(seal["seal_hash"]),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "evaluation_label_count": converted_label_count,
        "evaluation_case_count": len({row.case_id for row in requested.values()}),
        "support_label_count": 0,
        "whole_label_column_materialized": False,
        "labels_by_sample_id_persisted": False,
        "label_identity_hash": label_hash,
        "development_crossfit_labels_previously_opened": True,
        "outer_H_label_rows_used_to_build_plan_H": False,
        "terminal_target_scoring_opened_after_target_seal": True,
        "policy_or_action_update_after_label_access": False,
    }
    return labels, report


def _binary(value: object) -> int:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Target scoring label is not numeric.") from exc
    if number not in (0.0, 1.0):
        raise ProtocolError("Target scoring label is outside {0,1}.")
    return int(number)


__all__ = ("open_target_labels_after_global_seal",)
