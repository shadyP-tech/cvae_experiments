"""Terminal evaluation-label access after the global target seal."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ....data.features.uniform_b_routing_validation.config import MANIFEST_SHA256
from ...protocol import ProtocolError
from .artifact_io import read_json, sha256_file
from .target_seal import TARGET_PREDICTION_SEAL_MEMBER
from .target_seal import TargetScoringCapability


def open_target_labels_after_global_seal(
    config: object, partitions: object, *, root: Path,
    capability: TargetScoringCapability,
) -> tuple[dict[str, int], Mapping[str, object]]:
    if (
        not isinstance(capability, TargetScoringCapability)
        or capability.root != root.resolve()
    ):
        raise ProtocolError("Target labels require a reconstructively validated capability.")
    seal = read_json(root / TARGET_PREDICTION_SEAL_MEMBER)
    unhashed = {key: value for key, value in seal.items() if key != "seal_hash"}
    if (
        seal.get("status") != "SEALED_ALL_TARGET_ACTIONS_BEFORE_TERMINAL_TARGET_SCORING"
        or seal.get("seal_hash") != stable_hash(unhashed)
        or seal.get("all_actions_frozen") is not True
        or seal.get("all_predictions_materialized") is not True
        or seal.get("target_support_labels_opened") is not False
        or seal.get("target_evaluation_labels_opened") is not False
        or seal != dict(capability.payload)
    ):
        raise ProtocolError("Target labels require a valid durable global seal.")
    manifest = Path(getattr(config, "validation_manifest_path"))
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ProtocolError("Target scoring manifest hash drifted.")
    requested = {
        row.manifest_row_index: row
        for target in partitions.evaluation_rows_by_center
        for row in partitions.evaluation_rows_by_center[target]
    }
    support = {
        row.manifest_row_index
        for target in partitions.support_rows_by_center
        for row in partitions.support_rows_by_center[target]
    }
    if support & set(requested):
        raise ProtocolError("Target evaluation rows overlap fixed support.")
    labels: dict[str, int] = {}
    try:
        handle = manifest.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open terminal target manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Target scoring manifest fields drifted.")
        for index, raw in enumerate(reader):
            wanted = requested.get(index)
            if wanted is None:
                continue
            if (raw["sample_id"], raw["case_id"], raw["center"], raw["split"]) != (
                wanted.sample_id, wanted.case_id, wanted.center, wanted.split
            ):
                raise ProtocolError("Target scoring identity drifted.")
            labels[wanted.sample_id] = _binary(raw["label"])
    if set(labels) != {row.sample_id for row in requested.values()}:
        raise ProtocolError("Target scoring label coverage drifted.")
    label_hash = stable_hash([[key, labels[key]] for key in sorted(labels)])
    report = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_target_label_access_v1",
        "status": "PASS", "global_target_prediction_seal_hash": str(seal["seal_hash"]),
        "manifest_sha256": MANIFEST_SHA256, "evaluation_label_count": len(labels),
        "evaluation_case_count": len({row.case_id for row in requested.values()}),
        "support_label_count": 0, "whole_label_column_materialized": False,
        "labels_by_sample_id_persisted": False, "label_identity_hash": label_hash,
        "target_plan_frozen_before_label_access": True,
        "policy_or_action_update_after_label_access": False,
    }
    return labels, report


def _binary(value: object) -> int:
    try: number = float(str(value))
    except ValueError as exc: raise ProtocolError("Target label is not numeric.") from exc
    if number not in (0.0, 1.0): raise ProtocolError("Target label is outside {0,1}.")
    return int(number)


__all__ = ("open_target_labels_after_global_seal",)
