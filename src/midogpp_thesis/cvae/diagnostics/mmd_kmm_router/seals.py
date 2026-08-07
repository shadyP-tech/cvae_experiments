"""Global pre-label prediction capability for the MMD/KMM diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import MMDKMMRouterDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PREDICTION_CELL_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    row_identity_hash,
)
from .inputs import PartitionSurface
from .planning import RouterPlans
from .prediction import (
    TARGET_PREDICTION_ARRAY_MEMBER,
    TARGET_PREDICTION_INDEX_MEMBER,
    PredictionStore,
)


GLOBAL_PREDICTION_SEAL_MEMBER = "manifests/global_target_prediction_seal.json"


def _global_prediction_seal_payload(
    config: MMDKMMRouterDiagnosticConfig,
    partitions: PartitionSurface,
    plans: RouterPlans,
    predictions: PredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    expected_keys = tuple(
        (target, training_seed, generation_seed, arm)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for arm in ("equal_union_control", "mmd_kmm")
    )
    observed_keys = tuple(
        (
            str(row["target_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            str(row["arm_role"]),
        )
        for row in predictions.index_rows
    )
    if observed_keys != expected_keys or len(observed_keys) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("MMD/KMM global seal prediction coverage drifted.")
    rows_by_target = {
        target: [row.sample_id for row in partitions.evaluation_rows_by_center[target]]
        for target in CENTERS
    }
    row_hashes = {
        target: row_identity_hash(partitions.evaluation_rows_by_center[target])
        for target in CENTERS
    }
    cells = []
    for row in predictions.index_rows:
        target = str(row["target_center"])
        if (
            json.loads(str(row["evaluation_row_ids_json"])) != rows_by_target[target]
            or row["evaluation_row_identity_hash"] != row_hashes[target]
            or row["plan_hash"] != plans.plans_by_target[target]["plan_hash"]
            or _truthy(row["labels_available_to_fit_or_predict"])
            or _truthy(row["support_rows_used_to_predict"])
            or _truthy(row["seed_selection_performed"])
        ):
            raise ProtocolError("MMD/KMM global seal cell escaped its row/plan boundary.")
        cells.append(
            {
                "target_center": target,
                "arm_role": str(row["arm_role"]),
                "training_seed": int(row["training_seed"]),
                "generation_seed": int(row["generation_seed"]),
                "evaluation_row_identity_hash": str(row["evaluation_row_identity_hash"]),
                "prediction_sha256": str(row["prediction_sha256"]),
                "probability_sha256": str(row["probability_sha256"]),
                "composition_hash": str(row["composition_hash"]),
                "classifier_config_hash": str(row["classifier_config_hash"]),
                "plan_hash": str(row["plan_hash"]),
            }
        )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_global_target_prediction_seal_v1",
        "status": "COMPLETE_BEFORE_ANY_EVALUATION_LABEL_ACCESS",
        "config_contract_hash": config.contract_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "router_plan_lock_hash": plans.lock_hash,
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "prediction_index_member": TARGET_PREDICTION_INDEX_MEMBER,
        "prediction_index_sha256": _sha256_file(root / TARGET_PREDICTION_INDEX_MEMBER),
        "prediction_arrays_member": TARGET_PREDICTION_ARRAY_MEMBER,
        "prediction_arrays_sha256": _sha256_file(root / TARGET_PREDICTION_ARRAY_MEMBER),
        "evaluation_row_ids_by_target": rows_by_target,
        "evaluation_row_identity_hash_by_target": row_hashes,
        "target_count": len(CENTERS),
        "cell_count": len(cells),
        "cells": cells,
        "target_labels_opened": False,
        "support_labels_opened": False,
        "development_labels_opened": False,
        "all_predictions_persisted": True,
        "all_predictions_hashed": True,
    }
    return {**unhashed, "seal_hash": stable_hash(unhashed)}


def build_global_prediction_seal(
    config: MMDKMMRouterDiagnosticConfig,
    partitions: PartitionSurface,
    plans: RouterPlans,
    predictions: PredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    payload = _global_prediction_seal_payload(
        config,
        partitions,
        plans,
        predictions,
        root=root,
    )
    _write_json(root / GLOBAL_PREDICTION_SEAL_MEMBER, payload)
    return payload


def validate_global_prediction_seal(
    config: MMDKMMRouterDiagnosticConfig,
    partitions: PartitionSurface,
    plans: RouterPlans,
    predictions: PredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    observed = _json(root / GLOBAL_PREDICTION_SEAL_MEMBER)
    expected = _global_prediction_seal_payload(
        config,
        partitions,
        plans,
        predictions,
        root=root,
    )
    if observed != expected:
        raise ProtocolError(
            "MMD/KMM global prediction seal is not independently reconstructible."
        )
    return observed


def open_evaluation_labels(
    config: MMDKMMRouterDiagnosticConfig,
    partitions: PartitionSurface,
    *,
    root: Path,
) -> tuple[dict[str, int], Mapping[str, object]]:
    seal_path = root / GLOBAL_PREDICTION_SEAL_MEMBER
    seal = _json(seal_path)
    unhashed = {key: value for key, value in seal.items() if key != "seal_hash"}
    if (
        seal.get("seal_hash") != stable_hash(unhashed)
        or seal.get("status") != "COMPLETE_BEFORE_ANY_EVALUATION_LABEL_ACCESS"
        or seal.get("config_contract_hash") != config.contract_hash
        or seal.get("support_partition_lock_hash") != partitions.lock_hash
        or seal.get("validation_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or seal.get("cell_count") != EXPECTED_PREDICTION_CELL_COUNT
        or seal.get("prediction_index_sha256")
        != _sha256_file(root / TARGET_PREDICTION_INDEX_MEMBER)
        or seal.get("prediction_arrays_sha256")
        != _sha256_file(root / TARGET_PREDICTION_ARRAY_MEMBER)
    ):
        raise ProtocolError("MMD/KMM label capability seal failed validation.")
    rows = tuple(
        row for target in CENTERS for row in partitions.evaluation_rows_by_center[target]
    )
    if any(row.partition_role != "evaluation" for row in rows):
        raise ProtocolError("MMD/KMM label request contains support rows.")
    expected_ids = {
        target: [row.sample_id for row in partitions.evaluation_rows_by_center[target]]
        for target in CENTERS
    }
    expected_hashes = {
        target: row_identity_hash(partitions.evaluation_rows_by_center[target])
        for target in CENTERS
    }
    if (
        seal.get("evaluation_row_ids_by_target") != expected_ids
        or seal.get("evaluation_row_identity_hash_by_target") != expected_hashes
    ):
        raise ProtocolError("MMD/KMM label request differs from sealed evaluation rows.")
    labels = _stream_labels(
        config.validation_manifest_path,
        rows,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    by_sample = {row.sample_id: label for row, label in zip(rows, labels, strict=True)}
    label_hashes_by_target = {}
    cursor = 0
    for target in CENTERS:
        target_rows = partitions.evaluation_rows_by_center[target]
        target_labels = labels[cursor : cursor + len(target_rows)]
        cursor += len(target_rows)
        if set(target_labels) != {0, 1}:
            raise ProtocolError(f"MMD/KMM target {target} lacks binary evaluation support.")
        label_hashes_by_target[target] = stable_hash(
            {
                "target_center": target,
                "row_identity_hash": row_identity_hash(target_rows),
                "labels": list(target_labels),
                "prediction_seal_hash": seal["seal_hash"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            }
        )
    report: dict[str, object] = {
        "schema_version": "midogpp_mmd_kmm_label_access_report_v1",
        "status": "OPENED_AFTER_GLOBAL_PREDICTION_SEAL",
        "prediction_seal_hash": seal["seal_hash"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "opened_row_count": len(rows),
        "opened_target_count": len(CENTERS),
        "label_vector_hash_by_target": label_hashes_by_target,
        "support_label_count": 0,
        "train_label_count": 0,
        "test_label_count": 0,
        "excluded_center_label_count": 0,
        "labels_available_to_router": False,
        "labels_used_for_policy_or_hyperparameter_selection": False,
        "labels_used_for_scoring_only": True,
    }
    report["label_access_report_hash"] = stable_hash(report)
    return by_sample, report


def _stream_labels(
    manifest_path: Path,
    rows: Sequence[ValidationRowIdentity],
    *,
    expected_sha256: str,
) -> tuple[int, ...]:
    if _sha256_file(manifest_path) != expected_sha256:
        raise ProtocolError("MMD/KMM validation manifest hash drifted.")
    expected_by_index = {row.manifest_row_index: row for row in rows}
    if len(expected_by_index) != len(rows):
        raise ProtocolError("MMD/KMM requested manifest row indices duplicate.")
    labels: dict[int, int] = {}
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open MMD/KMM scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("MMD/KMM scoring manifest lacks required fields.")
        for index, raw in enumerate(reader):
            expected = expected_by_index.get(index)
            if expected is None:
                # This branch precedes the sole label read.  Support and every
                # non-evaluation row remain unopened.
                continue
            observed = (
                str(raw.get("sample_id", "")),
                str(raw.get("case_id", "")),
                str(raw.get("center", "")),
                str(raw.get("split", "")),
            )
            wanted = (expected.sample_id, expected.case_id, expected.center, expected.split)
            if observed != wanted:
                raise ProtocolError("MMD/KMM scoring-manifest identity drifted.")
            try:
                value = int(str(raw["label"]).strip())
            except (TypeError, ValueError) as exc:
                raise ProtocolError("MMD/KMM evaluation label is not binary.") from exc
            if value not in (0, 1):
                raise ProtocolError("MMD/KMM evaluation label is not binary.")
            labels[index] = value
    if set(labels) != set(expected_by_index):
        raise ProtocolError("MMD/KMM evaluation label coverage drifted.")
    return tuple(labels[row.manifest_row_index] for row in rows)


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM seal: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("MMD/KMM seal must be an object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "GLOBAL_PREDICTION_SEAL_MEMBER",
    "build_global_prediction_seal",
    "open_evaluation_labels",
    "validate_global_prediction_seal",
)
