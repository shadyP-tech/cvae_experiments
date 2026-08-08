"""Atomic persistence and reconstructive comparison primitives for bundles."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    PERMUTATION_CONTRAST,
    PRIMARY_CONTRASTS,
    FreshEvaluationReport,
)
from .prediction_cache import PREDICTION_INDEX_COLUMNS, PredictionCache
from .prediction_io import array_sha256


def primary_result_payload(report: FreshEvaluationReport) -> dict[str, object]:
    inference = {row.contrast_id: row for row in report.contrast_inference}
    required = ("R-B", "R-G_delta", "R-U", "R-P")
    if not set(required).issubset(inference):
        raise ProtocolError("Utility-aligned required confirmation contrasts are absent.")
    promote = all(inference[item].one_sided_95_lcb > 0.0 for item in required)
    return {
        "schema_version": "midogpp_utility_aligned_fresh_result_v1",
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "inference_unit": "target_center",
        "center_count": len(CENTERS),
        "primary_contrasts": [item[0] for item in PRIMARY_CONTRASTS],
        "permutation_contrast": PERMUTATION_CONTRAST[0],
        "contrast_inference": {key: asdict(value) for key, value in inference.items()},
        "decision": "PROMOTE_ROUTER" if promote else "DO_NOT_PROMOTE",
        "promotion_requires_positive_lcb_for": list(required),
        "prior_cardinality_transfer_role": "eligibility_only",
        "prior_expected_improvement_claimed": False,
        "oracle_diagnostics_terminal_only": True,
        "policy_update_emitted": False,
    }


def structural_checks() -> dict[str, object]:
    return {
        "status": "PASS",
        "logical_prediction_count": EXPECTED_LOGICAL_PREDICTION_COUNT,
        "ensemble_metric_count": EXPECTED_ENSEMBLE_METRIC_COUNT,
        "center_contrast_count": len(CENTERS) * 6,
        "center_count": len(CENTERS),
        "labels_opened_after_prediction_seal": True,
        "policy_update_emitted": False,
    }


def write_content_index(root: Path) -> None:
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    records = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        member = path.relative_to(root).as_posix()
        if member not in excluded:
            records.append(
                {
                    "relative_path": member,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "schema_version": "midogpp_utility_aligned_fresh_content_index_v1",
        "records": records,
    }
    atomic_json(
        root / "manifests/content_index.json",
        {**payload, "content_hash": stable_hash(payload)},
    )


def require_table(path: Path, expected: Sequence[Mapping[str, object]]) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            observed = tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read utility-aligned table: {path.name}.") from exc
    rendered = tuple(
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in expected
    )
    if observed != rendered:
        raise ProtocolError(f"Utility-aligned table drifted: {path.name}.")


def require_prediction_index(path: Path, cache: PredictionCache) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_INDEX_COLUMNS:
                raise ProtocolError("Utility-aligned prediction-index columns drifted.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot read utility-aligned prediction index.") from exc
    if len(rows) != EXPECTED_LOGICAL_PREDICTION_COUNT:
        raise ProtocolError("Utility-aligned prediction-index coverage drifted.")
    record_by_task = {record.task_id: record for record in cache.records}
    for index, (row, cell) in enumerate(zip(rows, cache.predictions, strict=True)):
        task_id = (
            f"target_{cell.target_center}__train_{cell.training_seed}"
            f"__gen_{cell.generation_seed}"
        )
        expected = {
            "schema_version": "midogpp_utility_aligned_prediction_index_row_v1",
            "target_center": cell.target_center,
            "training_seed": str(cell.training_seed),
            "generation_seed": str(cell.generation_seed),
            "action_id": cell.action_id,
            "action_hash": cell.action_hash,
            "composition_hash": cell.composition_hash,
            "evaluation_row_ids_hash": stable_hash(list(cell.evaluation_row_ids)),
            "probability_member": (
                "checkpoints/predictions/" + record_by_task[task_id].probability_member
            ),
            "probability_row": str(index % EXPECTED_ACTION_COUNT_PER_TARGET),
            "probability_sha256": array_sha256(cell.probabilities),
        }
        if row != expected:
            raise ProtocolError("Utility-aligned prediction-index row drifted.")


def write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Utility-aligned output table would be empty: {path.name}.")
    columns = tuple(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if tuple(row) != columns:
                raise ProtocolError("Utility-aligned table columns drifted.")
            writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with source.open("rb") as read_handle, temporary.open("wb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    os.replace(temporary, target)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Utility-aligned bundle JSON must be a mapping.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "atomic_copy", "atomic_json", "primary_result_payload", "read_json",
    "require_prediction_index", "require_table", "sha256_file", "structural_checks",
    "write_content_index", "write_table",
)
