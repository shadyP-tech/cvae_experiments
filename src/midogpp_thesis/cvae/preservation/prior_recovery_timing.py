"""Incremental non-evidentiary timing and run-state reports."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Mapping

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError


TIMING_COLUMNS = (
    "record_key",
    "mode",
    "outer_target_center",
    "inner_pseudo_target_center",
    "phase",
    "objective_id",
    "training_key_hash",
    "cache_status",
    "elapsed_seconds",
    "used_for_selection",
    "claim_scope",
)


class RuntimeTimingRecorder:
    def __init__(self, root: Path, *, protocol_hash: str, mode: str) -> None:
        self.root = Path(root)
        self.protocol_hash = str(protocol_hash)
        self.mode = str(mode)
        self.records: dict[str, dict[str, object]] = {}
        summary_path = self.root / "reports/runtime_summary.json"
        table_path = self.root / "tables/runtime_timings.csv"
        if summary_path.is_file() and table_path.is_file():
            summary = _read_json(summary_path)
            if summary.get("protocol_hash") == self.protocol_hash and summary.get("mode") == self.mode:
                for row in _read_csv(table_path):
                    self.records[row["record_key"]] = dict(row)
        self._write(status="RUNNING")

    def record(
        self,
        *,
        phase: str,
        elapsed_seconds: float,
        outer_target_center: str,
        inner_pseudo_target_center: str = "",
        objective_id: str = "",
        training_key_hash: str = "",
        cache_status: str = "not_applicable",
    ) -> None:
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ProtocolError("Runtime timing duration must be finite and nonnegative.")
        identity = {
            "mode": self.mode,
            "outer_target_center": str(outer_target_center),
            "inner_pseudo_target_center": str(inner_pseudo_target_center),
            "phase": str(phase),
            "objective_id": str(objective_id),
            "training_key_hash": str(training_key_hash),
        }
        record_key = stable_hash(identity)
        self.records[record_key] = {
            "record_key": record_key,
            **identity,
            "cache_status": str(cache_status),
            "elapsed_seconds": elapsed,
            "used_for_selection": "false",
            "claim_scope": "diagnostic_only",
        }
        self._write(status="RUNNING")

    def finalize(self) -> None:
        self._write(status="COMPLETE")

    def _write(self, *, status: str) -> None:
        rows = [self.records[key] for key in sorted(self.records)]
        _atomic_csv(self.root / "tables/runtime_timings.csv", rows)
        _atomic_json(
            self.root / "reports/runtime_summary.json",
            _summary_payload(
                rows,
                protocol_hash=self.protocol_hash,
                mode=self.mode,
                status=status,
            ),
        )


def write_run_state(root: Path, *, protocol_hash: str, mode: str, status: str) -> None:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ValueError(f"Unsupported run state: {status}")
    _atomic_json(
        Path(root) / "reports/run_state.json",
        {
            "schema_version": "midogpp_prior_recovery_run_state_v1",
            "protocol_hash": str(protocol_hash),
            "mode": str(mode),
            "status": status,
        },
    )


def mark_run_failed(root: Path, *, mode: str) -> None:
    """Mark ordinary failures while leaving externally interrupted runs RUNNING."""

    root = Path(root)
    state_path = root / "reports/run_state.json"
    table_path = root / "tables/runtime_timings.csv"
    if not state_path.is_file():
        return
    state = _read_json(state_path)
    protocol_hash = str(state.get("protocol_hash", ""))
    if state.get("mode") != mode or not protocol_hash:
        return
    rows = _read_csv(table_path) if table_path.is_file() else []
    _atomic_json(
        root / "reports/runtime_summary.json",
        _summary_payload(
            rows,
            protocol_hash=protocol_hash,
            mode=mode,
            status="FAILED",
        ),
    )
    write_run_state(
        root,
        protocol_hash=protocol_hash,
        mode=mode,
        status="FAILED",
    )


def validate_runtime_reports(
    root: Path,
    *,
    protocol_hash: str,
    mode: str,
    checkpoint_index: Mapping[str, object],
    frame_index: Mapping[str, object],
) -> None:
    root = Path(root)
    rows = _read_csv(root / "tables/runtime_timings.csv")
    summary = _read_json(root / "reports/runtime_summary.json")
    run_state = _read_json(root / "reports/run_state.json")
    if not rows or tuple(rows[0]) != TIMING_COLUMNS:
        raise ProtocolError("Runtime timing table is empty or has unexpected columns.")
    seen: set[str] = set()
    for row in rows:
        identity = {
            "mode": row.get("mode", ""),
            "outer_target_center": row.get("outer_target_center", ""),
            "inner_pseudo_target_center": row.get("inner_pseudo_target_center", ""),
            "phase": row.get("phase", ""),
            "objective_id": row.get("objective_id", ""),
            "training_key_hash": row.get("training_key_hash", ""),
        }
        expected_cache_statuses = (
            {"hit", "miss"}
            if row.get("phase") in {"pca_frame", "cvae_training"}
            else {"not_applicable"}
        )
        if (
            row.get("record_key") in seen
            or row.get("record_key") != stable_hash(identity)
            or row.get("mode") != mode
            or row.get("used_for_selection") != "false"
            or row.get("claim_scope") != "diagnostic_only"
            or not row.get("phase")
            or row.get("cache_status") not in expected_cache_statuses
        ):
            raise ProtocolError("Runtime timing row violates its diagnostic contract.")
        try:
            elapsed = float(row["elapsed_seconds"])
        except (KeyError, ValueError) as exc:
            raise ProtocolError("Runtime timing duration is malformed.") from exc
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ProtocolError("Runtime timing duration must be finite and nonnegative.")
        seen.add(row["record_key"])
    expected_summary = _summary_payload(
        rows,
        protocol_hash=str(protocol_hash),
        mode=str(mode),
        status="COMPLETE",
    )
    if summary != expected_summary:
        raise ProtocolError("Runtime summary does not match its timing rows.")
    if run_state != {
        "schema_version": "midogpp_prior_recovery_run_state_v1",
        "protocol_hash": str(protocol_hash),
        "mode": str(mode),
        "status": "COMPLETE",
    }:
        raise ProtocolError("Prior-recovery run state is not COMPLETE for this protocol.")
    checkpoint_records = checkpoint_index.get("records")
    frame_records = frame_index.get("records")
    if not isinstance(checkpoint_records, list) or not isinstance(frame_records, list):
        raise ProtocolError("Runtime timing coverage received malformed provenance indices.")
    expected_training_keys = {
        str(record.get("training_key_hash", ""))
        for record in checkpoint_records
        if isinstance(record, Mapping)
    }
    cvae_rows = [row for row in rows if row["phase"] == "cvae_training"]
    sampling_rows = [row for row in rows if row["phase"] == "sampling_and_scoring"]
    pca_rows = [row for row in rows if row["phase"] == "pca_frame"]
    if (
        len(cvae_rows) != len(expected_training_keys)
        or {row["training_key_hash"] for row in cvae_rows} != expected_training_keys
        or len(pca_rows) != len(frame_records)
    ):
        raise ProtocolError("Runtime timing coverage differs from checkpoint or PCA provenance.")
    if mode == "source_inner":
        nested_rows = [row for row in rows if row["phase"] == "nested_classifier_selection"]
        if (
            len(nested_rows) != len(frame_records)
            or len(sampling_rows) != len(expected_training_keys)
            or {row["training_key_hash"] for row in sampling_rows} != expected_training_keys
        ):
            raise ProtocolError("Source-inner runtime timing fold coverage is incomplete.")
    elif mode == "outer":
        fisher_rows = [row for row in rows if row["phase"] == "task_fisher_fit"]
        if (
            len(fisher_rows) != len(frame_records)
            or len(sampling_rows) * 2 != len(expected_training_keys)
        ):
            raise ProtocolError("Outer runtime timing fold/seed coverage is incomplete.")


def _summary_payload(
    rows: list[Mapping[str, object]],
    *,
    protocol_hash: str,
    mode: str,
    status: str,
) -> dict[str, object]:
    by_phase: dict[str, float] = {}
    cache_counts: dict[str, int] = {}
    total = 0.0
    for row in rows:
        elapsed = float(row.get("elapsed_seconds", 0.0))
        phase = str(row.get("phase", ""))
        cache_status = str(row.get("cache_status", ""))
        by_phase[phase] = by_phase.get(phase, 0.0) + elapsed
        cache_counts[cache_status] = cache_counts.get(cache_status, 0) + 1
        total += elapsed
    return {
        "schema_version": "midogpp_prior_recovery_runtime_summary_v1",
        "protocol_hash": str(protocol_hash),
        "mode": str(mode),
        "status": status,
        "claim_scope": "diagnostic_only",
        "used_for_selection": False,
        "n_records": len(rows),
        "total_recorded_seconds": total,
        "seconds_by_phase": {key: by_phase[key] for key in sorted(by_phase)},
        "cache_status_counts": {key: cache_counts[key] for key in sorted(cache_counts)},
    }


def _atomic_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Malformed runtime timing table: {path}") from exc


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed runtime report JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected runtime report JSON object: {path}")
    return payload
