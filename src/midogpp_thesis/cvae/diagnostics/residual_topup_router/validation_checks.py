"""Focused report, table, and claim checks for bundle reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import read_csv_rows, read_json
from .contracts import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
)


def compare_csv_rows(
    path: Path,
    expected: Sequence[Mapping[str, object]],
    *,
    role: str,
) -> None:
    observed = read_csv_rows(path)
    if len(observed) != len(expected):
        raise ProtocolError(f"Residual top-up {role} row count drifted.")
    for left, right in zip(observed, expected, strict=True):
        if set(left) != set(right):
            raise ProtocolError(f"Residual top-up {role} columns drifted.")
        for key, expected_value in right.items():
            raw = left[key]
            if isinstance(expected_value, bool):
                equal = raw.strip().lower() == str(expected_value).lower()
            elif isinstance(expected_value, (int, float)) and not isinstance(
                expected_value, bool
            ):
                try:
                    equal = bool(
                        np.isclose(
                            float(raw),
                            float(expected_value),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
                except ValueError:
                    equal = False
            else:
                equal = raw == str(expected_value)
            if not equal:
                raise ProtocolError(
                    f"Residual top-up {role} value drifted at {key!r}."
                )


def require_json_equal(path: Path, expected: Mapping[str, object], *, role: str) -> None:
    if read_json(path) != dict(expected):
        raise ProtocolError(f"Residual top-up {role} is not reconstructible.")


def validate_phase_reports(
    root: Path,
    expected_by_filename: Mapping[str, Mapping[str, object]],
) -> None:
    for filename, expected in expected_by_filename.items():
        observed = read_json(root / "reports" / filename)
        unhashed = {
            key: value for key, value in observed.items() if key != "phase_report_hash"
        }
        if (
            observed != dict(expected)
            or observed.get("phase_report_hash") != stable_hash(unhashed)
            or observed.get("status") != "COMPLETE"
            or observed.get("diagnostic_only") is not True
        ):
            raise ProtocolError(
                f"Residual top-up phase report drifted: {filename}."
            )


def validate_runtime_report(root: Path) -> Mapping[str, object]:
    payload = read_json(root / "reports/runtime_summary.json")
    preflight = payload.get("workstation_preflight")
    gpu_rows = preflight.get("gpus") if isinstance(preflight, Mapping) else None
    if (
        payload.get("schema_version")
        != "midogpp_residual_topup_runtime_summary_v1"
        or payload.get("status") != "PASS"
        or int(payload.get("source_task_count", -1)) != EXPECTED_SOURCE_TASK_COUNT
        or int(payload.get("source_block_count", -1)) != EXPECTED_SOURCE_BLOCK_COUNT
        or int(payload.get("prediction_task_count", -1))
        != EXPECTED_DEVELOPMENT_TASK_COUNT + EXPECTED_TARGET_TASK_COUNT
        or int(payload.get("prediction_cell_count", -1))
        != EXPECTED_PREDICTION_CELL_COUNT
        or not 1
        <= int(payload.get("unique_classifier_fit_count", -1))
        <= MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        or int(payload.get("gpu_worker_count", -1)) != 2
        or int(payload.get("classifier_worker_count", -1)) != CLASSIFIER_WORKERS
        or int(payload.get("classifier_threads_per_worker", -1))
        != CLASSIFIER_THREADS_PER_WORKER
        or payload.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or payload.get("dependency_versions_are_report_only") is not True
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or preflight.get("parent_cuda_context_initialized") is not False
        or int(preflight.get("classifier_worker_thread_product", -1))
        != CLASSIFIER_WORKERS * CLASSIFIER_THREADS_PER_WORKER
        or not isinstance(gpu_rows, list)
        or len(gpu_rows) != 2
        or {
            int(row.get("index", -1))
            for row in gpu_rows
            if isinstance(row, Mapping)
        }
        != {0, 1}
        or any(
            "RTX A5000" not in str(row.get("name"))
            for row in gpu_rows
            if isinstance(row, Mapping)
        )
    ):
        raise ProtocolError("Residual top-up runtime report drifted.")
    return payload


def validate_final_state(root: Path, *, allow_pending: bool) -> None:
    state = read_json(root / "reports/run_state.json")
    if allow_pending:
        if state.get("status") != "RUNNING" or state.get("phase") != "VALIDATING":
            raise ProtocolError("Residual top-up pending validation state drifted.")
        return
    report = read_json(root / "reports/validation_report.json")
    if (
        state.get("status") != "COMPLETE"
        or state.get("phase") != "COMPLETE"
        or report.get("status") != "PASS"
        or report.get("validator")
        != "validate_residual_topup_router_bundle"
        or not isinstance(report.get("checks"), Mapping)
    ):
        raise ProtocolError("Residual top-up final validation state is incomplete.")


__all__ = (
    "compare_csv_rows",
    "require_json_equal",
    "validate_final_state",
    "validate_phase_reports",
    "validate_runtime_report",
)
