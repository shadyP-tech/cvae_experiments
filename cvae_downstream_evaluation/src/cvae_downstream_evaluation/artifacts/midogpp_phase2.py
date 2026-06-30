"""Artifact helpers for MIDOG++ phase-2 target-support adaptation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas.midogpp_phase2 import (
    PHASE2_REQUIRED_DIRS,
    PHASE2_ROOT_NAME,
    assert_phase2_artifact_contract,
    assert_phase2_artifact_root,
    build_locked_phase2_support_eval_split,
)


def create_phase2_artifact_root(root: Path) -> dict[str, Path]:
    """Create the approved empty phase-2 artifact directory scaffold."""

    assert_phase2_artifact_root(root)
    paths: dict[str, Path] = {}
    for name in PHASE2_REQUIRED_DIRS:
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    assert_phase2_artifact_contract(root)
    return paths


def default_phase2_artifact_root(artifacts_root: Path) -> Path:
    """Return ``artifacts/midogpp/phase2_target_support_adaptation_virchow2_seed42``."""

    return Path(artifacts_root) / "midogpp" / PHASE2_ROOT_NAME


def write_phase2_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a phase-2 CSV after rejecting empty rows and forbidden matrix names."""

    _assert_phase2_output_path(path)
    if not rows:
        raise ProtocolError(f"Refusing to write empty phase-2 CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(str(column))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_phase2_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a phase-2 JSON report after rejecting forbidden artifact names."""

    _assert_phase2_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_locked_phase2_support_eval_split(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> tuple[Path, Path]:
    """Materialize locked support/eval manifests before any phase-2 scorer runs."""

    assert_phase2_artifact_root(root)
    support_rows, eval_rows = build_locked_phase2_support_eval_split(
        rows,
        heldout_center=heldout_center,
        support_size=support_size,
        support_seed=support_seed,
    )
    support_path = Path(root) / "manifests" / "support_sets.csv"
    eval_path = Path(root) / "manifests" / "eval_sets.csv"
    write_phase2_csv(support_path, support_rows)
    write_phase2_csv(eval_path, eval_rows)
    return support_path, eval_path


def phase2_validation_payload(
    *,
    artifacts_root: Path,
    status: str,
    checks: Mapping[str, object],
    error_message: str = "",
) -> dict[str, object]:
    """Build the standard phase-2 validation report payload."""

    return {
        "schema_version": "midogpp_phase2_validation_report_v1",
        "artifacts_root": str(artifacts_root),
        "status": status,
        "checks": dict(checks),
        "error_message": error_message,
    }


def _assert_phase2_output_path(path: Path) -> None:
    if Path(path).name == "target_support_downstream_matrix.csv":
        raise ProtocolError("target_support_downstream_matrix.csv is forbidden for MIDOG++ phase-2.")
