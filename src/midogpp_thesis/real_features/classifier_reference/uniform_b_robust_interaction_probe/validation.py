"""Independent validation for robust-Nyström versus bilinear B+."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..uniform_b_nonlinear_probe.statistics import binary_metrics
from .config import RobustInteractionConfig


REQUIRED = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/source_only_candidate_locks.json",
    "manifests/content_index.json",
    "tables/paired_error_audit.csv",
    "tables/error_group_summary.csv",
    "tables/robust_selector_cells.csv",
    "tables/bilinear_selector_cells.csv",
    "tables/family_candidate_summary.csv",
    "tables/outer_results.csv",
    "tables/outer_predictions.csv",
    "tables/stability_predictions.csv",
    "tables/center_family_comparison.csv",
    "reports/regression_audit_summary.json",
    "reports/family_decision.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/runtime_summary.json",
    "reports/leakage_provenance_report.json",
)


def validate_robust_interaction_bundle(
    root: str | Path,
    *,
    config: RobustInteractionConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(item for item in required if not (path / item).is_file())
    if missing:
        raise ProtocolError(f"Robust-interaction bundle is incomplete: {missing}.")
    frozen = _json(path / "manifests/frozen_protocol_snapshot.json")
    protocol = _json(path / "manifests/protocol_manifest.json")
    leakage = _json(path / "reports/leakage_provenance_report.json")
    if (
        stable_hash({key: value for key, value in frozen.items() if key != "protocol_hash"})
        != frozen.get("protocol_hash")
        or protocol.get("protocol_hash") != frozen.get("protocol_hash")
        or protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("validation_scored") is not False
        or protocol.get("test_scored") is not False
        or leakage.get("center_specific_thresholds") is not False
    ):
        raise ProtocolError("Robust-interaction protocol firewall failed.")
    robust = _csv(path / "tables/robust_selector_cells.csv")
    bilinear = _csv(path / "tables/bilinear_selector_cells.csv")
    audit = _csv(path / "tables/paired_error_audit.csv")
    predictions = _csv(path / "tables/outer_predictions.csv")
    stability = _csv(path / "tables/stability_predictions.csv")
    comparisons = _csv(path / "tables/center_family_comparison.csv")
    if (
        len(robust) != 9 * 8 * 3
        or len(bilinear) != 9 * 8 * 3
        or len(audit) != 751 + 550 + 1318
        or len(predictions) != 4 * 9648
        or len(stability) != 4 * 9648
        or len(comparisons) != 9 * 2 * 3
    ):
        raise ProtocolError("Robust-interaction artifact cardinality drifted.")
    for row in robust + bilinear:
        outer = row["outer_center"]
        inner = row["inner_center"]
        if (
            set(json.loads(row["train_centers"]))
            != set(config.heldout_centers).difference({outer, inner})
            or row["selection_used_outer_labels"].lower() != "false"
            or row["fit_used_outer_or_inner_center"].lower() != "false"
        ):
            raise ProtocolError("Robust-interaction selector leakage failed.")
    _validate_metrics(predictions, comparisons, config)
    _validate_content(path)
    checks = {
        "status": "PASS",
        "robust_selector_cells": len(robust),
        "bilinear_selector_cells": len(bilinear),
        "audited_error_rows": len(audit),
        "outer_predictions": len(predictions),
        "stability_predictions": len(stability),
        "validation_scored": False,
        "test_scored": False,
        "decision": _json(path / "reports/family_decision.json")["decision"],
    }
    if not allow_pending:
        report = _json(path / "reports/validation_report.json")
        if (
            protocol.get("status") != "PASS"
            or leakage.get("status") != "PASS"
            or report.get("status") != "PASS"
            or report.get("checks") != checks
        ):
            raise ProtocolError("Robust-interaction final validation failed.")
    return checks


def _validate_metrics(
    predictions: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    config: RobustInteractionConfig,
) -> None:
    for family in ("robust_nystroem", "bilinear"):
        for center in config.heldout_centers:
            rows = [
                row
                for row in predictions
                if row["family"] == family and row["outer_center"] == center
            ]
            metrics = binary_metrics(
                np.asarray([int(row["y_true"]) for row in rows]),
                np.asarray([int(row["y_pred"]) for row in rows]),
            )
            comparison = next(
                row
                for row in comparisons
                if row["family"] == family
                and row["outer_center"] == center
                and int(row["seed"]) == config.primary_seed
            )
            if not np.isclose(float(comparison["bacc"]), metrics["bacc"]):
                raise ProtocolError("Robust-interaction metric reconstruction failed.")


def _validate_content(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    if stable_hash({key: value for key, value in payload.items() if key != "content_hash"}) != payload.get("content_hash"):
        raise ProtocolError("Robust-interaction content hash drifted.")
    observed = set()
    for row in payload["files"]:
        member = root / row["path"]
        if not member.is_file() or _sha256(member) != row["sha256"]:
            raise ProtocolError(f"Robust-interaction member drifted: {row['path']}.")
        observed.add(row["path"])
    expected = {
        str(member.relative_to(root))
        for member in root.rglob("*")
        if member.is_file() and member.name != "content_index.json"
    }
    if observed != expected:
        raise ProtocolError("Robust-interaction content coverage drifted.")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
