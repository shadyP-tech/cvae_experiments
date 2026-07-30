"""Independent validation for the reviewed Uniform-B canonical reference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..schemas.matched_reference import assert_matched_reference_artifacts
from .config import (
    CONFIRMATION_SUMMARY_SHA256,
    EXPERIMENT_NAME,
    PROMOTION_REVIEW_ID,
    REPRESENTATION_ID,
    UniformBCanonicalReferenceConfig,
)


REQUIRED = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/uniform_b_canonical_representation_lock.json",
    "manifests/promotion_review_snapshot.json",
    "manifests/content_index.json",
    "reports/leakage_provenance_report.json",
    "reports/promotion_decision.json",
    "reports/promotion_report.md",
    "reports/test_consumption_ledger.json",
    "tables/source_inner_classifier_tuning.csv",
    "tables/classifier_tuned_source_results.csv",
    "tables/classifier_tuned_predictions.csv",
)


def validate_uniform_b_canonical_reference_bundle(
    root: str | Path,
    *,
    config: UniformBCanonicalReferenceConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B canonical reference is incomplete: {missing}.")
    assert_matched_reference_artifacts(path)
    protocol = _read_json(path / "manifests/protocol_manifest.json")
    lock = _read_json(path / "manifests/uniform_b_canonical_representation_lock.json")
    review = _read_json(path / "manifests/promotion_review_snapshot.json")
    decision = _read_json(path / "reports/promotion_decision.json")
    ledger = _read_json(path / "reports/test_consumption_ledger.json")
    if (
        protocol.get("experiment_name") != EXPERIMENT_NAME
        or protocol.get("claim_scope") != "real_feature_transfer_only"
        or lock.get("representation_id") != REPRESENTATION_ID
        or lock.get("confirmation_summary_sha256") != CONFIRMATION_SUMMARY_SHA256
        or lock.get("canonical_a_retained") is not True
        or lock.get("automatic_downstream_migration") is not False
        or review.get("review_id") != PROMOTION_REVIEW_ID
        or review.get("status") != "approved"
        or review.get("review_effect") != "authorizes_new_stage10_reference_only"
        or decision.get("decision") != "PROMOTED_AS_NEW_CANONICAL_REFERENCE"
        or decision.get("test_split_consumed_for_representation_adoption") is not True
        or decision.get("classifier_locks_imported_from_diagnostics") is not False
        or ledger.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or ledger.get("may_be_reused_as_fresh_representation_selection_evidence")
        is not False
    ):
        raise ProtocolError("Uniform-B canonical promotion boundary failed.")
    lock_unhashed = {
        key: value for key, value in lock.items() if key != "representation_lock_hash"
    }
    review_unhashed = {key: value for key, value in review.items() if key != "review_hash"}
    if (
        stable_hash(lock_unhashed) != lock.get("representation_lock_hash")
        or stable_hash(review_unhashed) != review.get("review_hash")
    ):
        raise ProtocolError("Uniform-B canonical promotion lock hash drifted.")
    if _sha256_file(
        config.confirmation_root / "reports/confirmation_summary.json"
    ) != CONFIRMATION_SUMMARY_SHA256:
        raise ProtocolError("Uniform-B canonical confirmation identity drifted.")
    _validate_content_index(path)
    results = _read_csv(path / "tables/classifier_tuned_source_results.csv")
    predictions = _read_csv(path / "tables/classifier_tuned_predictions.csv")
    tuning = _read_csv(path / "tables/source_inner_classifier_tuning.csv")
    checks = {
        "status": "PASS",
        "heldout_centers": len(results),
        "predictions": len(predictions),
        "tuning_rows": len(tuning),
        "representation_id": REPRESENTATION_ID,
        "canonical_a_retained": True,
        "automatic_downstream_migration": False,
    }
    if len(results) != 9 or len(tuning) != 90:
        raise ProtocolError("Uniform-B canonical reference coverage drifted.")
    if not allow_pending:
        validation = _read_json(path / "reports/validation_report.json")
        if validation.get("status") != "PASS" or validation.get("checks") != checks:
            raise ProtocolError("Uniform-B canonical validation report failed.")
    return checks


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if stable_hash(unhashed) != payload.get("content_hash"):
        raise ProtocolError("Uniform-B canonical content hash drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Uniform-B canonical content index is invalid.")
    expected = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Uniform-B canonical content-index row is invalid.")
        relative = str(row.get("path", ""))
        member = root / relative
        if not member.is_file() or _sha256_file(member) != row.get("sha256"):
            raise ProtocolError(f"Uniform-B canonical member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Uniform-B canonical content-index coverage drifted.")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Uniform-B canonical JSON must be an object: {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
