"""Cycle-free validation of the immutable Stage-70 reservation artifact.

The authorization package owns construction and its full validator.  Cache
construction needs a narrower capability: prove that the immutable artifact
root authorizes exactly the in-memory projector rows before model construction
or source-location access.  This reader validates the closed-world bundle,
content index, PASS report, identity lock, protocol lock, and neutral identity
table without importing the CVAE/authorization runtime.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    EVALUATION_SPLIT,
    FRESH_EVIDENCE,
    PURPOSE,
    TargetEvaluationReservation,
    semantic_sha256,
)

from .contracts import CACHE_ARTIFACT_ID, Stage70TestCacheError
from .io import file_sha256, read_json


RESERVATION_ARTIFACT_REQUIRED_FILES = (
    "config.resolved.yaml",
    "manifests/content_index.json",
    "manifests/evaluation_plan.json",
    "manifests/identity_lock.json",
    "manifests/input_binding.json",
    "manifests/protocol_manifest.json",
    "provenance/input_artifacts.json",
    "reports/authorization_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/target_identity.csv",
)
TARGET_IDENTITY_COLUMNS = (
    "evaluation_row_id",
    "contract_row_index",
    "target_center",
    "split",
)


@dataclass(frozen=True)
class ReservationArtifactBinding:
    artifact_root: Path
    content_hash: str
    identity_lock_hash: str
    target_identity_table_hash: str
    validation_report_sha256: str
    manifest_sha256: str
    reservation_id: str
    reservation_protocol_hash: str
    cache_extractor_protocol_hash: str
    row_count: int
    validation_status: str = "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "reservation_artifact_content_hash": self.content_hash,
            "reservation_identity_lock_hash": self.identity_lock_hash,
            "target_identity_table_hash": self.target_identity_table_hash,
            "reservation_validation_report_sha256": self.validation_report_sha256,
            "scoring_manifest_sha256": self.manifest_sha256,
            "target_evaluation_reservation_id": self.reservation_id,
            "target_evaluation_reservation_protocol_hash": (
                self.reservation_protocol_hash
            ),
            "cache_extractor_protocol_hash": self.cache_extractor_protocol_hash,
            "row_count": self.row_count,
            "reservation_artifact_validation_status": self.validation_status,
            "purpose": PURPOSE,
            "fresh_evidence": FRESH_EVIDENCE,
        }


def validate_reservation_artifact_binding(
    root: str | Path,
    *,
    reservation: TargetEvaluationReservation,
    expected_cache_extractor_protocol_hash: str,
) -> ReservationArtifactBinding:
    """Validate exact authorization evidence for one projected reservation."""

    artifact_root = Path(root)
    _assert_closed_world(artifact_root)
    content = _validate_content_index(artifact_root)
    identity = read_json(artifact_root / "manifests/identity_lock.json")
    protocol = read_json(artifact_root / "manifests/protocol_manifest.json")
    decision = read_json(artifact_root / "reports/authorization_decision.json")
    leakage = read_json(artifact_root / "reports/leakage_report.json")
    run_state = read_json(artifact_root / "reports/run_state.json")
    validation = read_json(artifact_root / "reports/validation_report.json")
    rows = _read_target_identity(artifact_root / "tables/target_identity.csv")
    expected_rows = [
        {
            "evaluation_row_id": row.evaluation_row_id,
            "contract_row_index": row.contract_row_index,
            "target_center": row.center,
            "split": row.split,
        }
        for row in reservation.rows
    ]
    if rows != expected_rows:
        raise Stage70TestCacheError(
            "Stage-70 reservation target-identity table differs from the projector."
        )
    target_identity_table_hash = stable_hash(rows)
    _validate_identity_lock(
        identity,
        reservation=reservation,
        expected_cache_extractor_protocol_hash=(
            expected_cache_extractor_protocol_hash
        ),
        target_identity_table_hash=target_identity_table_hash,
    )
    _validate_protocol_manifest(
        protocol,
        reservation=reservation,
        identity_lock_hash=str(identity["identity_lock_hash"]),
        expected_cache_extractor_protocol_hash=(
            expected_cache_extractor_protocol_hash
        ),
    )
    _validate_decision(decision)
    _validate_leakage(leakage)
    _validate_run_state(run_state)
    _validate_validation_report(
        validation,
        reservation=reservation,
        content_hash=str(content["content_hash"]),
        authorization_protocol_hash=str(protocol["protocol_hash"]),
        target_identity_table_hash=target_identity_table_hash,
    )
    return ReservationArtifactBinding(
        artifact_root=artifact_root,
        content_hash=str(content["content_hash"]),
        identity_lock_hash=str(identity["identity_lock_hash"]),
        target_identity_table_hash=target_identity_table_hash,
        validation_report_sha256=file_sha256(
            artifact_root / "reports/validation_report.json"
        ),
        manifest_sha256=reservation.manifest_sha256,
        reservation_id=reservation.reservation_id,
        reservation_protocol_hash=reservation.protocol_hash,
        cache_extractor_protocol_hash=expected_cache_extractor_protocol_hash,
        row_count=reservation.row_count,
    )


def resolve_reservation_artifact_binding(
    root: str | Path | None,
    *,
    reservation: TargetEvaluationReservation,
    expected_cache_extractor_protocol_hash: str,
    allow_test_fixture: bool = False,
) -> ReservationArtifactBinding:
    """Resolve production evidence, or an explicitly non-publishable test seam."""

    if root is not None:
        return validate_reservation_artifact_binding(
            root,
            reservation=reservation,
            expected_cache_extractor_protocol_hash=(
                expected_cache_extractor_protocol_hash
            ),
        )
    if not allow_test_fixture:
        raise Stage70TestCacheError(
            "Canonical Stage-70 cache construction requires a validated reservation artifact root."
        )
    fixture_identity = {
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "cache_extractor_protocol_hash": expected_cache_extractor_protocol_hash,
        "row_order_hash": reservation.row_order_hash,
    }
    return ReservationArtifactBinding(
        artifact_root=Path("<test-fixture-injected>"),
        content_hash=semantic_sha256({"fixture_reservation": fixture_identity}),
        identity_lock_hash=semantic_sha256({"fixture_identity": fixture_identity}),
        target_identity_table_hash=stable_hash(
            [
                {
                    "evaluation_row_id": row.evaluation_row_id,
                    "contract_row_index": row.contract_row_index,
                    "target_center": row.center,
                    "split": row.split,
                }
                for row in reservation.rows
            ]
        ),
        validation_report_sha256="0" * 64,
        manifest_sha256=reservation.manifest_sha256,
        reservation_id=reservation.reservation_id,
        reservation_protocol_hash=reservation.protocol_hash,
        cache_extractor_protocol_hash=expected_cache_extractor_protocol_hash,
        row_count=reservation.row_count,
        validation_status="TEST_FIXTURE_INJECTED",
    )


def _assert_closed_world(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise Stage70TestCacheError(
            "Stage-70 reservation artifact root is missing or unsafe."
        )
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise Stage70TestCacheError(
            "Stage-70 reservation artifact contains an unsafe symlink."
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(RESERVATION_ARTIFACT_REQUIRED_FILES)
    if actual != expected:
        raise Stage70TestCacheError(
            "Stage-70 reservation artifact closed-world coverage drifted: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _validate_content_index(root: Path) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "records", "content_hash"} or payload.get(
        "schema_version"
    ) != "midogpp_stage70_authorization_content_index_v1":
        raise Stage70TestCacheError(
            "Stage-70 reservation content-index schema drifted."
        )
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != stable_hash(unhashed):
        raise Stage70TestCacheError(
            "Stage-70 reservation content-index hash drifted."
        )
    records = payload.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise Stage70TestCacheError(
            "Stage-70 reservation content-index records are invalid."
        )
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    expected_members = set(RESERVATION_ARTIFACT_REQUIRED_FILES) - excluded
    observed_members: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise Stage70TestCacheError(
                "Stage-70 reservation content-index record schema drifted."
            )
        relative = str(record["relative_path"])
        member = root / relative
        if (
            relative in observed_members
            or relative not in expected_members
            or member.is_symlink()
            or not member.is_file()
            or record["sha256"] != file_sha256(member)
            or record["size_bytes"] != member.stat().st_size
        ):
            raise Stage70TestCacheError(
                f"Stage-70 reservation indexed member drifted: {relative}."
            )
        observed_members.add(relative)
    if observed_members != expected_members:
        raise Stage70TestCacheError(
            "Stage-70 reservation content-index member coverage drifted."
        )
    return payload


def _validate_identity_lock(
    identity: Mapping[str, object],
    *,
    reservation: TargetEvaluationReservation,
    expected_cache_extractor_protocol_hash: str,
    target_identity_table_hash: str,
) -> None:
    _assert_embedded_hash(identity, "identity_lock_hash")
    required = {
        "schema_version": "midogpp_stage70_target_identity_lock_v1",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "target_identity_table_hash": target_identity_table_hash,
        "row_count": reservation.row_count,
        "rows_by_center": reservation.rows_by_center,
        "split": EVALUATION_SPLIT,
        "opaque_evaluation_row_ids_only": True,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "target_label_values_persisted": False,
        "cache_artifact_id": CACHE_ARTIFACT_ID,
        "cache_extractor_protocol_hash": expected_cache_extractor_protocol_hash,
    }
    drift = [key for key, value in required.items() if identity.get(key) != value]
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 reservation identity-lock binding drifted: {drift}."
        )


def _validate_protocol_manifest(
    protocol: Mapping[str, object],
    *,
    reservation: TargetEvaluationReservation,
    identity_lock_hash: str,
    expected_cache_extractor_protocol_hash: str,
) -> None:
    _assert_embedded_hash(protocol, "protocol_hash")
    required = {
        "schema_version": "midogpp_stage70_target_evaluation_reservation_protocol_v1",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "descriptive_locked_model_scoring_allowed": True,
        "identity_lock_hash": identity_lock_hash,
        "cache_extractor_protocol_hash": expected_cache_extractor_protocol_hash,
        "target_labels_opened": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }
    drift = [key for key, value in required.items() if protocol.get(key) != value]
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 reservation protocol binding drifted: {drift}."
        )


def _validate_decision(decision: Mapping[str, object]) -> None:
    _assert_embedded_hash(decision, "decision_hash")
    required = {
        "schema_version": "midogpp_stage70_reservation_decision_v1",
        "status": "COMPLETE",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "cache_extraction_allowed": True,
        "prediction_allowed": False,
        "label_access_allowed": False,
        "metric_scoring_allowed": False,
        "generation_or_policy_refit_allowed": False,
    }
    if any(decision.get(key) != value for key, value in required.items()):
        raise Stage70TestCacheError(
            "Stage-70 reservation authorization decision drifted."
        )


def _validate_leakage(leakage: Mapping[str, object]) -> None:
    required = {
        "status": "PASS",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "previously_consumed_test_rows": True,
        "target_label_values_opened": False,
        "target_label_values_persisted": False,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "policy_or_seed_selection_performed": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }
    if any(leakage.get(key) != value for key, value in required.items()):
        raise Stage70TestCacheError(
            "Stage-70 reservation leakage report drifted."
        )


def _validate_run_state(run_state: Mapping[str, object]) -> None:
    required = {
        "schema_version": "midogpp_stage70_reservation_run_state_v1",
        "status": "COMPLETE",
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
    }
    if any(run_state.get(key) != value for key, value in required.items()):
        raise Stage70TestCacheError("Stage-70 reservation run state drifted.")


def _validate_validation_report(
    validation: Mapping[str, object],
    *,
    reservation: TargetEvaluationReservation,
    content_hash: str,
    authorization_protocol_hash: str,
    target_identity_table_hash: str,
) -> None:
    checks = validation.get("checks")
    if not isinstance(checks, Mapping):
        raise Stage70TestCacheError(
            "Stage-70 reservation validation report lacks checks."
        )
    required_report = {
        "schema_version": "midogpp_stage70_target_evaluation_reservation_validation_v1",
        "status": "PASS",
        "validator": "validate_target_evaluation_reservation",
    }
    required_checks = {
        "status": "PASS",
        "row_count": reservation.row_count,
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "target_identity_table_hash": target_identity_table_hash,
        "authorization_protocol_hash": authorization_protocol_hash,
        "content_hash": content_hash,
        "prediction_performed": False,
        "metric_scoring_performed": False,
        "target_labels_opened": False,
    }
    if any(validation.get(key) != value for key, value in required_report.items()) or any(
        checks.get(key) != value for key, value in required_checks.items()
    ):
        raise Stage70TestCacheError(
            "Stage-70 reservation validation report did not validate PASS for this reservation."
        )


def _read_target_identity(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != TARGET_IDENTITY_COLUMNS:
                raise Stage70TestCacheError(
                    "Stage-70 reservation target-identity schema drifted."
                )
            rows = [
                {
                    "evaluation_row_id": str(row["evaluation_row_id"]),
                    "contract_row_index": int(row["contract_row_index"]),
                    "target_center": str(row["target_center"]),
                    "split": str(row["split"]),
                }
                for row in reader
            ]
    except (OSError, TypeError, ValueError) as exc:
        raise Stage70TestCacheError(
            "Stage-70 reservation target-identity table is unreadable."
        ) from exc
    return rows


def _assert_embedded_hash(payload: Mapping[str, object], field: str) -> None:
    observed = payload.get(field)
    unhashed = {key: value for key, value in payload.items() if key != field}
    if observed != stable_hash(unhashed):
        raise Stage70TestCacheError(
            f"Stage-70 reservation embedded hash drifted: {field}."
        )


__all__ = (
    "RESERVATION_ARTIFACT_REQUIRED_FILES",
    "ReservationArtifactBinding",
    "TARGET_IDENTITY_COLUMNS",
    "resolve_reservation_artifact_binding",
    "validate_reservation_artifact_binding",
)
