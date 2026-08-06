"""Independent validator for the Stage-70 descriptive test cache."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    PURPOSE,
    TargetEvaluationReservation,
    TargetEvaluationRow,
    semantic_sha256,
)
from midogpp_thesis.data.contract.stage70_target_evaluation.validation import (
    validate_target_evaluation_reservation,
    validate_target_evaluation_reservation_against_manifest,
)

from .config import (
    Stage70TestCacheConfig,
    stage70_cache_config_protocol,
    validate_stage70_test_cache_config,
)
from .contracts import (
    CACHE_ARTIFACT_ID,
    CACHE_NAME,
    CACHE_SCHEMA_VERSION,
    FEATURE_DIM,
    FIXED_WINDOW_START,
    FORBIDDEN_METADATA_FIELDS,
    LEGACY_OUTCOME_PATTERN,
    POOLING_ID,
    REPRESENTATION_ID,
    SHARD_METADATA_FIELDS,
    Stage70TestCacheError,
    stage70_extractor_protocol,
    stage70_extractor_protocol_hash,
    validate_model_identity,
)
from .io import (
    ValidatedStage70TestCache,
    file_sha256,
    load_stage70_center_shard,
    read_json,
    validate_content_index,
)
from .reservation_binding import resolve_reservation_artifact_binding


BASE_REQUIRED_FILES = frozenset(
    {
        "manifests/frozen_build_protocol.json",
        "manifests/row_alignment.json",
        "manifests/content_index.json",
        "reports/cache_builder_report.json",
        "reports/validation_report.json",
    }
)
CANONICAL_SHARD_FILES = tuple(
    f"embeddings/by_center/center_{center}.pt" for center in ELIGIBLE_CENTERS
)
REQUIRED_FILES = tuple(sorted(BASE_REQUIRED_FILES.union(CANONICAL_SHARD_FILES)))
CACHE_REQUIRED_FILES = REQUIRED_FILES
PENDING_REQUIRED_FILES = tuple(
    relative for relative in REQUIRED_FILES if relative != "reports/validation_report.json"
)


def validate_stage70_test_cache(
    root: str | Path,
    *,
    expected_config: Stage70TestCacheConfig | None = None,
    expected_reservation: TargetEvaluationReservation | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct row, model, protocol, shard, and content-index claims."""

    cache_root = Path(root)
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise Stage70TestCacheError(
            f"Stage-70 test-cache root is missing or unsafe: {cache_root}."
        )
    canonical_validation = expected_config is None
    if expected_config is not None:
        validate_stage70_test_cache_config(
            expected_config,
            expected_reservation=expected_reservation,
        )
        centers = expected_config.eligible_centers
        expected_counts = {
            str(key): int(value)
            for key, value in expected_config.expected_rows_by_center.items()
        }
        expected_manifest_sha256 = expected_config.expected_manifest_sha256
        fixture = not expected_config.canonical_coverage_required
    else:
        centers = ELIGIBLE_CENTERS
        expected_counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
        expected_manifest_sha256 = CANONICAL_MANIFEST_SHA256
        fixture = False

    required = set(BASE_REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    required.update(
        f"embeddings/by_center/center_{center}.pt" for center in centers
    )
    actual = {
        str(path.relative_to(cache_root))
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    if actual != required or any(
        (cache_root / relative).is_symlink() for relative in actual
    ):
        raise Stage70TestCacheError(
            "Stage-70 test cache is incomplete or has an unexpected member set: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}."
        )
    content = validate_content_index(cache_root)
    frozen = read_json(cache_root / "manifests" / "frozen_build_protocol.json")
    alignment = read_json(cache_root / "manifests" / "row_alignment.json")
    report = read_json(cache_root / "reports" / "cache_builder_report.json")
    scan_cache_payload(frozen, role="frozen protocol")
    scan_cache_payload(alignment, role="row alignment")
    scan_cache_payload(report, role="builder report")

    _validate_frozen_protocol(
        frozen,
        expected_config=expected_config,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_counts=expected_counts,
        fixture=fixture,
    )
    _validate_builder_report(
        report,
        frozen=frozen,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_counts=expected_counts,
        allow_pending=allow_pending,
    )
    _validate_alignment_header(
        alignment,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_counts=expected_counts,
        centers=centers,
    )

    observed_rows: list[TargetEvaluationRow] = []
    shard_hashes: dict[str, str] = {}
    observed_model_identity: dict[str, object] | None = None
    for center in centers:
        shard_path = cache_root / "embeddings" / "by_center" / f"center_{center}.pt"
        shard = load_stage70_center_shard(shard_path, expected_center=center)
        expected_count = expected_counts[center]
        if len(shard.metadata) != expected_count:
            raise Stage70TestCacheError(
                f"Stage-70 cache row count drifted for center {center}."
            )
        _validate_feature_extractor(
            shard.feature_extractor,
            frozen=frozen,
            expected_config=expected_config,
        )
        model_identity = validate_model_identity(
            shard.feature_extractor.get("model_identity", {})
        )
        if observed_model_identity is None:
            observed_model_identity = model_identity
        elif semantic_sha256(observed_model_identity) != semantic_sha256(model_identity):
            raise Stage70TestCacheError(
                "Stage-70 cache model identity differs between center shards."
            )
        center_rows = tuple(_target_row(row) for row in shard.metadata)
        observed_rows.extend(center_rows)
        shard_hashes[center] = shard.shard_sha256
        centers_payload = _mapping(alignment.get("centers"), "centers")
        center_record = _mapping(
            centers_payload.get(center),
            f"center {center} alignment",
        )
        if (
            center_record.get("relative_member")
            != f"embeddings/by_center/center_{center}.pt"
            or center_record.get("sha256") != shard.shard_sha256
            or center_record.get("row_count") != expected_count
            or center_record.get("row_order_hash")
            != semantic_sha256(list(shard.evaluation_row_ids))
            or center_record.get("first_contract_row_index")
            != shard.contract_row_indices[0]
            or center_record.get("last_contract_row_index")
            != shard.contract_row_indices[-1]
        ):
            raise Stage70TestCacheError(
                f"Stage-70 cache shard/alignment drifted for center {center}."
            )

    if observed_model_identity is None:
        raise Stage70TestCacheError("Stage-70 cache has no model identity.")
    manifest_order_rows = tuple(
        sorted(observed_rows, key=lambda row: row.contract_row_index)
    )
    indices = [row.contract_row_index for row in manifest_order_rows]
    row_ids = [row.evaluation_row_id for row in manifest_order_rows]
    if len(indices) != len(set(indices)) or len(row_ids) != len(set(row_ids)):
        raise Stage70TestCacheError(
            "Stage-70 cache contains duplicated global row identities."
        )
    reconstructed = TargetEvaluationReservation(
        manifest_sha256=expected_manifest_sha256,
        protocol_hash=str(alignment.get("target_evaluation_reservation_protocol_hash", "")),
        reservation_id=str(alignment.get("target_evaluation_reservation_id", "")),
        rows=manifest_order_rows,
        coverage_scope="test_fixture_only" if fixture else "canonical",
    )
    validate_target_evaluation_reservation(
        reconstructed,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows_by_center=expected_counts,
        allow_test_fixture=fixture,
    )
    if expected_reservation is not None and reconstructed != expected_reservation:
        raise Stage70TestCacheError(
            "Stage-70 cache rows differ from the authorized reservation."
        )
    if expected_config is not None:
        validate_target_evaluation_reservation_against_manifest(
            expected_config.manifest_path,
            reconstructed,
            expected_rows_by_center=expected_counts,
            allow_test_fixture=fixture,
        )

    _validate_reservation_artifact_binding(
        frozen,
        report=report,
        reservation=reconstructed,
        expected_config=expected_config,
        fixture=fixture,
    )

    center_grouped_ids = [row.evaluation_row_id for row in observed_rows]
    if (
        alignment.get("row_order_hash") != reconstructed.row_order_hash
        or alignment.get("center_grouped_row_order_hash")
        != semantic_sha256(center_grouped_ids)
        or report.get("row_order_hash") != reconstructed.row_order_hash
        or report.get("shard_sha256_by_center") != shard_hashes
        or semantic_sha256(report.get("model_identity"))
        != semantic_sha256(observed_model_identity)
    ):
        raise Stage70TestCacheError(
            "Stage-70 cache row/model/hash report drifted."
        )

    summary: dict[str, object] = {
        "status": "PASS",
        "manifest_sha256": expected_manifest_sha256,
        "target_evaluation_reservation_id": reconstructed.reservation_id,
        "target_evaluation_reservation_protocol_hash": reconstructed.protocol_hash,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "row_count": reconstructed.row_count,
        "rows_by_center": reconstructed.rows_by_center,
        "row_order_hash": reconstructed.row_order_hash,
        "shard_sha256_by_center": shard_hashes,
        "content_hash": str(content.get("content_hash", "")),
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
    }
    if canonical_validation and reconstructed.row_count != EXPECTED_TEST_ROWS:
        raise Stage70TestCacheError(
            "Public Stage-70 cache validation requires canonical row coverage."
        )
    if not allow_pending:
        validation_report = read_json(cache_root / "reports" / "validation_report.json")
        scan_cache_payload(validation_report, role="validation report")
        durable_summary = {
            key: value for key, value in summary.items() if key != "content_hash"
        }
        if (
            validation_report.get("schema_version")
            != "midogpp_stage70_descriptive_test_cache_validation_v1"
            or validation_report.get("status") != "PASS"
            or validation_report.get("validator") != "validate_stage70_test_cache"
            or validation_report.get("checks") != durable_summary
        ):
            raise Stage70TestCacheError(
                "Stage-70 cache validation report does not reproduce validator checks."
            )
    return summary


def load_validated_stage70_test_cache(
    root: str | Path,
    *,
    expected_config: Stage70TestCacheConfig | None = None,
    expected_reservation: TargetEvaluationReservation | None = None,
) -> ValidatedStage70TestCache:
    """Return a read-only center loader only after full independent validation."""

    summary = validate_stage70_test_cache(
        root,
        expected_config=expected_config,
        expected_reservation=expected_reservation,
    )
    return ValidatedStage70TestCache(root=Path(root), summary=summary)


def scan_cache_payload(payload: object, *, role: str) -> None:
    """Reject field and string encodings forbidden from the cache boundary."""

    def visit(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key)
                if key.casefold() in FORBIDDEN_METADATA_FIELDS:
                    raise Stage70TestCacheError(
                        f"Stage-70 {role} contains a forbidden field at {location}."
                    )
                visit(nested, f"{location}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{location}[{index}]")
            return
        if isinstance(value, str) and LEGACY_OUTCOME_PATTERN.search(value):
            raise Stage70TestCacheError(
                f"Stage-70 {role} contains a legacy outcome encoding at {location}."
            )

    visit(payload, role)


def _validate_frozen_protocol(
    frozen: Mapping[str, object],
    *,
    expected_config: Stage70TestCacheConfig | None,
    expected_manifest_sha256: str,
    expected_counts: Mapping[str, int],
    fixture: bool,
) -> None:
    observed_hash = frozen.get("frozen_build_protocol_hash")
    unhashed = {
        key: value for key, value in frozen.items() if key != "frozen_build_protocol_hash"
    }
    if observed_hash != semantic_sha256(unhashed):
        raise Stage70TestCacheError(
            "Stage-70 frozen build-protocol hash drifted."
        )
    required = {
        "schema_version": "midogpp_stage70_descriptive_test_frozen_build_protocol_v1",
        "cache_name": CACHE_NAME,
        "cache_artifact_id": CACHE_ARTIFACT_ID,
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "scoring_manifest_sha256": expected_manifest_sha256,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "eligible_centers": list(expected_counts),
        "expected_row_count": sum(expected_counts.values()),
        "expected_rows_by_center": dict(expected_counts),
        "coverage_scope": "test_fixture_only" if fixture else "canonical",
        "cache_extractor_protocol": stage70_extractor_protocol(),
        "shard_metadata_fields": sorted(SHARD_METADATA_FIELDS),
        "outcome_access_during_extraction": "closed",
        "metric_computation": "absent",
    }
    drift = [key for key, value in required.items() if frozen.get(key) != value]
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 frozen build protocol drifted: {drift}."
        )
    _mapping(
        frozen.get("reservation_artifact_binding"),
        "reservation_artifact_binding",
    )
    if expected_config is not None and (
        frozen.get("config_protocol_hash") != expected_config.config_protocol_hash
        or {
            key: frozen.get(key)
            for key in stage70_cache_config_protocol(expected_config)
        }
        != stage70_cache_config_protocol(expected_config)
    ):
        raise Stage70TestCacheError(
            "Stage-70 frozen config-protocol binding drifted."
        )


def _validate_reservation_artifact_binding(
    frozen: Mapping[str, object],
    *,
    report: Mapping[str, object],
    reservation: TargetEvaluationReservation,
    expected_config: Stage70TestCacheConfig | None,
    fixture: bool,
) -> None:
    """Reconstruct the immutable reservation evidence bound into the cache."""

    observed = dict(
        _mapping(
            frozen.get("reservation_artifact_binding"),
            "reservation_artifact_binding",
        )
    )
    target_rows = [
        {
            "evaluation_row_id": row.evaluation_row_id,
            "contract_row_index": row.contract_row_index,
            "target_center": row.center,
            "split": row.split,
        }
        for row in reservation.rows
    ]
    required = {
        "scoring_manifest_sha256": reservation.manifest_sha256,
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "target_identity_table_hash": stable_hash(target_rows),
        "row_count": reservation.row_count,
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
    }
    expected_fields = {
        "reservation_artifact_content_hash",
        "reservation_identity_lock_hash",
        "target_identity_table_hash",
        "reservation_validation_report_sha256",
        "scoring_manifest_sha256",
        "target_evaluation_reservation_id",
        "target_evaluation_reservation_protocol_hash",
        "cache_extractor_protocol_hash",
        "row_count",
        "reservation_artifact_validation_status",
        "purpose",
        "fresh_evidence",
    }
    drift = [key for key, value in required.items() if observed.get(key) != value]
    if set(observed) != expected_fields or drift:
        raise Stage70TestCacheError(
            "Stage-70 cache reservation-artifact binding drifted: "
            f"fields={sorted(set(observed) ^ expected_fields)}, values={drift}."
        )

    if expected_config is not None:
        resolved = resolve_reservation_artifact_binding(
            expected_config.reservation_path,
            reservation=reservation,
            expected_cache_extractor_protocol_hash=(
                expected_config.expected_cache_extractor_protocol_hash
            ),
            allow_test_fixture=fixture,
        )
        if observed != resolved.to_dict():
            raise Stage70TestCacheError(
                "Stage-70 cache reservation-artifact evidence no longer validates."
            )
    else:
        if (
            observed.get("reservation_artifact_validation_status") != "PASS"
            or not _is_lower_hex(
                observed.get("reservation_artifact_content_hash"), length=16
            )
            or not _is_lower_hex(
                observed.get("reservation_identity_lock_hash"), length=16
            )
            or not _is_lower_hex(
                observed.get("target_identity_table_hash"), length=16
            )
            or not _is_lower_hex(
                observed.get("reservation_validation_report_sha256"), length=64
            )
        ):
            raise Stage70TestCacheError(
                "Public Stage-70 cache validation requires PASS reservation evidence."
            )

    flattened = {key: report.get(key) for key in expected_fields}
    if flattened != observed:
        raise Stage70TestCacheError(
            "Stage-70 builder report/reservation-artifact binding drifted."
        )


def _validate_builder_report(
    report: Mapping[str, object],
    *,
    frozen: Mapping[str, object],
    expected_manifest_sha256: str,
    expected_counts: Mapping[str, int],
    allow_pending: bool,
) -> None:
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    required = {
        "schema_version": "midogpp_stage70_descriptive_test_cache_builder_v1",
        "status": expected_status,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "representation_id": REPRESENTATION_ID,
        "pooling": POOLING_ID,
        "feature_dim": FEATURE_DIM,
        "split": EVALUATION_SPLIT,
        "row_count": sum(expected_counts.values()),
        "rows_by_center": dict(expected_counts),
        "manifest_sha256": expected_manifest_sha256,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "evidence_status": "previously_consumed_test",
        "allowed_use": "descriptive_locked_model_scoring_only",
        "outcome_access_during_extraction": "closed",
        "metric_computation": "absent",
    }
    drift = [key for key, value in required.items() if report.get(key) != value]
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 cache builder report drifted: {drift}."
        )
    validate_model_identity(_mapping(report.get("model_identity"), "model_identity"))
    if not allow_pending and report.get("independent_validation_status") != "PASS":
        raise Stage70TestCacheError(
            "Stage-70 cache lacks independent-validation completion."
        )


def _validate_alignment_header(
    alignment: Mapping[str, object],
    *,
    expected_manifest_sha256: str,
    expected_counts: Mapping[str, int],
    centers: Sequence[str],
) -> None:
    required = {
        "schema_version": "midogpp_stage70_descriptive_test_row_alignment_v1",
        "status": "PASS",
        "split": EVALUATION_SPLIT,
        "row_count": sum(expected_counts.values()),
        "rows_by_center": dict(expected_counts),
        "eligible_centers": list(centers),
        "excluded_centers": ["4"],
        "excluded_center_present": False,
        "manifest_sha256": expected_manifest_sha256,
    }
    drift = [key for key, value in required.items() if alignment.get(key) != value]
    centers_payload = alignment.get("centers")
    if not isinstance(centers_payload, Mapping) or set(centers_payload) != set(centers):
        drift.append("centers")
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 cache row-alignment header drifted: {drift}."
        )


def _validate_feature_extractor(
    extractor: Mapping[str, object],
    *,
    frozen: Mapping[str, object],
    expected_config: Stage70TestCacheConfig | None,
) -> None:
    protocol = stage70_extractor_protocol()
    required = {
        **protocol,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "frozen_build_protocol_hash": frozen.get("frozen_build_protocol_hash"),
    }
    # The actual pinned identity may additionally retain preprocessing detail.
    required.pop("model_identity", None)
    drift = [key for key, value in required.items() if extractor.get(key) != value]
    if expected_config is not None and extractor.get(
        "config_protocol_hash"
    ) != expected_config.config_protocol_hash:
        drift.append("config_protocol_hash")
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 cache feature-extractor protocol drifted: {drift}."
        )
    validate_model_identity(
        _mapping(extractor.get("model_identity"), "model_identity")
    )


def _target_row(metadata: Mapping[str, object]) -> TargetEvaluationRow:
    return TargetEvaluationRow(
        evaluation_row_id=str(metadata["evaluation_row_id"]),
        contract_row_index=int(metadata["contract_row_index"]),
        case_id=str(metadata["case_id"]),
        center=str(metadata["center"]),
        split=str(metadata["split"]),
    )


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage70TestCacheError(
            f"Stage-70 cache {role} must be a mapping."
        )
    return value


def _is_lower_hex(value: object, *, length: int) -> bool:
    rendered = str(value)
    return len(rendered) == length and all(
        character in "0123456789abcdef" for character in rendered
    )


__all__ = (
    "BASE_REQUIRED_FILES",
    "CACHE_REQUIRED_FILES",
    "CANONICAL_SHARD_FILES",
    "PENDING_REQUIRED_FILES",
    "REQUIRED_FILES",
    "load_validated_stage70_test_cache",
    "scan_cache_payload",
    "validate_stage70_test_cache",
)
