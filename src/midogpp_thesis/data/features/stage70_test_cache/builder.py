"""Atomic, label-sealed builder for the descriptive Stage-70 test cache."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import random
import time
from typing import Callable, Mapping

from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.contract.stage70_target_evaluation import (
    TargetEvaluationReservation,
    iter_bound_image_bytes,
    project_target_evaluation_manifest,
    validate_target_evaluation_reservation_against_manifest,
)

from .config import (
    Stage70TestCacheConfig,
    stage70_cache_config_protocol,
    validate_stage70_test_cache_config,
)
from .contracts import (
    CACHE_SCHEMA_VERSION,
    CHECKPOINT_FILE_SHA256,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    FEATURE_DIM,
    FEATURE_EXTRACTOR_SCHEMA_VERSION,
    FIXED_WINDOW_START,
    FRESH_EVIDENCE,
    MODEL_CONFIG_SHA256,
    MODEL_REF,
    MODEL_REVISION,
    POOLING_ID,
    PREPROCESSING_CONFIG_HASH,
    PURPOSE,
    REPRESENTATION_ID,
    SHARD_METADATA_FIELDS,
    STATE_DICT_SHA256,
    Stage70TestCacheError,
    stage70_extractor_protocol,
    stage70_extractor_protocol_hash,
    validate_model_identity,
)
from .io import (
    file_sha256,
    metadata_from_row,
    read_json,
    write_content_index,
    write_json,
    write_stage70_center_shard,
)
from .reservation_binding import (
    ReservationArtifactBinding,
    resolve_reservation_artifact_binding,
)


def build_stage70_test_cache(
    config: Stage70TestCacheConfig,
    *,
    reservation: TargetEvaluationReservation | None = None,
    extractor_factory: Callable[..., object] | None = None,
    image_reader: Callable[[Path], bytes] | None = None,
    access_log: object = None,
) -> Path:
    """Build, independently validate, and atomically publish one cache.

    A fake ``extractor_factory`` and a fixture-scoped config can exercise the
    complete build transaction with a tiny manifest.  Production defaults use
    the pinned Virchow2 extractor and canonical 9,928-row reservation.
    """

    resolved_reservation = _load_or_project_reservation(config, reservation)
    validate_stage70_test_cache_config(
        config,
        expected_reservation=resolved_reservation,
    )
    config.cache_root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(config.cache_root) as stage:
        staging_config = replace(config, cache_root=stage)
        _build_in_place(
            staging_config,
            resolved_reservation,
            extractor_factory=extractor_factory,
            image_reader=image_reader,
            access_log=access_log,
        )
        from .validation import validate_stage70_test_cache

        validate_stage70_test_cache(
            stage,
            expected_config=staging_config,
            expected_reservation=resolved_reservation,
            allow_pending=True,
        )
        _finalize_cache(staging_config, resolved_reservation)
        validate_stage70_test_cache(
            stage,
            expected_config=staging_config,
            expected_reservation=resolved_reservation,
        )
    return config.cache_root


def _build_in_place(
    config: Stage70TestCacheConfig,
    reservation: TargetEvaluationReservation,
    *,
    extractor_factory: Callable[..., object] | None,
    image_reader: Callable[[Path], bytes] | None,
    access_log: object,
) -> None:
    started = time.perf_counter()
    if any(config.cache_root.iterdir()):
        raise FileExistsError(
            f"Stage-70 cache staging directory is not empty: {config.cache_root}."
        )
    validate_stage70_test_cache_config(config, expected_reservation=reservation)
    validate_target_evaluation_reservation_against_manifest(
        config.manifest_path,
        reservation,
        expected_rows_by_center=config.expected_rows_by_center,
        allow_test_fixture=not config.canonical_coverage_required,
    )
    reservation_artifact = resolve_reservation_artifact_binding(
        config.reservation_path,
        reservation=reservation,
        expected_cache_extractor_protocol_hash=(
            config.expected_cache_extractor_protocol_hash
        ),
        allow_test_fixture=not config.canonical_coverage_required,
    )

    frozen = _frozen_build_protocol(config, reservation, reservation_artifact)
    # The lock is materialized before model construction and before any source
    # location is opened.  It contains only neutral reservation identities.
    write_json(
        config.cache_root / "manifests" / "frozen_build_protocol.json",
        frozen,
    )

    factory = extractor_factory
    if factory is None:
        from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor

        factory = Virchow2TokenExtractor
    extractor = factory(
        model_ref=MODEL_REF,
        model_revision=MODEL_REVISION,
        device=config.device,
        expected_model_config_sha256=MODEL_CONFIG_SHA256,
        expected_checkpoint_file_sha256=CHECKPOINT_FILE_SHA256,
        expected_state_dict_sha256=STATE_DICT_SHA256,
        expected_preprocessing_config_hash=PREPROCESSING_CONFIG_HASH,
    )
    extractor_identity = validate_model_identity(
        getattr(extractor, "identity", None)  # type: ignore[arg-type]
    )
    _seed(config.experiment_seed)

    grouped_embeddings: dict[str, list[object]] = {
        center: [] for center in config.eligible_centers
    }
    grouped_rows = {
        center: tuple(row for row in reservation.rows if row.center == center)
        for center in config.eligible_centers
    }
    pending_rows = []
    pending_images = []

    def flush() -> None:
        if not pending_rows:
            return
        try:
            extracted = getattr(extractor, "extract_spatial_windows")(
                pending_images,
                window_starts=[FIXED_WINDOW_START] * len(pending_images),
            )
            import torch

            tensor = torch.as_tensor(extracted).detach().cpu().float()
            if tuple(tensor.shape) != (len(pending_rows), FEATURE_DIM) or not bool(
                torch.isfinite(tensor).all()
            ):
                raise Stage70TestCacheError(
                    "Stage-70 pinned extraction returned invalid geometry/finiteness."
                )
            for batch_index, row in enumerate(pending_rows):
                grouped_embeddings[row.center].append(tensor[batch_index : batch_index + 1])
        finally:
            for image in pending_images:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            pending_rows.clear()
            pending_images.clear()

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Stage-70 raw-JPEG extraction requires Pillow.") from exc
    for opaque in iter_bound_image_bytes(
        config.manifest_path,
        reservation,
        repo_root=config.repo_root,
        image_reader=image_reader,
        access_log=access_log,  # type: ignore[arg-type]
        allow_test_fixture=not config.canonical_coverage_required,
    ):
        try:
            with Image.open(BytesIO(opaque.jpeg_bytes)) as source:
                if source.format != "JPEG":
                    raise Stage70TestCacheError(
                        "Stage-70 opaque source bytes are not a raw JPEG at contract row "
                        f"{opaque.row.contract_row_index}."
                    )
                image = source.convert("RGB")
        except Stage70TestCacheError:
            raise
        except Exception as exc:
            raise Stage70TestCacheError(
                "Stage-70 opaque JPEG decoding failed at contract row "
                f"{opaque.row.contract_row_index}."
            ) from exc
        pending_rows.append(opaque.row)
        pending_images.append(image)
        if len(pending_rows) == config.batch_size:
            flush()
    flush()

    import torch

    feature_extractor = {
        **stage70_extractor_protocol(),
        "model_identity": extractor_identity,
        "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
        "frozen_build_protocol_hash": frozen["frozen_build_protocol_hash"],
        "config_protocol_hash": config.config_protocol_hash,
    }
    center_alignment: dict[str, object] = {}
    shard_hashes: dict[str, str] = {}
    for center in config.eligible_centers:
        rows = grouped_rows[center]
        chunks = grouped_embeddings[center]
        if len(chunks) != len(rows) or not chunks:
            raise Stage70TestCacheError(
                f"Stage-70 extracted center coverage drifted for center {center}."
            )
        embeddings = torch.cat(tuple(chunks), dim=0)
        shard_path = (
            config.cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
        write_stage70_center_shard(
            shard_path,
            embeddings=embeddings,
            metadata=[metadata_from_row(row) for row in rows],
            feature_extractor=feature_extractor,
        )
        shard_hash = file_sha256(shard_path)
        shard_hashes[center] = shard_hash
        center_alignment[center] = {
            "relative_member": str(shard_path.relative_to(config.cache_root)),
            "sha256": shard_hash,
            "row_count": len(rows),
            "row_order_hash": _row_order_hash(rows),
            "first_contract_row_index": rows[0].contract_row_index,
            "last_contract_row_index": rows[-1].contract_row_index,
        }

    globally_ordered_ids = [row.evaluation_row_id for row in reservation.rows]
    center_grouped_ids = [
        row.evaluation_row_id
        for center in config.eligible_centers
        for row in grouped_rows[center]
    ]
    write_json(
        config.cache_root / "manifests" / "row_alignment.json",
        {
            "schema_version": "midogpp_stage70_descriptive_test_row_alignment_v1",
            "status": "PASS",
            "split": EVALUATION_SPLIT,
            "row_count": reservation.row_count,
            "rows_by_center": dict(config.expected_rows_by_center),
            "eligible_centers": list(config.eligible_centers),
            "excluded_centers": ["4"],
            "excluded_center_present": False,
            "manifest_sha256": reservation.manifest_sha256,
            "target_evaluation_reservation_id": reservation.reservation_id,
            "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
            "row_order_hash": reservation.row_order_hash,
            "center_grouped_row_order_hash": _semantic_hash(center_grouped_ids),
            "centers": center_alignment,
        },
    )
    write_json(
        config.cache_root / "reports" / "cache_builder_report.json",
        {
            "schema_version": "midogpp_stage70_descriptive_test_cache_builder_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "representation_id": REPRESENTATION_ID,
            "pooling": POOLING_ID,
            "feature_dim": FEATURE_DIM,
            "split": EVALUATION_SPLIT,
            "row_count": reservation.row_count,
            "rows_by_center": dict(config.expected_rows_by_center),
            "row_order_hash": _semantic_hash(globally_ordered_ids),
            "manifest_sha256": reservation.manifest_sha256,
            "target_evaluation_reservation_id": reservation.reservation_id,
            "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
            "cache_extractor_protocol_hash": stage70_extractor_protocol_hash(),
            **reservation_artifact.to_dict(),
            "model_identity": extractor_identity,
            "shard_sha256_by_center": shard_hashes,
            "purpose": PURPOSE,
            "fresh_evidence": FRESH_EVIDENCE,
            "evidence_status": "previously_consumed_test",
            "allowed_use": "descriptive_locked_model_scoring_only",
            "outcome_access_during_extraction": "closed",
            "metric_computation": "absent",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_content_index(config.cache_root)


def _load_or_project_reservation(
    config: Stage70TestCacheConfig,
    reservation: TargetEvaluationReservation | None,
) -> TargetEvaluationReservation:
    if reservation is not None:
        resolved = reservation
    else:
        # The configured reservation path is an immutable authorization
        # artifact root, never a serialized row source.  Reconstruct case
        # identity in memory from the hash-bound manifest, then bind its opaque
        # identities to the independently validated artifact root.
        resolved = project_target_evaluation_manifest(
            config.manifest_path,
            expected_manifest_sha256=config.expected_manifest_sha256,
            expected_rows_by_center=config.expected_rows_by_center,
            allow_test_fixture=not config.canonical_coverage_required,
        )
    validate_stage70_test_cache_config(config, expected_reservation=resolved)
    return resolved


def _frozen_build_protocol(
    config: Stage70TestCacheConfig,
    reservation: TargetEvaluationReservation,
    reservation_artifact: ReservationArtifactBinding,
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_stage70_descriptive_test_frozen_build_protocol_v1",
        **stage70_cache_config_protocol(config),
        "config_protocol_hash": config.config_protocol_hash,
        "cache_extractor_protocol": stage70_extractor_protocol(),
        "target_evaluation_row_order_hash": reservation.row_order_hash,
        "reservation_artifact_binding": reservation_artifact.to_dict(),
        "shard_metadata_fields": sorted(SHARD_METADATA_FIELDS),
        "source_location_lifetime": "bound_row_read_then_discard",
        "outcome_access_during_extraction": "closed",
        "metric_computation": "absent",
    }
    payload["frozen_build_protocol_hash"] = _semantic_hash(payload)
    return payload


def _finalize_cache(
    config: Stage70TestCacheConfig,
    reservation: TargetEvaluationReservation,
) -> None:
    from .validation import validate_stage70_test_cache

    checks = validate_stage70_test_cache(
        config.cache_root,
        expected_config=config,
        expected_reservation=reservation,
        allow_pending=True,
    )
    report_path = config.cache_root / "reports" / "cache_builder_report.json"
    report = read_json(report_path)
    report["status"] = "PASS"
    report["independent_validation_status"] = "PASS"
    write_json(report_path, report)
    durable_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"content_hash"}
    }
    write_json(
        config.cache_root / "reports" / "validation_report.json",
        {
            "schema_version": "midogpp_stage70_descriptive_test_cache_validation_v1",
            "status": "PASS",
            "validator": "validate_stage70_test_cache",
            "checks": durable_checks,
        },
    )
    write_content_index(config.cache_root)


def _row_order_hash(rows: object) -> str:
    return _semantic_hash([row.evaluation_row_id for row in rows])


def _semantic_hash(payload: object) -> str:
    from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
        semantic_sha256,
    )

    return semantic_sha256(payload)


def _seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ("build_stage70_test_cache",)
