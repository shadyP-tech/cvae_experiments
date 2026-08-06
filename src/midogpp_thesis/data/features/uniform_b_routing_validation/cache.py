"""Atomic builder for the immutable, unlabeled routing-validation cache."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Mapping, Sequence

from PIL import Image

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.contract.paths import resolve_contract_path
from midogpp_thesis.data.features.cache_io import CacheRows, load_cache_rows, write_center_shard
from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor
from midogpp_thesis.data.features.virchow2_tokens import (
    VIRCHOW2_TOKEN_LAYOUT,
    assert_preprocessing_spatial_identity,
)

from .config import (
    CANONICAL_A_DIM,
    CANONICAL_A_ID,
    CHECKPOINT_FILE_SHA256,
    ELIGIBLE_CENTERS,
    EXPECTED_CLASS_LABELS,
    EXPECTED_MANIFEST_ROWS_BY_SPLIT,
    EXPECTED_RUNTIME,
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_VALIDATION_ROWS_BY_CENTER,
    FEATURE_DIM,
    MODEL_CONFIG_SHA256,
    MODEL_REF,
    MODEL_REVISION,
    POOLING_ID,
    PREPROCESSING_CONFIG_HASH,
    REPRESENTATION_ID,
    ResolvedRoutingValidationCacheConfig,
    RoutingValidationCacheError,
    RoutingValidationCacheConfig,
    STATE_DICT_SHA256,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    resolve_routing_validation_cache_config,
    validate_routing_validation_cache_config,
)


CACHE_REQUIRED_FILES = (
    "manifests/frozen_build_protocol.json",
    "manifests/row_alignment.json",
    "manifests/consumption_lock.json",
    "manifests/content_index.json",
    "reports/cache_builder_report.json",
    "reports/validation_report.json",
)
UNLABELED_METADATA_KEYS = frozenset(
    {"sample_id", "case_id", "split", "center", "manifest_row_index"}
)


@dataclass(frozen=True)
class LabelBlindManifestRow:
    """Only fields allowed to influence validation-cache feature extraction."""

    sample_id: str
    case_id: str
    image_path: str
    split: str
    center: str
    manifest_row_index: int

    def alignment_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "manifest_row_index": self.manifest_row_index,
        }

    def shard_metadata(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "split": self.split,
            "center": self.center,
            "manifest_row_index": self.manifest_row_index,
        }


@dataclass(frozen=True)
class ManifestReservation:
    """Label-blind split reservation reconstructed from the frozen manifest."""

    rows: tuple[LabelBlindManifestRow, ...]
    eligible_train_rows: tuple[LabelBlindManifestRow, ...]
    validation_rows: tuple[LabelBlindManifestRow, ...]
    split_counts: Mapping[str, int]
    overlap_counts: Mapping[str, int]


def build_uniform_b_routing_validation_cache(
    config: RoutingValidationCacheConfig | ResolvedRoutingValidationCacheConfig,
    *,
    extractor_factory: Callable[..., object] = Virchow2TokenExtractor,
) -> Path:
    """Build, independently validate, and atomically publish one cache."""

    resolved = (
        resolve_routing_validation_cache_config(config)
        if isinstance(config, RoutingValidationCacheConfig)
        else config
    )
    validate_routing_validation_cache_config(resolved.contract)
    resolved.cache_root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(resolved.cache_root) as stage:
        staging = replace(resolved, cache_root=stage)
        _build_in_place(staging, extractor_factory=extractor_factory)
        from .validation import validate_uniform_b_routing_validation_cache

        validate_uniform_b_routing_validation_cache(
            stage,
            expected_config=staging,
            allow_pending=True,
        )
        _finalize_cache(staging)
        validate_uniform_b_routing_validation_cache(
            stage,
            expected_config=staging,
        )
    return resolved.cache_root


def _build_in_place(
    config: ResolvedRoutingValidationCacheConfig,
    *,
    extractor_factory: Callable[..., object],
) -> None:
    started = time.perf_counter()
    if any(config.cache_root.iterdir()):
        raise FileExistsError(
            f"Routing-validation cache staging root is not empty: {config.cache_root}."
        )
    contract = config.contract
    validate_routing_validation_cache_config(contract)
    _require_sha256(
        config.manifest_path,
        contract.expected_manifest_sha256,
        "annotation_patch_v1 manifest",
    )
    _require_sha256(
        config.canonical_train_cache_path,
        contract.expected_canonical_train_sha256,
        "canonical-A train cache",
    )
    _require_sha256(
        config.canonical_validation_cache_path,
        contract.expected_canonical_validation_sha256,
        "canonical-A validation cache",
    )
    expected_source_report = (
        config.source_train_b_cache_root / "reports" / "cache_builder_report.json"
    )
    if config.source_train_b_report_path.resolve() != expected_source_report.resolve():
        raise RoutingValidationCacheError(
            "Routing-validation source B report/root binding drifted."
        )
    _require_sha256(
        config.source_train_b_report_path,
        contract.expected_source_train_b_report_sha256,
        "source-train B report",
    )

    reservation = read_and_reserve_label_blind_manifest(config.manifest_path)
    source_report = _read_json(config.source_train_b_report_path)
    _validate_source_train_b_report(source_report)
    runtime = _runtime_identity()
    if runtime != dict(contract.expected_runtime):
        raise RoutingValidationCacheError(
            "Routing-validation runtime identity drifted: "
            f"observed={runtime}, expected={dict(contract.expected_runtime)}."
        )

    # This immutable lock is deliberately written before model construction or
    # the first feature-extraction call. No label field is present in the row
    # objects used above or below.
    frozen = _frozen_build_protocol(
        config,
        reservation,
        source_report=source_report,
        runtime_identity=runtime,
    )
    _write_json(
        config.cache_root / "manifests" / "frozen_build_protocol.json",
        frozen,
    )

    # Canonical cache metadata contains labels for later consumers, so even its
    # deserialization is kept on the locked side of the boundary. Alignment
    # below reads identity fields only.
    canonical_train = load_cache_rows(
        config.canonical_train_cache_path,
        expected_dim=CANONICAL_A_DIM,
    )
    canonical_validation = load_cache_rows(
        config.canonical_validation_cache_path,
        expected_dim=CANONICAL_A_DIM,
    )
    _validate_canonical_alignment(
        reservation,
        canonical_train=canonical_train,
        canonical_validation=canonical_validation,
    )

    _seed(contract.experiment_seed)
    extractor = extractor_factory(
        model_ref=contract.model_ref,
        model_revision=contract.model_revision,
        device=contract.device,
        expected_model_config_sha256=contract.expected_model_config_sha256,
        expected_checkpoint_file_sha256=contract.expected_checkpoint_file_sha256,
        expected_state_dict_sha256=contract.expected_state_dict_sha256,
        expected_preprocessing_config_hash=contract.expected_preprocessing_config_hash,
    )
    extractor_identity = getattr(extractor, "identity", None)
    if not isinstance(extractor_identity, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation extractor identity is missing."
        )
    source_model_identity = source_report.get("model_identity")
    if (
        not isinstance(source_model_identity, Mapping)
        or stable_hash(dict(extractor_identity)) != stable_hash(dict(source_model_identity))
    ):
        raise RoutingValidationCacheError(
            "Routing-validation extractor differs from source-train B."
        )
    preprocessing = extractor_identity.get("preprocessing_config")
    if not isinstance(preprocessing, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation preprocessing identity is missing."
        )
    spatial_identity = assert_preprocessing_spatial_identity(preprocessing)

    canonical_index = {
        sample_id: index
        for index, sample_id in enumerate(canonical_validation.sample_ids)
    }
    grouped = {
        center: tuple(row for row in reservation.validation_rows if row.center == center)
        for center in ELIGIBLE_CENTERS
    }
    minimum_cosine = 1.0
    maximum_relative_l2 = 0.0
    center_alignment: dict[str, object] = {}
    for center in ELIGIBLE_CENTERS:
        rows = grouped[center]
        chunks: list[object] = []
        canonical_chunks: list[object] = []
        for start in range(0, len(rows), contract.batch_size):
            batch = rows[start : start + contract.batch_size]
            images: list[Image.Image] = []
            try:
                for row in batch:
                    image_path = resolve_contract_path(config.repo_root, row.image_path)
                    if not image_path.is_file():
                        raise RoutingValidationCacheError(
                            f"Routing-validation JPEG is missing: {image_path}."
                        )
                    images.append(Image.open(image_path).convert("RGB"))
                extracted = extractor.extract_spatial_windows(
                    images,
                    window_starts=[(6, 6)] * len(images),
                )
                import torch

                b_tensor = torch.as_tensor(extracted).detach().cpu().float()
                if tuple(b_tensor.shape) != (len(batch), FEATURE_DIM):
                    raise RoutingValidationCacheError(
                        f"Routing-validation extracted shape drifted for center {center}."
                    )
                a_tensor = torch.stack(
                    tuple(
                        torch.as_tensor(
                            canonical_validation.embeddings[canonical_index[row.sample_id]]
                        )
                        for row in batch
                    ),
                    dim=0,
                ).detach().cpu().float()
                prefix = b_tensor[:, :CANONICAL_A_DIM]
                cosines = torch.nn.functional.cosine_similarity(prefix, a_tensor, dim=1)
                relative_l2 = torch.linalg.vector_norm(
                    prefix - a_tensor,
                    dim=1,
                ) / torch.clamp(
                    torch.linalg.vector_norm(a_tensor, dim=1),
                    min=1.0e-12,
                )
                minimum_cosine = min(minimum_cosine, float(cosines.min().item()))
                maximum_relative_l2 = max(
                    maximum_relative_l2,
                    float(relative_l2.max().item()),
                )
                chunks.append(b_tensor)
                canonical_chunks.append(a_tensor)
            finally:
                for image in images:
                    image.close()
        import torch

        embeddings = torch.cat(tuple(chunks), dim=0)
        canonical_embeddings = torch.cat(tuple(canonical_chunks), dim=0)
        shard_path = (
            config.cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
        write_center_shard(
            shard_path,
            embeddings=embeddings,
            canonical_a_embeddings=canonical_embeddings,
            metadata=[row.shard_metadata() for row in rows],
            feature_extractor={
                "schema_version": (
                    "midogpp_uniform_b_routing_validation_feature_extractor_v1"
                ),
                "representation_id": REPRESENTATION_ID,
                "canonical_comparator_id": CANONICAL_A_ID,
                "feature_dim": FEATURE_DIM,
                "pooling": POOLING_ID,
                "fixed_window_start": [6, 6],
                "token_layout": {
                    "cls_token_count": VIRCHOW2_TOKEN_LAYOUT.cls_token_count,
                    "register_token_count": VIRCHOW2_TOKEN_LAYOUT.register_token_count,
                    "patch_grid_side": VIRCHOW2_TOKEN_LAYOUT.patch_grid_side,
                    "window_side": VIRCHOW2_TOKEN_LAYOUT.window_side,
                    "patch_order": VIRCHOW2_TOKEN_LAYOUT.patch_order,
                },
                "model_identity": dict(extractor_identity),
                "runtime_identity": runtime,
                "preprocessing_spatial_identity": dict(spatial_identity),
                "frozen_build_protocol_hash": frozen["frozen_build_protocol_hash"],
                "feature_extraction_label_free": True,
                "validation_labels_used_for_feature_extraction": False,
            },
        )
        relative = str(shard_path.relative_to(config.cache_root))
        center_alignment[center] = {
            "path": relative,
            "sha256": _sha256_file(shard_path),
            "row_count": len(rows),
            "sample_id_order_hash": stable_hash([row.sample_id for row in rows]),
            "rows": [row.alignment_dict() for row in rows],
        }

    if (
        minimum_cosine < contract.minimum_canonical_a_prefix_cosine
        or maximum_relative_l2 > contract.maximum_canonical_a_prefix_relative_l2
    ):
        raise RoutingValidationCacheError(
            "Routing-validation canonical-A numeric bridge failed: "
            f"minimum_cosine={minimum_cosine}, "
            f"maximum_relative_l2={maximum_relative_l2}."
        )

    manifest_order = [row.sample_id for row in reservation.validation_rows]
    center_grouped_order = [
        row.sample_id for center in ELIGIBLE_CENTERS for row in grouped[center]
    ]
    _write_json(
        config.cache_root / "manifests" / "row_alignment.json",
        {
            "schema_version": (
                "midogpp_uniform_b_routing_validation_row_alignment_v1"
            ),
            "status": "PASS",
            "split": VALIDATION_SPLIT,
            "row_count": len(reservation.validation_rows),
            "rows_by_center": dict(EXPECTED_VALIDATION_ROWS_BY_CENTER),
            "eligible_centers": list(ELIGIBLE_CENTERS),
            "excluded_centers": ["4"],
            "center_4_present": False,
            "manifest_eligible_order_hash": stable_hash(manifest_order),
            "center_grouped_order_hash": stable_hash(center_grouped_order),
            "canonical_train_cache_sha256": canonical_train.cache_sha256,
            "canonical_validation_cache_sha256": canonical_validation.cache_sha256,
            "manifest_sha256": contract.expected_manifest_sha256,
            "split_identity_overlap_counts": dict(reservation.overlap_counts),
            "centers": center_alignment,
        },
    )
    _write_json(
        config.cache_root / "manifests" / "consumption_lock.json",
        _consumption_lock(contract),
    )
    _write_json(
        config.cache_root / "reports" / "cache_builder_report.json",
        {
            "schema_version": (
                "midogpp_uniform_b_routing_validation_cache_builder_v1"
            ),
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "representation_id": REPRESENTATION_ID,
            "split": VALIDATION_SPLIT,
            "row_count": len(reservation.validation_rows),
            "rows_by_center": dict(EXPECTED_VALIDATION_ROWS_BY_CENTER),
            "feature_dim": FEATURE_DIM,
            "pooling": POOLING_ID,
            "model_identity": dict(extractor_identity),
            "runtime_identity": runtime,
            "source_train_b_report_sha256": (
                contract.expected_source_train_b_report_sha256
            ),
            "minimum_canonical_a_prefix_cosine": minimum_cosine,
            "maximum_canonical_a_prefix_relative_l2": maximum_relative_l2,
            "required_minimum_canonical_a_prefix_cosine": (
                contract.minimum_canonical_a_prefix_cosine
            ),
            "required_maximum_canonical_a_prefix_relative_l2": (
                contract.maximum_canonical_a_prefix_relative_l2
            ),
            "validation_labels_unobserved_before_lock": True,
            "validation_labels_used_for_feature_extraction": False,
            "validation_labels_persisted_in_cache": False,
            "feature_extraction_label_free": True,
            "output_metric_computed": False,
            "routing_performed": False,
            "utility_computed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_content_index(config.cache_root)


def read_and_reserve_label_blind_manifest(path: str | Path) -> ManifestReservation:
    """Read only identity/geometry fields and reserve the exact eligible val rows."""

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "image_path", "split", "center"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RoutingValidationCacheError(
                "Routing-validation manifest lacks required label-blind identity fields."
            )
        rows = tuple(
            LabelBlindManifestRow(
                sample_id=str(row["sample_id"]),
                case_id=str(row["case_id"]),
                image_path=str(row["image_path"]),
                split=str(row["split"]),
                center=str(row["center"]),
                manifest_row_index=index,
            )
            for index, row in enumerate(reader)
        )
    if any(
        not row.sample_id
        or not row.case_id
        or not row.image_path
        or row.split not in {TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT}
        or row.center not in {*ELIGIBLE_CENTERS, "4"}
        for row in rows
    ):
        raise RoutingValidationCacheError(
            "Routing-validation manifest identity fields drifted."
        )
    if len({row.sample_id for row in rows}) != len(rows):
        raise RoutingValidationCacheError(
            "Routing-validation manifest sample IDs are duplicated."
        )
    split_counts = {
        split: sum(row.split == split for row in rows)
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    if split_counts != EXPECTED_MANIFEST_ROWS_BY_SPLIT:
        raise RoutingValidationCacheError(
            "Routing-validation manifest split counts drifted: "
            f"observed={split_counts}, expected={EXPECTED_MANIFEST_ROWS_BY_SPLIT}."
        )
    overlaps = manifest_split_overlap_counts(rows)
    if any(overlaps.values()):
        raise RoutingValidationCacheError(
            f"Routing-validation manifest split identities overlap: {overlaps}."
        )
    eligible_train = tuple(
        row for row in rows if row.split == TRAIN_SPLIT and row.center in ELIGIBLE_CENTERS
    )
    validation = tuple(
        row
        for row in rows
        if row.split == VALIDATION_SPLIT and row.center in ELIGIBLE_CENTERS
    )
    counts = {
        center: sum(row.center == center for row in validation)
        for center in ELIGIBLE_CENTERS
    }
    if (
        len(eligible_train) != 9648
        or len(validation) != EXPECTED_VALIDATION_ROWS
        or counts != EXPECTED_VALIDATION_ROWS_BY_CENTER
        or any(row.center == "4" for row in (*eligible_train, *validation))
    ):
        raise RoutingValidationCacheError(
            "Routing-validation eligible row reservation drifted."
        )
    return ManifestReservation(
        rows=rows,
        eligible_train_rows=eligible_train,
        validation_rows=validation,
        split_counts=split_counts,
        overlap_counts=overlaps,
    )


def manifest_split_overlap_counts(
    rows: Sequence[LabelBlindManifestRow],
) -> dict[str, int]:
    """Return manifest-level sample and case overlaps for all split pairs."""

    by_split = {
        split: tuple(row for row in rows if row.split == split)
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    }
    result: dict[str, int] = {}
    for left, right in (
        (TRAIN_SPLIT, VALIDATION_SPLIT),
        (TRAIN_SPLIT, TEST_SPLIT),
        (VALIDATION_SPLIT, TEST_SPLIT),
    ):
        left_samples = {row.sample_id for row in by_split[left]}
        right_samples = {row.sample_id for row in by_split[right]}
        left_cases = {row.case_id for row in by_split[left]}
        right_cases = {row.case_id for row in by_split[right]}
        prefix = f"{left}_{right}"
        result[f"{prefix}_sample_overlap"] = len(left_samples & right_samples)
        result[f"{prefix}_case_overlap"] = len(left_cases & right_cases)
    return result


def _validate_canonical_alignment(
    reservation: ManifestReservation,
    *,
    canonical_train: CacheRows,
    canonical_validation: CacheRows,
) -> None:
    train_manifest = tuple(row for row in reservation.rows if row.split == TRAIN_SPLIT)
    val_manifest = tuple(row for row in reservation.rows if row.split == VALIDATION_SPLIT)
    _assert_cache_rows_match_manifest(
        canonical_train,
        train_manifest,
        split=TRAIN_SPLIT,
    )
    _assert_cache_rows_match_manifest(
        canonical_validation,
        val_manifest,
        split=VALIDATION_SPLIT,
    )
    eligible_train_ids = tuple(
        sample_id
        for sample_id, metadata in zip(
            canonical_train.sample_ids,
            canonical_train.metadata,
            strict=True,
        )
        if str(metadata.get("center", "")) in ELIGIBLE_CENTERS
    )
    eligible_val_ids = tuple(
        sample_id
        for sample_id, metadata in zip(
            canonical_validation.sample_ids,
            canonical_validation.metadata,
            strict=True,
        )
        if str(metadata.get("center", "")) in ELIGIBLE_CENTERS
    )
    if (
        eligible_train_ids
        != tuple(row.sample_id for row in reservation.eligible_train_rows)
        or eligible_val_ids != tuple(row.sample_id for row in reservation.validation_rows)
    ):
        raise RoutingValidationCacheError(
            "Routing-validation canonical-A eligible order drifted."
        )


def _assert_cache_rows_match_manifest(
    cache: CacheRows,
    manifest_rows: Sequence[LabelBlindManifestRow],
    *,
    split: str,
) -> None:
    expected = tuple(
        (row.sample_id, row.case_id, row.split, row.center)
        for row in manifest_rows
    )
    observed = tuple(
        (
            str(row.get("sample_id", "")),
            str(row.get("case_id", "")),
            str(row.get("split", "")),
            str(row.get("center", "")),
        )
        for row in cache.metadata
    )
    if observed != expected or any(row[2] != split for row in observed):
        raise RoutingValidationCacheError(
            f"Routing-validation canonical-A {split} alignment drifted."
        )


def _validate_source_train_b_report(report: Mapping[str, object]) -> None:
    runtime = report.get("runtime_identity")
    model = report.get("model_identity")
    bridge = report.get("bridge")
    if (
        report.get("status") != "PASS"
        or report.get("representation_id") != REPRESENTATION_ID
        or report.get("feature_dim") != FEATURE_DIM
        or report.get("row_count") != 9648
        or report.get("pooling") != POOLING_ID
        or not isinstance(runtime, Mapping)
        or {key: str(runtime.get(key, "")) for key in EXPECTED_RUNTIME}
        != EXPECTED_RUNTIME
        or not isinstance(model, Mapping)
        or not isinstance(bridge, Mapping)
        or bridge.get("status") != "PASS"
        or float(bridge.get("minimum_cosine", -1.0)) < 0.99999
        or float(bridge.get("maximum_relative_l2", 1.0)) > 0.001
    ):
        raise RoutingValidationCacheError(
            "Routing-validation source-train B report drifted."
        )
    _validate_model_identity(model)


def _validate_model_identity(identity: Mapping[str, object]) -> None:
    exact = {
        "schema_version": "midogpp_virchow2_pinned_identity_v1",
        "model_ref": MODEL_REF,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "state_dict_sha256": STATE_DICT_SHA256,
        "preprocessing_config_hash": PREPROCESSING_CONFIG_HASH,
    }
    if any(identity.get(key) != value for key, value in exact.items()) or not isinstance(
        identity.get("preprocessing_config"), Mapping
    ):
        raise RoutingValidationCacheError(
            "Routing-validation pinned Virchow2 identity drifted."
        )


def _frozen_build_protocol(
    config: ResolvedRoutingValidationCacheConfig,
    reservation: ManifestReservation,
    *,
    source_report: Mapping[str, object],
    runtime_identity: Mapping[str, str],
) -> dict[str, object]:
    contract = config.contract
    payload: dict[str, object] = {
        "schema_version": (
            "midogpp_uniform_b_routing_validation_frozen_build_protocol_v1"
        ),
        "status": "LOCKED_BEFORE_FEATURE_EXTRACTION",
        "cache_name": contract.name,
        "dataset_contract": "midogpp_annotation_patch_v1",
        "representation_id": REPRESENTATION_ID,
        "canonical_comparator_id": CANONICAL_A_ID,
        "training_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "excluded_centers": ["4"],
        "expected_validation_rows": EXPECTED_VALIDATION_ROWS,
        "expected_validation_rows_by_center": dict(
            EXPECTED_VALIDATION_ROWS_BY_CENTER
        ),
        "expected_class_labels": list(EXPECTED_CLASS_LABELS),
        "all_expected_classes_present": True,
        "class_presence_basis": (
            "predeclared_hash_pinned_annotation_patch_v1_manifest"
        ),
        "pooling": POOLING_ID,
        "fixed_window_start": [6, 6],
        "feature_dim": FEATURE_DIM,
        "canonical_a_prefix_dim": CANONICAL_A_DIM,
        "model_identity": dict(source_report["model_identity"]),
        "runtime_identity": dict(runtime_identity),
        "input_hashes": {
            "manifest_sha256": contract.expected_manifest_sha256,
            "canonical_train_cache_sha256": contract.expected_canonical_train_sha256,
            "canonical_validation_cache_sha256": (
                contract.expected_canonical_validation_sha256
            ),
            "source_train_b_report_sha256": (
                contract.expected_source_train_b_report_sha256
            ),
        },
        "manifest_split_counts": dict(reservation.split_counts),
        "manifest_split_identity_overlap_counts": dict(
            reservation.overlap_counts
        ),
        "validation_labels_unobserved_before_lock": True,
        "validation_labels_used_for_feature_extraction": False,
        "validation_labels_persisted_in_cache": False,
        "feature_extraction_label_free": True,
        "output_metric_computed": False,
        "stage20_train_consumed": True,
        "test_representation_adoption_consumed": True,
        "validation_opened_only_for_later_source_inner_scoring": True,
        "center_4_excluded": True,
    }
    payload["frozen_build_protocol_hash"] = stable_hash(payload)
    return payload


def _consumption_lock(contract: RoutingValidationCacheConfig) -> dict[str, object]:
    """Return the exact split-consumption and label-disclosure firewall."""

    return {
        "schema_version": "midogpp_uniform_b_routing_validation_consumption_lock_v1",
        "status": "LOCKED",
        "representation_id": REPRESENTATION_ID,
        "training_split": {
            "name": TRAIN_SPLIT,
            "status": "CONSUMED_BY_STAGE20_SOURCE_INNER_EVIDENCE",
            "available_as_fresh_validation_evidence": False,
        },
        "test_split": {
            "name": TEST_SPLIT,
            "status": "CONSUMED_FOR_UNIFORM_B_REPRESENTATION_ADOPTION",
            "available_as_fresh_representation_selection_evidence": False,
        },
        "validation_split": {
            "name": VALIDATION_SPLIT,
            "status": "RESERVED_FOR_STAGE60_SOURCE_INNER_SCORING",
            "row_count": contract.expected_validation_rows,
            "rows_by_center": dict(contract.expected_validation_rows_by_center),
            "labels_unobserved_before_lock": True,
            "labels_used_for_feature_extraction": False,
            "labels_persisted_in_cache": False,
            "labels_may_be_joined_after_predictions_for_scoring_only": True,
            "labels_may_train_feature_extractor": False,
            "labels_may_select_representation": False,
            "labels_may_tune_experts": False,
            "opened_only_for_later_source_inner_scoring": True,
        },
        "cache_claim_boundary": {
            "prerequisite_only": True,
            "output_metric_computed": False,
            "compatibility_computed": False,
            "utility_computed": False,
            "regret_computed": False,
            "routing_performed": False,
            "expert_selected": False,
            "downstream_utility_claimed": False,
        },
    }


def _finalize_cache(config: ResolvedRoutingValidationCacheConfig) -> None:
    from .validation import validate_uniform_b_routing_validation_cache

    checks = validate_uniform_b_routing_validation_cache(
        config.cache_root,
        expected_config=config,
        allow_pending=True,
    )
    report_path = config.cache_root / "reports" / "cache_builder_report.json"
    report = _read_json(report_path)
    report["status"] = "PASS"
    report["independent_validation_status"] = "PASS"
    _write_json(report_path, report)
    _write_json(
        config.cache_root / "reports" / "validation_report.json",
        {
            "schema_version": (
                "midogpp_uniform_b_routing_validation_cache_validation_v1"
            ),
            "status": "PASS",
            "validator": "validate_uniform_b_routing_validation_cache",
            "checks": checks,
        },
    )
    write_content_index(config.cache_root)


def write_content_index(root: str | Path) -> None:
    cache_root = Path(root)
    rows = []
    for path in sorted(item for item in cache_root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(cache_root))
        if relative == "manifests/content_index.json":
            continue
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    payload: dict[str, object] = {
        "schema_version": (
            "midogpp_uniform_b_routing_validation_cache_content_index_v1"
        ),
        "files": rows,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(cache_root / "manifests" / "content_index.json", payload)


def _runtime_identity() -> dict[str, str]:
    import PIL
    import timm
    import torch

    return {
        "timm": str(timm.__version__),
        "torch": str(torch.__version__),
        "pillow": str(PIL.__version__),
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RoutingValidationCacheError(
            f"Routing-validation {label} is missing: {path}."
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise RoutingValidationCacheError(
            f"Routing-validation {label} hash drifted: "
            f"observed={actual}, expected={expected}."
        )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingValidationCacheError(
            f"Routing-validation JSON is unreadable: {path}."
        ) from exc
    if not isinstance(payload, dict):
        raise RoutingValidationCacheError(
            f"Routing-validation JSON must be an object: {path}."
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CACHE_REQUIRED_FILES",
    "LabelBlindManifestRow",
    "ManifestReservation",
    "UNLABELED_METADATA_KEYS",
    "build_uniform_b_routing_validation_cache",
    "manifest_split_overlap_counts",
    "read_and_reserve_label_blind_manifest",
    "write_content_index",
]
