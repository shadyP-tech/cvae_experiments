"""Independent fail-closed validation for the routing-validation cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.features.cache_io import load_cache_rows

from .cache import (
    CACHE_REQUIRED_FILES,
    UNLABELED_METADATA_KEYS,
    _read_json,
    _sha256_file,
    _validate_canonical_alignment,
    _validate_model_identity,
    _validate_source_train_b_report,
    read_and_reserve_label_blind_manifest,
)
from .config import (
    CANONICAL_A_DIM,
    CANONICAL_A_ID,
    CANONICAL_TRAIN_SHA256,
    CANONICAL_VALIDATION_SHA256,
    CACHE_NAME,
    ELIGIBLE_CENTERS,
    EXPECTED_CLASS_LABELS,
    EXPECTED_MANIFEST_ROWS_BY_SPLIT,
    EXPECTED_RUNTIME,
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_VALIDATION_ROWS_BY_CENTER,
    FEATURE_DIM,
    MANIFEST_SHA256,
    MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2,
    MINIMUM_CANONICAL_A_PREFIX_COSINE,
    POOLING_ID,
    REPRESENTATION_ID,
    ResolvedRoutingValidationCacheConfig,
    RoutingValidationCacheError,
    SOURCE_TRAIN_B_REPORT_SHA256,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    validate_routing_validation_cache_config,
)


# Keep the validator body readable while the public exception stays data-owned.
ProtocolError = RoutingValidationCacheError


@dataclass(frozen=True)
class UnlabeledValidationShard:
    """One validated center shard with no outcome fields in its metadata."""

    embeddings: Any
    canonical_a_embeddings: Any
    metadata: tuple[Mapping[str, object], ...]
    feature_extractor: Mapping[str, object]
    cache_sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(str(row["sample_id"]) for row in self.metadata)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(str(row["case_id"]) for row in self.metadata)

    @property
    def manifest_row_indices(self) -> tuple[int, ...]:
        return tuple(int(row["manifest_row_index"]) for row in self.metadata)


def load_unlabeled_validation_shard(
    path: str | Path,
    *,
    expected_center: str | None = None,
) -> UnlabeledValidationShard:
    """Load one shard and reject every non-whitelisted metadata field."""

    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Routing-validation cache loading requires numpy and torch.") from exc

    shard_path = Path(path)
    if not shard_path.is_file() or shard_path.is_symlink():
        raise RoutingValidationCacheError(
            f"Routing-validation shard is missing or unsafe: {shard_path}."
        )
    try:
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(shard_path, map_location="cpu")
    except Exception as exc:
        raise RoutingValidationCacheError(
            f"Routing-validation shard is unreadable: {shard_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation shard payload must be a mapping."
        )
    if set(payload) != {
        "embeddings",
        "canonical_a_embeddings",
        "metadata",
        "feature_extractor",
    }:
        raise RoutingValidationCacheError(
            "Routing-validation shard payload keys drifted."
        )
    metadata_raw = payload.get("metadata")
    if isinstance(metadata_raw, (str, bytes)) or not isinstance(metadata_raw, Sequence):
        raise RoutingValidationCacheError(
            "Routing-validation shard metadata must be a sequence."
        )
    metadata: list[dict[str, object]] = []
    for row in metadata_raw:
        if not isinstance(row, Mapping):
            raise RoutingValidationCacheError(
                "Routing-validation shard metadata row is invalid."
            )
        observed_keys = {str(key) for key in row}
        if observed_keys != UNLABELED_METADATA_KEYS:
            forbidden = sorted(observed_keys.difference(UNLABELED_METADATA_KEYS))
            raise RoutingValidationCacheError(
                "Routing-validation shard metadata is not strictly unlabeled: "
                f"unexpected={forbidden}."
            )
        normalized = {str(key).lower() for key in observed_keys}
        if normalized.intersection(
            {"label", "labels", "label_name", "y", "y_true", "target", "class", "class_id"}
        ):
            raise RoutingValidationCacheError(
                "Routing-validation shard metadata exposes an outcome."
            )
        copied = dict(row)
        if (
            not str(copied.get("sample_id", ""))
            or not str(copied.get("case_id", ""))
            or str(copied.get("split", "")) != VALIDATION_SPLIT
            or str(copied.get("center", "")) not in ELIGIBLE_CENTERS
            or isinstance(copied.get("manifest_row_index"), bool)
        ):
            raise RoutingValidationCacheError(
                "Routing-validation shard metadata identity drifted."
            )
        try:
            copied["manifest_row_index"] = int(copied["manifest_row_index"])
        except (TypeError, ValueError) as exc:
            raise RoutingValidationCacheError(
                "Routing-validation shard manifest row index is invalid."
            ) from exc
        metadata.append(copied)
    if expected_center is not None and any(
        str(row["center"]) != str(expected_center) for row in metadata
    ):
        raise RoutingValidationCacheError(
            f"Routing-validation shard contains rows outside center {expected_center}."
        )
    sample_ids = [str(row["sample_id"]) for row in metadata]
    if len(sample_ids) != len(set(sample_ids)):
        raise RoutingValidationCacheError(
            "Routing-validation shard sample IDs are duplicated."
        )

    embeddings = torch.as_tensor(payload.get("embeddings")).detach().cpu().float()
    canonical = (
        torch.as_tensor(payload.get("canonical_a_embeddings"))
        .detach()
        .cpu()
        .float()
    )
    if tuple(embeddings.shape) != (len(metadata), FEATURE_DIM):
        raise RoutingValidationCacheError(
            "Routing-validation shard embedding shape drifted."
        )
    if tuple(canonical.shape) != (len(metadata), CANONICAL_A_DIM):
        raise RoutingValidationCacheError(
            "Routing-validation shard canonical-A shape drifted."
        )
    if not bool(torch.isfinite(embeddings).all()) or not bool(torch.isfinite(canonical).all()):
        raise RoutingValidationCacheError(
            "Routing-validation shard contains non-finite embeddings."
        )
    extractor = payload.get("feature_extractor")
    if not isinstance(extractor, Mapping):
        raise RoutingValidationCacheError(
            "Routing-validation shard extractor identity is missing."
        )
    return UnlabeledValidationShard(
        embeddings=embeddings,
        canonical_a_embeddings=canonical,
        metadata=tuple(metadata),
        feature_extractor=dict(extractor),
        cache_sha256=_sha256_file(shard_path),
    )


def validate_uniform_b_routing_validation_cache(
    root: str | Path,
    *,
    expected_config: ResolvedRoutingValidationCacheConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct every cache claim and fail closed on any byte/schema drift."""

    cache_root = Path(root)
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise ProtocolError(
            f"Routing-validation cache root is missing or unsafe: {cache_root}."
        )
    if expected_config is not None:
        validate_routing_validation_cache_config(expected_config.contract)
    required = set(CACHE_REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    required.update(
        f"embeddings/by_center/center_{center}.pt" for center in ELIGIBLE_CENTERS
    )
    missing = sorted(
        relative
        for relative in required
        if not (cache_root / relative).is_file()
        or (cache_root / relative).is_symlink()
    )
    if missing:
        raise ProtocolError(
            f"Routing-validation cache is incomplete or unsafe: {missing}."
        )
    actual_members = {
        str(path.relative_to(cache_root))
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    if actual_members != required:
        raise ProtocolError(
            "Routing-validation cache member set drifted: "
            f"unexpected={sorted(actual_members.difference(required))}, "
            f"unindexed={sorted(required.difference(actual_members))}."
        )
    validate_content_index(cache_root)

    frozen = _read_json(cache_root / "manifests/frozen_build_protocol.json")
    alignment = _read_json(cache_root / "manifests/row_alignment.json")
    consumption = _read_json(cache_root / "manifests/consumption_lock.json")
    report = _read_json(cache_root / "reports/cache_builder_report.json")
    _validate_frozen_protocol(frozen)
    if consumption != _expected_consumption_lock():
        raise ProtocolError("Routing-validation split-consumption lock drifted.")
    _validate_builder_report(report, frozen=frozen, allow_pending=allow_pending)
    center_rows = _validate_alignment(alignment)

    minimum_cosine = 1.0
    maximum_relative_l2 = 0.0
    observed_ids: list[str] = []
    shard_hashes: dict[str, str] = {}
    for center in ELIGIBLE_CENTERS:
        shard_path = cache_root / f"embeddings/by_center/center_{center}.pt"
        shard = load_unlabeled_validation_shard(
            shard_path,
            expected_center=center,
        )
        expected_rows = center_rows[center]
        expected_metadata = tuple(
            {
                "sample_id": str(row["sample_id"]),
                "case_id": str(row["case_id"]),
                "split": VALIDATION_SPLIT,
                "center": center,
                "manifest_row_index": int(row["manifest_row_index"]),
            }
            for row in expected_rows
        )
        if shard.metadata != expected_metadata:
            raise ProtocolError(
                f"Routing-validation shard/index alignment drifted for center {center}."
            )
        center_record = alignment["centers"][center]
        if (
            shard.cache_sha256 != center_record.get("sha256")
            or shard.cache_sha256 != _sha256_file(shard_path)
            or center_record.get("path")
            != f"embeddings/by_center/center_{center}.pt"
            or center_record.get("sample_id_order_hash")
            != stable_hash(list(shard.sample_ids))
        ):
            raise ProtocolError(
                f"Routing-validation shard hash/order drifted for center {center}."
            )
        _validate_feature_extractor(shard.feature_extractor, frozen=frozen)
        import torch

        prefix = shard.embeddings[:, :CANONICAL_A_DIM]
        cosines = torch.nn.functional.cosine_similarity(
            prefix,
            shard.canonical_a_embeddings,
            dim=1,
        )
        relative_l2 = torch.linalg.vector_norm(
            prefix - shard.canonical_a_embeddings,
            dim=1,
        ) / torch.clamp(
            torch.linalg.vector_norm(shard.canonical_a_embeddings, dim=1),
            min=1.0e-12,
        )
        minimum_cosine = min(minimum_cosine, float(cosines.min().item()))
        maximum_relative_l2 = max(
            maximum_relative_l2,
            float(relative_l2.max().item()),
        )
        observed_ids.extend(shard.sample_ids)
        shard_hashes[center] = shard.cache_sha256

    if (
        len(observed_ids) != EXPECTED_VALIDATION_ROWS
        or len(observed_ids) != len(set(observed_ids))
        or minimum_cosine < MINIMUM_CANONICAL_A_PREFIX_COSINE
        or maximum_relative_l2 > MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2
    ):
        raise ProtocolError("Routing-validation coverage or canonical-A bridge failed.")
    if (
        report.get("minimum_canonical_a_prefix_cosine") != minimum_cosine
        or report.get("maximum_canonical_a_prefix_relative_l2")
        != maximum_relative_l2
    ):
        raise ProtocolError("Routing-validation bridge report is not reconstructible.")

    if expected_config is not None:
        _validate_external_inputs(
            expected_config,
            alignment=alignment,
            frozen=frozen,
            center_rows=center_rows,
            cache_root=cache_root,
        )

    checks: dict[str, object] = {
        "status": "PASS",
        "split": VALIDATION_SPLIT,
        "row_count": len(observed_ids),
        "center_count": len(ELIGIBLE_CENTERS),
        "feature_dim": FEATURE_DIM,
        "label_fields_absent": True,
        "output_metric_computed": False,
        "minimum_canonical_a_prefix_cosine": minimum_cosine,
        "maximum_canonical_a_prefix_relative_l2": maximum_relative_l2,
        "shard_sha256_by_center": shard_hashes,
    }
    if not allow_pending:
        validation = _read_json(cache_root / "reports/validation_report.json")
        if (
            set(validation) != {"schema_version", "status", "validator", "checks"}
            or validation.get("schema_version")
            != "midogpp_uniform_b_routing_validation_cache_validation_v1"
            or validation.get("status") != "PASS"
            or validation.get("validator")
            != "validate_uniform_b_routing_validation_cache"
            or validation.get("checks") != checks
        ):
            raise ProtocolError("Routing-validation independent report drifted.")
    return checks


def validate_content_index(root: str | Path) -> None:
    """Validate the closed set of cache members and every member SHA-256."""

    cache_root = Path(root)
    payload = _read_json(cache_root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if (
        payload.get("schema_version")
        != "midogpp_uniform_b_routing_validation_cache_content_index_v1"
        or stable_hash(unhashed) != payload.get("content_hash")
    ):
        raise ProtocolError("Routing-validation content-index identity drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Routing-validation content index must contain file rows.")
    expected = {
        str(path.relative_to(cache_root))
        for path in cache_root.rglob("*")
        if path.is_file()
        and str(path.relative_to(cache_root)) != "manifests/content_index.json"
    }
    observed: set[str] = set()
    root_resolved = cache_root.resolve()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ProtocolError("Routing-validation content-index row drifted.")
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in observed
        ):
            raise ProtocolError("Routing-validation content-index path is unsafe.")
        path = cache_root / relative_path
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root_resolved)
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError(
                f"Routing-validation cache member drifted: {relative}."
            )
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Routing-validation content-index coverage drifted.")


def _validate_frozen_protocol(frozen: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "cache_name",
        "dataset_contract",
        "representation_id",
        "canonical_comparator_id",
        "training_split",
        "validation_split",
        "test_split",
        "eligible_centers",
        "excluded_centers",
        "expected_validation_rows",
        "expected_validation_rows_by_center",
        "expected_class_labels",
        "all_expected_classes_present",
        "class_presence_basis",
        "pooling",
        "fixed_window_start",
        "feature_dim",
        "canonical_a_prefix_dim",
        "model_identity",
        "runtime_identity",
        "input_hashes",
        "manifest_split_counts",
        "manifest_split_identity_overlap_counts",
        "validation_labels_unobserved_before_lock",
        "validation_labels_used_for_feature_extraction",
        "validation_labels_persisted_in_cache",
        "feature_extraction_label_free",
        "output_metric_computed",
        "stage20_train_consumed",
        "test_representation_adoption_consumed",
        "validation_opened_only_for_later_source_inner_scoring",
        "center_4_excluded",
        "frozen_build_protocol_hash",
    }
    unhashed = {
        key: value for key, value in frozen.items() if key != "frozen_build_protocol_hash"
    }
    expected_overlap = {
        "train_val_sample_overlap": 0,
        "train_val_case_overlap": 0,
        "train_test_sample_overlap": 0,
        "train_test_case_overlap": 0,
        "val_test_sample_overlap": 0,
        "val_test_case_overlap": 0,
    }
    if (
        set(frozen) != expected_keys
        or stable_hash(unhashed) != frozen.get("frozen_build_protocol_hash")
        or frozen.get("schema_version")
        != "midogpp_uniform_b_routing_validation_frozen_build_protocol_v1"
        or frozen.get("status") != "LOCKED_BEFORE_FEATURE_EXTRACTION"
        or frozen.get("cache_name") != CACHE_NAME
        or frozen.get("dataset_contract") != "midogpp_annotation_patch_v1"
        or frozen.get("representation_id") != REPRESENTATION_ID
        or frozen.get("canonical_comparator_id") != CANONICAL_A_ID
        or frozen.get("training_split") != TRAIN_SPLIT
        or frozen.get("validation_split") != VALIDATION_SPLIT
        or frozen.get("test_split") != TEST_SPLIT
        or frozen.get("eligible_centers") != list(ELIGIBLE_CENTERS)
        or frozen.get("excluded_centers") != ["4"]
        or frozen.get("expected_validation_rows") != EXPECTED_VALIDATION_ROWS
        or frozen.get("expected_validation_rows_by_center")
        != EXPECTED_VALIDATION_ROWS_BY_CENTER
        or frozen.get("expected_class_labels") != list(EXPECTED_CLASS_LABELS)
        or frozen.get("all_expected_classes_present") is not True
        or frozen.get("class_presence_basis")
        != "predeclared_hash_pinned_annotation_patch_v1_manifest"
        or frozen.get("pooling") != POOLING_ID
        or frozen.get("fixed_window_start") != [6, 6]
        or frozen.get("feature_dim") != FEATURE_DIM
        or frozen.get("canonical_a_prefix_dim") != CANONICAL_A_DIM
        or frozen.get("runtime_identity") != EXPECTED_RUNTIME
        or frozen.get("input_hashes")
        != {
            "manifest_sha256": MANIFEST_SHA256,
            "canonical_train_cache_sha256": CANONICAL_TRAIN_SHA256,
            "canonical_validation_cache_sha256": CANONICAL_VALIDATION_SHA256,
            "source_train_b_report_sha256": SOURCE_TRAIN_B_REPORT_SHA256,
        }
        or frozen.get("manifest_split_counts") != EXPECTED_MANIFEST_ROWS_BY_SPLIT
        or frozen.get("manifest_split_identity_overlap_counts") != expected_overlap
        or frozen.get("validation_labels_unobserved_before_lock") is not True
        or frozen.get("validation_labels_used_for_feature_extraction") is not False
        or frozen.get("validation_labels_persisted_in_cache") is not False
        or frozen.get("feature_extraction_label_free") is not True
        or frozen.get("output_metric_computed") is not False
        or frozen.get("stage20_train_consumed") is not True
        or frozen.get("test_representation_adoption_consumed") is not True
        or frozen.get("validation_opened_only_for_later_source_inner_scoring")
        is not True
        or frozen.get("center_4_excluded") is not True
    ):
        raise ProtocolError("Routing-validation frozen protocol drifted.")
    model = frozen.get("model_identity")
    if not isinstance(model, Mapping):
        raise ProtocolError("Routing-validation frozen model identity is missing.")
    _validate_model_identity(model)


def _validate_builder_report(
    report: Mapping[str, object],
    *,
    frozen: Mapping[str, object],
    allow_pending: bool,
) -> None:
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    expected_keys = {
        "schema_version",
        "status",
        "representation_id",
        "split",
        "row_count",
        "rows_by_center",
        "feature_dim",
        "pooling",
        "model_identity",
        "runtime_identity",
        "source_train_b_report_sha256",
        "minimum_canonical_a_prefix_cosine",
        "maximum_canonical_a_prefix_relative_l2",
        "required_minimum_canonical_a_prefix_cosine",
        "required_maximum_canonical_a_prefix_relative_l2",
        "validation_labels_unobserved_before_lock",
        "validation_labels_used_for_feature_extraction",
        "validation_labels_persisted_in_cache",
        "feature_extraction_label_free",
        "output_metric_computed",
        "routing_performed",
        "utility_computed",
        "elapsed_seconds",
    }
    if not allow_pending:
        expected_keys.add("independent_validation_status")
    if (
        set(report) != expected_keys
        or report.get("schema_version")
        != "midogpp_uniform_b_routing_validation_cache_builder_v1"
        or report.get("status") != expected_status
        or report.get("representation_id") != REPRESENTATION_ID
        or report.get("split") != VALIDATION_SPLIT
        or report.get("row_count") != EXPECTED_VALIDATION_ROWS
        or report.get("rows_by_center") != EXPECTED_VALIDATION_ROWS_BY_CENTER
        or report.get("feature_dim") != FEATURE_DIM
        or report.get("pooling") != POOLING_ID
        or report.get("runtime_identity") != EXPECTED_RUNTIME
        or report.get("source_train_b_report_sha256")
        != SOURCE_TRAIN_B_REPORT_SHA256
        or report.get("required_minimum_canonical_a_prefix_cosine")
        != MINIMUM_CANONICAL_A_PREFIX_COSINE
        or report.get("required_maximum_canonical_a_prefix_relative_l2")
        != MAXIMUM_CANONICAL_A_PREFIX_RELATIVE_L2
        or report.get("validation_labels_unobserved_before_lock") is not True
        or report.get("validation_labels_used_for_feature_extraction") is not False
        or report.get("validation_labels_persisted_in_cache") is not False
        or report.get("feature_extraction_label_free") is not True
        or report.get("output_metric_computed") is not False
        or report.get("routing_performed") is not False
        or report.get("utility_computed") is not False
    ):
        raise ProtocolError("Routing-validation cache-builder claim boundary drifted.")
    if not allow_pending and report.get("independent_validation_status") != "PASS":
        raise ProtocolError("Routing-validation final validation status drifted.")
    model = report.get("model_identity")
    frozen_model = frozen.get("model_identity")
    if (
        not isinstance(model, Mapping)
        or not isinstance(frozen_model, Mapping)
        or stable_hash(dict(model)) != stable_hash(dict(frozen_model))
    ):
        raise ProtocolError("Routing-validation builder model identity drifted.")


def _validate_alignment(
    alignment: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    expected_overlap = {
        "train_val_sample_overlap": 0,
        "train_val_case_overlap": 0,
        "train_test_sample_overlap": 0,
        "train_test_case_overlap": 0,
        "val_test_sample_overlap": 0,
        "val_test_case_overlap": 0,
    }
    centers = alignment.get("centers")
    expected_keys = {
        "schema_version",
        "status",
        "split",
        "row_count",
        "rows_by_center",
        "eligible_centers",
        "excluded_centers",
        "center_4_present",
        "manifest_eligible_order_hash",
        "center_grouped_order_hash",
        "canonical_train_cache_sha256",
        "canonical_validation_cache_sha256",
        "manifest_sha256",
        "split_identity_overlap_counts",
        "centers",
    }
    if (
        set(alignment) != expected_keys
        or alignment.get("schema_version")
        != "midogpp_uniform_b_routing_validation_row_alignment_v1"
        or alignment.get("status") != "PASS"
        or alignment.get("split") != VALIDATION_SPLIT
        or alignment.get("row_count") != EXPECTED_VALIDATION_ROWS
        or alignment.get("rows_by_center") != EXPECTED_VALIDATION_ROWS_BY_CENTER
        or alignment.get("eligible_centers") != list(ELIGIBLE_CENTERS)
        or alignment.get("excluded_centers") != ["4"]
        or alignment.get("center_4_present") is not False
        or alignment.get("canonical_train_cache_sha256") != CANONICAL_TRAIN_SHA256
        or alignment.get("canonical_validation_cache_sha256")
        != CANONICAL_VALIDATION_SHA256
        or alignment.get("manifest_sha256") != MANIFEST_SHA256
        or alignment.get("split_identity_overlap_counts") != expected_overlap
        or not isinstance(alignment.get("manifest_eligible_order_hash"), str)
        or not isinstance(alignment.get("center_grouped_order_hash"), str)
        or not isinstance(centers, Mapping)
        or set(centers) != set(ELIGIBLE_CENTERS)
    ):
        raise ProtocolError("Routing-validation row-alignment manifest drifted.")
    result: dict[str, tuple[Mapping[str, object], ...]] = {}
    grouped_ids: list[str] = []
    seen_ids: set[str] = set()
    for center in ELIGIBLE_CENTERS:
        record = centers[center]
        if not isinstance(record, Mapping):
            raise ProtocolError("Routing-validation center alignment must be a mapping.")
        if set(record) != {
            "path",
            "sha256",
            "row_count",
            "sample_id_order_hash",
            "rows",
        }:
            raise ProtocolError(
                f"Routing-validation center alignment keys drifted: {center}."
            )
        rows = record.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_VALIDATION_ROWS_BY_CENTER[center]:
            raise ProtocolError(
                f"Routing-validation center alignment count drifted: {center}."
            )
        normalized: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "sample_id",
                "case_id",
                "manifest_row_index",
            }:
                raise ProtocolError(
                    "Routing-validation row index exposes labels or has schema drift."
                )
            sample_id = str(row.get("sample_id", ""))
            case_id = str(row.get("case_id", ""))
            try:
                manifest_row_index = int(row.get("manifest_row_index"))
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    "Routing-validation row index has an invalid manifest offset."
                ) from exc
            if not sample_id or not case_id or sample_id in seen_ids:
                raise ProtocolError("Routing-validation row index identity drifted.")
            seen_ids.add(sample_id)
            grouped_ids.append(sample_id)
            normalized.append(
                {
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "manifest_row_index": manifest_row_index,
                }
            )
        if (
            record.get("row_count") != len(rows)
            or record.get("sample_id_order_hash")
            != stable_hash([row["sample_id"] for row in normalized])
        ):
            raise ProtocolError(
                f"Routing-validation center row order drifted: {center}."
            )
        result[center] = tuple(normalized)
    if (
        len(grouped_ids) != EXPECTED_VALIDATION_ROWS
        or alignment.get("center_grouped_order_hash") != stable_hash(grouped_ids)
    ):
        raise ProtocolError("Routing-validation global row order drifted.")
    return result


def _validate_feature_extractor(
    extractor: Mapping[str, object],
    *,
    frozen: Mapping[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "representation_id",
        "canonical_comparator_id",
        "feature_dim",
        "pooling",
        "fixed_window_start",
        "token_layout",
        "model_identity",
        "runtime_identity",
        "preprocessing_spatial_identity",
        "frozen_build_protocol_hash",
        "feature_extraction_label_free",
        "validation_labels_used_for_feature_extraction",
    }
    if (
        set(extractor) != expected_keys
        or extractor.get("schema_version")
        != "midogpp_uniform_b_routing_validation_feature_extractor_v1"
        or extractor.get("representation_id") != REPRESENTATION_ID
        or extractor.get("canonical_comparator_id") != CANONICAL_A_ID
        or extractor.get("feature_dim") != FEATURE_DIM
        or extractor.get("pooling") != POOLING_ID
        or extractor.get("fixed_window_start") != [6, 6]
        or extractor.get("runtime_identity") != EXPECTED_RUNTIME
        or extractor.get("frozen_build_protocol_hash")
        != frozen.get("frozen_build_protocol_hash")
        or extractor.get("feature_extraction_label_free") is not True
        or extractor.get("validation_labels_used_for_feature_extraction") is not False
    ):
        raise ProtocolError("Routing-validation shard extractor contract drifted.")
    layout = extractor.get("token_layout")
    if layout != {
        "cls_token_count": 1,
        "register_token_count": 4,
        "patch_grid_side": 16,
        "window_side": 4,
        "patch_order": "row-major",
    }:
        raise ProtocolError("Routing-validation shard token-layout contract drifted.")
    spatial = extractor.get("preprocessing_spatial_identity")
    if not isinstance(spatial, Mapping) or spatial.get("spatial_identity") is not True:
        raise ProtocolError("Routing-validation preprocessing is not spatially identity.")
    model = extractor.get("model_identity")
    frozen_model = frozen.get("model_identity")
    if (
        not isinstance(model, Mapping)
        or not isinstance(frozen_model, Mapping)
        or stable_hash(dict(model)) != stable_hash(dict(frozen_model))
    ):
        raise ProtocolError("Routing-validation shard model identity drifted.")


def _validate_external_inputs(
    config: ResolvedRoutingValidationCacheConfig,
    *,
    alignment: Mapping[str, object],
    frozen: Mapping[str, object],
    center_rows: Mapping[str, tuple[Mapping[str, object], ...]],
    cache_root: Path,
) -> None:
    contract = config.contract
    external = (
        (config.manifest_path, contract.expected_manifest_sha256, "manifest"),
        (
            config.canonical_train_cache_path,
            contract.expected_canonical_train_sha256,
            "canonical train",
        ),
        (
            config.canonical_validation_cache_path,
            contract.expected_canonical_validation_sha256,
            "canonical validation",
        ),
        (
            config.source_train_b_report_path,
            contract.expected_source_train_b_report_sha256,
            "source B report",
        ),
    )
    for path, expected_hash, label in external:
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_hash:
            raise ProtocolError(
                f"Routing-validation external {label} identity drifted."
            )
    expected_report = config.source_train_b_cache_root / "reports/cache_builder_report.json"
    if config.source_train_b_report_path.resolve() != expected_report.resolve():
        raise ProtocolError("Routing-validation external source-report binding drifted.")
    reservation = read_and_reserve_label_blind_manifest(config.manifest_path)
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
    source_report = _read_json(config.source_train_b_report_path)
    _validate_source_train_b_report(source_report)
    source_model = source_report.get("model_identity")
    frozen_model = frozen.get("model_identity")
    if (
        not isinstance(source_model, Mapping)
        or not isinstance(frozen_model, Mapping)
        or stable_hash(dict(source_model)) != stable_hash(dict(frozen_model))
    ):
        raise ProtocolError("Routing-validation source/frozen extractor identity drifted.")
    expected_manifest_rows = {
        center: tuple(
            row.alignment_dict()
            for row in reservation.validation_rows
            if row.center == center
        )
        for center in ELIGIBLE_CENTERS
    }
    if center_rows != expected_manifest_rows:
        raise ProtocolError("Routing-validation manifest/index rows differ.")
    if alignment.get("manifest_eligible_order_hash") != stable_hash(
        [row.sample_id for row in reservation.validation_rows]
    ):
        raise ProtocolError("Routing-validation manifest-order reservation drifted.")

    import torch

    canonical_index = {
        sample_id: index
        for index, sample_id in enumerate(canonical_validation.sample_ids)
    }
    for center in ELIGIBLE_CENTERS:
        shard = load_unlabeled_validation_shard(
            cache_root / f"embeddings/by_center/center_{center}.pt",
            expected_center=center,
        )
        expected = torch.stack(
            tuple(
                torch.as_tensor(canonical_validation.embeddings[canonical_index[sample_id]])
                for sample_id in shard.sample_ids
            ),
            dim=0,
        ).detach().cpu().float()
        if not torch.equal(shard.canonical_a_embeddings, expected):
            raise ProtocolError(
                f"Routing-validation embedded canonical A differs for center {center}."
            )


def _expected_consumption_lock() -> dict[str, object]:
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
            "row_count": EXPECTED_VALIDATION_ROWS,
            "rows_by_center": dict(EXPECTED_VALIDATION_ROWS_BY_CENTER),
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


__all__ = [
    "UnlabeledValidationShard",
    "load_unlabeled_validation_shard",
    "validate_content_index",
    "validate_uniform_b_routing_validation_cache",
]
