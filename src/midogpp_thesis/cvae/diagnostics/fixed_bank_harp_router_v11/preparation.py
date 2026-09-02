"""Prepare source-train development and full-test evaluation inputs.

This module retains the established lifecycle API while delegating canonical
source authentication, durable cache publication, and role-manifest
and evaluation-release publication to focused helpers. The scoring manifest remains inaccessible
until the label-free cache and partition barrier have been flushed and
independently reconstructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    _read_evaluation_release_descriptor,
    _read_label_manifest,
)
from .preparation_cache_persistence import (
    independently_validate_label_blind_barrier as _independently_validate_label_blind_barrier,
    persist_label_blind_cache as _persist_label_blind_cache,
    validate_final_prepared_cache as _validate_final_prepared_cache,
    write_final_prepared_content_index as _write_final_content_index,
)
from .preparation_canonical_cache import (
    load_canonical_label_blind_cache,
    load_canonical_target_test_label_blind_cache,
    validate_canonical_label_blind_cache_identity,
)
from .preparation_source_train_cache import (
    load_canonical_source_train_label_blind_cache,
)
from .preparation_contracts import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_NAME,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_EXPERT_BANK_LOCK_HASH,
    CANONICAL_GENERATION_LOCK_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_PARENT_LEDGER_SHA256,
    CANONICAL_REPRESENTATION,
    CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
    CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
    CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
    CASE_PARTITION,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASES_BY_CENTER,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROWS_BY_CENTER,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    LEGACY_LABEL as _LEGACY_LABEL,
    METADATA_FIELDS as _METADATA_FIELDS,
    PARTITION_NAMESPACE,
    PREPARATION_RECEIPT,
    CanonicalFrameRow,
    CanonicalLabelBlindCacheIdentity,
    CanonicalLabelBlindFrame,
    HarpPreparationIdentity,
    HarpPreparedInputData,
    HarpV11PreparedInputs,
    V11_PREPARATION_IDENTITY,
)
from .preparation_durable_io import (
    atomic_text as _atomic_text,
    fsync_directory as _fsync_directory,
    fsync_file as _fsync_file,
    fsync_tree as _fsync_tree,
    single_inventory as _single_inventory,
    write_cache_rows as _write_cache_rows,
    write_content_index as _write_content_index,
)
from .preparation_role_manifests import (
    evaluation_row_id as _evaluation_row_id,
    publish_role_pure_manifests as _publish_role_pure_manifests,
)
from .safe_paths import safe_existing_member


def deterministic_case_partition(
    source_train_rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
    target_test_rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
) -> Mapping[tuple[str, str, str], str]:
    """Assign immutable roles by physical split, never by labels or outcomes."""

    output: dict[tuple[str, str, str], str] = {}
    if (
        tuple(source_train_rows_by_center) != CENTERS
        or tuple(target_test_rows_by_center) != CENTERS
    ):
        raise ProtocolError("HARP preparation center universe drifted.")
    source_cases: set[tuple[str, str]] = set()
    target_cases: set[tuple[str, str]] = set()
    for source_split, rows_by_center, role, observed in (
        ("train", source_train_rows_by_center, DEVELOPMENT_ROLE, source_cases),
        ("test", target_test_rows_by_center, EVALUATION_ROLE, target_cases),
    ):
        for center in CENTERS:
            rows = tuple(rows_by_center[center])
            cases = tuple(sorted({row.case_id for row in rows}))
            if (
                not cases
                or any(
                    row.center != center or row.source_split != source_split
                    for row in rows
                )
            ):
                raise ProtocolError("HARP v11 physical split identity drifted.")
            for case in cases:
                output[(source_split, center, case)] = role
                observed.add((center, case))
    if source_cases & target_cases:
        raise ProtocolError("HARP v11 source-train and target-test cases overlap.")
    if set(output.values()) != {DEVELOPMENT_ROLE, EVALUATION_ROLE}:
        raise ProtocolError("HARP v11 source-train/full-test roles are incomplete.")
    return output


def build_case_partition_payload(
    source_train_rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
    target_test_rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
    *,
    identity: HarpPreparationIdentity = V11_PREPARATION_IDENTITY,
) -> tuple[Mapping[tuple[str, str, str], str], dict[str, object], str]:
    """Build the byte-stable label-free partition contract for one revision."""

    partition = deterministic_case_partition(
        source_train_rows_by_center, target_test_rows_by_center
    )
    payload: dict[str, object] = {
        "schema_version": identity.partition_schema,
        "namespace": PARTITION_NAMESPACE,
        "assignments": [
            {
                "source_split": source_split,
                "center": center,
                "case_id": case,
                "split_role": role,
            }
            for (source_split, center, case), role in sorted(partition.items())
        ],
        "whole_case_disjoint": True,
        "source_development_origin": "canonical_train_only",
        "target_evaluation_origin": "canonical_test_all_cases",
        "test_development_cases": 0,
        "label_values_available": False,
        "canonical_scoring_manifest_opened": False,
    }
    return partition, payload, canonical_hash(payload)


def _prepare_harp_consumed_test_inputs(
    *,
    canonical_train_cache_root: str | Path,
    canonical_test_cache_root: str | Path,
    canonical_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    identity: HarpPreparationIdentity,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_parent_ledger_sha256: str = CANONICAL_PARENT_LEDGER_SHA256,
) -> HarpPreparedInputData:
    """Materialize the cache, development truth, and sealed evaluation release."""

    destination = Path(cache_root).resolve()
    development_path = Path(development_manifest_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    _assert_fresh_destinations(destination, development_path, evaluation_path)
    if expected_manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ProtocolError("HARP v11 canonical test-manifest binding drifted.")
    ledger = Path(parent_ledger_path).resolve()
    if (
        not ledger.is_file()
        or ledger.is_symlink()
        or sha256_file(ledger) != expected_parent_ledger_sha256
    ):
        raise ProtocolError("HARP preparation parent ledger is absent or drifted.")
    read_json(ledger)

    # Neither loader receives the scoring manifest.  The source projection
    # never indexes legacy metadata labels; the target projection contains no
    # labels at all.
    source_train_frame = load_canonical_source_train_label_blind_cache(
        Path(canonical_train_cache_root)
    )
    target_test_frame = load_canonical_target_test_label_blind_cache(
        Path(canonical_test_cache_root)
    )
    partition, partition_payload, partition_hash = build_case_partition_payload(
        source_train_frame.rows_by_center,
        target_test_frame.rows_by_center,
        identity=identity,
    )
    prepared_rows = _persist_label_blind_cache(
        destination,
        source_train_frame=source_train_frame,
        target_test_frame=target_test_frame,
        partition=partition,
        partition_payload=partition_payload,
        partition_hash=partition_hash,
        identity=identity,
    )
    _fsync_tree(destination)
    cache = _independently_validate_label_blind_barrier(
        destination,
        expected_partition_hash=partition_hash,
        identity=identity,
    )

    # This is the first operation allowed to open source-train labels.  The
    # target scoring manifest remains unopened until the terminal route seal.
    development_sha, evaluation_sha = _publish_role_pure_manifests(
        Path(canonical_manifest_path),
        expected_manifest_sha256=expected_manifest_sha256,
        cache=cache,
        canonical_train_cache_root=Path(canonical_train_cache_root),
        source_train_frame=source_train_frame,
        target_test_frame=target_test_frame,
        development_path=development_path,
        evaluation_path=evaluation_path,
        identity=identity,
    )
    _fsync_file(development_path)
    _fsync_file(evaluation_path)
    _read_label_manifest(
        development_path,
        expected_sha256=development_sha,
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
    )
    _read_evaluation_release_descriptor(
        evaluation_path,
        expected_sha256=evaluation_sha,
        cache=cache,
        require_final_cache=False,
    )
    receipt_base: dict[str, object] = {
        "schema_version": identity.preparation_receipt_schema,
        "experiment_id": identity.experiment_id,
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "canonical_source_train_tensor_sha256": source_train_frame.cache_content_hash,
        "canonical_source_train_row_order_hash": source_train_frame.row_order_hash,
        "canonical_target_test_cache_content_hash": target_test_frame.cache_content_hash,
        "canonical_target_test_row_order_hash": target_test_frame.row_order_hash,
        "canonical_test_manifest_sha256": expected_manifest_sha256,
        "parent_ledger_sha256": expected_parent_ledger_sha256,
        "partition_hash": partition_hash,
        "label_free_barrier_sha256": sha256_file(
            destination / identity.label_free_barrier
        ),
        "label_free_content_index_sha256": sha256_file(
            destination / identity.label_free_content_index
        ),
        "pre_manifest_cache_content_sha256": cache.content_sha256,
        "prepared_cache_index_hash": cache.cache_hash,
        "prepared_row_count": len(prepared_rows),
        "development_manifest_sha256": development_sha,
        "evaluation_manifest_sha256": evaluation_sha,
        "source_label_capability_artifact_kind": (
            "center_sharded_source_label_capability"
        ),
        "source_label_capability_shard_count": len(CENTERS),
        "source_label_capability_index_contains_labels": False,
        "source_label_capability_requires_fold_physical_surface_seal": True,
        "source_label_q_access_requires_prelabel_prediction_seal": True,
        "source_label_fit_scope_excludes_outer_H_and_heldout_q": True,
        "evaluation_artifact_kind": "sealed_label_free_release_descriptor",
        "evaluation_truth_rows_published_during_preparation": False,
        "evaluation_release_requires_frozen_route_receipt": True,
        "source_train_and_target_test_cases_disjoint": True,
        "source_development_row_count": 9648,
        "source_development_case_count": 216,
        "target_evaluation_row_count": 9928,
        "target_evaluation_case_count": 218,
        "test_development_case_count": 0,
        "mixed_patch_labels_within_case_supported": True,
        "partition_selected_without_labels": True,
        "cache_fsynced_and_independently_validated_before_manifest_open": True,
        "execution_amendment_created": False,
        "execution_authorized": False,
        "publication_status": identity.publication_status,
        "terminal_decision": identity.terminal_decision,
        "fresh_evidence": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_another_experiment": False,
    }
    receipt = {**receipt_base, "receipt_hash": canonical_hash(receipt_base)}
    atomic_json(destination / identity.preparation_receipt, receipt)
    _fsync_file(destination / identity.preparation_receipt)
    _write_final_content_index(destination, identity=identity)
    _fsync_tree(destination)
    cache = _validate_final_prepared_cache(destination, identity=identity)
    return HarpPreparedInputData(
        cache_root=destination,
        development_manifest_path=development_path,
        evaluation_manifest_path=evaluation_path,
        cache_content_sha256=cache.content_sha256,
        development_manifest_sha256=development_sha,
        evaluation_manifest_sha256=evaluation_sha,
        parent_ledger_sha256=expected_parent_ledger_sha256,
        partition_hash=partition_hash,
        preparation_receipt_hash=str(receipt["receipt_hash"]),
    )


def prepare_harp_consumed_test_inputs_v11(
    *,
    canonical_train_cache_root: str | Path,
    canonical_test_cache_root: str | Path,
    canonical_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_parent_ledger_sha256: str = CANONICAL_PARENT_LEDGER_SHA256,
) -> HarpV11PreparedInputs:
    """Reject the retired arbitrary-path v11 preparation surface.

    Only ``prepare_harp_v11_workstation_inputs`` may invoke the private builder,
    after catalog resolution and the exact preparation confirmation. Keep this
    fail-closed shim so stale callers cannot silently bypass that lifecycle.
    """

    raise ProtocolError(
        "HARP v11 arbitrary-path preparation is disabled; use the catalog-bound "
        "prepare-fixed-bank-harp-router-v11-inputs workstation lifecycle."
    )


def _assert_fresh_destinations(
    cache_root: Path, development_path: Path, evaluation_path: Path
) -> None:
    if len({cache_root, development_path, evaluation_path}) != 3:
        raise ProtocolError("HARP preparation output destinations overlap.")
    if cache_root.exists() or cache_root.is_symlink():
        raise ProtocolError("HARP prepared cache destination already exists.")
    for path in (development_path, evaluation_path):
        try:
            path.relative_to(cache_root)
        except ValueError:
            pass
        else:
            raise ProtocolError("HARP role manifests must remain outside the cache root.")
        if path.exists() or path.is_symlink():
            raise ProtocolError("HARP role manifest destination already exists.")


def _safe_member(root: Path, relative: str) -> Path:
    """Compatibility seam retained for callers of the former monolith."""

    return safe_existing_member(root, relative, role="canonical cache")


__all__ = (
    "CanonicalFrameRow",
    "CanonicalLabelBlindCacheIdentity",
    "CanonicalLabelBlindFrame",
    "HarpV11PreparedInputs",
    "V11_PREPARATION_IDENTITY",
    "build_case_partition_payload",
    "deterministic_case_partition",
    "load_canonical_label_blind_cache",
    "load_canonical_source_train_label_blind_cache",
    "load_canonical_target_test_label_blind_cache",
    "validate_canonical_label_blind_cache_identity",
)
