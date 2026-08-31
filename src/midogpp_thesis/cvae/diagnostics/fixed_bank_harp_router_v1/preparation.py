"""Prepare role-pure consumed-test inputs for terminal HARP sensitivity.

The case partition is derived solely from label-blind cache identities.  The
canonical scoring manifest is not opened until the new cache, row index, and
partition receipt have been flushed and independently reloaded.  Preparation
never creates or issues an execution amendment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_file
from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)
from .input_surfaces import (
    CACHE_INDEX,
    CACHE_ROWS,
    CONTENT_INDEX,
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpCacheRow,
    HarpConsumedCacheIdentity,
    V1_CACHE_IDENTITY,
    _read_label_manifest,
    load_cache_index,
)


CANONICAL_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
CANONICAL_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
CANONICAL_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
CANONICAL_PARENT_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
CANONICAL_CACHE_NAME = "uniform_b_v2_descriptive_test_cache_v1"
CANONICAL_REPRESENTATION = "annotation_jpeg_fixed_center_b_v3"
CANONICAL_EXPERT_BANK_LOCK_HASH = "9972a41dcd4814cd"
CANONICAL_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_ROW_COUNT = 9928
EXPECTED_CASE_COUNT = 218
EXPECTED_ROWS_BY_CENTER = {
    "0": 1532,
    "1": 866,
    "2": 3210,
    "3": 1278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}
EXPECTED_CASES_BY_CENTER = {
    "0": 23,
    "1": 20,
    "2": 24,
    "3": 39,
    "5": 23,
    "6": 23,
    "7": 21,
    "8": 22,
    "9": 23,
}
PARTITION_NAMESPACE = "midogpp_harp_consumed_test_case_partition_v1"
PREPARATION_RECEIPT = Path("manifests/harp_consumed_test_preparation_receipt.json")
LABEL_FREE_BARRIER = Path("manifests/label_free_partition_barrier.json")
LABEL_FREE_CONTENT_INDEX = Path("manifests/label_free_content_index.json")
CASE_PARTITION = Path("manifests/case_partition.json")
_METADATA_FIELDS = {
    "evaluation_row_id",
    "contract_row_index",
    "case_id",
    "center",
    "split",
}
_LEGACY_LABEL = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HarpPreparationIdentity:
    """Closed execution-revision identity for deterministic cache preparation."""

    experiment_id: str
    publication_status: str
    terminal_decision: str
    prepared_inputs_schema: str
    partition_schema: str
    preparation_receipt_schema: str
    label_free_barrier_schema: str
    cache_identity: HarpConsumedCacheIdentity
    preparation_receipt: Path = PREPARATION_RECEIPT
    label_free_barrier: Path = LABEL_FREE_BARRIER
    label_free_content_index: Path = LABEL_FREE_CONTENT_INDEX
    case_partition: Path = CASE_PARTITION


V1_PREPARATION_IDENTITY = HarpPreparationIdentity(
    experiment_id=EXPERIMENT_ID,
    publication_status=PUBLICATION_STATUS,
    terminal_decision=TERMINAL_DECISION,
    prepared_inputs_schema="midogpp_harp_consumed_test_prepared_inputs_v1",
    partition_schema="midogpp_harp_consumed_test_case_partition_v1",
    preparation_receipt_schema="midogpp_harp_consumed_test_preparation_receipt_v1",
    label_free_barrier_schema="midogpp_harp_consumed_test_label_free_barrier_v1",
    cache_identity=V1_CACHE_IDENTITY,
)


@dataclass(frozen=True, slots=True)
class HarpPreparedInputData:
    """Identity-neutral prepared-input receipt fields shared by v1 and v2."""

    cache_root: Path
    development_manifest_path: Path
    evaluation_manifest_path: Path
    cache_content_sha256: str
    development_manifest_sha256: str
    evaluation_manifest_sha256: str
    parent_ledger_sha256: str
    partition_hash: str
    preparation_receipt_hash: str


@dataclass(frozen=True, slots=True)
class CanonicalFrameRow:
    center: str
    case_id: str
    sample_id: str
    contract_row_index: int
    center_row_index: int


@dataclass(frozen=True, slots=True)
class CanonicalLabelBlindFrame:
    rows_by_center: Mapping[str, tuple[CanonicalFrameRow, ...]]
    embeddings_by_center: Mapping[str, np.ndarray]
    cache_content_hash: str
    row_order_hash: str
    source_member_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class HarpPreparedInputs:
    cache_root: Path
    development_manifest_path: Path
    evaluation_manifest_path: Path
    cache_content_sha256: str
    development_manifest_sha256: str
    evaluation_manifest_sha256: str
    parent_ledger_sha256: str
    partition_hash: str
    preparation_receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        amendment_binding = authorization_input_binding_payload(
            expert_bank_lock_hash=CANONICAL_EXPERT_BANK_LOCK_HASH,
            generation_lock_hash=CANONICAL_GENERATION_LOCK_HASH,
            test_cache_content_sha256=self.cache_content_sha256,
            development_manifest_sha256=self.development_manifest_sha256,
            evaluation_manifest_sha256=self.evaluation_manifest_sha256,
            parent_ledger_sha256=self.parent_ledger_sha256,
        )
        return {
            "schema_version": "midogpp_harp_consumed_test_prepared_inputs_v1",
            "cache_root": str(self.cache_root),
            "development_manifest_path": str(self.development_manifest_path),
            "evaluation_manifest_path": str(self.evaluation_manifest_path),
            "test_cache_content_sha256": self.cache_content_sha256,
            "development_manifest_sha256": self.development_manifest_sha256,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "partition_hash": self.partition_hash,
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "proposed_amendment_input_binding": amendment_binding,
            "execution_amendment_created": False,
            "execution_authorized": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }


def deterministic_case_partition(
    rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
) -> Mapping[tuple[str, str], str]:
    """Assign whole cases without accepting labels or manifest content."""

    output: dict[tuple[str, str], str] = {}
    if tuple(rows_by_center) != CENTERS:
        raise ProtocolError("HARP preparation center universe drifted.")
    for center in CENTERS:
        cases = tuple(sorted({row.case_id for row in rows_by_center[center]}))
        if len(cases) < 2:
            raise ProtocolError("HARP preparation needs at least two cases per center.")
        ranked = tuple(
            sorted(
                cases,
                key=lambda case: (
                    hashlib.sha256(
                        f"{PARTITION_NAMESPACE}\0{center}\0{case}".encode("utf-8")
                    ).hexdigest(),
                    case,
                ),
            )
        )
        development_count = len(ranked) // 2
        development = set(ranked[:development_count])
        for case in cases:
            output[(center, case)] = (
                DEVELOPMENT_ROLE if case in development else EVALUATION_ROLE
            )
    if set(output.values()) != {DEVELOPMENT_ROLE, EVALUATION_ROLE}:
        raise ProtocolError("HARP preparation case partition is incomplete.")
    return output


def build_case_partition_payload(
    rows_by_center: Mapping[str, Sequence[CanonicalFrameRow]],
    *,
    identity: HarpPreparationIdentity = V1_PREPARATION_IDENTITY,
) -> tuple[Mapping[tuple[str, str], str], dict[str, object], str]:
    """Build the byte-stable label-free partition contract for one revision."""

    partition = deterministic_case_partition(rows_by_center)
    payload: dict[str, object] = {
        "schema_version": identity.partition_schema,
        "namespace": PARTITION_NAMESPACE,
        "assignments": [
            {"center": center, "case_id": case, "split_role": role}
            for (center, case), role in sorted(partition.items())
        ],
        "whole_case_disjoint": True,
        "label_values_available": False,
        "canonical_scoring_manifest_opened": False,
    }
    return partition, payload, canonical_hash(payload)


def prepare_harp_consumed_test_inputs_with_identity(
    *,
    canonical_cache_root: str | Path,
    canonical_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    identity: HarpPreparationIdentity,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_parent_ledger_sha256: str = CANONICAL_PARENT_LEDGER_SHA256,
) -> HarpPreparedInputData:
    """Materialize the terminal HARP cache and role-pure label capabilities."""

    destination = Path(cache_root).resolve()
    development_path = Path(development_manifest_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    _assert_fresh_destinations(destination, development_path, evaluation_path)
    ledger = Path(parent_ledger_path).resolve()
    if (
        not ledger.is_file()
        or ledger.is_symlink()
        or sha256_file(ledger) != expected_parent_ledger_sha256
    ):
        raise ProtocolError("HARP preparation parent ledger is absent or drifted.")
    read_json(ledger)

    # No scoring manifest path is passed into this loader.
    frame = load_canonical_label_blind_cache(Path(canonical_cache_root))
    partition, partition_payload, partition_hash = build_case_partition_payload(
        frame.rows_by_center,
        identity=identity,
    )
    prepared_rows = _persist_label_blind_cache(
        destination,
        frame=frame,
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

    # This is the first operation allowed to open the canonical scoring
    # manifest.  The case assignment above is already immutable and validated.
    development_sha, evaluation_sha = _publish_role_pure_manifests(
        Path(canonical_manifest_path),
        expected_manifest_sha256=expected_manifest_sha256,
        cache=cache,
        frame=frame,
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
    _read_label_manifest(
        evaluation_path,
        expected_sha256=evaluation_sha,
        expected_role=EVALUATION_ROLE,
        cache=cache,
    )
    receipt_base: dict[str, object] = {
        "schema_version": identity.preparation_receipt_schema,
        "experiment_id": identity.experiment_id,
        "status": "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY",
        "canonical_cache_content_hash": frame.cache_content_hash,
        "canonical_cache_row_order_hash": frame.row_order_hash,
        "canonical_manifest_sha256": expected_manifest_sha256,
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
        "development_and_evaluation_cases_disjoint": True,
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


def prepare_harp_consumed_test_inputs(
    *,
    canonical_cache_root: str | Path,
    canonical_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_parent_ledger_sha256: str = CANONICAL_PARENT_LEDGER_SHA256,
) -> HarpPreparedInputs:
    """Materialize the v1 terminal HARP cache and label capabilities."""

    prepared = prepare_harp_consumed_test_inputs_with_identity(
        canonical_cache_root=canonical_cache_root,
        canonical_manifest_path=canonical_manifest_path,
        parent_ledger_path=parent_ledger_path,
        cache_root=cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        identity=V1_PREPARATION_IDENTITY,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_parent_ledger_sha256=expected_parent_ledger_sha256,
    )
    return HarpPreparedInputs(
        cache_root=prepared.cache_root,
        development_manifest_path=prepared.development_manifest_path,
        evaluation_manifest_path=prepared.evaluation_manifest_path,
        cache_content_sha256=prepared.cache_content_sha256,
        development_manifest_sha256=prepared.development_manifest_sha256,
        evaluation_manifest_sha256=prepared.evaluation_manifest_sha256,
        parent_ledger_sha256=prepared.parent_ledger_sha256,
        partition_hash=prepared.partition_hash,
        preparation_receipt_hash=prepared.preparation_receipt_hash,
    )


def load_canonical_label_blind_cache(root: Path) -> CanonicalLabelBlindFrame:
    """Authenticate and load the canonical consumed-test PT shards."""

    if root.is_symlink():
        raise ProtocolError("HARP canonical cache root is unsafe.")
    try:
        cache_root = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("HARP canonical cache root is absent.") from exc
    if not cache_root.is_dir():
        raise ProtocolError("HARP canonical cache root is unsafe.")
    content = read_json(cache_root / "manifests/content_index.json")
    files = content.get("files")
    content_base = {key: value for key, value in content.items() if key != "content_hash"}
    if (
        set(content) != {"schema_version", "files", "content_hash"}
        or not isinstance(files, list)
        or content.get("content_hash") != canonical_hash(content_base)
        or content.get("content_hash") != CANONICAL_CACHE_CONTENT_HASH
    ):
        raise ProtocolError("HARP canonical cache content index drifted.")
    indexed: dict[str, str] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ProtocolError("HARP canonical cache content member is malformed.")
        relative = str(row["path"])
        member = _safe_member(cache_root, relative)
        if relative in indexed or sha256_file(member) != row["sha256"]:
            raise ProtocolError("HARP canonical cache member bytes drifted.")
        indexed[relative] = str(row["sha256"])
    actual = {
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*")
        if path.is_file()
        and path.relative_to(cache_root).as_posix() != "manifests/content_index.json"
    }
    if actual != set(indexed):
        raise ProtocolError("HARP canonical cache closed-world inventory drifted.")
    frozen = read_json(cache_root / "manifests/frozen_build_protocol.json")
    alignment = read_json(cache_root / "manifests/row_alignment.json")
    report = read_json(cache_root / "reports/cache_builder_report.json")
    validation = read_json(cache_root / "reports/validation_report.json")
    extractor = frozen.get("cache_extractor_protocol")
    if (
        not isinstance(extractor, Mapping)
        or frozen.get("cache_name") != CANONICAL_CACHE_NAME
        or extractor.get("representation_id") != CANONICAL_REPRESENTATION
        or frozen.get("scoring_manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or alignment.get("row_order_hash") != CANONICAL_CACHE_ROW_ORDER_HASH
        or report.get("row_order_hash") != CANONICAL_CACHE_ROW_ORDER_HASH
        or report.get("row_count") != EXPECTED_ROW_COUNT
        or report.get("fresh_evidence") is not False
        or validation.get("status") != "PASS"
    ):
        raise ProtocolError("HARP canonical cache protocol drifted.")
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP cache preparation requires torch.") from exc
    rows_by_center: dict[str, tuple[CanonicalFrameRow, ...]] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        relative = f"embeddings/by_center/center_{center}.pt"
        path = _safe_member(cache_root, relative)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old workstation torch
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise ProtocolError("HARP canonical cache shard is unreadable.") from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "embeddings",
            "metadata",
            "feature_extractor",
        }:
            raise ProtocolError("HARP canonical cache shard schema drifted.")
        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, Sequence) or isinstance(
            raw_metadata, (str, bytes)
        ):
            raise ProtocolError("HARP canonical cache metadata is malformed.")
        center_rows: list[CanonicalFrameRow] = []
        for ordinal, raw in enumerate(raw_metadata):
            if not isinstance(raw, Mapping) or {str(key) for key in raw} != _METADATA_FIELDS:
                raise ProtocolError("HARP canonical cache metadata firewall failed.")
            sample = str(raw["evaluation_row_id"])
            if (
                not sample.startswith("eval_")
                or len(sample) != 69
                or _LEGACY_LABEL.search(sample)
                or str(raw["center"]) != center
                or str(raw["split"]) != "test"
                or not str(raw["case_id"])
                or type(raw["contract_row_index"]) is not int
                or int(raw["contract_row_index"]) < 0
            ):
                raise ProtocolError("HARP canonical cache row identity drifted.")
            center_rows.append(
                CanonicalFrameRow(
                    center=center,
                    case_id=str(raw["case_id"]),
                    sample_id=sample,
                    contract_row_index=int(raw["contract_row_index"]),
                    center_row_index=ordinal,
                )
            )
        values = np.ascontiguousarray(
            torch.as_tensor(payload["embeddings"]).detach().cpu().float().numpy(),
            dtype=np.float32,
        )
        if (
            values.shape != (len(center_rows), COMMON_OUTPUT_DIM)
            or len(center_rows) != EXPECTED_ROWS_BY_CENTER[center]
            or len({row.case_id for row in center_rows})
            != EXPECTED_CASES_BY_CENTER[center]
            or not np.isfinite(values).all()
        ):
            raise ProtocolError("HARP canonical cache shard geometry drifted.")
        rows_by_center[center] = tuple(center_rows)
        embeddings_by_center[center] = values
    rows = tuple(row for center in CENTERS for row in rows_by_center[center])
    if (
        len(rows) != EXPECTED_ROW_COUNT
        or len({(row.center, row.case_id) for row in rows}) != EXPECTED_CASE_COUNT
        or len({row.sample_id for row in rows}) != EXPECTED_ROW_COUNT
        or len({row.contract_row_index for row in rows}) != EXPECTED_ROW_COUNT
    ):
        raise ProtocolError("HARP canonical cache global geometry drifted.")
    return CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash=CANONICAL_CACHE_CONTENT_HASH,
        row_order_hash=CANONICAL_CACHE_ROW_ORDER_HASH,
        source_member_sha256=indexed,
    )


def _persist_label_blind_cache(
    root: Path,
    *,
    frame: CanonicalLabelBlindFrame,
    partition: Mapping[tuple[str, str], str],
    partition_payload: Mapping[str, object],
    partition_hash: str,
    identity: HarpPreparationIdentity,
) -> tuple[HarpCacheRow, ...]:
    root.mkdir(parents=True, exist_ok=False)
    shard_rows: list[dict[str, object]] = []
    shard_hashes: dict[str, str] = {}
    embedding_offset: dict[tuple[str, str], int] = {}
    for center in CENTERS:
        relative = f"embeddings/by_center/center_{center}.npy"
        path = root / relative
        atomic_npy(path, frame.embeddings_by_center[center])
        digest = sha256_file(path)
        shard_hashes[relative] = digest
        shard_rows.append(
            {
                "relative_path": relative,
                "file_sha256": digest,
                "shape": list(frame.embeddings_by_center[center].shape),
                "dtype": "float32",
            }
        )
        for row in frame.rows_by_center[center]:
            embedding_offset[(center, row.sample_id)] = row.center_row_index
    prepared: list[HarpCacheRow] = []
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in CENTERS:
            scoped = tuple(
                row
                for row in frame.rows_by_center[center]
                if partition[(center, row.case_id)] == role
            )
            for split_ordinal, row in enumerate(scoped):
                prepared.append(
                    HarpCacheRow(
                        center=center,
                        case_id=row.case_id,
                        sample_id=row.sample_id,
                        split_role=role,
                        split_row_index=split_ordinal,
                        embedding_file=f"embeddings/by_center/center_{center}.npy",
                        embedding_row_index=embedding_offset[(center, row.sample_id)],
                    )
                )
    if canonical_hash(partition_payload) != partition_hash:
        raise ProtocolError("HARP preparation partition payload drifted before write.")
    atomic_json(root / identity.case_partition, partition_payload)
    _write_cache_rows(
        root / CACHE_ROWS,
        prepared,
        row_schema=identity.cache_identity.row_schema,
    )
    index_base: dict[str, object] = {
        "schema_version": identity.cache_identity.cache_schema,
        "artifact_id": identity.cache_identity.artifact_id,
        "dataset_family": "MIDOG++",
        "representation_id": "midogpp_virchow2_common_3840_float32_v1",
        "feature_dim": COMMON_OUTPUT_DIM,
        "dtype": "float32",
        "labels_stored": False,
        "split_roles": [DEVELOPMENT_ROLE, EVALUATION_ROLE],
        "row_index_member": CACHE_ROWS.as_posix(),
        "shards": shard_rows,
    }
    atomic_json(root / CACHE_INDEX, {**index_base, "cache_index_hash": canonical_hash(index_base)})
    member_hashes = {
        CACHE_INDEX.as_posix(): sha256_file(root / CACHE_INDEX),
        CACHE_ROWS.as_posix(): sha256_file(root / CACHE_ROWS),
        identity.case_partition.as_posix(): sha256_file(
            root / identity.case_partition
        ),
        **shard_hashes,
    }
    barrier_base: dict[str, object] = {
        "schema_version": identity.label_free_barrier_schema,
        "status": "DURABLE_LABEL_FREE_CACHE_BEFORE_SCORING_MANIFEST_OPEN",
        "canonical_cache_content_hash": frame.cache_content_hash,
        "canonical_cache_row_order_hash": frame.row_order_hash,
        "partition_hash": partition_hash,
        "cache_index_sha256": sha256_file(root / CACHE_INDEX),
        "row_index_sha256": sha256_file(root / CACHE_ROWS),
        "case_partition_sha256": sha256_file(root / identity.case_partition),
        "embedding_shard_sha256": dict(sorted(shard_hashes.items())),
        "row_count": len(prepared),
        "case_count": len(partition),
        "whole_case_disjoint": True,
        "labels_stored": False,
        "canonical_scoring_manifest_opened": False,
        "mixed_patch_labels_within_case_supported": True,
    }
    atomic_json(
        root / identity.label_free_barrier,
        {**barrier_base, "barrier_hash": canonical_hash(barrier_base)},
    )
    label_free_members = {
        **member_hashes,
        identity.label_free_barrier.as_posix(): sha256_file(
            root / identity.label_free_barrier
        ),
    }
    _write_content_index(
        root / identity.label_free_content_index,
        label_free_members,
        content_schema=identity.cache_identity.content_schema,
    )
    _write_content_index(
        root / CONTENT_INDEX,
        {
            **label_free_members,
            identity.label_free_content_index.as_posix(): sha256_file(
                root / identity.label_free_content_index
            ),
        },
        content_schema=identity.cache_identity.content_schema,
    )
    return tuple(prepared)


def _independently_validate_label_blind_barrier(
    root: Path,
    *,
    expected_partition_hash: str,
    identity: HarpPreparationIdentity = V1_PREPARATION_IDENTITY,
):
    content = read_json(root / CONTENT_INDEX)
    expected_content_hash = str(content.get("content_index_hash"))
    config = SimpleNamespace(
        resolved_path=lambda role: root if role == "test_cache_root" else None,
        expected_hashes={"test_cache_content_sha256": expected_content_hash},
    )
    cache = load_cache_index(  # type: ignore[arg-type]
        config,
        cache_identity=identity.cache_identity,
    )
    barrier = read_json(root / identity.label_free_barrier)
    partition = read_json(root / identity.case_partition)
    label_free_content = read_json(root / identity.label_free_content_index)
    barrier_base = {key: value for key, value in barrier.items() if key != "barrier_hash"}
    label_free_base = {
        key: value
        for key, value in label_free_content.items()
        if key != "content_index_hash"
    }
    indexed_members = set(content["members"])
    expected_members = {
        CACHE_INDEX.as_posix(),
        CACHE_ROWS.as_posix(),
        identity.case_partition.as_posix(),
        identity.label_free_barrier.as_posix(),
        identity.label_free_content_index.as_posix(),
        *(f"embeddings/by_center/center_{center}.npy" for center in CENTERS),
    }
    label_free_expected = expected_members - {
        identity.label_free_content_index.as_posix()
    }
    assignments = partition.get("assignments")
    partition_roles = {
        (str(row.get("center")), str(row.get("case_id"))): str(
            row.get("split_role")
        )
        for row in assignments
        if isinstance(row, Mapping)
    } if isinstance(assignments, list) else {}
    cache_roles = {(row.center, row.case_id): row.split_role for row in cache.rows}
    if (
        barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or barrier.get("partition_hash") != expected_partition_hash
        or canonical_hash(partition) != expected_partition_hash
        or barrier.get("canonical_scoring_manifest_opened") is not False
        or indexed_members != expected_members
        or label_free_content.get("content_index_hash")
        != canonical_hash(label_free_base)
        or set(label_free_content.get("members", {})) != label_free_expected
        or partition_roles != cache_roles
        or len(cache.rows) != EXPECTED_ROW_COUNT
    ):
        raise ProtocolError("HARP label-free preparation barrier failed validation.")
    return cache


def _publish_role_pure_manifests(
    canonical_manifest: Path,
    *,
    expected_manifest_sha256: str,
    cache,
    frame: CanonicalLabelBlindFrame,
    development_path: Path,
    evaluation_path: Path,
    identity: HarpPreparationIdentity,
) -> tuple[str, str]:
    if (
        not canonical_manifest.is_file()
        or canonical_manifest.is_symlink()
        or sha256_file(canonical_manifest) != expected_manifest_sha256
    ):
        raise ProtocolError("HARP canonical scoring manifest is absent or drifted.")
    try:
        with canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = tuple(dict(row) for row in csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ProtocolError("HARP canonical scoring manifest is unreadable.") from exc
    required_fields = {"case_id", "center", "split", "label"}
    if not raw_rows or not required_fields.issubset(raw_rows[0]):
        raise ProtocolError("HARP canonical scoring manifest schema drifted.")
    by_index = {ordinal: row for ordinal, row in enumerate(raw_rows)}
    source_by_sample = {
        row.sample_id: row
        for center in CENTERS
        for row in frame.rows_by_center[center]
    }
    if len(source_by_sample) != len(cache.rows):
        raise ProtocolError("HARP canonical label-blind identity coverage drifted.")
    barrier = read_json(cache.root / identity.label_free_barrier)
    if barrier.get("canonical_scoring_manifest_opened") is not False:
        raise ProtocolError("HARP scoring manifest opened before the label-free barrier.")

    output_by_role: dict[str, list[tuple[str, str, str, int]]] = {
        DEVELOPMENT_ROLE: [],
        EVALUATION_ROLE: [],
    }
    used_manifest_rows: set[int] = set()
    for row in cache.rows:
        source = source_by_sample.get(row.sample_id)
        if source is None:
            raise ProtocolError("HARP cache/source identity alignment drifted.")
        ordinal = source.contract_row_index
        raw = by_index.get(ordinal)
        if (
            raw is None
            or source.center != row.center
            or source.case_id != row.case_id
            or source.center_row_index != row.embedding_row_index
            or raw.get("split") != "test"
            or str(raw.get("center")) != row.center
            or str(raw.get("case_id")) != row.case_id
            or row.sample_id != _evaluation_row_id(expected_manifest_sha256, ordinal)
            or ordinal in used_manifest_rows
            or str(raw.get("label")) not in {"0", "1"}
        ):
            raise ProtocolError("HARP cache/manifest identity alignment drifted.")
        used_manifest_rows.add(ordinal)
        output_by_role[row.split_role].append(
            (row.center, row.case_id, row.sample_id, int(raw["label"]))
        )
    if len(used_manifest_rows) != len(cache.rows):
        raise ProtocolError("HARP cache/manifest row coverage drifted.")
    for role, path in (
        (DEVELOPMENT_ROLE, development_path),
        (EVALUATION_ROLE, evaluation_path),
    ):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("center", "case_id", "sample_id", "label", "split_role"))
        writer.writerows((*values, role) for values in output_by_role[role])
        _atomic_text(path, buffer.getvalue())
    return sha256_file(development_path), sha256_file(evaluation_path)


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


def _write_cache_rows(
    path: Path,
    rows: Sequence[HarpCacheRow],
    *,
    row_schema: str = V1_CACHE_IDENTITY.row_schema,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "schema_version", "row_id", "center", "case_id", "split_role",
            "split_row_index", "embedding_file", "embedding_row_index",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row_schema, row.sample_id,
                row.center, row.case_id, row.split_role, row.split_row_index,
                row.embedding_file, row.embedding_row_index,
            )
        )
    _atomic_text(path, buffer.getvalue())


def _write_content_index(
    path: Path,
    members: Mapping[str, str],
    *,
    content_schema: str = V1_CACHE_IDENTITY.content_schema,
) -> None:
    base: dict[str, object] = {
        "schema_version": content_schema,
        "members": dict(sorted(members.items())),
    }
    atomic_json(path, {**base, "content_index_hash": canonical_hash(base)})


def _write_final_content_index(
    root: Path,
    *,
    identity: HarpPreparationIdentity = V1_PREPARATION_IDENTITY,
) -> None:
    members = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root) != CONTENT_INDEX
    }
    _write_content_index(
        root / CONTENT_INDEX,
        members,
        content_schema=identity.cache_identity.content_schema,
    )


def _validate_final_prepared_cache(
    root: Path,
    *,
    identity: HarpPreparationIdentity = V1_PREPARATION_IDENTITY,
):
    content = read_json(root / CONTENT_INDEX)
    config = SimpleNamespace(
        resolved_path=lambda role: root if role == "test_cache_root" else None,
        expected_hashes={
            "test_cache_content_sha256": str(content.get("content_index_hash"))
        },
    )
    cache = load_cache_index(  # type: ignore[arg-type]
        config,
        cache_identity=identity.cache_identity,
    )
    if identity.preparation_receipt.as_posix() not in cache.member_sha256:
        raise ProtocolError("HARP final prepared cache lacks its preparation receipt.")
    return cache


def _evaluation_row_id(manifest_sha256: str, contract_row_index: int) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "contract_row_index": contract_row_index,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"eval_{hashlib.sha256(encoded).hexdigest()}"


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ProtocolError("HARP canonical cache member path is unsafe.")
    path = (root / value).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("HARP canonical cache member escaped its root.") from exc
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP canonical cache member is unsafe.")
    return path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _fsync_file(path)
    directories = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
    )
    for path in (*directories, root):
        _fsync_directory(path)


__all__ = (
    "CanonicalFrameRow",
    "CanonicalLabelBlindFrame",
    "HarpPreparationIdentity",
    "HarpPreparedInputData",
    "HarpPreparedInputs",
    "V1_PREPARATION_IDENTITY",
    "build_case_partition_payload",
    "deterministic_case_partition",
    "load_canonical_label_blind_cache",
    "prepare_harp_consumed_test_inputs",
    "prepare_harp_consumed_test_inputs_with_identity",
)
