"""Direct, label-free parser for the immutable MIDOG++ test cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FEATURE_DIM,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .frame import LabelFreeTestFrame, TestRowIdentity


_CACHE_METADATA_FIELDS = {
    "evaluation_row_id",
    "contract_row_index",
    "case_id",
    "center",
    "split",
}
_LEGACY_OUTCOME = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


def load_label_free_test_frame(cache_root: str | Path) -> LabelFreeTestFrame:
    """Load all nine embedding shards without accepting a manifest path."""

    root = _safe_root(cache_root)
    content = _validate_content_index(root)
    if content.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH:
        raise ProtocolError("OE-PPUR v3 test-cache content identity drifted.")
    frozen = _read_json(root / "manifests/frozen_build_protocol.json")
    alignment = _read_json(root / "manifests/row_alignment.json")
    report = _read_json(root / "reports/cache_builder_report.json")
    validation = _read_json(root / "reports/validation_report.json")
    _validate_protocol(frozen, alignment, report, validation)

    rows: list[TestRowIdentity] = []
    embeddings: list[np.ndarray] = []
    by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    shard_hashes: dict[str, str] = {}
    ordinal = 0
    expected_rows = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    for center in CENTERS:
        values, metadata, shard_hash = _load_shard(root, center=center)
        if len(metadata) != expected_rows[center]:
            raise ProtocolError("OE-PPUR v3 cache center row count drifted.")
        center_rows = []
        for row in metadata:
            identity = TestRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=int(row["contract_row_index"]),
                sample_id=str(row["evaluation_row_id"]),
                case_id=str(row["case_id"]),
                center=center,
            )
            rows.append(identity)
            center_rows.append(identity)
            ordinal += 1
        embeddings.append(values)
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard_hash
    expected_cases = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    cases = {
        center: tuple(sorted({row.case_id for row in by_center[center]}))
        for center in CENTERS
    }
    all_cases = tuple((center, case) for center in CENTERS for case in cases[center])
    if (
        len(rows) != EXPECTED_TEST_ROW_COUNT
        or any(len(cases[center]) != expected_cases[center] for center in CENTERS)
        or len(all_cases) != EXPECTED_CASE_COUNT
        or len(set(all_cases)) != EXPECTED_CASE_COUNT
    ):
        raise ProtocolError("OE-PPUR v3 test-cache case inventory drifted.")
    binding = {
        "schema_version": "oe_ppur_v3_direct_test_cache_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "manifest_alias_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "underlying_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "representation_id": EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "shard_sha256_by_center": shard_hashes,
        "row_count": EXPECTED_TEST_ROW_COUNT,
        "case_count": EXPECTED_CASE_COUNT,
        "labels_persisted": False,
        "manifest_opened": False,
        "sample_paths_persisted": False,
        "fresh_evidence": False,
    }
    return LabelFreeTestFrame(
        np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32),
        tuple(rows),
        by_center,
        binding,
    )


def _validate_content_index(root: Path) -> dict[str, object]:
    payload = _read_json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "files", "content_hash"}:
        raise ProtocolError("OE-PPUR v3 cache content-index schema drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != canonical_hash(unhashed):
        raise ProtocolError("OE-PPUR v3 cache content-index hash drifted.")
    raw = payload.get("files")
    if not isinstance(raw, list):
        raise ProtocolError("OE-PPUR v3 cache content-index rows are absent.")
    indexed: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ProtocolError("OE-PPUR v3 cache content-index row drifted.")
        relative = str(row["path"])
        member = _safe_member(root, relative)
        if relative in indexed or _sha256_file(member) != row["sha256"]:
            raise ProtocolError("OE-PPUR v3 cache content-index member drifted.")
        indexed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "manifests/content_index.json"
    }
    if indexed != actual:
        raise ProtocolError("OE-PPUR v3 cache member coverage drifted.")
    return payload


def _validate_protocol(
    frozen: Mapping[str, object],
    alignment: Mapping[str, object],
    report: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    extractor = frozen.get("cache_extractor_protocol")
    if (
        not isinstance(extractor, Mapping)
        or frozen.get("cache_name") != EXPECTED_TEST_CACHE_SEMANTIC_ID
        or extractor.get("representation_id") != EXPECTED_TEST_CACHE_REPRESENTATION_ID
        or frozen.get("scoring_manifest_sha256") != EXPECTED_TEST_MANIFEST_SHA256
        or alignment.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or report.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or report.get("row_count") != EXPECTED_TEST_ROW_COUNT
        or report.get("fresh_evidence") is not False
        or validation.get("status") != "PASS"
    ):
        raise ProtocolError("OE-PPUR v3 test-cache protocol drifted.")


def _load_shard(
    root: Path, *, center: str
) -> tuple[np.ndarray, tuple[dict[str, object], ...], str]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("OE-PPUR v3 cache loading requires torch.") from exc
    path = _safe_member(root, f"embeddings/by_center/center_{center}.pt")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old workstation torch
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ProtocolError("OE-PPUR v3 test-cache shard is unreadable.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "embeddings",
        "metadata",
        "feature_extractor",
    }:
        raise ProtocolError("OE-PPUR v3 test-cache shard schema drifted.")
    raw = payload.get("metadata")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ProtocolError("OE-PPUR v3 test-cache metadata is malformed.")
    rows = []
    for value in raw:
        if not isinstance(value, Mapping) or {str(key) for key in value} != _CACHE_METADATA_FIELDS:
            raise ProtocolError("OE-PPUR v3 test-cache metadata firewall failed.")
        row = {str(key): nested for key, nested in value.items()}
        row_id = str(row["evaluation_row_id"])
        if (
            not row_id.startswith("eval_")
            or len(row_id) != 69
            or str(row["center"]) != center
            or str(row["split"]) != "test"
            or not str(row["case_id"])
            or _LEGACY_OUTCOME.search(row_id)
            or type(row["contract_row_index"]) is not int
            or int(row["contract_row_index"]) < 0
        ):
            raise ProtocolError("OE-PPUR v3 test-cache row identity drifted.")
        rows.append(row)
    indices = tuple(int(row["contract_row_index"]) for row in rows)
    row_ids = tuple(str(row["evaluation_row_id"]) for row in rows)
    if (
        indices != tuple(sorted(indices))
        or len(set(indices)) != len(indices)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise ProtocolError("OE-PPUR v3 test-cache row ordering drifted.")
    array = np.ascontiguousarray(
        torch.as_tensor(payload["embeddings"]).detach().cpu().float().numpy(),
        dtype=np.float32,
    )
    if array.shape != (len(rows), FEATURE_DIM) or not np.isfinite(array).all():
        raise ProtocolError("OE-PPUR v3 test-cache embedding geometry drifted.")
    return array, tuple(rows), _sha256_file(path)


def _safe_root(value: str | Path) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    _reject_symlink_chain(candidate)
    path = Path(value)
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 test-cache root is absent.") from exc
    if root != candidate or not root.is_dir() or root.is_symlink():
        raise ProtocolError("OE-PPUR v3 test-cache root is unsafe.")
    return root


def _safe_member(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise ProtocolError("OE-PPUR v3 cache member path is unsafe.")
    candidate = root / path
    _reject_symlink_chain(candidate, stop=root)
    try:
        member = candidate.resolve(strict=True)
        member.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 cache member escaped its root.") from exc
    if member == root or member.is_symlink() or not member.is_file():
        raise ProtocolError("OE-PPUR v3 cache member is unsafe.")
    return member


def _reject_symlink_chain(path: Path, *, stop: Path | None = None) -> None:
    current = path
    boundary = None if stop is None else Path(stop)
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 cache path contains a symlink.")
        if boundary is not None and current == boundary:
            return
        if current == current.parent:
            return
        current = current.parent


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 JSON member is absent or unsafe.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ProtocolError("OE-PPUR v3 JSON member is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v3 JSON member must contain an object.")
    return payload


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 hashed member is absent or unsafe.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("load_label_free_test_frame",)
