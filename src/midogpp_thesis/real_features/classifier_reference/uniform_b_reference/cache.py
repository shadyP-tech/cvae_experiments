"""Standardize the reviewed B train shards into one canonical train cache."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.features.cache_io import load_cache_rows

from ..protocol import ProtocolError
from ..real_feature_frame import load_midogpp_real_feature_frame
from .config import (
    EXPECTED_FEATURE_DIM,
    EXPECTED_TRAIN_ROWS,
    REPRESENTATION_ID,
    UniformBCanonicalCacheConfig,
)


def build_uniform_b_canonical_train_cache(
    config: UniformBCanonicalCacheConfig,
) -> Path:
    config.cache_root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(config.cache_root) as stage:
        staged = replace(config, cache_root=stage)
        _build_in_place(staged)
        validate_uniform_b_canonical_train_cache(
            stage, expected_config=config, allow_pending=True
        )
        _finalize(stage, config)
        validate_uniform_b_canonical_train_cache(stage, expected_config=config)
    return config.cache_root


def _build_in_place(config: UniformBCanonicalCacheConfig) -> None:
    import numpy as np
    import torch

    manifest = _read_manifest(config.manifest_path)
    eligible = set(config.eligible_centers)
    train_rows = [
        row
        for row in manifest
        if str(row.get("split", "")).lower() == "train"
        and str(row.get("center", "")) in eligible
    ]
    manifest_by_id = {str(row["sample_id"]): row for row in train_rows}
    manifest_order = {
        str(row["sample_id"]): index
        for index, row in enumerate(manifest)
        if str(row.get("sample_id", ""))
    }
    if len(train_rows) != config.expected_train_rows or len(manifest_by_id) != len(train_rows):
        raise ProtocolError("Uniform-B canonical train manifest coverage drifted.")

    embeddings: list[Any] = []
    metadata: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    for center in config.eligible_centers:
        path = (
            config.source_b_cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
        shard = load_cache_rows(path, expected_dim=config.expected_feature_dim)
        if any(
            str(row.get("center")) != center or str(row.get("split")) != "train"
            for row in shard.metadata
        ):
            raise ProtocolError(f"Uniform-B canonical source shard drifted: {center}.")
        embeddings.append(np.asarray(shard.embeddings, dtype=np.float32))
        metadata.extend(dict(row) for row in shard.metadata)
        source_hashes[center] = _sha256_file(path)
    if len(metadata) != config.expected_train_rows:
        raise ProtocolError("Uniform-B canonical source shard coverage drifted.")
    sample_ids = [str(row.get("sample_id", "")) for row in metadata]
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(manifest_by_id):
        raise ProtocolError("Uniform-B canonical source/manifest identities differ.")
    for row in metadata:
        source = manifest_by_id[str(row["sample_id"])]
        if (
            int(row["label"]) != int(float(str(source["label"])))
            or str(row["center"]) != str(source["center"])
            or str(row.get("case_id", "")) != str(source["case_id"])
        ):
            raise ProtocolError("Uniform-B canonical source metadata differs from manifest.")

    matrix = np.concatenate(embeddings, axis=0)
    order = np.asarray([manifest_order[sample_id] for sample_id in sample_ids])
    permutation = np.argsort(order, kind="stable")
    ordered_metadata = [metadata[int(index)] for index in permutation]
    ordered_matrix = matrix[permutation]
    frozen = {
        "schema_version": "midogpp_uniform_b_canonical_cache_protocol_v1",
        "cache_name": config.name,
        "representation_id": REPRESENTATION_ID,
        "transformation": "lossless_center_shard_concatenation_in_manifest_order",
        "split": "train",
        "eligible_centers": list(config.eligible_centers),
        "row_count": config.expected_train_rows,
        "feature_dim": config.expected_feature_dim,
        "source_shard_sha256": source_hashes,
        "labels_used_for_feature_construction": False,
        "test_rows_present": False,
        "validation_rows_present": False,
    }
    frozen["protocol_hash"] = stable_hash(frozen)
    _write_json(config.cache_root / "manifests/frozen_cache_protocol.json", frozen)
    output_metadata = [
        {
            "sample_id": str(row["sample_id"]),
            "case_id": str(row["case_id"]),
            "label": int(row["label"]),
            "split": "train",
            "center": str(row["center"]),
            "contract_row_index": int(manifest_order[str(row["sample_id"])]),
        }
        for row in ordered_metadata
    ]
    output = config.cache_root / "embeddings/train.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.from_numpy(ordered_matrix.copy()).float(),
            "metadata": output_metadata,
            "feature_extractor": {
                "schema_version": "midogpp_uniform_b_canonical_feature_extractor_v1",
                "model": "Virchow2",
                "dataset": "MIDOG++",
                "representation_id": REPRESENTATION_ID,
                "feature_dim": EXPECTED_FEATURE_DIM,
                "pooling": "fixed_center_rows6to9_cols6to9",
                "source_protocol_hash": frozen["protocol_hash"],
            },
        },
        output,
    )
    _write_json(
        config.cache_root / "reports/cache_builder_report.json",
        {
            "schema_version": "midogpp_uniform_b_canonical_cache_builder_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "representation_id": REPRESENTATION_ID,
            "split": "train",
            "row_count": len(output_metadata),
            "feature_dim": int(ordered_matrix.shape[1]),
            "source_shards": len(source_hashes),
            "numeric_transformation": "none",
        },
    )
    _write_content_index(config.cache_root)


def validate_uniform_b_canonical_train_cache(
    root: str | Path,
    *,
    expected_config: UniformBCanonicalCacheConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    import numpy as np

    path = Path(root)
    required = {
        "embeddings/train.pt",
        "manifests/frozen_cache_protocol.json",
        "manifests/content_index.json",
        "reports/cache_builder_report.json",
    }
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B canonical cache is incomplete: {missing}.")
    frozen = _read_json(path / "manifests/frozen_cache_protocol.json")
    report = _read_json(path / "reports/cache_builder_report.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    unhashed = {key: value for key, value in frozen.items() if key != "protocol_hash"}
    if (
        stable_hash(unhashed) != frozen.get("protocol_hash")
        or frozen.get("representation_id") != REPRESENTATION_ID
        or frozen.get("row_count") != EXPECTED_TRAIN_ROWS
        or frozen.get("feature_dim") != EXPECTED_FEATURE_DIM
        or frozen.get("labels_used_for_feature_construction") is not False
        or frozen.get("test_rows_present") is not False
        or report.get("status") != expected_status
        or report.get("numeric_transformation") != "none"
    ):
        raise ProtocolError("Uniform-B canonical cache protocol failed.")
    frame = load_midogpp_real_feature_frame(
        manifest_path=expected_config.manifest_path,
        feature_cache_path=path / "embeddings/train.pt",
        expected_feature_dim=EXPECTED_FEATURE_DIM,
        allow_excluded_center_omission=True,
    )
    if (
        len(frame.rows) != EXPECTED_TRAIN_ROWS
        or frame.eligible_centers != expected_config.eligible_centers
    ):
        raise ProtocolError("Uniform-B canonical frame coverage drifted.")

    observed_by_id = {
        row.sample_id: np.asarray(frame.embeddings[index], dtype=np.float32)
        for index, row in enumerate(frame.rows)
    }
    source_hashes = frozen.get("source_shard_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ProtocolError("Uniform-B canonical source hash index is invalid.")
    for center in expected_config.eligible_centers:
        source = (
            expected_config.source_b_cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
        if source_hashes.get(center) != _sha256_file(source):
            raise ProtocolError(f"Uniform-B canonical source hash drifted: {center}.")
        shard = load_cache_rows(source, expected_dim=EXPECTED_FEATURE_DIM)
        for index, row in enumerate(shard.metadata):
            sample_id = str(row["sample_id"])
            if not np.array_equal(
                observed_by_id[sample_id],
                np.asarray(shard.embeddings[index], dtype=np.float32),
            ):
                raise ProtocolError("Uniform-B canonical cache changed source feature values.")
    _validate_content_index(path)
    checks = {
        "status": "PASS",
        "row_count": EXPECTED_TRAIN_ROWS,
        "center_count": len(expected_config.eligible_centers),
        "feature_dim": EXPECTED_FEATURE_DIM,
        "numeric_identity": "EXACT",
    }
    if not allow_pending:
        validation = _read_json(path / "reports/validation_report.json")
        if validation.get("status") != "PASS" or validation.get("checks") != checks:
            raise ProtocolError("Uniform-B canonical cache validation report failed.")
    return checks


def _finalize(root: Path, config: UniformBCanonicalCacheConfig) -> None:
    checks = validate_uniform_b_canonical_train_cache(
        root, expected_config=config, allow_pending=True
    )
    report_path = root / "reports/cache_builder_report.json"
    report = _read_json(report_path)
    report["status"] = "PASS"
    report["independent_validation_status"] = "PASS"
    _write_json(report_path, report)
    _write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_canonical_cache_validation_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_canonical_train_cache",
            "checks": checks,
        },
    )
    _write_content_index(root)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Uniform-B canonical JSON must be an object: {path}.")
    return payload


def _write_content_index(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    payload = {
        "schema_version": "midogpp_uniform_b_canonical_cache_content_index_v1",
        "files": rows,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(root / "manifests/content_index.json", payload)


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if stable_hash(unhashed) != payload.get("content_hash"):
        raise ProtocolError("Uniform-B canonical cache content hash drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Uniform-B canonical cache content index is invalid.")
    expected = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Uniform-B canonical cache index row is invalid.")
        relative = str(row.get("path", ""))
        member = root / relative
        if not member.is_file() or _sha256_file(member) != row.get("sha256"):
            raise ProtocolError(f"Uniform-B canonical cache member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Uniform-B canonical cache index coverage drifted.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
