"""Strict shard and content-index I/O for the Stage-70 descriptive cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    TargetEvaluationRow,
    semantic_sha256,
)

from .contracts import (
    FEATURE_DIM,
    FORBIDDEN_METADATA_FIELDS,
    LEGACY_OUTCOME_PATTERN,
    SHARD_METADATA_FIELDS,
    Stage70TestCacheError,
)


@dataclass(frozen=True)
class Stage70CenterShard:
    embeddings: Any
    metadata: tuple[dict[str, object], ...]
    feature_extractor: dict[str, object]
    shard_sha256: str

    @property
    def evaluation_row_ids(self) -> tuple[str, ...]:
        return tuple(str(row["evaluation_row_id"]) for row in self.metadata)

    @property
    def contract_row_indices(self) -> tuple[int, ...]:
        return tuple(int(row["contract_row_index"]) for row in self.metadata)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(str(row["case_id"]) for row in self.metadata)


@dataclass(frozen=True)
class ValidatedStage70TestCache:
    root: Path
    summary: Mapping[str, object]

    def load_center(self, center: str) -> Stage70CenterShard:
        rendered = str(center)
        if rendered not in ELIGIBLE_CENTERS:
            raise Stage70TestCacheError(
                f"Stage-70 cache center is ineligible: {rendered}."
            )
        return load_stage70_center_shard(
            self.root / "embeddings" / "by_center" / f"center_{rendered}.pt",
            expected_center=rendered,
        )


def write_stage70_center_shard(
    path: str | Path,
    *,
    embeddings: object,
    metadata: Sequence[Mapping[str, object]],
    feature_extractor: Mapping[str, object],
) -> None:
    """Write a shard only after exact metadata and tensor validation."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Stage-70 cache writing requires torch.") from exc
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Stage-70 cache shard: {output}.")
    rows = tuple(_validated_metadata_row(row) for row in metadata)
    tensor = torch.as_tensor(embeddings).detach().cpu().float()
    if tuple(tensor.shape) != (len(rows), FEATURE_DIM) or not bool(
        torch.isfinite(tensor).all()
    ):
        raise Stage70TestCacheError(
            "Stage-70 cache shard embedding geometry/finiteness drifted."
        )
    if not isinstance(feature_extractor, Mapping):
        raise Stage70TestCacheError(
            "Stage-70 cache shard extractor identity is missing."
        )
    scan_forbidden_metadata(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": tensor,
            "metadata": [dict(row) for row in rows],
            "feature_extractor": dict(feature_extractor),
        },
        output,
    )


def load_stage70_center_shard(
    path: str | Path,
    *,
    expected_center: str | None = None,
) -> Stage70CenterShard:
    """Load one shard and reject outcome, source-location, or identity drift."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Stage-70 cache loading requires torch.") from exc
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Stage70TestCacheError(
            f"Stage-70 cache shard is missing or unsafe: {source}."
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(source, map_location="cpu")
    except Exception as exc:
        raise Stage70TestCacheError(
            f"Stage-70 cache shard is unreadable: {source}."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "embeddings",
        "metadata",
        "feature_extractor",
    }:
        raise Stage70TestCacheError("Stage-70 cache shard payload schema drifted.")
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, (str, bytes)) or not isinstance(
        raw_metadata, Sequence
    ):
        raise Stage70TestCacheError("Stage-70 cache shard metadata is invalid.")
    metadata = tuple(_validated_metadata_row(row) for row in raw_metadata)
    scan_forbidden_metadata(metadata)
    centers = {str(row["center"]) for row in metadata}
    if expected_center is not None and centers != {str(expected_center)}:
        raise Stage70TestCacheError(
            f"Stage-70 cache shard contains rows outside center {expected_center}."
        )
    row_ids = tuple(str(row["evaluation_row_id"]) for row in metadata)
    row_indices = tuple(int(row["contract_row_index"]) for row in metadata)
    if (
        len(row_ids) != len(set(row_ids))
        or len(row_indices) != len(set(row_indices))
        or row_indices != tuple(sorted(row_indices))
    ):
        raise Stage70TestCacheError(
            "Stage-70 cache shard row identities are duplicated or out of order."
        )
    embeddings = torch.as_tensor(payload["embeddings"]).detach().cpu().float()
    if tuple(embeddings.shape) != (len(metadata), FEATURE_DIM) or not bool(
        torch.isfinite(embeddings).all()
    ):
        raise Stage70TestCacheError(
            "Stage-70 cache shard embedding geometry/finiteness drifted."
        )
    extractor = payload.get("feature_extractor")
    if not isinstance(extractor, Mapping):
        raise Stage70TestCacheError(
            "Stage-70 cache shard extractor identity is invalid."
        )
    return Stage70CenterShard(
        embeddings=embeddings,
        metadata=metadata,
        feature_extractor=dict(extractor),
        shard_sha256=file_sha256(source),
    )


def scan_forbidden_metadata(metadata: Sequence[Mapping[str, object]]) -> None:
    """Scan keys and values for outcome/source/sample encodings."""

    for row_index, row in enumerate(metadata):
        keys = {str(key) for key in row}
        normalized = {key.casefold() for key in keys}
        if keys != SHARD_METADATA_FIELDS or normalized.intersection(
            FORBIDDEN_METADATA_FIELDS
        ):
            raise Stage70TestCacheError(
                f"Stage-70 shard metadata firewall failed at row {row_index}."
            )
        for value in row.values():
            if isinstance(value, str) and LEGACY_OUTCOME_PATTERN.search(value):
                raise Stage70TestCacheError(
                    "Stage-70 shard metadata contains a legacy outcome encoding."
                )


def write_content_index(root: str | Path) -> dict[str, object]:
    """Write an index over every material file except the index itself."""

    cache_root = Path(root)
    files = []
    for member in sorted(path for path in cache_root.rglob("*") if path.is_file()):
        relative = str(member.relative_to(cache_root))
        if relative == "manifests/content_index.json":
            continue
        if member.is_symlink():
            raise Stage70TestCacheError(
                f"Stage-70 cache contains an unsafe symlink: {relative}."
            )
        files.append({"path": relative, "sha256": file_sha256(member)})
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_descriptive_test_cache_content_index_v1",
        "files": files,
    }
    payload["content_hash"] = semantic_sha256(payload)
    write_json(cache_root / "manifests" / "content_index.json", payload)
    return payload


def validate_content_index(root: str | Path) -> dict[str, object]:
    """Rehash every indexed member and reject indexed-set drift."""

    cache_root = Path(root)
    payload = read_json(cache_root / "manifests" / "content_index.json")
    if set(payload) != {"schema_version", "files", "content_hash"}:
        raise Stage70TestCacheError("Stage-70 cache content-index schema drifted.")
    if payload.get("schema_version") != (
        "midogpp_stage70_descriptive_test_cache_content_index_v1"
    ):
        raise Stage70TestCacheError("Stage-70 cache content-index identity drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != semantic_sha256(unhashed):
        raise Stage70TestCacheError("Stage-70 cache content hash drifted.")
    raw_files = payload.get("files")
    if isinstance(raw_files, (str, bytes)) or not isinstance(raw_files, Sequence):
        raise Stage70TestCacheError("Stage-70 cache content-index files are invalid.")
    indexed: dict[str, str] = {}
    for record in raw_files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise Stage70TestCacheError(
                "Stage-70 cache content-index member schema drifted."
            )
        relative = str(record["path"])
        if relative in indexed or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise Stage70TestCacheError(
                "Stage-70 cache content-index contains an unsafe member."
            )
        indexed[relative] = str(record["sha256"])
    actual = {
        str(path.relative_to(cache_root))
        for path in cache_root.rglob("*")
        if path.is_file() and str(path.relative_to(cache_root)) != "manifests/content_index.json"
    }
    if set(indexed) != actual:
        raise Stage70TestCacheError("Stage-70 cache indexed member set drifted.")
    for relative, expected_sha256 in indexed.items():
        member = cache_root / relative
        if member.is_symlink() or file_sha256(member) != expected_sha256:
            raise Stage70TestCacheError(
                f"Stage-70 cache indexed member drifted: {relative}."
            )
    return payload


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage70TestCacheError(
            f"Stage-70 cache JSON is unreadable: {source}."
        ) from exc
    if not isinstance(payload, dict):
        raise Stage70TestCacheError(
            f"Stage-70 cache JSON must contain an object: {source}."
        )
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_from_row(row: TargetEvaluationRow) -> dict[str, object]:
    return row.to_dict()


def _validated_metadata_row(row: object) -> dict[str, object]:
    if not isinstance(row, Mapping) or {str(key) for key in row} != SHARD_METADATA_FIELDS:
        raise Stage70TestCacheError(
            "Stage-70 cache metadata must use the exact sealed row schema."
        )
    raw_index = row["contract_row_index"]
    if isinstance(raw_index, bool):
        raise Stage70TestCacheError("Stage-70 cache contract row index is invalid.")
    try:
        contract_row_index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise Stage70TestCacheError(
            "Stage-70 cache contract row index is invalid."
        ) from exc
    normalized = {
        "evaluation_row_id": str(row["evaluation_row_id"]),
        "contract_row_index": contract_row_index,
        "case_id": str(row["case_id"]),
        "center": str(row["center"]),
        "split": str(row["split"]),
    }
    if (
        not normalized["evaluation_row_id"].startswith("eval_")
        or not normalized["case_id"]
        or normalized["center"] not in ELIGIBLE_CENTERS
        or normalized["split"] != EVALUATION_SPLIT
        or contract_row_index < 0
    ):
        raise Stage70TestCacheError("Stage-70 cache metadata identity drifted.")
    return normalized


__all__ = (
    "Stage70CenterShard",
    "ValidatedStage70TestCache",
    "file_sha256",
    "load_stage70_center_shard",
    "metadata_from_row",
    "read_json",
    "scan_forbidden_metadata",
    "validate_content_index",
    "write_content_index",
    "write_json",
    "write_stage70_center_shard",
)
