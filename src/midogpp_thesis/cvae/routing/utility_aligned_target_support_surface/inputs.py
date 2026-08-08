"""Closed-world label-free reservation and cache admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from ..utility_aligned_identities import (
    CENTERS,
    POLICY_EXPERIMENT_ID,
    TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID,
)
from .config import TargetSupportSurfaceConfig
from .contracts import CACHE_ARTIFACT_ID, RESERVATION_ARTIFACT_ID


RESERVATION_MEMBER = "manifests/reservation.json"
CACHE_INDEX_MEMBER = "manifests/cache_index.json"
CACHE_CONTENT_INDEX_MEMBER = "manifests/content_index.json"


@dataclass(frozen=True)
class SupportRow:
    row_ordinal: int
    sample_id: str
    case_id: str
    center: str
    cache_shard_path: str
    cache_row_index: int


@dataclass(frozen=True)
class TargetSupportInputs:
    reservation_hash: str
    reservation_payload: Mapping[str, object]
    cache_binding_hash: str
    rows_by_target: Mapping[str, tuple[SupportRow, ...]]
    case_ids_by_target: Mapping[str, tuple[str, ...]]
    support_array_path_by_target: Mapping[str, Path]


def load_target_support_inputs(config: TargetSupportSurfaceConfig, *, execution_root: Path) -> TargetSupportInputs:
    reservation = _reservation(config.reservation_root / RESERVATION_MEMBER)
    admitted, binding = _cache(config.support_cache_root, reservation_hash=str(reservation["reservation_hash"]))
    rows = parse_support_rows(reservation)
    prepared = execution_root / "prepared_target_support"
    prepared.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for center in CENTERS:
        matrix = _materialize(config.support_cache_root, rows[center], admitted)
        destination = prepared / f"target_{center}_support.npy"
        _atomic_npy(destination, matrix)
        paths[center] = destination
    cases = {
        center: tuple(row.case_id for row in rows[center]) for center in CENTERS
    }
    return TargetSupportInputs(
        reservation_hash=str(reservation["reservation_hash"]),
        reservation_payload=MappingProxyType(dict(reservation)),
        cache_binding_hash=binding,
        rows_by_target=MappingProxyType(rows),
        case_ids_by_target=MappingProxyType(cases),
        support_array_path_by_target=MappingProxyType(paths),
    )


def _reservation(path: Path) -> Mapping[str, object]:
    raw = _json(path)
    required = {
        "schema_version", "artifact_id", "status", "authorized_consumer_experiment_ids",
        "dataset_family", "fresh_unconsumed_surface", "labels_present",
        "target_evaluation_rows_present", "support_case_ids_by_center",
        "support_rows_by_center", "reservation_id", "reservation_hash",
    }
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if (
        set(raw) != required
        or raw.get("schema_version") != "midogpp_utility_aligned_target_support_reservation_v1"
        or raw.get("artifact_id") != RESERVATION_ARTIFACT_ID or raw.get("status") != "ACTIVE"
        or raw.get("authorized_consumer_experiment_ids") != [
            TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID,
            POLICY_EXPERIMENT_ID,
        ]
        or raw.get("dataset_family") != "MIDOG++" or raw.get("fresh_unconsumed_surface") is not True
        or raw.get("labels_present") is not False or raw.get("target_evaluation_rows_present") is not False
        or raw.get("reservation_hash") != canonical_sha256(unhashed)
    ):
        raise ProtocolError("Target-support reservation failed closed.")
    parse_support_rows(raw)
    return raw


def parse_support_rows(raw: Mapping[str, object]) -> dict[str, tuple[SupportRow, ...]]:
    case_map = raw.get("support_case_ids_by_center")
    row_map = raw.get("support_rows_by_center")
    if not isinstance(case_map, Mapping) or not isinstance(row_map, Mapping) or tuple(str(key) for key in case_map) != CENTERS or tuple(str(key) for key in row_map) != CENTERS:
        raise ProtocolError("Target-support reservation center coverage drifted.")
    result: dict[str, tuple[SupportRow, ...]] = {}
    seen_samples: set[str] = set()
    seen_cache: set[tuple[str, int]] = set()
    seen_cases: set[str] = set()
    for center in CENTERS:
        case_values = case_map[center]
        raw_rows = row_map[center]
        if not isinstance(case_values, Sequence) or isinstance(case_values, (str, bytes)) or not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ProtocolError("Target-support reservation rows/cases are malformed.")
        cases = tuple(str(value) for value in case_values)
        if len(cases) < 8 or cases != tuple(sorted(cases)) or len(set(cases)) != len(cases) or any(not value for value in cases) or seen_cases.intersection(cases):
            raise ProtocolError("Target-support reservation requires eight unique cases.")
        seen_cases.update(cases)
        rows: list[SupportRow] = []
        for ordinal, value in enumerate(raw_rows):
            if not isinstance(value, Mapping) or set(value) != {"row_ordinal", "sample_id", "case_id", "center", "cache_shard_path", "cache_row_index"}:
                raise ProtocolError("Target-support reservation row schema drifted.")
            try:
                row = SupportRow(
                    row_ordinal=int(value["row_ordinal"]), sample_id=str(value["sample_id"]),
                    case_id=str(value["case_id"]), center=str(value["center"]),
                    cache_shard_path=str(value["cache_shard_path"]), cache_row_index=int(value["cache_row_index"]),
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProtocolError("Target-support reservation numeric field drifted.") from exc
            cache_key = (row.cache_shard_path, row.cache_row_index)
            if row.row_ordinal != ordinal or row.center != center or not row.sample_id or not row.case_id or not row.cache_shard_path or row.cache_row_index < 0 or row.sample_id in seen_samples or cache_key in seen_cache:
                raise ProtocolError("Target-support reservation row identity drifted.")
            seen_samples.add(row.sample_id); seen_cache.add(cache_key); rows.append(row)
        if {row.case_id for row in rows} != set(cases):
            raise ProtocolError("Target-support reservation row/case coverage drifted.")
        result[center] = tuple(rows)
    return result


def _cache(root: Path, *, reservation_hash: str) -> tuple[Mapping[str, Mapping[str, object]], str]:
    base = root.resolve()
    index = _json(base / CACHE_INDEX_MEMBER)
    required = {"schema_version", "artifact_id", "dataset_family", "representation_id", "feature_dim", "dtype", "labels_stored", "reservation_hash", "shards", "cache_index_hash"}
    unhashed = {key: value for key, value in index.items() if key != "cache_index_hash"}
    if set(index) != required or index.get("schema_version") != "midogpp_utility_aligned_target_support_cache_index_v1" or index.get("artifact_id") != CACHE_ARTIFACT_ID or index.get("dataset_family") != "MIDOG++" or index.get("representation_id") != "midogpp_virchow2_common_3840_float32_v1" or index.get("feature_dim") != COMMON_OUTPUT_DIM or index.get("dtype") != "float32" or index.get("labels_stored") is not False or index.get("reservation_hash") != reservation_hash or index.get("cache_index_hash") != canonical_sha256(unhashed):
        raise ProtocolError("Target-support cache index drifted.")
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes)) or not raw_shards:
        raise ProtocolError("Target-support cache has no shards.")
    shards: dict[str, Mapping[str, object]] = {}
    for raw in raw_shards:
        if not isinstance(raw, Mapping) or set(raw) != {"relative_path", "file_sha256", "shape", "dtype"}:
            raise ProtocolError("Target-support cache shard schema drifted.")
        relative = _relative(str(raw["relative_path"])); shape = raw["shape"]
        if relative in shards or not isinstance(shape, Sequence) or len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) != COMMON_OUTPUT_DIM or raw.get("dtype") != "float32":
            raise ProtocolError("Target-support cache shard geometry drifted.")
        member = _member(base, relative)
        if not member.is_file() or _sha(member) != raw["file_sha256"]:
            raise ProtocolError("Target-support cache shard bytes drifted.")
        array = _load_shard(member)
        if array.shape != (int(shape[0]), int(shape[1])) or array.dtype != np.float32:
            raise ProtocolError("Target-support cache shard header drifted.")
        shards[relative] = MappingProxyType(dict(raw))
    content_path = base / CACHE_CONTENT_INDEX_MEMBER
    content = _json(content_path)
    required_content = {"schema_version", "artifact_id", "cache_index_member", "member_sha256", "cache_binding_hash"}
    member_sha = content.get("member_sha256")
    if not isinstance(member_sha, Mapping):
        raise ProtocolError("Target-support cache content members are malformed.")
    normalized = {str(key): str(value) for key, value in member_sha.items()}
    discovered = tuple(base.rglob("*"))
    actual = {str(path.relative_to(base)) for path in discovered if path.is_file() and path.resolve() != content_path.resolve()}
    if any(path.is_symlink() for path in discovered) or set(normalized) != actual or actual != {CACHE_INDEX_MEMBER, *shards}:
        raise ProtocolError("Target-support cache is not closed-world label-free.")
    if any(_sha(_member(base, member)) != digest for member, digest in normalized.items()):
        raise ProtocolError("Target-support cache content bytes drifted.")
    content_unhashed = {key: value for key, value in content.items() if key != "cache_binding_hash"}
    if set(content) != required_content or content.get("schema_version") != "midogpp_utility_aligned_target_support_cache_content_index_v1" or content.get("artifact_id") != CACHE_ARTIFACT_ID or content.get("cache_index_member") != CACHE_INDEX_MEMBER or normalized.get(CACHE_INDEX_MEMBER) != _sha(base / CACHE_INDEX_MEMBER) or content.get("cache_binding_hash") != canonical_sha256(content_unhashed):
        raise ProtocolError("Target-support cache content binding drifted.")
    return MappingProxyType(shards), str(content["cache_binding_hash"])


def _materialize(root: Path, rows: Sequence[SupportRow], shards: Mapping[str, Mapping[str, object]]) -> np.ndarray:
    opened: dict[str, np.ndarray] = {}; values = []
    for row in rows:
        if row.cache_shard_path not in shards:
            raise ProtocolError("Target-support row references an unadmitted shard.")
        if row.cache_shard_path not in opened:
            opened[row.cache_shard_path] = _load_shard(_member(root.resolve(), row.cache_shard_path))
        shard = opened[row.cache_shard_path]
        if row.cache_row_index >= len(shard):
            raise ProtocolError("Target-support cache row is out of bounds.")
        values.append(np.asarray(shard[row.cache_row_index], dtype=np.float32))
    matrix = np.ascontiguousarray(np.stack(values), dtype=np.float32)
    if matrix.shape != (len(rows), COMMON_OUTPUT_DIM) or not np.isfinite(matrix).all():
        raise ProtocolError("Target-support prepared array geometry drifted.")
    return matrix


def _load_shard(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"embeddings"}:
                raise ProtocolError("Target-support NPZ shard schema drifted.")
            value = np.asarray(payload["embeddings"])
    else:
        raise ProtocolError("Target-support cache shards must be NPY or NPZ.")
    if value.ndim != 2 or value.shape[1] != COMMON_OUTPUT_DIM or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ProtocolError("Target-support cache shard values drifted.")
    return value


def _relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or not value.endswith((".npy", ".npz", ".json")):
        raise ProtocolError("Target-support cache member path is unsafe.")
    return str(path)


def _member(root: Path, value: str) -> Path:
    member = (root / _relative(value)).resolve()
    try: member.relative_to(root)
    except ValueError as exc: raise ProtocolError("Target-support cache member escaped root.") from exc
    return member


def _json(path: Path) -> dict[str, object]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ProtocolError(f"Cannot read target-support JSON: {path}.") from exc
    if not isinstance(value, dict): raise ProtocolError("Target-support JSON must be an object.")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle: np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


__all__ = (
    "SupportRow",
    "TargetSupportInputs",
    "load_target_support_inputs",
    "parse_support_rows",
)
