"""Authenticated label-blind cache and role-scoped consumed-test labels."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .config import HarpStage90Config


DEVELOPMENT_ROLE = "harp_consumed_test_development"
EVALUATION_ROLE = "harp_consumed_test_evaluation"
CACHE_INDEX = Path("manifests/cache_index.json")
CONTENT_INDEX = Path("manifests/content_index.json")
CACHE_ROWS = Path("tables/row_index.csv")


@dataclass(frozen=True, slots=True)
class HarpCacheRow:
    center: str
    case_id: str
    sample_id: str
    split_role: str
    split_row_index: int
    embedding_file: str
    embedding_row_index: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.center, self.case_id, self.sample_id)


@dataclass(frozen=True, slots=True)
class HarpConsumedCacheIndex:
    root: Path
    rows: tuple[HarpCacheRow, ...]
    shards: Mapping[str, tuple[int, int]]
    member_sha256: Mapping[str, str]
    content_sha256: str
    cache_hash: str

    def rows_for(self, center: str, role: str) -> tuple[HarpCacheRow, ...]:
        return tuple(
            row for row in self.rows if row.center == center and row.split_role == role
        )

    def load_embedding(self, row: HarpCacheRow) -> np.ndarray:
        path = _safe_member(self.root, row.embedding_file)
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError("Cannot load HARP Stage-90 label-blind embedding shard.") from exc
        if (
            values.dtype != np.float32
            or values.ndim != 2
            or values.shape[1] != COMMON_OUTPUT_DIM
            or row.embedding_row_index < 0
            or row.embedding_row_index >= len(values)
        ):
            raise ProtocolError("HARP Stage-90 embedding shard geometry drifted.")
        result = np.asarray(values[row.embedding_row_index], dtype=np.float32)
        if result.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(result).all():
            raise ProtocolError("HARP Stage-90 embedding row is nonfinite or malformed.")
        return result


def _expected(config: HarpStage90Config, role: str) -> str:
    value = config.expected_hashes.get(role)
    if value is None:
        raise ProtocolError(f"HARP Stage-90 authorized input hash is absent: {role}.")
    return value


def load_cache_index(config: HarpStage90Config) -> HarpConsumedCacheIndex:
    root = config.resolved_path("test_cache_root")
    index_path = root / CACHE_INDEX
    content_path = root / CONTENT_INDEX
    row_path = root / CACHE_ROWS
    index = read_json(index_path)
    content = read_json(content_path)
    content_base = {
        key: value for key, value in content.items() if key != "content_index_hash"
    }
    members = content.get("members")
    if (
        not isinstance(members, Mapping)
        or content.get("schema_version")
        != "midogpp_harp_consumed_test_content_index_v1"
        or content.get("content_index_hash") != canonical_hash(content_base)
        or content.get("content_index_hash")
        != _expected(config, "test_cache_content_sha256")
    ):
        raise ProtocolError("HARP Stage-90 cache content index drifted.")
    member_sha = {str(key): str(value) for key, value in members.items()}
    actual_members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root) != CONTENT_INDEX
    }
    if actual_members != set(member_sha):
        raise ProtocolError("HARP Stage-90 cache closed-world inventory drifted.")
    for relative, digest in member_sha.items():
        member = _safe_member(root, relative)
        if len(digest) != 64 or not member.is_file() or member.is_symlink() or sha256_file(member) != digest:
            raise ProtocolError("HARP Stage-90 cache content member drifted.")
    if member_sha.get(CACHE_INDEX.as_posix()) != sha256_file(index_path) or member_sha.get(
        CACHE_ROWS.as_posix()
    ) != sha256_file(row_path):
        raise ProtocolError("HARP Stage-90 cache index/row table is not content-bound.")
    index_base = {key: value for key, value in index.items() if key != "cache_index_hash"}
    shards_raw = index.get("shards")
    if (
        set(index)
        != {
            "schema_version", "artifact_id", "dataset_family", "representation_id",
            "feature_dim", "dtype", "labels_stored", "split_roles",
            "row_index_member", "shards", "cache_index_hash",
        }
        or index.get("schema_version")
        != "midogpp_harp_consumed_test_label_blind_frame_cache_v1"
        or index.get("artifact_id") != "midogpp_stage90_harp_consumed_test_cache_v1"
        or index.get("dataset_family") != "MIDOG++"
        or index.get("representation_id")
        != "midogpp_virchow2_common_3840_float32_v1"
        or index.get("feature_dim") != COMMON_OUTPUT_DIM
        or index.get("dtype") != "float32"
        or index.get("labels_stored") is not False
        or index.get("split_roles") != [DEVELOPMENT_ROLE, EVALUATION_ROLE]
        or index.get("row_index_member") != CACHE_ROWS.as_posix()
        or not isinstance(shards_raw, list)
        or not shards_raw
        or index.get("cache_index_hash") != canonical_hash(index_base)
    ):
        raise ProtocolError("HARP Stage-90 label-blind cache index drifted.")
    shards: dict[str, tuple[int, int]] = {}
    for raw in shards_raw:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path", "file_sha256", "shape", "dtype",
        }:
            raise ProtocolError("HARP Stage-90 cache shard schema drifted.")
        relative = str(raw["relative_path"])
        shape = raw["shape"]
        if (
            relative in shards
            or not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) <= 0
            or int(shape[1]) != COMMON_OUTPUT_DIM
            or raw.get("dtype") != "float32"
            or member_sha.get(relative) != raw.get("file_sha256")
        ):
            raise ProtocolError("HARP Stage-90 cache shard inventory drifted.")
        path = _safe_member(root, relative)
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError("Cannot read HARP Stage-90 cache shard.") from exc
        if array.dtype != np.float32 or array.shape != (int(shape[0]), int(shape[1])):
            raise ProtocolError("HARP Stage-90 cache shard header drifted.")
        shards[relative] = (int(shape[0]), int(shape[1]))
    rows = _read_cache_rows(row_path, shards)
    for center in CENTERS:
        for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
            if not any(row.center == center and row.split_role == role for row in rows):
                raise ProtocolError("HARP Stage-90 cache lacks center/split coverage.")
    development_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == DEVELOPMENT_ROLE
    }
    evaluation_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == EVALUATION_ROLE
    }
    if development_cases & evaluation_cases:
        raise ProtocolError("HARP Stage-90 development/evaluation cases are not disjoint.")
    return HarpConsumedCacheIndex(
        root=root, rows=rows, shards=shards, member_sha256=member_sha,
        content_sha256=str(content["content_index_hash"]),
        cache_hash=str(index["cache_index_hash"]),
    )


def _read_cache_rows(
    path: Path, shards: Mapping[str, tuple[int, int]]
) -> tuple[HarpCacheRow, ...]:
    expected_header = (
        "schema_version", "row_id", "center", "case_id", "split_role",
        "split_row_index", "embedding_file", "embedding_row_index",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ProtocolError("HARP Stage-90 cache row-index schema drifted.")
            output: list[HarpCacheRow] = []
            for raw in reader:
                if raw["schema_version"] != "midogpp_harp_consumed_test_frame_row_v1":
                    raise ProtocolError("HARP Stage-90 cache row schema version drifted.")
                output.append(
                    HarpCacheRow(
                        center=str(raw["center"]), case_id=str(raw["case_id"]),
                        sample_id=str(raw["row_id"]), split_role=str(raw["split_role"]),
                        split_row_index=int(raw["split_row_index"]),
                        embedding_file=str(raw["embedding_file"]),
                        embedding_row_index=int(raw["embedding_row_index"]),
                    )
                )
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read HARP Stage-90 cache row index.") from exc
    rows = tuple(output)
    expected_order = tuple(
        sorted(rows, key=lambda row: (row.split_role, row.center, row.split_row_index))
    )
    if (
        not rows or rows != expected_order or len({row.key for row in rows}) != len(rows)
        or any(
            row.center not in CENTERS
            or row.split_role not in {DEVELOPMENT_ROLE, EVALUATION_ROLE}
            or row.embedding_file not in shards
            or row.embedding_row_index < 0
            or row.embedding_row_index >= shards[row.embedding_file][0]
            for row in rows
        )
    ):
        raise ProtocolError("HARP Stage-90 cache row identities drifted.")
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in CENTERS:
            scoped = [row for row in rows if row.split_role == role and row.center == center]
            if [row.split_row_index for row in scoped] != list(range(len(scoped))):
                raise ProtocolError("HARP Stage-90 split row order is noncanonical.")
    return rows


def load_development_labels(
    config: HarpStage90Config, cache: HarpConsumedCacheIndex
) -> tuple[HarpSourceLabelRow, ...]:
    rows = _read_label_manifest(
        config.resolved_path("development_manifest_path"),
        expected_sha256=_expected(config, "development_manifest_sha256"),
        expected_role=DEVELOPMENT_ROLE, cache=cache,
    )
    return tuple(
        HarpSourceLabelRow(center=center, case_id=case, sample_id=sample, label=label)
        for center, case, sample, label in rows
    )


def load_evaluation_truth(
    config: HarpStage90Config, cache: HarpConsumedCacheIndex
) -> dict[tuple[str, str, str], int]:
    return {
        (center, case, sample): label
        for center, case, sample, label in _read_label_manifest(
            config.resolved_path("evaluation_manifest_path"),
            expected_sha256=_expected(config, "evaluation_manifest_sha256"),
            expected_role=EVALUATION_ROLE, cache=cache,
        )
    }


def _read_label_manifest(
    path: Path, *, expected_sha256: str, expected_role: str,
    cache: HarpConsumedCacheIndex,
) -> tuple[tuple[str, str, str, int], ...]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise ProtocolError("HARP Stage-90 role-scoped label manifest bytes drifted.")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != (
                "center", "case_id", "sample_id", "label", "split_role",
            ):
                raise ProtocolError("HARP Stage-90 role-scoped label schema drifted.")
            raw_rows = tuple(dict(row) for row in reader)
            if not raw_rows or any(
                row.get("split_role") != expected_role for row in raw_rows
            ):
                raise ProtocolError(
                    "HARP Stage-90 label capability file contains another split role."
                )
            output = tuple(
                (
                    str(row["center"]),
                    str(row["case_id"]),
                    str(row["sample_id"]),
                    int(row["label"]),
                )
                for row in raw_rows
            )
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read HARP Stage-90 role-scoped labels.") from exc
    cache_keys = tuple(row.key for row in cache.rows if row.split_role == expected_role)
    if (
        tuple(row[:3] for row in output) != cache_keys
        or any(label not in (0, 1) for *_key, label in output)
        or any(
            {label for center_, _case, _sample, label in output if center_ == center} != {0, 1}
            for center in CENTERS
        )
    ):
        raise ProtocolError("HARP Stage-90 labels do not exactly cover their cache split.")
    # A MIDOG whole-slide/case may legitimately contain both patch labels.
    # The replay estimand handles this with per-case-per-class denominators;
    # forcing one truth value per case would silently change the dataset.
    return output


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ProtocolError("HARP Stage-90 cache member path is unsafe.")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError("HARP Stage-90 cache member escaped its root.") from exc
    return path


__all__ = (
    "DEVELOPMENT_ROLE", "EVALUATION_ROLE", "HarpCacheRow",
    "HarpConsumedCacheIndex", "load_cache_index", "load_development_labels",
    "load_evaluation_truth",
)
