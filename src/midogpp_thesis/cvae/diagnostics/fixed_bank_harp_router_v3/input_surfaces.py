"""Revision-owned label-blind cache and role-scoped label readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .config import HarpStage90V3Config
from .safe_paths import safe_existing_member


DEVELOPMENT_ROLE = "harp_consumed_test_development"
EVALUATION_ROLE = "harp_consumed_test_evaluation"
CACHE_INDEX = Path("manifests/cache_index.json")
CONTENT_INDEX = Path("manifests/content_index.json")
CACHE_ROWS = Path("tables/row_index.csv")


@dataclass(frozen=True, slots=True)
class HarpConsumedCacheIdentity:
    artifact_id: str
    cache_schema: str
    row_schema: str
    content_schema: str


V3_CACHE_IDENTITY = HarpConsumedCacheIdentity(
    artifact_id="midogpp_stage90_harp_consumed_test_cache_v3",
    cache_schema="midogpp_harp_consumed_test_label_blind_frame_cache_v3",
    row_schema="midogpp_harp_consumed_test_frame_row_v3",
    content_schema="midogpp_harp_consumed_test_content_index_v3",
)


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
        return tuple(row for row in self.rows if row.center == center and row.split_role == role)

    def load_embedding(self, row: HarpCacheRow) -> np.ndarray:
        return self.load_embeddings((row,))[0].copy()

    def load_embeddings(self, rows: Sequence[HarpCacheRow]) -> np.ndarray:
        """Load ordered cache rows while opening each immutable shard once.

        The content index and every shard hash are authenticated when this
        object is built.  Grouping only the mmap handles here removes thousands
        of redundant ``np.load`` header parses during workstation staging; the
        requested row order remains the sole output order.
        """

        ordered = tuple(rows)
        if not ordered or any(type(row) is not HarpCacheRow for row in ordered):
            raise ProtocolError("HARP v3 embedding batch is malformed.")
        known = {row.key: row for row in self.rows}
        if any(known.get(row.key) != row for row in ordered):
            raise ProtocolError("HARP v3 embedding batch escaped the cache index.")

        opened: dict[str, np.ndarray] = {}
        result = np.empty((len(ordered), COMMON_OUTPUT_DIM), dtype=np.float32)
        for index, row in enumerate(ordered):
            values = opened.get(row.embedding_file)
            if values is None:
                path = _safe_member(self.root, row.embedding_file)
                try:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise ProtocolError("Cannot load HARP v3 embedding shard.") from exc
                expected_shape = self.shards.get(row.embedding_file)
                if (
                    expected_shape is None
                    or values.dtype != np.float32
                    or values.shape != expected_shape
                    or values.ndim != 2
                    or values.shape[1] != COMMON_OUTPUT_DIM
                ):
                    raise ProtocolError("HARP v3 embedding shard geometry drifted.")
                opened[row.embedding_file] = values
            if not 0 <= row.embedding_row_index < len(values):
                raise ProtocolError("HARP v3 embedding shard geometry drifted.")
            vector = np.asarray(values[row.embedding_row_index], dtype=np.float32)
            if vector.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(vector).all():
                raise ProtocolError("HARP v3 embedding row is malformed.")
            result[index] = vector
        return np.ascontiguousarray(result, dtype=np.float32)


def _expected(config: HarpStage90V3Config, role: str) -> str:
    value = config.expected_hashes.get(role)
    if type(value) is not str:
        raise ProtocolError(f"HARP v3 authorized input hash is absent: {role}.")
    return value


def load_cache_index(config: HarpStage90V3Config) -> HarpConsumedCacheIndex:
    if type(config) is not HarpStage90V3Config:
        raise ProtocolError("HARP v3 cache reader requires typed configuration.")
    root = config.resolved_path("test_cache_root")
    return _load_cache_index_from_root(
        root,
        expected_content_sha256=_expected(config, "test_cache_content_sha256"),
    )


def _load_cache_index_from_root(
    root: Path,
    *,
    expected_content_sha256: str,
) -> HarpConsumedCacheIndex:
    """Authenticate one fixed-identity v3 cache at an already bound root.

    The executable reader above remains restricted to the typed activated
    configuration.  Preparation uses this internal entry point only for its
    owned staging and final roots, before those paths can be represented by an
    activated configuration.  Cache identity is intentionally not a caller
    argument: every path is parsed as ``V3_CACHE_IDENTITY``.
    """

    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or type(expected_content_sha256) is not str
        or len(expected_content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_content_sha256
        )
    ):
        raise ProtocolError("HARP v3 cache root or expected identity is invalid.")
    index_path = root / CACHE_INDEX
    content_path = root / CONTENT_INDEX
    row_path = root / CACHE_ROWS
    index = read_json(index_path)
    content = read_json(content_path)
    content_base = {key: value for key, value in content.items() if key != "content_index_hash"}
    members = content.get("members")
    if (
        not isinstance(members, Mapping)
        or content.get("schema_version") != V3_CACHE_IDENTITY.content_schema
        or content.get("content_index_hash") != canonical_hash(content_base)
        or content.get("content_index_hash") != expected_content_sha256
    ):
        raise ProtocolError("HARP v3 cache content index drifted.")
    member_sha = {str(key): str(value) for key, value in members.items()}
    actual_members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root) != CONTENT_INDEX
    }
    if actual_members != set(member_sha):
        raise ProtocolError("HARP v3 cache closed-world inventory drifted.")
    for relative, digest in member_sha.items():
        member = _safe_member(root, relative)
        if (
            len(digest) != 64
            or not member.is_file()
            or member.is_symlink()
            or sha256_file(member) != digest
        ):
            raise ProtocolError("HARP v3 cache member drifted.")
    if member_sha.get(CACHE_INDEX.as_posix()) != sha256_file(index_path) or member_sha.get(
        CACHE_ROWS.as_posix()
    ) != sha256_file(row_path):
        raise ProtocolError("HARP v3 cache indexes are not content-bound.")

    index_base = {key: value for key, value in index.items() if key != "cache_index_hash"}
    shards_raw = index.get("shards")
    if (
        set(index)
        != {
            "schema_version",
            "artifact_id",
            "dataset_family",
            "representation_id",
            "feature_dim",
            "dtype",
            "labels_stored",
            "split_roles",
            "row_index_member",
            "shards",
            "cache_index_hash",
        }
        or index.get("schema_version") != V3_CACHE_IDENTITY.cache_schema
        or index.get("artifact_id") != V3_CACHE_IDENTITY.artifact_id
        or index.get("dataset_family") != "MIDOG++"
        or index.get("representation_id") != "midogpp_virchow2_common_3840_float32_v1"
        or index.get("feature_dim") != COMMON_OUTPUT_DIM
        or index.get("dtype") != "float32"
        or index.get("labels_stored") is not False
        or index.get("split_roles") != [DEVELOPMENT_ROLE, EVALUATION_ROLE]
        or index.get("row_index_member") != CACHE_ROWS.as_posix()
        or not isinstance(shards_raw, list)
        or not shards_raw
        or index.get("cache_index_hash") != canonical_hash(index_base)
    ):
        raise ProtocolError("HARP v3 label-blind cache index drifted.")

    shards: dict[str, tuple[int, int]] = {}
    for raw in shards_raw:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "file_sha256",
            "shape",
            "dtype",
        }:
            raise ProtocolError("HARP v3 cache shard schema drifted.")
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
            raise ProtocolError("HARP v3 cache shard inventory drifted.")
        try:
            array = np.load(_safe_member(root, relative), mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError("Cannot read HARP v3 cache shard.") from exc
        expected_shape = (int(shape[0]), int(shape[1]))
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise ProtocolError("HARP v3 cache shard header drifted.")
        shards[relative] = expected_shape
    rows = _read_cache_rows(row_path, shards)
    for center in CENTERS:
        for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
            if not any(row.center == center and row.split_role == role for row in rows):
                raise ProtocolError("HARP v3 cache lacks center/role coverage.")
    development_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == DEVELOPMENT_ROLE
    }
    evaluation_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == EVALUATION_ROLE
    }
    if development_cases & evaluation_cases:
        raise ProtocolError("HARP v3 development/evaluation cases overlap.")
    return HarpConsumedCacheIndex(
        root=root,
        rows=rows,
        shards=shards,
        member_sha256=member_sha,
        content_sha256=str(content["content_index_hash"]),
        cache_hash=str(index["cache_index_hash"]),
    )


def _read_cache_rows(
    path: Path, shards: Mapping[str, tuple[int, int]]
) -> tuple[HarpCacheRow, ...]:
    expected_header = (
        "schema_version",
        "row_id",
        "center",
        "case_id",
        "split_role",
        "split_row_index",
        "embedding_file",
        "embedding_row_index",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ProtocolError("HARP v3 cache row schema drifted.")
            rows = tuple(
                HarpCacheRow(
                    center=str(raw["center"]),
                    case_id=str(raw["case_id"]),
                    sample_id=str(raw["row_id"]),
                    split_role=str(raw["split_role"]),
                    split_row_index=int(raw["split_row_index"]),
                    embedding_file=str(raw["embedding_file"]),
                    embedding_row_index=int(raw["embedding_row_index"]),
                )
                for raw in reader
                if _require_row_schema(raw.get("schema_version"))
            )
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read HARP v3 cache row index.") from exc
    expected_order = tuple(sorted(rows, key=lambda row: (row.split_role, row.center, row.split_row_index)))
    if (
        not rows
        or rows != expected_order
        or len({row.key for row in rows}) != len(rows)
        or any(
            row.center not in CENTERS
            or row.split_role not in {DEVELOPMENT_ROLE, EVALUATION_ROLE}
            or row.embedding_file not in shards
            or not 0 <= row.embedding_row_index < shards[row.embedding_file][0]
            for row in rows
        )
    ):
        raise ProtocolError("HARP v3 cache row identities drifted.")
    for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE):
        for center in CENTERS:
            scoped = [row for row in rows if row.split_role == role and row.center == center]
            if [row.split_row_index for row in scoped] != list(range(len(scoped))):
                raise ProtocolError("HARP v3 split row order is noncanonical.")
    return rows


def _require_row_schema(value: object) -> bool:
    if value != V3_CACHE_IDENTITY.row_schema:
        raise ProtocolError("HARP v3 cache row schema version drifted.")
    return True


def load_development_labels(
    config: HarpStage90V3Config, cache: HarpConsumedCacheIndex
) -> tuple[HarpSourceLabelRow, ...]:
    rows = _read_label_manifest(
        config.resolved_path("development_manifest_path"),
        expected_sha256=_expected(config, "development_manifest_sha256"),
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
    )
    return tuple(
        HarpSourceLabelRow(center=center, case_id=case, sample_id=sample, label=label)
        for center, case, sample, label in rows
    )


def load_evaluation_truth(
    config: HarpStage90V3Config, cache: HarpConsumedCacheIndex
) -> dict[tuple[str, str, str], int]:
    return {
        (center, case, sample): label
        for center, case, sample, label in _read_label_manifest(
            config.resolved_path("evaluation_manifest_path"),
            expected_sha256=_expected(config, "evaluation_manifest_sha256"),
            expected_role=EVALUATION_ROLE,
            cache=cache,
        )
    }


def _read_label_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_role: str,
    cache: HarpConsumedCacheIndex,
) -> tuple[tuple[str, str, str, int], ...]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise ProtocolError("HARP v3 role label manifest drifted.")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != (
                "center",
                "case_id",
                "sample_id",
                "label",
                "split_role",
            ):
                raise ProtocolError("HARP v3 role label schema drifted.")
            raw_rows = tuple(dict(row) for row in reader)
        if not raw_rows or any(row.get("split_role") != expected_role for row in raw_rows):
            raise ProtocolError("HARP v3 label capability crossed split roles.")
        output = tuple(
            (str(row["center"]), str(row["case_id"]), str(row["sample_id"]), int(row["label"]))
            for row in raw_rows
        )
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read HARP v3 role labels.") from exc
    cache_keys = tuple(row.key for row in cache.rows if row.split_role == expected_role)
    if (
        tuple(row[:3] for row in output) != cache_keys
        or any(label not in (0, 1) for *_key, label in output)
        or any(
            {label for center_, _case, _sample, label in output if center_ == center} != {0, 1}
            for center in CENTERS
        )
    ):
        raise ProtocolError("HARP v3 labels do not exactly cover the cache role.")
    return output


def _safe_member(root: Path, relative: str) -> Path:
    return safe_existing_member(root, relative, role="prepared cache")


__all__ = (
    "CACHE_INDEX",
    "CACHE_ROWS",
    "CONTENT_INDEX",
    "DEVELOPMENT_ROLE",
    "EVALUATION_ROLE",
    "HarpCacheRow",
    "HarpConsumedCacheIdentity",
    "HarpConsumedCacheIndex",
    "V3_CACHE_IDENTITY",
    "load_cache_index",
    "load_development_labels",
    "load_evaluation_truth",
)
