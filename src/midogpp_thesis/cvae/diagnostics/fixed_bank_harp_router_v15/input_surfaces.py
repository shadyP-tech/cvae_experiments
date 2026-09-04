"""Revision-owned label-blind cache and role-scoped label readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from ...runtime.harp_v15_execution.contracts import FrozenRouteReceipt
from .config import HarpStage90V15Config
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .safe_paths import safe_existing_member


TARGET_TRAIN_SUPPORT_ROLE = "target_train_support"
TARGET_EVALUATION_ROLE = "target_test_evaluation"
# Compatibility aliases are deliberately limited to names used by the generic
# cache/config plumbing. Scientifically, v15 has target support, not a pooled
# source-development surface.
SUPPORT_ROLE = TARGET_TRAIN_SUPPORT_ROLE
DEVELOPMENT_ROLE = SUPPORT_ROLE
EVALUATION_ROLE = TARGET_EVALUATION_ROLE
CACHE_INDEX = Path("manifests/cache_index.json")
CONTENT_INDEX = Path("manifests/content_index.json")
CACHE_ROWS = Path("tables/row_index.csv")
EVALUATION_RELEASE_SCHEMA = "midogpp_harp_v15_evaluation_release_descriptor_v1"
EVALUATION_RELEASE_MEMBER = "release.json"
SOURCE_LABEL_INDEX_SCHEMA = (
    "midogpp_harp_v15_center_sharded_target_train_support_label_capability_v1"
)
SOURCE_LABEL_INDEX_MEMBER = "index.json"
CANONICAL_SCORING_MANIFEST_RELATIVE_PATH = Path(
    "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
)
_PREPARATION_RECEIPT_MEMBER = "manifests/harp_v15_consumed_test_preparation_receipt.json"
_LABEL_FREE_BARRIER_MEMBER = "manifests/harp_v15_label_free_partition_barrier.json"
_EVALUATION_RELEASE_KEYS = {
    "schema_version",
    "experiment_id",
    "artifact_role",
    "split_role",
    "canonical_scoring_manifest_relative_path",
    "canonical_scoring_manifest_sha256",
    "pre_manifest_cache_content_sha256",
    "cache_index_hash",
    "partition_hash",
    "ordered_cache_keys",
    "ordered_cache_key_hash",
    "row_count",
    "case_count",
    "evaluation_scope",
    "release_state",
    "publication_status",
    "terminal_decision",
    "fresh_evidence",
    "may_feed_stage60_or_stage70",
    "may_feed_another_experiment",
    "descriptor_hash",
}
_SOURCE_LABEL_INDEX_KEYS = {
    "schema_version",
    "experiment_id",
    "artifact_role",
    "split_role",
    "cache_index_hash",
    "pre_manifest_cache_content_sha256",
    "source_train_tensor_sha256",
    "shards",
    "row_count",
    "case_count",
    "labels_stored_in_index",
    "capability_state",
    "publication_status",
    "terminal_decision",
    "fresh_evidence",
    "may_feed_stage60_or_stage70",
    "may_feed_another_experiment",
    "index_hash",
}


@dataclass(frozen=True, slots=True)
class HarpConsumedCacheIdentity:
    artifact_id: str
    cache_schema: str
    row_schema: str
    content_schema: str


V15_CACHE_IDENTITY = HarpConsumedCacheIdentity(
    artifact_id="midogpp_stage90_harp_target_train_support_full_test_cache_v15",
    cache_schema="midogpp_harp_target_train_support_full_test_label_blind_frame_cache_v15",
    row_schema="midogpp_harp_target_train_support_full_test_frame_row_v15",
    content_schema="midogpp_harp_target_train_support_full_test_content_index_v15",
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
            raise ProtocolError("HARP v15 embedding batch is malformed.")
        known = {row.key: row for row in self.rows}
        if any(known.get(row.key) != row for row in ordered):
            raise ProtocolError("HARP v15 embedding batch escaped the cache index.")

        opened: dict[str, np.ndarray] = {}
        result = np.empty((len(ordered), COMMON_OUTPUT_DIM), dtype=np.float32)
        for index, row in enumerate(ordered):
            values = opened.get(row.embedding_file)
            if values is None:
                path = _safe_member(self.root, row.embedding_file)
                try:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise ProtocolError("Cannot load HARP v15 embedding shard.") from exc
                expected_shape = self.shards.get(row.embedding_file)
                if (
                    expected_shape is None
                    or values.dtype != np.float32
                    or values.shape != expected_shape
                    or values.ndim != 2
                    or values.shape[1] != COMMON_OUTPUT_DIM
                ):
                    raise ProtocolError("HARP v15 embedding shard geometry drifted.")
                opened[row.embedding_file] = values
            if not 0 <= row.embedding_row_index < len(values):
                raise ProtocolError("HARP v15 embedding shard geometry drifted.")
            vector = np.asarray(values[row.embedding_row_index], dtype=np.float32)
            if vector.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(vector).all():
                raise ProtocolError("HARP v15 embedding row is malformed.")
            result[index] = vector
        return np.ascontiguousarray(result, dtype=np.float32)


def _expected(config: HarpStage90V15Config, role: str) -> str:
    value = config.expected_hashes.get(role)
    if type(value) is not str:
        raise ProtocolError(f"HARP v15 authorized input hash is absent: {role}.")
    return value


def load_cache_index(config: HarpStage90V15Config) -> HarpConsumedCacheIndex:
    if type(config) is not HarpStage90V15Config:
        raise ProtocolError("HARP v15 cache reader requires typed configuration.")
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
    """Authenticate one fixed-identity v15 cache at an already bound root.

    The executable reader above remains restricted to the typed activated
    configuration.  Preparation uses this internal entry point only for its
    owned staging and final roots, before those paths can be represented by an
    activated configuration.  Cache identity is intentionally not a caller
    argument: every path is parsed as ``V15_CACHE_IDENTITY``.
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
        raise ProtocolError("HARP v15 cache root or expected identity is invalid.")
    index_path = root / CACHE_INDEX
    content_path = root / CONTENT_INDEX
    row_path = root / CACHE_ROWS
    index = read_json(index_path)
    content = read_json(content_path)
    content_base = {key: value for key, value in content.items() if key != "content_index_hash"}
    members = content.get("members")
    if (
        not isinstance(members, Mapping)
        or content.get("schema_version") != V15_CACHE_IDENTITY.content_schema
        or content.get("content_index_hash") != canonical_hash(content_base)
        or content.get("content_index_hash") != expected_content_sha256
    ):
        raise ProtocolError("HARP v15 cache content index drifted.")
    member_sha = {str(key): str(value) for key, value in members.items()}
    actual_members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root) != CONTENT_INDEX
    }
    if actual_members != set(member_sha):
        raise ProtocolError("HARP v15 cache closed-world inventory drifted.")
    for relative, digest in member_sha.items():
        member = _safe_member(root, relative)
        if (
            len(digest) != 64
            or not member.is_file()
            or member.is_symlink()
            or sha256_file(member) != digest
        ):
            raise ProtocolError("HARP v15 cache member drifted.")
    if member_sha.get(CACHE_INDEX.as_posix()) != sha256_file(index_path) or member_sha.get(
        CACHE_ROWS.as_posix()
    ) != sha256_file(row_path):
        raise ProtocolError("HARP v15 cache indexes are not content-bound.")

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
        or index.get("schema_version") != V15_CACHE_IDENTITY.cache_schema
        or index.get("artifact_id") != V15_CACHE_IDENTITY.artifact_id
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
        raise ProtocolError("HARP v15 label-blind cache index drifted.")

    shards: dict[str, tuple[int, int]] = {}
    for raw in shards_raw:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "file_sha256",
            "shape",
            "dtype",
        }:
            raise ProtocolError("HARP v15 cache shard schema drifted.")
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
            raise ProtocolError("HARP v15 cache shard inventory drifted.")
        try:
            array = np.load(_safe_member(root, relative), mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError("Cannot read HARP v15 cache shard.") from exc
        expected_shape = (int(shape[0]), int(shape[1]))
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise ProtocolError("HARP v15 cache shard header drifted.")
        shards[relative] = expected_shape
    rows = _read_cache_rows(row_path, shards)
    for center in CENTERS:
        for role in (SUPPORT_ROLE, TARGET_EVALUATION_ROLE):
            if not any(row.center == center and row.split_role == role for row in rows):
                raise ProtocolError("HARP v15 cache lacks center/role coverage.")
    development_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == DEVELOPMENT_ROLE
    }
    evaluation_cases = {
        (row.center, row.case_id) for row in rows if row.split_role == EVALUATION_ROLE
    }
    if development_cases & evaluation_cases:
        raise ProtocolError("HARP v15 development/evaluation cases overlap.")
    from .preparation_contracts import (
        EXPECTED_SOURCE_TRAIN_CASE_COUNT,
        EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER,
        EXPECTED_SOURCE_TRAIN_ROW_COUNT,
        EXPECTED_TARGET_TEST_CASE_COUNT,
        EXPECTED_TARGET_TEST_ROWS_BY_CENTER,
        EXPECTED_TARGET_TEST_ROW_COUNT,
    )

    source_rows = tuple(row for row in rows if row.split_role == SUPPORT_ROLE)
    target_rows = tuple(row for row in rows if row.split_role == TARGET_EVALUATION_ROLE)
    if (
        len(source_rows) != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or len(target_rows) != EXPECTED_TARGET_TEST_ROW_COUNT
        or len(development_cases) != EXPECTED_SOURCE_TRAIN_CASE_COUNT
        or len(evaluation_cases) != EXPECTED_TARGET_TEST_CASE_COUNT
        or {
            center: sum(row.center == center for row in source_rows)
            for center in CENTERS
        }
        != EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER
        or {
            center: sum(row.center == center for row in target_rows)
            for center in CENTERS
        }
        != EXPECTED_TARGET_TEST_ROWS_BY_CENTER
        or any(
            not row.sample_id.startswith("src_")
            or not row.embedding_file.startswith("embeddings/source_train/by_center/")
            for row in source_rows
        )
        or any(
            not row.sample_id.startswith("eval_")
            or not row.embedding_file.startswith("embeddings/target_test/by_center/")
            for row in target_rows
        )
    ):
        raise ProtocolError("HARP v15 source-train/full-test cache geometry drifted.")
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
                raise ProtocolError("HARP v15 cache row schema drifted.")
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
        raise ProtocolError("Cannot read HARP v15 cache row index.") from exc
    role_rank = {SUPPORT_ROLE: 0, TARGET_EVALUATION_ROLE: 1}
    expected_order = tuple(
        sorted(
            rows,
            key=lambda row: (
                role_rank.get(row.split_role, 99),
                row.center,
                row.split_row_index,
            ),
        )
    )
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
        raise ProtocolError("HARP v15 cache row identities drifted.")
    for role in (SUPPORT_ROLE, TARGET_EVALUATION_ROLE):
        for center in CENTERS:
            scoped = [row for row in rows if row.split_role == role and row.center == center]
            if [row.split_row_index for row in scoped] != list(range(len(scoped))):
                raise ProtocolError("HARP v15 split row order is noncanonical.")
    return rows


def _require_row_schema(value: object) -> bool:
    if value != V15_CACHE_IDENTITY.row_schema:
        raise ProtocolError("HARP v15 cache row schema version drifted.")
    return True


def load_support_labels(
    config: HarpStage90V15Config,
    cache: HarpConsumedCacheIndex,
    *,
    allowed_centers: Sequence[str] | None = None,
    source_label_capability: object | None = None,
) -> tuple[HarpSourceLabelRow, ...]:
    allowed = _allowed_source_centers(allowed_centers)
    _authenticate_support_label_scope(
        allowed,
        capability=source_label_capability,
    )
    rows = _read_label_manifest(
        config.resolved_path("development_manifest_path"),
        expected_sha256=_expected(config, "development_manifest_sha256"),
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
        allowed_centers=allowed,
    )
    return tuple(
        HarpSourceLabelRow(center=center, case_id=case, sample_id=sample, label=label)
        for center, case, sample, label in rows
    )


# Stable generic name for the preparation/runtime plumbing. New v15 science
# should call ``load_support_labels`` so the estimand cannot be misread.
load_development_labels = load_support_labels


def load_evaluation_truth(
    config: HarpStage90V15Config,
    cache: HarpConsumedCacheIndex,
    frozen_route_receipt: FrozenRouteReceipt,
) -> dict[tuple[str, str, str], int]:
    """Release evaluation truth only to one authenticated frozen route set."""

    _authenticate_frozen_route_receipt(config, cache, frozen_route_receipt)
    release_path = config.resolved_path("evaluation_manifest_path")
    descriptor = _read_evaluation_release_descriptor(
        release_path,
        expected_sha256=_expected(config, "evaluation_manifest_sha256"),
        cache=cache,
    )
    canonical_manifest = _canonical_manifest_from_release(
        release_path,
        expected_relative=str(
            descriptor["canonical_scoring_manifest_relative_path"]
        ),
    )
    expected_manifest_sha256 = str(
        descriptor["canonical_scoring_manifest_sha256"]
    )
    if sha256_file(canonical_manifest) != expected_manifest_sha256:
        raise ProtocolError("HARP v15 canonical evaluation truth source drifted.")
    try:
        with canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"case_id", "center", "split", "label"}.issubset(
                reader.fieldnames or ()
            ):
                raise ProtocolError("HARP v15 canonical evaluation schema drifted.")
            raw_rows = tuple(dict(row) for row in reader)
    except ProtocolError:
        raise
    except (OSError, csv.Error) as exc:
        raise ProtocolError("HARP v15 canonical evaluation truth is unreadable.") from exc

    expected_keys = tuple(
        tuple(str(value) for value in raw)
        for raw in descriptor["ordered_cache_keys"]  # type: ignore[union-attr]
    )
    expected_by_sample = {sample: (center, case) for center, case, sample in expected_keys}
    if len(expected_by_sample) != len(expected_keys):
        raise ProtocolError("HARP v15 evaluation release identities are not unique.")
    observed: dict[str, tuple[str, str, int]] = {}
    for ordinal, raw in enumerate(raw_rows):
        sample = evaluation_row_id(expected_manifest_sha256, ordinal)
        expected = expected_by_sample.get(sample)
        if expected is None:
            continue
        center = str(raw.get("center"))
        case = str(raw.get("case_id"))
        value = str(raw.get("label"))
        if (
            raw.get("split") != "test"
            or (center, case) != expected
            or value not in {"0", "1"}
            or sample in observed
        ):
            raise ProtocolError("HARP v15 evaluation cache/manifest alignment drifted.")
        observed[sample] = (center, case, int(value))
    if set(observed) != set(expected_by_sample):
        raise ProtocolError("HARP v15 evaluation truth does not cover its release.")
    if any(observed[key[2]][:2] != key[:2] for key in expected_keys):
        raise ProtocolError("HARP v15 evaluation truth identity binding drifted.")
    return {key: observed[key[2]][2] for key in expected_keys}


def _authenticate_frozen_route_receipt(
    config: HarpStage90V15Config,
    cache: HarpConsumedCacheIndex,
    receipt: FrozenRouteReceipt,
) -> None:
    """Authenticate the typed terminal capability before any truth-source read."""

    if type(config) is not HarpStage90V15Config or type(receipt) is not FrozenRouteReceipt:
        raise ProtocolError(
            "HARP v15 evaluation truth requires a typed frozen-route receipt."
        )
    centers = tuple(str(value) for value in config.protocol.get("centers", ()))
    evaluation_cases = {
        (row.center, row.case_id)
        for row in cache.rows
        if row.split_role == EVALUATION_ROLE
    }
    ordered_cases = tuple(sorted(evaluation_cases))
    ordered_case_identity_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_ordered_target_case_identity_v1",
            "ordered_cases": [list(value) for value in ordered_cases],
        }
    )
    samples_by_case: dict[tuple[str, str], list[str]] = {
        key: [] for key in ordered_cases
    }
    for row in cache.rows:
        if row.split_role == EVALUATION_ROLE:
            samples_by_case[(row.center, row.case_id)].append(row.sample_id)
    ordered_sample_identity_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_ordered_target_sample_identity_v1",
            "ordered_case_samples": [
                {
                    "outer_target_id": center,
                    "case_id": case,
                    "sample_ids": samples_by_case[(center, case)],
                }
                for center, case in ordered_cases
            ],
        }
    )
    from .preparation_contracts import EXPECTED_TARGET_TEST_CASE_COUNT

    if (
        config.execution_authorized is not True
        or receipt.config_hash != config.config_hash
        or receipt.expected_center_ids != centers
        or receipt.case_count != len(evaluation_cases)
        or len(evaluation_cases) != EXPECTED_TARGET_TEST_CASE_COUNT
        or receipt.ordered_case_identity_hash != ordered_case_identity_hash
        or receipt.ordered_sample_identity_hash != ordered_sample_identity_hash
        or not evaluation_cases
    ):
        raise ProtocolError("HARP v15 frozen-route receipt is not evaluation-bound.")


def _read_evaluation_release_descriptor(
    path: Path,
    *,
    expected_sha256: str,
    cache: HarpConsumedCacheIndex,
    require_final_cache: bool = True,
) -> Mapping[str, object]:
    """Authenticate the label-free direct input without opening outcome truth."""

    if (
        path.name != EVALUATION_RELEASE_MEMBER
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_sha256
    ):
        raise ProtocolError("HARP v15 evaluation release descriptor drifted.")
    descriptor = read_json(path)
    base = {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
    raw_keys = descriptor.get("ordered_cache_keys")
    if not isinstance(raw_keys, list):
        raise ProtocolError("HARP v15 evaluation release key inventory is malformed.")
    try:
        ordered_keys = tuple(
            tuple(str(value) for value in raw)
            for raw in raw_keys
            if isinstance(raw, list) and len(raw) == 3
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - strings only
        raise ProtocolError("HARP v15 evaluation release key inventory is malformed.") from exc
    cache_keys = tuple(
        row.key for row in cache.rows if row.split_role == EVALUATION_ROLE
    )
    barrier = read_json(cache.root / _LABEL_FREE_BARRIER_MEMBER)
    barrier_base = {
        key: value for key, value in barrier.items() if key != "barrier_hash"
    }
    pre_manifest_members = dict(cache.member_sha256)
    receipt_digest = pre_manifest_members.pop(_PREPARATION_RECEIPT_MEMBER, None)
    if require_final_cache and receipt_digest is None:
        raise ProtocolError("HARP v15 prepared cache lacks its release boundary.")
    pre_manifest_hash = (
        canonical_hash(
            {
                "schema_version": V15_CACHE_IDENTITY.content_schema,
                "members": dict(sorted(pre_manifest_members.items())),
            }
        )
        if receipt_digest is not None
        else cache.content_sha256
    )
    if (
        set(descriptor) != _EVALUATION_RELEASE_KEYS
        or descriptor.get("schema_version") != EVALUATION_RELEASE_SCHEMA
        or descriptor.get("experiment_id") != EXPERIMENT_ID
        or descriptor.get("artifact_role") != "sealed_evaluation_release_descriptor"
        or descriptor.get("split_role") != EVALUATION_ROLE
        or descriptor.get("canonical_scoring_manifest_relative_path")
        != CANONICAL_SCORING_MANIFEST_RELATIVE_PATH.as_posix()
        or descriptor.get("descriptor_hash") != canonical_hash(base)
        or descriptor.get("pre_manifest_cache_content_sha256")
        != pre_manifest_hash
        or descriptor.get("cache_index_hash") != cache.cache_hash
        or descriptor.get("partition_hash") != barrier.get("partition_hash")
        or barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or ordered_keys != cache_keys
        or len(ordered_keys) != len(raw_keys)
        or descriptor.get("ordered_cache_key_hash")
        != canonical_hash({"ordered_cache_keys": raw_keys})
        or descriptor.get("row_count") != len(cache_keys)
        or descriptor.get("case_count")
        != len({(center, case) for center, case, _sample in cache_keys})
        or descriptor.get("evaluation_scope") != "all_218_canonical_test_cases"
        or descriptor.get("release_state")
        != "SEALED_UNTIL_TYPED_FROZEN_ROUTE_RECEIPT"
        or descriptor.get("publication_status") != PUBLICATION_STATUS
        or descriptor.get("terminal_decision") != TERMINAL_DECISION
        or descriptor.get("fresh_evidence") is not False
        or descriptor.get("may_feed_stage60_or_stage70") is not False
        or descriptor.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("HARP v15 evaluation release binding drifted.")
    return descriptor


def _canonical_manifest_from_release(
    release_path: Path, *, expected_relative: str
) -> Path:
    release_relative = Path(
        "datasets/midogpp/contract/harp_full_test_evaluation_release_v15/release.json"
    )
    if (
        expected_relative != CANONICAL_SCORING_MANIFEST_RELATIVE_PATH.as_posix()
        or tuple(release_path.parts[-len(release_relative.parts) :])
        != release_relative.parts
    ):
        raise ProtocolError("HARP v15 evaluation release path escaped its catalog.")
    repository_root = release_path.parents[len(release_relative.parts) - 1]
    return safe_existing_member(
        repository_root,
        expected_relative,
        role="canonical evaluation truth",
    )


def evaluation_row_id(manifest_sha256: str, contract_row_index: int) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "contract_row_index": contract_row_index,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"eval_{hashlib.sha256(encoded).hexdigest()}"


def _read_label_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_role: str,
    cache: HarpConsumedCacheIndex,
    allowed_centers: Sequence[str] | None = None,
) -> tuple[tuple[str, str, str, int], ...]:
    allowed = _allowed_source_centers(allowed_centers)
    if (
        path.name != SOURCE_LABEL_INDEX_MEMBER
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_sha256
    ):
        raise ProtocolError("HARP v15 role label manifest drifted.")
    index = read_json(path)
    base = {key: value for key, value in index.items() if key != "index_hash"}
    raw_shards = index.get("shards")
    if (
        set(index) != _SOURCE_LABEL_INDEX_KEYS
        or index.get("schema_version") != SOURCE_LABEL_INDEX_SCHEMA
        or index.get("experiment_id") != EXPERIMENT_ID
        or index.get("artifact_role")
        != "center_sharded_target_train_support_label_capability"
        or index.get("split_role") != expected_role
        or index.get("cache_index_hash") != cache.cache_hash
        or index.get("pre_manifest_cache_content_sha256")
        != _pre_manifest_cache_content_sha256(cache)
        or index.get("source_train_tensor_sha256")
        != "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
        or not isinstance(raw_shards, list)
        or len(raw_shards) != len(CENTERS)
        or index.get("labels_stored_in_index") is not False
        or index.get("capability_state")
        != "SUPPORT_CENTER_SCOPED_OPEN_AFTER_SUPPORT_AND_TARGET_MENU_SEALS"
        or index.get("publication_status") != PUBLICATION_STATUS
        or index.get("terminal_decision") != TERMINAL_DECISION
        or index.get("fresh_evidence") is not False
        or index.get("may_feed_stage60_or_stage70") is not False
        or index.get("may_feed_another_experiment") is not False
        or index.get("index_hash") != canonical_hash(base)
    ):
        raise ProtocolError("HARP v15 source-label capability index drifted.")
    shard_by_center: dict[str, Mapping[str, object]] = {}
    for raw in raw_shards:
        if not isinstance(raw, Mapping) or set(raw) != {
            "center",
            "relative_path",
            "sha256",
            "row_count",
            "case_count",
            "ordered_key_hash",
        }:
            raise ProtocolError("HARP v15 source-label shard index drifted.")
        center = str(raw.get("center"))
        relative = str(raw.get("relative_path"))
        if (
            center not in CENTERS
            or center in shard_by_center
            or relative != f"by_center/center_{center}.csv"
            or not _is_sha256(raw.get("sha256"))
        ):
            raise ProtocolError("HARP v15 source-label shard identity drifted.")
        shard_by_center[center] = raw
    if tuple(shard_by_center) != CENTERS:
        raise ProtocolError("HARP v15 source-label shard coverage drifted.")
    output_rows: list[tuple[str, str, str, int]] = []
    for center in CENTERS:
        if center not in allowed:
            continue
        raw = shard_by_center[center]
        shard = safe_existing_member(
            path.parent,
            str(raw["relative_path"]),
            role="source-label shard",
        )
        if sha256_file(shard) != raw["sha256"]:
            raise ProtocolError("HARP v15 source-label shard bytes drifted.")
        rows = _read_source_label_shard(shard, center=center, expected_role=expected_role)
        cache_keys = tuple(
            row.key
            for row in cache.rows
            if row.split_role == expected_role and row.center == center
        )
        if (
            tuple(row[:3] for row in rows) != cache_keys
            or raw.get("row_count") != len(rows)
            or raw.get("case_count") != len({row[1] for row in rows})
            or raw.get("ordered_key_hash")
            != canonical_hash({"ordered_keys": [list(row[:3]) for row in rows]})
            or {row[3] for row in rows} != {0, 1}
        ):
            raise ProtocolError("HARP v15 labels do not exactly cover the cache role.")
        output_rows.extend(rows)
    output = tuple(output_rows)
    expected_keys = tuple(
        row.key
        for row in cache.rows
        if row.split_role == expected_role and row.center in allowed
    )
    if (
        tuple(row[:3] for row in output) != expected_keys
        or index.get("row_count")
        != sum(int(raw["row_count"]) for raw in raw_shards)
        or index.get("case_count")
        != sum(int(raw["case_count"]) for raw in raw_shards)
    ):
        raise ProtocolError("HARP v15 source-label capability coverage drifted.")
    return output


def _read_source_label_shard(
    path: Path,
    *,
    center: str,
    expected_role: str,
) -> tuple[tuple[str, str, str, int], ...]:
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
                raise ProtocolError("HARP v15 source-label shard schema drifted.")
            raw_rows = tuple(dict(row) for row in reader)
        if not raw_rows or any(
            row.get("center") != center or row.get("split_role") != expected_role
            for row in raw_rows
        ):
            raise ProtocolError("HARP v15 source-label shard crossed center roles.")
        return tuple(
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
        raise ProtocolError("Cannot read HARP v15 source-label shard.") from exc


def _allowed_source_centers(values: Sequence[str] | None) -> tuple[str, ...]:
    allowed = CENTERS if values is None else tuple(str(value) for value in values)
    if tuple(center for center in CENTERS if center in set(allowed)) != allowed or not allowed:
        raise ProtocolError("HARP v15 source-label center scope is noncanonical.")
    return allowed


def _authenticate_support_label_scope(
    allowed: Sequence[str],
    *,
    capability: object | None,
) -> None:
    from .source_label_capability import TargetSupportLabelCapability

    if type(capability) is not TargetSupportLabelCapability:
        raise ProtocolError(
            "HARP v15 support labels require a typed same-center menu capability."
        )
    capability.authorize(tuple(str(value) for value in allowed))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _pre_manifest_cache_content_sha256(cache: HarpConsumedCacheIndex) -> str:
    members = dict(cache.member_sha256)
    receipt = members.pop(_PREPARATION_RECEIPT_MEMBER, None)
    if receipt is None:
        return cache.content_sha256
    return canonical_hash(
        {
            "schema_version": V15_CACHE_IDENTITY.content_schema,
            "members": dict(sorted(members.items())),
        }
    )


def _safe_member(root: Path, relative: str) -> Path:
    return safe_existing_member(root, relative, role="prepared cache")


__all__ = (
    "CACHE_INDEX",
    "CACHE_ROWS",
    "CONTENT_INDEX",
    "CANONICAL_SCORING_MANIFEST_RELATIVE_PATH",
    "DEVELOPMENT_ROLE",
    "SUPPORT_ROLE",
    "TARGET_TRAIN_SUPPORT_ROLE",
    "EVALUATION_RELEASE_MEMBER",
    "EVALUATION_RELEASE_SCHEMA",
    "EVALUATION_ROLE",
    "SOURCE_LABEL_INDEX_MEMBER",
    "SOURCE_LABEL_INDEX_SCHEMA",
    "TARGET_EVALUATION_ROLE",
    "HarpCacheRow",
    "HarpConsumedCacheIdentity",
    "HarpConsumedCacheIndex",
    "V15_CACHE_IDENTITY",
    "load_cache_index",
    "load_development_labels",
    "load_support_labels",
    "load_evaluation_truth",
    "evaluation_row_id",
)
