"""Fresh reservation and label-free cache inputs for workstation execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ..metadata_compatibility import derive_compatibility_scores, derive_metadata_profiles
from ..metadata_compatibility.contracts import DOMAIN_MAPPING_MEMBER, DOMAIN_MAPPING_SHA256
from ..residual_topup.hashing import canonical_sha256
from .config import ExactTailUtilitySurfaceConfig, FreshInputAttestation
from .contracts import (
    CENTERS,
    DevelopmentPartition,
    EvaluationRowIdentity,
)


RESERVATION_MEMBER = "manifests/reservation.json"
RESERVATION_SCHEMA = "midogpp_utility_aligned_development_reservation_v1"
CACHE_INDEX_MEMBER = "manifests/cache_index.json"
CACHE_CONTENT_INDEX_MEMBER = "manifests/content_index.json"
CACHE_INDEX_SCHEMA = "midogpp_utility_aligned_development_cache_index_v1"
CACHE_CONTENT_INDEX_SCHEMA = (
    "midogpp_utility_aligned_development_cache_content_index_v1"
)
EXPECTED_CACHE_REPRESENTATION_ID = "midogpp_virchow2_common_3840_float32_v1"
REQUIRED_CACHE_FILES = (CACHE_INDEX_MEMBER, CACHE_CONTENT_INDEX_MEMBER)


@dataclass(frozen=True)
class DevelopmentReservation:
    partitions: Mapping[str, DevelopmentPartition]
    metadata_similarity_by_query_source: Mapping[str, Mapping[str, float]]
    reservation_hash: str
    raw_payload: Mapping[str, object]


@dataclass(frozen=True)
class PreparedDevelopmentInputs:
    reservation: DevelopmentReservation
    support_array_path_by_center: Mapping[str, Path]
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    evaluation_array_path_by_center: Mapping[str, Path]


def load_development_reservation(
    config: ExactTailUtilitySurfaceConfig,
    attestation: FreshInputAttestation,
) -> DevelopmentReservation:
    path = config.development_reservation_root / RESERVATION_MEMBER
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail development reservation is unreadable.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("Exact-tail development reservation must be an object.")
    required = {
        "schema_version",
        "status",
        "dataset_family",
        "center_universe",
        "partitions",
        "metadata_similarity_by_query_source",
        "metadata_profile_sha256",
        "reservation_cache_and_index_contain_labels",
        "whole_case_support_evaluation_disjoint",
        "development_target_evaluation_disjoint",
        "reservation_hash",
    }
    if set(raw) != required:
        raise ProtocolError("Exact-tail development reservation schema drifted.")
    observed_hash = str(raw.get("reservation_hash", ""))
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if (
        raw.get("schema_version") != RESERVATION_SCHEMA
        or raw.get("status") != "READY"
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("center_universe") != list(CENTERS)
        or raw.get("reservation_cache_and_index_contain_labels") is not False
        or raw.get("whole_case_support_evaluation_disjoint") is not True
        or raw.get("development_target_evaluation_disjoint") is not True
        or raw.get("metadata_profile_sha256") != attestation.metadata_profile_sha256
        or observed_hash != stable_hash(unhashed)
        or observed_hash != attestation.reservation_index_hash
    ):
        raise ProtocolError("Exact-tail development reservation identity drifted.")
    raw_partitions = raw.get("partitions")
    if not isinstance(raw_partitions, Sequence) or isinstance(
        raw_partitions, (str, bytes)
    ):
        raise ProtocolError("Exact-tail development partitions are absent.")
    partitions: dict[str, DevelopmentPartition] = {}
    for value in raw_partitions:
        partition = parse_development_partition(value)
        if partition.center in partitions:
            raise ProtocolError("Exact-tail development partition is duplicated.")
        partitions[partition.center] = partition
    if tuple(partitions) != CENTERS:
        raise ProtocolError("Exact-tail development reservation lacks all centers.")
    metadata = _metadata_similarity(raw.get("metadata_similarity_by_query_source"))
    profiles = derive_metadata_profiles(
        config.metadata_profile_root / DOMAIN_MAPPING_MEMBER,
        expected_sha256=DOMAIN_MAPPING_SHA256,
    )
    derived: dict[str, dict[str, float]] = {center: {} for center in CENTERS}
    for score in derive_compatibility_scores(profiles):
        derived[score.target_center][score.source_center] = (
            float(score.exact_match_count) / 3.0
        )
    if any(
        dict(metadata[query]) != derived[query]
        for query in CENTERS
    ):
        raise ProtocolError("Exact-tail metadata features escaped the hash-bound profile artifact.")
    return DevelopmentReservation(
        partitions=MappingProxyType(partitions),
        metadata_similarity_by_query_source=metadata,
        reservation_hash=observed_hash,
        raw_payload=MappingProxyType(dict(raw)),
    )


def prepare_development_cache_arrays(
    config: ExactTailUtilitySurfaceConfig,
    reservation: DevelopmentReservation,
    *,
    output_root: Path,
) -> PreparedDevelopmentInputs:
    """Publish compact float32 support/evaluation arrays for spawned workers."""

    admitted_shards = validate_development_cache_binding(config, reservation)
    prepared_root = output_root / "scratch/development_inputs"
    prepared_root.mkdir(parents=True, exist_ok=True)
    support_paths: dict[str, Path] = {}
    support_cases: dict[str, tuple[str, ...]] = {}
    evaluation_paths: dict[str, Path] = {}
    seen_samples: set[str] = set()
    seen_cache_cells: set[tuple[str, int]] = set()
    for center in CENTERS:
        partition = reservation.partitions[center]
        for rows, role in (
            (partition.support_rows, "support"),
            (partition.evaluation_rows, "evaluation"),
        ):
            for row in rows:
                cache_key = (row.cache_shard_path, row.cache_row_index)
                if row.sample_id in seen_samples or cache_key in seen_cache_cells:
                    raise ProtocolError(
                        "Exact-tail reservation duplicates sample/cache identities."
                    )
                seen_samples.add(row.sample_id)
                seen_cache_cells.add(cache_key)
            array = _materialize_rows(
                config.development_cache_root, rows, admitted_shards=admitted_shards
            )
            destination = prepared_root / f"center_{center}_{role}.npy"
            _atomic_save_npy(destination, array)
            if role == "support":
                support_paths[center] = destination
                support_cases[center] = tuple(row.case_id for row in rows)
            else:
                evaluation_paths[center] = destination
    return PreparedDevelopmentInputs(
        reservation=reservation,
        support_array_path_by_center=MappingProxyType(support_paths),
        support_case_ids_by_center=MappingProxyType(support_cases),
        evaluation_array_path_by_center=MappingProxyType(evaluation_paths),
    )


def validate_development_cache_binding(
    config: ExactTailUtilitySurfaceConfig,
    reservation: DevelopmentReservation,
) -> Mapping[str, Mapping[str, object]]:
    """Admit a closed-world, hash-indexed, label-free development cache."""

    root = config.development_cache_root.resolve()
    cache_index_path = root / CACHE_INDEX_MEMBER
    content_index_path = root / CACHE_CONTENT_INDEX_MEMBER
    index = _read_json(cache_index_path, "development cache index")
    required_index = {
        "schema_version",
        "artifact_id",
        "dataset_family",
        "representation_id",
        "feature_dim",
        "dtype",
        "labels_stored",
        "reservation_index_hash",
        "shards",
        "cache_index_hash",
    }
    if set(index) != required_index:
        raise ProtocolError("Exact-tail development cache-index schema drifted.")
    index_unhashed = {key: value for key, value in index.items() if key != "cache_index_hash"}
    if (
        index.get("schema_version") != CACHE_INDEX_SCHEMA
        or index.get("artifact_id") != config.input_artifact_ids[3]
        or index.get("dataset_family") != "MIDOG++"
        or index.get("representation_id") != EXPECTED_CACHE_REPRESENTATION_ID
        or index.get("feature_dim") != COMMON_OUTPUT_DIM
        or index.get("dtype") != "float32"
        or index.get("labels_stored") is not False
        or index.get("reservation_index_hash") != reservation.reservation_hash
        or index.get("cache_index_hash") != canonical_sha256(index_unhashed)
    ):
        raise ProtocolError("Exact-tail development cache-index identity drifted.")
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes)) or not raw_shards:
        raise ProtocolError("Exact-tail development cache index has no shards.")
    shards: dict[str, Mapping[str, object]] = {}
    shard_keys = {"relative_path", "file_sha256", "shape", "dtype"}
    for raw in raw_shards:
        if not isinstance(raw, Mapping) or set(raw) != shard_keys:
            raise ProtocolError("Exact-tail development cache shard schema drifted.")
        relative = _safe_relative(str(raw["relative_path"]))
        if relative in shards or not relative.endswith((".npy", ".npz")):
            raise ProtocolError("Exact-tail development cache shard identity drifted.")
        shape = raw["shape"]
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or len(shape) != 2
            or int(shape[0]) <= 0
            or int(shape[1]) != COMMON_OUTPUT_DIM
            or raw.get("dtype") != "float32"
        ):
            raise ProtocolError("Exact-tail development cache shard geometry drifted.")
        _require_sha256(str(raw["file_sha256"]), "cache shard")
        shard_path = _safe_member(root, relative)
        if not shard_path.is_file() or _sha256_file(shard_path) != raw["file_sha256"]:
            raise ProtocolError("Exact-tail development cache shard bytes drifted.")
        observed = _load_shard(shard_path)
        if observed.shape != (int(shape[0]), int(shape[1])) or observed.dtype != np.float32:
            raise ProtocolError("Exact-tail development cache shard header drifted.")
        shards[relative] = MappingProxyType(dict(raw))

    content = _read_json(content_index_path, "development cache content index")
    required_content = {
        "schema_version",
        "artifact_id",
        "cache_index_member",
        "member_sha256",
        "cache_binding_hash",
    }
    if set(content) != required_content:
        raise ProtocolError("Exact-tail development content-index schema drifted.")
    member_sha = content.get("member_sha256")
    if not isinstance(member_sha, Mapping):
        raise ProtocolError("Exact-tail development content-index members are malformed.")
    normalized_members = {str(key): str(value) for key, value in member_sha.items()}
    discovered = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in discovered):
        raise ProtocolError("Exact-tail development cache forbids symbolic links.")
    actual_members = {
        str(path.relative_to(root))
        for path in discovered
        if path.is_file() and path.resolve() != content_index_path.resolve()
    }
    expected_scientific = {CACHE_INDEX_MEMBER, *shards}
    if (
        set(normalized_members) != actual_members
        or actual_members != expected_scientific
    ):
        raise ProtocolError("Exact-tail development cache is not closed-world indexed.")
    for member, expected_sha in normalized_members.items():
        _require_sha256(expected_sha, "cache content member")
        if _sha256_file(_safe_member(root, member)) != expected_sha:
            raise ProtocolError("Exact-tail development cache content bytes drifted.")
    content_unhashed = {key: value for key, value in content.items() if key != "cache_binding_hash"}
    if (
        content.get("schema_version") != CACHE_CONTENT_INDEX_SCHEMA
        or content.get("artifact_id") != config.input_artifact_ids[3]
        or content.get("cache_index_member") != CACHE_INDEX_MEMBER
        or normalized_members.get(CACHE_INDEX_MEMBER) != _sha256_file(cache_index_path)
        or content.get("cache_binding_hash") != canonical_sha256(content_unhashed)
        or content.get("cache_binding_hash")
        != _attested_cache_binding(config.reservation_attestation_path)
    ):
        raise ProtocolError("Exact-tail development cache binding drifted.")
    reservation_shards = {
        row.cache_shard_path
        for partition in reservation.partitions.values()
        for row in (*partition.support_rows, *partition.evaluation_rows)
    }
    if not reservation_shards.issubset(shards):
        raise ProtocolError("Exact-tail reservation references an unadmitted cache shard.")
    return MappingProxyType(shards)


def parse_development_partition(raw: object) -> DevelopmentPartition:
    if not isinstance(raw, Mapping) or set(raw) != {
        "center",
        "support_case_ids",
        "support_rows",
        "evaluation_rows",
        "target_evaluation_case_ids",
        "labels_present",
        "reservation_hash",
    }:
        raise ProtocolError("Exact-tail development partition payload is malformed.")
    support_rows = _rows(raw["support_rows"], expected_role="development_support")
    evaluation_rows = _rows(
        raw["evaluation_rows"], expected_role="development_evaluation"
    )
    return DevelopmentPartition(
        center=str(raw["center"]),
        support_case_ids=tuple(str(value) for value in raw["support_case_ids"]),
        support_rows=support_rows,
        evaluation_rows=evaluation_rows,
        target_evaluation_case_ids=tuple(
            str(value) for value in raw["target_evaluation_case_ids"]
        ),
        reservation_hash=str(raw["reservation_hash"]),
        labels_present=raw["labels_present"] is True,
    )


def _rows(raw: object, *, expected_role: str) -> tuple[EvaluationRowIdentity, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ProtocolError("Exact-tail reservation row list is malformed.")
    rows: list[EvaluationRowIdentity] = []
    expected_keys = {
        "row_ordinal",
        "manifest_row_index",
        "sample_id",
        "case_id",
        "center",
        "split",
        "cache_shard_path",
        "cache_row_index",
        "partition_role",
    }
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ProtocolError("Exact-tail reservation row schema drifted.")
        if value.get("partition_role") != expected_role:
            raise ProtocolError("Exact-tail reservation row role drifted.")
        rows.append(
            EvaluationRowIdentity(
                row_ordinal=int(value["row_ordinal"]),
                manifest_row_index=int(value["manifest_row_index"]),
                sample_id=str(value["sample_id"]),
                case_id=str(value["case_id"]),
                center=str(value["center"]),
                split=str(value["split"]),
                cache_shard_path=str(value["cache_shard_path"]),
                cache_row_index=int(value["cache_row_index"]),
                partition_role=str(value["partition_role"]),
            )
        )
    return tuple(rows)


def _metadata_similarity(raw: object) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(raw, Mapping) or {str(key) for key in raw} != set(CENTERS):
        raise ProtocolError("Exact-tail metadata similarity grid lacks query centers.")
    normalized_raw = {str(key): value for key, value in raw.items()}
    result: dict[str, Mapping[str, float]] = {}
    for query in CENTERS:
        values = normalized_raw[query]
        expected_sources = tuple(center for center in CENTERS if center != query)
        if not isinstance(values, Mapping) or {str(key) for key in values} != set(
            expected_sources
        ):
            raise ProtocolError("Exact-tail metadata similarity candidate grid drifted.")
        normalized_values = {str(key): value for key, value in values.items()}
        scores = {source: float(normalized_values[source]) for source in expected_sources}
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores.values()):
            raise ProtocolError("Exact-tail metadata similarity must lie in [0,1].")
        result[query] = MappingProxyType(scores)
    return MappingProxyType(result)


def _materialize_rows(
    cache_root: Path,
    rows: Sequence[EvaluationRowIdentity],
    *,
    admitted_shards: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    shard_cache: dict[str, np.ndarray] = {}
    values: list[np.ndarray] = []
    root = cache_root.resolve()
    for row in rows:
        if row.cache_shard_path not in admitted_shards:
            raise ProtocolError("Exact-tail row references an unadmitted cache shard.")
        relative = Path(row.cache_shard_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError("Exact-tail cache shard path is unsafe.")
        shard_path = (root / relative).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as exc:
            raise ProtocolError("Exact-tail cache shard escaped its root.") from exc
        key = str(shard_path)
        if key not in shard_cache:
            shard_cache[key] = _load_shard(shard_path)
        shard = shard_cache[key]
        if row.cache_row_index >= len(shard):
            raise ProtocolError("Exact-tail cache row index is out of bounds.")
        values.append(np.asarray(shard[row.cache_row_index], dtype=np.float32))
    matrix = np.ascontiguousarray(np.stack(values), dtype=np.float32)
    if matrix.shape != (len(rows), COMMON_OUTPUT_DIM) or not np.isfinite(matrix).all():
        raise ProtocolError("Exact-tail prepared cache array geometry drifted.")
    return matrix


def _load_shard(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ProtocolError(f"Exact-tail cache shard is absent: {path}.")
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"embeddings"}:
                raise ProtocolError("Exact-tail NPZ cache shard schema drifted.")
            array = np.asarray(payload["embeddings"])
    else:
        raise ProtocolError("Exact-tail fresh cache shards must be NPY or NPZ.")
    if array.ndim != 2 or array.shape[1] != COMMON_OUTPUT_DIM or not np.isfinite(array).all():
        raise ProtocolError("Exact-tail cache shard geometry drifted.")
    return array


def _attested_cache_binding(path: Path) -> str:
    raw = _read_json(path, "fresh reservation attestation")
    value = str(raw.get("development_cache_binding_hash", ""))
    _require_sha256(value, "attested development cache binding")
    return value


def _read_json(path: Path, role: str) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Exact-tail {role} is unreadable.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError(f"Exact-tail {role} must be an object.")
    return raw


def _safe_relative(value: str) -> str:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError("Exact-tail cache member path is unsafe.")
    return str(relative)


def _safe_member(root: Path, relative: str) -> Path:
    member = (root / _safe_relative(relative)).resolve()
    try:
        member.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("Exact-tail cache member escaped its root.") from exc
    return member


def _require_sha256(value: str, role: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError(f"Exact-tail {role} SHA-256 is malformed.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
        handle.flush()
    temporary.replace(path)


__all__ = (
    "RESERVATION_MEMBER",
    "RESERVATION_SCHEMA",
    "CACHE_INDEX_MEMBER",
    "CACHE_CONTENT_INDEX_MEMBER",
    "REQUIRED_CACHE_FILES",
    "DevelopmentReservation",
    "PreparedDevelopmentInputs",
    "load_development_reservation",
    "parse_development_partition",
    "prepare_development_cache_arrays",
    "validate_development_cache_binding",
)
