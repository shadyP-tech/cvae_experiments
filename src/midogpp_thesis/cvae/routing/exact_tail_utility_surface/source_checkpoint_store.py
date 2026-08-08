"""Hash-valid checkpoint publication and loading for exact-tail source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM, SourceGenerationKey
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TRAINING_SEEDS,
)
from .source_contracts import (
    FeatureComponentRecord,
    GeneratedDevelopmentCache,
    SourceBlockRecord,
)


def load_component_arrays(
    cache: GeneratedDevelopmentCache,
    record: FeatureComponentRecord,
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    path = safe_member(cache.root, record.relative_path)
    if sha256_file(path) != record.file_sha256:
        raise ProtocolError("Exact-tail feature component file drifted.")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "reconstruction_0",
            "reconstruction_1",
            "kl_0",
            "kl_1",
        }:
            raise ProtocolError("Exact-tail feature component NPZ schema drifted.")
        reconstruction = {
            label: np.asarray(payload[f"reconstruction_{label}"], dtype=np.float64)
            for label in (0, 1)
        }
        kl = {
            label: np.asarray(payload[f"kl_{label}"], dtype=np.float64)
            for label in (0, 1)
        }
    return reconstruction, kl


def publish_source_record(
    raw: Mapping[str, object], *, canonical_root: Path, scratch_root: Path | None
) -> SourceBlockRecord:
    source_path = Path(str(raw["path"])).resolve()
    destination = canonical_root / f"arrays/source_{raw['stream_id']}.npy"
    publish_file(source_path, destination, str(raw["file_sha256"]), scratch_root)
    record = SourceBlockRecord(
        source_center=str(raw["source_center"]),
        training_seed=int(raw["training_seed"]),
        generation_seed=int(raw["generation_seed"]),
        stream_id=str(raw["stream_id"]),
        expert_lock_hash=str(raw["expert_lock_hash"]),
        relative_path=str(destination.relative_to(canonical_root)),
        file_sha256=str(raw["file_sha256"]),
        output_sha256=str(raw["output_sha256"]),
        rows_per_class=int(raw["rows_per_class"]),
        feature_dim=int(raw["feature_dim"]),
    )
    validate_source_record(record)
    atomic_json(
        canonical_root / f"metadata/source_{record.stream_id}.json",
        record.to_payload(),
    )
    return record


def publish_component_record(
    raw: Mapping[str, object], *, canonical_root: Path, scratch_root: Path | None
) -> FeatureComponentRecord:
    source_path = Path(str(raw["path"])).resolve()
    destination = canonical_root / (
        f"features/q{raw['query_center']}_e{raw['candidate_source']}_"
        f"train{raw['training_seed']}.npz"
    )
    publish_file(source_path, destination, str(raw["file_sha256"]), scratch_root)
    record = FeatureComponentRecord(
        query_center=str(raw["query_center"]),
        candidate_source=str(raw["candidate_source"]),
        training_seed=int(raw["training_seed"]),
        relative_path=str(destination.relative_to(canonical_root)),
        file_sha256=str(raw["file_sha256"]),
        case_equal_energy=float(raw["case_equal_energy"]),
        linear_kernel_mmd2_by_generation_seed={
            int(key): float(value)
            for key, value in dict(raw["linear_kernel_mmd2_by_generation_seed"]).items()
        },
        support_partition_hash=str(raw["support_partition_hash"]),
    )
    validate_component_record(record)
    atomic_json(
        canonical_root
        / (
            f"metadata/component_q{record.query_center}_e{record.candidate_source}_"
            f"train{record.training_seed}.json"
        ),
        record.to_payload(),
    )
    return record


def load_source_record(
    root: Path, key: SourceGenerationKey
) -> SourceBlockRecord | None:
    path = root / f"metadata/source_{key.stream_id}.json"
    if not path.is_file():
        return None
    try:
        raw = json_object(path)
        record = SourceBlockRecord(
            source_center=str(raw["source_center"]),
            training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]),
            stream_id=str(raw["stream_id"]),
            expert_lock_hash=str(raw["expert_lock_hash"]),
            relative_path=str(raw["relative_path"]),
            file_sha256=str(raw["file_sha256"]),
            output_sha256=str(raw["output_sha256"]),
            rows_per_class=int(raw["rows_per_class"]),
            feature_dim=int(raw["feature_dim"]),
        )
        validate_source_record(record)
        member = safe_member(root, record.relative_path)
        if (
            record.key != (key.source_center, key.training_seed, key.generation_seed)
            or record.stream_id != key.stream_id
            or record.expert_lock_hash != key.expert_lock_hash
            or not member.is_file()
            or sha256_file(member) != record.file_sha256
        ):
            raise ProtocolError(
                "Exact-tail source checkpoint binding or member hash drifted."
            )
        return record
    except ProtocolError:
        raise
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise ProtocolError("Exact-tail source checkpoint metadata is invalid.") from exc


def load_component_record(
    root: Path,
    *,
    query: str,
    source: str,
    training_seed: int,
    support_partition_hash: str,
) -> FeatureComponentRecord | None:
    path = root / f"metadata/component_q{query}_e{source}_train{training_seed}.json"
    if not path.is_file():
        return None
    try:
        raw = json_object(path)
        record = FeatureComponentRecord(
            query_center=str(raw["query_center"]),
            candidate_source=str(raw["candidate_source"]),
            training_seed=int(raw["training_seed"]),
            relative_path=str(raw["relative_path"]),
            file_sha256=str(raw["file_sha256"]),
            case_equal_energy=float(raw["case_equal_energy"]),
            linear_kernel_mmd2_by_generation_seed={
                int(key): float(value)
                for key, value in dict(
                    raw["linear_kernel_mmd2_by_generation_seed"]
                ).items()
            },
            support_partition_hash=str(raw["support_partition_hash"]),
        )
        validate_component_record(record)
        member = safe_member(root, record.relative_path)
        if (
            record.key != (query, source, training_seed)
            or record.support_partition_hash != support_partition_hash
            or not member.is_file()
            or sha256_file(member) != record.file_sha256
        ):
            raise ProtocolError(
                "Exact-tail component checkpoint binding or member hash drifted."
            )
        return record
    except ProtocolError:
        raise
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise ProtocolError(
            "Exact-tail component checkpoint metadata is invalid."
        ) from exc


def validate_source_record(record: SourceBlockRecord) -> None:
    if (
        record.source_center not in CENTERS
        or record.training_seed not in TRAINING_SEEDS
        or record.generation_seed not in GENERATION_SEEDS
        or record.rows_per_class != SOURCE_PREFIX_ROWS_PER_CLASS
        or record.feature_dim != COMMON_OUTPUT_DIM
        or len(record.file_sha256) != 64
        or len(record.output_sha256) != 64
    ):
        raise ProtocolError("Exact-tail source record identity drifted.")


def validate_component_record(record: FeatureComponentRecord) -> None:
    if (
        record.query_center not in CENTERS
        or record.candidate_source not in CENTERS
        or record.query_center == record.candidate_source
        or record.training_seed not in TRAINING_SEEDS
        or set(record.linear_kernel_mmd2_by_generation_seed) != set(GENERATION_SEEDS)
        or any(
            not np.isfinite(value) or value < 0.0
            for value in record.linear_kernel_mmd2_by_generation_seed.values()
        )
        or not np.isfinite(record.case_equal_energy)
        or len(record.file_sha256) != 64
    ):
        raise ProtocolError("Exact-tail feature component identity drifted.")


def publish_file(
    source: Path, destination: Path, digest: str, scratch_root: Path | None
) -> None:
    if not source.is_file() or sha256_file(source) != digest:
        raise ProtocolError("Exact-tail worker file hash drifted before publication.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        raise ProtocolError("Exact-tail scratch publication changed file bytes.")
    temporary.replace(destination)
    if scratch_root is not None:
        try:
            source.relative_to(scratch_root.resolve())
        except ValueError as exc:
            raise ProtocolError("Exact-tail worker output escaped scratch root.") from exc


def safe_member(root: Path, relative: str) -> Path:
    member = (root.resolve() / relative).resolve()
    try:
        member.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError("Exact-tail cache member escaped its root.") from exc
    return member


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
        handle.flush()
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Exact-tail JSON must be an object.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "atomic_json",
    "atomic_save_npy",
    "atomic_save_npz",
    "json_object",
    "load_component_arrays",
    "load_component_record",
    "load_source_record",
    "publish_component_record",
    "publish_file",
    "publish_source_record",
    "safe_member",
    "sha256_file",
    "validate_component_record",
    "validate_source_record",
)
