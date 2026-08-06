"""Durable content-addressed realization of Stage-40 source streams."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import errno
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import BinaryIO, Callable, Mapping
import zipfile

import numpy as np

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.serialization import (
    RoutingAuthorizedExpert,
    load_routing_authorized_expert,
)
from ..generation import (
    GenerationLock,
    GeneratedBlock,
    generate_source_block,
    source_generation_plan,
)
from ..generation.contracts import (
    COMMON_OUTPUT_DIM,
    TOTAL_PER_CLASS,
    SourceGenerationKey,
)
from ..protocol import ProtocolError
from .contracts import array_bundle_sha256


SOURCE_BLOCK_CACHE_SCHEMA = "midogpp_stage70_source_block_cache_v2"

_EVALUATION_SPLIT = "test_previously_consumed_for_representation_adoption"
_EXPECTED_SOURCE_BLOCK_COUNT = 81
_NPZ_MEMBER_NAMES = frozenset(
    {"embeddings.npy", "labels.npy", "metadata_json.npy"}
)
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cache_key",
        "source_generation_key",
        "checkpoint_hash",
        "bank_lock_hash",
        "generation_lock_hash",
        "dataset_contract_hash",
        "evaluation_split",
        "representation_id",
        "backbone_identity_hash",
        "budget_per_class",
        "output_sha256",
    }
)
_COPY_FALLBACK_ERRNOS = frozenset(
    value
    for value in (
        errno.EXDEV,
        errno.EPERM,
        errno.EACCES,
        errno.EMLINK,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


@dataclass(frozen=True)
class _CacheBinding:
    key: SourceGenerationKey
    cache_key: str
    protocol_version: str
    checkpoint_hash: str
    bank_lock_hash: str
    generation_lock_hash: str
    dataset_contract_hash: str
    evaluation_split: str
    representation_id: str
    backbone_identity_hash: str
    budget_per_class: int

    def identity_payload(self) -> dict[str, object]:
        return _cache_identity_payload(
            key=self.key,
            protocol_version=self.protocol_version,
            checkpoint_hash=self.checkpoint_hash,
            bank_lock_hash=self.bank_lock_hash,
            generation_lock_hash=self.generation_lock_hash,
            dataset_contract_hash=self.dataset_contract_hash,
            evaluation_split=self.evaluation_split,
            representation_id=self.representation_id,
            backbone_identity_hash=self.backbone_identity_hash,
            budget_per_class=self.budget_per_class,
        )

    def metadata_payload(self, *, output_sha256: str) -> dict[str, object]:
        return {
            "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
            "protocol_version": self.protocol_version,
            "cache_key": self.cache_key,
            "source_generation_key": self.key.to_payload(),
            "checkpoint_hash": self.checkpoint_hash,
            "bank_lock_hash": self.bank_lock_hash,
            "generation_lock_hash": self.generation_lock_hash,
            "dataset_contract_hash": self.dataset_contract_hash,
            "evaluation_split": self.evaluation_split,
            "representation_id": self.representation_id,
            "backbone_identity_hash": self.backbone_identity_hash,
            "budget_per_class": self.budget_per_class,
            "output_sha256": output_sha256,
        }


@dataclass(frozen=True)
class _StoredMember:
    path: Path
    binding: _CacheBinding
    output_sha256: str
    file_sha256: str


class SourceBlockStore(Mapping[str, GeneratedBlock]):
    """Lazy mapping that keeps cached source blocks off heap between uses."""

    def __init__(
        self,
        members: Mapping[str, _StoredMember],
    ) -> None:
        self._members = dict(members)
        self._cached_load = lru_cache(maxsize=10)(self._load)

    def __getitem__(self, stream_id: str) -> GeneratedBlock:
        """Load one stream with a bounded cache for one seed-pair traversal."""

        return self._cached_load(stream_id)

    def _load(self, stream_id: str) -> GeneratedBlock:
        """Validate and load one uncached member."""

        try:
            member = self._members[stream_id]
        except KeyError as exc:
            raise KeyError(stream_id) from exc
        block, observed_sha256 = _read_cached_block(
            member.path,
            binding=member.binding,
            expected_output_sha256=member.output_sha256,
        )
        if observed_sha256 != member.file_sha256:
            raise ProtocolError(
                "Stage-70 source-block member changed after materialization."
            )
        return block

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)


def source_block_cache_key(
    *,
    key: SourceGenerationKey,
    dataset_contract_hash: str,
    evaluation_split: str,
    representation_id: str,
    backbone_identity_hash: str,
    checkpoint_hash: str,
    bank_lock_hash: str,
    generation_lock_hash: str,
    protocol_version: str = SOURCE_BLOCK_CACHE_SCHEMA,
) -> str:
    """Return the complete persisted cache identity required by Stage 70."""

    return stable_hash(
        _cache_identity_payload(
            key=key,
            protocol_version=protocol_version,
            checkpoint_hash=checkpoint_hash,
            bank_lock_hash=bank_lock_hash,
            generation_lock_hash=generation_lock_hash,
            dataset_contract_hash=dataset_contract_hash,
            evaluation_split=evaluation_split,
            representation_id=representation_id,
            backbone_identity_hash=backbone_identity_hash,
            budget_per_class=TOTAL_PER_CLASS,
        )
    )


def materialize_source_blocks(
    *,
    generation_lock: GenerationLock,
    bank_root: str | Path,
    cache_root: str | Path,
    dataset_contract_hash: str,
    representation_id: str,
    backbone_identity_hash: str,
    device: str,
    publication_root: str | Path | None = None,
    expert_loader: Callable[
        ..., RoutingAuthorizedExpert
    ] = load_routing_authorized_expert,
    block_generator: Callable[..., GeneratedBlock] = generate_source_block,
) -> tuple[SourceBlockStore, tuple[dict[str, object], ...]]:
    """Materialize all source streams and optionally publish them.

    ``cache_root`` is the persistent, crash-reusable store.  When
    ``publication_root`` is supplied, each validated persistent member is
    hard-linked into that evaluator-artifact directory (with a verified
    atomic-copy fallback for filesystems that do not support links).  Omitting
    it keeps the historical behavior in which the cache directory is also the
    artifact member directory.
    """

    plan = tuple(source_generation_plan(generation_lock))
    if (
        len(plan) != _EXPECTED_SOURCE_BLOCK_COUNT
        or len({key.stream_id for key in plan}) != _EXPECTED_SOURCE_BLOCK_COUNT
    ):
        raise ProtocolError("Stage-70 maximum source-block coverage drifted.")
    persistent_root = _prepare_root(Path(cache_root), role="persistent cache")
    member_root = (
        persistent_root
        if publication_root is None
        else _prepare_root(Path(publication_root), role="publication")
    )
    members: dict[str, _StoredMember] = {}
    records: list[dict[str, object]] = []
    loaded_expert: RoutingAuthorizedExpert | None = None
    loaded_expert_key: tuple[str, int] | None = None
    for key in plan:
        requested_expert_key = (key.source_center, key.training_seed)
        if loaded_expert is None or loaded_expert_key != requested_expert_key:
            loaded_expert = expert_loader(
                bank_root,
                source_center=key.source_center,
                training_seed=key.training_seed,
                device=device,
            )
            loaded_expert_key = requested_expert_key
        expert = loaded_expert
        _validate_expert_binding(expert, key=key)
        cache_key = source_block_cache_key(
            key=key,
            dataset_contract_hash=dataset_contract_hash,
            evaluation_split=_EVALUATION_SPLIT,
            representation_id=representation_id,
            backbone_identity_hash=backbone_identity_hash,
            checkpoint_hash=expert.checkpoint_hash,
            bank_lock_hash=generation_lock.bank_lock_hash,
            generation_lock_hash=generation_lock.generation_lock_hash,
        )
        binding = _CacheBinding(
            key=key,
            cache_key=cache_key,
            protocol_version=SOURCE_BLOCK_CACHE_SCHEMA,
            checkpoint_hash=expert.checkpoint_hash,
            bank_lock_hash=generation_lock.bank_lock_hash,
            generation_lock_hash=generation_lock.generation_lock_hash,
            dataset_contract_hash=dataset_contract_hash,
            evaluation_split=_EVALUATION_SPLIT,
            representation_id=representation_id,
            backbone_identity_hash=backbone_identity_hash,
            budget_per_class=TOTAL_PER_CLASS,
        )
        filename = f"{cache_key}.npz"
        persistent_path = persistent_root / filename
        if _path_is_present(persistent_path):
            block, persistent_sha256 = _read_cached_block(
                persistent_path,
                binding=binding,
            )
            cache_status = "REUSED_VALIDATED"
        else:
            generated = block_generator(
                expert,
                key,
                per_class=TOTAL_PER_CLASS,
                device=device,
            )
            block, persistent_sha256, installed = _write_cached_block(
                persistent_path,
                block=generated,
                binding=binding,
            )
            cache_status = "GENERATED" if installed else "REUSED_VALIDATED"

        member_path = member_root / filename
        if member_path == persistent_path:
            member_sha256 = persistent_sha256
        else:
            block, member_sha256 = _publish_cached_block(
                persistent_path,
                member_path,
                binding=binding,
                expected_output_sha256=block.output_sha256,
                expected_file_sha256=persistent_sha256,
            )
        members[key.stream_id] = _StoredMember(
            path=member_path,
            binding=binding,
            output_sha256=block.output_sha256,
            file_sha256=member_sha256,
        )
        records.append(
            {
                "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
                "cache_key": cache_key,
                "cache_status": cache_status,
                "source_center": key.source_center,
                "training_seed": key.training_seed,
                "generation_seed": key.generation_seed,
                "source_stream_id": key.stream_id,
                "expert_lock_hash": key.expert_lock_hash,
                "checkpoint_hash": expert.checkpoint_hash,
                "bank_lock_hash": generation_lock.bank_lock_hash,
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "dataset_contract_hash": dataset_contract_hash,
                "evaluation_split": _EVALUATION_SPLIT,
                "representation_id": representation_id,
                "backbone_identity_hash": backbone_identity_hash,
                "budget_per_class": TOTAL_PER_CLASS,
                "output_sha256": block.output_sha256,
                "path": filename,
                "persistent_path": filename,
                "member_path": f"arrays/source_blocks/{filename}",
                "member_sha256": member_sha256,
            }
        )
    if (
        len(members) != _EXPECTED_SOURCE_BLOCK_COUNT
        or len(records) != _EXPECTED_SOURCE_BLOCK_COUNT
    ):
        raise ProtocolError("Stage-70 maximum source-block coverage drifted.")
    return SourceBlockStore(members), tuple(records)


def _cache_identity_payload(
    *,
    key: SourceGenerationKey,
    protocol_version: str,
    checkpoint_hash: str,
    bank_lock_hash: str,
    generation_lock_hash: str,
    dataset_contract_hash: str,
    evaluation_split: str,
    representation_id: str,
    backbone_identity_hash: str,
    budget_per_class: int,
) -> dict[str, object]:
    return {
        "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
        "protocol_version": protocol_version,
        "source_generation_key": key.to_payload(),
        "checkpoint_hash": checkpoint_hash,
        "bank_lock_hash": bank_lock_hash,
        "generation_lock_hash": generation_lock_hash,
        "dataset_contract_hash": dataset_contract_hash,
        "evaluation_split": evaluation_split,
        "representation_id": representation_id,
        "backbone_identity_hash": backbone_identity_hash,
        "budget_per_class": int(budget_per_class),
    }


def _validate_expert_binding(
    expert: RoutingAuthorizedExpert,
    *,
    key: SourceGenerationKey,
) -> None:
    if (
        expert.source_center != key.source_center
        or expert.training_seed != key.training_seed
        or expert.expert_lock_hash != key.expert_lock_hash
    ):
        raise ProtocolError(
            "Loaded expert does not match the Stage-70 source key."
        )


def _write_cached_block(
    path: Path,
    *,
    block: GeneratedBlock,
    binding: _CacheBinding,
) -> tuple[GeneratedBlock, str, bool]:
    canonical = _validate_generated_block(block, binding=binding)
    metadata = binding.metadata_payload(output_sha256=canonical.output_sha256)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            embeddings=canonical.embeddings,
            labels=canonical.labels,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _, temporary_sha256 = _read_cached_block(
            temporary,
            binding=binding,
            expected_output_sha256=canonical.output_sha256,
        )
        if _path_is_present(path):
            existing, existing_sha256 = _read_cached_block(
                path,
                binding=binding,
                expected_output_sha256=canonical.output_sha256,
            )
            return existing, existing_sha256, False
        temporary.replace(path)
        _fsync_file(path)
        _fsync_directory(path.parent)
        installed, installed_sha256 = _read_cached_block(
            path,
            binding=binding,
            expected_output_sha256=canonical.output_sha256,
        )
        if installed_sha256 != temporary_sha256:
            raise ProtocolError(
                "Stage-70 source-block atomic write changed file bytes."
            )
        return installed, installed_sha256, True
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _publish_cached_block(
    source: Path,
    destination: Path,
    *,
    binding: _CacheBinding,
    expected_output_sha256: str,
    expected_file_sha256: str,
) -> tuple[GeneratedBlock, str]:
    if _path_is_present(destination):
        existing, existing_sha256 = _read_cached_block(
            destination,
            binding=binding,
            expected_output_sha256=expected_output_sha256,
        )
        if existing_sha256 != expected_file_sha256:
            raise ProtocolError(
                "Existing Stage-70 publication member differs from its "
                "persistent cache."
            )
        return existing, existing_sha256

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            if exc.errno not in _COPY_FALLBACK_ERRNOS:
                raise
            _copy_file_durable(source, temporary)
        else:
            _fsync_file(temporary)

        _, temporary_sha256 = _read_cached_block(
            temporary,
            binding=binding,
            expected_output_sha256=expected_output_sha256,
        )
        if temporary_sha256 != expected_file_sha256:
            raise ProtocolError(
                "Stage-70 published source-block bytes differ from "
                "persistent cache."
            )
        if _path_is_present(destination):
            existing, existing_sha256 = _read_cached_block(
                destination,
                binding=binding,
                expected_output_sha256=expected_output_sha256,
            )
            if existing_sha256 != expected_file_sha256:
                raise ProtocolError(
                    "Existing Stage-70 publication member differs from its "
                    "persistent cache."
                )
            return existing, existing_sha256
        temporary.replace(destination)
        _fsync_file(destination)
        _fsync_directory(destination.parent)
        installed, installed_sha256 = _read_cached_block(
            destination,
            binding=binding,
            expected_output_sha256=expected_output_sha256,
        )
        if installed_sha256 != expected_file_sha256:
            raise ProtocolError(
                "Stage-70 publication atomic write changed file bytes."
            )
        return installed, installed_sha256
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _copy_file_durable(source: Path, destination: Path) -> None:
    try:
        with _open_regular_file(source) as source_handle:
            with destination.open("xb") as destination_handle:
                chunks = iter(
                    lambda: source_handle.read(1024 * 1024),
                    b"",
                )
                for chunk in chunks:
                    destination_handle.write(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError(
            "Stage-70 publication temporary path already exists."
        ) from exc


def _validate_generated_block(
    block: GeneratedBlock,
    *,
    binding: _CacheBinding,
) -> GeneratedBlock:
    if block.key.to_payload() != binding.key.to_payload():
        raise ProtocolError(
            "Generated Stage-70 source block has the wrong source key."
        )
    embeddings = np.asarray(block.embeddings)
    labels = np.asarray(block.labels)
    observed_hash = _validate_arrays(embeddings, labels)
    if block.output_sha256 != observed_hash:
        raise ProtocolError(
            "Generated Stage-70 source-block output hash drifted."
        )
    return GeneratedBlock(
        key=binding.key,
        embeddings=np.ascontiguousarray(embeddings),
        labels=np.ascontiguousarray(labels),
        output_sha256=observed_hash,
    )


def _read_cached_block(
    path: Path,
    *,
    binding: _CacheBinding,
    expected_output_sha256: str | None = None,
) -> tuple[GeneratedBlock, str]:
    if stable_hash(binding.identity_payload()) != binding.cache_key:
        raise ProtocolError(
            "Stage-70 source-block cache identity is not reproducible."
        )
    try:
        with _open_regular_file(path) as handle:
            _validate_npz_members(handle, path=path)
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["embeddings"])
                labels = np.asarray(payload["labels"])
                metadata_raw = np.asarray(payload["metadata_json"])
            if metadata_raw.shape != () or metadata_raw.dtype.kind != "U":
                raise ProtocolError(
                    "Stage-70 source-block metadata is not scalar Unicode."
                )
            metadata = json.loads(str(metadata_raw.item()))
            handle.seek(0)
            file_sha256 = _sha256_handle(handle)
    except ProtocolError:
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        EOFError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ProtocolError(
            f"Cannot read Stage-70 source-block cache: {path}."
        ) from exc
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
        raise ProtocolError(
            "Stage-70 source-block metadata is not closed-world."
        )
    output_sha256 = metadata.get("output_sha256")
    if not _is_sha256(output_sha256):
        raise ProtocolError("Stage-70 source-block output hash is malformed.")
    expected_metadata = binding.metadata_payload(
        output_sha256=str(output_sha256)
    )
    if dict(metadata) != expected_metadata:
        raise ProtocolError("Stage-70 source-block cache provenance drifted.")
    if (
        expected_output_sha256 is not None
        and output_sha256 != expected_output_sha256
    ):
        raise ProtocolError(
            "Stage-70 source-block expected output hash drifted."
        )
    observed_hash = _validate_arrays(embeddings, labels)
    if observed_hash != output_sha256:
        raise ProtocolError(
            "Stage-70 source-block cache content hash drifted."
        )
    return (
        GeneratedBlock(
            key=binding.key,
            embeddings=embeddings,
            labels=labels,
            output_sha256=observed_hash,
        ),
        file_sha256,
    )


def _validate_npz_members(handle: BinaryIO, *, path: Path) -> None:
    with zipfile.ZipFile(handle) as archive:
        members = archive.infolist()
    names = [member.filename for member in members]
    if (
        len(names) != len(_NPZ_MEMBER_NAMES)
        or set(names) != _NPZ_MEMBER_NAMES
    ):
        raise ProtocolError(
            "Stage-70 source-block archive is not closed-world."
        )
    for member in members:
        unix_mode = member.external_attr >> 16
        if member.is_dir() or stat.S_ISLNK(unix_mode):
            raise ProtocolError(
                "Stage-70 source-block archive contains an unsafe "
                f"member: {path}."
            )


def _validate_arrays(embeddings: np.ndarray, labels: np.ndarray) -> str:
    if (
        embeddings.dtype != np.dtype(np.float32)
        or labels.dtype != np.dtype(np.int64)
    ):
        raise ProtocolError("Stage-70 source-block array dtype drifted.")
    if (
        embeddings.shape != (2 * TOTAL_PER_CLASS, COMMON_OUTPUT_DIM)
        or labels.shape != (2 * TOTAL_PER_CLASS,)
    ):
        raise ProtocolError("Stage-70 cached source block geometry drifted.")
    if not np.isfinite(embeddings).all():
        raise ProtocolError("Stage-70 source-block embeddings are not finite.")
    if (
        int(np.sum(labels == 0)) != TOTAL_PER_CLASS
        or int(np.sum(labels == 1)) != TOTAL_PER_CLASS
        or set(int(value) for value in np.unique(labels)) != {0, 1}
    ):
        raise ProtocolError("Stage-70 source-block class balance drifted.")
    return array_bundle_sha256(embeddings, labels)


def _prepare_root(path: Path, *, role: str) -> Path:
    if path.is_symlink():
        raise ProtocolError(
            f"Stage-70 source-block {role} root is a symlink: {path}."
        )
    if path.exists() and not path.is_dir():
        raise ProtocolError(
            f"Stage-70 source-block {role} root is not a directory: {path}."
        )
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError(
            f"Stage-70 source-block {role} root is unsafe: {path}."
        )
    symlinks = sorted(
        member.name for member in path.iterdir() if member.is_symlink()
    )
    if symlinks:
        raise ProtocolError(
            f"Stage-70 source-block {role} root contains symlinks: {symlinks}."
        )
    return path


def _path_is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _open_regular_file(path: Path) -> BinaryIO:
    if path.is_symlink():
        raise ProtocolError(
            f"Stage-70 source-block member is a symlink: {path}."
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError(
            f"Cannot open Stage-70 source-block member: {path}."
        ) from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ProtocolError(
            f"Stage-70 source-block member is not a regular file: {path}."
        )
    return os.fdopen(descriptor, "rb")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with _open_regular_file(path) as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "SOURCE_BLOCK_CACHE_SCHEMA",
    "SourceBlockStore",
    "materialize_source_blocks",
    "source_block_cache_key",
)
