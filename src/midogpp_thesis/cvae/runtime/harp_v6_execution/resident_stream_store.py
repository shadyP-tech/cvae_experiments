"""Durable loading and workstation-local staging for resident streams."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import atomic_copy, read_json, sha256_file
from .resident_stream_contracts import (
    COMPATIBILITY_MEMBER,
    ResidentExpertStreamCache,
    ResidentExpertStreamRecord,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
)


def load_resident_expert_streams(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_generation_lock_hash: str | None = None,
    expected_support_binding_hash: str | None = None,
) -> ResidentExpertStreamCache:
    """Load a complete canonical or staged stream bundle fail-closed."""

    array_path = root / SOURCE_ARRAY_MEMBER
    index = read_json(root / SOURCE_INDEX_MEMBER)
    lock = read_json(root / SOURCE_LOCK_MEMBER)
    compatibility = read_json(root / COMPATIBILITY_MEMBER)
    raw_records = index.get("records")
    if not isinstance(raw_records, list):
        raise ProtocolError("HARP v6 resident expert index records are absent.")
    records = tuple(
        ResidentExpertStreamRecord(
            block_ordinal=int(row["block_ordinal"]),
            source_center=str(row["source_center"]),
            training_seed=int(row["training_seed"]),
            generation_seed=int(row["generation_seed"]),
            stream_id=str(row["stream_id"]),
            expert_lock_hash=str(row["expert_lock_hash"]),
            rows_per_class=int(row["rows_per_class"]),
            output_sha256=str(row["output_sha256"]),
        )
        for row in raw_records
        if isinstance(row, Mapping)
    )
    cache = ResidentExpertStreamCache(
        root=root,
        source_array_path=array_path,
        records=records,
        lock_payload=lock,
        compatibility_payload=compatibility,
    )
    index_unhashed = {
        key: value for key, value in index.items() if key != "source_stream_index_hash"
    }
    lock_unhashed = {
        key: value for key, value in lock.items() if key != "source_stream_lock_hash"
    }
    if (
        len(records) != len(raw_records)
        or index.get("source_stream_index_hash") != stable_hash(index_unhashed)
        or lock.get("source_stream_lock_hash") != stable_hash(lock_unhashed)
        or lock.get("source_stream_index_sha256")
        != sha256_file(root / SOURCE_INDEX_MEMBER)
        or lock.get("source_array_sha256") != sha256_file(array_path)
        or lock.get("source_stream_index_hash")
        != index.get("source_stream_index_hash")
        or lock.get("support_compatibility_sha256")
        != sha256_file(root / COMPATIBILITY_MEMBER)
        or lock.get("support_compatibility_hash")
        != compatibility.get("compatibility_hash")
        or compatibility.get("compatibility_hash")
        != canonical_hash(
            {
                key: value
                for key, value in compatibility.items()
                if key != "compatibility_hash"
            }
        )
        or (
            expected_config_hash is not None
            and lock.get("config_contract_hash") != expected_config_hash
        )
        or (
            expected_generation_lock_hash is not None
            and lock.get("generation_lock_hash") != expected_generation_lock_hash
        )
        or (
            expected_support_binding_hash is not None
            and lock.get("support_binding_hash") != expected_support_binding_hash
        )
    ):
        raise ProtocolError("HARP v6 resident expert stream lock failed validation.")
    return cache


def stage_resident_expert_streams(
    cache: ResidentExpertStreamCache,
    *,
    scratch_root: Path,
    canonical_root: Path,
    local_directory: str = "source_cache",
) -> ResidentExpertStreamCache:
    """Copy a validated canonical bundle to fast local storage idempotently."""

    canonical = Path(canonical_root).resolve()
    if cache.root.resolve() != canonical:
        raise ProtocolError(
            "HARP v6 resident expert staging received another canonical root."
        )
    destination = Path(scratch_root).resolve() / local_directory
    if destination == canonical:
        return cache
    if destination.is_symlink():
        raise ProtocolError("HARP v6 resident expert staging destination is a symlink.")
    destination.mkdir(parents=True, exist_ok=True)
    members = (
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        COMPATIBILITY_MEMBER,
        SOURCE_LOCK_MEMBER,
    )
    _assert_plain_parent_chain(destination, destination)
    for member in members:
        _assert_plain_parent_chain(destination, (destination / member).parent)
    expected = {
        SOURCE_ARRAY_MEMBER: str(cache.lock_payload["source_array_sha256"]),
        SOURCE_INDEX_MEMBER: str(cache.lock_payload["source_stream_index_sha256"]),
        COMPATIBILITY_MEMBER: str(
            cache.lock_payload["support_compatibility_sha256"]
        ),
        SOURCE_LOCK_MEMBER: sha256_file(canonical / SOURCE_LOCK_MEMBER),
    }
    for member in members:
        path = destination / member
        if path.is_symlink():
            raise ProtocolError("HARP v6 resident expert staging member is a symlink.")
        if path.exists():
            if not path.is_file() or sha256_file(path) != expected[member]:
                raise ProtocolError(
                    "Existing staged HARP v6 resident expert member differs; refusing repair."
                )
            continue
        atomic_copy(
            canonical / member,
            path,
            expected_sha256=expected[member],
        )
    staged = load_resident_expert_streams(
        destination,
        expected_config_hash=str(cache.lock_payload["config_contract_hash"]),
        expected_generation_lock_hash=str(cache.lock_payload["generation_lock_hash"]),
        expected_support_binding_hash=str(cache.lock_payload["support_binding_hash"]),
    )
    if dict(staged.lock_payload) != dict(cache.lock_payload):
        raise ProtocolError("Staged HARP v6 resident expert lock differs from canonical.")
    return staged


def _assert_plain_parent_chain(root: Path, parent: Path) -> None:
    """Reject symlinked/non-directory parents within an owned staging root."""

    base = Path(root)
    current = Path(parent)
    try:
        current.relative_to(base)
    except ValueError as exc:
        raise ProtocolError(
            "HARP v6 resident expert staging parent escapes its root."
        ) from exc
    while True:
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ProtocolError("HARP v6 resident expert staging parent is unsafe.")
        if current == base:
            break
        current = current.parent


__all__ = (
    "load_resident_expert_streams",
    "stage_resident_expert_streams",
)
