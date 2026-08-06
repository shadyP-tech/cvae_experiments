from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import zipfile

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.frozen_policy_downstream import source_blocks
from midogpp_thesis.cvae.frozen_policy_downstream.contracts import (
    array_bundle_sha256,
)
from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.generation.generation import GeneratedBlock
from midogpp_thesis.cvae.protocol import ProtocolError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Harness:
    keys: tuple[SourceGenerationKey, ...]
    calls: dict[str, int]
    generation_lock: object
    expert_loader: Callable[..., object]
    block_generator: Callable[..., GeneratedBlock]

    def run(
        self,
        cache_root: Path,
        *,
        publication_root: Path | None = None,
        block_generator: Callable[..., GeneratedBlock] | None = None,
    ):
        return source_blocks.materialize_source_blocks(
            generation_lock=self.generation_lock,  # type: ignore[arg-type]
            bank_root=cache_root.parent / "bank",
            cache_root=cache_root,
            publication_root=publication_root,
            dataset_contract_hash="d" * 64,
            representation_id="virchow2_mean_patch_embedding_3840",
            backbone_identity_hash="f" * 64,
            device="cpu",
            expert_loader=self.expert_loader,  # type: ignore[arg-type]
            block_generator=block_generator or self.block_generator,
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(source_blocks, "TOTAL_PER_CLASS", 2)
    monkeypatch.setattr(source_blocks, "COMMON_OUTPUT_DIM", 3)
    monkeypatch.setattr(source_blocks, "_EXPECTED_SOURCE_BLOCK_COUNT", 3)
    expert_lock_hash = stable_hash({"expert": "0-17"})
    keys = tuple(
        SourceGenerationKey(
            source_center="0",
            training_seed=17,
            generation_seed=generation_seed,
            expert_lock_hash=expert_lock_hash,
            stream_id=stable_hash({"stream": generation_seed}),
            class_seed_by_label={
                "0": 100 + generation_seed,
                "1": 200 + generation_seed,
            },
            max_samples_per_class=2,
            equal_union_prefix_per_class=1,
        )
        for generation_seed in (17, 29, 43)
    )
    monkeypatch.setattr(
        source_blocks,
        "source_generation_plan",
        lambda _lock: keys,
    )
    calls = {"load": 0, "generate": 0}

    def expert_loader(
        _root: Path,
        *,
        source_center: str,
        training_seed: int,
        device: str,
    ) -> object:
        assert device == "cpu"
        calls["load"] += 1
        return SimpleNamespace(
            source_center=source_center,
            training_seed=training_seed,
            expert_lock_hash=expert_lock_hash,
            checkpoint_hash="c" * 64,
        )

    def block_generator(
        _expert: object,
        key: SourceGenerationKey,
        *,
        per_class: int,
        device: str,
    ) -> GeneratedBlock:
        assert per_class == 2
        assert device == "cpu"
        calls["generate"] += 1
        offset = np.float32(key.generation_seed)
        embeddings = np.arange(12, dtype=np.float32).reshape(4, 3) + offset
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        return GeneratedBlock(
            key=key,
            embeddings=embeddings,
            labels=labels,
            output_sha256=array_bundle_sha256(embeddings, labels),
        )

    generation_lock = SimpleNamespace(
        bank_lock_hash="b" * 64,
        generation_lock_hash="g" * 64,
    )
    return _Harness(
        keys=keys,
        calls=calls,
        generation_lock=generation_lock,
        expert_loader=expert_loader,
        block_generator=block_generator,
    )


def test_generates_strict_bound_source_block_members(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"

    store, records = harness.run(cache_root)

    assert len(store) == 3
    assert len(records) == 3
    assert harness.calls == {"load": 1, "generate": 3}
    assert {record["cache_status"] for record in records} == {"GENERATED"}
    first_key = harness.keys[0]
    first = records[0]
    expected_identity = {
        "schema_version": source_blocks.SOURCE_BLOCK_CACHE_SCHEMA,
        "protocol_version": source_blocks.SOURCE_BLOCK_CACHE_SCHEMA,
        "source_generation_key": first_key.to_payload(),
        "checkpoint_hash": "c" * 64,
        "bank_lock_hash": "b" * 64,
        "generation_lock_hash": "g" * 64,
        "dataset_contract_hash": "d" * 64,
        "evaluation_split": (
            "test_previously_consumed_for_representation_adoption"
        ),
        "representation_id": "virchow2_mean_patch_embedding_3840",
        "backbone_identity_hash": "f" * 64,
        "budget_per_class": 2,
    }
    assert first["cache_key"] == stable_hash(expected_identity)
    assert first["path"] == f"{first['cache_key']}.npz"
    assert first["persistent_path"] == first["path"]
    assert first["member_path"] == f"arrays/source_blocks/{first['path']}"
    member = cache_root / str(first["path"])
    assert first["member_sha256"] == _file_sha256(member)
    with zipfile.ZipFile(member) as archive:
        assert set(archive.namelist()) == {
            "embeddings.npy",
            "labels.npy",
            "metadata_json.npy",
        }
    with np.load(member, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata == {
            **expected_identity,
            "cache_key": first["cache_key"],
            "output_sha256": first["output_sha256"],
        }
    loaded = store[first_key.stream_id]
    assert loaded.embeddings.shape == (4, 3)
    assert loaded.embeddings.dtype == np.dtype(np.float32)
    assert loaded.labels.tolist() == [0, 0, 1, 1]


def test_validated_reuse_never_regenerates(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, generated_records = harness.run(cache_root)
    harness.calls["generate"] = 0

    def forbidden_generator(
        *_args: object,
        **_kwargs: object,
    ) -> GeneratedBlock:
        pytest.fail("valid persistent cache members must not be regenerated")

    store, reused_records = harness.run(
        cache_root,
        block_generator=forbidden_generator,
    )

    assert harness.calls["generate"] == 0
    assert {record["cache_status"] for record in reused_records} == {
        "REUSED_VALIDATED"
    }
    assert [record["member_sha256"] for record in reused_records] == [
        record["member_sha256"] for record in generated_records
    ]
    assert (
        store[harness.keys[-1].stream_id].output_sha256
        == reused_records[-1]["output_sha256"]
    )


def test_publishes_validated_members_as_hard_links(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    publication_root = tmp_path / "artifact" / "arrays" / "source_blocks"

    store, records = harness.run(
        cache_root,
        publication_root=publication_root,
    )

    for record in records:
        persistent = cache_root / str(record["persistent_path"])
        member = publication_root / str(record["path"])
        assert persistent.stat().st_ino == member.stat().st_ino
        assert record["member_sha256"] == _file_sha256(member)
    assert store[harness.keys[1].stream_id].embeddings.shape == (4, 3)


def test_publication_uses_verified_copy_when_link_is_unsupported(
    tmp_path: Path,
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "persistent"
    publication_root = tmp_path / "artifact" / "arrays" / "source_blocks"

    def unsupported_link(_source: Path, _destination: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(source_blocks.os, "link", unsupported_link)
    _, records = harness.run(cache_root, publication_root=publication_root)

    for record in records:
        persistent = cache_root / str(record["persistent_path"])
        member = publication_root / str(record["path"])
        assert persistent.stat().st_ino != member.stat().st_ino
        assert _file_sha256(persistent) == _file_sha256(member)
        assert record["member_sha256"] == _file_sha256(member)


def test_tampered_member_fails_closed_without_regeneration(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    member = cache_root / str(records[0]["path"])
    with np.load(member, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"]).copy()
        labels = np.asarray(archive["labels"]).copy()
        metadata_json = np.asarray(archive["metadata_json"]).copy()
    embeddings[0, 0] += np.float32(1.0)
    np.savez_compressed(
        member,
        embeddings=embeddings,
        labels=labels,
        metadata_json=metadata_json,
    )
    tampered_sha256 = _file_sha256(member)
    harness.calls["generate"] = 0

    with pytest.raises(ProtocolError, match="content hash"):
        harness.run(cache_root)

    assert harness.calls["generate"] == 0
    assert _file_sha256(member) == tampered_sha256


def test_every_embedded_identity_field_is_bound(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    member = cache_root / str(records[0]["path"])
    with np.load(member, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"]).copy()
        labels = np.asarray(archive["labels"]).copy()
        original = json.loads(str(archive["metadata_json"].item()))
    top_level_mutations = {
        "schema_version": "wrong-schema",
        "protocol_version": "wrong-protocol",
        "cache_key": "0" * 16,
        "checkpoint_hash": "0" * 64,
        "bank_lock_hash": "0" * 64,
        "generation_lock_hash": "0" * 64,
        "dataset_contract_hash": "0" * 64,
        "evaluation_split": "validation",
        "representation_id": "wrong-representation",
        "backbone_identity_hash": "0" * 64,
        "budget_per_class": 3,
    }
    source_key_mutations = {
        "schema_version": "wrong-source-key-schema",
        "source_center": "8",
        "training_seed": 999,
        "generation_seed": 999,
        "expert_lock_hash": "0" * 16,
        "stream_id": "0" * 16,
        "class_seed_by_label": {"0": 999, "1": 999},
        "max_samples_per_class": 3,
        "equal_union_prefix_per_class": 2,
    }
    harness.calls["generate"] = 0

    for field, value in top_level_mutations.items():
        tampered = deepcopy(original)
        tampered[field] = value
        np.savez_compressed(
            member,
            embeddings=embeddings,
            labels=labels,
            metadata_json=np.asarray(json.dumps(tampered, sort_keys=True)),
        )
        with pytest.raises(ProtocolError, match="provenance drifted"):
            harness.run(cache_root)

    for field, value in source_key_mutations.items():
        tampered = deepcopy(original)
        tampered["source_generation_key"][field] = value
        np.savez_compressed(
            member,
            embeddings=embeddings,
            labels=labels,
            metadata_json=np.asarray(json.dumps(tampered, sort_keys=True)),
        )
        with pytest.raises(ProtocolError, match="provenance drifted"):
            harness.run(cache_root)

    assert harness.calls["generate"] == 0


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("dtype", "dtype drifted"),
        ("shape", "geometry drifted"),
        ("nonfinite", "not finite"),
        ("balance", "class balance drifted"),
        ("hash", "output hash drifted"),
    ),
)
def test_invalid_generated_arrays_are_rejected_before_persistence(
    tmp_path: Path,
    harness: _Harness,
    case: str,
    message: str,
) -> None:
    cache_root = tmp_path / "persistent"

    def invalid_generator(
        expert: object,
        key: SourceGenerationKey,
        *,
        per_class: int,
        device: str,
    ) -> GeneratedBlock:
        valid = harness.block_generator(
            expert,
            key,
            per_class=per_class,
            device=device,
        )
        embeddings = valid.embeddings.copy()
        labels = valid.labels.copy()
        if case == "dtype":
            embeddings = embeddings.astype(np.float64)
        elif case == "shape":
            embeddings = embeddings[:-1]
        elif case == "nonfinite":
            embeddings[0, 0] = np.nan
        elif case == "balance":
            labels[:] = np.asarray([0, 0, 0, 1], dtype=np.int64)
        output_sha256 = array_bundle_sha256(embeddings, labels)
        if case == "hash":
            output_sha256 = "0" * 64
        return GeneratedBlock(
            key=key,
            embeddings=embeddings,
            labels=labels,
            output_sha256=output_sha256,
        )

    with pytest.raises(ProtocolError, match=message):
        harness.run(cache_root, block_generator=invalid_generator)

    assert not tuple(cache_root.glob("*.npz"))


def test_closed_world_archive_rejects_an_extra_member(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    member = cache_root / str(records[0]["path"])
    with zipfile.ZipFile(member, mode="a") as archive:
        archive.writestr("unexpected.npy", b"not allowed")
    harness.calls["generate"] = 0

    with pytest.raises(ProtocolError, match="not closed-world"):
        harness.run(cache_root)

    assert harness.calls["generate"] == 0


def test_symlink_member_fails_closed_without_regeneration(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    member = cache_root / str(records[0]["path"])
    backing = tmp_path / "backing.npz"
    member.replace(backing)
    member.symlink_to(backing)
    harness.calls["generate"] = 0

    with pytest.raises(ProtocolError, match="contains symlinks"):
        harness.run(cache_root)

    assert harness.calls["generate"] == 0
    assert member.is_symlink()


def test_partial_member_is_never_overwritten_or_regenerated(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    member = cache_root / str(records[0]["path"])
    partial = b"PK\x03\x04partial-npz"
    member.write_bytes(partial)
    harness.calls["generate"] = 0

    with pytest.raises(ProtocolError, match="Cannot read"):
        harness.run(cache_root)

    assert harness.calls["generate"] == 0
    assert member.read_bytes() == partial


def test_invalid_existing_publication_is_never_overwritten(
    tmp_path: Path,
    harness: _Harness,
) -> None:
    cache_root = tmp_path / "persistent"
    _, records = harness.run(cache_root)
    publication_root = tmp_path / "artifact" / "arrays" / "source_blocks"
    publication_root.mkdir(parents=True)
    member = publication_root / str(records[0]["path"])
    invalid = b"present-but-invalid"
    member.write_bytes(invalid)
    harness.calls["generate"] = 0

    with pytest.raises(ProtocolError, match="Cannot read"):
        harness.run(cache_root, publication_root=publication_root)

    assert harness.calls["generate"] == 0
    assert member.read_bytes() == invalid
