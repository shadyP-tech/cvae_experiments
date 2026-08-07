from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.residual_topup_router import (
    _source_worker,
    source_cache,
    source_cache_scheduler,
    source_cache_store,
)
from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.protocol import ProtocolError


def test_source_task_grid_uses_one_expert_task_and_round_robin_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = tuple(
        _generation_key(source, training_seed, generation_seed)
        for source, training_seed, generation_seed in product(
            source_cache.CENTERS,
            source_cache.TRAINING_SEEDS,
            source_cache.GENERATION_SEEDS,
        )
    )
    monkeypatch.setattr(source_cache, "source_generation_plan", lambda _lock: keys)
    config = SimpleNamespace(expert_bank_root=tmp_path / "bank", contract_hash="config")
    lock = SimpleNamespace(generation_lock_hash="generation")
    tasks, key_map = source_cache._build_source_tasks(
        config,
        lock,
        checkpoint_root=tmp_path / "checkpoints",
        support_array_path=tmp_path / "support.npy",
        support_index_path=tmp_path / "support.json",
        support_scratch_hash="support",
    )

    assert len(tasks) == 27
    assert len(key_map) == 81
    assert [task["device"] for task in tasks[:6]] == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
    ]
    assert all(len(task["generation_keys"]) == 3 for task in tasks)
    assert len({(task["source_center"], task["training_seed"]) for task in tasks}) == 27


def test_source_cache_facade_preserves_public_types_and_module_boundaries() -> None:
    assert source_cache.SourceCache is source_cache_store.SourceCache
    assert source_cache.CachedSourceBlock is source_cache_store.CachedSourceBlock
    assert source_cache.CachedSourceKey is source_cache_store.CachedSourceKey
    assert source_cache._source_task_key is source_cache_scheduler.source_task_key

    facade = Path(source_cache.__file__).read_text(encoding="utf-8")
    assert len(facade.splitlines()) < 300
    assert "ProcessPoolExecutor" not in facade
    assert "open_memmap" not in facade


def test_durable_memmap_replace_fsyncs_file_then_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "cache.npy.tmp"
    destination = tmp_path / "cache.npy"
    temporary.write_bytes(b"complete-cache")
    events: list[tuple[str, object]] = []
    real_open = source_cache_store.os.open
    real_fsync = source_cache_store.os.fsync
    real_close = source_cache_store.os.close
    real_replace = source_cache_store.os.replace

    def tracked_open(path: object, flags: int) -> int:
        events.append(("open", Path(path)))
        return real_open(path, flags)

    def tracked_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))
        real_fsync(descriptor)

    def tracked_close(descriptor: int) -> None:
        events.append(("close", descriptor))
        real_close(descriptor)

    def tracked_replace(source: object, target: object) -> None:
        events.append(("replace", (Path(source), Path(target))))
        real_replace(source, target)

    monkeypatch.setattr(source_cache_store.os, "open", tracked_open)
    monkeypatch.setattr(source_cache_store.os, "fsync", tracked_fsync)
    monkeypatch.setattr(source_cache_store.os, "close", tracked_close)
    monkeypatch.setattr(source_cache_store.os, "replace", tracked_replace)

    source_cache_store.durable_replace(temporary, destination)

    assert destination.read_bytes() == b"complete-cache"
    assert [event for event, _value in events] == [
        "open",
        "fsync",
        "close",
        "replace",
        "open",
        "fsync",
        "close",
    ]
    assert events[0] == ("open", temporary)
    assert events[4] == ("open", tmp_path)


def test_spawn_worker_loads_expert_once_scores_all_supports_and_generates_all_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(
        monkeypatch,
        centers=("0", "1"),
        generation_seeds=(17, 42, 101),
        prefix=2,
        feature_dim=4,
    )
    support = np.arange(16, dtype=np.float32).reshape(4, 4)
    support_path = tmp_path / "support.npy"
    np.save(support_path, support, allow_pickle=False)
    support_unhashed = {
        "schema_version": "scratch",
        "shape": [4, 4],
        "dtype": "float32",
        "offsets": {
            "0": {"start": 0, "stop": 2, "case_ids": ["0-a", "0-b"]},
            "1": {"start": 2, "stop": 4, "case_ids": ["1-a", "1-b"]},
        },
        "array_sha256": _source_worker._sha256_array(support),
    }
    support_payload = {
        **support_unhashed,
        "support_scratch_hash": stable_hash(support_unhashed),
    }
    support_index_path = tmp_path / "support.json"
    support_index_path.write_text(json.dumps(support_payload), encoding="utf-8")

    calls = {"load": 0, "score": 0, "generate": 0}
    expert = SimpleNamespace(source_center="0", training_seed=17)

    def fake_load(*_args: object, **_kwargs: object) -> object:
        calls["load"] += 1
        return expert

    def fake_score(
        _expert: object, embeddings: np.ndarray, case_ids: tuple[str, ...]
    ) -> object:
        calls["score"] += 1
        values = np.arange(1, len(embeddings) + 1, dtype=np.float64)
        return SimpleNamespace(
            case_order=tuple(sorted(set(case_ids))),
            per_case={case_id: float(values[index]) for index, case_id in enumerate(case_ids)},
            per_class_energy={0: values, 1: values + 0.1},
            per_class_reconstruction_mse={0: values + 0.2, 1: values + 0.3},
            per_class_normalized_ps_kl={0: values + 0.4, 1: values + 0.5},
        )

    def fake_generate(
        _expert: object,
        key: SourceGenerationKey,
        *,
        per_class: int,
        device: str,
    ) -> object:
        assert per_class == 2
        assert device == "cpu"
        calls["generate"] += 1
        embeddings = np.full((4, 4), key.generation_seed, dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        return SimpleNamespace(
            embeddings=embeddings,
            output_sha256=_source_worker._array_bundle_sha256(embeddings, labels),
        )

    monkeypatch.setattr(_source_worker, "load_routing_authorized_expert", fake_load)
    monkeypatch.setattr(_source_worker, "score_variational_compatibility", fake_score)
    monkeypatch.setattr(_source_worker, "generate_source_block", fake_generate)
    keys = tuple(_generation_key("0", 17, seed) for seed in (17, 42, 101))
    task = {
        "task_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "generation_keys": keys,
        "device": "cpu",
        "expert_bank_root": str(tmp_path / "bank"),
        "support_array_path": str(support_path),
        "support_index_path": str(support_index_path),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "array_path": str(tmp_path / "blocks.npy"),
        "config_contract_hash": "config",
        "generation_lock_hash": "generation",
        "support_scratch_hash": support_payload["support_scratch_hash"],
    }

    payload = _source_worker.generate_source_task(task)
    assert calls == {"load": 1, "score": 2, "generate": 3}
    assert np.load(tmp_path / "blocks.npy").shape == (3, 4, 4)
    assert len(payload["compatibility_case_records"]) == 4
    assert _source_worker.load_generation_checkpoint(
        tmp_path / "checkpoint.json", task=task
    )["checkpoint_hash"] == payload["checkpoint_hash"]


def test_source_cache_shape_hash_roundtrip_and_block_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _mini_cache(tmp_path, monkeypatch)

    assert cache.array_path.stat().st_size > 0
    assert len(cache.index_rows) == 9
    assert len(cache.compatibility_case_rows) == 54
    block = cache.block("1", 42, 17)
    assert block.source_center == "1"
    assert block.training_seed == 42
    assert block.generation_seed == 17
    np.testing.assert_array_equal(block.labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(block.embeddings, np.full((4, 4), 4, np.float32))

    csv_roundtrip = source_cache.SourceCache(
        cache.array_path,
        tuple({key: str(value) for key, value in row.items()} for row in cache.index_rows),
        tuple(
            {key: str(value) for key, value in row.items()}
            for row in cache.compatibility_case_rows
        ),
    )
    assert csv_roundtrip.cache_hash == cache.cache_hash


def test_checkpoint_resume_accepts_valid_work_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _write_valid_checkpoint(tmp_path, monkeypatch)
    absent = {**task, "checkpoint_path": str(tmp_path / "absent.json")}

    completed, pending = source_cache._resume_source_tasks((task, absent))
    assert tuple(completed) == (("0", 17),)
    assert pending == [absent]

    path = Path(str(task["checkpoint_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["device"] = "cuda:9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="failed validation"):
        source_cache._resume_source_tasks((task,))


def test_source_cache_lock_detects_final_array_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _mini_cache(tmp_path, monkeypatch)
    source_cache._atomic_write_csv(
        tmp_path / source_cache.SOURCE_BLOCK_INDEX_MEMBER,
        cache.index_rows,
        columns=source_cache.SOURCE_BLOCK_INDEX_COLUMNS,
    )
    source_cache._atomic_write_csv(
        tmp_path / source_cache.COMPATIBILITY_CASE_MEMBER,
        cache.compatibility_case_rows,
        columns=source_cache.COMPATIBILITY_CASE_COLUMNS,
    )
    config = SimpleNamespace(contract_hash="config")
    generation = SimpleNamespace(bank_lock_hash="bank", generation_lock_hash="generation")
    frame = SimpleNamespace(cache_binding_hash="validation")
    partitions = SimpleNamespace(lock_hash="partitions")
    lock = source_cache.build_source_cache_lock(
        tmp_path,
        config=config,
        generation_lock=generation,
        frame=frame,
        partitions=partitions,
        source_cache=cache,
    )
    lock_path = tmp_path / source_cache.SOURCE_CACHE_LOCK_MEMBER
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    source_cache.validate_source_cache_lock(
        tmp_path,
        config=config,
        generation_lock=generation,
        frame=frame,
        partitions=partitions,
        source_cache=cache,
    )

    array = np.load(cache.array_path)
    array[0, 0, 0] += 1.0
    np.save(cache.array_path, array, allow_pickle=False)
    with pytest.raises(ProtocolError, match="not bound"):
        source_cache.validate_source_cache_lock(
            tmp_path,
            config=config,
            generation_lock=generation,
            frame=frame,
            partitions=partitions,
            source_cache=cache,
        )


def test_calibration_uses_only_exact_complete_candidate_replica_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _mini_cache(tmp_path, monkeypatch)
    baseline = cache.calibrated_energy_for("0", ("1",))

    changed_rows = []
    for row in cache.compatibility_case_rows:
        copied = dict(row)
        if copied["source_center"] == "2":
            copied["marginal_variational_energy"] = 1_000_000.0
        changed_rows.append(copied)
    changed = source_cache.SourceCache(
        cache.array_path,
        cache.index_rows,
        tuple(changed_rows),
    )
    observed = changed.calibrated_energy_for("0", ("1",))
    assert observed.mean_z_by_source == baseline.mean_z_by_source
    assert observed.candidate_sources == ("1",)

    with pytest.raises(ProtocolError, match="exact canonical candidate"):
        cache.calibrated_energy_for("0", ("2", "1"))
    with pytest.raises(ProtocolError, match="excludes.*query"):
        cache.calibrated_energy_for("0", ("0", "1"))

    incomplete = tuple(
        row
        for index, row in enumerate(cache.compatibility_case_rows)
        if index != 0
    )
    with pytest.raises(ProtocolError, match="compatibility (replica grid|support cases)"):
        source_cache.SourceCache(cache.array_path, cache.index_rows, incomplete)


def _generation_key(
    source: str, training_seed: int, generation_seed: int
) -> SourceGenerationKey:
    return SourceGenerationKey(
        source_center=source,
        training_seed=training_seed,
        generation_seed=generation_seed,
        expert_lock_hash=f"expert-{source}-{training_seed}",
        stream_id=f"stream-{source}-{training_seed}-{generation_seed}",
        class_seed_by_label={"0": generation_seed, "1": generation_seed + 1},
    )


def _patch_geometry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    centers: tuple[str, ...] = ("0", "1", "2"),
    generation_seeds: tuple[int, ...] = (17,),
    prefix: int = 2,
    feature_dim: int = 4,
) -> None:
    for module in (source_cache, source_cache_store, _source_worker):
        monkeypatch.setattr(module, "CENTERS", centers)
        monkeypatch.setattr(module, "TRAINING_SEEDS", (17, 42, 101))
        monkeypatch.setattr(module, "GENERATION_SEEDS", generation_seeds)
        monkeypatch.setattr(module, "MAX_SOURCE_PREFIX_PER_CLASS", prefix)
        monkeypatch.setattr(module, "COMMON_FEATURE_DIM", feature_dim)
    monkeypatch.setattr(source_cache, "EXPECTED_SOURCE_TASK_COUNT", len(centers) * 3)
    monkeypatch.setattr(
        source_cache,
        "EXPECTED_SOURCE_BLOCK_COUNT",
        len(centers) * 3 * len(generation_seeds),
    )


def _mini_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> source_cache.SourceCache:
    _patch_geometry(monkeypatch)
    keys = tuple(product(("0", "1", "2"), (17, 42, 101), (17,)))
    values = np.stack(
        [np.full((4, 4), ordinal, dtype=np.float32) for ordinal in range(len(keys))]
    )
    array_path = tmp_path / source_cache.SOURCE_BLOCK_ARRAY_MEMBER
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, values, allow_pickle=False)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    index_rows = tuple(
        {
            "schema_version": "midogpp_residual_topup_source_block_v1",
            "block_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "stream_id": f"stream-{source}-{training_seed}-{generation_seed}",
            "expert_lock_hash": f"expert-{source}-{training_seed}",
            "samples_per_class": 2,
            "row_count": 4,
            "feature_dim": 4,
            "output_sha256": _source_worker._array_bundle_sha256(
                values[ordinal], labels
            ),
        }
        for ordinal, (source, training_seed, generation_seed) in enumerate(keys)
    )
    case_rows = []
    for source, training_seed, query, case_index in product(
        ("0", "1", "2"), (17, 42, 101), ("0", "1", "2"), (0, 1)
    ):
        value = float(int(source) * 10 + training_seed / 100 + int(query) + case_index)
        case_rows.append(
            {
                "schema_version": "midogpp_residual_topup_compatibility_case_v1",
                "source_center": source,
                "training_seed": training_seed,
                "query_center": query,
                "case_id": f"{query}-case-{case_index}",
                "query_partition_role": "support",
                "row_count": 1,
                "marginal_variational_energy": value,
                "class_0_energy": value + 0.1,
                "class_1_energy": value + 0.2,
                "class_0_common_reconstruction_mse": value + 0.3,
                "class_1_common_reconstruction_mse": value + 0.4,
                "class_0_normalized_ps_kl": value + 0.5,
                "class_1_normalized_ps_kl": value + 0.6,
                "class_prior_json": "[0.5,0.5]",
                "labels_used": False,
                "exact_nelbo_claimed": False,
            }
        )
    return source_cache.SourceCache(array_path, index_rows, tuple(case_rows))


def _write_valid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    _patch_geometry(
        monkeypatch,
        centers=("0",),
        generation_seeds=(17,),
        prefix=2,
        feature_dim=4,
    )
    key = _generation_key("0", 17, 17)
    array = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    array_path = tmp_path / "task.npy"
    np.save(array_path, array, allow_pickle=False)
    support_index_path = tmp_path / "support.json"
    support_index_path.write_text(
        json.dumps(
            {
                "offsets": {
                    "0": {
                        "start": 0,
                        "stop": 2,
                        "case_ids": ["case-a", "case-b"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    task: dict[str, object] = {
        "task_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "generation_keys": (key,),
        "device": "cpu",
        "support_index_path": str(support_index_path),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "array_path": str(array_path),
        "config_contract_hash": "config",
        "generation_lock_hash": "generation",
        "support_scratch_hash": "support",
    }
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    compatibility = []
    for case_index, case_id in enumerate(("case-a", "case-b")):
        compatibility.append(
            {
                "source_center": "0",
                "training_seed": 17,
                "query_center": "0",
                "case_id": case_id,
                "row_count": 1,
                "marginal_variational_energy": 1.0 + case_index,
                "class_0_energy": 1.1 + case_index,
                "class_1_energy": 1.2 + case_index,
                "class_0_common_reconstruction_mse": 1.3 + case_index,
                "class_1_common_reconstruction_mse": 1.4 + case_index,
                "class_0_normalized_ps_kl": 1.5 + case_index,
                "class_1_normalized_ps_kl": 1.6 + case_index,
            }
        )
    unhashed = {
        "schema_version": "midogpp_residual_topup_source_checkpoint_v1",
        "status": "COMPLETE",
        "config_contract_hash": "config",
        "generation_lock_hash": "generation",
        "support_scratch_hash": "support",
        "task_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "device": "cpu",
        "array_path": str(array_path),
        "array_file_sha256": _source_worker._sha256_file(array_path),
        "blocks": [
            {
                "generation_seed": 17,
                "stream_id": key.stream_id,
                "output_sha256": _source_worker._array_bundle_sha256(
                    array[0], labels
                ),
            }
        ],
        "compatibility_case_records": compatibility,
        "target_labels_used": False,
        "support_labels_used": False,
        "evaluation_embeddings_used": False,
    }
    payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    Path(str(task["checkpoint_path"])).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return task
