from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
    source_block_sha256,
)
from midogpp_thesis.cvae.runtime.harp_v4_execution import classifier_worker_cache
from midogpp_thesis.cvae.runtime.harp_v4_execution import physical
from midogpp_thesis.cvae.runtime.harp_v4_execution.frame_binding import (
    persist_or_validate_frame_binding,
)
from midogpp_thesis.cvae.runtime.harp_probability_menu import (
    TARGET_SURFACE,
    HarpActionSpec,
)
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec


def _clear_worker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not change the test runner's real process environment while exercising
    # the spawn initializer directly.
    monkeypatch.setattr(classifier_worker_cache.os, "environ", {})
    classifier_worker_cache.initialize_classifier_worker(3)


def _bound_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, Path, Path, np.ndarray, np.ndarray]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(classifier_worker_cache, "EXPECTED_STREAM_COUNT", 1)
    monkeypatch.setattr(classifier_worker_cache, "CENTERS", ("1",))
    monkeypatch.setattr(
        classifier_worker_cache, "EXACT_NINE_SEED_PAIRS", ((17, 17),)
    )
    source_values = np.zeros(
        (
            1,
            2 * classifier_worker_cache.SOURCE_ROWS_PER_CLASS,
            classifier_worker_cache.COMMON_OUTPUT_DIM,
        ),
        dtype=np.float32,
    )
    source_values[0, 0, 0] = np.float32(1.25)
    frame_values = np.arange(
        2 * classifier_worker_cache.COMMON_OUTPUT_DIM, dtype=np.float32
    ).reshape(2, classifier_worker_cache.COMMON_OUTPUT_DIM)
    source_path = (tmp_path / "source.npy").resolve()
    frame_path = (tmp_path / "frame.npy").resolve()
    index_path = (tmp_path / "source-index.json").resolve()
    np.save(source_path, source_values, allow_pickle=False)
    np.save(frame_path, frame_values, allow_pickle=False)
    record = {
        "block_ordinal": 0,
        "source_center": "1",
        "training_seed": 17,
        "generation_seed": 17,
        "stream_id": "stream-1-17-17",
        "expert_lock_hash": stable_hash({"expert": "1-17"}),
        "rows_per_class": classifier_worker_cache.SOURCE_ROWS_PER_CLASS,
        "row_count": 2 * classifier_worker_cache.SOURCE_ROWS_PER_CLASS,
        "feature_dim": classifier_worker_cache.COMMON_OUTPUT_DIM,
        "output_sha256": source_block_sha256(source_values[0]),
    }
    source_index_unhashed = {
        "schema_version": "test_source_index_v1",
        "records": [record],
    }
    source_index_hash = stable_hash(source_index_unhashed)
    atomic_json(
        index_path,
        {
            **source_index_unhashed,
            "source_stream_index_hash": source_index_hash,
        },
    )
    frame_receipt = persist_or_validate_frame_binding(
        array_path=frame_path,
        receipt_path=(tmp_path / "frame-receipt.json").resolve(),
        shape=frame_values.shape,
    )
    source_lock_path = (tmp_path / "source-lock.json").resolve()
    source_lock_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_lock_v1",
        "source_array_sha256": sha256_file(source_path),
        "source_stream_index_sha256": sha256_file(index_path),
    }
    source_lock = {
        **source_lock_unhashed,
        "source_stream_lock_hash": stable_hash(source_lock_unhashed),
    }
    atomic_json(source_lock_path, source_lock)
    task: dict[str, object] = {
        "source_stream_lock_hash": source_lock["source_stream_lock_hash"],
        "source_stream_lock_sha256": sha256_file(source_lock_path),
        "source_array_path": str(source_path),
        "source_array_sha256": sha256_file(source_path),
        "source_index_path": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "source_stream_index_hash": source_index_hash,
        "frame_array_path": str(frame_path),
        "frame_array_sha256": sha256_file(frame_path),
        "frame_receipt_hash": frame_receipt.receipt_hash,
        "frame_receipt_sha256": frame_receipt.receipt_sha256,
        "training_seed": 17,
        "generation_seed": 17,
        "source_records": [record],
    }
    return task, source_path, index_path, frame_path, source_values, frame_values


def test_classifier_initializer_hides_cuda_binds_threads_and_clears_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, *_ = _bound_task(tmp_path, monkeypatch)
    _clear_worker_state(monkeypatch)
    classifier_worker_cache.load_worker_arrays(task)
    assert classifier_worker_cache._SOURCE_ARRAY_CACHE

    classifier_worker_cache.initialize_classifier_worker(3)
    assert classifier_worker_cache.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert {
        classifier_worker_cache.os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    } == {"3"}
    assert classifier_worker_cache.os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert classifier_worker_cache.os.environ["OMP_DYNAMIC"] == "FALSE"
    assert classifier_worker_cache.os.environ["MKL_DYNAMIC"] == "FALSE"
    assert not classifier_worker_cache._SOURCE_ARRAY_CACHE
    assert not classifier_worker_cache._FRAME_ARRAY_CACHE
    assert not classifier_worker_cache._SOURCE_BLOCK_HASH_CACHE

    with pytest.raises(ProtocolError, match="initializer drifted"):
        classifier_worker_cache.initialize_classifier_worker(1)


def test_worker_verifies_files_once_and_cached_values_equal_reference_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, source_path, index_path, frame_path, source_reference, frame_reference = (
        _bound_task(tmp_path, monkeypatch)
    )
    _clear_worker_state(monkeypatch)
    real_load = np.load
    real_sha256_file = sha256_file
    real_block_hash = source_block_sha256
    opened: list[str] = []
    digested: list[str] = []
    block_hash_calls = 0

    def tracked_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    def tracked_file_hash(path: Path) -> str:
        digested.append(str(path))
        return real_sha256_file(path)

    def tracked_block_hash(values: np.ndarray) -> str:
        nonlocal block_hash_calls
        block_hash_calls += 1
        return real_block_hash(values)

    monkeypatch.setattr(classifier_worker_cache.np, "load", tracked_load)
    monkeypatch.setattr(classifier_worker_cache, "sha256_file", tracked_file_hash)
    monkeypatch.setattr(
        classifier_worker_cache, "source_block_sha256", tracked_block_hash
    )

    first_source, first_frame, first_key = classifier_worker_cache.load_worker_arrays(
        task
    )
    first_blocks = classifier_worker_cache.load_source_blocks(
        (SimpleNamespace(source_order=("1",)),),
        task,
        source_values=first_source,
        source_key=first_key,
    )
    second_source, second_frame, second_key = classifier_worker_cache.load_worker_arrays(
        task
    )
    second_blocks = classifier_worker_cache.load_source_blocks(
        (SimpleNamespace(source_order=("1",)),),
        task,
        source_values=second_source,
        source_key=second_key,
    )

    assert first_source is second_source
    assert first_frame is second_frame
    assert first_key == second_key
    assert first_source.tobytes(order="C") == source_reference.tobytes(order="C")
    assert first_frame.tobytes(order="C") == frame_reference.tobytes(order="C")
    assert (
        first_blocks["1"]["embeddings"].tobytes(order="C")
        == source_reference[0].tobytes(order="C")
        == second_blocks["1"]["embeddings"].tobytes(order="C")
    )
    assert opened == [str(source_path), str(frame_path)]
    assert sorted(digested) == sorted(
        (str(source_path), str(index_path), str(frame_path))
    )
    assert block_hash_calls == 1

    drifted = deepcopy(task)
    assert isinstance(drifted["source_records"], list)
    drifted["source_records"][0]["output_sha256"] = "d" * 64
    with pytest.raises(ProtocolError, match="records drifted"):
        classifier_worker_cache.load_source_blocks(
            (SimpleNamespace(source_order=("1",)),),
            drifted,
            source_values=second_source,
            source_key=second_key,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_stream_lock_hash", "a" * 64, "source-stream lock hash"),
        ("source_stream_lock_sha256", "a" * 16, "source-stream lock SHA-256"),
        ("source_stream_index_hash", "a" * 64, "source-stream index hash"),
        ("source_array_sha256", "a" * 16, "source-array hash"),
        ("source_index_sha256", "a" * 16, "source-index hash"),
        ("frame_array_sha256", "a" * 16, "frame-array hash"),
        ("frame_receipt_hash", "a" * 64, "frame-receipt hash"),
        ("frame_receipt_sha256", "a" * 16, "frame-receipt SHA-256"),
    ),
)
def test_worker_rejects_malformed_semantic_and_file_hash_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    task, *_ = _bound_task(tmp_path, monkeypatch)
    _clear_worker_state(monkeypatch)
    task[field] = value

    with pytest.raises(ProtocolError, match=message):
        classifier_worker_cache.load_worker_arrays(task)


def test_worker_requires_real_width_expert_semantic_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, *_ = _bound_task(tmp_path, monkeypatch)
    _clear_worker_state(monkeypatch)
    source, _, source_key = classifier_worker_cache.load_worker_arrays(task)
    assert isinstance(task["source_records"], list)
    task["source_records"][0]["expert_lock_hash"] = "b" * 64
    source_index_unhashed = {"records": task["source_records"]}
    source_index_hash = stable_hash(source_index_unhashed)
    atomic_json(
        Path(str(task["source_index_path"])),
        {
            **source_index_unhashed,
            "source_stream_index_hash": source_index_hash,
        },
    )
    classifier_worker_cache.reset_worker_state()
    source, _, source_key = classifier_worker_cache.load_worker_arrays(
        {
            **task,
            "source_index_sha256": sha256_file(Path(str(task["source_index_path"]))),
            "source_stream_index_hash": source_index_hash,
        }
    )
    drifted = {
        **task,
        "source_index_sha256": sha256_file(Path(str(task["source_index_path"]))),
        "source_stream_index_hash": source_index_hash,
    }

    with pytest.raises(ProtocolError, match="expert-lock hash"):
        classifier_worker_cache.load_source_blocks(
            (SimpleNamespace(source_order=("1",)),),
            drifted,
            source_values=source,
            source_key=source_key,
        )


def test_worker_rejects_same_shape_source_mutation_after_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, source_path, *_ = _bound_task(tmp_path, monkeypatch)
    _clear_worker_state(monkeypatch)
    classifier_worker_cache.load_worker_arrays(task)
    prior = source_path.stat()
    changed = np.load(source_path, mmap_mode="r+", allow_pickle=False)
    changed[0, 0, 0] = np.float32(changed[0, 0, 0] + 1.0)
    changed.flush()
    os.utime(
        source_path,
        ns=(prior.st_atime_ns, prior.st_mtime_ns + 1_000_000),
    )

    with pytest.raises(ProtocolError, match="source-array file identity drifted"):
        classifier_worker_cache.load_worker_arrays(task)


def test_worker_rejects_same_shape_frame_mutation_before_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, _, _, frame_path, *_ = _bound_task(tmp_path, monkeypatch)
    changed = np.load(frame_path, mmap_mode="r+", allow_pickle=False)
    changed[0, 0] = np.float32(changed[0, 0] + 1.0)
    changed.flush()
    _clear_worker_state(monkeypatch)

    with pytest.raises(ProtocolError, match="frame-array bytes failed"):
        classifier_worker_cache.load_worker_arrays(task)


def test_worker_rejects_source_index_tamper_and_frame_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, _, index_path, _, *_ = _bound_task(tmp_path, monkeypatch)
    _clear_worker_state(monkeypatch)
    classifier_worker_cache.load_worker_arrays(task)

    index_bytes = index_path.read_bytes()
    assert b"stream-1-17-17" in index_bytes
    index_path.write_bytes(index_bytes.replace(b"stream-1-17-17", b"stream-2-17-17"))
    with pytest.raises(ProtocolError, match="source-index file identity drifted"):
        classifier_worker_cache.load_worker_arrays(task)

    task, _, _, _, *_ = _bound_task(tmp_path / "fresh", monkeypatch)
    _clear_worker_state(monkeypatch)
    classifier_worker_cache.load_worker_arrays(task)
    second_frame = (tmp_path / "fresh" / "frame-2.npy").resolve()
    np.save(
        second_frame,
        np.zeros((2, classifier_worker_cache.COMMON_OUTPUT_DIM), dtype=np.float32),
        allow_pickle=False,
    )
    drifted = dict(task)
    drifted["frame_array_path"] = str(second_frame)
    drifted["frame_array_sha256"] = sha256_file(second_frame)
    with pytest.raises(ProtocolError, match="frame-array path/binding drifted"):
        classifier_worker_cache.load_worker_arrays(drifted)


def test_task_identity_binds_frame_source_array_and_source_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source-cache"
    source_root.mkdir()
    source_array = (source_root / "source.npy").resolve()
    source_index = (source_root / SOURCE_INDEX_MEMBER).resolve()
    source_index.parent.mkdir(parents=True)
    source_array.write_bytes(b"source-array")
    source_index_unhashed = {
        "schema_version": "test_source_index_v1",
        "records": [],
    }
    source_index_payload = {
        **source_index_unhashed,
        "source_stream_index_hash": stable_hash(source_index_unhashed),
    }
    atomic_json(source_index, source_index_payload)
    frame_path = (tmp_path / "frames.npy").resolve()
    frame_path.write_bytes(b"frames")
    source_array_sha = sha256_file(source_array)
    source_index_sha = sha256_file(source_index)
    frame_sha = sha256_file(frame_path)
    frame_receipt_path = (tmp_path / "frame-receipt.json").resolve()
    frame_binding = persist_or_validate_frame_binding(
        array_path=frame_path,
        receipt_path=frame_receipt_path,
        shape=(1,),
    )
    source_lock_unhashed = {
        "schema_version": "midogpp_frozen_source_stream_lock_v1",
        "source_array_sha256": source_array_sha,
        "source_stream_index_sha256": source_index_sha,
        "source_stream_index_hash": source_index_payload[
            "source_stream_index_hash"
        ],
    }
    source_lock_payload = {
        **source_lock_unhashed,
        "source_stream_lock_hash": stable_hash(source_lock_unhashed),
    }
    source_lock = (source_root / SOURCE_LOCK_MEMBER).resolve()
    atomic_json(source_lock, source_lock_payload)
    source_lock_sha = sha256_file(source_lock)

    actions = tuple(
        SimpleNamespace(
            surface_kind="development",
            outer_target_id=outer,
            query_center_id=query,
            to_payload=lambda outer=outer, query=query: {
                "outer": outer,
                "query": query,
            },
        )
        for outer in tuple(str(value) for value in range(9))
        for query in tuple(str(value) for value in range(9))
    )
    monkeypatch.setattr(physical, "_all_actions", lambda: actions)
    contexts = {( "development", str(query)): (0, 1) for query in range(9)}
    samples = {
        ("development", str(query)): (f"sample-{query}",) for query in range(9)
    }
    cases = {
        ("development", str(query)): (f"case-{query}",) for query in range(9)
    }
    frames = physical._Frames(
        path=frame_path,
        receipt_path=frame_receipt_path,
        contexts=contexts,
        sample_ids=samples,
        case_ids=cases,
        sha256=frame_sha,
        receipt_hash=frame_binding.receipt_hash,
        receipt_sha256=frame_binding.receipt_sha256,
    )
    source_cache = SimpleNamespace(
        root=source_root,
        source_array_path=source_array,
        records=(),
        lock_hash=source_lock_payload["source_stream_lock_hash"],
        lock_payload=source_lock_payload,
    )
    inputs = SimpleNamespace(
        generation_hash="b" * 64,
        bank_hash="c" * 64,
        classifier=SimpleNamespace(to_payload=lambda: {"family": "test"}),
    )
    tasks = physical._build_tasks(
        scratch_root=tmp_path,
        frames=frames,
        source_cache=source_cache,
        inputs=inputs,
        workstation=physical._DEFAULT_WORKSTATION_PROFILE,
        development_role="development",
        evaluation_role="evaluation",
    )

    assert len(tasks) == 729
    for task in tasks:
        assert task["source_array_path"] == str(source_array)
        assert task["source_array_sha256"] == source_array_sha
        assert task["source_index_path"] == str(source_index)
        assert task["source_index_sha256"] == source_index_sha
        assert task["source_stream_index_hash"] == source_index_payload[
            "source_stream_index_hash"
        ]
        assert len(str(task["source_stream_index_hash"])) == 16
        assert task["source_stream_lock_hash"] == source_lock_payload[
            "source_stream_lock_hash"
        ]
        assert len(str(task["source_stream_lock_hash"])) == 16
        assert task["source_stream_lock_sha256"] == source_lock_sha
        assert len(str(task["source_stream_lock_sha256"])) == 64
        assert "source_cache_hash" not in task
        assert task["frame_array_path"] == str(frame_path)
        assert task["frame_array_sha256"] == frame_sha
        assert task["frame_receipt_hash"] == frame_binding.receipt_hash
        assert len(str(task["frame_receipt_hash"])) == 16
        assert task["frame_receipt_sha256"] == frame_binding.receipt_sha256
        assert len(str(task["frame_receipt_sha256"])) == 64
        body = {
            key: value
            for key, value in task.items()
            if key not in {"task_hash", "npz_path", "receipt_path"}
        }
        assert task["task_hash"] == canonical_hash(body)


def test_classifier_checkpoint_roundtrip_uses_real_semantic_scaler_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = HarpActionSpec(
        surface_kind=TARGET_SURFACE,
        outer_target_id="1",
        query_center_id="1",
        selected_source_id="2",
    )
    probabilities = np.asarray(
        ((0.75, 0.25), (0.20, 0.80)), dtype=np.float64
    )
    scaler_state_hash = stable_hash(
        {"mean_": [0.0], "var_": [1.0], "scale_": [1.0]}
    )
    composition_hash = canonical_hash({"composition": "real-width"})
    npz_path = (tmp_path / "checkpoint.npz").resolve()
    receipt_path = (tmp_path / "checkpoint.json").resolve()
    task_body: dict[str, object] = {
        "schema_version": "midogpp_harp_v4_label_free_classifier_task_v1",
        "ordinal": 0,
        "surface_kind": TARGET_SURFACE,
        "outer_target_id": "1",
        "query_center_id": "1",
        "training_seed": 17,
        "generation_seed": 17,
        "actions": [action.to_payload()],
        "frame_start": 0,
        "frame_stop": 2,
        "sample_ids": ["sample-0", "sample-1"],
        "case_ids": ["case-0", "case-1"],
        "generation_lock_hash": stable_hash({"generation": "lock"}),
        "classifier": ClassifierSpec().to_payload(),
        "threads_per_worker": 3,
        "workstation_profile_hash": physical._DEFAULT_WORKSTATION_PROFILE.profile_hash,
        "labels_available": False,
    }
    task = {
        **task_body,
        "task_hash": canonical_hash(task_body),
        "npz_path": str(npz_path),
        "receipt_path": str(receipt_path),
    }
    monkeypatch.setattr(
        physical,
        "_load_worker_arrays",
        lambda _task: (
            np.empty((0,), dtype=np.float32),
            np.zeros((2, physical.COMMON_OUTPUT_DIM), dtype=np.float32),
            object(),
        ),
    )
    monkeypatch.setattr(
        physical,
        "_load_source_blocks",
        lambda actions, *args, **kwargs: {
            source: {}
            for candidate in actions
            for source in candidate.source_order
        },
    )
    monkeypatch.setattr(
        physical,
        "compose_harp_action",
        lambda *args, **kwargs: SimpleNamespace(
            embeddings=np.zeros((2, physical.COMMON_OUTPUT_DIM), dtype=np.float32),
            labels=np.asarray((0, 1), dtype=np.int64),
            composition_hash=composition_hash,
        ),
    )
    monkeypatch.setattr(
        physical,
        "fit_logistic_classifier",
        lambda *args, **kwargs: SimpleNamespace(
            probabilities=probabilities,
            classes=(0, 1),
            converged=True,
            scaler_state_hash=scaler_state_hash,
        ),
    )

    physical._classifier_task(task)
    loaded = physical._load_task_checkpoint(task)

    assert loaded is not None
    assert loaded["actions"][0]["action_hash"] == action.action_hash
    assert len(loaded["actions"][0]["action_hash"]) == 64
    assert loaded["actions"][0]["composition_hash"] == composition_hash
    assert len(loaded["actions"][0]["composition_hash"]) == 64
    assert loaded["actions"][0]["scaler_state_hash"] == scaler_state_hash
    assert len(loaded["actions"][0]["scaler_state_hash"]) == 16
    assert loaded["actions"][0]["probability_sha256"] == hashlib.sha256(
        np.ascontiguousarray(probabilities[:, 1], dtype=np.float32).tobytes(order="C")
    ).hexdigest()

    malformed = read_json(receipt_path)
    malformed["actions"][0]["scaler_state_hash"] = "f" * 64
    body = {key: value for key, value in malformed.items() if key != "checkpoint_hash"}
    malformed["checkpoint_hash"] = canonical_hash(body)
    atomic_json(receipt_path, malformed)
    with pytest.raises(ProtocolError, match="action hash is malformed"):
        physical._load_task_checkpoint(task)


def test_frame_staging_requires_and_uses_grouped_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(physical, "CENTERS", ("1",))

    class Cache:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def rows_for(self, center: str, role: str) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    split_row_index=0,
                    sample_id=f"{role}-{center}-0",
                    case_id=f"case-{role}-{center}-0",
                ),
                SimpleNamespace(
                    split_row_index=1,
                    sample_id=f"{role}-{center}-1",
                    case_id=f"case-{role}-{center}-1",
                ),
            )

        def load_embeddings(self, rows: tuple[SimpleNamespace, ...]) -> np.ndarray:
            self.calls.append(tuple(row.sample_id for row in rows))
            value = float(len(self.calls))
            return np.full(
                (len(rows), physical.COMMON_OUTPUT_DIM), value, dtype=np.float32
            )

        def load_embedding(self, _row: object) -> np.ndarray:
            raise AssertionError("row-wise loading must not be used")

    cache = Cache()
    frames = physical._stage_frames(
        cache,
        scratch_root=tmp_path,
        roles=("development", "evaluation"),
    )
    values = np.load(frames.path, allow_pickle=False)

    assert len(cache.calls) == 2
    assert values.shape == (4, physical.COMMON_OUTPUT_DIM)
    assert np.array_equal(values[:2], np.full_like(values[:2], 1.0))
    assert np.array_equal(values[2:], np.full_like(values[2:], 2.0))

    with pytest.raises(ProtocolError, match="grouped-shard reader"):
        physical._stage_frames(
            object(),
            scratch_root=tmp_path / "missing-reader",
            roles=("development", "evaluation"),
        )
