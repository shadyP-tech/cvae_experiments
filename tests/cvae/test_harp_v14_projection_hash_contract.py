from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v14_execution import classifier_worker_cache
from midogpp_thesis.cvae.runtime.harp_v14_execution import crossfit_surface
from midogpp_thesis.cvae.runtime.harp_v14_execution import resident_stream_contracts
from midogpp_thesis.cvae.runtime.harp_v14_execution.resident_stream_contracts import (
    ResidentExpertStreamCache,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.classifier_tasks import (
    _validate_task_identity,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.compatibility_contracts import (
    ReplicaEnergyInput,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.execution_profile import (
    DEFAULT_WORKSTATION_PROFILE,
)


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in (17, 42, 101)
    for generation_seed in (17, 42, 101)
)
ROLE = "harp_source_train_development"


def _task_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], np.ndarray]:
    # Keep the producer/worker boundary real while shrinking only array geometry.
    monkeypatch.setattr(classifier_worker_cache, "SOURCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(classifier_worker_cache, "COMMON_OUTPUT_DIM", 2)
    monkeypatch.setattr(resident_stream_contracts, "SOURCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(resident_stream_contracts, "COMMON_OUTPUT_DIM", 2)

    source_values = np.arange(
        len(CENTERS) * len(SEED_PAIRS) * 4, dtype=np.float32
    ).reshape(len(CENTERS) * len(SEED_PAIRS), 2, 2)
    source_path = (tmp_path / "resident-sources.npy").resolve()
    np.save(source_path, source_values, allow_pickle=False)
    records = []
    for ordinal, (source, (training_seed, generation_seed)) in enumerate(
        (source, pair) for source in CENTERS for pair in SEED_PAIRS
    ):
        records.append(
            {
                "block_ordinal": ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "stream_id": f"source-{source}-{training_seed}-{generation_seed}",
                "expert_lock_hash": stable_hash(
                    {"source": source, "training_seed": training_seed}
                ),
                "rows_per_class": 1,
                "row_count": 2,
                "feature_dim": 2,
                "output_sha256": classifier_worker_cache.source_block_sha256(
                    source_values[ordinal]
                ),
            }
        )

    source_binding = SimpleNamespace(
        array_path=source_path,
        array_sha256=sha256_file(source_path),
        index_path=(tmp_path / "unused-full-index.json").resolve(),
        index_sha256="a" * 64,
        index_hash=stable_hash({"full_source_index": True}),
        records=tuple(records),
        lock_hash=stable_hash({"source_stream_lock": True}),
        lock_sha256="b" * 64,
    )
    monkeypatch.setattr(
        crossfit_surface,
        "validate_source_task_binding",
        lambda _cache: source_binding,
    )

    parent_frames = np.arange(len(CENTERS) * 4, dtype=np.float32).reshape(-1, 2)
    frame_path = (tmp_path / "source-frames.npy").resolve()
    np.save(frame_path, parent_frames, allow_pickle=False)
    frames = SimpleNamespace(
        path=frame_path,
        sha256=sha256_file(frame_path),
        receipt_hash=stable_hash({"parent_frame_receipt": True}),
        contexts=MappingProxyType(
            {
                (ROLE, center): (2 * index, 2 * index + 2)
                for index, center in enumerate(CENTERS)
            }
        ),
        sample_ids=MappingProxyType(
            {
                (ROLE, center): (f"{center}-sample-0", f"{center}-sample-1")
                for center in CENTERS
            }
        ),
        case_ids=MappingProxyType(
            {
                (ROLE, center): (f"{center}-case-0", f"{center}-case-1")
                for center in CENTERS
            }
        ),
    )
    inputs = SimpleNamespace(
        generation_hash=stable_hash({"generation": True}),
        bank_hash=stable_hash({"bank": True}),
        classifier=SimpleNamespace(to_payload=lambda: {"family": "fixture"}),
    )
    tasks = crossfit_surface.build_fold_conditioned_classifier_tasks(
        scratch_root=tmp_path,
        frames=frames,
        source_cache=object(),
        inputs=inputs,
        workstation=DEFAULT_WORKSTATION_PROFILE,
        source_role=ROLE,
        outer_targets=("0",),
    )
    task = next(
        dict(value)
        for value in tasks
        if value["heldout_center_id"] == "1"
        and value["current_query_center_id"] == "3"
        and value["training_seed"] == 17
        and value["generation_seed"] == 17
    )
    return task, source_values


def _rehash_task(task: dict[str, object]) -> None:
    body = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "npz_path", "receipt_path"}
    }
    task["task_hash"] = canonical_hash(body)


def _resident_cache_shell(lock_payload: object) -> ResidentExpertStreamCache:
    cache = object.__new__(ResidentExpertStreamCache)
    object.__setattr__(cache, "lock_payload", lock_payload)
    return cache


def test_projection_producer_and_worker_share_typed_digest_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, source_reference = _task_fixture(tmp_path, monkeypatch)

    for field in (
        "source_stream_lock_hash",
        "source_stream_index_hash",
        "source_record_projection_hash",
        "full_source_stream_index_hash",
        "frame_receipt_hash",
        "frame_projection_hash",
    ):
        assert len(str(task[field])) == 16
    for field in (
        "source_stream_lock_sha256",
        "source_array_sha256",
        "source_index_sha256",
        "frame_array_sha256",
        "frame_receipt_sha256",
    ):
        assert len(str(task[field])) == 64

    projection = read_json(Path(str(task["source_index_path"])))
    projection_body = {
        key: value
        for key, value in projection.items()
        if key != "source_record_projection_hash"
    }
    assert projection["source_record_projection_hash"] == stable_hash(projection_body)
    frame_receipt = read_json(Path(str(task["frame_receipt_path"])))
    frame_body = {
        key: value
        for key, value in frame_receipt.items()
        if key != "frame_projection_hash"
    }
    assert frame_receipt["frame_projection_hash"] == stable_hash(frame_body)

    _validate_task_identity(task, error_context="projection-contract test")
    classifier_worker_cache.reset_worker_state()
    source, frame, source_key = classifier_worker_cache.load_worker_arrays(task)
    selected_source = str(task["allowed_source_ids"][0])
    blocks = classifier_worker_cache.load_source_blocks(
        (SimpleNamespace(source_order=(selected_source,)),),
        task,
        source_values=source,
        source_key=source_key,
    )
    ordinal = next(
        int(record["block_ordinal"])
        for record in task["source_records"]
        if record["source_center"] == selected_source
        and record["training_seed"] == 17
        and record["generation_seed"] == 17
    )
    assert np.array_equal(blocks[selected_source]["embeddings"], source_reference[ordinal])
    assert frame.shape == (2, 2)


def test_resident_cache_exposes_typed_source_index_identity() -> None:
    cache = _resident_cache_shell(
        MappingProxyType(
            {
                "source_stream_lock_hash": "a" * 16,
                "source_stream_index_hash": "b" * 16,
            }
        )
    )

    assert cache.lock_hash == "a" * 16
    assert cache.index_hash == "b" * 16

    malformed = _resident_cache_shell(
        MappingProxyType(
            {
                "source_stream_lock_hash": "a" * 16,
                "source_stream_index_hash": "b" * 64,
            }
        )
    )
    with pytest.raises(ProtocolError, match="source-stream index hash is malformed"):
        _ = malformed.index_hash


@pytest.mark.parametrize("field", ("source_stream_index_hash", "frame_receipt_hash"))
def test_parent_rejects_canonical_sha_in_semantic_task_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    task, _ = _task_fixture(tmp_path, monkeypatch)
    task[field] = canonical_hash({"wrong_digest_role": field})
    _rehash_task(task)

    with pytest.raises(ProtocolError, match="digest role drifted"):
        _validate_task_identity(task, error_context="projection-contract test")


def test_worker_rejects_rehashed_projection_semantic_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, _ = _task_fixture(tmp_path, monkeypatch)
    projection_path = Path(str(task["source_index_path"]))
    projection = read_json(projection_path)
    projection["excluded_record_offsets_visible"] = True
    body = {
        key: value
        for key, value in projection.items()
        if key != "source_record_projection_hash"
    }
    projection["source_record_projection_hash"] = stable_hash(body)
    atomic_json(projection_path, projection)
    task["source_index_sha256"] = sha256_file(projection_path)
    task["source_stream_index_hash"] = projection[
        "source_record_projection_hash"
    ]
    task["source_record_projection_hash"] = projection[
        "source_record_projection_hash"
    ]
    _rehash_task(task)

    _validate_task_identity(task, error_context="projection-contract test")
    classifier_worker_cache.reset_worker_state()
    source, _, source_key = classifier_worker_cache.load_worker_arrays(task)
    with pytest.raises(ProtocolError, match="task records differ from the fold projection"):
        classifier_worker_cache.load_source_blocks(
            (SimpleNamespace(source_order=(str(task["allowed_source_ids"][0]),)),),
            task,
            source_values=source,
            source_key=source_key,
        )


def test_compatibility_replica_preserves_typed_upstream_identities() -> None:
    replica = ReplicaEnergyInput(
        candidate_source_id="1",
        training_seed=17,
        query_case_equal_energy=1.0,
        own_source_location=0.0,
        own_source_scale=1.0,
        checkpoint_hash="a" * 64,
        source_frame_hash="b" * 16,
        sampler_hash="c" * 16,
    )

    assert replica.checkpoint_hash == "a" * 64
    assert replica.source_frame_hash == "b" * 16
    assert replica.sampler_hash == "c" * 16


@pytest.mark.parametrize("field", ("source_frame_hash", "sampler_hash"))
def test_compatibility_replica_rejects_sha256_in_semantic_field(field: str) -> None:
    values = {
        "candidate_source_id": "1",
        "training_seed": 17,
        "query_case_equal_energy": 1.0,
        "own_source_location": 0.0,
        "own_source_scale": 1.0,
        "checkpoint_hash": "a" * 64,
        "source_frame_hash": "b" * 16,
        "sampler_hash": "c" * 16,
    }
    values[field] = "d" * 64

    with pytest.raises(ProtocolError, match="malformed"):
        ReplicaEnergyInput(**values)
