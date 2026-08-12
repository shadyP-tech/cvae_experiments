from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.runtime.frozen_source_streams import source_block_sha256
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import build_case_bootstrap_plan
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    SEED_PAIRS,
    TRAINING_SEEDS,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.feature_checkpoint_store import (
    load_feature_checkpoint,
    publish_feature_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.feature_execution import (
    assemble_seed_feature_production,
    combine_feature_runtime,
    materialize_label_free_seed_features,
    materialize_label_free_support_shifts,
    validate_feature_worker_topology,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.feature_runtime_contracts import (
    FeatureComponentRecord,
    build_feature_task,
    build_support_slice,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.feature_worker import (
    load_verified_source_means,
)


SHA = "a" * 64
SHORT_HASH = "a" * 16


def _feature_task_kwargs(tmp_path: Path) -> dict[str, object]:
    support_slices = tuple(
        build_support_slice(
            query_center=query,
            relative_array_path=f"support_q{query}.npy",
            array_sha256=SHA,
            case_ids=tuple(f"q{query}_case_{index}" for index in range(8)),
            row_identity_hash=SHA,
            center_partition_hash=SHA,
            feature_support_partition_hash=SHA,
        )
        for query in candidate_sources("0")
    )
    return {
        "source_center": "0",
        "training_seed": 17,
        "device": "cuda:0",
        "expert_bank_root": str(tmp_path / "bank"),
        "source_array_path": str(tmp_path / "source.npy"),
        "source_block_ordinal_by_generation_seed": {17: 0, 42: 1, 101: 2},
        "source_block_output_sha256_by_generation_seed": {
            seed: SHA for seed in GENERATION_SEEDS
        },
        "support_root": str(tmp_path),
        "support_slices": support_slices,
        "checkpoint_npz_path": str(tmp_path / "feature_e0_train17.npz"),
        "checkpoint_json_path": str(tmp_path / "feature_e0_train17.json"),
        "config_contract_hash": SHORT_HASH,
        "bank_lock_hash": SHORT_HASH,
        "source_stream_lock_hash": SHORT_HASH,
        "cache_binding_hash": SHA,
        "partition_lock_hash": SHA,
        "metadata_grid_hash": SHA,
    }


def test_default_feature_runtime_api_is_label_free_and_importable() -> None:
    for function in (
        materialize_label_free_seed_features,
        materialize_label_free_support_shifts,
        combine_feature_runtime,
    ):
        forbidden = {"labels", "outcomes", "responses", "evaluation_embeddings"}
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    source = inspect.getsource(
        __import__(
            "midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.feature_energy_adapter",
            fromlist=["*"],
        )
    )
    assert "utility_aligned_exact_tail_router" not in source
    assert "utility_aligned_ensemble_endpoint_router" not in source
    assert "score_variational_compatibility" in source


def test_synthetic_component_assembly_has_exact_geometry_and_strict_exclusion() -> None:
    cases = {
        query: tuple(f"case_{query}_{index}" for index in range(8))
        for query in CENTERS
    }
    metadata = {
        query: {
            source: (index + 1.0) / 9.0
            for index, source in enumerate(candidate_sources(query))
        }
        for query in CENTERS
    }
    records = tuple(
        _component_record(query, source, training_seed, cases[query])
        for query in CENTERS
        for source in candidate_sources(query)
        for training_seed in TRAINING_SEEDS
    )

    def load_arrays(record: FeatureComponentRecord):
        offset = float(record.training_seed) / 1_000.0
        base = np.arange(record.support_row_count, dtype=np.float64) / 100.0 + offset
        return ({0: base, 1: base + 0.01}, {0: base + 0.02, 1: base + 0.03})

    product = assemble_seed_feature_production(
        records,
        load_arrays,
        support_case_ids_by_query=cases,
        metadata_by_query=metadata,
        feature_input_seal_hash=SHA,
    )
    assert len(product.inner_rows) == 4_536
    assert len(product.target_rows) == 648
    assert len(product.component_records) == 216
    assert all(
        row.outer_target_id != row.query_id
        and row.candidate_source not in {row.outer_target_id, row.query_id}
        for row in product.inner_rows
    )
    assert all(
        row.outer_target_id == row.query_id
        and row.candidate_source != row.outer_target_id
        for row in product.target_rows
    )
    assert len(product.inner_table_rows()) == 4_536
    assert len(product.target_table_rows()) == 648
    with pytest.raises(ProtocolError):
        replace(product, production_hash="b" * 64)


def test_feature_checkpoint_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    task = build_feature_task(**_feature_task_kwargs(tmp_path))
    support_slices = task.support_slices
    arrays = {}
    for support in support_slices:
        for suffix in ("reconstruction_0", "reconstruction_1", "kl_0", "kl_1"):
            arrays[f"q{support.query_center}_{suffix}"] = np.arange(8, dtype=np.float64)
    components = tuple(
        {
            "case_equal_energy": float(index + 1),
            "linear_kernel_mmd2_by_generation_seed": {
                seed: float(index + seed) for seed in GENERATION_SEEDS
            },
        }
        for index, _ in enumerate(support_slices)
    )
    observed = publish_feature_checkpoint(task, arrays=arrays, component_payloads=components)
    assert len(observed) == 8
    assert load_feature_checkpoint(task) == observed
    path = Path(task.checkpoint_json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["task_hash"] = "b" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_feature_checkpoint(task)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("config_contract_hash", SHA),
        ("bank_lock_hash", SHA),
        ("source_stream_lock_hash", SHA),
        ("config_contract_hash", "A" * 16),
        ("cache_binding_hash", SHORT_HASH),
        ("partition_lock_hash", SHORT_HASH),
        ("metadata_grid_hash", SHORT_HASH),
    ),
)
def test_feature_task_enforces_exact_hash_families(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    kwargs = _feature_task_kwargs(tmp_path)
    kwargs[field] = wrong_value
    with pytest.raises(ProtocolError, match="feature task drifted"):
        build_feature_task(**kwargs)


def test_feature_task_rejects_duplicate_source_block_ordinals(tmp_path: Path) -> None:
    kwargs = _feature_task_kwargs(tmp_path)
    kwargs["source_block_ordinal_by_generation_seed"] = {17: 0, 42: 0, 101: 2}
    with pytest.raises(ProtocolError, match="feature task drifted"):
        build_feature_task(**kwargs)


def test_feature_task_hash_binds_source_block_output_hashes(tmp_path: Path) -> None:
    kwargs = _feature_task_kwargs(tmp_path)
    original = build_feature_task(**kwargs)
    restored = pickle.loads(pickle.dumps(original))
    assert restored.task_hash == original.task_hash
    assert (
        dict(restored.source_block_output_sha256_by_generation_seed)
        == kwargs["source_block_output_sha256_by_generation_seed"]
    )
    changed_hashes = dict(kwargs["source_block_output_sha256_by_generation_seed"])
    changed_hashes[GENERATION_SEEDS[0]] = "b" * 64
    kwargs["source_block_output_sha256_by_generation_seed"] = changed_hashes
    changed = build_feature_task(**kwargs)
    assert changed.task_hash != original.task_hash


def test_feature_worker_rejects_source_block_hash_tamper(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npy"
    blocks = np.zeros((3, 540, COMMON_OUTPUT_DIM), dtype=np.float32)
    for ordinal in range(3):
        blocks[ordinal, 0, 0] = float(ordinal)
    np.save(source_path, blocks, allow_pickle=False)

    kwargs = _feature_task_kwargs(tmp_path)
    kwargs["source_block_output_sha256_by_generation_seed"] = {
        seed: source_block_sha256(blocks[ordinal])
        for ordinal, seed in enumerate(GENERATION_SEEDS)
    }
    task = build_feature_task(**kwargs)
    means = load_verified_source_means(task)
    assert tuple(means) == GENERATION_SEEDS
    assert all(value.shape == (COMMON_OUTPUT_DIM,) for value in means.values())

    blocks[1, 0, 1] = 1.0
    np.save(source_path, blocks, allow_pickle=False)
    with pytest.raises(ProtocolError, match="semantic hash drifted"):
        load_verified_source_means(task)


def test_feature_checkpoint_single_atomic_orphan_is_recomputed(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "checkpoints/feature_runtime").resolve()
    checkpoint.mkdir(parents=True)
    task = type("Task", (), {
        "source_center": "0", "training_seed": 17,
        "checkpoint_npz_path": str(checkpoint / "feature_e0_train17.npz"),
        "checkpoint_json_path": str(checkpoint / "feature_e0_train17.json"),
    })()
    orphan = Path(task.checkpoint_npz_path)
    orphan.write_bytes(b"atomic-orphan")
    assert load_feature_checkpoint(task) is None
    assert not orphan.exists()


def test_workstation_feature_topology_is_two_persistent_spawned_workers() -> None:
    runtime = {
        "generation_devices": ["cuda:0", "cuda:1"],
        "generation_workers_per_device": 1,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "array_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
    }
    assert validate_feature_worker_topology(runtime) == ("cuda:0", "cuda:1")
    with pytest.raises(ProtocolError):
        validate_feature_worker_topology({**runtime, "generation_devices": ["cuda:0"]})


def _component_record(
    query: str,
    source: str,
    training_seed: int,
    case_ids: tuple[str, ...],
) -> FeatureComponentRecord:
    support_hash = build_case_bootstrap_plan(
        target_id=query, support_case_ids=case_ids
    ).support_partition_hash
    mmd = {seed: float(seed + training_seed) / 100.0 for seed in GENERATION_SEEDS}
    unhashed = {
        "schema_version": "midogpp_endpoint_router_feature_component_v1",
        "query_center": query,
        "candidate_source": source,
        "training_seed": training_seed,
        "relative_npz_path": f"component_q{query}_e{source}_t{training_seed}.npz",
        "npz_sha256": SHA,
        "array_prefix": f"q{query}",
        "support_row_count": len(case_ids),
        "support_case_count": 8,
        "support_partition_hash": support_hash,
        "support_row_identity_hash": SHA,
        "center_partition_hash": SHA,
        "case_equal_energy": float(training_seed) / 100.0,
        "linear_kernel_mmd2_by_generation_seed": {
            str(seed): mmd[seed] for seed in GENERATION_SEEDS
        },
        "task_hash": SHA,
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    return FeatureComponentRecord(
        query_center=query,
        candidate_source=source,
        training_seed=training_seed,
        relative_npz_path=str(unhashed["relative_npz_path"]),
        npz_sha256=SHA,
        array_prefix=f"q{query}",
        support_row_count=len(case_ids),
        support_case_count=8,
        support_partition_hash=support_hash,
        support_row_identity_hash=SHA,
        center_partition_hash=SHA,
        case_equal_energy=float(training_seed) / 100.0,
        linear_kernel_mmd2_by_generation_seed=mmd,
        task_hash=SHA,
        component_hash=canonical_sha256(unhashed),
    )
