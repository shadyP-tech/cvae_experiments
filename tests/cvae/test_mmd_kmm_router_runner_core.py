from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel

from midogpp_thesis.cvae.diagnostics.mmd_kmm_router import (
    inputs,
    planning,
    prediction,
    runner,
    source_products,
)
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.contracts import (
    COMMON_FRAME_HASH,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    INPUT_ARTIFACT_IDS,
    KERNEL_BATCH_ROWS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    NYSTROEM_GAMMA,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.mmd_kmm_mixture import FrozenNystroemFeatureMap


def test_workstation_schedule_and_cache_bounds_are_exact() -> None:
    assert EXPECTED_SOURCE_TASK_COUNT == 27
    assert EXPECTED_SOURCE_BLOCK_COUNT == 81
    assert EXPECTED_PREDICTION_CELL_COUNT == 162
    assert MAX_SOURCE_PREFIX_PER_CLASS == 256
    assert EXPECTED_SOURCE_BLOCK_COUNT * 2 * MAX_SOURCE_PREFIX_PER_CLASS * 3840 * 4 == 637_009_920


def test_workspace_sorted_provenance_is_returned_in_canonical_input_order(
    tmp_path: Path,
) -> None:
    configured_paths = [tmp_path / f"input-{index}" for index in range(5)]
    config = SimpleNamespace(
        expert_bank_root=configured_paths[0],
        generation_lock_root=configured_paths[1],
        equal_union_policy_root=configured_paths[2],
        validation_cache_root=configured_paths[3],
        validation_manifest_path=configured_paths[4] / "manifest.csv",
    )
    path_by_id = dict(zip(INPUT_ARTIFACT_IDS, configured_paths, strict=True))
    rows = [
        {
            "artifact_id": artifact_id,
            "resolved_path": str(path_by_id[artifact_id]),
            "exists": True,
            "semantic_identities": {},
            "file_integrity": {},
        }
        for artifact_id in sorted(INPUT_ARTIFACT_IDS)
    ]
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "input_artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": "midogpp_input_artifacts_v2",
                "dataset_id": "midogpp",
                "experiment_id": (
                    "midogpp.oracle.uniform_b_v2_consumed_validation_"
                    "mmd_kmm_router.v1"
                ),
                "stage": "90_oracles_and_diagnostics",
                "claim_scope": "diagnostic_only",
                "input_artifacts": rows,
            }
        ),
        encoding="utf-8",
    )
    observed = inputs.validate_workspace_provenance(tmp_path, config)
    assert tuple(observed) == INPUT_ARTIFACT_IDS


def test_source_products_hash_is_stable_across_csv_type_roundtrip(
    tmp_path: Path,
) -> None:
    index = {
        "schema_version": "index-v1",
        "block_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "generation_seed": 42,
        "stream_id": "stream",
        "expert_lock_hash": "expert",
        "samples_per_class": 256,
        "row_count": 512,
        "feature_dim": 3840,
        "output_sha256": "a" * 64,
    }
    score = {
        "schema_version": "score-v1",
        "query_center": "1",
        "source_center": "0",
        "training_seed_17_z": 0.1,
        "training_seed_42_z": 0.2,
        "training_seed_101_z": 0.3,
        "mean_calibrated_energy_z": 0.2,
        "query_support_case_count": 2,
        "replica_aggregation": "mean",
        "legal_target_candidate": True,
        "query_support_labels_used": False,
        "exact_nelbo_claimed": False,
    }
    typed = source_products.SourceProducts(
        array_path=tmp_path / "blocks.npy",
        index_rows=(index,),
        compatibility_case_rows=(),
        compatibility_score_rows=(score,),
        calibrated_energy_by_target={},
    )
    csv_typed = source_products.SourceProducts(
        array_path=typed.array_path,
        index_rows=({key: str(value) for key, value in index.items()},),
        compatibility_case_rows=(),
        compatibility_score_rows=(
            {key: str(value) for key, value in score.items()},
        ),
        calibrated_energy_by_target={},
    )
    assert typed.source_products_hash == csv_typed.source_products_hash


def test_rejected_concurrent_launch_cannot_unlink_or_bypass_run_lock(
    tmp_path: Path,
) -> None:
    with runner._exclusive_run_lock(tmp_path):
        assert (tmp_path / ".run.lock").is_file()
        with pytest.raises(ProtocolError, match="already owns"):
            with runner._exclusive_run_lock(tmp_path):
                pass
        assert (tmp_path / ".run.lock").is_file()
        with pytest.raises(ProtocolError, match="already owns"):
            with runner._exclusive_run_lock(tmp_path):
                pass
    with runner._exclusive_run_lock(tmp_path):
        assert (tmp_path / ".run.lock").is_file()


def test_stale_atomic_temps_are_pruned_without_touching_other_files(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "checkpoints"
    nested.mkdir()
    stale = nested / "cell.npz.123.tmp"
    retained = nested / "cell.npz"
    stale.write_bytes(b"partial")
    retained.write_bytes(b"durable")
    runner._prune_stale_temp_files(tmp_path)
    assert not stale.exists()
    assert retained.read_bytes() == b"durable"


def test_gpu_batched_nystroem_math_matches_cpu_on_cpu_device() -> None:
    rng = np.random.default_rng(7)
    fit = rng.normal(size=(20, 5))
    values = rng.normal(size=(11, 5))
    fitted = Nystroem(
        kernel="rbf",
        gamma=NYSTROEM_GAMMA,
        n_components=8,
        random_state=11,
    ).fit(fit)
    frozen = FrozenNystroemFeatureMap(
        components=fitted.components_,
        normalization=fitted.normalization_,
        gamma=NYSTROEM_GAMMA,
        common_frame_hash=COMMON_FRAME_HASH,
        preprocessing_hash="preprocessing",
        candidate_pool_fit_hash="pool",
        random_state=11,
    )
    observed, probe_error = planning._transform_nystroem_batched(
        values,
        frozen,
        device="cpu",
        batch_rows=KERNEL_BATCH_ROWS,
    )
    expected = rbf_kernel(values, frozen.components, gamma=frozen.gamma) @ frozen.normalization.T
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-4)
    assert probe_error <= 5e-4


def test_route_checkpoint_roundtrip_is_hash_validated(tmp_path: Path) -> None:
    unhashed = {
        "schema_version": "test",
        "target_center": "0",
        "config_contract_hash": "config",
        "support_partition_lock_hash": "support",
        "source_products_hash": "source-products",
        "source_products_lock_hash": "source-lock",
    }
    payload = {**unhashed, "plan_hash": planning.stable_hash(unhashed)}
    path = tmp_path / "route.npz"
    planning._write_route_checkpoint(
        path,
        payload=payload,
        state={"kernel_components": np.eye(3)},
    )
    task = {
        "target_center": "0",
        "config_contract_hash": "config",
        "support_partition_lock_hash": "support",
        "source_products_hash": "source-products",
        "source_products_lock_hash": "source-lock",
    }
    restored, state = planning._load_route_checkpoint(path, task=task)
    assert restored == payload
    np.testing.assert_array_equal(state["kernel_components"], np.eye(3))

    with np.load(path, allow_pickle=False) as raw:
        metadata = np.asarray(raw["checkpoint_json"])
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            checkpoint_json=metadata,
            kernel_components=np.zeros((3, 3)),
        )
    with pytest.raises(ProtocolError, match="failed validation"):
        planning._load_route_checkpoint(path, task=task)


def test_prediction_checkpoint_roundtrip_and_plan_binding(tmp_path: Path) -> None:
    result = {
        "schema_version": "test",
        "config_contract_hash": "config",
        "generation_lock_hash": "generation-lock",
        "source_products_lock_hash": "source-lock",
        "router_plan_lock_hash": "router-lock",
        "evaluation_row_identity_hash": "evaluation-rows",
        "target_center": "0",
        "training_seed": 17,
        "generation_seed": 17,
        "plan_hash": "plan",
        "equal_union_control_predictions": np.asarray([0, 1], dtype=np.uint8),
        "equal_union_control_probabilities": np.asarray([0.2, 0.8], dtype=np.float32),
        "equal_union_control_metadata": {"arm": "control"},
        "mmd_kmm_predictions": np.asarray([1, 1], dtype=np.uint8),
        "mmd_kmm_probabilities": np.asarray([0.6, 0.7], dtype=np.float32),
        "mmd_kmm_metadata": {"arm": "route"},
        "unique_classifier_fit_count": 2,
    }
    path = tmp_path / "prediction.npz"
    prediction._write_prediction_checkpoint(path, result)
    task = {
        "config_contract_hash": "config",
        "generation_lock_hash": "generation-lock",
        "source_products_lock_hash": "source-lock",
        "router_plan_lock_hash": "router-lock",
        "evaluation_row_identity_hash": "evaluation-rows",
        "target_center": "0",
        "training_seed": 17,
        "generation_seed": 17,
        "plan": {"plan_hash": "plan"},
    }
    restored = prediction._load_prediction_checkpoint(path, task=task)
    np.testing.assert_array_equal(
        restored["mmd_kmm_predictions"], result["mmd_kmm_predictions"]
    )
    changed = {**task, "plan": {"plan_hash": "different"}}
    with pytest.raises(ProtocolError, match="failed validation"):
        prediction._load_prediction_checkpoint(path, task=changed)


def test_corrupt_generation_checkpoint_fails_closed(tmp_path: Path) -> None:
    from midogpp_thesis.cvae.diagnostics.mmd_kmm_router import source_products

    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    task = {
        "config_contract_hash": "config",
        "task_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "device": "cuda:0",
        "array_path": str(tmp_path / "missing.npy"),
    }
    with pytest.raises(ProtocolError, match="failed validation"):
        source_products._load_generation_checkpoint(path, task=task)
