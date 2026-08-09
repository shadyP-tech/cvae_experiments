from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router import partitions as partition_module
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.artifact_io import atomic_json, atomic_npz, sha256_file
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.combined_prediction_io import (
    load_task_checkpoint,
    read_combined_store,
    write_combined_store,
    write_task_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.actions import inner_action_library_for
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.contracts import (
    CENTERS, GENERATION_SEEDS, SUPPORT_PARTITION_NAMESPACE, TRAINING_SEEDS,
    expected_target_action_ids,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.development_prediction_execution import validate_development_prediction_store
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.input_contracts import LabelFreeValidationFrame, ValidationRowIdentity
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.inputs import _assert_input_fence
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.prediction_contracts import (
    CombinedPredictionCell,
    array_sha256,
    build_store,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.target_prediction_execution import (
    TARGET_ARRAY_MEMBER, TARGET_INDEX_MEMBER, _persist_probe_seal, _probe_actions,
    _probe_library_hash, _write_target_index_table, materialize_target_probe_predictions,
    validate_target_probe_seal,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.target_label_access import open_target_labels_after_global_seal
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.target_seal import TargetScoringCapability, TARGET_PREDICTION_SEAL_MEMBER
from midogpp_thesis.cvae.protocol import ProtocolError


def _cell(index: int = 0, *, scope: str = "0", action: str = "B", action_hash: str = "a" * 64) -> CombinedPredictionCell:
    return CombinedPredictionCell(
        scope_id=scope, action_id=action, action_hash=action_hash,
        training_seed=17, generation_seed=17,
        support_row_identity_hash="s" * 64, evaluation_row_identity_hash="e" * 64,
        support_predictions=[0, 1], support_probabilities=[0.2, 0.8],
        evaluation_predictions=[1, 0, 1], evaluation_probabilities=[0.7, 0.3, 0.9],
        composition_hash=f"{index + 1:064x}", scaler_state_hash="c" * 64,
        fit_provenance_hash="f" * 64,
    )


def test_combined_cell_canonicalizes_and_locks_all_arrays() -> None:
    cell = _cell()
    for name in (
        "support_predictions", "support_probabilities",
        "evaluation_predictions", "evaluation_probabilities",
    ):
        value = getattr(cell, name)
        assert isinstance(value, np.ndarray)
        assert value.flags.c_contiguous
        assert value.flags.writeable is False


def test_combined_store_rejects_coherently_rehashed_metadata_tamper(tmp_path: Path) -> None:
    store = build_store(
        role="development", cells=(_cell(),), source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash="a" * 64,
        expected_cell_count=1, unique_classifier_fit_count=1,
    )
    arrays = tmp_path / "predictions.npz"
    index = tmp_path / "index.json"
    write_combined_store(arrays, index, store)
    assert read_combined_store(arrays, index).store_hash == store.store_hash

    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["store"]["action_library_hash"] = "b" * 64
    unhashed = {key: value for key, value in payload.items() if key != "prediction_index_hash"}
    payload["prediction_index_hash"] = stable_hash(unhashed)
    atomic_json(index, payload)
    with pytest.raises(ProtocolError, match="metadata"):
        read_combined_store(arrays, index)


def test_combined_store_rejects_offset_tamper_even_when_outer_hashes_match(tmp_path: Path) -> None:
    store = build_store(
        role="development", cells=(_cell(),), source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash="a" * 64,
        expected_cell_count=1, unique_classifier_fit_count=1,
    )
    arrays = tmp_path / "predictions.npz"
    index = tmp_path / "index.json"
    write_combined_store(arrays, index, store)
    with np.load(arrays, allow_pickle=False) as loaded:
        values = {name: np.asarray(loaded[name]) for name in loaded.files}
    values["support_offsets"] = np.asarray([0, 1], dtype=np.int64)
    atomic_npz(arrays, **values)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["array_sha256"] = sha256_file(arrays)
    unhashed = {key: value for key, value in payload.items() if key != "prediction_index_hash"}
    payload["prediction_index_hash"] = stable_hash(unhashed)
    atomic_json(index, payload)
    with pytest.raises(ProtocolError, match="layout"):
        read_combined_store(arrays, index)


@pytest.mark.parametrize(
    "tampered_member",
    (
        "support_predictions",
        "support_probabilities",
        "evaluation_predictions",
        "evaluation_probabilities",
    ),
)
def test_task_checkpoint_round_trip_loads_all_arrays_and_rejects_tamper(
    tmp_path: Path, tampered_member: str
) -> None:
    json_path = tmp_path / "task.json"
    npz_path = tmp_path / "task.npz"
    task = {
        "task_id": "development-H0-q1-train17-gen17",
        "task_hash": "t" * 64,
        "task_role": "development",
        "config_contract_hash": "c" * 64,
        "source_cache_lock_hash": "s" * 64,
        "partition_lock_hash": "p" * 64,
        "support_row_identity_hash": "u" * 64,
        "evaluation_row_identity_hash": "e" * 64,
        "support_row_count": 2,
        "evaluation_row_count": 3,
        "checkpoint_json_path": str(json_path),
        "checkpoint_npz_path": str(npz_path),
    }
    expected = {
        "support_predictions": np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        "support_probabilities": np.asarray(
            [[0.25, 0.75], [0.8, 0.2]], dtype=np.float32
        ),
        "evaluation_predictions": np.asarray(
            [[1, 0, 1], [0, 1, 0]], dtype=np.uint8
        ),
        "evaluation_probabilities": np.asarray(
            [[0.7, 0.3, 0.9], [0.1, 0.6, 0.4]], dtype=np.float32
        ),
    }
    action_rows = tuple(
        {
            "action_id": action_id,
            "support_prediction_sha256": array_sha256(
                expected["support_predictions"][ordinal]
            ),
            "support_probability_sha256": array_sha256(
                expected["support_probabilities"][ordinal]
            ),
            "evaluation_prediction_sha256": array_sha256(
                expected["evaluation_predictions"][ordinal]
            ),
            "evaluation_probability_sha256": array_sha256(
                expected["evaluation_probabilities"][ordinal]
            ),
        }
        for ordinal, action_id in enumerate(("B", "T"))
    )
    write_task_checkpoint(task, action_rows=action_rows, **expected)

    loaded = load_task_checkpoint(task)
    assert loaded is not None
    for member_name, values in expected.items():
        np.testing.assert_array_equal(loaded[member_name], values)

    with np.load(npz_path, allow_pickle=False) as arrays:
        tampered = {
            member_name: np.asarray(arrays[member_name]).copy()
            for member_name in arrays.files
        }
    if tampered_member.endswith("predictions"):
        tampered[tampered_member][0, 0] = 1 - tampered[tampered_member][0, 0]
    else:
        tampered[tampered_member][0, 0] += np.float32(0.125)
    atomic_npz(npz_path, **tampered)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["checkpoint_npz_sha256"] = sha256_file(npz_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    payload["checkpoint_hash"] = stable_hash(unhashed)
    atomic_json(json_path, payload)

    with pytest.raises(ProtocolError, match="vector bytes drifted"):
        load_task_checkpoint(task)


def test_target_probe_seal_is_reconstructively_validated(tmp_path: Path) -> None:
    cells = []
    for target in CENTERS:
        for training_seed in (17, 42, 101):
            for generation_seed in (17, 42, 101):
                for ordinal, action in enumerate(_probe_actions(target)):
                    base = _cell(ordinal, scope=target, action=str(action["action_id"]), action_hash=str(action["action_hash"]))
                    cells.append(CombinedPredictionCell(
                        **{**base.__dict__, "training_seed": training_seed, "generation_seed": generation_seed}
                    ))
    store = build_store(
        role="target_probe", cells=cells, source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash=_probe_library_hash(),
        expected_cell_count=729, unique_classifier_fit_count=729,
    )
    partitions = SimpleNamespace(lock_hash="p" * 64)
    _persist_probe_seal(tmp_path, store=store, partitions=partitions)
    assert validate_target_probe_seal(tmp_path, store, partitions)["status"].startswith("SEALED")

    path = tmp_path / "manifests/ensemble_endpoint_target_probe_seal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_plan_built"] = True
    unhashed = {key: value for key, value in payload.items() if key != "probe_seal_hash"}
    payload["probe_seal_hash"] = stable_hash(unhashed)
    atomic_json(path, payload)
    with pytest.raises(ProtocolError, match="probe seal"):
        validate_target_probe_seal(tmp_path, store, partitions)


def test_input_fence_rejects_prior_stage_outputs() -> None:
    config = SimpleNamespace(input_artifact_ids=(
        "bank", "generation", "cache", "manifest", "exact_tail_utility_surface_v2"
    ))
    with pytest.raises(ProtocolError, match="prior routing output"):
        _assert_input_fence(config)


def test_partition_builder_hashes_v2_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = []
    by_center = {}
    ordinal = 0
    for center_index, center in enumerate(CENTERS):
        count = 4 if center_index == 0 else 5
        selected = []
        for local in range(count):
            row = ValidationRowIdentity(
                row_ordinal=ordinal, manifest_row_index=ordinal,
                sample_id=f"sample::{center}::{local}", case_id=f"case::{center}::{local}",
                center=center,
            )
            rows.append(row); selected.append(row); ordinal += 1
        by_center[center] = tuple(selected)
    frame = LabelFreeValidationFrame(
        embeddings=np.zeros((44, 3840), dtype=np.float32), rows=tuple(rows),
        rows_by_center=by_center, cache_binding={"cache": "v2"},
    )
    monkeypatch.setattr(
        partition_module, "deterministic_case_partitions",
        lambda *args, **kwargs: SimpleNamespace(support_indices=(0, 1), partition_hash=stable_hash(kwargs)),
    )
    surface = partition_module.build_fixed_partition_surface(frame, config_contract_hash="c" * 64)
    assert surface.lock_payload["support_partition_namespace"] == SUPPORT_PARTITION_NAMESPACE
    assert {row["support_partition_namespace"] for row in surface.table_rows} == {SUPPORT_PARTITION_NAMESPACE}
    assert "exact_tail_router" not in surface.lock_payload["schema_version"]


def test_development_store_validator_rejects_canonical_order_tamper() -> None:
    cells = []
    for outer in CENTERS:
        for query in CENTERS:
            if outer == query:
                continue
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for ordinal, action in enumerate(inner_action_library_for(outer, query)):
                        base = _cell(ordinal, scope=f"{outer}::{query}", action=action.action_id, action_hash=action.action_hash)
                        cells.append(CombinedPredictionCell(
                            **{**base.__dict__, "training_seed": training_seed, "generation_seed": generation_seed}
                        ))
    library_hash = stable_hash("placeholder")
    # The public validator reconstructs the actual library hash internally.
    from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.actions import build_inner_ensemble_endpoint_action_library
    library_hash = build_inner_ensemble_endpoint_action_library().action_library_hash
    store = build_store(
        role="development", cells=cells, source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash=library_hash,
        expected_cell_count=5184, unique_classifier_fit_count=5184,
    )
    validate_development_prediction_store(
        store, source_cache_lock_hash="l" * 64, partition_lock_hash="p" * 64
    )
    reordered = list(cells)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    tampered = build_store(
        role="development", cells=reordered, source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash=library_hash,
        expected_cell_count=5184, unique_classifier_fit_count=5184,
    )
    with pytest.raises(ProtocolError, match="ordering"):
        validate_development_prediction_store(
            tampered, source_cache_lock_hash="l" * 64, partition_lock_hash="p" * 64
        )


def test_final_store_fast_path_reconstructs_probe_without_frame_or_workers(tmp_path: Path) -> None:
    final_cells = []
    for target in CENTERS:
        probe_by_id = {str(action["action_id"]): action for action in _probe_actions(target)}
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for ordinal, action_id in enumerate(expected_target_action_ids(target)):
                    probe_action = probe_by_id.get(action_id)
                    action_hash = str(probe_action["action_hash"]) if probe_action else f"{ordinal + 10:064x}"
                    base = _cell(ordinal, scope=target, action=action_id, action_hash=action_hash)
                    final_cells.append(CombinedPredictionCell(
                        **{**base.__dict__, "training_seed": training_seed, "generation_seed": generation_seed}
                    ))
    final = build_store(
        role="target_final", cells=final_cells, source_cache_lock_hash="l" * 64,
        partition_lock_hash="fold" * 16, action_library_hash="a" * 64,
        expected_cell_count=1053, unique_classifier_fit_count=810,
    )
    write_combined_store(tmp_path / TARGET_ARRAY_MEMBER, tmp_path / TARGET_INDEX_MEMBER, final)
    probe_cells = [cell for cell in final_cells if cell.action_id in {
        str(action["action_id"]) for target in CENTERS for action in _probe_actions(target)
    }]
    rebuilt = build_store(
        role="target_probe", cells=probe_cells, source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash=_probe_library_hash(),
        expected_cell_count=729, unique_classifier_fit_count=729,
    )
    partitions = SimpleNamespace(lock_hash="p" * 64)
    _persist_probe_seal(tmp_path, store=rebuilt, partitions=partitions)

    class NoFrame:
        def embeddings_for(self, rows):
            raise AssertionError("fast path must not materialize scratch or fit")

    observed = materialize_target_probe_predictions(
        SimpleNamespace(), SimpleNamespace(), NoFrame(), partitions,
        source_cache_lock_hash="l" * 64, root=tmp_path,
    )
    assert observed.store_hash == rebuilt.store_hash
    assert not (tmp_path / "checkpoints/ensemble_endpoint_target").exists()


def test_target_label_gate_rejects_missing_or_forged_capability_before_manifest(tmp_path: Path) -> None:
    config = SimpleNamespace(validation_manifest_path=tmp_path / "must-not-open.csv")
    partitions = SimpleNamespace()
    with pytest.raises(ProtocolError, match="reconstructively validated capability"):
        open_target_labels_after_global_seal(
            config, partitions, root=tmp_path, capability=None  # type: ignore[arg-type]
        )
    unhashed = {
        "status": "SEALED_ALL_TARGET_ACTIONS_BEFORE_TERMINAL_TARGET_SCORING",
        "all_actions_frozen": True, "all_predictions_materialized": True,
        "target_support_labels_opened": False, "target_evaluation_labels_opened": False,
    }
    seal = {**unhashed, "seal_hash": stable_hash(unhashed)}
    atomic_json(tmp_path / TARGET_PREDICTION_SEAL_MEMBER, seal)
    forged = TargetScoringCapability(root=tmp_path, payload={**seal, "seal_hash": "f" * 64})
    with pytest.raises(ProtocolError, match="valid durable global seal"):
        open_target_labels_after_global_seal(
            config, partitions, root=tmp_path, capability=forged
        )


def test_target_index_persistence_rejects_tamper_without_repair(tmp_path: Path) -> None:
    store = build_store(
        role="target_final", cells=(_cell(),), source_cache_lock_hash="l" * 64,
        partition_lock_hash="p" * 64, action_library_hash="a" * 64,
        expected_cell_count=1, unique_classifier_fit_count=1,
    )
    _write_target_index_table(tmp_path, store)
    path = tmp_path / "tables/target_prediction_index.csv"
    tampered = path.read_bytes() + b"tamper"
    path.write_bytes(tampered)
    with pytest.raises(ProtocolError, match="resumed CSV"):
        _write_target_index_table(tmp_path, store)
    assert path.read_bytes() == tampered
