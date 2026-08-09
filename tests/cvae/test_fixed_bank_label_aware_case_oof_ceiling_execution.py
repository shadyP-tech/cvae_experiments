from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.runtime import label_free_action_predictions as prediction_runtime
from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    atomic_npz,
    sha256_array,
    sha256_file,
)
from midogpp_thesis.cvae.runtime.label_free_action_predictions import (
    LabelFreePredictionStore,
    PredictionCell,
    _load_prediction_checkpoint,
    _load_validated_prediction_inputs,
    build_direct_target_actions,
)
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    SOURCE_ROWS_PER_CLASS,
    source_block_sha256,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def test_direct_target_action_library_is_B_plus_eight_target_excluded_hxe() -> None:
    for target in CENTERS:
        actions = build_direct_target_actions(target)
        assert len(actions) == 9
        assert actions[0].action_id == "B"
        assert actions[0].selected_source is None
        assert tuple(action.selected_source for action in actions[1:]) == tuple(
            center for center in CENTERS if center != target
        )
        assert all(action.selected_source != target for action in actions)
        assert all(
            sum(action.counts_by_class[str(label)].values())
            == (1024 if action.action_id == "B" else 1152)
            for action in actions
            for label in (0, 1)
        )


def test_prediction_checkpoint_rejects_coherent_probability_tamper_with_stale_hard_prediction_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _checkpoint_task(tmp_path)
    monkeypatch.setattr(
        prediction_runtime,
        "_load_validated_prediction_inputs",
        lambda _task: ({}, np.empty((3, COMMON_OUTPUT_DIM), dtype=np.float32)),
    )
    probabilities = np.full((9, 3), np.float32(0.25), dtype=np.float32)
    _write_checkpoint(task, probabilities)
    assert _load_prediction_checkpoint(task) is not None

    tampered = probabilities.copy()
    tampered[0, 0] = np.float32(0.75)
    # Update every probability/file/checkpoint hash an attacker could update,
    # while deliberately leaving the independently bound hard prediction hash.
    payload = _checkpoint_payload(task, tampered)
    payload["actions"][0]["prediction_sha256"] = sha256_array(
        (probabilities[0] >= np.float32(0.5)).astype(np.uint8)
    )
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    payload["checkpoint_hash"] = stable_hash(unhashed)
    atomic_json(Path(str(task["checkpoint_json_path"])), payload)
    with pytest.raises(ProtocolError, match="action drifted"):
        _load_prediction_checkpoint(task)


def test_existing_checkpoint_rebinds_target_slice_bytes_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, source_values = _bound_task(tmp_path)
    _patch_source_load(monkeypatch, task, source_values)
    probabilities = np.full((9, 3), np.float32(0.25), dtype=np.float32)
    _write_checkpoint(task, probabilities)
    assert _load_prediction_checkpoint(task) is not None

    target_path = Path(str(task["target_array_path"]))
    target = np.load(target_path, allow_pickle=False)
    target[0, 0] = np.float32(1.0)
    with target_path.open("wb") as handle:
        np.save(handle, target, allow_pickle=False)
    with pytest.raises(ProtocolError, match="target slice bytes drifted"):
        _load_prediction_checkpoint(task)


def test_worker_input_rejects_used_source_block_byte_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, source_values = _bound_task(tmp_path)
    _patch_source_load(monkeypatch, task, source_values)
    _load_validated_prediction_inputs(task)

    first_candidate = tuple(task["candidate_sources"])[0]
    source_values.blocks[first_candidate][0, 0] = np.float32(1.0)
    with pytest.raises(ProtocolError, match="source block bytes drifted"):
        _load_validated_prediction_inputs(task)


def test_prediction_cell_and_store_reject_noncanonical_action_identity_and_order() -> None:
    values = np.asarray([0.25], dtype=np.float32)
    probability_hash = sha256_array(values)
    prediction_hash = sha256_array(
        (values >= np.float32(0.5)).astype(np.uint8)
    )
    with pytest.raises(ProtocolError, match="cell drifted"):
        PredictionCell(
            target_center="0",
            action_id="Hxe::1",
            action_hash="0" * 16,
            training_seed=17,
            generation_seed=17,
            row_identity_hash="row",
            probabilities=values,
            probability_sha256=probability_hash,
            predictions_sha256=prediction_hash,
            composition_hash="composition",
            scaler_state_hash="scaler",
            fit_provenance_hash="fit",
        )

    cells: list[PredictionCell] = []
    for target in CENTERS:
        for training_seed in prediction_runtime.TRAINING_SEEDS:
            for generation_seed in prediction_runtime.GENERATION_SEEDS:
                for action in build_direct_target_actions(target):
                    cells.append(
                        PredictionCell(
                            target_center=target,
                            action_id=action.action_id,
                            action_hash=action.action_hash,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            row_identity_hash="row",
                            probabilities=values,
                            probability_sha256=probability_hash,
                            predictions_sha256=prediction_hash,
                            composition_hash="composition",
                            scaler_state_hash="scaler",
                            fit_provenance_hash="fit",
                        )
                    )
    cells[0], cells[1] = cells[1], cells[0]
    with pytest.raises(ProtocolError, match="inventory drifted"):
        LabelFreePredictionStore(
            cells=tuple(cells),
            rows_by_center={center: (f"row-{center}",) for center in CENTERS},
            case_ids_by_center={center: (f"case-{center}",) for center in CENTERS},
            source_stream_lock_hash="source-lock",
            action_library_hash=prediction_runtime._canonical_action_library_hash(),
            target_cache_binding_hash="cache-binding",
            store_hash="store",
        )


def test_neutral_runtime_has_no_diagnostic_or_routing_imports() -> None:
    runtime_root = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/runtime"
    )
    violations: list[str] = []
    for path in sorted(runtime_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "diagnostics" in module or ".routing" in module or module.endswith("routing"):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if ".diagnostics" in alias.name or ".routing" in alias.name:
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
    assert violations == []


def _checkpoint_task(tmp_path: Path) -> dict[str, object]:
    actions = [action.identity_payload() for action in build_direct_target_actions("0")]
    return {
        "task_id": "target_0_train_17_generation_17",
        "task_hash": "task-hash",
        "target_start": 0,
        "target_stop": 3,
        "actions": actions,
        "checkpoint_json_path": str(tmp_path / "checkpoint.json"),
        "checkpoint_npz_path": str(tmp_path / "checkpoint.npz"),
    }


class _FakeSourceValues:
    shape = (
        prediction_runtime.EXPECTED_STREAM_COUNT,
        2 * SOURCE_ROWS_PER_CLASS,
        COMMON_OUTPUT_DIM,
    )
    dtype = np.dtype(np.float32)

    def __init__(self, blocks: dict[str, np.ndarray], ordinals: dict[int, str]) -> None:
        self.blocks = blocks
        self.ordinals = ordinals

    def __getitem__(self, ordinal: int) -> np.ndarray:
        return self.blocks[self.ordinals[int(ordinal)]]


def _bound_task(tmp_path: Path) -> tuple[dict[str, object], _FakeSourceValues]:
    target = "0"
    candidates = tuple(center for center in CENTERS if center != target)
    block = np.zeros(
        (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM), dtype=np.float32
    )
    output_hash = source_block_sha256(block)
    source_rows: list[dict[str, object]] = []
    ordinal_sources: dict[int, str] = {}
    ordinal = 0
    for source in CENTERS:
        for training_seed in prediction_runtime.TRAINING_SEEDS:
            for generation_seed in prediction_runtime.GENERATION_SEEDS:
                source_rows.append(
                    {
                        "block_ordinal": ordinal,
                        "source_center": source,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "stream_id": f"{source}-{training_seed}-{generation_seed}",
                        "expert_lock_hash": "expert",
                        "rows_per_class": SOURCE_ROWS_PER_CLASS,
                        "row_count": 2 * SOURCE_ROWS_PER_CLASS,
                        "feature_dim": COMMON_OUTPUT_DIM,
                        "output_sha256": output_hash,
                    }
                )
                if training_seed == 17 and generation_seed == 17:
                    ordinal_sources[ordinal] = source
                ordinal += 1
    blocks = {source: block.copy() for source in CENTERS}
    source_values = _FakeSourceValues(blocks, ordinal_sources)

    target_values = np.zeros((3, COMMON_OUTPUT_DIM), dtype=np.float32)
    target_path = tmp_path / "targets.npy"
    with target_path.open("wb") as handle:
        np.save(handle, target_values, allow_pickle=False)
    actions = [action.identity_payload() for action in build_direct_target_actions(target)]
    unhashed = {
        "schema_version": "midogpp_label_free_action_prediction_task_v1",
        "task_id": "target_0_train_17_generation_17",
        "config_contract_hash": "config",
        "source_stream_lock_hash": "source-lock",
        "partition_lock_hash": "partition-lock",
        "action_library_hash": prediction_runtime._canonical_action_library_hash(),
        "target_center": target,
        "training_seed": 17,
        "generation_seed": 17,
        "candidate_sources": list(candidates),
        "source_array_path": str(tmp_path / "sources.npy"),
        "source_array_sha256": "a" * 64,
        "source_stream_index_hash": "index",
        "source_index_rows_hash": stable_hash(source_rows),
        "source_index_rows": source_rows,
        "target_array_path": str(target_path),
        "target_array_sha256": sha256_array(target_values),
        "target_array_shape": list(target_values.shape),
        "target_array_dtype": str(target_values.dtype),
        "target_scratch_hash": "scratch",
        "target_cache_binding_hash": "cache-binding",
        "target_start": 0,
        "target_stop": len(target_values),
        "target_row_identity_hash": "rows",
        "target_slice_sha256": sha256_array(target_values),
        "actions": actions,
        "classifier": {},
        "threads_per_fit": 3,
        "labels_available": False,
        "target_expert_available": False,
    }
    task = {
        **unhashed,
        "task_hash": stable_hash(unhashed),
        "checkpoint_json_path": str(tmp_path / "checkpoint.json"),
        "checkpoint_npz_path": str(tmp_path / "checkpoint.npz"),
    }
    return task, source_values


def _patch_source_load(
    monkeypatch: pytest.MonkeyPatch,
    task: dict[str, object],
    source_values: _FakeSourceValues,
) -> None:
    real_load = np.load
    source_path = Path(str(task["source_array_path"]))

    def load(path: object, *args: object, **kwargs: object) -> object:
        if Path(str(path)) == source_path:
            return source_values
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(prediction_runtime.np, "load", load)


def _write_checkpoint(task: dict[str, object], probabilities: np.ndarray) -> None:
    payload = _checkpoint_payload(task, probabilities)
    atomic_json(Path(str(task["checkpoint_json_path"])), payload)


def _checkpoint_payload(
    task: dict[str, object], probabilities: np.ndarray
) -> dict[str, object]:
    npz_path = Path(str(task["checkpoint_npz_path"]))
    atomic_npz(npz_path, probabilities=np.ascontiguousarray(probabilities, dtype=np.float32))
    actions = []
    for action, values in zip(task["actions"], probabilities, strict=True):
        actions.append(
            {
                "action_id": action["action_id"],
                "action_hash": action["action_hash"],
                "probability_sha256": sha256_array(values),
                "prediction_sha256": sha256_array(
                    (values >= np.float32(0.5)).astype(np.uint8)
                ),
                "composition_hash": "composition",
                "scaler_state_hash": "scaler",
                "fit_provenance_hash": "fit",
            }
        )
    unhashed = {
        "schema_version": "midogpp_label_free_action_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "shape": list(probabilities.shape),
        "dtype": str(probabilities.dtype),
        "actions": actions,
        "labels_consumed": False,
        "target_expert_used": False,
        "shared_representation_updated": False,
    }
    return {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
