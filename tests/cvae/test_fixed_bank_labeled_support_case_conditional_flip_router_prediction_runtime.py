from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import multiprocessing as mp
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.constants import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    atomic_npz,
    read_json,
    sha256_array,
    sha256_file,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    EXPECTED_CELL_COUNT,
    EXPECTED_TASK_COUNT,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_SEAL_MEMBER,
    PredictionCell,
    PredictionStore,
    prediction_store_hash,
    validate_action_library,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_planning import (
    build_prediction_tasks,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_store import (
    load_global_prediction_seal,
    write_prediction_store,
)
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_worker import (
    load_prediction_checkpoint,
)


CONFIG_HASH = "1" * 16
PARTITION_HASH = "2" * 64
SOURCE_HASH = "3" * 16
BINDING_HASH = "4" * 64


@dataclass(frozen=True)
class _FixtureRecord:
    ordinal: int

    def to_payload(self) -> dict[str, object]:
        return {"ordinal": self.ordinal}


@dataclass(frozen=True)
class _FixtureSource:
    source_array_path: Path
    records: tuple[_FixtureRecord, ...] = (_FixtureRecord(0),)
    lock_hash: str = SOURCE_HASH


@dataclass(frozen=True)
class _FixtureConfig:
    contract_hash: str = CONFIG_HASH
    classifier: Mapping[str, object] = None  # type: ignore[assignment]
    runtime: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "classifier",
            {
                "family": "logistic_regression",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 10,
                "class_weight": None,
                "random_state": 0,
                "l1_ratio": None,
                "threshold_policy": "positive_probability_gte_0p5",
                "scaler_fit": "unweighted",
            },
        )
        object.__setattr__(self, "runtime", {"classifier_threads_per_worker": 3})


def _contains_nonplain_mapping(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_nonplain_mapping(key) or _contains_nonplain_mapping(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_nonplain_mapping(item) for item in value)
    return type(value).__name__ == "mappingproxy"


def _planned_tasks(tmp_path: Path) -> tuple[dict[str, object], ...]:
    source_path = tmp_path / "source.npy"
    source_path.write_bytes(b"source fixture; workers do not open this in checkpoint tests")
    target_path = tmp_path / "target.npy"
    target_path.write_bytes(b"target fixture; workers do not open this in checkpoint tests")
    payload, library_hash = validate_action_library(action_library_by_target())
    offsets = {
        center: {
            "start": 0,
            "stop": 2,
            "row_identity_hash": stable_hash([f"{center}-row-0", f"{center}-row-1"]),
            "target_slice_sha256": "5" * 64,
        }
        for center in CENTERS
    }
    scratch = {
        "array_path": str(target_path),
        "array_sha256": sha256_file(target_path),
        "offsets": offsets,
    }
    return build_prediction_tasks(
        _FixtureConfig(),
        _FixtureSource(source_path),
        scratch,
        payload,
        library_hash,
        PARTITION_HASH,
        tmp_path,
    )


def _publish_checkpoint(task: Mapping[str, object]) -> None:
    values = np.stack(
        [
            np.asarray((0.05 + ordinal / 20.0, 0.95 - ordinal / 20.0), dtype=np.float32)
            for ordinal in range(10)
        ]
    )
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(npz_path, probabilities=values)
    actions = []
    for ordinal, raw in enumerate(task["actions"]):
        probabilities = values[ordinal]
        actions.append(
            {
                "action_id": raw["action_id"],
                "action_hash": raw["action_hash"],
                "probability_sha256": sha256_array(probabilities),
                "prediction_sha256": sha256_array(
                    (probabilities >= np.float32(0.5)).astype(np.uint8)
                ),
                "fit_provenance_hash": stable_hash(
                    {"task_hash": task["task_hash"], "action_id": raw["action_id"]}
                ),
            }
        )
    checkpoint = {
        "schema_version": "fixed_bank_a1_prediction_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "target_center": task["target_center"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "target_row_identity_hash": task["target_row_identity_hash"],
        "array_sha256": sha256_file(npz_path),
        "array_shape": list(values.shape),
        "array_dtype": str(values.dtype),
        "actions": actions,
        "labels_available": False,
        "target_expert_available": False,
    }
    atomic_json(
        json_path,
        {**checkpoint, "checkpoint_hash": stable_hash(checkpoint)},
    )


def _prediction_cells() -> tuple[
    tuple[PredictionCell, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    str,
]:
    library = action_library_by_target()
    _, library_hash = validate_action_library(library)
    rows = {center: (f"{center}-row-0", f"{center}-row-1") for center in CENTERS}
    cases = {center: (f"{center}-case-0", f"{center}-case-1") for center in CENTERS}
    cells: list[PredictionCell] = []
    ordinal = 0
    for target in CENTERS:
        row_hash = stable_hash(rows[target])
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for action in library[target]:
                    probability = np.asarray(
                        ((ordinal % 101) / 100.0, ((ordinal + 37) % 101) / 100.0),
                        dtype=np.float32,
                    )
                    cells.append(
                        PredictionCell(
                            target_center=target,
                            action_id=action.action_id,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            probabilities=probability,
                            action_hash=action.action_hash,
                            row_identity_hash=row_hash,
                            probability_sha256=sha256_array(probability),
                            prediction_sha256=sha256_array(
                                (probability >= np.float32(0.5)).astype(np.uint8)
                            ),
                            fit_provenance_hash=stable_hash(
                                {
                                    "target": target,
                                    "action": action.action_id,
                                    "training": training_seed,
                                    "generation": generation_seed,
                                }
                            ),
                        )
                    )
                    ordinal += 1
    return tuple(cells), rows, cases, library_hash


def test_action_library_and_prediction_store_have_exact_90_by_9_topology() -> None:
    library = action_library_by_target()
    payload, library_hash = validate_action_library(library)
    assert tuple(payload) == CENTERS
    assert sum(len(actions) for actions in payload.values()) == 90
    assert all(
        tuple(action["action_id"] for action in payload[target])
        == ("B", "U", *(f"A1::source={source}" for source in CENTERS if source != target))
        for target in CENTERS
    )

    cells, rows, cases, observed_library_hash = _prediction_cells()
    assert observed_library_hash == library_hash
    store_hash = prediction_store_hash(
        cells,
        rows,
        cases,
        SOURCE_HASH,
        library_hash,
        BINDING_HASH,
    )
    store = PredictionStore(
        cells,
        rows,
        cases,
        SOURCE_HASH,
        library_hash,
        BINDING_HASH,
        store_hash,
    )
    assert len(store.cells) == EXPECTED_CELL_COUNT == 810
    assert len({cell.key for cell in store.cells}) == 810
    assert store.exact_nine("0", "B").shape == (2,)


def test_prediction_store_rejects_duplicate_or_reordered_cells_even_if_rehashed() -> None:
    cells, rows, cases, library_hash = _prediction_cells()
    for drifted in (
        (replace(cells[0], action_id=cells[1].action_id), *cells[1:]),
        (cells[1], cells[0], *cells[2:]),
    ):
        drifted_hash = prediction_store_hash(
            drifted,
            rows,
            cases,
            SOURCE_HASH,
            library_hash,
            BINDING_HASH,
        )
        with pytest.raises(ProtocolError, match="prediction store drifted"):
            PredictionStore(
                tuple(drifted),
                rows,
                cases,
                SOURCE_HASH,
                library_hash,
                BINDING_HASH,
                drifted_hash,
            )


def test_plain_prediction_task_and_checkpoint_cross_a_real_spawn_boundary(
    tmp_path: Path,
) -> None:
    tasks = _planned_tasks(tmp_path)
    assert len(tasks) == EXPECTED_TASK_COUNT == 81
    assert all(type(task) is dict and not _contains_nonplain_mapping(task) for task in tasks)
    pickle.dumps(tasks[0])
    _publish_checkpoint(tasks[0])

    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
            payload = executor.submit(load_prediction_checkpoint, tasks[0]).result(timeout=30)
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"OS semaphore creation is unavailable in this sandbox: {exc}")

    assert payload is not None
    assert payload["task_id"] == tasks[0]["task_id"]
    assert len(payload["actions"]) == 10


def test_checkpoint_tamper_is_rejected_after_outer_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    task = _planned_tasks(tmp_path)[0]
    _publish_checkpoint(task)
    path = Path(str(task["checkpoint_json_path"]))
    payload = read_json(path)
    payload["actions"][0]["probability_sha256"] = "f" * 64
    payload["checkpoint_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    )
    atomic_json(path, payload)

    with pytest.raises(ProtocolError, match="checkpoint action record drifted"):
        load_prediction_checkpoint(task)


def test_prediction_index_tamper_is_rejected_after_outer_seal_is_recomputed(
    tmp_path: Path,
) -> None:
    cells, rows, cases, library_hash = _prediction_cells()
    store_hash = prediction_store_hash(
        cells,
        rows,
        cases,
        SOURCE_HASH,
        library_hash,
        BINDING_HASH,
    )
    write_prediction_store(
        tmp_path,
        cells,
        rows,
        cases,
        CONFIG_HASH,
        PARTITION_HASH,
        SOURCE_HASH,
        library_hash,
        BINDING_HASH,
        store_hash,
    )
    assert load_global_prediction_seal(
        tmp_path,
        expected_config_hash=CONFIG_HASH,
        expected_partition_hash=PARTITION_HASH,
        expected_source_lock_hash=SOURCE_HASH,
        expected_action_library_hash=library_hash,
        expected_target_cache_binding_hash=BINDING_HASH,
    ).store.store_hash == store_hash

    index_path = tmp_path / PREDICTION_INDEX_MEMBER
    seal_path = tmp_path / PREDICTION_SEAL_MEMBER
    index = read_json(index_path)
    index["cells"][0]["action_id"] = "U"
    atomic_json(index_path, index)
    seal = read_json(seal_path)
    seal["index_sha256"] = sha256_file(index_path)
    seal["global_prediction_seal_hash"] = stable_hash(
        {key: value for key, value in seal.items() if key != "global_prediction_seal_hash"}
    )
    atomic_json(seal_path, seal)

    with pytest.raises(ProtocolError, match="prediction store drifted"):
        load_global_prediction_seal(
            tmp_path,
            expected_config_hash=CONFIG_HASH,
            expected_partition_hash=PARTITION_HASH,
            expected_source_lock_hash=SOURCE_HASH,
            expected_action_library_hash=library_hash,
            expected_target_cache_binding_hash=BINDING_HASH,
        )
