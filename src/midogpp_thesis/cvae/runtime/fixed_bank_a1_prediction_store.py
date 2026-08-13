"""Checkpoint consolidation and immutable prediction-store seals."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ..protocol import ProtocolError
from .artifact_io import (
    atomic_json,
    atomic_npz,
    read_json,
    sha256_array,
    sha256_file,
)
from .fixed_bank_a1_prediction_contracts import (
    ACTION_COUNT_PER_TARGET,
    EXPECTED_CELL_COUNT,
    EXPECTED_TASK_COUNT,
    GlobalPredictionSeal,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_SEAL_MEMBER,
    PredictionCell,
    PredictionStore,
    prediction_store_hash,
)


def cells_from_checkpoints(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
) -> tuple[PredictionCell, ...]:
    result: list[PredictionCell] = []
    for task in tasks:
        payload = completed[str(task["task_id"])]
        with np.load(str(task["checkpoint_npz_path"]), allow_pickle=False) as archive:
            values = np.asarray(archive["probabilities"], dtype=np.float32)
        for ordinal, raw in enumerate(payload["actions"]):
            result.append(
                PredictionCell(
                    str(task["target_center"]),
                    str(raw["action_id"]),
                    int(task["training_seed"]),
                    int(task["generation_seed"]),
                    values[ordinal],
                    str(raw["action_hash"]),
                    str(task["target_row_identity_hash"]),
                    str(raw["probability_sha256"]),
                    str(raw["prediction_sha256"]),
                    str(raw["fit_provenance_hash"]),
                )
            )
    cells = tuple(result)
    if len(cells) != EXPECTED_CELL_COUNT or len({cell.key for cell in cells}) != EXPECTED_CELL_COUNT:
        raise ProtocolError("Fixed-bank A1 consolidation coverage drifted.")
    return cells


def write_prediction_store(
    root: Path,
    cells: Sequence[PredictionCell],
    rows: Mapping[str, Sequence[str]],
    cases: Mapping[str, Sequence[str]],
    config_hash: str,
    partition_hash: str,
    source_hash: str,
    library_hash: str,
    binding: str,
    store_hash: str,
) -> None:
    arrays_path = root / PREDICTION_ARRAY_MEMBER
    index_path = root / PREDICTION_INDEX_MEMBER
    seal_path = root / PREDICTION_SEAL_MEMBER
    arrays: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, object]] = []
    for ordinal, cell in enumerate(cells):
        member = f"cell_{ordinal:04d}"
        arrays[member] = cell.probabilities
        index_rows.append(
            {
                "target_center": cell.target_center,
                "action_id": cell.action_id,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "action_hash": cell.action_hash,
                "row_identity_hash": cell.row_identity_hash,
                "array_member": member,
                "probability_sha256": cell.probability_sha256,
                "prediction_sha256": cell.prediction_sha256,
                "fit_provenance_hash": cell.fit_provenance_hash,
            }
        )
    index: dict[str, object] = {
        "schema_version": "fixed_bank_a1_prediction_index_v1",
        "config_contract_hash": config_hash,
        "partition_hash": partition_hash,
        "source_stream_lock_hash": source_hash,
        "action_library_hash": library_hash,
        "target_cache_binding_hash": binding,
        "store_hash": store_hash,
        "rows_by_center": {key: list(value) for key, value in rows.items()},
        "case_ids_by_center": {key: list(value) for key, value in cases.items()},
        "cells": index_rows,
    }

    final_paths = (arrays_path, index_path, seal_path)
    _validate_final_member_paths(root, final_paths)
    existing = tuple(path.exists() for path in final_paths)
    # These are the only ordered crash boundaries.  An existing predecessor is
    # immutable: validate it against the reconstructed expectation and reuse
    # its bytes.  A seal without both predecessors is never repairable.
    if existing not in {
        (False, False, False),
        (True, False, False),
        (True, True, False),
    }:
        raise ProtocolError("Fixed-bank A1 final prediction trio is unsafe or sealed.")

    if existing[0]:
        _validate_existing_arrays(arrays_path, cells=cells, arrays=arrays)
    else:
        atomic_npz(arrays_path, **arrays)
    if existing[1]:
        _validate_existing_index(index_path, expected=index)
    else:
        atomic_json(index_path, index)

    seal = {
        "schema_version": "fixed_bank_a1_prediction_seal_v1",
        "status": "SEALED_ALL_810_LABEL_FREE_FIXED_BANK_A1_CELLS",
        "config_contract_hash": config_hash,
        "partition_hash": partition_hash,
        "source_stream_lock_hash": source_hash,
        "action_library_hash": library_hash,
        "target_cache_binding_hash": binding,
        "store_hash": store_hash,
        "arrays_sha256": sha256_file(arrays_path),
        "index_sha256": sha256_file(index_path),
        "cell_count": len(cells),
        "task_count": EXPECTED_TASK_COUNT,
        "action_count_per_target": ACTION_COUNT_PER_TARGET,
        "labels_opened": False,
        "target_expert_used": False,
        "exact_nine_seed_selection_used": False,
        "tf32_enabled": False,
        "amp_enabled": False,
    }
    atomic_json(
        seal_path,
        {**seal, "global_prediction_seal_hash": stable_hash(seal)},
    )


def _validate_final_member_paths(root: Path, paths: Sequence[Path]) -> None:
    directories = (root, *(path.parent for path in paths))
    if any(path.is_symlink() for path in (*directories, *paths)):
        raise ProtocolError("Fixed-bank A1 final prediction trio contains a symlink.")
    if any(path.exists() and not path.is_dir() for path in directories):
        raise ProtocolError("Fixed-bank A1 final prediction directory is unsafe.")
    if any(path.exists() and not path.is_file() for path in paths):
        raise ProtocolError("Fixed-bank A1 final prediction member is not a file.")


def _validate_existing_arrays(
    path: Path,
    *,
    cells: Sequence[PredictionCell],
    arrays: Mapping[str, np.ndarray],
) -> None:
    expected_members = tuple(arrays)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != expected_members:
                raise ProtocolError(
                    "Existing fixed-bank A1 prediction array members drifted; "
                    "refusing repair."
                )
            for member, cell in zip(expected_members, cells, strict=True):
                observed = np.asarray(archive[member])
                expected = arrays[member]
                if (
                    observed.dtype != expected.dtype
                    or observed.shape != expected.shape
                    or sha256_array(observed) != cell.probability_sha256
                ):
                    raise ProtocolError(
                        "Existing fixed-bank A1 prediction array cell drifted; "
                        "refusing repair."
                    )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(
            "Existing fixed-bank A1 prediction array is unreadable; refusing repair."
        ) from exc


def _validate_existing_index(
    path: Path, *, expected: Mapping[str, object]
) -> None:
    if read_json(path) != dict(expected):
        raise ProtocolError(
            "Existing fixed-bank A1 prediction index drifted; refusing repair."
        )


def load_global_prediction_seal(
    root: Path,
    *,
    expected_config_hash: str,
    expected_partition_hash: str,
    expected_source_lock_hash: str,
    expected_action_library_hash: str,
    expected_target_cache_binding_hash: str,
) -> GlobalPredictionSeal:
    arrays_path = root / PREDICTION_ARRAY_MEMBER
    index_path = root / PREDICTION_INDEX_MEMBER
    seal_path = root / PREDICTION_SEAL_MEMBER
    present = tuple(path.is_file() for path in (arrays_path, index_path, seal_path))
    if not all(present):
        raise ProtocolError("Fixed-bank A1 final prediction trio is incomplete.")
    if any(path.is_symlink() for path in (arrays_path, index_path, seal_path)):
        raise ProtocolError("Fixed-bank A1 final prediction trio contains a symlink.")
    index = read_json(index_path)
    seal = read_json(seal_path)
    if (
        seal.get("global_prediction_seal_hash")
        != stable_hash(
            {
                key: value
                for key, value in seal.items()
                if key != "global_prediction_seal_hash"
            }
        )
        or seal.get("status")
        != "SEALED_ALL_810_LABEL_FREE_FIXED_BANK_A1_CELLS"
        or seal.get("config_contract_hash") != expected_config_hash
        or seal.get("partition_hash") != expected_partition_hash
        or seal.get("source_stream_lock_hash") != expected_source_lock_hash
        or seal.get("action_library_hash") != expected_action_library_hash
        or seal.get("target_cache_binding_hash")
        != expected_target_cache_binding_hash
        or seal.get("arrays_sha256") != sha256_file(arrays_path)
        or seal.get("index_sha256") != sha256_file(index_path)
        or seal.get("cell_count") != EXPECTED_CELL_COUNT
        or seal.get("labels_opened") is not False
    ):
        raise ProtocolError("Fixed-bank A1 global prediction seal drifted.")
    raw_cells = index.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != EXPECTED_CELL_COUNT:
        raise ProtocolError("Fixed-bank A1 prediction index coverage drifted.")
    cells: list[PredictionCell] = []
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected_members = tuple(f"cell_{ordinal:04d}" for ordinal in range(EXPECTED_CELL_COUNT))
        if tuple(archive.files) != expected_members:
            raise ProtocolError("Fixed-bank A1 final array members drifted.")
        for ordinal, raw in enumerate(raw_cells):
            if not isinstance(raw, Mapping) or raw.get("array_member") != expected_members[ordinal]:
                raise ProtocolError("Fixed-bank A1 prediction index row is malformed.")
            cells.append(
                PredictionCell(
                    target_center=str(raw["target_center"]),
                    action_id=str(raw["action_id"]),
                    training_seed=int(raw["training_seed"]),
                    generation_seed=int(raw["generation_seed"]),
                    probabilities=np.asarray(archive[str(raw["array_member"])]),
                    action_hash=str(raw["action_hash"]),
                    row_identity_hash=str(raw["row_identity_hash"]),
                    probability_sha256=str(raw["probability_sha256"]),
                    prediction_sha256=str(raw["prediction_sha256"]),
                    fit_provenance_hash=str(raw["fit_provenance_hash"]),
                )
            )
    store = PredictionStore(
        tuple(cells),
        {key: tuple(value) for key, value in index["rows_by_center"].items()},
        {key: tuple(value) for key, value in index["case_ids_by_center"].items()},
        str(index["source_stream_lock_hash"]),
        str(index["action_library_hash"]),
        str(index["target_cache_binding_hash"]),
        str(index["store_hash"]),
    )
    if store.store_hash != seal.get("store_hash"):
        raise ProtocolError("Fixed-bank A1 store and seal hashes differ.")
    return GlobalPredictionSeal(
        store,
        MappingProxyType(seal),
        arrays_path,
        index_path,
        seal_path,
    )


def compute_store_hash(
    cells: Sequence[PredictionCell],
    rows: Mapping[str, Sequence[str]],
    cases: Mapping[str, Sequence[str]],
    source_hash: str,
    library_hash: str,
    binding: str,
) -> str:
    return prediction_store_hash(
        cells, rows, cases, source_hash, library_hash, binding
    )


__all__ = (
    "cells_from_checkpoints",
    "compute_store_hash",
    "load_global_prediction_seal",
    "write_prediction_store",
)
