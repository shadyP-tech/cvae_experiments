"""Persistence and replay validation for sealed action predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npz,
    read_json,
    sha256_file,
)
from .hashing import canonical_hash as stable_hash
from .prediction_contracts import (
    EXPECTED_CELL_COUNT,
    EXPECTED_TASK_COUNT,
    GLOBAL_PREDICTION_SEAL_MEMBER,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    ActionPredictionStore,
    GlobalActionPredictionSeal,
    PredictionCell,
)


def _validate_archive_members(members: Sequence[str]) -> None:
    expected = tuple(f"cell_{ordinal:04d}" for ordinal in range(EXPECTED_CELL_COUNT))
    if tuple(members) != expected:
        raise ProtocolError(
            "Actionability probability archive members drifted from the canonical "
            "1458-cell index."
        )


def write_final_store(
    arrays_path: Path,
    index_path: Path,
    *,
    cells: Sequence[PredictionCell],
    rows_by_center: Mapping[str, Sequence[str]],
    case_ids_by_center: Mapping[str, Sequence[str]],
    config_contract_hash: str,
    partition_hash: str,
    source_stream_lock_hash: str,
    action_library_hash: str,
    target_cache_binding_hash: str,
    store_hash: str,
) -> None:
    array_members = {
        f"cell_{ordinal:04d}": cell.probabilities
        for ordinal, cell in enumerate(cells)
    }
    atomic_npz(arrays_path, **array_members)
    unhashed = {
        "schema_version": "midogpp_actionability_prediction_index_v1",
        "config_contract_hash": config_contract_hash,
        "partition_hash": partition_hash,
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "target_cache_binding_hash": target_cache_binding_hash,
        "prediction_store_hash": store_hash,
        "rows_by_center": {
            center: list(rows_by_center[center]) for center in CENTERS
        },
        "case_ids_by_center": {
            center: list(case_ids_by_center[center]) for center in CENTERS
        },
        "cells": [
            cell.index_payload(array_member=f"cell_{ordinal:04d}")
            for ordinal, cell in enumerate(cells)
        ],
        "cell_count": len(cells),
        "labels_consumed": False,
        "target_expert_used": False,
    }
    atomic_json(index_path, {**unhashed, "index_hash": stable_hash(unhashed)})


def write_global_prediction_seal(
    seal_path: Path,
    *,
    arrays_path: Path,
    index_path: Path,
    config_contract_hash: str,
    partition_hash: str,
    prediction_store_hash: str,
    source_stream_lock_hash: str,
    action_library_hash: str,
    target_cache_binding_hash: str,
) -> None:
    seal_unhashed = {
        "schema_version": "midogpp_actionability_global_prediction_seal_v1",
        "status": "SEALED_ALL_1458_LABEL_FREE_ACTIONABILITY_CELLS",
        "config_contract_hash": config_contract_hash,
        "partition_hash": partition_hash,
        "prediction_store_hash": prediction_store_hash,
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "target_cache_binding_hash": target_cache_binding_hash,
        "prediction_array_sha256": sha256_file(arrays_path),
        "prediction_index_sha256": sha256_file(index_path),
        "cell_count": EXPECTED_CELL_COUNT,
        "task_count": EXPECTED_TASK_COUNT,
        "physical_action_count_per_target": PHYSICAL_ACTION_COUNT_PER_TARGET,
        "labels_opened": False,
        "target_expert_used": False,
        "seed_selection_used": False,
        "a1_sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
    }
    atomic_json(
        seal_path,
        {
            **seal_unhashed,
            "global_prediction_seal_hash": stable_hash(seal_unhashed),
        },
    )


def cell_from_index(
    row: Mapping[str, object], archive: Mapping[str, np.ndarray]
) -> PredictionCell:
    member = str(row["array_member"])
    if member not in archive:
        raise ProtocolError("Actionability probability array member is absent.")
    return PredictionCell(
        target_center=str(row["target_center"]),
        action_id=str(row["action_id"]),
        action_hash=str(row["action_hash"]),
        training_seed=int(row["training_seed"]),
        generation_seed=int(row["generation_seed"]),
        row_identity_hash=str(row["row_identity_hash"]),
        probabilities=np.asarray(archive[member], dtype=np.float32),
        probability_sha256=str(row["probability_sha256"]),
        predictions_sha256=str(row["predictions_sha256"]),
        composition_hash=str(row["composition_hash"]),
        scaler_state_hash=str(row["scaler_state_hash"]),
        fit_provenance_hash=str(row["fit_provenance_hash"]),
    )


def load_global_action_prediction_seal(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_lock_hash: str | None = None,
    expected_partition_hash: str | None = None,
    expected_action_library_hash: str | None = None,
    expected_target_cache_binding_hash: str | None = None,
) -> GlobalActionPredictionSeal:
    arrays_path = root / PREDICTION_ARRAY_MEMBER
    index_path = root / PREDICTION_INDEX_MEMBER
    seal_path = root / GLOBAL_PREDICTION_SEAL_MEMBER
    index = read_json(index_path)
    seal = read_json(seal_path)
    index_unhashed = {
        key: value for key, value in index.items() if key != "index_hash"
    }
    if (
        index.get("index_hash") != stable_hash(index_unhashed)
        or index.get("schema_version")
        != "midogpp_actionability_prediction_index_v1"
        or index.get("cell_count") != EXPECTED_CELL_COUNT
        or index.get("labels_consumed") is not False
        or index.get("target_expert_used") is not False
        or seal.get("prediction_array_sha256") != sha256_file(arrays_path)
        or seal.get("prediction_index_sha256") != sha256_file(index_path)
        or seal.get("prediction_store_hash") != index.get("prediction_store_hash")
        or seal.get("source_stream_lock_hash") != index.get("source_stream_lock_hash")
        or seal.get("action_library_hash") != index.get("action_library_hash")
        or seal.get("target_cache_binding_hash")
        != index.get("target_cache_binding_hash")
        or seal.get("partition_hash") != index.get("partition_hash")
        or (
            expected_config_hash is not None
            and seal.get("config_contract_hash") != expected_config_hash
        )
        or (
            expected_source_lock_hash is not None
            and seal.get("source_stream_lock_hash") != expected_source_lock_hash
        )
        or (
            expected_partition_hash is not None
            and seal.get("partition_hash") != expected_partition_hash
        )
        or (
            expected_action_library_hash is not None
            and seal.get("action_library_hash") != expected_action_library_hash
        )
        or (
            expected_target_cache_binding_hash is not None
            and seal.get("target_cache_binding_hash")
            != expected_target_cache_binding_hash
        )
    ):
        raise ProtocolError("Actionability prediction seal lineage drifted.")
    raw_cells = index.get("cells")
    raw_rows = index.get("rows_by_center")
    raw_cases = index.get("case_ids_by_center")
    if (
        not isinstance(raw_cells, list)
        or not isinstance(raw_rows, Mapping)
        or not isinstance(raw_cases, Mapping)
    ):
        raise ProtocolError("Actionability prediction index is malformed.")
    try:
        archive = np.load(arrays_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Actionability probability archive is unreadable.") from exc
    with archive:
        _validate_archive_members(archive.files)
        cells = tuple(
            cell_from_index(row, archive)
            for row in raw_cells
            if isinstance(row, Mapping)
        )
    if len(cells) != len(raw_cells):
        raise ProtocolError("Actionability prediction index contains malformed cells.")
    try:
        rows_by_center = {
            str(center): tuple(str(value) for value in raw_rows[center])
            for center in CENTERS
        }
        case_ids_by_center = {
            str(center): tuple(str(value) for value in raw_cases[center])
            for center in CENTERS
        }
    except (KeyError, TypeError) as exc:
        raise ProtocolError("Actionability prediction identity maps drifted.") from exc
    store = ActionPredictionStore(
        cells=cells,
        rows_by_center=rows_by_center,
        case_ids_by_center=case_ids_by_center,
        source_stream_lock_hash=str(index["source_stream_lock_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        target_cache_binding_hash=str(index["target_cache_binding_hash"]),
        store_hash=str(index["prediction_store_hash"]),
    )
    return GlobalActionPredictionSeal(
        store=store,
        seal_payload=seal,
        arrays_path=arrays_path,
        index_path=index_path,
        seal_path=seal_path,
    )


__all__ = (
    "cell_from_index",
    "load_global_action_prediction_seal",
    "write_final_store",
    "write_global_prediction_seal",
)
