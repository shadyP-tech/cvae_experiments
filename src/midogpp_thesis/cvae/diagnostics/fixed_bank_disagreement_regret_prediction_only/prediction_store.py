"""Durable classifier-bank and source/test prediction persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_file
from .actions import action_library_payload
from .constants import (
    ACTION_LIBRARY_MEMBER,
    CENTERS,
    CLASSIFIER_COEFFICIENT_MEMBER,
    CLASSIFIER_INDEX_MEMBER,
    CLASSIFIER_INTERCEPT_MEMBER,
    CLASSIFIER_MEAN_MEMBER,
    CLASSIFIER_SCALE_MEMBER,
    CLASSIFIER_SEAL_MEMBER,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_TASK_COUNT,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_SEAL_MEMBER,
    TEST_ARRAY_MEMBER,
    TEST_INDEX_MEMBER,
    TEST_SEAL_MEMBER,
)
from .hashing import canonical_hash
from .input_contracts import TestInferenceAdmission
from .prediction_contracts import (
    ActionClassifierBank,
    ActionPredictionStore,
    ClassifierBankCell,
    GlobalSourcePredictionSeal,
    GlobalTestPredictionSeal,
    PredictionCell,
    canonical_cell_keys,
    prediction_store_hash,
)


def write_action_library(root: Path) -> Mapping[str, object]:
    payload = action_library_payload()
    path = root / ACTION_LIBRARY_MEMBER
    if path.is_file():
        observed = read_json(path)
        if observed != payload:
            raise ProtocolError("Prediction-only persisted action library drifted.")
    else:
        atomic_json(path, payload)
    return payload


def write_classifier_bank_manifests(
    root: Path,
    *,
    cells: Sequence[ClassifierBankCell],
    config_contract_hash: str,
    source_stream_lock_hash: str,
    action_library_hash: str,
    source_cache_binding_hash: str,
) -> ActionClassifierBank:
    ordered = tuple(cells)
    if tuple(cell.key for cell in ordered) != canonical_cell_keys():
        raise ProtocolError("Prediction-only classifier cells are not canonical.")
    paths = {
        "scaler_mean": root / CLASSIFIER_MEAN_MEMBER,
        "scaler_scale": root / CLASSIFIER_SCALE_MEMBER,
        "coefficient": root / CLASSIFIER_COEFFICIENT_MEMBER,
        "intercept": root / CLASSIFIER_INTERCEPT_MEMBER,
    }
    if any(not path.is_file() for path in paths.values()):
        raise ProtocolError("Prediction-only classifier arrays are incomplete.")
    file_hashes = {name: sha256_file(path) for name, path in paths.items()}
    bank_unhashed = {
        "schema_version": "midogpp_prediction_only_action_classifier_bank_v1",
        "config_contract_hash": config_contract_hash,
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "fit_count": len(ordered),
        "cells": [cell.to_payload() for cell in ordered],
        "parameter_file_sha256": file_hashes,
        "fit_data": "frozen_generated_source_streams_only",
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
    }
    bank_hash = canonical_hash(bank_unhashed)
    index = {**bank_unhashed, "classifier_bank_hash": bank_hash}
    index_path = root / CLASSIFIER_INDEX_MEMBER
    atomic_json(index_path, index)
    seal_unhashed = {
        "schema_version": "midogpp_prediction_only_action_classifier_bank_seal_v1",
        "status": "SEALED_1458_SOURCE_ONLY_ACTION_CLASSIFIERS",
        "config_contract_hash": config_contract_hash,
        "classifier_bank_hash": bank_hash,
        "classifier_bank_index_sha256": sha256_file(index_path),
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "scaler_mean_file_sha256": file_hashes["scaler_mean"],
        "scaler_scale_file_sha256": file_hashes["scaler_scale"],
        "coefficient_file_sha256": file_hashes["coefficient"],
        "intercept_file_sha256": file_hashes["intercept"],
        "fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "task_count": EXPECTED_TASK_COUNT,
        "physical_action_count_per_task": PHYSICAL_ACTION_COUNT_PER_TARGET,
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
        "classifier_refit_required_for_test": False,
        "float64_frozen_parameter_arrays": True,
    }
    seal_path = root / CLASSIFIER_SEAL_MEMBER
    atomic_json(
        seal_path,
        {
            **seal_unhashed,
            "classifier_bank_seal_hash": canonical_hash(seal_unhashed),
        },
    )
    return load_action_classifier_bank(
        root,
        expected_config_hash=config_contract_hash,
        expected_source_stream_lock_hash=source_stream_lock_hash,
        expected_source_cache_binding_hash=source_cache_binding_hash,
    )


def load_action_classifier_bank(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_stream_lock_hash: str | None = None,
    expected_source_cache_binding_hash: str | None = None,
) -> ActionClassifierBank:
    index_path = root / CLASSIFIER_INDEX_MEMBER
    seal_path = root / CLASSIFIER_SEAL_MEMBER
    index = read_json(index_path)
    seal = read_json(seal_path)
    raw_cells = index.get("cells")
    if not isinstance(raw_cells, list):
        raise ProtocolError("Prediction-only classifier index cells are absent.")
    cells = tuple(_classifier_cell(row) for row in raw_cells if isinstance(row, Mapping))
    bank_unhashed = {
        key: value for key, value in index.items() if key != "classifier_bank_hash"
    }
    if (
        len(cells) != len(raw_cells)
        or index.get("classifier_bank_hash") != canonical_hash(bank_unhashed)
        or seal.get("classifier_bank_index_sha256") != sha256_file(index_path)
        or seal.get("classifier_bank_hash") != index.get("classifier_bank_hash")
        or seal.get("source_stream_lock_hash") != index.get("source_stream_lock_hash")
        or seal.get("action_library_hash") != index.get("action_library_hash")
        or seal.get("source_cache_binding_hash") != index.get("source_cache_binding_hash")
        or (
            expected_config_hash is not None
            and seal.get("config_contract_hash") != expected_config_hash
        )
        or (
            expected_source_stream_lock_hash is not None
            and seal.get("source_stream_lock_hash")
            != expected_source_stream_lock_hash
        )
        or (
            expected_source_cache_binding_hash is not None
            and seal.get("source_cache_binding_hash")
            != expected_source_cache_binding_hash
        )
    ):
        raise ProtocolError("Prediction-only classifier-bank lineage drifted.")
    return ActionClassifierBank(
        root=root,
        cells=cells,
        source_stream_lock_hash=str(index["source_stream_lock_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        source_cache_binding_hash=str(index["source_cache_binding_hash"]),
        config_contract_hash=str(index["config_contract_hash"]),
        bank_hash=str(index["classifier_bank_hash"]),
        seal_payload=seal,
    )


def write_prediction_store(
    root: Path,
    *,
    frame_role: str,
    cells: Sequence[PredictionCell],
    rows_by_outer_target: Mapping[str, Sequence[str]],
    case_ids_by_outer_target: Mapping[str, Sequence[str]],
    query_ids_by_outer_target: Mapping[str, Sequence[str]],
    frame_cache_binding_hash: str,
    action_library_hash: str,
    action_classifier_bank_seal_hash: str,
    config_contract_hash: str,
) -> ActionPredictionStore:
    array_member, index_member = _store_members(frame_role)
    ordered = tuple(cells)
    store_hash = prediction_store_hash(
        frame_role,
        ordered,
        rows_by_outer_target=rows_by_outer_target,
        case_ids_by_outer_target=case_ids_by_outer_target,
        query_ids_by_outer_target=query_ids_by_outer_target,
        frame_cache_binding_hash=frame_cache_binding_hash,
        action_library_hash=action_library_hash,
        action_classifier_bank_seal_hash=action_classifier_bank_seal_hash,
    )
    arrays_path = root / array_member
    index_path = root / index_member
    atomic_npz(
        arrays_path,
        **{
            f"cell_{ordinal:04d}": cell.probabilities
            for ordinal, cell in enumerate(ordered)
        },
    )
    unhashed = {
        "schema_version": "midogpp_prediction_only_probability_index_v1",
        "frame_role": frame_role,
        "config_contract_hash": config_contract_hash,
        "frame_cache_binding_hash": frame_cache_binding_hash,
        "action_library_hash": action_library_hash,
        "action_classifier_bank_seal_hash": action_classifier_bank_seal_hash,
        "prediction_store_hash": store_hash,
        "rows_by_outer_target": {
            target: list(rows_by_outer_target[target]) for target in CENTERS
        },
        "case_ids_by_outer_target": {
            target: list(case_ids_by_outer_target[target]) for target in CENTERS
        },
        "query_ids_by_outer_target": {
            target: list(query_ids_by_outer_target[target]) for target in CENTERS
        },
        "cells": [
            cell.index_payload(array_member=f"cell_{ordinal:04d}")
            for ordinal, cell in enumerate(ordered)
        ],
        "cell_count": len(ordered),
        "labels_consumed": False,
        "target_labels_available": False,
    }
    atomic_json(index_path, {**unhashed, "index_hash": canonical_hash(unhashed)})
    return load_prediction_store(
        root,
        frame_role=frame_role,
        expected_frame_cache_binding_hash=frame_cache_binding_hash,
        expected_classifier_bank_seal_hash=action_classifier_bank_seal_hash,
    )


def load_prediction_store(
    root: Path,
    *,
    frame_role: str,
    expected_frame_cache_binding_hash: str | None = None,
    expected_classifier_bank_seal_hash: str | None = None,
) -> ActionPredictionStore:
    array_member, index_member = _store_members(frame_role)
    arrays_path = root / array_member
    index = read_json(root / index_member)
    unhashed = {key: value for key, value in index.items() if key != "index_hash"}
    raw_cells = index.get("cells")
    raw_rows = index.get("rows_by_outer_target")
    raw_cases = index.get("case_ids_by_outer_target")
    raw_queries = index.get("query_ids_by_outer_target")
    if (
        index.get("index_hash") != canonical_hash(unhashed)
        or index.get("frame_role") != frame_role
        or index.get("cell_count") != EXPECTED_CLASSIFIER_FIT_COUNT
        or index.get("labels_consumed") is not False
        or index.get("target_labels_available") is not False
        or not isinstance(raw_cells, list)
        or not isinstance(raw_rows, Mapping)
        or not isinstance(raw_cases, Mapping)
        or not isinstance(raw_queries, Mapping)
        or (
            expected_frame_cache_binding_hash is not None
            and index.get("frame_cache_binding_hash")
            != expected_frame_cache_binding_hash
        )
        or (
            expected_classifier_bank_seal_hash is not None
            and index.get("action_classifier_bank_seal_hash")
            != expected_classifier_bank_seal_hash
        )
    ):
        raise ProtocolError("Prediction-only probability index drifted.")
    try:
        archive = np.load(arrays_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Prediction-only probability archive is unreadable.") from exc
    with archive:
        expected_members = tuple(
            f"cell_{ordinal:04d}" for ordinal in range(EXPECTED_CLASSIFIER_FIT_COUNT)
        )
        if tuple(archive.files) != expected_members:
            raise ProtocolError("Prediction-only probability members drifted.")
        cells = tuple(
            _prediction_cell(row, archive)
            for row in raw_cells
            if isinstance(row, Mapping)
        )
    if len(cells) != len(raw_cells):
        raise ProtocolError("Prediction-only probability index contains malformed cells.")
    try:
        rows = {target: tuple(str(value) for value in raw_rows[target]) for target in CENTERS}
        cases = {target: tuple(str(value) for value in raw_cases[target]) for target in CENTERS}
        queries = {target: tuple(str(value) for value in raw_queries[target]) for target in CENTERS}
    except (KeyError, TypeError) as exc:
        raise ProtocolError("Prediction-only probability identity maps drifted.") from exc
    return ActionPredictionStore(
        frame_role=frame_role,
        cells=cells,
        rows_by_outer_target=rows,
        case_ids_by_outer_target=cases,
        query_ids_by_outer_target=queries,
        frame_cache_binding_hash=str(index["frame_cache_binding_hash"]),
        action_library_hash=str(index["action_library_hash"]),
        action_classifier_bank_seal_hash=str(
            index["action_classifier_bank_seal_hash"]
        ),
        store_hash=str(index["prediction_store_hash"]),
    )


def write_source_prediction_seal(
    root: Path,
    *,
    classifier_bank: ActionClassifierBank,
    source_store: ActionPredictionStore,
    config_contract_hash: str,
) -> GlobalSourcePredictionSeal:
    arrays_path = root / SOURCE_ARRAY_MEMBER
    index_path = root / SOURCE_INDEX_MEMBER
    seal_path = root / SOURCE_SEAL_MEMBER
    unhashed = {
        "schema_version": "midogpp_prediction_only_source_prediction_seal_v1",
        "status": "SEALED_1458_SOURCE_ACTION_FITS_AND_PREDICTIONS",
        "config_contract_hash": config_contract_hash,
        "classifier_bank_seal_hash": classifier_bank.seal_hash,
        "source_prediction_store_hash": source_store.store_hash,
        "source_prediction_array_sha256": sha256_file(arrays_path),
        "source_prediction_index_sha256": sha256_file(index_path),
        "fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "source_prediction_cell_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
        "regret_model_bank_fitted": False,
    }
    atomic_json(seal_path, {**unhashed, "source_prediction_seal_hash": canonical_hash(unhashed)})
    return load_global_source_prediction_seal(
        root,
        expected_config_hash=config_contract_hash,
        expected_source_cache_binding_hash=source_store.frame_cache_binding_hash,
    )


def load_global_source_prediction_seal(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_cache_binding_hash: str | None = None,
) -> GlobalSourcePredictionSeal:
    classifier_bank = load_action_classifier_bank(
        root,
        expected_config_hash=expected_config_hash,
        expected_source_cache_binding_hash=expected_source_cache_binding_hash,
    )
    source_store = load_prediction_store(
        root,
        frame_role="source",
        expected_frame_cache_binding_hash=expected_source_cache_binding_hash,
        expected_classifier_bank_seal_hash=classifier_bank.seal_hash,
    )
    arrays_path = root / SOURCE_ARRAY_MEMBER
    index_path = root / SOURCE_INDEX_MEMBER
    seal_path = root / SOURCE_SEAL_MEMBER
    seal = read_json(seal_path)
    if (
        seal.get("source_prediction_array_sha256") != sha256_file(arrays_path)
        or seal.get("source_prediction_index_sha256") != sha256_file(index_path)
    ):
        raise ProtocolError("Prediction-only source prediction bytes drifted.")
    return GlobalSourcePredictionSeal(
        classifier_bank=classifier_bank,
        source_store=source_store,
        seal_payload=seal,
        arrays_path=arrays_path,
        index_path=index_path,
        seal_path=seal_path,
    )


def write_test_prediction_seal(
    root: Path,
    *,
    classifier_bank: ActionClassifierBank,
    test_store: ActionPredictionStore,
    admission: TestInferenceAdmission,
    config_contract_hash: str,
) -> GlobalTestPredictionSeal:
    arrays_path = root / TEST_ARRAY_MEMBER
    index_path = root / TEST_INDEX_MEMBER
    seal_path = root / TEST_SEAL_MEMBER
    unhashed = {
        "schema_version": "midogpp_prediction_only_test_prediction_seal_v1",
        "status": "SEALED_WHOLE_TEST_LABEL_FREE_INFERENCE",
        "config_contract_hash": config_contract_hash,
        "classifier_bank_seal_hash": classifier_bank.seal_hash,
        "test_prediction_store_hash": test_store.store_hash,
        "test_prediction_array_sha256": sha256_file(arrays_path),
        "test_prediction_index_sha256": sha256_file(index_path),
        "test_inference_admission_hash": admission.admission_hash,
        "source_prediction_seal_hash": admission.source_prediction_seal_hash,
        "regret_model_bank_seal_hash": admission.regret_model_bank_seal_hash,
        "fit_count_during_test_phase": 0,
        "target_labels_available": False,
        "target_scoring_permitted": False,
        "whole_consumed_test_row_count": sum(
            len(test_store.rows_by_outer_target[target]) for target in CENTERS
        ),
    }
    atomic_json(seal_path, {**unhashed, "test_prediction_seal_hash": canonical_hash(unhashed)})
    return load_global_test_prediction_seal(
        root,
        admission=admission,
        expected_config_hash=config_contract_hash,
        expected_test_cache_binding_hash=test_store.frame_cache_binding_hash,
    )


def load_global_test_prediction_seal(
    root: Path,
    *,
    admission: TestInferenceAdmission,
    expected_config_hash: str | None = None,
    expected_test_cache_binding_hash: str | None = None,
) -> GlobalTestPredictionSeal:
    classifier_bank = load_action_classifier_bank(root, expected_config_hash=expected_config_hash)
    if classifier_bank.seal_hash != admission.action_classifier_bank_seal_hash:
        raise ProtocolError("Prediction-only test admission names another classifier bank.")
    test_store = load_prediction_store(
        root,
        frame_role="test",
        expected_frame_cache_binding_hash=expected_test_cache_binding_hash,
        expected_classifier_bank_seal_hash=classifier_bank.seal_hash,
    )
    arrays_path = root / TEST_ARRAY_MEMBER
    index_path = root / TEST_INDEX_MEMBER
    seal_path = root / TEST_SEAL_MEMBER
    seal = read_json(seal_path)
    if (
        seal.get("test_prediction_array_sha256") != sha256_file(arrays_path)
        or seal.get("test_prediction_index_sha256") != sha256_file(index_path)
    ):
        raise ProtocolError("Prediction-only test prediction bytes drifted.")
    return GlobalTestPredictionSeal(
        classifier_bank=classifier_bank,
        test_store=test_store,
        admission=admission,
        seal_payload=seal,
        arrays_path=arrays_path,
        index_path=index_path,
        seal_path=seal_path,
    )


def _classifier_cell(raw: Mapping[str, object]) -> ClassifierBankCell:
    return ClassifierBankCell(
        cell_ordinal=int(raw["cell_ordinal"]),
        target_center=str(raw["target_center"]),
        action_id=str(raw["action_id"]),
        action_hash=str(raw["action_hash"]),
        training_seed=int(raw["training_seed"]),
        generation_seed=int(raw["generation_seed"]),
        composition_hash=str(raw["composition_hash"]),
        scaler_state_hash=str(raw["scaler_state_hash"]),
        parameter_sha256=str(raw["parameter_sha256"]),
        fit_provenance_hash=str(raw["fit_provenance_hash"]),
        classifier_config_hash=str(raw["classifier_config_hash"]),
        n_iter=tuple(int(value) for value in raw["n_iter"]),  # type: ignore[arg-type]
        converged=bool(raw["converged"]),
    )


def _prediction_cell(
    raw: Mapping[str, object], archive: Mapping[str, np.ndarray]
) -> PredictionCell:
    member = str(raw.get("array_member", ""))
    if member not in archive:
        raise ProtocolError("Prediction-only probability member is absent.")
    return PredictionCell(
        frame_role=str(raw["frame_role"]),
        target_center=str(raw["target_center"]),
        action_id=str(raw["action_id"]),
        action_hash=str(raw["action_hash"]),
        training_seed=int(raw["training_seed"]),
        generation_seed=int(raw["generation_seed"]),
        row_identity_hash=str(raw["row_identity_hash"]),
        probabilities=np.asarray(archive[member], dtype=np.float32),
        probability_sha256=str(raw["probability_sha256"]),
        predictions_sha256=str(raw["predictions_sha256"]),
        classifier_parameter_sha256=str(raw["classifier_parameter_sha256"]),
    )


def _store_members(frame_role: str) -> tuple[str, str]:
    if frame_role == "source":
        return SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER
    if frame_role == "test":
        return TEST_ARRAY_MEMBER, TEST_INDEX_MEMBER
    raise ProtocolError("Prediction-only probability frame role is invalid.")


__all__ = (
    "load_action_classifier_bank",
    "load_global_source_prediction_seal",
    "load_global_test_prediction_seal",
    "load_prediction_store",
    "write_action_library",
    "write_classifier_bank_manifests",
    "write_prediction_store",
    "write_source_prediction_seal",
    "write_test_prediction_seal",
)
