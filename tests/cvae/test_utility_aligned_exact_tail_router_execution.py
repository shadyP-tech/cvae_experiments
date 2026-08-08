from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.config import (
    CLASSIFIER,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.actions import (
    FrozenExactTailActionLibrary,
    build_inner_exact_tail_action_library,
    build_inner_exact_tail_actions,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.artifact_io import (
    atomic_json,
    read_json,
    sha256_file,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_target_action_ids,
    inner_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.development_label_access import (
    open_globally_sealed_development_labels,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.development_prediction_contracts import (
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    CoarseDevelopmentTask,
    PredictionCheckpointRecord,
    PredictionWorkerInput,
    SourceSlice,
    action_library_for,
    expected_coarse_task_keys,
    expected_prediction_keys,
    expected_utility_keys,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.development_prediction_store import (
    array_sha256,
    atomic_save_npz,
    load_prediction_checkpoint,
    write_prediction_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.development_prediction_worker import (
    compose_exact_tail_action,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.development_seal import (
    DevelopmentPredictionCapability,
    GlobalDevelopmentPredictionSeal,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.input_contracts import (
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_contracts import (
    EXPECTED_COMPONENT_RECORD_COUNT,
    SOURCE_ROWS_PER_CLASS,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_planning import (
    write_support_scratch,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.target_prediction_contracts import (
    TARGET_PREDICTION_CACHE_MEMBER,
    TARGET_PREDICTION_INDEX_COLUMNS,
    TARGET_PREDICTION_INDEX_MEMBER,
    TargetPredictionCell,
    TargetPredictionStore,
    array_sha256 as target_array_sha256,
    canonical_target_cell_keys,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.target_prediction_store import (
    read_target_prediction_store,
    write_target_prediction_store,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_exact_tail_geometry_uses_270_row_source_prefix() -> None:
    actions = action_library_for(outer_target="0", query_center="1")
    canonical = build_inner_exact_tail_actions("0", "1")
    sources = {
        source: np.arange(
            2 * SOURCE_ROWS_PER_CLASS * 4, dtype=np.float32
        ).reshape(2 * SOURCE_ROWS_PER_CLASS, 4)
        + 10_000 * ordinal
        for ordinal, source in enumerate(actions[0].source_order)
    }

    base_embeddings, base_labels = compose_exact_tail_action(sources, actions[0])
    tail_embeddings, tail_labels = compose_exact_tail_action(sources, actions[1])

    assert SOURCE_ROWS_PER_CLASS == 270
    assert EXPECTED_COMPONENT_RECORD_COUNT == 9 * 8 * 3 == 216
    assert tuple(action.action_id for action in actions) == tuple(
        action.action_id for action in canonical
    )
    assert tuple(action.action_hash for action in actions) == tuple(
        action.action_hash for action in canonical
    )
    assert len(build_inner_exact_tail_action_library().action_library_hash) == 64
    assert actions[0].total_per_class == 7 * 144 == 1008
    assert actions[1].total_per_class == 7 * 144 + 126 == 1134
    assert actions[1].counts_per_class[actions[1].selected_source] == 270
    assert base_embeddings.shape == (2 * 1008, 4)
    assert tail_embeddings.shape == (2 * 1134, 4)
    assert np.bincount(base_labels).tolist() == [1008, 1008]
    assert np.bincount(tail_labels).tolist() == [1134, 1134]


def test_development_grid_enforces_strict_H_q_e_exclusion() -> None:
    coarse = expected_coarse_task_keys()
    predictions = expected_prediction_keys()
    utilities = expected_utility_keys()

    assert len(coarse) == EXPECTED_COARSE_TASK_COUNT == 648
    assert len(predictions) == EXPECTED_PREDICTION_CELL_COUNT == 5184
    assert len(utilities) == EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT == 4536
    assert all(outer != query for outer, query, _, _ in coarse)
    assert all(
        source not in {outer, query}
        and source in inner_candidate_sources(outer, query)
        for outer, query, source, _, _ in utilities
    )


def test_prediction_checkpoint_resumes_and_rejects_tampered_member(
    tmp_path: Path,
) -> None:
    item = _worker_input(tmp_path)
    action_count = len(item.task.action_ids)
    predictions = np.zeros((action_count, 2), dtype=np.uint8)
    predictions[:, 1] = 1
    probabilities = np.tile(
        np.asarray([[0.25, 0.75]], dtype=np.float32), (action_count, 1)
    )
    prediction_hashes = {
        action: array_sha256(predictions[index])
        for index, action in enumerate(item.task.action_ids)
    }
    probability_hashes = {
        action: array_sha256(probabilities[index])
        for index, action in enumerate(item.task.action_ids)
    }
    semantic_hashes = {action: f"{index + 1:x}" * 64 for index, action in enumerate(item.task.action_ids)}
    scaler_hashes = {action: f"{index + 9:x}"[-1] * 64 for index, action in enumerate(item.task.action_ids)}

    written = write_prediction_checkpoint(
        item,
        predictions=predictions,
        probabilities=probabilities,
        classifier_config_hash=CLASSIFIER.config_hash,
        action_prediction_sha256=prediction_hashes,
        action_probability_sha256=probability_hashes,
        action_composition_sha256=semantic_hashes,
        action_scaler_state_hash=scaler_hashes,
    )
    resumed = load_prediction_checkpoint(item)
    assert resumed is not None
    assert resumed.checkpoint_hash == written.checkpoint_hash

    tampered = predictions.copy()
    tampered[0, 0] = 1
    atomic_save_npz(
        Path(item.checkpoint_npz_path),
        predictions=tampered,
        probabilities=probabilities,
    )
    with pytest.raises(ProtocolError, match="checkpoint binding"):
        load_prediction_checkpoint(item)


def test_spawn_contracts_rebuild_immutable_hash_maps(tmp_path: Path) -> None:
    item = _worker_input(tmp_path)
    hashes = {action: "a" * 64 for action in item.task.action_ids}
    record = PredictionCheckpointRecord(
        task=item.task,
        checkpoint_json_path=item.checkpoint_json_path,
        checkpoint_npz_path=item.checkpoint_npz_path,
        checkpoint_file_sha256="b" * 64,
        checkpoint_hash="c" * 64,
        evaluation_row_count=2,
        action_prediction_sha256=hashes,
        action_probability_sha256=hashes,
        action_composition_sha256=hashes,
        action_scaler_state_hash=hashes,
    )

    observed_item = pickle.loads(pickle.dumps(item))
    observed_record = pickle.loads(pickle.dumps(record))

    assert isinstance(observed_item.classifier_payload, MappingProxyType)
    assert isinstance(observed_record.action_prediction_sha256, MappingProxyType)
    assert observed_item.task == item.task


def test_support_scratch_contains_embeddings_and_no_label_vector(
    tmp_path: Path,
) -> None:
    rows_by_center = {}
    rows = []
    for center_ordinal, center in enumerate(CENTERS):
        center_rows = []
        for local in range(2):
            ordinal = 2 * center_ordinal + local
            row = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                sample_id=f"sample::{center}::{local}",
                case_id=f"case::{center}::{local}",
                center=center,
                partition_role="support",
            )
            rows.append(row)
            center_rows.append(row)
        rows_by_center[center] = tuple(center_rows)
    embeddings = np.arange(len(rows) * 5, dtype=np.float32).reshape(len(rows), 5)

    class Frame:
        cache_binding_hash = "cache-binding"

        def embeddings_for(self, selected):
            return embeddings[[row.row_ordinal for row in selected]]

    partitions = SimpleNamespace(
        support_rows_by_center=rows_by_center,
        lock_hash="partition-lock",
    )
    payload = write_support_scratch(
        tmp_path / "support.npy",
        tmp_path / "support.json",
        frame=Frame(),
        partitions=partitions,
    )

    assert np.load(tmp_path / "support.npy", allow_pickle=False).shape == (18, 5)
    assert payload["labels_consumed"] is False
    assert payload["evaluation_embeddings_consumed"] is False
    assert _find_label_like_values(payload) == [False]


def test_label_access_requires_a_durably_persisted_global_seal(
    tmp_path: Path,
) -> None:
    # __new__ is intentional: the missing persisted seal must fail before any
    # manifest or label-bearing object is examined.
    unpersisted = GlobalDevelopmentPredictionSeal.__new__(
        GlobalDevelopmentPredictionSeal
    )
    capability = DevelopmentPredictionCapability(
        store=None,  # type: ignore[arg-type]
        seal=unpersisted,
        seal_path=tmp_path / "missing-seal.json",
        prediction_index_path=tmp_path / "missing-index.json",
        prediction_arrays_path=tmp_path / "missing-arrays.npz",
    )
    with pytest.raises(ProtocolError, match="Cannot read Stage-90 JSON"):
        open_globally_sealed_development_labels(
            tmp_path / "never-opened-manifest.csv",
            SimpleNamespace(),
            capability=capability,
        )


def test_target_store_rejects_action_hash_not_bound_to_frozen_library(
    tmp_path: Path,
) -> None:
    library = _test_target_library()
    store = _test_target_store(library)
    write_target_prediction_store(tmp_path, store)
    read_target_prediction_store(
        tmp_path,
        library=library,
        source_cache_lock_hash="source-lock",
        case_fold_lock_hash="fold-lock",
    )

    index_path = tmp_path / TARGET_PREDICTION_INDEX_MEMBER
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["action_hash"] = "f" * 64
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TARGET_PREDICTION_INDEX_COLUMNS,
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = read_json(tmp_path / TARGET_PREDICTION_CACHE_MEMBER)
    metadata["prediction_index_sha256"] = sha256_file(index_path)
    atomic_json(tmp_path / TARGET_PREDICTION_CACHE_MEMBER, metadata)

    with pytest.raises(ProtocolError, match="index binding"):
        read_target_prediction_store(
            tmp_path,
            library=library,
            source_cache_lock_hash="source-lock",
            case_fold_lock_hash="fold-lock",
        )


def _worker_input(tmp_path: Path) -> PredictionWorkerInput:
    outer, query, training_seed, generation_seed = expected_coarse_task_keys()[0]
    actions = action_library_for(outer_target=outer, query_center=query)
    candidates = inner_candidate_sources(outer, query)
    task = CoarseDevelopmentTask(
        task_ordinal=0,
        outer_target=outer,
        query_center=query,
        training_seed=training_seed,
        generation_seed=generation_seed,
        candidate_sources=candidates,
        action_ids=tuple(action.action_id for action in actions),
        task_hash="1" * 64,
    )
    return PredictionWorkerInput(
        task=task,
        source_array_path=str(tmp_path / "source.npy"),
        source_slices=tuple(
            SourceSlice(
                source_center=source,
                block_ordinal=index,
                stream_id=f"stream::{source}",
                expert_lock_hash="2" * 64,
                output_sha256="3" * 64,
            )
            for index, source in enumerate(candidates)
        ),
        source_cache_lock_hash="4" * 64,
        evaluation_array_path=str(tmp_path / "evaluation.npy"),
        evaluation_array_sha256="5" * 64,
        evaluation_row_ids=("sample::0", "sample::1"),
        evaluation_row_identity_hash="6" * 64,
        support_partition_hash="7" * 64,
        partition_lock_hash="8" * 64,
        generation_lock_hash="9" * 64,
        config_contract_hash="a" * 64,
        classifier_payload=CLASSIFIER.to_payload(),
        checkpoint_json_path=str(tmp_path / "checkpoint.json"),
        checkpoint_npz_path=str(tmp_path / "checkpoint.npz"),
    )


def _test_target_library() -> FrozenExactTailActionLibrary:
    library = object.__new__(FrozenExactTailActionLibrary)
    actions = {
        target: tuple(
            SimpleNamespace(
                action_id=action_id,
                action_hash=stable_hash([target, action_id]),
            )
            for action_id in expected_target_action_ids(target)
        )
        for target in CENTERS
    }
    object.__setattr__(library, "actions_by_target", MappingProxyType(actions))
    object.__setattr__(
        library,
        "plan_hashes_by_target",
        MappingProxyType({target: stable_hash(["plan", target]) for target in CENTERS}),
    )
    object.__setattr__(library, "plan_set_hash", stable_hash("plans"))
    object.__setattr__(library, "action_library_hash", stable_hash("actions"))
    return library


def _test_target_store(
    library: FrozenExactTailActionLibrary,
) -> TargetPredictionStore:
    cells = tuple(
        TargetPredictionCell(
            target_center=target,
            action_id=action_id,
            action_hash=library.action(target, action_id).action_hash,
            training_seed=training_seed,
            generation_seed=generation_seed,
            evaluation_row_identity_hash=stable_hash(["rows", target]),
            predictions=np.asarray([0], dtype=np.uint8),
            probabilities=np.asarray([0.5], dtype=np.float32),
            composition_sha256=stable_hash(["composition", *key]),
            scaler_state_hash=stable_hash(["scaler", *key]),
            aliased_fit=False,
        )
        for key in canonical_target_cell_keys(library)
        for target, action_id, training_seed, generation_seed in (key,)
    )
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_target_prediction_store_v1",
        "action_library_hash": library.action_library_hash,
        "source_cache_lock_hash": "source-lock",
        "case_fold_lock_hash": "fold-lock",
        "cell_count": len(cells),
        "cell_keys": [list(cell.key) for cell in cells],
        "cell_action_hashes": [cell.action_hash for cell in cells],
        "cell_prediction_hashes": [target_array_sha256(cell.predictions) for cell in cells],
        "cell_probability_hashes": [target_array_sha256(cell.probabilities) for cell in cells],
        "composition_hashes": [cell.composition_sha256 for cell in cells],
        "unique_classifier_fit_count": len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "labels_stored": False,
    }
    return TargetPredictionStore(
        cells=cells,
        action_library_hash=library.action_library_hash,
        source_cache_lock_hash="source-lock",
        case_fold_lock_hash="fold-lock",
        unique_classifier_fit_count=int(unhashed["unique_classifier_fit_count"]),
        store_hash=stable_hash(unhashed),
    )


def _find_label_like_values(value: object) -> list[object]:
    observed = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if "label" in str(key).lower():
                observed.append(nested)
            observed.extend(_find_label_like_values(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            observed.extend(_find_label_like_values(nested))
    return observed
