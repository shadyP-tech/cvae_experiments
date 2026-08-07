from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router import (
    composition,
    prediction,
    prediction_store,
    prediction_worker,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.artifact_io import (
    atomic_write_json,
    read_json,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.contracts import (
    ARM_ROLES,
    CENTERS,
    CONTROL_ARM,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    GENERATION_SEEDS,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    ROUTED_ARM,
    TRAINING_SEEDS,
    candidate_sources,
    row_identity_hash,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.partitions import (
    CrossfitSurface,
    build_case_crossfit_surface,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.prediction import (
    CROSSFIT_PREDICTION_ARRAY_MEMBER,
    CROSSFIT_PREDICTION_INDEX_MEMBER,
    CrossfitPredictionStore,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.scoring import (
    score_case_crossfit_predictions,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.seals import (
    GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER,
    build_global_crossfit_prediction_seal,
    open_crossfit_evaluation_labels,
)
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.inputs import PartitionSurface
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.contracts import (
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.protocol import ProtocolError


EVALUATION_CASE_COUNTS = (3, 2, 4, 5, 3, 1, 2, 2, 4)


def _base_partitions() -> PartitionSurface:
    support: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    evaluation: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    ordinal = 0
    for target, case_count in zip(CENTERS, EVALUATION_CASE_COUNTS, strict=True):
        target_support = []
        for case_index in range(2):
            target_support.append(
                ValidationRowIdentity(
                    row_ordinal=ordinal,
                    manifest_row_index=ordinal,
                    sample_id=f"s_{target}_{case_index}",
                    case_id=f"support_{target}_{case_index}",
                    center=target,
                    partition_role="support",
                )
            )
            ordinal += 1
        target_evaluation = []
        for case_index in range(case_count):
            for class_index in range(2):
                target_evaluation.append(
                    ValidationRowIdentity(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        sample_id=f"e_{target}_{case_index}_{class_index}",
                        case_id=f"eval_{target}_{case_index}",
                        center=target,
                        partition_role="evaluation",
                    )
                )
                ordinal += 1
        support[target] = tuple(target_support)
        evaluation[target] = tuple(target_evaluation)
    return PartitionSurface(
        support_rows_by_center=support,
        evaluation_rows_by_center=evaluation,
        table_rows=(),
        lock_payload={"support_partition_lock_hash": "b" * 16},
    )


def _crossfit() -> CrossfitSurface:
    return build_case_crossfit_surface(
        _base_partitions(), config_contract_hash="a" * 16
    )


def _plans(crossfit: CrossfitSurface) -> SimpleNamespace:
    plans = {
        fold.fold_id: {
            "fold_id": fold.fold_id,
            "target_center": fold.target_center,
            "plan_hash": f"{fold.fold_ordinal:016x}",
        }
        for fold in crossfit.folds
    }
    return SimpleNamespace(plans_by_fold=plans, lock_hash="c" * 16)


def _prediction_store(
    crossfit: CrossfitSurface,
) -> tuple[CrossfitPredictionStore, dict[str, int]]:
    labels = {
        row.sample_id: int(row.sample_id.rsplit("_", 1)[-1])
        for fold in crossfit.folds
        for row in fold.heldout_rows
    }
    predictions = []
    probabilities = []
    rows = []
    cursor = 0
    for fold in crossfit.folds:
        ids = [row.sample_id for row in fold.heldout_rows]
        truth = np.asarray([labels[value] for value in ids], dtype=np.uint8)
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for arm in ARM_ROLES:
                    values = (
                        truth
                        if arm == CONTROL_ARM or fold.fold_ordinal % 2
                        else 1 - truth
                    ).astype(np.uint8)
                    probs = np.where(values == 1, 0.8, 0.2).astype(np.float32)
                    stop = cursor + len(values)
                    rows.append(
                        {
                            "schema_version": "midogpp_antisymmetric_residual_mmd_prediction_cell_v1",
                            "config_contract_hash": "a" * 16,
                            "generation_lock_hash": "b" * 16,
                            "source_products_lock_hash": "c" * 16,
                            "router_plan_lock_hash": "c" * 16,
                            "cell_ordinal": len(rows),
                            "fold_ordinal": fold.fold_ordinal,
                            "fold_id": fold.fold_id,
                            "fold_hash": fold.fold_hash,
                            "target_center": fold.target_center,
                            "heldout_case_id": fold.heldout_case_id,
                            "arm_role": arm,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "candidate_sources_json": prediction._compact(
                                list(candidate_sources(fold.target_center))
                            ),
                            "weights_by_class_json": prediction._compact({
                                "0": {},
                                "1": {},
                            }),
                            "allocations_by_class_json": prediction._compact({
                                "0": {},
                                "1": {},
                            }),
                            "shuffle_seed_by_class_json": prediction._compact({
                                "0": 1,
                                "1": 2,
                            }),
                            "evaluation_row_ids_json": prediction._compact(ids),
                            "evaluation_row_identity_hash": row_identity_hash(
                                fold.heldout_rows
                            ),
                            "prediction_offset_start": cursor,
                            "prediction_offset_stop": stop,
                            "prediction_sha256": prediction._sha256_array(values),
                            "probability_sha256": prediction._sha256_array(probs),
                            "composition_hash": "d" * 16,
                            "classifier_config_hash": "e" * 16,
                            "scaler_state_hash": "f" * 16,
                            "classifier_n_iter_json": "[3]",
                            "classifier_converged": True,
                            "router_support_row_identity_hash": row_identity_hash(
                                fold.router_support_rows
                            ),
                            "plan_hash": f"{fold.fold_ordinal:016x}",
                            "heldout_case_excluded_from_route": True,
                            "labels_available_to_fit_or_predict": False,
                            "support_labels_used": False,
                            "seed_selection_performed": False,
                            "control_fit_aliased": False,
                        }
                    )
                    predictions.append(values)
                    probabilities.append(probs)
                    cursor = stop
    return (
        CrossfitPredictionStore(
            y_pred=np.concatenate(predictions).astype(np.uint8),
            prob_pos=np.concatenate(probabilities).astype(np.float32),
            index_rows=tuple(rows),
            unique_classifier_fit_count=81,
        ),
        labels,
    )


def test_case_crossfit_surface_excludes_each_heldout_case_exactly_once() -> None:
    crossfit = _crossfit()
    assert len(crossfit.folds) == 26
    assert tuple(crossfit.folds_by_target) == CENTERS
    heldout_ids = []
    for fold in crossfit.folds:
        support_cases = {row.case_id for row in fold.router_support_rows}
        assert fold.heldout_case_id not in support_cases
        assert len({row.case_id for row in fold.router_support_rows if row.partition_role == "support"}) == 2
        assert not set(row.sample_id for row in fold.router_support_rows).intersection(
            row.sample_id for row in fold.heldout_rows
        )
        heldout_ids.extend(row.sample_id for row in fold.heldout_rows)
    base_ids = [
        row.sample_id
        for target in CENTERS
        for row in _base_partitions().evaluation_rows_by_center[target]
    ]
    assert set(heldout_ids) == set(base_ids)
    assert len(heldout_ids) == len(set(heldout_ids))
    center_6_fold = crossfit.folds_by_target["6"]
    assert len(center_6_fold) == 1
    assert len(center_6_fold[0].router_support_case_ids) == 2
    assert crossfit.lock_payload["support_labels_used"] is False
    assert crossfit.lock_payload["evaluation_labels_used"] is False


def test_real_crossfit_lock_is_atomically_json_serializable(tmp_path: Path) -> None:
    crossfit = _crossfit()
    path = tmp_path / "crossfit_surface_lock.json"

    atomic_write_json(path, crossfit.lock_payload)

    persisted = read_json(path)
    assert persisted == dict(crossfit.lock_payload)
    assert persisted["crossfit_surface_lock_hash"] == crossfit.lock_hash


def test_target_seed_task_reuses_control_and_duplicate_route_compositions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = "0"
    candidates = candidate_sources(target)
    blocks = np.empty((len(candidates), 512, 2), dtype=np.float32)
    for block_index, _source in enumerate(candidates):
        blocks[block_index, :, 0] = block_index
        blocks[block_index, :, 1] = np.arange(512, dtype=np.float32)
    source_path = tmp_path / "sources.npy"
    np.save(source_path, blocks, allow_pickle=False)
    evaluation_path = tmp_path / "evaluation.npy"
    np.save(
        evaluation_path,
        np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        allow_pickle=False,
    )
    control_allocations = {
        str(label): {source: 128 for source in candidates} for label in (0, 1)
    }
    routed_allocations = {
        "0": {source: 128 for source in candidates},
        "1": {source: 128 for source in candidates},
    }
    routed_allocations["0"][candidates[0]] = 129
    routed_allocations["0"][candidates[1]] = 127
    routed_allocations["1"][candidates[0]] = 127
    routed_allocations["1"][candidates[1]] = 129
    uniform = {source: 1.0 / len(candidates) for source in candidates}
    calls = []

    def fake_fit(_x, _y, evaluation, *, classifier, threads):
        calls.append(len(evaluation))
        values = np.arange(len(evaluation), dtype=np.uint8) % 2
        return {
            "predictions": values,
            "probabilities": np.where(values == 1, 0.75, 0.25).astype(np.float32),
            "classifier_config_hash": "f" * 16,
            "scaler_state_hash": "1" * 16,
            "n_iter": (3,),
            "converged": True,
        }

    monkeypatch.setattr(prediction, "_fit_classifier", fake_fit)
    fold_tasks = []
    for fold_ordinal, (start, stop) in enumerate(((0, 2), (2, 4))):
        fold_id = f"fold_{fold_ordinal}"
        plan = {
            "fold_id": fold_id,
            "target_center": target,
            "candidate_sources": list(candidates),
            "control_weights": uniform,
            "class_0_weights": uniform,
            "class_1_weights": uniform,
            "control_allocations_by_class": control_allocations,
            "routed_allocations_by_class": routed_allocations,
            "plan_hash": f"{fold_ordinal + 2:016x}",
        }
        fold_tasks.append(
            {
                "fold_ordinal": fold_ordinal,
                "fold_id": fold_id,
                "fold_hash": f"{fold_ordinal + 4:016x}",
                "heldout_case_id": f"case_{fold_ordinal}",
                "router_support_row_identity_hash": "2" * 16,
                "evaluation_row_ids": (f"row_{start}", f"row_{start + 1}"),
                "evaluation_row_identity_hash": "3" * 16,
                "evaluation_start": start,
                "evaluation_stop": stop,
                "plan": plan,
            }
        )
    result = prediction._prediction_task(
        {
            "config_contract_hash": "a" * 16,
            "generation_lock_hash": "b" * 16,
            "source_products_lock_hash": "c" * 16,
            "router_plan_lock_hash": "d" * 16,
            "crossfit_surface_lock_hash": "e" * 16,
            "heldout_scratch_hash": "f" * 16,
            "target_center": target,
            "training_seed": 17,
            "generation_seed": 17,
            "fold_tasks": tuple(fold_tasks),
            "fold_plan_hash": "1" * 16,
            "source_array_path": str(source_path),
            "source_index_rows": [
                {
                    "source_center": source,
                    "training_seed": 17,
                    "generation_seed": 17,
                    "block_ordinal": index,
                }
                for index, source in enumerate(candidates)
            ],
            "evaluation_array_path": str(evaluation_path),
            "target_evaluation_start": 0,
            "target_evaluation_stop": 4,
            "target_evaluation_row_identity_hash": "2" * 16,
            "classifier": SimpleNamespace(config_hash="f" * 16),
            "threads_per_worker": 3,
        }
    )
    assert calls == [4, 4]
    assert result["unique_classifier_fit_count"] == 2
    assert result[f"fold_0_{CONTROL_ARM}_predictions"].shape == (2,)
    assert result[f"fold_1_{ROUTED_ARM}_predictions"].shape == (2,)
    assert (
        result[f"fold_0_{ROUTED_ARM}_metadata"]["composition_hash"]
        == result[f"fold_1_{ROUTED_ARM}_metadata"]["composition_hash"]
    )


def test_prediction_facade_preserves_public_api_across_cohesive_modules() -> None:
    assert prediction.CrossfitPredictionStore is prediction_store.CrossfitPredictionStore
    assert (
        prediction.read_crossfit_prediction_store
        is prediction_store.read_crossfit_prediction_store
    )
    assert (
        prediction.validate_crossfit_prediction_store_binding
        is prediction_store.validate_crossfit_prediction_store_binding
    )
    assert prediction._ClassSpecificComposition is composition.ClassSpecificComposition
    assert prediction._fit_classifier is composition.fit_classifier
    assert (
        prediction._compose_class_specific_prefix_blocks
        is composition.compose_class_specific_prefix_blocks
    )
    assert (
        prediction._write_prediction_checkpoint
        is prediction_worker.write_prediction_checkpoint
    )
    assert prediction._load_prediction_checkpoint is prediction_worker.load_prediction_checkpoint
    assert prediction.materialize_case_crossfit_predictions.__module__.endswith(
        ".prediction"
    )


def test_prediction_task_builder_freezes_81_spawn_tasks_and_three_threads(
    tmp_path: Path,
) -> None:
    crossfit = _crossfit()
    plan_map = {}
    scratch_folds = {}
    scratch_targets = {}
    cursor = 0
    for target in CENTERS:
        target_start = cursor
        target_rows = []
        for fold in crossfit.folds_by_target[target]:
            candidates = candidate_sources(target)
            uniform = {source: 1.0 / len(candidates) for source in candidates}
            allocations = {
                str(label): {source: 128 for source in candidates}
                for label in (0, 1)
            }
            plan_map[fold.fold_id] = {
                "fold_id": fold.fold_id,
                "target_center": target,
                "candidate_sources": list(candidates),
                "control_weights": uniform,
                "class_0_weights": uniform,
                "class_1_weights": uniform,
                "control_allocations_by_class": allocations,
                "routed_allocations_by_class": allocations,
                "plan_hash": f"{fold.fold_ordinal + 1:016x}",
            }
            stop = cursor + len(fold.heldout_rows)
            scratch_folds[fold.fold_id] = {"start": cursor, "stop": stop}
            cursor = stop
            target_rows.extend(fold.heldout_rows)
        scratch_targets[target] = {
            "start": target_start,
            "stop": cursor,
            "row_identity_hash": row_identity_hash(target_rows),
        }
    classifier = SimpleNamespace(config_hash="f" * 16)
    config = SimpleNamespace(
        contract_hash="a" * 16,
        classifier=classifier,
        runtime={"classifier_threads_per_worker": 3},
    )
    tasks = prediction_worker.build_prediction_tasks(
        config,
        "b" * 16,
        SimpleNamespace(index_rows=(), array_path=tmp_path / "sources.npy"),
        plan_map,
        "c" * 16,
        crossfit,
        source_products_lock_hash="d" * 16,
        scratch={
            "heldout_scratch_hash": "e" * 16,
            "folds": scratch_folds,
            "targets": scratch_targets,
        },
        scratch_path=tmp_path / "heldout.npy",
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert len(tasks) == len(CENTERS) * EXPECTED_SEED_CELL_COUNT == 81
    assert {
        prediction_worker.task_key(task) for task in tasks
    } == {
        (target, training_seed, generation_seed)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    assert all(task["threads_per_worker"] == 3 for task in tasks)
    assert all(task["classifier"] is classifier for task in tasks)
    assert len({task["checkpoint_path"] for task in tasks}) == 81
    assert MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT == 315


def test_prediction_checkpoint_round_trip_preserves_hashes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    task = {
        "config_contract_hash": "a" * 16,
        "generation_lock_hash": "b" * 16,
        "source_products_lock_hash": "c" * 16,
        "router_plan_lock_hash": "d" * 16,
        "crossfit_surface_lock_hash": "e" * 16,
        "heldout_scratch_hash": "f" * 16,
        "target_center": "0",
        "training_seed": 17,
        "generation_seed": 17,
        "fold_plan_hash": "1" * 16,
        "target_evaluation_row_identity_hash": "2" * 16,
        "fold_tasks": ({"fold_ordinal": 0},),
    }
    result = {
        **{key: value for key, value in task.items() if key != "fold_tasks"},
        "schema_version": "midogpp_antisymmetric_residual_mmd_prediction_checkpoint_v1",
        "unique_classifier_fit_count": 2,
    }
    for arm_index, arm in enumerate(ARM_ROLES):
        result[f"fold_0_{arm}_predictions"] = np.asarray(
            [arm_index, 1 - arm_index], dtype=np.uint8
        )
        result[f"fold_0_{arm}_probabilities"] = np.asarray(
            [0.25, 0.75], dtype=np.float32
        )
    path = tmp_path / "checkpoint.npz"
    prediction_worker.write_prediction_checkpoint(path, result)
    loaded = prediction_worker.load_prediction_checkpoint(path, task=task)
    for arm in ARM_ROLES:
        np.testing.assert_array_equal(
            loaded[f"fold_0_{arm}_predictions"],
            result[f"fold_0_{arm}_predictions"],
        )
        np.testing.assert_array_equal(
            loaded[f"fold_0_{arm}_probabilities"],
            result[f"fold_0_{arm}_probabilities"],
        )

    with np.load(path, allow_pickle=False) as raw:
        arrays = {key: np.asarray(raw[key]) for key in raw.files}
    metadata = json.loads(str(np.asarray(arrays["checkpoint_json"]).item()))
    key = f"fold_0_{CONTROL_ARM}_predictions"
    assert metadata["array_hashes"][key] == prediction._sha256_array(arrays[key])
    arrays[key] = np.asarray(arrays[key]).copy()
    arrays[key][0] = 1 - arrays[key][0]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ProtocolError, match="failed validation"):
        prediction_worker.load_prediction_checkpoint(path, task=task)


def test_scoring_concatenates_cases_then_emits_162_metrics_and_81_deltas() -> None:
    crossfit = _crossfit()
    store, labels = _prediction_store(crossfit)
    metrics, deltas, report = score_case_crossfit_predictions(
        store, crossfit, labels_by_sample_id=labels
    )
    assert len(metrics) == 162
    assert len(deltas) == 81
    assert report["metric_row_count"] == 162
    assert report["paired_delta_row_count"] == 81
    assert report["case_predictions_concatenated_before_target_metric"] is True
    assert report["case_level_metrics_averaged"] is False
    assert all(row["case_metrics_averaged"] is False for row in metrics)
    with pytest.raises(ProtocolError, match="missing or extra rows"):
        score_case_crossfit_predictions(
            store,
            crossfit,
            labels_by_sample_id={**labels, "support_leak": 0},
        )
    last_start = int(store.index_rows[-1]["prediction_offset_start"])
    with pytest.raises(ProtocolError, match="malformed"):
        CrossfitPredictionStore(
            y_pred=store.y_pred[:last_start],
            prob_pos=store.prob_pos[:last_start],
            index_rows=store.index_rows[:-1],
            unique_classifier_fit_count=81,
        )


def test_global_seal_blocks_label_access_after_prediction_bytes_change(
    tmp_path: Path,
) -> None:
    crossfit = _crossfit()
    store, _labels = _prediction_store(crossfit)
    plans = _plans(crossfit)
    config = SimpleNamespace(
        contract_hash="a" * 16,
        validation_manifest_path=tmp_path / "must_not_open.csv",
    )
    array_path = tmp_path / CROSSFIT_PREDICTION_ARRAY_MEMBER
    index_path = tmp_path / CROSSFIT_PREDICTION_INDEX_MEMBER
    array_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    array_path.write_bytes(b"sealed arrays")
    index_path.write_bytes(b"sealed index")
    seal = build_global_crossfit_prediction_seal(
        config, crossfit, plans, store, root=tmp_path
    )
    assert seal["cell_count"] == EXPECTED_PREDICTION_CELL_COUNT
    assert (tmp_path / GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER).is_file()
    array_path.write_bytes(b"tampered arrays")
    with pytest.raises(ProtocolError, match="capability failed seal validation"):
        open_crossfit_evaluation_labels(config, crossfit, root=tmp_path)
