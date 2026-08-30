"""Public orchestration facade for the SCEPTRE v5 prediction pipeline.

The facade preserves the original import surface while delegating immutable
contracts, label-free frame staging, exact-B composition, spawn-bounded fitting,
and durable publication to focused modules. Production remains fixed at nine
seed cells, nine candidate sources, and 9,928 MIDOG++ evaluation rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file

from .prediction_composition import (
    candidate_exclusion_is_valid as _candidate_exclusion_is_valid,
    compose_exact_b as _compose_exact_b,
    compose_single_source as _compose_single_source,
    exact_b_source_centers,
)
from .prediction_contracts import (
    CANDIDATE_ARRAY_MEMBER,
    CANDIDATE_EXCLUSION_SENTINEL,
    CHECKPOINT_DIRECTORY,
    CPU_PREDICTION_WORKERS,
    EVALUATION_FRAME_MEMBER,
    EVALUATION_SCRATCH_MEMBER,
    EXACT_B_ARRAY_MEMBER,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FitOutcome,
    FitPredict,
    LOCKED_CLASSIFIER_SPEC,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
    PRODUCTION_PREDICTION_GEOMETRY,
    PredictionGeometry,
    PredictionRuntimeConfig,
    PredictionRuntimeTestMode,
    PredictionSurface,
    assert_owned_root as _assert_owned_root,
    assert_production_runtime as _assert_production_runtime,
    attempt_id as _attempt_id,
    config_hash as _config_hash,
    final_paths as _final_paths,
    geometry_for as _geometry,
    locked_classifier as _locked_classifier,
)
from .prediction_fitting import (
    build_tasks as _build_tasks,
    execute_cpu_tasks as _execute_cpu_tasks,
    fit_locked_logistic as _fit_locked_logistic,
    prediction_worker as _prediction_worker,
    task_key as _task_key,
)
from .prediction_frame import stage_evaluation_frame as _stage_evaluation_frame
from .prediction_io import canonical_sha256 as _canonical_sha256
from .prediction_io import persist_exact_json as _persist_exact_json
from .prediction_store import (
    load_checkpoint_if_complete as _load_checkpoint_if_complete,
    load_prediction_surface,
    publish_probability_arrays as _publish_probability_arrays,
    validate_checkpoint_tree as _validate_checkpoint_tree,
)
from .source_streams import SourceStreamStore


def materialize_prediction_surface(
    config: PredictionRuntimeConfig,
    source_store: SourceStreamStore,
    frame: object,
    *,
    root: Path,
    attempt_id: str | None = None,
    test_mode: PredictionRuntimeTestMode | None = None,
) -> PredictionSurface:
    """Fit all predeclared classifiers and seal both label-free tensors."""

    geometry = _geometry(test_mode)
    config_hash = _config_hash(config)
    inherited_attempt = attempt_id
    if (
        inherited_attempt is None
        and getattr(config, "attempt_id", None) is None
        and test_mode is not None
    ):
        inherited_attempt = source_store.attempt_id
    attempt = _attempt_id(
        config,
        explicit=inherited_attempt,
        root=Path(root),
        synthetic=test_mode is not None,
    )
    classifier = _locked_classifier(config)
    if source_store.geometry != geometry.source_geometry:
        raise ProtocolError("SCEPTRE v5 prediction/source geometry binding drifted.")
    if source_store.attempt_id != attempt:
        raise ProtocolError("SCEPTRE v5 prediction/source attempt binding drifted.")
    if test_mode is None:
        _assert_production_runtime(config.runtime)
    destination = Path(root)
    _assert_owned_root(destination)
    final_paths = _final_paths(destination)
    present = tuple(path.is_file() for path in final_paths)
    if any(path.is_symlink() for path in final_paths):
        raise ProtocolError("SCEPTRE v5 prediction final store contains a symlink.")
    if all(present):
        return load_prediction_surface(
            destination,
            expected_config_hash=config_hash,
            expected_source_receipt_hash=source_store.receipt_hash,
            expected_attempt_id=attempt,
            test_mode=test_mode,
        )
    if any(present):
        raise ProtocolError("SCEPTRE v5 prediction final store is an unsafe partial state.")

    frame_payload = _stage_evaluation_frame(
        destination,
        frame,
        geometry=geometry,
        attempt_id=attempt,
    )
    tasks = _build_tasks(
        config_hash=config_hash,
        classifier=classifier,
        source_store=source_store,
        frame_payload=frame_payload,
        root=destination,
        geometry=geometry,
        attempt_id=attempt,
    )
    completed: dict[tuple[int, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint = _load_checkpoint_if_complete(task, geometry=geometry)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[_task_key(task)] = checkpoint
    if pending:
        if test_mode is None:
            results = _execute_cpu_tasks(pending)
        else:
            fit = test_mode.fit_predict or _fit_locked_logistic
            results = tuple(
                _prediction_worker(task, geometry=geometry, fit_predict=fit)
                for task in pending
            )
        for result in results:
            key = (int(result["training_seed"]), int(result["generation_seed"]))
            task = next(task for task in pending if _task_key(task) == key)
            loaded = _load_checkpoint_if_complete(task, geometry=geometry)
            if loaded is None or loaded.get("checkpoint_sha256") != result.get(
                "checkpoint_sha256"
            ):
                raise ProtocolError("SCEPTRE v5 prediction checkpoint return drifted.")
            completed[key] = loaded
    if len(completed) != len(geometry.seed_cells):
        raise ProtocolError("SCEPTRE v5 prediction checkpoint coverage is incomplete.")

    candidate_path = destination / CANDIDATE_ARRAY_MEMBER
    exact_b_path = destination / EXACT_B_ARRAY_MEMBER
    fit_rows = _publish_probability_arrays(
        candidate_path,
        exact_b_path,
        tasks=tasks,
        completed=completed,
        geometry=geometry,
    )
    row_ids = tuple(str(value) for value in frame_payload["row_ids"])
    row_centers = tuple(str(value) for value in frame_payload["row_centers"])
    row_identity_sha256 = _canonical_sha256(
        [
            {"row_ordinal": ordinal, "row_id": row_id, "center": center}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    fit_index_sha256 = _canonical_sha256(fit_rows)
    index_unhashed = {
        "schema_version": "midogpp_sceptre_v5_physical_prediction_index_v1",
        "attempt_id": attempt,
        "config_hash": config_hash,
        "source_receipt_sha256": source_store.receipt_hash,
        "source_array_sha256": source_store.receipt["source_array_sha256"],
        "cache_binding_hash": frame_payload["cache_binding_hash"],
        "geometry": geometry.to_payload(),
        "classifier": classifier.to_payload(),
        "classifier_config_hash": classifier.config_hash,
        "row_ids": list(row_ids),
        "row_centers": list(row_centers),
        "row_identity_sha256": row_identity_sha256,
        "fit_rows": fit_rows,
        "fit_index_sha256": fit_index_sha256,
        "fit_count": len(fit_rows),
        "candidate_source_order": list(geometry.centers),
        "seed_cell_order": [list(value) for value in geometry.seed_cells],
        "exact_b_target_exclusion_verified": True,
        "candidate_target_exclusion_mode": "MASKED_BEFORE_SCORING",
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "all_seed_cells_retained": True,
        "seed_selection_performed": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    index = {**index_unhashed, "index_sha256": _canonical_sha256(index_unhashed)}
    index_path = destination / PREDICTION_INDEX_MEMBER
    _persist_exact_json(index_path, index)
    receipt_unhashed = {
        "schema_version": "midogpp_sceptre_v5_physical_prediction_receipt_v1",
        "status": "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS",
        "attempt_id": attempt,
        "config_hash": config_hash,
        "source_receipt_sha256": source_store.receipt_hash,
        "cache_binding_hash": frame_payload["cache_binding_hash"],
        "geometry": geometry.to_payload(),
        "classifier_config_hash": classifier.config_hash,
        "candidate_array_file_sha256": sha256_file(candidate_path),
        "exact_b_array_file_sha256": sha256_file(exact_b_path),
        "prediction_index_file_sha256": sha256_file(index_path),
        "prediction_index_sha256": index["index_sha256"],
        "row_identity_sha256": row_identity_sha256,
        "fit_index_sha256": fit_index_sha256,
        "fit_count": len(fit_rows),
        "candidate_shape": list(geometry.candidate_shape),
        "exact_b_shape": list(geometry.exact_b_shape),
        "dtype": "float32",
        "npy_memmap_mode": "read_only",
        "cpu_worker_count": CPU_PREDICTION_WORKERS if test_mode is None else 0,
        "blas_threads_per_worker": 1,
        "native_threads_per_worker": 1,
        "top_level_spawn_pool_only": test_mode is None,
        "target_expert_excluded_from_every_exact_b_fit": True,
        "target_expert_excluded_from_every_candidate_score": True,
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
        "classifier_refit_after_seal": False,
        "seed_selection_performed": False,
        "synthetic_test_mode": test_mode is not None,
    }
    receipt = {
        **receipt_unhashed,
        "receipt_sha256": _canonical_sha256(receipt_unhashed),
    }
    _persist_exact_json(destination / PREDICTION_RECEIPT_MEMBER, receipt)
    candidate_path.chmod(0o444)
    exact_b_path.chmod(0o444)
    surface = load_prediction_surface(
        destination,
        expected_config_hash=config_hash,
        expected_source_receipt_hash=source_store.receipt_hash,
        expected_attempt_id=attempt,
        test_mode=test_mode,
    )
    checkpoint_root = destination / CHECKPOINT_DIRECTORY
    _validate_checkpoint_tree(checkpoint_root, geometry=geometry)
    shutil.rmtree(checkpoint_root)
    return surface


__all__ = (
    "CANDIDATE_ARRAY_MEMBER",
    "CANDIDATE_EXCLUSION_SENTINEL",
    "CHECKPOINT_DIRECTORY",
    "CPU_PREDICTION_WORKERS",
    "EXACT_B_ARRAY_MEMBER",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "FitOutcome",
    "LOCKED_CLASSIFIER_SPEC",
    "PREDICTION_INDEX_MEMBER",
    "PREDICTION_RECEIPT_MEMBER",
    "PRODUCTION_PREDICTION_GEOMETRY",
    "PredictionGeometry",
    "PredictionRuntimeConfig",
    "PredictionRuntimeTestMode",
    "PredictionSurface",
    "exact_b_source_centers",
    "load_prediction_surface",
    "materialize_prediction_surface",
)
