"""Checkpoint authentication and durable probability-store publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import read_json, sha256_array, sha256_file

from .fit_semantics import (
    publish_fit_receipt,
    validate_published_fit_inventory,
    validate_seed_cell_fit_inventory,
)
from .prediction_composition import candidate_exclusion_is_valid
from .prediction_contracts import (
    CANDIDATE_EXCLUSION_SENTINEL,
    CHECKPOINT_DIRECTORY,
    CPU_PREDICTION_WORKERS,
    LOCKED_CLASSIFIER_SPEC,
    PRODUCTION_PREDICTION_GEOMETRY,
    PredictionGeometry,
    PredictionRuntimeTestMode,
    PredictionSurface,
    final_paths,
    geometry_for,
)
from .prediction_fitting import assert_prediction_task, task_key
from .prediction_io import canonical_sha256


def load_prediction_surface(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_source_receipt_hash: str | None = None,
    expected_attempt_id: str | None = None,
    test_mode: PredictionRuntimeTestMode | None = None,
) -> PredictionSurface:
    """Load and fully validate a completed read-only probability surface."""

    geometry = geometry_for(test_mode)
    destination = Path(root)
    candidate_path, exact_b_path, index_path, receipt_path = final_paths(destination)
    if any(
        path.is_symlink() or not path.is_file()
        for path in (candidate_path, exact_b_path, index_path, receipt_path)
    ):
        raise ProtocolError("SCEPTRE v5 prediction final store is absent or unsafe.")
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    index_unhashed = {key: value for key, value in index.items() if key != "index_sha256"}
    receipt_unhashed = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    row_ids = index.get("row_ids")
    row_centers = index.get("row_centers")
    fit_rows = index.get("fit_rows")
    if (
        not isinstance(row_ids, list)
        or not isinstance(row_centers, list)
        or not isinstance(fit_rows, list)
    ):
        raise ProtocolError("SCEPTRE v5 prediction index is malformed.")
    if (
        index.get("schema_version") != "midogpp_sceptre_v5_physical_prediction_index_v1"
        or receipt.get("schema_version")
        != "midogpp_sceptre_v5_physical_prediction_receipt_v1"
        or receipt.get("status")
        != "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS"
        or not str(index.get("attempt_id", ""))
        or receipt.get("attempt_id") != index.get("attempt_id")
        or index.get("geometry") != geometry.to_payload()
        or receipt.get("geometry") != geometry.to_payload()
        or index.get("index_sha256") != canonical_sha256(index_unhashed)
        or receipt.get("receipt_sha256") != canonical_sha256(receipt_unhashed)
        or receipt.get("candidate_array_file_sha256") != sha256_file(candidate_path)
        or receipt.get("exact_b_array_file_sha256") != sha256_file(exact_b_path)
        or receipt.get("prediction_index_file_sha256") != sha256_file(index_path)
        or receipt.get("prediction_index_sha256") != index.get("index_sha256")
        or receipt.get("cache_binding_hash") != index.get("cache_binding_hash")
        or receipt.get("row_identity_sha256") != index.get("row_identity_sha256")
        or receipt.get("fit_index_sha256") != index.get("fit_index_sha256")
        or index.get("fit_index_sha256") != canonical_sha256(fit_rows)
        or len(fit_rows) != geometry.fit_count
        or index.get("fit_count") != geometry.fit_count
        or receipt.get("fit_count") != geometry.fit_count
        or index.get("classifier") != LOCKED_CLASSIFIER_SPEC.to_payload()
        or index.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        or receipt.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        or index.get("candidate_source_order") != list(geometry.centers)
        or index.get("seed_cell_order") != [list(value) for value in geometry.seed_cells]
        or index.get("exact_b_target_exclusion_verified") is not True
        or receipt.get("target_expert_excluded_from_every_exact_b_fit") is not True
        or index.get("candidate_target_exclusion_mode") != "MASKED_BEFORE_SCORING"
        or index.get("candidate_exclusion_sentinel")
        != float(CANDIDATE_EXCLUSION_SENTINEL)
        or receipt.get("target_expert_excluded_from_every_candidate_score") is not True
        or receipt.get("candidate_exclusion_sentinel")
        != float(CANDIDATE_EXCLUSION_SENTINEL)
        or receipt.get("cpu_worker_count")
        != (CPU_PREDICTION_WORKERS if test_mode is None else 0)
        or receipt.get("blas_threads_per_worker") != 1
        or receipt.get("native_threads_per_worker") != 1
        or receipt.get("top_level_spawn_pool_only") is not (test_mode is None)
        or index.get("seed_selection_performed") is not False
        or receipt.get("seed_selection_performed") is not False
        or index.get("manifest_opened") is not False
        or receipt.get("manifest_opened") is not False
        or index.get("outcomes_available") is not False
        or receipt.get("outcomes_available") is not False
        or index.get("raw_sample_paths_available") is not False
        or receipt.get("raw_sample_paths_available") is not False
        or receipt.get("classifier_refit_after_seal") is not False
        or receipt.get("synthetic_test_mode") is not (test_mode is not None)
        or (
            expected_config_hash is not None
            and receipt.get("config_hash") != expected_config_hash
        )
        or (
            expected_source_receipt_hash is not None
            and receipt.get("source_receipt_sha256") != expected_source_receipt_hash
        )
        or (
            expected_attempt_id is not None
            and receipt.get("attempt_id") != expected_attempt_id
        )
    ):
        raise ProtocolError("SCEPTRE v5 prediction receipt failed validation.")
    expected_row_hash = canonical_sha256(
        [
            {"row_ordinal": ordinal, "row_id": str(row_id), "center": str(center)}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    if index.get("row_identity_sha256") != expected_row_hash:
        raise ProtocolError("SCEPTRE v5 prediction row identity hash drifted.")
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    exact_b = np.load(exact_b_path, mmap_mode="r", allow_pickle=False)
    if (
        candidate.shape != geometry.candidate_shape
        or exact_b.shape != geometry.exact_b_shape
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or candidate.flags.writeable
        or exact_b.flags.writeable
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or not candidate_exclusion_is_valid(candidate, geometry=geometry)
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
    ):
        raise ProtocolError("SCEPTRE v5 prediction tensor geometry or values drifted.")
    return PredictionSurface(
        root=destination,
        candidate_array_path=candidate_path,
        exact_b_array_path=exact_b_path,
        index_path=index_path,
        receipt_path=receipt_path,
        geometry=geometry,
        row_ids=tuple(str(value) for value in row_ids),
        row_centers=tuple(str(value) for value in row_centers),
        index=index,
        receipt=receipt,
    )


def load_checkpoint_if_complete(
    task: Mapping[str, object], *, geometry: PredictionGeometry
) -> Mapping[str, object] | None:
    candidate_path = Path(str(task["checkpoint_candidate_path"]))
    exact_b_path = Path(str(task["checkpoint_exact_b_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    paths = (candidate_path, exact_b_path, json_path)
    if any(path.is_symlink() for path in paths):
        raise ProtocolError("SCEPTRE v5 prediction checkpoint contains a symlink.")
    present = tuple(path.is_file() for path in paths)
    if present == (False, False, False):
        return None
    if present != (True, True, True):
        raise ProtocolError("SCEPTRE v5 prediction checkpoint is partial; refusing refit.")
    assert_prediction_task(task, geometry=geometry)
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    exact_b = np.load(exact_b_path, mmap_mode="r", allow_pickle=False)
    fit_rows = payload.get("fit_rows")
    if (
        payload.get("checkpoint_sha256") != canonical_sha256(unhashed)
        or payload.get("schema_version")
        != "midogpp_sceptre_v5_physical_prediction_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("attempt_id") != task["attempt_id"]
        or payload.get("task_sha256") != task["task_sha256"]
        or int(payload.get("training_seed", -1)) != task["training_seed"]
        or int(payload.get("generation_seed", -1)) != task["generation_seed"]
        or payload.get("candidate_array_file_sha256") != sha256_file(candidate_path)
        or payload.get("exact_b_array_file_sha256") != sha256_file(exact_b_path)
        or payload.get("candidate_array_sha256") != sha256_array(candidate)
        or payload.get("exact_b_array_sha256") != sha256_array(exact_b)
        or candidate.shape != (len(geometry.centers), geometry.evaluation_rows)
        or exact_b.shape != (geometry.evaluation_rows,)
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or not candidate_exclusion_is_valid(
            candidate[np.newaxis, ...], geometry=geometry
        )
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
        or not isinstance(fit_rows, list)
        or len(fit_rows) != 2 * len(geometry.centers)
        or payload.get("fit_index_sha256") != canonical_sha256(fit_rows)
        or payload.get("fit_count") != 2 * len(geometry.centers)
        or payload.get("manifest_opened") is not False
        or payload.get("outcomes_available") is not False
        or payload.get("raw_sample_paths_available") is not False
        or payload.get("target_expert_excluded_from_every_exact_b_fit") is not True
        or payload.get("target_expert_excluded_from_every_candidate_score") is not True
        or payload.get("candidate_exclusion_sentinel")
        != float(CANDIDATE_EXCLUSION_SENTINEL)
        or payload.get("seed_selection_performed") is not False
    ):
        raise ProtocolError("SCEPTRE v5 prediction checkpoint failed validation.")
    validate_seed_cell_fit_inventory(
        fit_rows,
        centers=geometry.centers,
        training_seed=int(task["training_seed"]),
        generation_seed=int(task["generation_seed"]),
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    return payload


def publish_probability_arrays(
    candidate_path: Path,
    exact_b_path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[int, int], Mapping[str, object]],
    geometry: PredictionGeometry,
) -> list[dict[str, object]]:
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temp = candidate_path.with_suffix(
        candidate_path.suffix + f".{os.getpid()}.tmp"
    )
    exact_b_temp = exact_b_path.with_suffix(exact_b_path.suffix + f".{os.getpid()}.tmp")
    fit_rows: list[dict[str, object]] = []
    try:
        candidate = np.lib.format.open_memmap(
            candidate_temp,
            mode="w+",
            dtype=np.float32,
            shape=geometry.candidate_shape,
        )
        exact_b = np.lib.format.open_memmap(
            exact_b_temp,
            mode="w+",
            dtype=np.float32,
            shape=geometry.exact_b_shape,
        )
        for seed_ordinal, task in enumerate(tasks):
            checkpoint = completed[task_key(task)]
            candidate[seed_ordinal] = np.load(
                Path(str(task["checkpoint_candidate_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            exact_b[seed_ordinal] = np.load(
                Path(str(task["checkpoint_exact_b_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            for fit_ordinal, raw in enumerate(checkpoint["fit_rows"]):
                fit_rows.append(
                    publish_fit_receipt(
                        raw,
                        global_fit_ordinal=len(fit_rows),
                        seed_cell_ordinal=seed_ordinal,
                        within_cell_fit_ordinal=fit_ordinal,
                        evaluation_rows=geometry.evaluation_rows,
                        rows_by_center=geometry.rows_by_center,
                        expected_classifier_config_hash=(
                            LOCKED_CLASSIFIER_SPEC.config_hash
                        ),
                    )
                )
        candidate.flush()
        exact_b.flush()
        del candidate, exact_b
        if candidate_path.exists() or exact_b_path.exists():
            raise ProtocolError("SCEPTRE v5 prediction final array appeared during publication.")
        os.replace(candidate_temp, candidate_path)
        os.replace(exact_b_temp, exact_b_path)
    except BaseException:
        candidate_temp.unlink(missing_ok=True)
        exact_b_temp.unlink(missing_ok=True)
        raise
    if len(fit_rows) != geometry.fit_count:
        raise ProtocolError("SCEPTRE v5 prediction fit coverage drifted.")
    validate_published_fit_inventory(
        fit_rows,
        centers=geometry.centers,
        seed_cells=geometry.seed_cells,
        evaluation_rows=geometry.evaluation_rows,
        rows_by_center=geometry.rows_by_center,
        expected_classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
    )
    return fit_rows


def validate_checkpoint_tree(directory: Path, *, geometry: PredictionGeometry) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("SCEPTRE v5 prediction checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("SCEPTRE v5 prediction checkpoint root is unsafe.")
    allowed_top = {"evaluation_embeddings.npy", "evaluation_frame.json", "tasks"}
    for path in directory.iterdir():
        if path.is_symlink() or path.name not in allowed_top:
            raise ProtocolError("SCEPTRE v5 prediction checkpoint tree has an unknown member.")
    tasks_root = directory / "tasks"
    if not tasks_root.exists():
        return
    if tasks_root.is_symlink() or not tasks_root.is_dir():
        raise ProtocolError("SCEPTRE v5 prediction task checkpoint root is unsafe.")
    expected = {
        f"train_{train}_generation_{generation}.{suffix}"
        for train, generation in geometry.seed_cells
        for suffix in ("candidate.npy", "exact_b.npy", "json")
    }
    for path in tasks_root.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ProtocolError("SCEPTRE v5 prediction task tree has an unknown member.")


__all__ = (
    "load_checkpoint_if_complete",
    "load_prediction_surface",
    "publish_probability_arrays",
    "validate_checkpoint_tree",
)
