"""Runtime orchestration for strict source-OOF development predictions."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .constants import (
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    SOURCE_CHECKPOINT_DIRECTORY,
)
from .development_actions import (
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
)
from .development_prediction_contracts import (
    COMPOSITE_PRELABEL_SEAL_MEMBER,
    COMPOSITE_PRELABEL_STATUS,
    DEVELOPMENT_CHECKPOINT_DIRECTORY,
    DEVELOPMENT_CLASSIFIER_INDEX_MEMBER,
    DEVELOPMENT_CLASSIFIER_SEAL_MEMBER,
    DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    DEVELOPMENT_PREDICTION_INDEX_MEMBER,
    DEVELOPMENT_PREDICTION_SEAL_MEMBER,
    CompositePrelabelPredictionSeal,
    DevelopmentPredictionConfig,
    DevelopmentSourcePredictionSeal,
)
from .development_prediction_plans import build_development_source_tasks
from .development_prediction_store import (
    assemble_development_source_products,
    execute_or_resume_development_source,
    load_development_source_prediction_seal,
    write_development_action_library,
    write_development_classifier_bank,
    write_development_prediction_store,
    write_development_source_prediction_seal,
)
from .frame_scratch import write_frame_scratch
from .hashing import canonical_hash
from .input_contracts import LabelFreeSourceFrame


def materialize_development_source_action_predictions(
    config: DevelopmentPredictionConfig,
    generated_sources: FrozenSourceStreamCache,
    source_frame: LabelFreeSourceFrame,
    *,
    root: Path,
) -> DevelopmentSourcePredictionSeal:
    """Fit 5,184 pair-symmetric classifiers and seal 10,368 H/q cells."""

    _assert_runtime(config.runtime)
    _assert_parent_cuda_free()
    library = write_development_action_library(root)
    library_hash = str(library["action_library_hash"])
    final_members = (
        DEVELOPMENT_CLASSIFIER_INDEX_MEMBER,
        DEVELOPMENT_CLASSIFIER_SEAL_MEMBER,
        DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
        DEVELOPMENT_PREDICTION_INDEX_MEMBER,
        DEVELOPMENT_PREDICTION_SEAL_MEMBER,
    )
    if all((root / member).is_file() for member in final_members):
        result = load_development_source_prediction_seal(
            root,
            expected_config_hash=config.contract_hash,
            expected_source_stream_lock_hash=generated_sources.lock_hash,
            expected_source_cache_binding_hash=source_frame.cache_binding_hash,
        )
        if result.action_library_hash != library_hash:
            raise ProtocolError("Strict source-OOF action library drifted on resume.")
        shutil.rmtree(root / DEVELOPMENT_CHECKPOINT_DIRECTORY, ignore_errors=True)
        shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return result
    scratch = write_frame_scratch(root, frame_role="source", frame=source_frame)
    tasks = build_development_source_tasks(
        config,
        generated_sources,
        scratch=scratch,
        action_library_hash=library_hash,
        root=root,
    )
    completed = execute_or_resume_development_source(
        tasks, workers=int(config.runtime["cpu_workers"])
    )
    classifier_cells, prediction_cells = assemble_development_source_products(
        root, tasks=tasks, completed=completed
    )
    classifier_bank = write_development_classifier_bank(
        root,
        cells=classifier_cells,
        config_contract_hash=config.contract_hash,
        source_stream_lock_hash=generated_sources.lock_hash,
        action_library_hash=library_hash,
        source_cache_binding_hash=source_frame.cache_binding_hash,
    )
    rows = {
        query: tuple(row.source_row_id for row in source_frame.rows_by_center[query])
        for query in CENTERS
    }
    cases = {
        query: tuple(row.case_id for row in source_frame.rows_by_center[query])
        for query in CENTERS
    }
    store = write_development_prediction_store(
        root,
        cells=prediction_cells,
        rows_by_query=rows,
        case_ids_by_query=cases,
        frame_cache_binding_hash=source_frame.cache_binding_hash,
        action_library_hash=library_hash,
        classifier_bank_seal_hash=classifier_bank.seal_hash,
        config_contract_hash=config.contract_hash,
    )
    result = write_development_source_prediction_seal(
        root,
        classifier_bank=classifier_bank,
        source_store=store,
        config_contract_hash=config.contract_hash,
    )
    shutil.rmtree(root / DEVELOPMENT_CHECKPOINT_DIRECTORY, ignore_errors=True)
    shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return result


def materialize_composite_prelabel_prediction_seal(
    strict_source_predictions: DevelopmentSourcePredictionSeal,
    target_classifier_bank: object,
    *,
    root: Path,
) -> CompositePrelabelPredictionSeal:
    """Bind strict H/q surfaces and the independent q=H target bank."""

    target_payload = dict(getattr(target_classifier_bank, "seal_payload", {}))
    if (
        target_payload.get("status")
        != "SEALED_1458_SOURCE_ONLY_ACTION_CLASSIFIERS"
        or target_payload.get("fit_count") != EXPECTED_CLASSIFIER_FIT_COUNT
        or target_payload.get("test_cache_admitted") is not False
        or getattr(target_classifier_bank, "source_cache_binding_hash", None)
        != strict_source_predictions.source_store.frame_cache_binding_hash
        or getattr(target_classifier_bank, "action_library_hash", None)
        == strict_source_predictions.classifier_bank.action_library_hash
    ):
        raise ProtocolError("Target-compatible classifier bank is not prelabel-ready.")
    unhashed = {
        "schema_version": "midogpp_composite_prelabel_prediction_seal_v1",
        "status": COMPOSITE_PRELABEL_STATUS,
        "strict_source_prediction_seal_hash": strict_source_predictions.seal_hash,
        "strict_source_oof_classifier_bank_seal_hash": (
            strict_source_predictions.classifier_bank.seal_hash
        ),
        "strict_source_oof_prediction_store_hash": (
            strict_source_predictions.source_store.store_hash
        ),
        "target_classifier_bank_seal_hash": str(
            getattr(target_classifier_bank, "seal_hash")
        ),
        "source_cache_binding_hash": (
            strict_source_predictions.source_store.frame_cache_binding_hash
        ),
        "strict_source_physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "strict_source_logical_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "query_excluded_from_every_source_composition": True,
        "outer_target_excluded_from_every_source_composition": True,
        "unordered_excluded_pair_fit_reuse": True,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
    }
    payload = {
        **unhashed,
        "composite_prelabel_prediction_seal_hash": canonical_hash(unhashed),
    }
    path = root / COMPOSITE_PRELABEL_SEAL_MEMBER
    if path.is_file():
        if read_json(path) != payload:
            raise ProtocolError("Persisted composite prelabel seal drifted.")
    else:
        atomic_json(path, payload)
    return CompositePrelabelPredictionSeal(
        strict_source_predictions=strict_source_predictions,
        target_classifier_bank=target_classifier_bank,
        seal_payload=payload,
        seal_path=path,
    )


def load_composite_prelabel_prediction_seal(
    strict_source_predictions: DevelopmentSourcePredictionSeal,
    target_classifier_bank: object,
    *,
    root: Path,
) -> CompositePrelabelPredictionSeal:
    return CompositePrelabelPredictionSeal(
        strict_source_predictions=strict_source_predictions,
        target_classifier_bank=target_classifier_bank,
        seal_payload=read_json(root / COMPOSITE_PRELABEL_SEAL_MEMBER),
        seal_path=root / COMPOSITE_PRELABEL_SEAL_MEMBER,
    )


def _assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("cpu_workers", -1)) != 4
        or int(runtime.get("threads_per_worker", -1)) != 3
        or int(runtime.get("maximum_total_cpu_threads", -1)) != 12
        or int(runtime.get("maximum_dense_fit_bytes", -1)) != 536_870_912
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("gpu_and_cpu_phases_disjoint") is not True
        or runtime.get("parent_cuda_context_forbidden_during_cpu_phase") is not True
        or runtime.get("scientific_reduction_dtype") != "float64"
        or runtime.get("surface_storage_dtype") != "float32"
        or runtime.get("hash_validated_resume") is not True
    ):
        raise ProtocolError("Strict source-OOF workstation runtime drifted.")


def _assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Strict source-OOF CPU parent must remain CUDA-free.")


__all__ = (
    "load_composite_prelabel_prediction_seal",
    "materialize_composite_prelabel_prediction_seal",
    "materialize_development_source_action_predictions",
)
