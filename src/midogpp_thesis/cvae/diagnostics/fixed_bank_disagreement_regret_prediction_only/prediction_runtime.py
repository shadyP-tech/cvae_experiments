"""Two-phase action runtime: source fit/seal, then frozen test inference."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .actions import action_library_payload
from .constants import (
    CENTERS,
    CLASSIFIER_COEFFICIENT_MEMBER,
    CLASSIFIER_INDEX_MEMBER,
    CLASSIFIER_INTERCEPT_MEMBER,
    CLASSIFIER_MEAN_MEMBER,
    CLASSIFIER_SCALE_MEMBER,
    CLASSIFIER_SEAL_MEMBER,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_TASK_COUNT,
    SOURCE_ARRAY_MEMBER,
    SOURCE_CHECKPOINT_DIRECTORY,
    SOURCE_INDEX_MEMBER,
    SOURCE_SEAL_MEMBER,
    TEST_ARRAY_MEMBER,
    TEST_CHECKPOINT_DIRECTORY,
    TEST_INDEX_MEMBER,
    TEST_SEAL_MEMBER,
)
from .input_contracts import (
    LabelFreeSourceFrame,
    LabelFreeTestFrame,
    TestInferenceAdmission,
)
from .prediction_contracts import (
    ActionClassifierBank,
    ActionPredictionConfig,
    GlobalSourcePredictionSeal,
    GlobalTestPredictionSeal,
)
from .prediction_store import (
    load_action_classifier_bank,
    load_global_source_prediction_seal,
    load_global_test_prediction_seal,
    write_action_library,
    write_classifier_bank_manifests,
    write_prediction_store,
    write_source_prediction_seal,
    write_test_prediction_seal,
)
from .prediction_tasks import (
    assemble_source_products,
    assemble_test_cells,
    build_source_tasks,
    build_test_tasks,
    execute_or_resume_source,
    execute_or_resume_test,
    write_frame_scratch,
)


class RegretModelBankSealLike(Protocol):
    seal_hash: str
    status: str


def materialize_target_action_classifier_bank(
    config: ActionPredictionConfig,
    generated_sources: FrozenSourceStreamCache,
    source_frame: LabelFreeSourceFrame,
    *,
    root: Path,
) -> ActionClassifierBank:
    """Fit and seal the 1,458 q=H-compatible classifiers before labels.

    The legacy task worker also computes temporary source projections while it
    fits each classifier.  Those projections are deliberately discarded and
    its checkpoint directory is removed after the parameter bank is sealed;
    only the strict H/q source-OOF runtime may publish source probabilities.
    """

    _assert_runtime(config.runtime)
    _assert_parent_cuda_free()
    library = write_action_library(root)
    library_hash = str(library["action_library_hash"])
    final_members = (
        CLASSIFIER_MEAN_MEMBER,
        CLASSIFIER_SCALE_MEMBER,
        CLASSIFIER_COEFFICIENT_MEMBER,
        CLASSIFIER_INTERCEPT_MEMBER,
        CLASSIFIER_INDEX_MEMBER,
        CLASSIFIER_SEAL_MEMBER,
    )
    if all((root / member).is_file() for member in final_members):
        result = load_action_classifier_bank(
            root,
            expected_config_hash=config.contract_hash,
            expected_source_stream_lock_hash=generated_sources.lock_hash,
            expected_source_cache_binding_hash=source_frame.cache_binding_hash,
        )
        if result.action_library_hash != library_hash:
            raise ProtocolError("Target-compatible action library drifted on resume.")
        shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return result
    scratch = write_frame_scratch(root, frame_role="source", frame=source_frame)
    tasks = build_source_tasks(
        config,
        generated_sources,
        scratch=scratch,
        action_library_hash=library_hash,
        root=root,
    )
    completed = execute_or_resume_source(
        tasks, workers=int(config.runtime["cpu_workers"])
    )
    classifier_cells, _unused_source_projections = assemble_source_products(
        root, tasks=tasks, completed=completed
    )
    result = write_classifier_bank_manifests(
        root,
        cells=classifier_cells,
        config_contract_hash=config.contract_hash,
        source_stream_lock_hash=generated_sources.lock_hash,
        action_library_hash=library_hash,
        source_cache_binding_hash=source_frame.cache_binding_hash,
    )
    shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return result


def materialize_source_action_predictions(
    config: ActionPredictionConfig,
    generated_sources: FrozenSourceStreamCache,
    source_frame: LabelFreeSourceFrame,
    *,
    root: Path,
) -> GlobalSourcePredictionSeal:
    """Fit exactly 1,458 classifiers and seal source predictions plus state."""

    _assert_runtime(config.runtime)
    _assert_parent_cuda_free()
    library = write_action_library(root)
    library_hash = str(library["action_library_hash"])
    final_members = (
        CLASSIFIER_INDEX_MEMBER,
        CLASSIFIER_SEAL_MEMBER,
        SOURCE_ARRAY_MEMBER,
        SOURCE_INDEX_MEMBER,
        SOURCE_SEAL_MEMBER,
    )
    if all((root / member).is_file() for member in final_members):
        result = load_global_source_prediction_seal(
            root,
            expected_config_hash=config.contract_hash,
            expected_source_cache_binding_hash=source_frame.cache_binding_hash,
        )
        if result.action_library_hash != library_hash:
            raise ProtocolError("Prediction-only source action library drifted on resume.")
        shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return result
    scratch = write_frame_scratch(root, frame_role="source", frame=source_frame)
    tasks = build_source_tasks(
        config,
        generated_sources,
        scratch=scratch,
        action_library_hash=library_hash,
        root=root,
    )
    completed = execute_or_resume_source(
        tasks, workers=int(config.runtime["cpu_workers"])
    )
    classifier_cells, probability_cells = assemble_source_products(
        root, tasks=tasks, completed=completed
    )
    classifier_bank = write_classifier_bank_manifests(
        root,
        cells=classifier_cells,
        config_contract_hash=config.contract_hash,
        source_stream_lock_hash=generated_sources.lock_hash,
        action_library_hash=library_hash,
        source_cache_binding_hash=source_frame.cache_binding_hash,
    )
    flat_rows = tuple(row for center in CENTERS for row in source_frame.rows_by_center[center])
    row_ids = tuple(row.source_row_id for row in flat_rows)
    case_ids = tuple(row.case_id for row in flat_rows)
    query_ids = tuple(row.center for row in flat_rows)
    source_store = write_prediction_store(
        root,
        frame_role="source",
        cells=probability_cells,
        rows_by_outer_target={target: row_ids for target in CENTERS},
        case_ids_by_outer_target={target: case_ids for target in CENTERS},
        query_ids_by_outer_target={target: query_ids for target in CENTERS},
        frame_cache_binding_hash=source_frame.cache_binding_hash,
        action_library_hash=library_hash,
        action_classifier_bank_seal_hash=classifier_bank.seal_hash,
        config_contract_hash=config.contract_hash,
    )
    result = write_source_prediction_seal(
        root,
        classifier_bank=classifier_bank,
        source_store=source_store,
        config_contract_hash=config.contract_hash,
    )
    shutil.rmtree(root / SOURCE_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return result


def issue_test_inference_admission(
    source_seal: object,
    regret_model_bank_seal: RegretModelBankSealLike | Mapping[str, object],
) -> TestInferenceAdmission:
    """Issue the only token accepted by the test-cache loader.

    The caller must provide the durable source-only regret model-bank seal, not
    just an arbitrary hash string.
    """

    if isinstance(regret_model_bank_seal, Mapping):
        model_hash = regret_model_bank_seal.get(
            "regret_model_bank_seal_hash", regret_model_bank_seal.get("seal_hash")
        )
        status = regret_model_bank_seal.get("status")
        source_only = regret_model_bank_seal.get("source_labels_only")
        test_admitted = regret_model_bank_seal.get("test_cache_admitted")
        target_labels = regret_model_bank_seal.get("target_labels_used")
    else:
        model_hash = getattr(regret_model_bank_seal, "seal_hash", None)
        status = getattr(regret_model_bank_seal, "status", None)
        source_only = getattr(regret_model_bank_seal, "source_labels_only", None)
        test_admitted = getattr(regret_model_bank_seal, "test_cache_admitted", None)
        target_labels = getattr(regret_model_bank_seal, "target_labels_used", None)
    target_classifier_bank = getattr(
        source_seal,
        "target_classifier_bank",
        getattr(source_seal, "classifier_bank", None),
    )
    source_payload = dict(getattr(source_seal, "seal_payload", {}))
    if (
        status != "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION"
        or source_only is not True
        or test_admitted is not False
        or target_labels is not False
        or source_payload.get("test_cache_admitted") is not False
        or source_payload.get("source_labels_opened") is not False
        or getattr(target_classifier_bank, "seal_hash", None) is None
    ):
        raise ProtocolError("Prediction-only regret model bank is not admission-ready.")
    unhashed = {
        "schema_version": "midogpp_prediction_only_test_inference_admission_v1",
        "source_prediction_seal_hash": str(getattr(source_seal, "seal_hash")),
        "action_classifier_bank_seal_hash": target_classifier_bank.seal_hash,
        "regret_model_bank_seal_hash": str(model_hash),
        "regret_model_bank_status": str(status),
        "target_labels_available": False,
        "test_scoring_permitted": False,
        "classifier_refit_permitted": False,
    }
    from .hashing import canonical_hash

    return TestInferenceAdmission(
        source_prediction_seal_hash=str(getattr(source_seal, "seal_hash")),
        action_classifier_bank_seal_hash=target_classifier_bank.seal_hash,
        regret_model_bank_seal_hash=str(model_hash),
        regret_model_bank_status=str(status),
        target_labels_available=False,
        test_scoring_permitted=False,
        admission_hash=canonical_hash(unhashed),
    )


def materialize_test_action_predictions(
    config: ActionPredictionConfig,
    source_seal: object,
    test_frame: LabelFreeTestFrame,
    *,
    root: Path,
) -> GlobalTestPredictionSeal:
    """Apply the frozen classifier bank to all test rows with zero refits."""

    _assert_runtime(config.runtime)
    _assert_parent_cuda_free()
    admission = test_frame.admission
    target_classifier_bank = getattr(
        source_seal,
        "target_classifier_bank",
        getattr(source_seal, "classifier_bank", None),
    )
    if (
        admission.source_prediction_seal_hash != getattr(source_seal, "seal_hash", None)
        or admission.action_classifier_bank_seal_hash
        != getattr(target_classifier_bank, "seal_hash", None)
        or getattr(target_classifier_bank, "config_contract_hash", None)
        != config.contract_hash
    ):
        raise ProtocolError("Prediction-only test frame names another frozen source bank.")
    final_members = (TEST_ARRAY_MEMBER, TEST_INDEX_MEMBER, TEST_SEAL_MEMBER)
    if all((root / member).is_file() for member in final_members):
        result = load_global_test_prediction_seal(
            root,
            admission=admission,
            expected_config_hash=config.contract_hash,
            expected_test_cache_binding_hash=test_frame.cache_binding_hash,
        )
        shutil.rmtree(root / TEST_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return result
    scratch = write_frame_scratch(root, frame_role="test", frame=test_frame)
    tasks = build_test_tasks(
        config,
        target_classifier_bank,
        scratch=scratch,
        root=root,
        source_prediction_seal_hash=source_seal.seal_hash,
        regret_model_bank_seal_hash=admission.regret_model_bank_seal_hash,
    )
    completed = execute_or_resume_test(
        tasks, workers=int(config.runtime["cpu_workers"])
    )
    # The test worker may only use the separately sealed q=H-compatible target
    # bank, never the strict H/q development classifier bank.
    cells = assemble_test_cells(
        tasks=tasks,
        completed=completed,
        classifier_bank=target_classifier_bank,
    )
    rows = {
        target: tuple(row.evaluation_row_id for row in test_frame.rows_by_center[target])
        for target in CENTERS
    }
    cases = {
        target: tuple(row.case_id for row in test_frame.rows_by_center[target])
        for target in CENTERS
    }
    queries = {
        target: tuple(target for _row in test_frame.rows_by_center[target])
        for target in CENTERS
    }
    store = write_prediction_store(
        root,
        frame_role="test",
        cells=cells,
        rows_by_outer_target=rows,
        case_ids_by_outer_target=cases,
        query_ids_by_outer_target=queries,
        frame_cache_binding_hash=test_frame.cache_binding_hash,
        action_library_hash=target_classifier_bank.action_library_hash,
        action_classifier_bank_seal_hash=target_classifier_bank.seal_hash,
        config_contract_hash=config.contract_hash,
    )
    result = write_test_prediction_seal(
        root,
        classifier_bank=target_classifier_bank,
        test_store=store,
        admission=admission,
        config_contract_hash=config.contract_hash,
    )
    shutil.rmtree(root / TEST_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return result


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
        raise ProtocolError("Prediction-only workstation runtime drifted.")


def _assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Prediction-only CPU parent must remain CUDA-free.")


__all__ = (
    "GlobalSourcePredictionSeal",
    "GlobalTestPredictionSeal",
    "RegretModelBankSealLike",
    "issue_test_inference_admission",
    "load_global_source_prediction_seal",
    "load_global_test_prediction_seal",
    "materialize_source_action_predictions",
    "materialize_target_action_classifier_bank",
    "materialize_test_action_predictions",
)
