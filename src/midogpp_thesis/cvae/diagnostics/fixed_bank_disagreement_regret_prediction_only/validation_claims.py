"""Exact fail-closed report replay for the terminal diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .constants import EXPECTED_CLASSIFIER_FIT_COUNT
from .development_actions import (
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
)
from .reports import leakage_report_payload, publication_decision_payload
from .validation_common import (
    expected_disjointness_report,
    read_object,
)


def validate_claim_reports(
    root: Path,
    *,
    capability: Mapping[str, object],
    source_prediction_seal_hash: str,
    test_prediction_seal_hash: str,
    model_collection_hash: str,
    frozen_prediction_hash: str,
    source_stream_lock_hash: str,
    source_oof_classifier_bank_seal_hash: str,
    target_classifier_bank_seal_hash: str,
    runtime: Mapping[str, object],
) -> None:
    expected_leakage = leakage_report_payload(
        source_prediction_seal_hash=source_prediction_seal_hash,
        test_prediction_seal_hash=test_prediction_seal_hash,
        source_label_capability_report=capability,
        model_bank_hash=model_collection_hash,
        frozen_test_prediction_hash=frozen_prediction_hash,
    )
    if read_object(root / "reports/leakage_report.json") != expected_leakage:
        raise ProtocolError("Prediction-only leakage report differs from replay.")
    expected_publication = publication_decision_payload(
        frozen_test_prediction_hash=frozen_prediction_hash
    )
    if read_object(root / "reports/publication_decision.json") != expected_publication:
        raise ProtocolError("Prediction-only publication boundary differs from replay.")
    expected_runtime = {
        "schema_version": "midogpp_prediction_only_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_stream_lock_hash,
        "source_oof_classifier_bank_seal_hash": (
            source_oof_classifier_bank_seal_hash
        ),
        "target_compatible_classifier_bank_seal_hash": (
            target_classifier_bank_seal_hash
        ),
        "source_prediction_seal_hash": source_prediction_seal_hash,
        "regret_model_bank_seal_hash": read_object(
            root / "manifests/model_bank_seal.json"
        )["regret_model_bank_seal_hash"],
        "test_prediction_seal_hash": test_prediction_seal_hash,
        "source_stream_count": 81,
        "source_oof_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "source_oof_oriented_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_compatible_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "test_classifier_fit_count": 0,
        "target_labels_available": False,
        "target_scores_computed": False,
        "test_cache_admitted_after_regret_model_bank_seal": True,
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "cpu_classifier_worker_count": int(runtime["cpu_workers"]),
        "blas_threads_per_classifier_worker": int(runtime["threads_per_worker"]),
        "float32_probability_store": True,
        "float64_exact_nine_reductions": True,
        "float64_frozen_classifier_parameters": True,
        "hash_validated_resume": True,
        "workstation_preflight": read_object(
            root / "reports/workstation_preflight.json"
        ),
        "train_test_disjointness": expected_disjointness_report(),
    }
    if read_object(root / "reports/runtime_summary.json") != expected_runtime:
        raise ProtocolError("Prediction-only runtime lineage report drifted.")


__all__ = ("validate_claim_reports",)
