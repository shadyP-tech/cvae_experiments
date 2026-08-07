from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router import (
    bundle_validation,
    plan_artifacts,
    plan_validation,
    planning,
    reports,
    runner,
    validation,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.artifact_io import (
    atomic_write_json,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.contracts import (
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _plans() -> dict[str, dict[str, object]]:
    return {
        f"fold-{ordinal}": {"used_uniform_fallback": ordinal % 2 == 0}
        for ordinal in range(EXPECTED_CROSS_FIT_FOLD_COUNT)
    }


def _scoring() -> dict[str, float]:
    return {
        "mean_equal_union_bacc_center_equal": 0.7,
        "mean_antisymmetric_residual_mmd_bacc_center_equal": 0.71,
        "mean_paired_bacc_delta_center_equal": 0.01,
    }


def _preflight() -> dict[str, object]:
    return {
        "status": "PASS",
        "gpus": [{"index": 0}, {"index": 1}],
        "classifier_worker_thread_product": 12,
        "parent_cuda_context_initialized": False,
    }


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        runtime={
            "workstation_profile": "dual_rtx_a5000_w2265",
            "generation_devices": ["cuda:0", "cuda:1"],
            "kernel_devices": ["cuda:0", "cuda:1"],
            "classifier_workers": 4,
            "classifier_threads_per_worker": 3,
            "maximum_unique_classifier_fit_count": (
                MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
            ),
            "resume_policy": "resume_hash_validated_checkpoints",
        }
    )


def test_facades_delegate_to_cohesive_modules_without_public_api_drift() -> None:
    assert planning.AntisymmetricRouterPlans is plan_artifacts.AntisymmetricRouterPlans
    assert (
        planning.load_antisymmetric_router_plans
        is plan_artifacts.load_antisymmetric_router_plans
    )
    assert planning.ROUTER_PLAN_COLUMNS is plan_artifacts.ROUTER_PLAN_COLUMNS
    assert runner._protocol_manifest is reports._protocol_manifest
    assert runner._write_content_index is reports._write_content_index
    assert validation._validate_plans is plan_validation._validate_plans
    assert (
        validation._validate_content_index
        is bundle_validation._validate_content_index
    )
    assert planning.__all__ == (
        "ROUTER_PLAN_COLUMNS",
        "ROUTER_PLAN_LOCK_MEMBER",
        "ROUTER_PLAN_TABLE_MEMBER",
        "ROUTER_STATE_MEMBER",
        "TARGET_ASSIGNMENT_COLUMNS",
        "TARGET_ASSIGNMENT_MEMBER",
        "AntisymmetricRouterPlans",
        "build_antisymmetric_router_plans",
        "load_antisymmetric_router_plans",
    )
    assert validation.__all__ == (
        "validate_antisymmetric_residual_mmd_router_bundle",
    )


def test_plan_checkpoint_round_trip_preserves_hash_and_task_bindings(
    tmp_path: Path,
) -> None:
    unhashed = {
        "fold_id": "fold-0",
        "target_center": "0",
        "config_contract_hash": "config",
        "crossfit_partition_lock_hash": "crossfit",
        "source_products_hash": "products",
        "source_products_lock_hash": "product-lock",
    }
    plan = {**unhashed, "plan_hash": stable_hash(unhashed)}
    task = {
        "folds": ({"fold_id": "fold-0"},),
        "target_center": "0",
        "config_contract_hash": "config",
        "crossfit_partition_lock_hash": "crossfit",
        "source_products_hash": "products",
        "source_products_lock_hash": "product-lock",
    }
    path = tmp_path / "checkpoints/routes/target_0.npz"
    state = {"kernel": np.asarray([[1.0, 2.0]], dtype=np.float64)}
    plan_artifacts._write_target_checkpoint(path, plans=[plan], state=state)

    loaded_plans, loaded_state = plan_artifacts._load_target_checkpoint(
        path,
        task=task,
    )
    assert loaded_plans == (plan,)
    np.testing.assert_array_equal(loaded_state["kernel"], state["kernel"])

    with pytest.raises(ProtocolError, match="binding drifted"):
        plan_artifacts._load_target_checkpoint(
            path,
            task={**task, "source_products_lock_hash": "tampered"},
        )


def test_reports_preserve_counts_claim_boundary_and_preflight_evidence(
    tmp_path: Path,
) -> None:
    assert EXPECTED_CROSS_FIT_FOLD_COUNT == 26
    assert EXPECTED_PREDICTION_CELL_COUNT == 468
    assert 9 * EXPECTED_SEED_CELL_COUNT * 2 == 162
    assert 9 * EXPECTED_SEED_CELL_COUNT == 81

    phase = reports._phase_payload("TEST_PHASE", count=26)
    phase_unhashed = {
        key: value for key, value in phase.items() if key != "phase_hash"
    }
    assert phase["phase_hash"] == stable_hash(phase_unhashed)
    assert phase["diagnostic_only"] is True
    assert phase["fresh_evidence"] is False
    assert phase["promotion_eligible"] is False

    publication = reports._publication_decision(_scoring(), plans=_plans())
    assert publication["crossfit_fold_count"] == 26
    assert publication["may_feed_stage60"] is False
    assert publication["may_feed_stage70"] is False
    assert publication["routing_quality_claimed"] is False

    runtime = reports._runtime_summary(
        _runtime_config(),
        elapsed_seconds=1.25,
        unique_classifier_fit_count=81,
        workstation_preflight=_preflight(),
    )
    assert runtime["source_block_count"] == 81
    assert runtime["prediction_task_count"] == 81
    assert runtime["workstation_preflight"] == _preflight()

    report_root = tmp_path / "reports"
    atomic_write_json(report_root / "leakage_report.json", reports._leakage_report())
    atomic_write_json(report_root / "publication_decision.json", publication)
    atomic_write_json(report_root / "runtime_summary.json", runtime)
    bundle_validation._validate_claim_reports(tmp_path)

    runtime["workstation_preflight"] = {
        **_preflight(),
        "parent_cuda_context_initialized": True,
    }
    atomic_write_json(report_root / "runtime_summary.json", runtime)
    with pytest.raises(ProtocolError, match="runtime contract drifted"):
        bundle_validation._validate_claim_reports(tmp_path)
