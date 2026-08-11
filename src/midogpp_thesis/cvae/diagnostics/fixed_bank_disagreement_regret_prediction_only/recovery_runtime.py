"""Label-free continuation after validated post-test prediction seals."""

from __future__ import annotations

from pathlib import Path

from .execution_adapter import aggregate_probability_rows, probability_views
from .inference import build_test_inference_products
from .persistence import persist_inference_products
from .recovery_contracts import rollback_post_test_seal_recovery
from .recovery_provenance import assert_repair_repository_state_unchanged
from .runner_runtime import (
    enter_cuda_free_cpu_phase,
    observe,
    write_state,
)
from .terminal_runtime import finalize_prediction_only_bundle


def resume_post_test_seal(
    root: Path,
    *,
    config: object,
    protocol: object,
    recovery: object,
    dependencies: object,
    default_validator: object,
) -> Path:
    """Continue only aggregate, inference persistence, and terminal validation."""

    deps = dependencies
    phase = "POST_TEST_SEAL_INFERENCE_RECOVERY"
    write_state(deps, root, status="RUNNING", phase=phase)
    try:
        enter_cuda_free_cpu_phase()
        observe(deps, "post_test_seal_recovery")
        test_predictions = getattr(recovery, "test_predictions")
        development = getattr(recovery, "development")
        test_views = probability_views(
            getattr(deps, "aggregate_test_probabilities")
            or aggregate_probability_rows,
            test_predictions,
            frame_role="test",
        )
        inference = (
            getattr(deps, "build_inference") or build_test_inference_products
        )(
            development,
            test_views,
            test_prediction_seal_hash=test_predictions.seal_hash,
            target_cache_content_hash=str(
                getattr(config, "expected_test_cache_content_hash")
            ),
            target_cache_order_hash=str(
                getattr(config, "expected_test_cache_row_order_hash")
            ),
        )
        (getattr(deps, "persist_inference") or persist_inference_products)(
            root, inference
        )
        observe(deps, "recovered_inference_persisted")

        phase = "VALIDATION_RECOVERY"
        finalize_prediction_only_bundle(
            root,
            config=config,
            protocol=protocol,
            dependencies=deps,
            default_validator=default_validator,
            generated_sources=getattr(recovery, "generated_sources"),
            source_predictions=getattr(recovery, "source_predictions"),
            test_predictions=test_predictions,
            source_label_capability_report=getattr(
                recovery, "source_label_capability_report"
            ),
            model_bank_hash=development.model_bank_hash,
            inference=inference,
            preflight=getattr(recovery, "workstation_preflight"),
            disjointness=getattr(recovery, "train_test_disjointness"),
            recovery_audit=getattr(recovery, "audit"),
            validation_phase=phase,
            revalidate_complete=False,
        )
        (
            getattr(deps, "validate_recovery_checkout")
            or assert_repair_repository_state_unchanged
        )(getattr(recovery, "audit"))
        return root
    except BaseException:
        rollback_post_test_seal_recovery(root)
        raise


__all__ = ("resume_post_test_seal",)
