"""Shared terminal report, index, and validation persistence."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .execution_adapter import runtime_summary_payload
from .persistence import (
    persist_reports,
    persist_validation_report,
)
from .reports import leakage_report_payload, publication_decision_payload
from .runner_runtime import write_state


def finalize_prediction_only_bundle(
    root: Path,
    *,
    config: object,
    protocol: object,
    dependencies: object,
    default_validator: object,
    generated_sources: object,
    source_predictions: object,
    test_predictions: object,
    source_label_capability_report: object,
    model_bank_hash: str,
    inference: object,
    preflight: object,
    disjointness: object,
    recovery_audit: object,
    validation_phase: str,
    revalidate_complete: bool,
) -> None:
    """Persist reports and validate only after inference is immutable."""

    deps = dependencies
    write_state(deps, root, status="RUNNING", phase=validation_phase)
    leakage = leakage_report_payload(
        source_prediction_seal_hash=source_predictions.seal_hash,
        test_prediction_seal_hash=test_predictions.seal_hash,
        source_label_capability_report=source_label_capability_report,
        model_bank_hash=model_bank_hash,
        frozen_test_prediction_hash=inference.frozen_prediction_hash,
    )
    publication = publication_decision_payload(
        frozen_test_prediction_hash=inference.frozen_prediction_hash
    )
    runtime_summary = dict(
        (getattr(deps, "build_runtime_summary") or runtime_summary_payload)(
            generated_sources=generated_sources,
            source_predictions=source_predictions,
            test_predictions=test_predictions,
            runtime=getattr(config, "runtime"),
        )
    )
    runtime_summary["workstation_preflight"] = dict(preflight)
    runtime_summary["train_test_disjointness"] = dict(disjointness)
    runtime_summary["post_test_seal_recovery"] = dict(recovery_audit)
    (getattr(deps, "persist_reports") or persist_reports)(
        root,
        leakage=leakage,
        publication=publication,
        runtime_summary=runtime_summary,
    )
    (getattr(deps, "write_content_index") or write_content_index)(
        root,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    validator = getattr(deps, "validate_bundle") or default_validator
    if not callable(validator):
        raise ProtocolError("Prediction-only bundle validator is not callable.")
    checks = validator(root, config=config)
    (getattr(deps, "persist_validation") or persist_validation_report)(
        root, checks
    )
    write_state(deps, root, status="COMPLETE", phase="COMPLETE")
    if revalidate_complete:
        validator(root, config=config)
    else:
        (
            getattr(deps, "validate_terminal_completion")
            or _assert_terminal_completion
        )(root)


def _assert_terminal_completion(root: Path) -> None:
    """Verify the final state transition without repeating the source refit."""

    assert_closed_world(root, allow_incomplete=False)
    state = read_json(root / "reports/run_state.json")
    validation = read_json(root / "reports/validation_report.json")
    if (
        state
        != {
            "schema_version": (
                "midogpp_disagreement_regret_prediction_only_run_state_v1"
            ),
            "status": "COMPLETE",
            "phase": "COMPLETE",
            "prediction_only": True,
            "test_labels_opened": False,
        }
        or validation.get("status") != "PASS"
        or validation.get("test_labels_opened") is not False
        or validation.get("test_metrics_computed") is not False
        or validation.get("routing_or_promotion_authorized") is not False
    ):
        raise ProtocolError("Prediction-only terminal completion drifted.")


__all__ = ("finalize_prediction_only_bundle",)
