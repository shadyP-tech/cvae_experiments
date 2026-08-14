"""Exact terminal-table, claim-boundary, and report reconstruction."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_rows import row_payload
from .artifact_writers import read_rows
from .constants import CENTERS, CONTROL_METHOD_IDS, METHOD_IDS
from .fresh_process_validation import (
    ATTESTATION_KEY,
    verify_attested_validation_checks,
)
from .persistence import TERMINAL_TABLES
from .reports import leakage_report_payload, publication_decision_payload
from .runtime_adapter import runtime_summary_payload


EXPECTED_TERMINAL_COUNTS = {
    "case_confusions": 218 * len(METHOD_IDS),
    "method_metrics": len(METHOD_IDS),
    "center_metrics": len(CENTERS) * len(METHOD_IDS),
    "contrasts": 4,
    "identification_metrics": 2,
    "calibration_metrics": len(METHOD_IDS),
    "delete_one_center": len(CENTERS),
    "attribution_controls": len(CONTROL_METHOD_IDS),
}


def validate_terminal_products(
    root: Path, *, reconstructed: Mapping[str, object]
) -> Mapping[str, object]:
    tables: dict[str, tuple[dict[str, object], ...]] = {}
    for key, filename in TERMINAL_TABLES.items():
        values = reconstructed.get(key)
        if not isinstance(values, (tuple, list)):
            raise ProtocolError(f"Dual-endpoint terminal table absent: {key}.")
        rows = tuple(row_payload(value) for value in values)
        if len(rows) != EXPECTED_TERMINAL_COUNTS[key]:
            raise ProtocolError(f"Dual-endpoint terminal topology drifted: {key}.")
        if read_rows(root / "tables" / filename) != rows:
            raise ProtocolError(f"Dual-endpoint terminal table drifted: {filename}.")
        tables[key] = rows
    _validate_topology(tables)
    seal = reconstructed.get("terminal_seal")
    if (
        not isinstance(seal, Mapping)
        or read_json(root / "manifests/terminal_evaluation_seal.json") != dict(seal)
        or seal.get("schema_version")
        != "fixed_bank_ogde_terminal_evaluation_seal_v1"
        or seal.get("method_ids") != list(METHOD_IDS)
        or seal.get("terminal_label_count") != 9_928
        or seal.get("raw_labels_persisted") is not False
        or seal.get("terminal_consumed_test_diagnostic") is not True
    ):
        raise ProtocolError("Dual-endpoint terminal seal is not reconstructive.")
    return {
        "terminal_evaluation_seal_hash": seal.get("seal_hash"),
        "terminal_tables_exact": True,
        "terminal_case_confusion_count": len(tables["case_confusions"]),
        "delete_center_full_pipeline_recomputed": True,
    }


def validate_terminal_reports(
    root: Path,
    *,
    config: object,
    preflight: Mapping[str, object],
    prelabel: Mapping[str, object],
    feature_seal: Mapping[str, object],
    aggregate_seal: Mapping[str, object],
    capability_report: Mapping[str, object],
    terminal: Mapping[str, object],
    allow_pending_validation: bool,
) -> None:
    terminal_seal = terminal.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Dual-endpoint terminal seal is absent from replay.")
    expected_leakage = leakage_report_payload(
        physical_prelabel_seal_hash=str(prelabel["physical_prelabel_seal_hash"]),
        feature_seal_hash=str(feature_seal["seal_hash"]),
        aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
        capability_report=capability_report,
    )
    expected_publication = publication_decision_payload(
        str(terminal_seal["seal_hash"]),
        diagnostic_summary=dict(terminal.get("diagnostic_summary", {})),
    )
    expected_runtime = runtime_summary_payload(
        source_cache=prelabel["source"],
        prediction=prelabel["prediction"],
        preflight=preflight,
        runtime=getattr(config, "runtime"),
    )
    if (
        read_json(root / "reports/label_capability_report.json")
        != dict(capability_report)
        or read_json(root / "reports/leakage_report.json") != expected_leakage
        or read_json(root / "reports/publication_decision.json")
        != expected_publication
        or read_json(root / "reports/runtime_summary.json") != expected_runtime
    ):
        raise ProtocolError("Dual-endpoint terminal reports are not reconstructive.")
    _validate_claim_boundary(expected_leakage, expected_publication, expected_runtime)
    _validate_run_state(root, pending=allow_pending_validation)


def validate_final_attestation(
    root: Path, *, checks: Mapping[str, object]
) -> Mapping[str, object]:
    attestation = read_json(root / "reports/fresh_process_attestation.json")
    report = read_json(root / "reports/validation_report.json")
    expected = {**dict(checks), ATTESTATION_KEY: attestation}
    verified = verify_attested_validation_checks(
        report,
        expected_reconstructed_checks=checks,
        persisted_attestation=attestation,
    )
    if dict(report) != dict(expected) or dict(verified) != dict(expected):
        raise ProtocolError("Dual-endpoint final validation report drifted.")
    return verified


def _validate_topology(
    tables: Mapping[str, Sequence[Mapping[str, object]]]
) -> None:
    if (
        tuple(row.get("method_id") for row in tables["method_metrics"])
        != METHOD_IDS
        or tuple(row.get("method_id") for row in tables["calibration_metrics"])
        != METHOD_IDS
        or Counter(row.get("method_id") for row in tables["case_confusions"])
        != Counter({method: 218 for method in METHOD_IDS})
        or Counter(row.get("method_id") for row in tables["center_metrics"])
        != Counter({method: len(CENTERS) for method in METHOD_IDS})
        or tuple(row.get("method_id") for row in tables["identification_metrics"])
        != ("I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED")
        or tuple(row.get("method_id") for row in tables["attribution_controls"])
        != CONTROL_METHOD_IDS
        or tuple(row.get("deleted_center") for row in tables["delete_one_center"])
        != CENTERS
    ):
        raise ProtocolError("Dual-endpoint terminal method topology drifted.")
    expected_contrasts = (
        ("OGDE_PORTFOLIO", "B"),
        ("OGDE_PORTFOLIO", "U"),
        ("OGDE_PORTFOLIO", "R_NINE_ARM_ROBUST"),
        ("OGDE_PORTFOLIO", "CALIBRATION_ONLY_B_R"),
    )
    if tuple(
        (row.get("candidate_method"), row.get("reference_method"))
        for row in tables["contrasts"]
    ) != expected_contrasts or any(
        row.get("G_recomputed") is not True
        or row.get("normalization_recomputed") is not True
        or row.get("identification_reselected") is not True
        or row.get("robust_nine_arms_reselected") is not True
        or row.get("portfolio_recomposed") is not True
        or row.get("deleted_center_removed_from_evaluation") is not True
        or row.get("deleted_center_removed_from_all_G_query_contributions")
        is not True
        or row.get("fixed_expert_bank_preserved") is not True
        for row in tables["delete_one_center"]
    ):
        raise ProtocolError("Dual-endpoint terminal sensitivity topology drifted.")


def _validate_claim_boundary(
    leakage: Mapping[str, object],
    publication: Mapping[str, object],
    runtime: Mapping[str, object],
) -> None:
    if (
        leakage.get("fresh_evidence") is not False
        or leakage.get("raw_labels_persisted") is not False
        or leakage.get("sample_or_image_paths_persisted") is not False
        or publication.get("status") != "POST_HOC_CONSUMED_TEST_SENSITIVITY"
        or publication.get("terminal_decision")
        != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or publication.get("significance_claim_authorized") is not False
        or publication.get("routing_success_claim_authorized") is not False
        or publication.get("fresh_evidence") is not False
        or publication.get("promotion_eligible") is not False
        or publication.get("may_feed_another_experiment") is not False
        or runtime.get("terminal_or_cross_run_recovery_used") is not False
        or runtime.get("prior_run_scratch_used_as_evidence") is not False
        or runtime.get(
            "predecessor_stage90_artifact_checkpoint_or_scratch_reused"
        )
        is not False
    ):
        raise ProtocolError("Dual-endpoint terminal claim boundary drifted.")


def _validate_run_state(root: Path, *, pending: bool) -> None:
    state = read_json(root / "reports/run_state.json")
    expected = {
        "schema_version",
        "status",
        "phase",
        "error",
        "error_class",
        "updated_at_utc",
        "cross_run_recovery_allowed",
        "terminal_recovery_allowed",
    }
    try:
        updated = datetime.fromisoformat(str(state.get("updated_at_utc")))
    except ValueError as exc:
        raise ProtocolError("Dual-endpoint run-state timestamp drifted.") from exc
    if (
        set(state) != expected
        or state.get("status") != ("RUNNING" if pending else "COMPLETE")
        or state.get("phase")
        != ("CONTENT_FIRST_TWO_FRESH_PROCESS_VALIDATION" if pending else "COMPLETE")
        or state.get("error") is not None
        or state.get("error_class") is not None
        or updated.tzinfo is None
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("Dual-endpoint run state drifted.")


__all__ = (
    "EXPECTED_TERMINAL_COUNTS",
    "validate_final_attestation",
    "validate_terminal_products",
    "validate_terminal_reports",
)
