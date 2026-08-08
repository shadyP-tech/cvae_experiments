"""Independent reconstructive validation of a completed case-OOF bundle."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import assert_closed_world, read_json
from .bundle import REQUIRED_FILES, validate_content_index
from .config import ResidualTopupCaseOOFConfig, load_residual_topup_case_oof_config
from .contracts import (
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPECTED_PROXY_SCORE_COUNT,
    EXPECTED_SEALED_PREDICTION_CELL_COUNT,
)
from .inference import (
    CONTRAST_INFERENCE_COLUMNS,
    ORACLE_HXE_COLUMNS,
    PRIMARY_CONTRAST_COLUMNS,
    build_center_contrasts,
    build_oracle_hxe_diagnostics,
    infer_center_contrasts,
)
from .inputs import (
    SUPPORT_PARTITION_COLUMNS,
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_access import open_evaluation_labels_after_global_seal
from .partitions import CASE_OOF_FOLD_COLUMNS, build_case_oof_surface
from .planning import build_case_oof_plan
from .prediction_planning import EXPECTED_PREDICTION_TASK_COUNT
from .prediction_store import read_prediction_store
from .prediction_validation import validate_prediction_store_binding
from .ranking import build_rank_surface
from .reports import (
    ACTION_ASSIGNMENT_COLUMNS,
    ACTION_PLAN_COLUMNS,
    PROXY_BALLOT_COLUMNS,
    PROXY_RANK_COLUMNS,
    action_assignment_rows,
    action_plan_rows,
    leakage_report_payload,
    phase_completion_payload,
    protocol_manifest_payload,
    proxy_ballot_rows,
    proxy_rank_rows,
    publication_decision_payload,
    runtime_summary_payload,
    scoring_summary_payload,
)
from .scoring import (
    CENTER_ENSEMBLE_METRIC_COLUMNS,
    CENTER_SEED_METRIC_COLUMNS,
    score_center_probability_ensembles,
    score_center_seed_cells,
)
from .seals import validate_global_prediction_seal
from .source_cache import (
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    load_source_cache,
    validate_source_cache_lock,
)


def validate_residual_topup_case_oof_bundle(
    root: str | Path,
    *,
    config: ResidualTopupCaseOOFConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Rebuild every scientific surface and reject any byte/content drift."""

    path = Path(root).resolve()
    _require_inventory(path, allow_pending=allow_pending)
    resolved = load_residual_topup_case_oof_config(path / "config.resolved.yaml")
    if config is not None:
        _validate_config_equivalence(resolved, config)

    workspace_binding = validate_active_diagnostic_workspace_binding(resolved)
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    firewall = {
        **validate_pre_gpu_firewall(resolved, frame),
        "workspace_binding": workspace_binding,
    }
    base = build_partition_surface(frame, config_contract_hash=resolved.contract_hash)
    crossfit = build_case_oof_surface(
        base, config_contract_hash=resolved.contract_hash
    )
    _assert_json(path / "manifests/support_partition_lock.json", base.lock_payload)
    _assert_csv(
        path / "tables/support_partitions.csv",
        base.table_rows,
        columns=SUPPORT_PARTITION_COLUMNS,
    )
    _assert_json(path / "manifests/crossfit_fold_lock.json", crossfit.lock_payload)
    _assert_csv(
        path / "tables/crossfit_folds.csv",
        crossfit.table_rows,
        columns=CASE_OOF_FOLD_COLUMNS,
    )
    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in resolved.input_artifact_ids
    }
    _assert_json(
        path / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            resolved,
            input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=frame.cache_binding_hash,
            pre_gpu_firewall=firewall,
        ),
    )

    source_cache = load_source_cache(path)
    source_lock = validate_source_cache_lock(
        path,
        config=resolved,
        generation_lock=locks.generation,
        frame=frame,
        crossfit=crossfit,
        source_cache=source_cache,
    )
    source_lock_hash = str(source_lock["source_cache_lock_hash"])
    proxy_rows = source_cache.proxy_score_rows(crossfit)
    if len(proxy_rows) != EXPECTED_PROXY_SCORE_COUNT:
        raise ProtocolError("Case-OOF reconstructed proxy grid drifted.")
    rank_surface = build_rank_surface(proxy_rows, crossfit)
    plan = build_case_oof_plan(
        rank_surface, crossfit, config_contract_hash=resolved.contract_hash
    )
    _assert_json(path / "manifests/action_library.json", plan.action_library_payload)
    _assert_json(path / "manifests/router_plan_lock.json", plan.lock_payload)
    _assert_csv(
        path / "tables/proxy_case_ballots.csv",
        proxy_ballot_rows(rank_surface),
        columns=PROXY_BALLOT_COLUMNS,
    )
    _assert_csv(
        path / "tables/proxy_rank_actions.csv",
        proxy_rank_rows(rank_surface),
        columns=PROXY_RANK_COLUMNS,
    )
    _assert_csv(
        path / "tables/action_plans.csv",
        action_plan_rows(plan),
        columns=ACTION_PLAN_COLUMNS,
    )
    _assert_csv(
        path / "tables/action_assignments.csv",
        action_assignment_rows(plan),
        columns=ACTION_ASSIGNMENT_COLUMNS,
    )

    predictions = read_prediction_store(path)
    validate_prediction_store_binding(
        predictions,
        config=resolved,
        generation_lock_hash=locks.generation.generation_lock_hash,
        source_cache=source_cache,
        source_cache_lock_hash=source_lock_hash,
        plan=plan,
        crossfit=crossfit,
    )
    seal = validate_global_prediction_seal(
        resolved,
        crossfit,
        plan,
        predictions,
        source_cache_lock_hash=source_lock_hash,
        root=path,
    )
    seal_hash = str(seal["seal_hash"])
    labels, label_report = open_evaluation_labels_after_global_seal(
        resolved,
        crossfit,
        plan,
        predictions,
        source_cache_lock_hash=source_lock_hash,
        root=path,
    )
    _assert_json(path / "reports/label_access_report.json", label_report)
    center_seed_rows = score_center_seed_cells(
        predictions, labels_by_sample_id=labels, crossfit=crossfit
    )
    ensemble_rows = score_center_probability_ensembles(
        predictions, labels_by_sample_id=labels, crossfit=crossfit
    )
    contrast_rows = build_center_contrasts(ensemble_rows)
    inference_rows = infer_center_contrasts(contrast_rows)
    oracle_rows = build_oracle_hxe_diagnostics(
        ensemble_rows, rank_surface=rank_surface
    )
    _assert_csv(
        path / "tables/center_seed_metrics.csv",
        center_seed_rows,
        columns=CENTER_SEED_METRIC_COLUMNS,
    )
    _assert_csv(
        path / "tables/center_ensemble_metrics.csv",
        ensemble_rows,
        columns=CENTER_ENSEMBLE_METRIC_COLUMNS,
    )
    _assert_csv(
        path / "tables/primary_contrasts.csv",
        contrast_rows,
        columns=PRIMARY_CONTRAST_COLUMNS,
    )
    _assert_csv(
        path / "tables/contrast_inference.csv",
        inference_rows,
        columns=CONTRAST_INFERENCE_COLUMNS,
    )
    _assert_csv(
        path / "tables/oracle_hxe_diagnostics.csv",
        oracle_rows,
        columns=ORACLE_HXE_COLUMNS,
    )

    summary = scoring_summary_payload(ensemble_rows, inference_rows, oracle_rows)
    _assert_json(
        path / "reports/leakage_report.json",
        leakage_report_payload(
            support_partition_lock_hash=base.lock_hash,
            crossfit_fold_lock_hash=crossfit.lock_hash,
            source_cache_lock_hash=source_lock_hash,
            router_plan_lock_hash=plan.lock_hash,
            global_prediction_seal_hash=seal_hash,
            pre_gpu_firewall=firewall,
        ),
    )
    _assert_json(
        path / "reports/publication_decision.json",
        publication_decision_payload(summary),
    )
    _validate_phase_reports(
        path,
        resolved=resolved,
        source_cache=source_cache,
        source_lock_hash=source_lock_hash,
        plan=plan,
        predictions=predictions,
        seal_hash=seal_hash,
        center_seed_rows=center_seed_rows,
        ensemble_rows=ensemble_rows,
        contrast_rows=contrast_rows,
        inference_rows=inference_rows,
        oracle_rows=oracle_rows,
    )
    _validate_runtime_report(path, predictions=predictions)
    validate_content_index(path, config_contract_hash=resolved.contract_hash)
    _validate_final_state(path, allow_pending=allow_pending)

    return {
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "workspace_binding_verified": True,
        "pre_gpu_data_firewall_verified": True,
        "fixed_support_case_oof_reconstructed": True,
        "source_cache_lock_verified": True,
        "proxy_rank_and_plan_reconstructed": True,
        "prediction_store_and_global_seal_verified": True,
        "terminal_scores_and_inference_recomputed": True,
        "content_index_verified": True,
        "fold_count": len(crossfit.folds),
        "prediction_cell_count": len(predictions.index_rows),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
    }


def _validate_phase_reports(
    path: Path,
    *,
    resolved: object,
    source_cache: object,
    source_lock_hash: str,
    plan: object,
    predictions: object,
    seal_hash: str,
    center_seed_rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
    contrast_rows: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> None:
    contract_hash = str(getattr(resolved, "contract_hash"))
    _assert_json(
        path / "reports/phase_01_source_cache_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_complete",
            config_contract_hash=contract_hash,
            bindings={"source_cache_lock_hash": source_lock_hash},
            counts={
                "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
                "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
                "compatibility_case_row_count": len(
                    getattr(source_cache, "compatibility_case_rows")
                ),
                "expanded_proxy_score_count": EXPECTED_PROXY_SCORE_COUNT,
            },
            labels_opened=False,
        ),
    )
    _assert_json(
        path / "reports/phase_02_all_predictions_sealed.json",
        phase_completion_payload(
            "phase_02_all_predictions_sealed",
            config_contract_hash=contract_hash,
            bindings={
                "router_plan_lock_hash": str(getattr(plan, "lock_hash")),
                "global_prediction_seal_hash": seal_hash,
            },
            counts={
                "frozen_action_count": EXPECTED_FROZEN_ACTION_COUNT,
                "prediction_cell_count": EXPECTED_SEALED_PREDICTION_CELL_COUNT,
                "unique_classifier_fit_count": int(
                    getattr(predictions, "unique_classifier_fit_count")
                ),
            },
            labels_opened=False,
        ),
    )
    _assert_json(
        path / "reports/phase_03_terminal_scoring_complete.json",
        phase_completion_payload(
            "phase_03_terminal_scoring_complete",
            config_contract_hash=contract_hash,
            bindings={"global_prediction_seal_hash": seal_hash},
            counts={
                "center_seed_metric_count": len(center_seed_rows),
                "center_ensemble_metric_count": len(ensemble_rows),
                "center_contrast_count": len(contrast_rows),
                "contrast_inference_count": len(inference_rows),
                "oracle_target_count": len(oracle_rows),
            },
            labels_opened=True,
        ),
    )


def _validate_runtime_report(path: Path, *, predictions: object) -> None:
    observed = read_json(path / "reports/runtime_summary.json")
    preflight = observed.get("workstation_preflight")
    if not isinstance(preflight, Mapping) or preflight.get("status") != "PASS":
        raise ProtocolError("Case-OOF runtime preflight report drifted.")
    expected = runtime_summary_payload(
        preflight,
        source_task_count=EXPECTED_SOURCE_TASK_COUNT,
        source_block_count=EXPECTED_SOURCE_BLOCK_COUNT,
        prediction_task_count=EXPECTED_PREDICTION_TASK_COUNT,
        prediction_cell_count=EXPECTED_SEALED_PREDICTION_CELL_COUNT,
        unique_classifier_fit_count=int(
            getattr(predictions, "unique_classifier_fit_count")
        ),
    )
    if observed != expected:
        raise ProtocolError("Case-OOF runtime summary drifted.")


def _require_inventory(path: Path, *, allow_pending: bool) -> None:
    assert_closed_world(path, required_files=REQUIRED_FILES, allow_incomplete=False)
    allowed_missing = {"reports/validation_report.json"} if allow_pending else set()
    missing = [
        member
        for member in REQUIRED_FILES
        if member not in allowed_missing and not (path / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Case-OOF bundle members are missing: {missing}.")


def _validate_final_state(path: Path, *, allow_pending: bool) -> None:
    state = read_json(path / "reports/run_state.json")
    if allow_pending:
        if state.get("status") not in {"RUNNING", "COMPLETE"}:
            raise ProtocolError("Case-OOF pending validation state drifted.")
    elif state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
        raise ProtocolError("Case-OOF final run state is not complete.")
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        if (
            report.get("status") != "PASS"
            or report.get("validator") != "validate_residual_topup_case_oof_bundle"
        ):
            raise ProtocolError("Case-OOF validation report drifted.")


def _validate_config_equivalence(
    resolved: ResidualTopupCaseOOFConfig,
    supplied: ResidualTopupCaseOOFConfig,
) -> None:
    path_fields = (
        "artifact_root",
        "expert_bank_root",
        "generation_lock_root",
        "equal_union_policy_root",
        "validation_cache_root",
        "validation_manifest_path",
    )
    if resolved.contract_hash != supplied.contract_hash or any(
        Path(getattr(resolved, field)).resolve()
        != Path(getattr(supplied, field)).resolve()
        for field in path_fields
    ):
        raise ProtocolError("Case-OOF supplied/resolved config binding drifted.")


def _assert_json(path: Path, expected: Mapping[str, object]) -> None:
    if read_json(path) != expected:
        raise ProtocolError(f"Case-OOF derived JSON drifted: {path.name}.")


def _assert_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    expected_handle = StringIO(newline="")
    writer = csv.DictWriter(
        expected_handle, fieldnames=list(columns), extrasaction="raise"
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ProtocolError("Case-OOF reconstructed CSV schema drifted.")
        writer.writerow(row)
    try:
        observed = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot read case-OOF CSV: {path}.") from exc
    if observed != expected_handle.getvalue():
        raise ProtocolError(f"Case-OOF derived CSV drifted: {path.name}.")


__all__ = ("validate_residual_topup_case_oof_bundle",)
