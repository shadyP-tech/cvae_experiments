"""Deterministic phase persistence for the ensemble-endpoint run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import atomic_json, persist_or_validate_csv, persist_or_validate_json
from .contracts import CENTERS
from .development_scoring import DEVELOPMENT_SEED_COLUMNS
from .partitions import SUPPORT_PARTITION_COLUMNS
from .reports import phase_completion_payload, protocol_manifest_payload, run_state_payload
from .terminal_scoring import TARGET_SEED_COLUMNS


CASE_FOLD_COLUMNS = (
    "schema_version", "fold_ordinal", "fold_id", "target_center", "heldout_case_id",
    "fixed_support_case_ids_json", "fixed_support_row_identity_hash",
    "heldout_row_identity_hash", "fold_hash", "support_partition_namespace",
    "support_labels_used", "evaluation_labels_used_for_route",
    "other_evaluation_embeddings_used_for_route",
)


def persist_initial_surfaces(
    root: Path, *, config: object, provenance: Mapping[str, Mapping[str, object]],
    frame: object, firewall: Mapping[str, object], partitions: object, case_folds: object,
) -> None:
    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    persist_or_validate_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config, input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=str(frame.cache_binding_hash), firewall=firewall,
        ),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv", partitions.table_rows, SUPPORT_PARTITION_COLUMNS
    )
    persist_or_validate_json(root / "manifests/support_partition_lock.json", partitions.lock_payload)
    persist_or_validate_json(root / "manifests/case_fold_lock.json", case_folds.lock_payload)
    persist_or_validate_csv(root / "tables/case_folds.csv", _case_fold_rows(case_folds), CASE_FOLD_COLUMNS)


def persist_feature_surfaces(
    root: Path, *, config_contract_hash: str, source_cache_lock_hash: str,
    seed_production: object, features: object, inner_shift_lock_hash: str,
    target_shift_lock_hash: str, target_probe_seal_hash: str,
) -> None:
    persist_or_validate_json(root / "manifests/feature_surface_set.json", features.to_payload())
    inner_rows = tuple(
        row for target in CENTERS for row in features.by_target[target].inner_m1.rows
    )
    target_rows = tuple(
        row for target in CENTERS for row in features.by_target[target].target_m1.rows
    )
    _persist_payload_csv(root / "tables/inner_ensemble_features.csv", inner_rows)
    _persist_payload_csv(root / "tables/target_ensemble_features.csv", target_rows)
    persist_or_validate_json(
        root / "reports/phase_01_source_cache_and_features_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_and_features_complete",
            config_contract_hash=config_contract_hash,
            bindings={
                "source_cache_lock_hash": source_cache_lock_hash,
                "seed_feature_production_hash": seed_production.production_hash,
                "feature_surface_set_hash": features.surface_hash,
                "source_inner_support_shift_lock_hash": inner_shift_lock_hash,
                "target_support_shift_lock_hash": target_shift_lock_hash,
                "target_probe_seal_hash": target_probe_seal_hash,
            },
            counts={"inner_ensemble_feature_count": len(inner_rows), "target_ensemble_feature_count": len(target_rows)},
            development_labels_opened=False, terminal_target_labels_opened=False,
        ),
    )


def persist_development_and_router_surfaces(
    root: Path, *, config_contract_hash: str, development_labels: object,
    utility_surface: object, seed_rows: Sequence[Mapping[str, object]], models: object,
    plans: object, actions: object, development_prediction_seal_hash: str,
) -> None:
    persist_or_validate_json(root / "reports/development_label_access_report.json", {
        "schema_version": "midogpp_stage90_ensemble_endpoint_development_label_access_report_v1",
        "status": "PASS", "prediction_seal_hash": development_labels.prediction_seal_hash,
        "manifest_sha256": development_labels.manifest_sha256,
        "capability_hash": development_labels.capability_hash,
        "evaluation_row_hash_by_center": dict(development_labels.evaluation_row_hash_by_center),
        "label_hash_by_center": dict(development_labels.label_hash_by_center),
        "support_labels_opened": False, "labels_persisted": False,
    })
    _persist_payload_csv(root / "tables/source_inner_ensemble_endpoints.csv", utility_surface.rows)
    persist_or_validate_csv(root / "tables/source_inner_seed_diagnostics.csv", seed_rows, DEVELOPMENT_SEED_COLUMNS)
    persist_or_validate_json(root / "manifests/model_set.json", models.to_payload())
    _persist_named_payload_csv(root / "tables/model_summary.csv", models.by_target)
    persist_or_validate_json(root / "manifests/diagnostic_plan_set.json", plans.to_payload())
    _persist_named_payload_csv(root / "tables/diagnostic_plans.csv", plans.by_target)
    persist_or_validate_json(root / "manifests/action_library.json", actions.to_payload())
    action_rows = tuple(action for target in CENTERS for action in actions.actions_by_target[target])
    _persist_payload_csv(root / "tables/target_actions.csv", action_rows)
    persist_or_validate_json(
        root / "reports/phase_02_development_scoring_and_action_lock_complete.json",
        phase_completion_payload(
            "phase_02_development_scoring_and_action_lock_complete",
            config_contract_hash=config_contract_hash,
            bindings={
                "development_prediction_seal_hash": development_prediction_seal_hash,
                "ensemble_utility_surface_hash": utility_surface.surface_hash,
                "model_set_hash": models.model_set_hash, "plan_set_hash": plans.plan_set_hash,
                "action_library_hash": actions.action_library_hash,
            },
            counts={"primary_endpoint_response_count": len(utility_surface.rows), "descriptive_seed_row_count": len(seed_rows), "target_action_count": actions.action_count},
            development_labels_opened=True, terminal_target_labels_opened=False,
        ),
    )


def persist_target_seal_phase(
    root: Path, *, config_contract_hash: str, capability: object,
    prediction_cell_count: int, unique_classifier_fit_count: int,
) -> None:
    payload = capability.payload
    persist_or_validate_json(
        root / "reports/phase_03_global_target_prediction_seal_complete.json",
        phase_completion_payload(
            "phase_03_global_target_prediction_seal_complete",
            config_contract_hash=config_contract_hash,
            bindings={"action_library_hash": payload["action_library_hash"], "global_target_prediction_seal_hash": payload["seal_hash"], "target_probe_seal_hash": payload["target_probe_seal_hash"]},
            counts={"target_prediction_cell_count": prediction_cell_count, "unique_classifier_fit_count": unique_classifier_fit_count},
            development_labels_opened=True, terminal_target_labels_opened=False,
        ),
    )


def persist_terminal_surfaces(
    root: Path, *, config_contract_hash: str, target_label_report: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, object]], scores: object,
    center_contrasts: Sequence[Mapping[str, object]], inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[object], leakage_report: Mapping[str, object],
    scoring_summary: Mapping[str, object], publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object], target_seal_hash: str,
) -> None:
    persist_or_validate_json(root / "reports/target_label_access_report.json", target_label_report)
    persist_or_validate_csv(root / "tables/target_seed_diagnostics.csv", seed_rows, TARGET_SEED_COLUMNS)
    _persist_payload_csv(root / "tables/target_ensemble_metrics.csv", scores.rows)
    _persist_mapping_csv(root / "tables/center_contrasts.csv", center_contrasts)
    _persist_mapping_csv(root / "tables/contrast_inference.csv", inference_rows)
    _persist_payload_csv(root / "tables/oracle_hxe_diagnostics.csv", oracle_rows)
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(root / "reports/scoring_summary.json", scoring_summary)
    persist_or_validate_json(root / "reports/publication_decision.json", publication_decision)
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)
    persist_or_validate_json(
        root / "reports/phase_04_terminal_scoring_complete.json",
        phase_completion_payload(
            "phase_04_terminal_scoring_complete", config_contract_hash=config_contract_hash,
            bindings={"global_target_prediction_seal_hash": target_seal_hash, "target_score_set_hash": scores.score_set_hash},
            counts={"target_seed_diagnostic_count": len(seed_rows), "target_endpoint_count": len(scores.rows), "center_contrast_count": len(center_contrasts), "oracle_target_count": len(oracle_rows)},
            development_labels_opened=True, terminal_target_labels_opened=True,
        ),
    )


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", checks)


def write_run_state(root: Path, *, status: str, phase: str, error: str | None = None) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _case_fold_rows(surface: object) -> tuple[dict[str, object], ...]:
    namespace = surface.lock_payload["support_partition_namespace"]
    return tuple({
        "schema_version": "midogpp_stage90_ensemble_endpoint_case_fold_row_v1",
        "fold_ordinal": fold.fold_ordinal, "fold_id": fold.fold_id,
        "target_center": fold.target_center, "heldout_case_id": fold.heldout_case_id,
        "fixed_support_case_ids_json": json.dumps(sorted({row.case_id for row in fold.fixed_support_rows}), separators=(",", ":")),
        "fixed_support_row_identity_hash": stable_hash([row.identity_payload() for row in fold.fixed_support_rows]),
        "heldout_row_identity_hash": fold.heldout_row_identity_hash, "fold_hash": fold.fold_hash,
        "support_partition_namespace": namespace, "support_labels_used": False,
        "evaluation_labels_used_for_route": False,
        "other_evaluation_embeddings_used_for_route": False,
    } for fold in surface.folds)


def _table_value(value: object) -> object:
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _payload(value: object) -> dict[str, object]:
    raw = value.to_payload()
    return {str(key): _table_value(item) for key, item in raw.items()}


def _persist_payload_csv(path: Path, values: Sequence[object]) -> None:
    rows = tuple(_payload(value) for value in values)
    if not rows: raise ValueError(f"Cannot persist empty scientific table: {path}.")
    persist_or_validate_csv(path, rows, tuple(rows[0]))


def _persist_named_payload_csv(path: Path, values: Mapping[str, object]) -> None:
    _persist_payload_csv(path, tuple(values[key] for key in CENTERS))


def _persist_mapping_csv(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    rows = tuple({str(key): _table_value(item) for key, item in row.items()} for row in values)
    if not rows: raise ValueError(f"Cannot persist empty scientific table: {path}.")
    persist_or_validate_csv(path, rows, tuple(rows[0]))


__all__ = (
    "CASE_FOLD_COLUMNS", "persist_development_and_router_surfaces",
    "persist_feature_surfaces", "persist_initial_surfaces", "persist_target_seal_phase",
    "persist_terminal_surfaces", "persist_validation_report", "write_run_state",
)
