"""Non-repairing persistence at every label-capability boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .artifact_io import persist_or_validate_csv, persist_or_validate_json
from .core_hashing import canonical_hash
from .experiment_contracts import CENTERS, EXPECTED_CENTER_FOLD_COUNT
from .reports import protocol_manifest_payload, publication_decision_payload, run_state_payload
from .scientific_constants import METHOD_IDS


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partition: object,
) -> None:
    input_hashes = {
        artifact_id: canonical_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    persist_or_validate_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            input_artifact_hashes=input_hashes,
            cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    partition_payload = _payload(partition)
    persist_or_validate_json(root / "manifests/case_oof_partition.json", partition_payload)
    rows: list[dict[str, object]] = []
    for fold in partition.folds:
        for role, cases in (
            ("support", fold.support_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                rows.append(
                    {
                        "target_center": fold.target_center,
                        "fold_ordinal": fold.fold_ordinal,
                        "fold_id": fold.fold_id,
                        "case_id": case_id,
                        "role": role,
                        "fold_hash": fold.fold_hash,
                        "partition_hash": partition.partition_hash,
                    }
                )
    _persist_rows(root / "tables/case_oof_partitions.csv", rows)


def persist_prediction_and_feature_surfaces(
    root: Path,
    *,
    prediction_capability: object,
    seed_rows: Sequence[object],
    probabilities: Sequence[object],
    probability_surface_hash: str,
    case_features: Sequence[object],
    source_controls: Sequence[object],
) -> Mapping[str, object]:
    _persist_rows(root / "tables/seed_probability_rows.csv", [_payload(row) for row in seed_rows])
    _persist_rows(
        root / "tables/aggregated_probability_rows.csv",
        [_payload(row) for row in probabilities],
    )
    _persist_rows(
        root / "tables/label_free_case_features.csv",
        [_payload(row) for row in case_features],
    )
    _persist_rows(
        root / "tables/label_free_source_controls.csv",
        [_payload(row) for row in source_controls],
    )
    probability_payload = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_probability_surface_v1",
        "row_count": len(probabilities),
        "rows": [_payload(row) for row in probabilities],
        "global_prediction_seal_hash": str(prediction_capability.seal_hash),
        "probability_store_hash": str(prediction_capability.store.store_hash),
        "surface_hash": probability_surface_hash,
        "exact_nine_seed_mean": True,
        "target_expert_used": False,
        "labels_used": False,
    }
    persist_or_validate_json(
        root / "manifests/sealed_probability_surface.json", probability_payload
    )
    feature_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_case_feature_surface_v1",
        "row_count": len(case_features),
        "rows": [_payload(row) for row in case_features],
        "probability_surface_hash": probability_surface_hash,
        "label_free": True,
        "metadata_used": False,
        "sealed_before_any_label_access": True,
    }
    feature_payload = {**feature_unhashed, "feature_surface_hash": canonical_hash(feature_unhashed)}
    control_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_source_control_surface_v1",
        "row_count": len(source_controls),
        "rows": [_payload(row) for row in source_controls],
        "definition": (
            "equal legal query mean of equal-case mean absolute baseline-anchored residual logit"
        ),
        "outer_mask": "q not in {H,e}",
        "nested_mask": "q not in {H,e,q0}",
        "training_context_mask": "u not in {H,e,s}",
        "nested_training_context_mask": "u not in {H,e,q0,s}",
        "probability_only": True,
        "metadata_used": False,
        "labels_used": False,
        "sealed_before_any_label_access": True,
    }
    control_payload = {**control_unhashed, "control_surface_hash": canonical_hash(control_unhashed)}
    persist_or_validate_json(
        root / "manifests/label_free_case_feature_surface.json", feature_payload
    )
    persist_or_validate_json(
        root / "manifests/label_free_source_control_surface.json", control_payload
    )
    phase_unhashed = {
        "schema_version": "midogpp_residual_stacker_prelabel_seal_v1",
        "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
        "global_prediction_seal_hash": str(prediction_capability.seal_hash),
        "probability_surface_hash": probability_surface_hash,
        "feature_surface_hash": feature_payload["feature_surface_hash"],
        "control_surface_hash": control_payload["control_surface_hash"],
        "all_729_probability_cells_sealed": len(prediction_capability.store.cells) == 729,
        "all_label_free_case_features_sealed": True,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "prior_stage90_prediction_surface_reused": False,
    }
    phase_payload = {**phase_unhashed, "prelabel_seal_hash": canonical_hash(phase_unhashed)}
    persist_or_validate_json(
        root / "reports/phase_01_prediction_and_feature_seal_complete.json",
        phase_payload,
    )
    for path, payload in (
        (root / "manifests/sealed_probability_surface.json", probability_payload),
        (root / "manifests/label_free_case_feature_surface.json", feature_payload),
        (root / "manifests/label_free_source_control_surface.json", control_payload),
        (root / "reports/phase_01_prediction_and_feature_seal_complete.json", phase_payload),
    ):
        if read_json(path) != payload:
            raise ProtocolError("Residual-stacker prelabel seal changed before labels.")
    return phase_payload


def persist_and_validate_loco_models(
    root: Path,
    *,
    donor_responses: Sequence[object],
    models: Sequence[object],
) -> Mapping[str, object]:
    if len(models) != len(CENTERS) * 3:
        raise ProtocolError("Residual stacker requires nine G, nine R, and nine P model seals.")
    _persist_rows(
        root / "tables/loco_donor_responses.csv",
        [_payload(row) for row in donor_responses],
    )
    components: list[dict[str, object]] = []
    for model in models:
        model_payload = _payload(model)
        for component in getattr(model, "candidate_models", ()):
            components.append(
                {
                    "target_center": model_payload.get("target_center"),
                    "model_family": model_payload.get("model_family"),
                    **_payload(component),
                    "outer_model_hash": model_payload.get("model_hash"),
                }
            )
    _persist_rows(root / "tables/loco_model_components.csv", components)
    unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_all_loco_models_v1",
        "model_count": len(models),
        "models": [_payload(model) for model in models],
        "all_G_R_and_P_models_sealed_before_same_H_support": True,
        "outer_H_labels_used": False,
        "target_expert_used": False,
        "separate_same_capacity_P_model": True,
    }
    payload = {**unhashed, "all_models_seal_hash": canonical_hash(unhashed)}
    path = root / "manifests/loco_hierarchical_model_seals.json"
    persist_or_validate_json(path, payload)
    if read_json(path) != payload:
        raise ProtocolError("Durable residual-stacker LOCO model seals drifted.")
    return payload


def persist_and_validate_preevaluation_seals(
    root: Path,
    *,
    calibrations: Sequence[object],
    decisions: Sequence[Mapping[str, object]],
    permutation_provenance: Mapping[str, object],
    config_contract_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    expected = EXPECTED_CENTER_FOLD_COUNT * len(METHOD_IDS)
    if len(decisions) != expected:
        raise ProtocolError(f"Residual stacker requires exactly {expected} method decisions.")
    _persist_rows(
        root / "tables/fold_calibrations.csv",
        [_payload(row) for row in calibrations],
    )
    _persist_rows(
        root / "tables/fold_method_decisions.csv",
        [_table_decision_row(row) for row in decisions],
    )
    manifest = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_fold_products_v1",
        "calibration_count": len(calibrations),
        "calibrations": [_payload(value) for value in calibrations],
        "decision_count": len(decisions),
        "decisions": [dict(value) for value in decisions],
        "method_ids": list(METHOD_IDS),
        "support_objective": "fixed_class_balanced_log_loss_only",
        "evaluation_labels_used": False,
    }
    persist_or_validate_json(
        root / "manifests/fold_calibrations_and_method_decisions.json", manifest
    )
    decision_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_all_decisions_v1",
        "config_contract_hash": config_contract_hash,
        "decision_count": len(decisions),
        "decision_hashes": [str(value["decision_hash"]) for value in decisions],
        "all_45_by_5_method_decisions_sealed_before_evaluation_labels": True,
        "evaluation_labels_used": False,
    }
    decision_seal = {
        **decision_unhashed,
        "decision_seal_hash": canonical_hash(decision_unhashed),
    }
    permutation_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_permutation_provenance_v1",
        "config_contract_hash": config_contract_hash,
        **dict(permutation_provenance),
        "applied_before_donor_fit": True,
        "applied_before_target_inference": True,
        "separate_same_capacity_model_fit": True,
        "labels_responses_residuals_and_g_preserved": True,
        "evaluation_labels_used": False,
    }
    permutation_seal = {
        **permutation_unhashed,
        "permutation_provenance_hash": canonical_hash(permutation_unhashed),
    }
    persist_or_validate_json(
        root / "manifests/all_fold_method_decisions_seal.json", decision_seal
    )
    persist_or_validate_json(
        root / "manifests/permutation_provenance_seal.json", permutation_seal
    )
    for path, expected_payload in (
        (root / "manifests/fold_calibrations_and_method_decisions.json", manifest),
        (root / "manifests/all_fold_method_decisions_seal.json", decision_seal),
        (root / "manifests/permutation_provenance_seal.json", permutation_seal),
    ):
        if read_json(path) != expected_payload:
            raise ProtocolError("Residual-stacker pre-evaluation seal drifted.")
    return decision_seal, permutation_seal


def persist_postseal_results(
    root: Path,
    *,
    evaluation: Mapping[str, object],
    confusion_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    contrast_rows: Sequence[Mapping[str, object]],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    persist_or_validate_json(
        root / "manifests/terminal_pooled_bacc_evaluation.json", dict(evaluation)
    )
    _persist_rows(
        root / "tables/oof_case_confusion_sufficient_statistics.csv",
        confusion_rows,
    )
    _persist_rows(root / "tables/oof_pooled_exact_bacc.csv", metric_rows)
    _persist_rows(root / "tables/paired_whole_case_cluster_contrasts.csv", contrast_rows)
    persist_or_validate_json(root / "reports/label_capability_report.json", capability_report)
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(evaluation),
    )
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _table_decision_row(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "target_center": value.get("target_center"),
        "fold_ordinal": value.get("fold_ordinal"),
        "method_id": value.get("method_id"),
        "decision_hash": value.get("decision_hash"),
        "prediction_count": value.get("prediction_count"),
        "prediction_hash": value.get("prediction_hash"),
        "decision_payload": json.dumps(dict(value), sort_keys=True, separators=(",", ":")),
    }


def _payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "to_payload"):
        raw = value.to_payload()
        if isinstance(raw, Mapping):
            return {str(key): _json_value(item) for key, item in raw.items()}
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in raw.items()
            if not str(key).startswith("_")
        }
    raise TypeError("Residual-stacker object must be mapping-like.")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_payload"):
        return _json_value(value.to_payload())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _persist_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    canonical = tuple({str(key): _json_value(value) for key, value in row.items()} for row in rows)
    if not canonical:
        raise ProtocolError(f"Residual-stacker table cannot be empty: {path}.")
    columns = tuple(canonical[0])
    if any(tuple(row) != columns for row in canonical):
        raise ProtocolError(f"Residual-stacker table schema drifted: {path}.")
    normalized = tuple(
        {
            key: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in canonical
    )
    persist_or_validate_csv(path, normalized, columns)


__all__ = (
    "persist_and_validate_loco_models",
    "persist_and_validate_preevaluation_seals",
    "persist_initial_surfaces",
    "persist_postseal_results",
    "persist_prediction_and_feature_surfaces",
    "persist_validation_report",
    "write_run_state",
)
