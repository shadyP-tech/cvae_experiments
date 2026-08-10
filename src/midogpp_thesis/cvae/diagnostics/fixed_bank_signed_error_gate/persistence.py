"""Atomic, non-repairing persistence at signed label-capability boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ..fixed_bank_hierarchical_residual_stacker.artifact_io import (
    persist_or_validate_csv,
    persist_or_validate_json,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from .execution import SignedFoldProducts, SignedModelProducts, SignedPrelabelProducts
from .protocol import SignedErrorGateProtocol
from .reports import (
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .terminal import SealedSignedGateEvaluationResult


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: SignedErrorGateProtocol,
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
            protocol=protocol,
            input_artifact_hashes=input_hashes,
            cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    persist_or_validate_json(
        root / "manifests/case_oof_partition.json", _payload(partition)
    )
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


def persist_prelabel_surfaces(
    root: Path,
    *,
    prediction_capability: object,
    seed_rows: Sequence[object],
    probabilities: Sequence[object],
    probability_surface_hash: str,
    prelabel: SignedPrelabelProducts,
) -> Mapping[str, object]:
    _persist_rows(
        root / "tables/seed_probability_rows.csv",
        [_payload(row) for row in seed_rows],
    )
    probability_rows = [_payload(row) for row in probabilities]
    _persist_rows(root / "tables/aggregated_probability_rows.csv", probability_rows)
    probability_payload = {
        "schema_version": "fixed_bank_signed_error_probability_surface_v1",
        "row_count": len(probability_rows),
        "rows": probability_rows,
        "global_prediction_seal_hash": str(prediction_capability.seal_hash),
        "probability_store_hash": str(prediction_capability.store.store_hash),
        "surface_hash": probability_surface_hash,
        "exact_nine_seed_mean": True,
        "target_expert_used": False,
        "labels_used": False,
    }
    feature_payload = {
        "schema_version": "fixed_bank_signed_error_prelabel_features_v1",
        "context_hashes": dict(prelabel.context_hashes),
        "feature_surface_hash": prelabel.feature_surface_hash,
        "protocol_contract_hash": prelabel.protocol_contract_hash,
        "context_feature_matrices_persisted": False,
        "context_features_rebuilt_and_hash_revalidated_per_target": True,
        "cross_target_context_cache_persisted": False,
        "baseline_predicted_class_branch_used": False,
        "sealed_before_any_label_access": True,
    }
    phase_unhashed = {
        "schema_version": "midogpp_fixed_bank_signed_error_prelabel_seal_v1",
        "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
        "global_prediction_seal_hash": str(prediction_capability.seal_hash),
        "probability_surface_hash": probability_surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "all_729_probability_cells_sealed": (
            len(prediction_capability.store.cells) == 729
        ),
        "outer_and_nested_context_hash_count": len(prelabel.context_hashes),
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "prior_stage90_prediction_surface_reused": False,
    }
    phase_payload = {
        **phase_unhashed,
        "prelabel_seal_hash": canonical_hash(phase_unhashed),
    }
    members = (
        ("manifests/sealed_probability_surface.json", probability_payload),
        ("manifests/signed_prelabel_feature_seal.json", feature_payload),
        (
            "reports/phase_01_prediction_and_feature_seal_complete.json",
            phase_payload,
        ),
    )
    for member, payload in members:
        persist_or_validate_json(root / member, payload)
        if read_json(root / member) != payload:
            raise ProtocolError("Signed-error prelabel seal changed before label access.")
    return phase_payload


def persist_and_validate_models(
    root: Path, *, products: SignedModelProducts
) -> Mapping[str, object]:
    if len(products.target_fits) != 9:
        raise ProtocolError("Signed-error diagnostic requires nine target model families.")
    model_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    target_payloads: list[dict[str, object]] = []
    for target_fit in products.target_fits:
        family_payloads: list[dict[str, object]] = []
        for fit in (
            target_fit.global_fit,
            target_fit.residual_fit,
            target_fit.permutation_fit,
        ):
            final = _model_payload(fit.final_model)
            model_rows.append(
                _model_table_row(
                    final,
                    role="final",
                    heldout_query_center="",
                    fit_hash=fit.fit_hash,
                )
            )
            nested_payloads: list[dict[str, object]] = []
            for nested in fit.nested_models:
                payload = _model_payload(nested.model)
                nested_payloads.append(
                    {
                        "heldout_query_center": nested.heldout_query_center,
                        "model": payload,
                    }
                )
                model_rows.append(
                    _model_table_row(
                        payload,
                        role="nested",
                        heldout_query_center=nested.heldout_query_center,
                        fit_hash=fit.fit_hash,
                    )
                )
            for alpha, mse in fit.validation_mse_by_alpha:
                alpha_rows.append(
                    {
                        "target_center": target_fit.target_center,
                        "family": fit.final_model.family,
                        "ridge_alpha": alpha,
                        "validation_mse": mse,
                        "selected": alpha == fit.final_model.ridge_alpha,
                        "fit_hash": fit.fit_hash,
                    }
                )
            family_payloads.append(
                {
                    "family": fit.final_model.family,
                    "final_model": final,
                    "nested_models": nested_payloads,
                    "validation_mse_by_alpha": [
                        [alpha, mse] for alpha, mse in fit.validation_mse_by_alpha
                    ],
                    "fit_hash": fit.fit_hash,
                }
            )
        for rows in (
            target_fit.global_corrections,
            target_fit.residual_corrections,
            target_fit.permutation_corrections,
        ):
            correction_rows.extend(row.to_payload() for row in rows)
        target_payloads.append(
            {
                "target_center": target_fit.target_center,
                "model_seal_hash": target_fit.model_seal_hash,
                "families": family_payloads,
                "target_labels_used": False,
            }
        )
    _persist_rows(root / "tables/signed_loco_models.csv", model_rows)
    _persist_rows(root / "tables/signed_alpha_path.csv", alpha_rows)
    _persist_rows(root / "tables/signed_corrections.csv", correction_rows)
    model_unhashed = {
        "schema_version": "fixed_bank_signed_error_all_loco_models_v1",
        "target_family_count": len(target_payloads),
        "targets": target_payloads,
        "protocol_contract_hash": products.protocol_contract_hash,
        "all_G_R_and_P_models_sealed_before_same_H_support": True,
        "outer_H_labels_used": False,
        "target_expert_used": False,
        "permutation_is_separate_same_capacity_refit": True,
    }
    model_manifest = {
        **model_unhashed,
        "all_models_seal_hash": canonical_hash(model_unhashed),
    }
    correction_unhashed = {
        "schema_version": "fixed_bank_signed_error_correction_surface_seals_v1",
        "correction_row_count": len(correction_rows),
        "R_raw_correction_surface_hash": products.raw_correction_surface_hash,
        "R_safe_correction_surface_hash": products.safe_correction_surface_hash,
        "control_correction_surface_hash": products.control_correction_surface_hash,
        "raw_and_safe_separately_sealed": True,
        "target_labels_used": False,
    }
    correction_manifest = {
        **correction_unhashed,
        "correction_manifest_hash": canonical_hash(correction_unhashed),
    }
    members = (
        ("manifests/signed_loco_model_seals.json", model_manifest),
        ("manifests/signed_correction_surface_seals.json", correction_manifest),
    )
    for member, payload in members:
        persist_or_validate_json(root / member, payload)
        if read_json(root / member) != payload:
            raise ProtocolError("Signed-error durable model/correction seal drifted.")
    return model_manifest


def persist_and_validate_fold_products(
    root: Path, *, products: SignedFoldProducts
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if len(products.decisions) != 45:
        raise ProtocolError("Signed-error diagnostic requires exactly 45 fold decisions.")
    decision_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []
    for decision in products.decisions:
        decision_rows.append(
            {
                "target_center": decision["target_center"],
                "fold_ordinal": decision["fold_ordinal"],
                "fold_hash": decision["fold_hash"],
                "partition_hash": decision["partition_hash"],
                "evaluation_case_ids": decision["evaluation_case_ids"],
                "intercept": decision["intercept"],
                "proposed_scale": decision["proposed_scale"],
                "selected_scale": decision["selected_scale"],
                "support_bacc_lcb": decision["support_bacc_lcb"],
                "fallback_reason": decision["fallback_reason"] or "",
                "evaluation_threshold_crossings": decision[
                    "evaluation_threshold_crossings"
                ],
                "model_seal_hash": decision["model_seal_hash"],
                "method_prediction_hashes": decision["method_prediction_hashes"],
                "method_decision_hashes": decision["method_decision_hashes"],
                "decision_hash": decision["decision_hash"],
            }
        )
        for path_row in decision["lambda_path"]:
            lambda_rows.append(
                {
                    "target_center": decision["target_center"],
                    "fold_ordinal": decision["fold_ordinal"],
                    **dict(path_row),
                    "decision_hash": decision["decision_hash"],
                }
            )
    prediction_rows = [
        row.to_payload()
        for method in products.predictions_by_method
        for row in products.predictions_by_method[method]
    ]
    _persist_rows(root / "tables/fold_decisions.csv", decision_rows)
    _persist_rows(root / "tables/lambda_path.csv", lambda_rows)
    _persist_rows(root / "tables/oof_predictions.csv", prediction_rows)
    manifest = {
        "schema_version": "fixed_bank_signed_error_fold_products_v1",
        "decision_count": len(products.decisions),
        "decisions": [dict(value) for value in products.decisions],
        "method_prediction_row_counts": {
            method: len(rows) for method, rows in products.predictions_by_method.items()
        },
        "method_prediction_surface_hashes": {
            method: canonical_hash([row.to_payload() for row in rows])
            for method, rows in products.predictions_by_method.items()
        },
        "partition_hash": products.partition_hash,
        "protocol_contract_hash": products.protocol_contract_hash,
        "evaluation_labels_used": False,
    }
    decision_seal = {
        "schema_version": "fixed_bank_signed_error_all_fold_decisions_v1",
        "decision_count": 45 * 6,
        "fold_decision_count": 45,
        "decision_hashes": [row["decision_hash"] for row in products.decisions],
        "decision_seal_hash": products.decision_seal_hash,
        "R_raw_and_R_safe_prediction_hashes_separate": True,
        "evaluation_labels_used": False,
    }
    permutation_seal = {
        "schema_version": "fixed_bank_signed_error_permutation_provenance_v1",
        "permutation_provenance_hash": products.permutation_provenance_hash,
        "complete_sample_feature_blocks_permuted": True,
        "labels_and_gradients_preserved": True,
        "separate_same_capacity_model_refit": True,
        "evaluation_labels_used": False,
    }
    members = (
        ("manifests/signed_fold_products.json", manifest),
        ("manifests/all_fold_method_decisions_seal.json", decision_seal),
        ("manifests/permutation_provenance_seal.json", permutation_seal),
    )
    for member, payload in members:
        persist_or_validate_json(root / member, payload)
        if read_json(root / member) != payload:
            raise ProtocolError("Signed-error pre-evaluation seal drifted.")
    return decision_seal, permutation_seal


def persist_postseal_results(
    root: Path,
    *,
    evaluation: SealedSignedGateEvaluationResult,
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    payload = evaluation.to_payload()
    confusion_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for method in evaluation.scientific_result.method_results:
        for row in method.case_confusions:
            confusion_rows.append(
                {
                    "method_id": row.method_id,
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "n_positive": row.n_positive,
                    "true_positive": row.true_positive,
                    "n_negative": row.n_negative,
                    "true_negative": row.true_negative,
                    "per_case_bacc": "",
                }
            )
        for center in method.center_metrics:
            metric_rows.append(center.to_payload())
    contrast_rows = [
        row.to_payload() for row in evaluation.scientific_result.contrasts
    ]
    _persist_rows(root / "tables/terminal_case_confusions.csv", confusion_rows)
    _persist_rows(root / "tables/terminal_center_metrics.csv", metric_rows)
    _persist_rows(root / "tables/terminal_contrasts.csv", contrast_rows)
    persist_or_validate_json(
        root / "reports/label_capability_report.json", capability_report
    )
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(payload),
    )
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)
    # Publish this last: its presence is the durable terminal-phase commit marker.
    persist_or_validate_json(root / "manifests/sealed_terminal_evaluation.json", payload)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _model_payload(model: object) -> dict[str, object]:
    standardization = getattr(model, "standardization")
    return {
        "schema_version": "fixed_bank_signed_error_model_v1",
        "target_center": getattr(model, "target_center"),
        "family": getattr(model, "family"),
        "ridge_alpha": getattr(model, "ridge_alpha"),
        "coefficients": list(getattr(model, "coefficients")),
        "means": list(standardization.means),
        "scales": list(standardization.scales),
        "donor_centers": list(getattr(model, "donor_centers")),
        "nested_model_hashes": list(getattr(model, "nested_model_hashes")),
        "response": "class_balanced_rescaled_negative_log_loss_logit_gradient",
        "ridge_objective": "unweighted_mse_on_rescaled_gradient_target",
        "target_labels_used": False,
        "model_hash": getattr(model, "model_hash"),
    }


def _model_table_row(
    payload: Mapping[str, object],
    *,
    role: str,
    heldout_query_center: str,
    fit_hash: str,
) -> dict[str, object]:
    return {
        "target_center": payload["target_center"],
        "family": payload["family"],
        "role": role,
        "heldout_query_center": heldout_query_center,
        "ridge_alpha": payload["ridge_alpha"],
        "coefficients": payload["coefficients"],
        "means": payload["means"],
        "scales": payload["scales"],
        "donor_centers": payload["donor_centers"],
        "nested_model_hashes": payload["nested_model_hashes"],
        "model_hash": payload["model_hash"],
        "fit_hash": fit_hash,
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
    raise TypeError("Signed-error object must be mapping-like.")


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
    canonical = tuple(
        {str(key): _json_value(value) for key, value in row.items()} for row in rows
    )
    if not canonical:
        raise ProtocolError(f"Signed-error table cannot be empty: {path}.")
    columns = tuple(canonical[0])
    if any(tuple(row) != columns for row in canonical):
        raise ProtocolError(f"Signed-error table schema drifted: {path}.")
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
    "persist_and_validate_fold_products",
    "persist_and_validate_models",
    "persist_initial_surfaces",
    "persist_postseal_results",
    "persist_prelabel_surfaces",
    "persist_validation_report",
    "write_run_state",
)
