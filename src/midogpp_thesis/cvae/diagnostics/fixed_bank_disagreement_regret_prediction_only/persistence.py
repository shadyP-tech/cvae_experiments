"""Phase-specific persistence for source development and test inference."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import PairwiseRegretModel
from ...runtime.artifact_io import read_json, sha256_file
from .artifact_io import (
    persist_or_validate_csv,
    persist_or_validate_json,
    persist_or_validate_npz,
)
from .experiment_contracts import GEOMETRY_IDS, MODEL_FAMILY_IDS
from .hashing import canonical_hash
from .products import DevelopmentProducts, InferenceProducts, ModelBankRecord, PrelabelProducts


FEATURE_FIELDS = tuple(f"feature_{index:02d}" for index in range(15))


def persist_initial_manifest(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, object],
    source_frame_binding: Mapping[str, object],
    test_input_binding: Mapping[str, object],
    firewall: Mapping[str, object],
) -> Mapping[str, object]:
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(getattr(protocol, "contract_hash")),
        "input_artifact_ids": list(getattr(config, "input_artifact_ids")),
        "input_provenance_hash": canonical_hash(dict(provenance)),
        "source_frame_binding": dict(source_frame_binding),
        "test_input_binding": dict(test_input_binding),
        "firewall": dict(firewall),
        "source_labels_opened": False,
        "test_cache_opened": False,
        "test_labels_available": False,
        "test_metrics_permitted": False,
        "prior_stage90_output_consumed": False,
    }
    payload = {**unhashed, "manifest_hash": canonical_hash(unhashed)}
    persist_or_validate_json(root / "manifests/protocol_manifest.json", payload)
    return payload


def persist_prelabel_products(root: Path, products: PrelabelProducts) -> None:
    rows: list[dict[str, object]] = []
    for record in sorted(products.feature_surfaces, key=lambda value: value.key):
        for row in record.surface.rows:
            rows.append(
                {
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "query_id": row.query_id,
                    "case_id": row.case_id,
                    "action_id": row.action_id,
                    "source_id": row.source_id,
                    "sample_count": row.sample_count,
                    "disagreement_count": row.disagreement_count,
                    "prediction_seal_hash": row.prediction_seal_hash,
                    "feature_origin_action_id": row.feature_origin_action_id,
                    "feature_hash": row.feature_hash,
                    **{
                        name: value
                        for name, value in zip(FEATURE_FIELDS, row.values, strict=True)
                    },
                }
            )
    fieldnames = (
        "outer_target_id",
        "geometry_id",
        "family",
        "query_id",
        "case_id",
        "action_id",
        "source_id",
        "sample_count",
        "disagreement_count",
        "prediction_seal_hash",
        "feature_origin_action_id",
        "feature_hash",
        *FEATURE_FIELDS,
    )
    persist_or_validate_csv(
        root / "tables/source_case_features.csv", fieldnames=fieldnames, rows=rows
    )
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_prelabel_feature_seal_v1",
        "source_prediction_seal_hash": products.source_prediction_seal_hash,
        "prelabel_feature_seal_hash": products.prelabel_feature_seal_hash,
        "surface_count": len(products.feature_surfaces),
        "row_count": len(rows),
        "table_sha256": sha256_file(root / "tables/source_case_features.csv"),
        "source_labels_opened": False,
        "test_cache_opened": False,
        "test_labels_opened": False,
    }
    persist_or_validate_json(
        root / "manifests/prelabel_feature_seal.json",
        {**unhashed, "seal_hash": canonical_hash(unhashed)},
    )


def persist_development_products(root: Path, products: DevelopmentProducts) -> None:
    capability = dict(products.source_label_capability_report)
    if (
        capability.get("raw_source_labels_persisted") is not False
        or capability.get("test_labels_opened") is not False
    ):
        raise ProtocolError("Cannot persist an unsafe source-label capability report.")
    persist_or_validate_json(
        root / "manifests/source_label_capability_report.json", capability
    )
    response_rows: list[dict[str, object]] = []
    for record in sorted(products.response_surfaces, key=lambda value: value.key):
        for row in record.surface.rows:
            response_rows.append(
                {
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "query_id": row.query_id,
                    "case_id": row.case_id,
                    "action_id": row.action_id,
                    "source_id": row.source_id,
                    "source_exact_bacc_gain_vs_control": row.exact_bacc_gain_vs_control,
                    "source_exact_regret_from_case_best": row.exact_regret_from_case_best,
                    "disagreement_count": row.disagreement_count,
                    "positive_class_count": row.positive_class_count,
                    "negative_class_count": row.negative_class_count,
                    "response_hash": row.response_hash,
                    "response_surface_hash": record.surface.surface_hash,
                }
            )
    persist_or_validate_csv(
        root / "tables/source_regret_responses.csv",
        fieldnames=tuple(response_rows[0]) if response_rows else (),
        rows=response_rows,
    )
    _persist_model_banks(root, products.model_banks, products.model_bank_hash)


def _persist_model_banks(
    root: Path,
    records: Sequence[ModelBankRecord],
    collection_hash: str,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    model_rows: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []
    ordinal = 0
    for record in sorted(records, key=lambda value: value.key):
        models = tuple(getattr(record.bank, "models"))
        bank_hash = str(
            getattr(record.bank, "model_bank_hash", getattr(record.bank, "bank_hash", ""))
        )
        bank_rows.append(
            {
                "outer_target_id": record.outer_target_id,
                "geometry_id": record.geometry_id,
                "family": record.family,
                "model_bank_hash": bank_hash,
                "model_count": len(models),
            }
        )
        for model in models:
            prefix = f"model_{ordinal:04d}"
            members = {
                "feature_mean": f"{prefix}_feature_mean",
                "feature_scale": f"{prefix}_feature_scale",
                "coefficients": f"{prefix}_coefficients",
                "coefficient_covariance": f"{prefix}_coefficient_covariance",
            }
            for name, member in members.items():
                arrays[member] = np.asarray(getattr(model, name), dtype=np.float64)
            model_rows.append(
                {
                    "ordinal": ordinal,
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "candidate_action_id": model.candidate_action_id,
                    "candidate_source_id": model.candidate_source_id,
                    "heldout_query_id": model.heldout_query_id,
                    "action_ids": list(model.action_ids),
                    "feature_names": list(model.feature_names),
                    "training_query_ids": list(model.training_query_ids),
                    "excluded_query_ids": list(model.excluded_query_ids),
                    "observation_count": model.observation_count,
                    "converged": model.converged,
                    "iteration_count": model.iteration_count,
                    "feature_surface_hash": model.feature_surface_hash,
                    "response_surface_hash": model.response_surface_hash,
                    "prediction_seal_hash": model.prediction_seal_hash,
                    "development_context_hash": model.development_context_hash,
                    "baseline_action_id": model.baseline_action_id,
                    "control_action_id": model.control_action_id,
                    "candidate_source_by_action": [
                        list(value) for value in model.candidate_source_by_action
                    ],
                    "training_feature_hash": model.training_feature_hash,
                    "training_response_hash": model.training_response_hash,
                    "shared_l2_penalty": model.shared_l2_penalty,
                    "action_l2_penalty": model.action_l2_penalty,
                    "max_newton_iterations": model.max_newton_iterations,
                    "gradient_tolerance": model.gradient_tolerance,
                    "source_history_mode": model.source_history_mode,
                    "training_scope": model.training_scope,
                    "training_surface_role": model.training_surface_role,
                    "array_members": members,
                    "model_hash": model.model_hash,
                    "model_bank_hash": bank_hash,
                }
            )
            ordinal += 1
    array_path = root / "arrays/model_bank.npz"
    persist_or_validate_npz(array_path, arrays)
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_model_bank_index_v1",
        "collection_hash": collection_hash,
        "banks": bank_rows,
        "models": model_rows,
        "bank_count": len(bank_rows),
        "model_count": len(model_rows),
        "array_sha256": sha256_file(array_path),
        "source_labels_used_for_training_only": True,
        "raw_source_labels_persisted": False,
        "test_cache_opened_before_model_seal": False,
        "test_labels_used": False,
    }
    index = {**unhashed, "index_hash": canonical_hash(unhashed)}
    index_path = root / "manifests/model_bank_index.json"
    persist_or_validate_json(index_path, index)
    seal_unhashed = {
        "schema_version": "midogpp_disagreement_regret_model_bank_seal_v1",
        "status": "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
        "collection_hash": collection_hash,
        "index_sha256": sha256_file(index_path),
        "array_sha256": sha256_file(array_path),
        "bank_count": len(bank_rows),
        "model_count": len(model_rows),
        "source_labels_only": True,
        "test_cache_admitted": False,
        "target_labels_used": False,
        "test_cache_opened": False,
        "test_labels_opened": False,
    }
    persist_or_validate_json(
        root / "manifests/model_bank_seal.json",
        {
            **seal_unhashed,
            "regret_model_bank_seal_hash": canonical_hash(seal_unhashed),
        },
    )
    persist_or_validate_csv(
        root / "tables/model_index.csv",
        fieldnames=(
            "ordinal",
            "outer_target_id",
            "geometry_id",
            "family",
            "candidate_action_id",
            "candidate_source_id",
            "observation_count",
            "iteration_count",
            "model_hash",
            "model_bank_hash",
        ),
        rows=model_rows,
    )


def load_model_bank_records(root: Path) -> tuple[ModelBankRecord, ...]:
    """Reload and validate every frozen model from the durable model bank."""

    index = read_json(root / "manifests/model_bank_index.json")
    unhashed = {key: value for key, value in index.items() if key != "index_hash"}
    if (
        index.get("index_hash") != canonical_hash(unhashed)
        or index.get("array_sha256") != sha256_file(root / "arrays/model_bank.npz")
        or not isinstance(index.get("models"), list)
        or not isinstance(index.get("banks"), list)
    ):
        raise ProtocolError("Persisted model-bank index drifted.")
    archive = np.load(root / "arrays/model_bank.npz", allow_pickle=False)
    grouped: dict[tuple[str, str, str], list[PairwiseRegretModel]] = {}
    with archive:
        for row in index["models"]:
            if not isinstance(row, Mapping) or not isinstance(row.get("array_members"), Mapping):
                raise ProtocolError("Persisted model row is malformed.")
            members = row["array_members"]
            try:
                model = PairwiseRegretModel(
                    family=str(row["family"]),
                    outer_target_id=str(row["outer_target_id"]),
                    candidate_action_id=str(row["candidate_action_id"]),
                    candidate_source_id=str(row["candidate_source_id"]),
                    heldout_query_id=(
                        None if row.get("heldout_query_id") is None else str(row["heldout_query_id"])
                    ),
                    action_ids=tuple(str(value) for value in row["action_ids"]),
                    feature_names=tuple(str(value) for value in row["feature_names"]),
                    feature_mean=np.asarray(archive[str(members["feature_mean"])], dtype=np.float64),
                    feature_scale=np.asarray(archive[str(members["feature_scale"])], dtype=np.float64),
                    coefficients=np.asarray(archive[str(members["coefficients"])], dtype=np.float64),
                    coefficient_covariance=np.asarray(
                        archive[str(members["coefficient_covariance"])], dtype=np.float64
                    ),
                    training_query_ids=tuple(str(value) for value in row["training_query_ids"]),
                    excluded_query_ids=tuple(str(value) for value in row["excluded_query_ids"]),
                    observation_count=int(row["observation_count"]),
                    converged=bool(row["converged"]),
                    iteration_count=int(row["iteration_count"]),
                    feature_surface_hash=str(row["feature_surface_hash"]),
                    response_surface_hash=str(row["response_surface_hash"]),
                    prediction_seal_hash=str(row["prediction_seal_hash"]),
                    development_context_hash=str(row["development_context_hash"]),
                    baseline_action_id=str(row["baseline_action_id"]),
                    control_action_id=str(row["control_action_id"]),
                    candidate_source_by_action=tuple(
                        (str(value[0]), str(value[1]))
                        for value in row["candidate_source_by_action"]
                    ),
                    training_feature_hash=str(row["training_feature_hash"]),
                    training_response_hash=str(row["training_response_hash"]),
                    shared_l2_penalty=float(row["shared_l2_penalty"]),
                    action_l2_penalty=float(row["action_l2_penalty"]),
                    max_newton_iterations=int(row["max_newton_iterations"]),
                    gradient_tolerance=float(row["gradient_tolerance"]),
                    training_scope=str(row["training_scope"]),
                    training_surface_role=str(row["training_surface_role"]),
                    source_history_mode=str(row["source_history_mode"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("Cannot reconstruct persisted pairwise model.") from exc
            if model.model_hash != row.get("model_hash"):
                raise ProtocolError("Persisted model hash drifted after reconstruction.")
            key = (
                str(row["outer_target_id"]),
                str(row["geometry_id"]),
                str(row["family"]),
            )
            grouped.setdefault(key, []).append(model)
    from ...routing.disagreement_regret_core import freeze_pairwise_model_bank

    records = tuple(
        ModelBankRecord(
            outer_target_id=key[0],
            geometry_id=key[1],
            family=key[2],
            bank=freeze_pairwise_model_bank(tuple(models)),
        )
        for key, models in sorted(grouped.items())
    )
    expected_bank_hashes = {
        (str(row["outer_target_id"]), str(row["geometry_id"]), str(row["family"])): str(
            row["model_bank_hash"]
        )
        for row in index["banks"]
        if isinstance(row, Mapping)
    }
    if any(
        str(getattr(record.bank, "model_bank_hash", getattr(record.bank, "bank_hash", "")))
        != expected_bank_hashes.get(record.key)
        for record in records
    ):
        raise ProtocolError("Persisted model-bank hash drifted after reconstruction.")
    return records


def persist_inference_products(root: Path, products: InferenceProducts) -> None:
    feature_rows: list[dict[str, object]] = []
    for record in sorted(products.feature_surfaces, key=lambda value: value.key):
        for row in record.surface.rows:
            feature_rows.append(
                {
                    "outer_target_id": record.outer_target_id,
                    "geometry_id": record.geometry_id,
                    "family": record.family,
                    "case_id": row.case_id,
                    "action_id": row.action_id,
                    "source_id": row.source_id,
                    "sample_count": row.sample_count,
                    "disagreement_count": row.disagreement_count,
                    "prediction_seal_hash": row.prediction_seal_hash,
                    "feature_origin_action_id": row.feature_origin_action_id,
                    "feature_hash": row.feature_hash,
                    **{
                        name: value
                        for name, value in zip(FEATURE_FIELDS, row.values, strict=True)
                    },
                }
            )
    persist_or_validate_csv(
        root / "tables/test_case_features.csv",
        fieldnames=tuple(feature_rows[0]) if feature_rows else (),
        rows=feature_rows,
    )
    contrast_rows = [
        {
            "geometry_id": record.geometry_id,
            "family": row.family,
            "target_query_id": row.target_query_id,
            "case_id": row.case_id,
            "candidate_action_id": row.candidate_action_id,
            "candidate_source_id": row.candidate_source_id,
            "predicted_preference_margin_vs_control": row.predicted_preference_margin_vs_control,
            "standard_error_vs_control": row.standard_error_vs_control,
            "predicted_preference_margin_vs_baseline": row.predicted_preference_margin_vs_baseline,
            "standard_error_vs_baseline": row.standard_error_vs_baseline,
            "model_hash": row.model_hash,
            "score_semantics": row.score_semantics,
        }
        for record in products.contrasts
        for row in (record.row,)
    ]
    persist_or_validate_csv(
        root / "tables/test_candidate_contrasts.csv",
        fieldnames=tuple(contrast_rows[0]) if contrast_rows else (),
        rows=contrast_rows,
    )
    selection_rows = [
        {
            "geometry_id": record.geometry_id,
            "family": row.family,
            "target_query_id": row.target_query_id,
            "case_id": row.case_id,
            "raw_action_id": row.raw_action_id,
            "safe_action_id": row.safe_action_id,
            "baseline_action_id": row.baseline_action_id,
            "control_action_id": row.control_action_id,
            "simultaneous_z_value": row.simultaneous_z_value,
            "safe_margin": row.safe_margin,
            "fallback_reason": row.fallback_reason,
            "claim_role": row.claim_role,
            "may_authorize_routing": False,
            "may_authorize_promotion": False,
        }
        for record in products.selections
        for row in (record.row,)
    ]
    persist_or_validate_csv(
        root / "tables/test_selection_diagnostics.csv",
        fieldnames=tuple(selection_rows[0]) if selection_rows else (),
        rows=selection_rows,
    )
    counts = Counter(
        (
            record.geometry_id,
            row.target_query_id,
            row.family,
            row.raw_action_id,
            row.safe_action_id,
            row.fallback_reason,
        )
        for record in products.selections
        for row in (record.row,)
    )
    summary_rows = [
        {
            "geometry_id": key[0],
            "target_query_id": key[1],
            "family": key[2],
            "raw_action_id": key[3],
            "safe_action_id": key[4],
            "fallback_reason": key[5],
            "case_count": count,
            "test_labels_used": False,
            "test_metric_computed": False,
        }
        for key, count in sorted(counts.items())
    ]
    persist_or_validate_csv(
        root / "tables/test_prediction_summary.csv",
        fieldnames=tuple(summary_rows[0]) if summary_rows else (),
        rows=summary_rows,
    )
    seal_unhashed = {
        "schema_version": "midogpp_disagreement_regret_frozen_test_prediction_seal_v1",
        "status": "SEALED_UNSCORED_PREDICTIONS_FOR_ALL_TEST_CASES",
        "model_bank_hash": products.model_bank_hash,
        "test_prediction_seal_hash": products.test_prediction_seal_hash,
        "frozen_test_prediction_hash": products.frozen_prediction_hash,
        "feature_surface_count": len(products.feature_surfaces),
        "contrast_row_count": len(products.contrasts),
        "selection_row_count": len(products.selections),
        "test_feature_table_sha256": sha256_file(root / "tables/test_case_features.csv"),
        "contrast_table_sha256": sha256_file(root / "tables/test_candidate_contrasts.csv"),
        "selection_table_sha256": sha256_file(root / "tables/test_selection_diagnostics.csv"),
        "summary_table_sha256": sha256_file(root / "tables/test_prediction_summary.csv"),
        "test_labels_opened": False,
        "test_metrics_computed": False,
        "routing_authorized": False,
        "may_feed_another_experiment": False,
    }
    persist_or_validate_json(
        root / "manifests/frozen_test_prediction_seal.json",
        {**seal_unhashed, "seal_hash": canonical_hash(seal_unhashed)},
    )


def persist_reports(
    root: Path,
    *,
    leakage: Mapping[str, object],
    publication: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    persist_or_validate_json(root / "reports/leakage_report.json", leakage)
    persist_or_validate_json(root / "reports/publication_decision.json", publication)
    persist_or_validate_json(root / "reports/runtime_summary.json", runtime_summary)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    payload: dict[str, object] = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_run_state_v1",
        "status": status,
        "phase": phase,
        "prediction_only": True,
        "test_labels_opened": False,
    }
    if error is not None:
        payload["error"] = error
    # Run state is operational and intentionally replaceable.
    from ...runtime.artifact_io import atomic_json

    atomic_json(root / "reports/run_state.json", payload)


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_validation_v1",
        "status": "PASS",
        "checks": dict(checks),
        "test_labels_opened": False,
        "test_metrics_computed": False,
        "routing_or_promotion_authorized": False,
    }
    persist_or_validate_json(
        root / "reports/validation_report.json",
        {**unhashed, "validation_hash": canonical_hash(unhashed)},
    )


__all__ = (
    "load_model_bank_records",
    "persist_development_products",
    "persist_inference_products",
    "persist_initial_manifest",
    "persist_prelabel_products",
    "persist_reports",
    "persist_validation_report",
    "write_run_state",
)
