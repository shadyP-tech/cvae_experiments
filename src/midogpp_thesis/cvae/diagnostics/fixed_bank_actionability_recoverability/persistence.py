"""Atomic persistence at every label-capability boundary.

The bundle stores sufficient statistics and hash-addressed scientific
products.  It intentionally never stores raw labels or a per-case BACC.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .actions import build_action_library
from .artifact_serialization import payload, persist_payload, persist_rows
from .hashing import canonical_hash
from .reports import (
    protocol_manifest_payload,
    run_state_payload,
)
from .terminal_persistence import (
    persist_label_capability_report,
    persist_postseal_results,
)


ACTION_LIBRARY_FIELDS = (
    "target_center",
    "action_id",
    "geometry_id",
    "selected_source",
    "counts_by_class",
    "sample_weight_by_source",
    "physical_fit_required",
    "action_hash",
)
PARTITION_FIELDS = (
    "target_center",
    "fold_ordinal",
    "fold_id",
    "case_id",
    "role",
    "fold_hash",
    "partition_hash",
)
SEED_PROBABILITY_FIELDS = (
    "target_center",
    "case_id",
    "sample_id",
    "action_id",
    "seed_pair_ordinal",
    "probability",
    "probability_store_hash",
)
AGGREGATED_PROBABILITY_FIELDS = (
    "target_center",
    "case_id",
    "sample_id",
    "action_id",
    "probability_mean",
    "probability_sd",
    "seed_pair_count",
    "seed_probability_hash",
    "row_hash",
)
CASE_ACTION_FEATURE_FIELDS = (
    "query_center",
    "case_id",
    "geometry_id",
    "selected_source",
    "feature_origin_source",
    "context_excluded_centers",
    "feature_names",
    "values",
    "labels_used",
    "action_id",
    "feature_hash",
)
MODEL_FIT_FIELDS = (
    "outer_target_center",
    "heldout_donor_center",
    "geometry_id",
    "selected_source",
    "family",
    "ridge_alpha",
    "feature_names",
    "means",
    "scales",
    "coefficients",
    "training_query_centers",
    "response_kind",
    "target_labels_used",
    "exclusion_rule",
    "model_hash",
)
MODEL_PREDICTION_FIELDS = (
    "prediction_role",
    "outer_target_center",
    "heldout_query_center",
    "target_center",
    "case_id",
    "geometry_id",
    "selected_source",
    "family",
    "observed_gain",
    "predicted_gain",
    "squared_error",
    "model_hash",
)
LOCO_UTILITY_FIELDS = (
    "outer_target_center",
    "query_center",
    "case_id",
    "geometry_id",
    "selected_source",
    "response",
    "response_kind",
    "utility_product_hash",
)
METHOD_DECISION_FIELDS = (
    "target_center",
    "case_id",
    "method_id",
    "action_id",
    "geometry_id",
    "predicted_gain",
    "decision_source",
    "evaluation_labels_used",
)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partition: object,
) -> None:
    input_ids = tuple(str(value) for value in getattr(config, "input_artifact_ids"))
    if set(provenance) != set(input_ids) or len(provenance) != 6:
        raise ProtocolError("Initial provenance must contain the exact six inputs.")
    input_hashes = {name: canonical_hash(provenance[name]) for name in input_ids}
    persist_payload(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol=protocol,
            input_artifact_hashes=input_hashes,
            cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
            firewall=firewall,
        ),
    )
    persist_payload(root / "manifests/case_oof_partition.json", payload(partition))
    partition_rows: list[dict[str, object]] = []
    for fold in getattr(partition, "folds"):
        for role, cases in (
            ("support", getattr(fold, "support_case_ids")),
            ("evaluation", getattr(fold, "evaluation_case_ids")),
        ):
            for case_id in cases:
                partition_rows.append(
                    {
                        "target_center": getattr(fold, "target_center"),
                        "fold_ordinal": getattr(fold, "fold_ordinal"),
                        "fold_id": getattr(fold, "fold_id"),
                        "case_id": case_id,
                        "role": role,
                        "fold_hash": getattr(fold, "fold_hash"),
                        "partition_hash": getattr(partition, "partition_hash"),
                    }
                )
    _persist_exact_rows(
        root / "tables/case_oof_partitions.csv",
        partition_rows,
        PARTITION_FIELDS,
    )
    actions = tuple(build_action_library())
    action_rows = [_project(payload(row), ACTION_LIBRARY_FIELDS) for row in actions]
    _persist_exact_rows(
        root / "tables/action_library.csv", action_rows, ACTION_LIBRARY_FIELDS
    )
    action_by_target = {
        target: [payload(row) for row in actions if row.target_center == target]
        for target in tuple(dict.fromkeys(row.target_center for row in actions))
    }
    action_unhashed = {
        "schema_version": "midogpp_fixed_bank_actionability_action_library_v1",
        "actions": [payload(row) for row in actions],
        "action_count": len(actions),
        "physical_actions_per_target": 18,
        "geometry_selected": False,
        "target_expert_used": False,
    }
    persist_payload(
        root / "manifests/action_library.json",
        {
            **action_unhashed,
            # This is exactly the payload consumed by prediction_runtime.
            "action_library_hash": canonical_hash(action_by_target),
        },
    )


def persist_prelabel_surfaces(
    root: Path,
    *,
    prediction_capability: object,
    seed_rows: Sequence[object],
    probability_surface: object,
    prelabel: object,
) -> Mapping[str, object]:
    seed_count = len(seed_rows)
    aggregate_rows = [
        _project(payload(row), AGGREGATED_PROBABILITY_FIELDS)
        for row in getattr(probability_surface, "rows")
    ]
    feature_rows = [
        _project(payload(row), CASE_ACTION_FEATURE_FIELDS)
        for row in getattr(prelabel, "features")
    ]
    _persist_exact_rows(
        root / "tables/seed_probability_rows.csv",
        (
            _project(payload(row), SEED_PROBABILITY_FIELDS)
            for row in seed_rows
        ),
        SEED_PROBABILITY_FIELDS,
    )
    _persist_exact_rows(
        root / "tables/aggregated_probability_rows.csv",
        aggregate_rows,
        AGGREGATED_PROBABILITY_FIELDS,
    )
    _persist_exact_rows(
        root / "tables/case_action_features.csv",
        feature_rows,
        CASE_ACTION_FEATURE_FIELDS,
    )
    prediction_hash = str(getattr(prediction_capability, "seal_hash"))
    store_hash = str(getattr(getattr(prediction_capability, "store"), "store_hash"))
    probability_payload = {
        "schema_version": "midogpp_fixed_bank_actionability_probability_surface_v1",
        "row_count": len(aggregate_rows),
        "seed_row_count": seed_count,
        "global_prediction_seal_hash": prediction_hash,
        "probability_store_hash": store_hash,
        "surface_hash": str(getattr(probability_surface, "surface_hash")),
        "exact_nine_seed_mean": True,
        "labels_used": False,
        "target_expert_used": False,
    }
    feature_payload = {
        "schema_version": "midogpp_fixed_bank_actionability_prelabel_feature_seal_v1",
        "feature_count": len(feature_rows),
        "feature_surface_hash": str(getattr(prelabel, "feature_surface_hash")),
        "probability_surface_hash": str(
            getattr(prelabel, "probability_surface_hash")
        ),
        "permutation_provenance_hash": str(
            getattr(prelabel, "permutation_provenance_hash")
        ),
        "protocol_contract_hash": str(getattr(prelabel, "protocol_contract_hash")),
        "prelabel_products_hash": str(getattr(prelabel, "prelabel_products_hash")),
        "sealed_before_any_label_access": True,
        "baseline_predicted_class_branch_used": False,
        "raw_labels_persisted": False,
    }
    action_library = read_json(root / "manifests/action_library.json")
    phase_unhashed = {
        "schema_version": "midogpp_fixed_bank_actionability_phase_01_v1",
        "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
        "global_prediction_seal_hash": prediction_hash,
        "probability_surface_hash": probability_payload["surface_hash"],
        "feature_surface_hash": feature_payload["feature_surface_hash"],
        "permutation_provenance_hash": feature_payload[
            "permutation_provenance_hash"
        ],
        "action_library_hash": action_library["action_library_hash"],
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "prior_stage90_prediction_surface_reused": False,
    }
    phase_payload = {
        **phase_unhashed,
        "phase_seal_hash": canonical_hash(phase_unhashed),
    }
    members = (
        ("manifests/sealed_probability_surface.json", probability_payload),
        ("manifests/prelabel_feature_seal.json", feature_payload),
        ("reports/phase_01_prelabel_seal_complete.json", phase_payload),
    )
    for member, value in members:
        persist_payload(root / member, value)
        if read_json(root / member) != value:
            raise ProtocolError("Prelabel boundary changed after publication.")
    return phase_payload


def persist_and_validate_models(
    root: Path,
    *,
    products: object,
    utility_products: Sequence[object],
    target_products: Sequence[object],
) -> Mapping[str, object]:
    utilities = tuple(utility_products)
    targets = tuple(target_products)
    if len(utilities) != 9 or len(targets) != 9:
        raise ProtocolError("Model persistence requires nine LOCO and target products.")
    utility_rows: list[dict[str, object]] = []
    for product in utilities:
        for row in getattr(product, "rows"):
            utility_rows.append(
                {
                    "outer_target_center": getattr(product, "outer_target_center"),
                    "query_center": row.query_center,
                    "case_id": row.case_id,
                    "geometry_id": row.geometry_id,
                    "selected_source": row.selected_source,
                    "response": row.response,
                    "response_kind": row.response_kind,
                    "utility_product_hash": getattr(product, "utility_product_hash"),
                }
            )
    _persist_exact_rows(
        root / "tables/loco_utility_targets.csv",
        utility_rows,
        LOCO_UTILITY_FIELDS,
    )
    models = tuple(getattr(products, "models"))
    scores = tuple(getattr(products, "scores"))
    nested = tuple(getattr(products, "nested_predictions"))
    model_rows = [_project(payload(row), MODEL_FIT_FIELDS) for row in models]
    score_rows = [
        {
            "prediction_role": "terminal_target_score",
            "outer_target_center": row.target_center,
            "heldout_query_center": "",
            "target_center": row.target_center,
            "case_id": row.case_id,
            "geometry_id": row.geometry_id,
            "selected_source": row.selected_source,
            "family": row.family,
            "observed_gain": "",
            "predicted_gain": row.predicted_gain,
            "squared_error": "",
            "model_hash": row.model_hash,
        }
        for row in scores
    ]
    score_rows.extend(
        {
            "prediction_role": "nested_query_diagnostic",
            "outer_target_center": row.outer_target_center,
            "heldout_query_center": row.heldout_query_center,
            "target_center": row.heldout_query_center,
            "case_id": row.case_id,
            "geometry_id": row.geometry_id,
            "selected_source": row.selected_source,
            "family": row.family,
            "observed_gain": row.observed_gain,
            "predicted_gain": row.predicted_gain,
            "squared_error": row.squared_error,
            "model_hash": row.model_hash,
        }
        for row in nested
    )
    _persist_exact_rows(root / "tables/model_fits.csv", model_rows, MODEL_FIT_FIELDS)
    _persist_exact_rows(
        root / "tables/model_predictions.csv", score_rows, MODEL_PREDICTION_FIELDS
    )
    manifest = {
        "schema_version": "midogpp_fixed_bank_actionability_model_seals_v1",
        "model_count": len(models),
        "score_count": len(scores),
        "nested_prediction_count": len(nested),
        "nested_mse": [payload(row) for row in getattr(products, "nested_mse")],
        "model_seals_by_target": {
            str(target): dict(values)
            for target, values in getattr(products, "model_seals_by_target").items()
        },
        "all_models_seal_hash": str(getattr(products, "all_models_seal_hash")),
        "permutation_provenance_hash": str(
            getattr(products, "permutation_provenance_hash")
        ),
        "protocol_contract_hash": str(getattr(products, "protocol_contract_hash")),
        "model_products_hash": str(getattr(products, "model_products_hash")),
        "all_G_R_P_models_sealed_before_same_H_support": True,
        "outer_H_labels_used": False,
        "raw_labels_persisted": False,
    }
    utility_manifest = {
        "schema_version": "midogpp_fixed_bank_actionability_loco_utility_seals_v1",
        "target_count": len(utilities),
        "targets": [
            {
                "outer_target_center": getattr(row, "outer_target_center"),
                "row_count": len(getattr(row, "rows")),
                "donor_label_surface_hash": getattr(row, "donor_label_surface_hash"),
                "probability_surface_hash": getattr(row, "probability_surface_hash"),
                "utility_product_hash": getattr(row, "utility_product_hash"),
                "target_product_hash": getattr(target, "target_product_hash"),
            }
            for row, target in zip(utilities, targets, strict=True)
        ],
        "response": "class_balanced_proper_loss_gain_vs_u",
        "outer_H_labels_used": False,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
    }
    persist_payload(root / "manifests/loco_utility_seals.json", utility_manifest)
    persist_payload(root / "manifests/model_seals.json", manifest)
    if (
        read_json(root / "manifests/model_seals.json") != manifest
        or read_json(root / "manifests/loco_utility_seals.json")
        != utility_manifest
    ):
        raise ProtocolError("Model seals changed before support access.")
    return manifest


def persist_pre_support_decisions(
    root: Path, *, products: object
) -> Mapping[str, object]:
    rows = [_project(_decision_payload(row), METHOD_DECISION_FIELDS) for row in getattr(products, "decisions")]
    # The final table is written after S_y is available.  This manifest is the
    # durable pre-support commit marker.
    manifest = {
        "schema_version": "midogpp_fixed_bank_actionability_pre_support_decisions_v1",
        "fold_decision_count": len(getattr(products, "fold_seals")),
        "decision_row_count": len(rows),
        "decisions": rows,
        "fold_seals": [payload(row) for row in getattr(products, "fold_seals")],
        "pre_support_seal_hash": str(getattr(products, "pre_support_seal_hash")),
        "permutation_provenance_hash": str(
            getattr(products, "permutation_provenance_hash")
        ),
        "partition_hash": str(getattr(products, "partition_hash")),
        "protocol_contract_hash": str(getattr(products, "protocol_contract_hash")),
        "evaluation_labels_used": False,
    }
    persist_payload(root / "manifests/pre_support_decisions_seal.json", manifest)
    if read_json(root / "manifests/pre_support_decisions_seal.json") != manifest:
        raise ProtocolError("Pre-support decisions changed after publication.")
    return manifest


def persist_all_decisions(root: Path, *, products: object) -> tuple[Mapping[str, object], Mapping[str, object]]:
    rows = [
        _project(_decision_payload(row), METHOD_DECISION_FIELDS)
        for row in getattr(products, "decisions")
    ]
    _persist_exact_rows(
        root / "tables/method_decisions.csv", rows, METHOD_DECISION_FIELDS
    )
    all_rows = [
        {
            "target_center": key[0],
            "fold_ordinal": key[1],
            "method_id": key[2],
            "geometry_id": key[3],
            "decision_hash": value,
        }
        for key, value in sorted(getattr(products, "all_decision_hashes").items())
    ]
    all_manifest = {
        "schema_version": "midogpp_fixed_bank_actionability_all_decisions_v1",
        "decision_cell_count": len(all_rows),
        "decision_row_count": len(rows),
        "decision_seals": all_rows,
        "support_action_scores": [
            payload(row) for row in getattr(products, "support_action_scores")
        ],
        "support_product_hashes": list(
            getattr(products, "support_product_hashes")
        ),
        "pre_support_seal_hash": str(getattr(products, "pre_support_seal_hash")),
        "all_decisions_seal_hash": str(
            getattr(products, "all_decisions_seal_hash")
        ),
        "partition_hash": str(getattr(products, "partition_hash")),
        "protocol_contract_hash": str(getattr(products, "protocol_contract_hash")),
        "evaluation_labels_used": False,
        "geometry_selected": False,
    }
    permutation = {
        "schema_version": "midogpp_fixed_bank_actionability_permutation_provenance_v1",
        "permutation_provenance_hash": str(
            getattr(products, "permutation_provenance_hash")
        ),
        "complete_candidate_feature_blocks_permuted": True,
        "labels_and_response_targets_preserved": True,
        "separate_same_capacity_models_refit": True,
        "evaluation_labels_used": False,
    }
    persist_payload(root / "manifests/all_method_decisions_seal.json", all_manifest)
    persist_payload(root / "manifests/permutation_provenance_seal.json", permutation)
    for member, expected in (
        ("manifests/all_method_decisions_seal.json", all_manifest),
        ("manifests/permutation_provenance_seal.json", permutation),
    ):
        if read_json(root / member) != expected:
            raise ProtocolError("Pre-evaluation decision boundary drifted.")
    return all_manifest, permutation


def persist_validation_report(root: Path, value: Mapping[str, object]) -> None:
    persist_payload(root / "reports/validation_report.json", value)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _decision_payload(row: object) -> dict[str, object]:
    return {
        "target_center": getattr(row, "target_center"),
        "case_id": getattr(row, "case_id"),
        "method_id": getattr(row, "method_id"),
        "action_id": getattr(row, "action_id"),
        "geometry_id": getattr(row, "geometry_id"),
        "predicted_gain": getattr(row, "predicted_gain"),
        "decision_source": getattr(row, "decision_source"),
        "evaluation_labels_used": getattr(row, "evaluation_labels_used"),
    }


def _project(row: Mapping[str, object], fields: Sequence[str]) -> dict[str, object]:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ProtocolError(f"Persisted row is missing fields: {missing}.")
    return {field: row[field] for field in fields}


def _persist_exact_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    persist_rows(path, (_project(row, fields) for row in rows))


__all__ = (
    "ACTION_LIBRARY_FIELDS",
    "AGGREGATED_PROBABILITY_FIELDS",
    "CASE_ACTION_FEATURE_FIELDS",
    "METHOD_DECISION_FIELDS",
    "LOCO_UTILITY_FIELDS",
    "MODEL_FIT_FIELDS",
    "MODEL_PREDICTION_FIELDS",
    "PARTITION_FIELDS",
    "SEED_PROBABILITY_FIELDS",
    "persist_all_decisions",
    "persist_and_validate_models",
    "persist_initial_surfaces",
    "persist_label_capability_report",
    "persist_postseal_results",
    "persist_pre_support_decisions",
    "persist_prelabel_surfaces",
    "persist_validation_report",
    "write_run_state",
)
