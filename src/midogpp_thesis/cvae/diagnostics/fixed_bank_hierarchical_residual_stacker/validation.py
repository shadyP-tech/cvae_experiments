"""Content-first, non-repairing validation of the residual-stacker bundle."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from ...generation.contracts import EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .execution_adapter import load_frozen_source_streams, load_global_prediction_seal
from .core_hashing import canonical_hash
from .experiment_contracts import CENTERS, EXPECTED_CENTER_FOLD_COUNT
from .reports import publication_decision_payload, run_state_payload
from .scientific_constants import METHOD_IDS, candidate_sources


def validate_fixed_bank_hierarchical_residual_stacker_bundle(
    root: str | Path, *, config: object
) -> Mapping[str, object]:
    root = Path(root)
    assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    # The byte index is deliberately the first scientific read.
    content = validate_content_index(root, config_contract_hash=str(config.contract_hash))

    protocol = read_json(root / "manifests/protocol_manifest.json")
    partition = read_json(root / "manifests/case_oof_partition.json")
    probability = read_json(root / "manifests/sealed_probability_surface.json")
    features = read_json(root / "manifests/label_free_case_feature_surface.json")
    controls = read_json(root / "manifests/label_free_source_control_surface.json")
    models = read_json(root / "manifests/loco_hierarchical_model_seals.json")
    fold_products = read_json(
        root / "manifests/fold_calibrations_and_method_decisions.json"
    )
    decision_seal = read_json(root / "manifests/all_fold_method_decisions_seal.json")
    permutation = read_json(root / "manifests/permutation_provenance_seal.json")
    evaluation = read_json(root / "manifests/terminal_pooled_bacc_evaluation.json")
    capability = read_json(root / "reports/label_capability_report.json")
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    prelabel = read_json(root / "reports/phase_01_prediction_and_feature_seal_complete.json")

    source_cache = load_frozen_source_streams(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    predictions = load_global_prediction_seal(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_source_lock_hash=source_cache.lock_hash,
        expected_partition_lock_hash=str(partition.get("partition_hash", "")),
        expected_target_cache_binding_hash=str(protocol.get("test_cache_binding_hash", "")),
    )
    rebuilt_prelabel = _rebuild_prelabel_surfaces(
        predictions,
        probability_payload=probability,
        feature_payload=features,
        control_payload=controls,
    )

    raw_models = _list(models.get("models"), "models")
    raw_decisions = _list(fold_products.get("decisions"), "decisions")
    raw_metrics = _list(evaluation.get("metrics"), "evaluation metrics")
    raw_contrasts = _list(evaluation.get("contrasts"), "evaluation contrasts")
    expected_decision_keys = {
        (center, fold, method)
        for center in CENTERS
        for fold in range(5)
        for method in METHOD_IDS
    }
    observed_decision_keys = {
        (
            str(row.get("target_center")),
            int(row.get("fold_ordinal", -1)),
            str(row.get("method_id")),
        )
        for row in raw_decisions
    }
    if (
        protocol.get("experiment_id") != config.experiment_id
        or protocol.get("config_contract_hash") != config.contract_hash
        or protocol.get("support_selection_objective")
        != "fixed_class_balanced_log_loss_only"
        or protocol.get("terminal_metric") != "pooled_exact_bacc"
        or protocol.get("soft_class_gate") is not True
        or protocol.get("per_case_bacc_defined_or_persisted") is not False
        or partition.get("fold_count") != 5
        or len(partition.get("folds", [])) != EXPECTED_CENTER_FOLD_COUNT
        or partition.get("support_evaluation_disjoint") is not True
        or probability.get("global_prediction_seal_hash") != predictions.seal_hash
        or probability.get("probability_store_hash") != predictions.store.store_hash
        or prelabel.get("global_prediction_seal_hash") != predictions.seal_hash
        or prelabel.get("feature_surface_hash") != features.get("feature_surface_hash")
        or prelabel.get("control_surface_hash") != controls.get("control_surface_hash")
        or prelabel.get("status") != "COMPLETE_BEFORE_ANY_LABEL_ACCESS"
        or len(raw_models) != len(CENTERS) * 3
        or models.get("all_G_R_and_P_models_sealed_before_same_H_support") is not True
        or fold_products.get("decision_count") != len(expected_decision_keys)
        or observed_decision_keys != expected_decision_keys
        or decision_seal.get("decision_count") != len(expected_decision_keys)
        or decision_seal.get(
            "all_45_by_5_method_decisions_sealed_before_evaluation_labels"
        ) is not True
        or permutation.get("applied_before_donor_fit") is not True
        or permutation.get("applied_before_target_inference") is not True
        or permutation.get("separate_same_capacity_model_fit") is not True
        or permutation.get("labels_responses_residuals_and_g_preserved") is not True
        or capability.get("status") != "PASS"
        or capability.get("evaluation_labels_opened") is not True
        or capability.get("fold_method_decision_count") != len(expected_decision_keys)
        or capability.get("all_decisions_seal_hash")
        != decision_seal.get("decision_seal_hash")
        or capability.get("permutation_provenance_hash")
        != permutation.get("permutation_provenance_hash")
        or leakage.get("status") != "PASS"
        or leakage.get("exact_bacc_used_for_grid_selection") is not False
        or leakage.get("prior_stage90_artifact_or_scratch_consumed") is not False
        or publication
        != publication_decision_payload(evaluation)
        or publication.get("decision") != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or publication.get("claim_role")
        != "known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic"
        or runtime.get("classifier_cell_count") != 729
        or runtime.get("scratch_root")
        != "/data/local/fixed_bank_hierarchical_residual_stacker_v1"
        or runtime.get("prior_stage90_artifact_or_scratch_reused") is not False
        or {str(row.get("method_id")) for row in raw_metrics} != set(METHOD_IDS)
        or {str(row.get("contrast_id")) for row in raw_contrasts}
        < {"R-B_cal", "R-G", "R-P"}
    ):
        raise ProtocolError("Residual-stacker scientific bundle invariants drifted.")

    rebuilt_models = _validate_models(
        root,
        rows=raw_models,
        manifest=models,
        features=rebuilt_prelabel.features,
    )
    _validate_source_control_contexts(
        _list(controls.get("rows"), "source-control rows")
    )
    _validate_preevaluation_hashes(
        fold_products=fold_products,
        decisions=raw_decisions,
        decision_seal=decision_seal,
        permutation=permutation,
        rebuilt_models=rebuilt_models,
        features=rebuilt_prelabel.features,
        config_contract_hash=str(config.contract_hash),
    )
    _require_closed_hashed_payload(
        evaluation,
        hash_key="scientific_result_hash",
        expected_keys={
            "schema_version",
            "method_ids",
            "metrics",
            "contrasts",
            "primary_contrasts",
            "primary_endpoint",
            "nonzero_lambda_coverage",
            "calibration_only_gain",
            "single_class_cases_retained",
            "per_case_bacc_stored_or_used",
            "evaluation_labels_opened_after_all_decision_seals",
            "fresh_evidence",
            "scientific_result_hash",
        },
        role="evaluation result",
    )
    _semantic_replay_counts(
        root,
        config=config,
        partition_payload=partition,
        models=raw_models,
        decisions=raw_decisions,
        decision_seal=decision_seal,
        permutation=permutation,
        calibrations=_list(fold_products.get("calibrations"), "calibrations"),
        expected_evaluation=evaluation,
        expected_capability=capability,
        expected_leakage=leakage,
        prediction_seal_hash=predictions.seal_hash,
        feature_seal_hash=str(features["feature_surface_hash"]),
    )
    _assert_no_per_case_bacc_columns(
        root / "tables/oof_pooled_exact_bacc.csv",
        root / "tables/paired_whole_case_cluster_contrasts.csv",
        root / "tables/oof_case_confusion_sufficient_statistics.csv",
    )
    result = {
        "schema_version": "midogpp_hierarchical_residual_stacker_validation_v1",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": config.contract_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": predictions.seal_hash,
        "probability_surface_hash": probability["surface_hash"],
        "feature_surface_hash": features["feature_surface_hash"],
        "control_surface_hash": controls["control_surface_hash"],
        "model_family_target_count": len(raw_models),
        "fold_method_decision_count": len(raw_decisions),
        "decision_seal_hash": decision_seal["decision_seal_hash"],
        "permutation_provenance_hash": permutation["permutation_provenance_hash"],
        "scientific_result_hash": evaluation["scientific_result_hash"],
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "support_selection_objective": "fixed_class_balanced_log_loss_only",
        "terminal_metric": "pooled_exact_bacc",
        "paired_whole_case_cluster_uncertainty": True,
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_stage90": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
        "raw_labels_persisted": False,
    }
    _validate_excluded_control_members(root, result)
    return result


def _validate_models(
    root: Path,
    *,
    rows: list[Mapping[str, object]],
    manifest: Mapping[str, object],
    features: tuple[object, ...],
) -> Mapping[tuple[str, str], object]:
    from .contracts import (
        CandidateClassModel,
        DonorResponseRow,
        HierarchicalResidualModel,
        Standardization,
    )
    from .case_features import feature_surface_hash
    from .donor_responses import response_surface_hash

    manifest_unhashed = {
        key: value for key, value in manifest.items() if key != "all_models_seal_hash"
    }
    expected_manifest_keys = {
        "schema_version",
        "model_count",
        "models",
        "all_G_R_and_P_models_sealed_before_same_H_support",
        "outer_H_labels_used",
        "target_expert_used",
        "separate_same_capacity_P_model",
        "all_models_seal_hash",
    }
    if (
        set(manifest) != expected_manifest_keys
        or canonical_hash(manifest_unhashed) != manifest.get("all_models_seal_hash")
        or manifest_unhashed.get("model_count") != len(CENTERS) * 3
        or manifest_unhashed.get("all_G_R_and_P_models_sealed_before_same_H_support") is not True
        or manifest_unhashed.get("outer_H_labels_used") is not False
        or manifest_unhashed.get("target_expert_used") is not False
        or manifest_unhashed.get("separate_same_capacity_P_model") is not True
    ):
        raise ProtocolError("Residual-stacker aggregate model seal drifted.")
    expected = {(center, family) for center in CENTERS for family in ("G", "R", "P")}
    observed = {(str(row.get("target_center")), str(row.get("model_family"))) for row in rows}
    if observed != expected:
        raise ProtocolError("Residual-stacker G/R/P outer-model coverage drifted.")
    feature_hash = feature_surface_hash(features)
    rebuilt: dict[tuple[str, str], object] = {}
    for row in rows:
        target = str(row.get("target_center"))
        family = str(row.get("model_family"))
        components = _list(row.get("candidate_models"), "candidate models")
        expected_components = {(source, side) for source in candidate_sources(target) for side in (0, 1)}
        observed_components = {
            (str(value.get("heldout_source_id")), int(value.get("class_side", -1)))
            for value in components
        }
        if observed_components != expected_components:
            raise ProtocolError("Residual-stacker candidate-class coverage drifted.")
        rebuilt_components = []
        for component in components:
            source = str(component.get("heldout_source_id"))
            donors = set(str(value) for value in component.get("donor_centers", []))
            if target in donors or source in donors:
                raise ProtocolError("Residual-stacker strict H/e donor mask drifted.")
            if family == "G" and component.get("source_id_term_present") is not False:
                raise ProtocolError("Global control gained a source identity term.")
            standardization = component.get("standardization")
            if not isinstance(standardization, Mapping):
                raise ProtocolError("Residual-stacker model standardization is malformed.")
            rebuilt_component = CandidateClassModel(
                target_center=target,
                heldout_source_id=source,
                class_side=int(component["class_side"]),
                ridge_alpha=float(component["ridge_alpha"]),
                coefficients=tuple(float(value) for value in component["coefficients"]),
                standardization=Standardization(
                    tuple(float(value) for value in standardization["means"]),
                    tuple(float(value) for value in standardization["scales"]),
                ),
                training_row_count=int(component["training_row_count"]),
                donor_centers=tuple(str(value) for value in component["donor_centers"]),
                nested_validation_mse=tuple(
                    (float(value[0]), float(value[1]))
                    for value in component["nested_validation_mse"]
                ),
                model_family=family,
            )
            if rebuilt_component.to_payload() != dict(component):
                raise ProtocolError("Residual-stacker candidate model failed typed replay.")
            rebuilt_components.append(rebuilt_component)
        rebuilt_model = HierarchicalResidualModel(
            target_center=target,
            candidate_models=tuple(rebuilt_components),
            feature_surface_hash=str(row["feature_surface_hash"]),
            response_surface_hash=str(row["response_surface_hash"]),
            model_family=family,
        )
        if rebuilt_model.to_payload() != dict(row) or rebuilt_model.feature_surface_hash != feature_hash:
            raise ProtocolError("Residual-stacker hierarchical model failed typed replay.")
        rebuilt[(target, family)] = rebuilt_model

    response_rows: dict[str, list[DonorResponseRow]] = {center: [] for center in CENTERS}
    with (root / "tables/loco_donor_responses.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for raw in csv.DictReader(handle):
            target = str(raw["outer_heldout_target"])
            response = DonorResponseRow(
                donor_center=str(raw["donor_center"]),
                case_id=str(raw["case_id"]),
                source_id=str(raw["source_id"]),
                class_side=int(raw["class_side"]),
                sample_count=int(raw["sample_count"]),
                smooth_response=float(raw["smooth_response"]),
            )
            if response.response_hash != raw["response_hash"]:
                raise ProtocolError("Residual-stacker donor response hash drifted.")
            response_rows[target].append(response)
    for target in CENTERS:
        expected_response_hash = response_surface_hash(response_rows[target])
        if any(
            rebuilt[(target, family)].response_surface_hash != expected_response_hash
            for family in ("G", "R", "P")
        ):
            raise ProtocolError("Residual-stacker model response-surface binding drifted.")
    return rebuilt


def _validate_source_control_contexts(rows: list[Mapping[str, object]]) -> None:
    observed: dict[tuple[str, str, str | None, tuple[str, ...]], Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("target_center")),
            str(row.get("source_id")),
            (
                None
                if row.get("excluded_query_center") is None
                else str(row.get("excluded_query_center"))
            ),
            tuple(str(value) for value in row.get("context_excluded_centers", [])),
        )
        if key in observed:
            raise ProtocolError("Residual-stacker source-control context is duplicated.")
        observed[key] = row
    expected: set[tuple[str, str, str | None, tuple[str, ...]]] = set()
    for target in CENTERS:
        for heldout_source in candidate_sources(target):
            expected.add((target, heldout_source, None, ()))
            for training_source in (
                value for value in CENTERS if value not in (target, heldout_source)
            ):
                expected.add((target, training_source, None, (heldout_source,)))
            for query in (
                value for value in CENTERS if value not in (target, heldout_source)
            ):
                expected.add((target, heldout_source, query, ()))
                for training_source in (
                    value
                    for value in CENTERS
                    if value not in (target, heldout_source, query)
                ):
                    expected.add(
                        (target, training_source, query, (heldout_source,))
                    )
    if set(observed) != expected:
        raise ProtocolError("Residual-stacker full H/e/q/s control provenance drifted.")
    for (target, source, query, context), row in observed.items():
        forbidden = {target, source, *context}
        if query is not None:
            forbidden.add(query)
        donors = tuple(str(value) for value in row.get("donor_query_centers", []))
        expected_donors = tuple(value for value in CENTERS if value not in forbidden)
        if donors != expected_donors:
            raise ProtocolError("Residual-stacker source-control donor mask drifted.")


def _validate_preevaluation_hashes(
    *,
    fold_products: Mapping[str, object],
    decisions: list[Mapping[str, object]],
    decision_seal: Mapping[str, object],
    permutation: Mapping[str, object],
    rebuilt_models: Mapping[tuple[str, str], object],
    features: tuple[object, ...],
    config_contract_hash: str,
) -> None:
    from .case_features import permute_case_features
    from .contracts import CalibrationChoice
    from .controls import ModelFamilyBundle

    calibrations = _list(fold_products.get("calibrations"), "calibrations")
    if len(calibrations) != EXPECTED_CENTER_FOLD_COUNT * 2:
        raise ProtocolError("Residual-stacker calibration inventory drifted.")
    for row in calibrations:
        choice = CalibrationChoice(
            method_id=str(row["method_id"]),
            intercept=float(row["intercept"]),
            residual_scale=float(row["residual_scale"]),
            objective_value=float(row["objective_value"]),
            support_case_count=int(row["support_case_count"]),
            lcb_gain_over_baseline_calibrated=(
                None
                if row.get("lcb_gain_over_baseline_calibrated") is None
                else float(row["lcb_gain_over_baseline_calibrated"])
            ),
        )
        scientific = {
            key: value
            for key, value in row.items()
            if key not in {"target_center", "fold_ordinal", "parameter_role"}
        }
        if choice.to_payload() != scientific:
            raise ProtocolError("Residual-stacker calibration hash failed typed replay.")
    expected_order = tuple(
        (center, fold, method)
        for center in CENTERS
        for fold in range(5)
        for method in METHOD_IDS
    )
    if tuple(
        (str(row["target_center"]), int(row["fold_ordinal"]), str(row["method_id"]))
        for row in decisions
    ) != expected_order:
        raise ProtocolError("Residual-stacker decision order binding drifted.")
    decision_keys = {
        "schema_version",
        "target_center",
        "fold_ordinal",
        "fold_hash",
        "method_id",
        "prediction_count",
        "prediction_hash",
        "predictions",
        "B_cal_intercept",
        "common_residual_scale",
        "support_objective",
        "evaluation_labels_used",
        "target_expert_used",
        "decision_hash",
    }
    if any(set(row) != decision_keys for row in decisions):
        raise ProtocolError("Residual-stacker fold-decision schema drifted.")
    expected_fold_manifest = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_fold_products_v1",
        "calibration_count": len(calibrations),
        "calibrations": calibrations,
        "decision_count": len(decisions),
        "decisions": decisions,
        "method_ids": list(METHOD_IDS),
        "support_objective": "fixed_class_balanced_log_loss_only",
        "evaluation_labels_used": False,
    }
    if dict(fold_products) != expected_fold_manifest:
        raise ProtocolError("Residual-stacker fold-products closed schema drifted.")
    decision_unhashed = {
        key: value for key, value in decision_seal.items() if key != "decision_seal_hash"
    }
    expected_decision_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_all_decisions_v1",
        "config_contract_hash": config_contract_hash,
        "decision_count": len(decisions),
        "decision_hashes": [str(value["decision_hash"]) for value in decisions],
        "all_45_by_5_method_decisions_sealed_before_evaluation_labels": True,
        "evaluation_labels_used": False,
    }
    if (
        decision_unhashed != expected_decision_unhashed
        or canonical_hash(decision_unhashed) != decision_seal.get("decision_seal_hash")
    ):
        raise ProtocolError("Residual-stacker aggregate decision seal drifted.")

    permuted = permute_case_features(features)
    bundles = tuple(
        ModelFamilyBundle(
            target_center=target,
            global_model=rebuilt_models[(target, "G")],
            residual_model=rebuilt_models[(target, "R")],
            permuted_model=rebuilt_models[(target, "P")],
            permuted_features=permuted,
        )
        for target in CENTERS
    )
    plan_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_permutation_plan_v1",
        "bundle_hashes": [bundle.bundle_hash for bundle in bundles],
        "P_model_hashes": [bundle.permuted_model.model_hash for bundle in bundles],
        "P_feature_hashes": [
            row.feature_hash for bundle in bundles for row in bundle.permuted_features
        ],
        "whole_case_candidate_phi_block_permutation": True,
        "probability_residuals_labels_responses_and_g_preserved": True,
    }
    plan_hash = canonical_hash(plan_unhashed)
    expected_permutation_unhashed = {
        **plan_unhashed,
        "plan_hash": plan_hash,
        "config_contract_hash": config_contract_hash,
        "applied_before_donor_fit": True,
        "applied_before_target_inference": True,
        "separate_same_capacity_model_fit": True,
        "labels_responses_residuals_and_g_preserved": True,
        "evaluation_labels_used": False,
    }
    observed_permutation_unhashed = {
        key: value
        for key, value in permutation.items()
        if key != "permutation_provenance_hash"
    }
    if (
        observed_permutation_unhashed != expected_permutation_unhashed
        or canonical_hash(observed_permutation_unhashed)
        != permutation.get("permutation_provenance_hash")
    ):
        raise ProtocolError("Residual-stacker permutation provenance seal drifted.")


def _semantic_replay_counts(
    root: Path,
    *,
    config: object,
    partition_payload: Mapping[str, object],
    models: list[Mapping[str, object]],
    decisions: list[Mapping[str, object]],
    decision_seal: Mapping[str, object],
    permutation: Mapping[str, object],
    calibrations: list[Mapping[str, object]],
    expected_evaluation: Mapping[str, object],
    expected_capability: Mapping[str, object],
    expected_leakage: Mapping[str, object],
    prediction_seal_hash: str,
    feature_seal_hash: str,
) -> None:
    """Reopen scoped labels only after replaying every durable prerequisite."""

    from .case_partitions import CaseFold, CaseOOFPartition
    from .contracts import PredictionRow
    from .inputs import load_label_free_test_frame
    from .input_contracts import TestRowIdentity
    from .label_capabilities import LabelCapabilityManager
    from .pooled_metrics import score_case_confusions
    from .execution_phases import evaluate_terminal_predictions

    identities = tuple(
        TestRowIdentity(
            row_ordinal=int(row["row_ordinal"]),
            manifest_row_index=int(row["manifest_row_index"]),
            evaluation_row_id=str(row["evaluation_row_id"]),
            case_id=str(row["case_id"]),
            center=str(row["center"]),
            split=str(row["split"]),
        )
        for row in _list(partition_payload.get("identities"), "partition identities")
    )
    folds = tuple(
        CaseFold(
            target_center=str(row["target_center"]),
            fold_ordinal=int(row["fold_ordinal"]),
            support_case_ids=tuple(str(value) for value in row["support_case_ids"]),
            evaluation_case_ids=tuple(str(value) for value in row["evaluation_case_ids"]),
        )
        for row in _list(partition_payload.get("folds"), "partition folds")
    )
    partition = CaseOOFPartition(
        identities=identities,
        folds=folds,
        partition_seed=int(partition_payload["partition_seed"]),
        partition_hash=str(partition_payload["partition_hash"]),
    )
    frame = load_label_free_test_frame(config)
    manager = LabelCapabilityManager(
        config.test_manifest_path,
        frame,
        partition,
        global_prediction_seal_hash=prediction_seal_hash,
        label_free_feature_seal_hash=feature_seal_hash,
    )
    model_by_key = {
        (str(row["target_center"]), str(row["model_family"])): str(row["model_hash"])
        for row in models
    }
    for target in CENTERS:
        manager.open_loco_donor_labels(target)
        manager.record_loco_model_seals(
            target,
            model_by_key[(target, "G")],
            model_by_key[(target, "R")],
            model_by_key[(target, "P")],
        )
    decision_by_key = {
        (str(row["target_center"]), int(row["fold_ordinal"]), str(row["method_id"])): row
        for row in decisions
    }
    for fold in partition.folds:
        manager.open_fold_support_labels(fold.target_center, fold.fold_ordinal)
        for method in METHOD_IDS:
            row = decision_by_key[(fold.target_center, fold.fold_ordinal, method)]
            unhashed = {key: value for key, value in row.items() if key != "decision_hash"}
            if canonical_hash(unhashed) != row.get("decision_hash"):
                raise ProtocolError("Residual-stacker fold decision hash failed replay.")
            manager.record_fold_method_decision(
                fold.target_center,
                fold.fold_ordinal,
                method,
                str(row["decision_hash"]),
            )
    manager.record_preevaluation_seals(
        str(decision_seal["decision_seal_hash"]),
        str(permutation["permutation_provenance_hash"]),
        decision_count=len(decisions),
    )
    labels = manager.open_oof_evaluation_labels()
    replayed_capability = manager.access_report()
    if dict(replayed_capability) != dict(expected_capability):
        raise ProtocolError("Residual-stacker label-capability report failed replay.")
    from .reports import leakage_report_payload

    replayed_leakage = leakage_report_payload(
        prediction_seal_hash=prediction_seal_hash,
        feature_seal_hash=feature_seal_hash,
        model_count=len(models),
        decision_count=len(decisions),
        capability_report=replayed_capability,
    )
    if replayed_leakage != dict(expected_leakage):
        raise ProtocolError("Residual-stacker leakage report failed replay.")
    expected_rows: list[dict[str, object]] = []
    predictions_by_method: dict[str, tuple[PredictionRow, ...]] = {}
    for method in METHOD_IDS:
        predictions: list[PredictionRow] = []
        for fold in partition.folds:
            raw = decision_by_key[(fold.target_center, fold.fold_ordinal, method)]
            for row in _list(raw.get("predictions"), "sealed predictions"):
                prediction = PredictionRow(
                    method_id=str(row["method_id"]),
                    target_center=str(row["target_center"]),
                    case_id=str(row["case_id"]),
                    sample_id=str(row["sample_id"]),
                    probability=float(row["probability"]),
                    hard_prediction=int(row["hard_prediction"]),
                )
                if prediction.prediction_hash != row.get("prediction_hash"):
                    raise ProtocolError("Residual-stacker prediction hash failed replay.")
                predictions.append(prediction)
        for row in score_case_confusions(tuple(sorted(predictions)), labels):
            expected_rows.append(
                {
                    "method_id": row.method_id,
                    "target_center": row.target_center,
                    "case_id": row.case_id,
                    "n_positive": row.n_positive,
                    "true_positive": row.true_positive,
                    "n_negative": row.n_negative,
                    "true_negative": row.true_negative,
                    "per_case_bacc_stored": False,
                }
            )
        predictions_by_method[method] = tuple(sorted(predictions))
    with (root / "tables/oof_case_confusion_sufficient_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        observed = list(csv.DictReader(handle))
    normalized = [
        {
            "method_id": row["method_id"],
            "target_center": row["target_center"],
            "case_id": row["case_id"],
            "n_positive": int(row["n_positive"]),
            "true_positive": int(row["true_positive"]),
            "n_negative": int(row["n_negative"]),
            "true_negative": int(row["true_negative"]),
            "per_case_bacc_stored": row["per_case_bacc_stored"] == "True",
        }
        for row in observed
    ]
    if normalized != expected_rows:
        raise ProtocolError(
            "Residual-stacker confusion sufficient statistics failed label replay."
        )
    rebuilt = evaluate_terminal_predictions(
        predictions_by_method=predictions_by_method,
        labels=labels,
        calibrations=calibrations,
        bootstrap_replicates=int(
            config.evaluation["whole_case_cluster_bootstrap_replicates"]
        ),
        bootstrap_seed=int(config.evaluation["whole_case_cluster_bootstrap_seed"]),
        bootstrap_workers=int(config.runtime["bootstrap_workers"]),
    )
    if dict(rebuilt.evaluation) != dict(expected_evaluation):
        raise ProtocolError("Residual-stacker terminal metrics failed semantic replay.")


def _rebuild_prelabel_surfaces(
    predictions: object,
    *,
    probability_payload: Mapping[str, object],
    feature_payload: Mapping[str, object],
    control_payload: Mapping[str, object],
) -> object:
    from .execution_adapter import seed_probability_rows
    from .execution_phases import build_prelabel_products

    rebuilt = build_prelabel_products(seed_probability_rows(predictions))
    probability_rows = [row.to_payload() for row in rebuilt.probabilities]
    feature_rows = [row.to_payload() for row in rebuilt.features]
    control_rows = [row.to_payload() for row in rebuilt.source_controls]
    expected_probability = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_probability_surface_v1",
        "row_count": len(probability_rows),
        "rows": probability_rows,
        "global_prediction_seal_hash": str(predictions.seal_hash),
        "probability_store_hash": str(predictions.store.store_hash),
        "surface_hash": rebuilt.probability_surface_hash,
        "exact_nine_seed_mean": True,
        "target_expert_used": False,
        "labels_used": False,
    }
    expected_feature_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_case_feature_surface_v1",
        "row_count": len(feature_rows),
        "rows": feature_rows,
        "probability_surface_hash": rebuilt.probability_surface_hash,
        "label_free": True,
        "metadata_used": False,
        "sealed_before_any_label_access": True,
    }
    expected_feature = {
        **expected_feature_unhashed,
        "feature_surface_hash": canonical_hash(expected_feature_unhashed),
    }
    expected_control_unhashed = {
        "schema_version": "fixed_bank_hierarchical_residual_stacker_source_control_surface_v1",
        "row_count": len(control_rows),
        "rows": control_rows,
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
    expected_control = {
        **expected_control_unhashed,
        "control_surface_hash": canonical_hash(expected_control_unhashed),
    }
    if (
        dict(probability_payload) != expected_probability
        or dict(feature_payload) != expected_feature
        or dict(control_payload) != expected_control
    ):
        raise ProtocolError(
            "Residual-stacker probability, feature, or g surface failed sealed-array replay."
        )
    return rebuilt


def _assert_no_per_case_bacc_columns(*paths: Path) -> None:
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        if any("case_bacc" in value.lower() for value in fields):
            raise ProtocolError("Residual-stacker bundle persisted per-case BACC.")


def _require_closed_hashed_payload(
    payload: Mapping[str, object],
    *,
    hash_key: str,
    expected_keys: set[str],
    role: str,
) -> None:
    unhashed = {key: value for key, value in payload.items() if key != hash_key}
    if set(payload) != expected_keys or canonical_hash(unhashed) != payload.get(hash_key):
        raise ProtocolError(f"Residual-stacker {role} closed schema or hash drifted.")


def _validate_excluded_control_members(
    root: Path, expected_validation: Mapping[str, object]
) -> None:
    state = read_json(root / "reports/run_state.json")
    validation_path = root / "reports/validation_report.json"
    if validation_path.is_file():
        if (
            state != run_state_payload("COMPLETE", "COMPLETE")
            or read_json(validation_path) != dict(expected_validation)
        ):
            raise ProtocolError("Residual-stacker run-state or validation report drifted.")
    elif state != run_state_payload(
        "RUNNING", "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
    ):
        raise ProtocolError("Residual-stacker first validation has the wrong running phase.")


def _list(value: object, role: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ProtocolError(f"Residual-stacker {role} are malformed.")
    return [dict(row) for row in value]


__all__ = (
    "_require_closed_hashed_payload",
    "_validate_excluded_control_members",
    "validate_fixed_bank_hierarchical_residual_stacker_bundle",
)
