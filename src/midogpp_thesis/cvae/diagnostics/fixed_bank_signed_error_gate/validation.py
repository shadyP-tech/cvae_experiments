"""Independent replay validator for the signed-error closed-world bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import load_frozen_source_streams
from ...runtime.label_free_action_predictions import load_global_prediction_seal
from ...runtime.preflight import REQUIRED_DISTRIBUTIONS, REQUIRED_THREAD_ENVIRONMENT
from ..fixed_bank_hierarchical_residual_stacker.artifact_io import read_json
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    PredictionRow,
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from .bundle import assert_closed_world, validate_content_index
from .contracts import CorrectionRow, SignedGateModel, Standardization
from .execution import (
    SignedFoldProducts,
    SignedModelProducts,
    TargetFamilyFits,
    build_signed_fold_products,
    build_signed_prelabel_products,
    fit_all_target_families,
)
from .features import build_signed_features, permute_feature_alignment
from .execution_adapter import runtime_summary_payload
from .label_capabilities import SignedErrorLabelCapabilityManager
from .model import (
    NestedSignedGateModel,
    SignedGateFit,
    correction_surface_hash,
    predict_corrections,
)
from .protocol import canonical_consumed_test_protocol
from .reports import (
    leakage_report_payload,
    protocol_manifest_payload,
    publication_decision_payload,
)
from .sealing import record_durable_fold_seals, record_durable_model_seals
from .terminal import evaluate_sealed_fold_products


def validate_fixed_bank_signed_error_gate_bundle(
    root: str | Path, *, config: object
) -> Mapping[str, object]:
    """Replay all label-free and terminal seals without trusting report summaries."""

    path = Path(root)
    validation_exists = (path / "reports/validation_report.json").is_file()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=not validation_exists,
    )
    protocol = canonical_consumed_test_protocol()
    validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )

    from .inputs import (
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )
    from .execution_adapter import build_case_partition

    workspace_binding = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    expected_input_hashes = {
        artifact_id: canonical_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace_binding
    if firewall.get("status") != "PASS":
        raise ProtocolError("Signed-error validation input firewall did not pass.")
    expected_protocol_manifest = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes=expected_input_hashes,
        cache_binding_hash=frame.cache_binding_hash,
        firewall=firewall,
    )
    if read_json(path / "manifests/protocol_manifest.json") != (
        expected_protocol_manifest
    ):
        raise ProtocolError("Signed-error protocol manifest differs from replay.")
    partition = build_case_partition(frame, config=config)
    if read_json(path / "manifests/case_oof_partition.json") != partition.to_payload():
        raise ProtocolError("Signed-error persisted partition drifted from replay.")
    _validate_partition_table(path, partition)

    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=locks.generation.generation_lock_hash,
    )
    prediction = load_global_prediction_seal(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_lock_hash=source.lock_hash,
        expected_partition_lock_hash=partition.partition_hash,
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    probabilities, probability_surface_hash = _load_probability_surface(
        path, prediction
    )
    prelabel = build_signed_prelabel_products(probabilities, protocol=protocol)
    persisted_prelabel = read_json(path / "manifests/signed_prelabel_feature_seal.json")
    expected_prelabel = {
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
    if persisted_prelabel != expected_prelabel:
        raise ProtocolError("Signed-error prelabel feature replay drifted.")
    phase_unhashed = {
        "schema_version": "midogpp_fixed_bank_signed_error_prelabel_seal_v1",
        "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
        "global_prediction_seal_hash": prediction.seal_hash,
        "probability_surface_hash": probability_surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "all_729_probability_cells_sealed": len(prediction.store.cells) == 729,
        "outer_and_nested_context_hash_count": len(prelabel.context_hashes),
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "prior_stage90_prediction_surface_reused": False,
    }
    expected_phase = {
        **phase_unhashed,
        "prelabel_seal_hash": canonical_hash(phase_unhashed),
    }
    if read_json(
        path / "reports/phase_01_prediction_and_feature_seal_complete.json"
    ) != expected_phase:
        raise ProtocolError("Signed-error prelabel phase report differs from replay.")

    persisted_models = _load_and_replay_models(
        path, probabilities, protocol.contract_hash
    )
    persisted_folds = _load_fold_products(path, protocol.contract_hash)
    if persisted_folds.partition_hash != partition.partition_hash:
        raise ProtocolError("Signed-error fold products use a different partition.")

    manager = SignedErrorLabelCapabilityManager(
        getattr(config, "test_manifest_path"),
        frame,
        partition,
        global_prediction_seal_hash=prediction.seal_hash,
        label_free_feature_seal_hash=prelabel.feature_surface_hash,
    )
    runtime = getattr(config, "runtime")
    replayed_models = fit_all_target_families(
        probabilities=probabilities,
        prelabel=prelabel,
        label_manager=manager,
        protocol=protocol,
        worker_count=int(runtime["model_workers"]),
        threads_per_worker=int(runtime["model_threads_per_worker"]),
    )
    if replayed_models != persisted_models:
        raise ProtocolError("Signed-error persisted models differ from donor-label refit.")
    record_durable_model_seals(manager, replayed_models)
    replayed_folds = build_signed_fold_products(
        probabilities=probabilities,
        model_products=replayed_models,
        partition=partition,
        label_manager=manager,
        protocol=protocol,
    )
    if replayed_folds != persisted_folds:
        raise ProtocolError(
            "Signed-error persisted fold decisions differ from support-label replay."
        )
    record_durable_fold_seals(manager, replayed_folds)
    terminal_labels = manager.open_oof_evaluation_labels()
    capability_report = manager.access_report()
    if dict(capability_report) != read_json(
        path / "reports/label_capability_report.json"
    ):
        raise ProtocolError("Signed-error label-capability replay drifted.")

    evaluation = getattr(config, "evaluation")
    replayed = evaluate_sealed_fold_products(
        fold_products=replayed_folds,
        capability_report=capability_report,
        terminal_labels=terminal_labels,
        protocol=protocol,
        bootstrap_replicates=int(evaluation["whole_case_cluster_bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["whole_case_cluster_bootstrap_seed"]),
        bootstrap_workers=int(runtime["bootstrap_workers"]),
        bootstrap_threads_per_worker=int(runtime["bootstrap_threads_per_worker"]),
        multiprocessing_start_method=str(runtime["multiprocessing_start_method"]),
    )
    if replayed.to_payload() != read_json(
        path / "manifests/sealed_terminal_evaluation.json"
    ):
        raise ProtocolError("Signed-error terminal evaluation replay drifted.")
    _validate_terminal_tables(path, replayed)
    _validate_reports(
        path,
        replayed.to_payload(),
        protocol.contract_hash,
        config=config,
        expected_input_hashes=expected_input_hashes,
        expected_firewall=firewall,
        source_cache=source,
        prediction_capability=prediction,
        expected_leakage=leakage_report_payload(
            prediction_seal_hash=prediction.seal_hash,
            feature_seal_hash=prelabel.feature_surface_hash,
            model_family_count=len(replayed_models.target_fits),
            decision_count=len(replayed_folds.decisions) * 6,
            capability_report=capability_report,
        ),
    )

    checks_unhashed = {
        "schema_version": "midogpp_fixed_bank_signed_error_validation_v1",
        "status": "PASS",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "partition_hash": partition.partition_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "probability_surface_hash": probability_surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "all_models_seal_hash": read_json(
            path / "manifests/signed_loco_model_seals.json"
        )["all_models_seal_hash"],
        "decision_seal_hash": replayed_folds.decision_seal_hash,
        "permutation_provenance_hash": replayed_folds.permutation_provenance_hash,
        "sealed_result_hash": replayed.sealed_result_hash,
        "closed_world_inventory": True,
        "content_index_validated_before_scientific_replay": True,
        "source_and_probability_arrays_reloaded": True,
        "prelabel_context_hashes_recomputed": True,
        "models_refit_from_donor_labels_and_corrections_replayed": True,
        "all_270_method_decisions_recomputed_from_support_and_resealed": True,
        "terminal_metrics_recomputed_from_sealed_predictions": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "routing_or_promotion_authorized": False,
        "may_feed_another_experiment": False,
    }
    checks = {
        **checks_unhashed,
        "validation_hash": canonical_hash(checks_unhashed),
    }
    state = read_json(path / "reports/run_state.json")
    expected_state_common = {
        "schema_version": "midogpp_fixed_bank_signed_error_run_state_v1",
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
    }
    status = state.get("status")
    phase = state.get("phase")
    if (
        set(state) != {*expected_state_common, "status", "phase"}
        or any(state.get(key) != value for key, value in expected_state_common.items())
        or status not in ("RUNNING", "COMPLETE")
        or (
            status == "RUNNING"
            and phase
            not in {
                "CLOSED_WORLD_CONTENT_FIRST_VALIDATION",
                "TERMINAL_PHASE_VALIDATION_RECOVERY",
                "CLOSED_WORLD_CONTENT_FIRST_VALIDATION_RECOVERY",
            }
        )
        or (status == "COMPLETE" and phase != "COMPLETE")
    ):
        raise ProtocolError("Signed-error run state is not validatable.")
    if validation_exists and read_json(
        path / "reports/validation_report.json"
    ) != checks:
        raise ProtocolError("Signed-error persisted validation report drifted.")
    if status == "COMPLETE" and not validation_exists:
        raise ProtocolError("Signed-error COMPLETE state lacks a validation report.")
    return checks


def _validate_partition_table(root: Path, partition: object) -> None:
    expected: list[dict[str, object]] = []
    for fold in partition.folds:
        for role, cases in (
            ("support", fold.support_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                expected.append(
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
    observed = _read_csv(root / "tables/case_oof_partitions.csv")
    canonical = tuple(
        {str(key): _csv_value(value) for key, value in row.items()}
        for row in expected
    )
    if observed != canonical:
        raise ProtocolError("Signed-error case partition table differs from replay.")


def _load_probability_surface(
    root: Path, prediction: object
) -> tuple[tuple[SampleActionProbability, ...], str]:
    from .execution_adapter import seed_probability_rows
    from .probability_surface import aggregate_exact_nine_probabilities

    payload = read_json(root / "manifests/sealed_probability_surface.json")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ProtocolError("Signed-error probability surface rows are absent.")
    rows = tuple(
        SampleActionProbability(
            str(row["target_center"]),
            str(row["case_id"]),
            str(row["sample_id"]),
            str(row["action_id"]),
            float(row["probability"]),
        )
        for row in raw_rows
        if isinstance(row, Mapping)
    )
    if (
        len(rows) != len(raw_rows)
        or payload.get("row_count") != len(rows)
        or [row.to_payload() for row in rows] != raw_rows
        or payload.get("global_prediction_seal_hash") != prediction.seal_hash
        or payload.get("probability_store_hash") != prediction.store.store_hash
    ):
        raise ProtocolError("Signed-error probability rows or runtime binding drifted.")
    replayed_seed_rows = seed_probability_rows(prediction)
    replayed_rows, surface_hash = aggregate_exact_nine_probabilities(
        replayed_seed_rows
    )
    expected_payload = {
        "schema_version": "fixed_bank_signed_error_probability_surface_v1",
        "row_count": len(rows),
        "rows": [row.to_payload() for row in rows],
        "global_prediction_seal_hash": prediction.seal_hash,
        "probability_store_hash": prediction.store.store_hash,
        "surface_hash": surface_hash,
        "exact_nine_seed_mean": True,
        "target_expert_used": False,
        "labels_used": False,
    }
    if payload != expected_payload or rows != replayed_rows:
        raise ProtocolError("Signed-error probability surface hash drifted.")
    seed_csv = _read_csv(root / "tables/seed_probability_rows.csv")
    expected_seed_csv = tuple(
        {
            str(key): _csv_value(value)
            for key, value in row.to_payload().items()
        }
        for row in replayed_seed_rows
    )
    if seed_csv != expected_seed_csv:
        raise ProtocolError("Signed-error seed probability CSV differs from replay.")
    csv_rows = _read_csv(root / "tables/aggregated_probability_rows.csv")
    expected_probability_csv = tuple(
        {
            str(key): _csv_value(value)
            for key, value in row.to_payload().items()
        }
        for row in rows
    )
    if csv_rows != expected_probability_csv:
        raise ProtocolError("Signed-error probability CSV differs from its seal.")
    return rows, surface_hash


def _load_and_replay_models(
    root: Path,
    probabilities: tuple[SampleActionProbability, ...],
    protocol_contract_hash: str,
) -> SignedModelProducts:
    manifest = read_json(root / "manifests/signed_loco_model_seals.json")
    correction_manifest = read_json(
        root / "manifests/signed_correction_surface_seals.json"
    )
    targets_raw = manifest.get("targets")
    if not isinstance(targets_raw, list):
        raise ProtocolError("Signed-error model target manifest is absent.")
    correction_rows = _load_corrections(root)
    by_key = {
        (row.target_center, row.family): [] for row in correction_rows
    }
    for row in correction_rows:
        by_key[(row.target_center, row.family)].append(row)
    target_fits: list[TargetFamilyFits] = []
    for target_payload in targets_raw:
        if not isinstance(target_payload, Mapping):
            raise ProtocolError("Signed-error target model payload is malformed.")
        target = str(target_payload["target_center"])
        families_raw = target_payload.get("families")
        if not isinstance(families_raw, list) or [
            row.get("family") for row in families_raw if isinstance(row, Mapping)
        ] != ["G", "R", "P"]:
            raise ProtocolError("Signed-error model family coverage drifted.")
        fits = tuple(_fit_from_payload(row) for row in families_raw)
        target_fit = TargetFamilyFits(
            target,
            fits[0],
            fits[1],
            fits[2],
            tuple(sorted(by_key[(target, "G")])),
            tuple(sorted(by_key[(target, "R")])),
            tuple(sorted(by_key[(target, "P")])),
        )
        expected_target_payload = {
            "target_center": target,
            "model_seal_hash": target_fit.model_seal_hash,
            "families": families_raw,
            "target_labels_used": False,
        }
        if dict(target_payload) != expected_target_payload:
            raise ProtocolError("Signed-error target model seal hash drifted.")
        _replay_target_corrections(probabilities, target_fit)
        target_fits.append(target_fit)
    residual_rows = tuple(
        row for row in correction_rows if row.family == "R"
    )
    control_rows = tuple(
        row for row in correction_rows if row.family in ("G", "P")
    )
    products = SignedModelProducts(
        tuple(target_fits),
        correction_surface_hash(residual_rows, surface="raw"),
        correction_surface_hash(residual_rows, surface="safe"),
        correction_surface_hash(control_rows, surface="combined"),
        protocol_contract_hash,
    )
    model_unhashed = {
        "schema_version": "fixed_bank_signed_error_all_loco_models_v1",
        "target_family_count": len(targets_raw),
        "targets": targets_raw,
        "protocol_contract_hash": protocol_contract_hash,
        "all_G_R_and_P_models_sealed_before_same_H_support": True,
        "outer_H_labels_used": False,
        "target_expert_used": False,
        "permutation_is_separate_same_capacity_refit": True,
    }
    expected_model_manifest = {
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
    expected_correction_manifest = {
        **correction_unhashed,
        "correction_manifest_hash": canonical_hash(correction_unhashed),
    }
    if (
        manifest != expected_model_manifest
        or correction_manifest != expected_correction_manifest
    ):
        raise ProtocolError("Signed-error model/correction manifest replay drifted.")
    _validate_model_tables(root, tuple(targets_raw))
    return products


def _validate_model_tables(
    root: Path, targets: tuple[object, ...]
) -> None:
    expected_models: list[dict[str, object]] = []
    expected_alpha: list[dict[str, object]] = []
    for target_payload in targets:
        if not isinstance(target_payload, Mapping):
            raise ProtocolError("Signed-error model-table source is malformed.")
        for family_payload in target_payload["families"]:  # type: ignore[union-attr]
            if not isinstance(family_payload, Mapping):
                raise ProtocolError("Signed-error model family table is malformed.")
            final = family_payload["final_model"]
            if not isinstance(final, Mapping):
                raise ProtocolError("Signed-error final model table is malformed.")
            expected_models.append(
                _model_csv_row(
                    final,
                    role="final",
                    query="",
                    fit_hash=str(family_payload["fit_hash"]),
                )
            )
            for nested in family_payload["nested_models"]:  # type: ignore[union-attr]
                if not isinstance(nested, Mapping) or not isinstance(
                    nested.get("model"), Mapping
                ):
                    raise ProtocolError("Signed-error nested model table is malformed.")
                expected_models.append(
                    _model_csv_row(
                        nested["model"],
                        role="nested",
                        query=str(nested["heldout_query_center"]),
                        fit_hash=str(family_payload["fit_hash"]),
                    )
                )
            for alpha, mse in family_payload["validation_mse_by_alpha"]:  # type: ignore[union-attr]
                expected_alpha.append(
                    {
                        "target_center": target_payload["target_center"],
                        "family": family_payload["family"],
                        "ridge_alpha": alpha,
                        "validation_mse": mse,
                        "selected": float(alpha) == float(final["ridge_alpha"]),
                        "fit_hash": family_payload["fit_hash"],
                    }
                )
    for member, expected in (
        ("tables/signed_loco_models.csv", expected_models),
        ("tables/signed_alpha_path.csv", expected_alpha),
    ):
        observed = _read_csv(root / member)
        canonical = tuple(
            {str(key): _csv_value(value) for key, value in row.items()}
            for row in expected
        )
        if observed != canonical:
            raise ProtocolError(f"Signed-error model table drifted: {member}.")


def _model_csv_row(
    payload: Mapping[str, object], *, role: str, query: str, fit_hash: str
) -> dict[str, object]:
    return {
        "target_center": payload["target_center"],
        "family": payload["family"],
        "role": role,
        "heldout_query_center": query,
        "ridge_alpha": payload["ridge_alpha"],
        "coefficients": payload["coefficients"],
        "means": payload["means"],
        "scales": payload["scales"],
        "donor_centers": payload["donor_centers"],
        "nested_model_hashes": payload["nested_model_hashes"],
        "model_hash": payload["model_hash"],
        "fit_hash": fit_hash,
    }


def _fit_from_payload(payload: Mapping[str, object]) -> SignedGateFit:
    final_raw = payload.get("final_model")
    nested_raw = payload.get("nested_models")
    path_raw = payload.get("validation_mse_by_alpha")
    if (
        not isinstance(final_raw, Mapping)
        or not isinstance(nested_raw, list)
        or not isinstance(path_raw, list)
        or len(nested_raw) != 8
        or not all(
            isinstance(row, Mapping) and isinstance(row.get("model"), Mapping)
            for row in nested_raw
        )
        or len(path_raw) != 3
        or not all(isinstance(row, list) and len(row) == 2 for row in path_raw)
    ):
        raise ProtocolError("Signed-error fit payload is malformed.")
    fit = SignedGateFit(
        _model_from_payload(final_raw),
        tuple(
            NestedSignedGateModel(
                str(row["heldout_query_center"]),
                _model_from_payload(row["model"]),
            )
            for row in nested_raw
        ),
        tuple((float(row[0]), float(row[1])) for row in path_raw),
    )
    expected = {
        "family": fit.final_model.family,
        "final_model": dict(final_raw),
        "nested_models": [dict(row) for row in nested_raw],
        "validation_mse_by_alpha": [
            [alpha, mse] for alpha, mse in fit.validation_mse_by_alpha
        ],
        "fit_hash": fit.fit_hash,
    }
    if dict(payload) != expected:
        raise ProtocolError("Signed-error fit hash drifted.")
    return fit


def _model_from_payload(payload: Mapping[str, object]) -> SignedGateModel:
    model = SignedGateModel(
        str(payload["target_center"]),
        str(payload["family"]),
        float(payload["ridge_alpha"]),
        tuple(float(value) for value in payload["coefficients"]),  # type: ignore[union-attr]
        Standardization(
            tuple(float(value) for value in payload["means"]),  # type: ignore[union-attr]
            tuple(float(value) for value in payload["scales"]),  # type: ignore[union-attr]
        ),
        tuple(str(value) for value in payload["donor_centers"]),  # type: ignore[union-attr]
        tuple(str(value) for value in payload["nested_model_hashes"]),  # type: ignore[union-attr]
    )
    expected = {
        "schema_version": "fixed_bank_signed_error_model_v1",
        "target_center": model.target_center,
        "family": model.family,
        "ridge_alpha": model.ridge_alpha,
        "coefficients": list(model.coefficients),
        "means": list(model.standardization.means),
        "scales": list(model.standardization.scales),
        "donor_centers": list(model.donor_centers),
        "nested_model_hashes": list(model.nested_model_hashes),
        "response": "class_balanced_rescaled_negative_log_loss_logit_gradient",
        "ridge_objective": "unweighted_mse_on_rescaled_gradient_target",
        "target_labels_used": False,
        "model_hash": model.model_hash,
    }
    if dict(payload) != expected:
        raise ProtocolError("Signed-error model hash drifted.")
    return model


def _load_corrections(root: Path) -> tuple[CorrectionRow, ...]:
    rows = _read_csv(root / "tables/signed_corrections.csv")
    corrections = tuple(
        CorrectionRow(
            row["target_center"],
            row["case_id"],
            row["sample_id"],
            row["family"],
            float(row["raw_correction"]),
            float(row["correction_standard_error"]),
            float(row["safe_correction"]),
            _bool(row["uncertainty_admitted"]),
        )
        for row in rows
    )
    expected = tuple(
        {str(key): _csv_value(value) for key, value in row.to_payload().items()}
        for row in corrections
    )
    if rows != expected:
        raise ProtocolError("Signed-error correction-row hash drifted.")
    return corrections


def _replay_target_corrections(
    probabilities: tuple[SampleActionProbability, ...], target_fit: TargetFamilyFits
) -> None:
    target = target_fit.target_center
    outer = build_signed_features(probabilities, excluded_candidate_centers=(target,))
    nested = {
        query: build_signed_features(
            probabilities, excluded_candidate_centers=(target, query)
        )
        for query in (
            value
            for value in ("0", "1", "2", "3", "5", "6", "7", "8", "9")
            if value != target
        )
    }
    outer_p = permute_feature_alignment(outer)
    nested_p = {query: permute_feature_alignment(rows) for query, rows in nested.items()}
    observed = (
        predict_corrections(
            target_fit.global_fit, outer, nested_prediction_features=nested
        ),
        predict_corrections(
            target_fit.residual_fit, outer, nested_prediction_features=nested
        ),
        predict_corrections(
            target_fit.permutation_fit,
            outer_p,
            nested_prediction_features=nested_p,
        ),
    )
    expected = (
        target_fit.global_corrections,
        target_fit.residual_corrections,
        target_fit.permutation_corrections,
    )
    if observed != expected:
        raise ProtocolError("Signed-error correction surface differs from model replay.")


def _load_fold_products(root: Path, protocol_contract_hash: str) -> SignedFoldProducts:
    manifest = read_json(root / "manifests/signed_fold_products.json")
    decisions = manifest.get("decisions")
    if not isinstance(decisions, list) or not all(
        isinstance(row, Mapping) for row in decisions
    ):
        raise ProtocolError("Signed-error fold decisions are malformed.")
    prediction_rows = _read_csv(root / "tables/oof_predictions.csv")
    predictions: dict[str, list[PredictionRow]] = {
        method: [] for method in ("B", "B_cal", "G", "R_raw", "R_safe", "P")
    }
    for row in prediction_rows:
        prediction = PredictionRow(
            row["method_id"],
            row["target_center"],
            row["case_id"],
            row["sample_id"],
            float(row["probability"]),
            int(row["hard_prediction"]),
        )
        if prediction.prediction_hash != row["prediction_hash"]:
            raise ProtocolError("Signed-error OOF prediction hash drifted.")
        predictions[prediction.method_id].append(prediction)
    decision_seal = read_json(
        root / "manifests/all_fold_method_decisions_seal.json"
    )
    permutation_seal = read_json(
        root / "manifests/permutation_provenance_seal.json"
    )
    products = SignedFoldProducts(
        tuple(dict(row) for row in decisions),
        {method: tuple(sorted(rows)) for method, rows in predictions.items()},
        str(decision_seal["decision_seal_hash"]),
        str(permutation_seal["permutation_provenance_hash"]),
        str(manifest["partition_hash"]),
        protocol_contract_hash,
    )
    expected_prediction_rows = tuple(
        {
            str(key): _csv_value(value)
            for key, value in row.to_payload().items()
        }
        for method in products.predictions_by_method
        for row in products.predictions_by_method[method]
    )
    if prediction_rows != expected_prediction_rows:
        raise ProtocolError("Signed-error OOF prediction table differs from replay.")
    expected_surface_hashes = {
        method: canonical_hash([row.to_payload() for row in rows])
        for method, rows in products.predictions_by_method.items()
    }
    expected_manifest = {
        "schema_version": "fixed_bank_signed_error_fold_products_v1",
        "decision_count": len(products.decisions),
        "decisions": [dict(value) for value in products.decisions],
        "method_prediction_row_counts": {
            method: len(rows)
            for method, rows in products.predictions_by_method.items()
        },
        "method_prediction_surface_hashes": expected_surface_hashes,
        "partition_hash": products.partition_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "evaluation_labels_used": False,
    }
    expected_decision_seal = {
        "schema_version": "fixed_bank_signed_error_all_fold_decisions_v1",
        "decision_count": 45 * 6,
        "fold_decision_count": 45,
        "decision_hashes": [row["decision_hash"] for row in products.decisions],
        "decision_seal_hash": products.decision_seal_hash,
        "R_raw_and_R_safe_prediction_hashes_separate": True,
        "evaluation_labels_used": False,
    }
    expected_permutation_seal = {
        "schema_version": "fixed_bank_signed_error_permutation_provenance_v1",
        "permutation_provenance_hash": products.permutation_provenance_hash,
        "complete_sample_feature_blocks_permuted": True,
        "labels_and_gradients_preserved": True,
        "separate_same_capacity_model_refit": True,
        "evaluation_labels_used": False,
    }
    if (
        manifest != expected_manifest
        or decision_seal != expected_decision_seal
        or permutation_seal != expected_permutation_seal
    ):
        raise ProtocolError("Signed-error prediction surface manifest drifted.")
    _validate_fold_tables(root, products.decisions)
    return products


def _validate_fold_tables(
    root: Path, decisions: tuple[Mapping[str, object], ...]
) -> None:
    expected_decisions: list[dict[str, object]] = []
    expected_lambda: list[dict[str, object]] = []
    for decision in decisions:
        expected_decisions.append(
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
        for row in decision["lambda_path"]:  # type: ignore[union-attr]
            expected_lambda.append(
                {
                    "target_center": decision["target_center"],
                    "fold_ordinal": decision["fold_ordinal"],
                    **dict(row),
                    "decision_hash": decision["decision_hash"],
                }
            )
    for member, expected in (
        ("tables/fold_decisions.csv", expected_decisions),
        ("tables/lambda_path.csv", expected_lambda),
    ):
        observed = _read_csv(root / member)
        canonical = tuple(
            {str(key): _csv_value(value) for key, value in row.items()}
            for row in expected
        )
        if observed != canonical:
            raise ProtocolError(f"Signed-error fold table drifted: {member}.")


def _validate_reports(
    root: Path,
    evaluation: Mapping[str, object],
    protocol_contract_hash: str,
    *,
    config: object,
    expected_input_hashes: Mapping[str, str],
    expected_firewall: Mapping[str, object],
    source_cache: object,
    prediction_capability: object,
    expected_leakage: Mapping[str, object],
) -> None:
    publication = read_json(root / "reports/publication_decision.json")
    leakage = read_json(root / "reports/leakage_report.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    preflight = read_json(root / "reports/workstation_preflight.json")
    protocol_manifest = read_json(root / "manifests/protocol_manifest.json")
    signed_protocol = protocol_manifest.get("signed_protocol")
    input_hashes = protocol_manifest.get("input_artifact_hashes")
    firewall = protocol_manifest.get("pre_gpu_firewall")
    local_staging = runtime.get("local_source_staging")
    config_runtime = getattr(config, "runtime")
    _validate_preflight_report(preflight, config_runtime)
    if Path(str(preflight["disk_probe_path"])).resolve() != root.resolve():
        raise ProtocolError("Signed-error preflight disk probe path drifted.")
    if not isinstance(local_staging, Mapping):
        raise ProtocolError("Signed-error runtime staging report is malformed.")
    _validate_local_staging(local_staging, preflight)
    expected_runtime = runtime_summary_payload(
        source_cache=source_cache,
        prediction_capability=prediction_capability,
        local_staging=local_staging,
        runtime=config_runtime,
    )
    if (
        publication != publication_decision_payload(evaluation)
        or leakage != dict(expected_leakage)
        or runtime != expected_runtime
        or not isinstance(signed_protocol, Mapping)
        or dict(signed_protocol) != canonical_consumed_test_protocol().to_payload()
        or signed_protocol.get("contract_hash") != protocol_contract_hash
        or not isinstance(input_hashes, Mapping)
        or dict(input_hashes) != dict(expected_input_hashes)
        or not isinstance(firewall, Mapping)
        or dict(firewall) != dict(expected_firewall)
        or protocol_manifest.get("experiment_id")
        != getattr(config, "experiment_id")
        or protocol_manifest.get("output_artifact_id")
        != getattr(config, "output_artifact_id")
        or protocol_manifest.get("config_contract_hash")
        != getattr(config, "contract_hash")
        or protocol_manifest.get("runtime") != dict(getattr(config, "runtime"))
        or protocol_manifest.get("evaluation")
        != dict(getattr(config, "evaluation"))
        or protocol_manifest.get("claim_boundary")
        != dict(getattr(config, "claim_boundary"))
        or protocol_manifest.get("fresh_evidence") is not False
    ):
        raise ProtocolError("Signed-error claim-bound reports drifted.")


def _validate_preflight_report(
    payload: Mapping[str, object], runtime: Mapping[str, object]
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "generation_devices",
        "persistent_gpu_workers",
        "classifier_workers",
        "blas_threads_per_classifier_worker",
        "gpu_then_cpu_phase_order",
        "phase_disjoint_gpu_and_cpu_pools",
        "parent_cuda_initialized",
        "tf32_enabled",
        "amp_enabled",
        "scratch_preference",
        "available_cpu_affinity_count",
        "physical_ram_bytes",
        "disk_probe_path",
        "disk_free_bytes_at_launch",
        "thread_environment",
        "cuda_visible_devices",
        "package_versions",
        "gpus",
        "source_generation_devices",
        "probability_materialization_device",
        "probability_materialization_workers",
        "probability_store_format",
        "context_feature_format",
        "maximum_concurrent_target_context_builds",
        "cross_target_context_cache_present",
        "resume_strategy",
    }
    versions = payload.get("package_versions")
    gpus = payload.get("gpus")
    fixed = {
        "schema_version": "midogpp_label_free_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "gpu_then_cpu_phase_order": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_initialized": False,
        "tf32_enabled": False,
        "amp_enabled": False,
        "scratch_preference": list(runtime["scratch_preference"]),
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": "0,1",
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": 4,
        "probability_store_format": "compressed_float32_npz",
        "context_feature_format": "bounded_process_local_float64_target_contexts",
        "maximum_concurrent_target_context_builds": 4,
        "cross_target_context_cache_present": False,
        "resume_strategy": runtime["resume_policy"],
    }
    if (
        set(payload) != expected_keys
        or any(payload.get(key) != value for key, value in fixed.items())
        or type(payload.get("available_cpu_affinity_count")) is not int
        or int(payload["available_cpu_affinity_count"])
        < int(runtime["minimum_logical_cpu_count"])
        or type(payload.get("physical_ram_bytes")) is not int
        or int(payload["physical_ram_bytes"])
        < int(runtime["minimum_physical_ram_bytes"])
        or type(payload.get("disk_free_bytes_at_launch")) is not int
        or int(payload["disk_free_bytes_at_launch"])
        < int(runtime["minimum_artifact_disk_free_bytes"])
        or not isinstance(payload.get("disk_probe_path"), str)
        or not payload["disk_probe_path"]
        or not isinstance(versions, Mapping)
        or set(versions) != set(REQUIRED_DISTRIBUTIONS)
        or any(not isinstance(value, str) or not value for value in versions.values())
        or not isinstance(gpus, list)
        or len(gpus) != 2
    ):
        raise ProtocolError("Signed-error workstation preflight report drifted.")
    for index, row in enumerate(gpus):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "index",
                "name",
                "memory_total_mib",
                "memory_free_mib",
            }
            or row.get("index") != index
            or "RTX A5000" not in str(row.get("name"))
            or type(row.get("memory_total_mib")) is not int
            or type(row.get("memory_free_mib")) is not int
            or int(row["memory_free_mib"])
            < int(runtime["minimum_gpu_free_mib_per_device"])
        ):
            raise ProtocolError("Signed-error workstation GPU report drifted.")


def _validate_local_staging(
    payload: Mapping[str, object], preflight: Mapping[str, object]
) -> None:
    allowed = {"attempted", "used", "status", "failure", "workstation_preflight"}
    if (
        not {"attempted", "used", "status", "workstation_preflight"}.issubset(
            payload
        )
        or not set(payload).issubset(allowed)
        or payload.get("attempted") is not True
        or type(payload.get("used")) is not bool
        or payload.get("workstation_preflight") != preflight
    ):
        raise ProtocolError("Signed-error local staging report drifted.")
    used = bool(payload["used"])
    failure = payload.get("failure")
    expected_status = (
        "CANONICAL_FALLBACK"
        if failure is not None
        else "STAGED_LOCAL_CPU_CACHE"
        if used
        else "CANONICAL_ALREADY_LOCAL"
    )
    if (
        payload.get("status") != expected_status
        or (failure is not None and (used or not isinstance(failure, str)))
    ):
        raise ProtocolError("Signed-error local staging status is inconsistent.")


def _validate_terminal_tables(root: Path, evaluation: object) -> None:
    scientific = evaluation.scientific_result
    confusion_expected: list[dict[str, object]] = []
    metric_expected: list[dict[str, object]] = []
    for method in scientific.method_results:
        for row in method.case_confusions:
            confusion_expected.append(
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
        metric_expected.extend(row.to_payload() for row in method.center_metrics)
    contrast_expected = [row.to_payload() for row in scientific.contrasts]
    for member, expected in (
        ("tables/terminal_case_confusions.csv", confusion_expected),
        ("tables/terminal_center_metrics.csv", metric_expected),
        ("tables/terminal_contrasts.csv", contrast_expected),
    ):
        observed = _read_csv(root / member)
        canonical = tuple(
            {str(key): _csv_value(value) for key, value in row.items()}
            for row in expected
        )
        if observed != canonical:
            raise ProtocolError(f"Signed-error terminal table drifted: {member}.")


def _csv_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read signed-error table: {path}.") from exc


def _bool(value: object) -> bool:
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise ProtocolError("Signed-error boolean table value is malformed.")


__all__ = ("validate_fixed_bank_signed_error_gate_bundle",)
