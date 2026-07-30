"""Atomic execution of prospective within-center uniform-B confirmation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_existing_directory

from ..artifacts import prepare_artifact_dirs, write_csv_rows, write_json
from ..classifiers import ClassifierSpec, fit_logistic_classifier
from ..downstream import balanced_accuracy, macro_f1
from ..physical_multiscale_center_pooling.decision_lock import (
    classifier_spec_from_lock,
    read_decision_lock,
)
from ..protocol import ProtocolError
from ..uniform_b_replay.frames import ABFrame, UniformBShardedStore
from .artifacts import (
    confirmation_input_hashes,
    prospective_paired_case_bootstrap,
    read_json,
    validate_confirmation_inputs,
    write_content_index,
)
from .config import CANONICAL_A, EVALUATION_SPLIT, UNIFORM_B, UniformBConfirmationConfig
from .workspace_binding import validate_production_workspace_binding


def run_uniform_b_confirmation(config: UniformBConfirmationConfig) -> Path:
    if not config.allow_partial_test_coverage:
        validate_production_workspace_binding(config)
        _validate_prepared_root(config.artifact_root)
        with staged_existing_directory(config.artifact_root) as stage:
            _run_in_place(replace(config, artifact_root=stage))
        return config.artifact_root
    return _run_in_place(config)


def _run_in_place(config: UniformBConfirmationConfig) -> Path:
    from .validation import validate_uniform_b_confirmation_bundle

    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root)
    validate_confirmation_inputs(config)
    hashes = confirmation_input_hashes(config)
    locks = tuple(
        read_decision_lock(
            config.source_v3_root / "manifests/decision_locks" / f"center_{center}.json"
        )
        for center in config.heldout_centers
    )
    _validate_locks(config, locks)
    frozen, representation_lock, source_index = _protocol_payloads(config, hashes, locks)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(root / "manifests/uniform_representation_lock.json", representation_lock)
    write_json(root / "manifests/source_lock_index.json", source_index)

    train_store = UniformBShardedStore(config.source_train_b_cache_root)
    test_store = UniformBShardedStore(config.test_b_cache_root)
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    splits: list[dict[str, object]] = []
    lock_rows: list[dict[str, object]] = []
    for lock in locks:
        heldout = str(lock.payload["outer_target_center"])
        source = train_store.source_frame(
            heldout=heldout, eligible_centers=config.heldout_centers
        )
        target = test_store.target_frame(heldout)
        sample_overlap = set(source.sample_ids) & set(target.sample_ids)
        case_overlap = set(source.case_ids) & set(target.case_ids)
        if sample_overlap or case_overlap or set(target.centers) != {heldout}:
            raise ProtocolError(f"Uniform-B prospective split isolation failed: {heldout}.")
        split_row = {
            "heldout_center": heldout,
            "training_split": "train",
            "evaluation_split": EVALUATION_SPLIT,
            "n_source_train": len(source.sample_ids),
            "n_target_test": len(target.sample_ids),
            "sample_overlap": 0,
            "case_overlap": 0,
            "target_center_absent_from_fit": heldout not in set(source.centers),
            "validation_split_used": False,
            "status": "PASS",
        }
        splits.append(split_row)
        fitted_by_role = {}
        for role, representation in (("canonical_a", CANONICAL_A), ("uniform_b", UNIFORM_B)):
            fitted = _fit(
                source,
                target,
                spec=classifier_spec_from_lock(lock, representation),
                representation=representation,
                role=role,
                heldout=heldout,
                source_decision_hash=lock.decision_hash,
            )
            fitted_by_role[role] = fitted
            results.append(fitted["result"])
            predictions.extend(fitted["predictions"])
            audits.append(fitted["audit"])
        a = fitted_by_role["canonical_a"]["result"]
        b = fitted_by_role["uniform_b"]["result"]
        comparisons.append(
            {
                "schema_version": "midogpp_uniform_b_prospective_center_comparison_v1",
                "heldout_center": heldout,
                "n_eval": b["n_eval"],
                "canonical_a_bacc": a["bacc"],
                "uniform_b_bacc": b["bacc"],
                "delta_bacc": float(b["bacc"]) - float(a["bacc"]),
                "canonical_a_macro_f1": a["macro_f1"],
                "uniform_b_macro_f1": b["macro_f1"],
                "delta_macro_f1": float(b["macro_f1"]) - float(a["macro_f1"]),
                "source_decision_hash": lock.decision_hash,
                "evaluation_split": EVALUATION_SPLIT,
                "outcome_was_untouched_at_protocol_lock": True,
            }
        )
        lock_rows.append(
            {
                "heldout_center": heldout,
                "source_decision_hash": lock.decision_hash,
                "canonical_a_classifier_hash": classifier_spec_from_lock(lock, CANONICAL_A).config_hash,
                "uniform_b_classifier_hash": classifier_spec_from_lock(lock, UNIFORM_B).config_hash,
                "classifier_selection_used_test_labels": False,
                "representation_locked_before_test_outcomes": True,
            }
        )
    bootstrap = prospective_paired_case_bootstrap(
        predictions,
        seed=config.bootstrap.seed,
        valid_replicates=config.bootstrap.valid_replicates,
        max_attempts=config.bootstrap.max_attempts,
    )
    summary = _summary(config, comparisons, bootstrap)
    write_csv_rows(root / "tables/source_lock_audit.csv", lock_rows)
    write_csv_rows(root / "tables/split_isolation_audit.csv", splits)
    write_csv_rows(root / "tables/prospective_test_results.csv", results)
    write_csv_rows(root / "tables/prospective_test_predictions.csv", predictions)
    write_csv_rows(root / "tables/paired_center_comparison.csv", comparisons)
    write_csv_rows(root / "tables/outer_fit_audit.csv", audits)
    write_json(root / "reports/conditional_bootstrap.json", bootstrap)
    write_json(root / "reports/confirmation_summary.json", summary)
    (root / "reports/confirmation_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_uniform_b_prospective_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "fit_leakage_status": "PASS",
            "train_test_sample_overlap": 0,
            "train_test_case_overlap": 0,
            "test_labels_used_for_fit_selection_or_feature_extraction": False,
            "test_labels_used_for_final_scoring_only": True,
            "validation_split_used": False,
            "independent_confirmation_within_observed_centers": True,
            "external_dataset_confirmation": False,
            "covers_new_center_uncertainty": False,
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_uniform_b_prospective_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - started,
            "outer_fit_count": len(results),
            "prediction_count": len(predictions),
        },
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_prospective_protocol_manifest_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": frozen["protocol_hash"],
            "uniform_lock_hash": representation_lock["uniform_lock_hash"],
            "source_lock_bundle_hash": source_index["source_lock_bundle_hash"],
            "input_hashes": hashes,
            "claim_scope": "diagnostic_only",
            "evidence_role": "prospective_within_center_uniform_b_confirmation",
            "evaluation_split": EVALUATION_SPLIT,
            "independent_confirmation_within_observed_centers": True,
            "external_dataset_confirmation": False,
            "covers_new_case_uncertainty_within_centers": True,
            "covers_new_center_uncertainty": False,
            "may_replace_canonical_reference": False,
            "may_feed_recipe_selection": False,
            "may_feed_deployable_selection": False,
        },
    )
    write_content_index(root)
    pending = validate_uniform_b_confirmation_bundle(root, config=config, allow_pending=True)
    _finalize(root, pending)
    validate_uniform_b_confirmation_bundle(root, config=config)
    return root


def _fit(
    source: ABFrame,
    target: ABFrame,
    *,
    spec: ClassifierSpec,
    representation: str,
    role: str,
    heldout: str,
    source_decision_hash: str,
) -> dict[str, object]:
    fitted = fit_logistic_classifier(
        source.embeddings[representation],
        source.labels,
        target.embeddings[representation],
        spec=spec,
    )
    if not fitted.converged:
        raise ProtocolError(f"Uniform-B prospective fit did not converge: {heldout}/{role}.")
    pred = [int(value) for value in fitted.predictions.tolist()]
    prob = [float(row[1]) for row in fitted.probabilities.tolist()]
    identity = stable_hash(
        {
            "heldout": heldout,
            "representation": representation,
            "classifier_hash": fitted.classifier_config_hash,
            "scaler_hash": fitted.scaler_state_hash,
            "source_samples": source.sample_ids,
            "source_decision_hash": source_decision_hash,
            "evaluation_split": EVALUATION_SPLIT,
        }
    )
    result = {
        "schema_version": "midogpp_uniform_b_prospective_result_v1",
        "heldout_center": heldout,
        "role": role,
        "representation_id": representation,
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "fit_identity": identity,
        "bacc": balanced_accuracy(target.labels.tolist(), pred),
        "macro_f1": macro_f1(target.labels.tolist(), pred),
        "n_train": len(source.sample_ids),
        "n_eval": len(target.sample_ids),
        "evaluation_split": EVALUATION_SPLIT,
        "source_decision_hash": source_decision_hash,
        "test_labels_used_for_scoring_only": True,
        "claim_scope": "diagnostic_only",
    }
    rows = [
        {
            "schema_version": "midogpp_uniform_b_prospective_prediction_v1",
            "heldout_center": heldout,
            "role": role,
            "representation_id": representation,
            "sample_id": sample,
            "case_id": case,
            "label": int(target.labels[index]),
            "prediction": pred[index],
            "probability_positive": prob[index],
            "evaluation_split": EVALUATION_SPLIT,
            "source_decision_hash": source_decision_hash,
            "test_labels_used_for_scoring_only": True,
        }
        for index, (sample, case) in enumerate(zip(target.sample_ids, target.case_ids, strict=True))
    ]
    audit = {
        "heldout_center": heldout,
        "role": role,
        "representation_id": representation,
        "fit_identity": identity,
        "source_centers": ",".join(sorted(set(source.centers), key=int)),
        "target_center_absent_from_fit": heldout not in set(source.centers),
        "scaler_fit_on_train_source_only": True,
        "test_labels_used_for_fit_or_selection": False,
    }
    return {"result": result, "predictions": rows, "audit": audit}


def _protocol_payloads(
    config: UniformBConfirmationConfig,
    hashes: Mapping[str, str],
    locks: tuple[object, ...],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    frozen = {
        "schema_version": "midogpp_uniform_b_prospective_frozen_protocol_v1",
        "experiment_name": config.name,
        "heldout_centers": list(config.heldout_centers),
        "representations": {CANONICAL_A: 2560, UNIFORM_B: 3840},
        "training_split": "train",
        "evaluation_split": EVALUATION_SPLIT,
        "validation_split_used": False,
        "primary_estimand": "equal_center_mean_test_bacc_uniform_b_minus_canonical_a",
        "confirmation_rule": config.confirmation_rule.__dict__,
        "bootstrap": config.bootstrap.__dict__,
        "representation_locked_before_test_b_extraction": True,
        "test_b_outcomes_previously_observed": False,
        "independent_confirmation_within_observed_centers": True,
        "external_dataset_confirmation": False,
        "covers_new_center_uncertainty": False,
        "input_hashes": dict(sorted(hashes.items())),
    }
    frozen["protocol_hash"] = stable_hash(frozen)
    representation = {
        "schema_version": "midogpp_uniform_b_prospective_representation_lock_v1",
        "representation_id": UNIFORM_B,
        "feature_dim": 3840,
        "applies_to_centers": list(config.heldout_centers),
        "evaluation_split": EVALUATION_SPLIT,
        "choice_informed_by_train_outer_scores": True,
        "choice_pre_test_b_extraction_and_scoring": True,
        "test_outcomes_used_for_choice": False,
        "protocol_hash": frozen["protocol_hash"],
    }
    representation["uniform_lock_hash"] = stable_hash(representation)
    rows = [
        {
            "heldout_center": str(lock.payload["outer_target_center"]),
            "decision_hash": str(lock.decision_hash),
        }
        for lock in locks
    ]
    source_index = {
        "schema_version": "midogpp_uniform_b_prospective_source_lock_index_v1",
        "locks": rows,
        "lock_count": len(rows),
        "classifier_locks_precede_test_b_extraction_and_scoring": True,
    }
    source_index["source_lock_bundle_hash"] = stable_hash(source_index)
    return frozen, representation, source_index


def _summary(
    config: UniformBConfirmationConfig,
    comparisons: list[dict[str, object]],
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    mean_a = sum(float(row["canonical_a_bacc"]) for row in comparisons) / len(comparisons)
    mean_b = sum(float(row["uniform_b_bacc"]) for row in comparisons) / len(comparisons)
    deltas = [float(row["delta_bacc"]) for row in comparisons]
    rule = config.confirmation_rule
    checks = {
        "mean_delta": mean_b - mean_a >= rule.minimum_mean_bacc_delta,
        "strict_center_wins": sum(delta > 0 for delta in deltas) >= rule.minimum_strict_center_wins,
        "worst_center_delta": min(deltas) >= rule.minimum_worst_center_delta,
        "bootstrap_lower_bound": (
            float(bootstrap["percentile_2_5"]) > 0
            if rule.require_bootstrap_lower_bound_above_zero
            else True
        ),
    }
    confirmed = all(checks.values())
    return {
        "schema_version": "midogpp_uniform_b_prospective_confirmation_summary_v1",
        "status": "COMPLETE",
        "decision": "CONFIRMED_WITHIN_CENTER" if confirmed else "NOT_CONFIRMED_WITHIN_CENTER",
        "confirmation_passed": confirmed,
        "claim_scope": "diagnostic_only",
        "evaluation_split": EVALUATION_SPLIT,
        "equal_center_mean_canonical_a_bacc": mean_a,
        "equal_center_mean_uniform_b_bacc": mean_b,
        "paired_mean_delta": mean_b - mean_a,
        "strict_wins": sum(delta > 0 for delta in deltas),
        "worst_center_delta": min(deltas),
        "confirmation_rule": rule.__dict__,
        "confirmation_checks": checks,
        "conditional_bootstrap": dict(bootstrap),
        "independent_confirmation_within_observed_centers": True,
        "external_dataset_confirmation": False,
        "covers_new_center_uncertainty": False,
        "may_replace_canonical_reference": False,
    }


def _render_report(summary: Mapping[str, object]) -> str:
    return "\n".join(
        (
            "# MIDOG++ Uniform-B v3 Prospective Test Confirmation",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            f"- Uniform-B test mean BACC: `{float(summary['equal_center_mean_uniform_b_bacc']):.6f}`",
            f"- Canonical-A test mean BACC: `{float(summary['equal_center_mean_canonical_a_bacc']):.6f}`",
            f"- Paired mean delta: `{float(summary['paired_mean_delta']):+.6f}`",
            f"- Strict center wins: `{int(summary['strict_wins'])}/9`",
            f"- Worst center delta: `{float(summary['worst_center_delta']):+.6f}`",
            "",
            "This is prospective confirmation on case-disjoint test rows from the same",
            "nine centers. It does not cover external-dataset or new-center uncertainty.",
            "",
        )
    )


def _validate_locks(config: UniformBConfirmationConfig, locks: tuple[object, ...]) -> None:
    centers = tuple(str(lock.payload["outer_target_center"]) for lock in locks)
    if centers != config.heldout_centers:
        raise ProtocolError("Uniform-B prospective lock coverage/order drifted.")
    for lock in locks:
        classifier_spec_from_lock(lock, CANONICAL_A)
        classifier_spec_from_lock(lock, UNIFORM_B)


def _finalize(root: Path, checks: Mapping[str, object]) -> None:
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = read_json(protocol_path)
    protocol["status"] = "PASS"
    protocol["independent_validation_status"] = "PASS"
    write_json(protocol_path, protocol)
    leakage_path = root / "reports/leakage_provenance_report.json"
    leakage = read_json(leakage_path)
    leakage["status"] = "PASS"
    leakage["independent_validation_status"] = "PASS"
    write_json(leakage_path, leakage)
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_prospective_validation_report_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_confirmation_bundle",
            "authoritative_bundle_verdict": True,
            "checks": dict(checks),
        },
    )
    write_content_index(root)


def _validate_prepared_root(root: Path) -> None:
    expected = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests",
        "reports",
        "tables",
        "provenance",
    }
    if not root.is_dir() or {str(path.relative_to(root)) for path in root.rglob("*")} != expected:
        raise ProtocolError("Uniform-B prospective production root was not prepared exactly.")
    if any(any((root / name).iterdir()) for name in ("manifests", "reports", "tables")):
        raise ProtocolError("Uniform-B prospective prepared output directories are not empty.")
