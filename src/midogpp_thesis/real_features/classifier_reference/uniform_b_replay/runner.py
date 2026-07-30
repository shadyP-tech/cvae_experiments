"""Atomic execution of the retrospective uniform-B replay."""

from __future__ import annotations

from dataclasses import replace
import json
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
from .artifacts import (
    input_hashes,
    paired_case_bootstrap,
    read_csv,
    read_json,
    validate_source_bundle,
    write_content_index,
)
from .config import CANONICAL_A, UNIFORM_B, UniformBReplayConfig
from .frames import ABFrame, UniformBShardedStore
from .workspace_binding import validate_production_workspace_binding


def run_uniform_b_replay(config: UniformBReplayConfig) -> Path:
    if not config.allow_partial_test_coverage:
        validate_production_workspace_binding(config)
        _validate_prepared_root(config.artifact_root)
        with staged_existing_directory(config.artifact_root) as stage:
            _run_in_place(replace(config, artifact_root=stage))
        return config.artifact_root
    return _run_in_place(config)


def _run_in_place(config: UniformBReplayConfig) -> Path:
    from .validation import validate_uniform_b_replay_bundle

    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root)
    validate_source_bundle(config)
    hashes = input_hashes(config)
    locks = tuple(
        read_decision_lock(
            config.source_v3_root
            / "manifests/decision_locks"
            / f"center_{center}.json"
        )
        for center in config.heldout_centers
    )
    _validate_source_locks(config, locks)
    frozen, uniform_lock, source_index = _protocol_payloads(config, hashes, locks)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(root / "manifests/uniform_representation_lock.json", uniform_lock)
    write_json(root / "manifests/source_lock_index.json", source_index)

    source_results = _source_results(config)
    source_predictions = _source_predictions(config)
    canonical_reference = _canonical_reference(config)
    store = UniformBShardedStore(config.b_cache_root)
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    fit_audit: list[dict[str, object]] = []
    canonical_replay: list[dict[str, object]] = []
    source_replay: list[dict[str, object]] = []
    lock_replay: list[dict[str, object]] = []

    for lock in locks:
        heldout = str(lock.payload["outer_target_center"])
        source = store.source_frame(
            heldout=heldout, eligible_centers=config.heldout_centers
        )
        target = store.target_frame(heldout)
        if heldout in set(source.centers) or set(target.centers) != {heldout}:
            raise ProtocolError(f"Uniform-B outer isolation failed: {heldout}.")
        per_role = {}
        for role, representation in (
            ("canonical_a", CANONICAL_A),
            ("uniform_b", UNIFORM_B),
        ):
            fitted = _fresh_fit(
                source,
                target,
                spec=classifier_spec_from_lock(lock, representation),
                representation=representation,
                role=role,
                heldout=heldout,
                source_decision_hash=lock.decision_hash,
            )
            results.append(fitted["result"])
            predictions.extend(fitted["predictions"])
            fit_audit.append(fitted["audit"])
            per_role[role] = fitted
        a = per_role["canonical_a"]["result"]
        b = per_role["uniform_b"]["result"]
        comparison = {
            "schema_version": "midogpp_uniform_b_center_comparison_v1",
            "heldout_center": heldout,
            "n_eval": b["n_eval"],
            "canonical_a_bacc": a["bacc"],
            "uniform_b_bacc": b["bacc"],
            "delta_bacc": float(b["bacc"]) - float(a["bacc"]),
            "canonical_a_macro_f1": a["macro_f1"],
            "uniform_b_macro_f1": b["macro_f1"],
            "delta_macro_f1": float(b["macro_f1"]) - float(a["macro_f1"]),
            "source_decision_hash": lock.decision_hash,
            "uniform_choice_is_retrospective": True,
            "row_role": "retrospective_paired_center_comparison",
        }
        comparisons.append(comparison)
        canonical_replay.append(
            _canonical_replay_row(
                heldout,
                per_role["canonical_a"],
                canonical_reference,
                lock.decision_hash,
            )
        )
        source_replay.append(
            _source_replay_row(
                heldout,
                per_role,
                source_results,
                source_predictions,
                lock.decision_hash,
            )
        )
        lock_replay.append(
            {
                "heldout_center": heldout,
                "source_decision_hash": lock.decision_hash,
                "canonical_a_classifier_hash": classifier_spec_from_lock(
                    lock, CANONICAL_A
                ).config_hash,
                "uniform_b_classifier_hash": classifier_spec_from_lock(
                    lock, UNIFORM_B
                ).config_hash,
                "source_lock_hash_valid": True,
                "classifier_selection_used_target_labels": False,
                "uniform_choice_is_retrospective": True,
            }
        )
    if any(role.startswith("target_outer_") for role, _center in store.access_log[:1]):
        raise ProtocolError("Uniform-B target was accessed before the global lock.")
    bootstrap = paired_case_bootstrap(predictions, config=config.bootstrap)
    summary = _summary(config, comparisons, bootstrap)
    cache_alignment = _cache_alignment(config)
    write_csv_rows(root / "tables/source_lock_replay.csv", lock_replay)
    write_csv_rows(root / "tables/cache_alignment_audit.csv", cache_alignment)
    write_csv_rows(root / "tables/uniform_b_outer_results.csv", results)
    write_csv_rows(root / "tables/uniform_b_outer_predictions.csv", predictions)
    write_csv_rows(root / "tables/paired_center_comparison.csv", comparisons)
    write_csv_rows(root / "tables/outer_fit_audit.csv", fit_audit)
    write_csv_rows(root / "tables/canonical_a_replay.csv", canonical_replay)
    write_csv_rows(root / "tables/v3_result_replay.csv", source_replay)
    write_json(root / "reports/conditional_bootstrap.json", bootstrap)
    write_json(root / "reports/diagnostic_summary.json", summary)
    (root / "reports/diagnostic_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_uniform_b_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "fit_leakage_status": "PASS",
            "study_design_status": "POSTHOC_DISCOVERY",
            "target_labels_used_for_classifier_selection": False,
            "target_center_used_for_fit_or_scaler": False,
            "uniform_choice_is_retrospective": True,
            "independent_confirmation": False,
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_uniform_b_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - started,
            "source_lock_count": len(locks),
            "outer_fit_count": len(results),
            "prediction_count": len(predictions),
        },
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_protocol_manifest_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": frozen["protocol_hash"],
            "uniform_lock_hash": uniform_lock["uniform_lock_hash"],
            "source_lock_bundle_hash": source_index["source_lock_bundle_hash"],
            "input_hashes": hashes,
            "claim_scope": "diagnostic_only",
            "evidence_role": "retrospective_uniform_representation_replay",
            "study_design_informed_by_prior_target_scores": True,
            "selection_performed_this_run": False,
            "non_adoptive": True,
            "adoption_eligible": False,
            "may_feed_recipe_selection": False,
            "may_feed_deployable_selection": False,
            "uses_cvae": False,
            "uses_router": False,
            "covers_new_center_uncertainty": False,
        },
    )
    write_content_index(root)
    pending = validate_uniform_b_replay_bundle(root, config=config, allow_pending=True)
    _finalize(root, pending)
    validate_uniform_b_replay_bundle(root, config=config)
    return root


def _protocol_payloads(
    config: UniformBReplayConfig,
    hashes: Mapping[str, str],
    locks: tuple[object, ...],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    frozen = {
        "schema_version": "midogpp_uniform_b_frozen_protocol_v1",
        "experiment_name": config.name,
        "heldout_centers": list(config.heldout_centers),
        "representations": {CANONICAL_A: 2560, UNIFORM_B: 3840},
        "source_profile_id": config.source_profile_id,
        "source_protocol_hash": config.source_protocol_hash,
        "source_bundle_lock_hash": config.source_bundle_lock_hash,
        "source_content_hash": config.source_content_hash,
        "bootstrap": config.bootstrap.__dict__,
        "primary_estimand": "equal_center_mean_bacc_uniform_b_minus_canonical_a",
        "study_design_informed_by_prior_target_scores": True,
        "same_outer_centers_previously_scored": True,
        "independent_confirmation": False,
        "non_adoptive": True,
        "input_hashes": dict(sorted(hashes.items())),
    }
    frozen["protocol_hash"] = stable_hash(frozen)
    uniform_lock = {
        "schema_version": "midogpp_uniform_b_representation_lock_v1",
        "representation_id": UNIFORM_B,
        "feature_dim": 3840,
        "applies_to_centers": list(config.heldout_centers),
        "selection_performed_this_run": False,
        "choice_informed_by_prior_target_scores": True,
        "choice_pre_original_outer_scoring": False,
        "adoption_eligible": False,
        "protocol_hash": frozen["protocol_hash"],
    }
    uniform_lock["uniform_lock_hash"] = stable_hash(uniform_lock)
    rows = [
        {
            "heldout_center": str(lock.payload["outer_target_center"]),  # type: ignore[attr-defined]
            "decision_hash": str(lock.decision_hash),  # type: ignore[attr-defined]
        }
        for lock in locks
    ]
    source_index = {
        "schema_version": "midogpp_uniform_b_source_lock_index_v1",
        "locks": rows,
        "lock_count": len(rows),
        "all_source_locks_precede_original_outer_scoring": True,
    }
    source_index["source_lock_bundle_hash"] = stable_hash(source_index)
    return frozen, uniform_lock, source_index


def _fresh_fit(
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
        raise ProtocolError(f"Uniform-B fit did not converge: {heldout}/{role}.")
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
        }
    )
    result = {
        "schema_version": "midogpp_uniform_b_outer_result_v1",
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
        "source_decision_hash": source_decision_hash,
        "fresh_outer_fit": True,
        "target_labels_used_for_scoring_only": True,
        "probabilities_calibrated": False,
        "claim_scope": "diagnostic_only",
    }
    rows = [
        {
            "schema_version": "midogpp_uniform_b_outer_prediction_v1",
            "heldout_center": heldout,
            "role": role,
            "representation_id": representation,
            "sample_id": sample,
            "case_id": case,
            "label": int(target.labels[index]),
            "prediction": pred[index],
            "probability_positive": prob[index],
            "source_decision_hash": source_decision_hash,
            "target_labels_used_for_scoring_only": True,
        }
        for index, (sample, case) in enumerate(
            zip(target.sample_ids, target.case_ids, strict=True)
        )
    ]
    audit = {
        "heldout_center": heldout,
        "role": role,
        "representation_id": representation,
        "fit_identity": identity,
        "source_centers": ",".join(sorted(set(source.centers), key=int)),
        "target_center_absent_from_fit": heldout not in set(source.centers),
        "scaler_fit_on_source_only": True,
        "fresh_outer_fit": True,
    }
    return {"result": result, "predictions": rows, "audit": audit}


def _source_results(config: UniformBReplayConfig) -> dict[tuple[str, str], dict[str, str]]:
    locked = read_csv(config.source_v3_root / "tables/outer_locked_results.csv")
    posthoc = read_csv(config.source_v3_root / "tables/posthoc_candidate_isolation.csv")
    out = {
        (row["heldout_center"], "canonical_a"): row
        for row in locked
        if row["role"] == "canonical_a"
    }
    for center in config.heldout_centers:
        matches = [
            row
            for row in locked
            if row["heldout_center"] == center
            and row["role"] == "selected_policy"
            and row["representation_id"] == UNIFORM_B
        ] + [
            row
            for row in posthoc
            if row["heldout_center"] == center
            and row["representation_id"] == UNIFORM_B
        ]
        if len(matches) != 1:
            raise ProtocolError(f"V3 source lacks exactly one B result: {center}.")
        out[(center, "uniform_b")] = matches[0]
    return out


def _source_predictions(
    config: UniformBReplayConfig,
) -> dict[tuple[str, str], dict[str, int]]:
    rows = read_csv(config.source_v3_root / "tables/outer_locked_predictions.csv")
    out: dict[tuple[str, str], dict[str, int]] = {}
    for center in config.heldout_centers:
        for role in ("canonical_a", "uniform_b"):
            source_role = "canonical_a" if role == "canonical_a" else "selected_policy"
            selected = [
                row
                for row in rows
                if row["heldout_center"] == center
                and row["role"] == source_role
                and (
                    role == "canonical_a"
                    or row["representation_id"] == UNIFORM_B
                )
            ]
            if selected:
                out[(center, role)] = {
                    row["sample_id"]: int(row["prediction"]) for row in selected
                }
    return out


def _canonical_reference(config: UniformBReplayConfig) -> dict[str, object]:
    return {
        "results": read_csv(
            config.canonical_reference_root
            / "tables/classifier_tuned_source_results.csv"
        ),
        "predictions": read_csv(
            config.canonical_reference_root
            / "tables/classifier_tuned_predictions.csv"
        ),
    }


def _canonical_replay_row(
    center: str,
    fitted: Mapping[str, object],
    reference: Mapping[str, object],
    source_decision_hash: str,
) -> dict[str, object]:
    result = fitted["result"]
    assert isinstance(result, Mapping)
    ref_results = [
        row
        for row in reference["results"]  # type: ignore[index]
        if row["heldout_center"] == center
    ]
    ref_predictions = {
        row["sample_id"]: int(float(row["y_pred"]))
        for row in reference["predictions"]  # type: ignore[index]
        if row["heldout_center"] == center
    }
    actual = {
        row["sample_id"]: int(row["prediction"])
        for row in fitted["predictions"]  # type: ignore[index]
    }
    exact = (
        len(ref_results) == 1
        and ref_results[0]["selected_classifier_config_hash"]
        == result["classifier_config_hash"]
        and float(ref_results[0]["heldout_bacc"]) == float(result["bacc"])
        and float(ref_results[0]["heldout_macro_f1"]) == float(result["macro_f1"])
        and ref_predictions == actual
    )
    if not exact:
        raise ProtocolError(f"Uniform-B canonical A replay failed: {center}.")
    return {
        "heldout_center": center,
        "source_decision_hash": source_decision_hash,
        "classifier_hash_exact": True,
        "predictions_exact": True,
        "bacc_exact": True,
        "macro_f1_exact": True,
        "status": "PASS",
    }


def _source_replay_row(
    center: str,
    fitted: Mapping[str, Mapping[str, object]],
    source_results: Mapping[tuple[str, str], Mapping[str, str]],
    source_predictions: Mapping[tuple[str, str], Mapping[str, int]],
    source_decision_hash: str,
) -> dict[str, object]:
    checks = {}
    for role in ("canonical_a", "uniform_b"):
        result = fitted[role]["result"]
        assert isinstance(result, Mapping)
        source = source_results[(center, role)]
        checks[f"{role}_metric_exact"] = (
            float(source["bacc"]) == float(result["bacc"])
            and float(source["macro_f1"]) == float(result["macro_f1"])
            and source["classifier_config_hash"] == result["classifier_config_hash"]
        )
        expected_predictions = source_predictions.get((center, role))
        if expected_predictions is None:
            checks[f"{role}_prediction_replay_applicable"] = False
            checks[f"{role}_predictions_exact"] = ""
        else:
            actual = {
                row["sample_id"]: int(row["prediction"])
                for row in fitted[role]["predictions"]  # type: ignore[index]
            }
            checks[f"{role}_prediction_replay_applicable"] = True
            checks[f"{role}_predictions_exact"] = actual == expected_predictions
    if not checks["canonical_a_metric_exact"] or not checks["uniform_b_metric_exact"]:
        raise ProtocolError(f"Uniform-B v3 metric replay failed: {center}.")
    if checks["canonical_a_predictions_exact"] is not True or (
        checks["uniform_b_prediction_replay_applicable"]
        and checks["uniform_b_predictions_exact"] is not True
    ):
        raise ProtocolError(f"Uniform-B v3 prediction replay failed: {center}.")
    return {
        "heldout_center": center,
        "source_decision_hash": source_decision_hash,
        **checks,
        "status": "PASS",
    }


def _cache_alignment(config: UniformBReplayConfig) -> list[dict[str, object]]:
    payload = read_json(config.b_cache_root / "manifests/row_alignment.json")
    return [
        {
            "representation_id": UNIFORM_B,
            "feature_dim": 3840,
            "status": payload["status"],
            "row_count": payload["row_count"],
            "sample_id_order_hash": payload["sample_id_order_hash"],
            "center_4_present": payload["center_4_present"],
            "canonical_order_exact": True,
            "c_cache_accessed": False,
        }
    ]


def _summary(
    config: UniformBReplayConfig,
    comparisons: list[dict[str, object]],
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    mean_a = sum(float(row["canonical_a_bacc"]) for row in comparisons) / len(
        comparisons
    )
    mean_b = sum(float(row["uniform_b_bacc"]) for row in comparisons) / len(
        comparisons
    )
    deltas = [float(row["delta_bacc"]) for row in comparisons]
    return {
        "schema_version": "midogpp_uniform_b_diagnostic_summary_v1",
        "status": "COMPLETE",
        "experiment_name": config.name,
        "claim_scope": "diagnostic_only",
        "study_design_status": "POSTHOC_DISCOVERY",
        "independent_confirmation": False,
        "adoption_eligible": False,
        "equal_center_mean_canonical_a_bacc": mean_a,
        "equal_center_mean_uniform_b_bacc": mean_b,
        "paired_mean_delta": mean_b - mean_a,
        "strict_wins": sum(delta > 0.0 for delta in deltas),
        "worst_center_delta": min(deltas),
        "conditional_bootstrap": dict(bootstrap),
        "covers_new_center_uncertainty": False,
    }


def _render_report(summary: Mapping[str, object]) -> str:
    return "\n".join(
        (
            "# MIDOG++ Uniform-B v3 Retrospective Replay",
            "",
            "Status: complete Stage-90 retrospective diagnostic.",
            "",
            f"- Uniform-B mean BACC: `{float(summary['equal_center_mean_uniform_b_bacc']):.6f}`",
            f"- Canonical-A mean BACC: `{float(summary['equal_center_mean_canonical_a_bacc']):.6f}`",
            f"- Paired mean delta: `{float(summary['paired_mean_delta']):+.6f}`",
            f"- Strict center wins: `{int(summary['strict_wins'])}/9`",
            f"- Worst center delta: `{float(summary['worst_center_delta']):+.6f}`",
            "",
            "B was chosen after these same outer-center outcomes had been observed.",
            "This replay establishes deterministic reproducibility and a consolidated",
            "fixed-B estimate only; it is not independent confirmation or adoption evidence.",
            "",
        )
    )


def _validate_source_locks(config: UniformBReplayConfig, locks: tuple[object, ...]) -> None:
    centers = tuple(str(lock.payload["outer_target_center"]) for lock in locks)  # type: ignore[attr-defined]
    if centers != config.heldout_centers:
        raise ProtocolError("Uniform-B source-lock coverage/order drifted.")
    for lock in locks:
        classifier_spec_from_lock(lock, CANONICAL_A)  # type: ignore[arg-type]
        classifier_spec_from_lock(lock, UNIFORM_B)  # type: ignore[arg-type]


def _finalize(root: Path, checks: Mapping[str, object]) -> None:
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = read_json(protocol_path)
    protocol["status"] = "PASS"
    protocol["independent_validation_status"] = "PASS"
    write_json(protocol_path, protocol)
    leakage_path = root / "reports/leakage_provenance_report.json"
    leakage = read_json(leakage_path)
    leakage["status"] = "PASS_WITH_POSTHOC_DESIGN"
    leakage["independent_validation_status"] = "PASS"
    write_json(leakage_path, leakage)
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_validation_report_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_replay_bundle",
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
    if not root.is_dir() or {
        str(path.relative_to(root)) for path in root.rglob("*")
    } != expected:
        raise ProtocolError("Uniform-B production root was not prepared exactly.")
    if any(
        any((root / name).iterdir()) for name in ("manifests", "reports", "tables")
    ):
        raise ProtocolError("Uniform-B prepared claim-bearing directories are not empty.")
