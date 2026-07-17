"""Fail-closed table, metric, summary, runtime, and bundle validation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .downstream import balanced_accuracy, macro_f1
from .fixed_c_risk_protocol_validation import (
    FixedCRiskValidationInputs,
    ReconstructedFixedCRiskFold,
    validate_fixed_c_risk_protocol_and_inputs,
)
from .fixed_c_risk_reporting import (
    build_diagnostic_summary,
    render_diagnostic_report,
)
from .protocol import ProtocolError
from .real_feature_frame import RealFeatureFrame
from .schemas.fixed_c_risk_diagnostic import (
    FIXED_C_RISK_METHOD,
    FIXED_C_RISK_PAIRED_COLUMNS,
    FIXED_C_RISK_PAIRED_SCHEMA_VERSION,
    FIXED_C_RISK_PREDICTION_COLUMNS,
    FIXED_C_RISK_PREDICTION_SCHEMA_VERSION,
    FIXED_C_RISK_REQUIRED_OUTPUTS,
    FIXED_C_RISK_RESULT_COLUMNS,
    FIXED_C_RISK_RESULT_SCHEMA_VERSION,
    FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS,
    FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION,
    FIXED_CLASSIFIER_CONFIG_HASH,
    PRIMARY_CONTRAST,
    PRIOR_METHOD,
    RISK_POLICY_FORMULAS,
    RISK_POLICY_IDS,
    SELECTION_SOURCE,
    WEIGHT_NORMALIZATION,
    ZERO_CELL_POLICY,
    fixed_c_risk_bundle_hash,
    risk_policy_hash,
)
from .artifacts import stable_hash


def assert_fixed_c_risk_artifacts(
    root: Path,
    already_loaded_frame: RealFeatureFrame | None = None,
) -> None:
    """Validate every persisted contract and independently recomputable value."""

    root = Path(root)
    missing = [
        relative
        for relative in FIXED_C_RISK_REQUIRED_OUTPUTS
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(f"Fixed-C risk artifact missing outputs: {missing}")

    protocol = _read_json(root / "manifests/protocol_manifest.json")
    frozen = _read_json(root / "manifests/frozen_protocol_snapshot.json")
    leakage = _read_json(root / "reports/leakage_provenance_report.json")
    summary = _read_json(root / "reports/diagnostic_summary.json")
    runtime = _read_json(root / "reports/runtime_summary.json")
    results = _read_csv(root / "tables/fixed_c_risk_results.csv")
    predictions = _read_csv(root / "tables/fixed_c_risk_predictions.csv")
    audits = _read_csv(root / "tables/fixed_c_risk_weight_audit.csv")
    paired = _read_csv(root / "tables/fixed_c_risk_paired_comparison.csv")

    reconstructed = validate_fixed_c_risk_protocol_and_inputs(
        root,
        protocol,
        frozen,
        leakage,
        already_loaded_frame=already_loaded_frame,
    )
    _assert_columns(results, FIXED_C_RISK_RESULT_COLUMNS, "fixed_c_risk_results.csv")
    _assert_columns(
        predictions,
        FIXED_C_RISK_PREDICTION_COLUMNS,
        "fixed_c_risk_predictions.csv",
    )
    _assert_columns(
        audits,
        FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS,
        "fixed_c_risk_weight_audit.csv",
    )
    _assert_columns(
        paired,
        FIXED_C_RISK_PAIRED_COLUMNS,
        "fixed_c_risk_paired_comparison.csv",
    )
    bundle_hash = fixed_c_risk_bundle_hash(results, predictions, audits, paired)
    if protocol.get("bundle_hash") != bundle_hash:
        raise ProtocolError("Fixed-C risk content bundle hash mismatch.")
    for report in (leakage, summary, runtime):
        if (
            report.get("protocol_hash") != protocol.get("protocol_hash")
            or report.get("bundle_hash") != bundle_hash
        ):
            raise ProtocolError("Fixed-C risk report is not bound to protocol/bundle.")

    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    expected_keys = {
        (heldout, policy) for heldout in heldouts for policy in RISK_POLICY_IDS
    }
    results_by_key = _unique_by_pair(results, "fixed-C result")
    audits_by_key = _unique_by_pair(audits, "fixed-C weight audit")
    if set(results_by_key) != expected_keys or set(audits_by_key) != expected_keys:
        raise ProtocolError("Fixed-C risk result/audit coverage is incomplete.")
    if len(results) != int(protocol["expected_fit_count"]):
        raise ProtocolError("Fixed-C risk fit coverage differs from protocol.")
    prediction_keys = {
        (row.get("heldout_center", ""), row.get("risk_policy_id", ""))
        for row in predictions
    }
    if not prediction_keys or not prediction_keys.issubset(expected_keys):
        raise ProtocolError("Fixed-C risk prediction keys are invalid.")
    if len(paired) != len(heldouts):
        raise ProtocolError("Fixed-C risk paired coverage differs from heldouts.")

    for heldout in heldouts:
        fold = reconstructed.folds[heldout]
        fold_results = [results_by_key[(heldout, policy)] for policy in RISK_POLICY_IDS]
        fold_audits = [audits_by_key[(heldout, policy)] for policy in RISK_POLICY_IDS]
        _validate_fold_identity(fold, fold_results, fold_audits)
        for policy in RISK_POLICY_IDS:
            row = results_by_key[(heldout, policy)]
            audit = audits_by_key[(heldout, policy)]
            _validate_result_row(row, protocol, fold, policy)
            _validate_weight_audit(audit, row, protocol, fold, policy)
            fold_predictions = [
                item
                for item in predictions
                if item.get("heldout_center") == heldout
                and item.get("risk_policy_id") == policy
            ]
            _validate_predictions(
                fold_predictions,
                row,
                protocol,
                fold,
                policy,
            )
    if len(predictions) != sum(int(row["n_eval"]) for row in results):
        raise ProtocolError("Fixed-C risk total prediction coverage mismatch.")
    _validate_weight_vectors_against_inputs(reconstructed, audits_by_key)

    paired_by_center = {
        str(row.get("heldout_center", "")): row for row in paired
    }
    if set(paired_by_center) != set(heldouts) or len(paired_by_center) != len(paired):
        raise ProtocolError("Fixed-C risk paired rows are duplicate or incomplete.")
    for heldout in heldouts:
        _validate_paired(
            paired_by_center[heldout],
            results_by_key[(heldout, "domain_class")],
            results_by_key[(heldout, "pooled")],
            protocol,
        )
    _validate_summary(summary, results, paired, protocol)
    expected_report = render_diagnostic_report(summary)
    if (
        root / "reports/diagnostic_report.md"
    ).read_text(encoding="utf-8") != expected_report:
        raise ProtocolError("Fixed-C risk diagnostic report does not match summary.")
    _validate_runtime(runtime, protocol)


def _validate_fold_identity(
    fold: ReconstructedFixedCRiskFold,
    results: Sequence[Mapping[str, str]],
    audits: Sequence[Mapping[str, str]],
) -> None:
    for field in (
        "train_centers",
        "fit_row_hash",
        "eval_row_hash",
        "training_frame_hash",
        "scaler_state_hash",
        "n_train",
        "n_eval",
    ):
        values = {row.get(field, "") for row in results}
        if len(values) != 1:
            raise ProtocolError(
                f"Fixed-C risk cross-arm {field} mismatch in results for "
                f"{fold.heldout_center}."
            )
    for field in (
        "train_centers",
        "fit_row_hash",
        "training_frame_hash",
        "scaler_state_hash",
        "n_fit",
        "n_domains",
    ):
        values = {row.get(field, "") for row in audits}
        if len(values) != 1:
            raise ProtocolError(
                f"Fixed-C risk cross-arm {field} mismatch in weight audits for "
                f"{fold.heldout_center}."
            )
    try:
        result_train_centers = tuple(json.loads(results[0]["train_centers"]))
        audit_train_centers = tuple(json.loads(audits[0]["train_centers"]))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("Fixed-C risk train-center payload is malformed.") from exc
    if (
        result_train_centers != fold.train_centers
        or audit_train_centers != fold.train_centers
        or results[0]["fit_row_hash"] != fold.fit_row_hash
        or audits[0]["fit_row_hash"] != fold.fit_row_hash
        or results[0]["eval_row_hash"] != fold.eval_row_hash
        or results[0]["training_frame_hash"] != fold.training_frame_hash
        or audits[0]["training_frame_hash"] != fold.training_frame_hash
        or int(results[0]["n_train"]) != len(fold.fit_rows)
        or int(audits[0]["n_fit"]) != len(fold.fit_rows)
        or int(results[0]["n_eval"]) != len(fold.eval_rows)
        or int(audits[0]["n_domains"]) != len(fold.train_centers)
    ):
        raise ProtocolError("Fixed-C risk reconstructed fold identity mismatch.")
    for result, audit in zip(results, audits):
        for field in (
            "fit_row_hash",
            "training_frame_hash",
            "scaler_state_hash",
            "weight_vector_hash",
        ):
            if result[field] != audit[field]:
                raise ProtocolError(f"Fixed-C risk result/audit {field} mismatch.")


def _validate_result_row(
    row: Mapping[str, str],
    protocol: Mapping[str, object],
    fold: ReconstructedFixedCRiskFold,
    policy: str,
) -> None:
    expected = {
        "schema_version": FIXED_C_RISK_RESULT_SCHEMA_VERSION,
        "method": FIXED_C_RISK_METHOD,
        "protocol_hash": str(protocol["protocol_hash"]),
        "experiment_seed": str(protocol["experiment_seed"]),
        "classifier_seed": str(protocol["classifier_seed"]),
        "heldout_center": fold.heldout_center,
        "risk_policy_id": policy,
        "risk_policy_formula": RISK_POLICY_FORMULAS[policy],
        "risk_policy_hash": risk_policy_hash(policy),
        "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
        "status": "ok",
        "manifest_hash": str(protocol["manifest_hash"]),
        "feature_cache_hash": str(protocol["feature_cache_hash"]),
        "prior_method": PRIOR_METHOD,
        "threshold_policy": "predict",
        "selection_source": SELECTION_SOURCE,
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "target_rows_used_for_fit": "false",
        "generated_embeddings_used": "false",
        "cvae_checkpoint_used": "false",
        "is_router": "false",
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "risk_weighting_diagnostic",
        "row_role": "diagnostic_result",
        "diagnostic_only": "true",
        "non_adoptive": "true",
        "adoption_eligible": "false",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "leakage_status": "PASS",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProtocolError(f"Fixed-C risk result field {field} mismatch.")
    if row.get("sample_weight_passed_to_fit") != (
        "false" if policy == "pooled" else "true"
    ):
        raise ProtocolError("Fixed-C risk sample-weight fit flag mismatch.")
    try:
        persisted_spec = json.loads(row["fixed_classifier_spec"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("Fixed-C risk result classifier spec is malformed.") from exc
    if (
        not isinstance(persisted_spec, Mapping)
        or dict(persisted_spec) != dict(protocol["fixed_classifier_spec"])  # type: ignore[arg-type]
    ):
        raise ProtocolError("Fixed-C risk result classifier spec drifted.")
    if row.get("converged") != "true":
        raise ProtocolError("Fixed-C risk classifier did not converge.")
    try:
        n_iter = json.loads(row["n_iter"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("Fixed-C risk result n_iter is malformed.") from exc
    max_iter = int(persisted_spec["max_iter"])
    if (
        not isinstance(n_iter, list)
        or not n_iter
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value >= max_iter
            for value in n_iter
        )
    ):
        raise ProtocolError("Fixed-C risk result n_iter is invalid.")
    for field in (
        "weight_vector_hash",
        "fit_row_hash",
        "eval_row_hash",
        "training_frame_hash",
        "scaler_state_hash",
    ):
        if not row.get(field):
            raise ProtocolError(f"Fixed-C risk result lacks {field}.")
    for field in ("heldout_bacc", "heldout_macro_f1"):
        value = float(row[field])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError(f"Fixed-C risk result {field} is invalid.")


def _validate_predictions(
    rows: Sequence[Mapping[str, str]],
    result: Mapping[str, str],
    protocol: Mapping[str, object],
    fold: ReconstructedFixedCRiskFold,
    policy: str,
) -> None:
    if len(rows) != int(result["n_eval"]) or len(rows) != len(fold.eval_rows):
        raise ProtocolError("Fixed-C risk prediction coverage mismatch.")
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []
    for row, expected_eval in zip(rows, fold.eval_rows):
        expected = {
            "schema_version": FIXED_C_RISK_PREDICTION_SCHEMA_VERSION,
            "method": FIXED_C_RISK_METHOD,
            "protocol_hash": str(protocol["protocol_hash"]),
            "heldout_center": fold.heldout_center,
            "risk_policy_id": policy,
            "risk_policy_hash": risk_policy_hash(policy),
            "weight_vector_hash": result["weight_vector_hash"],
            "sample_id": expected_eval.sample_id,
            "case_id": expected_eval.case_id,
            "center": fold.heldout_center,
            "y_true": str(expected_eval.label),
            "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
            "fit_row_hash": result["fit_row_hash"],
            "eval_row_hash": result["eval_row_hash"],
            "training_frame_hash": result["training_frame_hash"],
            "scaler_state_hash": result["scaler_state_hash"],
            "prior_method": PRIOR_METHOD,
            "selection_source": SELECTION_SOURCE,
            "claim_role": "risk_weighting_diagnostic",
            "row_role": "diagnostic_prediction",
            "diagnostic_only": "true",
            "non_adoptive": "true",
            "adoption_eligible": "false",
            "support_labels_used": "false",
            "oracle_eligible": "false",
            "target_eval_labels_used_for_scoring_only": "true",
            "leakage_status": "PASS",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ProtocolError(f"Fixed-C risk prediction field {field} mismatch.")
        truth, predicted, probability = (
            int(row["y_true"]),
            int(row["y_pred"]),
            float(row["prob_pos"]),
        )
        if (
            truth not in {0, 1}
            or predicted not in {0, 1}
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or predicted != int(probability > 0.5)
        ):
            raise ProtocolError("Invalid fixed-C risk prediction values.")
        y_true.append(truth)
        y_pred.append(predicted)
        sample_ids.append(row["sample_id"])
    if len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError("Fixed-C risk predictions contain duplicate sample IDs.")
    if tuple(sample_ids) != tuple(row.sample_id for row in fold.eval_rows):
        raise ProtocolError("Fixed-C risk prediction row order differs from inputs.")
    if not math.isclose(
        balanced_accuracy(y_true, y_pred),
        float(result["heldout_bacc"]),
        abs_tol=1e-12,
    ):
        raise ProtocolError("Fixed-C risk heldout BACC does not recompute.")
    if not math.isclose(
        macro_f1(y_true, y_pred),
        float(result["heldout_macro_f1"]),
        abs_tol=1e-12,
    ):
        raise ProtocolError("Fixed-C risk heldout macro-F1 does not recompute.")


def _validate_weight_audit(
    audit: Mapping[str, str],
    result: Mapping[str, str],
    protocol: Mapping[str, object],
    fold: ReconstructedFixedCRiskFold,
    policy: str,
) -> None:
    expected = {
        "schema_version": FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION,
        "method": FIXED_C_RISK_METHOD,
        "protocol_hash": str(protocol["protocol_hash"]),
        "heldout_center": fold.heldout_center,
        "risk_policy_id": policy,
        "risk_policy_formula": RISK_POLICY_FORMULAS[policy],
        "risk_policy_hash": risk_policy_hash(policy),
        "weight_vector_hash": result["weight_vector_hash"],
        "fit_row_hash": result["fit_row_hash"],
        "training_frame_hash": result["training_frame_hash"],
        "scaler_state_hash": result["scaler_state_hash"],
        "fixed_classifier_config_hash": FIXED_CLASSIFIER_CONFIG_HASH,
        "normalization": WEIGHT_NORMALIZATION,
        "zero_cell_policy": ZERO_CELL_POLICY,
        "all_weights_finite": "true",
        "all_weights_positive": "true",
        "target_rows_used": "false",
        "scaler_fit_used_sample_weight": "false",
        "sample_weight_passed_to_fit": "false" if policy == "pooled" else "true",
        "prior_method": PRIOR_METHOD,
        "selection_source": SELECTION_SOURCE,
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "risk_weighting_diagnostic",
        "row_role": "fit_level_weight_audit",
        "diagnostic_only": "true",
        "non_adoptive": "true",
        "adoption_eligible": "false",
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "status": "PASS",
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ProtocolError(f"Fixed-C risk audit field {field} mismatch.")
    counts = _json_number_map(audit["group_counts"], integer=True)
    weights = _json_number_map(audit["group_weights"], integer=False)
    masses = _json_number_map(audit["group_masses"], integer=False)
    if set(counts) != set(weights) or set(counts) != set(masses):
        raise ProtocolError("Fixed-C risk audit group keys differ.")
    n_fit = int(audit["n_fit"])
    n_domains = int(audit["n_domains"])
    if (
        n_fit != int(result["n_train"])
        or n_fit != len(fold.fit_rows)
        or n_domains != len(fold.train_centers)
        or sum(int(value) for value in counts.values()) != n_fit
    ):
        raise ProtocolError("Fixed-C risk audit group counts do not sum to N.")
    expected_group_count = {
        "pooled": 1,
        "global_class": 2,
        "domain": n_domains,
        "domain_class": 2 * n_domains,
    }[policy]
    if len(counts) != expected_group_count:
        raise ProtocolError("Fixed-C risk audit group coverage mismatch.")
    expected_weight = {
        key: _formula_weight(policy, n_fit, n_domains, int(count))
        for key, count in counts.items()
    }
    for key in counts:
        if int(counts[key]) <= 0:
            raise ProtocolError("Fixed-C risk audit contains a zero group cell.")
        if not math.isclose(weights[key], expected_weight[key], abs_tol=1e-12):
            raise ProtocolError("Fixed-C risk audit formula invariant failed.")
        expected_mass = float(counts[key]) * expected_weight[key]
        if not math.isclose(masses[key], expected_mass, abs_tol=1e-12):
            raise ProtocolError("Fixed-C risk audit group-mass invariant failed.")
    expanded_min = min(weights.values())
    expanded_max = max(weights.values())
    total = sum(masses.values())
    if (
        not math.isclose(float(audit["weight_min"]), expanded_min, abs_tol=1e-12)
        or not math.isclose(float(audit["weight_max"]), expanded_max, abs_tol=1e-12)
        or not math.isclose(float(audit["weight_sum"]), total, abs_tol=1e-10)
        or not math.isclose(
            float(audit["expected_weight_sum"]),
            float(n_fit),
            abs_tol=1e-12,
        )
        or not math.isclose(total, float(n_fit), abs_tol=1e-10)
    ):
        raise ProtocolError("Fixed-C risk audit normalization invariant failed.")


def _validate_weight_vectors_against_inputs(
    reconstructed: FixedCRiskValidationInputs,
    audits: Mapping[tuple[str, str], Mapping[str, str]],
) -> None:
    for heldout, fold in reconstructed.folds.items():
        fit_hash = fold.fit_row_hash
        for policy in RISK_POLICY_IDS:
            audit = audits[(heldout, policy)]
            weights_by_group = _json_number_map(
                audit["group_weights"],
                integer=False,
            )
            actual_counts: dict[str, int] = {}
            ordered_weights: list[float] = []
            for row in fold.fit_rows:
                if policy == "pooled":
                    key = "all"
                elif policy == "global_class":
                    key = f"class={row.label}"
                elif policy == "domain":
                    key = f"domain={row.center}"
                else:
                    key = f"domain={row.center}|class={row.label}"
                if key not in weights_by_group:
                    raise ProtocolError(
                        "Fixed-C risk audit omits an observed fit group."
                    )
                actual_counts[key] = actual_counts.get(key, 0) + 1
                ordered_weights.append(weights_by_group[key])
            persisted_counts = {
                key: int(value)
                for key, value in _json_number_map(
                    audit["group_counts"],
                    integer=True,
                ).items()
            }
            if actual_counts != persisted_counts:
                raise ProtocolError(
                    "Fixed-C risk audit group counts differ from bound fit rows."
                )
            expected_vector_hash = stable_hash(
                {
                    "fit_row_hash": fit_hash,
                    "ordered_weights": ordered_weights,
                }
            )
            if (
                audit["fit_row_hash"] != fit_hash
                or audit["training_frame_hash"] != fold.training_frame_hash
                or audit["weight_vector_hash"] != expected_vector_hash
            ):
                raise ProtocolError(
                    "Fixed-C risk ordered weight-vector/frame hash mismatch."
                )


def _validate_paired(
    row: Mapping[str, str],
    primary: Mapping[str, str],
    baseline: Mapping[str, str],
    protocol: Mapping[str, object],
) -> None:
    expected = {
        "schema_version": FIXED_C_RISK_PAIRED_SCHEMA_VERSION,
        "method": FIXED_C_RISK_METHOD,
        "protocol_hash": str(protocol["protocol_hash"]),
        "heldout_center": primary["heldout_center"],
        "contrast_id": PRIMARY_CONTRAST,
        "primary_risk_policy_id": "domain_class",
        "baseline_risk_policy_id": "pooled",
        "primary_risk_policy_hash": primary["risk_policy_hash"],
        "baseline_risk_policy_hash": baseline["risk_policy_hash"],
        "primary_weight_vector_hash": primary["weight_vector_hash"],
        "baseline_weight_vector_hash": baseline["weight_vector_hash"],
        "eval_row_hash": primary["eval_row_hash"],
        "training_frame_hash": primary["training_frame_hash"],
        "scaler_state_hash": primary["scaler_state_hash"],
        "selection_source": SELECTION_SOURCE,
        "claim_role": "risk_weighting_diagnostic",
        "row_role": "primary_paired_comparison",
        "claim_scope": "real_feature_transfer_only",
        "diagnostic_only": "true",
        "non_adoptive": "true",
        "adoption_eligible": "false",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "target_eval_labels_used_for_scoring_only": "true",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProtocolError(f"Fixed-C risk paired field {field} mismatch.")
    if any(
        primary[field] != baseline[field]
        for field in ("eval_row_hash", "training_frame_hash", "scaler_state_hash")
    ):
        raise ProtocolError("Fixed-C risk paired source frames differ.")
    metric_fields = (
        ("primary_bacc", primary, "heldout_bacc"),
        ("baseline_bacc", baseline, "heldout_bacc"),
        ("primary_macro_f1", primary, "heldout_macro_f1"),
        ("baseline_macro_f1", baseline, "heldout_macro_f1"),
    )
    for output_field, source, source_field in metric_fields:
        if not math.isclose(
            float(row[output_field]),
            float(source[source_field]),
            abs_tol=1e-12,
        ):
            raise ProtocolError("Fixed-C risk paired source metric mismatch.")
    if not math.isclose(
        float(row["delta_bacc"]),
        float(primary["heldout_bacc"]) - float(baseline["heldout_bacc"]),
        abs_tol=1e-12,
    ) or not math.isclose(
        float(row["delta_macro_f1"]),
        float(primary["heldout_macro_f1"]) - float(baseline["heldout_macro_f1"]),
        abs_tol=1e-12,
    ):
        raise ProtocolError("Fixed-C risk paired delta does not recompute.")


def _validate_summary(
    summary: Mapping[str, object],
    results: Sequence[Mapping[str, str]],
    paired: Sequence[Mapping[str, str]],
    protocol: Mapping[str, object],
) -> None:
    expected = build_diagnostic_summary(
        results,
        paired,
        protocol_hash=str(protocol["protocol_hash"]),
        bundle_hash=str(protocol["bundle_hash"]),
        heldout_count=len(protocol["heldout_centers"]),  # type: ignore[arg-type]
    )
    if dict(summary) != expected:
        raise ProtocolError("Fixed-C risk diagnostic summary does not recompute.")


def _validate_runtime(
    runtime: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    elapsed = float(runtime.get("elapsed_seconds", math.nan))
    if (
        runtime.get("schema_version") != "midogpp_fixed_c_risk_runtime_v1"
        or runtime.get("status") != "COMPLETE"
        or runtime.get("used_for_selection") is not False
        or runtime.get("claim_scope") != "real_feature_transfer_only"
        or runtime.get("diagnostic_only") is not True
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or int(runtime.get("n_fits", -1)) != int(protocol["expected_fit_count"])
    ):
        raise ProtocolError("Fixed-C risk runtime summary is invalid.")


def _formula_weight(
    policy: str,
    n_fit: int,
    n_domains: int,
    count: int,
) -> float:
    if count <= 0 or n_fit <= 0 or n_domains <= 0:
        raise ProtocolError("Fixed-C risk formula received a zero denominator.")
    if policy == "pooled":
        return 1.0
    if policy == "global_class":
        return float(n_fit) / float(2 * count)
    if policy == "domain":
        return float(n_fit) / float(n_domains * count)
    if policy == "domain_class":
        return float(n_fit) / float(2 * n_domains * count)
    raise ProtocolError(f"Unknown fixed-C risk policy: {policy!r}")


def _unique_by_pair(
    rows: Sequence[Mapping[str, str]],
    label: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    output: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row.get("heldout_center", ""), row.get("risk_policy_id", ""))
        if not all(key) or key in output:
            raise ProtocolError(f"Duplicate or empty key in {label}.")
        output[key] = row
    return output


def _json_number_map(raw: str, *, integer: bool) -> dict[str, float]:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError(
            "Fixed-C risk audit group mapping is malformed."
        ) from exc
    if not isinstance(payload, Mapping) or not payload:
        raise ProtocolError("Fixed-C risk audit group mapping is empty/malformed.")
    output: dict[str, float] = {}
    for key, value in payload.items():
        number = float(value)
        if not math.isfinite(number) or (integer and not number.is_integer()):
            raise ProtocolError("Fixed-C risk audit group value is invalid.")
        output[str(key)] = number
    return output


def _assert_columns(
    rows: Sequence[Mapping[str, object]],
    required: Sequence[str],
    label: str,
) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ProtocolError(f"{label} missing columns: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


__all__ = ["assert_fixed_c_risk_artifacts"]
