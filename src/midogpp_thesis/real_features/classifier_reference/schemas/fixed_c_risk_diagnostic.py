"""Declarative contracts and public facade for the fixed-C risk diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from ..artifacts import FrozenProtocolSnapshot, stable_hash
from ..classifiers import ClassifierSpec
from ..protocol import ProtocolError

if TYPE_CHECKING:
    from ..real_feature_frame import RealFeatureFrame


FIXED_C_RISK_SCHEMA_VERSION = "midogpp_fixed_c_risk_diagnostic_v1"
FIXED_C_RISK_RESULT_SCHEMA_VERSION = "midogpp_fixed_c_risk_result_v1"
FIXED_C_RISK_PREDICTION_SCHEMA_VERSION = "midogpp_fixed_c_risk_prediction_v1"
FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION = "midogpp_fixed_c_risk_weight_audit_v1"
FIXED_C_RISK_PAIRED_SCHEMA_VERSION = "midogpp_fixed_c_risk_paired_comparison_v1"
FIXED_C_RISK_METHOD = "fixed_c_risk_diagnostic"
FIXED_C_RISK_EXPERIMENT_ID = "midogpp.real_feature.fixed_c_risk_diagnostic.v1"
FIXED_C_RISK_EXPERIMENT_NAME = "fixed_c_risk_diagnostic_v1"
FIXED_C_RISK_CODE_VERSION = "fixed_c_risk_diagnostic_v1"
FIXED_C_RISK_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_real_feature_fixed_c_risk_diagnostic_v1"
)
FIXED_CLASSIFIER_CONFIG_HASH = "86378e6ceb12136e"
RISK_POLICY_IDS = ("pooled", "global_class", "domain", "domain_class")
RISK_POLICY_FORMULAS = {
    "pooled": "1",
    "global_class": "N/(2*n_y)",
    "domain": "N/(D*n_d)",
    "domain_class": "N/(2*D*n_dy)",
}
WEIGHT_NORMALIZATION = "sum_to_n_fit"
ZERO_CELL_POLICY = "fail_closed"
PRIMARY_CONTRAST = "domain_class_minus_pooled"
SELECTION_SOURCE = "predeclared_fixed_no_selection"
PRIOR_METHOD = "not_applicable_real_features"

FIXED_C_RISK_RESULT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "risk_policy_id",
    "risk_policy_formula",
    "risk_policy_hash",
    "weight_vector_hash",
    "train_centers",
    "n_train",
    "n_eval",
    "fit_row_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "fixed_classifier_config_hash",
    "fixed_classifier_spec",
    "heldout_bacc",
    "heldout_macro_f1",
    "converged",
    "n_iter",
    "status",
    "manifest_hash",
    "feature_cache_hash",
    "prior_method",
    "threshold_policy",
    "selection_source",
    "sample_weight_passed_to_fit",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "target_rows_used_for_fit",
    "generated_embeddings_used",
    "cvae_checkpoint_used",
    "is_router",
    "claim_scope",
    "claim_role",
    "row_role",
    "diagnostic_only",
    "non_adoptive",
    "adoption_eligible",
    "support_labels_used",
    "oracle_eligible",
    "leakage_status",
)

FIXED_C_RISK_PREDICTION_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "risk_policy_id",
    "risk_policy_hash",
    "weight_vector_hash",
    "sample_id",
    "case_id",
    "center",
    "y_true",
    "y_pred",
    "prob_pos",
    "fixed_classifier_config_hash",
    "fit_row_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "prior_method",
    "selection_source",
    "claim_role",
    "row_role",
    "diagnostic_only",
    "non_adoptive",
    "adoption_eligible",
    "support_labels_used",
    "oracle_eligible",
    "target_eval_labels_used_for_scoring_only",
    "leakage_status",
)

FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "risk_policy_id",
    "risk_policy_formula",
    "risk_policy_hash",
    "weight_vector_hash",
    "train_centers",
    "n_fit",
    "n_domains",
    "fit_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "fixed_classifier_config_hash",
    "group_counts",
    "group_weights",
    "group_masses",
    "weight_min",
    "weight_max",
    "weight_sum",
    "expected_weight_sum",
    "normalization",
    "zero_cell_policy",
    "all_weights_finite",
    "all_weights_positive",
    "target_rows_used",
    "scaler_fit_used_sample_weight",
    "sample_weight_passed_to_fit",
    "prior_method",
    "selection_source",
    "claim_scope",
    "claim_role",
    "row_role",
    "diagnostic_only",
    "non_adoptive",
    "adoption_eligible",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "support_labels_used",
    "oracle_eligible",
    "status",
)

FIXED_C_RISK_PAIRED_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "contrast_id",
    "primary_risk_policy_id",
    "baseline_risk_policy_id",
    "primary_risk_policy_hash",
    "baseline_risk_policy_hash",
    "primary_weight_vector_hash",
    "baseline_weight_vector_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "primary_bacc",
    "baseline_bacc",
    "delta_bacc",
    "primary_macro_f1",
    "baseline_macro_f1",
    "delta_macro_f1",
    "selection_source",
    "claim_role",
    "row_role",
    "claim_scope",
    "diagnostic_only",
    "non_adoptive",
    "adoption_eligible",
    "support_labels_used",
    "oracle_eligible",
    "target_eval_labels_used_for_scoring_only",
)

FIXED_C_RISK_REQUIRED_OUTPUTS = (
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/runtime_summary.json",
    "tables/fixed_c_risk_results.csv",
    "tables/fixed_c_risk_predictions.csv",
    "tables/fixed_c_risk_weight_audit.csv",
    "tables/fixed_c_risk_paired_comparison.csv",
)


def canonical_fixed_classifier_spec() -> ClassifierSpec:
    """Return the one frozen classifier payload used by every diagnostic arm."""

    spec = ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
    )
    if spec.config_hash != FIXED_CLASSIFIER_CONFIG_HASH:
        raise ProtocolError(
            "Canonical fixed-C classifier hash drift: "
            f"expected={FIXED_CLASSIFIER_CONFIG_HASH} actual={spec.config_hash}"
        )
    return spec


def risk_policy_hash(policy_id: str) -> str:
    """Hash one predeclared risk-weighting policy."""

    policy = str(policy_id)
    if policy not in RISK_POLICY_IDS:
        raise ProtocolError(f"Unknown fixed-C risk policy: {policy!r}")
    return stable_hash(
        {
            "risk_policy_id": policy,
            "formula": RISK_POLICY_FORMULAS[policy],
            "normalization": WEIGHT_NORMALIZATION,
            "zero_cell_policy": ZERO_CELL_POLICY,
        }
    )


def fixed_c_risk_bundle_hash(
    result_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    weight_audit_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> str:
    """Hash the four canonical tables without their circular protocol binding."""

    return stable_hash(
        {
            "results": _canonical_table(result_rows, ignored=("protocol_hash",)),
            "predictions": _canonical_table(
                prediction_rows, ignored=("protocol_hash",)
            ),
            "weight_audits": _canonical_table(
                weight_audit_rows, ignored=("protocol_hash",)
            ),
            "paired": _canonical_table(paired_rows, ignored=("protocol_hash",)),
        }
    )


def expected_frozen_snapshot(protocol: Mapping[str, object]) -> FrozenProtocolSnapshot:
    """Build the frozen declarative snapshot for a protocol payload."""

    return FrozenProtocolSnapshot(
        candidate_pool_hash=stable_hash(
            {
                "eligible_centers": list(protocol["eligible_centers"]),
                "excluded_centers": list(protocol["excluded_centers"]),
                "heldout_centers": list(protocol["heldout_centers"]),
                "coverage_mode": protocol["coverage_mode"],
            }
        ),
        generation_config_hash=stable_hash(
            {
                "prior_method": PRIOR_METHOD,
                "generated_embeddings_used": False,
                "cvae_checkpoint_used": False,
            }
        ),
        classifier_config_hash=FIXED_CLASSIFIER_CONFIG_HASH,
        metric_config_hash=stable_hash(
            {
                "metrics": ["balanced_accuracy", "macro_f1"],
                "threshold_policy": "predict",
                "primary_contrast": PRIMARY_CONTRAST,
                "paired_by": "heldout_center",
            }
        ),
        feature_config_hash=stable_hash(
            {
                "manifest_hash": protocol["manifest_hash"],
                "feature_cache_hash": protocol["feature_cache_hash"],
                "expected_feature_dim": protocol["expected_feature_dim"],
            }
        ),
        routing_config_hash=stable_hash(
            {
                "is_router": False,
                "risk_policies": [
                    {
                        "risk_policy_id": policy,
                        "formula": RISK_POLICY_FORMULAS[policy],
                        "risk_policy_hash": risk_policy_hash(policy),
                    }
                    for policy in RISK_POLICY_IDS
                ],
                "normalization": WEIGHT_NORMALIZATION,
                "zero_cell_policy": ZERO_CELL_POLICY,
                "selection_source": SELECTION_SOURCE,
                "selection_performed": False,
                "scaler_fit_scope": "outer_source_train_only",
                "scaler_weighting": "unweighted",
                "sample_weight_scope": "logistic_regression_fit_only",
            }
        ),
    )


def assert_fixed_c_risk_artifacts(
    root: Path,
    already_loaded_frame: RealFeatureFrame | None = None,
) -> None:
    """Validate a bundle through the stable schema facade."""

    from ..fixed_c_risk_artifact_validation import (
        assert_fixed_c_risk_artifacts as _assert_fixed_c_risk_artifacts,
    )

    _assert_fixed_c_risk_artifacts(
        root,
        already_loaded_frame=already_loaded_frame,
    )


def render_diagnostic_report(summary: Mapping[str, object]) -> str:
    """Render a diagnostic report through the stable schema facade."""

    from ..fixed_c_risk_reporting import render_diagnostic_report as _render

    return _render(summary)


def _canonical_table(
    rows: Sequence[Mapping[str, object]],
    *,
    ignored: Sequence[str] = (),
) -> list[dict[str, str]]:
    ignored_set = set(ignored)
    fields = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key) not in ignored_set
        }
    )
    normalized = [
        {
            field: "" if row.get(field) is None else str(row.get(field))
            for field in fields
        }
        for row in rows
    ]
    return sorted(normalized, key=stable_hash)


__all__ = [
    "FIXED_C_RISK_CODE_VERSION",
    "FIXED_C_RISK_EXPERIMENT_ID",
    "FIXED_C_RISK_EXPERIMENT_NAME",
    "FIXED_C_RISK_METHOD",
    "FIXED_C_RISK_OUTPUT_ARTIFACT_ID",
    "FIXED_C_RISK_PAIRED_COLUMNS",
    "FIXED_C_RISK_PAIRED_SCHEMA_VERSION",
    "FIXED_C_RISK_PREDICTION_COLUMNS",
    "FIXED_C_RISK_PREDICTION_SCHEMA_VERSION",
    "FIXED_C_RISK_REQUIRED_OUTPUTS",
    "FIXED_C_RISK_RESULT_COLUMNS",
    "FIXED_C_RISK_RESULT_SCHEMA_VERSION",
    "FIXED_C_RISK_SCHEMA_VERSION",
    "FIXED_C_RISK_WEIGHT_AUDIT_COLUMNS",
    "FIXED_C_RISK_WEIGHT_AUDIT_SCHEMA_VERSION",
    "FIXED_CLASSIFIER_CONFIG_HASH",
    "PRIMARY_CONTRAST",
    "PRIOR_METHOD",
    "RISK_POLICY_FORMULAS",
    "RISK_POLICY_IDS",
    "SELECTION_SOURCE",
    "WEIGHT_NORMALIZATION",
    "ZERO_CELL_POLICY",
    "assert_fixed_c_risk_artifacts",
    "canonical_fixed_classifier_spec",
    "expected_frozen_snapshot",
    "fixed_c_risk_bundle_hash",
    "render_diagnostic_report",
    "risk_policy_hash",
]
