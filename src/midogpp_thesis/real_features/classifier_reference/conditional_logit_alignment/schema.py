"""Declarative artifact contracts for the conditional-logit alignment diagnostic.

This module intentionally contains no filesystem or numerical code.  It is the
single source of truth for table names, column order, semantic roles, and the
claim boundary used by both the writer and the independent validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..artifacts import stable_hash


CLA_SCHEMA_VERSION = "midogpp_conditional_logit_alignment_v1"
CLA_CODE_VERSION = "conditional_logit_alignment_v1"
CLA_EXPERIMENT_ID = "midogpp.real_feature.conditional_logit_alignment.v1"
CLA_EXPERIMENT_NAME = "conditional_logit_alignment_v1"
CLA_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_real_feature_conditional_logit_alignment_v1"
)
CLA_STAGE = "10_real_feature_reference"
CLA_INPUT_ARTIFACT_IDS = (
    "midogpp_dataset_contract_annotation_patch_v1",
    "midogpp_virchow2_xyxy_feature_cache_seed42",
)
CLA_CANONICAL_OUTPUT_PATH = (
    "artifacts/midogpp/10_real_feature_reference/"
    "conditional_logit_alignment_v1/seed42"
)
CLA_EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
CLA_EXPECTED_FEATURE_CACHE_SHA256 = (
    "f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2"
)
CLA_METHOD = "conditional_logit_alignment"
CLA_CLAIM_SCOPE = "real_feature_transfer_only"
CLA_CLAIM_ROLE = "conditional_logit_alignment_diagnostic"
CLA_PRIOR_METHOD = "not_applicable_real_features"
CLA_SELECTION_SOURCE = (
    "source_inner_equal_center_mean_bacc_smallest_gamma_tie_break"
)

SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION = "midogpp_cla_inner_fold_score_v1"
SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION = "midogpp_cla_gamma_summary_v1"
OUTER_RESULT_SCHEMA_VERSION = "midogpp_cla_outer_result_v1"
OUTER_PREDICTION_SCHEMA_VERSION = "midogpp_cla_outer_prediction_v1"
CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION = "midogpp_cla_frame_audit_v1"
SOLVER_AUDIT_SCHEMA_VERSION = "midogpp_cla_solver_audit_v1"
OUTER_COMPARISON_SCHEMA_VERSION = "midogpp_cla_outer_comparison_v1"

FROZEN_PROTOCOL_SCHEMA_VERSION = "midogpp_cla_frozen_protocol_v1"
PROTOCOL_MANIFEST_SCHEMA_VERSION = "midogpp_cla_protocol_manifest_v1"
LEAKAGE_REPORT_SCHEMA_VERSION = "midogpp_cla_leakage_report_v1"
DECISION_SUMMARY_SCHEMA_VERSION = "midogpp_cla_decision_summary_v1"
RUNTIME_SUMMARY_SCHEMA_VERSION = "midogpp_cla_runtime_summary_v1"
CONTENT_INDEX_SCHEMA_VERSION = "midogpp_cla_content_index_v1"
WORKSPACE_BINDING_SCHEMA_VERSION = "midogpp_cla_workspace_binding_v1"

OUTER_EVALUATION_ROLES = ("selected", "gamma0")
PRIMARY_CONTRAST = "selected_minus_gamma0"
FOLD_SCOPES = ("source_inner", "outer")

PRODUCTION_OUTER_FOLD_COUNT = 9
PRODUCTION_INNER_FOLDS_PER_OUTER = 8
PRODUCTION_GAMMA_COUNT = 7
PRODUCTION_INNER_SCORE_COUNT = 504
PRODUCTION_GAMMA_SUMMARY_COUNT = 63
PRODUCTION_OUTER_RESULT_COUNT = 18
PRODUCTION_FRAME_AUDIT_COUNT = 81
PRODUCTION_OUTER_COMPARISON_COUNT = 9


# These fields occur on every scientific/audit row.  Repetition is deliberate:
# each extracted table remains fail-closed when separated from the bundle.
CLAIM_COLUMNS = (
    "prior_method",
    "selection_source",
    "claim_scope",
    "claim_role",
    "row_role",
    "diagnostic_only",
    "non_adoptive",
    "adoption_eligible",
    "may_feed_recipe_selection",
    "may_feed_deployable_selection",
    "source_inner_labels_used",
    "support_labels_used",
    "oracle_eligible",
    "target_eval_labels_used_for_scoring_only",
    "target_eval_labels_used_for_fit",
    "target_eval_labels_used_for_selection",
    "uses_generated_embeddings",
    "uses_cvae_checkpoint",
    "uses_encoder_posterior",
    "uses_decoder_likelihood",
    "uses_prior",
    "uses_nelbo",
    "uses_latent_representation",
    "models_embedding_distribution",
    "uses_expert_bank",
    "uses_router",
    "performs_expert_selection",
    "performs_expert_weighting",
    "performs_aggregation",
    "leakage_status",
)

FAIL_CLOSED_CLAIM_VALUES: Mapping[str, str] = {
    "prior_method": CLA_PRIOR_METHOD,
    "selection_source": CLA_SELECTION_SOURCE,
    "claim_scope": CLA_CLAIM_SCOPE,
    "claim_role": CLA_CLAIM_ROLE,
    "diagnostic_only": "true",
    "non_adoptive": "true",
    "adoption_eligible": "false",
    "may_feed_recipe_selection": "false",
    "may_feed_deployable_selection": "false",
    "source_inner_labels_used": "true",
    "support_labels_used": "false",
    "oracle_eligible": "false",
    "target_eval_labels_used_for_scoring_only": "true",
    "target_eval_labels_used_for_fit": "false",
    "target_eval_labels_used_for_selection": "false",
    "uses_generated_embeddings": "false",
    "uses_cvae_checkpoint": "false",
    "uses_encoder_posterior": "false",
    "uses_decoder_likelihood": "false",
    "uses_prior": "false",
    "uses_nelbo": "false",
    "uses_latent_representation": "false",
    "models_embedding_distribution": "false",
    "uses_expert_bank": "false",
    "uses_router": "false",
    "performs_expert_selection": "false",
    "performs_expert_weighting": "false",
    "performs_aggregation": "false",
    "leakage_status": "PASS",
}


SOURCE_INNER_FOLD_SCORE_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "inner_center",
    "gamma",
    "fit_identity",
    "conditional_frame_identity",
    "fit_centers",
    "n_fit",
    "n_eval",
    "fit_row_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "penalty_operator_hash",
    "classifier_config_hash",
    "inner_bacc",
    "inner_macro_f1",
    "converged",
    "n_iter",
    "status",
    *CLAIM_COLUMNS,
)

SOURCE_INNER_GAMMA_SUMMARY_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "gamma",
    "n_inner_centers",
    "mean_inner_bacc",
    "mean_inner_macro_f1",
    "minimum_inner_bacc",
    "selected",
    "selection_rank",
    "tie_atol",
    "tie_rtol",
    "status",
    *CLAIM_COLUMNS,
)

OUTER_RESULT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "evaluation_role",
    "gamma",
    "selected_gamma",
    "fit_identity",
    "conditional_frame_identity",
    "shared_fit",
    "fit_centers",
    "n_fit",
    "n_eval",
    "fit_row_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "penalty_operator_hash",
    "classifier_config_hash",
    "heldout_bacc",
    "heldout_macro_f1",
    "converged",
    "n_iter",
    "status",
    "manifest_hash",
    "feature_cache_hash",
    *CLAIM_COLUMNS,
)

OUTER_PREDICTION_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "evaluation_role",
    "gamma",
    "selected_gamma",
    "fit_identity",
    "conditional_frame_identity",
    "shared_fit",
    "sample_id",
    "case_id",
    "center",
    "y_true",
    "y_pred",
    "prob_pos",
    "fit_row_hash",
    "eval_row_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "penalty_operator_hash",
    "classifier_config_hash",
    *CLAIM_COLUMNS,
)

CONDITIONAL_FRAME_AUDIT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "fold_scope",
    "heldout_center",
    "inner_center",
    "conditional_frame_identity",
    "fit_centers",
    "n_fit",
    "n_domains",
    "fit_row_hash",
    "eval_row_hash",
    "fit_case_hash",
    "eval_case_hash",
    "fit_image_path_hash",
    "eval_image_path_hash",
    "fit_row_index_hash",
    "eval_row_index_hash",
    "training_frame_hash",
    "scaler_state_hash",
    "penalty_operator_hash",
    "operator_rank",
    "maximum_operator_rank",
    "operator_trace",
    "required_cell_count",
    "observed_cell_count",
    "missing_cell_count",
    "factor_representation",
    "normalization",
    "dense_matrix_materialized",
    "heldout_center_excluded",
    "inner_center_excluded",
    "fit_eval_sample_overlap_count",
    "fit_eval_case_overlap_count",
    "fit_eval_image_path_overlap_count",
    "fit_eval_row_index_overlap_count",
    "target_rows_used_for_scaler",
    "target_rows_used_for_operator",
    "target_rows_used_for_fit",
    "status",
    *CLAIM_COLUMNS,
)

SOLVER_AUDIT_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "fold_scope",
    "heldout_center",
    "inner_center",
    "gamma",
    "fit_identity",
    "conditional_frame_identity",
    "fit_row_hash",
    "scaler_state_hash",
    "penalty_operator_hash",
    "classifier_config_hash",
    "backend",
    "warm_start",
    "objective_value",
    "gradient_inf_norm",
    "n_iter",
    "converged",
    "optimizer_status",
    "l2_normalization",
    "intercept_penalized",
    "gamma_zero_shared_sklearn_path",
    "status",
    *CLAIM_COLUMNS,
)

OUTER_COMPARISON_COLUMNS = (
    "schema_version",
    "method",
    "protocol_hash",
    "heldout_center",
    "contrast_id",
    "selected_gamma",
    "selected_fit_identity",
    "gamma0_fit_identity",
    "shared_fit",
    "eval_row_hash",
    "selected_bacc",
    "gamma0_bacc",
    "delta_bacc",
    "selected_macro_f1",
    "gamma0_macro_f1",
    "delta_macro_f1",
    "status",
    *CLAIM_COLUMNS,
)


TABLE_PATHS: Mapping[str, str] = {
    "source_inner_fold_scores": "tables/source_inner_fold_scores.csv",
    "source_inner_gamma_summary": "tables/source_inner_gamma_summary.csv",
    "outer_results": "tables/outer_results.csv",
    "outer_predictions": "tables/outer_predictions.csv",
    "conditional_frame_audit": "tables/conditional_frame_audit.csv",
    "solver_audit": "tables/solver_audit.csv",
    "outer_comparison": "tables/outer_comparison.csv",
}

TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "source_inner_fold_scores": SOURCE_INNER_FOLD_SCORE_COLUMNS,
    "source_inner_gamma_summary": SOURCE_INNER_GAMMA_SUMMARY_COLUMNS,
    "outer_results": OUTER_RESULT_COLUMNS,
    "outer_predictions": OUTER_PREDICTION_COLUMNS,
    "conditional_frame_audit": CONDITIONAL_FRAME_AUDIT_COLUMNS,
    "solver_audit": SOLVER_AUDIT_COLUMNS,
    "outer_comparison": OUTER_COMPARISON_COLUMNS,
}

CLA_WORKSPACE_REQUIRED_OUTPUTS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
)

CLA_RUNNER_REQUIRED_OUTPUTS = (
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/content_index.json",
    "reports/leakage_provenance_report.json",
    "reports/decision_summary.json",
    "reports/decision_report.md",
    "reports/runtime_summary.json",
    *TABLE_PATHS.values(),
)

CLA_COMPLETE_REQUIRED_OUTPUTS = (
    *CLA_WORKSPACE_REQUIRED_OUTPUTS,
    *CLA_RUNNER_REQUIRED_OUTPUTS,
)

# Public required-output contract means a complete catalog bundle.
CLA_REQUIRED_OUTPUTS = CLA_COMPLETE_REQUIRED_OUTPUTS


@dataclass(frozen=True)
class AlignmentArtifactTables:
    """The seven normalized logical table payloads."""

    source_inner_fold_scores: tuple[Mapping[str, object], ...]
    source_inner_gamma_summary: tuple[Mapping[str, object], ...]
    outer_results: tuple[Mapping[str, object], ...]
    outer_predictions: tuple[Mapping[str, object], ...]
    conditional_frame_audit: tuple[Mapping[str, object], ...]
    solver_audit: tuple[Mapping[str, object], ...]
    outer_comparison: tuple[Mapping[str, object], ...]

    def as_mapping(self) -> Mapping[str, tuple[Mapping[str, object], ...]]:
        return {
            name: getattr(self, name)
            for name in TABLE_PATHS
        }

    @classmethod
    def from_mapping(
        cls,
        tables: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> "AlignmentArtifactTables":
        missing = sorted(set(TABLE_PATHS).difference(tables))
        extra = sorted(set(tables).difference(TABLE_PATHS))
        if missing or extra:
            raise ValueError(
                f"CLA table mapping mismatch: missing={missing}, extra={extra}"
            )
        return cls(
            **{name: tuple(tables[name]) for name in TABLE_PATHS}  # type: ignore[arg-type]
        )


def claim_fields(*, row_role: str) -> dict[str, object]:
    """Return the immutable per-row claim boundary with one semantic role."""

    return dict(FAIL_CLOSED_CLAIM_VALUES) | {"row_role": str(row_role)}


def canonical_table(
    rows: Sequence[Mapping[str, object]],
    *,
    ignored: Sequence[str] = ("protocol_hash",),
) -> list[dict[str, str]]:
    """Normalize a logical table independent of row order and CSV typing."""

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


def table_hashes(tables: AlignmentArtifactTables) -> dict[str, str]:
    """Hash each table without its circular protocol binding."""

    return {
        name: stable_hash(canonical_table(rows))
        for name, rows in tables.as_mapping().items()
    }


def table_bundle_hash(tables: AlignmentArtifactTables) -> str:
    """Hash the ordered collection of seven canonical table identities."""

    return stable_hash(
        {
            "schema_version": CLA_SCHEMA_VERSION,
            "table_hashes": table_hashes(tables),
        }
    )


__all__ = [
    "AlignmentArtifactTables",
    "CLA_CLAIM_ROLE",
    "CLA_CLAIM_SCOPE",
    "CLA_CANONICAL_OUTPUT_PATH",
    "CLA_CODE_VERSION",
    "CLA_COMPLETE_REQUIRED_OUTPUTS",
    "CLA_EXPECTED_FEATURE_CACHE_SHA256",
    "CLA_EXPECTED_MANIFEST_SHA256",
    "CLA_EXPERIMENT_ID",
    "CLA_EXPERIMENT_NAME",
    "CLA_INPUT_ARTIFACT_IDS",
    "CLA_METHOD",
    "CLA_OUTPUT_ARTIFACT_ID",
    "CLA_PRIOR_METHOD",
    "CLA_REQUIRED_OUTPUTS",
    "CLA_RUNNER_REQUIRED_OUTPUTS",
    "CLA_SCHEMA_VERSION",
    "CLA_SELECTION_SOURCE",
    "CLA_STAGE",
    "CLA_WORKSPACE_REQUIRED_OUTPUTS",
    "CLAIM_COLUMNS",
    "CONDITIONAL_FRAME_AUDIT_COLUMNS",
    "CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION",
    "CONTENT_INDEX_SCHEMA_VERSION",
    "DECISION_SUMMARY_SCHEMA_VERSION",
    "FAIL_CLOSED_CLAIM_VALUES",
    "FOLD_SCOPES",
    "FROZEN_PROTOCOL_SCHEMA_VERSION",
    "LEAKAGE_REPORT_SCHEMA_VERSION",
    "OUTER_COMPARISON_COLUMNS",
    "OUTER_COMPARISON_SCHEMA_VERSION",
    "OUTER_EVALUATION_ROLES",
    "OUTER_PREDICTION_COLUMNS",
    "OUTER_PREDICTION_SCHEMA_VERSION",
    "OUTER_RESULT_COLUMNS",
    "OUTER_RESULT_SCHEMA_VERSION",
    "PRIMARY_CONTRAST",
    "PRODUCTION_FRAME_AUDIT_COUNT",
    "PRODUCTION_GAMMA_COUNT",
    "PRODUCTION_GAMMA_SUMMARY_COUNT",
    "PRODUCTION_INNER_FOLDS_PER_OUTER",
    "PRODUCTION_INNER_SCORE_COUNT",
    "PRODUCTION_OUTER_COMPARISON_COUNT",
    "PRODUCTION_OUTER_FOLD_COUNT",
    "PRODUCTION_OUTER_RESULT_COUNT",
    "PROTOCOL_MANIFEST_SCHEMA_VERSION",
    "RUNTIME_SUMMARY_SCHEMA_VERSION",
    "SOLVER_AUDIT_COLUMNS",
    "SOLVER_AUDIT_SCHEMA_VERSION",
    "SOURCE_INNER_FOLD_SCORE_COLUMNS",
    "SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION",
    "SOURCE_INNER_GAMMA_SUMMARY_COLUMNS",
    "SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION",
    "TABLE_COLUMNS",
    "TABLE_PATHS",
    "WORKSPACE_BINDING_SCHEMA_VERSION",
    "canonical_table",
    "claim_fields",
    "table_bundle_hash",
    "table_hashes",
]
