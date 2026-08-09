"""Query, outer-center, family-summary, and terminal result DTOs."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DIAGNOSTIC_RESPONSE_NAMES,
    EXPERIMENT_ID,
    OUTER_INFERENCE_UNIT_COUNT,
    PRIMARY_RESPONSE_NAME,
    PUBLICATION_STATUS,
)
from .crossfit_contracts import CaseAwareCrossfitResult


@dataclass(frozen=True)
class QueryMetricRow:
    family_id: str
    response_name: str
    outer_target_id: str
    query_id: str
    candidate_count: int
    exact_top1: float
    tie_aware_top1: float
    spearman: float
    spearman_defined: bool
    normalized_oracle_regret: float
    pairwise_accuracy: float
    rmse: float
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_query_metrics_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "inference_unit": "descriptive_query_nested_within_outer_H",
            "candidate_rows_are_inference_units": False,
            "response_is_primary": self.response_name == PRIMARY_RESPONSE_NAME,
            "smooth_response_is_diagnostic_only": (
                self.response_name != PRIMARY_RESPONSE_NAME
            ),
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class OuterMetricRow:
    family_id: str
    response_name: str
    outer_target_id: str
    query_count: int
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    mean_normalized_oracle_regret: float
    mean_pairwise_accuracy: float
    rmse: float
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_outer_metrics_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "inference_unit": "outer_target_center",
            "query_rows_are_nested_descriptive_units": True,
            "candidate_and_seed_rows_are_not_inference_units": True,
            "response_is_primary": self.response_name == PRIMARY_RESPONSE_NAME,
            "smooth_response_is_diagnostic_only": (
                self.response_name != PRIMARY_RESPONSE_NAME
            ),
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class FamilySummaryRow:
    family_id: str
    response_name: str
    family_role: str
    predictor_count: int
    outer_count: int
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    spearman_ci95_lower: float
    spearman_ci95_upper: float
    mean_normalized_oracle_regret: float
    regret_ci95_lower: float
    regret_ci95_upper: float
    mean_pairwise_accuracy: float
    pairwise_ci95_lower: float
    pairwise_ci95_upper: float
    mean_rmse: float
    beats_all_controls: bool
    screening_eligible: bool
    screening_passed: bool
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_family_summary_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "inference_unit": "outer_target_center",
            "outer_center_count": OUTER_INFERENCE_UNIT_COUNT,
            "response_is_primary": self.response_name == PRIMARY_RESPONSE_NAME,
            "smooth_response_is_diagnostic_only": (
                self.response_name != PRIMARY_RESPONSE_NAME
            ),
            "screening_gate_may_authorize_policy": False,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class CaseAwareProxyInformationAuditResult:
    crossfit: CaseAwareCrossfitResult
    query_metrics: tuple[QueryMetricRow, ...]
    outer_metrics: tuple[OuterMetricRow, ...]
    family_summaries: tuple[FamilySummaryRow, ...]
    primary_proxy_information_gate_passed: bool
    informative_family_ids: tuple[str, ...]
    result_hash: str

    @property
    def crossfit_table_rows(self) -> tuple[dict[str, object], ...]:
        return self.crossfit.table_rows

    @property
    def fold_audit_table_rows(self) -> tuple[dict[str, object], ...]:
        return self.crossfit.fold_audit_table_rows

    @property
    def query_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.query_metrics)

    @property
    def outer_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.outer_metrics)

    @property
    def family_summary_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.family_summaries)

    def fold_lock_payload(self) -> dict[str, object]:
        return self.crossfit.fold_lock_payload()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_proxy_audit_result_v1",
            "experiment_id": EXPERIMENT_ID,
            "crossfit_result_hash": self.crossfit.result_hash,
            "family_summary_row_hashes": [
                row.row_hash for row in self.family_summaries
            ],
            "primary_proxy_information_gate_passed": (
                self.primary_proxy_information_gate_passed
            ),
            "informative_family_ids": list(self.informative_family_ids),
            "primary_response": PRIMARY_RESPONSE_NAME,
            "diagnostic_responses": list(DIAGNOSTIC_RESPONSE_NAMES),
            "outer_target_centers_are_inference_units": True,
            "outer_inference_unit_count": OUTER_INFERENCE_UNIT_COUNT,
            "consumed_test_data": True,
            "publication_status": PUBLICATION_STATUS,
            "terminal_diagnostic_only": True,
            "routing_quality_claimed": False,
            "policy_update_authorized": False,
            "stage60_feed_authorized": False,
            "stage70_feed_authorized": False,
            "target_actions_authorized": False,
            "promotion_eligible": False,
            "audit_result_hash": self.result_hash,
        }


__all__ = (
    "CaseAwareProxyInformationAuditResult",
    "FamilySummaryRow",
    "OuterMetricRow",
    "QueryMetricRow",
)
