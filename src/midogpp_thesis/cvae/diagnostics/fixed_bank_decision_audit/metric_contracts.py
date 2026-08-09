"""Decision, metric, and terminal audit DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    ABSTENTION_DECISION_SCHEMA,
    ABSTENTION_SUMMARY_SCHEMA,
    AUDIT_RESULT_SCHEMA,
    EXACT_FAMILY_IDS,
    FAMILY_SUMMARY_SCHEMA,
    OUTER_METRIC_SCHEMA,
    PRIMARY_R_FAMILY_ID,
    QUERY_METRIC_SCHEMA,
)
from .experiment_contracts import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
)
from .model_contracts import ExactCrossfitResult, SmoothCrossfitResult
from .serialization import canonical_hash, finite, require_sha256


@dataclass(frozen=True)
class QueryMetricRow:
    family_id: str
    outer_target_id: str
    query_id: str
    candidate_count: int
    selected_source: str
    global_selected_source: str
    selected_exact_gain: float
    global_selected_exact_gain: float
    r_minus_g_exact_gain: float
    exact_top1: float
    tie_aware_top1: float
    spearman: float
    spearman_defined: bool
    pairwise_accuracy: float
    normalized_oracle_regret: float
    global_normalized_oracle_regret: float
    regret_minus_g: float
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "selected_exact_gain",
            "global_selected_exact_gain",
            "r_minus_g_exact_gain",
            "exact_top1",
            "tie_aware_top1",
            "spearman",
            "pairwise_accuracy",
            "normalized_oracle_regret",
            "global_normalized_oracle_regret",
            "regret_minus_g",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_METRIC_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "response_name": "exact_bacc_delta",
            "inference_unit": "descriptive_query_nested_within_outer_H",
            "candidate_rows_are_inference_units": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class OuterMetricRow:
    family_id: str
    outer_target_id: str
    query_count: int
    mean_selected_exact_gain: float
    mean_global_selected_exact_gain: float
    mean_r_minus_g_exact_gain: float
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    mean_pairwise_accuracy: float
    mean_normalized_oracle_regret: float
    mean_global_normalized_oracle_regret: float
    mean_regret_minus_g: float
    source_max_selection_share: float
    source_selection_entropy: float
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name.startswith("mean_") or name.startswith("source_"):
                object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": OUTER_METRIC_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "response_name": "exact_bacc_delta",
            "inference_unit": "outer_target_center",
            "query_rows_are_nested_descriptive_units": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class FamilySummaryRow:
    family_id: str
    scientific_role: str
    local_predictor_count: int
    source_effects_included: bool
    outer_count: int
    mean_selected_exact_gain: float
    selected_gain_ci95_lower: float
    selected_gain_ci95_upper: float
    mean_r_minus_g_exact_gain: float
    r_minus_g_ci95_lower: float
    r_minus_g_ci95_upper: float
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    spearman_ci95_lower: float
    spearman_ci95_upper: float
    mean_pairwise_accuracy: float
    pairwise_ci95_lower: float
    pairwise_ci95_upper: float
    mean_normalized_oracle_regret: float
    regret_ci95_lower: float
    regret_ci95_upper: float
    mean_regret_minus_g: float
    regret_minus_g_ci95_lower: float
    regret_minus_g_ci95_upper: float
    source_selection_counts: tuple[tuple[str, int], ...]
    source_max_selection_share: float
    source_selection_entropy: float
    publication_gate_eligible: bool
    exact_gate_passed: bool
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.publication_gate_eligible is not (
            self.family_id == PRIMARY_R_FAMILY_ID
        ):
            raise ProtocolError("Fixed-bank publication-gate eligibility drifted.")
        if self.exact_gate_passed and not self.publication_gate_eligible:
            raise ProtocolError("A non-primary family cannot pass the exact gate.")
        if sum(count for _, count in self.source_selection_counts) != 72:
            raise ProtocolError("Family source-selection coverage drifted.")
        for name in self.__dataclass_fields__:
            if (
                name.startswith("mean_")
                or "ci95" in name
                or name.startswith("source_max")
                or name == "source_selection_entropy"
            ):
                object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": FAMILY_SUMMARY_SCHEMA,
            **{
                name: (
                    [list(value) for value in self.source_selection_counts]
                    if name == "source_selection_counts"
                    else getattr(self, name)
                )
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "response_name": "exact_bacc_delta",
            "inference_unit": "outer_target_center",
            "predeclared_primary_r_family": PRIMARY_R_FAMILY_ID,
            "secondary_challengers_cannot_replace_primary_posthoc": True,
            "consumed_test_data": True,
            "policy_update_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class AbstentionDecisionRow:
    family_id: str
    outer_target_id: str
    query_id: str
    selected_source: str
    predicted_exact_gain: float
    prediction_standard_error: float
    lower_confidence_bound: float
    minimum_route_gain: float
    routed: bool
    observed_selected_exact_gain: float
    deployed_exact_gain: float
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = (
            self.observed_selected_exact_gain if self.routed else 0.0
        )
        if abs(self.deployed_exact_gain - expected) > 1.0e-12:
            raise ProtocolError("Abstention fallback-to-B utility drifted.")
        for name in (
            "predicted_exact_gain",
            "prediction_standard_error",
            "lower_confidence_bound",
            "minimum_route_gain",
            "observed_selected_exact_gain",
            "deployed_exact_gain",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": ABSTENTION_DECISION_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "fallback_action": "exact_B",
            "diagnostic_only": True,
            "target_actions_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class AbstentionSummaryRow:
    family_id: str
    query_count: int
    routed_query_count: int
    route_coverage: float
    mean_deployed_exact_gain: float
    deployed_gain_ci95_lower: float
    deployed_gain_ci95_upper: float
    mean_routed_exact_gain: float
    confidence_multiplier: float
    minimum_route_gain: float
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.query_count != 72 or not 0 <= self.routed_query_count <= 72:
            raise ProtocolError("Abstention summary query coverage drifted.")
        for name in (
            "route_coverage",
            "mean_deployed_exact_gain",
            "deployed_gain_ci95_lower",
            "deployed_gain_ci95_upper",
            "mean_routed_exact_gain",
            "confidence_multiplier",
            "minimum_route_gain",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": ABSTENTION_SUMMARY_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "inference_unit": "outer_target_center",
            "fallback_action": "exact_B",
            "diagnostic_only": True,
            "target_actions_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class FixedBankDecisionAuditResult:
    exact_crossfit: ExactCrossfitResult
    smooth_descriptive_crossfit: SmoothCrossfitResult | None
    query_metrics: tuple[QueryMetricRow, ...]
    outer_metrics: tuple[OuterMetricRow, ...]
    family_summaries: tuple[FamilySummaryRow, ...]
    abstention_decisions: tuple[AbstentionDecisionRow, ...]
    abstention_summaries: tuple[AbstentionSummaryRow, ...]
    primary_exact_gate_passed: bool
    exact_decision_hash: str = field(init=False)
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        family_count = len(EXACT_FAMILY_IDS)
        if (
            self.exact_crossfit.family_ids != EXACT_FAMILY_IDS
            or len(self.query_metrics) != family_count * 72
            or len(self.outer_metrics) != family_count * 9
            or len(self.family_summaries) != family_count
            or len(self.abstention_decisions) != family_count * 72
            or len(self.abstention_summaries) != family_count
        ):
            raise ProtocolError("Terminal fixed-bank audit coverage drifted.")
        primary = tuple(
            row
            for row in self.family_summaries
            if row.family_id == PRIMARY_R_FAMILY_ID
        )
        if len(primary) != 1 or self.primary_exact_gate_passed is not primary[0].exact_gate_passed:
            raise ProtocolError("Primary fixed-bank exact-gate result drifted.")
        if any(
            row.routed
            and (
                not self.primary_exact_gate_passed
                or row.family_id != PRIMARY_R_FAMILY_ID
            )
            for row in self.abstention_decisions
        ):
            raise ProtocolError("Abstention diagnostic violated fail-closed routing.")
        object.__setattr__(
            self, "exact_decision_hash", canonical_hash(self._exact_unhashed())
        )
        object.__setattr__(self, "result_hash", canonical_hash(self._unhashed()))

    def _exact_unhashed(self) -> dict[str, object]:
        return {
            "schema_version": f"{AUDIT_RESULT_SCHEMA}_exact_decision",
            "experiment_id": EXPERIMENT_ID,
            "exact_crossfit_hash": self.exact_crossfit.result_hash,
            "query_metric_row_hashes": [row.row_hash for row in self.query_metrics],
            "outer_metric_row_hashes": [row.row_hash for row in self.outer_metrics],
            "family_summary_row_hashes": [
                row.row_hash for row in self.family_summaries
            ],
            "abstention_decision_row_hashes": [
                row.row_hash for row in self.abstention_decisions
            ],
            "abstention_summary_row_hashes": [
                row.row_hash for row in self.abstention_summaries
            ],
            "predeclared_primary_r_family": PRIMARY_R_FAMILY_ID,
            "primary_exact_gate_passed": self.primary_exact_gate_passed,
            "smooth_response_used": False,
            "policy_update_authorized": False,
        }

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_RESULT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "exact_crossfit_hash": self.exact_crossfit.result_hash,
            "smooth_descriptive_crossfit_hash": (
                None
                if self.smooth_descriptive_crossfit is None
                else self.smooth_descriptive_crossfit.result_hash
            ),
            "query_metric_row_hashes": [row.row_hash for row in self.query_metrics],
            "outer_metric_row_hashes": [row.row_hash for row in self.outer_metrics],
            "family_summary_row_hashes": [
                row.row_hash for row in self.family_summaries
            ],
            "abstention_decision_row_hashes": [
                row.row_hash for row in self.abstention_decisions
            ],
            "abstention_summary_row_hashes": [
                row.row_hash for row in self.abstention_summaries
            ],
            "predeclared_primary_r_family": PRIMARY_R_FAMILY_ID,
            "primary_exact_gate_passed": self.primary_exact_gate_passed,
            "exact_decision_hash": self.exact_decision_hash,
            "publication_status": PUBLICATION_STATUS,
            "consumed_test_data": True,
            "terminal_diagnostic_only": True,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
            "smooth_influences_exact_model_or_decision": False,
            "routing_quality_claimed": False,
            "policy_update_authorized": False,
            "stage60_feed_authorized": False,
            "stage70_feed_authorized": False,
            "target_actions_authorized": False,
            "promotion_eligible": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "audit_result_hash": self.result_hash}

    @property
    def query_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.query_metrics)

    @property
    def outer_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.outer_metrics)

    @property
    def family_summary_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.family_summaries)

    @property
    def abstention_decision_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.abstention_decisions)

    @property
    def abstention_summary_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.abstention_summaries)


__all__ = (
    "AbstentionDecisionRow",
    "AbstentionSummaryRow",
    "FamilySummaryRow",
    "FixedBankDecisionAuditResult",
    "OuterMetricRow",
    "QueryMetricRow",
)
