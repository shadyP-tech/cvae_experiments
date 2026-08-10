"""Immutable terminal result contracts and persistence table projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS
from .contracts import CaseConfusionCounts, PooledBacc
from .hashing import canonical_hash, finite, require_sha256
from .metrics import PairwiseComplementarity, RankStabilityResult
from .terminal_inference import TerminalContrast


METHOD_ORDER = ("U", "G", "R", "P", "S_y", "O_static", "O_case")


def pooled_payload(value: PooledBacc) -> dict[str, object]:
    return dict(value.__dict__)


def count_payload(value: CaseConfusionCounts) -> dict[str, object]:
    return dict(value.__dict__)


def rank_payload(value: RankStabilityResult) -> dict[str, object]:
    return {
        "action_ids": list(value.action_ids),
        "support_ranks": list(value.support_ranks),
        "evaluation_ranks": list(value.evaluation_ranks),
        "spearman": value.spearman,
        "identifiable": value.identifiable,
    }


@dataclass(frozen=True)
class CenterMetric:
    target_center: str
    geometry_id: str | None
    method_id: str
    pooled_bacc: PooledBacc

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Center metric has an unknown target center.")
        if self.method_id == "B":
            if self.geometry_id is not None:
                raise ProtocolError("Global B center metric cannot carry a geometry.")
        elif self.geometry_id not in GEOMETRY_IDS or self.method_id not in METHOD_ORDER:
            raise ProtocolError("Geometry center metric identity is invalid.")

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center, "geometry_id": self.geometry_id,
            "method_id": self.method_id, "pooled_bacc": pooled_payload(self.pooled_bacc),
        }


@dataclass(frozen=True)
class TerminalMethodSummary:
    geometry_id: str | None
    method_id: str
    case_confusions: tuple[CaseConfusionCounts, ...]
    center_metrics: tuple[CenterMetric, ...]
    all_center_pooled_bacc: PooledBacc
    equal_center_exact_bacc: float
    single_class_case_count: int
    method_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.method_id == "B":
            if self.geometry_id is not None:
                raise ProtocolError("Global B summary cannot carry a geometry.")
        elif self.geometry_id not in GEOMETRY_IDS or self.method_id not in METHOD_ORDER:
            raise ProtocolError("Terminal method summary identity is invalid.")
        rows, centers = tuple(self.case_confusions), tuple(self.center_metrics)
        if not rows or tuple(x.target_center for x in centers) != MIDOGPP_CENTERS:
            raise ProtocolError("Terminal method summary must cover all centers.")
        if any(x.geometry_id != self.geometry_id or x.method_id != self.method_id for x in centers):
            raise ProtocolError("Center metrics drifted from their terminal method.")
        expected = math.fsum(x.pooled_bacc.exact_bacc for x in centers) / 9
        observed = finite(self.equal_center_exact_bacc, "equal_center_exact_bacc")
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise ProtocolError("Equal-center BACC drifted from center metrics.")
        if self.single_class_case_count != sum(x.n_positive == 0 or x.n_negative == 0 for x in rows):
            raise ProtocolError("Single-class whole-case accounting drifted.")
        object.__setattr__(self, "case_confusions", rows)
        object.__setattr__(self, "center_metrics", centers)
        object.__setattr__(self, "equal_center_exact_bacc", observed)
        object.__setattr__(self, "method_result_hash", canonical_hash(self._unhashed()))

    @property
    def method_key(self) -> str:
        return self.method_id if self.geometry_id is None else f"{self.geometry_id}:{self.method_id}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_terminal_method_v1",
            "geometry_id": self.geometry_id, "method_id": self.method_id,
            "case_confusions": [count_payload(x) for x in self.case_confusions],
            "center_metrics": [x.to_payload() for x in self.center_metrics],
            "all_center_pooled_bacc": pooled_payload(self.all_center_pooled_bacc),
            "equal_center_exact_bacc": self.equal_center_exact_bacc,
            "single_class_case_count": self.single_class_case_count,
            "single_class_cases_retained": True, "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "method_result_hash": self.method_result_hash}


@dataclass(frozen=True)
class FoldRankStability:
    target_center: str
    fold_ordinal: int
    geometry_id: str
    support_scores: tuple[tuple[str, float], ...]
    evaluation_scores: tuple[tuple[str, float], ...]
    result: RankStabilityResult
    rank_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or self.fold_ordinal not in range(5) or self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Fold rank-stability identity is invalid.")
        if tuple(x[0] for x in self.support_scores) != self.result.action_ids or tuple(x[0] for x in self.evaluation_scores) != self.result.action_ids:
            raise ProtocolError("Fold rank-stability vectors are misaligned.")
        object.__setattr__(self, "rank_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_fold_rank_stability_v1",
            "target_center": self.target_center, "fold_ordinal": self.fold_ordinal,
            "geometry_id": self.geometry_id,
            "support_scores": [list(x) for x in self.support_scores],
            "evaluation_scores": [list(x) for x in self.evaluation_scores],
            "result": rank_payload(self.result),
            "support_evaluation_whole_case_disjoint": True, "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "rank_hash": self.rank_hash}


@dataclass(frozen=True, order=True)
class NormalizedOracleGap:
    geometry_id: str
    method_id: str
    selected_equal_center_bacc: float
    uniform_equal_center_bacc: float
    static_oracle_equal_center_bacc: float
    normalized_gap: float
    degenerate_static_headroom: bool

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS or self.method_id not in ("G", "R", "P", "S_y"):
            raise ProtocolError("Normalized oracle-gap identity is invalid.")
        for name in ("selected_equal_center_bacc", "uniform_equal_center_bacc", "static_oracle_equal_center_bacc", "normalized_gap"):
            finite(getattr(self, name), name)
        headroom = self.static_oracle_equal_center_bacc - self.uniform_equal_center_bacc
        if self.degenerate_static_headroom is not (headroom <= 1e-12):
            raise ProtocolError("Normalized oracle-gap degeneracy flag drifted.")


@dataclass(frozen=True)
class TerminalGeometryResult:
    geometry_id: str
    method_summaries: tuple[TerminalMethodSummary, ...]
    contrasts: tuple[TerminalContrast, ...]
    complementarity: tuple[PairwiseComplementarity, ...]
    fold_rank_stability: tuple[FoldRankStability, ...]
    normalized_oracle_gaps: tuple[NormalizedOracleGap, ...]
    geometry_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS or tuple(x.method_id for x in self.method_summaries) != METHOD_ORDER:
            raise ProtocolError("Terminal geometry method surface is incomplete.")
        if len(self.contrasts) != 9 or any(x.geometry_id != self.geometry_id for x in self.contrasts):
            raise ProtocolError("Terminal geometry contrasts are incomplete.")
        if len(self.fold_rank_stability) != 45:
            raise ProtocolError("Rank stability must cover every center-fold cell.")
        if tuple(x.method_id for x in self.normalized_oracle_gaps) != ("G", "R", "P", "S_y"):
            raise ProtocolError("Normalized oracle gaps are incomplete.")
        object.__setattr__(self, "geometry_result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_terminal_geometry_v1",
            "geometry_id": self.geometry_id,
            "method_summaries": [x.to_payload() for x in self.method_summaries],
            "contrasts": [x.to_payload() for x in self.contrasts],
            "complementarity": [dict(x.__dict__) for x in self.complementarity],
            "fold_rank_stability": [x.to_payload() for x in self.fold_rank_stability],
            "normalized_oracle_gaps": [dict(x.__dict__) for x in self.normalized_oracle_gaps],
            "cross_geometry_selection_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "geometry_result_hash": self.geometry_result_hash}


@dataclass(frozen=True)
class TerminalScientificResult:
    global_b: TerminalMethodSummary
    geometries: tuple[TerminalGeometryResult, ...]
    terminal_label_surface_hash: str
    scientific_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.global_b.method_id != "B" or self.global_b.geometry_id is not None:
            raise ProtocolError("Terminal result lacks global B.")
        if tuple(x.geometry_id for x in self.geometries) != GEOMETRY_IDS:
            raise ProtocolError("Terminal result requires parallel A0/A1 surfaces.")
        require_sha256(self.terminal_label_surface_hash, "terminal_label_surface_hash")
        object.__setattr__(self, "scientific_result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_terminal_scientific_result_v1",
            "global_b": self.global_b.to_payload(),
            "geometries": [x.to_payload() for x in self.geometries],
            "terminal_label_surface_hash": self.terminal_label_surface_hash,
            "primary_endpoint": "center_pooled_exact_bacc_equal_center_aggregate",
            "whole_case_cluster_uncertainty": True, "single_class_cases_retained": True,
            "per_case_bacc_stored_or_used": False, "geometry_selected": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "scientific_result_hash": self.scientific_result_hash}


@dataclass(frozen=True)
class TerminalSealedEnvelope:
    scientific_result: TerminalScientificResult
    probability_surface_hash: str
    all_decisions_seal_hash: str
    permutation_provenance_hash: str
    partition_hash: str
    label_capability_report_hash: str
    protocol_contract_hash: str
    sealed_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.probability_surface_hash, "probability_surface_hash"),
            (self.all_decisions_seal_hash, "all_decisions_seal_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.partition_hash, "partition_hash"),
            (self.label_capability_report_hash, "label_capability_report_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
        ):
            require_sha256(value, name)
        object.__setattr__(self, "sealed_result_hash", canonical_hash(self._unhashed()))

    @property
    def method_summaries(self) -> tuple[TerminalMethodSummary, ...]:
        return (self.scientific_result.global_b, *(x for g in self.scientific_result.geometries for x in g.method_summaries))

    @property
    def case_confusions(self) -> tuple[CaseConfusionCounts, ...]:
        return tuple(x for method in self.method_summaries for x in method.case_confusions)

    @property
    def center_metrics(self) -> tuple[CenterMetric, ...]:
        return tuple(x for method in self.method_summaries for x in method.center_metrics)

    def table_rows(self) -> Mapping[str, tuple[dict[str, object], ...]]:
        confusion = tuple(
            {"geometry_id": method.geometry_id, "method_id": method.method_id,
             **count_payload(row), "method_result_hash": method.method_result_hash}
            for method in self.method_summaries for row in method.case_confusions
        )
        centers = tuple(
            {"geometry_id": method.geometry_id, "method_id": method.method_id,
             "target_center": row.target_center, **pooled_payload(row.pooled_bacc),
             "method_result_hash": method.method_result_hash}
            for method in self.method_summaries for row in method.center_metrics
        )
        methods = tuple(
            {"geometry_id": x.geometry_id, "method_id": x.method_id,
             "equal_center_exact_bacc": x.equal_center_exact_bacc,
             "all_center_pooled_exact_bacc": x.all_center_pooled_bacc.exact_bacc,
             "case_count": len(x.case_confusions), "single_class_case_count": x.single_class_case_count,
             "method_result_hash": x.method_result_hash}
            for x in self.method_summaries
        )
        contrasts = tuple(
            {
                "contrast_id": row.contrast_id,
                "contrast_family": row.contrast_family,
                "geometry_id": row.geometry_id,
                "challenger_method": row.challenger_method,
                "reference_method": row.reference_method,
                "center_differences": row.center_differences,
                "equal_center_difference": row.equal_center_difference,
                "center_t_ci95_lower": row.center_t_ci95_lower,
                "center_t_ci95_upper": row.center_t_ci95_upper,
                "bootstrap_mean": row.bootstrap.bootstrap_mean,
                "bootstrap_ci95_lower": row.bootstrap.ci95_lower,
                "bootstrap_ci95_upper": row.bootstrap.ci95_upper,
                "bootstrap_replicate_count": row.bootstrap.replicate_count,
                "bootstrap_seed": row.bootstrap.seed,
                "bootstrap_invalid_draw_count": row.bootstrap.invalid_draw_count,
                "bootstrap_hash": row.bootstrap.bootstrap_hash,
                "contrast_hash": row.contrast_hash,
                "geometry_result_hash": geometry.geometry_result_hash,
            }
            for geometry in self.scientific_result.geometries
            for row in geometry.contrasts
        )
        gaps = tuple(
            {**dict(x.__dict__), "geometry_result_hash": geometry.geometry_result_hash}
            for geometry in self.scientific_result.geometries for x in geometry.normalized_oracle_gaps
        )
        complementarity = tuple(
            {"geometry_id": geometry.geometry_id, **dict(x.__dict__),
             "geometry_result_hash": geometry.geometry_result_hash}
            for geometry in self.scientific_result.geometries for x in geometry.complementarity
        )
        ranks = tuple(
            {"target_center": x.target_center, "fold_ordinal": x.fold_ordinal,
             "geometry_id": x.geometry_id, "spearman": x.result.spearman,
             "identifiable": x.result.identifiable, "rank_hash": x.rank_hash}
            for geometry in self.scientific_result.geometries for x in geometry.fold_rank_stability
        )
        permutation = tuple(
            {"geometry_id": geometry.geometry_id, "method_id": "P",
             "P_equal_center_exact_bacc": next(x.equal_center_exact_bacc for x in geometry.method_summaries if x.method_id == "P"),
             "R_minus_P_equal_center": next(x.equal_center_difference for x in geometry.contrasts if x.challenger_method == "R" and x.reference_method == "P"),
             "permutation_provenance_hash": self.permutation_provenance_hash,
             "geometry_result_hash": geometry.geometry_result_hash}
            for geometry in self.scientific_result.geometries
        )
        return MappingProxyType({
            "tables/terminal_case_confusions.csv": confusion,
            "tables/terminal_center_metrics.csv": centers,
            "tables/terminal_method_summary.csv": methods,
            "tables/terminal_contrasts.csv": contrasts,
            "tables/oracle_rank_metrics.csv": gaps,
            "tables/complementarity.csv": complementarity,
            "tables/rank_stability.csv": ranks,
            "tables/permutation_metrics.csv": permutation,
        })

    @property
    def diagnostic_tables(self) -> Mapping[str, tuple[dict[str, object], ...]]:
        return self.table_rows()

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_terminal_sealed_envelope_v1",
            "scientific_result": self.scientific_result.to_payload(),
            "probability_surface_hash": self.probability_surface_hash,
            "all_decisions_seal_hash": self.all_decisions_seal_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "partition_hash": self.partition_hash,
            "label_capability_report_hash": self.label_capability_report_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "sealed_result_hash": self.sealed_result_hash}


__all__ = (
    "CenterMetric", "FoldRankStability", "METHOD_ORDER", "NormalizedOracleGap",
    "TerminalGeometryResult", "TerminalMethodSummary", "TerminalScientificResult",
    "TerminalSealedEnvelope", "count_payload", "pooled_payload",
)
