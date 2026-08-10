"""Pure evaluation result contracts for the consumed-test diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS
from .contracts import PooledBacc
from .hashing import canonical_hash, finite
from .metrics import PairwiseComplementarity, RankStabilityResult


_EVALUATION_METHOD_ORDER = ("U", "G", "R", "P", "S_y", "O_static", "O_case")


def _pooled_payload(value: PooledBacc) -> dict[str, object]:
    return {
        "action_or_method_id": value.action_or_method_id,
        "case_count": value.case_count,
        "n_positive": value.n_positive,
        "true_positive": value.true_positive,
        "n_negative": value.n_negative,
        "true_negative": value.true_negative,
        "sensitivity": value.sensitivity,
        "specificity": value.specificity,
        "exact_bacc": value.exact_bacc,
    }


@dataclass(frozen=True)
class MethodEvaluationResult:
    geometry_id: str
    method_id: str
    pooled_bacc: PooledBacc
    delta_vs_b: float
    delta_vs_u: float
    decision_count: int
    terminal_diagnostic_only: bool

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS or self.method_id not in _EVALUATION_METHOD_ORDER:
            raise ProtocolError("Method evaluation has an invalid geometry/method.")
        if self.decision_count <= 0:
            raise ProtocolError("Method evaluation requires case decisions.")
        object.__setattr__(self, "delta_vs_b", finite(self.delta_vs_b, "delta_vs_b"))
        object.__setattr__(self, "delta_vs_u", finite(self.delta_vs_u, "delta_vs_u"))
        expected_terminal = self.method_id in ("O_static", "O_case")
        if self.terminal_diagnostic_only is not expected_terminal:
            raise ProtocolError("Only O_static/O_case may be marked terminal oracle rows.")

    def to_payload(self) -> dict[str, object]:
        return {
            "geometry_id": self.geometry_id,
            "method_id": self.method_id,
            "pooled_bacc": _pooled_payload(self.pooled_bacc),
            "delta_vs_b": self.delta_vs_b,
            "delta_vs_u": self.delta_vs_u,
            "decision_count": self.decision_count,
            "terminal_diagnostic_only": self.terminal_diagnostic_only,
        }


@dataclass(frozen=True)
class GeometryEvaluationResult:
    geometry_id: str
    methods: tuple[MethodEvaluationResult, ...]
    rank_stability: RankStabilityResult
    complementarity: tuple[PairwiseComplementarity, ...]
    normalized_oracle_gaps: Mapping[str, float]
    geometry_result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Unknown evaluation geometry.")
        methods = tuple(self.methods)
        if tuple(value.method_id for value in methods) != _EVALUATION_METHOD_ORDER or any(
            value.geometry_id != self.geometry_id for value in methods
        ):
            raise ProtocolError("Geometry result must contain U/G/R/P/S_y/O_static/O_case exactly once.")
        gaps = {
            str(method): finite(value, f"normalized oracle gap {method}")
            for method, value in self.normalized_oracle_gaps.items()
        }
        if set(gaps) != {"G", "R", "P", "S_y"}:
            raise ProtocolError("Normalized oracle gaps must cover G/R/P/S_y.")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "normalized_oracle_gaps", MappingProxyType(gaps))
        object.__setattr__(self, "geometry_result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_geometry_result_v1",
            "geometry_id": self.geometry_id,
            "methods": [value.to_payload() for value in self.methods],
            "rank_stability": {
                "action_ids": list(self.rank_stability.action_ids),
                "support_ranks": list(self.rank_stability.support_ranks),
                "evaluation_ranks": list(self.rank_stability.evaluation_ranks),
                "spearman": self.rank_stability.spearman,
                "identifiable": self.rank_stability.identifiable,
            },
            "complementarity": [
                {
                    key: getattr(value, key)
                    for key in value.__dataclass_fields__
                }
                for value in self.complementarity
            ],
            "normalized_oracle_gaps": dict(self.normalized_oracle_gaps),
            "cross_geometry_selection_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "geometry_result_hash": self.geometry_result_hash}


@dataclass(frozen=True)
class DiagnosticEvaluationResult:
    global_b: PooledBacc
    geometries: tuple[GeometryEvaluationResult, ...]
    reused_test_dataset: bool = True
    claim_status: str = "EXPLORATORY_CONSUMED_DATA_ONLY"
    routing_or_promotion_authorized: bool = False
    another_experiment_authorized: bool = False
    evaluation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        geometries = tuple(self.geometries)
        if tuple(value.geometry_id for value in geometries) != GEOMETRY_IDS:
            raise ProtocolError("Diagnostic result requires parallel A0 then A1 surfaces.")
        if (
            self.reused_test_dataset is not True
            or self.claim_status != "EXPLORATORY_CONSUMED_DATA_ONLY"
            or self.routing_or_promotion_authorized is not False
            or self.another_experiment_authorized is not False
        ):
            raise ProtocolError("Consumed-test result cannot authorize routing, promotion, or another experiment.")
        object.__setattr__(self, "geometries", geometries)
        object.__setattr__(self, "evaluation_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_recoverability_result_v1",
            "global_b": _pooled_payload(self.global_b),
            "geometries": [value.to_payload() for value in self.geometries],
            "reused_test_dataset": True,
            "claim_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
            "routing_or_promotion_authorized": False,
            "another_experiment_authorized": False,
            "test_labels_used_for_terminal_scoring_only": True,
            "geometry_selected": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "evaluation_hash": self.evaluation_hash}


__all__ = (
    "DiagnosticEvaluationResult",
    "GeometryEvaluationResult",
    "MethodEvaluationResult",
)
