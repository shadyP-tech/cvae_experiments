"""Label-free persisted DTOs for terminal aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..artifacts.hashing import canonical_hash, require_sha256
from ..protocol import GovernanceError


@dataclass(frozen=True, slots=True)
class CenterMetrics:
    target_center: str
    row_count: int
    bacc: float
    brier: float
    log_loss: float
    metrics_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (float(self.bacc), float(self.brier), float(self.log_loss))
        if (
            not str(self.target_center)
            or type(self.row_count) is not int
            or self.row_count <= 0
            or not all(math.isfinite(value) for value in values)
            or not 0.0 <= values[0] <= 1.0
            or values[1] < 0.0
            or values[2] < 0.0
        ):
            raise GovernanceError("SCALE-BP v2 center terminal metrics drifted.")
        object.__setattr__(self, "bacc", values[0])
        object.__setattr__(self, "brier", values[1])
        object.__setattr__(self, "log_loss", values[2])
        object.__setattr__(
            self,
            "metrics_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_center_terminal_metrics_v1",
            "target_center": self.target_center,
            "row_count": self.row_count,
            "bacc": self.bacc,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }
        if include_hash:
            payload["metrics_hash"] = self.metrics_hash
        return payload


@dataclass(frozen=True, slots=True)
class TerminalMetrics:
    method_id: str
    row_count: int
    prediction_hash: str
    decision_seal_hash: str
    pooled_bacc: float
    pooled_brier: float
    pooled_log_loss: float
    equal_center_bacc: float
    equal_center_brier: float
    equal_center_log_loss: float
    center_metrics: tuple[CenterMetrics, ...]
    metrics_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(self.center_metrics)
        values = (
            float(self.pooled_bacc),
            float(self.pooled_brier),
            float(self.pooled_log_loss),
            float(self.equal_center_bacc),
            float(self.equal_center_brier),
            float(self.equal_center_log_loss),
        )
        if (
            not str(self.method_id)
            or type(self.row_count) is not int
            or self.row_count <= 0
            or not centers
            or len({row.target_center for row in centers}) != len(centers)
            or sum(row.row_count for row in centers) != self.row_count
            or not all(math.isfinite(value) for value in values)
            or any(not 0.0 <= value <= 1.0 for value in (values[0], values[3]))
            or any(value < 0.0 for value in (values[1], values[2], values[4], values[5]))
        ):
            raise GovernanceError("SCALE-BP v2 terminal metrics drifted.")
        require_sha256(self.prediction_hash, "terminal prediction hash")
        require_sha256(self.decision_seal_hash, "terminal decision seal")
        for name, value in zip(
            (
                "pooled_bacc",
                "pooled_brier",
                "pooled_log_loss",
                "equal_center_bacc",
                "equal_center_brier",
                "equal_center_log_loss",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "center_metrics", centers)
        object.__setattr__(
            self,
            "metrics_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_terminal_metrics_v1",
            "method_id": self.method_id,
            "row_count": self.row_count,
            "prediction_hash": self.prediction_hash,
            "decision_seal_hash": self.decision_seal_hash,
            "pooled": {
                "bacc": self.pooled_bacc,
                "brier": self.pooled_brier,
                "log_loss": self.pooled_log_loss,
            },
            "equal_center": {
                "bacc": self.equal_center_bacc,
                "brier": self.equal_center_brier,
                "log_loss": self.equal_center_log_loss,
            },
            "centers": [row.to_payload() for row in self.center_metrics],
            "raw_labels_persisted": False,
        }
        if include_hash:
            payload["metrics_hash"] = self.metrics_hash
        return payload


@dataclass(frozen=True, slots=True)
class TerminalComparison:
    method_id: str
    protected_method_id: str
    pooled_delta_bacc: float
    pooled_delta_brier: float
    pooled_delta_log_loss: float
    equal_center_delta_bacc: float
    equal_center_delta_brier: float
    equal_center_delta_log_loss: float
    comparison_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(
            float(value)
            for value in (
                self.pooled_delta_bacc,
                self.pooled_delta_brier,
                self.pooled_delta_log_loss,
                self.equal_center_delta_bacc,
                self.equal_center_delta_brier,
                self.equal_center_delta_log_loss,
            )
        )
        if (
            not self.method_id
            or not self.protected_method_id
            or self.method_id == self.protected_method_id
            or not all(math.isfinite(value) for value in values)
        ):
            raise GovernanceError("SCALE-BP v2 terminal comparison drifted.")
        for name, value in zip(
            (
                "pooled_delta_bacc",
                "pooled_delta_brier",
                "pooled_delta_log_loss",
                "equal_center_delta_bacc",
                "equal_center_delta_brier",
                "equal_center_delta_log_loss",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "comparison_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_terminal_comparison_v1",
            "method_id": self.method_id,
            "protected_method_id": self.protected_method_id,
            "pooled_delta": {
                "bacc": self.pooled_delta_bacc,
                "brier": self.pooled_delta_brier,
                "log_loss": self.pooled_delta_log_loss,
            },
            "equal_center_delta": {
                "bacc": self.equal_center_delta_bacc,
                "brier": self.equal_center_delta_brier,
                "log_loss": self.equal_center_delta_log_loss,
            },
        }
        if include_hash:
            payload["comparison_hash"] = self.comparison_hash
        return payload


@dataclass(frozen=True, slots=True)
class TerminalAggregate:
    decision_seal_hash: str
    protected_method_id: str
    metrics: tuple[TerminalMetrics, ...]
    comparisons: tuple[TerminalComparison, ...]
    terminal_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        comparisons = tuple(self.comparisons)
        method_ids = tuple(row.method_id for row in metrics)
        if (
            not metrics
            or method_ids[0] != self.protected_method_id
            or len(set(method_ids)) != len(method_ids)
            or any(row.decision_seal_hash != self.decision_seal_hash for row in metrics)
            or tuple(row.method_id for row in comparisons) != method_ids[1:]
            or any(row.protected_method_id != self.protected_method_id for row in comparisons)
        ):
            raise GovernanceError("SCALE-BP v2 terminal aggregate topology drifted.")
        require_sha256(self.decision_seal_hash, "terminal decision seal")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(
            self,
            "terminal_seal_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    @property
    def terminal_metrics_hash(self) -> str:
        return canonical_hash([row.metrics_hash for row in self.metrics])

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_terminal_aggregate_v1",
            "decision_seal_hash": self.decision_seal_hash,
            "protected_method_id": self.protected_method_id,
            "methods": [row.to_payload() for row in self.metrics],
            "comparisons_to_protected_p": [row.to_payload() for row in self.comparisons],
            "method_count": len(self.metrics),
            "terminal_metrics_hash": self.terminal_metrics_hash,
            "raw_labels_persisted": False,
            "row_level_labels_persisted": False,
            "terminal_scoring_only": True,
        }
        if include_hash:
            payload["terminal_seal_hash"] = self.terminal_seal_hash
        return payload


__all__ = (
    "CenterMetrics",
    "TerminalAggregate",
    "TerminalComparison",
    "TerminalMetrics",
)
