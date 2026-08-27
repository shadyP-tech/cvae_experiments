"""Clustered uncertainty contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .shared import P_ACTION_ID, ProtocolError, UNCERTAINTY_METRICS, _text, canonical_sha256

@dataclass(frozen=True, slots=True)
class OOFResidualObservation:
    """Ephemeral source-OOF residual row for one action/pair and metric."""

    center_id: str
    case_id: str
    oof_held_center: str
    action_id: str
    comparator_id: str
    metric: str
    predicted: float
    observed: float
    source_scope_receipt_hash: str

    def __post_init__(self) -> None:
        center = _text(self.center_id, role="residual center")
        held = _text(self.oof_held_center, role="OOF held center")
        metric = str(self.metric)
        if held != center:
            raise ProtocolError("Residual calibration requires genuine held-center OOF predictions.")
        if metric not in UNCERTAINTY_METRICS:
            raise ProtocolError("Unknown residual-calibration metric.")
        predicted = float(self.predicted)
        observed = float(self.observed)
        if not math.isfinite(predicted) or not math.isfinite(observed):
            raise ProtocolError("Residual-calibration values must be finite.")
        comparator = _text(self.comparator_id, role="residual comparator")
        if metric != "pairwise" and comparator != P_ACTION_ID:
            raise ProtocolError("Action utility residuals must be calibrated against P.")
        if metric == "pairwise" and comparator == self.action_id:
            raise ProtocolError("Pairwise residual calibration needs distinct actions.")
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "case_id", _text(self.case_id, role="residual case"))
        object.__setattr__(self, "oof_held_center", held)
        object.__setattr__(self, "action_id", _text(self.action_id, role="residual action"))
        object.__setattr__(self, "comparator_id", comparator)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="residual source scope receipt hash"),
        )
@dataclass(frozen=True, slots=True)
class UncertaintyComponent:
    """One action/pair-specific one-sided residual offset."""

    action_id: str
    comparator_id: str
    metric: str
    side: str
    offset: float
    alpha: float
    center_count: int
    case_count: int
    scope_receipt_hashes: tuple[str, ...]
    component_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.metric not in UNCERTAINTY_METRICS or self.side not in ("lower", "upper"):
            raise ProtocolError("Uncertainty component metric or side is invalid.")
        if (
            not math.isfinite(float(self.offset))
            or float(self.offset) < 0.0
            or not 0.0 < float(self.alpha) < 0.5
            or int(self.center_count) < 4
            or int(self.case_count) < int(self.center_count)
        ):
            raise ProtocolError("Uncertainty component is not calibratable.")
        object.__setattr__(self, "action_id", _text(self.action_id, role="uncertainty action"))
        object.__setattr__(self, "comparator_id", _text(self.comparator_id, role="uncertainty comparator"))
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "center_count", int(self.center_count))
        object.__setattr__(self, "case_count", int(self.case_count))
        scopes = tuple(sorted(_text(v, role="uncertainty scope receipt hash") for v in self.scope_receipt_hashes))
        if not scopes or len(set(scopes)) != len(scopes):
            raise ProtocolError("Uncertainty component scope receipts are invalid.")
        object.__setattr__(self, "scope_receipt_hashes", scopes)
        object.__setattr__(self, "component_hash", canonical_sha256({"schema": "action_pair_center_clustered_residual_component_v1", "action_id": self.action_id, "comparator_id": self.comparator_id, "metric": self.metric, "side": self.side, "alpha": self.alpha, "offset": self.offset, "center_count": self.center_count, "case_count": self.case_count, "scope_receipt_hashes": scopes, "raw_residuals_persisted": False}))


@dataclass(frozen=True, slots=True)
class UncertaintyCalibration:
    """Source-OOF action/pair-specific residual calibration surface."""

    components: tuple[UncertaintyComponent, ...]
    outer_target_center: str
    calibration_scope_receipt_hashes: tuple[str, ...]
    source_scope_receipt_hash: str = field(init=False)
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        components = tuple(self.components)
        keys = tuple(
            (component.action_id, component.comparator_id, component.metric, component.side)
            for component in components
        )
        if not components or len(set(keys)) != len(keys) or tuple(sorted(keys)) != keys:
            raise ProtocolError("Uncertainty components must be unique and canonically ordered.")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self, "outer_target_center", _text(self.outer_target_center, role="uncertainty outer target H")
        )
        scopes = tuple(sorted(_text(v, role="calibration scope receipt hash") for v in self.calibration_scope_receipt_hashes))
        if len(scopes) < 4 or len(set(scopes)) != len(scopes):
            raise ProtocolError("Uncertainty calibration scope receipts are invalid.")
        object.__setattr__(self, "calibration_scope_receipt_hashes", scopes)
        source_hash = canonical_sha256({"schema": "rotating_L_calibration_scope_v1", "scope_receipt_hashes": scopes, "outer_target_H": self.outer_target_center})
        object.__setattr__(self, "source_scope_receipt_hash", source_hash)
        object.__setattr__(self, "calibration_hash", canonical_sha256({"schema": "clustered_uncertainty_calibration_v1", "combined_scope_hash": source_hash, "outer_target_H": self.outer_target_center, "component_hashes": tuple(c.component_hash for c in components), "global_pooled_fallback": False}))

    def component(
        self, action_id: object, comparator_id: object, metric: object, side: object
    ) -> UncertaintyComponent:
        key = (str(action_id), str(comparator_id), str(metric), str(side))
        for component in self.components:
            if (
                component.action_id,
                component.comparator_id,
                component.metric,
                component.side,
            ) == key:
                return component
        raise ProtocolError(f"Missing action/pair-specific uncertainty component: {key}")


@dataclass(frozen=True, slots=True)
class CalibratedBound:
    """Label-free mean plus one source-OOF one-sided bound."""

    mean: float
    bound: float
    side: str
    component_hash: str

    def __post_init__(self) -> None:
        if self.side not in ("lower", "upper") or not all(
            math.isfinite(float(value)) for value in (self.mean, self.bound)
        ):
            raise ProtocolError("Calibrated bound is invalid.")
        if self.side == "lower" and self.bound > self.mean + 1.0e-15:
            raise ProtocolError("A lower bound cannot exceed its mean.")
        if self.side == "upper" and self.bound < self.mean - 1.0e-15:
            raise ProtocolError("An upper bound cannot be below its mean.")
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "bound", float(self.bound))
        object.__setattr__(self, "component_hash", _text(self.component_hash, role="uncertainty component hash"))
