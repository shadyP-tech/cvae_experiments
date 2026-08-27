"""Source-only row-posterior contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .shared import ProtocolError, _finite_tuple, _sorted_unique, _text, canonical_sha256

@dataclass(frozen=True, slots=True)
class SourceScopeReceipt:
    """Exact H/J/K/L/d exclusion receipt for source-only posterior fitting.

    H is the outer target, J the source query, K the hyperparameter-validation
    center, L the residual-calibration center, and d the held-out case.  K and
    L are deliberately distinct roles; neither may enter estimator fitting.
    """

    outer_target_center: str
    query_center: str
    hyperparameter_center: str
    calibration_center: str
    heldout_case_center: str
    heldout_case_id: str
    training_center_ids: tuple[str, ...]
    training_case_keys: tuple[tuple[str, str], ...]
    role: str = "SOURCE_ONLY_H_J_K_L_D_EXCLUDED"
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = _text(self.outer_target_center, role="outer target center H")
        j = _text(self.query_center, role="query center J")
        k = _text(self.hyperparameter_center, role="hyperparameter center K")
        ell = _text(self.calibration_center, role="residual calibration center L")
        d_center = _text(self.heldout_case_center, role="held-out case center")
        d = _text(self.heldout_case_id, role="held-out case d")
        centers = _sorted_unique(self.training_center_ids, role="training center")
        raw_case_keys = tuple(self.training_case_keys)
        case_keys = tuple(
            sorted(
                (
                    _text(center, role="training case center"),
                    _text(case, role="training case id"),
                )
                for center, case in raw_case_keys
            )
        )
        if len({h, j, k, ell}) != 4:
            raise ProtocolError("Source-only H/J/K/L center roles must be distinct.")
        if (
            d_center != j
            or
            len(set(case_keys)) != len(case_keys)
            or {center for center, _ in case_keys} != set(centers)
            or set(centers).intersection({h, j, k, ell})
            or (d_center, d) in case_keys
        ):
            raise ProtocolError("Source-only receipt violates exact H/J/K/L/d exclusion.")
        if self.role != "SOURCE_ONLY_H_J_K_L_D_EXCLUDED":
            raise ProtocolError("Source-only receipt role drifted.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "query_center", j)
        object.__setattr__(self, "hyperparameter_center", k)
        object.__setattr__(self, "calibration_center", ell)
        object.__setattr__(self, "heldout_case_center", d_center)
        object.__setattr__(self, "heldout_case_id", d)
        object.__setattr__(self, "training_center_ids", centers)
        object.__setattr__(self, "training_case_keys", case_keys)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "source_scope_receipt_v2",
                    "H": h,
                    "J": j,
                    "K": k,
                    "L": ell,
                    "d_center": d_center,
                    "d": d,
                    "training_centers": centers,
                    "training_case_keys": case_keys,
                    "source_labels_only": True,
                    "target_labels_used": False,
                    "role": self.role,
                }
            ),
        )
@dataclass(frozen=True, slots=True)
class RowPosteriorObservation:
    """Ephemeral source-training row; labels never enter fitted records."""

    center_id: str
    case_id: str
    row_id: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    outcome: int

    def __post_init__(self) -> None:
        names = tuple(_text(name, role="row feature name") for name in self.feature_names)
        values = _finite_tuple(self.feature_values, role="row feature values")
        if len(names) != len(values) or len(set(names)) != len(names):
            raise ProtocolError("Row-posterior feature schema is empty, duplicated, or misaligned.")
        outcome = int(self.outcome)
        if outcome not in (0, 1):
            raise ProtocolError("Row-posterior source outcome must be binary.")
        object.__setattr__(self, "center_id", _text(self.center_id, role="source center"))
        object.__setattr__(self, "case_id", _text(self.case_id, role="source case"))
        object.__setattr__(self, "row_id", _text(self.row_id, role="source row"))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "outcome", outcome)


@dataclass(frozen=True, slots=True)
class RowPosteriorModel:
    """Low-capacity source-only logistic posterior without training outcomes."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge_alpha: float
    training_row_count: int
    training_center_count: int
    training_case_count: int
    source_scope_receipt_hash: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        mean = tuple(float(value) for value in self.feature_mean)
        scale = tuple(float(value) for value in self.feature_scale)
        coefficients = tuple(float(value) for value in self.coefficients)
        if (
            not names
            or len(names) != len(mean)
            or len(names) != len(scale)
            or len(names) != len(coefficients)
            or not all(math.isfinite(value) for value in (*mean, *scale, *coefficients, self.intercept))
            or any(value <= 0.0 for value in scale)
            or float(self.ridge_alpha) <= 0.0
            or min(self.training_row_count, self.training_center_count, self.training_case_count) <= 0
        ):
            raise ProtocolError("Row-posterior model contract is invalid.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "ridge_alpha", float(self.ridge_alpha))
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="source scope receipt hash"),
        )
        object.__setattr__(
            self,
            "model_hash",
            canonical_sha256(
                {
                    "schema": "source_row_posterior_model_v1",
                    "feature_names": names,
                    "feature_mean": mean,
                    "feature_scale": scale,
                    "intercept": self.intercept,
                    "coefficients": coefficients,
                    "ridge_alpha": self.ridge_alpha,
                    "row_count": self.training_row_count,
                    "center_count": self.training_center_count,
                    "case_count": self.training_case_count,
                    "scope_receipt_hash": self.source_scope_receipt_hash,
                    "training_outcomes_persisted": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RowPosteriorPrediction:
    """Label-free row probability prediction."""

    eta: float
    model_hash: str
    source_scope_receipt_hash: str

    def __post_init__(self) -> None:
        eta = float(self.eta)
        if not math.isfinite(eta) or eta <= 0.0 or eta >= 1.0:
            raise ProtocolError("Row-posterior prediction must be strictly inside (0, 1).")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "model_hash", _text(self.model_hash, role="row model hash"))
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="source scope receipt hash"),
        )


@dataclass(frozen=True, slots=True)
class RowPosteriorOOFPrediction:
    """Source-OOF prediction record; no realized outcome is retained."""

    center_id: str
    case_id: str
    row_id: str
    eta: float
    model_hash: str
    source_scope_receipt_hash: str

    def __post_init__(self) -> None:
        eta = float(self.eta)
        if not math.isfinite(eta) or eta <= 0.0 or eta >= 1.0:
            raise ProtocolError("Source-OOF eta must lie strictly inside (0, 1).")
        object.__setattr__(self, "center_id", _text(self.center_id, role="OOF center"))
        object.__setattr__(self, "case_id", _text(self.case_id, role="OOF case"))
        object.__setattr__(self, "row_id", _text(self.row_id, role="OOF row"))
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "model_hash", _text(self.model_hash, role="OOF model hash"))
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="OOF source scope receipt hash"),
        )
