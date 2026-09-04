"""Case-balanced endpoint model used by the HARP v16 hierarchy.

All physical actions contribute to all four heads.  The hierarchy is applied
after endpoint prediction, avoiding the predecessor's rank-first bottleneck
where a single selected-action acceptor suppressed useful relative signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CasePrediction,
    EndpointPrediction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SurfaceRole,
    canonical_text,
)
from .features import FittedFeatureMap, case_balanced_weights, fit_feature_map
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class RidgeHead:
    endpoint: str
    coefficients: tuple[float, ...]
    ridge_alpha: float
    head_hash: str = field(init=False)

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint)
        coefficients = tuple(float(value) for value in self.coefficients)
        alpha = float(self.ridge_alpha)
        if (
            endpoint not in {"gain", "harm", "brier", "log_loss"}
            or not coefficients
            or any(not math.isfinite(value) for value in coefficients)
            or not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            raise ProtocolError("HARP v16 endpoint head is malformed.")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(
            self,
            "head_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_ridge_head_v16",
                    "endpoint": endpoint,
                    "coefficients": coefficients,
                    "ridge_alpha": alpha,
                    "case_balanced_fit": True,
                }
            ),
        )

    def predict(self, vector: np.ndarray) -> float:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if vector.shape != coefficients.shape:
            raise ProtocolError("HARP v16 endpoint feature dimension drifted.")
        output = float(np.dot(vector, coefficients))
        if not math.isfinite(output):
            raise ProtocolError("HARP v16 endpoint prediction is non-finite.")
        return output

    def public_payload(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "coefficients": list(self.coefficients),
            "ridge_alpha": self.ridge_alpha,
            "head_hash": self.head_hash,
            "case_balanced_fit": True,
        }


@dataclass(frozen=True, slots=True)
class SupportEndpointModel:
    outer_target_id: str
    feature_map: FittedFeatureMap
    gain_head: RidgeHead
    harm_head: RidgeHead
    brier_head: RidgeHead
    log_loss_head: RidgeHead
    training_case_ids: tuple[str, ...]
    ridge_alpha: float
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(str(value) for value in self.training_case_ids))
        heads = (self.gain_head, self.harm_head, self.brier_head, self.log_loss_head)
        expected = ("gain", "harm", "brier", "log_loss")
        if (
            not self.outer_target_id
            or not cases
            or len(cases) != len(set(cases))
            or cases != self.feature_map.fitted_case_ids
            or tuple(row.endpoint for row in heads) != expected
            or any(len(row.coefficients) != len(self.feature_map.vector_names) for row in heads)
            or any(not math.isclose(row.ridge_alpha, self.ridge_alpha) for row in heads)
        ):
            raise ProtocolError("HARP v16 support endpoint model is malformed.")
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_endpoint_model_v16",
                    "outer_target_id": self.outer_target_id,
                    "feature_map_hash": self.feature_map.feature_map_hash,
                    "head_hashes": tuple(row.head_hash for row in heads),
                    "training_case_ids": cases,
                    "fit_surface": SurfaceRole.TARGET_TRAIN_SUPPORT.value,
                    "evaluation_labels_consumed": False,
                    "all_actions_fit": True,
                }
            ),
        )

    def predict_menu(
        self,
        menu: LabelFreeCaseMenu,
        *,
        out_of_fold: bool,
    ) -> CasePrediction:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.outer_target_id != self.outer_target_id
        ):
            raise ProtocolError("HARP v16 prediction crossed an outer-target boundary.")
        predictions = tuple(
            self.predict_action(action, menu_hash=menu.menu_hash, out_of_fold=out_of_fold)
            for action in menu.actions
        )
        return CasePrediction(menu_hash=menu.menu_hash, action_predictions=predictions)

    @property
    def is_null(self) -> bool:
        return False

    def predict_action(
        self,
        action: LabelFreeAction,
        *,
        menu_hash: str,
        out_of_fold: bool,
    ) -> EndpointPrediction:
        if action.outer_target_id != self.outer_target_id:
            raise ProtocolError("HARP v16 action crossed an outer-target boundary.")
        vector = self.feature_map.transform(action)
        return EndpointPrediction(
            action=action,
            menu_hash=menu_hash,
            predicted_gain=self.gain_head.predict(vector),
            predicted_harm_probability=min(max(self.harm_head.predict(vector), 0.0), 1.0),
            predicted_brier_delta=self.brier_head.predict(vector),
            predicted_log_loss_delta=self.log_loss_head.predict(vector),
            training_case_ids=self.training_case_ids,
            feature_map_hash=self.feature_map.feature_map_hash,
            model_hash=self.model_hash,
            out_of_fold=bool(out_of_fold),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_endpoint_model_v16",
            "outer_target_id": self.outer_target_id,
            "feature_map": self.feature_map.public_payload(),
            "gain_head": self.gain_head.public_payload(),
            "harm_head": self.harm_head.public_payload(),
            "brier_head": self.brier_head.public_payload(),
            "log_loss_head": self.log_loss_head.public_payload(),
            "training_case_ids": list(self.training_case_ids),
            "ridge_alpha": self.ridge_alpha,
            "model_hash": self.model_hash,
            "fit_surface": SurfaceRole.TARGET_TRAIN_SUPPORT.value,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class NullSupportEndpointModel:
    """Deterministic always-B model for a support fold with no active action."""

    outer_target_id: str
    training_case_ids: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    reason: str = "NO_ACTIVE_SUPPORT_ACTION"
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = canonical_text(self.outer_target_id, name="null model outer target H")
        cases = tuple(
            sorted(
                canonical_text(value, name="null model training case id")
                for value in self.training_case_ids
            )
        )
        candidates = tuple(
            sorted(
                canonical_text(value, name="null model candidate source")
                for value in self.candidate_source_ids
            )
        )
        reason = canonical_text(self.reason, name="null model reason")
        if (
            not cases
            or len(cases) != len(set(cases))
            or len(candidates) != len(set(candidates))
            or outer in candidates
        ):
            raise ProtocolError("HARP v16 null support endpoint model is malformed.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "candidate_source_ids", candidates)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_null_endpoint_model_v16",
                    "outer_target_id": outer,
                    "training_case_ids": cases,
                    "candidate_source_ids": candidates,
                    "reason": reason,
                    "always_exact_b": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def is_null(self) -> bool:
        return True

    def predict_menu(
        self,
        menu: LabelFreeCaseMenu,
        *,
        out_of_fold: bool,
    ) -> CasePrediction:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.outer_target_id != self.outer_target_id
            or (out_of_fold and menu.case_id in self.training_case_ids)
        ):
            raise ProtocolError("HARP v16 null prediction crossed a support boundary.")
        # No endpoint prediction is invented when the fold contains no action
        # from which an endpoint head could be estimated.  The policy represents
        # the case explicitly and deterministically chooses exact B.
        return CasePrediction(menu_hash=menu.menu_hash, action_predictions=())

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_null_endpoint_model_v16",
            "outer_target_id": self.outer_target_id,
            "training_case_ids": list(self.training_case_ids),
            "candidate_source_ids": list(self.candidate_source_ids),
            "reason": self.reason,
            "model_hash": self.model_hash,
            "always_exact_b": True,
            "fit_surface": SurfaceRole.TARGET_TRAIN_SUPPORT.value,
            "evaluation_labels_consumed": False,
        }


def _fit_ridge_head(
    endpoint: str,
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> RidgeHead:
    if (
        matrix.ndim != 2
        or target.shape != (matrix.shape[0],)
        or weights.shape != target.shape
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ProtocolError("HARP v16 ridge inputs are malformed.")
    root_weight = np.sqrt(weights)
    weighted_x = matrix * root_weight[:, None]
    weighted_y = target * root_weight
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    normal = weighted_x.T @ weighted_x + penalty
    right = weighted_x.T @ weighted_y
    try:
        coefficients = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ProtocolError("HARP v16 ridge fit produced non-finite coefficients.")
    return RidgeHead(
        endpoint=endpoint,
        coefficients=tuple(float(value) for value in coefficients.tolist()),
        ridge_alpha=float(alpha),
    )


def fit_support_endpoint_model(
    outcomes: Sequence[SupportActionOutcome],
    *,
    config: RouterFitConfig,
    candidate_source_ids: Sequence[str] | None = None,
    training_case_ids: Sequence[str] | None = None,
    outer_target_id: str | None = None,
) -> SupportEndpointModel | NullSupportEndpointModel:
    rows = tuple(sorted(outcomes, key=lambda row: (row.action.case_id, row.action.action_id)))
    if not isinstance(config, RouterFitConfig):
        raise ProtocolError("HARP v16 endpoint fitting requires a typed configuration.")
    observed_outer_ids = {row.action.outer_target_id for row in rows}
    if outer_target_id is None:
        if len(observed_outer_ids) != 1:
            raise ProtocolError("HARP v16 endpoint fitting lacks one outer target H.")
        outer = next(iter(observed_outer_ids))
    else:
        outer = canonical_text(outer_target_id, name="endpoint model outer target H")
        if observed_outer_ids and observed_outer_ids != {outer}:
            raise ProtocolError("HARP v16 endpoint outcomes crossed outer targets.")
    observed_cases = {row.action.case_id for row in rows}
    cases = tuple(
        sorted(
            canonical_text(value, name="endpoint training case id")
            for value in (
                observed_cases if training_case_ids is None else tuple(training_case_ids)
            )
        )
    )
    observed_candidates = {
        row.action.candidate_source_id
        for row in rows
        if row.action.candidate_source_id is not None
    }
    candidates = tuple(
        sorted(
            canonical_text(value, name="endpoint candidate source")
            for value in (
                observed_candidates
                if candidate_source_ids is None
                else tuple(candidate_source_ids)
            )
        )
    )
    action_hashes = {row.action.action_hash for row in rows}
    if (
        len(action_hashes) != len(rows)
        or len(cases) < 2
        or len(cases) != len(set(cases))
        or not observed_cases.issubset(cases)
        or len(candidates) != len(set(candidates))
        or not observed_candidates.issubset(candidates)
        or outer in candidates
        or any(row.action.surface_role is not SurfaceRole.TARGET_TRAIN_SUPPORT for row in rows)
        or any(not row.has_class_local_components for row in rows)
        or (rows and len({row.normalization_hash for row in rows}) != 1)
        or (rows and next(iter({row.normalization_hash for row in rows})) is None)
    ):
        raise ProtocolError("HARP v16 endpoint support inventory is malformed.")
    if len(observed_cases) < 2:
        return NullSupportEndpointModel(
            outer_target_id=outer,
            training_case_ids=cases,
            candidate_source_ids=candidates,
            reason=(
                "NO_ACTIVE_SUPPORT_ACTION"
                if not rows
                else "INSUFFICIENT_ACTIVE_SUPPORT_CASES"
            ),
        )
    actions = tuple(row.action for row in rows)
    feature_map = fit_feature_map(
        actions,
        maximum_numeric_features=config.maximum_numeric_features,
        candidate_source_ids=candidates,
        fitted_case_ids=cases,
    )
    matrix = np.vstack([feature_map.transform(row) for row in actions])
    weights = np.asarray(case_balanced_weights(actions), dtype=np.float64)
    targets = {
        "gain": np.asarray([row.bacc_gain for row in rows], dtype=np.float64),
        "harm": np.asarray([float(row.harmed) for row in rows], dtype=np.float64),
        "brier": np.asarray([row.brier_delta for row in rows], dtype=np.float64),
        "log_loss": np.asarray([row.log_loss_delta for row in rows], dtype=np.float64),
    }
    heads = {
        endpoint: _fit_ridge_head(
            endpoint,
            matrix,
            target,
            weights,
            alpha=config.ridge_alpha,
        )
        for endpoint, target in targets.items()
    }
    return SupportEndpointModel(
        outer_target_id=outer,
        feature_map=feature_map,
        gain_head=heads["gain"],
        harm_head=heads["harm"],
        brier_head=heads["brier"],
        log_loss_head=heads["log_loss"],
        training_case_ids=cases,
        ridge_alpha=config.ridge_alpha,
    )


__all__ = (
    "NullSupportEndpointModel",
    "RidgeHead",
    "SupportEndpointModel",
    "fit_support_endpoint_model",
)
