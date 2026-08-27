"""Antisymmetric pairwise-ranker contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .shared import P_ACTION_ID, ProtocolError, _finite_tuple, _text, canonical_sha256
from .utility import NormalizedUtility


@dataclass(frozen=True, slots=True)
class CandidatePoolReceipt:
    """Exact C-minus-H expert inventory bound to one immutable fixed bank."""

    outer_target_center: str
    all_center_ids: tuple[str, ...]
    candidate_center_ids: tuple[str, ...]
    expert_inventory: tuple[tuple[str, str], ...]
    bank_lock_hash: str
    source_surface_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = _text(self.outer_target_center, role="candidate-pool outer target H")
        raw_all = tuple(_text(value, role="bank center") for value in self.all_center_ids)
        raw_candidates = tuple(
            _text(value, role="candidate center") for value in self.candidate_center_ids
        )
        all_centers = tuple(sorted(raw_all))
        candidate_centers = tuple(sorted(raw_candidates))
        inventory = tuple(
            sorted(
                (
                    _text(expert_id, role="expert id"),
                    _text(center_id, role="expert source center"),
                )
                for expert_id, center_id in self.expert_inventory
            )
        )
        expected = tuple(center for center in all_centers if center != h)
        if (
            h not in all_centers
            or len(set(raw_all)) != len(raw_all)
            or len(set(raw_candidates)) != len(raw_candidates)
            or candidate_centers != expected
            or len({expert for expert, _ in inventory}) != len(inventory)
            or {center for _, center in inventory} != set(candidate_centers)
            or len(inventory) != len(candidate_centers)
        ):
            raise ProtocolError("Candidate-pool receipt is not the exact C-minus-H expert inventory.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "all_center_ids", all_centers)
        object.__setattr__(self, "candidate_center_ids", candidate_centers)
        object.__setattr__(self, "expert_inventory", inventory)
        object.__setattr__(self, "bank_lock_hash", _text(self.bank_lock_hash, role="bank lock hash"))
        object.__setattr__(
            self,
            "source_surface_receipt_hash",
            _text(self.source_surface_receipt_hash, role="source surface receipt hash"),
        )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "candidate_pool_C_minus_H_receipt_v1",
                    "H": h,
                    "all_centers": all_centers,
                    "candidate_centers": candidate_centers,
                    "expert_inventory": inventory,
                    "bank_lock_hash": self.bank_lock_hash,
                    "source_surface_receipt_hash": self.source_surface_receipt_hash,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BaccRankingPolicy:
    """Frozen action-invariant ranking estimand."""

    metric: str = "EXPECTED_BACC_GAIN"
    denominator_policy: str = "ACTION_INVARIANT_EXPECTED_CLASS_TOTALS"
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.metric != "EXPECTED_BACC_GAIN"
            or self.denominator_policy != "ACTION_INVARIANT_EXPECTED_CLASS_TOTALS"
        ):
            raise ProtocolError("BACC ranking policy drifted.")
        object.__setattr__(
            self,
            "policy_hash",
            canonical_sha256(
                {"metric": self.metric, "denominator_policy": self.denominator_policy}
            ),
        )

@dataclass(frozen=True, slots=True)
class ActionUtilityObservation:
    """Source-only case/action value used to construct pairwise contrasts."""

    center_id: str
    case_id: str
    action_id: str
    family: str
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    response: NormalizedUtility
    source_scope_receipt_hash: str
    candidate_pool_receipt_hash: str
    opportunity_case_receipt_hash: str

    def __post_init__(self) -> None:
        action_id = _text(self.action_id, role="pairwise action id")
        if action_id == P_ACTION_ID:
            raise ProtocolError("P is implicit and fixed to zero in pairwise training.")
        names = tuple(_text(name, role="pairwise feature name") for name in self.feature_names)
        values = _finite_tuple(self.feature_values, role="pairwise feature values")
        if (
            len(names) != len(values)
            or len(set(names)) != len(names)
            or not isinstance(self.response, NormalizedUtility)
        ):
            raise ProtocolError("Pairwise action observation is invalid.")
        object.__setattr__(self, "center_id", _text(self.center_id, role="pairwise source center"))
        object.__setattr__(self, "case_id", _text(self.case_id, role="pairwise source case"))
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "family", _text(self.family, role="pairwise action family"))
        object.__setattr__(self, "direction", _text(self.direction, role="pairwise action direction"))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="pairwise source scope receipt hash"),
        )
        object.__setattr__(
            self,
            "opportunity_case_receipt_hash",
            _text(self.opportunity_case_receipt_hash, role="opportunity-case receipt hash"),
        )
        object.__setattr__(
            self,
            "candidate_pool_receipt_hash",
            _text(self.candidate_pool_receipt_hash, role="candidate pool receipt hash"),
        )
@dataclass(frozen=True, slots=True)
class PairwiseRankerModel:
    """Antisymmetric latent-score ridge model with P fixed to zero."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    action_schema: tuple[tuple[str, str, str], ...]
    candidate_action_ids: tuple[str, ...]
    design_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    selected_alpha: float
    alpha_grid: tuple[float, ...]
    delete_center_losses: tuple[tuple[float, str, float], ...]
    alpha_selection_summary: tuple[tuple[float, float, float], ...]
    training_center_ids: tuple[str, ...]
    training_case_count: int
    training_contrast_count: int
    source_scope_receipt_hash: str
    candidate_pool_receipt_hash: str
    opportunity_surface_receipt_hash: str
    bacc_ranking_policy_hash: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        means = tuple(float(value) for value in self.feature_mean)
        scales = tuple(float(value) for value in self.feature_scale)
        schema = tuple(tuple(str(value) for value in row) for row in self.action_schema)
        candidate_action_ids = tuple(sorted(str(value) for value in self.candidate_action_ids))
        design_names = tuple(self.design_names)
        coefficients = tuple(float(value) for value in self.coefficients)
        grid = tuple(float(value) for value in self.alpha_grid)
        losses = tuple(
            (float(alpha), str(center), float(loss))
            for alpha, center, loss in self.delete_center_losses
        )
        summaries = tuple(
            (float(alpha), float(worst), float(mean_loss))
            for alpha, worst, mean_loss in self.alpha_selection_summary
        )
        centers = tuple(self.training_center_ids)
        if (
            not names
            or len(names) != len(means)
            or len(names) != len(scales)
            or any(value <= 0.0 or not math.isfinite(value) for value in scales)
            or len(schema) == 0
            or len(set(candidate_action_ids)) != len(candidate_action_ids)
            or not set(row[0] for row in schema).issubset(candidate_action_ids)
            or any(len(row) != 3 or row[0] == P_ACTION_ID for row in schema)
            or len({row[0] for row in schema}) != len(schema)
            or len(design_names) != len(coefficients)
            or not coefficients
            or not all(math.isfinite(value) for value in coefficients)
            or self.selected_alpha not in grid
            or tuple(alpha for alpha, _, _ in summaries) != grid
            or not all(math.isfinite(loss) and loss >= 0.0 for _, _, loss in losses)
            or not all(
                math.isfinite(worst)
                and math.isfinite(mean_loss)
                and worst >= mean_loss >= 0.0
                for _, worst, mean_loss in summaries
            )
            or {alpha for alpha, _, _ in losses} != set(grid)
            or len({center for _, center, _ in losses}) < 2
            or len(centers) < 3
            or self.training_case_count <= 0
            or self.training_contrast_count <= 0
        ):
            raise ProtocolError("Pairwise ranker model contract is invalid.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", means)
        object.__setattr__(self, "feature_scale", scales)
        object.__setattr__(self, "action_schema", schema)
        object.__setattr__(self, "candidate_action_ids", candidate_action_ids)
        object.__setattr__(self, "design_names", design_names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "selected_alpha", float(self.selected_alpha))
        object.__setattr__(self, "alpha_grid", grid)
        object.__setattr__(self, "delete_center_losses", losses)
        object.__setattr__(self, "alpha_selection_summary", summaries)
        object.__setattr__(self, "training_center_ids", centers)
        object.__setattr__(
            self,
            "source_scope_receipt_hash",
            _text(self.source_scope_receipt_hash, role="pairwise source scope receipt hash"),
        )
        object.__setattr__(
            self,
            "opportunity_surface_receipt_hash",
            _text(self.opportunity_surface_receipt_hash, role="opportunity surface receipt hash"),
        )
        object.__setattr__(
            self,
            "bacc_ranking_policy_hash",
            _text(self.bacc_ranking_policy_hash, role="BACC ranking policy hash"),
        )
        object.__setattr__(
            self,
            "candidate_pool_receipt_hash",
            _text(self.candidate_pool_receipt_hash, role="candidate pool receipt hash"),
        )
        object.__setattr__(
            self,
            "model_hash",
            canonical_sha256(
                {
                    "schema": "center_jackknife_pairwise_ranker_model_v2",
                    "P_anchor": 0.0,
                    "feature_names": names,
                    "feature_mean": means,
                    "feature_scale": scales,
                    "action_schema": schema,
                    "candidate_action_ids": candidate_action_ids,
                    "design_names": design_names,
                    "coefficients": coefficients,
                    "selected_alpha": self.selected_alpha,
                    "alpha_grid": grid,
                    "delete_center_losses": losses,
                    "alpha_selection_summary": summaries,
                    "training_centers": centers,
                    "case_count": self.training_case_count,
                    "contrast_count": self.training_contrast_count,
                    "source_scope_receipt_hash": self.source_scope_receipt_hash,
                    "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
                    "opportunity_surface_receipt_hash": self.opportunity_surface_receipt_hash,
                    "bacc_ranking_policy_hash": self.bacc_ranking_policy_hash,
                    "training_values_persisted": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ActionQuery:
    """Label-free action descriptor for pairwise inference."""

    action_id: str
    family: str
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        action_id = _text(self.action_id, role="query action id")
        names = tuple(_text(name, role="query feature name") for name in self.feature_names)
        values = _finite_tuple(self.feature_values, role="query feature values")
        if len(names) != len(values) or len(set(names)) != len(names):
            raise ProtocolError("Pairwise query feature schema is invalid.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "family", _text(self.family, role="query family"))
        object.__setattr__(self, "direction", _text(self.direction, role="query direction"))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)

    @classmethod
    def p_anchor(cls, feature_names: Sequence[str]) -> "ActionQuery":
        names = tuple(str(name) for name in feature_names)
        return cls(P_ACTION_ID, "P", "P", names, tuple(0.0 for _ in names))


@dataclass(frozen=True, slots=True)
class PairwisePrediction:
    """Ordered label-free contrast; reversing inputs negates the mean exactly."""

    left_action_id: str
    right_action_id: str
    mean_contrast: float
    model_hash: str

    def __post_init__(self) -> None:
        if self.left_action_id == self.right_action_id or not math.isfinite(float(self.mean_contrast)):
            raise ProtocolError("Pairwise prediction requires two distinct actions and a finite mean.")
        object.__setattr__(self, "left_action_id", _text(self.left_action_id, role="left action"))
        object.__setattr__(self, "right_action_id", _text(self.right_action_id, role="right action"))
        object.__setattr__(self, "mean_contrast", float(self.mean_contrast))
        object.__setattr__(self, "model_hash", _text(self.model_hash, role="pairwise model hash"))
