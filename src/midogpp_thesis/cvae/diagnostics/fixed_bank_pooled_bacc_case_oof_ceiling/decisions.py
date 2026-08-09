"""B/G/R fold decisions and the complete 45-fold pre-evaluation seal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .case_partitions import CaseFold, CaseOOFPartition
from .core_contracts import SealedProbabilitySurface
from .core_hashing import canonical_hash, finite, require_sha256
from .pooled_posterior import PooledFoldPosterior
from .pooled_prior import PooledLocoPrior
from .scientific_constants import (
    BASELINE_ACTION_ID,
    DEFAULT_MINIMUM_GAIN,
    DEFAULT_TIE_TOLERANCE,
    EXPECTED_FOLD_COUNT,
    EXPECTED_FOLD_DECISION_COUNT,
    MIDOGPP_CENTERS,
    action_ids,
    routing_challengers,
)


@dataclass(frozen=True)
class DecisionConfig:
    minimum_gain: float = DEFAULT_MINIMUM_GAIN
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_gain", finite(self.minimum_gain, "minimum_gain"))
        object.__setattr__(self, "tie_tolerance", finite(self.tie_tolerance, "tie_tolerance"))
        if self.minimum_gain != 0.0:
            raise ProtocolError("The v2 route switch is frozen at a strict zero lower bound.")
        if self.tie_tolerance != DEFAULT_TIE_TOLERANCE:
            raise ProtocolError("The v2 lexicographic tie tolerance drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "minimum_gain": self.minimum_gain,
            "tie_tolerance": self.tie_tolerance,
            "thresholds_fixed_prelabel": True,
            "posthoc_tuning_used": False,
            "strict_positive_lcb_required": True,
        }


@dataclass(frozen=True, order=True)
class FoldDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    evaluation_case_ids: tuple[str, ...]
    baseline_action_id: str
    global_action_id: str
    selected_challenger_action_id: str
    routed_action_id: str
    route_tier: str
    selected_posterior_mean_gain_vs_g: float
    selected_lower_confidence_bound: float
    global_lower_confidence_bound_vs_b: float
    global_prior_hash: str
    posterior_hash: str
    decision_hash: str = field(init=False, compare=True)
    evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Fold decision uses an unknown target center.")
        if (
            isinstance(self.fold_ordinal, bool)
            or not isinstance(self.fold_ordinal, int)
            or self.fold_ordinal < 0
            or self.fold_ordinal >= EXPECTED_FOLD_COUNT
        ):
            raise ProtocolError("Fold decision ordinal violates the five-fold lock.")
        for name in ("fold_hash", "global_prior_hash", "posterior_hash"):
            require_sha256(getattr(self, name), name)
        evaluation = tuple(sorted(self.evaluation_case_ids))
        if not evaluation or self.baseline_action_id != BASELINE_ACTION_ID:
            raise ProtocolError("Fold decision lacks its baseline/evaluation contract.")
        if self.global_action_id not in action_ids(self.target_center):
            raise ProtocolError("Fold decision G_H is outside the fixed bank.")
        if self.selected_challenger_action_id not in routing_challengers(
            self.target_center, self.global_action_id
        ):
            raise ProtocolError("Fold decision selected G_H itself or an invalid challenger.")
        if self.routed_action_id not in (self.global_action_id, self.selected_challenger_action_id):
            raise ProtocolError("Fold decision must retain G_H or use its selected challenger.")
        if self.target_center in (
            self.global_action_id,
            self.selected_challenger_action_id,
            self.routed_action_id,
        ):
            raise ProtocolError("Fold decision leaked the held-out target expert.")
        expected_tier = (
            "R"
            if self.routed_action_id != self.global_action_id
            else "G"
            if self.global_action_id != BASELINE_ACTION_ID
            else "B"
        )
        if self.route_tier != expected_tier or self.evaluation_labels_used is not False:
            raise ProtocolError("Fold decision tier/label boundary drifted.")
        for name in (
            "selected_posterior_mean_gain_vs_g",
            "selected_lower_confidence_bound",
            "global_lower_confidence_bound_vs_b",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if (self.routed_action_id != self.global_action_id) != (
            self.selected_lower_confidence_bound > 0.0
        ):
            raise ProtocolError("Fold route does not match the strict-positive LCB gate.")
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def fold_id(self) -> str:
        return f"H{self.target_center}::fold{self.fold_ordinal}"

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_fold_decision_v2",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "baseline_action_id": self.baseline_action_id,
            "global_action_id": self.global_action_id,
            "selected_challenger_action_id": self.selected_challenger_action_id,
            "routed_action_id": self.routed_action_id,
            "route_tier": self.route_tier,
            "selected_posterior_mean_gain_vs_g": self.selected_posterior_mean_gain_vs_g,
            "selected_lower_confidence_bound": self.selected_lower_confidence_bound,
            "global_lower_confidence_bound_vs_b": self.global_lower_confidence_bound_vs_b,
            "global_prior_hash": self.global_prior_hash,
            "posterior_hash": self.posterior_hash,
            "evaluation_labels_used": False,
            "per_case_bacc_used": False,
            "strict_positive_lcb_required": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "fold_id": self.fold_id, "decision_hash": self.decision_hash}


@dataclass(frozen=True)
class DecisionSeal:
    decisions: tuple[FoldDecision, ...]
    partition_hash: str
    probability_surface_hash: str
    decision_seal_hash: str
    all_fold_decisions_sealed_before_evaluation_labels: bool = True
    fold_decision_count: int = EXPECTED_FOLD_DECISION_COUNT

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        for name in ("partition_hash", "probability_surface_hash", "decision_seal_hash"):
            require_sha256(getattr(self, name), name)
        expected_keys = tuple(
            (center, fold)
            for center in MIDOGPP_CENTERS
            for fold in range(EXPECTED_FOLD_COUNT)
        )
        if (
            tuple((value.target_center, value.fold_ordinal) for value in decisions)
            != expected_keys
            or self.fold_decision_count != EXPECTED_FOLD_DECISION_COUNT
            or self.all_fold_decisions_sealed_before_evaluation_labels is not True
        ):
            raise ProtocolError("Decision seal must contain all 45 decisions before labels.")
        if canonical_hash(self._unhashed()) != self.decision_seal_hash:
            raise ProtocolError("All-fold decision seal hash drifted.")
        object.__setattr__(self, "decisions", decisions)

    def decision(self, target_center: str, fold_ordinal: int) -> FoldDecision:
        for value in self.decisions:
            if value.target_center == str(target_center) and value.fold_ordinal == int(fold_ordinal):
                return value
        raise KeyError((target_center, fold_ordinal))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_all_fold_decision_seal_v2",
            "partition_hash": self.partition_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "fold_decision_count": self.fold_decision_count,
            "decisions": [value.to_payload() for value in self.decisions],
            "all_fold_decisions_sealed_before_evaluation_labels": True,
            "target_evaluation_labels_readable_before_seal": False,
            "target_expert_used": False,
            "policy_update_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_seal_hash": self.decision_seal_hash}


def make_fold_decision(
    fold: CaseFold,
    posterior: PooledFoldPosterior,
    global_prior: PooledLocoPrior,
    *,
    config: DecisionConfig = DecisionConfig(),
) -> FoldDecision:
    if (
        posterior.target_center != fold.target_center
        or posterior.fold_ordinal != fold.fold_ordinal
        or posterior.fold_hash != fold.fold_hash
        or posterior.global_prior_hash != global_prior.prior_hash
        or global_prior.target_center != fold.target_center
        or posterior.global_action_id != global_prior.global_action_id
    ):
        raise ProtocolError("Fold decision received mismatched posterior/prior inputs.")
    maximum = max(value.lower_confidence_bound for value in posterior.estimates)
    eligible = tuple(
        value
        for value in posterior.estimates
        if maximum - value.lower_confidence_bound <= config.tie_tolerance
    )
    order = routing_challengers(fold.target_center, global_prior.global_action_id)
    best = min(eligible, key=lambda value: order.index(value.action_id))
    local_switch = best.lower_confidence_bound > config.minimum_gain
    routed = best.action_id if local_switch else global_prior.global_action_id
    global_estimate = global_prior.estimate(global_prior.global_action_id)
    global_lcb = 0.0 if global_estimate is None else global_estimate.lower_confidence_bound
    return FoldDecision(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        fold_hash=fold.fold_hash,
        evaluation_case_ids=fold.evaluation_case_ids,
        baseline_action_id=BASELINE_ACTION_ID,
        global_action_id=global_prior.global_action_id,
        selected_challenger_action_id=best.action_id,
        routed_action_id=routed,
        route_tier="R" if local_switch else "G" if routed != BASELINE_ACTION_ID else "B",
        selected_posterior_mean_gain_vs_g=best.posterior_mean,
        selected_lower_confidence_bound=best.lower_confidence_bound,
        global_lower_confidence_bound_vs_b=global_lcb,
        global_prior_hash=global_prior.prior_hash,
        posterior_hash=posterior.posterior_hash,
    )


def seal_fold_decisions(
    decisions: Sequence[FoldDecision],
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
) -> DecisionSeal:
    canonical = tuple(
        sorted(
            tuple(decisions),
            key=lambda value: (MIDOGPP_CENTERS.index(value.target_center), value.fold_ordinal),
        )
    )
    if len(canonical) != EXPECTED_FOLD_DECISION_COUNT:
        raise ProtocolError("All 45 fold decisions are required for the evaluation seal.")
    for decision, fold in zip(canonical, partition.folds):
        if (
            decision.target_center != fold.target_center
            or decision.fold_ordinal != fold.fold_ordinal
            or decision.fold_hash != fold.fold_hash
            or decision.evaluation_case_ids != fold.evaluation_case_ids
        ):
            raise ProtocolError("Fold decision drifted from the locked partition.")
    identity_keys = {
        (identity.target_center, identity.case_id, identity.sample_id)
        for identity in partition.identities
    }
    probability_keys = {
        (identity.target_center, identity.case_id, identity.sample_id)
        for identity in probabilities.identities
    }
    if identity_keys != probability_keys:
        raise ProtocolError("Decision partition and probability rows are not closed-world aligned.")
    payload = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_decision_seal_v2",
        "partition_hash": partition.partition_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "fold_decision_count": EXPECTED_FOLD_DECISION_COUNT,
        "decisions": [value.to_payload() for value in canonical],
        "all_fold_decisions_sealed_before_evaluation_labels": True,
        "target_evaluation_labels_readable_before_seal": False,
        "target_expert_used": False,
        "policy_update_authorized": False,
    }
    return DecisionSeal(
        decisions=canonical,
        partition_hash=partition.partition_hash,
        probability_surface_hash=probabilities.surface_hash,
        decision_seal_hash=canonical_hash(payload),
    )


__all__ = (
    "DecisionConfig",
    "DecisionSeal",
    "FoldDecision",
    "make_fold_decision",
    "seal_fold_decisions",
)
