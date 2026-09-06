"""Immutable candidate estimates and a single, label-free winner decision."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .composition import build_baseline_composite
from .contracts import LabelFreeCaseMenu, SoftTopKComposite
from .hashing import canonical_hash
from .splitting import CaseKey

POLICY_ARM_ID = "SAFE_WINNER_ACTION_POLICY"


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate: object
    prediction: object | None
    hard_prediction_changed: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_prediction_changed", bool(
            self.candidate.composite is not None and self.candidate.composite.prediction_changed))
        if self.prediction is not None and (self.candidate.composite is None
            or self.prediction.composite_hash != self.candidate.composite.composite_hash):
            raise ProtocolError("HARP v21 outcome estimates do not bind their actual composite.")

    @property
    def arm_id(self) -> str:
        return self.candidate.arm_id

    @property
    def eligible_for_winner(self) -> bool:
        return bool(self.arm_id != "B" and self.candidate.eligible
                    and self.candidate.duplicate_of is None
                    and self.hard_prediction_changed and self.prediction is not None
                    and self.prediction.predicted_gain > 0.0
                    and self.prediction.predicted_brier_delta <= .002
                    and self.prediction.predicted_logloss_delta <= .005)

    @property
    def screened(self) -> bool:
        """Compatibility spelling: structural eligibility plus positive benefit."""
        return self.eligible_for_winner and self.risk_adjusted_score > 0.0

    @property
    def risk_adjusted_score(self) -> float:
        return 0.0 if self.prediction is None else float(self.prediction.predicted_gain)

    @property
    def route_score(self) -> float:
        """Candidate diagnostic score; the actual policy uses its winner gate."""
        return float(self.eligible_for_winner)

    def public_payload(self) -> dict[str, object]:
        return {"arm_id": self.arm_id, "eligible": self.candidate.eligible,
                "ineligible_reason": self.candidate.ineligible_reason,
                "duplicate_of": self.candidate.duplicate_of,
                "composite_hash": None if self.candidate.composite is None else self.candidate.composite.composite_hash,
                "prediction": None if self.prediction is None else self.prediction.public_payload(),
                "hard_prediction_changed": self.hard_prediction_changed,
                "eligible_for_winner": self.eligible_for_winner,
                "positive_risk_adjusted_gain": self.screened,
                "individual_predicted_proper_loss_screen_used": True,
                "candidate_harm_probability_estimated": False}


def unthresholded_winner(candidates: Sequence[CandidatePrediction]) -> CandidatePrediction | None:
    eligible = tuple(row for row in candidates if row.eligible_for_winner)
    return min(eligible, key=lambda row: (-row.risk_adjusted_score, row.arm_id)) if eligible else None


@dataclass(frozen=True, slots=True)
class HeldCandidatePrediction:
    fold: int
    menu: LabelFreeCaseMenu
    candidates: tuple[CandidatePrediction, ...]
    training_case_keys: tuple[CaseKey, ...]
    model_hash: str
    winner_prediction: object | None = None
    patch_control: object | None = None
    prediction_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.menu.center_id, self.menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v21 held candidate entered complete learner fitting.")
        winner = unthresholded_winner(self.candidates)
        if self.winner_prediction is not None and (winner is None
            or self.winner_prediction.composite_hash != winner.candidate.composite.composite_hash):
            raise ProtocolError("HARP v21 gate prediction does not bind the selected winner.")
        object.__setattr__(self, "prediction_seal_hash", canonical_hash({
            "schema_version": "harp_v21_held_complete_prediction_seal",
            "fold": self.fold, "menu_hash": self.menu.menu_hash,
            "training_case_keys": self.training_case_keys, "model_hash": self.model_hash,
            "candidates": tuple(row.public_payload() for row in self.candidates),
            "winner_prediction": None if self.winner_prediction is None else self.winner_prediction.public_payload(),
            "patch_control_prediction_hash": None if self.patch_control is None else self.patch_control.prediction_hash,
            "held_truth_joined": False}))


def choose_candidate(menu: LabelFreeCaseMenu, candidates: Sequence[CandidatePrediction],
                     threshold: float, *, enabled: bool = True, winner_prediction: object | None = None
                     ) -> tuple[SoftTopKComposite, float, str | None]:
    """Select exactly one winner, then gate it. A veto never tries runner-up."""
    if not enabled:
        return build_baseline_composite(menu), 0.0, "NO_SAFE_INNER_OOF_POLICY"
    winner = unthresholded_winner(candidates)
    if winner is None:
        return build_baseline_composite(menu), 0.0, "NO_FEASIBLE_POSITIVE_GAIN_CANDIDATE"
    if winner.risk_adjusted_score <= 0.0:
        return build_baseline_composite(menu), 0.0, "NONPOSITIVE_PREDICTED_GAIN"
    if winner_prediction is None:
        return build_baseline_composite(menu), 0.0, "MISSING_COMPLETE_WINNER_GATE"
    if winner_prediction.composite_hash != winner.candidate.composite.composite_hash:
        raise ProtocolError("HARP v21 gate is bound to another candidate winner.")
    if not winner_prediction.calibration_available:
        return build_baseline_composite(menu), 0.0, "WINNER_GATE_UNAVAILABLE"
    score = float(winner_prediction.route_score)
    if score < threshold:
        return build_baseline_composite(menu), score, "WINNER_GATE_BELOW_THRESHOLD"
    return winner.candidate.composite, score, None
