"""Pre-truth unthresholded winner seals, including failures and empty menus."""
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .candidate_prediction import unthresholded_winner
from .hashing import canonical_hash


def winner_features(menu, candidates):
    """Low-dimensional gate context, never a source class-support normalizer."""
    from .features import exact_composite_features
    winner = unthresholded_winner(candidates)
    if winner is None:
        return ()
    available = sorted((row for row in candidates if row.eligible_for_winner),
                       key=lambda row: (-row.risk_adjusted_score, row.arm_id))
    prediction = winner.prediction
    exact = exact_composite_features(menu, winner.candidate.composite)
    # Exact executed features are deterministic, and their schema is frozen by
    # the candidate feature module. Keep the gate small rather than refit all
    # candidate features on the much smaller selected population.
    selected_names = ("sample_count", "lambda", "hard_change_fraction",
        *(f"{direction}_{name}" for direction in ("d01", "d10") for name in
          ("flip_count", "flip_fraction", "baseline_margin_q10", "action_margin_q10",
           "selected_donor_disagreement_on_flips")))
    features = {
        "risk_adjusted_score": winner.risk_adjusted_score,
        "candidate_safe_probability": prediction.safe_positive_probability,
        "candidate_harm_probability": prediction.predicted_harm,
        "predicted_brier_delta": prediction.predicted_brier_delta,
        "predicted_logloss_delta": prediction.predicted_logloss_delta,
        "candidate_count": float(len(available)),
        "winner_runnerup_gap": winner.risk_adjusted_score - available[1].risk_adjusted_score if len(available) > 1 else 0.0,
        "singleton_menu": float(len(available) == 1),
        **{f"family::{name}": float(winner.candidate.kind.value == name)
           for name in ("U_FULL", "D01_ONLY", "D10_ONLY", "BOTH")},
        **{f"executed::{name}": exact[name] for name in selected_names},
    }
    return tuple(sorted(features.items()))


@dataclass(frozen=True, slots=True)
class SealedWinner:
    menu: object
    candidates: tuple
    training_case_keys: tuple
    proposer_hash: str
    fold: int
    winner: object = field(init=False)
    features: tuple = field(init=False)
    winner_seal_hash: str = field(init=False)

    def __post_init__(self):
        if (self.menu.center_id, self.menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v20 winner case entered its proposer fit.")
        object.__setattr__(self, "winner", unthresholded_winner(self.candidates))
        object.__setattr__(self, "features", winner_features(self.menu, self.candidates))
        object.__setattr__(self, "winner_seal_hash", canonical_hash(self._payload()))

    @property
    def case_key(self):
        return self.menu.center_id, self.menu.case_id

    def _payload(self):
        return {"case_key": self.case_key, "menu_hash": self.menu.menu_hash,
            "fold": self.fold, "training_case_keys": self.training_case_keys,
            "proposer_hash": self.proposer_hash,
            "candidate_prediction_hashes": tuple(canonical_hash(row.public_payload()) for row in self.candidates),
            "winner_composite_hash": None if self.winner is None else self.winner.candidate.composite.composite_hash,
            "winner_arm_id": None if self.winner is None else self.winner.arm_id,
            "winner_risk_adjusted_score": None if self.winner is None else self.winner.risk_adjusted_score,
            "features": self.features, "winner_selected_before_truth": True,
            "threshold_or_outcome_filter_applied": False}

    def public_payload(self):
        return {**self._payload(), "winner_seal_hash": self.winner_seal_hash}
