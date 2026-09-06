"""Pre-truth unthresholded winner seals, including failures and empty menus."""
from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .candidate_prediction import unthresholded_winner
from .hashing import canonical_hash


def winner_features(menu, candidates):
    """Six prespecified case summaries for the frozen selected action."""
    from .features import exact_composite_features
    winner = unthresholded_winner(candidates)
    if winner is None:
        return ()
    prediction = winner.prediction
    exact = exact_composite_features(menu, winner.candidate.composite)
    features = {
        "predicted_gain": prediction.predicted_gain,
        "predicted_brier_delta": prediction.predicted_brier_delta,
        "predicted_logloss_delta": prediction.predicted_logloss_delta,
        "hard_change_fraction": exact["hard_change_fraction"],
        "d01_disagreement_on_flips": exact["d01_selected_donor_disagreement_on_flips"],
        "d10_disagreement_on_flips": exact["d10_selected_donor_disagreement_on_flips"],
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
            raise ProtocolError("HARP v21 winner case entered its proposer fit.")
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
            "threshold_or_outcome_filter_applied": False,
            "label_free_positive_gain_and_proper_loss_screen_applied": True}

    def public_payload(self):
        return {**self._payload(), "winner_seal_hash": self.winner_seal_hash}
