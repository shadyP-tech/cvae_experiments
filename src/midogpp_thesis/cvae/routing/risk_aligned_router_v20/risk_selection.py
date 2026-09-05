"""Risk-aligned action utility; penalties are chosen only inside source CV."""
from dataclasses import replace
from .hashing import canonical_hash

RISK_PENALTY_WEIGHTS = (.05, 1.0, .25)

def apply_risk_penalty(prediction, scale):
    h,b,l = RISK_PENALTY_WEIGHTS
    score = prediction.predicted_gain - scale * (
        h*(prediction.predicted_harm-.25) + b*(prediction.predicted_brier_delta-.002)
        + l*(prediction.predicted_logloss_delta-.005))
    return replace(prediction, risk_adjusted_score=float(score), approximate_gain_lower_score=float(score))


def selection_contract(scale):
    return dict(risk_penalty_scale=scale, risk_penalty_weights=RISK_PENALTY_WEIGHTS,
        score_definition='GAIN_MINUS_WEIGHTED_HARM_BRIER_LOGLOSS_EXCESS',
        baseline_score=0., score_is_safety_bound=False, risk_applied_before_winner_selection=True)
