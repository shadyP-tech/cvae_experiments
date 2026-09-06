"""Separate mean-effect feasibility from action utility; no compensating risk credit."""

POLICY_RULE = "FEASIBLE_POSITIVE_GAIN_THEN_ONE_MINUS_WINNER_HARM_GE_TAU_ELSE_EXACT_B"


def selection_contract():
    return dict(score_definition="EXACT_FLIP_AGGREGATED_PREDICTED_BACC_GAIN",
        predicted_gain_strictly_positive=True, predicted_brier_delta_max=.002,
        predicted_logloss_delta_max=.005, baseline_score=0.,
        score_is_safety_bound=False, candidate_harm_probability_estimated=False,
        proper_loss_screen_before_winner=True,
        candidate_tie_rule="MAX_PREDICTED_GAIN_THEN_LEXICAL_ARM_ID",
        policy_rule=POLICY_RULE, runnerup_after_gate_veto=False)
