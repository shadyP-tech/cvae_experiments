"""Candidate/outcome and complete-winner joins after a shared prediction seal."""
from .candidate_prediction import choose_candidate, unthresholded_winner
from .hashing import canonical_hash
from .features import exact_composite_features


def detailed_prediction_joins(held, outcomes, thresholds, stage, pretruth_hash):
    candidate_rows, winner_rows = [], []
    for row in held:
        winner = unthresholded_winner(row.candidates)
        gate = row.winner_prediction
        decisions = tuple({"threshold": float(threshold), "selected_arm_id": composite.arm_id,
            "gate_score": score, "fallback_reason": reason, "route_selected": composite.route_selected}
            for threshold in thresholds
            for composite, score, reason in (choose_candidate(row.menu, row.candidates, threshold,
                                                             winner_prediction=gate),))
        common = {"stage": stage, "fold": row.fold,
            "center_id": row.menu.center_id, "case_id": row.menu.case_id,
            "menu_hash": row.menu.menu_hash, "training_case_keys": row.training_case_keys,
            "complete_model_hash": row.model_hash,
            "evidence_variant": None if row.patch_control is None else row.patch_control.evidence_variant,
            "held_prediction_seal_hash": row.prediction_seal_hash,
            "pretruth_frontier_seal_hash": pretruth_hash, "raw_labels_persisted": False}
        for candidate in row.candidates:
            composite = candidate.candidate.composite
            outcome = None if composite is None else outcomes[composite.composite_hash]
            payload = {**common, **candidate.public_payload(),
                "outcome": None if outcome is None else outcome.public_payload(),
                "exact_executed_features": None if composite is None else exact_composite_features(row.menu, composite),
                "probability_vector_hash": None if composite is None else canonical_hash(composite.probability_hex),
                "composite_recipe": None if composite is None else {
                    "kind": composite.kind.value, "k": composite.k, "mixing_lambda": composite.mixing_lambda,
                    "d01_action_ids": composite.d01_action_ids, "d10_action_ids": composite.d10_action_ids,
                    "donor_ids": composite.donor_ids},
                "is_unthresholded_winner": candidate is winner}
            candidate_rows.append({**payload, "join_hash": canonical_hash(payload)})
        outcome = None if winner is None else outcomes[winner.candidate.composite.composite_hash]
        payload = {**common, "winner_arm_id": None if winner is None else winner.arm_id,
            "winner_risk_adjusted_score": None if winner is None else winner.risk_adjusted_score,
            "raw_candidate_harm_probability": None,
            "candidate_harm_probability_estimated": False,
            "winner_gate_prediction": None if gate is None else gate.public_payload(),
            "outcome": None if outcome is None else outcome.public_payload(),
            "positive_score_routing_eligible": winner is not None and winner.risk_adjusted_score > 0,
            "threshold_decisions": decisions,
            "threshold_decisions_are_enabled_policy_diagnostics": True,
            "actual_nested_policy_admission_applied": False,
            "winner_satisfies_positive_gain_and_proper_loss_screens": winner is not None, "runnerup_after_veto": False}
        winner_rows.append({**payload, "diagnostic_hash": canonical_hash(payload)})
    return tuple(candidate_rows), tuple(winner_rows)
