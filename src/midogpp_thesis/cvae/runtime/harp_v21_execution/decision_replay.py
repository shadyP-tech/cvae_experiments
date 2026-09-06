"""Restore public numeric parameters and replay the complete label-free selector."""
from collections.abc import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.correction_mass_router_v21.hashing import canonical_hash


def restore_frozen_proposer(payload):
    """Load numeric JSON only; every advertised public component is revalidated."""
    from ...routing.correction_mass_router_v21.modeling import FittedFeatureTransform, PairwiseRanker, ProposalModel
    from ...routing.correction_mass_router_v21.outcome_model import ActionOutcomeModel
    from ...routing.correction_mass_router_v21.proposer import FittedProposer
    try:
        proposal = payload["proposal_model"]
        transform = proposal["transform"]
        ranker = proposal["ranker"]
        fitted_transform = FittedFeatureTransform(
            tuple(transform["feature_names"]), tuple(transform["means"]), tuple(transform["scales"]),
            tuple(transform["donor_ids"]), tuple(tuple(k) for k in transform["training_case_keys"]))
        fitted_ranker = PairwiseRanker(tuple(ranker["coefficients"]), ranker["ridge_alpha"],
            tuple(ranker["design_names"]), ranker["transform_hash"], ranker["unique_comparison_count"],
            tuple(tuple(k) for k in ranker["training_case_keys"]))
        fitted_proposal = ProposalModel(fitted_transform, fitted_ranker)
        fitted = FittedProposer(fitted_proposal, ActionOutcomeModel.from_payload(payload["action_model"]),
            tuple(tuple(k) for k in payload["training_case_keys"]), tuple(payload["stacking_receipts"]))
        for observed, restored in ((transform, fitted_transform), (ranker, fitted_ranker),
                                   (proposal, fitted_proposal), (payload, fitted)):
            if canonical_hash(observed) != canonical_hash(restored.public_payload()):
                raise ProtocolError("HARP v21 frozen proposer numeric replay identity drifted.")
        return fitted
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProtocolError("HARP v21 frozen proposer numeric projection is malformed.") from exc


def restore_target_menu(payload: Mapping, patch_features):
    from ...routing.correction_mass_router_v21.contracts import LabelFreeAction, LabelFreeCaseMenu, Direction, SurfaceRole
    try:
        actions = []
        for row in payload["actions"]:
            action = LabelFreeAction(SurfaceRole(row["surface_role"]), row["center_id"], row["case_id"],
                row["arm_id"], Direction(row["direction"]), row["donor_id"], tuple(row["feature_names"]),
                tuple(row["feature_values"]), tuple(row["sample_ids"]),
                tuple(row["baseline_probability_hex"]), tuple(row["action_probability_hex"]))
            if canonical_hash(action.public_payload()) != canonical_hash(row):
                raise ProtocolError("HARP v21 target candidate input identity drifted.")
            actions.append(action)
        menu = LabelFreeCaseMenu(SurfaceRole(payload["surface_role"]), payload["center_id"], payload["case_id"],
            tuple(payload["sample_ids"]), tuple(payload["baseline_probability_hex"]), tuple(actions), patch_features)
        if menu.surface_role is not SurfaceRole.TARGET_EVALUATION or canonical_hash(menu.public_payload()) != canonical_hash(payload):
            raise ProtocolError("HARP v21 target selector input escaped its sealed menu.")
        return menu
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProtocolError("HARP v21 target selector input projection is malformed.") from exc


def replay_selected_winner(menu, payload, *, proposer, config):
    """Recompute the screened argmax and its gate inputs from actual primitives."""
    from ...routing.correction_mass_router_v21.candidate_prediction import unthresholded_winner
    from ...routing.correction_mass_router_v21.winner_records import winner_features
    candidates = proposer.candidate_predictions(menu, config)
    winner = unthresholded_winner(candidates)
    if winner is None:
        if payload.get("winner_composite_hash") is not None:
            raise ProtocolError("HARP v21 recorded winner has no feasible positive-gain candidate.")
        return
    if (payload.get("winner_composite_hash") != winner.candidate.composite.composite_hash
        or payload.get("winner_arm_id") != winner.arm_id
        or not np.isclose(payload.get("winner_risk_adjusted_score", float("nan")),
                          winner.risk_adjusted_score, rtol=1e-10, atol=1e-12)):
        raise ProtocolError("HARP v21 recorded winner differs from frozen candidate replay.")
    transcript = payload.get("winner_gate_prediction_payload")
    if not isinstance(transcript, Mapping):
        raise ProtocolError("HARP v21 selected winner lacks gate replay inputs.")
    expected = dict(winner_features(menu, candidates))
    names = tuple(transcript.get("feature_names", ()))
    if transcript.get("calibration_available") is False and not names and not transcript.get("feature_values"):
        return
    if (names != tuple(sorted(expected))
        or not np.allclose(np.asarray(transcript.get("feature_values", ()), dtype=float),
                           [expected[name] for name in names], rtol=1e-10, atol=1e-12)):
        raise ProtocolError("HARP v21 sealed gate features differ from frozen winner replay.")
