"""Inventory-backed HARP v12 admission and whole-policy OOF replay."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.policy_calibrated_residual_router_v12 import (
    replay_policy_decisions,
)
from .contracts import ArtifactValue
from .model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    admission_manifest,
    build_source_only_admission,
)
from .production_validation import require_sha256, require_state
from .source_development import SourceDevelopmentState


def _policy_oof_replay(
    fitted: RouterFitState,
    admitted: RouterAdmissionState,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    numeric: list[tuple[float, ...]] = []
    nested_numeric: list[tuple[float, ...]] = []
    for bundle in fitted.bundles:
        policy = admitted.for_outer(bundle.outer_target_id)
        universe = policy.source_outcome_universe
        inventory = policy.source_oof_inventory
        if (
            policy.source_outcome_universe_hash != universe.universe_hash
            or policy.source_oof_inventory_hash != inventory.inventory_hash
            or tuple(
                context.prediction.prediction_hash
                for context in inventory.contexts
            )
            != tuple(
                prediction.prediction_hash
                for prediction in bundle.lodo.oof_predictions
            )
        ):
            raise ProtocolError("HARP v12 admission/replay inventory identity drifted.")
        deployed = replay_policy_decisions(
            inventory,
            acceptance_threshold=policy.calibration.acceptance_threshold,
            rank_margin_threshold=policy.calibration.rank_margin_threshold,
            policy_enabled=policy.policy_enabled,
        )
        heldout_thresholds = {
            center: (threshold, margin)
            for center, threshold, margin in policy.calibration.heldout_thresholds
        }
        folds = {
            fold.heldout_center_id: fold
            for fold in bundle.lodo.nested_policy_folds
        }
        nested_contexts = {}
        nested_decisions = {}
        for center in sorted(folds):
            try:
                threshold, margin = heldout_thresholds[center]
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v12 nested replay lacks a held-source threshold."
                ) from exc
            heldout_inventory = universe.bind_predictions(
                folds[center].heldout_predictions
            )
            for context in heldout_inventory.contexts:
                if context.key in nested_contexts:
                    raise ProtocolError("HARP v12 nested OOF contexts overlap.")
                nested_contexts[context.key] = context
            decisions = replay_policy_decisions(
                heldout_inventory,
                acceptance_threshold=threshold,
                rank_margin_threshold=margin,
            )
            if set(decisions) & set(nested_decisions):
                raise ProtocolError("HARP v12 nested OOF decisions overlap.")
            nested_decisions.update(decisions)
        expected_keys = {context.key for context in inventory.contexts}
        if (
            set(deployed) != expected_keys
            or set(nested_contexts) != expected_keys
            or set(nested_decisions) != expected_keys
        ):
            raise ProtocolError("HARP v12 OOF replay case inventory is incomplete.")

        for context in inventory.contexts:
            prediction = context.prediction
            decision = deployed[context.key]
            nested_context = nested_contexts[context.key]
            nested_prediction = nested_context.prediction
            nested_decision = nested_decisions[context.key]
            fold = folds[prediction.query_center_id]
            nested_threshold, nested_margin = heldout_thresholds[
                prediction.query_center_id
            ]
            best_gain = context.best_bacc_gain
            regret = best_gain - decision.bacc_gain
            nested_regret = best_gain - nested_decision.bacc_gain
            reason = (
                "SOURCE_OOF_ROUTED_POLICY_ACCEPTED_TOP1"
                if decision.routed
                else "EXACT_B_SOURCE_POLICY_ABSTENTION"
            )
            rows.append(
                {
                    "outer_target_id": bundle.outer_target_id,
                    "query_center_id": prediction.query_center_id,
                    "case_id": prediction.case_id,
                    "prediction": dict(prediction.public_payload()),
                    "active_action_ids": [
                        action.action_id for action in context.menu.actions
                    ],
                    "active_action_hashes": [
                        action.action_hash for action in context.menu.actions
                    ],
                    "active_outcome_hashes": [
                        outcome.outcome_hash for outcome in context.outcomes
                    ],
                    "case_inventory_kind": context.label_free.kind.value,
                    "is_exact_b_control": context.is_exact_b_control,
                    "label_free_case_context_hash": (
                        context.label_free.context_hash
                    ),
                    "case_outcome_context_hash": context.context_hash,
                    "source_outcome_universe_hash": universe.universe_hash,
                    "source_oof_label_free_inventory_hash": (
                        inventory.label_free_inventory_hash
                    ),
                    "source_oof_inventory_hash": inventory.inventory_hash,
                    "selected_action_id": decision.selected_action_id,
                    "reason": reason,
                    "observed_bacc_gain": decision.bacc_gain,
                    "observed_brier_delta": decision.brier_delta,
                    "observed_log_delta": decision.log_delta,
                    "best_observed_bacc_gain": best_gain,
                    "regret": regret,
                    "acceptance_threshold": policy.calibration.acceptance_threshold,
                    "rank_margin_threshold": policy.calibration.rank_margin_threshold,
                    "nested_selected_action_id": nested_decision.selected_action_id,
                    "nested_acceptance_probability": (
                        nested_prediction.acceptance_probability
                    ),
                    "nested_rank_margin": nested_prediction.rank_margin,
                    "nested_observed_bacc_gain": nested_decision.bacc_gain,
                    "nested_observed_brier_delta": nested_decision.brier_delta,
                    "nested_observed_log_delta": nested_decision.log_delta,
                    "nested_regret": nested_regret,
                    "nested_acceptance_threshold": nested_threshold,
                    "nested_rank_margin_threshold": nested_margin,
                    "nested_threshold_training_center_ids": list(
                        fold.training_center_ids
                    ),
                    "nested_policy_fold_hash": fold.fold_hash,
                    "nested_policy_replay_hash": (
                        policy.calibration.nested_replay.replay_hash
                    ),
                    "heldout_model_hash": prediction.model_hash,
                    "nested_heldout_model_hash": nested_prediction.model_hash,
                    "nested_prediction_hash": nested_prediction.prediction_hash,
                    "menu_hash": prediction.menu_hash,
                    "prediction_hash": prediction.prediction_hash,
                }
            )
            numeric.append(
                (
                    float(prediction.acceptance_probability),
                    float(prediction.rank_margin),
                    float(decision.routed),
                    decision.bacc_gain,
                    decision.brier_delta,
                    decision.log_delta,
                    best_gain,
                    regret,
                )
            )
            nested_numeric.append(
                (
                    float(nested_prediction.acceptance_probability),
                    float(nested_prediction.rank_margin),
                    float(nested_decision.routed),
                    nested_decision.bacc_gain,
                    nested_decision.brier_delta,
                    nested_decision.log_delta,
                    best_gain,
                    nested_regret,
                )
            )
    return (
        rows,
        np.asarray(numeric, dtype=np.float64).reshape((-1, 8)),
        np.asarray(nested_numeric, dtype=np.float64).reshape((-1, 8)),
    )


def build_source_admission_artifact(
    fitted: ArtifactValue,
    development: ArtifactValue,
    *,
    config: object,
    admission_fn: Callable[..., RouterAdmissionState] = build_source_only_admission,
) -> ArtifactValue:
    """Persist local admission and inventory-backed whole-policy OOF replay."""

    fit_state = require_state(fitted, RouterFitState, role="fitted router")
    development_state = require_state(
        development, SourceDevelopmentState, role="source-development surface"
    )
    admitted = admission_fn(
        fit_state,
        development_state,
        model_config=getattr(config, "model"),
    )
    replay_rows, replay_values, nested_replay_values = _policy_oof_replay(
        fit_state, admitted
    )
    body = {
        **admission_manifest(admitted),
        "model_hash": require_sha256(
            fitted.manifest.get("model_hash"), role="model hash"
        ),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "source_policy_oof_rows": replay_rows,
        "source_policy_oof_case_count": len(replay_rows),
        "typed_case_outcome_inventory_replayed": True,
        "empty_effective_menus_retained_as_exact_B_controls": True,
        "complete_rank_accept_route_or_exact_b_replayed": True,
        "nested_held_source_threshold_policy_replayed": True,
        "per_action_worst_center_certificate_used": False,
        "per_outer_local_admission": True,
        "global_kill_switch_used": False,
        "source_only": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=admitted,
        manifest={**body, "admission_hash": canonical_hash(body)},
        arrays={
            "source_policy_oof_values": replay_values,
            "nested_source_policy_oof_values": nested_replay_values,
        },
    )


__all__ = ("build_source_admission_artifact",)
