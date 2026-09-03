"""Durable HARP v13 model scores, separate from policy replay artifacts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ArtifactValue
from .model_adapter import RouterFitState, fit_outer_routers, model_manifest
from .production_validation import require_sha256, require_state
from .source_development import SourceDevelopmentState


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def _numeric_oof_arrays(state: RouterFitState) -> dict[str, np.ndarray]:
    case_rows: list[tuple[float, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    score_offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    float(prediction.acceptance_probability),
                    float(prediction.rank_margin),
                    float(len(prediction.action_scores)),
                    float(prediction.top_action_id not in (None, "B")),
                )
            )
            score_rows.extend(
                (
                    float(score.pairwise_score),
                    float(score.predicted_budget_gain),
                    float(score.predicted_allocation_gain),
                    float(score.predicted_total_gain),
                    float(score.predicted_harm_probability),
                    float(score.predicted_brier_delta),
                    float(score.predicted_log_delta),
                    float(score.acceptance_probability),
                    float(score.model_available),
                )
                for score in prediction.action_scores
            )
            score_offsets.append(len(score_rows))
    return {
        "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 4)),
        "oof_action_scores": np.asarray(score_rows, dtype=np.float64).reshape((-1, 9)),
        "oof_action_score_offsets": np.asarray(score_offsets, dtype=np.int64),
    }


def build_source_router_artifact(
    development: ArtifactValue,
    compatibility: ArtifactValue,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
    fit_fn: Callable[..., RouterFitState] = fit_outer_routers,
) -> ArtifactValue:
    """Fit all outer routers and persist every numeric nested-LODO score."""

    development_state = require_state(
        development, SourceDevelopmentState, role="source-development surface"
    )
    compatibility_state = compatibility_loader(compatibility)
    known_menu_hashes = {menu.menu_hash for menu in compatibility_state.effective_menus}
    if any(
        menu.menu_hash not in known_menu_hashes
        for menu in development_state.effective_menus
    ):
        raise ProtocolError("HARP v13 model rows escaped the sealed effective menu.")
    fitted = fit_fn(
        development_state,
        model_config=getattr(config, "model"),
        runtime_config=getattr(config, "runtime"),
    )
    body = {
        **model_manifest(fitted),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"), role="compatibility hash"
        ),
        "effective_menu_hash": canonical_hash(
            [menu.menu_hash for menu in development_state.effective_menus]
        ),
        "all_preprocessing_fit_inside_source_lodo": True,
        "regularization_hyperparameters_predeclared_fixed_before_source_lodo": True,
        "regularization_hyperparameter_selection_performed": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=fitted,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=_numeric_oof_arrays(fitted),
    )


__all__ = ("build_source_router_artifact",)
