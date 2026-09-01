"""Durable source-only model and learnability-admission artifacts for HARP v5."""

from __future__ import annotations

from collections.abc import Callable

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    ActionKind as RouterActionKind,
    SourceActionObservation,
)
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ArtifactValue
from .model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    admission_manifest,
    build_source_only_admission,
    fit_outer_routers,
    model_manifest,
)
from .production_validation import require_sha256, require_state


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def build_source_router_artifact(
    development: ArtifactValue,
    compatibility: ArtifactValue,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
    fit_fn: Callable[..., RouterFitState] = fit_outer_routers,
) -> ArtifactValue:
    """Fit source-only outer routers and bind them to sealed input artifacts."""

    observations = require_state(
        development, tuple, role="source-development surface"
    )
    if not observations or any(
        not isinstance(row, SourceActionObservation) for row in observations
    ):
        raise ProtocolError("HARP v5 source-development observations are untyped.")
    compatibility_state = compatibility_loader(compatibility)
    known_receipts = {row.receipt_hash for row in compatibility_state.receipts}
    if any(
        row.feature.compatibility_receipt_hash not in known_receipts
        for row in observations
        if row.feature.action_kind is RouterActionKind.HXE
    ):
        raise ProtocolError("HARP v5 model rows escaped sealed compatibility receipts.")
    fitted = fit_fn(
        observations,
        model_config=getattr(config, "model"),
        runtime_config=getattr(config, "runtime"),
    )
    body = {
        **model_manifest(fitted),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"),
            role="compatibility hash",
        ),
        "alpha_selected_inside_source_lodo": True,
        "policy_hyperparameters_frozen_preexecution": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=fitted,
        manifest={**body, "model_hash": canonical_hash(body)},
    )


def build_source_admission_artifact(
    fitted: ArtifactValue,
    development: ArtifactValue,
    *,
    config: object,
    admission_fn: Callable[..., RouterAdmissionState] = build_source_only_admission,
) -> ArtifactValue:
    """Apply the fixed source-only learnability gate and bind its receipt."""

    fit_state = require_state(fitted, RouterFitState, role="fitted router")
    require_state(development, tuple, role="source-development surface")
    threshold = float(getattr(config, "model")["opportunity_probability_threshold"])
    admitted = admission_fn(fit_state, opportunity_threshold=threshold)
    body = {
        **admission_manifest(admitted),
        "model_hash": require_sha256(
            fitted.manifest.get("model_hash"), role="model hash"
        ),
        "development_surface_hash": require_sha256(
            development.manifest.get("surface_hash"), role="development surface hash"
        ),
        "opportunity_probability_threshold": threshold,
        "source_only": True,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=admitted,
        manifest={**body, "admission_hash": canonical_hash(body)},
    )


__all__ = (
    "build_source_admission_artifact",
    "build_source_router_artifact",
)
