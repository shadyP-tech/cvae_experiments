"""Runnable policy-lock orchestration with fail-closed fresh-input gates."""

from __future__ import annotations

import json
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .bundle import REQUIRED_FILES, persist_policy_bundle, validate_policy_bundle
from .config import (
    UtilityAlignedResidualPolicyConfig,
    require_policy_inputs_ready,
)
from .inputs import load_policy_inputs
from .policy_building import build_policy_bundle
from .workspace_binding import validate_production_workspace_binding


def run_utility_aligned_residual_policy_lock(
    config: UtilityAlignedResidualPolicyConfig,
    *,
    workspace_validator: Callable[[UtilityAlignedResidualPolicyConfig], None] = (
        validate_production_workspace_binding
    ),
) -> dict[str, object]:
    """Fit and lock the declared policy; never opens any target label."""

    workspace_validator(config)
    if all((config.artifact_root / member).is_file() for member in REQUIRED_FILES):
        return dict(validate_policy_bundle(config.artifact_root, config=config))
    state_path = config.artifact_root / "reports/run_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("Utility-aligned partial run-state is unreadable.") from exc
        if isinstance(state, Mapping) and state.get("status") == "COMPLETE":
            raise ProtocolError(
                "Utility-aligned COMPLETE artifact is incomplete; refusing silent refit."
            )
    require_policy_inputs_ready(config)
    inputs = load_policy_inputs(config)
    bundle = build_policy_bundle(config, inputs)
    persist_policy_bundle(config.artifact_root, bundle, config=config)
    return dict(bundle.policy_lock)


__all__ = ("run_utility_aligned_residual_policy_lock",)
