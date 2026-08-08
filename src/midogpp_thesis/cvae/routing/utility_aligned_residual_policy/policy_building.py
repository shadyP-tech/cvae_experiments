"""Thin policy-model and artifact assembly facade."""

from __future__ import annotations

from .config import UtilityAlignedResidualPolicyConfig
from .inputs import PolicyInputs
from .model_workers import fit_all_targets
from .policy_artifacts import BuiltPolicyBundle, build_policy_artifacts


def build_policy_bundle(
    config: UtilityAlignedResidualPolicyConfig,
    inputs: PolicyInputs,
    *,
    spawn_workers: bool = True,
) -> BuiltPolicyBundle:
    return build_policy_artifacts(
        config,
        inputs,
        fit_all_targets(config, inputs, spawn_workers=spawn_workers),
    )


__all__ = ("BuiltPolicyBundle", "build_policy_bundle")
