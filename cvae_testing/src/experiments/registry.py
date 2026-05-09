from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.hybrid import HybridAblationExperiment
from src.experiments.learned_utility_routing import LearnedUtilityRoutingExperiment


EXPERIMENT_REGISTRY = {
    "hybrid_ablation": HybridAblationExperiment,
    "learned_utility_routing": LearnedUtilityRoutingExperiment,
}

QUARANTINED_EXPERIMENT_MODES = {
    "legacy_routed_cvae": "legacy routed-CVAE allowed target-expert candidates and is quarantined",
    "latent_compatibility": "latent compatibility is diagnostic-only and is quarantined as a normal run mode",
}


def create_experiment(mode: str) -> BaseExperiment:
    if mode in QUARANTINED_EXPERIMENT_MODES:
        raise ValueError(
            f"experiment.mode '{mode}' is quarantined: {QUARANTINED_EXPERIMENT_MODES[mode]}. "
            f"Use one of: {sorted(EXPERIMENT_REGISTRY)}"
        )
    exp_cls = EXPERIMENT_REGISTRY.get(mode)
    if exp_cls is None:
        raise ValueError(f"Unsupported experiment.mode: {mode}. Available: {sorted(EXPERIMENT_REGISTRY)}")
    return exp_cls()
