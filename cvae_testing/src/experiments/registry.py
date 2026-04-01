from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.hybrid import HybridAblationExperiment
from src.experiments.latent_compatibility import LatentCompatibilityExperiment
from src.experiments.learned_utility_routing import LearnedUtilityRoutingExperiment
from src.experiments.legacy_routed import LegacyRoutedExperiment


EXPERIMENT_REGISTRY = {
    "legacy_routed_cvae": LegacyRoutedExperiment,
    "hybrid_ablation": HybridAblationExperiment,
    "latent_compatibility": LatentCompatibilityExperiment,
    "learned_utility_routing": LearnedUtilityRoutingExperiment,
}


def create_experiment(mode: str) -> BaseExperiment:
    exp_cls = EXPERIMENT_REGISTRY.get(mode)
    if exp_cls is None:
        raise ValueError(f"Unsupported experiment.mode: {mode}. Available: {sorted(EXPERIMENT_REGISTRY)}")
    return exp_cls()
