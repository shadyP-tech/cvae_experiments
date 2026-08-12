"""Terminal consumed-test multi-challenger hierarchical routing diagnostic."""

from __future__ import annotations

from pathlib import Path

from .config import (
    FixedBankMultiChallengerHierarchicalFlipRouterConfig,
    load_fixed_bank_multi_challenger_hierarchical_flip_router_config,
)


def run_fixed_bank_multi_challenger_hierarchical_flip_router(
    config: FixedBankMultiChallengerHierarchicalFlipRouterConfig,
    *,
    artifact_root: Path,
) -> Path:
    from .runner import run_fixed_bank_multi_challenger_hierarchical_flip_router as run

    return run(config, artifact_root=artifact_root)


__all__ = (
    "FixedBankMultiChallengerHierarchicalFlipRouterConfig",
    "load_fixed_bank_multi_challenger_hierarchical_flip_router_config",
    "run_fixed_bank_multi_challenger_hierarchical_flip_router",
)
