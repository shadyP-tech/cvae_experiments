"""Terminal held-case directional correctness and abstention diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    FixedBankCaseDirectionalCorrectnessAbstentionRouterConfig,
    load_fixed_bank_case_directional_correctness_abstention_router_config,
)


def run_fixed_bank_case_directional_correctness_abstention_router(
    config: FixedBankCaseDirectionalCorrectnessAbstentionRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    from .runner import (
        run_fixed_bank_case_directional_correctness_abstention_router as run,
    )

    return run(config, artifact_root=artifact_root, **kwargs)


__all__ = (
    "FixedBankCaseDirectionalCorrectnessAbstentionRouterConfig",
    "load_fixed_bank_case_directional_correctness_abstention_router_config",
    "run_fixed_bank_case_directional_correctness_abstention_router",
)
