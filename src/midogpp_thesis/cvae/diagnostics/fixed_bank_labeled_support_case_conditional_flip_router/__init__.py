"""Terminal consumed-test labeled-support threshold-flip diagnostic."""

from __future__ import annotations

from pathlib import Path

from .config import (
    FixedBankLabeledSupportCaseConditionalFlipRouterConfig,
    load_fixed_bank_labeled_support_case_conditional_flip_router_config,
)


def run_fixed_bank_labeled_support_case_conditional_flip_router(
    config: FixedBankLabeledSupportCaseConditionalFlipRouterConfig,
    *,
    artifact_root: Path,
) -> Path:
    from .runner import run_fixed_bank_labeled_support_case_conditional_flip_router as run

    return run(config, artifact_root=artifact_root)


__all__ = (
    "FixedBankLabeledSupportCaseConditionalFlipRouterConfig",
    "load_fixed_bank_labeled_support_case_conditional_flip_router_config",
    "run_fixed_bank_labeled_support_case_conditional_flip_router",
)
