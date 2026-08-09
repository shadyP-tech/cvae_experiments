"""Terminal consumed-test diagnostic for a known fixed MIDOG++ expert bank."""

from __future__ import annotations

from .audit import run_fixed_bank_decision_core
from .contracts import (
    FixedBankDataset,
    FixedBankDecisionAuditResult,
    FixedBankFeatureRow,
    FixedBankResponseRow,
)
from .features import build_fixed_bank_dataset


def __getattr__(name: str) -> object:
    if name in {"FixedBankDecisionAuditConfig", "load_fixed_bank_decision_audit_config"}:
        from .config import (  # noqa: PLC0415
            FixedBankDecisionAuditConfig,
            load_fixed_bank_decision_audit_config,
        )

        return {
            "FixedBankDecisionAuditConfig": FixedBankDecisionAuditConfig,
            "load_fixed_bank_decision_audit_config": (
                load_fixed_bank_decision_audit_config
            ),
        }[name]
    if name == "run_fixed_bank_decision_audit":
        from .runner import run_fixed_bank_decision_audit  # noqa: PLC0415

        return run_fixed_bank_decision_audit
    raise AttributeError(name)


__all__ = (
    "FixedBankDataset",
    "FixedBankDecisionAuditConfig",
    "FixedBankDecisionAuditResult",
    "FixedBankFeatureRow",
    "FixedBankResponseRow",
    "build_fixed_bank_dataset",
    "load_fixed_bank_decision_audit_config",
    "run_fixed_bank_decision_core",
    "run_fixed_bank_decision_audit",
)
