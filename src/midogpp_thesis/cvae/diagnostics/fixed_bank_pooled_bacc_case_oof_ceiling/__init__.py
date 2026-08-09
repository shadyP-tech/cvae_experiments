"""Terminal pooled-BACC case-OOF ceiling over the known MIDOG++ bank."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name in {
        "FixedBankPooledBaccCaseOofCeilingConfig",
        "load_fixed_bank_pooled_bacc_case_oof_ceiling_config",
    }:
        from .config import (  # noqa: PLC0415
            FixedBankPooledBaccCaseOofCeilingConfig,
            load_fixed_bank_pooled_bacc_case_oof_ceiling_config,
        )

        return {
            "FixedBankPooledBaccCaseOofCeilingConfig": (
                FixedBankPooledBaccCaseOofCeilingConfig
            ),
            "load_fixed_bank_pooled_bacc_case_oof_ceiling_config": (
                load_fixed_bank_pooled_bacc_case_oof_ceiling_config
            ),
        }[name]
    if name == "run_fixed_bank_pooled_bacc_case_oof_ceiling":
        from .runner import (  # noqa: PLC0415
            run_fixed_bank_pooled_bacc_case_oof_ceiling,
        )

        return run_fixed_bank_pooled_bacc_case_oof_ceiling
    raise AttributeError(name)


__all__ = (
    "FixedBankPooledBaccCaseOofCeilingConfig",
    "load_fixed_bank_pooled_bacc_case_oof_ceiling_config",
    "run_fixed_bank_pooled_bacc_case_oof_ceiling",
)
