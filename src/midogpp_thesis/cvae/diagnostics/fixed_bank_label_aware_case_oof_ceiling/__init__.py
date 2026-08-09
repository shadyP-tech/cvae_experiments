"""Terminal label-aware case-OOF ceiling over the known MIDOG++ bank."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name in {
        "FixedBankLabelAwareCaseOofCeilingConfig",
        "load_fixed_bank_label_aware_case_oof_ceiling_config",
    }:
        from .config import (  # noqa: PLC0415
            FixedBankLabelAwareCaseOofCeilingConfig,
            load_fixed_bank_label_aware_case_oof_ceiling_config,
        )

        return {
            "FixedBankLabelAwareCaseOofCeilingConfig": (
                FixedBankLabelAwareCaseOofCeilingConfig
            ),
            "load_fixed_bank_label_aware_case_oof_ceiling_config": (
                load_fixed_bank_label_aware_case_oof_ceiling_config
            ),
        }[name]
    if name == "run_fixed_bank_label_aware_case_oof_ceiling":
        from .runner import (  # noqa: PLC0415
            run_fixed_bank_label_aware_case_oof_ceiling,
        )

        return run_fixed_bank_label_aware_case_oof_ceiling
    raise AttributeError(name)


__all__ = (
    "FixedBankLabelAwareCaseOofCeilingConfig",
    "load_fixed_bank_label_aware_case_oof_ceiling_config",
    "run_fixed_bank_label_aware_case_oof_ceiling",
)
