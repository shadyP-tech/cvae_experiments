"""Terminal MIDOG++ fixed-bank hierarchical residual-stacker diagnostic."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name in {
        "FixedBankHierarchicalResidualStackerConfig",
        "load_fixed_bank_hierarchical_residual_stacker_config",
    }:
        from .config import (  # noqa: PLC0415
            FixedBankHierarchicalResidualStackerConfig,
            load_fixed_bank_hierarchical_residual_stacker_config,
        )

        return {
            "FixedBankHierarchicalResidualStackerConfig": (
                FixedBankHierarchicalResidualStackerConfig
            ),
            "load_fixed_bank_hierarchical_residual_stacker_config": (
                load_fixed_bank_hierarchical_residual_stacker_config
            ),
        }[name]
    if name == "run_fixed_bank_hierarchical_residual_stacker":
        from .runner import (  # noqa: PLC0415
            run_fixed_bank_hierarchical_residual_stacker,
        )

        return run_fixed_bank_hierarchical_residual_stacker
    raise AttributeError(name)


__all__ = (
    "FixedBankHierarchicalResidualStackerConfig",
    "load_fixed_bank_hierarchical_residual_stacker_config",
    "run_fixed_bank_hierarchical_residual_stacker",
)
