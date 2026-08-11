"""Post-hoc source-trained, whole-test prediction-only diagnostic."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Keep config and workstation imports lazy at package import."""

    if name in {
        "FixedBankDisagreementRegretPredictionOnlyConfig",
        "load_fixed_bank_disagreement_regret_prediction_only_config",
    }:
        from .config import (
            FixedBankDisagreementRegretPredictionOnlyConfig,
            load_fixed_bank_disagreement_regret_prediction_only_config,
        )

        return {
            "FixedBankDisagreementRegretPredictionOnlyConfig": (
                FixedBankDisagreementRegretPredictionOnlyConfig
            ),
            "load_fixed_bank_disagreement_regret_prediction_only_config": (
                load_fixed_bank_disagreement_regret_prediction_only_config
            ),
        }[name]
    if name == "run_fixed_bank_disagreement_regret_prediction_only":
        from .runner import run_fixed_bank_disagreement_regret_prediction_only

        return run_fixed_bank_disagreement_regret_prediction_only
    raise AttributeError(name)


__all__ = (
    "FixedBankDisagreementRegretPredictionOnlyConfig",
    "load_fixed_bank_disagreement_regret_prediction_only_config",
    "run_fixed_bank_disagreement_regret_prediction_only",
)
