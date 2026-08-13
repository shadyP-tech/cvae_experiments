"""Quarantined fixed-bank whole-case LOO directional diagnostic."""

from .config import (
    FixedBankLooDirectionalShrinkageEnsembleConfig,
    load_fixed_bank_loo_directional_shrinkage_ensemble_config,
)


def run_fixed_bank_loo_directional_shrinkage_ensemble(
    *args: object, **kwargs: object
):
    from .runner import (
        run_fixed_bank_loo_directional_shrinkage_ensemble as implementation,
    )

    return implementation(*args, **kwargs)


def validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle(
    *args: object, **kwargs: object
):
    from .validation import (
        validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle as implementation,
    )

    return implementation(*args, **kwargs)


__all__ = (
    "FixedBankLooDirectionalShrinkageEnsembleConfig",
    "load_fixed_bank_loo_directional_shrinkage_ensemble_config",
    "run_fixed_bank_loo_directional_shrinkage_ensemble",
    "validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle",
)
