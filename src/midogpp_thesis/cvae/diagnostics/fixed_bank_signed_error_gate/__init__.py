"""Public API for the signed sample-level error-gate mechanism diagnostic."""

from .calibration import fit_signed_gate_decision
from .composition import compose_signed_predictions, margin_gate
from .contracts import (
    CorrectionRow,
    GradientTargetRow,
    LambdaPathRow,
    SignedFeatureRow,
    SignedGateDecision,
    SignedGateModel,
    Standardization,
)
from .execution import (
    SignedFoldProducts,
    SignedModelProducts,
    SignedPrelabelProducts,
    TargetFamilyFits,
    build_signed_fold_products,
    build_signed_prelabel_products,
    fit_all_target_families,
    fit_target_families,
)
from .features import (
    build_signed_features,
    feature_context_hash,
    permute_feature_alignment,
)
from .gradients import build_gradient_targets
from .label_capabilities import (
    SignedErrorLabelCapability,
    SignedErrorLabelCapabilityManager,
)
from .model import (
    NestedSignedGateModel,
    SignedGateFit,
    correction_surface_hash,
    fit_signed_gate,
    predict_corrections,
)
from .protocol import (
    SignedErrorGateProtocol,
    assert_consumed_test_diagnostic_only,
    canonical_consumed_test_protocol,
)
from .sealing import record_durable_fold_seals, record_durable_model_seals
from .terminal import SealedSignedGateEvaluationResult, evaluate_sealed_fold_products


def __getattr__(name: str) -> object:
    """Keep config/runner imports lazy at the package boundary."""

    if name in {
        "FixedBankSignedErrorGateConfig",
        "load_fixed_bank_signed_error_gate_config",
    }:
        from .config import (  # noqa: PLC0415
            FixedBankSignedErrorGateConfig,
            load_fixed_bank_signed_error_gate_config,
        )

        return {
            "FixedBankSignedErrorGateConfig": FixedBankSignedErrorGateConfig,
            "load_fixed_bank_signed_error_gate_config": (
                load_fixed_bank_signed_error_gate_config
            ),
        }[name]
    if name == "run_fixed_bank_signed_error_gate":
        from .runner import run_fixed_bank_signed_error_gate  # noqa: PLC0415

        return run_fixed_bank_signed_error_gate
    raise AttributeError(name)


__all__ = (
    "CorrectionRow",
    "FixedBankSignedErrorGateConfig",
    "GradientTargetRow",
    "LambdaPathRow",
    "NestedSignedGateModel",
    "SignedErrorGateProtocol",
    "SignedErrorLabelCapability",
    "SignedErrorLabelCapabilityManager",
    "SignedFeatureRow",
    "SignedFoldProducts",
    "SignedGateDecision",
    "SignedGateFit",
    "SignedGateModel",
    "SignedModelProducts",
    "SignedPrelabelProducts",
    "SealedSignedGateEvaluationResult",
    "Standardization",
    "TargetFamilyFits",
    "assert_consumed_test_diagnostic_only",
    "build_gradient_targets",
    "build_signed_features",
    "build_signed_fold_products",
    "build_signed_prelabel_products",
    "canonical_consumed_test_protocol",
    "compose_signed_predictions",
    "correction_surface_hash",
    "evaluate_sealed_fold_products",
    "feature_context_hash",
    "fit_all_target_families",
    "fit_signed_gate",
    "fit_signed_gate_decision",
    "fit_target_families",
    "margin_gate",
    "load_fixed_bank_signed_error_gate_config",
    "permute_feature_alignment",
    "predict_corrections",
    "record_durable_fold_seals",
    "record_durable_model_seals",
    "run_fixed_bank_signed_error_gate",
)
