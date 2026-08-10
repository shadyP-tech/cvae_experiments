"""Public orchestration facade for the modular actionability diagnostic."""

from .decision_contracts import (
    DecisionProducts,
    FoldActionScore,
    FoldDecisionSeal,
    PreSupportDecisionProducts,
    SupportFoldProduct,
)
from .decision_execution import (
    build_pre_support_decision_products,
    build_support_fold_product,
    combine_decision_products,
)
from .model_execution import (
    ModelProducts,
    NestedMseSummary,
    NestedPredictionDiagnostic,
    TargetModelProduct,
    combine_model_products,
    fit_target_model_product,
)
from .utility_execution import (
    LocoUtilityProduct,
    PrelabelProducts,
    build_loco_utility_product,
    build_prelabel_products,
)


__all__ = (
    "DecisionProducts",
    "FoldActionScore",
    "FoldDecisionSeal",
    "LocoUtilityProduct",
    "ModelProducts",
    "NestedMseSummary",
    "NestedPredictionDiagnostic",
    "PreSupportDecisionProducts",
    "PrelabelProducts",
    "SupportFoldProduct",
    "TargetModelProduct",
    "build_loco_utility_product",
    "build_pre_support_decision_products",
    "build_prelabel_products",
    "build_support_fold_product",
    "combine_decision_products",
    "combine_model_products",
    "fit_target_model_product",
)
