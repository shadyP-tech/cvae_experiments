"""Pure scientific core for the terminal case-aware proxy audit."""

from .audit import (
    run_case_aware_proxy_information_audit,
    run_proxy_information_audit,
)
from .case_features import (
    build_case_aware_feature_row,
    build_case_aware_feature_surface,
    build_case_aware_proxy_feature_row,
    feature_row_from_payload,
)
from .config import (
    CaseAwareProxyInformationAuditConfig,
    load_utility_aligned_case_aware_proxy_information_audit_config,
)
from .contracts import (
    CaseAwareFeatureSurface,
    CaseAwareProxyFeatureRow,
    CaseAwareProxyInformationAuditResult,
    CaseAwareResponseRow,
    CaseAwareResponseSurface,
    SupportCaseVectors,
    expected_strict_training_row_count,
)
from .crossfit import crossfit_fold_audit_from_payload, crossfit_proxy_families
from .family_designs import (
    FAMILY_PREDICTORS,
    PROXY_FAMILY_SPECS,
    build_family_designs,
)
from .metrics import summarize_crossfit
from .response_surfaces import (
    ExactNineEvaluationVectors,
    balanced_accuracy,
    build_response_row,
    build_response_surface,
    exact_nine_response_values,
    mean_exact_nine_probabilities,
    soft_balanced_accuracy,
)


def run_utility_aligned_case_aware_proxy_information_audit(
    *args: object, **kwargs: object
):
    """Lazy orchestration facade that keeps the scientific core importable."""

    from .runner import run_utility_aligned_case_aware_proxy_information_audit as run

    return run(*args, **kwargs)


__all__ = (
    "CaseAwareFeatureSurface",
    "CaseAwareProxyFeatureRow",
    "CaseAwareProxyInformationAuditResult",
    "CaseAwareProxyInformationAuditConfig",
    "CaseAwareResponseRow",
    "CaseAwareResponseSurface",
    "ExactNineEvaluationVectors",
    "FAMILY_PREDICTORS",
    "PROXY_FAMILY_SPECS",
    "SupportCaseVectors",
    "balanced_accuracy",
    "build_case_aware_feature_row",
    "build_case_aware_feature_surface",
    "build_case_aware_proxy_feature_row",
    "build_family_designs",
    "build_response_row",
    "build_response_surface",
    "crossfit_proxy_families",
    "crossfit_fold_audit_from_payload",
    "exact_nine_response_values",
    "expected_strict_training_row_count",
    "feature_row_from_payload",
    "mean_exact_nine_probabilities",
    "load_utility_aligned_case_aware_proxy_information_audit_config",
    "run_case_aware_proxy_information_audit",
    "run_proxy_information_audit",
    "run_utility_aligned_case_aware_proxy_information_audit",
    "soft_balanced_accuracy",
    "summarize_crossfit",
)
