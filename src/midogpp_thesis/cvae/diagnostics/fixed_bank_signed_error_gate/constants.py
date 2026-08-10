"""Frozen mathematical constants for the signed-error mechanism diagnostic."""

from __future__ import annotations


FEATURE_NAMES = (
    "intercept",
    "absolute_baseline_logit_margin",
    "residual_logit_mean",
    "residual_logit_abs_mean",
    "residual_logit_std",
    "positive_residual_mass",
    "negative_residual_mass",
    "hard_disagreement_rate",
    "candidate_probability_std",
    "residual_mean_x_near_threshold",
    "disagreement_x_near_threshold",
)

# G is the strictly case-independent intercept-only control. The fixed margin
# envelope is applied later and therefore remains identical across G/R/P.
GLOBAL_FEATURE_INDICES = (0,)

RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0)
INTERCEPT_GRID = (-0.1, -0.05, 0.0, 0.05, 0.1)
LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.25)
PROBABILITY_EPSILON = 1.0e-4
HARD_THRESHOLD = 0.5
MARGIN_BANDWIDTH_LOGIT = 1.0
MAX_ABSOLUTE_CORRECTION_LOGIT = 2.0
UNCERTAINTY_Z = 1.96
MIN_NESTED_MODELS = 3
STANDARDIZATION_SCALE_FLOOR = 1.0e-3
PERMUTATION_NAMESPACE = "midogpp_signed_error_gate_sample_features_v1"
METHOD_IDS = ("B", "B_cal", "G", "R_raw", "R_safe", "P")


__all__ = tuple(name for name in globals() if name.isupper())
