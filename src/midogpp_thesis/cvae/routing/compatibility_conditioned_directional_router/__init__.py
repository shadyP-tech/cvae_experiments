"""Stage-neutral compatibility-conditioned directional routing science."""

from .admission import DEFAULT_THRESHOLDS, evaluate_source_only_admission
from .compatibility import build_compatibility_receipts, compatibility_by_candidate
from .composition import compose_directional_probability_bytes, compose_route
from .contracts import *  # noqa: F403
from .contracts import __all__ as _contract_exports
from .features import (
    COMPATIBILITY_FEATURE_NAMES,
    build_candidate_feature,
    build_label_free_opportunity,
    probability_hash,
)
from .hashing import canonical_bytes, canonical_hash, probability_bytes_hash, require_sha256
from .pairwise import (
    action_key,
    crossfit_source_predictions,
    fit_hurdle_pairwise_model,
    predict_action,
)
from .policy import (
    DEFAULT_MIXTURE_LAMBDA,
    DEFAULT_OPPORTUNITY_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    select_baseline_anchored_route,
)
from .pools import build_source_candidate_pool, build_target_candidate_pool
from .uncertainty import (
    DEFAULT_QUANTILE,
    apply_endpoint_bounds,
    bound_action_vs_baseline,
    build_oof_endpoint_rows,
    calibrate_endpoint_uncertainty,
)


__all__ = (
    *_contract_exports,
    "COMPATIBILITY_FEATURE_NAMES",
    "DEFAULT_MIXTURE_LAMBDA",
    "DEFAULT_OPPORTUNITY_THRESHOLD",
    "DEFAULT_QUANTILE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_TOP_K",
    "action_key",
    "apply_endpoint_bounds",
    "bound_action_vs_baseline",
    "build_candidate_feature",
    "build_compatibility_receipts",
    "build_label_free_opportunity",
    "build_oof_endpoint_rows",
    "build_source_candidate_pool",
    "build_target_candidate_pool",
    "calibrate_endpoint_uncertainty",
    "canonical_bytes",
    "canonical_hash",
    "compatibility_by_candidate",
    "compose_directional_probability_bytes",
    "compose_route",
    "crossfit_source_predictions",
    "evaluate_source_only_admission",
    "fit_hurdle_pairwise_model",
    "predict_action",
    "probability_bytes_hash",
    "probability_hash",
    "require_sha256",
    "select_baseline_anchored_route",
)
