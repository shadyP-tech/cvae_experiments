"""Stable facade for the terminal hierarchical residual-stacker science."""

from .calibration import class_balanced_log_loss, fit_baseline_intercept, fit_residual_scale
from .case_features import (
    compute_case_features,
    compute_source_control,
    compute_source_controls,
    context_permute_training_features,
    feature_surface_hash,
    permute_case_features,
)
from .composition import (
    baseline_predictions,
    calibrated_baseline_predictions,
    calibrated_baseline_probability,
    compose_probabilities,
    soft_class_residual,
)
from .contracts import (
    BinaryLabel,
    CalibrationChoice,
    CandidateClassModel,
    CaseClassWeights,
    CaseConfusionCounts,
    CaseFeatureRow,
    DonorResponseRow,
    HierarchicalResidualModel,
    PairedClusterEstimate,
    PooledExactBacc,
    PredictionRow,
    SampleActionProbability,
    SourceControlRow,
    Standardization,
)
from .controls import (
    ModelFamilyBundle,
    build_method_predictions,
    fit_model_families,
    predict_family_weights,
)
from .donor_responses import (
    build_donor_responses,
    response_class_coverage,
    response_surface_hash,
)
from .hierarchical_model import (
    fit_loco_hierarchical_model,
    fit_standardization,
    interaction_design,
    predict_candidate_gain,
    predict_case_weights,
    strict_transfer_training_rows,
    top2_sparse_simplex,
)
from .pooled_metrics import (
    paired_whole_case_cluster_lcb,
    pooled_exact_bacc,
    score_case_confusions,
)
from .residuals import clipped_probability, logit_clip, residual_logit, sigmoid
from .scientific_constants import *  # noqa: F403
from .uncertainty import (
    BootstrapContrast,
    EqualCenterContrast,
    equal_center_contrast,
    whole_case_bootstrap,
)


__all__ = tuple(name for name in globals() if not name.startswith("_"))
