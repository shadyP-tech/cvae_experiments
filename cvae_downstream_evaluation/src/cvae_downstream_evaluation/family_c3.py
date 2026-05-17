"""Family C3 rich latent-sampler downstream evaluation.

Family C3 keeps the independently trained Family C label-conditioned CVAE
experts frozen. It changes only the class-conditional latent sampler used for
synthetic embedding generation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .family_c import (
    FAMILY_C_BUDGET_PER_CLASS,
    FAMILY_C_CLASSIFIER_SEEDS,
    FAMILY_C_DATASET_NAME,
    FAMILY_C_GENERATION_SEEDS,
    FAMILY_C_HIDDEN_DIM,
    FAMILY_C_INPUT_DIM,
    FAMILY_C_LABEL_VALUES,
    FAMILY_C_LATENT_DIM,
    FAMILY_C_SOURCE_TRANSFER_METHOD,
    FAMILY_C_SUPPORT_SEEDS,
    FAMILY_C_SUPPORT_SIZES,
    FamilyCDownstreamRow,
    TorchLabelConditionedExpertBank,
    TrainedClassifier,
    _as_int_tuple,
    _as_numpy_synthetic_batch,
    _batch_arrays,
    _classifier_manifest_row,
    _dedupe_rows,
    _domain_from_meta,
    _ensure_cvae_testing_imports,
    _evaluate_matrix_row,
    _generation_manifest_row,
    _label_from_meta,
    _mapping,
    _nanmean,
    _ordered_keys,
    _protocol_audit_rows,
    _read_csv,
    _recreate_eval_splits,
    _resolve,
    _write_csv,
    _write_json,
    classifier_cache_key,
    default_family_c_config,
    read_family_c_downstream_matrix,
    resolve_family_c_checkpoint_paths,
    train_locked_synthetic_classifier,
    validate_family_c_checkpoint_provenance,
    validate_family_c_protocol_audit,
    write_family_c_downstream_matrix,
)
from .family_c2 import (
    FAMILY_C2_PRIMARY_GENERATION_MODE,
    _center_level_mean,
    _compute_oracles_for_mode,
    _summary_row,
    preflight_family_c2_downstream_inputs,
)
from .generation import SyntheticBatch
from .protocol import ArtifactSyncError, ProtocolError
from .schemas import SINGLE_EXPERT_ROW_TYPE


FAMILY_C3_EXPERIMENT_NAME = "family_c3_rich_latent_sampler_downstream_v1"
FAMILY_C3_SOURCE_TRANSFER_METHOD = "family_c3_source_transfer_sampler_expert_prior"
FAMILY_C3_BOOTSTRAP_MU_MODE = "class_conditional_posterior_stratified_bootstrap_mu"
FAMILY_C3_BOOTSTRAP_T1_MODE = "class_conditional_posterior_stratified_bootstrap_t1"
FAMILY_C3_GMM_MODE = "class_conditional_gmm_mu_diag_bic"
FAMILY_C3_MODE_TIE_BREAK_ORDER = (
    FAMILY_C3_BOOTSTRAP_MU_MODE,
    FAMILY_C3_BOOTSTRAP_T1_MODE,
    FAMILY_C3_GMM_MODE,
)
FAMILY_C3_GENERATION_MODES = FAMILY_C3_MODE_TIE_BREAK_ORDER
FAMILY_C3_MIN_SOURCE_TRAIN_PER_CLASS = 16
FAMILY_C3_GMM_K_CANDIDATES = (1, 2, 4)
FAMILY_C3_GMM_REG_COVAR = 1e-4
FAMILY_C3_GMM_MIN_COMPONENT_WEIGHT = 1e-6
FAMILY_C3_DUPLICATE_EPS = 1e-6
FAMILY_C3_SAMPLER_RELEASE_LEVEL = "per_sample_posterior_bank"


@dataclass(frozen=True)
class FamilyC3DownstreamConfig:
    family_c_reports_dir: str
    family_c_run_root: str
    family_c_standard_artifacts_root: str
    family_c2_artifacts_root: str
    artifacts_root: str
    train_cache: str
    val_cache: str
    test_cache: str
    checkpoints_dir: str
    support_sizes: tuple[int, ...] = FAMILY_C_SUPPORT_SIZES
    support_seeds: tuple[int, ...] = FAMILY_C_SUPPORT_SEEDS
    generation_seeds: tuple[int, ...] = FAMILY_C_GENERATION_SEEDS
    classifier_seeds: tuple[int, ...] = FAMILY_C_CLASSIFIER_SEEDS
    budget_per_class: int = FAMILY_C_BUDGET_PER_CLASS
    hidden_dim: int = FAMILY_C_HIDDEN_DIM
    latent_dim: int = FAMILY_C_LATENT_DIM
    input_dim: int = FAMILY_C_INPUT_DIM
    label_values: tuple[int, ...] = FAMILY_C_LABEL_VALUES
    generation_modes: tuple[str, ...] = FAMILY_C3_GENERATION_MODES
    mode_tie_break_order: tuple[str, ...] = FAMILY_C3_MODE_TIE_BREAK_ORDER
    min_source_train_per_class: int = FAMILY_C3_MIN_SOURCE_TRAIN_PER_CLASS
    gmm_k_candidates: tuple[int, ...] = FAMILY_C3_GMM_K_CANDIDATES
    gmm_reg_covar: float = FAMILY_C3_GMM_REG_COVAR
    gmm_min_component_weight: float = FAMILY_C3_GMM_MIN_COMPONENT_WEIGHT
    duplicate_eps: float = FAMILY_C3_DUPLICATE_EPS
    smoke: bool = False


@dataclass(frozen=True)
class PosteriorBank:
    expert: str
    class_label: int
    mu: object
    logvar: object
    n_source_train: int
    available: int
    diagnostics: Mapping[str, object]
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class GmmLatentPrior:
    expert: str
    class_label: int
    selected_k: int
    weights: object
    means: object
    covariances: object
    available: int
    diagnostics: Mapping[str, object]


FAMILY_C3_GENERATION_MANIFEST_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "budget_per_class",
    "generation_mode",
    "label_values",
    "class_counts",
    "synthetic_data_hash",
    "generated_nan_count",
    "generated_inf_count",
    "generated_norm_mean",
    "generated_norm_std",
    "real_source_norm_mean",
    "real_source_norm_std",
)

FAMILY_C3_CLASSIFIER_MANIFEST_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "classifier_path_or_hash",
    "synthetic_data_hash",
    "scaler_fit_scope",
)

FAMILY_C3_POSTERIOR_BANK_COLUMNS = (
    "expert",
    "class_label",
    "posterior_bank_fit_split",
    "source_domain",
    "n_source_train",
    "latent_dim",
    "min_source_train_per_class",
    "mu_norm_mean",
    "mu_norm_std",
    "logvar_mean",
    "sampler_release_level",
    "available",
)

FAMILY_C3_GMM_DIAGNOSTIC_COLUMNS = (
    "expert",
    "class_label",
    "mode",
    "n_source_train",
    "valid_k_candidates",
    "gmm_converged",
    "gmm_n_iter",
    "gmm_bic_by_k",
    "gmm_selected_k",
    "gmm_min_component_weight",
    "mode_available",
    "unavailable_reason",
)

FAMILY_C3_SAMPLER_DIAGNOSTIC_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "generation_mode",
    "mode_available",
    "unavailable_reason",
    "sampler_release_level",
    "generated_effective_rank",
    "generated_cov_trace",
    "generated_pairwise_distance_mean",
    "generated_duplicate_rate",
    "duplicate_eps",
    "latent_sample_norm_mean",
    "latent_sample_norm_std",
    "generated_class_centroid_distance",
    "generated_class_linear_probe_train_accuracy",
    "real_source_val_accuracy_from_generated_train",
)

FAMILY_C3_ALIGNMENT_COLUMNS = (
    "heldout_center",
    "method",
    "selected_expert",
    "selected_generation_mode",
    "generation_seed",
    "classifier_seed",
    "budget_per_class",
    "generation_mode",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "downstream_oracle_generation_mode",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "top1_downstream_oracle_hit",
    "spearman_neg_support_score_vs_bacc",
    "available",
    "selection_source",
)

FAMILY_C3_SOURCE_TRANSFER_AUDIT_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_mode",
    "prior_score",
    "prior_score_std_across_source_centers",
    "prior_score_min_across_source_centers",
    "prior_score_max_across_source_centers",
    "selected_expert",
    "selected_generation_mode",
    "n_source_centers_used",
    "source_centers_used",
    "n_rows_used",
    "min_required_source_centers",
    "coverage_ok",
    "target_heldout_rows_used",
    "target_eval_labels_used",
    "target_heldout_rows_used_for_sampler_prior",
    "target_mode_oracle_used_for_selection",
    "target_expert_oracle_used_for_selection",
    "selection_source",
    "available",
)

FAMILY_C3_PROTOCOL_AUDIT_COLUMNS = (
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "target_expert_excluded",
    "support_eval_disjoint",
    "support_labels_used_for_routing",
    "routing_uses_eval_score",
    "posterior_bank_fit_split",
    "target_support_labels_used_for_generation",
    "target_eval_embeddings_used_for_generation",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "target_heldout_rows_used_for_sampler_prior",
    "target_mode_oracle_used_for_selection",
    "target_expert_oracle_used_for_selection",
    "metric_valid_bacc",
    "metric_valid_macro_f1",
)

FAMILY_C3_FIXED_COMPARISON_COLUMNS = (
    "heldout_center",
    "candidate_expert",
    "generation_seed",
    "classifier_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "c2_generation_mode",
    "c3_generation_mode",
    "bacc_c2_fitted_prior",
    "bacc_c3",
    "delta_bacc_c3_minus_c2",
    "macro_f1_c2_fitted_prior",
    "macro_f1_c3",
    "delta_macro_f1_c3_minus_c2",
)

FAMILY_C3_SELECTED_POLICY_COMPARISON_COLUMNS = (
    "heldout_center",
    "generation_seed",
    "classifier_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "c2_method",
    "c2_selected_expert",
    "c2_generation_mode",
    "c2_selected_bacc",
    "c2_oracle_gap_bacc",
    "c3_method",
    "c3_selected_expert",
    "c3_generation_mode",
    "c3_selected_bacc",
    "c3_oracle_gap_bacc",
    "delta_bacc_c3_minus_c2",
    "delta_oracle_gap_c3_minus_c2",
)


def default_family_c3_config() -> FamilyC3DownstreamConfig:
    default_c = default_family_c_config()
    return FamilyC3DownstreamConfig(
        family_c_reports_dir=default_c.family_c_reports_dir,
        family_c_run_root=default_c.family_c_run_root,
        family_c_standard_artifacts_root=(
            "cvae_downstream_evaluation/artifacts/family_c_label_conditioned_downstream_v1"
        ),
        family_c2_artifacts_root=(
            "cvae_downstream_evaluation/artifacts/family_c2_fitted_latent_prior_downstream_v1"
        ),
        artifacts_root=(
            "cvae_downstream_evaluation/artifacts/"
            "family_c3_rich_latent_sampler_downstream_v1"
        ),
        train_cache=default_c.train_cache,
        val_cache=default_c.val_cache,
        test_cache=default_c.test_cache,
        checkpoints_dir=default_c.checkpoints_dir,
    )


def assert_family_c3_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_C3_EXPERIMENT_NAME}",
        FAMILY_C3_BOOTSTRAP_MU_MODE,
        FAMILY_C3_BOOTSTRAP_T1_MODE,
        FAMILY_C3_GMM_MODE,
        FAMILY_C3_SOURCE_TRANSFER_METHOD,
        "posterior_bank_fit_split: source_train",
        "sampler_release_level: per_sample_posterior_bank",
        "family_c3_downstream_decision_summary.json",
        "target_eval_labels_for_training: forbidden",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ProtocolError(f"Family C3 config missing required fields: {missing}")
    forbidden = (
        "target_support_empirical",
        "target_eval_empirical",
        "hyperparameter_tuning: allowed",
        "support_coral:\n    role: primary",
    )
    present = [value for value in forbidden if value in text]
    if present:
        raise ProtocolError(f"Family C3 config contains forbidden fields: {present}")


def load_family_c3_downstream_config(path: Path) -> FamilyC3DownstreamConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_c3_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_c3_config()
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ProtocolError("Family C3 downstream config must be a YAML mapping.")
    return family_c3_config_from_mapping(loaded)


def family_c3_config_from_mapping(config: Mapping[str, Any]) -> FamilyC3DownstreamConfig:
    exp = _mapping(config.get("experiment"), "experiment")
    if exp.get("name") != FAMILY_C3_EXPERIMENT_NAME:
        raise ProtocolError(f"experiment.name must be {FAMILY_C3_EXPERIMENT_NAME}")
    if str(exp.get("dataset", "")).strip() != FAMILY_C_DATASET_NAME:
        raise ProtocolError("Family C3 is Camelyon17 only.")
    inputs = _mapping(config.get("inputs"), "inputs")
    routing = _mapping(config.get("routing"), "routing")
    generation = _mapping(config.get("generation"), "generation")
    downstream = _mapping(config.get("downstream"), "downstream")
    posterior = _mapping(generation.get("posterior_bank"), "generation.posterior_bank")
    gmm = _mapping(generation.get("gmm"), "generation.gmm")
    labels = tuple(int(v) for v in generation.get("label_values", FAMILY_C_LABEL_VALUES))
    if labels != FAMILY_C_LABEL_VALUES:
        raise ProtocolError("Family C3 requires label_values [0, 1].")
    modes = tuple(str(v) for v in generation.get("modes", FAMILY_C3_GENERATION_MODES))
    if modes != FAMILY_C3_GENERATION_MODES:
        raise ProtocolError("generation.modes must contain the locked C3 modes in order.")
    tie_order = tuple(str(v) for v in generation.get("mode_tie_break_order", FAMILY_C3_MODE_TIE_BREAK_ORDER))
    if tie_order != FAMILY_C3_MODE_TIE_BREAK_ORDER:
        raise ProtocolError("generation.mode_tie_break_order must be mu -> t1 -> gmm.")
    if int(generation.get("budget_per_class", FAMILY_C_BUDGET_PER_CLASS)) != FAMILY_C_BUDGET_PER_CLASS:
        raise ProtocolError("Family C3 locks budget_per_class to 128.")
    classifier = _mapping(downstream.get("classifier"), "downstream.classifier")
    expected_classifier = {
        "family": "sklearn_logistic_regression",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
        "scaler_fit": "synthetic_train_only",
        "hyperparameter_tuning": "forbidden",
    }
    for key, expected in expected_classifier.items():
        if classifier.get(key) != expected:
            raise ProtocolError(f"downstream.classifier.{key} must be {expected!r}")

    default = default_family_c3_config()
    return FamilyC3DownstreamConfig(
        family_c_reports_dir=str(inputs.get("family_c_reports_dir", default.family_c_reports_dir)),
        family_c_run_root=str(inputs.get("family_c_run_root", default.family_c_run_root)),
        family_c_standard_artifacts_root=str(
            inputs.get("family_c_standard_artifacts_root", default.family_c_standard_artifacts_root)
        ),
        family_c2_artifacts_root=str(inputs.get("family_c2_artifacts_root", default.family_c2_artifacts_root)),
        artifacts_root=str(_mapping(config.get("artifacts"), "artifacts").get("root", default.artifacts_root)),
        train_cache=str(inputs.get("train_cache", default.train_cache)),
        val_cache=str(inputs.get("val_cache", default.val_cache)),
        test_cache=str(inputs.get("test_cache", default.test_cache)),
        checkpoints_dir=str(inputs.get("checkpoints_dir", default.checkpoints_dir)),
        support_sizes=_as_int_tuple(routing.get("support_sizes"), FAMILY_C_SUPPORT_SIZES),
        support_seeds=_as_int_tuple(routing.get("support_seeds"), FAMILY_C_SUPPORT_SEEDS),
        generation_seeds=_as_int_tuple(generation.get("generation_seeds"), FAMILY_C_GENERATION_SEEDS),
        classifier_seeds=_as_int_tuple(downstream.get("classifier_seeds"), FAMILY_C_CLASSIFIER_SEEDS),
        budget_per_class=int(generation.get("budget_per_class", FAMILY_C_BUDGET_PER_CLASS)),
        hidden_dim=int(generation.get("hidden_dim", FAMILY_C_HIDDEN_DIM)),
        latent_dim=int(generation.get("latent_dim", FAMILY_C_LATENT_DIM)),
        input_dim=int(generation.get("input_dim", FAMILY_C_INPUT_DIM)),
        label_values=labels,
        generation_modes=modes,
        mode_tie_break_order=tie_order,
        min_source_train_per_class=int(
            posterior.get("min_source_train_per_class", FAMILY_C3_MIN_SOURCE_TRAIN_PER_CLASS)
        ),
        gmm_k_candidates=_as_int_tuple(gmm.get("k_candidates"), FAMILY_C3_GMM_K_CANDIDATES),
        gmm_reg_covar=float(gmm.get("reg_covar", FAMILY_C3_GMM_REG_COVAR)),
        gmm_min_component_weight=float(gmm.get("min_component_weight", FAMILY_C3_GMM_MIN_COMPONENT_WEIGHT)),
        duplicate_eps=float(generation.get("duplicate_eps", FAMILY_C3_DUPLICATE_EPS)),
        smoke=bool(exp.get("smoke", False)),
    )


def preflight_family_c3_downstream_inputs(
    config: FamilyC3DownstreamConfig,
    *,
    repo_root: Path,
    require_heavy_artifacts: bool,
) -> dict[str, object]:
    c2_like = preflight_family_c2_downstream_inputs(
        config,
        repo_root=repo_root,
        require_heavy_artifacts=require_heavy_artifacts,
    )
    c2_root = _resolve(repo_root, config.family_c2_artifacts_root)
    required_c2 = [
        c2_root / "tables" / "family_c2_all_expert_downstream_matrix.csv",
        c2_root / "tables" / "family_c2_downstream_selection_alignment.csv",
        c2_root / "tables" / "family_c2_downstream_baseline_comparison.csv",
        c2_root / "reports" / "family_c2_downstream_decision_summary.json",
    ]
    missing_c2 = [path for path in required_c2 if not path.exists()]
    if missing_c2:
        raise ArtifactSyncError(_missing_message("Missing C2 comparison artifacts", missing_c2))
    estimates = estimate_family_c3_workload(config)
    return {
        **c2_like,
        "c2_artifacts_root": str(c2_root),
        **estimates,
    }


def estimate_family_c3_workload(config: FamilyC3DownstreamConfig) -> dict[str, int]:
    n_centers = 5
    n_candidates_per_center = 4
    n_generation_jobs = (
        n_centers
        * n_candidates_per_center
        * len(config.generation_modes)
        * len(config.generation_seeds)
    )
    n_classifier_jobs = n_generation_jobs * len(config.classifier_seeds)
    n_eval_rows = n_classifier_jobs * len(config.support_sizes) * len(config.support_seeds)
    return {
        "n_expected_generation_jobs": int(n_generation_jobs),
        "n_expected_classifier_jobs": int(n_classifier_jobs),
        "n_expected_eval_rows": int(n_eval_rows),
    }


def fit_posterior_bank_from_arrays(
    mu: object,
    logvar: object,
    *,
    min_count: int,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    mu_arr = np.asarray(mu, dtype=float)
    logvar_arr = np.asarray(logvar, dtype=float)
    if mu_arr.ndim != 2 or logvar_arr.ndim != 2 or mu_arr.shape != logvar_arr.shape:
        raise ValueError("mu and logvar must be 2D arrays with matching shapes.")
    n_rows = int(mu_arr.shape[0])
    latent_dim = int(mu_arr.shape[1]) if mu_arr.ndim == 2 else 0
    mu_norms = np.linalg.norm(mu_arr, axis=1) if n_rows else np.asarray([], dtype=float)
    return {
        "mu": mu_arr,
        "logvar": logvar_arr,
        "available": int(n_rows >= int(min_count)),
        "n_source_train": n_rows,
        "latent_dim": latent_dim,
        "mu_norm_mean": float(mu_norms.mean()) if n_rows else math.nan,
        "mu_norm_std": float(mu_norms.std()) if n_rows else math.nan,
        "logvar_mean": float(logvar_arr.mean()) if n_rows else math.nan,
    }


def valid_gmm_k_candidates(n_source_train: int, k_candidates: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(k)
        for k in k_candidates
        if int(k) > 0 and int(n_source_train) >= max(32, 8 * int(k))
    )


def fit_gmm_prior_from_mu(
    mu: object,
    *,
    k_candidates: Sequence[int],
    reg_covar: float,
    min_component_weight: float,
    random_state: int = 0,
) -> dict[str, object]:
    import numpy as np  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim != 2:
        raise ValueError("mu must be a 2D array.")
    valid_k = valid_gmm_k_candidates(int(mu_arr.shape[0]), k_candidates)
    if not valid_k:
        return _unavailable_gmm("no_valid_k", int(mu_arr.shape[0]), valid_k)
    bic_by_k: dict[str, float] = {}
    models: dict[int, object] = {}
    for k in valid_k:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=float(reg_covar),
            random_state=int(random_state) + int(k) * 1009,
        )
        model.fit(mu_arr)
        bic_by_k[str(k)] = float(model.bic(mu_arr))
        models[int(k)] = model
    selected_k = min(valid_k, key=lambda k: (bic_by_k[str(k)], int(k)))
    selected = models[int(selected_k)]
    weights = np.asarray(selected.weights_, dtype=float)
    means = np.asarray(selected.means_, dtype=float)
    covariances = np.asarray(selected.covariances_, dtype=float)
    converged = int(bool(getattr(selected, "converged_", False)))
    finite = int(np.isfinite(weights).all() and np.isfinite(means).all() and np.isfinite(covariances).all())
    positive_cov = int((covariances > 0).all())
    min_weight = float(weights.min()) if weights.size else math.nan
    available = int(
        converged == 1
        and finite == 1
        and positive_cov == 1
        and min_weight >= float(min_component_weight)
    )
    reason = ""
    if available != 1:
        if converged != 1:
            reason = "gmm_not_converged"
        elif finite != 1:
            reason = "non_finite_gmm_parameters"
        elif positive_cov != 1:
            reason = "non_positive_covariance"
        elif min_weight < float(min_component_weight):
            reason = "component_weight_below_threshold"
    return {
        "selected_k": int(selected_k),
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "available": available,
        "diagnostics": {
            "n_source_train": int(mu_arr.shape[0]),
            "valid_k_candidates": "|".join(str(k) for k in valid_k),
            "gmm_converged": converged,
            "gmm_n_iter": int(getattr(selected, "n_iter_", 0)),
            "gmm_bic_by_k": json.dumps(bic_by_k, sort_keys=True),
            "gmm_selected_k": int(selected_k),
            "gmm_min_component_weight": min_weight,
            "mode_available": available,
            "unavailable_reason": reason,
        },
    }


def _unavailable_gmm(reason: str, n_source_train: int, valid_k: Sequence[int]) -> dict[str, object]:
    return {
        "selected_k": 0,
        "weights": [],
        "means": [],
        "covariances": [],
        "available": 0,
        "diagnostics": {
            "n_source_train": int(n_source_train),
            "valid_k_candidates": "|".join(str(k) for k in valid_k),
            "gmm_converged": 0,
            "gmm_n_iter": 0,
            "gmm_bic_by_k": "{}",
            "gmm_selected_k": 0,
            "gmm_min_component_weight": math.nan,
            "mode_available": 0,
            "unavailable_reason": reason,
        },
    }


def fit_family_c3_posterior_banks(
    backend: TorchLabelConditionedExpertBank,
    *,
    train_x: object,
    train_domains: object,
    train_labels: object,
    label_values: Sequence[int],
    min_source_train_per_class: int,
    gmm_k_candidates: Sequence[int],
    gmm_reg_covar: float,
    gmm_min_component_weight: float,
) -> tuple[
    dict[tuple[str, int], PosteriorBank],
    dict[tuple[str, int], GmmLatentPrior],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x = np.asarray(train_x, dtype=float)
    domains = np.asarray(train_domains, dtype=np.int64)
    labels = np.asarray(train_labels, dtype=np.int64)
    banks: dict[tuple[str, int], PosteriorBank] = {}
    gmm_priors: dict[tuple[str, int], GmmLatentPrior] = {}
    bank_rows: list[dict[str, object]] = []
    gmm_rows: list[dict[str, object]] = []
    for expert in sorted(backend.models):
        model = backend.models[int(expert)]
        for class_label in label_values:
            mask = (domains == int(expert)) & (labels == int(class_label))
            x_class = x[mask]
            if x_class.size:
                tensor_x = torch.as_tensor(x_class, dtype=torch.float32, device=backend.device)
                y = torch.zeros((tensor_x.shape[0], int(backend.class_condition_dim)), dtype=torch.float32, device=backend.device)
                y[:, int(class_label)] = 1.0
                with torch.no_grad():
                    mu_t, logvar_t = model.encode(tensor_x, y=y)
                mu = mu_t.detach().cpu().numpy()
                logvar = logvar_t.detach().cpu().numpy()
            else:
                mu = np.zeros((0, int(backend.latent_dim)), dtype=float)
                logvar = np.zeros((0, int(backend.latent_dim)), dtype=float)
            fitted = fit_posterior_bank_from_arrays(
                mu,
                logvar,
                min_count=int(min_source_train_per_class),
            )
            mu_arr = np.asarray(fitted["mu"], dtype=float)
            logvar_arr = np.asarray(fitted["logvar"], dtype=float)
            provenance = {
                "expert": str(expert),
                "class_label": int(class_label),
                "posterior_bank_fit_split": "source_train",
                "source_domain": str(expert),
                "n_source_train": int(fitted["n_source_train"]),
                "latent_dim": int(fitted["latent_dim"]),
                "min_source_train_per_class": int(min_source_train_per_class),
                "mu_norm_mean": float(fitted["mu_norm_mean"]),
                "mu_norm_std": float(fitted["mu_norm_std"]),
                "logvar_mean": float(fitted["logvar_mean"]),
                "sampler_release_level": FAMILY_C3_SAMPLER_RELEASE_LEVEL,
                "available": int(fitted["available"]),
            }
            bank = PosteriorBank(
                expert=str(expert),
                class_label=int(class_label),
                mu=mu_arr,
                logvar=logvar_arr,
                n_source_train=int(fitted["n_source_train"]),
                available=int(fitted["available"]),
                diagnostics=provenance,
                provenance=provenance,
            )
            banks[(str(expert), int(class_label))] = bank
            bank_rows.append(provenance)

            gmm_fit = fit_gmm_prior_from_mu(
                mu_arr,
                k_candidates=gmm_k_candidates,
                reg_covar=float(gmm_reg_covar),
                min_component_weight=float(gmm_min_component_weight),
                random_state=int(expert) * 1000 + int(class_label),
            )
            gmm_diag = {
                "expert": str(expert),
                "class_label": int(class_label),
                "mode": FAMILY_C3_GMM_MODE,
                **dict(gmm_fit["diagnostics"]),
            }
            gmm_priors[(str(expert), int(class_label))] = GmmLatentPrior(
                expert=str(expert),
                class_label=int(class_label),
                selected_k=int(gmm_fit["selected_k"]),
                weights=gmm_fit["weights"],
                means=gmm_fit["means"],
                covariances=gmm_fit["covariances"],
                available=int(gmm_fit["available"]),
                diagnostics=gmm_diag,
            )
            gmm_rows.append(gmm_diag)
    return banks, gmm_priors, bank_rows, gmm_rows


def sample_posterior_bootstrap_embeddings(
    backend: TorchLabelConditionedExpertBank,
    banks: Mapping[tuple[str, int], PosteriorBank],
    *,
    expert_domain: int,
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int],
    temperature: float,
    generation_mode: str,
) -> tuple[SyntheticBatch, dict[str, object]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    chunks: list[object] = []
    labels: list[int] = []
    latent_chunks: list[object] = []
    expert = str(int(expert_domain))
    for offset, class_label in enumerate(int(v) for v in label_values):
        bank = banks.get((expert, int(class_label)))
        if bank is None or int(bank.available) != 1:
            raise ProtocolError(f"Posterior bank unavailable for expert={expert}, class={class_label}.")
        rng = np.random.default_rng(int(generation_seed) + (offset + 1) * 7919)
        mu = np.asarray(bank.mu, dtype=np.float32)
        logvar = np.asarray(bank.logvar, dtype=np.float32)
        indices = _stratified_bootstrap_indices(mu.shape[0], int(budget_per_class), rng)
        z_np = mu[indices]
        if float(temperature) != 0.0:
            eps = rng.normal(size=z_np.shape).astype(np.float32)
            z_np = z_np + float(temperature) * eps * np.sqrt(np.exp(logvar[indices]))
        z = torch.as_tensor(z_np, dtype=torch.float32, device=backend.device)
        y = torch.zeros((int(budget_per_class), int(backend.class_condition_dim)), dtype=torch.float32, device=backend.device)
        y[:, int(class_label)] = 1.0
        model = backend.models[int(expert_domain)]
        with torch.no_grad():
            decoded = model.decode(z, y=y)
        chunks.append(decoded.detach().cpu().numpy())
        latent_chunks.append(z_np)
        labels.extend([int(class_label)] * int(budget_per_class))
    batch = SyntheticBatch(
        expert_domain=expert,
        generation_mode=str(generation_mode),
        projection_frame="dinov2_embedding_posterior_bootstrap",
        embeddings=chunks,
        labels=labels,
    )
    return batch, _latent_stats(latent_chunks)


def sample_gmm_embeddings(
    backend: TorchLabelConditionedExpertBank,
    gmm_priors: Mapping[tuple[str, int], GmmLatentPrior],
    *,
    expert_domain: int,
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int],
) -> tuple[SyntheticBatch, dict[str, object]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    chunks: list[object] = []
    labels: list[int] = []
    latent_chunks: list[object] = []
    expert = str(int(expert_domain))
    for offset, class_label in enumerate(int(v) for v in label_values):
        prior = gmm_priors.get((expert, int(class_label)))
        if prior is None or int(prior.available) != 1:
            raise ProtocolError(f"GMM latent prior unavailable for expert={expert}, class={class_label}.")
        rng = np.random.default_rng(int(generation_seed) + (offset + 1) * 7919)
        weights = np.asarray(prior.weights, dtype=float)
        means = np.asarray(prior.means, dtype=np.float32)
        covariances = np.asarray(prior.covariances, dtype=np.float32)
        components = rng.choice(len(weights), size=int(budget_per_class), replace=True, p=weights / weights.sum())
        eps = rng.normal(size=(int(budget_per_class), means.shape[1])).astype(np.float32)
        z_np = means[components] + eps * np.sqrt(covariances[components])
        z = torch.as_tensor(z_np, dtype=torch.float32, device=backend.device)
        y = torch.zeros((int(budget_per_class), int(backend.class_condition_dim)), dtype=torch.float32, device=backend.device)
        y[:, int(class_label)] = 1.0
        model = backend.models[int(expert_domain)]
        with torch.no_grad():
            decoded = model.decode(z, y=y)
        chunks.append(decoded.detach().cpu().numpy())
        latent_chunks.append(z_np)
        labels.extend([int(class_label)] * int(budget_per_class))
    batch = SyntheticBatch(
        expert_domain=expert,
        generation_mode=FAMILY_C3_GMM_MODE,
        projection_frame="dinov2_embedding_gmm_mu_diag_bic",
        embeddings=chunks,
        labels=labels,
    )
    return batch, _latent_stats(latent_chunks)


def sample_family_c3_embeddings(
    backend: TorchLabelConditionedExpertBank,
    banks: Mapping[tuple[str, int], PosteriorBank],
    gmm_priors: Mapping[tuple[str, int], GmmLatentPrior],
    *,
    expert_domain: int,
    generation_seed: int,
    budget_per_class: int,
    label_values: Sequence[int],
    generation_mode: str,
) -> tuple[SyntheticBatch, dict[str, object]]:
    if generation_mode == FAMILY_C3_BOOTSTRAP_MU_MODE:
        return sample_posterior_bootstrap_embeddings(
            backend,
            banks,
            expert_domain=expert_domain,
            generation_seed=generation_seed,
            budget_per_class=budget_per_class,
            label_values=label_values,
            temperature=0.0,
            generation_mode=generation_mode,
        )
    if generation_mode == FAMILY_C3_BOOTSTRAP_T1_MODE:
        return sample_posterior_bootstrap_embeddings(
            backend,
            banks,
            expert_domain=expert_domain,
            generation_seed=generation_seed,
            budget_per_class=budget_per_class,
            label_values=label_values,
            temperature=1.0,
            generation_mode=generation_mode,
        )
    if generation_mode == FAMILY_C3_GMM_MODE:
        return sample_gmm_embeddings(
            backend,
            gmm_priors,
            expert_domain=expert_domain,
            generation_seed=generation_seed,
            budget_per_class=budget_per_class,
            label_values=label_values,
        )
    raise ProtocolError(f"Unknown Family C3 generation mode: {generation_mode}")


def run_family_c3_downstream(
    config: FamilyC3DownstreamConfig,
    *,
    repo_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    preflight = preflight_family_c3_downstream_inputs(
        config,
        repo_root=repo_root,
        require_heavy_artifacts=not dry_run,
    )
    if dry_run:
        return {"status": "dry_run_passed", **preflight}

    _ensure_cvae_testing_imports(repo_root)
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from src.eval.evaluators.support_set_calibration import make_support_eval_split  # type: ignore
    from src.torch_utils import safe_torch_load  # type: ignore

    reports_dir = _resolve(repo_root, config.family_c_reports_dir)
    artifacts_root = _resolve(repo_root, config.artifacts_root)
    c2_root = _resolve(repo_root, config.family_c2_artifacts_root)
    tables_dir = artifacts_root / "tables"
    reports_out_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"

    provenance_rows = _read_csv(reports_dir / "label_conditioned_checkpoint_provenance.csv")
    protocol_rows = _read_csv(reports_dir / "label_marginal_protocol_audit.csv")
    validate_family_c_checkpoint_provenance(provenance_rows)
    validate_family_c_protocol_audit(protocol_rows)

    train_payload = safe_torch_load(_resolve(repo_root, config.train_cache), map_location="cpu")
    val_payload = safe_torch_load(_resolve(repo_root, config.val_cache), map_location="cpu")
    test_payload = safe_torch_load(_resolve(repo_root, config.test_cache), map_location="cpu")
    train_x = train_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    val_x = val_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    test_x = test_payload["embeddings"].detach().cpu().numpy().astype(float, copy=False)
    train_meta = list(train_payload["metadata"])
    val_meta = list(val_payload["metadata"])
    test_meta = list(test_payload["metadata"])
    train_domains = np.asarray([_domain_from_meta(row) for row in train_meta], dtype=np.int64)
    train_labels = np.asarray([_label_from_meta(row) for row in train_meta], dtype=np.int64)
    val_domains = np.asarray([_domain_from_meta(row) for row in val_meta], dtype=np.int64)
    val_labels = np.asarray([_label_from_meta(row) for row in val_meta], dtype=np.int64)
    test_domains = np.asarray([_domain_from_meta(row) for row in test_meta], dtype=np.int64)
    test_labels = np.asarray([_label_from_meta(row) for row in test_meta], dtype=np.int64)
    labels_by_index = {idx: int(label) for idx, label in enumerate(test_labels.tolist())}

    checkpoint_paths = resolve_family_c_checkpoint_paths(
        provenance_rows,
        checkpoints_dir=_resolve(repo_root, config.checkpoints_dir),
        require_exists=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = TorchLabelConditionedExpertBank.load(
        checkpoint_paths=checkpoint_paths,
        input_dim=int(config.input_dim),
        hidden_dim=int(config.hidden_dim),
        latent_dim=int(config.latent_dim),
        class_condition_dim=len(config.label_values),
        device=device,
        repo_root=repo_root,
    )

    banks, gmm_priors, bank_rows, gmm_rows = fit_family_c3_posterior_banks(
        backend,
        train_x=train_x,
        train_domains=train_domains,
        train_labels=train_labels,
        label_values=config.label_values,
        min_source_train_per_class=config.min_source_train_per_class,
        gmm_k_candidates=config.gmm_k_candidates,
        gmm_reg_covar=config.gmm_reg_covar,
        gmm_min_component_weight=config.gmm_min_component_weight,
    )

    splits = _recreate_eval_splits(
        test_domains=test_domains,
        labels_by_index=labels_by_index,
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
        make_support_eval_split=make_support_eval_split,
    )
    unique_eval_contexts = sorted(
        splits.values(),
        key=lambda item: (item["heldout_center"], item["support_size"], item["support_seed"]),
    )

    classifier_cache: dict[tuple[object, ...], TrainedClassifier] = {}
    generation_manifest: list[dict[str, object]] = []
    classifier_manifest: list[dict[str, object]] = []
    downstream_rows: list[FamilyCDownstreamRow] = []
    sampler_diagnostics: list[dict[str, object]] = []

    heldouts = sorted(set(str(int(v)) for v in test_domains.tolist()), key=lambda value: int(value))
    for heldout in heldouts:
        for generation_mode in config.generation_modes:
            source_experts = [str(domain) for domain in sorted(checkpoint_paths) if str(domain) != heldout]
            candidate_experts = []
            for expert in source_experts:
                if _mode_available(
                    generation_mode,
                    banks=banks,
                    gmm_priors=gmm_priors,
                    expert=expert,
                    label_values=config.label_values,
                ):
                    candidate_experts.append(expert)
                    continue
                for generation_seed in config.generation_seeds:
                    sampler_diagnostics.append(
                        _unavailable_sampler_diagnostic_row(
                            heldout_center=heldout,
                            candidate_expert=expert,
                            generation_seed=int(generation_seed),
                            generation_mode=generation_mode,
                            reason=_mode_unavailable_reason(
                                generation_mode,
                                banks=banks,
                                gmm_priors=gmm_priors,
                                expert=expert,
                                label_values=config.label_values,
                            ),
                            duplicate_eps=config.duplicate_eps,
                        )
                    )
            if heldout in candidate_experts:
                raise ProtocolError(f"Target expert {heldout} leaked into C3 candidate pool.")
            for generation_seed in config.generation_seeds:
                for expert in candidate_experts:
                    raw_batch, latent_stats = sample_family_c3_embeddings(
                        backend,
                        banks,
                        gmm_priors,
                        expert_domain=int(expert),
                        generation_seed=int(generation_seed),
                        budget_per_class=int(config.budget_per_class),
                        label_values=config.label_values,
                        generation_mode=generation_mode,
                    )
                    batch = _as_numpy_synthetic_batch(raw_batch)
                    real_source_x = val_x[val_domains == int(expert)]
                    real_source_y = val_labels[val_domains == int(expert)]
                    generation_manifest.append(
                        _generation_manifest_row_c3(
                            heldout,
                            expert,
                            int(generation_seed),
                            batch,
                            real_x=real_source_x,
                        )
                    )
                    sampler_diagnostics.append(
                        _sampler_diagnostic_row(
                            heldout_center=heldout,
                            candidate_expert=expert,
                            generation_seed=int(generation_seed),
                            generation_mode=generation_mode,
                            batch=batch,
                            latent_stats=latent_stats,
                            real_x=real_source_x,
                            real_labels=real_source_y,
                            duplicate_eps=config.duplicate_eps,
                            classifier_seed=int(config.classifier_seeds[0]),
                        )
                    )
                    for classifier_seed in config.classifier_seeds:
                        trained = _train_or_get_c3_classifier(
                            classifier_cache,
                            heldout_center=heldout,
                            candidate_expert=expert,
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            budget_per_class=int(config.budget_per_class),
                            generation_mode=generation_mode,
                            batch=batch,
                        )
                        classifier_manifest.append(
                            _classifier_manifest_row_c3(
                                heldout,
                                expert,
                                int(generation_seed),
                                int(classifier_seed),
                                trained,
                                generation_mode=generation_mode,
                            )
                        )
                        for split in unique_eval_contexts:
                            if split["heldout_center"] != heldout:
                                continue
                            downstream_rows.append(
                                _evaluate_matrix_row(
                                    heldout_center=heldout,
                                    candidate_expert=expert,
                                    trained=trained,
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    budget_per_class=int(config.budget_per_class),
                                    generation_mode=generation_mode,
                                    split=split,
                                    test_x=test_x,
                                    test_labels=test_labels,
                                    row_type=SINGLE_EXPERT_ROW_TYPE,
                                )
                            )

    source_transfer_audit = build_c3_source_transfer_sampler_prior_audit_rows(
        downstream_rows=downstream_rows,
        generation_modes=config.generation_modes,
        mode_tie_break_order=config.mode_tie_break_order,
    )
    alignment_rows = build_c3_source_transfer_selection_alignment_rows(
        source_transfer_audit_rows=source_transfer_audit,
        downstream_rows=downstream_rows,
    )
    c2_alignment = _read_csv(c2_root / "tables" / "family_c2_downstream_selection_alignment.csv")
    c2_baseline = _read_csv(c2_root / "tables" / "family_c2_downstream_baseline_comparison.csv")
    c2_matrix = read_family_c_downstream_matrix(c2_root / "tables" / "family_c2_all_expert_downstream_matrix.csv")
    fixed_comparison = build_c3_fixed_expert_generation_mode_comparison_rows(
        c2_rows=c2_matrix,
        c3_rows=downstream_rows,
    )
    selected_policy_comparison = build_c3_selected_policy_comparison_rows(
        c2_alignment_rows=c2_alignment,
        c3_alignment_rows=alignment_rows,
    )
    baseline_rows = build_c3_baseline_rows(
        c3_alignment_rows=alignment_rows,
        c2_baseline_rows=c2_baseline,
    )
    protocol_audit = _protocol_audit_rows_c3(protocol_rows, downstream_rows)
    decision_summary = classify_family_c3_decision(
        c3_alignment_rows=alignment_rows,
        c3_rows=downstream_rows,
        c2_alignment_rows=c2_alignment,
        c2_rows=c2_matrix,
        selected_policy_comparison=selected_policy_comparison,
        protocol_rows=protocol_audit,
        source_transfer_audit_rows=source_transfer_audit,
        workload=estimate_family_c3_workload(config),
    )

    _write_csv(manifests_dir / "family_c3_generation_manifest.csv", FAMILY_C3_GENERATION_MANIFEST_COLUMNS, generation_manifest)
    _write_csv(manifests_dir / "family_c3_trained_classifier_manifest.csv", FAMILY_C3_CLASSIFIER_MANIFEST_COLUMNS, _dedupe_rows(classifier_manifest))
    write_family_c_downstream_matrix(tables_dir / "family_c3_all_expert_downstream_matrix.csv", downstream_rows)
    _write_csv(tables_dir / "family_c3_downstream_selection_alignment.csv", FAMILY_C3_ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(tables_dir / "family_c3_downstream_baseline_comparison.csv", tuple(_ordered_keys(baseline_rows)), baseline_rows)
    _write_csv(tables_dir / "family_c3_source_transfer_sampler_prior_audit.csv", FAMILY_C3_SOURCE_TRANSFER_AUDIT_COLUMNS, source_transfer_audit)
    _write_csv(tables_dir / "family_c3_posterior_bank_provenance.csv", FAMILY_C3_POSTERIOR_BANK_COLUMNS, bank_rows)
    _write_csv(tables_dir / "family_c3_gmm_prior_diagnostics.csv", FAMILY_C3_GMM_DIAGNOSTIC_COLUMNS, gmm_rows)
    _write_csv(tables_dir / "family_c3_sampler_diagnostics.csv", FAMILY_C3_SAMPLER_DIAGNOSTIC_COLUMNS, sampler_diagnostics)
    _write_csv(tables_dir / "family_c3_fixed_expert_generation_mode_comparison.csv", FAMILY_C3_FIXED_COMPARISON_COLUMNS, fixed_comparison)
    _write_csv(tables_dir / "family_c3_selected_policy_comparison.csv", FAMILY_C3_SELECTED_POLICY_COMPARISON_COLUMNS, selected_policy_comparison)
    _write_csv(reports_out_dir / "family_c3_downstream_protocol_audit.csv", FAMILY_C3_PROTOCOL_AUDIT_COLUMNS, protocol_audit)
    _write_json(reports_out_dir / "family_c3_downstream_decision_summary.json", decision_summary)
    return {
        "status": "complete",
        "artifacts_root": str(artifacts_root),
        "n_downstream_rows": len(downstream_rows),
        "n_alignment_rows": len(alignment_rows),
        "n_fixed_comparison_rows": len(fixed_comparison),
        "n_selected_policy_comparison_rows": len(selected_policy_comparison),
        "decision": decision_summary.get("classification"),
        "oracle_status": decision_summary.get("oracle_status"),
    }


def build_c3_source_transfer_sampler_prior_audit_rows(
    *,
    downstream_rows: Sequence[FamilyCDownstreamRow],
    generation_modes: Sequence[str],
    mode_tie_break_order: Sequence[str],
    min_required_source_centers: int = 3,
) -> list[dict[str, object]]:
    valid_rows = [
        row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
        and row.generation_mode in set(generation_modes)
        and int(row.metric_valid_bacc) == 1
        and str(row.candidate_expert).isdigit()
        and not math.isnan(float(row.bacc))
    ]
    heldout_centers = sorted({str(row.heldout_center) for row in valid_rows}, key=lambda value: int(value))
    candidate_experts = sorted({str(row.candidate_expert) for row in valid_rows}, key=lambda value: int(value))
    mode_rank = {mode: index for index, mode in enumerate(mode_tie_break_order)}
    out: list[dict[str, object]] = []
    for heldout in heldout_centers:
        candidate_rows: list[dict[str, object]] = []
        for mode in generation_modes:
            for candidate in candidate_experts:
                if candidate == heldout:
                    continue
                grouped: dict[str, list[float]] = {}
                n_rows = 0
                for row in valid_rows:
                    if str(row.candidate_expert) != candidate or str(row.generation_mode) != mode:
                        continue
                    if str(row.heldout_center) in {heldout, candidate}:
                        continue
                    grouped.setdefault(str(row.heldout_center), []).append(float(row.bacc))
                    n_rows += 1
                source_scores = {source: _nanmean(values) for source, values in grouped.items()}
                values = [value for value in source_scores.values() if not math.isnan(value)]
                available = int(len(values) >= int(min_required_source_centers))
                candidate_rows.append(
                    {
                        "heldout_center": heldout,
                        "candidate_expert": candidate,
                        "generation_mode": mode,
                        "prior_score": _nanmean(values),
                        "prior_score_std_across_source_centers": _std(values),
                        "prior_score_min_across_source_centers": min(values) if values else math.nan,
                        "prior_score_max_across_source_centers": max(values) if values else math.nan,
                        "selected_expert": "",
                        "selected_generation_mode": "",
                        "n_source_centers_used": len(values),
                        "source_centers_used": "|".join(sorted(source_scores, key=lambda value: int(value))),
                        "n_rows_used": n_rows,
                        "min_required_source_centers": int(min_required_source_centers),
                        "coverage_ok": available,
                        "target_heldout_rows_used": 0,
                        "target_eval_labels_used": 0,
                        "target_heldout_rows_used_for_sampler_prior": 0,
                        "target_mode_oracle_used_for_selection": 0,
                        "target_expert_oracle_used_for_selection": 0,
                        "selection_source": "source_transfer_sampler_expert_prior_loto",
                        "available": available,
                    }
                )
        selectable = [
            row
            for row in candidate_rows
            if int(row["available"]) == 1 and not math.isnan(float(row["prior_score"]))
        ]
        selected_expert = ""
        selected_mode = ""
        if selectable:
            winner = max(
                selectable,
                key=lambda row: (
                    float(row["prior_score"]),
                    -_safe_std(row["prior_score_std_across_source_centers"]),
                    -int(str(row["candidate_expert"])),
                    -int(mode_rank.get(str(row["generation_mode"]), 999)),
                ),
            )
            selected_expert = str(winner["candidate_expert"])
            selected_mode = str(winner["generation_mode"])
        for row in candidate_rows:
            row["selected_expert"] = selected_expert
            row["selected_generation_mode"] = selected_mode
            out.append(row)
    return out


def build_c3_source_transfer_selection_alignment_rows(
    *,
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    all_mode_oracles = _compute_all_mode_oracles(downstream_rows)
    single_index = _single_index_all_modes(downstream_rows)
    selected_by_heldout: dict[str, tuple[str, str]] = {}
    for row in source_transfer_audit_rows:
        if int(float(row.get("available", 0) or 0)) != 1:
            continue
        if (
            str(row.get("candidate_expert", "")) == str(row.get("selected_expert", ""))
            and str(row.get("generation_mode", "")) == str(row.get("selected_generation_mode", ""))
        ):
            selected_by_heldout[str(row.get("heldout_center", ""))] = (
                str(row.get("selected_expert", "")),
                str(row.get("selected_generation_mode", "")),
            )
    out: list[dict[str, object]] = []
    for context, oracle in sorted(all_mode_oracles.items()):
        heldout, generation_seed, classifier_seed, budget, support_size, support_seed, split_id = context
        selected = selected_by_heldout.get(heldout)
        if selected is None:
            continue
        selected_expert, selected_mode = selected
        selected_row = single_index.get(
            (heldout, selected_expert, generation_seed, classifier_seed, budget, selected_mode, support_size, support_seed, split_id)
        )
        if selected_row is None:
            continue
        out.append(
            {
                "heldout_center": heldout,
                "method": FAMILY_C3_SOURCE_TRANSFER_METHOD,
                "selected_expert": selected_expert,
                "selected_generation_mode": selected_mode,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "budget_per_class": budget,
                "generation_mode": selected_mode,
                "support_size": support_size,
                "support_seed": support_seed,
                "support_eval_split_id": split_id,
                "selected_bacc": float(selected_row.bacc),
                "selected_macro_f1": float(selected_row.macro_f1),
                "downstream_oracle_expert": oracle["expert"],
                "downstream_oracle_generation_mode": oracle["generation_mode"],
                "oracle_bacc": oracle["bacc"],
                "oracle_macro_f1": oracle["macro_f1"],
                "downstream_oracle_gap_bacc": oracle["bacc"] - float(selected_row.bacc),
                "downstream_oracle_gap_macro_f1": oracle["macro_f1"] - float(selected_row.macro_f1),
                "top1_downstream_oracle_hit": int(
                    selected_expert == oracle["expert"] and selected_mode == oracle["generation_mode"]
                ),
                "spearman_neg_support_score_vs_bacc": math.nan,
                "available": 1,
                "selection_source": "source_transfer_sampler_expert_prior_loto",
            }
        )
    return out


def build_c3_fixed_expert_generation_mode_comparison_rows(
    *,
    c2_rows: Sequence[FamilyCDownstreamRow],
    c3_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    c2_index = {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in c2_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.generation_mode == FAMILY_C2_PRIMARY_GENERATION_MODE
    }
    out: list[dict[str, object]] = []
    for row in c3_rows:
        if row.row_type != SINGLE_EXPERT_ROW_TYPE:
            continue
        c2 = c2_index.get(
            (
                row.heldout_center,
                row.candidate_expert,
                row.generation_seed,
                row.classifier_seed,
                row.budget_per_class,
                row.support_size,
                row.support_seed,
                row.support_eval_split_id,
            )
        )
        if c2 is None:
            continue
        out.append(
            {
                "heldout_center": row.heldout_center,
                "candidate_expert": row.candidate_expert,
                "generation_seed": row.generation_seed,
                "classifier_seed": row.classifier_seed,
                "support_size": row.support_size,
                "support_seed": row.support_seed,
                "support_eval_split_id": row.support_eval_split_id,
                "c2_generation_mode": FAMILY_C2_PRIMARY_GENERATION_MODE,
                "c3_generation_mode": row.generation_mode,
                "bacc_c2_fitted_prior": float(c2.bacc),
                "bacc_c3": float(row.bacc),
                "delta_bacc_c3_minus_c2": float(row.bacc) - float(c2.bacc),
                "macro_f1_c2_fitted_prior": float(c2.macro_f1),
                "macro_f1_c3": float(row.macro_f1),
                "delta_macro_f1_c3_minus_c2": float(row.macro_f1) - float(c2.macro_f1),
            }
        )
    return out


def build_c3_selected_policy_comparison_rows(
    *,
    c2_alignment_rows: Sequence[Mapping[str, str]],
    c3_alignment_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    c3_index = {
        (
            str(row["heldout_center"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
            int(row["support_size"]),
            int(row["support_seed"]),
            str(row["support_eval_split_id"]),
        ): row
        for row in c3_alignment_rows
        if str(row.get("method")) == FAMILY_C3_SOURCE_TRANSFER_METHOD
    }
    out: list[dict[str, object]] = []
    for c2 in c2_alignment_rows:
        if str(c2.get("method")) != FAMILY_C_SOURCE_TRANSFER_METHOD:
            continue
        key = (
            str(c2["heldout_center"]),
            int(float(c2["generation_seed"])),
            int(float(c2["classifier_seed"])),
            int(float(c2["support_size"])),
            int(float(c2["support_seed"])),
            str(c2["support_eval_split_id"]),
        )
        c3 = c3_index.get(key)
        if c3 is None:
            continue
        c2_bacc = float(c2["selected_bacc"])
        c2_gap = float(c2["downstream_oracle_gap_bacc"])
        c3_bacc = float(c3["selected_bacc"])
        c3_gap = float(c3["downstream_oracle_gap_bacc"])
        out.append(
            {
                "heldout_center": key[0],
                "generation_seed": key[1],
                "classifier_seed": key[2],
                "support_size": key[3],
                "support_seed": key[4],
                "support_eval_split_id": key[5],
                "c2_method": FAMILY_C_SOURCE_TRANSFER_METHOD,
                "c2_selected_expert": str(c2["selected_expert"]),
                "c2_generation_mode": str(c2["generation_mode"]),
                "c2_selected_bacc": c2_bacc,
                "c2_oracle_gap_bacc": c2_gap,
                "c3_method": FAMILY_C3_SOURCE_TRANSFER_METHOD,
                "c3_selected_expert": str(c3["selected_expert"]),
                "c3_generation_mode": str(c3["selected_generation_mode"]),
                "c3_selected_bacc": c3_bacc,
                "c3_oracle_gap_bacc": c3_gap,
                "delta_bacc_c3_minus_c2": c3_bacc - c2_bacc,
                "delta_oracle_gap_c3_minus_c2": c3_gap - c2_gap,
            }
        )
    return out


def build_c3_baseline_rows(
    *,
    c3_alignment_rows: Sequence[Mapping[str, object]],
    c2_baseline_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    rows = [_summary_row(FAMILY_C3_SOURCE_TRANSFER_METHOD, "selection_method", c3_alignment_rows)]
    for row in c2_baseline_rows:
        converted: dict[str, object] = dict(row)
        converted["method"] = f"c2_{row.get('method', '')}"
        converted["row_type"] = str(row.get("row_type", "c2_reference"))
        rows.append(converted)
    return rows


def classify_family_c3_decision(
    *,
    c3_alignment_rows: Sequence[Mapping[str, object]],
    c3_rows: Sequence[FamilyCDownstreamRow],
    c2_alignment_rows: Sequence[Mapping[str, str]],
    c2_rows: Sequence[FamilyCDownstreamRow],
    selected_policy_comparison: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
    workload: Mapping[str, int],
) -> dict[str, object]:
    c2_selector = [row for row in c2_alignment_rows if row.get("method") == FAMILY_C_SOURCE_TRANSFER_METHOD]
    c3_selector = [row for row in c3_alignment_rows if row.get("method") == FAMILY_C3_SOURCE_TRANSFER_METHOD]
    c3_bacc_center = _center_level_mean(c3_selector, "selected_bacc")
    c3_gap_center = _center_level_mean(c3_selector, "downstream_oracle_gap_bacc")
    c2_bacc_center = _center_level_mean(c2_selector, "selected_bacc")
    c2_gap_center = _center_level_mean(c2_selector, "downstream_oracle_gap_bacc")
    delta_center = c3_bacc_center - c2_bacc_center
    gap_delta = c3_gap_center - c2_gap_center
    oracle_center = _fixed_mode_expert_oracle_center_mean(c3_rows)
    c2_oracle_center = _oracle_center_level_mean_c2(c2_rows)
    protocol_pass = _c3_protocol_pass(protocol_rows, source_transfer_audit_rows)
    downstream_strong = (
        protocol_pass
        and c3_bacc_center >= 0.80
        and delta_center >= 0.01
        and gap_delta <= 0.005
    )
    generation_oracle_strong = oracle_center >= 0.80
    generation_improved = (
        protocol_pass
        and delta_center >= 0.005
        and gap_delta <= 0.0
    )
    selector_bottleneck = generation_oracle_strong and c3_bacc_center < 0.80
    if downstream_strong:
        classification = "DOWNSTREAM_STRONG"
    elif selector_bottleneck:
        classification = "SELECTOR_BOTTLENECK"
    elif generation_improved:
        classification = "GENERATION_IMPROVED"
    elif protocol_pass and delta_center > 0:
        classification = "PROMISING_DIAGNOSTIC"
    elif protocol_pass:
        classification = "DIAGNOSTIC_ONLY"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "oracle_status": "GENERATION_ORACLE_STRONG" if generation_oracle_strong else "GENERATION_ORACLE_NOT_STRONG",
        "primary_method": FAMILY_C3_SOURCE_TRANSFER_METHOD,
        "primary_generation_modes": list(FAMILY_C3_GENERATION_MODES),
        "metrics": {
            "row_level_mean_bacc": _nanmean(float(row.get("selected_bacc", math.nan)) for row in c3_selector),
            "center_level_mean_bacc": c3_bacc_center,
            "center_level_delta_vs_c2": delta_center,
            "center_level_oracle_gap_bacc": c3_gap_center,
            "center_level_oracle_gap_delta_vs_c2": gap_delta,
            "c2_center_level_mean_bacc": c2_bacc_center,
            "c2_center_level_oracle_gap_bacc": c2_gap_center,
            "c3_fixed_mode_expert_oracle_center_level_mean_bacc": oracle_center,
            "c2_fixed_expert_oracle_center_level_mean_bacc": c2_oracle_center,
            "protocol_audit_pass": int(protocol_pass),
            "selected_policy_center_delta_bacc": _center_level_mean(
                selected_policy_comparison,
                "delta_bacc_c3_minus_c2",
            ),
            **{key: int(value) for key, value in workload.items()},
        },
        "decision_thresholds": {
            "downstream_strong_min_center_level_bacc": 0.80,
            "min_bacc_improvement_vs_c2": 0.01,
            "generation_improved_min_bacc_delta_vs_c2": 0.005,
            "max_allowed_oracle_gap_worsening": 0.005,
            "generation_oracle_strong_min_center_level_bacc": 0.80,
        },
        "claim_boundary": {
            "allowed": (
                "Richer source-train class-conditional latent sampling can improve synthetic "
                "embedding utility for independently trained label-conditioned CVAE experts."
            ),
            "forbidden": (
                "C3 does not prove full medical image generation quality, improve support-NELBO "
                "routing, or satisfy privacy release requirements for posterior-bank samplers."
            ),
        },
    }


def _generation_manifest_row_c3(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    batch: SyntheticBatch,
    *,
    real_x: object,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    row = _generation_manifest_row(heldout_center, candidate_expert, generation_seed, batch)
    x, _ = _batch_arrays(batch)
    real = np.asarray(real_x, dtype=float)
    row.update(
        {
            "generation_mode": str(batch.generation_mode),
            "generated_norm_mean": float(np.linalg.norm(x, axis=1).mean()) if x.size else math.nan,
            "generated_norm_std": float(np.linalg.norm(x, axis=1).std()) if x.size else math.nan,
            "real_source_norm_mean": float(np.linalg.norm(real, axis=1).mean()) if real.size else math.nan,
            "real_source_norm_std": float(np.linalg.norm(real, axis=1).std()) if real.size else math.nan,
        }
    )
    return row


def _classifier_manifest_row_c3(
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    trained: TrainedClassifier,
    *,
    generation_mode: str,
) -> dict[str, object]:
    return _classifier_manifest_row(
        heldout_center,
        candidate_expert,
        generation_seed,
        classifier_seed,
        trained,
        generation_mode=generation_mode,
    )


def _train_or_get_c3_classifier(
    cache: dict[tuple[object, ...], TrainedClassifier],
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generation_mode: str,
    batch: SyntheticBatch,
) -> TrainedClassifier:
    key = classifier_cache_key(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=int(generation_seed),
        classifier_seed=int(classifier_seed),
        budget_per_class=int(budget_per_class),
        generation_mode=generation_mode,
    )
    if key not in cache:
        x_syn, y_syn = _batch_arrays(batch)
        cache[key] = train_locked_synthetic_classifier(
            x_syn,
            y_syn,
            classifier_seed=int(classifier_seed),
        )
    return cache[key]


def _protocol_audit_rows_c3(
    protocol_rows: Sequence[Mapping[str, str]],
    downstream_rows: Sequence[FamilyCDownstreamRow],
) -> list[dict[str, object]]:
    base_rows = _protocol_audit_rows(protocol_rows, downstream_rows)
    out: list[dict[str, object]] = []
    for row in base_rows:
        out.append(
            {
                "heldout_center": row["heldout_center"],
                "support_size": row["support_size"],
                "support_seed": row["support_seed"],
                "support_eval_split_id": row["support_eval_split_id"],
                "target_expert_excluded": row["target_expert_excluded"],
                "support_eval_disjoint": row["support_eval_disjoint"],
                "support_labels_used_for_routing": row["support_labels_used_for_routing"],
                "routing_uses_eval_score": row["routing_uses_eval_score"],
                "posterior_bank_fit_split": "source_train",
                "target_support_labels_used_for_generation": 0,
                "target_eval_embeddings_used_for_generation": 0,
                "target_eval_labels_used_for_training": 0,
                "target_eval_labels_used_for_final_metric_only": 1,
                "target_heldout_rows_used_for_sampler_prior": 0,
                "target_mode_oracle_used_for_selection": 0,
                "target_expert_oracle_used_for_selection": 0,
                "metric_valid_bacc": row["metric_valid_bacc"],
                "metric_valid_macro_f1": row["metric_valid_macro_f1"],
            }
        )
    return out


def _c3_protocol_pass(
    protocol_rows: Sequence[Mapping[str, object]],
    source_transfer_audit_rows: Sequence[Mapping[str, object]],
) -> bool:
    if not protocol_rows or not source_transfer_audit_rows:
        return False
    for row in protocol_rows:
        if str(row.get("posterior_bank_fit_split")) != "source_train":
            return False
        required_one = ("target_expert_excluded", "support_eval_disjoint", "target_eval_labels_used_for_final_metric_only")
        required_zero = (
            "support_labels_used_for_routing",
            "routing_uses_eval_score",
            "target_support_labels_used_for_generation",
            "target_eval_embeddings_used_for_generation",
            "target_eval_labels_used_for_training",
            "target_heldout_rows_used_for_sampler_prior",
            "target_mode_oracle_used_for_selection",
            "target_expert_oracle_used_for_selection",
        )
        for key in required_one:
            if int(float(row.get(key, 0))) != 1:
                return False
        for key in required_zero:
            if int(float(row.get(key, 1))) != 0:
                return False
    for row in source_transfer_audit_rows:
        for key in (
            "target_heldout_rows_used",
            "target_eval_labels_used",
            "target_heldout_rows_used_for_sampler_prior",
            "target_mode_oracle_used_for_selection",
            "target_expert_oracle_used_for_selection",
        ):
            if int(float(row.get(key, 1))) != 0:
                return False
    return True


def _mode_available(
    generation_mode: str,
    *,
    banks: Mapping[tuple[str, int], PosteriorBank],
    gmm_priors: Mapping[tuple[str, int], GmmLatentPrior],
    expert: str,
    label_values: Sequence[int],
) -> bool:
    if generation_mode in {FAMILY_C3_BOOTSTRAP_MU_MODE, FAMILY_C3_BOOTSTRAP_T1_MODE}:
        return all(int(banks.get((str(expert), int(label)), _empty_bank(expert, label)).available) == 1 for label in label_values)
    if generation_mode == FAMILY_C3_GMM_MODE:
        return all(int(gmm_priors.get((str(expert), int(label)), _empty_gmm(expert, label)).available) == 1 for label in label_values)
    return False


def _empty_bank(expert: str, label: int) -> PosteriorBank:
    return PosteriorBank(str(expert), int(label), [], [], 0, 0, {}, {})


def _empty_gmm(expert: str, label: int) -> GmmLatentPrior:
    return GmmLatentPrior(str(expert), int(label), 0, [], [], [], 0, {})


def _stratified_bootstrap_indices(n_source: int, n_samples: int, rng: object) -> object:
    import numpy as np  # type: ignore

    if int(n_source) <= 0:
        raise ProtocolError("Cannot bootstrap from an empty posterior bank.")
    chunks: list[object] = []
    remaining = int(n_samples)
    while remaining > 0:
        perm = rng.permutation(int(n_source))
        take = min(remaining, int(n_source))
        chunks.append(perm[:take])
        remaining -= take
    return np.concatenate(chunks, axis=0)


def _latent_stats(latent_chunks: Sequence[object]) -> dict[str, float]:
    import numpy as np  # type: ignore

    z = np.concatenate([np.asarray(chunk, dtype=float) for chunk in latent_chunks], axis=0)
    norms = np.linalg.norm(z, axis=1)
    return {
        "latent_sample_norm_mean": float(norms.mean()) if norms.size else math.nan,
        "latent_sample_norm_std": float(norms.std()) if norms.size else math.nan,
    }


def _sampler_diagnostic_row(
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    generation_mode: str,
    batch: SyntheticBatch,
    latent_stats: Mapping[str, object],
    real_x: object,
    real_labels: object,
    duplicate_eps: float,
    classifier_seed: int,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    x, y = _batch_arrays(batch)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    real = np.asarray(real_x, dtype=float)
    real_y = np.asarray(real_labels, dtype=np.int64)
    train_acc = _generated_train_accuracy(x, y, classifier_seed=classifier_seed)
    real_acc = _real_source_accuracy_from_generated(x, y, real, real_y, classifier_seed=classifier_seed)
    return {
        "heldout_center": str(heldout_center),
        "candidate_expert": str(candidate_expert),
        "generation_seed": int(generation_seed),
        "generation_mode": str(generation_mode),
        "mode_available": 1,
        "unavailable_reason": "",
        "sampler_release_level": FAMILY_C3_SAMPLER_RELEASE_LEVEL,
        "generated_effective_rank": _effective_rank(x),
        "generated_cov_trace": _cov_trace(x),
        "generated_pairwise_distance_mean": _pairwise_distance_mean(x),
        "generated_duplicate_rate": _duplicate_rate(x, eps=float(duplicate_eps)),
        "duplicate_eps": float(duplicate_eps),
        "latent_sample_norm_mean": float(latent_stats.get("latent_sample_norm_mean", math.nan)),
        "latent_sample_norm_std": float(latent_stats.get("latent_sample_norm_std", math.nan)),
        "generated_class_centroid_distance": _class_centroid_distance(x, y),
        "generated_class_linear_probe_train_accuracy": train_acc,
        "real_source_val_accuracy_from_generated_train": real_acc,
    }


def _unavailable_sampler_diagnostic_row(
    *,
    heldout_center: str,
    candidate_expert: str,
    generation_seed: int,
    generation_mode: str,
    reason: str,
    duplicate_eps: float,
) -> dict[str, object]:
    return {
        "heldout_center": str(heldout_center),
        "candidate_expert": str(candidate_expert),
        "generation_seed": int(generation_seed),
        "generation_mode": str(generation_mode),
        "mode_available": 0,
        "unavailable_reason": str(reason),
        "sampler_release_level": FAMILY_C3_SAMPLER_RELEASE_LEVEL,
        "generated_effective_rank": math.nan,
        "generated_cov_trace": math.nan,
        "generated_pairwise_distance_mean": math.nan,
        "generated_duplicate_rate": math.nan,
        "duplicate_eps": float(duplicate_eps),
        "latent_sample_norm_mean": math.nan,
        "latent_sample_norm_std": math.nan,
        "generated_class_centroid_distance": math.nan,
        "generated_class_linear_probe_train_accuracy": math.nan,
        "real_source_val_accuracy_from_generated_train": math.nan,
    }


def _mode_unavailable_reason(
    generation_mode: str,
    *,
    banks: Mapping[tuple[str, int], PosteriorBank],
    gmm_priors: Mapping[tuple[str, int], GmmLatentPrior],
    expert: str,
    label_values: Sequence[int],
) -> str:
    if generation_mode in {FAMILY_C3_BOOTSTRAP_MU_MODE, FAMILY_C3_BOOTSTRAP_T1_MODE}:
        missing = [
            int(label)
            for label in label_values
            if int(banks.get((str(expert), int(label)), _empty_bank(expert, label)).available) != 1
        ]
        return "posterior_bank_unavailable_classes=" + "|".join(str(value) for value in missing)
    if generation_mode == FAMILY_C3_GMM_MODE:
        missing = [
            int(label)
            for label in label_values
            if int(gmm_priors.get((str(expert), int(label)), _empty_gmm(expert, label)).available) != 1
        ]
        return "gmm_unavailable_classes=" + "|".join(str(value) for value in missing)
    return "unknown_generation_mode"


def _generated_train_accuracy(x: object, y: object, *, classifier_seed: int) -> float:
    import numpy as np  # type: ignore

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int64)
    if len(set(y_arr.tolist())) < 2:
        return math.nan
    trained = train_locked_synthetic_classifier(x_arr, y_arr, classifier_seed=int(classifier_seed))
    pred = trained.classifier.predict(trained.scaler.transform(x_arr))
    return float((pred == y_arr).mean())


def _real_source_accuracy_from_generated(
    x: object,
    y: object,
    real_x: object,
    real_y: object,
    *,
    classifier_seed: int,
) -> float:
    import numpy as np  # type: ignore

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=np.int64)
    real_arr = np.asarray(real_x, dtype=float)
    real_labels = np.asarray(real_y, dtype=np.int64)
    if real_arr.size == 0 or len(set(y_arr.tolist())) < 2:
        return math.nan
    trained = train_locked_synthetic_classifier(x_arr, y_arr, classifier_seed=int(classifier_seed))
    pred = trained.classifier.predict(trained.scaler.transform(real_arr))
    return float((pred == real_labels).mean())


def _effective_rank(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return math.nan
    cov = np.cov(arr, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(math.exp(-float(np.sum(p * np.log(p)))))


def _cov_trace(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return math.nan
    return float(np.trace(np.cov(arr, rowvar=False)))


def _pairwise_distance_mean(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return math.nan
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    tri = dist[np.triu_indices(arr.shape[0], k=1)]
    return float(tri.mean()) if tri.size else math.nan


def _duplicate_rate(x: object, *, eps: float) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    dist[range(arr.shape[0]), range(arr.shape[0])] = math.inf
    return float((dist.min(axis=1) < float(eps)).mean())


def _class_centroid_distance(x: object, y: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=np.int64)
    values = sorted(set(labels.tolist()))
    if len(values) != 2:
        return math.nan
    a = arr[labels == values[0]].mean(axis=0)
    b = arr[labels == values[1]].mean(axis=0)
    return float(np.linalg.norm(a - b))


def _compute_all_mode_oracles(
    rows: Sequence[FamilyCDownstreamRow],
) -> dict[tuple[str, int, int, int, int, int, str], dict[str, object]]:
    grouped: dict[tuple[str, int, int, int, int, int, str], list[FamilyCDownstreamRow]] = {}
    for row in rows:
        if row.row_type != SINGLE_EXPERT_ROW_TYPE or int(row.metric_valid_bacc) != 1:
            continue
        key = (
            row.heldout_center,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        )
        grouped.setdefault(key, []).append(row)
    out: dict[tuple[str, int, int, int, int, int, str], dict[str, object]] = {}
    mode_rank = {mode: idx for idx, mode in enumerate(FAMILY_C3_MODE_TIE_BREAK_ORDER)}
    for key, group in grouped.items():
        winner = max(
            group,
            key=lambda row: (
                float(row.bacc),
                float(row.macro_f1),
                -int(row.candidate_expert),
                -int(mode_rank.get(row.generation_mode, 999)),
            ),
        )
        out[key] = {
            "expert": str(winner.candidate_expert),
            "generation_mode": str(winner.generation_mode),
            "bacc": float(winner.bacc),
            "macro_f1": float(winner.macro_f1),
        }
    return out


def _single_index_all_modes(
    rows: Sequence[FamilyCDownstreamRow],
) -> dict[tuple[str, str, int, int, int, str, int, int, str], FamilyCDownstreamRow]:
    return {
        (
            row.heldout_center,
            row.candidate_expert,
            row.generation_seed,
            row.classifier_seed,
            row.budget_per_class,
            row.generation_mode,
            row.support_size,
            row.support_seed,
            row.support_eval_split_id,
        ): row
        for row in rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
    }


def _fixed_mode_expert_oracle_center_mean(rows: Sequence[FamilyCDownstreamRow]) -> float:
    centers = sorted({row.heldout_center for row in rows}, key=lambda value: int(value))
    best_by_center: list[float] = []
    for center in centers:
        by_mode_expert: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row.heldout_center != center or row.row_type != SINGLE_EXPERT_ROW_TYPE:
                continue
            by_mode_expert.setdefault((row.generation_mode, row.candidate_expert), []).append(float(row.bacc))
        means = [_nanmean(values) for values in by_mode_expert.values()]
        if means:
            best_by_center.append(max(means))
    return _nanmean(best_by_center)


def _oracle_center_level_mean_c2(rows: Sequence[FamilyCDownstreamRow]) -> float:
    oracles = _compute_oracles_for_mode(rows, FAMILY_C2_PRIMARY_GENERATION_MODE)
    pseudo = [{"heldout_center": key[0], "oracle_bacc": oracle.bacc} for key, oracle in oracles.items()]
    return _center_level_mean(pseudo, "oracle_bacc")


def _std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    if not vals:
        return math.nan
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((value - mean) ** 2 for value in vals) / len(vals))


def _safe_std(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.inf
    return parsed if not math.isnan(parsed) else math.inf


def _missing_message(prefix: str, paths: Sequence[Path]) -> str:
    joined = "\n".join(f"  - {path}" for path in paths)
    return f"{prefix}:\n{joined}"
