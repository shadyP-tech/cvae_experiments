"""Family E1 direct embedding sampler downstream diagnostics.

Family E1 is a non-CVAE diagnostic baseline. It keeps the source-center
expert boundary, but fits direct class-conditional samplers in DINO embedding
space from source-train rows only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .downstream import fit_locked_logistic_classifier
from .generation import allocate_equal_total_ensemble_budget
from .matrix import (
    EmbeddingCache,
    TargetEvalPool,
    _domain,
    _experiment_seed_from_run,
    _failure_status,
    _label,
    _load_embedding_cache,
    _make_support_eval_split,
    _read_samples_manifest,
    _records_for_split,
    _sample_id,
    _to_numpy,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import ArtifactSyncError, ProtocolError
from .routing import SupportSelectionUnit, assert_target_excluded, parse_expert_scores_json
from .schemas import (
    CAMELYON17_CENTERS,
    CLASSIFIER_SEEDS,
    EXPERIMENT_SEEDS,
    GENERATION_SEEDS,
    SUPPORT_NELBO_METHOD,
    SUPPORT_SEEDS,
    SUPPORT_SIZES,
)


FAMILY_E1_NAME = "family_e1_direct_embedding_sampler_downstream_v1"

E1_GMM_MODE = "family_e1_class_conditional_gmm_diag_bic"
E1_KDE_MODE = "family_e1_class_conditional_kde_gaussian"
E1_SMOTE_MODE = "family_e1_class_conditional_smote_interpolate"
E1_BOOTSTRAP_MODE = "family_e1_source_bootstrap_upper_bound"
E1_REAL_SOURCE_MODE = "real_source_train_classifier_baseline"

E1_SYNTHETIC_MODES = (E1_GMM_MODE, E1_KDE_MODE, E1_SMOTE_MODE)
E1_ALL_MODES = E1_SYNTHETIC_MODES + (E1_BOOTSTRAP_MODE, E1_REAL_SOURCE_MODE)
E1_PRIMARY_GMM_MODES = (E1_GMM_MODE,)
E1_SENSITIVITY_MODES = (E1_GMM_MODE, E1_KDE_MODE, E1_SMOTE_MODE)

E1_GMM_SELECTOR = "family_e1_gmm_source_transfer_expert_prior"
E1_SAMPLER_SELECTOR = "family_e1_source_transfer_sampler_expert_prior"
E1_GMM_ENSEMBLE_METHOD = "family_e1_gmm_same_budget_ensemble"
E1_GMM_ORACLE_METHOD = "family_e1_gmm_fixed_expert_oracle"
E1_SAMPLER_ORACLE_METHOD = "family_e1_fixed_mode_expert_oracle"
E1_BOOTSTRAP_ORACLE_METHOD = "family_e1_bootstrap_upper_bound_oracle"
E1_REAL_SOURCE_ORACLE_METHOD = "family_e1_real_source_train_upper_bound_oracle"

E1_ENSEMBLE_EXPERT_ID = "__family_e1_gmm_same_budget_ensemble__"
E1_SINGLE_EXPERT_ROW_TYPE = "single_expert_sampler"
E1_METHOD_BASELINE_ROW_TYPE = "method_baseline"
E1_DIAGNOSTIC_UPPER_BOUND_ROW_TYPE = "diagnostic_upper_bound"
E1_SCHEMA_VERSION = "family_e1_direct_embedding_sampler_downstream_v1"

E1_MODE_ORDER = {
    E1_GMM_MODE: 0,
    E1_KDE_MODE: 1,
    E1_SMOTE_MODE: 2,
}

E1_RELEASE_LEVELS = {
    E1_GMM_MODE: "aggregate_statistics",
    E1_KDE_MODE: "per_sample_source_bank",
    E1_SMOTE_MODE: "per_sample_source_bank",
    E1_BOOTSTRAP_MODE: "per_sample_source_bank",
    E1_REAL_SOURCE_MODE: "real_source_non_synthetic",
}

E1_REQUIRED_OUTPUTS = (
    "family_e1_sampler_provenance.csv",
    "family_e1_sampler_diagnostics.csv",
    "family_e1_generation_manifest.csv",
    "family_e1_trained_classifier_manifest.csv",
    "family_e1_all_expert_downstream_matrix.csv",
    "family_e1_downstream_selection_alignment.csv",
    "family_e1_downstream_baseline_comparison.csv",
    "family_e1_source_transfer_sampler_prior_audit.csv",
    "family_e1_generation_mode_comparison_vs_c2.csv",
    "family_e1_downstream_protocol_audit.csv",
    "family_e1_downstream_decision_summary.json",
)

E1_MATRIX_COLUMNS = (
    "schema_version",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generation_mode",
    "budget_per_class",
    "generation_seed",
    "classifier_seed",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "row_type",
    "n_train",
    "n_target_eval",
    "target_eval_pool_id",
    "target_eval_label_counts_json",
    "target_eval_has_all_classes",
    "candidate_experts_hash",
    "sampler_release_level",
    "available",
    "status",
    "error_message",
)

E1_MATRIX_PRIMARY_KEY = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "candidate_expert",
    "generation_mode",
    "budget_per_class",
    "generation_seed",
    "classifier_seed",
    "row_type",
    "candidate_experts_hash",
)

E1_ALIGNMENT_COLUMNS = (
    "selector",
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed",
    "classifier_seed",
    "selected_mode",
    "selected_expert",
    "prior_score",
    "selected_bacc",
    "selected_macro_f1",
    "oracle_mode",
    "oracle_expert",
    "oracle_bacc",
    "oracle_macro_f1",
    "oracle_gap_bacc",
    "oracle_gap_macro_f1",
    "target_heldout_rows_used_for_source_transfer_prior",
    "available",
    "status",
)

E1_PRIOR_AUDIT_COLUMNS = (
    "heldout_center",
    "selector",
    "mode",
    "candidate_expert",
    "prior_score",
    "source_centers_used",
    "source_center_scores_json",
    "n_source_centers_used",
    "target_heldout_rows_used",
    "selected",
    "tie_break_mode_rank",
)

E1_PROTOCOL_AUDIT_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generation_mode",
    "sampler_fit_split",
    "target_expert_excluded",
    "support_eval_disjoint",
    "target_labels_used_for_sampler_fit",
    "target_support_labels_used_for_generation",
    "target_eval_embeddings_used_for_generation",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "target_eval_label_counts_json",
    "target_eval_has_all_classes",
    "target_oracle_used_for_selection",
    "target_heldout_rows_used_for_source_transfer_prior",
    "sampler_release_level",
    "available",
)


@dataclass(frozen=True)
class FamilyE1Config:
    dataset_name: str
    domain_key: str
    candidate_domains: tuple[str, ...]
    experiment_seeds: tuple[int, ...]
    support_sizes: tuple[int, ...]
    support_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    classifier_seeds: tuple[int, ...]
    class_labels: tuple[int, ...]
    budget_per_class: int
    modes: tuple[str, ...]
    support_selection_glob: str
    artifacts_root: str
    pca_enabled: bool
    pca_n_components: int
    gmm_k_candidates: tuple[int, ...]
    gmm_reg_covar: float
    gmm_valid_min_samples: int
    gmm_valid_samples_per_component: int
    kde_min_bandwidth: float
    smote_jitter_scale: float
    c2_artifacts_root: str


@dataclass(frozen=True)
class FamilyE1BuildLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FamilyE1SupportArtifacts:
    experiment_seed: int
    run_dir: Path
    train_cache: Path
    test_cache: Path
    samples_manifest: Path
    config_resolved: Path
    split_manifest: Path
    support_selection_path: Path


@dataclass(frozen=True)
class SourceClassData:
    source_center: str
    class_label: int
    embeddings: Any
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class SamplerFitResult:
    mode: str
    source_center: str
    class_label: int
    n_source_train: int
    embedding_dim: int
    available: bool
    release_level: str
    model: Any
    source_embeddings: Any
    source_sample_ids: tuple[str, ...]
    diagnostics: Mapping[str, object]
    error_message: str = ""


@dataclass(frozen=True)
class GeneratedBatch:
    embeddings: Any
    labels: tuple[int, ...]
    generation_rows: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FamilyE1MatrixRow:
    experiment_seed: int
    heldout_center: str
    support_size: int
    support_seed: int
    support_eval_split_id: str
    candidate_expert: str
    generation_mode: str
    budget_per_class: int
    generation_seed: int
    classifier_seed: int
    bacc: float
    macro_f1: float
    auroc: float = math.nan
    auprc: float = math.nan
    row_type: str = E1_SINGLE_EXPERT_ROW_TYPE
    n_train: int = 0
    n_target_eval: int = 0
    target_eval_pool_id: str = ""
    target_eval_label_counts_json: str = "{}"
    target_eval_has_all_classes: int = 0
    candidate_experts_hash: str = "__single_expert__"
    sampler_release_level: str = ""
    available: int = 1
    status: str = "ok"
    error_message: str = ""
    schema_version: str = E1_SCHEMA_VERSION

    def primary_key(self) -> tuple[object, ...]:
        return tuple(getattr(self, field) for field in E1_MATRIX_PRIMARY_KEY)

    def context_key(self) -> tuple[int, str, int, int, int, int]:
        return (
            int(self.experiment_seed),
            self.heldout_center,
            int(self.support_size),
            int(self.support_seed),
            int(self.generation_seed),
            int(self.classifier_seed),
        )

    def is_selector_eligible(self) -> bool:
        return (
            self.row_type == E1_SINGLE_EXPERT_ROW_TYPE
            and self.generation_mode in E1_SENSITIVITY_MODES
            and self.status == "ok"
            and int(self.available) == 1
        )

    def to_csv_row(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_seed": self.experiment_seed,
            "heldout_center": self.heldout_center,
            "support_size": self.support_size,
            "support_seed": self.support_seed,
            "support_eval_split_id": self.support_eval_split_id,
            "candidate_expert": self.candidate_expert,
            "generation_mode": self.generation_mode,
            "budget_per_class": self.budget_per_class,
            "generation_seed": self.generation_seed,
            "classifier_seed": self.classifier_seed,
            "bacc": self.bacc,
            "macro_f1": self.macro_f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "row_type": self.row_type,
            "n_train": self.n_train,
            "n_target_eval": self.n_target_eval,
            "target_eval_pool_id": self.target_eval_pool_id,
            "target_eval_label_counts_json": self.target_eval_label_counts_json,
            "target_eval_has_all_classes": self.target_eval_has_all_classes,
            "candidate_experts_hash": self.candidate_experts_hash,
            "sampler_release_level": self.sampler_release_level,
            "available": self.available,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class SourceTransferPrior:
    heldout_center: str
    selector: str
    scores: Mapping[tuple[str, str], float]
    source_center_scores: Mapping[tuple[str, str], Mapping[str, float]]


@dataclass(frozen=True)
class SourceTransferSelection:
    selector: str
    heldout_center: str
    mode: str
    expert: str
    prior_score: float


def default_family_e1_config() -> FamilyE1Config:
    return FamilyE1Config(
        dataset_name="camelyon17",
        domain_key="center",
        candidate_domains=CAMELYON17_CENTERS,
        experiment_seeds=EXPERIMENT_SEEDS,
        support_sizes=SUPPORT_SIZES,
        support_seeds=SUPPORT_SEEDS,
        generation_seeds=GENERATION_SEEDS,
        classifier_seeds=CLASSIFIER_SEEDS,
        class_labels=(0, 1),
        budget_per_class=128,
        modes=E1_ALL_MODES,
        support_selection_glob=(
            "cvae_testing/outputs/camelyon17/"
            "camelyon17_support_estimated_utility_routing_v2/"
            "support_utility_v2_seed*/reports/support_response_sample_selections.csv"
        ),
        artifacts_root=(
            "cvae_downstream_evaluation/artifacts/"
            "family_e1_direct_embedding_sampler_downstream_v1"
        ),
        pca_enabled=False,
        pca_n_components=64,
        gmm_k_candidates=(1, 2, 4, 8),
        gmm_reg_covar=1e-4,
        gmm_valid_min_samples=32,
        gmm_valid_samples_per_component=8,
        kde_min_bandwidth=1e-6,
        smote_jitter_scale=0.01,
        c2_artifacts_root="cvae_downstream_evaluation/artifacts",
    )


def load_family_e1_config(path: Path) -> FamilyE1Config:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_e1_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_e1_config()
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise ProtocolError("Family E1 config must decode to a mapping.")
    return _family_e1_config_from_mapping(loaded)


def assert_family_e1_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_E1_NAME}",
        E1_GMM_MODE,
        E1_KDE_MODE,
        E1_SMOTE_MODE,
        E1_BOOTSTRAP_MODE,
        E1_REAL_SOURCE_MODE,
        "covariance_type: diag",
        "k_candidates: [1, 2, 4, 8]",
        "reg_covar: 1e-4",
        "budget_per_class: 128",
        "enabled: false",
        "n_components: 64",
        E1_GMM_SELECTOR,
        E1_SAMPLER_SELECTOR,
        "real_source_non_synthetic",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"Family E1 config missing locked fields: {', '.join(missing)}")
    forbidden = (
        "conditional_cvae_decoder",
        "support_labels_for_generation: required",
        "target_eval_labels_for_selection",
        "pca_before_sampler:\n    enabled: true",
    )
    present = [snippet for snippet in forbidden if snippet in text]
    if present:
        raise ProtocolError(f"Family E1 config contains forbidden fields: {', '.join(present)}")


def valid_gmm_k_candidates(
    n_source_train: int,
    k_candidates: Sequence[int],
    *,
    min_samples: int = 32,
    samples_per_component: int = 8,
) -> tuple[int, ...]:
    return tuple(
        int(k)
        for k in k_candidates
        if int(k) > 0 and int(n_source_train) >= max(int(min_samples), int(samples_per_component) * int(k))
    )


def select_lowest_bic(bic_by_k: Mapping[int, float]) -> int:
    if not bic_by_k:
        raise ProtocolError("Cannot select GMM k without valid BIC values.")
    return min((int(k) for k in bic_by_k), key=lambda k: (float(bic_by_k[k]), int(k)))


def fit_family_e1_sampler_bank(
    *,
    train_cache: EmbeddingCache,
    config: FamilyE1Config,
    random_state: int = 0,
) -> dict[tuple[str, str, int], SamplerFitResult]:
    bank: dict[tuple[str, str, int], SamplerFitResult] = {}
    for source_center in config.candidate_domains:
        for class_label in config.class_labels:
            source = extract_source_class_data(
                train_cache=train_cache,
                source_center=str(source_center),
                class_label=int(class_label),
            )
            for mode in config.modes:
                if mode == E1_GMM_MODE:
                    result = fit_gmm_diag_bic_sampler(source, config=config, random_state=random_state)
                elif mode == E1_KDE_MODE:
                    result = fit_kde_gaussian_sampler(source, config=config)
                elif mode == E1_SMOTE_MODE:
                    result = fit_smote_interpolate_sampler(source, config=config)
                elif mode == E1_BOOTSTRAP_MODE:
                    result = fit_source_bootstrap_sampler(source)
                elif mode == E1_REAL_SOURCE_MODE:
                    result = fit_real_source_sampler(source)
                else:
                    raise ProtocolError(f"Unknown Family E1 mode: {mode}")
                bank[(mode, str(source_center), int(class_label))] = result
    return bank


def extract_source_class_data(
    *,
    train_cache: EmbeddingCache,
    source_center: str,
    class_label: int,
) -> SourceClassData:
    embeddings = _as_numpy_2d(train_cache.embeddings)
    indices = [
        idx
        for idx, row in enumerate(train_cache.metadata)
        if str(_domain(row)) == str(source_center) and _label(row) == int(class_label)
    ]
    sample_ids = tuple(str(_sample_id(train_cache.metadata[idx])) for idx in indices)
    return SourceClassData(
        source_center=str(source_center),
        class_label=int(class_label),
        embeddings=embeddings[indices],
        sample_ids=sample_ids,
    )


def candidate_experts_for_heldout(candidate_domains: Sequence[str], heldout_center: str) -> tuple[str, ...]:
    candidates = tuple(str(domain) for domain in candidate_domains if str(domain) != str(heldout_center))
    if str(heldout_center) in candidates:
        raise ProtocolError(f"Target expert {heldout_center} leaked into candidate pool.")
    if not candidates:
        raise ProtocolError("Family E1 candidate pool is empty after target exclusion.")
    return candidates


def fit_gmm_diag_bic_sampler(
    source: SourceClassData,
    *,
    config: FamilyE1Config,
    random_state: int = 0,
) -> SamplerFitResult:
    x = _as_numpy_2d(source.embeddings)
    n, dim = _shape2(x)
    valid = valid_gmm_k_candidates(
        n,
        config.gmm_k_candidates,
        min_samples=config.gmm_valid_min_samples,
        samples_per_component=config.gmm_valid_samples_per_component,
    )
    base = _base_fit_diagnostics(source, mode=E1_GMM_MODE, embedding_dim=dim)
    if not valid:
        return SamplerFitResult(
            mode=E1_GMM_MODE,
            source_center=source.source_center,
            class_label=source.class_label,
            n_source_train=n,
            embedding_dim=dim,
            available=False,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            model=None,
            source_embeddings=x,
            source_sample_ids=source.sample_ids,
            diagnostics={
                **base,
                "gmm_selected_k": "",
                "gmm_bic_by_k": "{}",
                "gmm_converged": 0,
                "gmm_n_iter": 0,
                "gmm_min_component_weight": math.nan,
                "gmm_cov_min": math.nan,
                "gmm_cov_max": math.nan,
            },
            error_message="unavailable_no_valid_k",
        )
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Family E1 GMM sampler requires scikit-learn.") from exc

    models: dict[int, Any] = {}
    bic_by_k: dict[int, float] = {}
    for k in valid:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=float(config.gmm_reg_covar),
            random_state=int(random_state),
        )
        model.fit(x)
        models[int(k)] = model
        bic_by_k[int(k)] = float(model.bic(x))
    selected_k = select_lowest_bic(bic_by_k)
    selected = models[selected_k]
    covariances = _as_numpy_2d(selected.covariances_)
    weights = _as_numpy_1d(selected.weights_)
    return SamplerFitResult(
        mode=E1_GMM_MODE,
        source_center=source.source_center,
        class_label=source.class_label,
        n_source_train=n,
        embedding_dim=dim,
        available=True,
        release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
        model=selected,
        source_embeddings=x,
        source_sample_ids=source.sample_ids,
        diagnostics={
            **base,
            "gmm_selected_k": int(selected_k),
            "gmm_bic_by_k": json.dumps({str(k): float(v) for k, v in sorted(bic_by_k.items())}, sort_keys=True),
            "gmm_converged": int(bool(getattr(selected, "converged_", False))),
            "gmm_n_iter": int(getattr(selected, "n_iter_", 0)),
            "gmm_min_component_weight": float(weights.min()) if weights.size else math.nan,
            "gmm_cov_min": float(covariances.min()) if covariances.size else math.nan,
            "gmm_cov_max": float(covariances.max()) if covariances.size else math.nan,
        },
    )


def fit_kde_gaussian_sampler(source: SourceClassData, *, config: FamilyE1Config) -> SamplerFitResult:
    x = _as_numpy_2d(source.embeddings)
    n, dim = _shape2(x)
    base = _base_fit_diagnostics(source, mode=E1_KDE_MODE, embedding_dim=dim)
    if n <= 0:
        return SamplerFitResult(
            mode=E1_KDE_MODE,
            source_center=source.source_center,
            class_label=source.class_label,
            n_source_train=n,
            embedding_dim=dim,
            available=False,
            release_level=E1_RELEASE_LEVELS[E1_KDE_MODE],
            model=None,
            source_embeddings=x,
            source_sample_ids=source.sample_ids,
            diagnostics={**base, "kde_bandwidth": math.nan},
            error_message="unavailable_empty_source_class",
        )
    try:
        from sklearn.neighbors import KernelDensity  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Family E1 KDE sampler requires scikit-learn.") from exc
    bandwidth = source_only_kde_bandwidth(x, min_bandwidth=float(config.kde_min_bandwidth))
    model = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    model.fit(x)
    return SamplerFitResult(
        mode=E1_KDE_MODE,
        source_center=source.source_center,
        class_label=source.class_label,
        n_source_train=n,
        embedding_dim=dim,
        available=True,
        release_level=E1_RELEASE_LEVELS[E1_KDE_MODE],
        model=model,
        source_embeddings=x,
        source_sample_ids=source.sample_ids,
        diagnostics={**base, "kde_bandwidth": float(bandwidth)},
    )


def fit_smote_interpolate_sampler(source: SourceClassData, *, config: FamilyE1Config) -> SamplerFitResult:
    x = _as_numpy_2d(source.embeddings)
    n, dim = _shape2(x)
    base = _base_fit_diagnostics(source, mode=E1_SMOTE_MODE, embedding_dim=dim)
    jitter_std = smote_jitter_std(x, jitter_scale=float(config.smote_jitter_scale))
    return SamplerFitResult(
        mode=E1_SMOTE_MODE,
        source_center=source.source_center,
        class_label=source.class_label,
        n_source_train=n,
        embedding_dim=dim,
        available=n > 0,
        release_level=E1_RELEASE_LEVELS[E1_SMOTE_MODE],
        model={"jitter_std": float(jitter_std)},
        source_embeddings=x,
        source_sample_ids=source.sample_ids,
        diagnostics={**base, "smote_jitter_std": float(jitter_std)},
        error_message="" if n > 0 else "unavailable_empty_source_class",
    )


def fit_source_bootstrap_sampler(source: SourceClassData) -> SamplerFitResult:
    x = _as_numpy_2d(source.embeddings)
    n, dim = _shape2(x)
    return SamplerFitResult(
        mode=E1_BOOTSTRAP_MODE,
        source_center=source.source_center,
        class_label=source.class_label,
        n_source_train=n,
        embedding_dim=dim,
        available=n > 0,
        release_level=E1_RELEASE_LEVELS[E1_BOOTSTRAP_MODE],
        model=None,
        source_embeddings=x,
        source_sample_ids=source.sample_ids,
        diagnostics=_base_fit_diagnostics(source, mode=E1_BOOTSTRAP_MODE, embedding_dim=dim),
        error_message="" if n > 0 else "unavailable_empty_source_class",
    )


def fit_real_source_sampler(source: SourceClassData) -> SamplerFitResult:
    x = _as_numpy_2d(source.embeddings)
    n, dim = _shape2(x)
    return SamplerFitResult(
        mode=E1_REAL_SOURCE_MODE,
        source_center=source.source_center,
        class_label=source.class_label,
        n_source_train=n,
        embedding_dim=dim,
        available=n > 0,
        release_level=E1_RELEASE_LEVELS[E1_REAL_SOURCE_MODE],
        model=None,
        source_embeddings=x,
        source_sample_ids=source.sample_ids,
        diagnostics=_base_fit_diagnostics(source, mode=E1_REAL_SOURCE_MODE, embedding_dim=dim),
        error_message="" if n > 0 else "unavailable_empty_source_class",
    )


def source_only_kde_bandwidth(source_embeddings: Any, *, min_bandwidth: float = 1e-6) -> float:
    x = _as_numpy_2d(source_embeddings)
    n, dim = _shape2(x)
    if n <= 1:
        return float(max(min_bandwidth, 1.0))
    median_dist = median_pairwise_distance(x)
    if math.isnan(median_dist) or median_dist <= 0.0:
        spread = float(_np().std(x))
        median_dist = spread if spread > 0.0 else 1.0
    scale = float(n) ** (-1.0 / float(dim + 4))
    return float(max(float(min_bandwidth), float(median_dist) * scale))


def smote_jitter_std(source_embeddings: Any, *, jitter_scale: float = 0.01) -> float:
    x = _as_numpy_2d(source_embeddings)
    n, dim = _shape2(x)
    if n <= 1 or dim <= 0:
        return 0.0
    median_dist = median_pairwise_distance(x)
    if math.isnan(median_dist) or median_dist <= 0.0:
        return 0.0
    return float(max(0.0, float(jitter_scale)) * median_dist / math.sqrt(float(dim)))


def generate_from_sampler(
    fit: SamplerFitResult,
    *,
    n_samples: int,
    seed: int,
) -> tuple[Any, Mapping[str, object]]:
    if int(n_samples) <= 0:
        raise ProtocolError("n_samples must be positive.")
    if not fit.available:
        raise ProtocolError(f"Sampler unavailable for {fit.mode}/{fit.source_center}/class{fit.class_label}")
    np = _np()
    rng = np.random.default_rng(int(seed))
    if fit.mode == E1_GMM_MODE:
        generated, _ = fit.model.sample(int(n_samples))
        return np.asarray(generated, dtype=float), {"smote_alpha_mean": math.nan}
    if fit.mode == E1_KDE_MODE:
        generated = fit.model.sample(int(n_samples), random_state=int(seed))
        return np.asarray(generated, dtype=float), {"smote_alpha_mean": math.nan}
    source = _as_numpy_2d(fit.source_embeddings)
    n_source = int(source.shape[0])
    if fit.mode == E1_SMOTE_MODE:
        idx_a = rng.integers(0, n_source, size=int(n_samples))
        if n_source > 1:
            idx_b = rng.integers(0, n_source, size=int(n_samples))
        else:
            idx_b = idx_a
        alpha = rng.random(int(n_samples))
        generated = (1.0 - alpha[:, None]) * source[idx_a] + alpha[:, None] * source[idx_b]
        jitter_std = float((fit.model or {}).get("jitter_std", 0.0))
        if jitter_std > 0.0:
            generated = generated + rng.normal(0.0, jitter_std, size=generated.shape)
        return generated, {"smote_alpha_mean": float(alpha.mean()) if alpha.size else math.nan}
    if fit.mode == E1_BOOTSTRAP_MODE:
        idx = rng.integers(0, n_source, size=int(n_samples))
        return source[idx].copy(), {"smote_alpha_mean": math.nan}
    raise ProtocolError(f"Mode {fit.mode} does not generate synthetic embeddings.")


def generate_class_balanced_batch(
    *,
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    mode: str,
    source_center: str,
    class_labels: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
) -> GeneratedBatch:
    np = _np()
    chunks: list[Any] = []
    labels: list[int] = []
    generation_rows: list[Mapping[str, object]] = []
    real_by_class: dict[int, Any] = {}
    extra_diag: dict[str, object] = {}
    for class_label in sorted(int(v) for v in class_labels):
        fit = sampler_bank[(mode, str(source_center), int(class_label))]
        generated, diag = generate_from_sampler(
            fit,
            n_samples=int(budget_per_class),
            seed=int(generation_seed) + int(class_label) * 104729,
        )
        chunks.append(generated)
        labels.extend([int(class_label)] * int(budget_per_class))
        real_by_class[int(class_label)] = _as_numpy_2d(fit.source_embeddings)
        for key, value in diag.items():
            if key not in extra_diag or math.isnan(float(extra_diag.get(key, math.nan))):
                extra_diag[key] = value
        generation_rows.append(
            {
                "source_center": source_center,
                "generation_mode": mode,
                "class_label": class_label,
                "generation_seed": int(generation_seed),
                "budget_per_class": int(budget_per_class),
                "n_generated": int(generated.shape[0]),
                "n_source_train": int(fit.n_source_train),
                "sampler_release_level": fit.release_level,
                "available": int(fit.available),
                "source_sample_ids_hash": _hash_values(fit.source_sample_ids),
            }
        )
    embeddings = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=float)
    diagnostics = {
        **generation_diagnostics(
            generated_embeddings=embeddings,
            generated_labels=labels,
            real_source_by_class=real_by_class,
        ),
        **extra_diag,
    }
    return GeneratedBatch(
        embeddings=embeddings,
        labels=tuple(labels),
        generation_rows=tuple(generation_rows),
        diagnostics=diagnostics,
    )


def build_real_source_train_batch(
    *,
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    source_center: str,
    class_labels: Sequence[int],
) -> GeneratedBatch:
    np = _np()
    chunks: list[Any] = []
    labels: list[int] = []
    real_by_class: dict[int, Any] = {}
    generation_rows: list[Mapping[str, object]] = []
    for class_label in sorted(int(v) for v in class_labels):
        fit = sampler_bank[(E1_REAL_SOURCE_MODE, str(source_center), int(class_label))]
        if not fit.available:
            raise ProtocolError(f"Real-source train class is unavailable for source {source_center}, label {class_label}")
        x = _as_numpy_2d(fit.source_embeddings)
        chunks.append(x)
        labels.extend([int(class_label)] * int(x.shape[0]))
        real_by_class[int(class_label)] = x
        generation_rows.append(
            {
                "source_center": source_center,
                "generation_mode": E1_REAL_SOURCE_MODE,
                "class_label": class_label,
                "generation_seed": "",
                "budget_per_class": "",
                "n_generated": int(x.shape[0]),
                "n_source_train": int(fit.n_source_train),
                "sampler_release_level": fit.release_level,
                "available": int(fit.available),
                "source_sample_ids_hash": _hash_values(fit.source_sample_ids),
            }
        )
    embeddings = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=float)
    return GeneratedBatch(
        embeddings=embeddings,
        labels=tuple(labels),
        generation_rows=tuple(generation_rows),
        diagnostics=generation_diagnostics(
            generated_embeddings=embeddings,
            generated_labels=labels,
            real_source_by_class=real_by_class,
        ),
    )


def family_e1_source_transfer_prior(
    rows: Sequence[FamilyE1MatrixRow],
    *,
    heldout_center: str,
    candidate_experts: Sequence[str],
    modes: Sequence[str],
    selector: str,
) -> SourceTransferPrior:
    candidate_set = {str(v) for v in candidate_experts}
    mode_set = {str(v) for v in modes}
    if str(heldout_center) in candidate_set:
        raise ProtocolError("Target expert leaked into source-transfer prior candidate set.")

    source_grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if not row.is_selector_eligible():
            continue
        source_center = str(row.heldout_center)
        expert = str(row.candidate_expert)
        mode = str(row.generation_mode)
        if source_center == str(heldout_center):
            continue
        if source_center == expert:
            continue
        if expert not in candidate_set or mode not in mode_set:
            continue
        source_grouped.setdefault((source_center, mode, expert), []).append(float(row.bacc))

    by_candidate_source: dict[tuple[str, str], dict[str, float]] = {}
    for (source_center, mode, expert), values in source_grouped.items():
        by_candidate_source.setdefault((mode, expert), {})[source_center] = _nanmean(values)

    scores: dict[tuple[str, str], float] = {}
    for mode in mode_set:
        for expert in candidate_set:
            per_source = by_candidate_source.get((mode, expert), {})
            scores[(mode, expert)] = _nanmean(per_source.values())
    return SourceTransferPrior(
        heldout_center=str(heldout_center),
        selector=str(selector),
        scores=scores,
        source_center_scores=by_candidate_source,
    )


def select_source_transfer_candidate(
    prior: SourceTransferPrior,
    *,
    modes: Sequence[str],
    candidate_experts: Sequence[str],
) -> SourceTransferSelection:
    mode_set = tuple(str(v) for v in modes)
    candidate_set = tuple(str(v) for v in candidate_experts)
    choices: list[tuple[float, int, int, str, str]] = []
    for mode in mode_set:
        for expert in candidate_set:
            score = float(prior.scores.get((mode, expert), math.nan))
            if math.isnan(score):
                continue
            choices.append((-score, int(E1_MODE_ORDER.get(mode, 999)), _expert_sort_value(expert), mode, expert))
    if not choices:
        raise ProtocolError(f"No source-transfer prior candidates available for heldout={prior.heldout_center}")
    _, _, _, mode, expert = min(choices)
    return SourceTransferSelection(
        selector=prior.selector,
        heldout_center=prior.heldout_center,
        mode=mode,
        expert=expert,
        prior_score=float(prior.scores[(mode, expert)]),
    )


def build_family_e1_alignment_rows(
    *,
    rows: Sequence[FamilyE1MatrixRow],
    candidate_domains: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    row_by_key = {
        (
            row.context_key(),
            row.generation_mode,
            row.candidate_expert,
        ): row
        for row in rows
        if row.is_selector_eligible()
    }
    contexts = sorted({row.context_key() for row in rows if row.is_selector_eligible()})
    alignment: list[dict[str, object]] = []
    prior_audit: list[dict[str, object]] = []
    for context in contexts:
        _, heldout, _, _, _, _ = context
        candidates = candidate_experts_for_heldout(candidate_domains, heldout)
        selector_specs = (
            (E1_GMM_SELECTOR, E1_PRIMARY_GMM_MODES),
            (E1_SAMPLER_SELECTOR, E1_SENSITIVITY_MODES),
        )
        for selector, modes in selector_specs:
            prior = family_e1_source_transfer_prior(
                rows,
                heldout_center=heldout,
                candidate_experts=candidates,
                modes=modes,
                selector=selector,
            )
            selection = select_source_transfer_candidate(
                prior,
                modes=modes,
                candidate_experts=candidates,
            )
            selected_row = row_by_key.get((context, selection.mode, selection.expert))
            if selected_row is None:
                status = "missing_selected_row"
                selected_bacc = math.nan
                selected_macro = math.nan
                available = 0
            else:
                status = selected_row.status
                selected_bacc = float(selected_row.bacc)
                selected_macro = float(selected_row.macro_f1)
                available = int(selected_row.available)
            oracle = _oracle_for_context(rows, context=context, modes=modes)
            alignment.append(
                {
                    "selector": selector,
                    "heldout_center": heldout,
                    "experiment_seed": context[0],
                    "support_size": context[2],
                    "support_seed": context[3],
                    "support_eval_split_id": selected_row.support_eval_split_id if selected_row else "",
                    "generation_seed": context[4],
                    "classifier_seed": context[5],
                    "selected_mode": selection.mode,
                    "selected_expert": selection.expert,
                    "prior_score": selection.prior_score,
                    "selected_bacc": selected_bacc,
                    "selected_macro_f1": selected_macro,
                    "oracle_mode": oracle.generation_mode if oracle else "",
                    "oracle_expert": oracle.candidate_expert if oracle else "",
                    "oracle_bacc": float(oracle.bacc) if oracle else math.nan,
                    "oracle_macro_f1": float(oracle.macro_f1) if oracle else math.nan,
                    "oracle_gap_bacc": float(oracle.bacc) - selected_bacc if oracle else math.nan,
                    "oracle_gap_macro_f1": float(oracle.macro_f1) - selected_macro if oracle else math.nan,
                    "target_heldout_rows_used_for_source_transfer_prior": 0,
                    "available": available,
                    "status": status,
                }
            )
            prior_audit.extend(
                _prior_audit_rows(prior, selection=selection, modes=modes, candidate_experts=candidates)
            )
    return alignment, prior_audit


def build_family_e1_baseline_comparison_rows(
    *,
    rows: Sequence[FamilyE1MatrixRow],
    alignment_rows: Sequence[Mapping[str, object]],
    c2_comparison: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    c2 = dict(c2_comparison or {})
    output: list[dict[str, object]] = []
    for selector in (E1_GMM_SELECTOR, E1_SAMPLER_SELECTOR):
        subset = [row for row in alignment_rows if row.get("selector") == selector]
        output.append(
            _comparison_row(
                method=selector,
                row_type="selector",
                mean_bacc=center_level_mean(subset, metric="selected_bacc"),
                mean_macro_f1=center_level_mean(subset, metric="selected_macro_f1"),
                mean_oracle_gap_bacc=center_level_mean(subset, metric="oracle_gap_bacc"),
                available=1 if subset else 0,
            )
        )
    oracle_specs = (
        (E1_GMM_ORACLE_METHOD, (E1_GMM_MODE,)),
        (E1_SAMPLER_ORACLE_METHOD, E1_SENSITIVITY_MODES),
        (E1_BOOTSTRAP_ORACLE_METHOD, (E1_BOOTSTRAP_MODE,)),
        (E1_REAL_SOURCE_ORACLE_METHOD, (E1_REAL_SOURCE_MODE,)),
    )
    for method, modes in oracle_specs:
        oracle_rows = _oracle_rows_by_context(rows, modes=modes)
        output.append(
            _comparison_row(
                method=method,
                row_type="diagnostic_oracle",
                mean_bacc=center_level_mean(oracle_rows, metric="bacc"),
                mean_macro_f1=center_level_mean(oracle_rows, metric="macro_f1"),
                mean_oracle_gap_bacc=0.0 if oracle_rows else math.nan,
                available=1 if oracle_rows else 0,
            )
        )
    ensemble_rows = [
        row for row in rows if row.candidate_expert == E1_ENSEMBLE_EXPERT_ID and row.status == "ok"
    ]
    output.append(
        _comparison_row(
            method=E1_GMM_ENSEMBLE_METHOD,
            row_type=E1_METHOD_BASELINE_ROW_TYPE,
            mean_bacc=center_level_mean(ensemble_rows, metric="bacc"),
            mean_macro_f1=center_level_mean(ensemble_rows, metric="macro_f1"),
            mean_oracle_gap_bacc=math.nan,
            available=1 if ensemble_rows else 0,
        )
    )
    for method in (
        "C2 source-transfer selector",
        "C2 fixed-expert oracle",
        "C3 fixed mode+expert oracle",
        "Family D fixed-expert oracle",
        "Family C label-marginal selector",
        "metadata/static/random/source-global baselines",
    ):
        output.append(
            _comparison_row(
                method=method,
                row_type="external_baseline",
                mean_bacc=float(c2.get(method, math.nan)),
                mean_macro_f1=math.nan,
                mean_oracle_gap_bacc=math.nan,
                available=1 if method in c2 else 0,
            )
        )
    return output


def classify_family_e1_decision(
    *,
    rows: Sequence[FamilyE1MatrixRow],
    alignment_rows: Sequence[Mapping[str, object]],
    protocol_audit_rows: Sequence[Mapping[str, object]],
    c2_metrics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    c2 = dict(c2_metrics or {})
    protocol_pass = int(protocol_audit_pass(protocol_audit_rows))
    gmm_selected = [row for row in alignment_rows if row.get("selector") == E1_GMM_SELECTOR]
    sampler_selected = [row for row in alignment_rows if row.get("selector") == E1_SAMPLER_SELECTOR]
    gmm_selected_bacc = center_level_mean(gmm_selected, metric="selected_bacc")
    sampler_selected_bacc = center_level_mean(sampler_selected, metric="selected_bacc")
    gmm_gap = center_level_mean(gmm_selected, metric="oracle_gap_bacc")
    gmm_oracle_rows = _oracle_rows_by_context(rows, modes=(E1_GMM_MODE,))
    sampler_oracle_rows = _oracle_rows_by_context(rows, modes=E1_SENSITIVITY_MODES)
    bootstrap_oracle_rows = _oracle_rows_by_context(rows, modes=(E1_BOOTSTRAP_MODE,))
    real_oracle_rows = _oracle_rows_by_context(rows, modes=(E1_REAL_SOURCE_MODE,))
    gmm_oracle_bacc = center_level_mean(gmm_oracle_rows, metric="bacc")
    sampler_oracle_bacc = center_level_mean(sampler_oracle_rows, metric="bacc")
    bootstrap_oracle_bacc = center_level_mean(bootstrap_oracle_rows, metric="bacc")
    real_oracle_bacc = center_level_mean(real_oracle_rows, metric="bacc")
    c2_selected = float(c2.get("c2_selected_center_level_mean_bacc", math.nan))
    c2_oracle = float(c2.get("c2_oracle_center_level_mean_bacc", math.nan))
    c2_gap = float(c2.get("c2_oracle_gap_bacc", math.nan))
    delta_vs_c2 = gmm_selected_bacc - c2_selected if not math.isnan(c2_selected) else math.nan
    gmm_oracle_delta_vs_c2 = gmm_oracle_bacc - c2_oracle if not math.isnan(c2_oracle) else math.nan
    gap_delta_vs_c2 = gmm_gap - c2_gap if not math.isnan(c2_gap) else math.nan

    classification = "DIAGNOSTIC_ONLY"
    pass_fail = "FAIL"
    if (
        protocol_pass
        and gmm_selected_bacc >= 0.80
        and not math.isnan(delta_vs_c2)
        and delta_vs_c2 >= 0.02
        and not math.isnan(gap_delta_vs_c2)
        and gap_delta_vs_c2 <= 0.005
    ):
        classification = "DOWNSTREAM_STRONG"
        pass_fail = "PASS"
    elif gmm_oracle_bacc >= 0.80:
        classification = "GENERATION_ORACLE_STRONG"
        pass_fail = "PASS"
    elif (
        protocol_pass
        and (
            (not math.isnan(gmm_oracle_delta_vs_c2) and gmm_oracle_delta_vs_c2 >= 0.01)
            or (not math.isnan(delta_vs_c2) and delta_vs_c2 >= 0.005)
        )
    ):
        classification = "GENERATION_IMPROVED"
        pass_fail = "PASS"
    elif sampler_selected_bacc >= 0.80 and gmm_selected_bacc < 0.80:
        classification = "SENSITIVITY_STRONG"
        pass_fail = "PASS"
    elif max(_nan_to_neg_inf(bootstrap_oracle_bacc), _nan_to_neg_inf(real_oracle_bacc)) >= 0.80 and max(
        _nan_to_neg_inf(gmm_selected_bacc), _nan_to_neg_inf(sampler_selected_bacc), _nan_to_neg_inf(sampler_oracle_bacc)
    ) < 0.80:
        classification = "BOOTSTRAP_ONLY_UPPER_BOUND"
        pass_fail = "FAIL"
    elif sampler_oracle_bacc >= 0.80 and max(_nan_to_neg_inf(gmm_selected_bacc), _nan_to_neg_inf(sampler_selected_bacc)) < 0.80:
        classification = "SELECTOR_BOTTLENECK"
        pass_fail = "FAIL"

    recommendation = ""
    weak_gmm = (
        gmm_oracle_bacc < 0.80
        and any(
            row.generation_mode == E1_GMM_MODE
            and row.available == 0
            or (
                row.generation_mode == E1_GMM_MODE
                and row.status != "ok"
            )
            for row in rows
        )
    )
    low_ratio = _low_effective_sample_ratio(rows)
    if weak_gmm or low_ratio:
        recommendation = "Follow-up E1.1 PCA-to-64 + GMM is recommended; v1 kept PCA disabled."
    return {
        "schema_version": E1_SCHEMA_VERSION,
        "decision_classification": classification,
        "pass_fail": pass_fail,
        "protocol_audit_pass": protocol_pass,
        "metrics": {
            "gmm_selected_center_level_mean_bacc": gmm_selected_bacc,
            "gmm_selected_delta_vs_c2": delta_vs_c2,
            "gmm_selected_oracle_gap_bacc": gmm_gap,
            "gmm_oracle_center_level_mean_bacc": gmm_oracle_bacc,
            "gmm_oracle_delta_vs_c2": gmm_oracle_delta_vs_c2,
            "sampler_selector_center_level_mean_bacc": sampler_selected_bacc,
            "sampler_oracle_center_level_mean_bacc": sampler_oracle_bacc,
            "bootstrap_oracle_center_level_mean_bacc": bootstrap_oracle_bacc,
            "real_source_oracle_center_level_mean_bacc": real_oracle_bacc,
        },
        "claim_boundary": (
            "Family E1 is a non-CVAE diagnostic baseline. GMM is the only "
            "primary thesis-facing E1 mode; KDE, SMOTE, bootstrap, and "
            "real-source rows are sensitivity or upper-bound diagnostics."
        ),
        "recommendation": recommendation,
    }


def build_family_e1_all_expert_downstream_matrix(
    *,
    config: FamilyE1Config,
    repo_root: Path,
    artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    resume: bool = False,
    limits: FamilyE1BuildLimits = FamilyE1BuildLimits(),
) -> dict[str, Path]:
    artifacts = discover_family_e1_support_artifacts(config=config, repo_root=repo_root)
    artifacts = _limit_artifacts(artifacts, limits.experiment_seeds)
    support_units = _limit_support_units(support_units, limits)
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldouts = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    matrix_path = artifacts_root / "family_e1_all_expert_downstream_matrix.csv"
    completed = _read_completed_e1_keys(matrix_path) if resume else set()

    matrix_rows: list[FamilyE1MatrixRow] = []
    provenance_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []

    units_by_seed = _support_units_by_seed(support_units)
    for artifact in artifacts:
        seed_units = units_by_seed.get(int(artifact.experiment_seed), ())
        if not seed_units:
            raise ProtocolError(f"No Family E1 support contexts for experiment_seed={artifact.experiment_seed}")
        samples = _read_samples_manifest(artifact.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(artifact.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(artifact.test_cache, test_records, repo_root=repo_root)
        sampler_bank = fit_family_e1_sampler_bank(train_cache=train_cache, config=config)
        provenance_rows.extend(_sampler_provenance_rows(sampler_bank, experiment_seed=artifact.experiment_seed))
        diagnostic_rows.extend(_sampler_fit_diagnostic_rows(sampler_bank, experiment_seed=artifact.experiment_seed))

        for heldout in selected_heldouts:
            heldout = str(heldout)
            if heldout not in {str(v) for v in config.candidate_domains}:
                raise ProtocolError(f"Unknown Family E1 heldout center: {heldout}")
            candidates = candidate_experts_for_heldout(config.candidate_domains, heldout)
            heldout_units = sorted(
                (
                    unit
                    for unit in seed_units
                    if unit.heldout_center == heldout
                    and (limits.support_sizes is None or int(unit.support_size) in set(limits.support_sizes))
                    and (limits.support_seeds is None or int(unit.support_seed) in set(limits.support_seeds))
                ),
                key=lambda u: (int(u.support_size), int(u.support_seed), u.support_eval_split_id),
            )
            if not heldout_units:
                raise ProtocolError(
                    f"No Family E1 support contexts for seed={artifact.experiment_seed}, heldout={heldout}"
                )
            for unit in heldout_units:
                target_pool = build_family_e1_target_eval_pool(
                    test_metadata=test_cache.metadata,
                    heldout_center=heldout,
                    support_size=int(unit.support_size),
                    support_seed=int(unit.support_seed),
                    support_eval_split_id=unit.support_eval_split_id,
                )
                target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
                if not target_labels:
                    raise ProtocolError(
                        f"Family E1 target eval pool is empty for seed={artifact.experiment_seed}, "
                        f"heldout={heldout}, support_size={unit.support_size}, support_seed={unit.support_seed}"
                    )
                if not set(target_labels).issubset(set(config.class_labels)):
                    raise ProtocolError(
                        f"Family E1 expects target labels {config.class_labels}, got {sorted(set(target_labels))}"
                    )
                target_embeddings = _as_numpy_2d(test_cache.embeddings)[list(target_pool.eval_indices)]
                for candidate in candidates:
                    for mode in config.modes:
                        for generation_seed in selected_generation_seeds:
                            for classifier_seed in selected_classifier_seeds:
                                row, generated_meta, diag, clf_manifest, audit = score_family_e1_candidate(
                                    sampler_bank=sampler_bank,
                                    experiment_seed=artifact.experiment_seed,
                                    heldout_center=heldout,
                                    support_unit=unit,
                                    candidate_expert=candidate,
                                    mode=mode,
                                    budget_per_class=int(config.budget_per_class),
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    class_labels=config.class_labels,
                                    target_embeddings=target_embeddings,
                                    target_labels=target_labels,
                                    target_pool=target_pool,
                                )
                                if resume and row.primary_key() in completed:
                                    continue
                                matrix_rows.append(row)
                                completed.add(row.primary_key())
                                generation_rows.extend(generated_meta)
                                diagnostic_rows.append(diag)
                                classifier_rows.append(clf_manifest)
                                protocol_rows.append(audit)
                for generation_seed in selected_generation_seeds:
                    for classifier_seed in selected_classifier_seeds:
                        row, generated_meta, diag, clf_manifest, audit = score_family_e1_gmm_same_budget_ensemble(
                            sampler_bank=sampler_bank,
                            experiment_seed=artifact.experiment_seed,
                            heldout_center=heldout,
                            support_unit=unit,
                            candidate_experts=candidates,
                            budget_per_class=int(config.budget_per_class),
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            class_labels=config.class_labels,
                            target_embeddings=target_embeddings,
                            target_labels=target_labels,
                            target_pool=target_pool,
                        )
                        if resume and row.primary_key() in completed:
                            continue
                        matrix_rows.append(row)
                        completed.add(row.primary_key())
                        generation_rows.extend(generated_meta)
                        diagnostic_rows.append(diag)
                        classifier_rows.append(clf_manifest)
                        protocol_rows.append(audit)

    if resume and matrix_path.exists():
        existing = read_family_e1_matrix(matrix_path)
        matrix_rows = existing + matrix_rows

    validate_family_e1_matrix(matrix_rows)
    validate_family_e1_protocol_audit(protocol_rows)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "family_e1_sampler_provenance.csv", _provenance_columns(), provenance_rows)
    _write_csv(artifacts_root / "family_e1_sampler_diagnostics.csv", _diagnostic_columns(), diagnostic_rows)
    _write_csv(artifacts_root / "family_e1_generation_manifest.csv", _generation_manifest_columns(), generation_rows)
    _write_csv(artifacts_root / "family_e1_trained_classifier_manifest.csv", _classifier_manifest_columns(), classifier_rows)
    write_family_e1_matrix(matrix_path, matrix_rows)
    _write_csv(artifacts_root / "family_e1_downstream_protocol_audit.csv", E1_PROTOCOL_AUDIT_COLUMNS, protocol_rows)
    return {
        "sampler_provenance": artifacts_root / "family_e1_sampler_provenance.csv",
        "sampler_diagnostics": artifacts_root / "family_e1_sampler_diagnostics.csv",
        "generation_manifest": artifacts_root / "family_e1_generation_manifest.csv",
        "classifier_manifest": artifacts_root / "family_e1_trained_classifier_manifest.csv",
        "matrix": matrix_path,
        "protocol_audit": artifacts_root / "family_e1_downstream_protocol_audit.csv",
    }


def score_family_e1_candidate(
    *,
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    experiment_seed: int,
    heldout_center: str,
    support_unit: SupportSelectionUnit,
    candidate_expert: str,
    mode: str,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    class_labels: Sequence[int],
    target_embeddings: Any,
    target_labels: Sequence[int],
    target_pool: TargetEvalPool,
) -> tuple[FamilyE1MatrixRow, tuple[Mapping[str, object], ...], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    base = _row_base(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        support_unit=support_unit,
        candidate_expert=candidate_expert,
        mode=mode,
        budget_per_class=budget_per_class,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        target_pool=target_pool,
        row_type=E1_DIAGNOSTIC_UPPER_BOUND_ROW_TYPE if mode in {E1_BOOTSTRAP_MODE, E1_REAL_SOURCE_MODE} else E1_SINGLE_EXPERT_ROW_TYPE,
        candidate_hash="__single_expert__",
    )
    release_level = E1_RELEASE_LEVELS[mode]
    target_status = validate_target_eval_class_coverage(
        target_labels=target_labels,
        required_class_labels=class_labels,
    )
    if not target_status["target_eval_has_all_classes"]:
        return _unavailable_target_eval_result(
            base=base,
            release_level=release_level,
            target_status=target_status,
            train_source="invalid_target_eval_single_class",
        )
    try:
        if mode == E1_REAL_SOURCE_MODE:
            batch = build_real_source_train_batch(
                sampler_bank=sampler_bank,
                source_center=candidate_expert,
                class_labels=class_labels,
            )
        else:
            batch = generate_class_balanced_batch(
                sampler_bank=sampler_bank,
                mode=mode,
                source_center=candidate_expert,
                class_labels=class_labels,
                budget_per_class=budget_per_class,
                generation_seed=generation_seed,
            )
        prediction = fit_locked_logistic_classifier(
            batch.embeddings,
            batch.labels,
            target_embeddings,
            target_labels,
            classifier_seed=classifier_seed,
        )
        row = FamilyE1MatrixRow(
            **base,
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            n_train=len(batch.labels),
            target_eval_label_counts_json=str(target_status["target_eval_label_counts_json"]),
            target_eval_has_all_classes=int(target_status["target_eval_has_all_classes"]),
            sampler_release_level=release_level,
        )
        diag = _context_diagnostic_row(
            base=base,
            release_level=release_level,
            available=1,
            status="ok",
            diagnostics=batch.diagnostics,
        )
        clf_manifest = _classifier_manifest_row(base=base, row=row, train_source="real_source_train" if mode == E1_REAL_SOURCE_MODE else "generated_embedding_sampler")
        audit = _protocol_audit_row(
            base=base,
            release_level=release_level,
            available=1,
            target_status=target_status,
        )
        return row, batch.generation_rows, diag, clf_manifest, audit
    except Exception as exc:
        row = FamilyE1MatrixRow(
            **base,
            bacc=math.nan,
            macro_f1=math.nan,
            n_train=0,
            target_eval_label_counts_json=str(target_status["target_eval_label_counts_json"]),
            target_eval_has_all_classes=int(target_status["target_eval_has_all_classes"]),
            sampler_release_level=release_level,
            available=0,
            status=_failure_status(exc),
            error_message=str(exc),
        )
        diag = _context_diagnostic_row(
            base=base,
            release_level=release_level,
            available=0,
            status=row.status,
            diagnostics={},
        )
        clf_manifest = _classifier_manifest_row(base=base, row=row, train_source="unavailable")
        audit = _protocol_audit_row(
            base=base,
            release_level=release_level,
            available=0,
            target_status=target_status,
        )
        return row, tuple(), diag, clf_manifest, audit


def score_family_e1_gmm_same_budget_ensemble(
    *,
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    experiment_seed: int,
    heldout_center: str,
    support_unit: SupportSelectionUnit,
    candidate_experts: Sequence[str],
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    class_labels: Sequence[int],
    target_embeddings: Any,
    target_labels: Sequence[int],
    target_pool: TargetEvalPool,
) -> tuple[FamilyE1MatrixRow, tuple[Mapping[str, object], ...], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    np = _np()
    candidate_hash = hash_candidate_experts(candidate_experts)
    base = _row_base(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        support_unit=support_unit,
        candidate_expert=E1_ENSEMBLE_EXPERT_ID,
        mode=E1_GMM_MODE,
        budget_per_class=budget_per_class,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        target_pool=target_pool,
        row_type=E1_METHOD_BASELINE_ROW_TYPE,
        candidate_hash=candidate_hash,
    )
    target_status = validate_target_eval_class_coverage(
        target_labels=target_labels,
        required_class_labels=class_labels,
    )
    if not target_status["target_eval_has_all_classes"]:
        return _unavailable_target_eval_result(
            base=base,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            target_status=target_status,
            train_source="invalid_target_eval_single_class",
        )
    try:
        allocation = allocate_family_e1_ensemble_budget(
            total_per_class=budget_per_class,
            candidate_experts=candidate_experts,
        )
        chunks: list[Any] = []
        labels: list[int] = []
        generation_rows: list[Mapping[str, object]] = []
        diagnostics: list[Mapping[str, object]] = []
        for expert in sorted(str(v) for v in candidate_experts):
            batch = generate_class_balanced_batch(
                sampler_bank=sampler_bank,
                mode=E1_GMM_MODE,
                source_center=expert,
                class_labels=class_labels,
                budget_per_class=int(allocation[expert]),
                generation_seed=int(generation_seed) + int(_expert_sort_value(expert)) * 7919,
            )
            chunks.append(batch.embeddings)
            labels.extend(batch.labels)
            generation_rows.extend(batch.generation_rows)
            diagnostics.append(batch.diagnostics)
        synthetic = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=float)
        prediction = fit_locked_logistic_classifier(
            synthetic,
            labels,
            target_embeddings,
            target_labels,
            classifier_seed=classifier_seed,
        )
        row = FamilyE1MatrixRow(
            **base,
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            n_train=len(labels),
            target_eval_label_counts_json=str(target_status["target_eval_label_counts_json"]),
            target_eval_has_all_classes=int(target_status["target_eval_has_all_classes"]),
            sampler_release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
        )
        diag = _context_diagnostic_row(
            base=base,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            available=1,
            status="ok",
            diagnostics=_mean_diagnostics(diagnostics),
        )
        clf_manifest = _classifier_manifest_row(base=base, row=row, train_source="gmm_same_budget_pooled_ensemble")
        audit = _protocol_audit_row(
            base=base,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            available=1,
            target_status=target_status,
        )
        return row, tuple(generation_rows), diag, clf_manifest, audit
    except Exception as exc:
        row = FamilyE1MatrixRow(
            **base,
            bacc=math.nan,
            macro_f1=math.nan,
            n_train=0,
            target_eval_label_counts_json=str(target_status["target_eval_label_counts_json"]),
            target_eval_has_all_classes=int(target_status["target_eval_has_all_classes"]),
            sampler_release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            available=0,
            status=_failure_status(exc),
            error_message=str(exc),
        )
        diag = _context_diagnostic_row(
            base=base,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            available=0,
            status=row.status,
            diagnostics={},
        )
        clf_manifest = _classifier_manifest_row(base=base, row=row, train_source="unavailable")
        audit = _protocol_audit_row(
            base=base,
            release_level=E1_RELEASE_LEVELS[E1_GMM_MODE],
            available=0,
            target_status=target_status,
        )
        return row, tuple(), diag, clf_manifest, audit


def allocate_family_e1_ensemble_budget(*, total_per_class: int, candidate_experts: Sequence[str]) -> dict[str, int]:
    allocation = allocate_equal_total_ensemble_budget(
        total_per_class=int(total_per_class),
        candidate_experts=tuple(str(v) for v in candidate_experts),
    )
    if sum(int(v) for v in allocation.values()) > int(total_per_class):
        raise ProtocolError("Family E1 same-budget ensemble exceeds single-expert per-class budget.")
    return allocation


def validate_target_eval_class_coverage(
    *,
    target_labels: Sequence[int],
    required_class_labels: Sequence[int],
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for label in target_labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    required = tuple(str(int(label)) for label in sorted(int(v) for v in required_class_labels))
    has_all = all(int(counts.get(label, 0)) > 0 for label in required)
    return {
        "target_eval_label_counts_json": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        "target_eval_has_all_classes": int(has_all),
    }


def build_family_e1_reports(
    *,
    artifacts_root: Path,
    candidate_domains: Sequence[str],
    c2_metrics: Mapping[str, float] | None = None,
) -> dict[str, Path]:
    matrix_path = artifacts_root / "family_e1_all_expert_downstream_matrix.csv"
    audit_path = artifacts_root / "family_e1_downstream_protocol_audit.csv"
    rows = read_family_e1_matrix(matrix_path)
    protocol_rows = _read_dict_csv(audit_path) if audit_path.exists() else []
    alignment_rows, prior_rows = build_family_e1_alignment_rows(
        rows=rows,
        candidate_domains=candidate_domains,
    )
    baseline_rows = build_family_e1_baseline_comparison_rows(
        rows=rows,
        alignment_rows=alignment_rows,
        c2_comparison=c2_metrics,
    )
    comparison_vs_c2 = build_family_e1_generation_mode_comparison_vs_c2(
        rows=rows,
        alignment_rows=alignment_rows,
        c2_metrics=c2_metrics or {},
    )
    summary = classify_family_e1_decision(
        rows=rows,
        alignment_rows=alignment_rows,
        protocol_audit_rows=protocol_rows,
        c2_metrics=c2_metrics or {},
    )
    _write_csv(artifacts_root / "family_e1_downstream_selection_alignment.csv", E1_ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(artifacts_root / "family_e1_source_transfer_sampler_prior_audit.csv", E1_PRIOR_AUDIT_COLUMNS, prior_rows)
    _write_csv(artifacts_root / "family_e1_downstream_baseline_comparison.csv", _baseline_columns(), baseline_rows)
    _write_csv(
        artifacts_root / "family_e1_generation_mode_comparison_vs_c2.csv",
        _comparison_vs_c2_columns(),
        comparison_vs_c2,
    )
    (artifacts_root / "family_e1_downstream_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "alignment": artifacts_root / "family_e1_downstream_selection_alignment.csv",
        "prior_audit": artifacts_root / "family_e1_source_transfer_sampler_prior_audit.csv",
        "baseline_comparison": artifacts_root / "family_e1_downstream_baseline_comparison.csv",
        "comparison_vs_c2": artifacts_root / "family_e1_generation_mode_comparison_vs_c2.csv",
        "decision_summary": artifacts_root / "family_e1_downstream_decision_summary.json",
    }


def build_family_e1_generation_mode_comparison_vs_c2(
    *,
    rows: Sequence[FamilyE1MatrixRow],
    alignment_rows: Sequence[Mapping[str, object]],
    c2_metrics: Mapping[str, float],
) -> list[dict[str, object]]:
    gmm_selected = [row for row in alignment_rows if row.get("selector") == E1_GMM_SELECTOR]
    sampler_selected = [row for row in alignment_rows if row.get("selector") == E1_SAMPLER_SELECTOR]
    gmm_oracle = _oracle_rows_by_context(rows, modes=(E1_GMM_MODE,))
    sampler_oracle = _oracle_rows_by_context(rows, modes=E1_SENSITIVITY_MODES)
    return [
        {
            "comparison": "gmm_selected_vs_c2_selected",
            "family_e1_bacc": center_level_mean(gmm_selected, metric="selected_bacc"),
            "c2_bacc": float(c2_metrics.get("c2_selected_center_level_mean_bacc", math.nan)),
            "delta_bacc": _delta_or_nan(
                center_level_mean(gmm_selected, metric="selected_bacc"),
                float(c2_metrics.get("c2_selected_center_level_mean_bacc", math.nan)),
            ),
            "available": int("c2_selected_center_level_mean_bacc" in c2_metrics),
        },
        {
            "comparison": "gmm_oracle_vs_c2_oracle",
            "family_e1_bacc": center_level_mean(gmm_oracle, metric="bacc"),
            "c2_bacc": float(c2_metrics.get("c2_oracle_center_level_mean_bacc", math.nan)),
            "delta_bacc": _delta_or_nan(
                center_level_mean(gmm_oracle, metric="bacc"),
                float(c2_metrics.get("c2_oracle_center_level_mean_bacc", math.nan)),
            ),
            "available": int("c2_oracle_center_level_mean_bacc" in c2_metrics),
        },
        {
            "comparison": "sampler_selector_vs_gmm_selector",
            "family_e1_bacc": center_level_mean(sampler_selected, metric="selected_bacc"),
            "c2_bacc": center_level_mean(gmm_selected, metric="selected_bacc"),
            "delta_bacc": _delta_or_nan(
                center_level_mean(sampler_selected, metric="selected_bacc"),
                center_level_mean(gmm_selected, metric="selected_bacc"),
            ),
            "available": 1,
        },
        {
            "comparison": "sampler_oracle_vs_gmm_oracle",
            "family_e1_bacc": center_level_mean(sampler_oracle, metric="bacc"),
            "c2_bacc": center_level_mean(gmm_oracle, metric="bacc"),
            "delta_bacc": _delta_or_nan(
                center_level_mean(sampler_oracle, metric="bacc"),
                center_level_mean(gmm_oracle, metric="bacc"),
            ),
            "available": 1,
        },
    ]


def generation_diagnostics(
    *,
    generated_embeddings: Any,
    generated_labels: Sequence[int],
    real_source_by_class: Mapping[int, Any],
) -> dict[str, object]:
    np = _np()
    x = _as_numpy_2d(generated_embeddings)
    labels = np.asarray([int(v) for v in generated_labels], dtype=int)
    real_chunks = [_as_numpy_2d(v) for _, v in sorted(real_source_by_class.items()) if _as_numpy_2d(v).size]
    real = np.vstack(real_chunks) if real_chunks else np.empty((0, x.shape[1] if x.ndim == 2 else 0))
    nearest_mean, nearest_min = generated_to_nearest_source_distances(x, labels, real_source_by_class)
    gen_centroid = class_centroid_distance(x, labels)
    real_labels: list[int] = []
    for label, values in sorted(real_source_by_class.items()):
        arr = _as_numpy_2d(values)
        real_labels.extend([int(label)] * int(arr.shape[0]))
    real_centroid = class_centroid_distance(real, real_labels)
    ratio = gen_centroid / real_centroid if real_centroid and not math.isnan(real_centroid) else math.nan
    return {
        "generated_to_nearest_source_distance_mean": nearest_mean,
        "generated_to_nearest_source_distance_min": nearest_min,
        "generated_effective_rank": effective_rank(x),
        "generated_cov_trace": covariance_trace(x),
        "generated_pairwise_distance_mean": mean_pairwise_distance(x),
        "generated_nan_count": int(np.isnan(x).sum()) if x.size else 0,
        "generated_inf_count": int(np.isinf(x).sum()) if x.size else 0,
        "generated_norm_mean": norm_mean(x),
        "generated_norm_std": norm_std(x),
        "real_source_norm_mean": norm_mean(real),
        "real_source_norm_std": norm_std(real),
        "generated_class_centroid_distance": gen_centroid,
        "real_class_centroid_distance": real_centroid,
        "centroid_distance_ratio": ratio,
    }


def generated_to_nearest_source_distances(
    generated_embeddings: Any,
    generated_labels: Sequence[int],
    real_source_by_class: Mapping[int, Any],
) -> tuple[float, float]:
    np = _np()
    x = _as_numpy_2d(generated_embeddings)
    labels = np.asarray([int(v) for v in generated_labels], dtype=int)
    distances: list[float] = []
    for label in sorted(set(labels.tolist())):
        generated = x[labels == int(label)]
        real = _as_numpy_2d(real_source_by_class.get(int(label), np.empty((0, x.shape[1]))))
        if generated.size == 0 or real.size == 0:
            continue
        for chunk in _chunks(generated, 256):
            dist = _pairwise_distances(chunk, real)
            distances.extend(np.min(dist, axis=1).tolist())
    if not distances:
        return math.nan, math.nan
    return float(np.mean(distances)), float(np.min(distances))


def median_pairwise_distance(x: Any, *, max_points: int = 512) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.shape[0] < 2:
        return math.nan
    sample = arr[: min(int(max_points), arr.shape[0])]
    dist = _pairwise_distances(sample, sample)
    tri = dist[np.triu_indices(sample.shape[0], k=1)]
    return float(np.median(tri)) if tri.size else math.nan


def mean_pairwise_distance(x: Any, *, max_points: int = 512) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.shape[0] < 2:
        return math.nan
    sample = arr[: min(int(max_points), arr.shape[0])]
    dist = _pairwise_distances(sample, sample)
    tri = dist[np.triu_indices(sample.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else math.nan


def effective_rank(x: Any) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        return math.nan
    cov = np.cov(arr, rowvar=False)
    eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))
    eigvals = np.asarray([float(v) for v in eigvals if float(v) > 0.0], dtype=float)
    total = float(eigvals.sum())
    if total <= 0.0:
        return 0.0
    probs = eigvals / total
    return float(math.exp(-float(np.sum(probs * np.log(probs)))))


def covariance_trace(x: Any) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.shape[0] < 2:
        return math.nan
    cov = np.cov(arr, rowvar=False)
    return float(np.trace(np.atleast_2d(cov)))


def norm_mean(x: Any) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.size == 0:
        return math.nan
    return float(np.linalg.norm(arr, axis=1).mean())


def norm_std(x: Any) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    if arr.size == 0:
        return math.nan
    return float(np.linalg.norm(arr, axis=1).std())


def class_centroid_distance(x: Any, labels: Sequence[int]) -> float:
    np = _np()
    arr = _as_numpy_2d(x)
    y = np.asarray([int(v) for v in labels], dtype=int)
    classes = sorted(set(y.tolist()))
    if len(classes) != 2:
        return math.nan
    centroids = []
    for cls in classes:
        subset = arr[y == int(cls)]
        if subset.size == 0:
            return math.nan
        centroids.append(subset.mean(axis=0))
    return float(np.linalg.norm(centroids[1] - centroids[0]))


def build_family_e1_target_eval_pool(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
) -> TargetEvalPool:
    require_patient = "patient_disjoint" in str(support_eval_split_id)
    if not require_patient:
        return build_target_eval_pool(
            test_metadata=test_metadata,
            heldout_center=heldout_center,
            support_sizes=(int(support_size),),
            support_seeds=(int(support_seed),),
        )
    target_indices = tuple(
        idx for idx, row in enumerate(test_metadata) if str(_domain(row)) == str(heldout_center)
    )
    labels_by_index = {idx: _label(test_metadata[idx]) for idx in target_indices}
    patient_ids = {
        idx: str(test_metadata[idx].get("patient_id", "") or test_metadata[idx].get("patient", ""))
        for idx in target_indices
    }
    try:
        split = _make_support_eval_split(
            target_domain=int(heldout_center),
            target_indices=target_indices,
            labels_by_index=labels_by_index,
            support_size=int(support_size),
            sampling_policy="random",
            support_seed=int(support_seed),
            patient_ids_by_index=patient_ids,
            require_patient_disjoint=True,
        )
    except TypeError:
        split = _make_support_eval_split(
            target_domain=int(heldout_center),
            target_indices=target_indices,
            labels_by_index=labels_by_index,
            support_size=int(support_size),
            sampling_policy="random",
            support_seed=int(support_seed),
        )
    support_ids = {str(_sample_id(test_metadata[idx])) for idx in split.support_indices}
    eval_indices = tuple(int(idx) for idx in split.eval_indices)
    if set(support_ids).intersection(str(_sample_id(test_metadata[idx])) for idx in eval_indices):
        raise ProtocolError("Family E1 support/eval split overlap detected.")
    digest = hashlib.sha256("|".join(sorted(support_ids)).encode("utf-8")).hexdigest()[:12]
    return TargetEvalPool(
        eval_indices=eval_indices,
        excluded_support_sample_ids=tuple(sorted(support_ids)),
        target_eval_pool_id=f"{support_eval_split_id}_exclude_support_{digest}",
    )


def discover_family_e1_support_artifacts(
    *,
    config: FamilyE1Config,
    repo_root: Path,
) -> tuple[FamilyE1SupportArtifacts, ...]:
    import glob

    support_paths = sorted(Path(p) for p in glob.glob(str(repo_root / config.support_selection_glob)))
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    artifacts: list[FamilyE1SupportArtifacts] = []
    for support_path in support_paths:
        run_dir = support_path.parent.parent
        config_resolved = run_dir / "config_resolved.yaml"
        artifact = FamilyE1SupportArtifacts(
            experiment_seed=_experiment_seed_from_run(run_dir, config_resolved),
            run_dir=run_dir,
            train_cache=run_dir / "embeddings" / "train.pt",
            test_cache=run_dir / "embeddings" / "test.pt",
            samples_manifest=run_dir / "manifests" / "samples.csv",
            config_resolved=config_resolved,
            split_manifest=run_dir / "reports" / "support_response_split_manifest.csv",
            support_selection_path=support_path,
        )
        missing = [
            path
            for path in (
                artifact.train_cache,
                artifact.test_cache,
                artifact.samples_manifest,
                artifact.config_resolved,
                artifact.split_manifest,
                artifact.support_selection_path,
            )
            if not path.exists()
        ]
        if missing:
            preview = "\n".join(f"- {path}" for path in missing)
            raise ArtifactSyncError(
                "Missing frozen support-run artifacts required for Family E1:\n"
                f"{preview}"
            )
        artifacts.append(artifact)
    return tuple(sorted(artifacts, key=lambda item: item.experiment_seed))


def read_family_e1_support_units(paths: Iterable[Path]) -> list[SupportSelectionUnit]:
    units: list[SupportSelectionUnit] = []
    seen: set[tuple[object, ...]] = set()
    for path in sorted(Path(p) for p in paths):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("method", "")) != SUPPORT_NELBO_METHOD:
                    continue
                unit = _support_unit_from_row(row)
                assert_target_excluded(unit)
                key = (
                    int(unit.experiment_seed),
                    unit.heldout_center,
                    int(unit.support_size),
                    int(unit.support_seed),
                    unit.support_eval_split_id,
                )
                if key in seen:
                    continue
                seen.add(key)
                units.append(unit)
    if not units:
        raise ProtocolError("No support-NELBO support contexts found for Family E1.")
    return sorted(units, key=lambda u: (int(u.experiment_seed), u.heldout_center, int(u.support_size), int(u.support_seed), u.support_eval_split_id))


def validate_family_e1_matrix(rows: Sequence[FamilyE1MatrixRow]) -> None:
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        if row.schema_version != E1_SCHEMA_VERSION:
            raise ProtocolError(f"Unexpected Family E1 matrix schema: {row.schema_version}")
        key = row.primary_key()
        if key in seen:
            raise ProtocolError(f"Duplicate Family E1 matrix row: {key}")
        seen.add(key)
        if row.row_type not in {E1_SINGLE_EXPERT_ROW_TYPE, E1_METHOD_BASELINE_ROW_TYPE, E1_DIAGNOSTIC_UPPER_BOUND_ROW_TYPE}:
            raise ProtocolError(f"Unknown Family E1 row_type: {row.row_type}")
        if row.heldout_center == row.candidate_expert:
            raise ProtocolError(f"Target expert leakage in Family E1 matrix row: {key}")
        if row.status == "ok" and int(row.available) == 1 and int(row.target_eval_has_all_classes) != 1:
            raise ProtocolError(f"Family E1 ok matrix row lacks all target eval classes: {key}")


def validate_family_e1_protocol_audit(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        checks = {
            "sampler_fit_split": str(row.get("sampler_fit_split", "")) == "source_train",
            "target_expert_excluded": int(row.get("target_expert_excluded", 0)) == 1,
            "support_eval_disjoint": int(row.get("support_eval_disjoint", 0)) == 1,
            "target_labels_used_for_sampler_fit": int(row.get("target_labels_used_for_sampler_fit", 1)) == 0,
            "target_support_labels_used_for_generation": int(row.get("target_support_labels_used_for_generation", 1)) == 0,
            "target_eval_embeddings_used_for_generation": int(row.get("target_eval_embeddings_used_for_generation", 1)) == 0,
            "target_eval_labels_used_for_training": int(row.get("target_eval_labels_used_for_training", 1)) == 0,
            "target_eval_labels_used_for_final_metric_only": int(row.get("target_eval_labels_used_for_final_metric_only", 0)) == 1,
            "target_eval_has_all_classes": _audit_target_class_check(row),
            "target_oracle_used_for_selection": int(row.get("target_oracle_used_for_selection", 1)) == 0,
            "target_heldout_rows_used_for_source_transfer_prior": int(row.get("target_heldout_rows_used_for_source_transfer_prior", 1)) == 0,
        }
        failed = [key for key, ok in checks.items() if not ok]
        if failed:
            raise ProtocolError(f"Family E1 protocol audit failed fields {failed}: {row}")


def _audit_target_class_check(row: Mapping[str, object]) -> bool:
    if int(row.get("available", 0)) == 0:
        return True
    return int(row.get("target_eval_has_all_classes", 0)) == 1


def protocol_audit_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    try:
        validate_family_e1_protocol_audit(rows)
    except (ProtocolError, ValueError, TypeError):
        return False
    return bool(rows)


def read_family_e1_matrix(path: Path) -> list[FamilyE1MatrixRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [_e1_row_from_csv(row) for row in csv.DictReader(handle)]
    validate_family_e1_matrix(rows)
    return rows


def write_family_e1_matrix(path: Path, rows: Sequence[FamilyE1MatrixRow]) -> None:
    validate_family_e1_matrix(rows)
    _write_csv(path, E1_MATRIX_COLUMNS, [row.to_csv_row() for row in rows])


def center_level_mean(rows: Sequence[Mapping[str, object] | FamilyE1MatrixRow], *, metric: str) -> float:
    by_center: dict[str, list[float]] = {}
    for row in rows:
        center = str(_row_get(row, "heldout_center"))
        value = _to_float(_row_get(row, metric))
        if math.isnan(value):
            continue
        by_center.setdefault(center, []).append(value)
    center_means = [_nanmean(values) for values in by_center.values() if values]
    return _nanmean(center_means)


def _family_e1_config_from_mapping(config: Mapping[str, Any]) -> FamilyE1Config:
    base = default_family_e1_config()
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("name") != FAMILY_E1_NAME:
        raise ProtocolError(f"Unexpected Family E1 experiment.name: {experiment.get('name')!r}")
    datasets = _mapping(config.get("datasets"), "datasets")
    camelyon = _mapping(datasets.get("camelyon17"), "datasets.camelyon17")
    if not bool(camelyon.get("enabled")):
        raise ProtocolError("Family E1 v1 must enable camelyon17.")
    generation = _mapping(config.get("generation"), "generation")
    samplers = _mapping(config.get("samplers"), "samplers")
    pca = _mapping(samplers.get("pca_before_sampler"), "samplers.pca_before_sampler")
    if bool(pca.get("enabled")):
        raise ProtocolError("Family E1 v1 must keep pca_before_sampler.enabled=false.")
    gmm = _mapping(samplers.get("gmm_diag_bic"), "samplers.gmm_diag_bic")
    if gmm.get("covariance_type") != "diag":
        raise ProtocolError("Family E1 GMM covariance_type must be diag.")
    release_levels = _mapping(samplers.get("release_levels"), "samplers.release_levels")
    for mode, expected in E1_RELEASE_LEVELS.items():
        key = _mode_short_name(mode)
        if release_levels.get(key) != expected:
            raise ProtocolError(f"Family E1 release level for {key} must be {expected!r}.")
    support_inputs = _mapping(config.get("support_inputs"), "support_inputs")
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    comparison = _mapping(config.get("external_comparisons"), "external_comparisons")
    return FamilyE1Config(
        dataset_name="camelyon17",
        domain_key=str(camelyon.get("domain_key", base.domain_key)),
        candidate_domains=tuple(str(v) for v in camelyon.get("candidate_domains", base.candidate_domains)),
        experiment_seeds=tuple(int(v) for v in camelyon.get("experiment_seeds", base.experiment_seeds)),
        support_sizes=tuple(int(v) for v in camelyon.get("support_sizes", base.support_sizes)),
        support_seeds=tuple(int(v) for v in camelyon.get("support_seeds", base.support_seeds)),
        generation_seeds=tuple(int(v) for v in camelyon.get("generation_seeds", base.generation_seeds)),
        classifier_seeds=tuple(int(v) for v in camelyon.get("classifier_seeds", base.classifier_seeds)),
        class_labels=tuple(int(v) for v in generation.get("labels", base.class_labels)),
        budget_per_class=int(generation.get("budget_per_class", base.budget_per_class)),
        modes=tuple(str(v) for v in samplers.get("modes", base.modes)),
        support_selection_glob=str(support_inputs.get("selection_glob", base.support_selection_glob)),
        artifacts_root=str(artifacts.get("root", base.artifacts_root)),
        pca_enabled=bool(pca.get("enabled", False)),
        pca_n_components=int(pca.get("n_components", base.pca_n_components)),
        gmm_k_candidates=tuple(int(v) for v in gmm.get("k_candidates", base.gmm_k_candidates)),
        gmm_reg_covar=float(gmm.get("reg_covar", base.gmm_reg_covar)),
        gmm_valid_min_samples=int(gmm.get("valid_k_min_samples", base.gmm_valid_min_samples)),
        gmm_valid_samples_per_component=int(gmm.get("valid_k_samples_per_component", base.gmm_valid_samples_per_component)),
        kde_min_bandwidth=float(_mapping(samplers.get("kde_gaussian", {}), "samplers.kde_gaussian").get("min_bandwidth", base.kde_min_bandwidth)),
        smote_jitter_scale=float(_mapping(samplers.get("smote_interpolate", {}), "samplers.smote_interpolate").get("jitter_scale", base.smote_jitter_scale)),
        c2_artifacts_root=str(comparison.get("c2_artifacts_root", base.c2_artifacts_root)),
    )


def _support_unit_from_row(row: Mapping[str, str]) -> SupportSelectionUnit:
    heldout = str(row.get("fold_query_domain") or row.get("query_domain") or row.get("target_domain") or row.get("heldout_center") or "").strip()
    candidates = tuple(str(part).strip() for part in str(row.get("candidate_experts", "")).split("|") if str(part).strip())
    if not heldout or not candidates:
        raise ProtocolError("Family E1 support row lacks heldout center or candidates.")
    selected = str(row.get("selected_expert", "")).strip()
    if not selected:
        selected = candidates[0]
    return SupportSelectionUnit(
        heldout_center=heldout,
        experiment_seed=int(row.get("seed") or row.get("experiment_seed") or row.get("run_seed") or 0),
        support_size=int(row.get("support_size_requested") or row.get("support_size") or row.get("support_n") or 0),
        support_seed=int(row.get("support_seed") or 0),
        method=SUPPORT_NELBO_METHOD,
        selected_expert=selected,
        candidate_experts=candidates,
        support_nelbo_by_expert=parse_expert_scores_json(row.get("support_nelbo_by_expert_json", "{}")),
        target_expert_excluded=str(row.get("target_expert_excluded", "")).strip().lower() in {"1", "true", "yes"},
        support_eval_split_id=str(row.get("support_eval_split_id", "")).strip(),
    )


def _row_base(
    *,
    experiment_seed: int,
    heldout_center: str,
    support_unit: SupportSelectionUnit,
    candidate_expert: str,
    mode: str,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    target_pool: TargetEvalPool,
    row_type: str,
    candidate_hash: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_size": int(support_unit.support_size),
        "support_seed": int(support_unit.support_seed),
        "support_eval_split_id": str(support_unit.support_eval_split_id),
        "candidate_expert": str(candidate_expert),
        "generation_mode": str(mode),
        "budget_per_class": int(budget_per_class),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "row_type": str(row_type),
        "n_target_eval": len(target_pool.eval_indices),
        "target_eval_pool_id": target_pool.target_eval_pool_id,
        "candidate_experts_hash": str(candidate_hash),
    }


def _unavailable_target_eval_result(
    *,
    base: Mapping[str, object],
    release_level: str,
    target_status: Mapping[str, object],
    train_source: str,
) -> tuple[FamilyE1MatrixRow, tuple[Mapping[str, object], ...], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    row = FamilyE1MatrixRow(
        **base,
        bacc=math.nan,
        macro_f1=math.nan,
        n_train=0,
        target_eval_label_counts_json=str(target_status["target_eval_label_counts_json"]),
        target_eval_has_all_classes=int(target_status["target_eval_has_all_classes"]),
        sampler_release_level=release_level,
        available=0,
        status="failed_single_class_target_eval",
        error_message=(
            "Target eval pool must contain every required class before downstream "
            f"utility scoring; observed {target_status['target_eval_label_counts_json']}."
        ),
    )
    diag = _context_diagnostic_row(
        base=base,
        release_level=release_level,
        available=0,
        status=row.status,
        diagnostics={
            "target_eval_label_counts_json": str(target_status["target_eval_label_counts_json"]),
            "target_eval_has_all_classes": int(target_status["target_eval_has_all_classes"]),
        },
    )
    clf_manifest = _classifier_manifest_row(base=base, row=row, train_source=train_source)
    audit = _protocol_audit_row(
        base=base,
        release_level=release_level,
        available=0,
        target_status=target_status,
    )
    return row, tuple(), diag, clf_manifest, audit


def _protocol_audit_row(
    *,
    base: Mapping[str, object],
    release_level: str,
    available: int,
    target_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    target_status = target_status or {
        "target_eval_label_counts_json": "{}",
        "target_eval_has_all_classes": 0,
    }
    return {
        "experiment_seed": base["experiment_seed"],
        "heldout_center": base["heldout_center"],
        "support_size": base["support_size"],
        "support_seed": base["support_seed"],
        "support_eval_split_id": base["support_eval_split_id"],
        "candidate_expert": base["candidate_expert"],
        "generation_mode": base["generation_mode"],
        "sampler_fit_split": "source_train",
        "target_expert_excluded": int(str(base["heldout_center"]) != str(base["candidate_expert"])),
        "support_eval_disjoint": 1,
        "target_labels_used_for_sampler_fit": 0,
        "target_support_labels_used_for_generation": 0,
        "target_eval_embeddings_used_for_generation": 0,
        "target_eval_labels_used_for_training": 0,
        "target_eval_labels_used_for_final_metric_only": 1,
        "target_eval_label_counts_json": str(target_status["target_eval_label_counts_json"]),
        "target_eval_has_all_classes": int(target_status["target_eval_has_all_classes"]),
        "target_oracle_used_for_selection": 0,
        "target_heldout_rows_used_for_source_transfer_prior": 0,
        "sampler_release_level": release_level,
        "available": int(available),
    }


def _classifier_manifest_row(
    *,
    base: Mapping[str, object],
    row: FamilyE1MatrixRow,
    train_source: str,
) -> dict[str, object]:
    return {
        "experiment_seed": base["experiment_seed"],
        "heldout_center": base["heldout_center"],
        "support_size": base["support_size"],
        "support_seed": base["support_seed"],
        "candidate_expert": base["candidate_expert"],
        "generation_mode": base["generation_mode"],
        "generation_seed": base["generation_seed"],
        "classifier_seed": base["classifier_seed"],
        "train_source": train_source,
        "scaler_fit": "synthetic_train_only" if train_source != "real_source_train" else "real_source_train_only",
        "classifier_family": "sklearn_logistic_regression",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": "",
        "n_train": row.n_train,
        "n_target_eval": row.n_target_eval,
        "target_eval_pool_id": row.target_eval_pool_id,
        "target_eval_label_counts_json": row.target_eval_label_counts_json,
        "target_eval_has_all_classes": row.target_eval_has_all_classes,
        "status": row.status,
        "available": row.available,
    }


def _context_diagnostic_row(
    *,
    base: Mapping[str, object],
    release_level: str,
    available: int,
    status: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    row = {
        "experiment_seed": base["experiment_seed"],
        "heldout_center": base["heldout_center"],
        "support_size": base["support_size"],
        "support_seed": base["support_seed"],
        "candidate_expert": base["candidate_expert"],
        "generation_mode": base["generation_mode"],
        "generation_seed": base["generation_seed"],
        "classifier_seed": base["classifier_seed"],
        "class_label": "",
        "sampler_release_level": release_level,
        "available": int(available),
        "status": status,
    }
    row.update(_blank_diagnostics())
    row.update(diagnostics)
    return row


def _sampler_provenance_rows(
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    *,
    experiment_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, _, _), fit in sorted(sampler_bank.items(), key=lambda item: (item[1].mode, _expert_sort_value(item[1].source_center), item[1].class_label)):
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "source_center": fit.source_center,
                "class_label": fit.class_label,
                "generation_mode": fit.mode,
                "sampler_fit_split": "source_train",
                "n_source_train": fit.n_source_train,
                "embedding_dim": fit.embedding_dim,
                "sampler_release_level": fit.release_level,
                "available": int(fit.available),
                "source_sample_ids_hash": _hash_values(fit.source_sample_ids),
                "target_rows_used_for_fit": 0,
                "pca_before_sampler_enabled": 0,
                "pca_n_components": "",
                "error_message": fit.error_message,
            }
        )
    return rows


def _sampler_fit_diagnostic_rows(
    sampler_bank: Mapping[tuple[str, str, int], SamplerFitResult],
    *,
    experiment_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, _, _), fit in sorted(sampler_bank.items(), key=lambda item: (item[1].mode, _expert_sort_value(item[1].source_center), item[1].class_label)):
        row = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": "",
            "support_size": "",
            "support_seed": "",
            "candidate_expert": fit.source_center,
            "generation_mode": fit.mode,
            "generation_seed": "",
            "classifier_seed": "",
            "class_label": fit.class_label,
            "sampler_release_level": fit.release_level,
            "available": int(fit.available),
            "status": "ok" if fit.available else fit.error_message,
        }
        row.update(_blank_diagnostics())
        row.update(fit.diagnostics)
        rows.append(row)
    return rows


def _base_fit_diagnostics(source: SourceClassData, *, mode: str, embedding_dim: int) -> dict[str, object]:
    x = _as_numpy_2d(source.embeddings)
    n = int(x.shape[0])
    return {
        "n_source_train": n,
        "embedding_dim": int(embedding_dim),
        "effective_sample_to_dim_ratio": float(n) / float(max(1, int(embedding_dim))),
        "median_pairwise_distance": median_pairwise_distance(x),
        "gmm_selected_k": "",
        "gmm_bic_by_k": "{}",
        "gmm_converged": "",
        "gmm_n_iter": "",
        "gmm_min_component_weight": math.nan,
        "gmm_cov_min": math.nan,
        "gmm_cov_max": math.nan,
        "kde_bandwidth": math.nan,
        "smote_jitter_std": math.nan,
    }


def _oracle_for_context(
    rows: Sequence[FamilyE1MatrixRow],
    *,
    context: tuple[int, str, int, int, int, int],
    modes: Sequence[str],
) -> FamilyE1MatrixRow | None:
    mode_set = {str(v) for v in modes}
    candidates = [
        row
        for row in rows
        if row.context_key() == context
        and row.generation_mode in mode_set
        and row.status == "ok"
        and int(row.available) == 1
        and row.row_type in {E1_SINGLE_EXPERT_ROW_TYPE, E1_DIAGNOSTIC_UPPER_BOUND_ROW_TYPE}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row.bacc), float(row.macro_f1), -E1_MODE_ORDER.get(row.generation_mode, 999), -_expert_sort_value(row.candidate_expert)))


def _oracle_rows_by_context(
    rows: Sequence[FamilyE1MatrixRow],
    *,
    modes: Sequence[str],
) -> list[FamilyE1MatrixRow]:
    out: list[FamilyE1MatrixRow] = []
    for context in sorted({row.context_key() for row in rows}):
        oracle = _oracle_for_context(rows, context=context, modes=modes)
        if oracle is not None:
            out.append(oracle)
    return out


def _prior_audit_rows(
    prior: SourceTransferPrior,
    *,
    selection: SourceTransferSelection,
    modes: Sequence[str],
    candidate_experts: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in modes:
        for expert in candidate_experts:
            per_source = prior.source_center_scores.get((mode, expert), {})
            rows.append(
                {
                    "heldout_center": prior.heldout_center,
                    "selector": prior.selector,
                    "mode": mode,
                    "candidate_expert": expert,
                    "prior_score": prior.scores.get((mode, expert), math.nan),
                    "source_centers_used": "|".join(sorted(per_source)),
                    "source_center_scores_json": json.dumps({str(k): float(v) for k, v in sorted(per_source.items())}, sort_keys=True),
                    "n_source_centers_used": len(per_source),
                    "target_heldout_rows_used": 0,
                    "selected": int(selection.mode == mode and selection.expert == expert),
                    "tie_break_mode_rank": E1_MODE_ORDER.get(mode, 999),
                }
            )
    return rows


def _comparison_row(
    *,
    method: str,
    row_type: str,
    mean_bacc: float,
    mean_macro_f1: float,
    mean_oracle_gap_bacc: float,
    available: int,
) -> dict[str, object]:
    return {
        "method": method,
        "row_type": row_type,
        "center_level_mean_bacc": mean_bacc,
        "center_level_mean_macro_f1": mean_macro_f1,
        "center_level_mean_oracle_gap_bacc": mean_oracle_gap_bacc,
        "available": int(available),
    }


def _low_effective_sample_ratio(rows: Sequence[FamilyE1MatrixRow]) -> bool:
    _ = rows
    return False


def _read_completed_e1_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    return {row.primary_key() for row in read_family_e1_matrix(path)}


def _e1_row_from_csv(row: Mapping[str, str]) -> FamilyE1MatrixRow:
    return FamilyE1MatrixRow(
        schema_version=str(row.get("schema_version") or E1_SCHEMA_VERSION),
        experiment_seed=int(row.get("experiment_seed") or 0),
        heldout_center=str(row["heldout_center"]),
        support_size=int(row.get("support_size") or 0),
        support_seed=int(row.get("support_seed") or 0),
        support_eval_split_id=str(row.get("support_eval_split_id") or ""),
        candidate_expert=str(row["candidate_expert"]),
        generation_mode=str(row["generation_mode"]),
        budget_per_class=int(row.get("budget_per_class") or 0),
        generation_seed=int(row.get("generation_seed") or 0),
        classifier_seed=int(row.get("classifier_seed") or 0),
        bacc=_to_float(row.get("bacc")),
        macro_f1=_to_float(row.get("macro_f1")),
        auroc=_to_float(row.get("auroc")),
        auprc=_to_float(row.get("auprc")),
        row_type=str(row.get("row_type") or E1_SINGLE_EXPERT_ROW_TYPE),
        n_train=int(row.get("n_train") or 0),
        n_target_eval=int(row.get("n_target_eval") or 0),
        target_eval_pool_id=str(row.get("target_eval_pool_id") or ""),
        target_eval_label_counts_json=str(row.get("target_eval_label_counts_json") or "{}"),
        target_eval_has_all_classes=int(row.get("target_eval_has_all_classes") or 0),
        candidate_experts_hash=str(row.get("candidate_experts_hash") or "__single_expert__"),
        sampler_release_level=str(row.get("sampler_release_level") or ""),
        available=int(row.get("available") or 0),
        status=str(row.get("status") or "ok"),
        error_message=str(row.get("error_message") or ""),
    )


def _support_units_by_seed(units: Sequence[SupportSelectionUnit]) -> dict[int, tuple[SupportSelectionUnit, ...]]:
    grouped: dict[int, list[SupportSelectionUnit]] = {}
    for unit in units:
        grouped.setdefault(int(unit.experiment_seed), []).append(unit)
    return {seed: tuple(values) for seed, values in grouped.items()}


def _limit_artifacts(
    artifacts: Sequence[FamilyE1SupportArtifacts],
    experiment_seeds: Sequence[int] | None,
) -> tuple[FamilyE1SupportArtifacts, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.experiment_seed) in allowed)


def _limit_support_units(
    units: Sequence[SupportSelectionUnit],
    limits: FamilyE1BuildLimits,
) -> tuple[SupportSelectionUnit, ...]:
    out = []
    for unit in units:
        if limits.experiment_seeds is not None and int(unit.experiment_seed) not in set(limits.experiment_seeds):
            continue
        if limits.heldout_centers is not None and unit.heldout_center not in set(limits.heldout_centers):
            continue
        if limits.support_sizes is not None and int(unit.support_size) not in set(limits.support_sizes):
            continue
        if limits.support_seeds is not None and int(unit.support_seed) not in set(limits.support_seeds):
            continue
        out.append(unit)
    return tuple(out)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_dict_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _provenance_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "source_center",
        "class_label",
        "generation_mode",
        "sampler_fit_split",
        "n_source_train",
        "embedding_dim",
        "sampler_release_level",
        "available",
        "source_sample_ids_hash",
        "target_rows_used_for_fit",
        "pca_before_sampler_enabled",
        "pca_n_components",
        "error_message",
    )


def _generation_manifest_columns() -> tuple[str, ...]:
    return (
        "source_center",
        "generation_mode",
        "class_label",
        "generation_seed",
        "budget_per_class",
        "n_generated",
        "n_source_train",
        "sampler_release_level",
        "available",
        "source_sample_ids_hash",
    )


def _classifier_manifest_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "support_size",
        "support_seed",
        "candidate_expert",
        "generation_mode",
        "generation_seed",
        "classifier_seed",
        "train_source",
        "scaler_fit",
        "classifier_family",
        "solver",
        "C",
        "max_iter",
        "class_weight",
        "n_train",
        "n_target_eval",
        "target_eval_pool_id",
        "target_eval_label_counts_json",
        "target_eval_has_all_classes",
        "status",
        "available",
    )


def _diagnostic_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "support_size",
        "support_seed",
        "candidate_expert",
        "generation_mode",
        "generation_seed",
        "classifier_seed",
        "class_label",
        "sampler_release_level",
        "available",
        "status",
        "target_eval_label_counts_json",
        "target_eval_has_all_classes",
        "n_source_train",
        "embedding_dim",
        "gmm_selected_k",
        "gmm_bic_by_k",
        "gmm_converged",
        "gmm_n_iter",
        "gmm_min_component_weight",
        "gmm_cov_min",
        "gmm_cov_max",
        "effective_sample_to_dim_ratio",
        "kde_bandwidth",
        "median_pairwise_distance",
        "generated_to_nearest_source_distance_mean",
        "generated_to_nearest_source_distance_min",
        "smote_alpha_mean",
        "smote_jitter_std",
        "generated_effective_rank",
        "generated_cov_trace",
        "generated_pairwise_distance_mean",
        "generated_nan_count",
        "generated_inf_count",
        "generated_norm_mean",
        "generated_norm_std",
        "real_source_norm_mean",
        "real_source_norm_std",
        "generated_class_centroid_distance",
        "real_class_centroid_distance",
        "centroid_distance_ratio",
    )


def _blank_diagnostics() -> dict[str, object]:
    blank = {column: math.nan for column in _diagnostic_columns() if column not in {
        "experiment_seed",
        "heldout_center",
        "support_size",
        "support_seed",
        "candidate_expert",
        "generation_mode",
        "generation_seed",
        "classifier_seed",
        "class_label",
        "sampler_release_level",
        "available",
        "status",
        "gmm_bic_by_k",
        "target_eval_label_counts_json",
    }} | {"gmm_bic_by_k": "{}"}
    blank["target_eval_label_counts_json"] = "{}"
    return blank


def _baseline_columns() -> tuple[str, ...]:
    return (
        "method",
        "row_type",
        "center_level_mean_bacc",
        "center_level_mean_macro_f1",
        "center_level_mean_oracle_gap_bacc",
        "available",
    )


def _comparison_vs_c2_columns() -> tuple[str, ...]:
    return ("comparison", "family_e1_bacc", "c2_bacc", "delta_bacc", "available")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _mode_short_name(mode: str) -> str:
    mapping = {
        E1_GMM_MODE: "gmm_diag_bic",
        E1_KDE_MODE: "kde_gaussian",
        E1_SMOTE_MODE: "smote_interpolate",
        E1_BOOTSTRAP_MODE: "source_bootstrap_upper_bound",
        E1_REAL_SOURCE_MODE: "real_source_train_classifier_baseline",
    }
    return mapping[mode]


def _as_numpy_2d(value: Any) -> Any:
    np = _np()
    arr = np.asarray(_to_numpy(value), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D embedding array.")
    return arr


def _as_numpy_1d(value: Any) -> Any:
    np = _np()
    return np.asarray(_to_numpy(value), dtype=float).reshape(-1)


def _shape2(value: Any) -> tuple[int, int]:
    arr = _as_numpy_2d(value)
    return int(arr.shape[0]), int(arr.shape[1])


def _pairwise_distances(a: Any, b: Any) -> Any:
    np = _np()
    aa = _as_numpy_2d(a)
    bb = _as_numpy_2d(b)
    diff = aa[:, None, :] - bb[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _chunks(x: Any, size: int) -> Iterable[Any]:
    for start in range(0, int(x.shape[0]), int(size)):
        yield x[start : start + int(size)]


def _np() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Family E1 direct embedding samplers require numpy.") from exc
    return np


def _hash_values(values: Sequence[object]) -> str:
    payload = "|".join(str(v) for v in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _expert_sort_value(value: str) -> int:
    try:
        return int(str(value).replace("center_", "").replace("x", ""))
    except ValueError:
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


def _to_float(value: object) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        return math.nan
    return float(text)


def _row_get(row: Mapping[str, object] | FamilyE1MatrixRow, key: str) -> object:
    if isinstance(row, FamilyE1MatrixRow):
        return getattr(row, key)
    return row.get(key, "")


def _nanmean(values: Iterable[float]) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    return float(mean(cleaned)) if cleaned else math.nan


def _nan_to_neg_inf(value: float) -> float:
    return float("-inf") if math.isnan(float(value)) else float(value)


def _delta_or_nan(left: float, right: float) -> float:
    if math.isnan(float(left)) or math.isnan(float(right)):
        return math.nan
    return float(left) - float(right)


def _mean_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    out: dict[str, object] = {}
    for key in keys:
        values: list[float] = []
        for row in rows:
            try:
                value = float(row.get(key, math.nan))
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                values.append(value)
        if values:
            out[key] = _nanmean(values)
    return out
