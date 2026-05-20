"""PCA-64 class-conditional CVAE downstream diagnostic.

This experiment is isolated from the locked unconditioned PCA64 CVAE run.  It
keeps the PCA64/scaler/classifier protocol fixed and changes only the source
expert generator: the CVAE is conditioned on source-train class labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .downstream import fit_locked_logistic_classifier
from .family_c_pca64 import (
    FamilyCPca64BuildLimits,
    NelboScore,
    Pca64Preprocessor,
    Pca64TargetSplit,
    build_pca64_target_split,
    fit_source_train_pca64_preprocessor,
    inverse_transform_pca64,
    source_normalized_nelbo,
    transform_pca64,
)
from .matrix import (
    EmbeddingCache,
    SupportRunArtifacts,
    _domain,
    _failure_status,
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _read_support_run_dimensions,
    _records_for_split,
    _resolve_torch_device,
    _torch_generator,
    build_class_reference_pools,
    discover_support_run_artifacts,
)
from .protocol import ProtocolError
from .routing import SupportSelectionUnit, read_support_selection_units
from .schemas import CLASSIFIER_SEEDS, EXPERIMENT_SEEDS, GENERATION_SEEDS, SUPPORT_NELBO_METHOD, SUPPORT_SEEDS, SUPPORT_SIZES


FAMILY_C_PCA64_CC_NAME = "family_c_pca64_class_conditional_cvae_downstream_v1"
FAMILY_C_PCA64_CC_SCHEMA_VERSION = "family_c_pca64_class_conditional_cvae_downstream_v1"
PCA64_CC_CVAE_MODE = "family_c_pca64_class_conditional_cvae_reference_posterior_resampling"
PCA64_CC_REAL_RECONSTRUCTION_MODE = "pca64_real_reconstruction_upper"
PCA64_CC_RAW_SELECTOR = "family_c_pca64_cc_uniform_support_nelbo"
PCA64_CC_SOURCE_PRIOR_SELECTOR = "family_c_pca64_cc_source_prior_support_nelbo"
PCA64_CC_GLOBAL_PRIOR_SELECTOR = "family_c_pca64_cc_global_source_prior_support_nelbo"
PCA64_CC_SINGLE_EXPERT_ROW_TYPE = "single_expert_pca64_class_conditional_cvae"
PCA64_CC_REAL_UPPER_ROW_TYPE = "real_reconstruction_upper"
PCA64_CC_FEATURE_SPACE = "standardized_pca64_class_conditional"
PCA64_CC_PCA_DIM = 64
PCA64_CC_BUDGET_PER_CLASS = 128
PCA64_CC_CLASS_LABELS = (0, 1)
PCA64_CC_CONDITION_DIM = 2
PCA64_CC_NORMALIZED_EPS = 1e-8
PCA64_CC_SMALL_STD_THRESHOLD = 1e-8


PCA64_CC_MATRIX_COLUMNS = (
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
    "support_nelbo_raw",
    "support_nelbo_source_prior",
    "support_nelbo_global_source_prior",
    "support_nelbo_source_normalized",
    "target_eval_nelbo_unlabeled",
    "target_eval_nelbo_source_prior",
    "target_eval_nelbo_global_source_prior",
    "available",
    "status",
    "error_message",
)


PCA64_CC_ALIGNMENT_COLUMNS = (
    "selector",
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed",
    "classifier_seed",
    "selected_expert",
    "selected_score",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "downstream_oracle_bacc",
    "downstream_oracle_macro_f1",
    "density_oracle_expert",
    "density_oracle_nelbo",
    "oracle_agreement",
    "oracle_gap_bacc",
    "top1_oracle_hit",
    "spearman_neg_nelbo_vs_bacc",
    "available",
    "status",
)


PCA64_CC_PREPROCESSING_COLUMNS = (
    "experiment_seed",
    "source_center",
    "feature_extractor",
    "split_id",
    "pca_dim",
    "standardized",
    "pca_artifact_id",
    "scaler_artifact_id",
    "preprocessing_artifact_key",
    "n_fit_samples",
    "embedding_dim",
    "pca_explained_variance_ratio_sum",
    "pca_coord_mean_before_scaling",
    "pca_coord_std_before_scaling",
    "scaler_mean",
    "scaler_scale",
    "pca_fit_centers",
    "target_rows_used_for_fit",
    "available",
    "status",
)


PCA64_CC_CHECKPOINT_COLUMNS = (
    "experiment_seed",
    "source_center",
    "checkpoint_path",
    "input_dim",
    "hidden_dim",
    "latent_dim",
    "feature_space",
    "conditioning",
    "metadata_dim",
    "pca_artifact_id",
    "scaler_artifact_id",
    "source_class_prior_json",
    "source_train_nelbo_mean",
    "source_train_nelbo_std",
    "kl_beta",
    "decoder_output_variance_assumption",
    "available",
    "status",
)


PCA64_CC_NELBO_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "source_center",
    "split_role",
    "n_samples",
    "support_recon_term",
    "support_kl_term",
    "support_total_nelbo",
    "support_nelbo_source_prior",
    "support_nelbo_global_source_prior",
    "source_train_nelbo_mean",
    "source_train_nelbo_std",
    "source_normalized_nelbo",
    "normalized_available",
    "target_eval_nelbo_unlabeled_diagnostic_only",
    "target_eval_nelbo_source_prior_diagnostic_only",
    "target_eval_nelbo_global_source_prior_diagnostic_only",
    "nelbo_reduction",
    "recon_reduction",
    "primary_class_prior",
)


PCA64_CC_GENERATION_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "candidate_expert",
    "generation_mode",
    "generation_seed",
    "class_label",
    "class_condition",
    "num_generated_per_class_actual",
    "class_generation_failures",
    "generated_pca_std_mean",
    "generated_dino_std_mean",
    "source_pca_std_mean",
    "source_dino_std_mean",
    "generated_dino_std_ratio",
    "generated_shape",
    "generated_nan_count",
    "generated_inf_count",
    "generated_class_centroid_distance",
    "source_class_centroid_distance",
    "generated_class_centroid_ratio",
    "generated_class_overlap_score",
    "missing_class_generation",
    "low_generated_dino_std_ratio",
    "low_generated_pca_std_ratio",
    "low_generated_class_centroid_ratio",
    "generated_class_overlap_too_high",
    "inverse_scaler_used",
    "inverse_pca_used",
    "classifier_space",
)


PCA64_CC_PROTOCOL_AUDIT_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generation_mode",
    "generation_seed",
    "classifier_seed",
    "lineage_key",
    "preprocessing_artifact_key",
    "pca_artifact_id",
    "scaler_artifact_id",
    "cvae_input_dim",
    "class_condition_dim",
    "generated_embedding_dim",
    "target_expert_excluded",
    "support_eval_disjoint",
    "pca_fit_split",
    "scaler_fit_split",
    "cvae_fit_split",
    "target_center_excluded_from_pca",
    "target_rows_used_for_pca",
    "target_rows_used_for_scaler",
    "source_labels_used_for_cvae_training",
    "target_labels_used_for_cvae_training",
    "support_labels_used_for_nelbo",
    "target_eval_labels_used_for_selection",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "classifier_scaler_fit",
    "checkpoint_feature_space",
    "checkpoint_conditioning",
    "available",
    "status",
)


PCA64_CC_BASELINE_COLUMNS = (
    "method",
    "row_type",
    "center_level_mean_bacc",
    "center_level_median_bacc",
    "center_level_mean_macro_f1",
    "center_level_mean_oracle_gap_bacc",
    "top1_oracle_hit_rate",
    "oracle_agreement_rate",
    "spearman_neg_nelbo_vs_bacc",
    "delta_vs_pca64_unconditioned_selected_bacc",
    "delta_vs_pca64_unconditioned_oracle_bacc",
    "delta_vs_pca_gmm_oracle_bacc",
    "available",
)


@dataclass(frozen=True)
class FamilyCPca64ClassConditionalConfig:
    experiment_name: str = FAMILY_C_PCA64_CC_NAME
    candidate_domains: tuple[str, ...] = ("0", "1", "2", "3", "4")
    experiment_seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    support_sizes: tuple[int, ...] = SUPPORT_SIZES
    support_seeds: tuple[int, ...] = SUPPORT_SEEDS
    generation_seeds: tuple[int, ...] = GENERATION_SEEDS
    classifier_seeds: tuple[int, ...] = CLASSIFIER_SEEDS
    support_selection_glob: str = (
        "cvae_testing/outputs/camelyon17/"
        "camelyon17_support_estimated_utility_routing_v2/"
        "support_utility_v2_seed*/reports/support_response_sample_selections.csv"
    )
    artifacts_root: str = "cvae_downstream_evaluation/artifacts/family_c_pca64_class_conditional_cvae_downstream_v1"
    pca_dim: int = PCA64_CC_PCA_DIM
    budget_per_class: int = PCA64_CC_BUDGET_PER_CLASS
    hidden_dim: int | None = None
    latent_dim: int | None = None
    lr: float = 1e-3
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    val_fraction: float = 0.2
    kl_beta: float = 1.0
    normalized_nelbo_eps: float = PCA64_CC_NORMALIZED_EPS
    small_std_threshold: float = PCA64_CC_SMALL_STD_THRESHOLD
    collapse_ratio_threshold: float = 0.25
    meaningful_oracle_gain: float = 0.02
    high_disagreement_threshold: float = 0.60
    material_oracle_gap: float = 0.02
    pca64_unconditioned_selected_bacc: float = 0.48615812746936227
    pca64_unconditioned_oracle_bacc: float = 0.6159814144466809
    pca_gmm_oracle_bacc: float = 0.844213323150479


@dataclass(frozen=True)
class ClassConditionalNelboScores:
    uniform: NelboScore
    source_prior: NelboScore
    global_source_prior: NelboScore


@dataclass(frozen=True)
class Pca64ClassConditionalExpert:
    experiment_seed: int
    source_center: str
    model: Any
    preprocessor: Pca64Preprocessor
    source_class_prior: dict[int, float]
    global_source_prior: dict[int, float]
    source_train_nelbo_mean: float
    source_train_nelbo_std: float
    checkpoint_path: Path
    input_dim: int
    hidden_dim: int
    latent_dim: int
    kl_beta: float


def default_family_c_pca64_cc_config() -> FamilyCPca64ClassConditionalConfig:
    return FamilyCPca64ClassConditionalConfig()


def load_family_c_pca64_cc_config(path: Path) -> FamilyCPca64ClassConditionalConfig:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_c_pca64_cc_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_c_pca64_cc_config()
    loaded = yaml.safe_load(text) or {}
    experiment = _mapping(loaded.get("experiment"), "experiment")
    datasets = _mapping(loaded.get("datasets"), "datasets")
    camelyon = _mapping(datasets.get("camelyon17"), "datasets.camelyon17")
    preprocessing = _mapping(loaded.get("preprocessing"), "preprocessing")
    training = _mapping(loaded.get("training"), "training")
    generation = _mapping(loaded.get("generation"), "generation")
    artifacts = _mapping(loaded.get("artifacts"), "artifacts")
    support_inputs = _mapping(loaded.get("support_inputs"), "support_inputs")
    decision = _mapping(loaded.get("decision_rule"), "decision_rule")
    baselines = _mapping(loaded.get("baseline_references"), "baseline_references")
    return FamilyCPca64ClassConditionalConfig(
        experiment_name=str(experiment.get("name", FAMILY_C_PCA64_CC_NAME)),
        candidate_domains=tuple(str(v) for v in camelyon.get("candidate_domains", ("0", "1", "2", "3", "4"))),
        experiment_seeds=tuple(int(v) for v in camelyon.get("experiment_seeds", EXPERIMENT_SEEDS)),
        support_sizes=tuple(int(v) for v in camelyon.get("support_sizes", SUPPORT_SIZES)),
        support_seeds=tuple(int(v) for v in camelyon.get("support_seeds", SUPPORT_SEEDS)),
        generation_seeds=tuple(int(v) for v in camelyon.get("generation_seeds", GENERATION_SEEDS)),
        classifier_seeds=tuple(int(v) for v in camelyon.get("classifier_seeds", CLASSIFIER_SEEDS)),
        support_selection_glob=str(support_inputs.get("selection_glob", default_family_c_pca64_cc_config().support_selection_glob)),
        artifacts_root=str(artifacts.get("root", default_family_c_pca64_cc_config().artifacts_root)),
        pca_dim=int(preprocessing.get("pca_dim", PCA64_CC_PCA_DIM)),
        budget_per_class=int(generation.get("budget_per_class", PCA64_CC_BUDGET_PER_CLASS)),
        hidden_dim=_optional_int(training.get("hidden_dim")),
        latent_dim=_optional_int(training.get("latent_dim")),
        lr=float(training.get("lr", 1e-3)),
        epochs=int(training.get("epochs", 80)),
        patience=int(training.get("patience", 10)),
        batch_size=int(training.get("batch_size", 128)),
        val_fraction=float(training.get("val_fraction", 0.2)),
        kl_beta=float(training.get("kl_beta", 1.0)),
        normalized_nelbo_eps=float(training.get("normalized_nelbo_eps", PCA64_CC_NORMALIZED_EPS)),
        small_std_threshold=float(decision.get("small_std_threshold", PCA64_CC_SMALL_STD_THRESHOLD)),
        collapse_ratio_threshold=float(decision.get("collapse_ratio_threshold", 0.25)),
        meaningful_oracle_gain=float(decision.get("meaningful_oracle_gain", 0.02)),
        high_disagreement_threshold=float(decision.get("high_disagreement_threshold", 0.60)),
        material_oracle_gap=float(decision.get("material_oracle_gap", 0.02)),
        pca64_unconditioned_selected_bacc=float(baselines.get("pca64_unconditioned_selected_bacc", 0.48615812746936227)),
        pca64_unconditioned_oracle_bacc=float(baselines.get("pca64_unconditioned_oracle_bacc", 0.6159814144466809)),
        pca_gmm_oracle_bacc=float(baselines.get("pca_gmm_oracle_bacc", 0.844213323150479)),
    )


def assert_family_c_pca64_cc_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_C_PCA64_CC_NAME}",
        "feature_space: standardized_pca64_class_conditional",
        "conditioning: class_label_one_hot",
        "pca_dim: 64",
        "uniform_class_prior",
        "source_prior_diagnostic",
        "support_labels_for_primary_routing: forbidden",
        "target_eval_labels_used_for_selection: forbidden",
        "budget_per_class: 128",
        "meaningful_oracle_gain: 0.02",
        "family_c_pca64_cc_all_expert_downstream_matrix.csv",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"Family C PCA64 class-conditional config missing required fields: {missing}")


def assert_pca64_cc_checkpoint_metadata(metadata: Mapping[str, object], *, expected: Mapping[str, object]) -> None:
    required = {
        "input_dim": 64,
        "feature_space": PCA64_CC_FEATURE_SPACE,
        "conditioning": "class_label_one_hot",
        "metadata_dim": PCA64_CC_CONDITION_DIM,
        "pca_artifact_id": expected.get("pca_artifact_id"),
        "scaler_artifact_id": expected.get("scaler_artifact_id"),
        "source_center": str(expected.get("source_center")),
        "experiment_seed": int(expected.get("experiment_seed", 0)),
    }
    for key, value in required.items():
        actual = metadata.get(key)
        if key in {"input_dim", "metadata_dim", "experiment_seed"}:
            actual = int(actual) if str(actual).strip() else actual
        else:
            actual = str(actual)
            value = str(value)
        if actual != value:
            raise ProtocolError(
                f"Incompatible PCA64 class-conditional CVAE checkpoint metadata for {key}: "
                f"got {actual!r}, expected {value!r}"
            )


def class_one_hot(labels: Sequence[int], *, torch: Any | None = None, device: Any | None = None) -> Any:
    values = [int(v) for v in labels]
    for value in values:
        if value not in PCA64_CC_CLASS_LABELS:
            raise ProtocolError(f"Class-conditional PCA64 CVAE expects labels {PCA64_CC_CLASS_LABELS}, got {value}")
    if torch is None:
        import numpy as np  # type: ignore

        out = np.zeros((len(values), PCA64_CC_CONDITION_DIM), dtype="float32")
        for i, value in enumerate(values):
            out[i, value] = 1.0
        return out
    out = torch.zeros((len(values), PCA64_CC_CONDITION_DIM), dtype=torch.float32, device=device)
    for i, value in enumerate(values):
        out[i, value] = 1.0
    return out


def score_label_marginal_nelbo(
    model: Any,
    x_standardized_pca64: Any,
    *,
    class_prior: Mapping[int, float],
    torch: Any,
    device: Any,
    kl_beta: float = 1.0,
) -> NelboScore:
    x_np = _as_numpy(x_standardized_pca64)
    if int(x_np.ndim) != 2 or int(x_np.shape[1]) != PCA64_CC_PCA_DIM:
        raise ProtocolError(f"Expected standardized PCA64 input shape [n,64], got {tuple(x_np.shape)}")
    if int(x_np.shape[0]) == 0:
        return NelboScore(total=math.nan, recon=math.nan, kl=math.nan, n_samples=0)
    prior = _normalized_prior(class_prior)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    log_terms = []
    recon_terms = []
    kl_terms = []
    model.eval()
    with torch.no_grad():
        for class_label in PCA64_CC_CLASS_LABELS:
            m = class_one_hot([class_label] * int(x.shape[0]), torch=torch, device=device)
            mu, logvar = model.encode(x, m=m)
            recon = model.decode(mu, m=m)
            recon_term = torch.mean((recon - x).pow(2), dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total = recon_term + (float(kl_beta) * kl)
            log_terms.append(math.log(float(prior[class_label])) - total)
            recon_terms.append(recon_term)
            kl_terms.append(kl)
        log_stack = torch.stack(log_terms, dim=0)
        weights = torch.softmax(log_stack, dim=0)
        marginal = -torch.logsumexp(log_stack, dim=0)
        recon_weighted = torch.sum(weights * torch.stack(recon_terms, dim=0), dim=0)
        kl_weighted = torch.sum(weights * torch.stack(kl_terms, dim=0), dim=0)
    return NelboScore(
        total=float(torch.mean(marginal).detach().cpu().item()),
        recon=float(torch.mean(recon_weighted).detach().cpu().item()),
        kl=float(torch.mean(kl_weighted).detach().cpu().item()),
        n_samples=int(x_np.shape[0]),
    )


def build_family_c_pca64_cc_all_expert_downstream_matrix(
    *,
    config: FamilyCPca64ClassConditionalConfig,
    repo_root: Path,
    artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str = "auto",
    resume: bool = False,
    limits: FamilyCPca64BuildLimits = FamilyCPca64BuildLimits(),
) -> dict[str, Path]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifacts_root)
    completed = _read_completed_matrix_keys(paths["matrix"]) if resume else set()
    artifacts = _limit_artifacts(discover_support_run_artifacts(config=config, repo_root=repo_root), limits.experiment_seeds)
    primary_units = _primary_support_units(support_units, limits=limits)
    units_by_seed = _units_by_seed(primary_units)
    selected_generation_seeds = limits.generation_seeds or config.generation_seeds
    selected_classifier_seeds = limits.classifier_seeds or config.classifier_seeds
    selected_heldout_centers = limits.heldout_centers or config.candidate_domains

    for artifact in artifacts:
        seed_units = units_by_seed.get(int(artifact.experiment_seed), ())
        if not seed_units:
            raise ProtocolError(f"No support units for experiment_seed={artifact.experiment_seed}")
        samples = _read_samples_manifest(artifact.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(artifact.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(artifact.test_cache, test_records, repo_root=repo_root)
        dimensions = _read_support_run_dimensions(artifact.config_resolved)
        feature_extractor = str(dimensions.get("feature_extractor_checkpoint", ""))
        global_prior = _class_prior_for_indices(range(len(train_cache.metadata)), train_cache.metadata)
        bank, prep_rows, checkpoint_rows = fit_or_load_pca64_cc_cvae_bank(
            config=config,
            artifact=artifact,
            train_cache=train_cache,
            dimensions=dimensions,
            feature_extractor=feature_extractor,
            global_source_prior=global_prior,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            device=device,
        )
        _append_unique_dict_rows(paths["preprocessing"], PCA64_CC_PREPROCESSING_COLUMNS, prep_rows, _preprocessing_manifest_key)
        _append_unique_dict_rows(paths["checkpoints"], PCA64_CC_CHECKPOINT_COLUMNS, checkpoint_rows, _checkpoint_manifest_key)

        for unit in seed_units:
            heldout = str(unit.heldout_center)
            if heldout not in {str(v) for v in selected_heldout_centers}:
                continue
            candidates = tuple(str(c) for c in unit.candidate_experts if str(c) != heldout)
            if heldout in candidates or not unit.target_expert_excluded:
                raise ProtocolError(f"Target expert leakage in support unit {unit.support_eval_split_id}")
            target_split = build_pca64_target_split(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_size=int(unit.support_size),
                support_seed=int(unit.support_seed),
                support_eval_split_id=str(unit.support_eval_split_id),
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_split.eval_indices]
            target_status = _target_label_status(target_labels)
            support_scores: dict[str, ClassConditionalNelboScores] = {}
            target_density_scores: dict[str, ClassConditionalNelboScores] = {}
            nelbo_rows: list[dict[str, object]] = []
            for expert in candidates:
                pca_expert = bank[str(expert)]
                support_x = transform_pca64(pca_expert.preprocessor, _slice_embeddings(test_cache.embeddings, target_split.support_indices))
                target_x = transform_pca64(pca_expert.preprocessor, _slice_embeddings(test_cache.embeddings, target_split.eval_indices))
                support_score = _score_all_priors(pca_expert, support_x)
                target_score = _score_all_priors(pca_expert, target_x)
                support_scores[str(expert)] = support_score
                target_density_scores[str(expert)] = target_score
                normalized, norm_available = source_normalized_nelbo(
                    support_score.uniform.total,
                    mean=pca_expert.source_train_nelbo_mean,
                    std=pca_expert.source_train_nelbo_std,
                    eps=config.normalized_nelbo_eps,
                )
                nelbo_rows.append(
                    _nelbo_row(
                        unit=unit,
                        heldout_center=heldout,
                        source_center=expert,
                        split_role="target_support",
                        scores=support_score,
                        source_mean=pca_expert.source_train_nelbo_mean,
                        source_std=pca_expert.source_train_nelbo_std,
                        normalized=normalized,
                        normalized_available=norm_available,
                        target_eval_scores=target_score,
                    )
                )
            _append_unique_dict_rows(paths["nelbo"], PCA64_CC_NELBO_COLUMNS, nelbo_rows, _nelbo_manifest_key)

            for expert in candidates:
                reference_pools = _standardized_reference_pools(
                    train_cache=train_cache,
                    expert=bank[str(expert)],
                    required_labels=PCA64_CC_CLASS_LABELS,
                )
                for generation_seed in selected_generation_seeds:
                    for classifier_seed in selected_classifier_seeds:
                        row, generation_rows, audit = score_pca64_cc_cvae_candidate(
                            config=config,
                            experiment_seed=int(artifact.experiment_seed),
                            heldout_center=heldout,
                            unit=unit,
                            candidate_expert=str(expert),
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            target_split=target_split,
                            target_status=target_status,
                            target_labels=target_labels,
                            target_embeddings_original=_slice_embeddings(test_cache.embeddings, target_split.eval_indices),
                            expert=bank[str(expert)],
                            reference_pools=reference_pools,
                            support_scores=support_scores[str(expert)],
                            target_density_scores=target_density_scores[str(expert)],
                        )
                        if not (resume and _matrix_key(row) in completed):
                            _append_dict_rows(paths["matrix"], PCA64_CC_MATRIX_COLUMNS, [row])
                            _append_unique_dict_rows(paths["generation"], PCA64_CC_GENERATION_COLUMNS, generation_rows, _generation_manifest_key)
                            _append_unique_dict_rows(paths["protocol_audit"], PCA64_CC_PROTOCOL_AUDIT_COLUMNS, [audit], _protocol_audit_key)
                            completed.add(_matrix_key(row))

                        recon_row, recon_audit = score_pca64_cc_real_reconstruction_candidate(
                            config=config,
                            experiment_seed=int(artifact.experiment_seed),
                            heldout_center=heldout,
                            unit=unit,
                            candidate_expert=str(expert),
                            classifier_seed=int(classifier_seed),
                            generation_seed=int(generation_seed),
                            target_split=target_split,
                            target_status=target_status,
                            target_labels=target_labels,
                            target_embeddings_original=_slice_embeddings(test_cache.embeddings, target_split.eval_indices),
                            train_cache=train_cache,
                            expert=bank[str(expert)],
                            support_scores=support_scores[str(expert)],
                            target_density_scores=target_density_scores[str(expert)],
                        )
                        if not (resume and _matrix_key(recon_row) in completed):
                            _append_dict_rows(paths["matrix"], PCA64_CC_MATRIX_COLUMNS, [recon_row])
                            _append_unique_dict_rows(paths["protocol_audit"], PCA64_CC_PROTOCOL_AUDIT_COLUMNS, [recon_audit], _protocol_audit_key)
                            completed.add(_matrix_key(recon_row))
    return paths


def fit_or_load_pca64_cc_cvae_bank(
    *,
    config: FamilyCPca64ClassConditionalConfig,
    artifact: SupportRunArtifacts,
    train_cache: EmbeddingCache,
    dimensions: Mapping[str, object],
    feature_extractor: str,
    global_source_prior: Mapping[int, float],
    repo_root: Path,
    artifacts_root: Path,
    device: str,
) -> tuple[dict[str, Pca64ClassConditionalExpert], list[dict[str, object]], list[dict[str, object]]]:
    torch, CVAEExpert, load_model_checkpoint, wrap_model_state_dict = _pca64_torch_imports(repo_root)
    resolved_device = _resolve_torch_device(torch, device)
    bank: dict[str, Pca64ClassConditionalExpert] = {}
    prep_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    hidden_dim = int(config.hidden_dim if config.hidden_dim is not None else dimensions["hidden_dim"])
    latent_dim = int(config.latent_dim if config.latent_dim is not None else dimensions["latent_dim"])
    split_id = f"source_train_seed{artifact.experiment_seed}"
    for source_center in config.candidate_domains:
        prep = fit_source_train_pca64_preprocessor(
            experiment_seed=int(artifact.experiment_seed),
            source_center=str(source_center),
            train_cache=train_cache,
            feature_extractor=feature_extractor,
            split_id=split_id,
            pca_dim=config.pca_dim,
        )
        prep_rows.append(_preprocessing_row(prep))
        source_indices = _center_indices(train_cache, source_center)
        source_labels = [_label(train_cache.metadata[idx]) for idx in source_indices]
        _assert_has_all_source_classes(source_labels, source_center)
        source_x = transform_pca64(prep, _slice_embeddings(train_cache.embeddings, source_indices))
        source_prior = _class_prior_for_indices(source_indices, train_cache.metadata)
        checkpoint_path = (
            artifacts_root
            / "checkpoints"
            / f"seed{int(artifact.experiment_seed)}"
            / f"expert_{source_center}_standardized_pca64_class_conditional.pt"
        )
        metadata = {
            "schema_version": FAMILY_C_PCA64_CC_SCHEMA_VERSION,
            "input_dim": int(config.pca_dim),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "feature_space": PCA64_CC_FEATURE_SPACE,
            "conditioning": "class_label_one_hot",
            "metadata_dim": PCA64_CC_CONDITION_DIM,
            "pca_artifact_id": prep.pca_artifact_id,
            "scaler_artifact_id": prep.scaler_artifact_id,
            "source_center": str(source_center),
            "experiment_seed": int(artifact.experiment_seed),
            "pca_dim": int(config.pca_dim),
            "standardized": True,
            "kl_beta": float(config.kl_beta),
            "source_class_prior_json": _json_prior(source_prior),
        }
        if checkpoint_path.exists():
            loaded = load_model_checkpoint(checkpoint_path, map_location=resolved_device)
            assert_pca64_cc_checkpoint_metadata(loaded.checkpoint_metadata, expected=metadata)
            model = CVAEExpert(
                int(config.pca_dim),
                hidden_dim,
                latent_dim,
                metadata_dim=PCA64_CC_CONDITION_DIM,
                aux_metadata_dim=PCA64_CC_CONDITION_DIM,
            ).to(resolved_device)
            model.load_state_dict(loaded.model_state_dict)
            status = "loaded"
        else:
            model = CVAEExpert(
                int(config.pca_dim),
                hidden_dim,
                latent_dim,
                metadata_dim=PCA64_CC_CONDITION_DIM,
                aux_metadata_dim=PCA64_CC_CONDITION_DIM,
            ).to(resolved_device)
            _train_pca64_cc_cvae(
                model=model,
                torch=torch,
                device=resolved_device,
                x=source_x,
                labels=source_labels,
                config=config,
                seed=_stable_seed(artifact.experiment_seed, source_center, "class_conditional_train"),
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(wrap_model_state_dict(model.state_dict(), metadata), checkpoint_path)
            status = "trained"
        model.eval()
        model._pca64_torch = torch
        model._pca64_device = resolved_device
        expert_shell = Pca64ClassConditionalExpert(
            experiment_seed=int(artifact.experiment_seed),
            source_center=str(source_center),
            model=model,
            preprocessor=prep,
            source_class_prior=dict(source_prior),
            global_source_prior=dict(global_source_prior),
            source_train_nelbo_mean=math.nan,
            source_train_nelbo_std=math.nan,
            checkpoint_path=checkpoint_path,
            input_dim=int(config.pca_dim),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            kl_beta=float(config.kl_beta),
        )
        source_score = score_label_marginal_nelbo(
            model,
            source_x,
            class_prior={0: 0.5, 1: 0.5},
            torch=torch,
            device=resolved_device,
            kl_beta=config.kl_beta,
        )
        source_values = _sample_label_marginal_nelbo_values(
            model,
            source_x,
            class_prior={0: 0.5, 1: 0.5},
            torch=torch,
            device=resolved_device,
            kl_beta=config.kl_beta,
        )
        source_std = _std(source_values)
        expert = Pca64ClassConditionalExpert(
            **{**expert_shell.__dict__, "source_train_nelbo_mean": float(source_score.total), "source_train_nelbo_std": source_std}
        )
        bank[str(source_center)] = expert
        checkpoint_rows.append(
            {
                "experiment_seed": int(artifact.experiment_seed),
                "source_center": str(source_center),
                "checkpoint_path": str(checkpoint_path),
                "input_dim": int(config.pca_dim),
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "feature_space": PCA64_CC_FEATURE_SPACE,
                "conditioning": "class_label_one_hot",
                "metadata_dim": PCA64_CC_CONDITION_DIM,
                "pca_artifact_id": prep.pca_artifact_id,
                "scaler_artifact_id": prep.scaler_artifact_id,
                "source_class_prior_json": _json_prior(source_prior),
                "source_train_nelbo_mean": float(source_score.total),
                "source_train_nelbo_std": source_std,
                "kl_beta": float(config.kl_beta),
                "decoder_output_variance_assumption": "unit_variance_mse_proxy",
                "available": 1,
                "status": status,
            }
        )
    return bank, prep_rows, checkpoint_rows


def score_pca64_cc_cvae_candidate(
    *,
    config: FamilyCPca64ClassConditionalConfig,
    experiment_seed: int,
    heldout_center: str,
    unit: SupportSelectionUnit,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    target_split: Pca64TargetSplit,
    target_status: Mapping[str, object],
    target_labels: Sequence[int],
    target_embeddings_original: Any,
    expert: Pca64ClassConditionalExpert,
    reference_pools: Mapping[int, Any],
    support_scores: ClassConditionalNelboScores,
    target_density_scores: ClassConditionalNelboScores,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    base = _row_base(
        schema_version=FAMILY_C_PCA64_CC_SCHEMA_VERSION,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        unit=unit,
        candidate_expert=candidate_expert,
        generation_mode=PCA64_CC_CVAE_MODE,
        budget_per_class=config.budget_per_class,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        row_type=PCA64_CC_SINGLE_EXPERT_ROW_TYPE,
        n_target_eval=len(target_split.eval_indices),
        target_eval_pool_id=target_split.target_eval_pool_id,
        target_status=target_status,
        support_scores=support_scores,
        target_density_scores=target_density_scores,
        source_normalized=source_normalized_nelbo(
            support_scores.uniform.total,
            mean=expert.source_train_nelbo_mean,
            std=expert.source_train_nelbo_std,
            eps=config.normalized_nelbo_eps,
        )[0],
    )
    generation_rows: list[dict[str, object]] = []
    try:
        synthetic, labels, generation_rows = generate_pca64_cc_cvae_class_balanced(
            expert=expert,
            reference_pools=reference_pools,
            class_labels=PCA64_CC_CLASS_LABELS,
            budget_per_class=config.budget_per_class,
            generation_seed=generation_seed,
            context=base,
            small_std_threshold=config.small_std_threshold,
            collapse_ratio_threshold=config.collapse_ratio_threshold,
        )
        prediction = fit_locked_logistic_classifier(
            synthetic,
            labels,
            target_embeddings_original,
            target_labels,
            classifier_seed=classifier_seed,
        )
        row = {
            **base,
            "bacc": float(prediction.score.balanced_accuracy),
            "macro_f1": float(prediction.score.macro_f1),
            "auroc": float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            "auprc": float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            "n_train": len(labels),
            "available": 1,
            "status": "ok",
            "error_message": "",
        }
    except Exception as exc:
        row = _failed_row(base, exc)
    audit = _audit_row(row=row, expert=expert, target_expert_excluded=int(str(heldout_center) != str(candidate_expert)))
    return row, generation_rows, audit


def score_pca64_cc_real_reconstruction_candidate(
    *,
    config: FamilyCPca64ClassConditionalConfig,
    experiment_seed: int,
    heldout_center: str,
    unit: SupportSelectionUnit,
    candidate_expert: str,
    generation_seed: int,
    classifier_seed: int,
    target_split: Pca64TargetSplit,
    target_status: Mapping[str, object],
    target_labels: Sequence[int],
    target_embeddings_original: Any,
    train_cache: EmbeddingCache,
    expert: Pca64ClassConditionalExpert,
    support_scores: ClassConditionalNelboScores,
    target_density_scores: ClassConditionalNelboScores,
) -> tuple[dict[str, object], dict[str, object]]:
    base = _row_base(
        schema_version=FAMILY_C_PCA64_CC_SCHEMA_VERSION,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        unit=unit,
        candidate_expert=candidate_expert,
        generation_mode=PCA64_CC_REAL_RECONSTRUCTION_MODE,
        budget_per_class=0,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        row_type=PCA64_CC_REAL_UPPER_ROW_TYPE,
        n_target_eval=len(target_split.eval_indices),
        target_eval_pool_id=target_split.target_eval_pool_id,
        target_status=target_status,
        support_scores=support_scores,
        target_density_scores=target_density_scores,
        source_normalized=source_normalized_nelbo(
            support_scores.uniform.total,
            mean=expert.source_train_nelbo_mean,
            std=expert.source_train_nelbo_std,
            eps=config.normalized_nelbo_eps,
        )[0],
    )
    try:
        idxs = _center_indices(train_cache, candidate_expert)
        labels = [_label(train_cache.metadata[idx]) for idx in idxs]
        x_source = _slice_embeddings(train_cache.embeddings, idxs)
        x_reconstructed = inverse_transform_pca64(expert.preprocessor, transform_pca64(expert.preprocessor, x_source))
        prediction = fit_locked_logistic_classifier(
            x_reconstructed,
            labels,
            target_embeddings_original,
            target_labels,
            classifier_seed=classifier_seed,
        )
        row = {
            **base,
            "bacc": float(prediction.score.balanced_accuracy),
            "macro_f1": float(prediction.score.macro_f1),
            "auroc": float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            "auprc": float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            "n_train": len(labels),
            "available": 1,
            "status": "ok",
            "error_message": "",
        }
    except Exception as exc:
        row = _failed_row(base, exc)
    audit = _audit_row(row=row, expert=expert, target_expert_excluded=int(str(heldout_center) != str(candidate_expert)))
    return row, audit


def generate_pca64_cc_cvae_class_balanced(
    *,
    expert: Pca64ClassConditionalExpert,
    reference_pools: Mapping[int, Any],
    class_labels: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
    context: Mapping[str, object],
    small_std_threshold: float,
    collapse_ratio_threshold: float,
) -> tuple[Any, list[int], list[dict[str, object]]]:
    import numpy as np  # type: ignore

    torch = expert.model._pca64_torch
    device = expert.model._pca64_device
    chunks: dict[int, Any] = {}
    source_reconstructed: dict[int, Any] = {}
    labels: list[int] = []
    rows: list[dict[str, object]] = []
    for class_label in tuple(int(v) for v in class_labels):
        refs = reference_pools.get(class_label)
        refs_np = _as_numpy(refs)
        if refs is None or int(refs_np.shape[0]) <= 0:
            raise ProtocolError(f"Empty standardized PCA64 reference pool for class {class_label}")
        idx_gen = torch.Generator(device="cpu").manual_seed(int(generation_seed) + int(class_label))
        idx = torch.randint(int(refs_np.shape[0]), (int(budget_per_class),), generator=idx_gen, device="cpu")
        xb = torch.as_tensor(refs_np[idx.numpy()], dtype=torch.float32, device=device)
        m = class_one_hot([class_label] * int(budget_per_class), torch=torch, device=device)
        gen = _torch_generator(torch, device, int(generation_seed) + 104729 + int(class_label))
        with torch.no_grad():
            mu, logvar = expert.model.encode(xb, m=m)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape, generator=gen, device=device, dtype=std.dtype)
            z = mu + eps * std
            generated_pca = expert.model.decode(z, m=m).detach().cpu().numpy()
        generated_dino = inverse_transform_pca64(expert.preprocessor, generated_pca)
        if int(generated_dino.shape[1]) != expert.preprocessor.embedding_dim:
            raise ProtocolError(
                f"Generated inverse PCA embeddings have wrong dim {generated_dino.shape[1]}, "
                f"expected {expert.preprocessor.embedding_dim}"
            )
        source_reconstructed[class_label] = inverse_transform_pca64(expert.preprocessor, refs_np)
        chunks[class_label] = generated_dino
        labels.extend([class_label] * int(budget_per_class))
    generated = np.vstack([chunks[int(label)] for label in class_labels])
    gen_labels = [int(label) for label in class_labels for _ in range(int(chunks[int(label)].shape[0]))]
    source = np.vstack([source_reconstructed[int(label)] for label in class_labels])
    source_labels = [int(label) for label in class_labels for _ in range(int(source_reconstructed[int(label)].shape[0]))]
    global_diag = _generation_global_diagnostics(
        generated=generated,
        generated_labels=gen_labels,
        source=source,
        source_labels=source_labels,
        threshold=collapse_ratio_threshold,
    )
    for class_label in tuple(int(v) for v in class_labels):
        generated_dino = chunks[class_label]
        generated_pca = transform_pca64(expert.preprocessor, generated_dino)
        source_dino = source_reconstructed[class_label]
        pca_std = float(np.std(generated_pca))
        source_pca_std = float(np.std(reference_pools[class_label]))
        dino_std = float(np.std(generated_dino))
        source_dino_std = float(np.std(source_dino))
        dino_std_ratio = dino_std / max(source_dino_std, 1e-12)
        pca_std_ratio = pca_std / max(source_pca_std, 1e-12)
        nan_count = int(np.isnan(generated_dino).sum() + np.isnan(generated_pca).sum())
        inf_count = int(np.isinf(generated_dino).sum() + np.isinf(generated_pca).sum())
        failures = []
        if int(generated_dino.shape[0]) != int(budget_per_class):
            failures.append("missing_class_generation")
        if pca_std <= float(small_std_threshold):
            failures.append("collapsed_pca_variance")
        if dino_std <= float(small_std_threshold):
            failures.append("collapsed_dino_variance")
        if pca_std_ratio < float(collapse_ratio_threshold):
            failures.append("low_generated_pca_std_ratio")
        if dino_std_ratio < float(collapse_ratio_threshold):
            failures.append("low_generated_dino_std_ratio")
        if int(global_diag["low_generated_class_centroid_ratio"]):
            failures.append("low_generated_class_centroid_ratio")
        if int(global_diag["generated_class_overlap_too_high"]):
            failures.append("generated_class_overlap_too_high")
        if nan_count > 0:
            failures.append("generated_nan")
        if inf_count > 0:
            failures.append("generated_inf")
        rows.append(
            {
                "experiment_seed": context["experiment_seed"],
                "heldout_center": context["heldout_center"],
                "support_size": context["support_size"],
                "support_seed": context["support_seed"],
                "candidate_expert": context["candidate_expert"],
                "generation_mode": context["generation_mode"],
                "generation_seed": context["generation_seed"],
                "class_label": class_label,
                "class_condition": json.dumps(_condition_list(class_label), separators=(",", ":")),
                "num_generated_per_class_actual": int(generated_dino.shape[0]),
                "class_generation_failures": "|".join(failures),
                "generated_pca_std_mean": pca_std,
                "generated_dino_std_mean": dino_std,
                "source_pca_std_mean": source_pca_std,
                "source_dino_std_mean": source_dino_std,
                "generated_dino_std_ratio": dino_std_ratio,
                "generated_shape": json.dumps([int(v) for v in generated_dino.shape]),
                "generated_nan_count": nan_count,
                "generated_inf_count": inf_count,
                **global_diag,
                "missing_class_generation": int(generated_dino.shape[0] != int(budget_per_class)),
                "low_generated_dino_std_ratio": int(dino_std_ratio < float(collapse_ratio_threshold)),
                "low_generated_pca_std_ratio": int(pca_std_ratio < float(collapse_ratio_threshold)),
                "inverse_scaler_used": 1,
                "inverse_pca_used": 1,
                "classifier_space": "original_dino_after_inverse_transform",
            }
        )
    return generated, labels, rows


def build_family_c_pca64_cc_reports(
    *,
    artifacts_root: Path,
    candidate_domains: Sequence[str],
    config: FamilyCPca64ClassConditionalConfig | None = None,
) -> dict[str, Path]:
    cfg = config or default_family_c_pca64_cc_config()
    paths = _artifact_paths(artifacts_root)
    rows = _read_dict_rows(paths["matrix"])
    align = build_family_c_pca64_cc_alignment_rows(rows=rows, candidate_domains=candidate_domains)
    baseline = build_family_c_pca64_cc_baseline_rows(rows=rows, alignment_rows=align, config=cfg)
    summary = classify_family_c_pca64_cc_decision(rows=rows, alignment_rows=align, config=cfg)
    _write_dict_csv(paths["alignment"], PCA64_CC_ALIGNMENT_COLUMNS, align)
    _write_dict_csv(paths["baseline"], PCA64_CC_BASELINE_COLUMNS, baseline)
    paths["decision_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def build_family_c_pca64_cc_alignment_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_domains: Sequence[str],
) -> list[dict[str, object]]:
    _ = tuple(candidate_domains)
    generated = [
        r
        for r in rows
        if r.get("generation_mode") == PCA64_CC_CVAE_MODE
        and str(r.get("status")) == "ok"
        and str(r.get("target_eval_has_all_classes")) == "1"
    ]
    by_context: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in generated:
        by_context.setdefault(_alignment_context_key(row), []).append(row)
    out: list[dict[str, object]] = []
    for _, group in sorted(by_context.items(), key=lambda item: item[0]):
        downstream_oracle = max(group, key=lambda row: (_float(row.get("bacc")), _float(row.get("macro_f1")), _reverse_sort(str(row.get("candidate_expert")))))
        density_oracle = min(group, key=lambda row: (_float(row.get("target_eval_nelbo_unlabeled")), str(row.get("candidate_expert"))))
        for selector, score_column in (
            (PCA64_CC_RAW_SELECTOR, "support_nelbo_raw"),
            (PCA64_CC_SOURCE_PRIOR_SELECTOR, "support_nelbo_source_prior"),
            (PCA64_CC_GLOBAL_PRIOR_SELECTOR, "support_nelbo_global_source_prior"),
        ):
            available_group = [row for row in group if not math.isnan(_float(row.get(score_column)))]
            if not available_group:
                selected = group[0]
                status = f"unavailable_{selector}"
                available = 0
            else:
                selected = min(available_group, key=lambda row: (_float(row.get(score_column)), str(row.get("candidate_expert"))))
                status = "ok"
                available = 1
            out.append(
                {
                    "selector": selector,
                    "heldout_center": selected.get("heldout_center", ""),
                    "experiment_seed": selected.get("experiment_seed", ""),
                    "support_size": selected.get("support_size", ""),
                    "support_seed": selected.get("support_seed", ""),
                    "support_eval_split_id": selected.get("support_eval_split_id", ""),
                    "generation_seed": selected.get("generation_seed", ""),
                    "classifier_seed": selected.get("classifier_seed", ""),
                    "selected_expert": selected.get("candidate_expert", ""),
                    "selected_score": selected.get(score_column, ""),
                    "selected_bacc": selected.get("bacc", math.nan),
                    "selected_macro_f1": selected.get("macro_f1", math.nan),
                    "downstream_oracle_expert": downstream_oracle.get("candidate_expert", ""),
                    "downstream_oracle_bacc": downstream_oracle.get("bacc", math.nan),
                    "downstream_oracle_macro_f1": downstream_oracle.get("macro_f1", math.nan),
                    "density_oracle_expert": density_oracle.get("candidate_expert", ""),
                    "density_oracle_nelbo": density_oracle.get("target_eval_nelbo_unlabeled", math.nan),
                    "oracle_agreement": int(str(density_oracle.get("candidate_expert")) == str(downstream_oracle.get("candidate_expert"))),
                    "oracle_gap_bacc": _float(downstream_oracle.get("bacc")) - _float(selected.get("bacc")),
                    "top1_oracle_hit": int(str(selected.get("candidate_expert")) == str(downstream_oracle.get("candidate_expert"))),
                    "spearman_neg_nelbo_vs_bacc": _spearman(
                        [-_float(row.get(score_column)) for row in group],
                        [_float(row.get("bacc")) for row in group],
                    ),
                    "available": available,
                    "status": status,
                }
            )
    return out


def build_family_c_pca64_cc_baseline_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    config: FamilyCPca64ClassConditionalConfig,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for selector in (PCA64_CC_RAW_SELECTOR, PCA64_CC_SOURCE_PRIOR_SELECTOR, PCA64_CC_GLOBAL_PRIOR_SELECTOR):
        subset = [row for row in alignment_rows if row.get("selector") == selector and int(row.get("available", 0)) == 1]
        selected_bacc = _center_level_mean(subset, "selected_bacc")
        out.append(
            {
                "method": selector,
                "row_type": "selector",
                "center_level_mean_bacc": selected_bacc,
                "center_level_median_bacc": _median(_float(row.get("selected_bacc")) for row in subset),
                "center_level_mean_macro_f1": _center_level_mean(subset, "selected_macro_f1"),
                "center_level_mean_oracle_gap_bacc": _center_level_mean(subset, "oracle_gap_bacc"),
                "top1_oracle_hit_rate": _mean(_float(row.get("top1_oracle_hit")) for row in subset),
                "oracle_agreement_rate": _mean(_float(row.get("oracle_agreement")) for row in subset),
                "spearman_neg_nelbo_vs_bacc": _mean(_float(row.get("spearman_neg_nelbo_vs_bacc")) for row in subset),
                "delta_vs_pca64_unconditioned_selected_bacc": selected_bacc - float(config.pca64_unconditioned_selected_bacc),
                "delta_vs_pca64_unconditioned_oracle_bacc": math.nan,
                "delta_vs_pca_gmm_oracle_bacc": math.nan,
                "available": int(bool(subset)),
            }
        )
    generated = [
        row
        for row in rows
        if row.get("generation_mode") == PCA64_CC_CVAE_MODE
        and row.get("row_type") == PCA64_CC_SINGLE_EXPERT_ROW_TYPE
        and row.get("status") == "ok"
        and str(row.get("target_eval_has_all_classes")) == "1"
    ]
    recon = [
        row
        for row in rows
        if row.get("generation_mode") == PCA64_CC_REAL_RECONSTRUCTION_MODE
        and row.get("row_type") == PCA64_CC_REAL_UPPER_ROW_TYPE
        and row.get("status") == "ok"
        and str(row.get("target_eval_has_all_classes")) == "1"
    ]
    oracle = _oracle_center_mean(generated)
    out.append(_oracle_baseline_row(generated, method="family_c_pca64_cc_downstream_oracle", row_type="diagnostic_oracle", config=config, oracle_value=oracle))
    out.append(_oracle_baseline_row(recon, method="PCA64_real_reconstruction_upper", row_type="diagnostic_upper_bound", config=config, oracle_value=_oracle_center_mean(recon)))
    return out


def classify_family_c_pca64_cc_decision(
    *,
    rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    config: FamilyCPca64ClassConditionalConfig,
) -> dict[str, object]:
    raw_rows = [row for row in alignment_rows if row.get("selector") == PCA64_CC_RAW_SELECTOR and int(row.get("available", 0)) == 1]
    selected = _center_level_mean(raw_rows, "selected_bacc")
    oracle = _center_level_mean(raw_rows, "downstream_oracle_bacc")
    gap = _center_level_mean(raw_rows, "oracle_gap_bacc")
    top1 = _mean(_float(row.get("top1_oracle_hit")) for row in raw_rows)
    agreement = _mean(_float(row.get("oracle_agreement")) for row in raw_rows)
    selected_median = _median(_float(row.get("selected_bacc")) for row in raw_rows)
    oracle_median = _median(_float(row.get("downstream_oracle_bacc")) for row in raw_rows)
    recon_oracle = _oracle_center_mean(
        [
            row
            for row in rows
            if row.get("generation_mode") == PCA64_CC_REAL_RECONSTRUCTION_MODE
            and row.get("status") == "ok"
            and str(row.get("target_eval_has_all_classes")) == "1"
        ]
    )
    delta_selected = selected - float(config.pca64_unconditioned_selected_bacc)
    delta_oracle = oracle - float(config.pca64_unconditioned_oracle_bacc)
    delta_pca_gmm_oracle = oracle - float(config.pca_gmm_oracle_bacc)
    stable_gain = delta_oracle >= float(config.meaningful_oracle_gain) and oracle_median >= float(config.pca64_unconditioned_oracle_bacc) + float(config.meaningful_oracle_gain)
    high_disagreement = top1 <= float(config.high_disagreement_threshold) or agreement <= float(config.high_disagreement_threshold)
    material_gap = gap >= float(config.material_oracle_gap)
    if delta_oracle < float(config.meaningful_oracle_gain) or not stable_gain:
        classification = "NO_MEANINGFUL_GAIN"
    elif oracle >= 0.80 and selected < 0.80 and material_gap and high_disagreement:
        classification = "ROUTING_BOTTLENECK"
    elif oracle >= 0.80 and stable_gain:
        classification = "GENERATOR_SOLVED"
    elif oracle < 0.80 and recon_oracle >= 0.80:
        classification = "CLASS_CONDITIONING_INSUFFICIENT"
    elif oracle < 0.80 and recon_oracle < 0.80:
        classification = "PCA64_OR_SOURCE_LIMIT"
    else:
        classification = "DIAGNOSTIC_ONLY"
    all_rows = [row for row in rows if row.get("status") == "ok"]
    all_class_rows = [row for row in all_rows if str(row.get("target_eval_has_all_classes")) == "1"]
    return {
        "schema_version": FAMILY_C_PCA64_CC_SCHEMA_VERSION,
        "decision_classification": classification,
        "pass_fail": "PASS" if classification in {"GENERATOR_SOLVED", "ROUTING_BOTTLENECK"} else "FAIL",
        "metrics": {
            "pca64_cc_cvae_selected_center_level_mean_bacc": selected,
            "pca64_cc_cvae_selected_median_bacc": selected_median,
            "pca64_cc_cvae_downstream_oracle_center_level_mean_bacc": oracle,
            "pca64_cc_cvae_downstream_oracle_median_bacc": oracle_median,
            "pca64_cc_cvae_selected_oracle_gap_bacc": gap,
            "pca64_real_reconstruction_upper_center_level_mean_bacc": recon_oracle,
            "delta_vs_pca64_unconditioned_selected_bacc": delta_selected,
            "delta_vs_pca64_unconditioned_oracle_bacc": delta_oracle,
            "delta_vs_pca_gmm_oracle_bacc": delta_pca_gmm_oracle,
            "top1_oracle_hit_rate": top1,
            "oracle_agreement_rate": agreement,
            "stable_meaningful_oracle_gain": int(stable_gain),
            "material_oracle_gap": int(material_gap),
            "high_oracle_selected_disagreement": int(high_disagreement),
            "headline_rows_exclude_single_class_target_eval": 1,
            "n_ok_rows": len(all_rows),
            "n_headline_all_class_rows": len(all_class_rows),
            "n_single_class_target_eval_rows_excluded_from_headline": len(all_rows) - len(all_class_rows),
        },
        "thresholds": {
            "meaningful_oracle_gain": float(config.meaningful_oracle_gain),
            "high_disagreement_threshold": float(config.high_disagreement_threshold),
            "material_oracle_gap": float(config.material_oracle_gap),
        },
        "claim_boundary": (
            "Family C PCA64 class-conditional CVAE is a generator diagnostic. It tests whether "
            "source-label conditioning recovers downstream utility; it does not introduce a new router."
        ),
    }


def read_family_c_pca64_cc_support_units(paths: Sequence[Path]) -> list[SupportSelectionUnit]:
    return [unit for unit in read_support_selection_units(paths, methods=(SUPPORT_NELBO_METHOD,)) if unit.method == SUPPORT_NELBO_METHOD]


def _train_pca64_cc_cvae(
    *,
    model: Any,
    torch: Any,
    device: Any,
    x: Any,
    labels: Sequence[int],
    config: FamilyCPca64ClassConditionalConfig,
    seed: int,
) -> None:
    import numpy as np  # type: ignore

    torch.manual_seed(int(seed))
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray([int(v) for v in labels], dtype=np.int64)
    if x_np.ndim != 2 or x_np.shape[1] != int(config.pca_dim):
        raise ProtocolError(f"Training tensors must be [n,{config.pca_dim}], got {tuple(x_np.shape)}")
    if x_np.shape[0] != y_np.shape[0]:
        raise ProtocolError("Class-conditional CVAE training labels must align with source_train rows.")
    _assert_has_all_source_classes(y_np.tolist(), "training")
    train_idx, val_idx = _stratified_train_val_indices(y_np, val_fraction=config.val_fraction, seed=seed)
    train_x = torch.as_tensor(x_np[train_idx], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_np[train_idx], dtype=torch.long, device=device)
    val_x = torch.as_tensor(x_np[val_idx], dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_np[val_idx], dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = math.inf
    bad = 0
    for epoch in range(int(config.epochs)):
        model.train()
        for batch_x, batch_y in _iter_balanced_condition_batches(
            train_x,
            train_y,
            int(config.batch_size),
            torch=torch,
            seed=int(seed) + int(epoch),
        ):
            optimizer.zero_grad()
            loss = _pca64_cc_cvae_loss(model, batch_x, batch_y, kl_beta=float(config.kl_beta), torch=torch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(_pca64_cc_cvae_loss(model, val_x, val_y, kl_beta=float(config.kl_beta), torch=torch).detach().cpu().item())
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(config.patience):
                break
    model.load_state_dict(best_state)


def _pca64_cc_cvae_loss(model: Any, x: Any, labels: Any, *, kl_beta: float, torch: Any) -> Any:
    m = class_one_hot([int(v) for v in labels.detach().cpu().tolist()], torch=torch, device=x.device)
    recon, mu, logvar = model(x, m=m)
    recon_term = torch.mean((recon - x).pow(2), dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(recon_term + (float(kl_beta) * kl))


def _iter_balanced_condition_batches(x: Any, y: Any, batch_size: int, *, torch: Any, seed: int) -> Any:
    labels = [int(v) for v in y.detach().cpu().tolist()]
    by_class = {label: [idx for idx, value in enumerate(labels) if value == label] for label in PCA64_CC_CLASS_LABELS}
    if any(not by_class[label] for label in PCA64_CC_CLASS_LABELS):
        raise ProtocolError("Class-balanced CVAE batches require both source classes.")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n_batches = max(1, math.ceil(int(len(labels)) / max(1, int(batch_size))))
    per_class = max(1, int(batch_size) // len(PCA64_CC_CLASS_LABELS))
    for _ in range(n_batches):
        idxs: list[int] = []
        for label in PCA64_CC_CLASS_LABELS:
            pool = by_class[label]
            sampled = torch.randint(len(pool), (per_class,), generator=gen, device="cpu").tolist()
            idxs.extend(pool[int(i)] for i in sampled)
        yield x[idxs], y[idxs]


def _score_all_priors(expert: Pca64ClassConditionalExpert, x: Any) -> ClassConditionalNelboScores:
    torch = expert.model._pca64_torch
    device = expert.model._pca64_device
    return ClassConditionalNelboScores(
        uniform=score_label_marginal_nelbo(expert.model, x, class_prior={0: 0.5, 1: 0.5}, torch=torch, device=device, kl_beta=expert.kl_beta),
        source_prior=score_label_marginal_nelbo(expert.model, x, class_prior=expert.source_class_prior, torch=torch, device=device, kl_beta=expert.kl_beta),
        global_source_prior=score_label_marginal_nelbo(expert.model, x, class_prior=expert.global_source_prior, torch=torch, device=device, kl_beta=expert.kl_beta),
    )


def _sample_label_marginal_nelbo_values(
    model: Any,
    x_standardized_pca64: Any,
    *,
    class_prior: Mapping[int, float],
    torch: Any,
    device: Any,
    kl_beta: float,
) -> list[float]:
    x_np = _as_numpy(x_standardized_pca64)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    prior = _normalized_prior(class_prior)
    log_terms = []
    model.eval()
    with torch.no_grad():
        for class_label in PCA64_CC_CLASS_LABELS:
            m = class_one_hot([class_label] * int(x.shape[0]), torch=torch, device=device)
            mu, logvar = model.encode(x, m=m)
            recon = model.decode(mu, m=m)
            recon_term = torch.mean((recon - x).pow(2), dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total = recon_term + (float(kl_beta) * kl)
            log_terms.append(math.log(float(prior[class_label])) - total)
        marginal = -torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)
    return [float(v) for v in marginal.detach().cpu().numpy().tolist()]


def _standardized_reference_pools(*, train_cache: EmbeddingCache, expert: Pca64ClassConditionalExpert, required_labels: Sequence[int]) -> dict[int, Any]:
    pools = build_class_reference_pools(
        train_cache=train_cache,
        candidate_expert=expert.source_center,
        required_labels=required_labels,
    )
    out: dict[int, Any] = {}
    for label, embeddings in pools.items():
        out[int(label)] = None if embeddings is None else transform_pca64(expert.preprocessor, embeddings)
    return out


def _preprocessing_row(prep: Pca64Preprocessor) -> dict[str, object]:
    return {
        "experiment_seed": prep.experiment_seed,
        "source_center": prep.source_center,
        "feature_extractor": prep.feature_extractor,
        "split_id": prep.split_id,
        "pca_dim": prep.pca_dim,
        "standardized": 1,
        "pca_artifact_id": prep.pca_artifact_id,
        "scaler_artifact_id": prep.scaler_artifact_id,
        "preprocessing_artifact_key": prep.artifact_key,
        "n_fit_samples": prep.n_fit_samples,
        "embedding_dim": prep.embedding_dim,
        "pca_explained_variance_ratio_sum": prep.pca_explained_variance_ratio_sum,
        "pca_coord_mean_before_scaling": prep.pca_coord_mean_before_scaling,
        "pca_coord_std_before_scaling": prep.pca_coord_std_before_scaling,
        "scaler_mean": _json_float_list(prep.scaler.mean_),
        "scaler_scale": _json_float_list(prep.scaler.scale_),
        "pca_fit_centers": prep.source_center,
        "target_rows_used_for_fit": 0,
        "available": 1,
        "status": "ok",
    }


def _nelbo_row(
    *,
    unit: SupportSelectionUnit,
    heldout_center: str,
    source_center: str,
    split_role: str,
    scores: ClassConditionalNelboScores,
    source_mean: float,
    source_std: float,
    normalized: float,
    normalized_available: int,
    target_eval_scores: ClassConditionalNelboScores,
) -> dict[str, object]:
    return {
        "experiment_seed": int(unit.experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(unit.support_size),
        "support_seed": int(unit.support_seed),
        "support_eval_split_id": unit.support_eval_split_id,
        "source_center": source_center,
        "split_role": split_role,
        "n_samples": int(scores.uniform.n_samples),
        "support_recon_term": float(scores.uniform.recon),
        "support_kl_term": float(scores.uniform.kl),
        "support_total_nelbo": float(scores.uniform.total),
        "support_nelbo_source_prior": float(scores.source_prior.total),
        "support_nelbo_global_source_prior": float(scores.global_source_prior.total),
        "source_train_nelbo_mean": float(source_mean),
        "source_train_nelbo_std": float(source_std),
        "source_normalized_nelbo": normalized,
        "normalized_available": normalized_available,
        "target_eval_nelbo_unlabeled_diagnostic_only": float(target_eval_scores.uniform.total),
        "target_eval_nelbo_source_prior_diagnostic_only": float(target_eval_scores.source_prior.total),
        "target_eval_nelbo_global_source_prior_diagnostic_only": float(target_eval_scores.global_source_prior.total),
        "nelbo_reduction": "mean_over_samples",
        "recon_reduction": "mean_per_sample_per_dim",
        "primary_class_prior": "uniform_class_prior",
    }


def _row_base(
    *,
    schema_version: str,
    experiment_seed: int,
    heldout_center: str,
    unit: SupportSelectionUnit,
    candidate_expert: str,
    generation_mode: str,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    row_type: str,
    n_target_eval: int,
    target_eval_pool_id: str,
    target_status: Mapping[str, object],
    support_scores: ClassConditionalNelboScores,
    target_density_scores: ClassConditionalNelboScores,
    source_normalized: float,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_size": int(unit.support_size),
        "support_seed": int(unit.support_seed),
        "support_eval_split_id": str(unit.support_eval_split_id),
        "candidate_expert": str(candidate_expert),
        "generation_mode": str(generation_mode),
        "budget_per_class": int(budget_per_class),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "row_type": row_type,
        "n_target_eval": int(n_target_eval),
        "target_eval_pool_id": str(target_eval_pool_id),
        "target_eval_label_counts_json": target_status["target_eval_label_counts_json"],
        "target_eval_has_all_classes": int(target_status["target_eval_has_all_classes"]),
        "support_nelbo_raw": float(support_scores.uniform.total),
        "support_nelbo_source_prior": float(support_scores.source_prior.total),
        "support_nelbo_global_source_prior": float(support_scores.global_source_prior.total),
        "support_nelbo_source_normalized": source_normalized,
        "target_eval_nelbo_unlabeled": float(target_density_scores.uniform.total),
        "target_eval_nelbo_source_prior": float(target_density_scores.source_prior.total),
        "target_eval_nelbo_global_source_prior": float(target_density_scores.global_source_prior.total),
    }


def _failed_row(base: Mapping[str, object], exc: Exception) -> dict[str, object]:
    return {
        **base,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "auroc": math.nan,
        "auprc": math.nan,
        "n_train": 0,
        "available": 0,
        "status": _failure_status(exc),
        "error_message": str(exc),
    }


def _audit_row(*, row: Mapping[str, object], expert: Pca64ClassConditionalExpert, target_expert_excluded: int) -> dict[str, object]:
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "support_size": row.get("support_size", ""),
        "support_seed": row.get("support_seed", ""),
        "support_eval_split_id": row.get("support_eval_split_id", ""),
        "candidate_expert": row.get("candidate_expert", ""),
        "generation_mode": row.get("generation_mode", ""),
        "generation_seed": row.get("generation_seed", ""),
        "classifier_seed": row.get("classifier_seed", ""),
        "lineage_key": _lineage_key(row=row, expert=expert),
        "preprocessing_artifact_key": expert.preprocessor.artifact_key,
        "pca_artifact_id": expert.preprocessor.pca_artifact_id,
        "scaler_artifact_id": expert.preprocessor.scaler_artifact_id,
        "cvae_input_dim": expert.input_dim,
        "class_condition_dim": PCA64_CC_CONDITION_DIM,
        "generated_embedding_dim": expert.preprocessor.embedding_dim,
        "target_expert_excluded": int(target_expert_excluded),
        "support_eval_disjoint": 1,
        "pca_fit_split": "source_train",
        "scaler_fit_split": "source_train_pca_coordinates",
        "cvae_fit_split": "source_train",
        "target_center_excluded_from_pca": int(target_expert_excluded),
        "target_rows_used_for_pca": 0,
        "target_rows_used_for_scaler": 0,
        "source_labels_used_for_cvae_training": 1,
        "target_labels_used_for_cvae_training": 0,
        "support_labels_used_for_nelbo": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_labels_used_for_training": 0,
        "target_eval_labels_used_for_final_metric_only": 1,
        "classifier_scaler_fit": "synthetic_train_only" if row.get("generation_mode") != PCA64_CC_REAL_RECONSTRUCTION_MODE else "reconstructed_source_train_only",
        "checkpoint_feature_space": PCA64_CC_FEATURE_SPACE,
        "checkpoint_conditioning": "class_label_one_hot",
        "available": row.get("available", 0),
        "status": row.get("status", ""),
    }


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "matrix": root / "family_c_pca64_cc_all_expert_downstream_matrix.csv",
        "preprocessing": root / "family_c_pca64_cc_preprocessing_manifest.csv",
        "checkpoints": root / "family_c_pca64_cc_expert_checkpoint_manifest.csv",
        "nelbo": root / "family_c_pca64_cc_nelbo_diagnostics.csv",
        "generation": root / "family_c_pca64_cc_generation_manifest.csv",
        "protocol_audit": root / "family_c_pca64_cc_protocol_audit.csv",
        "alignment": root / "family_c_pca64_cc_routing_alignment.csv",
        "baseline": root / "family_c_pca64_cc_baseline_comparison.csv",
        "decision_summary": root / "family_c_pca64_cc_decision_summary.json",
    }


def _pca64_torch_imports(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    cvae_testing_root = repo_root / "cvae_testing"
    if str(cvae_testing_root) not in sys.path:
        sys.path.insert(0, str(cvae_testing_root))
    import torch  # type: ignore
    from src.models.cvae_expert import CVAEExpert  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint, wrap_model_state_dict  # type: ignore

    return torch, CVAEExpert, load_model_checkpoint, wrap_model_state_dict


def _read_completed_matrix_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    return {_matrix_key(row) for row in _read_dict_rows(path)}


def _matrix_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["heldout_center"]),
        int(row["support_size"]),
        int(row["support_seed"]),
        str(row["candidate_expert"]),
        str(row["generation_mode"]),
        int(row["budget_per_class"]),
        int(row["generation_seed"]),
        int(row["classifier_seed"]),
        str(row["row_type"]),
    )


def _preprocessing_manifest_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (_int_value(row.get("experiment_seed")), str(row.get("source_center", "")), str(row.get("preprocessing_artifact_key", "")))


def _checkpoint_manifest_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (_int_value(row.get("experiment_seed")), str(row.get("source_center", "")), str(row.get("checkpoint_path", "")))


def _nelbo_manifest_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int_value(row.get("experiment_seed")),
        str(row.get("heldout_center", "")),
        _int_value(row.get("support_size")),
        _int_value(row.get("support_seed")),
        str(row.get("support_eval_split_id", "")),
        str(row.get("source_center", "")),
        str(row.get("split_role", "")),
    )


def _generation_manifest_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int_value(row.get("experiment_seed")),
        str(row.get("heldout_center", "")),
        _int_value(row.get("support_size")),
        _int_value(row.get("support_seed")),
        str(row.get("candidate_expert", "")),
        str(row.get("generation_mode", "")),
        _int_value(row.get("generation_seed")),
        _int_value(row.get("class_label")),
    )


def _protocol_audit_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int_value(row.get("experiment_seed")),
        str(row.get("heldout_center", "")),
        _int_value(row.get("support_size")),
        _int_value(row.get("support_seed")),
        str(row.get("support_eval_split_id", "")),
        str(row.get("candidate_expert", "")),
        str(row.get("generation_mode", "")),
        _int_value(row.get("generation_seed")),
        _int_value(row.get("classifier_seed")),
    )


def _alignment_context_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["heldout_center"]),
        int(row["support_size"]),
        int(row["support_seed"]),
        str(row["support_eval_split_id"]),
        int(row["generation_seed"]),
        int(row["classifier_seed"]),
    )


def _primary_support_units(units: Sequence[SupportSelectionUnit], *, limits: FamilyCPca64BuildLimits) -> tuple[SupportSelectionUnit, ...]:
    out = [
        unit
        for unit in units
        if unit.method == SUPPORT_NELBO_METHOD
        and (limits.experiment_seeds is None or int(unit.experiment_seed) in set(int(v) for v in limits.experiment_seeds))
        and (limits.heldout_centers is None or str(unit.heldout_center) in set(str(v) for v in limits.heldout_centers))
        and (limits.support_sizes is None or int(unit.support_size) in set(int(v) for v in limits.support_sizes))
        and (limits.support_seeds is None or int(unit.support_seed) in set(int(v) for v in limits.support_seeds))
    ]
    if not out:
        raise ProtocolError("No primary support-NELBO units remain after limits.")
    return tuple(out)


def _units_by_seed(units: Sequence[SupportSelectionUnit]) -> dict[int, tuple[SupportSelectionUnit, ...]]:
    grouped: dict[int, list[SupportSelectionUnit]] = {}
    for unit in units:
        grouped.setdefault(int(unit.experiment_seed), []).append(unit)
    return {key: tuple(value) for key, value in grouped.items()}


def _limit_artifacts(artifacts: Sequence[SupportRunArtifacts], seeds: Sequence[int] | None) -> tuple[SupportRunArtifacts, ...]:
    if seeds is None:
        return tuple(artifacts)
    allowed = {int(v) for v in seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.experiment_seed) in allowed)


def _append_dict_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _append_unique_dict_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]], key_fn: Any) -> None:
    if not rows:
        return
    existing = {key_fn(row) for row in _read_dict_rows(path)} if path.exists() and path.stat().st_size > 0 else set()
    unique: list[Mapping[str, object]] = []
    for row in rows:
        key = key_fn(row)
        if key in existing:
            continue
        existing.add(key)
        unique.append(row)
    _append_dict_rows(path, columns, unique)


def _write_dict_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_dict_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _oracle_baseline_row(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    row_type: str,
    config: FamilyCPca64ClassConditionalConfig,
    oracle_value: float,
) -> dict[str, object]:
    return {
        "method": method,
        "row_type": row_type,
        "center_level_mean_bacc": oracle_value,
        "center_level_median_bacc": _median(_float(row.get("bacc")) for row in rows),
        "center_level_mean_macro_f1": _oracle_center_mean(rows, metric="macro_f1"),
        "center_level_mean_oracle_gap_bacc": 0.0,
        "top1_oracle_hit_rate": math.nan,
        "oracle_agreement_rate": math.nan,
        "spearman_neg_nelbo_vs_bacc": math.nan,
        "delta_vs_pca64_unconditioned_selected_bacc": math.nan,
        "delta_vs_pca64_unconditioned_oracle_bacc": oracle_value - float(config.pca64_unconditioned_oracle_bacc),
        "delta_vs_pca_gmm_oracle_bacc": oracle_value - float(config.pca_gmm_oracle_bacc),
        "available": int(bool(rows)),
    }


def _oracle_center_mean(rows: Sequence[Mapping[str, object]], metric: str = "bacc") -> float:
    by_context: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        by_context.setdefault(_alignment_context_key(row), []).append(row)
    winners = [max(group, key=lambda row: (_float(row.get(metric)), _reverse_sort(str(row.get("candidate_expert"))))) for group in by_context.values()]
    return _center_level_mean(winners, metric)


def _center_level_mean(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    by_center: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if math.isnan(value):
            continue
        by_center.setdefault(str(row.get("heldout_center")), []).append(value)
    return _mean(_mean(values) for values in by_center.values())


def _generation_global_diagnostics(*, generated: Any, generated_labels: Sequence[int], source: Any, source_labels: Sequence[int], threshold: float) -> dict[str, object]:
    import numpy as np  # type: ignore

    gen_centroid = _class_centroid_distance(generated, generated_labels)
    source_centroid = _class_centroid_distance(source, source_labels)
    ratio = gen_centroid / max(source_centroid, 1e-12)
    overlap = gen_centroid / max(_mean_class_std(generated, generated_labels), 1e-12)
    return {
        "generated_class_centroid_distance": gen_centroid,
        "source_class_centroid_distance": source_centroid,
        "generated_class_centroid_ratio": ratio,
        "generated_class_overlap_score": overlap,
        "low_generated_class_centroid_ratio": int(ratio < float(threshold)),
        "generated_class_overlap_too_high": int(overlap < float(threshold)),
    }


def _class_centroid_distance(x: Any, labels: Sequence[int]) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    y = np.asarray([int(v) for v in labels], dtype=int)
    if not all(np.any(y == label) for label in PCA64_CC_CLASS_LABELS):
        return math.nan
    means = [arr[y == label].mean(axis=0) for label in PCA64_CC_CLASS_LABELS]
    return float(np.linalg.norm(means[0] - means[1]))


def _mean_class_std(x: Any, labels: Sequence[int]) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    y = np.asarray([int(v) for v in labels], dtype=int)
    vals = [float(np.mean(np.std(arr[y == label], axis=0))) for label in PCA64_CC_CLASS_LABELS if np.any(y == label)]
    return _mean(vals)


def _stratified_train_val_indices(labels: Any, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    val: list[int] = []
    labels_arr = np.asarray(labels, dtype=int)
    for class_label in PCA64_CC_CLASS_LABELS:
        idxs = np.where(labels_arr == int(class_label))[0]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * float(val_fraction)))) if len(idxs) > 1 else 0
        val.extend(int(v) for v in idxs[:n_val])
        train.extend(int(v) for v in idxs[n_val:])
    if not train:
        train = val[:]
    if not val:
        val = train[:]
    return train, val


def _class_prior_for_indices(indices: Any, metadata: Sequence[Mapping[str, object]]) -> dict[int, float]:
    counts = {label: 0 for label in PCA64_CC_CLASS_LABELS}
    total = 0
    for idx in indices:
        label = _label(metadata[int(idx)])
        if label in counts:
            counts[int(label)] += 1
            total += 1
    if total <= 0 or any(counts[label] <= 0 for label in PCA64_CC_CLASS_LABELS):
        return {0: 0.5, 1: 0.5}
    return {label: counts[label] / total for label in PCA64_CC_CLASS_LABELS}


def _normalized_prior(prior: Mapping[int, float]) -> dict[int, float]:
    vals = {label: max(float(prior.get(label, 0.0)), 0.0) for label in PCA64_CC_CLASS_LABELS}
    total = sum(vals.values())
    if total <= 0.0:
        return {0: 0.5, 1: 0.5}
    return {label: vals[label] / total for label in PCA64_CC_CLASS_LABELS}


def _center_indices(cache: EmbeddingCache, center: str) -> tuple[int, ...]:
    return tuple(idx for idx, row in enumerate(cache.metadata) if str(_domain(row)) == str(center))


def _slice_embeddings(embeddings: Any, indices: Sequence[int]) -> Any:
    if hasattr(embeddings, "__getitem__"):
        return embeddings[list(indices)]
    raise ProtocolError("Embedding cache object is not indexable.")


def _assert_has_all_source_classes(labels: Sequence[int], source_center: str) -> None:
    observed = {int(v) for v in labels}
    missing = [label for label in PCA64_CC_CLASS_LABELS if label not in observed]
    if missing:
        raise ProtocolError(f"Source center {source_center} missing class labels for class-conditional CVAE: {missing}")


def _target_label_status(target_labels: Sequence[int]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for label in target_labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return {
        "target_eval_label_counts_json": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        "target_eval_has_all_classes": int(all(counts.get(str(label), 0) > 0 for label in PCA64_CC_CLASS_LABELS)),
    }


def _lineage_key(*, row: Mapping[str, object], expert: Pca64ClassConditionalExpert) -> str:
    return _hash_json(
        {
            "experiment_seed": int(row.get("experiment_seed", expert.experiment_seed)),
            "heldout_center": str(row.get("heldout_center", "")),
            "source_center": str(expert.source_center),
            "support_size": int(row.get("support_size", 0)),
            "support_seed": int(row.get("support_seed", 0)),
            "support_eval_split_id": str(row.get("support_eval_split_id", "")),
            "generation_mode": str(row.get("generation_mode", "")),
            "generation_seed": int(row.get("generation_seed", 0)),
            "classifier_seed": int(row.get("classifier_seed", 0)),
            "feature_space": PCA64_CC_FEATURE_SPACE,
            "conditioning": "class_label_one_hot",
            "pca_artifact_id": expert.preprocessor.pca_artifact_id,
            "scaler_artifact_id": expert.preprocessor.scaler_artifact_id,
        }
    )


def _mean(values: Any) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def _median(values: Any) -> float:
    vals = sorted(float(v) for v in values if not math.isnan(float(v)))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return math.nan
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return math.nan
    rx = _ranks([x for x, _ in pairs])
    ry = _ranks([y for _, y in pairs])
    mx, my = _mean(rx), _mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den_x * den_y) if den_x > 0 and den_y > 0 else math.nan


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), idx))
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _int_value(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _reverse_sort(text: str) -> tuple[int, str]:
    try:
        return (-int(text), text)
    except Exception:
        return (0, text)


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _json_float_list(values: Any) -> str:
    return json.dumps([float(v) for v in list(values)], separators=(",", ":"))


def _json_prior(prior: Mapping[int, float]) -> str:
    return json.dumps({str(k): float(v) for k, v in sorted(prior.items())}, sort_keys=True, separators=(",", ":"))


def _condition_list(class_label: int) -> list[int]:
    return [1, 0] if int(class_label) == 0 else [0, 1]


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:8], 16)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"inherit", "inherit_from_support_run", "null", "none"}:
        return None
    return int(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value
