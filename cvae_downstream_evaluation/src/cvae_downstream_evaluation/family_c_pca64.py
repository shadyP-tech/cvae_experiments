"""PCA-64 standardized CVAE downstream diagnostic.

This module is intentionally isolated from the locked v1 downstream pipeline.
It trains source-domain CVAE experts in source-train-only standardized PCA
space, inverse-transforms generated samples back to the original DINO feature
space, and evaluates the unchanged locked downstream classifier.
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
from .matrix import (
    EmbeddingCache,
    SupportRunArtifacts,
    _domain,
    _failure_status,
    _label,
    _load_embedding_cache,
    _make_support_eval_split,
    _read_samples_manifest,
    _read_support_run_dimensions,
    _records_for_split,
    _resolve_torch_device,
    _sample_id,
    _torch_generator,
    build_class_reference_pools,
    discover_support_run_artifacts,
)
from .protocol import ProtocolError
from .routing import SupportSelectionUnit, read_support_selection_units
from .schemas import CLASSIFIER_SEEDS, EXPERIMENT_SEEDS, GENERATION_SEEDS, SUPPORT_NELBO_METHOD, SUPPORT_SEEDS, SUPPORT_SIZES
from .splits import assert_disjoint_ids


FAMILY_C_PCA64_NAME = "family_c_pca64_standardized_cvae_downstream_v1"
FAMILY_C_PCA64_SCHEMA_VERSION = "family_c_pca64_standardized_cvae_downstream_v1"
PCA64_CVAE_MODE = "family_c_pca64_standardized_cvae_reference_posterior_resampling"
PCA64_REAL_RECONSTRUCTION_MODE = "pca64_real_reconstruction_upper"
PCA64_RAW_SELECTOR = "family_c_pca64_raw_support_nelbo"
PCA64_NORMALIZED_SELECTOR = "family_c_pca64_source_normalized_support_nelbo"
PCA64_SINGLE_EXPERT_ROW_TYPE = "single_expert_pca64_cvae"
PCA64_REAL_UPPER_ROW_TYPE = "real_reconstruction_upper"
PCA64_CHECKPOINT_FEATURE_SPACE = "standardized_pca64"
PCA64_PCA_DIM = 64
PCA64_BUDGET_PER_CLASS = 128
PCA64_NORMALIZED_EPS = 1e-8
PCA64_SMALL_STD_THRESHOLD = 1e-8


PCA64_MATRIX_COLUMNS = (
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
    "support_nelbo_source_normalized",
    "target_eval_nelbo_unlabeled",
    "available",
    "status",
    "error_message",
)


PCA64_ALIGNMENT_COLUMNS = (
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


PCA64_PREPROCESSING_COLUMNS = (
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


PCA64_CHECKPOINT_COLUMNS = (
    "experiment_seed",
    "source_center",
    "checkpoint_path",
    "input_dim",
    "hidden_dim",
    "latent_dim",
    "feature_space",
    "pca_artifact_id",
    "scaler_artifact_id",
    "source_train_nelbo_mean",
    "source_train_nelbo_std",
    "kl_beta",
    "decoder_output_variance_assumption",
    "available",
    "status",
)


PCA64_NELBO_COLUMNS = (
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
    "source_train_nelbo_mean",
    "source_train_nelbo_std",
    "source_normalized_nelbo",
    "normalized_available",
    "target_eval_nelbo_unlabeled_diagnostic_only",
    "nelbo_reduction",
    "recon_reduction",
    "uniform_class_prior",
)


PCA64_GENERATION_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "candidate_expert",
    "generation_mode",
    "generation_seed",
    "class_label",
    "num_generated_per_class_actual",
    "class_generation_failures",
    "generated_pca_std_mean",
    "generated_dino_std_mean",
    "generated_shape",
    "inverse_scaler_used",
    "inverse_pca_used",
    "classifier_space",
)


PCA64_PROTOCOL_AUDIT_COLUMNS = (
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
    "generated_embedding_dim",
    "target_expert_excluded",
    "support_eval_disjoint",
    "pca_fit_split",
    "scaler_fit_split",
    "target_center_excluded_from_pca",
    "target_rows_used_for_pca",
    "target_rows_used_for_scaler",
    "support_labels_used_for_nelbo",
    "target_eval_labels_used_for_selection",
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_final_metric_only",
    "classifier_scaler_fit",
    "checkpoint_feature_space",
    "available",
    "status",
)


PCA64_BASELINE_COLUMNS = (
    "method",
    "row_type",
    "center_level_mean_bacc",
    "center_level_mean_macro_f1",
    "center_level_mean_oracle_gap_bacc",
    "top1_oracle_hit_rate",
    "spearman_neg_nelbo_vs_bacc",
    "available",
)


@dataclass(frozen=True)
class FamilyCPca64Config:
    experiment_name: str = FAMILY_C_PCA64_NAME
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
    artifacts_root: str = "cvae_downstream_evaluation/artifacts/family_c_pca64_standardized_cvae_downstream_v1"
    pca_dim: int = PCA64_PCA_DIM
    standardized: bool = True
    budget_per_class: int = PCA64_BUDGET_PER_CLASS
    hidden_dim: int | None = None
    latent_dim: int | None = None
    lr: float = 1e-3
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    val_fraction: float = 0.2
    kl_beta: float = 1.0
    normalized_nelbo_eps: float = PCA64_NORMALIZED_EPS
    small_std_threshold: float = PCA64_SMALL_STD_THRESHOLD
    smoke_oracle_drop_tolerance: float = 0.05
    current_cvae_oracle_bacc: float = 0.7547755213552805


@dataclass(frozen=True)
class FamilyCPca64BuildLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Pca64Preprocessor:
    experiment_seed: int
    source_center: str
    feature_extractor: str
    split_id: str
    pca_dim: int
    pca: Any
    scaler: Any
    pca_artifact_id: str
    scaler_artifact_id: str
    artifact_key: str
    pca_explained_variance_ratio_sum: float
    pca_coord_mean_before_scaling: float
    pca_coord_std_before_scaling: float
    n_fit_samples: int
    embedding_dim: int


@dataclass(frozen=True)
class NelboScore:
    total: float
    recon: float
    kl: float
    n_samples: int


@dataclass(frozen=True)
class Pca64CvaeExpert:
    experiment_seed: int
    source_center: str
    model: Any
    preprocessor: Pca64Preprocessor
    source_train_nelbo_mean: float
    source_train_nelbo_std: float
    checkpoint_path: Path
    input_dim: int
    hidden_dim: int
    latent_dim: int
    kl_beta: float


@dataclass(frozen=True)
class Pca64TargetSplit:
    support_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    target_eval_pool_id: str


def default_family_c_pca64_config() -> FamilyCPca64Config:
    return FamilyCPca64Config()


def load_family_c_pca64_config(path: Path) -> FamilyCPca64Config:
    text = Path(path).read_text(encoding="utf-8")
    assert_family_c_pca64_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_family_c_pca64_config()
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
    return FamilyCPca64Config(
        experiment_name=str(experiment.get("name", FAMILY_C_PCA64_NAME)),
        candidate_domains=tuple(str(v) for v in camelyon.get("candidate_domains", ("0", "1", "2", "3", "4"))),
        experiment_seeds=tuple(int(v) for v in camelyon.get("experiment_seeds", EXPERIMENT_SEEDS)),
        support_sizes=tuple(int(v) for v in camelyon.get("support_sizes", SUPPORT_SIZES)),
        support_seeds=tuple(int(v) for v in camelyon.get("support_seeds", SUPPORT_SEEDS)),
        generation_seeds=tuple(int(v) for v in camelyon.get("generation_seeds", GENERATION_SEEDS)),
        classifier_seeds=tuple(int(v) for v in camelyon.get("classifier_seeds", CLASSIFIER_SEEDS)),
        support_selection_glob=str(support_inputs.get("selection_glob", default_family_c_pca64_config().support_selection_glob)),
        artifacts_root=str(artifacts.get("root", default_family_c_pca64_config().artifacts_root)),
        pca_dim=int(preprocessing.get("pca_dim", PCA64_PCA_DIM)),
        standardized=bool(preprocessing.get("standardized", True)),
        budget_per_class=int(generation.get("budget_per_class", PCA64_BUDGET_PER_CLASS)),
        hidden_dim=_optional_int(training.get("hidden_dim")),
        latent_dim=_optional_int(training.get("latent_dim")),
        lr=float(training.get("lr", 1e-3)),
        epochs=int(training.get("epochs", 80)),
        patience=int(training.get("patience", 10)),
        batch_size=int(training.get("batch_size", 128)),
        val_fraction=float(training.get("val_fraction", 0.2)),
        kl_beta=float(training.get("kl_beta", 1.0)),
        normalized_nelbo_eps=float(training.get("normalized_nelbo_eps", PCA64_NORMALIZED_EPS)),
        small_std_threshold=float(decision.get("small_std_threshold", PCA64_SMALL_STD_THRESHOLD)),
        smoke_oracle_drop_tolerance=float(decision.get("smoke_oracle_drop_tolerance", 0.05)),
        current_cvae_oracle_bacc=float(decision.get("current_cvae_oracle_bacc", 0.7547755213552805)),
    )


def assert_family_c_pca64_config_text(text: str) -> None:
    required = (
        f"name: {FAMILY_C_PCA64_NAME}",
        "feature_space: standardized_pca64",
        "pca_dim: 64",
        "standardized: true",
        "uniform_class_prior",
        "support_labels_for_primary_routing: forbidden",
        "target_eval_labels_used_for_selection: forbidden",
        "budget_per_class: 128",
        "family_c_pca64_all_expert_downstream_matrix.csv",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"Family C PCA64 config missing required fields: {missing}")


def fit_source_train_pca64_preprocessor(
    *,
    experiment_seed: int,
    source_center: str,
    train_cache: EmbeddingCache,
    feature_extractor: str,
    split_id: str,
    pca_dim: int = PCA64_PCA_DIM,
    target_center: str | None = None,
) -> Pca64Preprocessor:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    idxs = [idx for idx, row in enumerate(train_cache.metadata) if str(_domain(row)) == str(source_center)]
    if not idxs:
        raise ProtocolError(f"No source_train rows for center {source_center}")
    fit_centers = {str(_domain(train_cache.metadata[idx])) for idx in idxs}
    if fit_centers != {str(source_center)}:
        raise ProtocolError(f"PCA fit centers {fit_centers} do not match source center {source_center}")
    if target_center is not None and str(target_center) in fit_centers:
        raise ProtocolError(f"Target center {target_center} leaked into PCA fit for source {source_center}")
    x = _as_numpy(train_cache.embeddings[idxs])
    if int(x.ndim) != 2:
        raise ProtocolError("PCA fit embeddings must be a 2D array.")
    n_fit, embedding_dim = int(x.shape[0]), int(x.shape[1])
    if n_fit < int(pca_dim) + 1:
        raise ProtocolError(f"PCA64 requires at least {int(pca_dim) + 1} source_train rows, got {n_fit}")
    if embedding_dim < int(pca_dim):
        raise ProtocolError(f"PCA64 requires embedding_dim >= {pca_dim}, got {embedding_dim}")
    pca = PCA(n_components=int(pca_dim), whiten=False, svd_solver="randomized", random_state=_stable_seed(experiment_seed, source_center, "pca"))
    x_pca = pca.fit_transform(x)
    scaler = StandardScaler()
    scaler.fit(x_pca)
    pca_id = _hash_json(
        {
            "experiment_seed": int(experiment_seed),
            "source_center": str(source_center),
            "feature_extractor": str(feature_extractor),
            "split_id": str(split_id),
            "pca_dim": int(pca_dim),
            "standardized": True,
            "kind": "pca",
        }
    )
    scaler_id = _hash_json(
        {
            "experiment_seed": int(experiment_seed),
            "source_center": str(source_center),
            "feature_extractor": str(feature_extractor),
            "split_id": str(split_id),
            "pca_dim": int(pca_dim),
            "standardized": True,
            "kind": "scaler",
        }
    )
    key = preprocessing_artifact_key(
        experiment_seed=experiment_seed,
        source_center=source_center,
        feature_extractor=feature_extractor,
        split_id=split_id,
        pca_dim=pca_dim,
        standardized=True,
    )
    return Pca64Preprocessor(
        experiment_seed=int(experiment_seed),
        source_center=str(source_center),
        feature_extractor=str(feature_extractor),
        split_id=str(split_id),
        pca_dim=int(pca_dim),
        pca=pca,
        scaler=scaler,
        pca_artifact_id=pca_id,
        scaler_artifact_id=scaler_id,
        artifact_key=key,
        pca_explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
        pca_coord_mean_before_scaling=float(np.mean(x_pca)),
        pca_coord_std_before_scaling=float(np.std(x_pca)),
        n_fit_samples=n_fit,
        embedding_dim=embedding_dim,
    )


def preprocessing_artifact_key(
    *,
    experiment_seed: int,
    source_center: str,
    feature_extractor: str,
    split_id: str,
    pca_dim: int,
    standardized: bool,
) -> str:
    return _hash_json(
        {
            "experiment_seed": int(experiment_seed),
            "source_center": str(source_center),
            "feature_extractor": str(feature_extractor),
            "split_id": str(split_id),
            "pca_dim": int(pca_dim),
            "standardized": bool(standardized),
        }
    )


def transform_pca64(preprocessor: Pca64Preprocessor, embeddings: Any) -> Any:
    x = _as_numpy(embeddings)
    x_pca = preprocessor.pca.transform(x)
    return preprocessor.scaler.transform(x_pca)


def inverse_transform_pca64(preprocessor: Pca64Preprocessor, standardized_pca: Any) -> Any:
    x_pca = preprocessor.scaler.inverse_transform(_as_numpy(standardized_pca))
    return preprocessor.pca.inverse_transform(x_pca)


def assert_pca64_checkpoint_metadata(metadata: Mapping[str, object], *, expected: Mapping[str, object]) -> None:
    required = {
        "input_dim": 64,
        "feature_space": PCA64_CHECKPOINT_FEATURE_SPACE,
        "pca_artifact_id": expected.get("pca_artifact_id"),
        "scaler_artifact_id": expected.get("scaler_artifact_id"),
        "source_center": str(expected.get("source_center")),
        "experiment_seed": int(expected.get("experiment_seed", 0)),
    }
    for key, value in required.items():
        actual = metadata.get(key)
        if key in {"input_dim", "experiment_seed"}:
            actual = int(actual) if str(actual).strip() else actual
        else:
            actual = str(actual)
            value = str(value)
        if actual != value:
            raise ProtocolError(
                f"Incompatible PCA64 CVAE checkpoint metadata for {key}: got {actual!r}, expected {value!r}"
            )


def score_unlabeled_nelbo(
    model: Any,
    x_standardized_pca64: Any,
    *,
    torch: Any,
    device: Any,
    kl_beta: float = 1.0,
) -> NelboScore:
    x_np = _as_numpy(x_standardized_pca64)
    if int(x_np.ndim) != 2 or int(x_np.shape[1]) != PCA64_PCA_DIM:
        raise ProtocolError(f"Expected standardized PCA64 input shape [n,64], got {tuple(x_np.shape)}")
    if int(x_np.shape[0]) == 0:
        return NelboScore(total=math.nan, recon=math.nan, kl=math.nan, n_samples=0)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x)
        recon = model.decode(mu)
        recon_term = torch.mean((recon - x).pow(2), dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        total = recon_term + (float(kl_beta) * kl)
    return NelboScore(
        total=float(torch.mean(total).detach().cpu().item()),
        recon=float(torch.mean(recon_term).detach().cpu().item()),
        kl=float(torch.mean(kl).detach().cpu().item()),
        n_samples=int(x_np.shape[0]),
    )


def build_family_c_pca64_all_expert_downstream_matrix(
    *,
    config: FamilyCPca64Config,
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
    artifacts = discover_support_run_artifacts(config=config, repo_root=repo_root)
    artifacts = _limit_artifacts(artifacts, limits.experiment_seeds)
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
        bank, prep_rows, checkpoint_rows = fit_or_load_pca64_cvae_bank(
            config=config,
            artifact=artifact,
            train_cache=train_cache,
            dimensions=dimensions,
            feature_extractor=feature_extractor,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            device=device,
        )
        _append_unique_dict_rows(paths["preprocessing"], PCA64_PREPROCESSING_COLUMNS, prep_rows, _preprocessing_manifest_key)
        _append_unique_dict_rows(paths["checkpoints"], PCA64_CHECKPOINT_COLUMNS, checkpoint_rows, _checkpoint_manifest_key)

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
            support_embeddings_by_expert: dict[str, Any] = {}
            target_embeddings_by_expert: dict[str, Any] = {}
            support_scores: dict[str, NelboScore] = {}
            target_density_scores: dict[str, NelboScore] = {}
            nelbo_rows: list[dict[str, object]] = []
            for expert in candidates:
                pca_expert = bank[str(expert)]
                support_x = transform_pca64(
                    pca_expert.preprocessor,
                    _slice_embeddings(test_cache.embeddings, target_split.support_indices),
                )
                target_x = transform_pca64(
                    pca_expert.preprocessor,
                    _slice_embeddings(test_cache.embeddings, target_split.eval_indices),
                )
                support_embeddings_by_expert[str(expert)] = support_x
                target_embeddings_by_expert[str(expert)] = target_x
                support_score = score_unlabeled_nelbo(
                    pca_expert.model,
                    support_x,
                    torch=pca_expert.model._pca64_torch,
                    device=pca_expert.model._pca64_device,
                    kl_beta=pca_expert.kl_beta,
                )
                target_score = score_unlabeled_nelbo(
                    pca_expert.model,
                    target_x,
                    torch=pca_expert.model._pca64_torch,
                    device=pca_expert.model._pca64_device,
                    kl_beta=pca_expert.kl_beta,
                )
                support_scores[str(expert)] = support_score
                target_density_scores[str(expert)] = target_score
                normalized, norm_available = source_normalized_nelbo(
                    support_score.total,
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
                        score=support_score,
                        source_mean=pca_expert.source_train_nelbo_mean,
                        source_std=pca_expert.source_train_nelbo_std,
                        normalized=normalized,
                        normalized_available=norm_available,
                        target_eval_score=target_score,
                    )
                )
            _append_unique_dict_rows(paths["nelbo"], PCA64_NELBO_COLUMNS, nelbo_rows, _nelbo_manifest_key)

            for expert in candidates:
                reference_pools = _standardized_reference_pools(
                    train_cache=train_cache,
                    expert=bank[str(expert)],
                    required_labels=(0, 1),
                )
                for generation_seed in selected_generation_seeds:
                    for classifier_seed in selected_classifier_seeds:
                        row, generation_rows, audit = score_pca64_cvae_candidate(
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
                            support_score=support_scores[str(expert)],
                            target_density_score=target_density_scores[str(expert)],
                        )
                        if not (resume and _matrix_key(row) in completed):
                            _append_dict_rows(paths["matrix"], PCA64_MATRIX_COLUMNS, [row])
                            _append_unique_dict_rows(paths["generation"], PCA64_GENERATION_COLUMNS, generation_rows, _generation_manifest_key)
                            _append_unique_dict_rows(paths["protocol_audit"], PCA64_PROTOCOL_AUDIT_COLUMNS, [audit], _protocol_audit_key)
                            completed.add(_matrix_key(row))

                        recon_row, recon_audit = score_pca64_real_reconstruction_candidate(
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
                            support_score=support_scores[str(expert)],
                            target_density_score=target_density_scores[str(expert)],
                        )
                        if not (resume and _matrix_key(recon_row) in completed):
                            _append_dict_rows(paths["matrix"], PCA64_MATRIX_COLUMNS, [recon_row])
                            _append_unique_dict_rows(paths["protocol_audit"], PCA64_PROTOCOL_AUDIT_COLUMNS, [recon_audit], _protocol_audit_key)
                            completed.add(_matrix_key(recon_row))

    return paths


def fit_or_load_pca64_cvae_bank(
    *,
    config: FamilyCPca64Config,
    artifact: SupportRunArtifacts,
    train_cache: EmbeddingCache,
    dimensions: Mapping[str, object],
    feature_extractor: str,
    repo_root: Path,
    artifacts_root: Path,
    device: str,
) -> tuple[dict[str, Pca64CvaeExpert], list[dict[str, object]], list[dict[str, object]]]:
    torch, CVAEExpert, load_model_checkpoint, wrap_model_state_dict = _pca64_torch_imports(repo_root)
    resolved_device = _resolve_torch_device(torch, device)
    bank: dict[str, Pca64CvaeExpert] = {}
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
        source_x = transform_pca64(prep, _center_embeddings(train_cache, source_center))
        checkpoint_path = (
            artifacts_root
            / "checkpoints"
            / f"seed{int(artifact.experiment_seed)}"
            / f"expert_{source_center}_standardized_pca64.pt"
        )
        metadata = {
            "schema_version": FAMILY_C_PCA64_SCHEMA_VERSION,
            "input_dim": int(config.pca_dim),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "feature_space": PCA64_CHECKPOINT_FEATURE_SPACE,
            "pca_artifact_id": prep.pca_artifact_id,
            "scaler_artifact_id": prep.scaler_artifact_id,
            "source_center": str(source_center),
            "experiment_seed": int(artifact.experiment_seed),
            "pca_dim": int(config.pca_dim),
            "standardized": True,
            "kl_beta": float(config.kl_beta),
        }
        if checkpoint_path.exists():
            loaded = load_model_checkpoint(checkpoint_path, map_location=resolved_device)
            assert_pca64_checkpoint_metadata(loaded.checkpoint_metadata, expected=metadata)
            model = CVAEExpert(int(config.pca_dim), hidden_dim, latent_dim).to(resolved_device)
            model.load_state_dict(loaded.model_state_dict)
            status = "loaded"
        else:
            model = CVAEExpert(int(config.pca_dim), hidden_dim, latent_dim).to(resolved_device)
            _train_pca64_cvae(
                model=model,
                torch=torch,
                device=resolved_device,
                x=source_x,
                config=config,
                seed=_stable_seed(artifact.experiment_seed, source_center, "train"),
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(wrap_model_state_dict(model.state_dict(), metadata), checkpoint_path)
            status = "trained"
        model.eval()
        model._pca64_torch = torch
        model._pca64_device = resolved_device
        source_score = score_unlabeled_nelbo(
            model,
            source_x,
            torch=torch,
            device=resolved_device,
            kl_beta=config.kl_beta,
        )
        sample_scores = _sample_nelbo_values(model, source_x, torch=torch, device=resolved_device, kl_beta=config.kl_beta)
        source_std = float(_std(sample_scores))
        bank[str(source_center)] = Pca64CvaeExpert(
            experiment_seed=int(artifact.experiment_seed),
            source_center=str(source_center),
            model=model,
            preprocessor=prep,
            source_train_nelbo_mean=float(source_score.total),
            source_train_nelbo_std=source_std,
            checkpoint_path=checkpoint_path,
            input_dim=int(config.pca_dim),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            kl_beta=float(config.kl_beta),
        )
        checkpoint_rows.append(
            {
                "experiment_seed": int(artifact.experiment_seed),
                "source_center": str(source_center),
                "checkpoint_path": str(checkpoint_path),
                "input_dim": int(config.pca_dim),
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "feature_space": PCA64_CHECKPOINT_FEATURE_SPACE,
                "pca_artifact_id": prep.pca_artifact_id,
                "scaler_artifact_id": prep.scaler_artifact_id,
                "source_train_nelbo_mean": float(source_score.total),
                "source_train_nelbo_std": source_std,
                "kl_beta": float(config.kl_beta),
                "decoder_output_variance_assumption": "unit_variance_mse_proxy",
                "available": 1,
                "status": status,
            }
        )
    return bank, prep_rows, checkpoint_rows


def score_pca64_cvae_candidate(
    *,
    config: FamilyCPca64Config,
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
    expert: Pca64CvaeExpert,
    reference_pools: Mapping[int, Any],
    support_score: NelboScore,
    target_density_score: NelboScore,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    base = _row_base(
        schema_version=FAMILY_C_PCA64_SCHEMA_VERSION,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        unit=unit,
        candidate_expert=candidate_expert,
        generation_mode=PCA64_CVAE_MODE,
        budget_per_class=config.budget_per_class,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        row_type=PCA64_SINGLE_EXPERT_ROW_TYPE,
        n_target_eval=len(target_split.eval_indices),
        target_eval_pool_id=target_split.target_eval_pool_id,
        target_status=target_status,
        support_score=support_score,
        target_density_score=target_density_score,
        source_normalized=source_normalized_nelbo(
            support_score.total,
            mean=expert.source_train_nelbo_mean,
            std=expert.source_train_nelbo_std,
            eps=config.normalized_nelbo_eps,
        )[0],
    )
    generation_rows: list[dict[str, object]] = []
    try:
        synthetic, labels, generation_rows = generate_pca64_cvae_class_balanced(
            expert=expert,
            reference_pools=reference_pools,
            class_labels=(0, 1),
            budget_per_class=config.budget_per_class,
            generation_seed=generation_seed,
            context=base,
            small_std_threshold=config.small_std_threshold,
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
        row = {
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
    audit = _audit_row(
        row=row,
        expert=expert,
        target_expert_excluded=int(str(heldout_center) != str(candidate_expert)),
        support_eval_disjoint=1,
        checkpoint_feature_space=PCA64_CHECKPOINT_FEATURE_SPACE,
    )
    return row, generation_rows, audit


def score_pca64_real_reconstruction_candidate(
    *,
    config: FamilyCPca64Config,
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
    expert: Pca64CvaeExpert,
    support_score: NelboScore,
    target_density_score: NelboScore,
) -> tuple[dict[str, object], dict[str, object]]:
    base = _row_base(
        schema_version=FAMILY_C_PCA64_SCHEMA_VERSION,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        unit=unit,
        candidate_expert=candidate_expert,
        generation_mode=PCA64_REAL_RECONSTRUCTION_MODE,
        budget_per_class=0,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        row_type=PCA64_REAL_UPPER_ROW_TYPE,
        n_target_eval=len(target_split.eval_indices),
        target_eval_pool_id=target_split.target_eval_pool_id,
        target_status=target_status,
        support_score=support_score,
        target_density_score=target_density_score,
        source_normalized=source_normalized_nelbo(
            support_score.total,
            mean=expert.source_train_nelbo_mean,
            std=expert.source_train_nelbo_std,
            eps=config.normalized_nelbo_eps,
        )[0],
    )
    try:
        idxs = [idx for idx, row in enumerate(train_cache.metadata) if str(_domain(row)) == str(candidate_expert)]
        labels = [_label(train_cache.metadata[idx]) for idx in idxs]
        x_source = _slice_embeddings(train_cache.embeddings, tuple(idxs))
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
        row = {
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
    audit = _audit_row(
        row=row,
        expert=expert,
        target_expert_excluded=int(str(heldout_center) != str(candidate_expert)),
        support_eval_disjoint=1,
        checkpoint_feature_space=PCA64_CHECKPOINT_FEATURE_SPACE,
    )
    return row, audit


def generate_pca64_cvae_class_balanced(
    *,
    expert: Pca64CvaeExpert,
    reference_pools: Mapping[int, Any],
    class_labels: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
    context: Mapping[str, object],
    small_std_threshold: float,
) -> tuple[Any, list[int], list[dict[str, object]]]:
    import numpy as np  # type: ignore

    torch = expert.model._pca64_torch
    device = expert.model._pca64_device
    chunks: list[Any] = []
    labels: list[int] = []
    generation_rows: list[dict[str, object]] = []
    for class_label in tuple(int(v) for v in class_labels):
        refs = reference_pools.get(class_label)
        if refs is None or int(_as_numpy(refs).shape[0]) <= 0:
            raise ProtocolError(f"Empty standardized PCA64 reference pool for class {class_label}")
        refs_np = _as_numpy(refs)
        idx_gen = torch.Generator(device="cpu").manual_seed(int(generation_seed) + int(class_label))
        idx = torch.randint(int(refs_np.shape[0]), (int(budget_per_class),), generator=idx_gen, device="cpu")
        xb = torch.as_tensor(refs_np[idx.numpy()], dtype=torch.float32, device=device)
        gen = _torch_generator(torch, device, int(generation_seed) + 104729 + int(class_label))
        with torch.no_grad():
            mu, logvar = expert.model.encode(xb)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape, generator=gen, device=device, dtype=std.dtype)
            z = mu + eps * std
            generated_pca = expert.model.decode(z).detach().cpu().numpy()
        generated_dino = inverse_transform_pca64(expert.preprocessor, generated_pca)
        if int(generated_dino.shape[1]) != expert.preprocessor.embedding_dim:
            raise ProtocolError(
                f"Generated inverse PCA embeddings have wrong dim {generated_dino.shape[1]}, "
                f"expected {expert.preprocessor.embedding_dim}"
            )
        pca_std = float(np.std(generated_pca))
        dino_std = float(np.std(generated_dino))
        failures = []
        if pca_std <= float(small_std_threshold):
            failures.append("collapsed_pca_variance")
        if dino_std <= float(small_std_threshold):
            failures.append("collapsed_dino_variance")
        generation_rows.append(
            {
                "experiment_seed": context["experiment_seed"],
                "heldout_center": context["heldout_center"],
                "support_size": context["support_size"],
                "support_seed": context["support_seed"],
                "candidate_expert": context["candidate_expert"],
                "generation_mode": context["generation_mode"],
                "generation_seed": context["generation_seed"],
                "class_label": class_label,
                "num_generated_per_class_actual": int(generated_dino.shape[0]),
                "class_generation_failures": "|".join(failures),
                "generated_pca_std_mean": pca_std,
                "generated_dino_std_mean": dino_std,
                "generated_shape": json.dumps([int(v) for v in generated_dino.shape]),
                "inverse_scaler_used": 1,
                "inverse_pca_used": 1,
                "classifier_space": "original_dino_after_inverse_transform",
            }
        )
        chunks.append(generated_dino)
        labels.extend([class_label] * int(budget_per_class))
    return np.vstack(chunks), labels, generation_rows


def build_pca64_target_split(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
) -> Pca64TargetSplit:
    target_indices = tuple(idx for idx, row in enumerate(test_metadata) if str(_domain(row)) == str(heldout_center))
    labels_by_index = {idx: _label(test_metadata[idx]) for idx in target_indices}
    split = _make_support_eval_split(
        target_domain=int(heldout_center),
        target_indices=target_indices,
        labels_by_index=labels_by_index,
        support_size=int(support_size),
        sampling_policy="random",
        support_seed=int(support_seed),
    )
    support_indices = tuple(int(idx) for idx in split.support_indices)
    support_ids = tuple(str(_sample_id(test_metadata[idx])) for idx in support_indices)
    support_id_set = set(support_ids)
    eval_indices = tuple(
        idx for idx in target_indices if str(_sample_id(test_metadata[idx])) not in support_id_set
    )
    assert_disjoint_ids(support_ids, (str(_sample_id(test_metadata[idx])) for idx in eval_indices))
    return Pca64TargetSplit(
        support_indices=support_indices,
        eval_indices=eval_indices,
        support_sample_ids=tuple(sorted(support_ids)),
        target_eval_pool_id=str(support_eval_split_id),
    )


def source_normalized_nelbo(value: float, *, mean: float, std: float, eps: float = PCA64_NORMALIZED_EPS) -> tuple[float, int]:
    if math.isnan(float(value)) or math.isnan(float(mean)) or math.isnan(float(std)) or float(std) < float(eps):
        return math.nan, 0
    return (float(value) - float(mean)) / float(std), 1


def build_family_c_pca64_reports(*, artifacts_root: Path, candidate_domains: Sequence[str]) -> dict[str, Path]:
    paths = _artifact_paths(artifacts_root)
    rows = _read_dict_rows(paths["matrix"])
    align = build_family_c_pca64_alignment_rows(rows=rows, candidate_domains=candidate_domains)
    baseline = build_family_c_pca64_baseline_rows(rows=rows, alignment_rows=align)
    summary = classify_family_c_pca64_decision(rows=rows, alignment_rows=align)
    _write_dict_csv(paths["alignment"], PCA64_ALIGNMENT_COLUMNS, align)
    _write_dict_csv(paths["baseline"], PCA64_BASELINE_COLUMNS, baseline)
    paths["decision_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def build_family_c_pca64_alignment_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_domains: Sequence[str],
) -> list[dict[str, object]]:
    _ = tuple(candidate_domains)
    generated = [r for r in rows if r.get("generation_mode") == PCA64_CVAE_MODE and str(r.get("status")) == "ok"]
    by_context: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in generated:
        by_context.setdefault(_alignment_context_key(row), []).append(row)
    out: list[dict[str, object]] = []
    for key, group in sorted(by_context.items(), key=lambda item: item[0]):
        downstream_oracle = max(group, key=lambda row: (_float(row.get("bacc")), _float(row.get("macro_f1")), _reverse_sort(str(row.get("candidate_expert")))))
        density_oracle = min(group, key=lambda row: (_float(row.get("target_eval_nelbo_unlabeled")), str(row.get("candidate_expert"))))
        for selector, score_column in (
            (PCA64_RAW_SELECTOR, "support_nelbo_raw"),
            (PCA64_NORMALIZED_SELECTOR, "support_nelbo_source_normalized"),
        ):
            available_group = [row for row in group if not math.isnan(_float(row.get(score_column)))]
            if not available_group:
                selected = group[0]
                status = "unavailable_normalized_nelbo" if selector == PCA64_NORMALIZED_SELECTOR else "unavailable_raw_nelbo"
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
                        [-_float(row.get("support_nelbo_raw")) for row in group],
                        [_float(row.get("bacc")) for row in group],
                    ),
                    "available": available,
                    "status": status,
                }
            )
    return out


def build_family_c_pca64_baseline_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for selector in (PCA64_RAW_SELECTOR, PCA64_NORMALIZED_SELECTOR):
        subset = [row for row in alignment_rows if row.get("selector") == selector and int(row.get("available", 0)) == 1]
        out.append(
            {
                "method": selector,
                "row_type": "selector",
                "center_level_mean_bacc": _center_level_mean(subset, "selected_bacc"),
                "center_level_mean_macro_f1": _center_level_mean(subset, "selected_macro_f1"),
                "center_level_mean_oracle_gap_bacc": _center_level_mean(subset, "oracle_gap_bacc"),
                "top1_oracle_hit_rate": _mean(_float(row.get("top1_oracle_hit")) for row in subset),
                "spearman_neg_nelbo_vs_bacc": _mean(_float(row.get("spearman_neg_nelbo_vs_bacc")) for row in subset),
                "available": int(bool(subset)),
            }
        )
    generated = [row for row in rows if row.get("generation_mode") == PCA64_CVAE_MODE and row.get("row_type") == PCA64_SINGLE_EXPERT_ROW_TYPE and row.get("status") == "ok"]
    recon = [row for row in rows if row.get("generation_mode") == PCA64_REAL_RECONSTRUCTION_MODE and row.get("row_type") == PCA64_REAL_UPPER_ROW_TYPE and row.get("status") == "ok"]
    out.append(_oracle_baseline_row(generated, method="family_c_pca64_downstream_oracle", row_type="diagnostic_oracle"))
    out.append(_oracle_baseline_row(recon, method="PCA64_real_reconstruction_upper", row_type="diagnostic_upper_bound"))
    return out


def classify_family_c_pca64_decision(
    *,
    rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    raw_rows = [row for row in alignment_rows if row.get("selector") == PCA64_RAW_SELECTOR and int(row.get("available", 0)) == 1]
    selected = _center_level_mean(raw_rows, "selected_bacc")
    oracle = _center_level_mean(raw_rows, "downstream_oracle_bacc")
    gap = _center_level_mean(raw_rows, "oracle_gap_bacc")
    recon_oracle = _oracle_center_mean(
        [row for row in rows if row.get("generation_mode") == PCA64_REAL_RECONSTRUCTION_MODE and row.get("status") == "ok"]
    )
    if oracle >= 0.80 and selected < 0.80:
        classification = "PCA64_CVAE_ORACLE_STRONG_ROUTING_BOTTLENECK"
    elif oracle >= 0.80:
        classification = "PCA64_CVAE_GEOMETRY_BOTTLENECK_CONFIRMED"
    elif oracle < 0.80 and recon_oracle >= 0.80:
        classification = "PCA64_CVAE_MODELING_BOTTLENECK"
    elif oracle < 0.80 and recon_oracle < 0.80:
        classification = "PCA64_COMPRESSION_OR_SOURCE_SUBSPACE_LIMIT"
    else:
        classification = "DIAGNOSTIC_ONLY"
    return {
        "schema_version": FAMILY_C_PCA64_SCHEMA_VERSION,
        "decision_classification": classification,
        "pass_fail": "PASS" if classification == "PCA64_CVAE_GEOMETRY_BOTTLENECK_CONFIRMED" else "FAIL",
        "metrics": {
            "pca64_cvae_selected_center_level_mean_bacc": selected,
            "pca64_cvae_downstream_oracle_center_level_mean_bacc": oracle,
            "pca64_cvae_selected_oracle_gap_bacc": gap,
            "pca64_real_reconstruction_upper_center_level_mean_bacc": recon_oracle,
        },
        "claim_boundary": (
            "Family C PCA64 standardized CVAE is a diagnostic follow-up. It tests expert "
            "geometry/modeling limits and must not be presented as replacing Family C/C2 routing."
        ),
    }


def read_family_c_pca64_support_units(paths: Sequence[Path]) -> list[SupportSelectionUnit]:
    return [unit for unit in read_support_selection_units(paths, methods=(SUPPORT_NELBO_METHOD,)) if unit.method == SUPPORT_NELBO_METHOD]


def _train_pca64_cvae(*, model: Any, torch: Any, device: Any, x: Any, config: FamilyCPca64Config, seed: int) -> None:
    import numpy as np  # type: ignore

    torch.manual_seed(int(seed))
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[1] != int(config.pca_dim):
        raise ProtocolError(f"Training tensors must be [n,{config.pca_dim}], got {tuple(x_np.shape)}")
    n = int(x_np.shape[0])
    if n < 2:
        raise ProtocolError("Need at least two source_train rows to train PCA64 CVAE.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_val = max(1, int(round(n * float(config.val_fraction))))
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n - n_val > 0 else order
    train_x = torch.as_tensor(x_np[train_idx], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(x_np[val_idx], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = math.inf
    bad = 0
    for _epoch in range(int(config.epochs)):
        model.train()
        for batch in _iter_batches(train_x, int(config.batch_size), torch=torch, seed=int(seed) + int(_epoch)):
            optimizer.zero_grad()
            loss = _pca64_cvae_loss(model, batch, kl_beta=float(config.kl_beta), torch=torch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(_pca64_cvae_loss(model, val_x, kl_beta=float(config.kl_beta), torch=torch).detach().cpu().item())
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(config.patience):
                break
    model.load_state_dict(best_state)


def _pca64_cvae_loss(model: Any, x: Any, *, kl_beta: float, torch: Any) -> Any:
    recon, mu, logvar = model(x)
    recon_term = torch.mean((recon - x).pow(2), dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(recon_term + (float(kl_beta) * kl))


def _iter_batches(x: Any, batch_size: int, *, torch: Any, seed: int) -> Any:
    n = int(x.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(n, generator=gen).tolist()
    for start in range(0, n, int(batch_size)):
        idx = order[start : start + int(batch_size)]
        yield x[idx]


def _sample_nelbo_values(model: Any, x_standardized_pca64: Any, *, torch: Any, device: Any, kl_beta: float) -> list[float]:
    x_np = _as_numpy(x_standardized_pca64)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x)
        recon = model.decode(mu)
        recon_term = torch.mean((recon - x).pow(2), dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        total = recon_term + (float(kl_beta) * kl)
    return [float(v) for v in total.detach().cpu().numpy().tolist()]


def _standardized_reference_pools(*, train_cache: EmbeddingCache, expert: Pca64CvaeExpert, required_labels: Sequence[int]) -> dict[int, Any]:
    pools = build_class_reference_pools(
        train_cache=train_cache,
        candidate_expert=expert.source_center,
        required_labels=required_labels,
    )
    out: dict[int, Any] = {}
    for label, embeddings in pools.items():
        out[int(label)] = None if embeddings is None else transform_pca64(expert.preprocessor, embeddings)
    return out


def _center_embeddings(cache: EmbeddingCache, center: str) -> Any:
    idxs = [idx for idx, row in enumerate(cache.metadata) if str(_domain(row)) == str(center)]
    return _slice_embeddings(cache.embeddings, tuple(idxs))


def _slice_embeddings(embeddings: Any, indices: Sequence[int]) -> Any:
    if hasattr(embeddings, "__getitem__"):
        return embeddings[list(indices)]
    raise ProtocolError("Embedding cache object is not indexable.")


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
    score: NelboScore,
    source_mean: float,
    source_std: float,
    normalized: float,
    normalized_available: int,
    target_eval_score: NelboScore,
) -> dict[str, object]:
    return {
        "experiment_seed": int(unit.experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(unit.support_size),
        "support_seed": int(unit.support_seed),
        "support_eval_split_id": unit.support_eval_split_id,
        "source_center": source_center,
        "split_role": split_role,
        "n_samples": int(score.n_samples),
        "support_recon_term": float(score.recon),
        "support_kl_term": float(score.kl),
        "support_total_nelbo": float(score.total),
        "source_train_nelbo_mean": float(source_mean),
        "source_train_nelbo_std": float(source_std),
        "source_normalized_nelbo": normalized,
        "normalized_available": normalized_available,
        "target_eval_nelbo_unlabeled_diagnostic_only": float(target_eval_score.total),
        "nelbo_reduction": "mean_over_samples",
        "recon_reduction": "mean_per_sample_per_dim",
        "uniform_class_prior": 1,
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
    support_score: NelboScore,
    target_density_score: NelboScore,
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
        "support_nelbo_raw": float(support_score.total),
        "support_nelbo_source_normalized": source_normalized,
        "target_eval_nelbo_unlabeled": float(target_density_score.total),
    }


def _audit_row(
    *,
    row: Mapping[str, object],
    expert: Pca64CvaeExpert,
    target_expert_excluded: int,
    support_eval_disjoint: int,
    checkpoint_feature_space: str,
) -> dict[str, object]:
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
        "generated_embedding_dim": expert.preprocessor.embedding_dim,
        "target_expert_excluded": int(target_expert_excluded),
        "support_eval_disjoint": int(support_eval_disjoint),
        "pca_fit_split": "source_train",
        "scaler_fit_split": "source_train_pca_coordinates",
        "target_center_excluded_from_pca": int(target_expert_excluded),
        "target_rows_used_for_pca": 0,
        "target_rows_used_for_scaler": 0,
        "support_labels_used_for_nelbo": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_labels_used_for_training": 0,
        "target_eval_labels_used_for_final_metric_only": 1,
        "classifier_scaler_fit": "synthetic_train_only" if row.get("generation_mode") != PCA64_REAL_RECONSTRUCTION_MODE else "reconstructed_source_train_only",
        "checkpoint_feature_space": checkpoint_feature_space,
        "available": row.get("available", 0),
        "status": row.get("status", ""),
    }


def _lineage_key(*, row: Mapping[str, object], expert: Pca64CvaeExpert) -> str:
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
            "feature_space": PCA64_CHECKPOINT_FEATURE_SPACE,
            "pca_artifact_id": expert.preprocessor.pca_artifact_id,
            "scaler_artifact_id": expert.preprocessor.scaler_artifact_id,
        }
    )


def _target_label_status(target_labels: Sequence[int]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for label in target_labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return {
        "target_eval_label_counts_json": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        "target_eval_has_all_classes": int(all(counts.get(str(label), 0) > 0 for label in (0, 1))),
    }


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "matrix": root / "family_c_pca64_all_expert_downstream_matrix.csv",
        "preprocessing": root / "family_c_pca64_preprocessing_manifest.csv",
        "checkpoints": root / "family_c_pca64_expert_checkpoint_manifest.csv",
        "nelbo": root / "family_c_pca64_nelbo_diagnostics.csv",
        "generation": root / "family_c_pca64_generation_manifest.csv",
        "protocol_audit": root / "family_c_pca64_protocol_audit.csv",
        "alignment": root / "family_c_pca64_routing_alignment.csv",
        "baseline": root / "family_c_pca64_baseline_comparison.csv",
        "decision_summary": root / "family_c_pca64_decision_summary.json",
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
    return (
        _int_value(row.get("experiment_seed")),
        str(row.get("source_center", "")),
        str(row.get("preprocessing_artifact_key", "")),
    )


def _checkpoint_manifest_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int_value(row.get("experiment_seed")),
        str(row.get("source_center", "")),
        str(row.get("checkpoint_path", "")),
    )


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


def _primary_support_units(
    units: Sequence[SupportSelectionUnit],
    *,
    limits: FamilyCPca64BuildLimits,
) -> tuple[SupportSelectionUnit, ...]:
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


def _append_unique_dict_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    key_fn: Any,
) -> None:
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


def _oracle_baseline_row(rows: Sequence[Mapping[str, object]], *, method: str, row_type: str) -> dict[str, object]:
    return {
        "method": method,
        "row_type": row_type,
        "center_level_mean_bacc": _oracle_center_mean(rows),
        "center_level_mean_macro_f1": _oracle_center_mean(rows, metric="macro_f1"),
        "center_level_mean_oracle_gap_bacc": 0.0,
        "top1_oracle_hit_rate": math.nan,
        "spearman_neg_nelbo_vs_bacc": math.nan,
        "available": int(bool(rows)),
    }


def _oracle_center_mean(rows: Sequence[Mapping[str, object]], metric: str = "bacc") -> float:
    by_context: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        by_context.setdefault(_alignment_context_key(row), []).append(row)
    winners = []
    for group in by_context.values():
        winners.append(max(group, key=lambda row: (_float(row.get(metric)), _reverse_sort(str(row.get("candidate_expert"))))))
    return _center_level_mean(winners, metric)


def _center_level_mean(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    by_center: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if math.isnan(value):
            continue
        by_center.setdefault(str(row.get("heldout_center")), []).append(value)
    return _mean(_mean(values) for values in by_center.values())


def _mean(values: Any) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


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
