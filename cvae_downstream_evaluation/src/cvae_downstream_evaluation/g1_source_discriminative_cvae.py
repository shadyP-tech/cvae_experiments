"""G1 source-only discriminative CVAE objective diagnostic.

G1 retrains source-only class-conditioned PCA64 heteroscedastic CVAEs with
conservative source-discriminative auxiliary losses. It is not a routing
experiment: target labels are consumed only after fixed generation/aggregation
decisions are made for final downstream scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .c41_heteroscedastic import (
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    SourceTrainPCAProjection,
    build_source_train_reference_pools,
    decoder_logvar_diagnostics_by_class,
    fit_source_train_pca_projection,
    generate_posterior_sampled_embeddings,
    labels_from_metadata,
)
from .c41_workstation import (
    C41TrainingProfile,
    _indices_for_domain,
    _limit_c41_artifacts,
    _load_c41_model,
    _profile_for_support_config,
    _support_conditions,
    _write_csv,
    discover_c41_run_artifacts,
)
from .c61_mixture import (
    _GenerationCache,
    _file_hash,
    _mean as _c61_mean,
    _quantile,
    _to_numpy,
    load_csv_rows,
)
from .c62_late_ensemble import (
    C62_LEGACY_SUPPORT_UNITS,
    EnsembleMemberSpec,
    GLOBAL_CLASS_ORDER,
    POLICY_SAFE_MULTI as C62_POLICY_SAFE_MULTI,
    _generate_member as _generate_c62_member,
    _fit_member_probabilities,
    align_probabilities_to_class_order,
    fixed_predictions_from_probabilities,
)
from .c63_geometric_ensemble import (
    C63_ARTIFACTS_ROOT,
    GEOMETRIC_GENERATOR_FAMILY,
    LOG_PROBABILITY_EPSILON,
    POLICY_GEOM_SAFE_MULTI,
    build_c63_ensemble_plans,
    geometric_pool_probabilities,
)
from .c71a_source_probe_ce import (
    SourceProbe,
    _between_class_distance_ratio,
    _bootstrap_ci,
    _class_centroid_shift_norm,
    _class_cov_trace_ratios,
    _class_effective_rank_ratios,
    _ensure_cvae_testing_path,
    _grad_norm,
    _nearest_neighbor_concentration,
    _pairwise_mean,
    _probe_bacc_and_ce,
    _rbf_mmd,
    _std,
    _within_class_distance_ratio,
    source_probe_diagnostics,
    train_source_probe,
)
from .downstream import fit_locked_logistic_classifier
from .matrix import (
    MatrixBuildLimits,
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _records_for_split,
    _resolve_torch_device,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    BASELINE_ROUTING_FAMILY_USED,
    BASELINE_SELECTED_EXPERT_IDS_SOURCE,
    ENSEMBLE_EXPERT_ID,
    METHOD_BASELINE_ROW_TYPE,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


G1_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/g1_source_discriminative_cvae_v1"
G1_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
G1_DEFAULT_C42_ROOT = "cvae_downstream_evaluation/artifacts/c42_latent_gmm_prior_v1"
G1_DEFAULT_C63_ROOT = C63_ARTIFACTS_ROOT
G1_GENERATOR_FAMILY = "family_g1_pca64_class_conditional_source_discriminative_cvae_downstream_v1"

VARIANT_BASE = "G1_base_retrain"
VARIANT_PROBE_CE = "G1_probe_ce_only"
VARIANT_DISTILL = "G1_distill_only"
VARIANT_DISTILL_MARGIN = "G1_distill_margin_no_ce"
VARIANT_TEACHER_DISTILL_MARGIN = "G1_teacher_distill_margin"
G1_STAGE1_VARIANTS = (VARIANT_BASE, VARIANT_DISTILL, VARIANT_TEACHER_DISTILL_MARGIN)
G1_FULL_VARIANTS = (
    VARIANT_BASE,
    VARIANT_PROBE_CE,
    VARIANT_DISTILL,
    VARIANT_DISTILL_MARGIN,
    VARIANT_TEACHER_DISTILL_MARGIN,
)
G1_DISCRIMINATIVE_VARIANTS = tuple(v for v in G1_FULL_VARIANTS if v != VARIANT_BASE)

TEACHER_TEMPERATURE = 2.0
LAMBDA_CE_MAX = 0.03
LAMBDA_DISTILL_MAX = 0.03
LAMBDA_MARGIN_MAX = 0.02
AUX_WARMUP_EPOCHS = 5
AUX_RAMP_EPOCHS = 10
MAX_WEIGHTED_AUX_TO_NLL_RATIO = 0.20
SOURCE_VAL_NELBO_DEGRADATION_TOLERANCE = 0.10
CENTROID_MARGIN_FRACTION = 0.25

COLLAPSE_EFFECTIVE_RANK_RATIO_MIN = 0.85
COLLAPSE_COV_TRACE_RATIO_MIN = 0.75
COLLAPSE_COV_TRACE_RATIO_MAX = 1.25
COLLAPSE_NN_CONCENTRATION_BASE_MULTIPLIER = 1.20
COLLAPSE_NN_CONCENTRATION_ABS_MAX = 2.0

POLICY_G1_ONLY_GEOMETRIC = "g1_equal_weight_geometric_late_ensemble"
POLICY_C63_PLUS_G1_GEOMETRIC = "c63_safe_multiseed_plus_g1_equal_weight_geometric_late_ensemble"

DECISION_GENERATOR_SUCCESS = "G1_GENERATOR_OBJECTIVE_SUCCESS"
DECISION_COMBINED_USEFUL = "G1_COMBINED_OBJECTIVE_USEFUL"
DECISION_THESIS_PROGRESS = "G1_THESIS_PROGRESS"
FAILURE_SOURCE_GEOMETRY = "SOURCE_GEOMETRY_NOT_TARGET_UTILITY"
FAILURE_COLLAPSE = "DISCRIMINATIVE_LOSS_COLLAPSES_VARIANCE"
FAILURE_AUX_DOMINATES = "AUX_LOSS_DOMINATES_ELBO"
FAILURE_PROBE_WEAK = "SOURCE_PROBE_TOO_WEAK"
FAILURE_NO_GAIN = "G1_NO_GAIN_OVER_BASE"
FAILURE_SIMPLE_DISTILL = "G1_SIMPLE_DISTILLATION_BEATS_COMBINED"
FAILURE_SINGLE_NOT_ENSEMBLE = "G1_IMPROVES_SINGLE_EXPERT_NOT_ENSEMBLE"
FAILURE_AUGMENT_DILUTION = "C63_AUGMENTATION_DILUTION"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"

FORBIDDEN_PREJOIN_SUBSTRINGS = (
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "oracle",
    "regret",
    "target_eval",
    "target_label",
    "support_label",
    "current_heldout_utility",
)


@dataclass(frozen=True)
class G1RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None
    variants: tuple[str, ...] | None = None


@dataclass(frozen=True)
class G1VariantSpec:
    name: str
    ce_max: float = 0.0
    distill_max: float = 0.0
    margin_max: float = 0.0


@dataclass(frozen=True)
class G1TrainResult:
    checkpoint_path: Path
    history_rows: tuple[dict[str, object], ...]
    selected_val_nelbo: float
    selected_geometry: dict[str, float]


@dataclass(frozen=True)
class G1Member:
    candidate_expert: str
    variant: str
    generation_seed: int
    synthetic_pca: torch.Tensor
    synthetic_dino: torch.Tensor
    synthetic_labels: tuple[int, ...]
    checkpoint_path: Path
    projection_path: Path
    member_key: str
    weight: float = 1.0


TRAINING_COLUMNS = (
    "experiment_seed",
    "candidate_expert",
    "variant",
    "epoch",
    "aux_ce_weight",
    "aux_distill_weight",
    "aux_margin_weight",
    "train_loss",
    "val_loss",
    "train_nelbo_raw",
    "train_nll_raw",
    "train_kl_raw",
    "train_source_probe_ce_raw",
    "train_distill_kl_raw",
    "train_centroid_margin_raw",
    "train_source_val_composite_source_only",
    "train_weighted_aux_to_nll_ratio",
    "train_weighted_aux_to_total_ratio",
    "train_aux_cap_scale",
    "train_grad_norm_decoder_from_nll",
    "train_grad_norm_decoder_from_aux",
    "val_nelbo_raw",
    "val_nll_raw",
    "val_kl_raw",
    "val_source_probe_ce_raw",
    "val_distill_kl_raw",
    "val_centroid_margin_raw",
    "val_source_val_composite_source_only",
    "source_val_nelbo_degradation_vs_base",
    "source_val_nelbo_constraint_passed",
    "checkpoint_selection_metric",
    "checkpoint_selected",
    "checkpoint_selected_by_aux_metric",
    "geometry_collapse_warning",
    "effective_rank_ratio_min",
    "cov_trace_ratio_min",
    "cov_trace_ratio_max",
    "synthetic_nn_concentration_ratio",
)

GEOMETRY_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "candidate_expert",
    "variant",
    "generation_seed",
    "generation_mode",
    "generated_source_probe_bacc",
    "real_source_probe_bacc",
    "source_val_probe_bacc",
    "per_class_generated_cov_trace_ratio",
    "per_class_generated_cov_trace_ratio_min",
    "per_class_generated_cov_trace_ratio_max",
    "per_class_generated_cov_trace_ratio_mean",
    "per_class_generated_effective_rank_ratio",
    "per_class_generated_effective_rank_ratio_min",
    "per_class_generated_effective_rank_ratio_mean",
    "real_vs_generated_mmd_rbf_pca64",
    "class_centroid_shift_norm",
    "within_class_distance_ratio",
    "between_class_distance_ratio",
    "synthetic_nearest_neighbor_concentration",
    "real_nearest_neighbor_concentration",
    "synthetic_nn_concentration_ratio",
    "base_synthetic_nn_concentration_ratio",
    "class_geometry_collapse_warning",
    "decoder_logvar_mean",
    "decoder_logvar_at_max_frac",
)

SINGLE_MATRIX_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "variant",
    "generator_family",
    "generation_mode",
    "budget_per_class",
    "generation_seed",
    "classifier_seed",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "row_type",
    "n_synthetic_train",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_experts_hash",
    "utility_depends_on_support",
    "selection_depends_on_support",
    "routing_family_used",
    "routing_recomputed_for_g1",
    "selected_expert_ids_source",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "target_eval_labels_used_for_metrics_only",
    "status",
    "error_message",
)

ALIGNMENT_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "variant",
    "generation_seed",
    "classifier_seed",
    "selected_expert",
    "selected_bacc",
    "oracle_expert",
    "oracle_bacc",
    "oracle_gap",
    "paired_delta_vs_g1_base_retrain",
    "routing_family_used",
    "routing_recomputed_for_g1",
    "selected_expert_ids_source",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "status",
)

ENSEMBLE_COLUMNS = (
    "ensemble_policy",
    "variant",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "classifier_seed",
    "generator_family",
    "generation_mode",
    "budget_per_class",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "ensemble_ge_080",
    "oracle_bacc_reference",
    "regret_bacc",
    "row_type",
    "n_synthetic_train",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_expert",
    "candidate_experts_hash",
    "member_keys",
    "num_members",
    "aggregation_rule",
    "log_probability_epsilon",
    "geometric_softmax_temperature",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "paired_delta_vs_c63_replay",
    "status",
    "error_message",
)

SUMMARY_COLUMNS = (
    "summary_scope",
    "variant",
    "ensemble_policy",
    "heldout_center",
    "n_rows",
    "mean_bacc",
    "ge_080_rate",
    "mean_oracle_bacc",
    "mean_oracle_gap",
    "paired_delta_vs_g1_base_retrain",
    "paired_delta_vs_c63_replay",
    "positive_paired_delta_rate_vs_base",
    "strong_center_degrade_gt_002",
    "oracle_gap_delta_vs_base",
    "geometry_collapse_count",
    "decision_label",
)

PROTOCOL_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "variant",
    "heldout_source_excluded",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "target_eval_labels_used_for_metrics_only",
    "source_val_labels_used_for_checkpointing",
    "checkpoint_selection_metric",
    "downstream_metrics_used_for_checkpoint_selection",
    "routing_recomputed",
    "status",
)

PROVENANCE_COLUMNS = (
    "experiment_seed",
    "candidate_expert",
    "variant",
    "generator_family",
    "projection_source",
    "checkpoint_path",
    "checkpoint_hash",
    "source_probe_split",
    "source_val_split",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
)


VARIANT_SPECS: dict[str, G1VariantSpec] = {
    VARIANT_BASE: G1VariantSpec(VARIANT_BASE),
    VARIANT_PROBE_CE: G1VariantSpec(VARIANT_PROBE_CE, ce_max=LAMBDA_CE_MAX),
    VARIANT_DISTILL: G1VariantSpec(VARIANT_DISTILL, distill_max=LAMBDA_DISTILL_MAX),
    VARIANT_DISTILL_MARGIN: G1VariantSpec(
        VARIANT_DISTILL_MARGIN,
        distill_max=LAMBDA_DISTILL_MAX,
        margin_max=LAMBDA_MARGIN_MAX,
    ),
    VARIANT_TEACHER_DISTILL_MARGIN: G1VariantSpec(
        VARIANT_TEACHER_DISTILL_MARGIN,
        ce_max=LAMBDA_CE_MAX,
        distill_max=LAMBDA_DISTILL_MAX,
        margin_max=LAMBDA_MARGIN_MAX,
    ),
}


def g1_aux_weights(epoch: int, spec: G1VariantSpec) -> tuple[float, float, float]:
    """Predeclared warmup/ramp schedule for G1 auxiliary losses."""

    if int(epoch) < AUX_WARMUP_EPOCHS:
        scale = 0.0
    elif AUX_RAMP_EPOCHS <= 0:
        scale = 1.0
    else:
        progress = (int(epoch) - AUX_WARMUP_EPOCHS + 1) / float(AUX_RAMP_EPOCHS)
        scale = min(max(progress, 0.0), 1.0)
    return (
        float(spec.ce_max) * scale,
        float(spec.distill_max) * scale,
        float(spec.margin_max) * scale,
    )


def source_val_composite_source_only(
    *,
    source_val_nelbo: float,
    source_probe_ce: float,
    distill_kl: float,
    centroid_margin: float,
    ce_weight: float,
    distill_weight: float,
    margin_weight: float,
) -> float:
    return float(source_val_nelbo) + float(ce_weight) * float(source_probe_ce) + float(distill_weight) * float(distill_kl) + float(margin_weight) * float(centroid_margin)


def assert_g1_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if any(fragment in str(key).lower() for fragment in FORBIDDEN_PREJOIN_SUBSTRINGS)
        }
    )
    allowed = {"target_support_labels_used", "target_eval_labels_used_for_selection"}
    bad = [key for key in bad if key not in allowed]
    if bad:
        raise ProtocolError(f"G1 pre-join rows contain forbidden target/utility columns: {bad}")


def run_g1_source_discriminative_cvae(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c63_artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    training_profile: C41TrainingProfile,
    limits: G1RunLimits = G1RunLimits(),
) -> dict[str, Path]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    tables = artifacts_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    variants = limits.variants or G1_FULL_VARIANTS
    _validate_variants(variants)
    artifacts = _limit_c41_artifacts(discover_c41_run_artifacts(config=config, repo_root=repo_root), limits.experiment_seeds)
    generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    torch_device = _resolve_torch_device(torch, device)
    c63_reference = _load_c63_reference(c63_artifacts_root)

    training_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    g1_ensemble_rows: list[dict[str, object]] = []
    augmented_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        profile = _profile_for_support_config(training_profile, artifact.support.config_resolved)
        experiment_seed = int(artifact.support.experiment_seed)
        samples = _read_samples_manifest(artifact.support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        val_records = _records_for_split(samples, "val")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(artifact.support.train_cache, train_records, repo_root=repo_root)
        val_cache = _load_embedding_cache(artifact.val_cache, val_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(artifact.support.test_cache, test_records, repo_root=repo_root)
        c63_generation_cache = _GenerationCache(
            repo_root=repo_root,
            c41_artifacts_root=c41_artifacts_root,
            c42_artifacts_root=c42_artifacts_root,
            experiment_seed=experiment_seed,
            train_embeddings=train_cache.embeddings,
            train_metadata=train_cache.metadata,
            device=device,
        )

        candidate_state: dict[str, dict[str, Any]] = {}
        for candidate in tuple(str(c) for c in config.candidate_domains):
            projection = _fit_or_load_g1_projection(
                artifacts_root=artifacts_root,
                train_cache=train_cache,
                candidate_expert=candidate,
                seed=experiment_seed,
                n_components=profile.pca_components,
                resume=resume,
            )
            train_projected = projection.transform(train_cache.embeddings)
            val_projected = projection.transform(val_cache.embeddings)
            train_idx = _indices_for_domain(train_cache.metadata, candidate)
            val_idx = _indices_for_domain(val_cache.metadata, candidate)
            if not train_idx or not val_idx:
                raise ProtocolError(f"G1 requires nonempty source train/val rows for candidate={candidate}.")
            train_x = train_projected[train_idx]
            val_x = val_projected[val_idx]
            train_y = labels_from_metadata([train_cache.metadata[idx] for idx in train_idx])
            val_y = labels_from_metadata([val_cache.metadata[idx] for idx in val_idx])
            source_probe = train_source_probe(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                seed=experiment_seed + int(candidate),
                device=torch_device,
            )
            probe_rows.append(
                source_probe_diagnostics(
                    probe=source_probe,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    experiment_seed=experiment_seed,
                    candidate_expert=candidate,
                    device=torch_device,
                )
            )
            source_centroids = _class_centroids(train_x, train_y, device=torch_device)
            base_result = train_g1_cvae(
                repo_root=repo_root,
                artifacts_root=artifacts_root,
                experiment_seed=experiment_seed,
                candidate_expert=candidate,
                variant=VARIANT_BASE,
                train_x=train_x,
                val_x=val_x,
                train_y=train_y,
                val_y=val_y,
                source_probe=source_probe,
                source_centroids=source_centroids,
                profile=profile,
                device=torch_device,
                resume=resume,
                base_val_nelbo=None,
                base_geometry=None,
            )
            variant_ckpts = {VARIANT_BASE: base_result.checkpoint_path}
            training_rows.extend(base_result.history_rows)
            for variant in variants:
                if variant == VARIANT_BASE:
                    continue
                result = train_g1_cvae(
                    repo_root=repo_root,
                    artifacts_root=artifacts_root,
                    experiment_seed=experiment_seed,
                    candidate_expert=candidate,
                    variant=variant,
                    train_x=train_x,
                    val_x=val_x,
                    train_y=train_y,
                    val_y=val_y,
                    source_probe=source_probe,
                    source_centroids=source_centroids,
                    profile=profile,
                    device=torch_device,
                    resume=resume,
                    base_val_nelbo=base_result.selected_val_nelbo,
                    base_geometry=base_result.selected_geometry,
                )
                variant_ckpts[variant] = result.checkpoint_path
                training_rows.extend(result.history_rows)
            candidate_state[candidate] = {
                "projection": projection,
                "projection_path": _g1_projection_path(artifacts_root, experiment_seed, candidate),
                "train_embeddings": train_cache.embeddings,
                "train_metadata": train_cache.metadata,
                "reference_pools": build_source_train_reference_pools(
                    train_projected_embeddings=train_projected,
                    train_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=GLOBAL_CLASS_ORDER,
                ),
                "train_projected": train_projected,
                "val_projected": val_projected,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_y": train_y,
                "val_y": val_y,
                "source_probe": source_probe,
                "variant_ckpts": variant_ckpts,
            }
            for variant, checkpoint_path in sorted(variant_ckpts.items()):
                provenance_rows.append(
                    {
                        "experiment_seed": experiment_seed,
                        "candidate_expert": candidate,
                        "variant": variant,
                        "generator_family": G1_GENERATOR_FAMILY,
                        "projection_source": "reused_c41_full_source_train_pca64_semantics",
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_hash": _file_hash(checkpoint_path),
                        "source_probe_split": "source_train",
                        "source_val_split": "source_val_checkpointing_diagnostics_only",
                        "target_support_labels_used": 0,
                        "target_eval_labels_used_for_selection": 0,
                    }
                )

        for heldout_center in heldout_centers:
            heldout = str(heldout_center)
            if heldout not in {str(c) for c in config.candidate_domains}:
                raise ProtocolError(f"Unknown heldout center requested: {heldout}")
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _filtered_support_conditions(
                support_units,
                experiment_seed=experiment_seed,
                heldout_center=heldout,
                support_sizes=limits.support_sizes,
                support_seeds=limits.support_seeds,
            )
            if not support_conditions:
                raise ProtocolError(f"No support conditions for seed={experiment_seed}, heldout={heldout}.")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            target_dino = test_cache.embeddings[list(target_pool.eval_indices)]

            for variant in variants:
                for generation_seed in generation_seeds:
                    members = [
                        generate_g1_member(
                            state=candidate_state[candidate],
                            variant=variant,
                            experiment_seed=experiment_seed,
                            candidate_expert=candidate,
                            generation_seed=int(generation_seed),
                            budget_per_class=int(config.primary_budget_per_class),
                            device=device,
                        )
                        for candidate in candidates
                    ]
                    for member in members:
                        geometry_rows.append(
                            g1_generated_geometry_diagnostics(
                                member=member,
                                state=candidate_state[member.candidate_expert],
                                experiment_seed=experiment_seed,
                                heldout_center=heldout,
                            )
                        )
                    for support_size, support_seed in support_conditions:
                        unit = _support_unit(
                            support_units,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                        )
                        support_eval_split_id = unit.support_eval_split_id if unit else ""
                        oracle_reference = _c63_oracle_reference(
                            c63_reference,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                        )
                        c63_by_classifier = _c63_bacc_by_classifier(
                            c63_reference,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                        )
                        for classifier_seed in classifier_seeds:
                            single_rows.extend(
                                score_g1_single_members(
                                    members=members,
                                    experiment_seed=experiment_seed,
                                    heldout_center=heldout,
                                    support_size=int(support_size),
                                    support_seed=int(support_seed),
                                    support_eval_split_id=support_eval_split_id,
                                    classifier_seed=int(classifier_seed),
                                    target_dino=target_dino,
                                    target_labels=target_labels,
                                    target_eval_pool_id=target_pool.target_eval_pool_id,
                                    budget_per_class=int(config.primary_budget_per_class),
                                )
                            )
                            g1_ensemble_rows.append(
                                score_geometric_members(
                                    policy=POLICY_G1_ONLY_GEOMETRIC,
                                    variant=variant,
                                    members=members,
                                    experiment_seed=experiment_seed,
                                    heldout_center=heldout,
                                    support_size=int(support_size),
                                    support_seed=int(support_seed),
                                    support_eval_split_id=support_eval_split_id,
                                    generation_seed_group=str(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    budget_per_class=int(config.primary_budget_per_class),
                                    target_dino=target_dino,
                                    target_labels=target_labels,
                                    target_eval_pool_id=target_pool.target_eval_pool_id,
                                    oracle_reference=oracle_reference,
                                    c63_replay_bacc=c63_by_classifier.get(int(classifier_seed), math.nan),
                                )
                            )
                            c63_members = _c63_safe_members(
                                cache=c63_generation_cache,
                                candidates=candidates,
                                generation_seeds=generation_seeds,
                                total_budget_per_class=int(config.primary_budget_per_class),
                            )
                            augmented_rows.append(
                                score_geometric_members(
                                    policy=POLICY_C63_PLUS_G1_GEOMETRIC,
                                    variant=variant,
                                    members=tuple(c63_members) + tuple(members),
                                    experiment_seed=experiment_seed,
                                    heldout_center=heldout,
                                    support_size=int(support_size),
                                    support_seed=int(support_seed),
                                    support_eval_split_id=support_eval_split_id,
                                    generation_seed_group="c63_safe_plus_g1",
                                    classifier_seed=int(classifier_seed),
                                    budget_per_class=int(config.primary_budget_per_class),
                                    target_dino=target_dino,
                                    target_labels=target_labels,
                                    target_eval_pool_id=target_pool.target_eval_pool_id,
                                    oracle_reference=oracle_reference,
                                    c63_replay_bacc=c63_by_classifier.get(int(classifier_seed), math.nan),
                                )
                            )
                            protocol_rows.append(
                                {
                                    "experiment_seed": experiment_seed,
                                    "heldout_center": heldout,
                                    "variant": variant,
                                    "heldout_source_excluded": int(heldout not in {str(c) for c in candidates}),
                                    "target_support_labels_used": 0,
                                    "target_eval_labels_used_for_selection": 0,
                                    "target_eval_labels_used_for_metrics_only": 1,
                                    "source_val_labels_used_for_checkpointing": 1,
                                    "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only_with_source_only_constraints",
                                    "downstream_metrics_used_for_checkpoint_selection": 0,
                                    "routing_recomputed": 0,
                                    "status": "PASS",
                                }
                            )

    geometry_rows = annotate_geometry_collapse_vs_base(geometry_rows)
    alignment_rows = build_g1_locked_selected_alignment(single_rows, support_units)
    summary_rows = build_g1_summary_rows(
        alignment_rows=alignment_rows,
        g1_ensemble_rows=g1_ensemble_rows,
        augmented_rows=augmented_rows,
        geometry_rows=geometry_rows,
    )
    threshold_rows = build_g1_threshold_rows(summary_rows)
    assert_g1_prejoin_rows_safe(provenance_rows)

    outputs = {
        "training": tables / "g1_training_diagnostics.csv",
        "probe": tables / "g1_source_probe_diagnostics.csv",
        "geometry": tables / "g1_generated_geometry_diagnostics.csv",
        "single_matrix": tables / "g1_single_expert_downstream_matrix.csv",
        "alignment": tables / "g1_locked_selected_alignment.csv",
        "g1_ensemble": tables / "g1_geometric_ensemble_matrix.csv",
        "augmented_ensemble": tables / "g1_c63_augmented_ensemble_matrix.csv",
        "center": tables / "g1_center_summary.csv",
        "threshold": tables / "g1_threshold_audit.csv",
        "protocol": tables / "g1_protocol_audit.csv",
        "provenance": tables / "g1_generator_provenance.csv",
    }
    _write_csv(outputs["training"], TRAINING_COLUMNS, training_rows)
    _write_csv(outputs["probe"], tuple(probe_rows[0].keys()) if probe_rows else (), probe_rows)
    _write_csv(outputs["geometry"], GEOMETRY_COLUMNS, geometry_rows)
    _write_csv(outputs["single_matrix"], SINGLE_MATRIX_COLUMNS, single_rows)
    _write_csv(outputs["alignment"], ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(outputs["g1_ensemble"], ENSEMBLE_COLUMNS, g1_ensemble_rows)
    _write_csv(outputs["augmented_ensemble"], ENSEMBLE_COLUMNS, augmented_rows)
    _write_csv(outputs["center"], SUMMARY_COLUMNS, summary_rows)
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, threshold_rows)
    _write_csv(outputs["protocol"], PROTOCOL_COLUMNS, protocol_rows)
    _write_csv(outputs["provenance"], PROVENANCE_COLUMNS, provenance_rows)
    return outputs


def train_g1_cvae(
    *,
    repo_root: Path,
    artifacts_root: Path,
    experiment_seed: int,
    candidate_expert: str,
    variant: str,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    source_probe: SourceProbe,
    source_centroids: Mapping[int, torch.Tensor],
    profile: C41TrainingProfile,
    device: torch.device,
    resume: bool,
    base_val_nelbo: float | None,
    base_geometry: Mapping[str, float] | None,
) -> G1TrainResult:
    _ensure_cvae_testing_path(repo_root)
    from src.models.cvae_expert import (  # type: ignore
        CVAEExpert,
        DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        RECON_LOSS_GAUSSIAN_NLL_DIAG,
        REDUCTION_MEAN,
    )
    from src.train.checkpoint_provenance import load_model_checkpoint, wrap_model_state_dict  # type: ignore

    if variant not in VARIANT_SPECS:
        raise ProtocolError(f"Unknown G1 variant: {variant}")
    spec = VARIANT_SPECS[variant]
    out_dir = artifacts_root / "checkpoints" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / _variant_slug(variant)
    ckpt = out_dir / f"{_variant_slug(variant)}_class_conditional_pca64.pt"
    history_path = out_dir / "training_history.csv"
    if ckpt.exists() and resume:
        loaded = load_model_checkpoint(ckpt, map_location=device)
        rows = _read_csv_dicts(history_path)
        selected = _selected_nelbo_from_history(rows)
        geometry = _selected_geometry_from_history(rows)
        return G1TrainResult(checkpoint_path=ckpt, history_rows=tuple(rows), selected_val_nelbo=selected, selected_geometry=geometry)
    if ckpt.exists() and not resume:
        raise ProtocolError(f"G1 checkpoint already exists; use --resume or a clean artifact root: {ckpt}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(experiment_seed) + int(candidate_expert) + _variant_seed_offset(variant))
    model = CVAEExpert(
        input_dim=int(train_x.shape[1]),
        hidden_dim=int(profile.hidden_dim),
        latent_dim=int(profile.latent_dim),
        class_condition_dim=2,
        decoder_likelihood=DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        decoder_logvar_min=-9.21,
        decoder_logvar_max=2.0,
        decoder_min_variance=1.0e-4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(profile.lr))
    train_loader = DataLoader(
        TensorDataset(train_x.float(), train_y.long()),
        batch_size=int(profile.batch_size),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(experiment_seed) + int(candidate_expert)),
    )
    val_loader = DataLoader(TensorDataset(val_x.float(), val_y.long()), batch_size=int(profile.batch_size), shuffle=False)
    decoder_params = [param for name, param in model.named_parameters() if name.startswith("dec")]
    history: list[dict[str, object]] = []
    best_sort: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(int(profile.epochs)):
        ce_weight, distill_weight, margin_weight = g1_aux_weights(epoch, spec)
        train_stats = _run_g1_epoch(
            model=model,
            source_probe=source_probe,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            source_centroids=source_centroids,
            ce_weight=ce_weight,
            distill_weight=distill_weight,
            margin_weight=margin_weight,
            decoder_params=decoder_params,
            train=True,
        )
        val_stats = _run_g1_epoch(
            model=model,
            source_probe=source_probe,
            loader=val_loader,
            optimizer=None,
            device=device,
            source_centroids=source_centroids,
            ce_weight=ce_weight,
            distill_weight=distill_weight,
            margin_weight=margin_weight,
            decoder_params=decoder_params,
            train=False,
        )
        geometry = _source_val_decoder_geometry(
            model=model,
            source_probe=source_probe,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            device=device,
            base_geometry=base_geometry,
        )
        val_nelbo = float(val_stats["nelbo_raw"])
        nelbo_degradation = val_nelbo - float(base_val_nelbo) if base_val_nelbo is not None else 0.0
        nelbo_constraint_passed = int(base_val_nelbo is None or nelbo_degradation <= SOURCE_VAL_NELBO_DEGRADATION_TOLERANCE)
        geometry_ok = int(not bool(geometry["geometry_collapse_warning"]))
        selectable = bool(nelbo_constraint_passed and geometry_ok)
        sort_key = (
            val_nelbo if selectable else float("inf"),
            float(val_stats["source_val_composite_source_only"]),
        )
        selected = 0
        if best_sort is None or sort_key < best_sort:
            best_sort = sort_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = int(epoch)
            selected = 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(profile.patience):
                # If all constrained checkpoints failed, keep training until at least
                # patience is exhausted, then fall back below.
                break
        row = {
            "experiment_seed": int(experiment_seed),
            "candidate_expert": candidate_expert,
            "variant": variant,
            "epoch": int(epoch),
            "aux_ce_weight": ce_weight,
            "aux_distill_weight": distill_weight,
            "aux_margin_weight": margin_weight,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_nelbo_raw": train_stats["nelbo_raw"],
            "train_nll_raw": train_stats["nll_raw"],
            "train_kl_raw": train_stats["kl_raw"],
            "train_source_probe_ce_raw": train_stats["source_probe_ce_raw"],
            "train_distill_kl_raw": train_stats["distill_kl_raw"],
            "train_centroid_margin_raw": train_stats["centroid_margin_raw"],
            "train_source_val_composite_source_only": train_stats["source_val_composite_source_only"],
            "train_weighted_aux_to_nll_ratio": train_stats["weighted_aux_to_nll_ratio"],
            "train_weighted_aux_to_total_ratio": train_stats["weighted_aux_to_total_ratio"],
            "train_aux_cap_scale": train_stats["aux_cap_scale"],
            "train_grad_norm_decoder_from_nll": train_stats["grad_norm_decoder_from_nll"],
            "train_grad_norm_decoder_from_aux": train_stats["grad_norm_decoder_from_aux"],
            "val_nelbo_raw": val_stats["nelbo_raw"],
            "val_nll_raw": val_stats["nll_raw"],
            "val_kl_raw": val_stats["kl_raw"],
            "val_source_probe_ce_raw": val_stats["source_probe_ce_raw"],
            "val_distill_kl_raw": val_stats["distill_kl_raw"],
            "val_centroid_margin_raw": val_stats["centroid_margin_raw"],
            "val_source_val_composite_source_only": val_stats["source_val_composite_source_only"],
            "source_val_nelbo_degradation_vs_base": nelbo_degradation,
            "source_val_nelbo_constraint_passed": nelbo_constraint_passed,
            "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only_with_source_only_constraints",
            "checkpoint_selected": selected,
            "checkpoint_selected_by_aux_metric": 0,
            **geometry,
        }
        history.append(row)
    if best_state is None:
        # Constrained selection found no candidate. Fall back to the lowest NELBO
        # row so the failure is auditable rather than silently dropping a model.
        best_row = min(history, key=lambda r: float(r["val_nelbo_raw"]))
        best_epoch = int(best_row["epoch"])
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for row in history:
        row["checkpoint_selected"] = int(int(row["epoch"]) == int(best_epoch))
    metadata = {
        "generator_family": G1_GENERATOR_FAMILY,
        "experiment_id": "G1",
        "variant": variant,
        "experiment_seed": int(experiment_seed),
        "candidate_expert": str(candidate_expert),
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(profile.hidden_dim),
        "latent_dim": int(profile.latent_dim),
        "class_condition_dim": 2,
        "decoder_likelihood": DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        "decoder_logvar_min": -9.21,
        "decoder_logvar_max": 2.0,
        "decoder_min_variance": 1.0e-4,
        "reconstruction_loss": RECON_LOSS_GAUSSIAN_NLL_DIAG,
        "recon_reduction": REDUCTION_MEAN,
        "kl_reduction": REDUCTION_MEAN,
        "beta_effective": 1.0,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "lambda_ce_max": spec.ce_max,
        "lambda_distill_max": spec.distill_max,
        "lambda_margin_max": spec.margin_max,
        "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only_with_source_only_constraints",
        "source_val_labels_used_for_checkpointing": 1,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
    }
    torch.save(wrap_model_state_dict(best_state, metadata), ckpt)
    _write_csv(history_path, TRAINING_COLUMNS, history)
    selected_row = next((row for row in history if int(row["checkpoint_selected"]) == 1), history[-1])
    selected_geometry = {
        "effective_rank_ratio_min": float(selected_row["effective_rank_ratio_min"]),
        "cov_trace_ratio_min": float(selected_row["cov_trace_ratio_min"]),
        "cov_trace_ratio_max": float(selected_row["cov_trace_ratio_max"]),
        "synthetic_nn_concentration_ratio": float(selected_row["synthetic_nn_concentration_ratio"]),
    }
    return G1TrainResult(
        checkpoint_path=ckpt,
        history_rows=tuple(history),
        selected_val_nelbo=float(selected_row["val_nelbo_raw"]),
        selected_geometry=selected_geometry,
    )


def _run_g1_epoch(
    *,
    model: Any,
    source_probe: SourceProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    source_centroids: Mapping[int, torch.Tensor],
    ce_weight: float,
    distill_weight: float,
    margin_weight: float,
    decoder_params: Sequence[torch.nn.Parameter],
    train: bool,
) -> dict[str, float]:
    from src.models.cvae_expert import RECON_LOSS_GAUSSIAN_NLL_DIAG, REDUCTION_MEAN, elbo_loss_terms  # type: ignore

    model.train(bool(train))
    sums = {
        "loss": 0.0,
        "nelbo_raw": 0.0,
        "nll_raw": 0.0,
        "kl_raw": 0.0,
        "source_probe_ce_raw": 0.0,
        "distill_kl_raw": 0.0,
        "centroid_margin_raw": 0.0,
        "source_val_composite_source_only": 0.0,
        "weighted_aux_to_nll_ratio": 0.0,
        "weighted_aux_to_total_ratio": 0.0,
        "aux_cap_scale": 0.0,
        "grad_norm_decoder_from_nll": 0.0,
        "grad_norm_decoder_from_aux": 0.0,
    }
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            recon_payload, mu_z, logvar_z, _aux = model(xb, y=yb, return_aux=True, return_distribution=True)
            recon_mu, recon_logvar = recon_payload
            terms = elbo_loss_terms(
                recon_mu,
                xb,
                mu_z,
                logvar_z,
                recon_logvar_x=recon_logvar,
                reconstruction_loss=RECON_LOSS_GAUSSIAN_NLL_DIAG,
                recon_reduction=REDUCTION_MEAN,
                kl_reduction=REDUCTION_MEAN,
                kl_weight=1.0,
            )
            nelbo = terms["loss"].mean()
            decoder_mean, _decoder_logvar = model.decode(mu_z, y=yb, return_distribution=True)
            student_logits = source_probe(decoder_mean)
            with torch.no_grad():
                teacher_logits = source_probe(xb)
            source_probe_ce = F.cross_entropy(student_logits, yb)
            distill_kl = F.kl_div(
                F.log_softmax(student_logits / TEACHER_TEMPERATURE, dim=1),
                F.softmax(teacher_logits / TEACHER_TEMPERATURE, dim=1),
                reduction="batchmean",
            ) * (TEACHER_TEMPERATURE ** 2)
            margin_loss = centroid_margin_hinge(decoder_mean, yb, source_centroids)
            weighted_aux_raw = float(ce_weight) * source_probe_ce + float(distill_weight) * distill_kl + float(margin_weight) * margin_loss
            aux_cap = MAX_WEIGHTED_AUX_TO_NLL_RATIO * nelbo.detach().abs().clamp_min(1.0e-12)
            aux_scale = torch.clamp(aux_cap / weighted_aux_raw.detach().clamp_min(1.0e-12), max=1.0) if float(weighted_aux_raw.detach().item()) > 0 else torch.tensor(1.0, device=device)
            weighted_aux = weighted_aux_raw * aux_scale
            loss = nelbo + weighted_aux
            composite = source_val_composite_source_only(
                source_val_nelbo=float(nelbo.detach().item()),
                source_probe_ce=float(source_probe_ce.detach().item()),
                distill_kl=float(distill_kl.detach().item()),
                centroid_margin=float(margin_loss.detach().item()),
                ce_weight=float(ce_weight),
                distill_weight=float(distill_weight),
                margin_weight=float(margin_weight),
            )
            batch_n = int(xb.shape[0])
            if train and optimizer is not None:
                nll_grad = _grad_norm(nelbo, decoder_params, retain_graph=True)
                aux_grad = _grad_norm(weighted_aux_raw, decoder_params, retain_graph=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                nll_grad = math.nan
                aux_grad = math.nan
            sums["loss"] += float(loss.item()) * batch_n
            sums["nelbo_raw"] += float(nelbo.item()) * batch_n
            sums["nll_raw"] += float(terms["recon_nll"].mean().item()) * batch_n
            sums["kl_raw"] += float(terms["kl"].mean().item()) * batch_n
            sums["source_probe_ce_raw"] += float(source_probe_ce.item()) * batch_n
            sums["distill_kl_raw"] += float(distill_kl.item()) * batch_n
            sums["centroid_margin_raw"] += float(margin_loss.item()) * batch_n
            sums["source_val_composite_source_only"] += float(composite) * batch_n
            sums["weighted_aux_to_nll_ratio"] += float((weighted_aux.detach() / nelbo.detach().abs().clamp_min(1.0e-12)).item()) * batch_n
            sums["weighted_aux_to_total_ratio"] += float((weighted_aux.detach() / loss.detach().abs().clamp_min(1.0e-12)).item()) * batch_n
            sums["aux_cap_scale"] += float(aux_scale.detach().item()) * batch_n
            if not math.isnan(nll_grad):
                sums["grad_norm_decoder_from_nll"] += nll_grad * batch_n
                sums["grad_norm_decoder_from_aux"] += aux_grad * batch_n
            count += batch_n
    return {key: value / float(max(count, 1)) for key, value in sums.items()}


def centroid_margin_hinge(
    x: torch.Tensor,
    y: torch.Tensor,
    centroids: Mapping[int, torch.Tensor],
    *,
    margin_fraction: float = CENTROID_MARGIN_FRACTION,
) -> torch.Tensor:
    labels = sorted(int(v) for v in centroids)
    if len(labels) < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    c0 = centroids[labels[0]].to(device=x.device, dtype=x.dtype)
    c1 = centroids[labels[1]].to(device=x.device, dtype=x.dtype)
    base_margin = float((c0 - c1).norm().detach().item()) * float(margin_fraction)
    d0 = (x - c0).pow(2).sum(dim=1).sqrt()
    d1 = (x - c1).pow(2).sum(dim=1).sqrt()
    true_dist = torch.where(y.long() == labels[0], d0, d1)
    other_dist = torch.where(y.long() == labels[0], d1, d0)
    return torch.relu(float(base_margin) + true_dist - other_dist).mean()


def generate_g1_member(
    *,
    state: Mapping[str, Any],
    variant: str,
    experiment_seed: int,
    candidate_expert: str,
    generation_seed: int,
    budget_per_class: int,
    device: str,
) -> G1Member:
    projection: SourceTrainPCAProjection = state["projection"]
    checkpoint = state["variant_ckpts"][variant]
    model = _load_g1_model(checkpoint, device=_resolve_torch_device(torch, device))
    chunks = []
    labels: list[int] = []
    for label in GLOBAL_CLASS_ORDER:
        generated = generate_posterior_sampled_embeddings(
            model=model,
            reference_pool=state["reference_pools"][int(label)].to(next(model.parameters()).device),
            class_label=int(label),
            n_samples=int(budget_per_class),
            seed=int(generation_seed) + int(label),
            generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        )
        chunks.append(generated.embeddings)
        labels.extend(int(v) for v in generated.labels.tolist())
    synthetic_pca = torch.cat(chunks, dim=0).detach().cpu().float()
    synthetic_dino = projection.inverse_transform(synthetic_pca).detach().cpu().float()
    return G1Member(
        candidate_expert=candidate_expert,
        variant=variant,
        generation_seed=int(generation_seed),
        synthetic_pca=synthetic_pca,
        synthetic_dino=synthetic_dino,
        synthetic_labels=tuple(labels),
        checkpoint_path=checkpoint,
        projection_path=state["projection_path"],
        member_key=f"{variant}::expert_{candidate_expert}::posterior_mean::seed_{int(generation_seed)}",
        weight=1.0,
    )


def score_g1_single_members(
    *,
    members: Sequence[G1Member],
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    classifier_seed: int,
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    budget_per_class: int,
) -> list[dict[str, object]]:
    rows = []
    for member in members:
        base = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "support_size": int(support_size),
            "support_seed": int(support_seed),
            "support_eval_split_id": support_eval_split_id,
            "candidate_expert": member.candidate_expert,
            "variant": member.variant,
            "generator_family": G1_GENERATOR_FAMILY,
            "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
            "budget_per_class": int(budget_per_class),
            "generation_seed": int(member.generation_seed),
            "classifier_seed": int(classifier_seed),
            "row_type": SINGLE_EXPERT_ROW_TYPE,
            "n_synthetic_train": len(member.synthetic_labels),
            "n_target_eval": len(target_labels),
            "target_eval_pool_id": target_eval_pool_id,
            "candidate_experts_hash": hash_candidate_experts([member.candidate_expert]),
            "utility_depends_on_support": 0,
            "selection_depends_on_support": 0,
            "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
            "routing_recomputed_for_g1": 0,
            "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
            "target_support_labels_used": 0,
            "target_eval_labels_used_for_selection": 0,
            "target_eval_labels_used_for_metrics_only": 1,
        }
        try:
            prediction = fit_locked_logistic_classifier(
                _to_numpy(member.synthetic_dino),
                member.synthetic_labels,
                _to_numpy(target_dino),
                target_labels,
                classifier_seed=int(classifier_seed),
            )
            rows.append(
                {
                    **base,
                    "bacc": prediction.score.balanced_accuracy,
                    "macro_f1": prediction.score.macro_f1,
                    "auroc": prediction.score.secondary_metrics.get("auroc", math.nan),
                    "auprc": prediction.score.secondary_metrics.get("auprc", math.nan),
                    "status": "ok",
                    "error_message": "",
                }
            )
        except Exception as exc:
            rows.append({**base, "bacc": math.nan, "macro_f1": math.nan, "auroc": math.nan, "auprc": math.nan, "status": "failed_g1_single_scoring", "error_message": str(exc)})
    return rows


def score_geometric_members(
    *,
    policy: str,
    variant: str,
    members: Sequence[Any],
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    generation_seed_group: str,
    classifier_seed: int,
    budget_per_class: int,
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    oracle_reference: float,
    c63_replay_bacc: float,
) -> dict[str, object]:
    base = {
        "ensemble_policy": policy,
        "variant": variant,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed_group": generation_seed_group,
        "classifier_seed": int(classifier_seed),
        "generator_family": G1_GENERATOR_FAMILY if policy == POLICY_G1_ONLY_GEOMETRIC else f"{GEOMETRIC_GENERATOR_FAMILY}+{G1_GENERATOR_FAMILY}",
        "generation_mode": policy,
        "budget_per_class": int(budget_per_class),
        "row_type": METHOD_BASELINE_ROW_TYPE,
        "n_target_eval": len(target_labels),
        "target_eval_pool_id": target_eval_pool_id,
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "candidate_experts_hash": hash_candidate_experts(_member_key(member) for member in members),
        "member_keys": ";".join(_member_key(member) for member in members),
        "num_members": len(members),
        "aggregation_rule": "weighted_log_probability_geometric_pooling",
        "log_probability_epsilon": LOG_PROBABILITY_EPSILON,
        "geometric_softmax_temperature": 1.0,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
    }
    try:
        probs = []
        weights = []
        total_train = 0
        for member in members:
            dino, labels, weight = _member_payload(member)
            pred = _fit_member_probabilities(
                synthetic_embeddings=_to_numpy(dino),
                synthetic_labels=labels,
                target_embeddings=_to_numpy(target_dino),
                classifier_seed=int(classifier_seed),
            )
            probs.append(align_probabilities_to_class_order(pred["probabilities"], pred["classes"], GLOBAL_CLASS_ORDER))
            weights.append(float(weight))
            total_train += len(labels)
        stacked = _np_stack(probs)
        scores, geometric_prob = geometric_pool_probabilities(stacked, weights)
        pred_idx = _np_argmax(scores)
        predictions = [int(GLOBAL_CLASS_ORDER[int(idx)]) for idx in pred_idx]
        from .c62_late_ensemble import _score_predictions_and_probabilities  # type: ignore

        metrics = _score_predictions_and_probabilities(target_labels, predictions, geometric_prob, GLOBAL_CLASS_ORDER)
        bacc = float(metrics["bacc"])
        return {
            **base,
            "bacc": bacc,
            "macro_f1": metrics["macro_f1"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "ensemble_ge_080": int(bacc >= 0.80),
            "oracle_bacc_reference": oracle_reference,
            "regret_bacc": float(oracle_reference) - bacc if not math.isnan(float(oracle_reference)) else math.nan,
            "n_synthetic_train": total_train,
            "paired_delta_vs_c63_replay": bacc - float(c63_replay_bacc) if not math.isnan(float(c63_replay_bacc)) else math.nan,
            "status": "ok",
            "error_message": "",
        }
    except Exception as exc:
        return {
            **base,
            "bacc": math.nan,
            "macro_f1": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "ensemble_ge_080": 0,
            "oracle_bacc_reference": oracle_reference,
            "regret_bacc": math.nan,
            "n_synthetic_train": sum(len(_member_payload(member)[1]) for member in members),
            "paired_delta_vs_c63_replay": math.nan,
            "status": "failed_g1_geometric_scoring",
            "error_message": str(exc),
        }


def g1_generated_geometry_diagnostics(
    *,
    member: G1Member,
    state: Mapping[str, Any],
    experiment_seed: int,
    heldout_center: str,
) -> dict[str, object]:
    train_idx = state["train_idx"]
    val_idx = state["val_idx"]
    train_projected = state["train_projected"]
    val_projected = state["val_projected"]
    source_train_pca = train_projected[train_idx].float()
    source_val_pca = val_projected[val_idx].float()
    train_y = state["train_y"].long()
    val_y = state["val_y"].long()
    synthetic_pca = member.synthetic_pca.float()
    synthetic_y = torch.tensor(member.synthetic_labels, dtype=torch.long)
    probe = state["source_probe"]
    device = next(probe.parameters()).device
    gen_bacc, _gen_ce = _probe_bacc_and_ce(probe, synthetic_pca, synthetic_y, device=device)
    real_bacc, _real_ce = _probe_bacc_and_ce(probe, source_train_pca, train_y, device=device)
    val_bacc, _val_ce = _probe_bacc_and_ce(probe, source_val_pca, val_y, device=device)
    ratios = _class_cov_trace_ratios(synthetic_pca, synthetic_y, source_train_pca, train_y)
    rank_ratios = _class_effective_rank_ratios(synthetic_pca, synthetic_y, source_train_pca, train_y)
    nn_syn = _nearest_neighbor_concentration(synthetic_pca, source_train_pca)
    nn_real = _nearest_neighbor_concentration(source_val_pca, source_train_pca)
    nn_ratio = nn_syn / max(nn_real, 1.0e-12)
    model = _load_g1_model(member.checkpoint_path, device=device)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "candidate_expert": member.candidate_expert,
        "variant": member.variant,
        "generation_seed": int(member.generation_seed),
        "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        "generated_source_probe_bacc": gen_bacc,
        "real_source_probe_bacc": real_bacc,
        "source_val_probe_bacc": val_bacc,
        "per_class_generated_cov_trace_ratio": _format_class_map(ratios),
        "per_class_generated_cov_trace_ratio_min": min(ratios.values()),
        "per_class_generated_cov_trace_ratio_max": max(ratios.values()),
        "per_class_generated_cov_trace_ratio_mean": _mean(ratios.values()),
        "per_class_generated_effective_rank_ratio": _format_class_map(rank_ratios),
        "per_class_generated_effective_rank_ratio_min": min(rank_ratios.values()),
        "per_class_generated_effective_rank_ratio_mean": _mean(rank_ratios.values()),
        "real_vs_generated_mmd_rbf_pca64": _rbf_mmd(synthetic_pca, source_train_pca),
        "class_centroid_shift_norm": _class_centroid_shift_norm(synthetic_pca, synthetic_y, source_train_pca, train_y),
        "within_class_distance_ratio": _within_class_distance_ratio(synthetic_pca, synthetic_y, source_train_pca, train_y),
        "between_class_distance_ratio": _between_class_distance_ratio(synthetic_pca, synthetic_y, source_train_pca, train_y),
        "synthetic_nearest_neighbor_concentration": nn_syn,
        "real_nearest_neighbor_concentration": nn_real,
        "synthetic_nn_concentration_ratio": nn_ratio,
        "base_synthetic_nn_concentration_ratio": math.nan,
        "class_geometry_collapse_warning": 0,
        **decoder_logvar_diagnostics_by_class(model=model, reference_pools=state["reference_pools"]),
    }


def build_g1_locked_selected_alignment(
    single_rows: Sequence[Mapping[str, object]],
    support_units: Sequence[SupportSelectionUnit],
) -> list[dict[str, object]]:
    out = []
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in single_rows:
        if str(row.get("status")) != "ok":
            continue
        key = (
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            int(row["support_size"]),
            int(row["support_seed"]),
            str(row["variant"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
        )
        grouped.setdefault(key, []).append(row)
    base_by_key = {
        (key[0], key[1], key[2], key[3], key[5], key[6]): rows
        for key, rows in grouped.items()
        if key[4] == VARIANT_BASE
    }
    for key, rows in sorted(grouped.items()):
        exp, heldout, support_size, support_seed, variant, generation_seed, classifier_seed = key
        unit = _support_unit(support_units, experiment_seed=exp, heldout_center=heldout, support_size=support_size, support_seed=support_seed)
        selected = str(unit.selected_expert) if unit else ""
        selected_rows = [row for row in rows if str(row["candidate_expert"]) == selected]
        selected_bacc = float(selected_rows[0]["bacc"]) if selected_rows else math.nan
        oracle_row = max(rows, key=lambda row: float(row["bacc"]))
        oracle_bacc = float(oracle_row["bacc"])
        base_rows = base_by_key.get((exp, heldout, support_size, support_seed, generation_seed, classifier_seed), [])
        base_selected = [row for row in base_rows if str(row["candidate_expert"]) == selected]
        base_bacc = float(base_selected[0]["bacc"]) if base_selected else math.nan
        out.append(
            {
                "experiment_seed": exp,
                "heldout_center": heldout,
                "support_size": support_size,
                "support_seed": support_seed,
                "support_eval_split_id": unit.support_eval_split_id if unit else "",
                "variant": variant,
                "generation_seed": generation_seed,
                "classifier_seed": classifier_seed,
                "selected_expert": selected,
                "selected_bacc": selected_bacc,
                "oracle_expert": oracle_row["candidate_expert"],
                "oracle_bacc": oracle_bacc,
                "oracle_gap": oracle_bacc - selected_bacc if not math.isnan(selected_bacc) else math.nan,
                "paired_delta_vs_g1_base_retrain": selected_bacc - base_bacc if not math.isnan(base_bacc) and variant != VARIANT_BASE else math.nan,
                "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                "routing_recomputed_for_g1": 0,
                "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
                "status": "ok",
            }
        )
    return out


def build_g1_summary_rows(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    g1_ensemble_rows: Sequence[Mapping[str, object]],
    augmented_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out = []
    out.extend(_summary_for_alignment(alignment_rows, geometry_rows))
    out.extend(_summary_for_ensemble("g1_geometric_ensemble", g1_ensemble_rows, geometry_rows))
    out.extend(_summary_for_ensemble("g1_c63_augmented_ensemble", augmented_rows, geometry_rows))
    return apply_g1_decision_labels(out)


def build_g1_threshold_rows(summary_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in summary_rows if str(row.get("summary_scope")) != "center"]


def annotate_geometry_collapse_vs_base(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    base_nn: dict[tuple[object, ...], float] = {}
    for row in rows:
        if str(row.get("variant")) == VARIANT_BASE:
            key = (int(row["experiment_seed"]), str(row["heldout_center"]), str(row["candidate_expert"]), int(row["generation_seed"]))
            base_nn[key] = float(row["synthetic_nn_concentration_ratio"])
    out = []
    for row in rows:
        item = dict(row)
        key = (int(item["experiment_seed"]), str(item["heldout_center"]), str(item["candidate_expert"]), int(item["generation_seed"]))
        base_value = base_nn.get(key, math.nan)
        item["base_synthetic_nn_concentration_ratio"] = base_value
        nn_limit = COLLAPSE_NN_CONCENTRATION_ABS_MAX if math.isnan(base_value) else max(COLLAPSE_NN_CONCENTRATION_ABS_MAX, COLLAPSE_NN_CONCENTRATION_BASE_MULTIPLIER * base_value)
        collapse = (
            float(item["per_class_generated_effective_rank_ratio_min"]) < COLLAPSE_EFFECTIVE_RANK_RATIO_MIN
            or float(item["per_class_generated_cov_trace_ratio_min"]) < COLLAPSE_COV_TRACE_RATIO_MIN
            or float(item["per_class_generated_cov_trace_ratio_max"]) > COLLAPSE_COV_TRACE_RATIO_MAX
            or float(item["synthetic_nn_concentration_ratio"]) > nn_limit
        )
        item["class_geometry_collapse_warning"] = int(collapse)
        out.append(item)
    return out


def _summary_for_alignment(
    alignment_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out = []
    for center in sorted({str(row["heldout_center"]) for row in alignment_rows}):
        for variant in sorted({str(row["variant"]) for row in alignment_rows}):
            subset = [row for row in alignment_rows if str(row["heldout_center"]) == center and str(row["variant"]) == variant]
            if subset:
                out.append(_summary_row("center", variant, "locked_selected_single_expert", center, subset, geometry_rows))
    for variant in sorted({str(row["variant"]) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row["variant"]) == variant]
        out.append(_summary_row("locked_selected_single_expert", variant, "locked_selected_single_expert", "ALL", subset, geometry_rows))
    return out


def _summary_for_ensemble(
    scope: str,
    rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out = []
    for center in sorted({str(row["heldout_center"]) for row in rows}):
        for variant in sorted({str(row["variant"]) for row in rows}):
            subset = [row for row in rows if str(row["heldout_center"]) == center and str(row["variant"]) == variant and str(row.get("status")) == "ok"]
            if subset:
                out.append(_summary_row("center", variant, scope, center, subset, geometry_rows))
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if str(row["variant"]) == variant and str(row.get("status")) == "ok"]
        if subset:
            out.append(_summary_row(scope, variant, scope, "ALL", subset, geometry_rows))
    return out


def _summary_row(
    scope: str,
    variant: str,
    policy: str,
    center: str,
    rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    value_key = "selected_bacc" if "selected_bacc" in rows[0] else "bacc"
    baccs = [_float(row.get(value_key)) for row in rows]
    oracle_key = "oracle_bacc" if "oracle_bacc" in rows[0] else "oracle_bacc_reference"
    gap_key = "oracle_gap" if "oracle_gap" in rows[0] else "regret_bacc"
    base_delta = [_float(row.get("paired_delta_vs_g1_base_retrain")) for row in rows if not math.isnan(_float(row.get("paired_delta_vs_g1_base_retrain")))]
    c63_delta = [_float(row.get("paired_delta_vs_c63_replay")) for row in rows if not math.isnan(_float(row.get("paired_delta_vs_c63_replay")))]
    collapse = _geometry_collapse_count(geometry_rows, variant=variant, heldout_center=center if center != "ALL" else None)
    mean_gap = _mean(_float(row.get(gap_key)) for row in rows)
    base_gap = _mean(_float(row.get(gap_key)) for row in rows if str(row.get("variant")) == VARIANT_BASE)
    row = {
        "summary_scope": scope,
        "variant": variant,
        "ensemble_policy": policy,
        "heldout_center": center,
        "n_rows": len(rows),
        "mean_bacc": _mean(baccs),
        "ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in baccs),
        "mean_oracle_bacc": _mean(_float(row.get(oracle_key)) for row in rows),
        "mean_oracle_gap": mean_gap,
        "paired_delta_vs_g1_base_retrain": _mean(base_delta),
        "paired_delta_vs_c63_replay": _mean(c63_delta),
        "positive_paired_delta_rate_vs_base": _mean(1.0 if value > 0 else 0.0 for value in base_delta),
        "strong_center_degrade_gt_002": 0,
        "oracle_gap_delta_vs_base": mean_gap - base_gap if not math.isnan(base_gap) else math.nan,
        "geometry_collapse_count": collapse,
        "decision_label": "",
    }
    row["decision_label"] = _g1_base_decision_label(row)
    return row


def apply_g1_decision_labels(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = [dict(row) for row in rows]
    by_key = {
        (
            str(row.get("summary_scope")),
            str(row.get("ensemble_policy")),
            str(row.get("heldout_center")),
            str(row.get("variant")),
        ): row
        for row in out
    }
    for row in out:
        scope = str(row.get("summary_scope"))
        policy = str(row.get("ensemble_policy"))
        center = str(row.get("heldout_center"))
        variant = str(row.get("variant"))
        base = by_key.get((scope, policy, center, VARIANT_BASE))
        if base is not None and variant != VARIANT_BASE:
            row["oracle_gap_delta_vs_base"] = _float(row.get("mean_oracle_gap")) - _float(base.get("mean_oracle_gap"))
            row["paired_delta_vs_g1_base_retrain"] = _float(row.get("mean_bacc")) - _float(base.get("mean_bacc"))
            row["decision_label"] = _g1_base_decision_label(row)
        if variant == VARIANT_TEACHER_DISTILL_MARGIN and row.get("decision_label") == DECISION_GENERATOR_SUCCESS:
            distill = by_key.get((scope, policy, center, VARIANT_DISTILL))
            if distill is not None and _float(row.get("mean_bacc")) < _float(distill.get("mean_bacc")) - 0.005:
                row["decision_label"] = FAILURE_SIMPLE_DISTILL
            else:
                row["decision_label"] = DECISION_COMBINED_USEFUL
        if scope == "g1_c63_augmented_ensemble" and center == "ALL" and variant != VARIANT_BASE:
            center1 = by_key.get(("center", policy, "1", variant))
            center3 = by_key.get(("center", policy, "3", variant))
            weak_center_values = [
                value
                for value in (
                    _float(center1.get("paired_delta_vs_c63_replay")) if center1 else math.nan,
                    _float(center3.get("paired_delta_vs_c63_replay")) if center3 else math.nan,
                )
                if not math.isnan(value)
            ]
            weak_center_gain = max(weak_center_values) if weak_center_values else math.nan
            if (
                _float(row.get("mean_bacc")) >= 0.80
                and _float(row.get("paired_delta_vs_c63_replay")) > 0.0
                and weak_center_gain >= 0.02
            ):
                row["decision_label"] = DECISION_THESIS_PROGRESS
    return out


def _g1_base_decision_label(row: Mapping[str, object]) -> str:
    variant = str(row.get("variant"))
    if variant == VARIANT_BASE:
        return "DIAGNOSTIC_CONTROL"
    if int(row.get("geometry_collapse_count", 0) or 0) > 0:
        return FAILURE_COLLAPSE
    delta = _float(row.get("paired_delta_vs_g1_base_retrain"))
    positive_rate = _float(row.get("positive_paired_delta_rate_vs_base"))
    gap_delta = _float(row.get("oracle_gap_delta_vs_base"))
    if not math.isnan(delta) and delta >= 0.01 and positive_rate >= 0.60 and (math.isnan(gap_delta) or gap_delta <= 0.0):
        return DECISION_GENERATOR_SUCCESS
    if not math.isnan(delta) and delta > 0:
        return FAILURE_SINGLE_NOT_ENSEMBLE if "ensemble" in str(row.get("summary_scope")) else FAILURE_SOURCE_GEOMETRY
    return FAILURE_NO_GAIN


def _source_val_decoder_geometry(
    *,
    model: Any,
    source_probe: SourceProbe,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    base_geometry: Mapping[str, float] | None,
) -> dict[str, float]:
    with torch.no_grad():
        xb = val_x.float().to(device)
        yb = val_y.long().to(device)
        mu_z, _logvar_z = model.encode(xb, y=yb)
        decoded, _ = model.decode(mu_z, y=yb, return_distribution=True)
    decoded_cpu = decoded.detach().cpu().float()
    train_cpu = train_x.detach().cpu().float()
    train_y_cpu = train_y.detach().cpu().long()
    val_y_cpu = val_y.detach().cpu().long()
    ratios = _class_cov_trace_ratios(decoded_cpu, val_y_cpu, train_cpu, train_y_cpu)
    rank_ratios = _class_effective_rank_ratios(decoded_cpu, val_y_cpu, train_cpu, train_y_cpu)
    nn_syn = _nearest_neighbor_concentration(decoded_cpu, train_cpu)
    nn_real = _nearest_neighbor_concentration(val_x.detach().cpu().float(), train_cpu)
    nn_ratio = nn_syn / max(nn_real, 1.0e-12)
    base_nn = float((base_geometry or {}).get("synthetic_nn_concentration_ratio", math.nan))
    nn_limit = COLLAPSE_NN_CONCENTRATION_ABS_MAX if math.isnan(base_nn) else max(COLLAPSE_NN_CONCENTRATION_ABS_MAX, COLLAPSE_NN_CONCENTRATION_BASE_MULTIPLIER * base_nn)
    collapse = (
        min(rank_ratios.values()) < COLLAPSE_EFFECTIVE_RANK_RATIO_MIN
        or min(ratios.values()) < COLLAPSE_COV_TRACE_RATIO_MIN
        or max(ratios.values()) > COLLAPSE_COV_TRACE_RATIO_MAX
        or nn_ratio > nn_limit
    )
    return {
        "geometry_collapse_warning": float(int(collapse)),
        "effective_rank_ratio_min": float(min(rank_ratios.values())),
        "cov_trace_ratio_min": float(min(ratios.values())),
        "cov_trace_ratio_max": float(max(ratios.values())),
        "synthetic_nn_concentration_ratio": float(nn_ratio),
    }


def _class_centroids(x: torch.Tensor, y: torch.Tensor, *, device: torch.device) -> dict[int, torch.Tensor]:
    out = {}
    for label in GLOBAL_CLASS_ORDER:
        subset = x[y.long() == int(label)]
        if int(subset.shape[0]) == 0:
            raise ProtocolError(f"G1 cannot compute source centroid for empty label {label}.")
        out[int(label)] = subset.float().to(device).mean(dim=0)
    return out


def _c63_safe_members(
    *,
    cache: _GenerationCache,
    candidates: Sequence[str],
    generation_seeds: Sequence[int],
    total_budget_per_class: int,
) -> tuple[Any, ...]:
    plans = build_c63_ensemble_plans(
        policies=(POLICY_GEOM_SAFE_MULTI,),
        candidates=candidates,
        total_budget_per_class=int(total_budget_per_class),
        generation_seeds=tuple(int(seed) for seed in generation_seeds),
        support_rows=(),
    )
    if not plans:
        return ()
    return tuple(_generate_c62_member(cache=cache, spec=spec, label_values=GLOBAL_CLASS_ORDER) for spec in plans[0].specs)


def _member_payload(member: Any) -> tuple[torch.Tensor, tuple[int, ...], float]:
    if isinstance(member, G1Member):
        return member.synthetic_dino, member.synthetic_labels, float(member.weight)
    return member.generated.synthetic_dino, member.generated.synthetic_labels, float(member.spec.weight)


def _member_key(member: Any) -> str:
    if isinstance(member, G1Member):
        return member.member_key
    return member.spec.member_key


def _fit_or_load_g1_projection(
    *,
    artifacts_root: Path,
    train_cache: Any,
    candidate_expert: str,
    seed: int,
    n_components: int,
    resume: bool,
) -> SourceTrainPCAProjection:
    path = _g1_projection_path(artifacts_root, seed, candidate_expert)
    if resume and path.exists():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    projection = fit_source_train_pca_projection(
        train_embeddings=train_cache.embeddings,
        train_metadata=train_cache.metadata,
        source_domain=candidate_expert,
        seed=int(seed),
        n_components=int(n_components),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projection, path)
    return projection


def _g1_projection_path(root: Path, seed: int, candidate_expert: str) -> Path:
    return root / "projections" / f"seed{int(seed)}" / f"expert_{candidate_expert}" / "pca64.pt"


def _load_g1_model(checkpoint_path: Path, *, device: torch.device) -> Any:
    _ensure_cvae_testing_path(Path.cwd())
    from src.models.cvae_expert import build_cvae_from_metadata  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

    loaded = load_model_checkpoint(checkpoint_path, map_location=device)
    model = build_cvae_from_metadata(loaded.checkpoint_metadata).to(device)
    model.load_state_dict(loaded.model_state_dict)
    model.eval()
    return model


def _filtered_support_conditions(
    units: Sequence[SupportSelectionUnit],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_sizes: Sequence[int] | None,
    support_seeds: Sequence[int] | None,
) -> tuple[tuple[int, int], ...]:
    pairs = _support_conditions(units, experiment_seed=experiment_seed, heldout_center=heldout_center)
    if support_sizes is not None:
        allowed_sizes = {int(v) for v in support_sizes}
        pairs = tuple(pair for pair in pairs if int(pair[0]) in allowed_sizes)
    if support_seeds is not None:
        allowed_seeds = {int(v) for v in support_seeds}
        pairs = tuple(pair for pair in pairs if int(pair[1]) in allowed_seeds)
    return tuple((int(a), int(b)) for a, b in pairs)


def _support_unit(
    units: Sequence[SupportSelectionUnit],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> SupportSelectionUnit | None:
    for unit in units:
        if (
            int(unit.experiment_seed) == int(experiment_seed)
            and str(unit.heldout_center) == str(heldout_center)
            and int(unit.support_size) == int(support_size)
            and int(unit.support_seed) == int(support_seed)
            and unit.method == SUPPORT_NELBO_METHOD
        ):
            return unit
    return None


def _load_c63_reference(root: Path) -> dict[tuple[int, str, int, int, int], Mapping[str, object]]:
    path = root / "tables" / "c63_geometric_late_ensemble_downstream_matrix.csv"
    if not path.exists():
        return {}
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        if str(row.get("ensemble_policy")) != POLICY_GEOM_SAFE_MULTI or str(row.get("status")) != "ok":
            continue
        key = (
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            int(row["support_size"]),
            int(row["support_seed"]),
            int(row["classifier_seed"]),
        )
        out[key] = row
    return out


def _c63_bacc_by_classifier(
    rows: Mapping[tuple[int, str, int, int, int], Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> dict[int, float]:
    out = {}
    for key, row in rows.items():
        exp, center, size, seed, classifier_seed = key
        if exp == int(experiment_seed) and center == str(heldout_center) and size == int(support_size) and seed == int(support_seed):
            out[int(classifier_seed)] = _float(row.get("bacc"))
    return out


def _c63_oracle_reference(
    rows: Mapping[tuple[int, str, int, int, int], Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> float:
    vals = [
        _float(row.get("oracle_bacc_reference"))
        for key, row in rows.items()
        if key[0] == int(experiment_seed)
        and key[1] == str(heldout_center)
        and key[2] == int(support_size)
        and key[3] == int(support_seed)
    ]
    return _mean(vals)


def _selected_nelbo_from_history(rows: Sequence[Mapping[str, object]]) -> float:
    selected = [row for row in rows if int(float(row.get("checkpoint_selected", 0) or 0)) == 1]
    row = selected[-1] if selected else (rows[-1] if rows else {})
    return _float(row.get("val_nelbo_raw"))


def _selected_geometry_from_history(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    selected = [row for row in rows if int(float(row.get("checkpoint_selected", 0) or 0)) == 1]
    row = selected[-1] if selected else (rows[-1] if rows else {})
    return {
        "effective_rank_ratio_min": _float(row.get("effective_rank_ratio_min")),
        "cov_trace_ratio_min": _float(row.get("cov_trace_ratio_min")),
        "cov_trace_ratio_max": _float(row.get("cov_trace_ratio_max")),
        "synthetic_nn_concentration_ratio": _float(row.get("synthetic_nn_concentration_ratio")),
    }


def _validate_variants(variants: Sequence[str]) -> None:
    unknown = sorted(set(str(v) for v in variants).difference(VARIANT_SPECS))
    if unknown:
        raise ProtocolError(f"Unknown G1 variants requested: {unknown}")
    if VARIANT_BASE not in set(variants):
        raise ProtocolError("G1 variants must include G1_base_retrain.")


def _variant_slug(variant: str) -> str:
    return str(variant).lower().replace(".", "").replace(" ", "_")


def _variant_seed_offset(variant: str) -> int:
    return {
        VARIANT_BASE: 0,
        VARIANT_PROBE_CE: 101,
        VARIANT_DISTILL: 211,
        VARIANT_DISTILL_MARGIN: 307,
        VARIANT_TEACHER_DISTILL_MARGIN: 401,
    }[variant]


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _np_stack(items: Sequence[object]) -> Any:
    import numpy as np  # type: ignore

    return np.stack([np.asarray(item, dtype=float) for item in items], axis=0)


def _np_argmax(scores: object) -> list[int]:
    import numpy as np  # type: ignore

    return np.argmax(np.asarray(scores, dtype=float), axis=1).tolist()


def _format_class_map(values: Mapping[int, float]) -> str:
    return "|".join(f"{int(key)}:{float(value):.6g}" for key, value in sorted(values.items()))


def _geometry_collapse_count(
    rows: Sequence[Mapping[str, object]],
    *,
    variant: str,
    heldout_center: str | None,
) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("variant")) == str(variant)
        and (heldout_center is None or str(row.get("heldout_center")) == str(heldout_center))
        and int(row.get("class_geometry_collapse_warning", 0) or 0) == 1
    )


def _mean(values: Iterable[float]) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    return sum(cleaned) / float(len(cleaned)) if cleaned else math.nan


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out
