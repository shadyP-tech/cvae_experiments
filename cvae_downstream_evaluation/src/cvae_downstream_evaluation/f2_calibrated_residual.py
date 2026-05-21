"""F2 source-anchored calibrated residual CVAE downstream diagnostic.

F2 is a generator-repair experiment. It keeps locked C4.1 support-NELBO
selected experts fixed, reuses C4.1 source-train PCA64 projections, and changes
only source-local residual generation. Source-val is used only for frozen
source-side calibration.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .c41_heteroscedastic import SourceTrainPCAProjection, _generator_for_device, _randn_like
from .c41_workstation import (
    C41TrainingProfile,
    _indices_for_domain,
    _support_conditions,
    _write_csv,
    _write_dict_csv,
    discover_c41_run_artifacts,
    safe_support_selection_units_from_paths,
)
from .downstream import (
    CandidateDownstreamRow,
    balanced_accuracy,
    fit_locked_logistic_classifier,
    macro_f1,
    read_candidate_downstream_matrix,
)
from .f1_source_anchored import (
    ANCHOR_NEIGHBOR_K,
    NEAR_DUPLICATE_EPS,
    TRAIN_PAIRS_PER_SAMPLE,
    VAL_PAIRS_PER_SAMPLE,
    AnchorPairDataset,
    AnchoredResidualCVAE,
    SourceAnchorIndex,
    _aggregate_float_dicts,
    _anchor_pair_diagnostics,
    _as_float,
    _c41_projection_path,
    _generated_distribution_diagnostics,
    _limit_c41_artifacts,
    _load_c41_projection,
    _mean,
    _nanmean,
    _pairwise_distance_mean,
    _reverse_lex,
    _sample_generation_pairs,
    _subset,
    _trace_cov,
    _write_csv_with_header,
    build_anchor_pair_dataset,
    build_source_anchor_index,
    kl_latent_dim_mean,
    residual_gaussian_nll_terms,
)
from .matrix import (
    MatrixBuildLimits,
    TargetEvalPool,
    _label,
    _load_embedding_cache,
    _read_completed_keys,
    _read_samples_manifest,
    _records_for_split,
    _resolve_torch_device,
    _to_numpy,
    append_matrix_row,
    build_target_eval_pool,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    BASELINE_ROUTING_FAMILY_USED,
    BASELINE_SELECTED_EXPERT_IDS_SOURCE,
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    PRIMARY_BUDGET_PER_CLASS,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


F2_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/f2_source_anchored_calibrated_residual_cvae_v1"
F2_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
F2_DEFAULT_F1_ROOT = "cvae_downstream_evaluation/artifacts/f1_source_anchored_residual_cvae_v1"
F2_DEFAULT_C61_ROOT = "cvae_downstream_evaluation/artifacts/c61_cvae_mixture_downstream_v1"
F2_GENERATOR_FAMILY = "family_f2_pca64_class_conditional_source_anchored_calibrated_residual_cvae_downstream_v1"

F2_MODE_CALIBRATED_NOISE = "anchor_posterior_residual_calibrated_noise"
F2_MODE_UNCALIBRATED_NOISE = "anchor_posterior_residual_uncalibrated_noise"
F2_MODE_GLOBAL_CALIBRATED_NOISE = "anchor_posterior_residual_global_calibrated_noise"
F2_MODE_CALIBRATED_MEAN = "anchor_posterior_residual_calibrated_mean"
F2_MODE_PRIOR_CALIBRATED_NOISE = "anchor_prior_residual_calibrated_noise"
F2_MODE_UNCALIBRATED_NOISE_NO_PENALTY = "anchor_posterior_residual_uncalibrated_noise_no_energy_cov_penalty"
F2_MODE_CALIBRATED_NOISE_NO_PENALTY = "anchor_posterior_residual_calibrated_noise_no_energy_cov_penalty"
F2_MODE_IDENTITY_BOOTSTRAP = "anchor_identity_bootstrap"
F2_MODE_EMPIRICAL_BOOTSTRAP = "anchor_residual_empirical_bootstrap"
F2_MODE_TRANSFER_BOOTSTRAP = "anchor_empirical_residual_transfer_bootstrap"

F2_GENERATION_MODES = (
    F2_MODE_CALIBRATED_NOISE,
    F2_MODE_UNCALIBRATED_NOISE,
    F2_MODE_GLOBAL_CALIBRATED_NOISE,
    F2_MODE_CALIBRATED_MEAN,
    F2_MODE_PRIOR_CALIBRATED_NOISE,
    F2_MODE_UNCALIBRATED_NOISE_NO_PENALTY,
    F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
    F2_MODE_IDENTITY_BOOTSTRAP,
    F2_MODE_EMPIRICAL_BOOTSTRAP,
    F2_MODE_TRANSFER_BOOTSTRAP,
)
F2_DIAGNOSTIC_ONLY_MODES = tuple(mode for mode in F2_GENERATION_MODES if mode != F2_MODE_CALIBRATED_NOISE)
F2_NO_PENALTY_MODES = {F2_MODE_UNCALIBRATED_NOISE_NO_PENALTY, F2_MODE_CALIBRATED_NOISE_NO_PENALTY}
F2_BOOTSTRAP_MODES = {F2_MODE_IDENTITY_BOOTSTRAP, F2_MODE_EMPIRICAL_BOOTSTRAP, F2_MODE_TRANSFER_BOOTSTRAP}

CALIBRATION_MIN_VAL_PAIRS = 16
CALIBRATION_MIN_SCALE = 0.5
CALIBRATION_MAX_SCALE = 2.5
TOP5_NN_SHARE_FAILURE_THRESHOLD = 0.30
MEDIAN_NN_COPY_RATIO_THRESHOLD = 0.25

DECISION_REPAIR_SUCCESS = "F2_GENERATOR_REPAIR_SUCCESS"
DECISION_SUPERIORITY_SUCCESS = "F2_GENERATOR_SUPERIORITY_SUCCESS"
DECISION_THESIS_SUCCESS = "F2_THESIS_SUCCESS"
DECISION_CALIBRATION_NO_GAIN = "RESIDUAL_CALIBRATION_NO_GAIN"
DECISION_MOMENTS_ONLY = "RESIDUAL_MOMENTS_IMPROVE_BACC_DOES_NOT"
DECISION_UNDERDISPERSION = "RESIDUAL_UNDERDISPERSION"
DECISION_OVERDISPERSION = "RESIDUAL_OVERDISPERSION"
DECISION_BOOTSTRAP_STRONGER = "BOOTSTRAP_STILL_STRONGER"
DECISION_NEAR_COPY = "NEAR_COPY_FAILURE"
DECISION_ENSEMBLE_REQUIRED = "ENSEMBLE_REQUIRED_FOR_080"
DECISION_PROTOCOL_FAILURE = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"

F2_ALIGNMENT_COLUMNS = (
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "generator_family",
    "generation_mode",
    "generation_seed",
    "classifier_seed",
    "method",
    "selected_expert",
    "selected_bacc",
    "selected_macro_f1",
    "downstream_oracle_expert",
    "oracle_bacc",
    "oracle_macro_f1",
    "downstream_oracle_gap_bacc",
    "downstream_oracle_gap_macro_f1",
    "relative_downstream_oracle_gap_pct",
    "top1_downstream_hit",
    "spearman_neg_nelbo_vs_bacc",
    "metadata_bacc",
    "delta_vs_metadata",
    "selection_depends_on_support",
    "routing_family_used",
    "routing_scores_recomputed_for_f2",
    "selected_expert_ids_source",
    "projection_source",
    "generation_conditioning",
)

F2_DELTA_COLUMNS = (
    "heldout_center",
    "support_size",
    "generation_mode",
    "selected_bacc_f2",
    "oracle_bacc_f2",
    "oracle_gap_f2",
    "selected_bacc_f1_posterior_residual_mean",
    "oracle_bacc_f1_posterior_residual_mean",
    "oracle_gap_f1_posterior_residual_mean",
    "selected_bacc_delta_vs_f1_posterior_residual_mean",
    "oracle_bacc_delta_vs_f1_posterior_residual_mean",
    "oracle_gap_delta_vs_f1_posterior_residual_mean",
    "selected_bacc_anchor_identity_bootstrap",
    "selected_bacc_anchor_residual_empirical_bootstrap",
    "selected_bacc_anchor_empirical_residual_transfer_bootstrap",
    "beats_identity_bootstrap",
    "beats_empirical_bootstrap",
    "beats_transfer_bootstrap",
    "selected_ge_080",
    "median_center_seed_delta_gt_0",
    "center_seed_improvement_rate",
    "near_copy_failure",
    "residual_moment_improved_vs_f1_mean",
    "diagnostic_only",
    "decision_label",
)


@dataclass(frozen=True)
class ResidualCalibration:
    class_scales: Mapping[int, float]
    global_scale: float
    rows: tuple[dict[str, object], ...]

    def scale_for(self, class_label: int, *, global_only: bool = False) -> float:
        if global_only:
            return float(self.global_scale)
        return float(self.class_scales.get(int(class_label), self.global_scale))


@dataclass(frozen=True)
class F2GeneratedBatch:
    embeddings: torch.Tensor
    labels: torch.Tensor
    generation_mode: str
    diagnostics: Mapping[str, float]
    provenance_rows: tuple[dict[str, object], ...]


def anchored_residual_loss_terms_f2(
    *,
    delta_mu: torch.Tensor,
    delta_true: torch.Tensor,
    delta_logvar: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.25,
    energy_weight: float = 0.5,
    cov_trace_weight: float = 0.25,
    use_moment_penalties: bool = True,
) -> dict[str, torch.Tensor]:
    recon_terms = residual_gaussian_nll_terms(
        delta_mu=delta_mu,
        delta_true=delta_true,
        delta_logvar=delta_logvar,
    )
    kl = kl_latent_dim_mean(mu, logvar).mean()
    recon = recon_terms["recon_nll"].mean()
    if use_moment_penalties:
        gen_delta = delta_mu + torch.exp(0.5 * delta_logvar) * torch.randn_like(delta_mu)
        energy_penalty = _log_ratio_square(
            gen_delta.pow(2).sum(dim=1).mean(),
            delta_true.pow(2).sum(dim=1).mean(),
        )
        cov_penalty = _log_ratio_square(_trace_cov_tensor(gen_delta), _trace_cov_tensor(delta_true))
    else:
        energy_penalty = torch.zeros((), dtype=delta_mu.dtype, device=delta_mu.device)
        cov_penalty = torch.zeros((), dtype=delta_mu.dtype, device=delta_mu.device)
    loss = recon + (float(beta) * kl) + (float(energy_weight) * energy_penalty) + (float(cov_trace_weight) * cov_penalty)
    return {
        "loss": loss,
        "recon_nll": recon,
        "logvar_term": recon_terms["logvar_term"].mean(),
        "squared_error_scaled": recon_terms["squared_error_scaled"].mean(),
        "kl": kl,
        "residual_energy_penalty": energy_penalty,
        "residual_cov_trace_penalty": cov_penalty,
    }


def train_f2_anchored_residual_cvae(
    *,
    train_pairs: AnchorPairDataset,
    val_pairs: AnchorPairDataset,
    out_dir: Path,
    model_name: str,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    resume: bool,
    use_moment_penalties: bool,
    checkpoint_metadata: Mapping[str, object],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_name}.pt"
    if ckpt.exists() and resume:
        return ckpt
    if ckpt.exists() and not resume:
        raise ProtocolError(f"F2 checkpoint already exists; use --resume or a clean artifact root: {ckpt}")

    torch_device = _resolve_torch_device(torch, device)
    model = AnchoredResidualCVAE(
        input_dim=int(train_pairs.pair_targets.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        class_condition_dim=2,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_pairs.pair_targets, train_pairs.anchors, train_pairs.labels),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_pairs.pair_targets, val_pairs.anchors, val_pairs.labels),
        batch_size=int(batch_size),
        shuffle=False,
    )
    best_val = float("inf")
    bad_epochs = 0
    history: dict[str, list[float]] = {"train": [], "val": []}
    for _epoch in range(int(epochs)):
        train_loss = _run_f2_epoch(model, train_loader, torch_device, optimizer=optimizer, use_moment_penalties=use_moment_penalties)
        val_loss = _run_f2_epoch(model, val_loader, torch_device, optimizer=None, use_moment_penalties=use_moment_penalties)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "checkpoint_metadata": {
                        **dict(checkpoint_metadata),
                        "model_class": "AnchoredResidualCVAE",
                        "input_dim": int(train_pairs.pair_targets.shape[1]),
                        "hidden_dim": int(hidden_dim),
                        "latent_dim": int(latent_dim),
                        "class_condition_dim": 2,
                        "decoder_likelihood": "gaussian_diag_residual",
                        "decoder_logvar_min": -9.21,
                        "decoder_logvar_max": 2.0,
                        "decoder_min_variance": 1.0e-4,
                        "recon_reduction": "dim_mean",
                        "kl_reduction": "latent_dim_mean",
                        "beta_effective": 0.25,
                        "energy_weight": 0.5 if use_moment_penalties else 0.0,
                        "cov_trace_weight": 0.25 if use_moment_penalties else 0.0,
                        "use_moment_penalties": int(bool(use_moment_penalties)),
                        "history": history,
                        "best_val": best_val,
                    },
                },
                ckpt,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    return ckpt


def fit_residual_calibration(
    *,
    model: AnchoredResidualCVAE,
    val_pairs: AnchorPairDataset,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    model_variant: str,
    device: str,
) -> ResidualCalibration:
    torch_device = _resolve_torch_device(torch, device)
    model = model.to(torch_device)
    model.eval()
    x_source_residual_ref = val_pairs.pair_targets.to(torch_device)
    x_anchor = val_pairs.anchors.to(torch_device)
    y = val_pairs.labels.to(torch_device)
    with torch.no_grad():
        mu_z, _ = model.encode(x_source_residual_ref, x_anchor, y)
        delta_mu, delta_logvar = model.decode_residual(mu_z, x_anchor, y)
        gen = _generator_for_device(torch_device, int(experiment_seed) + 4909)
        synthetic_delta = delta_mu + torch.exp(0.5 * delta_logvar) * _randn_like(delta_mu, generator=gen)
    true_delta = x_source_residual_ref - x_anchor
    rows: list[dict[str, object]] = []
    class_scales: dict[int, float] = {}
    global_stats = _scale_stats(true_delta.detach().cpu(), synthetic_delta.detach().cpu())
    global_scale = _clip_scale(global_stats["scale_geomean"])
    for class_label in sorted(set(int(v) for v in val_pairs.labels.tolist())):
        mask = val_pairs.labels == int(class_label)
        n_val = int(mask.sum().item())
        if n_val >= CALIBRATION_MIN_VAL_PAIRS:
            stats = _scale_stats(true_delta.detach().cpu()[mask], synthetic_delta.detach().cpu()[mask])
            fallback = 0
            raw_scale = stats["scale_geomean"]
        else:
            stats = dict(global_stats)
            fallback = 1
            raw_scale = global_stats["scale_geomean"]
        scale = _clip_scale(raw_scale)
        class_scales[int(class_label)] = scale
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "candidate_expert": candidate_expert,
                "model_variant": model_variant,
                "class_label": int(class_label),
                "scale_norm": stats["scale_norm"],
                "scale_cov_trace": stats["scale_cov_trace"],
                "scale_geomean": scale,
                "scale_clipped_flag": int(scale != float(raw_scale)),
                "n_val_pairs": n_val,
                "fallback_used": fallback,
                "source_val_moment_improved": int(_moment_error(scale, stats) < _moment_error(1.0, stats)),
                "calibration_split": "source_val",
            }
        )
    return ResidualCalibration(class_scales=class_scales, global_scale=global_scale, rows=tuple(rows))


def generate_f2_anchor_residual_embeddings(
    *,
    model: AnchoredResidualCVAE,
    anchor_index: SourceAnchorIndex,
    calibration: ResidualCalibration,
    class_label: int,
    n_samples: int,
    seed: int,
    generation_mode: str,
    experiment_seed: int = 0,
    heldout_center: str = "",
    candidate_expert: str = "",
    support_size: int = 0,
    support_seed: int = 0,
) -> F2GeneratedBatch:
    if generation_mode not in F2_GENERATION_MODES:
        raise ProtocolError(f"Unknown F2 generation mode: {generation_mode}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")

    device = next(model.parameters()).device
    pair = _sample_generation_pairs(anchor_index, int(class_label), int(n_samples), int(seed), transfer=False)
    transfer_pair = (
        _sample_generation_pairs(anchor_index, int(class_label), int(n_samples), int(seed) + 7919, transfer=True)
        if generation_mode == F2_MODE_TRANSFER_BOOTSTRAP
        else None
    )
    x_source_residual_ref = pair["pair_targets"].to(device)
    x_anchor = pair["anchors"].to(device)
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)
    decoder_noise = torch.zeros_like(x_anchor)
    delta_logvar = torch.zeros_like(x_anchor)
    reference_delta = x_source_residual_ref - x_anchor
    posterior_or_prior_source = "source_train_residual_reference_posterior"
    provenance_pair = pair
    scale_used = 1.0

    with torch.no_grad():
        if generation_mode in {
            F2_MODE_CALIBRATED_NOISE,
            F2_MODE_UNCALIBRATED_NOISE,
            F2_MODE_GLOBAL_CALIBRATED_NOISE,
            F2_MODE_CALIBRATED_MEAN,
            F2_MODE_UNCALIBRATED_NOISE_NO_PENALTY,
            F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
        }:
            mu_z, _ = model.encode(x_source_residual_ref, x_anchor, y)
            delta_mu, delta_logvar = model.decode_residual(mu_z, x_anchor, y)
            if generation_mode in {F2_MODE_CALIBRATED_NOISE, F2_MODE_CALIBRATED_NOISE_NO_PENALTY}:
                scale_used = calibration.scale_for(int(class_label))
            elif generation_mode == F2_MODE_GLOBAL_CALIBRATED_NOISE:
                scale_used = calibration.scale_for(int(class_label), global_only=True)
            else:
                scale_used = 1.0
            if generation_mode != F2_MODE_CALIBRATED_MEAN:
                gen = _generator_for_device(device, int(seed) + 104729)
                decoder_noise = torch.exp(0.5 * delta_logvar) * _randn_like(delta_mu, generator=gen)
            embeddings = x_anchor + (float(scale_used) * (delta_mu + decoder_noise))
        elif generation_mode == F2_MODE_PRIOR_CALIBRATED_NOISE:
            gen = _generator_for_device(device, int(seed) + 209759)
            z = torch.randn((int(n_samples), int(model.latent_dim)), generator=gen, device=device, dtype=x_anchor.dtype)
            delta_mu, delta_logvar = model.decode_residual(z, x_anchor, y)
            noise_gen = _generator_for_device(device, int(seed) + 104729)
            decoder_noise = torch.exp(0.5 * delta_logvar) * _randn_like(delta_mu, generator=noise_gen)
            scale_used = calibration.scale_for(int(class_label))
            embeddings = x_anchor + (float(scale_used) * (delta_mu + decoder_noise))
            posterior_or_prior_source = "standard_normal_prior"
        elif generation_mode == F2_MODE_IDENTITY_BOOTSTRAP:
            embeddings = x_anchor
            posterior_or_prior_source = "identity_source_train_anchor"
        elif generation_mode == F2_MODE_EMPIRICAL_BOOTSTRAP:
            embeddings = x_anchor + reference_delta
            posterior_or_prior_source = "empirical_paired_source_train_residual"
        elif generation_mode == F2_MODE_TRANSFER_BOOTSTRAP:
            assert transfer_pair is not None
            transfer_ref = transfer_pair["pair_targets"].to(device)
            transfer_anchor = transfer_pair["anchors"].to(device)
            x_anchor = transfer_pair["anchor_a"].to(device)
            x_source_residual_ref = transfer_ref
            reference_delta = transfer_ref - transfer_anchor
            embeddings = x_anchor + reference_delta
            posterior_or_prior_source = "empirical_transfer_source_train_residual"
            provenance_pair = transfer_pair
        else:
            raise ProtocolError(f"Unhandled F2 generation mode: {generation_mode}")

    embeddings_cpu = embeddings.detach().cpu().float()
    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    diagnostics = _f2_generation_diagnostics(
        embeddings=embeddings_cpu,
        anchors=x_anchor.detach().cpu().float(),
        source_refs=x_source_residual_ref.detach().cpu().float(),
        reference_delta=reference_delta.detach().cpu().float(),
        delta_logvar=delta_logvar.detach().cpu().float(),
        decoder_noise=decoder_noise.detach().cpu().float(),
        anchor_ids=tuple(str(v) for v in provenance_pair["anchor_ids"]),
        scale_used=scale_used,
    )
    provenance_rows = _f2_provenance_rows(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        support_size=support_size,
        support_seed=support_seed,
        generation_mode=generation_mode,
        generation_seed=seed,
        class_label=class_label,
        pair=provenance_pair,
        posterior_or_prior_source=posterior_or_prior_source,
        scale_used=scale_used,
    )
    return F2GeneratedBatch(
        embeddings=embeddings_cpu,
        labels=labels,
        generation_mode=generation_mode,
        diagnostics=diagnostics,
        provenance_rows=tuple(provenance_rows),
    )


def build_f2_downstream_matrix(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    training_profile: C41TrainingProfile,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
) -> Path:
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed_keys(matrix_path) if resume else set()
    artifacts = _limit_c41_artifacts(discover_c41_run_artifacts(config=config, repo_root=repo_root), limits.experiment_seeds)
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)

    anchor_diag_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    generator_diag_rows: list[dict[str, object]] = []
    residual_moment_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    sample_provenance_rows: list[dict[str, object]] = []
    late_ensemble_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        support = artifact.support
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        val_records = _records_for_split(samples, "val")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        val_cache = _load_embedding_cache(artifact.val_cache, val_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)

        for heldout_center in selected_heldout_centers:
            heldout = str(heldout_center)
            if heldout not in {str(c) for c in config.candidate_domains}:
                raise ProtocolError(f"Unknown heldout center requested: {heldout}")
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions(support_units, experiment_seed=support.experiment_seed, heldout_center=heldout)
            if not support_conditions:
                raise ProtocolError(f"No locked support-selection conditions for seed={support.experiment_seed}, heldout={heldout}.")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            label_values = tuple(sorted(set(target_labels).union({0, 1})))
            if label_values != (0, 1):
                raise ProtocolError(f"F2 expects binary labels 0/1, got {label_values}")
            late_context: dict[tuple[int, int], list[object]] = {}

            for candidate in candidates:
                projection = _load_c41_projection(c41_artifacts_root, support.experiment_seed, candidate)
                train_projected_all = projection.transform(train_cache.embeddings)
                val_projected_all = projection.transform(val_cache.embeddings)
                candidate_train_idx = _indices_for_domain(train_cache.metadata, candidate)
                candidate_val_idx = _indices_for_domain(val_cache.metadata, candidate)
                if not candidate_train_idx or not candidate_val_idx:
                    raise ProtocolError(f"F2 requires nonempty source train/val rows for candidate={candidate}.")
                source_anchor_index = build_source_anchor_index(
                    source_projected_embeddings=train_projected_all,
                    source_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                    neighbor_k=ANCHOR_NEIGHBOR_K,
                )
                train_pairs = build_anchor_pair_dataset(
                    pair_projected_embeddings=train_projected_all,
                    pair_metadata=train_cache.metadata,
                    anchor_projected_embeddings=train_projected_all,
                    anchor_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                    pairs_per_sample=TRAIN_PAIRS_PER_SAMPLE,
                    neighbor_k=ANCHOR_NEIGHBOR_K,
                    seed=int(support.experiment_seed) + int(candidate),
                    pair_split="source_train",
                    anchor_split="source_train",
                )
                val_pairs = build_anchor_pair_dataset(
                    pair_projected_embeddings=val_projected_all,
                    pair_metadata=val_cache.metadata,
                    anchor_projected_embeddings=train_projected_all,
                    anchor_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                    pairs_per_sample=VAL_PAIRS_PER_SAMPLE,
                    neighbor_k=ANCHOR_NEIGHBOR_K,
                    seed=int(support.experiment_seed) + int(candidate) + 17,
                    pair_split="source_val",
                    anchor_split="source_train",
                )
                anchor_diag_rows.append(
                    _anchor_pair_diagnostics(
                        experiment_seed=support.experiment_seed,
                        heldout_center=heldout,
                        candidate_expert=candidate,
                        train_pairs=train_pairs,
                        val_pairs=val_pairs,
                        source_anchor_index=source_anchor_index,
                    )
                )
                models: dict[str, tuple[Path, AnchoredResidualCVAE, ResidualCalibration]] = {}
                for variant, use_penalties in (("penalty", True), ("no_energy_cov_penalty", False)):
                    ckpt = train_f2_anchored_residual_cvae(
                        train_pairs=train_pairs,
                        val_pairs=val_pairs,
                        out_dir=artifacts_root / "checkpoints" / f"seed{int(support.experiment_seed)}" / f"expert_{candidate}" / f"f2_{variant}",
                        model_name=f"f2_source_anchored_calibrated_residual_{variant}_pca64",
                        hidden_dim=training_profile.hidden_dim,
                        latent_dim=training_profile.latent_dim,
                        lr=training_profile.lr,
                        epochs=training_profile.epochs,
                        patience=training_profile.patience,
                        batch_size=training_profile.batch_size,
                        device=device,
                        resume=resume,
                        use_moment_penalties=use_penalties,
                        checkpoint_metadata={
                            "generator_family": F2_GENERATOR_FAMILY,
                            "experiment_id": "F2",
                            "experiment_seed": int(support.experiment_seed),
                            "candidate_expert": str(candidate),
                            "model_variant": variant,
                            "anchor_strategy": "source_train_same_class_nn",
                            "anchor_neighbor_k": ANCHOR_NEIGHBOR_K,
                            "train_pairs_per_sample": TRAIN_PAIRS_PER_SAMPLE,
                            "projection_source": "reused_c41_full_source_train_pca64",
                        },
                    )
                    model = _load_f2_model(ckpt, device=device)
                    calibration = fit_residual_calibration(
                        model=model,
                        val_pairs=val_pairs,
                        experiment_seed=int(support.experiment_seed),
                        heldout_center=heldout,
                        candidate_expert=candidate,
                        model_variant=variant,
                        device=device,
                    )
                    calibration_rows.extend(calibration.rows)
                    models[variant] = (ckpt, model, calibration)
                    provenance_rows.append(
                        {
                            "experiment_seed": int(support.experiment_seed),
                            "heldout_center": heldout,
                            "candidate_expert": candidate,
                            "model_variant": variant,
                            "generator_family": F2_GENERATOR_FAMILY,
                            "checkpoint_path": str(ckpt),
                            "projection_path": str(_c41_projection_path(c41_artifacts_root, support.experiment_seed, candidate)),
                            "projection_source": "reused_c41_full_source_train_pca64",
                            "generation_conditioning": "source_train_residual_reference_posterior",
                            "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                            "routing_scores_recomputed_for_f2": 0,
                            "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                        }
                    )

                for generation_mode in F2_GENERATION_MODES:
                    variant = "no_energy_cov_penalty" if generation_mode in F2_NO_PENALTY_MODES else "penalty"
                    _ckpt, model, calibration = models[variant]
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            base_row, diagnostics, duplicates, sample_rows, prediction = _score_f2_candidate(
                                model=model,
                                calibration=calibration,
                                projection=projection,
                                anchor_index=source_anchor_index,
                                generation_mode=generation_mode,
                                experiment_seed=support.experiment_seed,
                                heldout_center=heldout,
                                candidate_expert=candidate,
                                target_eval_pool=target_pool,
                                target_labels=target_labels,
                                test_cache=test_cache,
                                train_cache=train_cache,
                                val_cache=val_cache,
                                label_values=label_values,
                                budget_per_class=config.primary_budget_per_class,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                            )
                            generator_diag_rows.append(diagnostics)
                            residual_moment_rows.append(_residual_moment_row(diagnostics))
                            duplicate_rows.append(duplicates)
                            sample_provenance_rows.extend(sample_rows)
                            if generation_mode == F2_MODE_CALIBRATED_NOISE and prediction is not None:
                                late_context.setdefault((int(generation_seed), int(classifier_seed)), []).append(prediction)
                            for support_size, support_seed in support_conditions:
                                row = replace(base_row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and row.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, row)
                                completed.add(row.primary_key())

            for (generation_seed, classifier_seed), predictions in late_context.items():
                if len(predictions) != len(candidates):
                    continue
                late_ensemble_rows.extend(
                    _late_ensemble_rows(
                        predictions=predictions,
                        target_labels=target_labels,
                        experiment_seed=int(support.experiment_seed),
                        heldout_center=heldout,
                        support_conditions=support_conditions,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                    )
                )

    _write_csv_with_header(artifacts_root / "tables" / "f2_anchor_pair_diagnostics.csv", anchor_diag_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_calibration_diagnostics.csv", calibration_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_generator_distribution_diagnostics.csv", generator_diag_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_residual_moment_diagnostics.csv", residual_moment_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_duplicate_diagnostics.csv", duplicate_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_synthetic_sample_provenance.csv", sample_provenance_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f2_late_ensemble_summary.csv", late_ensemble_rows)
    _write_csv_with_header(artifacts_root / "manifests" / "f2_generator_provenance.csv", provenance_rows)
    return matrix_path


def build_f2_routing_alignment_rows(
    *,
    selections: Sequence[SupportSelectionUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    f2_rows = [row for row in downstream_rows if row.generator_family == F2_GENERATOR_FAMILY and row.status == "ok"]
    single_rows = {
        (
            int(row.experiment_seed),
            row.heldout_center,
            int(row.support_size),
            int(row.support_seed),
            row.candidate_expert,
            row.generation_mode,
            int(row.budget_per_class),
            int(row.generation_seed),
            int(row.classifier_seed),
        ): row
        for row in f2_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
    }
    contexts = sorted(
        {
            (
                int(row.experiment_seed),
                row.heldout_center,
                row.generation_mode,
                int(row.budget_per_class),
                int(row.generation_seed),
                int(row.classifier_seed),
            )
            for row in f2_rows
            if int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        }
    )
    oracle_by_context = _f2_oracles(f2_rows)
    rows: list[dict[str, object]] = []
    for unit in selections:
        if unit.method != SUPPORT_NELBO_METHOD:
            continue
        for experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed in contexts:
            if int(experiment_seed) != int(unit.experiment_seed) or heldout != unit.heldout_center:
                continue
            selected_key = (
                int(unit.experiment_seed),
                unit.heldout_center,
                int(unit.support_size),
                int(unit.support_seed),
                unit.selected_expert,
                generation_mode,
                int(budget),
                int(generation_seed),
                int(classifier_seed),
            )
            selected = single_rows.get(selected_key) or single_rows.get(
                (
                    int(unit.experiment_seed),
                    unit.heldout_center,
                    0,
                    0,
                    unit.selected_expert,
                    generation_mode,
                    int(budget),
                    int(generation_seed),
                    int(classifier_seed),
                )
            )
            if selected is None:
                raise ProtocolError(f"Missing F2 downstream row for selected expert key {selected_key}")
            oracle = oracle_by_context.get((experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed))
            if oracle is None:
                raise ProtocolError("Missing F2 downstream oracle")
            oracle_gap = float(oracle.bacc) - float(selected.bacc)
            rows.append(
                {
                    "heldout_center": unit.heldout_center,
                    "experiment_seed": int(unit.experiment_seed),
                    "support_size": int(unit.support_size),
                    "support_seed": int(unit.support_seed),
                    "generator_family": F2_GENERATOR_FAMILY,
                    "generation_mode": generation_mode,
                    "generation_seed": int(generation_seed),
                    "classifier_seed": int(classifier_seed),
                    "method": unit.method,
                    "selected_expert": unit.selected_expert,
                    "selected_bacc": float(selected.bacc),
                    "selected_macro_f1": float(selected.macro_f1),
                    "downstream_oracle_expert": oracle.candidate_expert,
                    "oracle_bacc": float(oracle.bacc),
                    "oracle_macro_f1": float(oracle.macro_f1),
                    "downstream_oracle_gap_bacc": oracle_gap,
                    "downstream_oracle_gap_macro_f1": float(oracle.macro_f1) - float(selected.macro_f1),
                    "relative_downstream_oracle_gap_pct": (oracle_gap / float(oracle.bacc)) * 100.0 if float(oracle.bacc) else math.nan,
                    "top1_downstream_hit": int(str(unit.selected_expert) == str(oracle.candidate_expert)),
                    "spearman_neg_nelbo_vs_bacc": math.nan,
                    "metadata_bacc": math.nan,
                    "delta_vs_metadata": math.nan,
                    "selection_depends_on_support": 1,
                    "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                    "routing_scores_recomputed_for_f2": 0,
                    "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                    "projection_source": "reused_c41_full_source_train_pca64",
                    "generation_conditioning": "source_train_residual_reference_posterior",
                }
            )
    return rows


def build_f2_delta_summary_rows(
    *,
    f2_alignment_rows: Sequence[Mapping[str, object]],
    f1_alignment_rows: Sequence[Mapping[str, object]],
    duplicate_rows: Sequence[Mapping[str, object]] = (),
    residual_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    f2_support = [row for row in f2_alignment_rows if str(row.get("method")) == SUPPORT_NELBO_METHOD]
    f1_primary = _f1_rows_for_mode(f1_alignment_rows, "anchor_posterior_residual_mean")
    groups = sorted({(str(row["heldout_center"]), int(row["support_size"])) for row in f2_support})
    rows: list[dict[str, object]] = []
    for heldout, support_size in groups:
        f1_subset = _subset(f1_primary, heldout, support_size, "anchor_posterior_residual_mean")
        if not f1_subset:
            continue
        f1_selected = _mean(f1_subset, "selected_bacc")
        f1_oracle = _mean(f1_subset, "oracle_bacc")
        f1_gap = _mean(f1_subset, "downstream_oracle_gap_bacc")
        identity_selected = _mean(_subset(f2_support, heldout, support_size, F2_MODE_IDENTITY_BOOTSTRAP), "selected_bacc")
        empirical_selected = _mean(_subset(f2_support, heldout, support_size, F2_MODE_EMPIRICAL_BOOTSTRAP), "selected_bacc")
        transfer_selected = _mean(_subset(f2_support, heldout, support_size, F2_MODE_TRANSFER_BOOTSTRAP), "selected_bacc")
        for mode in F2_GENERATION_MODES:
            mode_subset = _subset(f2_support, heldout, support_size, mode)
            if not mode_subset:
                continue
            selected = _mean(mode_subset, "selected_bacc")
            oracle = _mean(mode_subset, "oracle_bacc")
            gap = _mean(mode_subset, "downstream_oracle_gap_bacc")
            deltas = [float(row.get("selected_bacc", math.nan)) - f1_selected for row in mode_subset]
            clean_deltas = [value for value in deltas if not math.isnan(value)]
            improvement_rate = sum(1 for value in clean_deltas if value > 0.0) / float(max(len(clean_deltas), 1))
            median_delta = _median(clean_deltas)
            near_copy = _f2_near_copy_failure(duplicate_rows, heldout, mode)
            moments_improved = _residual_moment_improved(residual_rows, heldout, mode)
            decision = _f2_decision_label(
                mode=mode,
                selected=selected,
                f1_selected=f1_selected,
                identity_selected=identity_selected,
                empirical_selected=empirical_selected,
                transfer_selected=transfer_selected,
                gap=gap,
                f1_gap=f1_gap,
                near_copy=near_copy,
                moments_improved=moments_improved,
                median_delta=median_delta,
                improvement_rate=improvement_rate,
            )
            rows.append(
                {
                    "heldout_center": heldout,
                    "support_size": support_size,
                    "generation_mode": mode,
                    "selected_bacc_f2": selected,
                    "oracle_bacc_f2": oracle,
                    "oracle_gap_f2": gap,
                    "selected_bacc_f1_posterior_residual_mean": f1_selected,
                    "oracle_bacc_f1_posterior_residual_mean": f1_oracle,
                    "oracle_gap_f1_posterior_residual_mean": f1_gap,
                    "selected_bacc_delta_vs_f1_posterior_residual_mean": selected - f1_selected,
                    "oracle_bacc_delta_vs_f1_posterior_residual_mean": oracle - f1_oracle,
                    "oracle_gap_delta_vs_f1_posterior_residual_mean": gap - f1_gap,
                    "selected_bacc_anchor_identity_bootstrap": identity_selected,
                    "selected_bacc_anchor_residual_empirical_bootstrap": empirical_selected,
                    "selected_bacc_anchor_empirical_residual_transfer_bootstrap": transfer_selected,
                    "beats_identity_bootstrap": int(selected > identity_selected),
                    "beats_empirical_bootstrap": int(selected > empirical_selected),
                    "beats_transfer_bootstrap": int(selected > transfer_selected),
                    "selected_ge_080": int(selected >= 0.80),
                    "median_center_seed_delta_gt_0": int(median_delta > 0.0),
                    "center_seed_improvement_rate": improvement_rate,
                    "near_copy_failure": int(near_copy),
                    "residual_moment_improved_vs_f1_mean": int(moments_improved),
                    "diagnostic_only": int(mode in F2_DIAGNOSTIC_ONLY_MODES),
                    "decision_label": decision,
                }
            )
    return rows


def build_f2_calibration_to_utility_join_rows(
    *,
    calibration_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    deltas = {
        mode: {
            "selected_bacc_delta_vs_f1_posterior_residual_mean": _mean(mode_rows, "selected_bacc_delta_vs_f1_posterior_residual_mean"),
            "oracle_bacc_delta_vs_f1_posterior_residual_mean": _mean(mode_rows, "oracle_bacc_delta_vs_f1_posterior_residual_mean"),
        }
        for mode, mode_rows in _group_by_mode(delta_rows).items()
    }
    rows: list[dict[str, object]] = []
    for cal in calibration_rows:
        for mode in (F2_MODE_CALIBRATED_NOISE, F2_MODE_CALIBRATED_NOISE_NO_PENALTY):
            delta = deltas.get(mode, {})
            rows.append(
                {
                    "expert_id": cal.get("candidate_expert"),
                    "class_label": cal.get("class_label"),
                    "model_variant": cal.get("model_variant"),
                    "scale_geomean": cal.get("scale_geomean"),
                    "scale_clipped_flag": cal.get("scale_clipped_flag"),
                    "source_val_moment_improved": cal.get("source_val_moment_improved"),
                    "generation_mode": mode,
                    "selected_bacc_delta_after_final_scoring": delta.get("selected_bacc_delta_vs_f1_posterior_residual_mean", math.nan),
                    "oracle_bacc_delta_after_final_scoring": delta.get("oracle_bacc_delta_vs_f1_posterior_residual_mean", math.nan),
                }
            )
    return rows


def _group_by_mode(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("generation_mode")), []).append(row)
    return grouped


def write_f2_alignment_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F2_ALIGNMENT_COLUMNS, rows)


def write_f2_delta_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F2_DELTA_COLUMNS, rows)


def load_f2_diagnostics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _run_f2_epoch(
    model: AnchoredResidualCVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    use_moment_penalties: bool,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    with torch.enable_grad() if optimizer is not None else torch.no_grad():
        for x_source_residual_ref, x_anchor, y in loader:
            x_source_residual_ref = x_source_residual_ref.to(device)
            x_anchor = x_anchor.to(device)
            y = y.to(device)
            delta_true = x_source_residual_ref - x_anchor
            delta_mu, delta_logvar, mu, logvar = model(x_source_residual_ref, x_anchor, y)
            terms = anchored_residual_loss_terms_f2(
                delta_mu=delta_mu,
                delta_true=delta_true,
                delta_logvar=delta_logvar,
                mu=mu,
                logvar=logvar,
                use_moment_penalties=use_moment_penalties,
            )
            loss = terms["loss"]
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * int(x_source_residual_ref.shape[0])
            count += int(x_source_residual_ref.shape[0])
    return total / float(max(count, 1))


def _score_f2_candidate(
    *,
    model: AnchoredResidualCVAE,
    calibration: ResidualCalibration,
    projection: SourceTrainPCAProjection,
    anchor_index: SourceAnchorIndex,
    generation_mode: str,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    test_cache: object,
    train_cache: object,
    val_cache: object,
    label_values: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
) -> tuple[CandidateDownstreamRow, dict[str, object], dict[str, object], list[dict[str, object]], object | None]:
    try:
        chunks: list[torch.Tensor] = []
        labels: list[int] = []
        diagnostic_parts: list[Mapping[str, float]] = []
        provenance_rows: list[dict[str, object]] = []
        for label in label_values:
            generated = generate_f2_anchor_residual_embeddings(
                model=model,
                anchor_index=anchor_index,
                calibration=calibration,
                class_label=int(label),
                n_samples=int(budget_per_class),
                seed=int(generation_seed) + int(label),
                generation_mode=generation_mode,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                candidate_expert=candidate_expert,
            )
            chunks.append(generated.embeddings)
            labels.extend(int(v) for v in generated.labels.tolist())
            diagnostic_parts.append(generated.diagnostics)
            provenance_rows.extend(generated.provenance_rows)
        synthetic_embeddings = torch.cat(chunks, dim=0)
        target_embeddings = projection.transform(test_cache.embeddings[list(target_eval_pool.eval_indices)])
        prediction = fit_locked_logistic_classifier(
            _to_numpy(synthetic_embeddings),
            labels,
            _to_numpy(target_embeddings),
            target_labels,
            classifier_seed=classifier_seed,
        )
        source_train_idx = _indices_for_domain(train_cache.metadata, candidate_expert)
        source_val_idx = _indices_for_domain(val_cache.metadata, candidate_expert)
        source_train_pca = projection.transform(train_cache.embeddings[source_train_idx])
        source_val_pca = projection.transform(val_cache.embeddings[source_val_idx])
        source_train_dino = train_cache.embeddings[source_train_idx]
        source_train_labels = [_label(train_cache.metadata[idx]) for idx in source_train_idx]
        diagnostics = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": F2_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "diagnostic_only": int(generation_mode in F2_DIAGNOSTIC_ONLY_MODES),
            **_aggregate_float_dicts(diagnostic_parts),
            **_generated_distribution_diagnostics(
                synthetic_embeddings=synthetic_embeddings,
                synthetic_labels=labels,
                source_train_pca=source_train_pca,
                source_train_dino=source_train_dino,
                projection=projection,
            ),
            **_f2_geometry_diagnostics(
                synthetic_embeddings=synthetic_embeddings,
                synthetic_labels=labels,
                source_train_pca=source_train_pca,
                source_train_labels=source_train_labels,
                source_val_pca=source_val_pca,
            ),
        }
        duplicate = {
            key: diagnostics.get(key, math.nan)
            for key in (
                "experiment_seed",
                "heldout_center",
                "candidate_expert",
                "generator_family",
                "generation_mode",
                "generation_seed",
                "classifier_seed",
                "min_dist_to_anchor",
                "min_dist_to_reference_source_sample",
                "fraction_exact_or_near_duplicate_anchor",
                "fraction_exact_or_near_duplicate_reference",
                "mean_interpolation_ratio",
                "anchor_reuse_rate",
                "median_nn_dist_synthetic_to_source_train",
                "median_nn_dist_source_val_to_source_train",
                "median_nn_copy_ratio",
                "top_5_source_nn_share_per_class",
                "hard_near_copy_failure",
            )
        }
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=F2_GENERATOR_FAMILY,
            generation_mode=generation_mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=int(budget_per_class) * len(label_values),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            candidate_experts_hash=SINGLE_EXPERT_HASH,
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            plain_baseline_source="f2_uses_reused_c41_projection",
            plain_baseline_artifact_path="",
            plain_baseline_training_profile="f2_source_anchored_calibrated_residual",
            plain_baseline_matches_locked_hparams=0,
            routing_family_used=BASELINE_ROUTING_FAMILY_USED,
            routing_scores_recomputed_for_heteroscedastic=0,
            selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        )
        return row, diagnostics, duplicate, provenance_rows, prediction
    except Exception as exc:
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=F2_GENERATOR_FAMILY,
            generation_mode=generation_mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=math.nan,
            macro_f1=math.nan,
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=int(budget_per_class) * len(label_values),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            status="failed_f2_candidate_scoring",
            error_message=str(exc),
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            routing_family_used=BASELINE_ROUTING_FAMILY_USED,
            routing_scores_recomputed_for_heteroscedastic=0,
            selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        )
        diagnostics = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": F2_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "status": row.status,
            "error_message": row.error_message,
        }
        return row, diagnostics, dict(diagnostics), [], None


def _f2_generation_diagnostics(
    *,
    embeddings: torch.Tensor,
    anchors: torch.Tensor,
    source_refs: torch.Tensor,
    reference_delta: torch.Tensor,
    delta_logvar: torch.Tensor,
    decoder_noise: torch.Tensor,
    anchor_ids: Sequence[str],
    scale_used: float,
) -> dict[str, float]:
    gen_residual = embeddings - anchors
    dist_anchor = gen_residual.norm(dim=1)
    dist_ref = (embeddings - source_refs).norm(dim=1)
    ref_norm = reference_delta.norm(dim=1).clamp_min(1.0e-12)
    generated_norm = gen_residual.norm(dim=1)
    logvar_at_min = (delta_logvar <= -9.21 + 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    logvar_at_max = (delta_logvar >= 2.0 - 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    noise_energy = decoder_noise.pow(2).sum(dim=1).mean() if decoder_noise.numel() else torch.tensor(0.0)
    mean_energy = embeddings.pow(2).sum(dim=1).mean().clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(gen_residual, reference_delta, dim=1).mean() if gen_residual.numel() else torch.tensor(float("nan"))
    return {
        "min_dist_to_anchor": float(dist_anchor.min().item()),
        "min_dist_to_reference_source_sample": float(dist_ref.min().item()),
        "fraction_exact_or_near_duplicate_anchor": float((dist_anchor <= NEAR_DUPLICATE_EPS).float().mean().item()),
        "fraction_exact_or_near_duplicate_reference": float((dist_ref <= NEAR_DUPLICATE_EPS).float().mean().item()),
        "mean_interpolation_ratio": float((generated_norm / ref_norm).mean().item()),
        "anchor_reuse_rate": 1.0 - (float(len(set(str(v) for v in anchor_ids))) / float(max(len(anchor_ids), 1))),
        "real_pair_delta_norm_mean": float(ref_norm.mean().item()),
        "generated_residual_norm_mean": float(generated_norm.mean().item()),
        "residual_energy_ratio": float(gen_residual.pow(2).sum(dim=1).mean().item() / max(reference_delta.pow(2).sum(dim=1).mean().item(), 1.0e-12)),
        "residual_cov_trace_ratio": float(_trace_cov(gen_residual) / max(_trace_cov(reference_delta), 1.0e-12)),
        "residual_cosine_alignment_real_vs_synthetic": float(cosine.item()),
        "decoder_logvar_mean": float(delta_logvar.mean().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_min": float(delta_logvar.min().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_max": float(delta_logvar.max().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_at_min_frac": float(logvar_at_min.item()),
        "decoder_logvar_at_max_frac": float(logvar_at_max.item()),
        "decoder_noise_energy_ratio": float((noise_energy / mean_energy).item()),
        "residual_calibration_scale_used": float(scale_used),
    }


def _f2_geometry_diagnostics(
    *,
    synthetic_embeddings: torch.Tensor,
    synthetic_labels: Sequence[int],
    source_train_pca: torch.Tensor,
    source_train_labels: Sequence[int],
    source_val_pca: torch.Tensor,
) -> dict[str, float]:
    synthetic = synthetic_embeddings.detach().cpu().float()
    source = source_train_pca.detach().cpu().float()
    source_val = source_val_pca.detach().cpu().float()
    if synthetic.numel() == 0 or source.numel() == 0:
        return {}
    distances = torch.cdist(synthetic, source)
    min_dist, nn_idx = distances.min(dim=1)
    source_val_min = torch.cdist(source_val, source).min(dim=1).values if source_val.numel() else torch.tensor([math.nan])
    median_syn = float(min_dist.median().item())
    median_val = float(source_val_min.median().item()) if torch.isfinite(source_val_min).any() else math.nan
    top5_share = _top5_nn_share_per_class(nn_idx.tolist(), synthetic_labels)
    hard_copy = int((not math.isnan(median_val) and median_syn < MEDIAN_NN_COPY_RATIO_THRESHOLD * median_val) or top5_share > TOP5_NN_SHARE_FAILURE_THRESHOLD)
    return {
        "synthetic_to_train_nn_dist_p10": _quantile(min_dist, 0.10),
        "synthetic_to_train_nn_dist_p50": median_syn,
        "synthetic_to_train_nn_dist_p90": _quantile(min_dist, 0.90),
        "median_nn_dist_synthetic_to_source_train": median_syn,
        "median_nn_dist_source_val_to_source_train": median_val,
        "median_nn_copy_ratio": median_syn / median_val if median_val and not math.isnan(median_val) else math.nan,
        "top_5_source_nn_share_per_class": top5_share,
        "hard_near_copy_failure": hard_copy,
        "within_class_cov_frobenius_error": _within_class_cov_error(synthetic, synthetic_labels, source, source_train_labels),
        "class_centroid_shift_error": _class_centroid_shift_error(synthetic, synthetic_labels, source, source_train_labels),
        "between_class_margin_preservation": _between_class_margin_ratio(synthetic, synthetic_labels, source, source_train_labels),
        "residual_eigenvalue_topk_ratio": _eigen_topk_ratio(synthetic - synthetic.mean(dim=0), source - source.mean(dim=0), top_k=5),
        "anchor_to_synthetic_nn_distance_ratio": _pairwise_distance_mean(synthetic) / max(_pairwise_distance_mean(source), 1.0e-12),
    }


def _f2_provenance_rows(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    support_size: int,
    support_seed: int,
    generation_mode: str,
    generation_seed: int,
    class_label: int,
    pair: Mapping[str, object],
    posterior_or_prior_source: str,
    scale_used: float,
) -> list[dict[str, object]]:
    source_ref_ids = tuple(str(v) for v in pair["pair_ids"])
    anchor_ids = tuple(str(v) for v in pair["anchor_ids"])
    anchor_a_ids = tuple(str(v) for v in pair.get("anchor_a_ids", anchor_ids))
    residual_anchor_ids = tuple(str(v) for v in pair.get("residual_anchor_ids", anchor_ids))
    rows: list[dict[str, object]] = []
    for idx, (source_ref_id, anchor_id) in enumerate(zip(source_ref_ids, anchor_ids)):
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "candidate_expert": candidate_expert,
                "support_size": int(support_size),
                "support_seed": int(support_seed),
                "generation_mode": generation_mode,
                "generation_seed": int(generation_seed),
                "synthetic_index": idx,
                "class_label": int(class_label),
                "synthetic_anchor_id": anchor_a_ids[idx] if generation_mode == F2_MODE_TRANSFER_BOOTSTRAP else anchor_id,
                "residual_reference_sample_id": source_ref_id,
                "residual_anchor_id": residual_anchor_ids[idx],
                "anchor_split": "source_train",
                "residual_reference_split": "source_train",
                "calibration_split": "source_val",
                "same_class_anchor": 1,
                "posterior_or_prior_source": posterior_or_prior_source,
                "generation_conditioning": "source_train_residual_reference_posterior",
                "residual_calibration_scale_used": float(scale_used),
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
        )
    return rows


def _residual_moment_row(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "experiment_seed",
        "heldout_center",
        "candidate_expert",
        "generator_family",
        "generation_mode",
        "generation_seed",
        "classifier_seed",
        "real_pair_delta_norm_mean",
        "generated_residual_norm_mean",
        "residual_energy_ratio",
        "residual_cov_trace_ratio",
        "residual_cosine_alignment_real_vs_synthetic",
        "residual_eigenvalue_topk_ratio",
        "within_class_cov_frobenius_error",
        "class_centroid_shift_error",
        "between_class_margin_preservation",
    )
    return {key: row.get(key, math.nan) for key in keys}


def _late_ensemble_rows(
    *,
    predictions: Sequence[object],
    target_labels: Sequence[int],
    experiment_seed: int,
    heldout_center: str,
    support_conditions: Sequence[tuple[int, int]],
    generation_seed: int,
    classifier_seed: int,
) -> list[dict[str, object]]:
    import numpy as np

    proba = np.mean([np.asarray(pred.probabilities, dtype=float) for pred in predictions], axis=0)
    classes = tuple(int(v) for v in predictions[0].classes)
    if classes != (0, 1):
        raise ProtocolError(f"F2 late ensemble expects classes (0, 1), got {classes}")
    y_pred = (proba[:, 1] >= 0.5).astype(int).tolist()
    bacc = balanced_accuracy(target_labels, y_pred)
    f1 = macro_f1(target_labels, y_pred)
    rows: list[dict[str, object]] = []
    for support_size, support_seed in support_conditions:
        rows.append(
            {
                "mixture_policy": "f2_fixed_all_source_calibrated_residual_late_ensemble",
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "support_size": int(support_size),
                "support_seed": int(support_seed),
                "generation_mode": F2_MODE_CALIBRATED_NOISE,
                "generation_seed": int(generation_seed),
                "classifier_seed": int(classifier_seed),
                "mean_bacc": float(bacc),
                "macro_f1": float(f1),
                "f2_late_ge_080": int(float(bacc) >= 0.80),
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
                "heldout_source_excluded": 1,
                "decision_label": DECISION_THESIS_SUCCESS if float(bacc) >= 0.80 else DECISION_ENSEMBLE_REQUIRED,
            }
        )
    return rows


def _load_f2_model(path: Path, *, device: str) -> AnchoredResidualCVAE:
    torch_device = _resolve_torch_device(torch, device)
    try:
        payload = torch.load(path, map_location=torch_device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=torch_device)
    metadata = dict(payload.get("checkpoint_metadata", {}))
    model = AnchoredResidualCVAE(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        latent_dim=int(metadata["latent_dim"]),
        class_condition_dim=int(metadata.get("class_condition_dim", 2)),
    ).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _f2_oracles(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[int, str, str, int, int, int], CandidateDownstreamRow]:
    grouped: dict[tuple[int, str, str, int, int, int], list[CandidateDownstreamRow]] = {}
    for row in rows:
        if row.generator_family != F2_GENERATOR_FAMILY or row.status != "ok":
            continue
        key = (
            int(row.experiment_seed),
            row.heldout_center,
            row.generation_mode,
            int(row.budget_per_class),
            int(row.generation_seed),
            int(row.classifier_seed),
        )
        grouped.setdefault(key, []).append(row)
    return {
        key: max(group, key=lambda row: (float(row.bacc), float(row.macro_f1), _reverse_lex(row.candidate_expert)))
        for key, group in grouped.items()
    }


def _f2_decision_label(
    *,
    mode: str,
    selected: float,
    f1_selected: float,
    identity_selected: float,
    empirical_selected: float,
    transfer_selected: float,
    gap: float,
    f1_gap: float,
    near_copy: bool,
    moments_improved: bool,
    median_delta: float,
    improvement_rate: float,
) -> str:
    if mode != F2_MODE_CALIBRATED_NOISE:
        return DECISION_DIAGNOSTIC_ONLY
    if near_copy:
        return DECISION_NEAR_COPY
    repair = (
        selected > f1_selected
        and median_delta > 0.0
        and improvement_rate >= 0.60
        and selected > identity_selected
        and moments_improved
    )
    if repair and selected > empirical_selected and selected > transfer_selected and gap <= f1_gap:
        return DECISION_SUPERIORITY_SUCCESS
    if repair:
        return DECISION_REPAIR_SUCCESS
    if moments_improved and selected <= f1_selected:
        return DECISION_MOMENTS_ONLY
    if selected < transfer_selected:
        return DECISION_BOOTSTRAP_STRONGER
    return DECISION_CALIBRATION_NO_GAIN


def _f1_rows_for_mode(rows: Sequence[Mapping[str, object]], mode: str) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("method")) == SUPPORT_NELBO_METHOD
        and str(row.get("generator_family", "")).startswith("family_f1_")
        and str(row.get("generation_mode")) == mode
    ]


def _f2_near_copy_failure(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not subset:
        return False
    hard = max(_as_float(row.get("hard_near_copy_failure", 0)) for row in subset)
    return bool(hard >= 1)


def _residual_moment_improved(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not subset:
        return False
    energy_error = abs(math.log(max(_mean(subset, "residual_energy_ratio"), 1.0e-12)))
    cov_error = abs(math.log(max(_mean(subset, "residual_cov_trace_ratio"), 1.0e-12)))
    return bool((energy_error + cov_error) < 1.0)


def _scale_stats(true_delta: torch.Tensor, synthetic_delta: torch.Tensor) -> dict[str, float]:
    true_norm = true_delta.norm(dim=1).mean().item()
    synthetic_norm = synthetic_delta.norm(dim=1).mean().item()
    scale_norm = true_norm / max(synthetic_norm, 1.0e-12)
    true_trace = _trace_cov(true_delta)
    synthetic_trace = _trace_cov(synthetic_delta)
    scale_cov = math.sqrt(true_trace / max(synthetic_trace, 1.0e-12))
    return {
        "scale_norm": float(scale_norm),
        "scale_cov_trace": float(scale_cov),
        "scale_geomean": float(math.sqrt(max(scale_norm, 1.0e-12) * max(scale_cov, 1.0e-12))),
    }


def _moment_error(scale: float, stats: Mapping[str, float]) -> float:
    norm_error = abs(math.log(max(float(scale) / max(float(stats["scale_norm"]), 1.0e-12), 1.0e-12)))
    cov_error = abs(math.log(max(float(scale) / max(float(stats["scale_cov_trace"]), 1.0e-12), 1.0e-12)))
    return norm_error + cov_error


def _clip_scale(value: float) -> float:
    return min(CALIBRATION_MAX_SCALE, max(CALIBRATION_MIN_SCALE, float(value)))


def _log_ratio_square(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.log(numerator.clamp_min(1.0e-12) / denominator.clamp_min(1.0e-12)).pow(2)


def _trace_cov_tensor(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[0]) <= 1:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return x.var(dim=0, unbiased=True).sum()


def _trace_cov(x: torch.Tensor) -> float:
    arr = x.detach().cpu().float()
    if int(arr.shape[0]) <= 1:
        return 0.0
    return float(arr.var(dim=0, unbiased=True).sum().item())


def _quantile(values: torch.Tensor, q: float) -> float:
    if not values.numel():
        return math.nan
    return float(torch.quantile(values.detach().cpu().float(), float(q)).item())


def _top5_nn_share_per_class(nn_indices: Sequence[int], labels: Sequence[int]) -> float:
    max_share = 0.0
    for label in sorted(set(int(v) for v in labels)):
        idxs = [int(nn) for nn, y in zip(nn_indices, labels) if int(y) == int(label)]
        if not idxs:
            continue
        counts: dict[int, int] = {}
        for idx in idxs:
            counts[idx] = counts.get(idx, 0) + 1
        top5 = sum(sorted(counts.values(), reverse=True)[:5])
        max_share = max(max_share, float(top5) / float(len(idxs)))
    return max_share


def _within_class_cov_error(
    synthetic: torch.Tensor,
    synthetic_labels: Sequence[int],
    source: torch.Tensor,
    source_labels: Sequence[int],
) -> float:
    errors: list[float] = []
    for label in sorted(set(int(v) for v in synthetic_labels).intersection(int(v) for v in source_labels)):
        syn = synthetic[[idx for idx, y in enumerate(synthetic_labels) if int(y) == label]]
        src = source[[idx for idx, y in enumerate(source_labels) if int(y) == label]]
        if int(syn.shape[0]) <= 1 or int(src.shape[0]) <= 1:
            continue
        syn_cov = _cov_matrix(syn)
        src_cov = _cov_matrix(src)
        errors.append(float(torch.linalg.matrix_norm(syn_cov - src_cov, ord="fro").item() / max(torch.linalg.matrix_norm(src_cov, ord="fro").item(), 1.0e-12)))
    return sum(errors) / float(len(errors)) if errors else math.nan


def _class_centroid_shift_error(
    synthetic: torch.Tensor,
    synthetic_labels: Sequence[int],
    source: torch.Tensor,
    source_labels: Sequence[int],
) -> float:
    shifts: list[float] = []
    for label in sorted(set(int(v) for v in synthetic_labels).intersection(int(v) for v in source_labels)):
        syn = synthetic[[idx for idx, y in enumerate(synthetic_labels) if int(y) == label]]
        src = source[[idx for idx, y in enumerate(source_labels) if int(y) == label]]
        if not int(syn.shape[0]) or not int(src.shape[0]):
            continue
        shifts.append(float((syn.mean(dim=0) - src.mean(dim=0)).norm().item()))
    return sum(shifts) / float(len(shifts)) if shifts else math.nan


def _between_class_margin_ratio(
    synthetic: torch.Tensor,
    synthetic_labels: Sequence[int],
    source: torch.Tensor,
    source_labels: Sequence[int],
) -> float:
    if len(set(int(v) for v in synthetic_labels)) < 2 or len(set(int(v) for v in source_labels)) < 2:
        return math.nan
    syn0 = synthetic[[idx for idx, y in enumerate(synthetic_labels) if int(y) == 0]]
    syn1 = synthetic[[idx for idx, y in enumerate(synthetic_labels) if int(y) == 1]]
    src0 = source[[idx for idx, y in enumerate(source_labels) if int(y) == 0]]
    src1 = source[[idx for idx, y in enumerate(source_labels) if int(y) == 1]]
    if min(int(syn0.shape[0]), int(syn1.shape[0]), int(src0.shape[0]), int(src1.shape[0])) == 0:
        return math.nan
    syn_margin = (syn0.mean(dim=0) - syn1.mean(dim=0)).norm().item()
    src_margin = (src0.mean(dim=0) - src1.mean(dim=0)).norm().item()
    return float(syn_margin / max(src_margin, 1.0e-12))


def _eigen_topk_ratio(generated: torch.Tensor, reference: torch.Tensor, *, top_k: int) -> float:
    if int(generated.shape[0]) <= 1 or int(reference.shape[0]) <= 1:
        return math.nan
    gen_vals = torch.linalg.eigvalsh(_cov_matrix(generated)).clamp_min(0.0)
    ref_vals = torch.linalg.eigvalsh(_cov_matrix(reference)).clamp_min(0.0)
    k = min(int(top_k), int(gen_vals.numel()), int(ref_vals.numel()))
    return float(torch.sort(gen_vals, descending=True).values[:k].sum().item() / max(torch.sort(ref_vals, descending=True).values[:k].sum().item(), 1.0e-12))


def _cov_matrix(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(max(int(x.shape[0]) - 1, 1))


def _median(values: Sequence[float]) -> float:
    clean = sorted(float(v) for v in values if not math.isnan(float(v)))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0
