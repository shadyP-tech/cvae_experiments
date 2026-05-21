"""F2.1 direction-preserving source-anchored residual diagnostic.

F2.1 is a generator-construction diagnostic. It keeps locked C4.1
support-NELBO selected experts fixed and changes only source-local residual
generation. The method replays source-train residual directions and lets the
no-penalty F2 residual CVAE contribute calibrated residual magnitudes.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .c41_heteroscedastic import SourceTrainPCAProjection, _generator_for_device, _randn_like
from .c41_workstation import (
    C41TrainingProfile,
    _indices_for_domain,
    _support_conditions,
    _write_csv,
    discover_c41_run_artifacts,
)
from .downstream import CandidateDownstreamRow, fit_locked_logistic_classifier, read_candidate_downstream_matrix
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
)
from .f2_calibrated_residual import (
    CALIBRATION_MAX_SCALE,
    CALIBRATION_MIN_SCALE,
    F2_MODE_CALIBRATED_NOISE_NO_PENALTY,
    F2_MODE_IDENTITY_BOOTSTRAP,
    F2_MODE_TRANSFER_BOOTSTRAP,
    MEDIAN_NN_COPY_RATIO_THRESHOLD,
    TOP5_NN_SHARE_FAILURE_THRESHOLD,
    ResidualCalibration,
    _between_class_margin_ratio,
    _class_centroid_shift_error,
    _clip_scale,
    _eigen_topk_ratio,
    _f2_geometry_diagnostics,
    _load_f2_model,
    _quantile,
    _scale_stats,
    _top5_nn_share_per_class,
    _within_class_cov_error,
    fit_residual_calibration,
    generate_f2_anchor_residual_embeddings,
    train_f2_anchored_residual_cvae,
)
from .matrix import (
    MatrixBuildLimits,
    TargetEvalPool,
    _label,
    _load_embedding_cache,
    _read_completed_keys,
    _read_samples_manifest,
    _records_for_split,
    _to_numpy,
    append_matrix_row,
    build_target_eval_pool,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    BASELINE_ROUTING_FAMILY_USED,
    BASELINE_SELECTED_EXPERT_IDS_SOURCE,
    PRIMARY_BUDGET_PER_CLASS,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


F21_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/f21_direction_preserving_residual_cvae_v1"
F21_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
F21_DEFAULT_F1_ROOT = "cvae_downstream_evaluation/artifacts/f1_source_anchored_residual_cvae_v1"
F21_DEFAULT_F2_ROOT = "cvae_downstream_evaluation/artifacts/f2_source_anchored_calibrated_residual_cvae_v1"
F21_GENERATOR_FAMILY = "family_f21_pca64_class_conditional_source_anchored_direction_preserving_residual_cvae_downstream_v1"

F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE = "anchor_empirical_direction_cvae_magnitude"
F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE_JITTER = "anchor_empirical_direction_cvae_magnitude_jitter"
F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE = "anchor_empirical_direction_empirical_magnitude"
F21_MODE_CVAE_DIRECTION_CVAE_MAGNITUDE = "anchor_cvae_direction_cvae_magnitude"
F21_MODE_F2_NO_PENALTY_REPLAY = "f2_no_penalty_calibrated_noise_replay"
F21_MODE_TRANSFER_BOOTSTRAP = "anchor_empirical_residual_transfer_bootstrap"
F21_MODE_IDENTITY_BOOTSTRAP = "anchor_identity_bootstrap"

F21_GENERATION_MODES = (
    F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE,
    F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE_JITTER,
    F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE,
    F21_MODE_CVAE_DIRECTION_CVAE_MAGNITUDE,
    F21_MODE_F2_NO_PENALTY_REPLAY,
    F21_MODE_TRANSFER_BOOTSTRAP,
    F21_MODE_IDENTITY_BOOTSTRAP,
)
F21_DIAGNOSTIC_ONLY_MODES = tuple(mode for mode in F21_GENERATION_MODES if mode != F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE)
F21_DIRECTION_MODES = {
    F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE,
    F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE_JITTER,
    F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE,
    F21_MODE_CVAE_DIRECTION_CVAE_MAGNITUDE,
}

F21_DIRECTION_BANK_MIN_VALID = 64
F21_DIRECTION_NEAR_ZERO_EPS = 1.0e-12
F21_DIRECTION_JITTER_SCALE = 0.10
F21_TOP5_DIRECTION_SHARE_FAILURE_THRESHOLD = 0.30
F21_SAME_DIRECTION_ANCHOR_REUSE_FAILURE_THRESHOLD = 0.05

STATUS_DIRECTION_BANK_INVALID = "F21_DIRECTION_BANK_INVALID"

DECISION_DIRECTION_REPAIR_SUCCESS = "F21_DIRECTION_REPAIR_SUCCESS"
DECISION_TRANSFER_SUPERIORITY_SUCCESS = "F21_TRANSFER_SUPERIORITY_SUCCESS"
DECISION_MAGNITUDE_MODEL_NO_GAIN = "F21_MAGNITUDE_MODEL_NO_GAIN"
DECISION_NO_GAIN_DIRECTION_BANK_ONLY = "F21_NO_GAIN_DIRECTION_BANK_ONLY"
DECISION_DIRECTION_CONCENTRATION_FAILURE = "F21_DIRECTION_CONCENTRATION_FAILURE"
DECISION_NEAR_COPY_FAILURE = "F21_NEAR_COPY_FAILURE"
DECISION_SEED_UNSTABLE_GAIN = "F21_SEED_UNSTABLE_GAIN"
DECISION_PROTOCOL_FAILURE = "F21_PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"

F21_ALIGNMENT_COLUMNS = (
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
    "selected_status",
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
    "routing_scores_recomputed_for_f21",
    "selected_expert_ids_source",
    "projection_source",
    "generation_conditioning",
    "decision_label",
)

F21_DELTA_COLUMNS = (
    "heldout_center",
    "support_size",
    "generation_mode",
    "selected_bacc_f21",
    "oracle_bacc_f21",
    "oracle_gap_f21",
    "selected_bacc_f2_no_penalty_calibrated_noise",
    "oracle_bacc_f2_no_penalty_calibrated_noise",
    "oracle_gap_f2_no_penalty_calibrated_noise",
    "selected_bacc_delta_vs_f2_no_penalty_calibrated_noise",
    "oracle_bacc_delta_vs_f2_no_penalty_calibrated_noise",
    "oracle_gap_delta_vs_f2_no_penalty_calibrated_noise",
    "selected_bacc_anchor_empirical_direction_empirical_magnitude",
    "selected_bacc_anchor_empirical_residual_transfer_bootstrap",
    "beats_empirical_direction_empirical_magnitude",
    "beats_transfer_bootstrap",
    "selected_ge_080",
    "valid_cell_count",
    "invalid_cell_count",
    "center_seed_improvement_rate",
    "seed_gain_rate",
    "strong_center_degraded",
    "near_copy_failure",
    "direction_concentration_failure",
    "mechanistic_direction_or_margin_improved",
    "diagnostic_only",
    "decision_label",
)


@dataclass(frozen=True)
class DirectionBank:
    anchor_index: SourceAnchorIndex
    label_values: tuple[int, ...]
    directions_by_class: Mapping[int, torch.Tensor]
    magnitudes_by_class: Mapping[int, torch.Tensor]
    reference_local_indices_by_class: Mapping[int, tuple[int, ...]]
    anchor_local_indices_by_class: Mapping[int, tuple[int, ...]]
    reference_ids_by_class: Mapping[int, tuple[str, ...]]
    anchor_ids_by_class: Mapping[int, tuple[str, ...]]
    valid_by_class: Mapping[int, bool]
    rows: tuple[dict[str, object], ...]
    min_valid_directions: int

    def valid_for(self, labels: Sequence[int]) -> bool:
        return all(bool(self.valid_by_class.get(int(label), False)) for label in labels)

    def invalid_reason_for(self, labels: Sequence[int]) -> str:
        invalid = [str(int(label)) for label in labels if not bool(self.valid_by_class.get(int(label), False))]
        return "" if not invalid else f"invalid_direction_bank_classes={','.join(invalid)}"


@dataclass(frozen=True)
class F21GeneratedBatch:
    embeddings: torch.Tensor
    labels: torch.Tensor
    generation_mode: str
    diagnostics: Mapping[str, float]
    provenance_rows: tuple[dict[str, object], ...]


def build_f21_direction_bank(
    *,
    anchor_index: SourceAnchorIndex,
    label_values: Sequence[int],
    experiment_seed: int = 0,
    heldout_center: str = "",
    candidate_expert: str = "",
    min_valid_directions: int = F21_DIRECTION_BANK_MIN_VALID,
    near_zero_eps: float = F21_DIRECTION_NEAR_ZERO_EPS,
) -> DirectionBank:
    directions_by_class: dict[int, torch.Tensor] = {}
    magnitudes_by_class: dict[int, torch.Tensor] = {}
    ref_indices_by_class: dict[int, tuple[int, ...]] = {}
    anchor_indices_by_class: dict[int, tuple[int, ...]] = {}
    ref_ids_by_class: dict[int, tuple[str, ...]] = {}
    anchor_ids_by_class: dict[int, tuple[str, ...]] = {}
    valid_by_class: dict[int, bool] = {}
    rows: list[dict[str, object]] = []

    for class_label in tuple(int(v) for v in label_values):
        directions: list[torch.Tensor] = []
        magnitudes: list[torch.Tensor] = []
        ref_indices: list[int] = []
        anchor_indices: list[int] = []
        ref_ids: list[str] = []
        anchor_ids: list[str] = []
        near_zero_rejected = 0
        total_candidates = 0
        for ref_local, label in enumerate(anchor_index.labels.tolist()):
            if int(label) != int(class_label):
                continue
            for anchor_local in anchor_index.neighbor_indices[ref_local]:
                total_candidates += 1
                if int(anchor_local) == int(ref_local):
                    continue
                if str(anchor_index.sample_ids[anchor_local]) == str(anchor_index.sample_ids[ref_local]):
                    continue
                delta = anchor_index.embeddings[ref_local] - anchor_index.embeddings[int(anchor_local)]
                norm = delta.norm()
                if float(norm.item()) <= float(near_zero_eps):
                    near_zero_rejected += 1
                    continue
                directions.append((delta / norm).detach().cpu().float())
                magnitudes.append(norm.detach().cpu().float())
                ref_indices.append(int(ref_local))
                anchor_indices.append(int(anchor_local))
                ref_ids.append(str(anchor_index.sample_ids[ref_local]))
                anchor_ids.append(str(anchor_index.sample_ids[int(anchor_local)]))

        n_valid = len(directions)
        valid = n_valid >= int(min_valid_directions)
        valid_by_class[class_label] = bool(valid)
        directions_by_class[class_label] = torch.stack(directions).float() if directions else torch.empty((0, anchor_index.embeddings.shape[1]))
        magnitudes_by_class[class_label] = torch.stack(magnitudes).float() if magnitudes else torch.empty((0,))
        ref_indices_by_class[class_label] = tuple(ref_indices)
        anchor_indices_by_class[class_label] = tuple(anchor_indices)
        ref_ids_by_class[class_label] = tuple(ref_ids)
        anchor_ids_by_class[class_label] = tuple(anchor_ids)
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "candidate_expert": candidate_expert,
                "class_label": int(class_label),
                "n_valid_directions_per_expert_class": int(n_valid),
                "effective_sample_size_per_expert_class": _effective_sample_size(ref_ids),
                "near_zero_rejection_rate": float(near_zero_rejected) / float(max(total_candidates, 1)),
                "direction_reference_reuse_rate": 1.0 - (float(len(set(ref_ids))) / float(max(len(ref_ids), 1))),
                "top_5_direction_reference_share_per_class": _top5_string_share(ref_ids),
                "direction_bank_valid": int(valid),
                "invalid_reason": "" if valid else f"n_valid_directions<{int(min_valid_directions)}",
                "direction_reference_split": "source_train",
                "direction_anchor_split": "source_train",
            }
        )
    return DirectionBank(
        anchor_index=anchor_index,
        label_values=tuple(int(v) for v in label_values),
        directions_by_class=directions_by_class,
        magnitudes_by_class=magnitudes_by_class,
        reference_local_indices_by_class=ref_indices_by_class,
        anchor_local_indices_by_class=anchor_indices_by_class,
        reference_ids_by_class=ref_ids_by_class,
        anchor_ids_by_class=anchor_ids_by_class,
        valid_by_class=valid_by_class,
        rows=tuple(rows),
        min_valid_directions=int(min_valid_directions),
    )


def generate_f21_direction_preserving_embeddings(
    *,
    model: AnchoredResidualCVAE,
    anchor_index: SourceAnchorIndex,
    direction_bank: DirectionBank,
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
) -> F21GeneratedBatch:
    if generation_mode not in F21_GENERATION_MODES:
        raise ProtocolError(f"Unknown F2.1 generation mode: {generation_mode}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")

    if generation_mode in {F21_MODE_F2_NO_PENALTY_REPLAY, F21_MODE_TRANSFER_BOOTSTRAP, F21_MODE_IDENTITY_BOOTSTRAP}:
        return _generate_f21_replay_control(
            model=model,
            anchor_index=anchor_index,
            calibration=calibration,
            class_label=int(class_label),
            n_samples=int(n_samples),
            seed=int(seed),
            generation_mode=generation_mode,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidate_expert=candidate_expert,
            support_size=support_size,
            support_seed=support_seed,
        )

    if not direction_bank.valid_by_class.get(int(class_label), False):
        raise ProtocolError(direction_bank.invalid_reason_for((int(class_label),)) or STATUS_DIRECTION_BANK_INVALID)

    device = next(model.parameters()).device
    sample = _sample_direction_generation_payload(direction_bank, int(class_label), int(n_samples), int(seed))
    x_anchor_apply = sample["synthetic_anchors"].to(device)
    u_empirical = sample["directions"].to(device)
    direction_magnitude = sample["direction_magnitudes"].to(device)
    x_source_residual_ref = sample["residual_refs"].to(device)
    x_residual_anchor = sample["residual_anchors"].to(device)
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)
    decoder_noise = torch.zeros_like(x_anchor_apply)
    delta_logvar = torch.zeros_like(x_anchor_apply)
    scale_used = calibration.scale_for(int(class_label))
    posterior_or_prior_source = "source_train_residual_reference_posterior"

    with torch.no_grad():
        mu_z, _ = model.encode(x_source_residual_ref, x_residual_anchor, y)
        delta_mu, delta_logvar = model.decode_residual(mu_z, x_residual_anchor, y)
        gen = _generator_for_device(device, int(seed) + 104729)
        decoder_noise = torch.exp(0.5 * delta_logvar) * _randn_like(delta_mu, generator=gen)
        delta_cvae = float(scale_used) * (delta_mu + decoder_noise)
        cvae_magnitude = delta_cvae.norm(dim=1).clamp_min(1.0e-12)

        if generation_mode == F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE:
            direction = u_empirical
            magnitude = cvae_magnitude
        elif generation_mode == F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE_JITTER:
            direction = _jitter_direction(u_empirical, seed=int(seed) + 3847, scale=F21_DIRECTION_JITTER_SCALE)
            magnitude = cvae_magnitude
        elif generation_mode == F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE:
            direction = u_empirical
            magnitude = direction_magnitude
            posterior_or_prior_source = "empirical_direction_empirical_magnitude"
        elif generation_mode == F21_MODE_CVAE_DIRECTION_CVAE_MAGNITUDE:
            direction = _normalize_rows(delta_cvae)
            magnitude = cvae_magnitude
        else:
            raise ProtocolError(f"Unhandled F2.1 generation mode: {generation_mode}")
        embeddings = x_anchor_apply + direction * magnitude.unsqueeze(1)

    embeddings_cpu = embeddings.detach().cpu().float()
    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    direction_delta = sample["directions"].float() * sample["direction_magnitudes"].float().unsqueeze(1)
    diagnostics = _f21_generation_diagnostics(
        embeddings=embeddings_cpu,
        synthetic_anchors=sample["synthetic_anchors"].float(),
        direction_reference_embeddings=sample["direction_refs"].float(),
        direction_delta=direction_delta.float(),
        decoder_noise=decoder_noise.detach().cpu().float(),
        delta_logvar=delta_logvar.detach().cpu().float(),
        synthetic_anchor_ids=tuple(str(v) for v in sample["synthetic_anchor_ids"]),
        direction_reference_ids=tuple(str(v) for v in sample["direction_reference_ids"]),
        direction_anchor_ids=tuple(str(v) for v in sample["direction_anchor_ids"]),
        residual_reference_ids=tuple(str(v) for v in sample["residual_reference_ids"]),
        scale_used=float(scale_used),
    )
    provenance_rows = _f21_provenance_rows(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        support_size=support_size,
        support_seed=support_seed,
        generation_mode=generation_mode,
        generation_seed=seed,
        class_label=class_label,
        sample=sample,
        posterior_or_prior_source=posterior_or_prior_source,
        scale_used=float(scale_used),
    )
    return F21GeneratedBatch(
        embeddings=embeddings_cpu,
        labels=labels,
        generation_mode=generation_mode,
        diagnostics=diagnostics,
        provenance_rows=tuple(provenance_rows),
    )


def build_f21_downstream_matrix(
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
    direction_bank_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    sample_provenance_rows: list[dict[str, object]] = []

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
                raise ProtocolError(f"F2.1 expects binary labels 0/1, got {label_values}")

            for candidate in candidates:
                projection = _load_c41_projection(c41_artifacts_root, support.experiment_seed, candidate)
                train_projected_all = projection.transform(train_cache.embeddings)
                val_projected_all = projection.transform(val_cache.embeddings)
                candidate_train_idx = _indices_for_domain(train_cache.metadata, candidate)
                candidate_val_idx = _indices_for_domain(val_cache.metadata, candidate)
                if not candidate_train_idx or not candidate_val_idx:
                    raise ProtocolError(f"F2.1 requires nonempty source train/val rows for candidate={candidate}.")

                source_anchor_index = build_source_anchor_index(
                    source_projected_embeddings=train_projected_all,
                    source_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                    neighbor_k=ANCHOR_NEIGHBOR_K,
                )
                direction_bank = build_f21_direction_bank(
                    anchor_index=source_anchor_index,
                    label_values=label_values,
                    experiment_seed=int(support.experiment_seed),
                    heldout_center=heldout,
                    candidate_expert=candidate,
                )
                direction_bank_rows.extend(direction_bank.rows)
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
                ckpt = train_f2_anchored_residual_cvae(
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    out_dir=artifacts_root / "checkpoints" / f"seed{int(support.experiment_seed)}" / f"expert_{candidate}" / "f21_no_energy_cov_penalty",
                    model_name="f21_direction_preserving_no_energy_cov_penalty_pca64",
                    hidden_dim=training_profile.hidden_dim,
                    latent_dim=training_profile.latent_dim,
                    lr=training_profile.lr,
                    epochs=training_profile.epochs,
                    patience=training_profile.patience,
                    batch_size=training_profile.batch_size,
                    device=device,
                    resume=resume,
                    use_moment_penalties=False,
                    checkpoint_metadata={
                        "generator_family": F21_GENERATOR_FAMILY,
                        "experiment_id": "F2.1",
                        "experiment_seed": int(support.experiment_seed),
                        "candidate_expert": str(candidate),
                        "model_variant": "no_energy_cov_penalty",
                        "anchor_strategy": "source_train_same_class_nn",
                        "direction_strategy": "source_train_same_class_empirical_residual_direction",
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
                    model_variant="f21_no_energy_cov_penalty",
                    device=device,
                )
                calibration_rows.extend(_f21_calibration_rows(calibration.rows))
                provenance_rows.append(
                    {
                        "experiment_seed": int(support.experiment_seed),
                        "heldout_center": heldout,
                        "candidate_expert": candidate,
                        "model_variant": "no_energy_cov_penalty",
                        "generator_family": F21_GENERATOR_FAMILY,
                        "checkpoint_path": str(ckpt),
                        "projection_path": str(_c41_projection_path(c41_artifacts_root, support.experiment_seed, candidate)),
                        "projection_source": "reused_c41_full_source_train_pca64",
                        "generation_conditioning": "source_train_residual_reference_posterior",
                        "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                        "routing_scores_recomputed_for_f21": 0,
                        "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                    }
                )

                for generation_mode in F21_GENERATION_MODES:
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            base_row, diagnostics, duplicates, sample_rows = _score_f21_candidate(
                                model=model,
                                calibration=calibration,
                                projection=projection,
                                anchor_index=source_anchor_index,
                                direction_bank=direction_bank,
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
                            geometry_rows.append(_f21_geometry_row(diagnostics))
                            duplicate_rows.append(duplicates)
                            sample_provenance_rows.extend(sample_rows)
                            for support_size, support_seed in support_conditions:
                                row = replace(base_row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and row.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, row)
                                completed.add(row.primary_key())

    _write_csv_with_header(artifacts_root / "tables" / "f21_anchor_pair_diagnostics.csv", anchor_diag_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_direction_bank_diagnostics.csv", direction_bank_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_magnitude_calibration_diagnostics.csv", calibration_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_geometry_diagnostics.csv", geometry_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_duplicate_diagnostics.csv", duplicate_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_synthetic_sample_provenance.csv", sample_provenance_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f21_generator_provenance.csv", provenance_rows)
    _write_csv_with_header(artifacts_root / "manifests" / "f21_generator_provenance.csv", provenance_rows)
    return matrix_path


def build_f21_routing_alignment_rows(
    *,
    selections: Sequence[SupportSelectionUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    f21_rows = [row for row in downstream_rows if row.generator_family == F21_GENERATOR_FAMILY]
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
        for row in f21_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE
    }
    ok_rows = [row for row in f21_rows if row.status == "ok"]
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
            for row in f21_rows
            if int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        }
    )
    oracle_by_context = _f21_oracles(ok_rows)
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
                raise ProtocolError(f"Missing F2.1 downstream row for selected expert key {selected_key}")
            oracle = oracle_by_context.get((experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed))
            selected_ok = selected.status == "ok"
            oracle_gap = float(oracle.bacc) - float(selected.bacc) if oracle is not None and selected_ok else math.nan
            invalid = selected.status == STATUS_DIRECTION_BANK_INVALID
            rows.append(
                {
                    "heldout_center": unit.heldout_center,
                    "experiment_seed": int(unit.experiment_seed),
                    "support_size": int(unit.support_size),
                    "support_seed": int(unit.support_seed),
                    "generator_family": F21_GENERATOR_FAMILY,
                    "generation_mode": generation_mode,
                    "generation_seed": int(generation_seed),
                    "classifier_seed": int(classifier_seed),
                    "method": unit.method,
                    "selected_expert": unit.selected_expert,
                    "selected_status": selected.status,
                    "selected_bacc": float(selected.bacc),
                    "selected_macro_f1": float(selected.macro_f1),
                    "downstream_oracle_expert": oracle.candidate_expert if oracle is not None else "",
                    "oracle_bacc": float(oracle.bacc) if oracle is not None else math.nan,
                    "oracle_macro_f1": float(oracle.macro_f1) if oracle is not None else math.nan,
                    "downstream_oracle_gap_bacc": oracle_gap,
                    "downstream_oracle_gap_macro_f1": (float(oracle.macro_f1) - float(selected.macro_f1)) if oracle is not None and selected_ok else math.nan,
                    "relative_downstream_oracle_gap_pct": (oracle_gap / float(oracle.bacc)) * 100.0 if oracle is not None and float(oracle.bacc) else math.nan,
                    "top1_downstream_hit": int(selected_ok and oracle is not None and str(unit.selected_expert) == str(oracle.candidate_expert)),
                    "spearman_neg_nelbo_vs_bacc": math.nan,
                    "metadata_bacc": math.nan,
                    "delta_vs_metadata": math.nan,
                    "selection_depends_on_support": 1,
                    "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                    "routing_scores_recomputed_for_f21": 0,
                    "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                    "projection_source": "reused_c41_full_source_train_pca64",
                    "generation_conditioning": "source_train_residual_reference_posterior",
                    "decision_label": STATUS_DIRECTION_BANK_INVALID if invalid else "",
                }
            )
    return rows


def build_f21_delta_summary_rows(
    *,
    f21_alignment_rows: Sequence[Mapping[str, object]],
    f2_alignment_rows: Sequence[Mapping[str, object]],
    duplicate_rows: Sequence[Mapping[str, object]] = (),
    geometry_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    f21_support = [row for row in f21_alignment_rows if str(row.get("method")) == SUPPORT_NELBO_METHOD]
    f2_baseline = [
        row
        for row in f2_alignment_rows
        if str(row.get("method")) == SUPPORT_NELBO_METHOD
        and str(row.get("generation_mode")) == F2_MODE_CALIBRATED_NOISE_NO_PENALTY
    ]
    groups = sorted({(str(row["heldout_center"]), int(row["support_size"])) for row in f21_support})
    strong_centers = _strong_centers_from_f2_transfer(f2_alignment_rows)
    rows: list[dict[str, object]] = []
    for heldout, support_size in groups:
        f2_subset = _subset(f2_baseline, heldout, support_size, F2_MODE_CALIBRATED_NOISE_NO_PENALTY)
        if not f2_subset:
            continue
        f2_selected = _mean(f2_subset, "selected_bacc")
        f2_oracle = _mean(f2_subset, "oracle_bacc")
        f2_gap = _mean(f2_subset, "downstream_oracle_gap_bacc")
        empirical_direction_selected = _mean(_valid_subset(f21_support, heldout, support_size, F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE), "selected_bacc")
        transfer_selected = _mean(_valid_subset(f21_support, heldout, support_size, F21_MODE_TRANSFER_BOOTSTRAP), "selected_bacc")
        for mode in F21_GENERATION_MODES:
            mode_subset_all = _subset(f21_support, heldout, support_size, mode)
            mode_subset = [row for row in mode_subset_all if str(row.get("selected_status") or "ok") == "ok"]
            if not mode_subset_all:
                continue
            invalid_count = len(mode_subset_all) - len(mode_subset)
            selected = _mean(mode_subset, "selected_bacc")
            oracle = _mean(mode_subset, "oracle_bacc")
            gap = _mean(mode_subset, "downstream_oracle_gap_bacc")
            deltas = [float(row.get("selected_bacc", math.nan)) - f2_selected for row in mode_subset]
            clean_deltas = [value for value in deltas if not math.isnan(value)]
            improvement_rate = sum(1 for value in clean_deltas if value > 0.0) / float(max(len(clean_deltas), 1))
            seed_gain_rate = _seed_gain_rate(mode_subset, f2_subset)
            near_copy = _f21_near_copy_failure(duplicate_rows, heldout, mode)
            direction_concentration = _f21_direction_concentration_failure(duplicate_rows, heldout, mode)
            mechanistic = _f21_mechanistic_improved(geometry_rows, heldout, mode)
            strong_center_degraded = int(heldout in strong_centers and selected < strong_centers[heldout] - 0.02)
            if invalid_count and not mode_subset:
                decision = STATUS_DIRECTION_BANK_INVALID
            else:
                decision = _f21_decision_label(
                    mode=mode,
                    selected=selected,
                    f2_selected=f2_selected,
                    empirical_direction_selected=empirical_direction_selected,
                    transfer_selected=transfer_selected,
                    gap=gap,
                    f2_gap=f2_gap,
                    improvement_rate=improvement_rate,
                    seed_gain_rate=seed_gain_rate,
                    strong_center_degraded=bool(strong_center_degraded),
                    near_copy=near_copy,
                    direction_concentration=direction_concentration,
                )
            rows.append(
                {
                    "heldout_center": heldout,
                    "support_size": support_size,
                    "generation_mode": mode,
                    "selected_bacc_f21": selected,
                    "oracle_bacc_f21": oracle,
                    "oracle_gap_f21": gap,
                    "selected_bacc_f2_no_penalty_calibrated_noise": f2_selected,
                    "oracle_bacc_f2_no_penalty_calibrated_noise": f2_oracle,
                    "oracle_gap_f2_no_penalty_calibrated_noise": f2_gap,
                    "selected_bacc_delta_vs_f2_no_penalty_calibrated_noise": selected - f2_selected,
                    "oracle_bacc_delta_vs_f2_no_penalty_calibrated_noise": oracle - f2_oracle,
                    "oracle_gap_delta_vs_f2_no_penalty_calibrated_noise": gap - f2_gap,
                    "selected_bacc_anchor_empirical_direction_empirical_magnitude": empirical_direction_selected,
                    "selected_bacc_anchor_empirical_residual_transfer_bootstrap": transfer_selected,
                    "beats_empirical_direction_empirical_magnitude": int(selected > empirical_direction_selected),
                    "beats_transfer_bootstrap": int(selected > transfer_selected),
                    "selected_ge_080": int(selected >= 0.80),
                    "valid_cell_count": len(mode_subset),
                    "invalid_cell_count": invalid_count,
                    "center_seed_improvement_rate": improvement_rate,
                    "seed_gain_rate": seed_gain_rate,
                    "strong_center_degraded": strong_center_degraded,
                    "near_copy_failure": int(near_copy),
                    "direction_concentration_failure": int(direction_concentration),
                    "mechanistic_direction_or_margin_improved": int(mechanistic),
                    "diagnostic_only": int(mode in F21_DIAGNOSTIC_ONLY_MODES),
                    "decision_label": decision,
                }
            )
    return rows


def write_f21_alignment_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F21_ALIGNMENT_COLUMNS, rows)


def write_f21_delta_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F21_DELTA_COLUMNS, rows)


def load_f21_diagnostics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _score_f21_candidate(
    *,
    model: AnchoredResidualCVAE,
    calibration: ResidualCalibration,
    projection: SourceTrainPCAProjection,
    anchor_index: SourceAnchorIndex,
    direction_bank: DirectionBank,
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
) -> tuple[CandidateDownstreamRow, dict[str, object], dict[str, object], list[dict[str, object]]]:
    try:
        if generation_mode in F21_DIRECTION_MODES and not direction_bank.valid_for(label_values):
            raise _DirectionBankInvalid(direction_bank.invalid_reason_for(label_values))
        chunks: list[torch.Tensor] = []
        labels: list[int] = []
        diagnostic_parts: list[Mapping[str, float]] = []
        provenance_rows: list[dict[str, object]] = []
        for label in label_values:
            generated = generate_f21_direction_preserving_embeddings(
                model=model,
                anchor_index=anchor_index,
                direction_bank=direction_bank,
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
            "generator_family": F21_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "diagnostic_only": int(generation_mode in F21_DIAGNOSTIC_ONLY_MODES),
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
        duplicate = _f21_duplicate_row(diagnostics)
        row = _f21_candidate_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidate_expert=candidate_expert,
            generation_mode=generation_mode,
            budget_per_class=budget_per_class,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
            target_eval_pool=target_eval_pool,
            label_values=label_values,
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
        )
        return row, diagnostics, duplicate, provenance_rows
    except _DirectionBankInvalid as exc:
        row = _f21_candidate_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidate_expert=candidate_expert,
            generation_mode=generation_mode,
            budget_per_class=budget_per_class,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
            target_eval_pool=target_eval_pool,
            label_values=label_values,
            bacc=math.nan,
            macro_f1=math.nan,
            status=STATUS_DIRECTION_BANK_INVALID,
            error_message=str(exc),
        )
        diagnostics = _f21_failure_diagnostics(row)
        return row, diagnostics, dict(diagnostics), []
    except Exception as exc:
        row = _f21_candidate_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidate_expert=candidate_expert,
            generation_mode=generation_mode,
            budget_per_class=budget_per_class,
            generation_seed=generation_seed,
            classifier_seed=classifier_seed,
            target_eval_pool=target_eval_pool,
            label_values=label_values,
            bacc=math.nan,
            macro_f1=math.nan,
            status="failed_f21_candidate_scoring",
            error_message=str(exc),
        )
        diagnostics = _f21_failure_diagnostics(row)
        return row, diagnostics, dict(diagnostics), []


def _generate_f21_replay_control(
    *,
    model: AnchoredResidualCVAE,
    anchor_index: SourceAnchorIndex,
    calibration: ResidualCalibration,
    class_label: int,
    n_samples: int,
    seed: int,
    generation_mode: str,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    support_size: int,
    support_seed: int,
) -> F21GeneratedBatch:
    if generation_mode == F21_MODE_F2_NO_PENALTY_REPLAY:
        f2_mode = F2_MODE_CALIBRATED_NOISE_NO_PENALTY
    elif generation_mode == F21_MODE_TRANSFER_BOOTSTRAP:
        f2_mode = F2_MODE_TRANSFER_BOOTSTRAP
    elif generation_mode == F21_MODE_IDENTITY_BOOTSTRAP:
        f2_mode = F2_MODE_IDENTITY_BOOTSTRAP
    else:
        raise ProtocolError(f"Unsupported F2.1 replay control: {generation_mode}")
    generated = generate_f2_anchor_residual_embeddings(
        model=model,
        anchor_index=anchor_index,
        calibration=calibration,
        class_label=int(class_label),
        n_samples=int(n_samples),
        seed=int(seed),
        generation_mode=f2_mode,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        support_size=support_size,
        support_seed=support_seed,
    )
    rows: list[dict[str, object]] = []
    for row in generated.provenance_rows:
        rewritten = dict(row)
        rewritten["generation_mode"] = generation_mode
        rewritten["direction_reference_sample_id"] = ""
        rewritten["direction_anchor_id"] = ""
        rewritten["direction_reference_split"] = ""
        rewritten["direction_anchor_split"] = ""
        rewritten["routing_scores_recomputed_for_f21"] = 0
        rewritten["selected_expert_ids_source"] = BASELINE_SELECTED_EXPERT_IDS_SOURCE
        rows.append(rewritten)
    diagnostics = dict(generated.diagnostics)
    diagnostics["generation_mode"] = generation_mode
    diagnostics["top_5_direction_reference_share_per_class"] = 0.0
    diagnostics["direction_reference_reuse_rate"] = 0.0
    diagnostics["fraction_same_direction_and_anchor_pair_reused"] = 0.0
    diagnostics["hard_direction_concentration_failure"] = 0.0
    diagnostics["residual_magnitude_ratio_to_real"] = diagnostics.get("mean_interpolation_ratio", math.nan)
    return F21GeneratedBatch(
        embeddings=generated.embeddings,
        labels=generated.labels,
        generation_mode=generation_mode,
        diagnostics=diagnostics,
        provenance_rows=tuple(rows),
    )


def _sample_direction_generation_payload(
    direction_bank: DirectionBank,
    class_label: int,
    n_samples: int,
    seed: int,
) -> dict[str, object]:
    anchor_index = direction_bank.anchor_index
    class_rows = [idx for idx, label in enumerate(anchor_index.labels.tolist()) if int(label) == int(class_label)]
    directions = direction_bank.directions_by_class[int(class_label)]
    magnitudes = direction_bank.magnitudes_by_class[int(class_label)]
    ref_indices = direction_bank.reference_local_indices_by_class[int(class_label)]
    anchor_indices = direction_bank.anchor_local_indices_by_class[int(class_label)]
    if len(class_rows) < 4:
        raise ProtocolError(f"F2.1 strict self-exclusion requires at least four source-train rows for class={class_label}")
    if int(directions.shape[0]) <= 0:
        raise _DirectionBankInvalid(f"empty direction bank for class={class_label}")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    synthetic_anchors: list[torch.Tensor] = []
    sampled_dirs: list[torch.Tensor] = []
    sampled_magnitudes: list[torch.Tensor] = []
    direction_refs: list[torch.Tensor] = []
    residual_refs: list[torch.Tensor] = []
    residual_anchors: list[torch.Tensor] = []
    synthetic_anchor_ids: list[str] = []
    direction_reference_ids: list[str] = []
    direction_anchor_ids: list[str] = []
    residual_reference_ids: list[str] = []
    residual_anchor_ids: list[str] = []

    for _ in range(int(n_samples)):
        chosen = _sample_strict_direction_indices(
            class_rows=class_rows,
            ref_indices=ref_indices,
            anchor_indices=anchor_indices,
            sample_ids=anchor_index.sample_ids,
            generator=gen,
        )
        synthetic_anchor_local, direction_pos, residual_pos = chosen
        dir_ref_local = int(ref_indices[direction_pos])
        dir_anchor_local = int(anchor_indices[direction_pos])
        res_ref_local = int(ref_indices[residual_pos])
        res_anchor_local = int(anchor_indices[residual_pos])

        synthetic_anchors.append(anchor_index.embeddings[synthetic_anchor_local])
        sampled_dirs.append(directions[direction_pos])
        sampled_magnitudes.append(magnitudes[direction_pos])
        direction_refs.append(anchor_index.embeddings[dir_ref_local])
        residual_refs.append(anchor_index.embeddings[res_ref_local])
        residual_anchors.append(anchor_index.embeddings[res_anchor_local])
        synthetic_anchor_ids.append(anchor_index.sample_ids[synthetic_anchor_local])
        direction_reference_ids.append(anchor_index.sample_ids[dir_ref_local])
        direction_anchor_ids.append(anchor_index.sample_ids[dir_anchor_local])
        residual_reference_ids.append(anchor_index.sample_ids[res_ref_local])
        residual_anchor_ids.append(anchor_index.sample_ids[res_anchor_local])

    return {
        "synthetic_anchors": torch.stack(synthetic_anchors).float(),
        "directions": torch.stack(sampled_dirs).float(),
        "direction_magnitudes": torch.stack(sampled_magnitudes).float(),
        "direction_refs": torch.stack(direction_refs).float(),
        "residual_refs": torch.stack(residual_refs).float(),
        "residual_anchors": torch.stack(residual_anchors).float(),
        "synthetic_anchor_ids": tuple(synthetic_anchor_ids),
        "direction_reference_ids": tuple(direction_reference_ids),
        "direction_anchor_ids": tuple(direction_anchor_ids),
        "residual_reference_ids": tuple(residual_reference_ids),
        "residual_anchor_ids": tuple(residual_anchor_ids),
    }


def _sample_strict_direction_indices(
    *,
    class_rows: Sequence[int],
    ref_indices: Sequence[int],
    anchor_indices: Sequence[int],
    sample_ids: Sequence[str],
    generator: torch.Generator,
) -> tuple[int, int, int]:
    n_dirs = len(ref_indices)
    for _attempt in range(2048):
        synthetic_anchor_local = int(class_rows[int(torch.randint(len(class_rows), (1,), generator=generator).item())])
        direction_pos = int(torch.randint(n_dirs, (1,), generator=generator).item())
        residual_pos = int(torch.randint(n_dirs, (1,), generator=generator).item())
        synthetic_id = str(sample_ids[synthetic_anchor_local])
        direction_ref_id = str(sample_ids[int(ref_indices[direction_pos])])
        direction_anchor_id = str(sample_ids[int(anchor_indices[direction_pos])])
        residual_ref_id = str(sample_ids[int(ref_indices[residual_pos])])
        if synthetic_id == direction_ref_id:
            continue
        if synthetic_id == direction_anchor_id:
            continue
        if synthetic_id == residual_ref_id:
            continue
        if direction_ref_id == residual_ref_id:
            continue
        return synthetic_anchor_local, direction_pos, residual_pos
    raise ProtocolError("Unable to sample F2.1 strict self-excluded direction/residual references")


def _jitter_direction(direction: torch.Tensor, *, seed: int, scale: float) -> torch.Tensor:
    gen = _generator_for_device(direction.device, int(seed))
    noise = _randn_like(direction, generator=gen)
    projection = (noise * direction).sum(dim=1, keepdim=True) * direction
    orthogonal = noise - projection
    orthogonal = _normalize_rows(orthogonal)
    return _normalize_rows(direction + float(scale) * orthogonal)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    return x / norm


def _f21_generation_diagnostics(
    *,
    embeddings: torch.Tensor,
    synthetic_anchors: torch.Tensor,
    direction_reference_embeddings: torch.Tensor,
    direction_delta: torch.Tensor,
    delta_logvar: torch.Tensor,
    decoder_noise: torch.Tensor,
    synthetic_anchor_ids: Sequence[str],
    direction_reference_ids: Sequence[str],
    direction_anchor_ids: Sequence[str],
    residual_reference_ids: Sequence[str],
    scale_used: float,
) -> dict[str, float]:
    gen_residual = embeddings - synthetic_anchors
    dist_anchor = gen_residual.norm(dim=1)
    dist_ref = (embeddings - direction_reference_embeddings).norm(dim=1)
    ref_norm = direction_delta.norm(dim=1).clamp_min(1.0e-12)
    generated_norm = gen_residual.norm(dim=1)
    logvar_at_min = (delta_logvar <= -9.21 + 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    logvar_at_max = (delta_logvar >= 2.0 - 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    noise_energy = decoder_noise.pow(2).sum(dim=1).mean() if decoder_noise.numel() else torch.tensor(0.0)
    mean_energy = embeddings.pow(2).sum(dim=1).mean().clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(gen_residual, direction_delta, dim=1).mean() if gen_residual.numel() else torch.tensor(float("nan"))
    top5_direction_share = _top5_string_share(direction_reference_ids)
    direction_anchor_triplets = [
        f"{anchor}|{ref}|{direction_anchor}"
        for anchor, ref, direction_anchor in zip(synthetic_anchor_ids, direction_reference_ids, direction_anchor_ids)
    ]
    same_triplet_reuse = 1.0 - (float(len(set(direction_anchor_triplets))) / float(max(len(direction_anchor_triplets), 1)))
    hard_direction = int(
        top5_direction_share > F21_TOP5_DIRECTION_SHARE_FAILURE_THRESHOLD
        or same_triplet_reuse > F21_SAME_DIRECTION_ANCHOR_REUSE_FAILURE_THRESHOLD
    )
    return {
        "min_dist_to_anchor": float(dist_anchor.min().item()),
        "min_dist_to_reference_source_sample": float(dist_ref.min().item()),
        "fraction_exact_or_near_duplicate_anchor": float((dist_anchor <= NEAR_DUPLICATE_EPS).float().mean().item()),
        "fraction_exact_or_near_duplicate_reference": float((dist_ref <= NEAR_DUPLICATE_EPS).float().mean().item()),
        "mean_interpolation_ratio": float((generated_norm / ref_norm).mean().item()),
        "anchor_reuse_rate": 1.0 - (float(len(set(str(v) for v in synthetic_anchor_ids))) / float(max(len(synthetic_anchor_ids), 1))),
        "direction_reference_reuse_rate": 1.0 - (float(len(set(str(v) for v in direction_reference_ids))) / float(max(len(direction_reference_ids), 1))),
        "top_5_direction_reference_share_per_class": top5_direction_share,
        "fraction_same_direction_and_anchor_pair_reused": same_triplet_reuse,
        "hard_direction_concentration_failure": hard_direction,
        "real_pair_delta_norm_mean": float(ref_norm.mean().item()),
        "generated_residual_norm_mean": float(generated_norm.mean().item()),
        "residual_magnitude_ratio_to_real": float((generated_norm / ref_norm).mean().item()),
        "residual_energy_ratio": float(gen_residual.pow(2).sum(dim=1).mean().item() / max(direction_delta.pow(2).sum(dim=1).mean().item(), 1.0e-12)),
        "residual_cov_trace_ratio": float(_trace_cov(gen_residual) / max(_trace_cov(direction_delta), 1.0e-12)),
        "residual_direction_cosine_real_vs_synthetic": float(cosine.item()),
        "decoder_logvar_mean": float(delta_logvar.mean().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_min": float(delta_logvar.min().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_max": float(delta_logvar.max().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_at_min_frac": float(logvar_at_min.item()),
        "decoder_logvar_at_max_frac": float(logvar_at_max.item()),
        "decoder_noise_energy_ratio": float((noise_energy / mean_energy).item()),
        "residual_calibration_scale_used": float(scale_used),
        "direction_conditioning_is_source_train_only": 1.0,
        "strict_self_exclusion_enforced": 1.0,
    }


def _f21_provenance_rows(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    support_size: int,
    support_seed: int,
    generation_mode: str,
    generation_seed: int,
    class_label: int,
    sample: Mapping[str, object],
    posterior_or_prior_source: str,
    scale_used: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    synthetic_anchor_ids = tuple(str(v) for v in sample["synthetic_anchor_ids"])
    direction_reference_ids = tuple(str(v) for v in sample["direction_reference_ids"])
    direction_anchor_ids = tuple(str(v) for v in sample["direction_anchor_ids"])
    residual_reference_ids = tuple(str(v) for v in sample["residual_reference_ids"])
    residual_anchor_ids = tuple(str(v) for v in sample["residual_anchor_ids"])
    for idx, synthetic_anchor_id in enumerate(synthetic_anchor_ids):
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
                "synthetic_anchor_id": synthetic_anchor_id,
                "direction_reference_sample_id": direction_reference_ids[idx],
                "direction_anchor_id": direction_anchor_ids[idx],
                "residual_reference_sample_id": residual_reference_ids[idx],
                "residual_anchor_id": residual_anchor_ids[idx],
                "anchor_split": "source_train",
                "direction_reference_split": "source_train",
                "direction_anchor_split": "source_train",
                "residual_reference_split": "source_train",
                "calibration_split": "source_val",
                "same_class_anchor": 1,
                "posterior_or_prior_source": posterior_or_prior_source,
                "generation_conditioning": "source_train_residual_reference_posterior",
                "residual_calibration_scale_used": float(scale_used),
                "routing_scores_recomputed_for_f21": 0,
                "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
        )
    return rows


def _f21_candidate_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    generation_mode: str,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    target_eval_pool: TargetEvalPool,
    label_values: Sequence[int],
    bacc: float,
    macro_f1: float,
    auroc: float = math.nan,
    auprc: float = math.nan,
    status: str = "ok",
    error_message: str = "",
) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        experiment_seed=int(experiment_seed),
        heldout_center=heldout_center,
        support_size=0,
        support_seed=0,
        candidate_expert=candidate_expert,
        generator_family=F21_GENERATOR_FAMILY,
        generation_mode=generation_mode,
        budget_per_class=int(budget_per_class),
        generation_seed=int(generation_seed),
        classifier_seed=int(classifier_seed),
        bacc=float(bacc),
        macro_f1=float(macro_f1),
        auroc=float(auroc),
        auprc=float(auprc),
        row_type=SINGLE_EXPERT_ROW_TYPE,
        n_synthetic_train=int(budget_per_class) * len(label_values),
        n_target_eval=len(target_eval_pool.eval_indices),
        target_eval_pool_id=target_eval_pool.target_eval_pool_id,
        candidate_experts_hash=SINGLE_EXPERT_HASH,
        utility_depends_on_support=0,
        selection_depends_on_support=0,
        plain_baseline_source="f21_uses_reused_c41_projection",
        plain_baseline_artifact_path="",
        plain_baseline_training_profile="f21_direction_preserving_residual",
        plain_baseline_matches_locked_hparams=0,
        routing_family_used=BASELINE_ROUTING_FAMILY_USED,
        routing_scores_recomputed_for_heteroscedastic=0,
        selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        status=status,
        error_message=error_message,
    )


def _f21_failure_diagnostics(row: CandidateDownstreamRow) -> dict[str, object]:
    return {
        "experiment_seed": int(row.experiment_seed),
        "heldout_center": row.heldout_center,
        "candidate_expert": row.candidate_expert,
        "generator_family": F21_GENERATOR_FAMILY,
        "generation_mode": row.generation_mode,
        "generation_seed": int(row.generation_seed),
        "classifier_seed": int(row.classifier_seed),
        "status": row.status,
        "error_message": row.error_message,
    }


def _f21_duplicate_row(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
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
        "direction_reference_reuse_rate",
        "top_5_direction_reference_share_per_class",
        "fraction_same_direction_and_anchor_pair_reused",
        "median_nn_dist_synthetic_to_source_train",
        "median_nn_dist_source_val_to_source_train",
        "median_nn_copy_ratio",
        "top_5_source_nn_share_per_class",
        "hard_near_copy_failure",
        "hard_direction_concentration_failure",
    )
    return {key: row.get(key, math.nan) for key in keys}


def _f21_geometry_row(row: Mapping[str, object]) -> dict[str, object]:
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
        "residual_magnitude_ratio_to_real",
        "residual_energy_ratio",
        "residual_cov_trace_ratio",
        "residual_direction_cosine_real_vs_synthetic",
        "residual_eigenvalue_topk_ratio",
        "within_class_cov_frobenius_error",
        "class_centroid_shift_error",
        "between_class_margin_preservation",
        "synthetic_to_train_nn_dist_p10",
        "synthetic_to_train_nn_dist_p50",
        "synthetic_to_train_nn_dist_p90",
        "anchor_to_synthetic_nn_distance_ratio",
    )
    return {key: row.get(key, math.nan) for key in keys}


def _f21_calibration_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["generator_family"] = F21_GENERATOR_FAMILY
        item["calibration_type"] = "magnitude_only"
        item["scale_min"] = CALIBRATION_MIN_SCALE
        item["scale_max"] = CALIBRATION_MAX_SCALE
        out.append(item)
    return out


def _f21_oracles(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[int, str, str, int, int, int], CandidateDownstreamRow]:
    grouped: dict[tuple[int, str, str, int, int, int], list[CandidateDownstreamRow]] = {}
    for row in rows:
        if row.generator_family != F21_GENERATOR_FAMILY or row.status != "ok":
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


def _valid_subset(rows: Sequence[Mapping[str, object]], heldout: str, support_size: int, mode: str) -> list[Mapping[str, object]]:
    return [row for row in _subset(rows, heldout, support_size, mode) if str(row.get("selected_status") or "ok") == "ok"]


def _strong_centers_from_f2_transfer(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("method")) != SUPPORT_NELBO_METHOD:
            continue
        if str(row.get("generation_mode")) != F21_MODE_TRANSFER_BOOTSTRAP:
            continue
        value = _as_float(row.get("selected_bacc", math.nan))
        if math.isnan(value):
            continue
        values.setdefault(str(row.get("heldout_center")), []).append(value)
    return {
        heldout: sum(vals) / float(len(vals))
        for heldout, vals in values.items()
        if vals and sum(vals) / float(len(vals)) >= 0.78
    }


def _seed_gain_rate(mode_subset: Sequence[Mapping[str, object]], f2_subset: Sequence[Mapping[str, object]]) -> float:
    f2_by_seed: dict[int, list[float]] = {}
    for row in f2_subset:
        seed = int(row.get("experiment_seed", 0))
        value = _as_float(row.get("selected_bacc", math.nan))
        if not math.isnan(value):
            f2_by_seed.setdefault(seed, []).append(value)
    mode_by_seed: dict[int, list[float]] = {}
    for row in mode_subset:
        seed = int(row.get("experiment_seed", 0))
        value = _as_float(row.get("selected_bacc", math.nan))
        if not math.isnan(value):
            mode_by_seed.setdefault(seed, []).append(value)
    gains = []
    for seed, values in mode_by_seed.items():
        if seed not in f2_by_seed:
            continue
        gains.append((sum(values) / float(len(values))) > (sum(f2_by_seed[seed]) / float(len(f2_by_seed[seed]))))
    return sum(1 for value in gains if value) / float(len(gains)) if gains else math.nan


def _f21_near_copy_failure(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not subset:
        return False
    hard = max(_as_float(row.get("hard_near_copy_failure", 0)) for row in subset)
    return bool(hard >= 1)


def _f21_direction_concentration_failure(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not subset:
        return False
    hard = max(_as_float(row.get("hard_direction_concentration_failure", 0)) for row in subset)
    return bool(hard >= 1)


def _f21_mechanistic_improved(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    baseline = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == F21_MODE_F2_NO_PENALTY_REPLAY]
    current = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not baseline or not current:
        return False
    return bool(
        _mean(current, "residual_direction_cosine_real_vs_synthetic") > _mean(baseline, "residual_direction_cosine_real_vs_synthetic")
        or _mean(current, "between_class_margin_preservation") > _mean(baseline, "between_class_margin_preservation")
    )


def _f21_decision_label(
    *,
    mode: str,
    selected: float,
    f2_selected: float,
    empirical_direction_selected: float,
    transfer_selected: float,
    gap: float,
    f2_gap: float,
    improvement_rate: float,
    seed_gain_rate: float,
    strong_center_degraded: bool,
    near_copy: bool,
    direction_concentration: bool,
) -> str:
    if mode != F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE:
        return DECISION_DIAGNOSTIC_ONLY
    if near_copy:
        return DECISION_NEAR_COPY_FAILURE
    if direction_concentration:
        return DECISION_DIRECTION_CONCENTRATION_FAILURE
    repair = (
        selected > f2_selected
        and improvement_rate >= 0.60
        and (math.isnan(seed_gain_rate) or seed_gain_rate >= (2.0 / 3.0))
        and not strong_center_degraded
    )
    if not repair and selected > f2_selected and not (math.isnan(seed_gain_rate) or seed_gain_rate >= (2.0 / 3.0)):
        return DECISION_SEED_UNSTABLE_GAIN
    if selected <= empirical_direction_selected:
        return DECISION_MAGNITUDE_MODEL_NO_GAIN
    if repair and selected > transfer_selected and gap <= f2_gap:
        return DECISION_TRANSFER_SUPERIORITY_SUCCESS
    if repair:
        return DECISION_DIRECTION_REPAIR_SUCCESS
    if selected <= transfer_selected:
        return DECISION_NO_GAIN_DIRECTION_BANK_ONLY
    return DECISION_NO_GAIN_DIRECTION_BANK_ONLY


def _effective_sample_size(ids: Sequence[str]) -> float:
    if not ids:
        return 0.0
    counts: dict[str, int] = {}
    for item in ids:
        counts[str(item)] = counts.get(str(item), 0) + 1
    total = float(len(ids))
    return 1.0 / sum((count / total) ** 2 for count in counts.values())


def _top5_string_share(ids: Sequence[str]) -> float:
    if not ids:
        return 0.0
    counts: dict[str, int] = {}
    for item in ids:
        counts[str(item)] = counts.get(str(item), 0) + 1
    return float(sum(sorted(counts.values(), reverse=True)[:5])) / float(len(ids))


class _DirectionBankInvalid(Exception):
    pass
