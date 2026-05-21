"""C7.1a source-probe CE geometry-regularized CVAE diagnostic.

C7.1a is a generator-objective diagnostic. It retrains the C4.1
heteroscedastic class-conditioned PCA64 CVAE under the same source-only split
discipline, then adds one conservative auxiliary loss: a frozen source-trained
linear probe is applied to posterior-mean decoder outputs. Routing and late
composition are fixed; target labels are consumed only for final metrics.
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
    GENERATOR_FAMILY_HETEROSCEDASTIC,
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
from .c62_late_ensemble import (
    GLOBAL_CLASS_ORDER,
    _fit_member_probabilities,
    _score_predictions_and_probabilities,
    align_probabilities_to_class_order,
)
from .c63_geometric_ensemble import geometric_pool_probabilities
from .matrix import (
    MatrixBuildLimits,
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _records_for_split,
    _resolve_torch_device,
    _to_numpy,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import ENSEMBLE_EXPERT_ID, SUPPORT_NELBO_METHOD


C71A_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c71a_source_probe_ce_cvae_v1"
C71A_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
C71A_DEFAULT_C63_ROOT = "cvae_downstream_evaluation/artifacts/c63_geometric_late_ensemble_v1"
C71A_GENERATOR_FAMILY = "family_c_pca64_class_conditional_source_probe_ce_cvae_downstream_v1"

VARIANT_BASE = "C7.1_base"
VARIANT_SOURCE_PROBE_CE = "C7.1_source_probe_ce"
VARIANT_C63_ORIGINAL_HETERO_MEAN_REPLAY = "C6.3_original_c41_hetero_mean_only_replay"
C71A_VARIANTS = (VARIANT_C63_ORIGINAL_HETERO_MEAN_REPLAY, VARIANT_BASE, VARIANT_SOURCE_PROBE_CE)

LAMBDA_CLS = 0.05
AUX_WARMUP_EPOCHS = 5
AUX_RAMP_EPOCHS = 10
PROBE_EPOCHS = 100
PROBE_LR = 1.0e-2

COLLAPSE_EFFECTIVE_RANK_RATIO_MIN = 0.70
COLLAPSE_COV_TRACE_RATIO_MIN = 0.60
COLLAPSE_NN_CONCENTRATION_RATIO_MAX = 2.0

FAILURE_SOURCE_GEOMETRY_NOT_TARGET_UTILITY = "SOURCE_GEOMETRY_NOT_TARGET_UTILITY"
FAILURE_PROBE_GAIN_DOWNSTREAM_DROP = "SOURCE_PROBE_BACC_GAIN_WITH_DOWNSTREAM_DROP"
FAILURE_GEOMETRY_COLLAPSE = "AUX_GEOMETRY_COLLAPSES_CLASS_VARIANCE"
FAILURE_PROBE_WEAK = "SOURCE_PROBE_TOO_WEAK_FOR_GEOMETRY_CRITIC"
FAILURE_AUX_OVERFITS = "CLASSIFIER_AUX_OVERFITS_SOURCE"
FAILURE_AUX_DOMINATES = "AUX_LOSS_DOMINATES_ELBO"
FAILURE_NO_GAIN = "C71A_NO_GAIN_OVER_RETRAINED_BASE"
FAILURE_WEAK_CENTERS = "WEAK_CENTERS_REMAIN_GENERATOR_CEILING"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_USEFUL = "C71A_SOURCE_PROBE_CE_USEFUL"
DECISION_STRONG = "C71A_SOURCE_PROBE_CE_STRONG"

DOWNSTREAM_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed",
    "classifier_seed",
    "paired_unit_key",
    "variant",
    "generator_family",
    "generation_mode",
    "composer",
    "comparison_role",
    "candidate_expert",
    "member_keys",
    "candidate_experts_hash",
    "num_members",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "target_eval_pool_id",
    "n_target_eval",
    "c63_full_context_bacc",
    "delta_vs_c63_full_context",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "target_eval_labels_used_for_metrics_only",
    "source_val_labels_used_for_early_stopping",
    "checkpoint_selection_metric",
    "status",
    "error_message",
)

TRAINING_COLUMNS = (
    "experiment_seed",
    "candidate_expert",
    "variant",
    "epoch",
    "aux_weight",
    "train_loss",
    "val_loss",
    "train_nll_raw",
    "train_kl_raw",
    "train_source_probe_ce_raw",
    "train_weighted_aux_to_nll_ratio",
    "train_weighted_aux_to_total_ratio",
    "train_grad_norm_decoder_from_nll",
    "train_grad_norm_decoder_from_aux",
    "val_nll_raw",
    "val_kl_raw",
    "val_source_probe_ce_raw",
    "checkpoint_selection_metric",
    "checkpoint_selected_by_aux_metric",
)

PROBE_COLUMNS = (
    "experiment_seed",
    "candidate_expert",
    "source_probe_train_bacc",
    "source_probe_val_bacc",
    "source_probe_generalization_gap",
    "source_probe_train_ce",
    "source_probe_val_ce",
    "source_probe_epochs",
    "source_probe_split",
    "source_probe_val_split",
    "source_probe_too_weak_for_geometry_critic",
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
    "synthetic_nearest_neighbor_concentration_ratio",
    "class_geometry_collapse_warning",
    "decoder_logvar_mean",
    "decoder_logvar_at_max_frac",
)

CENTER_COLUMNS = (
    "heldout_center",
    "variant",
    "mean_bacc",
    "std_bacc",
    "classifier_seed_mean_bacc",
    "classifier_seed_std_bacc",
    "delta_vs_c71a_base",
    "delta_vs_c63_full_context",
    "paired_positive_count_vs_base",
    "paired_unit_count_vs_base",
    "paired_delta_std",
    "paired_delta_bootstrap_ci_low",
    "paired_delta_bootstrap_ci_high",
    "center_mean_drop_vs_base",
    "decision_label",
)

THRESHOLD_COLUMNS = (
    "variant",
    "mean_bacc",
    "delta_vs_c71a_base",
    "delta_vs_c63_full_context",
    "paired_positive_count_vs_base",
    "paired_unit_count_vs_base",
    "paired_positive_rate_vs_base",
    "seed_level_catastrophic_drop_count",
    "mean_paired_delta",
    "paired_delta_std",
    "paired_delta_bootstrap_ci_low",
    "paired_delta_bootstrap_ci_high",
    "ge_080_rate",
    "ge_090_rate_diagnostic_only",
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
    "source_val_labels_used_for_early_stopping",
    "checkpoint_selection_metric",
    "auxiliary_metrics_used_for_checkpoint_selection",
    "routing_recomputed",
    "status",
)

FORBIDDEN_PREJOIN_SUBSTRINGS = (
    "target_label",
    "support_label",
    "target_eval",
    "oracle",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "regret",
    "current_heldout_utility",
)


@dataclass(frozen=True)
class C71AMember:
    candidate_expert: str
    variant: str
    generation_seed: int
    synthetic_pca: torch.Tensor
    synthetic_dino: torch.Tensor
    synthetic_labels: tuple[int, ...]
    checkpoint_path: Path
    projection_path: Path
    member_key: str


@dataclass(frozen=True)
class C71ATrainResult:
    checkpoint_path: Path
    history_rows: tuple[dict[str, object], ...]


class SourceProbe(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(int(input_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def c71a_aux_weight(epoch: int, *, lambda_cls: float = LAMBDA_CLS) -> float:
    if int(epoch) < AUX_WARMUP_EPOCHS:
        return 0.0
    if AUX_RAMP_EPOCHS <= 0:
        return float(lambda_cls)
    progress = (int(epoch) - AUX_WARMUP_EPOCHS + 1) / float(AUX_RAMP_EPOCHS)
    return float(lambda_cls) * min(max(progress, 0.0), 1.0)


def assert_c71a_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        forbidden = sorted(
            key
            for key in row
            if any(fragment in str(key).lower() for fragment in FORBIDDEN_PREJOIN_SUBSTRINGS)
        )
        if forbidden:
            raise ProtocolError(f"C7.1a pre-join row contains forbidden target/utility columns: {forbidden}")


def run_c71a_source_probe_ce(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c63_artifacts_root: Path | None = None,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    training_profile: C41TrainingProfile,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
) -> dict[str, Path]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    tables = artifacts_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    artifacts = _limit_c41_artifacts(discover_c41_run_artifacts(config=config, repo_root=repo_root), limits.experiment_seeds)
    generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    torch_device = _resolve_torch_device(torch, device)

    downstream_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    c63_full_context = load_c63_full_context_bacc(c63_artifacts_root)

    for artifact in artifacts:
        profile = _profile_for_support_config(training_profile, artifact.support.config_resolved)
        samples = _read_samples_manifest(artifact.support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        val_records = _records_for_split(samples, "val")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(artifact.support.train_cache, train_records, repo_root=repo_root)
        val_cache = _load_embedding_cache(artifact.val_cache, val_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(artifact.support.test_cache, test_records, repo_root=repo_root)

        candidate_state: dict[str, dict[str, Any]] = {}
        for candidate in tuple(str(c) for c in config.candidate_domains):
            projection = _fit_or_load_c71a_projection(
                artifacts_root=artifacts_root,
                train_cache=train_cache,
                candidate_expert=candidate,
                seed=int(artifact.support.experiment_seed),
                n_components=profile.pca_components,
                resume=resume,
            )
            train_projected = projection.transform(train_cache.embeddings)
            val_projected = projection.transform(val_cache.embeddings)
            train_idx = _indices_for_domain(train_cache.metadata, candidate)
            val_idx = _indices_for_domain(val_cache.metadata, candidate)
            if not train_idx or not val_idx:
                raise ProtocolError(f"C7.1a requires nonempty source train/val rows for candidate={candidate}.")
            train_x = train_projected[train_idx]
            val_x = val_projected[val_idx]
            train_y = labels_from_metadata([train_cache.metadata[idx] for idx in train_idx])
            val_y = labels_from_metadata([val_cache.metadata[idx] for idx in val_idx])
            source_probe = train_source_probe(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                seed=int(artifact.support.experiment_seed) + int(candidate),
                device=torch_device,
            )
            probe_rows.append(
                source_probe_diagnostics(
                    probe=source_probe,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    experiment_seed=int(artifact.support.experiment_seed),
                    candidate_expert=candidate,
                    device=torch_device,
                )
            )
            variant_ckpts: dict[str, Path] = {}
            for variant in (VARIANT_BASE, VARIANT_SOURCE_PROBE_CE):
                result = train_c71a_cvae(
                    repo_root=repo_root,
                    artifacts_root=artifacts_root,
                    experiment_seed=int(artifact.support.experiment_seed),
                    candidate_expert=candidate,
                    variant=variant,
                    train_x=train_x,
                    val_x=val_x,
                    train_y=train_y,
                    val_y=val_y,
                    source_probe=source_probe,
                    profile=profile,
                    device=torch_device,
                    resume=resume,
                )
                variant_ckpts[variant] = result.checkpoint_path
                training_rows.extend(result.history_rows)
            candidate_state[candidate] = {
                "projection": projection,
                "projection_path": _c71a_projection_path(artifacts_root, artifact.support.experiment_seed, candidate),
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

        for heldout_center in heldout_centers:
            heldout = str(heldout_center)
            if heldout not in {str(c) for c in config.candidate_domains}:
                raise ProtocolError(f"Unknown heldout center requested: {heldout}")
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions(
                support_units,
                experiment_seed=int(artifact.support.experiment_seed),
                heldout_center=heldout,
            )
            if not support_conditions:
                raise ProtocolError(f"No support conditions for seed={artifact.support.experiment_seed}, heldout={heldout}.")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            if tuple(sorted(set(target_labels).union({0, 1}))) != GLOBAL_CLASS_ORDER:
                raise ProtocolError(f"C7.1a expects binary labels {GLOBAL_CLASS_ORDER}, got {sorted(set(target_labels))}")
            target_dino = test_cache.embeddings[list(target_pool.eval_indices)]
            for variant in C71A_VARIANTS:
                for generation_seed in generation_seeds:
                    members = [
                        _generate_c71a_member(
                            repo_root=repo_root,
                            c41_artifacts_root=c41_artifacts_root,
                            state=candidate_state[candidate],
                            variant=variant,
                            experiment_seed=int(artifact.support.experiment_seed),
                            candidate_expert=candidate,
                            generation_seed=int(generation_seed),
                            budget_per_class=int(config.primary_budget_per_class),
                            device=device,
                        )
                        for candidate in candidates
                    ]
                    for member in members:
                        if member.variant != VARIANT_C63_ORIGINAL_HETERO_MEAN_REPLAY:
                            geometry_rows.append(
                                generated_geometry_diagnostics(
                                    member=member,
                                    state=candidate_state[member.candidate_expert],
                                    experiment_seed=int(artifact.support.experiment_seed),
                                    heldout_center=heldout,
                                )
                            )
                    for support_size, support_seed in support_conditions:
                        unit = _support_unit(
                            support_units,
                            experiment_seed=int(artifact.support.experiment_seed),
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                        )
                        for classifier_seed in classifier_seeds:
                            row = _score_c71a_ensemble(
                                members=members,
                                experiment_seed=int(artifact.support.experiment_seed),
                                heldout_center=heldout,
                                support_size=int(support_size),
                                support_seed=int(support_seed),
                                support_eval_split_id=unit.support_eval_split_id if unit else "",
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                target_dino=target_dino,
                                target_labels=target_labels,
                                target_eval_pool_id=target_pool.target_eval_pool_id,
                                c63_full_context_bacc=c63_full_context.get(
                                    (
                                        int(artifact.support.experiment_seed),
                                        heldout,
                                        int(support_size),
                                        int(support_seed),
                                        int(classifier_seed),
                                    ),
                                    math.nan,
                                ),
                            )
                            downstream_rows.append(row)
                            protocol_rows.append(
                                _protocol_row(
                                    row=row,
                                    candidates=candidates,
                                    heldout_center=heldout,
                                    experiment_seed=int(artifact.support.experiment_seed),
                                )
                            )

    center_rows = build_c71a_center_summary_rows(downstream_rows)
    threshold_rows = build_c71a_threshold_rows(downstream_rows)
    outputs = {
        "training": tables / "c71a_training_diagnostics.csv",
        "probe": tables / "c71a_source_probe_diagnostics.csv",
        "geometry": tables / "c71a_generated_geometry_diagnostics.csv",
        "matrix": tables / "c71a_downstream_matrix.csv",
        "center": tables / "c71a_center_summary.csv",
        "threshold": tables / "c71a_threshold_audit.csv",
        "protocol": tables / "c71a_protocol_audit.csv",
    }
    _write_csv(outputs["training"], TRAINING_COLUMNS, training_rows)
    _write_csv(outputs["probe"], PROBE_COLUMNS, probe_rows)
    _write_csv(outputs["geometry"], GEOMETRY_COLUMNS, geometry_rows)
    _write_csv(outputs["matrix"], DOWNSTREAM_COLUMNS, downstream_rows)
    _write_csv(outputs["center"], CENTER_COLUMNS, center_rows)
    _write_csv(outputs["threshold"], THRESHOLD_COLUMNS, threshold_rows)
    _write_csv(outputs["protocol"], PROTOCOL_COLUMNS, protocol_rows)
    return outputs


def load_c63_full_context_bacc(c63_artifacts_root: Path | None) -> dict[tuple[int, str, int, int, int], float]:
    if c63_artifacts_root is None:
        return {}
    path = Path(c63_artifacts_root) / "tables" / "c63_geometric_late_ensemble_downstream_matrix.csv"
    if not path.exists():
        return {}
    grouped: dict[tuple[int, str, int, int, int], list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("ensemble_policy")) != "fixed_all_source_safe_multiseed_geometric_late_ensemble":
                continue
            if str(row.get("status", "ok")) != "ok":
                continue
            try:
                key = (
                    int(row["experiment_seed"]),
                    str(row["heldout_center"]),
                    int(row["support_size"]),
                    int(row["support_seed"]),
                    int(row["classifier_seed"]),
                )
                grouped.setdefault(key, []).append(float(row["bacc"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {key: _mean(values) for key, values in grouped.items()}


def train_source_probe(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int = PROBE_EPOCHS,
) -> SourceProbe:
    torch.manual_seed(int(seed))
    probe = SourceProbe(input_dim=int(train_x.shape[1]), num_classes=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    loader = DataLoader(
        TensorDataset(train_x.float(), train_y.long()),
        batch_size=min(128, max(1, int(train_x.shape[0]))),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    for _epoch in range(int(epochs)):
        probe.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


def source_probe_diagnostics(
    *,
    probe: SourceProbe,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    experiment_seed: int,
    candidate_expert: str,
    device: torch.device,
) -> dict[str, object]:
    train_bacc, train_ce = _probe_bacc_and_ce(probe, train_x, train_y, device=device)
    val_bacc, val_ce = _probe_bacc_and_ce(probe, val_x, val_y, device=device)
    gap = train_bacc - val_bacc
    return {
        "experiment_seed": int(experiment_seed),
        "candidate_expert": candidate_expert,
        "source_probe_train_bacc": train_bacc,
        "source_probe_val_bacc": val_bacc,
        "source_probe_generalization_gap": gap,
        "source_probe_train_ce": train_ce,
        "source_probe_val_ce": val_ce,
        "source_probe_epochs": PROBE_EPOCHS,
        "source_probe_split": "source_train",
        "source_probe_val_split": "source_val_diagnostics_only",
        "source_probe_too_weak_for_geometry_critic": int(val_bacc < 0.60 or gap > 0.25),
    }


def train_c71a_cvae(
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
    profile: C41TrainingProfile,
    device: torch.device,
    resume: bool,
) -> C71ATrainResult:
    _ensure_cvae_testing_path(repo_root)
    from src.models.cvae_expert import (  # type: ignore
        CVAEExpert,
        DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        RECON_LOSS_GAUSSIAN_NLL_DIAG,
        REDUCTION_MEAN,
        elbo_loss_terms,
    )
    from src.train.checkpoint_provenance import load_model_checkpoint, wrap_model_state_dict  # type: ignore

    if variant not in (VARIANT_BASE, VARIANT_SOURCE_PROBE_CE):
        raise ProtocolError(f"Unknown C7.1a train variant: {variant}")
    out_dir = artifacts_root / "checkpoints" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / _variant_slug(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{_variant_slug(variant)}_class_conditional_pca64.pt"
    history_path = out_dir / "training_history.csv"
    if ckpt.exists() and resume:
        loaded = load_model_checkpoint(ckpt, map_location=device)
        rows = _read_csv_dicts(history_path)
        return C71ATrainResult(checkpoint_path=ckpt, history_rows=tuple(rows))
    if ckpt.exists() and not resume:
        raise ProtocolError(f"C7.1a checkpoint already exists; use --resume or a clean artifact root: {ckpt}")

    torch.manual_seed(int(experiment_seed) + int(candidate_expert))
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
    best_val_nelbo = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    for epoch in range(int(profile.epochs)):
        aux_weight = c71a_aux_weight(epoch) if variant == VARIANT_SOURCE_PROBE_CE else 0.0
        train_stats = _run_c71a_epoch(
            model=model,
            source_probe=source_probe,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            aux_weight=aux_weight,
            decoder_params=decoder_params,
            train=True,
        )
        val_stats = _run_c71a_epoch(
            model=model,
            source_probe=source_probe,
            loader=val_loader,
            optimizer=None,
            device=device,
            aux_weight=aux_weight,
            decoder_params=decoder_params,
            train=False,
        )
        row = {
            "experiment_seed": int(experiment_seed),
            "candidate_expert": candidate_expert,
            "variant": variant,
            "epoch": int(epoch),
            "aux_weight": float(aux_weight),
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_nll_raw": train_stats["nll_raw"],
            "train_kl_raw": train_stats["kl_raw"],
            "train_source_probe_ce_raw": train_stats["source_probe_ce_raw"],
            "train_weighted_aux_to_nll_ratio": train_stats["weighted_aux_to_nll_ratio"],
            "train_weighted_aux_to_total_ratio": train_stats["weighted_aux_to_total_ratio"],
            "train_grad_norm_decoder_from_nll": train_stats["grad_norm_decoder_from_nll"],
            "train_grad_norm_decoder_from_aux": train_stats["grad_norm_decoder_from_aux"],
            "val_nll_raw": val_stats["nll_raw"],
            "val_kl_raw": val_stats["kl_raw"],
            "val_source_probe_ce_raw": val_stats["source_probe_ce_raw"],
            "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only",
            "checkpoint_selected_by_aux_metric": 0,
        }
        history.append(row)
        val_nelbo = float(val_stats["nelbo_raw"])
        if val_nelbo < best_val_nelbo:
            best_val_nelbo = val_nelbo
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(profile.patience):
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    metadata = {
        "generator_family": C71A_GENERATOR_FAMILY,
        "experiment_id": "C7.1a",
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
        "source_probe_ce_lambda": LAMBDA_CLS if variant == VARIANT_SOURCE_PROBE_CE else 0.0,
        "aux_warmup_epochs": AUX_WARMUP_EPOCHS,
        "aux_ramp_epochs": AUX_RAMP_EPOCHS,
        "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only",
        "source_val_labels_used_for_early_stopping": 1,
    }
    torch.save(wrap_model_state_dict(best_state, metadata), ckpt)
    _write_csv(history_path, TRAINING_COLUMNS, history)
    return C71ATrainResult(checkpoint_path=ckpt, history_rows=tuple(history))


def generated_geometry_diagnostics(
    *,
    member: C71AMember,
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
    collapse = int(
        min(ratios.values()) < COLLAPSE_COV_TRACE_RATIO_MIN
        or min(rank_ratios.values()) < COLLAPSE_EFFECTIVE_RANK_RATIO_MIN
        or nn_ratio > COLLAPSE_NN_CONCENTRATION_RATIO_MAX
    )
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
        "synthetic_nearest_neighbor_concentration_ratio": nn_ratio,
        "class_geometry_collapse_warning": collapse,
        **decoder_logvar_diagnostics_by_class(
            model=_load_c71a_model(state["variant_ckpts"][member.variant], device=device),
            reference_pools=state["reference_pools"],
        ),
    }


def build_c71a_center_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    matrix = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    base_by_key = {
        str(row["paired_unit_key"]): float(row["bacc"])
        for row in matrix
        if str(row["variant"]) == VARIANT_BASE
    }
    out: list[dict[str, object]] = []
    for center in sorted({str(row["heldout_center"]) for row in matrix}):
        for variant in C71A_VARIANTS:
            subset = [row for row in matrix if str(row["heldout_center"]) == center and str(row["variant"]) == variant]
            if not subset:
                continue
            baccs = [float(row["bacc"]) for row in subset]
            deltas = [
                float(row["bacc"]) - base_by_key[str(row["paired_unit_key"])]
                for row in subset
                if str(row["paired_unit_key"]) in base_by_key and str(row["variant"]) != VARIANT_BASE
            ]
            c63_deltas = [
                float(row["delta_vs_c63_full_context"])
                for row in subset
                if not math.isnan(float(row.get("delta_vs_c63_full_context", math.nan)))
            ]
            ci_low, ci_high = _bootstrap_ci(deltas)
            label = _c71a_decision_label(variant=variant, mean_delta=_mean(deltas), paired_positive=sum(1 for d in deltas if d > 0), paired_n=len(deltas))
            out.append(
                {
                    "heldout_center": center,
                    "variant": variant,
                    "mean_bacc": _mean(baccs),
                    "std_bacc": _std(baccs),
                    "classifier_seed_mean_bacc": _mean(baccs),
                    "classifier_seed_std_bacc": _std(baccs),
                    "delta_vs_c71a_base": _mean(deltas),
                    "delta_vs_c63_full_context": _mean(c63_deltas),
                    "paired_positive_count_vs_base": sum(1 for d in deltas if d > 0),
                    "paired_unit_count_vs_base": len(deltas),
                    "paired_delta_std": _std(deltas),
                    "paired_delta_bootstrap_ci_low": ci_low,
                    "paired_delta_bootstrap_ci_high": ci_high,
                    "center_mean_drop_vs_base": min(_mean(deltas), 0.0) if deltas else math.nan,
                    "decision_label": label,
                }
            )
    return out


def build_c71a_threshold_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    matrix = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    base_by_key = {
        str(row["paired_unit_key"]): float(row["bacc"])
        for row in matrix
        if str(row["variant"]) == VARIANT_BASE
    }
    out: list[dict[str, object]] = []
    for variant in C71A_VARIANTS:
        subset = [row for row in matrix if str(row["variant"]) == variant]
        if not subset:
            continue
        baccs = [float(row["bacc"]) for row in subset]
        deltas = [
            float(row["bacc"]) - base_by_key[str(row["paired_unit_key"])]
            for row in subset
            if str(row["paired_unit_key"]) in base_by_key and variant != VARIANT_BASE
        ]
        c63_deltas = [
            float(row["delta_vs_c63_full_context"])
            for row in subset
            if not math.isnan(float(row.get("delta_vs_c63_full_context", math.nan)))
        ]
        ci_low, ci_high = _bootstrap_ci(deltas)
        positive = sum(1 for delta in deltas if delta > 0.0)
        decision = _c71a_decision_label(
            variant=variant,
            mean_delta=_mean(deltas),
            paired_positive=positive,
            paired_n=len(deltas),
        )
        out.append(
            {
                "variant": variant,
                "mean_bacc": _mean(baccs),
                "delta_vs_c71a_base": _mean(deltas),
                "delta_vs_c63_full_context": _mean(c63_deltas),
                "paired_positive_count_vs_base": positive,
                "paired_unit_count_vs_base": len(deltas),
                "paired_positive_rate_vs_base": positive / float(max(len(deltas), 1)),
                "seed_level_catastrophic_drop_count": sum(1 for delta in deltas if delta < -0.05),
                "mean_paired_delta": _mean(deltas),
                "paired_delta_std": _std(deltas),
                "paired_delta_bootstrap_ci_low": ci_low,
                "paired_delta_bootstrap_ci_high": ci_high,
                "ge_080_rate": sum(1 for value in baccs if value >= 0.80) / float(max(len(baccs), 1)),
                "ge_090_rate_diagnostic_only": sum(1 for value in baccs if value >= 0.90) / float(max(len(baccs), 1)),
                "decision_label": decision,
            }
        )
    return out


def _run_c71a_epoch(
    *,
    model: Any,
    source_probe: SourceProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    aux_weight: float,
    decoder_params: Sequence[torch.nn.Parameter],
    train: bool,
) -> dict[str, float]:
    from src.models.cvae_expert import (  # type: ignore
        RECON_LOSS_GAUSSIAN_NLL_DIAG,
        REDUCTION_MEAN,
        elbo_loss_terms,
    )

    model.train(bool(train))
    sums: dict[str, float] = {
        "loss": 0.0,
        "nelbo_raw": 0.0,
        "nll_raw": 0.0,
        "kl_raw": 0.0,
        "source_probe_ce_raw": 0.0,
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
            source_probe_ce = F.cross_entropy(source_probe(decoder_mean), yb)
            loss = nelbo + float(aux_weight) * source_probe_ce
            batch_n = int(xb.shape[0])
            if train and optimizer is not None:
                nll_grad = _grad_norm(nelbo, decoder_params, retain_graph=True)
                aux_grad = _grad_norm(source_probe_ce, decoder_params, retain_graph=True)
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
            if not math.isnan(nll_grad):
                sums["grad_norm_decoder_from_nll"] += nll_grad * batch_n
                sums["grad_norm_decoder_from_aux"] += aux_grad * batch_n
            count += batch_n
    out = {key: value / float(max(count, 1)) for key, value in sums.items()}
    weighted_aux = float(aux_weight) * out["source_probe_ce_raw"]
    out["weighted_aux_to_nll_ratio"] = weighted_aux / max(abs(out["nll_raw"]), 1.0e-12)
    out["weighted_aux_to_total_ratio"] = weighted_aux / max(abs(out["loss"]), 1.0e-12)
    return out


def _generate_c71a_member(
    *,
    repo_root: Path,
    c41_artifacts_root: Path,
    state: Mapping[str, Any],
    variant: str,
    experiment_seed: int,
    candidate_expert: str,
    generation_seed: int,
    budget_per_class: int,
    device: str,
) -> C71AMember:
    projection: SourceTrainPCAProjection = state["projection"]
    if variant == VARIANT_C63_ORIGINAL_HETERO_MEAN_REPLAY:
        checkpoint = _c41_hetero_checkpoint_path(c41_artifacts_root, experiment_seed, candidate_expert)
        projection = _load_projection(_c41_projection_path(c41_artifacts_root, experiment_seed, candidate_expert))
        reference_pools = build_source_train_reference_pools(
            train_projected_embeddings=projection.transform(state["train_embeddings"]),
            train_metadata=state["train_metadata"],
            source_domain=candidate_expert,
            label_values=GLOBAL_CLASS_ORDER,
        )
        projection_path = _c41_projection_path(c41_artifacts_root, experiment_seed, candidate_expert)
        model = _load_c41_model(repo_root, checkpoint, device=device)
    else:
        checkpoint = state["variant_ckpts"][variant]
        reference_pools = state["reference_pools"]
        projection_path = state["projection_path"]
        model = _load_c71a_model(checkpoint, device=_resolve_torch_device(torch, device))
    chunks = []
    labels: list[int] = []
    for label in GLOBAL_CLASS_ORDER:
        generated = generate_posterior_sampled_embeddings(
            model=model,
            reference_pool=reference_pools[int(label)].to(next(model.parameters()).device),
            class_label=int(label),
            n_samples=int(budget_per_class),
            seed=int(generation_seed) + int(label),
            generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        )
        chunks.append(generated.embeddings)
        labels.extend(int(v) for v in generated.labels.tolist())
    synthetic_pca = torch.cat(chunks, dim=0).detach().cpu().float()
    synthetic_dino = projection.inverse_transform(synthetic_pca).detach().cpu().float()
    return C71AMember(
        candidate_expert=candidate_expert,
        variant=variant,
        generation_seed=int(generation_seed),
        synthetic_pca=synthetic_pca,
        synthetic_dino=synthetic_dino,
        synthetic_labels=tuple(labels),
        checkpoint_path=checkpoint,
        projection_path=projection_path,
        member_key=f"{variant}::expert_{candidate_expert}::hetero_mean::seed_{int(generation_seed)}",
    )


def _score_c71a_ensemble(
    *,
    members: Sequence[C71AMember],
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    generation_seed: int,
    classifier_seed: int,
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    c63_full_context_bacc: float = math.nan,
) -> dict[str, object]:
    variant = members[0].variant if members else ""
    paired_key = _paired_unit_key(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        support_size=support_size,
        support_seed=support_seed,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
    )
    base = {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "paired_unit_key": paired_key,
        "variant": variant,
        "generator_family": C71A_GENERATOR_FAMILY if variant != VARIANT_C63_ORIGINAL_HETERO_MEAN_REPLAY else GENERATOR_FAMILY_HETEROSCEDASTIC,
        "generation_mode": GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        "composer": "c63_style_equal_weight_geometric_late_ensemble",
        "comparison_role": _comparison_role(variant),
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "member_keys": ";".join(member.member_key for member in members),
        "candidate_experts_hash": hash_candidate_experts([member.candidate_expert for member in members]),
        "num_members": len(members),
        "target_eval_pool_id": target_eval_pool_id,
        "n_target_eval": len(target_labels),
        "c63_full_context_bacc": float(c63_full_context_bacc),
        "delta_vs_c63_full_context": math.nan,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_labels_used_for_metrics_only": 1,
        "source_val_labels_used_for_early_stopping": 1,
        "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only",
    }
    try:
        probabilities = []
        for member in members:
            pred = _fit_member_probabilities(
                synthetic_embeddings=_to_numpy(member.synthetic_dino),
                synthetic_labels=member.synthetic_labels,
                target_embeddings=_to_numpy(target_dino),
                classifier_seed=int(classifier_seed),
            )
            probabilities.append(align_probabilities_to_class_order(pred["probabilities"], pred["classes"], GLOBAL_CLASS_ORDER))
        stacked = _np_stack(probabilities)
        scores, geometric_prob = geometric_pool_probabilities(stacked, [1.0 for _ in members])
        pred_idx = _np_argmax(scores)
        predictions = [int(GLOBAL_CLASS_ORDER[int(idx)]) for idx in pred_idx]
        metrics = _score_predictions_and_probabilities(target_labels, predictions, geometric_prob, GLOBAL_CLASS_ORDER)
        return {
            **base,
            "bacc": metrics["bacc"],
            "macro_f1": metrics["macro_f1"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "delta_vs_c63_full_context": metrics["bacc"] - float(c63_full_context_bacc)
            if not math.isnan(float(c63_full_context_bacc))
            else math.nan,
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
            "status": "failed_c71a_ensemble_scoring",
            "error_message": str(exc),
        }


def _fit_or_load_c71a_projection(
    *,
    artifacts_root: Path,
    train_cache: Any,
    candidate_expert: str,
    seed: int,
    n_components: int,
    resume: bool,
) -> SourceTrainPCAProjection:
    path = _c71a_projection_path(artifacts_root, seed, candidate_expert)
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


def _c71a_projection_path(root: Path, seed: int, candidate_expert: str) -> Path:
    return root / "projections" / f"seed{int(seed)}" / f"expert_{candidate_expert}" / "pca64.pt"


def _c41_projection_path(root: Path, seed: int, candidate_expert: str) -> Path:
    return root / "projections" / f"seed{int(seed)}" / f"expert_{candidate_expert}" / "pca64.pt"


def _c41_hetero_checkpoint_path(root: Path, seed: int, candidate_expert: str) -> Path:
    return root / "checkpoints" / f"seed{int(seed)}" / f"expert_{candidate_expert}" / "heteroscedastic" / "heteroscedastic_class_conditional_pca64.pt"


def _load_projection(path: Path) -> SourceTrainPCAProjection:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_c71a_model(checkpoint_path: Path, *, device: torch.device) -> Any:
    _ensure_cvae_testing_path(Path.cwd())
    from src.models.cvae_expert import build_cvae_from_metadata  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

    loaded = load_model_checkpoint(checkpoint_path, map_location=device)
    model = build_cvae_from_metadata(loaded.checkpoint_metadata).to(device)
    model.load_state_dict(loaded.model_state_dict)
    model.eval()
    return model


def _protocol_row(
    *,
    row: Mapping[str, object],
    candidates: Sequence[str],
    heldout_center: str,
    experiment_seed: int,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "variant": row["variant"],
        "heldout_source_excluded": int(str(heldout_center) not in {str(v) for v in candidates}),
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_labels_used_for_metrics_only": 1,
        "source_val_labels_used_for_early_stopping": 1,
        "checkpoint_selection_metric": "source_val_nelbo_reconstruction_kl_only",
        "auxiliary_metrics_used_for_checkpoint_selection": 0,
        "routing_recomputed": 0,
        "status": "PASS" if str(row.get("status")) == "ok" else row.get("status", "failed"),
    }


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


def _probe_bacc_and_ce(probe: SourceProbe, x: torch.Tensor, y: torch.Tensor, *, device: torch.device) -> tuple[float, float]:
    from .downstream import balanced_accuracy

    with torch.no_grad():
        logits = probe(x.float().to(device))
        targets = y.long().to(device)
        ce = F.cross_entropy(logits, targets)
        pred = logits.argmax(dim=1).detach().cpu().tolist()
    return balanced_accuracy([int(v) for v in y.tolist()], [int(v) for v in pred]), float(ce.item())


def _grad_norm(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], *, retain_graph: bool) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _class_cov_trace_ratios(gen: torch.Tensor, gen_y: torch.Tensor, real: torch.Tensor, real_y: torch.Tensor) -> dict[int, float]:
    out = {}
    for label in GLOBAL_CLASS_ORDER:
        out[int(label)] = _trace_cov(gen[gen_y == int(label)]) / max(_trace_cov(real[real_y == int(label)]), 1.0e-12)
    return out


def _class_effective_rank_ratios(gen: torch.Tensor, gen_y: torch.Tensor, real: torch.Tensor, real_y: torch.Tensor) -> dict[int, float]:
    out = {}
    for label in GLOBAL_CLASS_ORDER:
        out[int(label)] = _effective_rank(gen[gen_y == int(label)]) / max(_effective_rank(real[real_y == int(label)]), 1.0e-12)
    return out


def _trace_cov(x: torch.Tensor) -> float:
    if int(x.shape[0]) <= 1:
        return 0.0
    centered = x.float() - x.float().mean(dim=0, keepdim=True)
    return float(centered.pow(2).sum(dim=1).mean().item())


def _effective_rank(x: torch.Tensor) -> float:
    if int(x.shape[0]) <= 2:
        return 0.0
    centered = x.float() - x.float().mean(dim=0, keepdim=True)
    cov = centered.T @ centered / float(max(int(x.shape[0]) - 1, 1))
    vals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = vals.sum().clamp_min(1.0e-12)
    probs = vals / total
    entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
    return float(torch.exp(entropy).item())


def _rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    combined = torch.cat([x, y], dim=0)
    dists = torch.pdist(combined).pow(2)
    bandwidth = torch.median(dists[dists > 0]).clamp_min(1.0e-6) if (dists > 0).any() else torch.tensor(1.0)
    kxx = torch.exp(-torch.cdist(x, x).pow(2) / bandwidth).mean()
    kyy = torch.exp(-torch.cdist(y, y).pow(2) / bandwidth).mean()
    kxy = torch.exp(-torch.cdist(x, y).pow(2) / bandwidth).mean()
    return float((kxx + kyy - 2.0 * kxy).item())


def _class_centroid_shift_norm(gen: torch.Tensor, gen_y: torch.Tensor, real: torch.Tensor, real_y: torch.Tensor) -> float:
    shifts = []
    for label in GLOBAL_CLASS_ORDER:
        shifts.append(float((gen[gen_y == int(label)].mean(dim=0) - real[real_y == int(label)].mean(dim=0)).norm().item()))
    return _mean(shifts)


def _within_class_distance_ratio(gen: torch.Tensor, gen_y: torch.Tensor, real: torch.Tensor, real_y: torch.Tensor) -> float:
    ratios = []
    for label in GLOBAL_CLASS_ORDER:
        ratios.append(_pairwise_mean(gen[gen_y == int(label)]) / max(_pairwise_mean(real[real_y == int(label)]), 1.0e-12))
    return _mean(ratios)


def _between_class_distance_ratio(gen: torch.Tensor, gen_y: torch.Tensor, real: torch.Tensor, real_y: torch.Tensor) -> float:
    gen_dist = (gen[gen_y == 0].mean(dim=0) - gen[gen_y == 1].mean(dim=0)).norm()
    real_dist = (real[real_y == 0].mean(dim=0) - real[real_y == 1].mean(dim=0)).norm().clamp_min(1.0e-12)
    return float((gen_dist / real_dist).item())


def _pairwise_mean(x: torch.Tensor) -> float:
    if int(x.shape[0]) <= 1:
        return 0.0
    return float(torch.pdist(x.float()).mean().item())


def _nearest_neighbor_concentration(query: torch.Tensor, reference: torch.Tensor) -> float:
    if int(query.shape[0]) == 0 or int(reference.shape[0]) == 0:
        return math.nan
    nearest = torch.cdist(query.float(), reference.float()).argmin(dim=1).tolist()
    counts: dict[int, int] = {}
    for idx in nearest:
        counts[int(idx)] = counts.get(int(idx), 0) + 1
    top = sum(sorted(counts.values(), reverse=True)[:5])
    return float(top) / float(len(nearest))


def _format_class_map(values: Mapping[int, float]) -> str:
    return "|".join(f"{int(key)}:{float(value):.6g}" for key, value in sorted(values.items()))


def _np_stack(items: Sequence[object]) -> Any:
    import numpy as np  # type: ignore

    return np.stack([np.asarray(item, dtype=float) for item in items], axis=0)


def _np_argmax(scores: object) -> list[int]:
    import numpy as np  # type: ignore

    return np.argmax(np.asarray(scores, dtype=float), axis=1).tolist()


def _paired_unit_key(
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    generation_seed: int,
    classifier_seed: int,
) -> str:
    payload = json.dumps(
        [int(experiment_seed), str(heldout_center), int(support_size), int(support_seed), int(generation_seed), int(classifier_seed)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _variant_slug(variant: str) -> str:
    return "source_probe_ce" if variant == VARIANT_SOURCE_PROBE_CE else "base"


def _comparison_role(variant: str) -> str:
    if variant == VARIANT_SOURCE_PROBE_CE:
        return "primary_source_probe_ce"
    if variant == VARIANT_BASE:
        return "retrained_base_control"
    return "original_c41_hetero_mean_replay_context"


def _c71a_decision_label(*, variant: str, mean_delta: float, paired_positive: int, paired_n: int) -> str:
    if variant != VARIANT_SOURCE_PROBE_CE:
        return "DIAGNOSTIC_CONTROL"
    if math.isnan(float(mean_delta)):
        return FAILURE_NO_GAIN
    if float(mean_delta) >= 0.02 and paired_positive >= 10:
        return DECISION_STRONG
    if float(mean_delta) >= 0.01 and paired_positive >= 10:
        return DECISION_USEFUL
    if paired_n and paired_positive < max(1, paired_n // 2):
        return FAILURE_SOURCE_GEOMETRY_NOT_TARGET_UTILITY
    return FAILURE_NO_GAIN


def _ensure_cvae_testing_path(repo_root: Path) -> None:
    path = str(Path(repo_root) / "cvae_testing")
    if path not in sys.path:
        sys.path.insert(0, path)


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mean(values: Iterable[float]) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    return sum(cleaned) / float(len(cleaned)) if cleaned else math.nan


def _std(values: Iterable[float]) -> float:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    if len(cleaned) <= 1:
        return 0.0 if cleaned else math.nan
    mean = _mean(cleaned)
    return math.sqrt(sum((v - mean) ** 2 for v in cleaned) / float(len(cleaned) - 1))


def _bootstrap_ci(values: Sequence[float]) -> tuple[float, float]:
    cleaned = [float(v) for v in values if not math.isnan(float(v))]
    if not cleaned:
        return math.nan, math.nan
    # Deterministic lightweight bootstrap over paired deltas.
    import random

    rng = random.Random(17)
    means = []
    for _ in range(500):
        sample = [cleaned[rng.randrange(len(cleaned))] for _ in cleaned]
        means.append(_mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]
