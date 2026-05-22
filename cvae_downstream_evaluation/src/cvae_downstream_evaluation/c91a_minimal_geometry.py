"""C9.1a minimal source-geometry CVAE objective.

C9.1a changes only source-expert generator training. It uses source-train
class labels to add a small frozen-probe/prototype signal and keeps target
support/eval data out of training and checkpoint selection.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .c41_heteroscedastic import (
    SourceTrainPCAProjection,
    build_source_train_reference_pools,
    decoder_sample_diagnostics,
    fit_source_train_pca_projection,
    labels_from_metadata,
)
from .c41_workstation import (
    C41RunArtifacts,
    C41TrainingProfile,
    _profile_for_support_config,
    discover_c41_run_artifacts,
    safe_support_selection_units_from_paths,
)
from .c63_geometric_ensemble import (
    GLOBAL_CLASS_ORDER,
    LOG_PROBABILITY_EPSILON,
    GEOMETRIC_SOFTMAX_TEMPERATURE,
    geometric_pool_probabilities,
)
from .downstream import CandidateDownstreamRow, fit_locked_logistic_classifier, macro_f1
from .matrix import (
    EmbeddingCache,
    TargetEvalPool,
    _domain,
    _label,
    _load_embedding_cache,
    _read_completed_keys,
    _read_samples_manifest,
    _records_for_split,
    _resolve_torch_device,
    _to_numpy,
    append_matrix_row,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .reporting import write_alignment_csv
from .routing import SupportSelectionUnit, support_units_from_csv, write_support_selection_units
from .schemas import (
    BASELINE_ROUTING_FAMILY_USED,
    BASELINE_SELECTED_EXPERT_IDS_SOURCE,
    PRIMARY_BUDGET_PER_CLASS,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


C91A_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c91a_minimal_class_geometry_cvae_v1"
C91A_GENERATOR_FAMILY = "family_c91a_pca64_class_conditional_minimal_geometry_cvae_downstream_v1"
C91A_ELBO_ONLY_MODE = "c91a_elbo_only_posterior_decoder_mean"
C91A_PROBE_ONLY_MODE = "c91a_probe_only_posterior_decoder_mean"
C91A_PROBE_PROTO_MODE = "c91a_probe_proto_posterior_decoder_mean"
C91A_GENERATION_MODES = (C91A_ELBO_ONLY_MODE, C91A_PROBE_ONLY_MODE, C91A_PROBE_PROTO_MODE)
C91A_ENSEMBLE_FAMILY = "family_c91a_pca64_geometric_late_ensemble_downstream_v1"
C91A_ENSEMBLE_POLICY = "fixed_all_source_c91a_geometric_late_ensemble"

GEOMETRY_WARMUP_EPOCHS = 5
GEOMETRY_RAMP_EPOCHS = 10
PROBE_WEIGHT = 0.05
PROTOTYPE_WEIGHT = 0.05
BETA = 1.0

DECISION_SIGNAL = "C91A_SIGNAL_FOUND"
DECISION_PROXY_ONLY = "C91A_SOURCE_PROXY_ONLY"
DECISION_NO_SIGNAL = "C91A_NO_SIGNAL"
DECISION_PROTOCOL = "C91A_PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"


@dataclass(frozen=True)
class C91aRunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class C91aVariant:
    mode: str
    probe_weight: float
    prototype_weight: float

    @property
    def slug(self) -> str:
        return str(self.mode).replace("c91a_", "").replace("_posterior_decoder_mean", "")


@dataclass(frozen=True)
class C91aTrainResult:
    checkpoint_path: Path
    history_rows: tuple[dict[str, object], ...]
    probe_diagnostics: dict[str, object]


@dataclass(frozen=True)
class _GeneratedC91a:
    synthetic_pca: torch.Tensor
    synthetic_dino: torch.Tensor
    synthetic_labels: tuple[int, ...]
    diagnostics: dict[str, object]


def c91a_variants(*, include_probe_only: bool = True) -> tuple[C91aVariant, ...]:
    variants = [C91aVariant(C91A_ELBO_ONLY_MODE, 0.0, 0.0)]
    if include_probe_only:
        variants.append(C91aVariant(C91A_PROBE_ONLY_MODE, PROBE_WEIGHT, 0.0))
    variants.append(C91aVariant(C91A_PROBE_PROTO_MODE, PROBE_WEIGHT, PROTOTYPE_WEIGHT))
    return tuple(variants)


def geometry_weight_for_epoch(epoch: int) -> float:
    if int(epoch) < GEOMETRY_WARMUP_EPOCHS:
        return 0.0
    ramp_index = int(epoch) - GEOMETRY_WARMUP_EPOCHS + 1
    if ramp_index >= GEOMETRY_RAMP_EPOCHS:
        return 1.0
    return float(ramp_index) / float(GEOMETRY_RAMP_EPOCHS)


def normalized_prototype_centroid_loss(
    generated_mu: torch.Tensor,
    labels: torch.Tensor,
    class_centroids: Mapping[int, torch.Tensor],
    class_variance_traces: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for label, centroid in class_centroids.items():
        mask = labels.long() == int(label)
        if int(mask.sum().item()) <= 0:
            continue
        pred = generated_mu[mask].mean(dim=0)
        denom = class_variance_traces[int(label)].to(generated_mu.device).clamp_min(1.0e-6)
        losses.append((pred - centroid.to(generated_mu.device)).pow(2).sum() / denom)
    if not losses:
        return generated_mu.sum() * 0.0
    return torch.stack(losses).mean()


def train_c91a_model(
    *,
    repo_root: Path,
    artifacts_root: Path,
    experiment_seed: int,
    candidate_expert: str,
    variant: C91aVariant,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    profile: C41TrainingProfile,
    device: str,
    resume: bool,
) -> C91aTrainResult:
    _ensure_cvae_testing_path(repo_root)
    from src.models.cvae_expert import (  # type: ignore
        CVAEExpert,
        DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        RECON_LOSS_GAUSSIAN_NLL_DIAG,
        REDUCTION_MEAN,
        elbo_loss_terms,
    )
    from src.train.checkpoint_provenance import load_model_checkpoint, wrap_model_state_dict  # type: ignore

    torch_device = _resolve_torch_device(torch, device)
    out_dir = (
        artifacts_root
        / "checkpoints"
        / f"seed{int(experiment_seed)}"
        / f"expert_{candidate_expert}"
        / variant.slug
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{variant.slug}_class_conditional_pca64.pt"
    if ckpt.exists() and not resume:
        raise ProtocolError(f"C9.1a checkpoint already exists; use --resume or a clean root: {ckpt}")

    model = CVAEExpert(
        input_dim=int(train_x.shape[1]),
        hidden_dim=int(profile.hidden_dim),
        latent_dim=int(profile.latent_dim),
        class_condition_dim=2,
        decoder_likelihood=DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        decoder_logvar_min=-9.21,
        decoder_logvar_max=2.0,
        decoder_min_variance=1.0e-4,
    ).to(torch_device)
    if resume and ckpt.exists():
        loaded = load_model_checkpoint(ckpt, map_location=torch_device)
        model.load_state_dict(loaded.model_state_dict)
        return C91aTrainResult(
            checkpoint_path=ckpt,
            history_rows=tuple(),
            probe_diagnostics={"checkpoint_reused": 1, "source_probe_status": "not_refit_on_resume"},
        )

    probe, probe_diag = _fit_frozen_source_probe(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        seed=int(experiment_seed),
        device=torch_device,
    )
    centroids, traces = _class_centroid_stats(train_x, train_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(profile.lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(experiment_seed) + 9100)
    loader = DataLoader(
        TensorDataset(train_x.float(), train_y.long()),
        batch_size=int(profile.batch_size),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(val_x.float(), val_y.long()), batch_size=int(profile.batch_size), shuffle=False)

    best_val = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, object]] = []
    for epoch in range(int(profile.epochs)):
        model.train()
        epoch_weight = geometry_weight_for_epoch(epoch)
        train_sums: dict[str, float] = {}
        train_n = 0
        grad_elbo = math.nan
        grad_geometry = math.nan
        for batch_idx, (xb_cpu, yb_cpu) in enumerate(loader):
            xb = xb_cpu.to(torch_device)
            yb = yb_cpu.to(torch_device)
            recon_payload, mu_z, logvar_z = model(xb, y=yb, return_distribution=True)
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
            )
            nelbo = (terms["recon_nll"] + BETA * terms["kl"]).mean()
            aux_mu, _aux_logvar = model.decode(mu_z, y=yb, return_distribution=True)
            probe_ce = F.cross_entropy(probe(aux_mu), yb) if float(variant.probe_weight) > 0 else aux_mu.sum() * 0.0
            proto = (
                normalized_prototype_centroid_loss(aux_mu, yb, centroids, traces)
                if float(variant.prototype_weight) > 0
                else aux_mu.sum() * 0.0
            )
            weighted_probe = float(variant.probe_weight) * float(epoch_weight) * probe_ce
            weighted_proto = float(variant.prototype_weight) * float(epoch_weight) * proto
            geometry_loss = weighted_probe + weighted_proto
            if batch_idx == 0:
                params = [param for param in model.parameters() if param.requires_grad]
                grad_elbo = _grad_norm(nelbo, params, retain_graph=True)
                grad_geometry = _grad_norm(geometry_loss, params, retain_graph=True) if float(epoch_weight) > 0 else 0.0
            loss = nelbo + geometry_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            train_n += batch_n
            _accumulate(train_sums, "train_loss", loss, batch_n)
            _accumulate(train_sums, "train_recon_nll_mean", terms["recon_nll"].mean(), batch_n)
            _accumulate(train_sums, "train_kl_mean", terms["kl"].mean(), batch_n)
            _accumulate(train_sums, "train_probe_ce_mean", probe_ce, batch_n)
            _accumulate(train_sums, "train_prototype_loss_mean", proto, batch_n)
            _accumulate(train_sums, "weighted_probe_ce", weighted_probe, batch_n)
            _accumulate(train_sums, "weighted_proto", weighted_proto, batch_n)

        val_diag = _evaluate_c91a_val(
            model=model,
            val_loader=val_loader,
            device=torch_device,
            probe=probe,
            centroids=centroids,
            traces=traces,
            variant=variant,
        )
        train_means = {key: value / float(max(train_n, 1)) for key, value in train_sums.items()}
        row = {
            "experiment_seed": int(experiment_seed),
            "candidate_expert": str(candidate_expert),
            "generation_mode": variant.mode,
            "epoch": int(epoch),
            "geometry_weight_current_epoch": float(epoch_weight),
            "decoder_grad_norm_from_elbo": float(grad_elbo),
            "decoder_grad_norm_from_geometry": float(grad_geometry),
            **train_means,
            **val_diag,
        }
        row["weighted_probe_ce_to_nll_ratio"] = _safe_ratio(
            float(row.get("weighted_probe_ce", 0.0)),
            float(row.get("train_recon_nll_mean", math.nan)),
        )
        row["weighted_proto_to_nll_ratio"] = _safe_ratio(
            float(row.get("weighted_proto", 0.0)),
            float(row.get("train_recon_nll_mean", math.nan)),
        )
        history_rows.append(row)

        val_metric = float(val_diag["val_elbo_nll_checkpoint_metric"])
        if val_metric < best_val:
            best_val = val_metric
            bad_epochs = 0
            metadata = {
                "generator_family": C91A_GENERATOR_FAMILY,
                "generation_mode": variant.mode,
                "experiment_seed": int(experiment_seed),
                "candidate_expert": str(candidate_expert),
                "projection_family": "source_train_pca64",
                "class_condition_dim": 2,
                "input_dim": int(train_x.shape[1]),
                "hidden_dim": int(profile.hidden_dim),
                "latent_dim": int(profile.latent_dim),
                "decoder_likelihood": DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
                "decoder_logvar_min": -9.21,
                "decoder_logvar_max": 2.0,
                "decoder_min_variance": 1.0e-4,
                "reconstruction_loss": RECON_LOSS_GAUSSIAN_NLL_DIAG,
                "recon_reduction": REDUCTION_MEAN,
                "kl_reduction": REDUCTION_MEAN,
                "beta_effective": BETA,
                "geometry_probe_weight": float(variant.probe_weight),
                "geometry_prototype_weight": float(variant.prototype_weight),
                "checkpoint_selection": "source_val_elbo_nll",
                "source_val_probe_bacc_used_for_checkpoint": 0,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
            torch.save(wrap_model_state_dict(model.state_dict(), metadata), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= int(profile.patience):
                break

    return C91aTrainResult(checkpoint_path=ckpt, history_rows=tuple(history_rows), probe_diagnostics=probe_diag)


def build_c91a_downstream(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    training_profile: C41TrainingProfile,
    limits: C91aRunLimits = C91aRunLimits(),
    include_probe_only: bool = True,
) -> dict[str, Path]:
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = _limit_artifacts(discover_c41_run_artifacts(config=config, repo_root=repo_root), limits.experiment_seeds)
    completed = _read_completed_keys(matrix_path) if resume else set()
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    variants = c91a_variants(include_probe_only=include_probe_only)

    train_diag_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        support = artifact.support
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        val_records = _records_for_split(samples, "val")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        val_cache = _load_embedding_cache(artifact.val_cache, val_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)
        profile = _profile_for_support_config(training_profile, support.config_resolved)
        model_cache: dict[tuple[str, str], Any] = {}
        projection_cache: dict[str, SourceTrainPCAProjection] = {}
        reference_cache: dict[str, Mapping[int, torch.Tensor]] = {}
        generated_cache: dict[tuple[str, str, int, int], _GeneratedC91a] = {}
        trained_results: dict[tuple[str, str], C91aTrainResult] = {}

        for heldout in selected_heldout:
            heldout = str(heldout)
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions(
                support_units,
                experiment_seed=support.experiment_seed,
                heldout_center=heldout,
                support_sizes=limits.support_sizes,
                support_seeds=limits.support_seeds,
            )
            if not support_conditions:
                raise ProtocolError(f"No support conditions for C9.1a seed={support.experiment_seed}, heldout={heldout}")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            target_dino = test_cache.embeddings[list(target_pool.eval_indices)].detach().cpu().float()
            if tuple(sorted(set(target_labels).union({0, 1}))) != GLOBAL_CLASS_ORDER:
                raise ProtocolError(f"C9.1a expects binary labels {GLOBAL_CLASS_ORDER}, got {sorted(set(target_labels))}")

            for candidate in candidates:
                projection = _fit_or_load_projection(
                    artifacts_root=artifacts_root,
                    train_cache=train_cache,
                    source_domain=candidate,
                    seed=int(support.experiment_seed),
                    n_components=int(profile.pca_components),
                    resume=resume,
                )
                projection_cache[candidate] = projection
                train_projected_all = projection.transform(train_cache.embeddings)
                val_projected_all = projection.transform(val_cache.embeddings)
                train_idx = _indices_for_domain(train_cache.metadata, candidate)
                val_idx = _indices_for_domain(val_cache.metadata, candidate)
                if not train_idx or not val_idx:
                    raise ProtocolError(f"C9.1a requires source train/val rows for candidate={candidate}.")
                train_x = train_projected_all[train_idx]
                val_x = val_projected_all[val_idx]
                train_y = labels_from_metadata([train_cache.metadata[idx] for idx in train_idx])
                val_y = labels_from_metadata([val_cache.metadata[idx] for idx in val_idx])
                reference_pools = build_source_train_reference_pools(
                    train_projected_embeddings=train_projected_all,
                    train_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=GLOBAL_CLASS_ORDER,
                )
                reference_cache[candidate] = reference_pools
                source_train_pca = train_projected_all[train_idx].detach().cpu().float()
                source_train_dino = train_cache.embeddings[train_idx].detach().cpu().float()
                source_val_pca = val_projected_all[val_idx].detach().cpu().float()
                source_val_labels = tuple(int(v) for v in val_y.tolist())

                provenance_rows.append(
                    {
                        "experiment_seed": int(support.experiment_seed),
                        "candidate_expert": candidate,
                        "heldout_center_context": heldout,
                        "generator_family": C91A_GENERATOR_FAMILY,
                        "projection_source": "source_train_pca64_fit_in_c91a",
                        **projection.provenance(),
                        "target_support_labels_used": 0,
                        "target_eval_labels_used_for_selection": 0,
                    }
                )

                for variant in variants:
                    train_key = (candidate, variant.mode)
                    if train_key not in trained_results:
                        result = train_c91a_model(
                            repo_root=repo_root,
                            artifacts_root=artifacts_root,
                            experiment_seed=int(support.experiment_seed),
                            candidate_expert=candidate,
                            variant=variant,
                            train_x=train_x,
                            val_x=val_x,
                            train_y=train_y,
                            val_y=val_y,
                            profile=profile,
                            device=device,
                            resume=resume,
                        )
                        trained_results[train_key] = result
                        train_diag_rows.extend(result.history_rows)
                        probe_rows.append(
                            {
                                "experiment_seed": int(support.experiment_seed),
                                "candidate_expert": candidate,
                                "generation_mode": variant.mode,
                                **result.probe_diagnostics,
                            }
                        )
                    result = trained_results[train_key]
                    model = _load_c91a_model(repo_root, result.checkpoint_path, device=device)
                    model_cache[(candidate, variant.mode)] = model
                    probe_bacc = _source_val_synthetic_probe_bacc(
                        model=model,
                        reference_pools=reference_pools,
                        val_x=source_val_pca,
                        val_y=source_val_labels,
                        budget_per_class=int(config.primary_budget_per_class),
                        generation_seed=int(selected_generation_seeds[0]),
                    )
                    probe_rows.append(
                        {
                            "experiment_seed": int(support.experiment_seed),
                            "candidate_expert": candidate,
                            "generation_mode": variant.mode,
                            "source_val_synthetic_probe_bacc": probe_bacc,
                            "source_val_probe_bacc_used_for_checkpoint": 0,
                        }
                    )

                    for generation_seed in selected_generation_seeds:
                        generated = _generate_c91a_batch(
                            model=model,
                            projection=projection,
                            reference_pools=reference_pools,
                            source_train_pca=source_train_pca,
                            source_train_dino=source_train_dino,
                            source_train_labels=tuple(int(v) for v in train_y.tolist()),
                            budget_per_class=int(config.primary_budget_per_class),
                            generation_seed=int(generation_seed),
                            mode=variant.mode,
                        )
                        generated_cache[(candidate, variant.mode, int(generation_seed), int(config.primary_budget_per_class))] = generated
                        geometry_rows.append(
                            {
                                "experiment_seed": int(support.experiment_seed),
                                "heldout_center": heldout,
                                "candidate_expert": candidate,
                                "generation_mode": variant.mode,
                                "generation_seed": int(generation_seed),
                                **generated.diagnostics,
                            }
                        )
                        duplicate_rows.append(
                            {
                                "experiment_seed": int(support.experiment_seed),
                                "heldout_center": heldout,
                                "candidate_expert": candidate,
                                "generation_mode": variant.mode,
                                "generation_seed": int(generation_seed),
                                **_duplicate_diagnostics(generated.synthetic_pca, source_train_pca, generated.synthetic_labels, train_y),
                            }
                        )
                        for classifier_seed in selected_classifier_seeds:
                            row = _score_single_c91a(
                                projection=projection,
                                generated=generated,
                                experiment_seed=int(support.experiment_seed),
                                heldout_center=heldout,
                                candidate_expert=candidate,
                                generation_mode=variant.mode,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                budget_per_class=int(config.primary_budget_per_class),
                                target_eval_pool=target_pool,
                                target_labels=target_labels,
                                test_cache=test_cache,
                                checkpoint_path=result.checkpoint_path,
                                training_profile=profile,
                            )
                            for support_size, support_seed, _split_id in support_conditions:
                                replicated = replace(row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and replicated.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, replicated)
                                completed.add(replicated.primary_key())

            for support_size, support_seed, support_eval_split_id in support_conditions:
                for mode in [variant.mode for variant in variants]:
                    for classifier_seed in selected_classifier_seeds:
                        ensemble_rows.append(
                            _score_c91a_geometric_ensemble(
                                generated_cache=generated_cache,
                                model_cache=model_cache,
                                projection_cache=projection_cache,
                                reference_cache=reference_cache,
                                candidates=candidates,
                                mode=mode,
                                experiment_seed=int(support.experiment_seed),
                                heldout_center=heldout,
                                support_size=int(support_size),
                                support_seed=int(support_seed),
                                support_eval_split_id=support_eval_split_id,
                                generation_seeds=selected_generation_seeds,
                                classifier_seed=int(classifier_seed),
                                budget_per_class=int(config.primary_budget_per_class),
                                target_dino=target_dino,
                                target_labels=target_labels,
                                target_eval_pool_id=target_pool.target_eval_pool_id,
                            )
                        )
                protocol_rows.append(
                    {
                        "experiment_seed": int(support.experiment_seed),
                        "heldout_center": heldout,
                        "support_size": int(support_size),
                        "support_seed": int(support_seed),
                        "support_eval_split_id": support_eval_split_id,
                        "target_support_labels_used": 0,
                        "target_eval_features_used_for_training": 0,
                        "target_eval_labels_used_for_selection": 0,
                        "source_val_probe_bacc_used_for_checkpoint": 0,
                        "heldout_source_excluded": int(heldout not in candidates),
                        "independent_per_source_experts": 1,
                        "protocol_status": "pass",
                    }
                )

    alignment_rows = build_c91a_alignment_rows(
        support_units=support_units,
        downstream_rows=_read_c91a_matrix_rows(matrix_path),
        modes=tuple(variant.mode for variant in variants),
    )
    outputs = {
        "matrix": matrix_path,
        "alignment": artifacts_root / "tables" / "routing_to_downstream_alignment.csv",
        "ensemble": artifacts_root / "tables" / "c91a_geometric_ensemble_downstream_matrix.csv",
        "center": artifacts_root / "tables" / "c91a_center_summary.csv",
        "threshold": artifacts_root / "tables" / "c91a_threshold_audit.csv",
        "training": artifacts_root / "tables" / "c91a_training_objective_diagnostics.csv",
        "probe": artifacts_root / "tables" / "c91a_source_probe_diagnostics.csv",
        "geometry": artifacts_root / "tables" / "c91a_geometry_diagnostics.csv",
        "duplicate": artifacts_root / "tables" / "c91a_duplicate_diagnostics.csv",
        "provenance": artifacts_root / "tables" / "c91a_generator_provenance.csv",
        "protocol": artifacts_root / "tables" / "c91a_protocol_audit.csv",
    }
    write_alignment_csv(outputs["alignment"], alignment_rows)
    _write_dict_csv(outputs["ensemble"], ensemble_rows)
    _write_dict_csv(outputs["center"], build_c91a_center_summary(alignment_rows))
    _write_dict_csv(outputs["threshold"], build_c91a_threshold_audit(alignment_rows, probe_rows))
    _write_dict_csv(outputs["training"], train_diag_rows)
    _write_dict_csv(outputs["probe"], probe_rows)
    _write_dict_csv(outputs["geometry"], geometry_rows)
    _write_dict_csv(outputs["duplicate"], duplicate_rows)
    _write_dict_csv(outputs["provenance"], provenance_rows)
    _write_dict_csv(outputs["protocol"], protocol_rows)
    return outputs


def build_c91a_alignment_rows(
    *,
    support_units: Sequence[SupportSelectionUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
    modes: Sequence[str] = C91A_GENERATION_MODES,
) -> list[dict[str, object]]:
    single_rows = {
        (
            int(row.experiment_seed),
            str(row.heldout_center),
            int(row.support_size),
            int(row.support_seed),
            str(row.candidate_expert),
            str(row.generation_mode),
            int(row.budget_per_class),
            int(row.generation_seed),
            int(row.classifier_seed),
        ): row
        for row in downstream_rows
        if row.row_type == SINGLE_EXPERT_ROW_TYPE and row.status == "ok"
    }
    oracle = _c91a_oracles(downstream_rows, modes=modes)
    contexts = sorted(oracle)
    rows: list[dict[str, object]] = []
    for unit in support_units:
        if unit.method != SUPPORT_NELBO_METHOD:
            continue
        for context in contexts:
            experiment_seed, heldout, mode, budget, generation_seed, classifier_seed = context
            if int(experiment_seed) != int(unit.experiment_seed) or str(heldout) != str(unit.heldout_center):
                continue
            selected_key = (
                int(unit.experiment_seed),
                str(unit.heldout_center),
                int(unit.support_size),
                int(unit.support_seed),
                str(unit.selected_expert),
                str(mode),
                int(budget),
                int(generation_seed),
                int(classifier_seed),
            )
            selected = single_rows.get(selected_key)
            if selected is None:
                selected_key_utility = (
                    int(unit.experiment_seed),
                    str(unit.heldout_center),
                    0,
                    0,
                    str(unit.selected_expert),
                    str(mode),
                    int(budget),
                    int(generation_seed),
                    int(classifier_seed),
                )
                selected = single_rows.get(selected_key_utility)
            if selected is None:
                raise ProtocolError(f"Missing C9.1a downstream row for selected key {selected_key}")
            winner = oracle[context]
            gap = float(winner.bacc) - float(selected.bacc)
            rows.append(
                {
                    "heldout_center": unit.heldout_center,
                    "experiment_seed": int(unit.experiment_seed),
                    "support_size": int(unit.support_size),
                    "support_seed": int(unit.support_seed),
                    "generator_family": C91A_GENERATOR_FAMILY,
                    "generation_mode": mode,
                    "generation_seed": int(generation_seed),
                    "classifier_seed": int(classifier_seed),
                    "method": unit.method,
                    "selected_expert": unit.selected_expert,
                    "selected_bacc": float(selected.bacc),
                    "selected_macro_f1": float(selected.macro_f1),
                    "downstream_oracle_expert": winner.candidate_expert,
                    "oracle_bacc": float(winner.bacc),
                    "oracle_macro_f1": float(winner.macro_f1),
                    "downstream_oracle_gap_bacc": gap,
                    "downstream_oracle_gap_macro_f1": float(winner.macro_f1) - float(selected.macro_f1),
                    "relative_downstream_oracle_gap_pct": gap / float(winner.bacc) if abs(float(winner.bacc)) > 1.0e-12 else math.nan,
                    "top1_downstream_hit": int(unit.selected_expert == winner.candidate_expert),
                    "spearman_neg_nelbo_vs_bacc": math.nan,
                    "metadata_bacc": math.nan,
                    "delta_vs_metadata": math.nan,
                    "selection_depends_on_support": 1,
                }
            )
    return rows


def build_c91a_threshold_audit(
    alignment_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = _mean_selected(alignment_rows, C91A_ELBO_ONLY_MODE)
    rows: list[dict[str, object]] = []
    proxy_by_mode = {
        mode: _nanmean(_float(row.get("source_val_synthetic_probe_bacc")) for row in probe_rows if str(row.get("generation_mode")) == mode)
        for mode in C91A_GENERATION_MODES
    }
    baseline_proxy = proxy_by_mode.get(C91A_ELBO_ONLY_MODE, math.nan)
    for mode in C91A_GENERATION_MODES:
        subset = [row for row in alignment_rows if str(row.get("generation_mode")) == mode]
        selected = _nanmean(_float(row.get("selected_bacc")) for row in subset)
        delta = selected - baseline if not math.isnan(baseline) else math.nan
        positive_units = _paired_positive_units(alignment_rows, mode)
        center_drops = _center_drop_count(alignment_rows, mode, threshold=-0.02)
        proxy_delta = proxy_by_mode.get(mode, math.nan) - baseline_proxy if not math.isnan(baseline_proxy) else math.nan
        decision = _decision_label(delta=delta, positives=positive_units, center_drops=center_drops, proxy_delta=proxy_delta)
        rows.append(
            {
                "generation_mode": mode,
                "n_rows": len(subset),
                "mean_selected_bacc": selected,
                "delta_vs_c91a_elbo_only": delta,
                "paired_positive_heldout_seed_cells": positive_units,
                "paired_total_heldout_seed_cells": _paired_total_units(alignment_rows, mode),
                "center_drop_gt_002_count": center_drops,
                "source_val_probe_bacc": proxy_by_mode.get(mode, math.nan),
                "source_val_probe_bacc_delta_vs_elbo": proxy_delta,
                "decision_label": decision,
            }
        )
    return rows


def build_c91a_center_summary(alignment_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for center in sorted({str(row["heldout_center"]) for row in alignment_rows}):
        base = _nanmean(
            _float(row.get("selected_bacc"))
            for row in alignment_rows
            if str(row.get("heldout_center")) == center and str(row.get("generation_mode")) == C91A_ELBO_ONLY_MODE
        )
        for mode in C91A_GENERATION_MODES:
            subset = [row for row in alignment_rows if str(row.get("heldout_center")) == center and str(row.get("generation_mode")) == mode]
            selected = _nanmean(_float(row.get("selected_bacc")) for row in subset)
            rows.append(
                {
                    "heldout_center": center,
                    "generation_mode": mode,
                    "n_rows": len(subset),
                    "mean_selected_bacc": selected,
                    "delta_vs_c91a_elbo_only": selected - base if not math.isnan(base) else math.nan,
                    "mean_oracle_bacc": _nanmean(_float(row.get("oracle_bacc")) for row in subset),
                    "mean_oracle_gap": _nanmean(_float(row.get("downstream_oracle_gap_bacc")) for row in subset),
                }
            )
    return rows


def _score_single_c91a(
    *,
    projection: SourceTrainPCAProjection,
    generated: _GeneratedC91a,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    generation_mode: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    test_cache: EmbeddingCache,
    checkpoint_path: Path,
    training_profile: C41TrainingProfile,
) -> CandidateDownstreamRow:
    try:
        target_pca = projection.transform(test_cache.embeddings[list(target_eval_pool.eval_indices)])
        prediction = fit_locked_logistic_classifier(
            _to_numpy(generated.synthetic_pca),
            generated.synthetic_labels,
            _to_numpy(target_pca),
            target_labels,
            classifier_seed=int(classifier_seed),
        )
        return CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=C91A_GENERATOR_FAMILY,
            generation_mode=generation_mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=len(generated.synthetic_labels),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            candidate_experts_hash=SINGLE_EXPERT_HASH,
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            plain_baseline_source="retrained_in_c91a_elbo_only" if generation_mode == C91A_ELBO_ONLY_MODE else "paired_c91a_elbo_only",
            plain_baseline_artifact_path=str(checkpoint_path),
            plain_baseline_training_profile=training_profile.name,
            plain_baseline_matches_locked_hparams=training_profile.matches_locked_hparams,
            routing_family_used=BASELINE_ROUTING_FAMILY_USED,
            routing_scores_recomputed_for_heteroscedastic=0,
            selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        )
    except Exception as exc:
        return CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=C91A_GENERATOR_FAMILY,
            generation_mode=generation_mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=math.nan,
            macro_f1=math.nan,
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=len(generated.synthetic_labels),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            status="failed_c91a_single_expert_scoring",
            error_message=str(exc),
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            routing_family_used=BASELINE_ROUTING_FAMILY_USED,
            routing_scores_recomputed_for_heteroscedastic=0,
            selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        )


def _score_c91a_geometric_ensemble(
    *,
    generated_cache: Mapping[tuple[str, str, int, int], _GeneratedC91a],
    model_cache: Mapping[tuple[str, str], Any],
    projection_cache: Mapping[str, SourceTrainPCAProjection],
    reference_cache: Mapping[str, Mapping[int, torch.Tensor]],
    candidates: Sequence[str],
    mode: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    generation_seeds: Sequence[int],
    classifier_seed: int,
    budget_per_class: int,
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
) -> dict[str, object]:
    try:
        probs = []
        member_keys = []
        weights = []
        for candidate in sorted(str(v) for v in candidates):
            for generation_seed in generation_seeds:
                key = (candidate, mode, int(generation_seed), int(budget_per_class))
                generated = generated_cache.get(key)
                if generated is None:
                    generated = _generate_c91a_batch(
                        model=model_cache[(candidate, mode)],
                        projection=projection_cache[candidate],
                        reference_pools=reference_cache[candidate],
                        source_train_pca=torch.empty(0),
                        source_train_dino=torch.empty(0),
                        source_train_labels=(),
                        budget_per_class=int(budget_per_class),
                        generation_seed=int(generation_seed),
                        mode=mode,
                    )
                prediction = fit_locked_logistic_classifier(
                    _to_numpy(generated.synthetic_dino),
                    generated.synthetic_labels,
                    _to_numpy(target_dino),
                    target_labels,
                    classifier_seed=int(classifier_seed),
                )
                aligned = _align_probabilities(prediction.probabilities, prediction.classes, GLOBAL_CLASS_ORDER)
                probs.append(aligned)
                member_keys.append(f"expert_{candidate}::{mode}::seed_{int(generation_seed)}")
                weights.append(1.0)
        import numpy as np  # type: ignore

        stacked = np.stack(probs, axis=0)
        norm_weights = np.ones((len(weights),), dtype=float) / float(max(len(weights), 1))
        scores, geom_prob = geometric_pool_probabilities(
            stacked,
            norm_weights,
            epsilon=LOG_PROBABILITY_EPSILON,
            temperature=GEOMETRIC_SOFTMAX_TEMPERATURE,
        )
        pred = [int(GLOBAL_CLASS_ORDER[int(idx)]) for idx in np.argmax(scores, axis=1).tolist()]
        bacc = _balanced_accuracy(target_labels, pred)
        mf1 = macro_f1(target_labels, pred)
        entropy = -np.sum(geom_prob * np.log(np.clip(geom_prob, 1.0e-12, 1.0)), axis=1).mean()
        status = "ok"
        error = ""
    except Exception as exc:
        bacc = math.nan
        mf1 = math.nan
        entropy = math.nan
        member_keys = []
        status = "failed_c91a_geometric_ensemble"
        error = str(exc)
    return {
        "ensemble_policy": C91A_ENSEMBLE_POLICY,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed_group": "all:" + "|".join(str(int(seed)) for seed in generation_seeds),
        "classifier_seed": int(classifier_seed),
        "generator_family": C91A_ENSEMBLE_FAMILY,
        "generation_mode": mode,
        "budget_per_class": int(budget_per_class),
        "bacc": float(bacc),
        "macro_f1": float(mf1),
        "ensemble_entropy": float(entropy),
        "row_type": "method_baseline",
        "candidate_expert": "__ensemble__",
        "candidate_experts_hash": hash_candidate_experts(member_keys),
        "member_keys": ";".join(member_keys),
        "num_members": len(member_keys),
        "aggregation_rule": "weighted_log_probability_geometric_pooling",
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "status": status,
        "error_message": error,
        "target_eval_pool_id": target_eval_pool_id,
    }


def _generate_c91a_batch(
    *,
    model: Any,
    projection: SourceTrainPCAProjection,
    reference_pools: Mapping[int, torch.Tensor],
    source_train_pca: torch.Tensor,
    source_train_dino: torch.Tensor,
    source_train_labels: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
    mode: str,
) -> _GeneratedC91a:
    chunks = []
    labels: list[int] = []
    sample_diagnostics = []
    device = next(model.parameters()).device
    for label in GLOBAL_CLASS_ORDER:
        pool = reference_pools[int(label)].to(device)
        index_gen = torch.Generator(device="cpu")
        index_gen.manual_seed(int(generation_seed) + int(label))
        indices = torch.randint(pool.shape[0], (int(budget_per_class),), generator=index_gen, device="cpu")
        xb = pool[indices.to(device)]
        yb = torch.full((int(budget_per_class),), int(label), dtype=torch.long, device=device)
        with torch.no_grad():
            mu_z, _logvar_z = model.encode(xb, y=yb)
            mu_x, logvar_x = model.decode(mu_z, y=yb, return_distribution=True)
            noise = torch.zeros_like(mu_x)
        chunks.append(mu_x.detach().cpu().float())
        labels.extend([int(label)] * int(budget_per_class))
        sample_diagnostics.append(decoder_sample_diagnostics(model=model, mu_x=mu_x, logvar_x=logvar_x, noise=noise))
    synthetic_pca = torch.cat(chunks, dim=0)
    synthetic_dino = projection.inverse_transform(synthetic_pca).detach().cpu().float()
    diagnostics = {
        **_aggregate_dict_means(sample_diagnostics),
        **_geometry_diagnostics(
            synthetic_pca,
            synthetic_dino,
            labels,
            source_train_pca,
            source_train_dino,
            source_train_labels,
        ),
        "generation_mode": mode,
    }
    return _GeneratedC91a(
        synthetic_pca=synthetic_pca,
        synthetic_dino=synthetic_dino,
        synthetic_labels=tuple(labels),
        diagnostics=diagnostics,
    )


def _evaluate_c91a_val(
    *,
    model: Any,
    val_loader: DataLoader,
    device: torch.device,
    probe: nn.Module,
    centroids: Mapping[int, torch.Tensor],
    traces: Mapping[int, torch.Tensor],
    variant: C91aVariant,
) -> dict[str, float]:
    from src.models.cvae_expert import RECON_LOSS_GAUSSIAN_NLL_DIAG, REDUCTION_MEAN, elbo_loss_terms  # type: ignore

    model.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for xb_cpu, yb_cpu in val_loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            recon_payload, mu_z, logvar_z = model(xb, y=yb, return_distribution=True)
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
            )
            aux_mu, _ = model.decode(mu_z, y=yb, return_distribution=True)
            probe_ce = F.cross_entropy(probe(aux_mu), yb) if float(variant.probe_weight) > 0 else aux_mu.sum() * 0.0
            proto = (
                normalized_prototype_centroid_loss(aux_mu, yb, centroids, traces)
                if float(variant.prototype_weight) > 0
                else aux_mu.sum() * 0.0
            )
            batch_n = int(xb.shape[0])
            count += batch_n
            _accumulate(sums, "val_recon_nll_mean", terms["recon_nll"].mean(), batch_n)
            _accumulate(sums, "val_kl_mean", terms["kl"].mean(), batch_n)
            _accumulate(sums, "val_probe_ce_mean", probe_ce, batch_n)
            _accumulate(sums, "val_prototype_loss_mean", proto, batch_n)
    out = {key: value / float(max(count, 1)) for key, value in sums.items()}
    out["val_elbo_nll_checkpoint_metric"] = float(out.get("val_recon_nll_mean", math.nan)) + BETA * float(out.get("val_kl_mean", math.nan))
    return out


def _fit_frozen_source_probe(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(int(seed) + 7001)
    probe = nn.Linear(int(train_x.shape[1]), 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1.0e-2)
    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(int(seed) + 7002)
    loader = DataLoader(
        TensorDataset(train_x.float(), train_y.long()),
        batch_size=min(128, int(train_x.shape[0])),
        shuffle=True,
        generator=loader_gen,
    )
    for _epoch in range(100):
        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        train_pred = probe(train_x.to(device).float()).argmax(dim=1).detach().cpu().tolist()
        val_pred = probe(val_x.to(device).float()).argmax(dim=1).detach().cpu().tolist()
    return probe, {
        "source_probe_train_bacc": _balanced_accuracy(train_y.tolist(), train_pred),
        "source_probe_val_bacc": _balanced_accuracy(val_y.tolist(), val_pred),
        "source_probe_frozen": 1,
        "source_probe_fit_split": "source_train",
        "source_probe_val_split": "source_val",
    }


def _source_val_synthetic_probe_bacc(
    *,
    model: Any,
    reference_pools: Mapping[int, torch.Tensor],
    val_x: torch.Tensor,
    val_y: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
) -> float:
    chunks = []
    labels = []
    device = next(model.parameters()).device
    for label in GLOBAL_CLASS_ORDER:
        pool = reference_pools[int(label)].to(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(generation_seed) + int(label))
        indices = torch.randint(pool.shape[0], (int(budget_per_class),), generator=gen, device="cpu")
        xb = pool[indices.to(device)]
        yb = torch.full((int(budget_per_class),), int(label), dtype=torch.long, device=device)
        with torch.no_grad():
            mu_z, _ = model.encode(xb, y=yb)
            mu_x, _ = model.decode(mu_z, y=yb, return_distribution=True)
        chunks.append(mu_x.detach().cpu().float())
        labels.extend([int(label)] * int(budget_per_class))
    prediction = fit_locked_logistic_classifier(
        _to_numpy(torch.cat(chunks, dim=0)),
        labels,
        _to_numpy(val_x),
        val_y,
        classifier_seed=17,
    )
    return float(prediction.score.balanced_accuracy)


def _class_centroid_stats(x: torch.Tensor, y: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    centroids: dict[int, torch.Tensor] = {}
    traces: dict[int, torch.Tensor] = {}
    for label in sorted(set(int(v) for v in y.tolist())):
        subset = x[y.long() == int(label)].float()
        if subset.numel() == 0:
            continue
        centroids[int(label)] = subset.mean(dim=0)
        traces[int(label)] = subset.var(dim=0, unbiased=False).sum().clamp_min(1.0e-6)
    return centroids, traces


def _geometry_diagnostics(
    synthetic_pca: torch.Tensor,
    synthetic_dino: torch.Tensor,
    labels: Sequence[int],
    source_train_pca: torch.Tensor,
    source_train_dino: torch.Tensor,
    source_train_labels: Sequence[int],
) -> dict[str, object]:
    y = torch.tensor([int(v) for v in labels], dtype=torch.long)
    y_ref = torch.tensor([int(v) for v in source_train_labels], dtype=torch.long)
    out: dict[str, object] = {
        "generated_pca_std_ratio": _std_ratio(synthetic_pca, source_train_pca),
        "generated_dino_std_ratio": _std_ratio(synthetic_dino, source_train_dino),
        "per_class_generated_pca_std_ratio": json.dumps(
            _per_class_std_ratio(synthetic_pca, y, source_train_pca, y_ref),
            sort_keys=True,
        ),
    }
    class_centroids = []
    real_centroid_errors = []
    within_traces = []
    for label in GLOBAL_CLASS_ORDER:
        subset = synthetic_pca[y == int(label)]
        if subset.shape[0] <= 0:
            continue
        centroid = subset.mean(dim=0)
        class_centroids.append(centroid)
        within_traces.append(float(subset.var(dim=0, unbiased=False).sum().item()))
        if source_train_pca.numel() and y_ref.numel() == source_train_pca.shape[0]:
            ref_subset = source_train_pca[y_ref == int(label)]
            if ref_subset.shape[0] > 0:
                real_centroid_errors.append(float(torch.dist(centroid, ref_subset.mean(dim=0)).item()))
    if len(class_centroids) >= 2:
        out["generated_between_class_centroid_distance"] = float(torch.dist(class_centroids[0], class_centroids[1]).item())
    else:
        out["generated_between_class_centroid_distance"] = math.nan
    out["generated_within_class_trace"] = _nanmean(within_traces)
    out["generated_fisher_ratio"] = float(out["generated_between_class_centroid_distance"]) / max(float(out["generated_within_class_trace"]), 1.0e-12)
    out["real_vs_generated_centroid_l2"] = _nanmean(real_centroid_errors)
    return out


def _duplicate_diagnostics(
    synthetic_pca: torch.Tensor,
    source_train_pca: torch.Tensor,
    synthetic_labels: Sequence[int],
    source_train_labels: torch.Tensor,
) -> dict[str, object]:
    if synthetic_pca.numel() == 0 or source_train_pca.numel() == 0:
        return {"near_duplicate_frac": math.nan, "per_class_near_duplicate_frac": "{}"}
    distances = torch.cdist(synthetic_pca.float(), source_train_pca.float()).min(dim=1).values
    near = distances <= 1.0e-6
    y_syn = torch.tensor([int(v) for v in synthetic_labels], dtype=torch.long)
    per_class = {}
    for label in GLOBAL_CLASS_ORDER:
        mask = y_syn == int(label)
        per_class[str(label)] = float(near[mask].float().mean().item()) if int(mask.sum().item()) else math.nan
    return {
        "near_duplicate_frac": float(near.float().mean().item()),
        "per_class_near_duplicate_frac": json.dumps(per_class, sort_keys=True),
        "median_nn_dist_synthetic_to_source_train": float(distances.median().item()),
    }


def _c91a_oracles(
    rows: Sequence[CandidateDownstreamRow],
    *,
    modes: Sequence[str],
) -> dict[tuple[int, str, str, int, int, int], CandidateDownstreamRow]:
    grouped: dict[tuple[int, str, str, int, int, int], list[CandidateDownstreamRow]] = {}
    allowed = set(str(mode) for mode in modes)
    for row in rows:
        if row.row_type != SINGLE_EXPERT_ROW_TYPE or row.status != "ok" or str(row.generation_mode) not in allowed:
            continue
        key = (
            int(row.experiment_seed),
            str(row.heldout_center),
            str(row.generation_mode),
            int(row.budget_per_class),
            int(row.generation_seed),
            int(row.classifier_seed),
        )
        grouped.setdefault(key, []).append(row)
    return {key: max(group, key=lambda row: (float(row.bacc), float(row.macro_f1), str(row.candidate_expert))) for key, group in grouped.items()}


def _fit_or_load_projection(
    *,
    artifacts_root: Path,
    train_cache: EmbeddingCache,
    source_domain: str,
    seed: int,
    n_components: int,
    resume: bool,
) -> SourceTrainPCAProjection:
    path = artifacts_root / "projections" / f"seed{int(seed)}" / f"expert_{source_domain}" / "pca64.pt"
    if resume and path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    projection = fit_source_train_pca_projection(
        train_embeddings=train_cache.embeddings,
        train_metadata=train_cache.metadata,
        source_domain=source_domain,
        seed=int(seed),
        n_components=int(n_components),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projection, path)
    return projection


def _load_c91a_model(repo_root: Path, checkpoint_path: Path, *, device: str) -> Any:
    _ensure_cvae_testing_path(repo_root)
    from src.models.cvae_expert import build_cvae_from_metadata  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

    torch_device = _resolve_torch_device(torch, device)
    loaded = load_model_checkpoint(checkpoint_path, map_location=torch_device)
    model = build_cvae_from_metadata(loaded.checkpoint_metadata).to(torch_device)
    model.load_state_dict(loaded.model_state_dict)
    model.eval()
    return model


def _support_conditions(
    units: Sequence[SupportSelectionUnit],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_sizes: Sequence[int] | None,
    support_seeds: Sequence[int] | None,
) -> tuple[tuple[int, int, str], ...]:
    allowed_sizes = set(int(v) for v in support_sizes) if support_sizes else None
    allowed_seeds = set(int(v) for v in support_seeds) if support_seeds else None
    values = {
        (int(unit.support_size), int(unit.support_seed), str(unit.support_eval_split_id))
        for unit in units
        if int(unit.experiment_seed) == int(experiment_seed)
        and str(unit.heldout_center) == str(heldout_center)
        and str(unit.method) == SUPPORT_NELBO_METHOD
        and (allowed_sizes is None or int(unit.support_size) in allowed_sizes)
        and (allowed_seeds is None or int(unit.support_seed) in allowed_seeds)
    }
    return tuple(sorted(values))


def _limit_artifacts(
    artifacts: Sequence[C41RunArtifacts],
    experiment_seeds: Sequence[int] | None,
) -> tuple[C41RunArtifacts, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.support.experiment_seed) in allowed)


def _indices_for_domain(metadata: Sequence[Mapping[str, object]], domain: str) -> list[int]:
    return [idx for idx, row in enumerate(metadata) if str(_domain(row)) == str(domain)]


def _read_c91a_matrix_rows(path: Path) -> list[CandidateDownstreamRow]:
    from .downstream import read_candidate_downstream_matrix

    return read_candidate_downstream_matrix(path)


def _align_probabilities(probabilities: object, classes: Sequence[int], class_order: Sequence[int]) -> Any:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    out = np.zeros((probs.shape[0], len(class_order)), dtype=float)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    for out_idx, cls in enumerate(class_order):
        if int(cls) in class_to_idx:
            out[:, out_idx] = probs[:, class_to_idx[int(cls)]]
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1.0e-12, None)
    return out / row_sums


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    values = []
    for label in sorted(set(int(v) for v in y_true).union(int(v) for v in y_pred)):
        denom = sum(1 for y in y_true if int(y) == label)
        values.append(sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == label and int(yp) == label) / float(denom) if denom else 0.0)
    return sum(values) / float(len(values)) if values else math.nan


def _grad_norm(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], *, retain_graph: bool) -> float:
    if not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    total = torch.tensor(0.0, device=loss.device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return float(torch.sqrt(total).item())


def _accumulate(sums: dict[str, float], key: str, value: torch.Tensor, n: int) -> None:
    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item()) * int(n)


def _std_ratio(generated: torch.Tensor, reference: torch.Tensor) -> float:
    if generated.numel() == 0 or reference.numel() == 0:
        return math.nan
    return float((generated.float().std(dim=0, unbiased=False).mean() / reference.float().std(dim=0, unbiased=False).mean().clamp_min(1.0e-12)).item())


def _per_class_std_ratio(
    generated: torch.Tensor,
    labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
) -> dict[str, float]:
    out = {}
    for label in GLOBAL_CLASS_ORDER:
        subset = generated[labels == int(label)]
        ref_subset = reference[reference_labels == int(label)] if reference_labels.numel() == reference.shape[0] else reference
        out[str(label)] = _std_ratio(subset, ref_subset) if subset.numel() else math.nan
    return out


def _aggregate_dict_means(items: Sequence[Mapping[str, object]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: _nanmean(_float(item.get(key)) for item in items) for key in keys}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if math.isnan(float(numerator)) or math.isnan(float(denominator)) or abs(float(denominator)) <= 1.0e-12:
        return math.nan
    return float(numerator) / abs(float(denominator))


def _mean_selected(rows: Sequence[Mapping[str, object]], mode: str) -> float:
    return _nanmean(_float(row.get("selected_bacc")) for row in rows if str(row.get("generation_mode")) == mode)


def _paired_positive_units(rows: Sequence[Mapping[str, object]], mode: str) -> int:
    base = _paired_means(rows, C91A_ELBO_ONLY_MODE)
    other = _paired_means(rows, mode)
    return sum(1 for key, value in other.items() if key in base and value > base[key])


def _paired_total_units(rows: Sequence[Mapping[str, object]], mode: str) -> int:
    base = _paired_means(rows, C91A_ELBO_ONLY_MODE)
    other = _paired_means(rows, mode)
    return sum(1 for key in other if key in base)


def _paired_means(rows: Sequence[Mapping[str, object]], mode: str) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if str(row.get("generation_mode")) != mode:
            continue
        key = (str(row.get("heldout_center")), int(row.get("experiment_seed", -1)))
        grouped.setdefault(key, []).append(_float(row.get("selected_bacc")))
    return {key: _nanmean(values) for key, values in grouped.items()}


def _center_drop_count(rows: Sequence[Mapping[str, object]], mode: str, *, threshold: float) -> int:
    centers = sorted({str(row.get("heldout_center")) for row in rows})
    count = 0
    for center in centers:
        base = _nanmean(_float(row.get("selected_bacc")) for row in rows if str(row.get("heldout_center")) == center and str(row.get("generation_mode")) == C91A_ELBO_ONLY_MODE)
        other = _nanmean(_float(row.get("selected_bacc")) for row in rows if str(row.get("heldout_center")) == center and str(row.get("generation_mode")) == mode)
        if not math.isnan(base) and not math.isnan(other) and (other - base) < float(threshold):
            count += 1
    return count


def _decision_label(*, delta: float, positives: int, center_drops: int, proxy_delta: float) -> str:
    if not math.isnan(proxy_delta) and proxy_delta > 0.0 and (math.isnan(delta) or delta <= 0.0):
        return DECISION_PROXY_ONLY
    if not math.isnan(delta) and delta >= 0.01 and positives >= 8 and center_drops == 0:
        return DECISION_SIGNAL
    return DECISION_NO_SIGNAL


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _nanmean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return sum(clean) / float(len(clean)) if clean else math.nan


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _ensure_cvae_testing_path(repo_root: Path) -> None:
    path = str(repo_root / "cvae_testing")
    if path not in sys.path:
        sys.path.insert(0, path)
