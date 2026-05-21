"""F1 source-anchored residual CVAE downstream diagnostic.

F1 is a generator-construction experiment. It keeps locked C4.1 support-NELBO
selected experts fixed, reuses source-train PCA64 projections, and changes only
how synthetic PCA64 embeddings are generated.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c41_heteroscedastic import (
    GeneratedBatch,
    SourceTrainPCAProjection,
    _generator_for_device,
    _randn_like,
    labels_from_metadata,
)
from .c41_workstation import (
    C41TrainingProfile,
    _indices_for_domain,
    _support_conditions,
    _write_csv,
    _write_dict_csv,
    discover_c41_run_artifacts,
    safe_support_selection_units_from_paths,
)
from .downstream import CandidateDownstreamRow, fit_locked_logistic_classifier, read_candidate_downstream_matrix
from .matrix import (
    MatrixBuildLimits,
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


F1_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/f1_source_anchored_residual_cvae_v1"
F1_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
F1_GENERATOR_FAMILY = "family_f1_pca64_class_conditional_source_anchored_residual_cvae_downstream_v1"

F1_MODE_POSTERIOR_MEAN = "anchor_posterior_residual_mean"
F1_MODE_POSTERIOR_NOISE = "anchor_posterior_residual_noise"
F1_MODE_PRIOR_MEAN = "anchor_prior_residual_mean"
F1_MODE_IDENTITY_BOOTSTRAP = "anchor_identity_bootstrap"
F1_MODE_EMPIRICAL_BOOTSTRAP = "anchor_residual_empirical_bootstrap"
F1_MODE_TRANSFER_BOOTSTRAP = "anchor_empirical_residual_transfer_bootstrap"
F1_GENERATION_MODES = (
    F1_MODE_POSTERIOR_MEAN,
    F1_MODE_POSTERIOR_NOISE,
    F1_MODE_PRIOR_MEAN,
    F1_MODE_IDENTITY_BOOTSTRAP,
    F1_MODE_EMPIRICAL_BOOTSTRAP,
    F1_MODE_TRANSFER_BOOTSTRAP,
)
F1_DIAGNOSTIC_ONLY_MODES = (
    F1_MODE_POSTERIOR_NOISE,
    F1_MODE_PRIOR_MEAN,
    F1_MODE_IDENTITY_BOOTSTRAP,
    F1_MODE_EMPIRICAL_BOOTSTRAP,
    F1_MODE_TRANSFER_BOOTSTRAP,
)

ANCHOR_NEIGHBOR_K = 8
TRAIN_PAIRS_PER_SAMPLE = 4
VAL_PAIRS_PER_SAMPLE = 1
NEAR_DUPLICATE_EPS = 1.0e-6

DECISION_GENERATOR_SUCCESS = "GENERATOR_SUCCESS"
DECISION_SOURCE_BOOTSTRAP = "SOURCE_GEOMETRY_BOOTSTRAP_SUCCESS"
DECISION_NO_GAIN = "ANCHOR_RESIDUAL_NO_UTILITY_GAIN"
DECISION_OVERDISPERSION = "ANCHOR_RESIDUAL_OVERDISPERSION"
DECISION_UNDERDISPERSION = "ANCHOR_RESIDUAL_UNDERDISPERSION"
DECISION_NEAR_COPY = "ANCHOR_NEAR_COPY_FAILURE"
DECISION_ROUTING_BOTTLENECK = "ROUTING_BOTTLENECK_PERSISTS"
DECISION_PROTOCOL_FAILURE = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"

F1_DELTA_COLUMNS = (
    "heldout_center",
    "support_size",
    "generation_mode",
    "selected_bacc_f1",
    "oracle_bacc_f1",
    "oracle_gap_f1",
    "selected_bacc_c41_hetero_mean",
    "oracle_bacc_c41_hetero_mean",
    "oracle_gap_c41_hetero_mean",
    "selected_bacc_delta_vs_c41_hetero_mean",
    "oracle_bacc_delta_vs_c41_hetero_mean",
    "oracle_gap_delta_vs_c41_hetero_mean",
    "selected_bacc_anchor_identity_bootstrap",
    "selected_bacc_anchor_residual_empirical_bootstrap",
    "selected_bacc_anchor_empirical_residual_transfer_bootstrap",
    "beats_identity_bootstrap",
    "beats_empirical_bootstrap",
    "beats_transfer_bootstrap",
    "selected_ge_080",
    "diagnostic_only",
    "decision_label",
)

F1_ALIGNMENT_COLUMNS = (
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
    "routing_scores_recomputed_for_f1",
    "selected_expert_ids_source",
    "projection_source",
)


@dataclass(frozen=True)
class AnchorPairDataset:
    pair_targets: torch.Tensor
    anchors: torch.Tensor
    labels: torch.Tensor
    pair_sample_ids: tuple[str, ...]
    anchor_sample_ids: tuple[str, ...]
    anchor_neighbor_ranks: tuple[int, ...]
    pair_split: str
    anchor_split: str


@dataclass(frozen=True)
class SourceAnchorIndex:
    embeddings: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    neighbor_indices: tuple[tuple[int, ...], ...]
    neighbor_ranks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class F1GeneratedBatch:
    embeddings: torch.Tensor
    labels: torch.Tensor
    generation_mode: str
    diagnostics: Mapping[str, float]
    provenance_rows: tuple[dict[str, object], ...]


class AnchoredResidualCVAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        class_condition_dim: int = 2,
        decoder_logvar_min: float = -9.21,
        decoder_logvar_max: float = 2.0,
        decoder_min_variance: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.class_condition_dim = int(class_condition_dim)
        self.decoder_logvar_min = float(decoder_logvar_min)
        self.decoder_logvar_max = float(decoder_logvar_max)
        self.decoder_min_variance = float(decoder_min_variance)
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and latent_dim must be positive")
        if self.class_condition_dim <= 0:
            raise ValueError("class_condition_dim must be positive")
        if self.decoder_logvar_min > self.decoder_logvar_max:
            raise ValueError("decoder_logvar_min must be <= decoder_logvar_max")
        if self.decoder_min_variance <= 0:
            raise ValueError("decoder_min_variance must be > 0")

        self.enc = nn.Linear((2 * self.input_dim) + self.class_condition_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.dec1 = nn.Linear(self.latent_dim + self.input_dim + self.class_condition_dim, self.hidden_dim)
        self.dec_delta = nn.Linear(self.hidden_dim, self.input_dim)
        self.dec_logvar = nn.Linear(self.hidden_dim, self.input_dim)

    def _one_hot(self, y: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if y.ndim == 1:
            targets = y.long()
            if targets.numel():
                min_target = int(targets.min().item())
                max_target = int(targets.max().item())
                if min_target < 0 or max_target >= self.class_condition_dim:
                    raise ValueError(
                        "Class-condition indices are out of range: "
                        f"min={min_target}, max={max_target}, class_condition_dim={self.class_condition_dim}"
                    )
            return F.one_hot(targets.to(device=device), num_classes=self.class_condition_dim).to(torch.float32)
        if y.ndim == 2 and int(y.shape[1]) == self.class_condition_dim:
            return y.to(device=device, dtype=torch.float32)
        raise ValueError("y must be 1D class indices or matching one-hot rows")

    def encode(self, x_pair_target: torch.Tensor, x_anchor: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair_shapes(x_pair_target, x_anchor)
        y_one_hot = self._one_hot(y, device=x_pair_target.device)
        if int(y_one_hot.shape[0]) != int(x_pair_target.shape[0]):
            raise ValueError("y batch size must match x_pair_target")
        h = F.relu(self.enc(torch.cat([x_pair_target, x_anchor, y_one_hot], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode_residual(
        self,
        z: torch.Tensor,
        x_anchor: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or x_anchor.ndim != 2:
            raise ValueError("z and x_anchor must be 2D tensors")
        if int(z.shape[0]) != int(x_anchor.shape[0]):
            raise ValueError("z and x_anchor batch sizes must match")
        y_one_hot = self._one_hot(y, device=x_anchor.device)
        if int(y_one_hot.shape[0]) != int(x_anchor.shape[0]):
            raise ValueError("y batch size must match x_anchor")
        h = F.relu(self.dec1(torch.cat([z, x_anchor, y_one_hot], dim=1)))
        delta_mu = self.dec_delta(h)
        logvar = torch.clamp(self.dec_logvar(h), min=self.decoder_logvar_min, max=self.decoder_logvar_max)
        min_logvar = math.log(self.decoder_min_variance)
        if min_logvar > self.decoder_logvar_min:
            logvar = torch.clamp(logvar, min=min_logvar)
        return delta_mu, logvar

    def forward(
        self,
        x_pair_target: torch.Tensor,
        x_anchor: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_pair_target, x_anchor, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        delta_mu, delta_logvar = self.decode_residual(z, x_anchor, y)
        return delta_mu, delta_logvar, mu, logvar


def residual_gaussian_nll_terms(
    *,
    delta_mu: torch.Tensor,
    delta_true: torch.Tensor,
    delta_logvar: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_pair_shapes(delta_mu, delta_true)
    _validate_pair_shapes(delta_mu, delta_logvar)
    var = torch.exp(delta_logvar)
    logvar_term = 0.5 * delta_logvar.mean(dim=1)
    squared_error_scaled = 0.5 * ((delta_true - delta_mu).pow(2) / var).mean(dim=1)
    constant = torch.full_like(logvar_term, 0.5 * math.log(2.0 * math.pi))
    return {
        "recon_nll": logvar_term + squared_error_scaled + constant,
        "logvar_term": logvar_term,
        "squared_error_scaled": squared_error_scaled,
    }


def kl_latent_dim_mean(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have matching shapes")
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean(dim=1)


def anchored_residual_loss_terms(
    *,
    delta_mu: torch.Tensor,
    delta_true: torch.Tensor,
    delta_logvar: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    recon_terms = residual_gaussian_nll_terms(
        delta_mu=delta_mu,
        delta_true=delta_true,
        delta_logvar=delta_logvar,
    )
    kl = kl_latent_dim_mean(mu, logvar)
    loss = recon_terms["recon_nll"] + (float(beta) * kl)
    return {
        "loss": loss,
        "recon_nll": recon_terms["recon_nll"],
        "logvar_term": recon_terms["logvar_term"],
        "squared_error_scaled": recon_terms["squared_error_scaled"],
        "kl": kl,
    }


def build_anchor_pair_dataset(
    *,
    pair_projected_embeddings: torch.Tensor,
    pair_metadata: Sequence[Mapping[str, object]],
    anchor_projected_embeddings: torch.Tensor,
    anchor_metadata: Sequence[Mapping[str, object]],
    source_domain: str,
    label_values: Sequence[int],
    pairs_per_sample: int,
    neighbor_k: int,
    seed: int,
    pair_split: str,
    anchor_split: str,
) -> AnchorPairDataset:
    """Build same-class source pairs. Generation must pass source_train/source_train."""

    if int(pairs_per_sample) <= 0:
        raise ValueError("pairs_per_sample must be positive")
    anchor_index = build_source_anchor_index(
        source_projected_embeddings=anchor_projected_embeddings,
        source_metadata=anchor_metadata,
        source_domain=source_domain,
        label_values=label_values,
        neighbor_k=neighbor_k,
    )
    pair_rows = [
        idx
        for idx, row in enumerate(pair_metadata)
        if str(_domain(row)) == str(source_domain)
        and int(row.get("label", -1)) in {int(v) for v in label_values}
    ]
    if not pair_rows:
        raise ProtocolError(f"No pair rows for source_domain={source_domain}, split={pair_split}")

    anchor_by_label: dict[int, list[int]] = {}
    for local_idx, label in enumerate(anchor_index.labels.tolist()):
        anchor_by_label.setdefault(int(label), []).append(local_idx)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    targets: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    labels: list[int] = []
    pair_ids: list[str] = []
    anchor_ids: list[str] = []
    ranks: list[int] = []
    for pair_idx in pair_rows:
        row = pair_metadata[pair_idx]
        label = int(row["label"])
        pair_id = _sample_id(row, fallback=pair_idx)
        candidates = _same_class_anchor_candidates(
            pair_embedding=pair_projected_embeddings[pair_idx],
            pair_sample_id=pair_id,
            label=label,
            anchor_index=anchor_index,
            neighbor_k=neighbor_k,
        )
        if not candidates:
            raise ProtocolError(f"No valid same-class anchors for sample_id={pair_id}, label={label}")
        for _ in range(int(pairs_per_sample)):
            choice_pos = int(torch.randint(len(candidates), (1,), generator=gen).item())
            anchor_local, rank = candidates[choice_pos]
            targets.append(pair_projected_embeddings[pair_idx])
            anchors.append(anchor_index.embeddings[anchor_local])
            labels.append(label)
            pair_ids.append(pair_id)
            anchor_ids.append(anchor_index.sample_ids[anchor_local])
            ranks.append(int(rank))
    return AnchorPairDataset(
        pair_targets=torch.stack(targets).float(),
        anchors=torch.stack(anchors).float(),
        labels=torch.tensor(labels, dtype=torch.long),
        pair_sample_ids=tuple(pair_ids),
        anchor_sample_ids=tuple(anchor_ids),
        anchor_neighbor_ranks=tuple(ranks),
        pair_split=str(pair_split),
        anchor_split=str(anchor_split),
    )


def build_source_anchor_index(
    *,
    source_projected_embeddings: torch.Tensor,
    source_metadata: Sequence[Mapping[str, object]],
    source_domain: str,
    label_values: Sequence[int],
    neighbor_k: int,
) -> SourceAnchorIndex:
    rows = [
        idx
        for idx, row in enumerate(source_metadata)
        if str(_domain(row)) == str(source_domain)
        and int(row.get("label", -1)) in {int(v) for v in label_values}
    ]
    if not rows:
        raise ProtocolError(f"No source-train rows for source_domain={source_domain}")
    embeddings = source_projected_embeddings[rows].detach().cpu().float()
    labels = torch.tensor([int(source_metadata[idx]["label"]) for idx in rows], dtype=torch.long)
    sample_ids = tuple(_sample_id(source_metadata[idx], fallback=idx) for idx in rows)
    for label in sorted(set(labels.tolist())):
        if int((labels == int(label)).sum().item()) < 2:
            raise ProtocolError(
                f"F1 requires at least two source-train samples per class; "
                f"source_domain={source_domain}, label={label}"
            )

    neighbor_indices: list[tuple[int, ...]] = []
    neighbor_ranks: list[tuple[int, ...]] = []
    for idx, (embedding, label, sample_id) in enumerate(zip(embeddings, labels.tolist(), sample_ids)):
        candidates = _same_class_anchor_candidates(
            pair_embedding=embedding,
            pair_sample_id=sample_id,
            label=int(label),
            anchor_index=None,
            anchor_embeddings=embeddings,
            anchor_labels=labels,
            anchor_sample_ids=sample_ids,
            neighbor_k=neighbor_k,
        )
        neighbor_indices.append(tuple(item[0] for item in candidates))
        neighbor_ranks.append(tuple(item[1] for item in candidates))
    return SourceAnchorIndex(
        embeddings=embeddings,
        labels=labels,
        sample_ids=sample_ids,
        neighbor_indices=tuple(neighbor_indices),
        neighbor_ranks=tuple(neighbor_ranks),
    )


def train_anchored_residual_cvae(
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
    checkpoint_metadata: Mapping[str, object],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_name}.pt"
    if ckpt.exists() and resume:
        return ckpt
    if ckpt.exists() and not resume:
        raise ProtocolError(f"F1 checkpoint already exists; use --resume or a clean artifact root: {ckpt}")

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
        train_loss = _run_f1_epoch(model, train_loader, torch_device, optimizer=optimizer)
        val_loss = _run_f1_epoch(model, val_loader, torch_device, optimizer=None)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            payload = {
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
                    "beta_effective": 1.0,
                    "history": history,
                    "best_val": best_val,
                },
            }
            torch.save(payload, ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    return ckpt


def generate_anchor_residual_embeddings(
    *,
    model: AnchoredResidualCVAE,
    anchor_index: SourceAnchorIndex,
    class_label: int,
    n_samples: int,
    seed: int,
    generation_mode: str,
    experiment_seed: int = 0,
    heldout_center: str = "",
    candidate_expert: str = "",
    support_size: int = 0,
    support_seed: int = 0,
) -> F1GeneratedBatch:
    if generation_mode not in F1_GENERATION_MODES:
        raise ProtocolError(f"Unknown F1 generation mode: {generation_mode}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")

    device = next(model.parameters()).device
    pair = _sample_generation_pairs(anchor_index, int(class_label), int(n_samples), int(seed), transfer=False)
    transfer_pair = (
        _sample_generation_pairs(anchor_index, int(class_label), int(n_samples), int(seed) + 7919, transfer=True)
        if generation_mode == F1_MODE_TRANSFER_BOOTSTRAP
        else None
    )
    x_pair_target = pair["pair_targets"].to(device)
    x_anchor = pair["anchors"].to(device)
    y = torch.full((int(n_samples),), int(class_label), dtype=torch.long, device=device)
    decoder_noise = torch.zeros_like(x_anchor)
    delta_logvar = torch.zeros_like(x_anchor)
    posterior_or_prior_source = "source_train_posterior"
    reference_delta = x_pair_target - x_anchor
    provenance_pair = pair

    with torch.no_grad():
        if generation_mode in {F1_MODE_POSTERIOR_MEAN, F1_MODE_POSTERIOR_NOISE}:
            mu_z, _logvar_z = model.encode(x_pair_target, x_anchor, y)
            delta_mu, delta_logvar = model.decode_residual(mu_z, x_anchor, y)
            if generation_mode == F1_MODE_POSTERIOR_NOISE:
                gen = _generator_for_device(device, int(seed) + 104729)
                decoder_noise = torch.exp(0.5 * delta_logvar) * _randn_like(delta_mu, generator=gen)
            embeddings = x_anchor + delta_mu + decoder_noise
        elif generation_mode == F1_MODE_PRIOR_MEAN:
            gen = _generator_for_device(device, int(seed) + 209759)
            z = torch.randn((int(n_samples), int(model.latent_dim)), generator=gen, device=device, dtype=x_anchor.dtype)
            delta_mu, delta_logvar = model.decode_residual(z, x_anchor, y)
            embeddings = x_anchor + delta_mu
            posterior_or_prior_source = "standard_normal_prior"
        elif generation_mode == F1_MODE_IDENTITY_BOOTSTRAP:
            embeddings = x_anchor
            posterior_or_prior_source = "identity_source_train_anchor"
        elif generation_mode == F1_MODE_EMPIRICAL_BOOTSTRAP:
            embeddings = x_anchor + reference_delta
            posterior_or_prior_source = "empirical_paired_source_train_residual"
        elif generation_mode == F1_MODE_TRANSFER_BOOTSTRAP:
            assert transfer_pair is not None
            transfer_ref = transfer_pair["pair_targets"].to(device)
            transfer_anchor = transfer_pair["anchors"].to(device)
            x_anchor = transfer_pair["anchor_a"].to(device)
            x_pair_target = transfer_ref
            reference_delta = transfer_ref - transfer_anchor
            embeddings = x_anchor + reference_delta
            posterior_or_prior_source = "empirical_transfer_source_train_residual"
            provenance_pair = transfer_pair
        else:
            raise ProtocolError(f"Unhandled F1 generation mode: {generation_mode}")

    embeddings_cpu = embeddings.detach().cpu().float()
    labels = torch.full((int(n_samples),), int(class_label), dtype=torch.long)
    anchors_cpu = x_anchor.detach().cpu().float()
    pair_targets_cpu = x_pair_target.detach().cpu().float()
    reference_delta_cpu = reference_delta.detach().cpu().float()
    diagnostics = _generation_diagnostics(
        embeddings=embeddings_cpu,
        anchors=anchors_cpu,
        pair_targets=pair_targets_cpu,
        reference_delta=reference_delta_cpu,
        delta_logvar=delta_logvar.detach().cpu().float(),
        decoder_noise=decoder_noise.detach().cpu().float(),
        anchor_ids=pair["anchor_ids"],
        reference_ids=pair["pair_ids"],
    )
    provenance_rows = _provenance_rows(
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
        transfer_pair=transfer_pair,
    )
    return F1GeneratedBatch(
        embeddings=embeddings_cpu,
        labels=labels,
        generation_mode=generation_mode,
        diagnostics=diagnostics,
        provenance_rows=tuple(provenance_rows),
    )


def build_f1_downstream_matrix(
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
    generator_diag_rows: list[dict[str, object]] = []
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
            support_conditions = _support_conditions(
                support_units,
                experiment_seed=support.experiment_seed,
                heldout_center=heldout,
            )
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
                raise ProtocolError(f"F1 expects binary labels 0/1, got {label_values}")

            for candidate in candidates:
                projection = _load_c41_projection(c41_artifacts_root, support.experiment_seed, candidate)
                train_projected_all = projection.transform(train_cache.embeddings)
                val_projected_all = projection.transform(val_cache.embeddings)
                candidate_train_idx = _indices_for_domain(train_cache.metadata, candidate)
                candidate_val_idx = _indices_for_domain(val_cache.metadata, candidate)
                if not candidate_train_idx or not candidate_val_idx:
                    raise ProtocolError(f"F1 requires nonempty source train/val rows for candidate={candidate}.")
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
                ckpt = train_anchored_residual_cvae(
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    out_dir=artifacts_root / "checkpoints" / f"seed{int(support.experiment_seed)}" / f"expert_{candidate}" / "f1_source_anchored_residual",
                    model_name="f1_source_anchored_residual_pca64",
                    hidden_dim=training_profile.hidden_dim,
                    latent_dim=training_profile.latent_dim,
                    lr=training_profile.lr,
                    epochs=training_profile.epochs,
                    patience=training_profile.patience,
                    batch_size=training_profile.batch_size,
                    device=device,
                    resume=resume,
                    checkpoint_metadata={
                        "generator_family": F1_GENERATOR_FAMILY,
                        "experiment_id": "F1",
                        "experiment_seed": int(support.experiment_seed),
                        "candidate_expert": str(candidate),
                        "anchor_strategy": "source_train_same_class_nn",
                        "anchor_neighbor_k": ANCHOR_NEIGHBOR_K,
                        "train_pairs_per_sample": TRAIN_PAIRS_PER_SAMPLE,
                        "projection_source": "reused_c41_full_source_train_pca64",
                    },
                )
                provenance_rows.append(
                    {
                        "experiment_seed": int(support.experiment_seed),
                        "heldout_center": heldout,
                        "candidate_expert": candidate,
                        "generator_family": F1_GENERATOR_FAMILY,
                        "checkpoint_path": str(ckpt),
                        "projection_path": str(_c41_projection_path(c41_artifacts_root, support.experiment_seed, candidate)),
                        "projection_source": "reused_c41_full_source_train_pca64",
                        "routing_family_used": BASELINE_ROUTING_FAMILY_USED,
                        "routing_scores_recomputed_for_f1": 0,
                        "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                    }
                )
                model = _load_f1_model(ckpt, device=device)
                for generation_mode in F1_GENERATION_MODES:
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            base_row, diagnostics, duplicates, sample_rows = _score_f1_candidate(
                                model=model,
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
                                label_values=label_values,
                                budget_per_class=config.primary_budget_per_class,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                            )
                            generator_diag_rows.append(diagnostics)
                            duplicate_rows.append(duplicates)
                            sample_provenance_rows.extend(sample_rows)
                            for support_size, support_seed in support_conditions:
                                row = replace(base_row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and row.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, row)
                                completed.add(row.primary_key())

    _write_csv_with_header(artifacts_root / "tables" / "f1_anchor_pair_diagnostics.csv", anchor_diag_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f1_generator_distribution_diagnostics.csv", generator_diag_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f1_duplicate_diagnostics.csv", duplicate_rows)
    _write_csv_with_header(artifacts_root / "tables" / "f1_synthetic_sample_provenance.csv", sample_provenance_rows)
    _write_csv_with_header(artifacts_root / "manifests" / "f1_generator_provenance.csv", provenance_rows)
    return matrix_path


def build_f1_routing_alignment_rows(
    *,
    selections: Sequence[SupportSelectionUnit],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    f1_rows = [row for row in downstream_rows if row.generator_family == F1_GENERATOR_FAMILY and row.status == "ok"]
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
        for row in f1_rows
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
            for row in f1_rows
            if int(row.budget_per_class) == PRIMARY_BUDGET_PER_CLASS
        }
    )
    oracle_by_context = _f1_oracles(f1_rows)
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
                raise ProtocolError(f"Missing F1 downstream row for selected expert key {selected_key}")
            oracle = oracle_by_context.get((experiment_seed, heldout, generation_mode, budget, generation_seed, classifier_seed))
            if oracle is None:
                raise ProtocolError("Missing F1 downstream oracle")
            oracle_gap = float(oracle.bacc) - float(selected.bacc)
            rows.append(
                {
                    "heldout_center": unit.heldout_center,
                    "experiment_seed": int(unit.experiment_seed),
                    "support_size": int(unit.support_size),
                    "support_seed": int(unit.support_seed),
                    "generator_family": F1_GENERATOR_FAMILY,
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
                    "routing_scores_recomputed_for_f1": 0,
                    "selected_expert_ids_source": BASELINE_SELECTED_EXPERT_IDS_SOURCE,
                    "projection_source": "reused_c41_full_source_train_pca64",
                }
            )
    return rows


def build_f1_delta_summary_rows(
    *,
    f1_alignment_rows: Sequence[Mapping[str, object]],
    c41_alignment_rows: Sequence[Mapping[str, object]],
    duplicate_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    f1_support = [row for row in f1_alignment_rows if str(row.get("method")) == SUPPORT_NELBO_METHOD]
    c41_baseline = [
        row
        for row in c41_alignment_rows
        if str(row.get("method")) == SUPPORT_NELBO_METHOD
        and str(row.get("generator_family")) == HETEROSCEDASTIC_GENERATOR_FAMILY
        and str(row.get("generation_mode")) == POSTERIOR_DECODER_MEAN_GENERATION_MODE
    ]
    groups = sorted({(str(row["heldout_center"]), int(row["support_size"])) for row in f1_support})
    rows: list[dict[str, object]] = []
    for heldout, support_size in groups:
        c41_subset = _subset(c41_baseline, heldout, support_size, POSTERIOR_DECODER_MEAN_GENERATION_MODE)
        if not c41_subset:
            continue
        c41_selected = _mean(c41_subset, "selected_bacc")
        c41_oracle = _mean(c41_subset, "oracle_bacc")
        c41_gap = _mean(c41_subset, "downstream_oracle_gap_bacc")
        identity_selected = _mean(_subset(f1_support, heldout, support_size, F1_MODE_IDENTITY_BOOTSTRAP), "selected_bacc")
        empirical_selected = _mean(_subset(f1_support, heldout, support_size, F1_MODE_EMPIRICAL_BOOTSTRAP), "selected_bacc")
        transfer_selected = _mean(_subset(f1_support, heldout, support_size, F1_MODE_TRANSFER_BOOTSTRAP), "selected_bacc")
        for mode in F1_GENERATION_MODES:
            mode_subset = _subset(f1_support, heldout, support_size, mode)
            if not mode_subset:
                continue
            selected = _mean(mode_subset, "selected_bacc")
            oracle = _mean(mode_subset, "oracle_bacc")
            gap = _mean(mode_subset, "downstream_oracle_gap_bacc")
            near_copy = _near_copy_failure(duplicate_rows, heldout, mode)
            decision = _f1_decision_label(
                mode=mode,
                selected=selected,
                oracle_gap=gap,
                c41_selected=c41_selected,
                c41_gap=c41_gap,
                identity_selected=identity_selected,
                empirical_selected=empirical_selected,
                transfer_selected=transfer_selected,
                near_copy=near_copy,
            )
            rows.append(
                {
                    "heldout_center": heldout,
                    "support_size": support_size,
                    "generation_mode": mode,
                    "selected_bacc_f1": selected,
                    "oracle_bacc_f1": oracle,
                    "oracle_gap_f1": gap,
                    "selected_bacc_c41_hetero_mean": c41_selected,
                    "oracle_bacc_c41_hetero_mean": c41_oracle,
                    "oracle_gap_c41_hetero_mean": c41_gap,
                    "selected_bacc_delta_vs_c41_hetero_mean": selected - c41_selected,
                    "oracle_bacc_delta_vs_c41_hetero_mean": oracle - c41_oracle,
                    "oracle_gap_delta_vs_c41_hetero_mean": gap - c41_gap,
                    "selected_bacc_anchor_identity_bootstrap": identity_selected,
                    "selected_bacc_anchor_residual_empirical_bootstrap": empirical_selected,
                    "selected_bacc_anchor_empirical_residual_transfer_bootstrap": transfer_selected,
                    "beats_identity_bootstrap": int(selected > identity_selected),
                    "beats_empirical_bootstrap": int(selected > empirical_selected),
                    "beats_transfer_bootstrap": int(selected > transfer_selected),
                    "selected_ge_080": int(selected >= 0.80),
                    "diagnostic_only": int(mode in F1_DIAGNOSTIC_ONLY_MODES),
                    "decision_label": decision,
                }
            )
    return rows


def write_f1_alignment_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F1_ALIGNMENT_COLUMNS, rows)


def write_f1_delta_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, F1_DELTA_COLUMNS, rows)


def load_f1_diagnostics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _run_f1_epoch(
    model: AnchoredResidualCVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    with torch.enable_grad() if optimizer is not None else torch.no_grad():
        for x_pair_target, x_anchor, y in loader:
            x_pair_target = x_pair_target.to(device)
            x_anchor = x_anchor.to(device)
            y = y.to(device)
            delta_true = x_pair_target - x_anchor
            delta_mu, delta_logvar, mu, logvar = model(x_pair_target, x_anchor, y)
            terms = anchored_residual_loss_terms(
                delta_mu=delta_mu,
                delta_true=delta_true,
                delta_logvar=delta_logvar,
                mu=mu,
                logvar=logvar,
            )
            loss = terms["loss"].mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * int(x_pair_target.shape[0])
            count += int(x_pair_target.shape[0])
    return total / float(max(count, 1))


def _score_f1_candidate(
    *,
    model: AnchoredResidualCVAE,
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
    label_values: Sequence[int],
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
) -> tuple[CandidateDownstreamRow, dict[str, object], dict[str, object], list[dict[str, object]]]:
    try:
        chunks: list[torch.Tensor] = []
        labels: list[int] = []
        diagnostic_parts: list[Mapping[str, float]] = []
        provenance_rows: list[dict[str, object]] = []
        for label in label_values:
            generated = generate_anchor_residual_embeddings(
                model=model,
                anchor_index=anchor_index,
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
        source_train_pca = projection.transform(train_cache.embeddings[source_train_idx])
        source_train_dino = train_cache.embeddings[source_train_idx]
        diagnostics = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": F1_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "diagnostic_only": int(generation_mode in F1_DIAGNOSTIC_ONLY_MODES),
            **_aggregate_float_dicts(diagnostic_parts),
            **_generated_distribution_diagnostics(
                synthetic_embeddings=synthetic_embeddings,
                synthetic_labels=labels,
                source_train_pca=source_train_pca,
                source_train_dino=source_train_dino,
                projection=projection,
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
            )
        }
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=F1_GENERATOR_FAMILY,
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
            plain_baseline_source="f1_uses_reused_c41_projection",
            plain_baseline_artifact_path="",
            plain_baseline_training_profile="f1_source_anchored_residual",
            plain_baseline_matches_locked_hparams=0,
            routing_family_used=BASELINE_ROUTING_FAMILY_USED,
            routing_scores_recomputed_for_heteroscedastic=0,
            selected_expert_ids_source=BASELINE_SELECTED_EXPERT_IDS_SOURCE,
        )
        return row, diagnostics, duplicate, provenance_rows
    except Exception as exc:
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=F1_GENERATOR_FAMILY,
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
            status="failed_f1_candidate_scoring",
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
            "generator_family": F1_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "status": row.status,
            "error_message": row.error_message,
        }
        return row, diagnostics, dict(diagnostics), []


def _sample_generation_pairs(
    anchor_index: SourceAnchorIndex,
    class_label: int,
    n_samples: int,
    seed: int,
    *,
    transfer: bool,
) -> dict[str, object]:
    class_rows = [idx for idx, label in enumerate(anchor_index.labels.tolist()) if int(label) == int(class_label)]
    if len(class_rows) < 2:
        raise ProtocolError(f"F1 generation requires at least two source-train rows for class={class_label}")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pair_targets: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    anchor_a: list[torch.Tensor] = []
    pair_ids: list[str] = []
    anchor_ids: list[str] = []
    anchor_a_ids: list[str] = []
    residual_anchor_ids: list[str] = []
    ranks: list[int] = []
    for _ in range(int(n_samples)):
        pair_local = class_rows[int(torch.randint(len(class_rows), (1,), generator=gen).item())]
        neighbors = anchor_index.neighbor_indices[pair_local]
        neighbor_ranks = anchor_index.neighbor_ranks[pair_local]
        if not neighbors:
            raise ProtocolError(f"Source row has no same-class anchor: {anchor_index.sample_ids[pair_local]}")
        choice = int(torch.randint(len(neighbors), (1,), generator=gen).item())
        anchor_local = int(neighbors[choice])
        pair_targets.append(anchor_index.embeddings[pair_local])
        anchors.append(anchor_index.embeddings[anchor_local])
        pair_ids.append(anchor_index.sample_ids[pair_local])
        anchor_ids.append(anchor_index.sample_ids[anchor_local])
        residual_anchor_ids.append(anchor_index.sample_ids[anchor_local])
        ranks.append(int(neighbor_ranks[choice]))
        if transfer:
            anchor_a_local = class_rows[int(torch.randint(len(class_rows), (1,), generator=gen).item())]
            anchor_a.append(anchor_index.embeddings[anchor_a_local])
            anchor_a_ids.append(anchor_index.sample_ids[anchor_a_local])
    payload: dict[str, object] = {
        "pair_targets": torch.stack(pair_targets).float(),
        "anchors": torch.stack(anchors).float(),
        "pair_ids": tuple(pair_ids),
        "anchor_ids": tuple(anchor_ids),
        "residual_anchor_ids": tuple(residual_anchor_ids),
        "anchor_neighbor_ranks": tuple(ranks),
    }
    if transfer:
        payload["anchor_a"] = torch.stack(anchor_a).float()
        payload["anchor_a_ids"] = tuple(anchor_a_ids)
    return payload


def _generation_diagnostics(
    *,
    embeddings: torch.Tensor,
    anchors: torch.Tensor,
    pair_targets: torch.Tensor,
    reference_delta: torch.Tensor,
    delta_logvar: torch.Tensor,
    decoder_noise: torch.Tensor,
    anchor_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> dict[str, float]:
    gen_residual = embeddings - anchors
    dist_anchor = gen_residual.norm(dim=1)
    dist_ref = (embeddings - pair_targets).norm(dim=1)
    ref_norm = reference_delta.norm(dim=1).clamp_min(1.0e-12)
    generated_norm = gen_residual.norm(dim=1)
    logvar_at_min = (delta_logvar <= -9.21 + 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    logvar_at_max = (delta_logvar >= 2.0 - 1.0e-6).float().mean() if delta_logvar.numel() else torch.tensor(0.0)
    noise_energy = decoder_noise.pow(2).sum(dim=1).mean() if decoder_noise.numel() else torch.tensor(0.0)
    mean_energy = embeddings.pow(2).sum(dim=1).mean().clamp_min(1.0e-12)
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
        "decoder_logvar_mean": float(delta_logvar.mean().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_min": float(delta_logvar.min().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_max": float(delta_logvar.max().item()) if delta_logvar.numel() else math.nan,
        "decoder_logvar_at_min_frac": float(logvar_at_min.item()),
        "decoder_logvar_at_max_frac": float(logvar_at_max.item()),
        "decoder_noise_energy_ratio": float((noise_energy / mean_energy).item()),
    }


def _generated_distribution_diagnostics(
    *,
    synthetic_embeddings: torch.Tensor,
    synthetic_labels: Sequence[int],
    source_train_pca: torch.Tensor,
    source_train_dino: torch.Tensor,
    projection: SourceTrainPCAProjection,
) -> dict[str, float]:
    synthetic_dino = projection.inverse_transform(synthetic_embeddings)
    return {
        "generated_pca_std_ratio": _std_ratio(synthetic_embeddings, source_train_pca),
        "generated_dino_std_ratio": _std_ratio(synthetic_dino, source_train_dino),
        "synthetic_pca64_cov_trace_ratio_to_source_train": _trace_cov(synthetic_embeddings) / max(_trace_cov(source_train_pca), 1.0e-12),
        "synthetic_pairwise_distance_ratio_to_source_train": _pairwise_distance_mean(synthetic_embeddings) / max(_pairwise_distance_mean(source_train_pca), 1.0e-12),
        "synthetic_count_class_0": float(sum(1 for label in synthetic_labels if int(label) == 0)),
        "synthetic_count_class_1": float(sum(1 for label in synthetic_labels if int(label) == 1)),
        "nan_or_inf_generated": float(int(not torch.isfinite(synthetic_embeddings).all().item())),
    }


def _provenance_rows(
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
    transfer_pair: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    pair_ids = tuple(str(v) for v in pair["pair_ids"])
    anchor_ids = tuple(str(v) for v in pair["anchor_ids"])
    anchor_a_ids = tuple(str(v) for v in pair.get("anchor_a_ids", anchor_ids))
    residual_anchor_ids = tuple(str(v) for v in pair.get("residual_anchor_ids", anchor_ids))
    rows: list[dict[str, object]] = []
    for idx, (pair_id, anchor_id) in enumerate(zip(pair_ids, anchor_ids)):
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
                "synthetic_anchor_id": anchor_a_ids[idx] if generation_mode == F1_MODE_TRANSFER_BOOTSTRAP else anchor_id,
                "residual_reference_sample_id": pair_id,
                "residual_anchor_id": residual_anchor_ids[idx],
                "anchor_split": "source_train",
                "residual_reference_split": "source_train",
                "same_class_anchor": 1,
                "posterior_or_prior_source": posterior_or_prior_source,
            }
        )
    return rows


def _anchor_pair_diagnostics(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    train_pairs: AnchorPairDataset,
    val_pairs: AnchorPairDataset,
    source_anchor_index: SourceAnchorIndex,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "candidate_expert": candidate_expert,
        "anchor_strategy": "source_train_same_class_nn",
        "anchor_neighbor_k": ANCHOR_NEIGHBOR_K,
        "train_pairs_per_sample": TRAIN_PAIRS_PER_SAMPLE,
        "train_pair_count": int(train_pairs.labels.numel()),
        "val_pair_count": int(val_pairs.labels.numel()),
        "source_train_anchor_count": int(source_anchor_index.labels.numel()),
        "train_anchor_neighbor_rank_mean": _mean_float(train_pairs.anchor_neighbor_ranks),
        "val_anchor_neighbor_rank_mean": _mean_float(val_pairs.anchor_neighbor_ranks),
        "source_train_count_class_0": int((source_anchor_index.labels == 0).sum().item()),
        "source_train_count_class_1": int((source_anchor_index.labels == 1).sum().item()),
        "train_anchor_split": train_pairs.anchor_split,
        "val_anchor_split": val_pairs.anchor_split,
        "generation_anchor_split": "source_train",
    }


def _same_class_anchor_candidates(
    *,
    pair_embedding: torch.Tensor,
    pair_sample_id: str,
    label: int,
    neighbor_k: int,
    anchor_index: SourceAnchorIndex | None = None,
    anchor_embeddings: torch.Tensor | None = None,
    anchor_labels: torch.Tensor | None = None,
    anchor_sample_ids: Sequence[str] | None = None,
) -> list[tuple[int, int]]:
    if anchor_index is not None:
        anchor_embeddings = anchor_index.embeddings
        anchor_labels = anchor_index.labels
        anchor_sample_ids = anchor_index.sample_ids
    if anchor_embeddings is None or anchor_labels is None or anchor_sample_ids is None:
        raise ValueError("anchor embeddings, labels, and sample ids are required")
    candidate_indices = [
        idx
        for idx, anchor_label in enumerate(anchor_labels.tolist())
        if int(anchor_label) == int(label) and str(anchor_sample_ids[idx]) != str(pair_sample_id)
    ]
    if not candidate_indices:
        return []
    candidates = anchor_embeddings[candidate_indices]
    distances = (candidates - pair_embedding.detach().cpu().float()).pow(2).sum(dim=1)
    order = torch.argsort(distances).tolist()
    limited = order[: max(1, int(neighbor_k))]
    return [(int(candidate_indices[pos]), int(rank + 1)) for rank, pos in enumerate(limited)]


def _load_c41_projection(c41_root: Path, experiment_seed: int, candidate_expert: str) -> SourceTrainPCAProjection:
    path = _c41_projection_path(c41_root, experiment_seed, candidate_expert)
    if not path.exists():
        raise ProtocolError(f"Missing C4.1 source-train PCA projection for F1: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _c41_projection_path(c41_root: Path, experiment_seed: int, candidate_expert: str) -> Path:
    return c41_root / "projections" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / "pca64.pt"


def _load_f1_model(path: Path, *, device: str) -> AnchoredResidualCVAE:
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


def _f1_oracles(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[int, str, str, int, int, int], CandidateDownstreamRow]:
    grouped: dict[tuple[int, str, str, int, int, int], list[CandidateDownstreamRow]] = {}
    for row in rows:
        if row.generator_family != F1_GENERATOR_FAMILY or row.status != "ok":
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


def _f1_decision_label(
    *,
    mode: str,
    selected: float,
    oracle_gap: float,
    c41_selected: float,
    c41_gap: float,
    identity_selected: float,
    empirical_selected: float,
    transfer_selected: float,
    near_copy: bool,
) -> str:
    if mode != F1_MODE_POSTERIOR_MEAN:
        return DECISION_DIAGNOSTIC_ONLY
    if near_copy:
        return DECISION_NEAR_COPY
    beats_bootstrap = selected > identity_selected and selected > empirical_selected and selected > transfer_selected
    if selected >= 0.80 and selected > c41_selected and beats_bootstrap and oracle_gap <= c41_gap:
        return DECISION_GENERATOR_SUCCESS
    if selected >= 0.80 or max(identity_selected, empirical_selected, transfer_selected) >= 0.80:
        return DECISION_SOURCE_BOOTSTRAP
    if selected > c41_selected and not beats_bootstrap:
        return DECISION_SOURCE_BOOTSTRAP
    if oracle_gap > c41_gap + 0.02:
        return DECISION_ROUTING_BOTTLENECK
    return DECISION_NO_GAIN


def _near_copy_failure(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> bool:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    if not subset:
        return False
    anchor_dup = _mean(subset, "fraction_exact_or_near_duplicate_anchor")
    ref_dup = _mean(subset, "fraction_exact_or_near_duplicate_reference")
    return bool(max(anchor_dup, ref_dup) > 0.95)


def _subset(rows: Sequence[Mapping[str, object]], heldout: str, support_size: int, mode: str) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("heldout_center")) == str(heldout)
        and int(row.get("support_size", 0)) == int(support_size)
        and str(row.get("generation_mode")) == str(mode)
    ]


def _aggregate_float_dicts(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: _nanmean(float(item[key]) for item in items if key in item) for key in keys}


def _std_ratio(generated: torch.Tensor, reference: torch.Tensor) -> float:
    gen_std = generated.detach().cpu().float().std(dim=0, unbiased=False).mean()
    ref_std = reference.detach().cpu().float().std(dim=0, unbiased=False).mean().clamp_min(1.0e-12)
    return float((gen_std / ref_std).item())


def _trace_cov(x: torch.Tensor) -> float:
    arr = x.detach().cpu().float()
    if int(arr.shape[0]) <= 1:
        return 0.0
    return float(arr.var(dim=0, unbiased=True).sum().item())


def _pairwise_distance_mean(x: torch.Tensor, *, max_points: int = 512) -> float:
    arr = x.detach().cpu().float()
    if int(arr.shape[0]) > int(max_points):
        arr = arr[: int(max_points)]
    if int(arr.shape[0]) <= 1:
        return 0.0
    return float(torch.pdist(arr).mean().item())


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return _nanmean(_as_float(row.get(key, math.nan)) for row in rows)


def _mean_float(values: Iterable[float | int]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / float(len(vals)) if vals else math.nan


def _nanmean(values: Iterable[float]) -> float:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    return sum(cleaned) / float(len(cleaned)) if cleaned else math.nan


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _sample_id(row: Mapping[str, object], *, fallback: object) -> str:
    value = str(row.get("sample_id", "")).strip()
    return value if value else str(fallback)


def _validate_pair_shapes(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("F1 tensors must be 2D")
    if left.shape != right.shape:
        raise ValueError(f"F1 tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")


def _reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(ch)) for ch in str(value))


def _write_csv_with_header(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if rows:
        _write_dict_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _limit_c41_artifacts(artifacts: Sequence[object], experiment_seeds: Sequence[int] | None) -> tuple[object, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.support.experiment_seed) in allowed)
