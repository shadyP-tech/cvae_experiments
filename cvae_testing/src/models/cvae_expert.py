from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DECODER_LIKELIHOOD_MSE = "mse"
DECODER_LIKELIHOOD_GAUSSIAN_DIAG = "gaussian_diag"
RECON_LOSS_MSE = "mse"
RECON_LOSS_GAUSSIAN_NLL_DIAG = "gaussian_nll_diag"
REDUCTION_SUM = "sum"
REDUCTION_MEAN = "mean"


class CVAEExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        metadata_dim: int = 0,
        metadata_constraint_cfg: dict[str, object] | None = None,
        aux_metadata_dim: int | None = None,
        class_condition_dim: int = 0,
        decoder_likelihood: str = DECODER_LIKELIHOOD_MSE,
        decoder_logvar_min: float = -9.21,
        decoder_logvar_max: float = 2.0,
        decoder_min_variance: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.metadata_dim = int(metadata_dim)
        if self.metadata_dim < 0:
            raise ValueError("metadata_dim must be >= 0")
        self.class_condition_dim = int(class_condition_dim)
        if self.class_condition_dim < 0:
            raise ValueError("class_condition_dim must be >= 0")
        if self.metadata_dim > 0 and self.class_condition_dim > 0:
            raise ValueError("CVAEExpert does not support mixing metadata conditioning and class conditioning")

        self.decoder_likelihood = str(decoder_likelihood).strip().lower()
        allowed_decoder_likelihoods = {DECODER_LIKELIHOOD_MSE, DECODER_LIKELIHOOD_GAUSSIAN_DIAG}
        if self.decoder_likelihood not in allowed_decoder_likelihoods:
            raise ValueError(
                "decoder_likelihood must be one of "
                f"{sorted(allowed_decoder_likelihoods)}. Got: {self.decoder_likelihood}"
            )
        self.decoder_logvar_min = float(decoder_logvar_min)
        self.decoder_logvar_max = float(decoder_logvar_max)
        if self.decoder_logvar_min > self.decoder_logvar_max:
            raise ValueError("decoder_logvar_min must be <= decoder_logvar_max")
        self.decoder_min_variance = float(decoder_min_variance)
        if self.decoder_min_variance <= 0:
            raise ValueError("decoder_min_variance must be > 0")

        self.conditioning_enabled = self.metadata_dim > 0
        self.class_conditioning_enabled = self.class_condition_dim > 0
        enc_input_dim = (
            self.input_dim
            + (self.metadata_dim if self.conditioning_enabled else 0)
            + (self.class_condition_dim if self.class_conditioning_enabled else 0)
        )
        dec_input_dim = (
            self.latent_dim
            + (self.metadata_dim if self.conditioning_enabled else 0)
            + (self.class_condition_dim if self.class_conditioning_enabled else 0)
        )

        self.enc = nn.Linear(enc_input_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(dec_input_dim, self.hidden_dim)
        self.dec2 = nn.Linear(self.hidden_dim, self.input_dim)
        self.dec_logvar: nn.Module | None = None
        if self.decoder_likelihood == DECODER_LIKELIHOOD_GAUSSIAN_DIAG:
            self.dec_logvar = nn.Linear(self.hidden_dim, self.input_dim)

        constraint_cfg = metadata_constraint_cfg or {}
        self.metadata_constraint_enabled = bool(constraint_cfg.get("enabled", False))
        self.metadata_constraint_variant = str(constraint_cfg.get("variant", "aux_head")).strip().lower()
        allowed_constraint_variants = {"aux_head", "conditional_prior"}
        if self.metadata_constraint_enabled and self.metadata_constraint_variant not in allowed_constraint_variants:
            raise ValueError(
                "metadata_constraint.variant must be one of "
                f"{sorted(allowed_constraint_variants)}. "
                f"Got: {self.metadata_constraint_variant}"
            )

        self.metadata_constraint_weight = float(constraint_cfg.get("aux_weight", 0.0))
        if self.metadata_constraint_weight < 0:
            raise ValueError("metadata_constraint.aux_weight must be >= 0")

        self.metadata_constraint_use_mu = bool(constraint_cfg.get("use_mu", True))

        self.aux_metadata_dim = int(aux_metadata_dim if aux_metadata_dim is not None else self.metadata_dim)
        if self.aux_metadata_dim < 0:
            raise ValueError("aux_metadata_dim must be >= 0")

        self.metadata_aux_head: nn.Module | None = None
        self.metadata_prior_mu_head: nn.Module | None = None
        self.metadata_prior_logvar_head: nn.Module | None = None
        self.metadata_prior_logvar_min = float(constraint_cfg.get("prior_logvar_min", -6.0))
        self.metadata_prior_logvar_max = float(constraint_cfg.get("prior_logvar_max", 2.0))
        if self.metadata_prior_logvar_min > self.metadata_prior_logvar_max:
            raise ValueError(
                "metadata_constraint.prior_logvar_min must be <= "
                "metadata_constraint.prior_logvar_max"
            )

        if self.metadata_constraint_enabled:
            if self.aux_metadata_dim <= 1:
                raise ValueError(
                    "metadata_constraint requires aux_metadata_dim > 1 for classification. "
                    f"Got: {self.aux_metadata_dim}"
                )
            if self.metadata_constraint_variant == "aux_head":
                aux_head_hidden_dim = int(constraint_cfg.get("head_hidden_dim", 0))
                if aux_head_hidden_dim < 0:
                    raise ValueError("metadata_constraint.head_hidden_dim must be >= 0")
                if aux_head_hidden_dim > 0:
                    self.metadata_aux_head = nn.Sequential(
                        nn.Linear(self.latent_dim, aux_head_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(aux_head_hidden_dim, self.aux_metadata_dim),
                    )
                else:
                    self.metadata_aux_head = nn.Linear(self.latent_dim, self.aux_metadata_dim)
            elif self.metadata_constraint_variant == "conditional_prior":
                prior_hidden_dim = int(constraint_cfg.get("prior_hidden_dim", 0))
                if prior_hidden_dim < 0:
                    raise ValueError("metadata_constraint.prior_hidden_dim must be >= 0")
                if prior_hidden_dim > 0:
                    self.metadata_prior_mu_head = nn.Sequential(
                        nn.Linear(self.aux_metadata_dim, prior_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(prior_hidden_dim, self.latent_dim),
                    )
                    self.metadata_prior_logvar_head = nn.Sequential(
                        nn.Linear(self.aux_metadata_dim, prior_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(prior_hidden_dim, self.latent_dim),
                    )
                else:
                    self.metadata_prior_mu_head = nn.Linear(self.aux_metadata_dim, self.latent_dim)
                    self.metadata_prior_logvar_head = nn.Linear(self.aux_metadata_dim, self.latent_dim)

    def _concat_metadata(self, x: torch.Tensor, m: torch.Tensor | None, stage: str) -> torch.Tensor:
        if not self.conditioning_enabled:
            return x
        if m is None:
            raise ValueError(f"Metadata tensor is required for conditioning at stage '{stage}'.")
        if x.ndim != 2 or m.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors for conditioning at stage '{stage}', got x.ndim={x.ndim}, m.ndim={m.ndim}."
            )
        if x.shape[0] != m.shape[0]:
            raise ValueError(
                f"Batch-size mismatch at stage '{stage}': x has {x.shape[0]} rows, metadata has {m.shape[0]} rows."
            )
        if m.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Metadata width mismatch at stage '{stage}': expected {self.metadata_dim}, got {m.shape[1]}."
            )
        return torch.cat([x, m], dim=1)

    def _class_condition_to_one_hot(self, y: torch.Tensor | None, stage: str, device: torch.device) -> torch.Tensor | None:
        if not self.class_conditioning_enabled:
            return None
        if y is None:
            raise ValueError(f"Class-condition tensor is required for conditioning at stage '{stage}'.")
        if y.ndim == 1:
            targets = y.long()
            min_target = int(targets.min().item()) if targets.numel() else 0
            max_target = int(targets.max().item()) if targets.numel() else -1
            if min_target < 0 or max_target >= self.class_condition_dim:
                raise ValueError(
                    "Class-condition indices are out of range: "
                    f"min={min_target}, max={max_target}, class_condition_dim={self.class_condition_dim}"
                )
            return F.one_hot(targets.to(device=device), num_classes=self.class_condition_dim).to(dtype=torch.float32)
        if y.ndim == 2:
            if y.shape[1] != self.class_condition_dim:
                raise ValueError(
                    f"Class-condition width mismatch at stage '{stage}': "
                    f"expected {self.class_condition_dim}, got {y.shape[1]}."
                )
            return y.to(device=device, dtype=torch.float32)
        raise ValueError(
            f"Class-condition tensor must be 1D indices or 2D one-hot at stage '{stage}', got y.ndim={y.ndim}."
        )

    def _concat_conditioning(
        self,
        x: torch.Tensor,
        m: torch.Tensor | None,
        y: torch.Tensor | None,
        stage: str,
    ) -> torch.Tensor:
        out = self._concat_metadata(x, m, stage=stage)
        y_one_hot = self._class_condition_to_one_hot(y, stage=stage, device=x.device)
        if y_one_hot is None:
            return out
        if out.ndim != 2 or y_one_hot.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors for class conditioning at stage '{stage}', "
                f"got x.ndim={out.ndim}, y.ndim={y_one_hot.ndim}."
            )
        if out.shape[0] != y_one_hot.shape[0]:
            raise ValueError(
                f"Batch-size mismatch at stage '{stage}': x has {out.shape[0]} rows, "
                f"class condition has {y_one_hot.shape[0]} rows."
            )
        return torch.cat([out, y_one_hot], dim=1)

    def encode(
        self,
        x: torch.Tensor,
        m: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ):
        x_enc = self._concat_conditioning(x, m=m, y=y, stage="encoder")
        h = F.relu(self.enc(x_enc))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_distribution(
        self,
        z: torch.Tensor,
        m: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z_dec = self._concat_conditioning(z, m=m, y=y, stage="decoder")
        h = F.relu(self.dec1(z_dec))
        mean = self.dec2(h)
        if self.decoder_likelihood != DECODER_LIKELIHOOD_GAUSSIAN_DIAG:
            return mean, None
        if self.dec_logvar is None:
            raise ValueError("Gaussian decoder likelihood requires a logvar head")
        logvar = torch.clamp(
            self.dec_logvar(h),
            min=self.decoder_logvar_min,
            max=self.decoder_logvar_max,
        )
        min_logvar = math.log(self.decoder_min_variance)
        if min_logvar > self.decoder_logvar_min:
            logvar = torch.clamp(logvar, min=min_logvar)
        return mean, logvar

    def decode(
        self,
        z: torch.Tensor,
        m: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        return_distribution: bool = False,
    ):
        mean, recon_logvar = self.decode_distribution(z, m=m, y=y)
        if return_distribution:
            return mean, recon_logvar
        return mean

    def _metadata_targets_to_indices(self, metadata_targets: torch.Tensor | None) -> torch.Tensor:
        if metadata_targets is None:
            raise ValueError("metadata targets are required for metadata-constraint computations")

        if metadata_targets.ndim == 2:
            if metadata_targets.shape[1] != self.aux_metadata_dim:
                raise ValueError(
                    "Metadata target width mismatch for metadata-constraint computations: "
                    f"expected {self.aux_metadata_dim}, got {metadata_targets.shape[1]}."
                )
            targets = metadata_targets.argmax(dim=1)
        elif metadata_targets.ndim == 1:
            targets = metadata_targets.long()
        else:
            raise ValueError(
                "metadata targets must be a 1D index tensor or 2D one-hot tensor"
            )

        max_target = int(targets.max().item()) if targets.numel() > 0 else -1
        min_target = int(targets.min().item()) if targets.numel() > 0 else 0
        if min_target < 0 or max_target >= self.aux_metadata_dim:
            raise ValueError(
                "metadata target indices are out of range for metadata constraint: "
                f"min={min_target}, max={max_target}, aux_metadata_dim={self.aux_metadata_dim}"
            )

        return targets.long()

    def _metadata_targets_to_one_hot(self, metadata_targets: torch.Tensor | None) -> torch.Tensor:
        targets = self._metadata_targets_to_indices(metadata_targets)
        return F.one_hot(targets, num_classes=self.aux_metadata_dim).to(dtype=torch.float32)

    def metadata_constraint_logits(self, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor | None:
        if not self.metadata_constraint_enabled or self.metadata_constraint_variant != "aux_head":
            return None
        if self.metadata_aux_head is None:
            return None
        latent = mu if self.metadata_constraint_use_mu else z
        return self.metadata_aux_head(latent)

    def metadata_constraint_prior(
        self,
        metadata_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
        if not self.metadata_constraint_enabled or self.metadata_constraint_variant != "conditional_prior":
            return None, None, 1.0
        if self.metadata_prior_mu_head is None or self.metadata_prior_logvar_head is None:
            raise ValueError("Conditional prior heads are not initialized for variant='conditional_prior'")

        metadata_one_hot = self._metadata_targets_to_one_hot(metadata_targets)
        prior_mu = self.metadata_prior_mu_head(metadata_one_hot)
        prior_logvar_raw = self.metadata_prior_logvar_head(metadata_one_hot)
        prior_logvar = torch.clamp(
            prior_logvar_raw,
            min=self.metadata_prior_logvar_min,
            max=self.metadata_prior_logvar_max,
        )
        kl_weight = float(self.metadata_constraint_weight) if self.metadata_constraint_weight > 0 else 1.0
        return prior_mu, prior_logvar, kl_weight

    def metadata_constraint_loss(
        self,
        aux_logits: torch.Tensor | None,
        metadata_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.metadata_constraint_enabled:
            raise ValueError("metadata_constraint_loss called while metadata constraint is disabled")
        if self.metadata_constraint_variant != "aux_head":
            raise ValueError("metadata_constraint_loss is only defined for metadata_constraint.variant='aux_head'")
        if aux_logits is None:
            raise ValueError("aux_logits must not be None when metadata constraint is enabled")
        targets = self._metadata_targets_to_indices(metadata_targets)

        if targets.shape[0] != aux_logits.shape[0]:
            raise ValueError(
                "Batch-size mismatch for metadata auxiliary loss: "
                f"targets={targets.shape[0]}, logits={aux_logits.shape[0]}"
            )

        return F.cross_entropy(aux_logits, targets)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        return_aux: bool = False,
        return_distribution: bool = False,
    ):
        mu, logvar = self.encode(x, m=m, y=y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, m=m, y=y, return_distribution=return_distribution)
        aux_logits = self.metadata_constraint_logits(mu=mu, z=z)
        if return_aux:
            return recon, mu, logvar, aux_logits
        return recon, mu, logvar


def gaussian_nll_diag_terms(
    recon_mu: torch.Tensor,
    x: torch.Tensor,
    recon_logvar: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if recon_mu.shape != x.shape or recon_logvar.shape != x.shape:
        raise ValueError("recon_mu, recon_logvar, and x must share shape for gaussian_nll_diag")
    var = torch.exp(recon_logvar)
    logvar_term = 0.5 * recon_logvar.mean(dim=1)
    squared_error_scaled = 0.5 * ((x - recon_mu).pow(2) / var).mean(dim=1)
    constant = torch.full_like(logvar_term, 0.5 * math.log(2.0 * math.pi))
    return {
        "recon_nll": logvar_term + squared_error_scaled + constant,
        "logvar_term": logvar_term,
        "squared_error_scaled": squared_error_scaled,
    }


def _reduce_feature_terms(values: torch.Tensor, reduction: str, name: str) -> torch.Tensor:
    reduction_norm = str(reduction).strip().lower()
    if reduction_norm == REDUCTION_SUM:
        return values.sum(dim=1)
    if reduction_norm == REDUCTION_MEAN:
        return values.mean(dim=1)
    raise ValueError(f"{name} must be one of ['sum', 'mean'], got: {reduction}")


def kl_divergence_diag_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor | None = None,
    prior_logvar: torch.Tensor | None = None,
    reduction: str = REDUCTION_SUM,
) -> torch.Tensor:
    if prior_mu is None or prior_logvar is None:
        terms = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    else:
        if prior_mu.shape != mu.shape or prior_logvar.shape != logvar.shape:
            raise ValueError(
                "prior_mu and prior_logvar must have the same shape as mu/logvar for conditional prior KL"
            )
        prior_var_inv = torch.exp(-prior_logvar)
        terms = 0.5 * (
            prior_logvar
            - logvar
            + torch.exp(logvar - prior_logvar)
            + (mu - prior_mu).pow(2) * prior_var_inv
            - 1.0
        )
    return _reduce_feature_terms(terms, reduction=reduction, name="kl_reduction")


def elbo_components(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor | None = None,
    prior_logvar: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    recon_logvar_x: torch.Tensor | None = None,
    reconstruction_loss: str = RECON_LOSS_MSE,
    recon_reduction: str = REDUCTION_SUM,
    kl_reduction: str = REDUCTION_SUM,
):
    reconstruction_loss_norm = str(reconstruction_loss).strip().lower()
    if reconstruction_loss_norm == RECON_LOSS_MSE:
        mse = F.mse_loss(recon_x, x, reduction="none")
        recon = _reduce_feature_terms(mse, reduction=recon_reduction, name="recon_reduction")
    elif reconstruction_loss_norm == RECON_LOSS_GAUSSIAN_NLL_DIAG:
        if recon_logvar_x is None:
            raise ValueError("recon_logvar_x is required for gaussian_nll_diag reconstruction")
        if str(recon_reduction).strip().lower() != REDUCTION_MEAN:
            raise ValueError("gaussian_nll_diag currently requires recon_reduction='mean'")
        recon = gaussian_nll_diag_terms(recon_x, x, recon_logvar_x)["recon_nll"]
    else:
        raise ValueError(
            "reconstruction_loss must be one of "
            f"{[RECON_LOSS_MSE, RECON_LOSS_GAUSSIAN_NLL_DIAG]}, got: {reconstruction_loss}"
        )
    kl = kl_divergence_diag_gaussian(
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        reduction=kl_reduction,
    )
    if float(kl_weight) != 1.0:
        kl = kl * float(kl_weight)
    return recon, kl


def elbo_loss_terms(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor | None = None,
    prior_logvar: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    recon_logvar_x: torch.Tensor | None = None,
    reconstruction_loss: str = RECON_LOSS_MSE,
    recon_reduction: str = REDUCTION_SUM,
    kl_reduction: str = REDUCTION_SUM,
) -> dict[str, torch.Tensor]:
    recon, kl = elbo_components(
        recon_x,
        x,
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        kl_weight=kl_weight,
        recon_logvar_x=recon_logvar_x,
        reconstruction_loss=reconstruction_loss,
        recon_reduction=recon_reduction,
        kl_reduction=kl_reduction,
    )
    terms = {
        "loss": recon + kl,
        "recon": recon,
        "kl": kl,
    }
    if str(reconstruction_loss).strip().lower() == RECON_LOSS_GAUSSIAN_NLL_DIAG:
        if recon_logvar_x is None:
            raise ValueError("recon_logvar_x is required for gaussian_nll_diag diagnostics")
        terms.update(gaussian_nll_diag_terms(recon_x, x, recon_logvar_x))
    return terms


def negative_elbo(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor | None = None,
    prior_logvar: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    recon_logvar_x: torch.Tensor | None = None,
    reconstruction_loss: str = RECON_LOSS_MSE,
    recon_reduction: str = REDUCTION_SUM,
    kl_reduction: str = REDUCTION_SUM,
) -> torch.Tensor:
    recon, kl = elbo_components(
        recon_x,
        x,
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        kl_weight=kl_weight,
        recon_logvar_x=recon_logvar_x,
        reconstruction_loss=reconstruction_loss,
        recon_reduction=recon_reduction,
        kl_reduction=kl_reduction,
    )
    return (recon + kl).mean()


def build_cvae_from_metadata(metadata: dict[str, object]) -> CVAEExpert:
    """Construct a CVAEExpert from checkpoint provenance metadata."""

    input_dim = int(metadata.get("input_dim", metadata.get("embedding_dim", 0)))
    hidden_dim = int(metadata.get("hidden_dim", 0))
    latent_dim = int(metadata.get("latent_dim", 0))
    if input_dim <= 0 or hidden_dim <= 0 or latent_dim <= 0:
        raise ValueError(
            "checkpoint metadata must define positive input_dim/embedding_dim, hidden_dim, and latent_dim"
        )
    return CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=int(metadata.get("metadata_dim", 0)),
        class_condition_dim=int(metadata.get("class_condition_dim", 0)),
        decoder_likelihood=str(metadata.get("decoder_likelihood", DECODER_LIKELIHOOD_MSE)),
        decoder_logvar_min=float(metadata.get("decoder_logvar_min", -9.21)),
        decoder_logvar_max=float(metadata.get("decoder_logvar_max", 2.0)),
        decoder_min_variance=float(metadata.get("decoder_min_variance", 1.0e-4)),
    )
