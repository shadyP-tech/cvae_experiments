"""Class-conditional low-rank Gaussian-mixture latent priors.

The implementation is deliberately separate from the locked v1/v2 prior
families.  It provides the explicit density, differentiable rate bound, and
deterministic sampling contract required by the Stage-20 aggregate-posterior
mixture study.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASSES = 2
DEFAULT_COMPONENTS = 2
DEFAULT_RANK = 2
DEFAULT_WEIGHT_FLOOR = 0.05
DEFAULT_VARIANCE_FLOOR = 1e-4


@dataclass(frozen=True)
class MixtureInitialization:
    """Auditable result of source-only aggregate-posterior initialization."""

    assignments: tuple[int, ...]
    component_row_counts: tuple[tuple[int, ...], ...]
    component_case_counts: tuple[tuple[int, ...], ...]
    assignment_fallbacks: tuple[bool, ...]
    covariance_fallbacks: tuple[tuple[bool, ...], ...]
    shrinkage: float
    minimum_component_rows: int
    minimum_component_cases: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_aggregate_posterior_mixture_initialization_v1",
            "assignments": list(self.assignments),
            "component_row_counts": [
                list(values) for values in self.component_row_counts
            ],
            "component_case_counts": [
                list(values) for values in self.component_case_counts
            ],
            "assignment_fallbacks": list(self.assignment_fallbacks),
            "covariance_fallbacks": [
                list(values) for values in self.covariance_fallbacks
            ],
            "shrinkage": self.shrinkage,
            "minimum_component_rows": self.minimum_component_rows,
            "minimum_component_cases": self.minimum_component_cases,
        }


@dataclass(frozen=True)
class MixturePriorDiagnostics:
    """Numerical and component-health diagnostics for one prior state."""

    finite: bool
    minimum_weight: float
    maximum_condition_number: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    weight_floor_respected: bool
    covariance_positive_definite: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_low_rank_mixture_prior_diagnostics_v1",
            "finite": self.finite,
            "minimum_weight": self.minimum_weight,
            "maximum_condition_number": self.maximum_condition_number,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "weight_floor_respected": self.weight_floor_respected,
            "covariance_positive_definite": self.covariance_positive_definite,
        }


class ClassConditionalLowRankMixturePrior(nn.Module):
    """Binary class-conditional ``K``-component ``diag + UU^T`` prior.

    Component KL values are computed in full latent units.  Only after the
    mixture variational bound is formed is the result normalized by latent
    dimension.  This ordering is part of the scientific contract.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        n_components: int = DEFAULT_COMPONENTS,
        rank: int = DEFAULT_RANK,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    ) -> None:
        super().__init__()
        latent_dim = int(latent_dim)
        n_components = int(n_components)
        rank = int(rank)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if n_components < 2:
            raise ValueError("A mixture prior requires at least two components.")
        if rank <= 0 or rank > latent_dim:
            raise ValueError("rank must lie in [1, latent_dim].")
        if not 0.0 <= float(weight_floor) < 1.0 / n_components:
            raise ValueError("weight_floor must lie in [0, 1 / n_components).")
        if not float(variance_floor) > 0.0:
            raise ValueError("variance_floor must be positive.")

        self.latent_dim = latent_dim
        self.n_classes = N_CLASSES
        self.n_components = n_components
        self.rank = rank
        self.weight_floor = float(weight_floor)
        self.variance_floor = float(variance_floor)

        self.mixture_logits = nn.Parameter(
            torch.zeros(N_CLASSES, n_components, dtype=torch.float32)
        )
        self.component_means = nn.Parameter(
            torch.zeros(
                N_CLASSES,
                n_components,
                latent_dim,
                dtype=torch.float32,
            )
        )
        unit_rho = _inverse_softplus(
            torch.tensor(1.0 - self.variance_floor, dtype=torch.float32)
        )
        self.diag_rho = nn.Parameter(
            unit_rho.expand(
                N_CLASSES,
                n_components,
                latent_dim,
            ).clone()
        )
        self.low_rank = nn.Parameter(
            torch.zeros(
                N_CLASSES,
                n_components,
                latent_dim,
                rank,
                dtype=torch.float32,
            )
        )

    def weights(self) -> torch.Tensor:
        """Return component weights with an exact positive floor."""

        free_mass = 1.0 - self.n_components * self.weight_floor
        return self.weight_floor + free_mass * torch.softmax(
            self.mixture_logits,
            dim=-1,
        )

    def diagonal_variance(self) -> torch.Tensor:
        return self.variance_floor + F.softplus(self.diag_rho)

    def covariance(self) -> torch.Tensor:
        diagonal = torch.diag_embed(self.diagonal_variance())
        return diagonal + self.low_rank @ self.low_rank.transpose(-1, -2)

    def component_kl(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return summed ``KL(q || p_k)`` for every row and component."""

        posterior_mu, posterior_logvar, y = self._validated_posterior(
            posterior_mu,
            posterior_logvar,
            labels,
        )
        means, diagonal, factors = self._parameters_for_labels(y)
        q_variance = posterior_logvar.exp().unsqueeze(1)
        difference = posterior_mu.unsqueeze(1) - means
        (
            log_determinant,
            trace_term,
            quadratic_term,
        ) = _low_rank_gaussian_terms(
            difference=difference,
            q_variance=q_variance,
            diagonal=diagonal,
            factors=factors,
        )
        q_log_determinant = posterior_logvar.sum(dim=-1, keepdim=True)
        kl = 0.5 * (
            log_determinant
            - q_log_determinant
            - float(self.latent_dim)
            + trace_term
            + quadratic_term
        )
        return kl.clamp_min(0.0)

    def kl_upper_bound(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return the latent-normalized variational upper bound to the mixture.

        ``KL(q || sum_k pi_k p_k)`` is bounded above by

        ``-log sum_k pi_k exp(-KL(q || p_k))``.
        """

        y = _validated_labels(labels, device=posterior_mu.device)
        component_kl = self.component_kl(posterior_mu, posterior_logvar, y)
        log_weights = self.weights()[y].to(
            dtype=component_kl.dtype,
            device=component_kl.device,
        ).log()
        bound = -torch.logsumexp(log_weights - component_kl, dim=-1)
        return bound.clamp_min(0.0) / float(self.latent_dim)

    def log_prob(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return exact class-conditional mixture log density."""

        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape [rows,{self.latent_dim}]."
            )
        y = _validated_labels(
            labels,
            n_rows=int(latent.shape[0]),
            device=latent.device,
        )
        means, diagonal, factors = self._parameters_for_labels(y)
        difference = latent.unsqueeze(1) - means
        dummy_q_variance = torch.zeros_like(difference)
        log_determinant, _, quadratic = _low_rank_gaussian_terms(
            difference=difference,
            q_variance=dummy_q_variance,
            diagonal=diagonal,
            factors=factors,
        )
        component_log_prob = -0.5 * (
            float(self.latent_dim) * math.log(2.0 * math.pi)
            + log_determinant
            + quadratic
        )
        log_weights = self.weights()[y].to(
            dtype=component_log_prob.dtype,
            device=component_log_prob.device,
        ).log()
        return torch.logsumexp(log_weights + component_log_prob, dim=-1)

    def sample(
        self,
        labels: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
        component_uniform: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample with explicit Gaussian noise and categorical uniforms.

        Explicit inputs form the reproducibility contract used to pair arms.
        """

        y = _validated_labels(labels, device=self.component_means.device)
        n_rows = int(y.numel())
        if (epsilon is not None or component_uniform is not None) and generator is not None:
            raise ValueError("Provide explicit sampling tensors or generator, not both.")
        dtype = self.component_means.dtype
        device = self.component_means.device
        if epsilon is None:
            epsilon = torch.randn(
                (n_rows, self.latent_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        else:
            epsilon = torch.as_tensor(epsilon, dtype=dtype, device=device)
        if component_uniform is None:
            component_uniform = torch.rand(
                (n_rows,),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        else:
            component_uniform = torch.as_tensor(
                component_uniform,
                dtype=dtype,
                device=device,
            ).reshape(-1)
        if epsilon.shape != (n_rows, self.latent_dim):
            raise ValueError("epsilon has the wrong shape.")
        if component_uniform.shape != (n_rows,):
            raise ValueError("component_uniform has the wrong shape.")
        if n_rows and (
            bool((component_uniform < 0.0).any())
            or bool((component_uniform >= 1.0).any())
        ):
            raise ValueError("component_uniform values must lie in [0,1).")

        weights = self.weights()[y]
        components = (
            component_uniform.unsqueeze(-1) >= weights.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(self.n_components - 1)
        means = self.component_means[y, components]
        # Factor the small class-by-component covariance table once, then
        # gather. Repeating a 32x32 Cholesky decomposition for every generated
        # row is unnecessarily expensive in the full source-inner grid.
        cholesky = torch.linalg.cholesky(self.covariance())[y, components]
        return means + torch.matmul(cholesky, epsilon.unsqueeze(-1)).squeeze(-1)

    def initialize_from_aggregate_posterior(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
        *,
        case_ids: Sequence[str] | None = None,
        random_state: int,
        shrinkage: float = 0.10,
        minimum_component_rows: int = 8,
        minimum_component_cases: int = 2,
    ) -> MixtureInitialization:
        """Initialize from source-only posterior sufficient statistics.

        K-means is applied only to posterior means.  Each component covariance
        then adds the average posterior variance, preserving the total
        aggregate-posterior moment rather than fitting means alone.
        """

        if not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0,1].")
        posterior_mu, posterior_logvar, y = self._validated_posterior(
            posterior_mu,
            posterior_logvar,
            labels,
        )
        n_rows = int(posterior_mu.shape[0])
        cases = (
            tuple(str(value) for value in case_ids)
            if case_ids is not None
            else tuple(f"row-{index}" for index in range(n_rows))
        )
        if len(cases) != n_rows:
            raise ValueError("case_ids are not aligned with posterior rows.")

        from sklearn.cluster import KMeans

        mu_np = posterior_mu.detach().cpu().double().numpy()
        logvar_np = posterior_logvar.detach().cpu().double().numpy()
        labels_np = y.detach().cpu().numpy()
        assignments = np.full(n_rows, -1, dtype=np.int64)
        weights = np.empty((N_CLASSES, self.n_components), dtype=np.float64)
        means = np.empty(
            (N_CLASSES, self.n_components, self.latent_dim),
            dtype=np.float64,
        )
        diagonals = np.empty_like(means)
        factors = np.zeros(
            (
                N_CLASSES,
                self.n_components,
                self.latent_dim,
                self.rank,
            ),
            dtype=np.float64,
        )
        row_counts: list[tuple[int, ...]] = []
        case_counts: list[tuple[int, ...]] = []
        fallbacks: list[tuple[bool, ...]] = []
        assignment_fallbacks: list[bool] = []

        for class_label in range(N_CLASSES):
            class_indices = np.flatnonzero(labels_np == class_label)
            minimum_rows_for_weight = (
                int(math.floor(self.weight_floor * len(class_indices))) + 1
            )
            required_component_rows = max(
                int(minimum_component_rows),
                minimum_rows_for_weight,
            )
            if len(class_indices) < self.n_components * required_component_rows:
                raise ValueError(
                    f"Class {class_label} has too few rows for the locked mixture."
                )
            fitted = KMeans(
                n_clusters=self.n_components,
                init="k-means++",
                n_init=10,
                random_state=int(random_state) + class_label,
                algorithm="lloyd",
            ).fit(mu_np[class_indices])
            local_assignments = np.asarray(fitted.labels_, dtype=np.int64)
            assignment_fallback = False
            if not _assignment_is_viable(
                local_assignments,
                class_indices=class_indices,
                cases=cases,
                n_components=self.n_components,
                minimum_rows=required_component_rows,
                minimum_cases=int(minimum_component_cases),
            ):
                if self.n_components != 2:
                    raise ValueError(
                        "Constrained assignment repair is defined only for K=2."
                    )
                local_assignments = _balanced_case_projection_split(
                    mu_np[class_indices],
                    case_ids=tuple(cases[index] for index in class_indices),
                    minimum_rows=required_component_rows,
                    minimum_cases=int(minimum_component_cases),
                )
                assignment_fallback = True
            local_centers = np.stack(
                [
                    mu_np[class_indices][local_assignments == component].mean(
                        axis=0
                    )
                    for component in range(self.n_components)
                ]
            )
            order = sorted(
                range(self.n_components),
                key=lambda component: tuple(
                    np.round(local_centers[component], decimals=12)
                ),
            )
            remap = {old: new for new, old in enumerate(order)}
            local_assignments = np.asarray(
                [remap[int(value)] for value in local_assignments],
                dtype=np.int64,
            )
            assignments[class_indices] = local_assignments

            class_rows: list[int] = []
            class_cases: list[int] = []
            class_fallbacks: list[bool] = []
            for component in range(self.n_components):
                local_mask = local_assignments == component
                selected_indices = class_indices[local_mask]
                selected_mu = mu_np[selected_indices]
                selected_var = np.exp(logvar_np[selected_indices])
                component_rows = int(len(selected_indices))
                component_cases = len({cases[index] for index in selected_indices})
                if component_rows < required_component_rows:
                    raise ValueError(
                        f"Class {class_label} component {component} has "
                        f"{component_rows} rows; minimum is "
                        f"{required_component_rows} after the weight-floor guard."
                    )
                if component_cases < int(minimum_component_cases):
                    raise ValueError(
                        f"Class {class_label} component {component} has "
                        f"{component_cases} cases; minimum is {minimum_component_cases}."
                    )
                mean = selected_mu.mean(axis=0)
                centered = selected_mu - mean
                between = centered.T @ centered / float(component_rows)
                total = between + np.diag(selected_var.mean(axis=0))
                target = np.diag(np.diag(total))
                covariance = (
                    (1.0 - float(shrinkage)) * total
                    + float(shrinkage) * target
                )
                covariance = 0.5 * (covariance + covariance.T)
                diagonal, factor, fallback = _factorize_covariance(
                    covariance,
                    rank=self.rank,
                    variance_floor=self.variance_floor,
                )
                weights[class_label, component] = (
                    component_rows / float(len(class_indices))
                )
                means[class_label, component] = mean
                diagonals[class_label, component] = diagonal
                factors[class_label, component] = factor
                class_rows.append(component_rows)
                class_cases.append(component_cases)
                class_fallbacks.append(fallback)
            row_counts.append(tuple(class_rows))
            case_counts.append(tuple(class_cases))
            fallbacks.append(tuple(class_fallbacks))
            assignment_fallbacks.append(assignment_fallback)

        adjusted_weights = (
            weights - self.weight_floor
        ) / (1.0 - self.n_components * self.weight_floor)
        if np.any(adjusted_weights <= 0.0):
            raise ValueError("Empirical component weights violate the locked floor.")
        adjusted_weights /= adjusted_weights.sum(axis=1, keepdims=True)
        diagonal_free = np.maximum(
            diagonals - self.variance_floor,
            np.finfo(np.float32).eps,
        )
        with torch.no_grad():
            self.mixture_logits.copy_(
                torch.as_tensor(
                    np.log(adjusted_weights),
                    dtype=self.mixture_logits.dtype,
                    device=self.mixture_logits.device,
                )
            )
            self.component_means.copy_(
                torch.as_tensor(
                    means,
                    dtype=self.component_means.dtype,
                    device=self.component_means.device,
                )
            )
            self.diag_rho.copy_(
                _inverse_softplus(
                    torch.as_tensor(
                        diagonal_free,
                        dtype=self.diag_rho.dtype,
                        device=self.diag_rho.device,
                    )
                )
            )
            self.low_rank.copy_(
                torch.as_tensor(
                    factors,
                    dtype=self.low_rank.dtype,
                    device=self.low_rank.device,
                )
            )
        return MixtureInitialization(
            assignments=tuple(int(value) for value in assignments.tolist()),
            component_row_counts=tuple(row_counts),
            component_case_counts=tuple(case_counts),
            assignment_fallbacks=tuple(assignment_fallbacks),
            covariance_fallbacks=tuple(fallbacks),
            shrinkage=float(shrinkage),
            minimum_component_rows=int(minimum_component_rows),
            minimum_component_cases=int(minimum_component_cases),
        )

    def state_diagnostics(self) -> MixturePriorDiagnostics:
        with torch.no_grad():
            weights = self.weights()
            covariance = self.covariance()
            eigenvalues = torch.linalg.eigvalsh(covariance)
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    weights,
                    self.component_means,
                    self.diagonal_variance(),
                    self.low_rank,
                    covariance,
                    eigenvalues,
                )
            )
            minimum_eigenvalue = float(eigenvalues.min().cpu())
            maximum_eigenvalue = float(eigenvalues.max().cpu())
            condition = eigenvalues[..., -1] / eigenvalues[..., 0]
            minimum_weight = float(weights.min().cpu())
            return MixturePriorDiagnostics(
                finite=finite,
                minimum_weight=minimum_weight,
                maximum_condition_number=float(condition.max().cpu()),
                minimum_eigenvalue=minimum_eigenvalue,
                maximum_eigenvalue=maximum_eigenvalue,
                weight_floor_respected=(
                    minimum_weight + 1e-7 >= self.weight_floor
                ),
                covariance_positive_definite=minimum_eigenvalue > 0.0,
            )

    def assert_healthy(self, *, maximum_condition_number: float) -> None:
        """Fail closed when a fitted prior is unsafe for training/generation."""

        if not math.isfinite(float(maximum_condition_number)) or float(
            maximum_condition_number
        ) <= 1.0:
            raise ValueError("maximum_condition_number must be finite and > 1.")
        diagnostics = self.state_diagnostics()
        if (
            not diagnostics.finite
            or not diagnostics.weight_floor_respected
            or not diagnostics.covariance_positive_definite
            or diagnostics.maximum_condition_number
            > float(maximum_condition_number)
        ):
            raise FloatingPointError(
                "Mixture prior failed its finite/weight/SPD/condition gate."
            )

    def state_payload(self) -> Mapping[str, object]:
        return {
            "schema_version": "midogpp_class_conditional_low_rank_mixture_prior_v1",
            "n_classes": self.n_classes,
            "n_components": self.n_components,
            "latent_dim": self.latent_dim,
            "rank": self.rank,
            "weight_parameterization": (
                "floor+(1-K*floor)*softmax(mixture_logits)"
            ),
            "variance_parameterization": "floor+softplus(diag_rho)",
            "weight_floor": self.weight_floor,
            "variance_floor": self.variance_floor,
            "mixture_logits": self.mixture_logits.detach().cpu().tolist(),
            "weights": self.weights().detach().cpu().tolist(),
            "component_means": self.component_means.detach().cpu().tolist(),
            "diag_rho": self.diag_rho.detach().cpu().tolist(),
            "diagonal_variance": self.diagonal_variance().detach().cpu().tolist(),
            "low_rank": self.low_rank.detach().cpu().tolist(),
        }

    def _validated_posterior(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            posterior_mu.ndim != 2
            or posterior_mu.shape[1] != self.latent_dim
            or posterior_logvar.shape != posterior_mu.shape
        ):
            raise ValueError(
                f"Posterior tensors must have shape [rows,{self.latent_dim}]."
            )
        if not bool(torch.isfinite(posterior_mu).all()) or not bool(
            torch.isfinite(posterior_logvar).all()
        ):
            raise FloatingPointError("Posterior parameters are nonfinite.")
        y = _validated_labels(
            labels,
            n_rows=int(posterior_mu.shape[0]),
            device=posterior_mu.device,
        )
        return posterior_mu, posterior_logvar, y

    def _parameters_for_labels(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.component_means[labels],
            self.diagonal_variance()[labels],
            self.low_rank[labels],
        )


def _low_rank_gaussian_terms(
    *,
    difference: torch.Tensor,
    q_variance: torch.Tensor,
    diagonal: torch.Tensor,
    factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logdet, trace, and quadratic terms via Woodbury identities."""

    inverse_diagonal = diagonal.reciprocal()
    scaled_factors = factors * inverse_diagonal.unsqueeze(-1)
    inner = torch.einsum("bkdr,bkds->bkrs", factors, scaled_factors)
    identity = torch.eye(
        factors.shape[-1],
        dtype=factors.dtype,
        device=factors.device,
    )
    inner = inner + identity
    cholesky = torch.linalg.cholesky(inner)
    log_determinant = diagonal.log().sum(dim=-1) + 2.0 * (
        torch.diagonal(cholesky, dim1=-2, dim2=-1).log().sum(dim=-1)
    )

    base_trace = (q_variance * inverse_diagonal).sum(dim=-1)
    trace_matrix = torch.einsum(
        "bkdr,bkd,bkds->bkrs",
        factors,
        q_variance * inverse_diagonal.square(),
        factors,
    )
    trace_correction = torch.diagonal(
        torch.cholesky_solve(trace_matrix, cholesky),
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    trace_term = base_trace - trace_correction

    base_quadratic = (difference.square() * inverse_diagonal).sum(dim=-1)
    projected = torch.einsum(
        "bkdr,bkd->bkr",
        factors,
        difference * inverse_diagonal,
    )
    solved = torch.cholesky_solve(
        projected.unsqueeze(-1),
        cholesky,
    ).squeeze(-1)
    quadratic = base_quadratic - (projected * solved).sum(dim=-1)
    return log_determinant, trace_term, quadratic


def _factorize_covariance(
    covariance: np.ndarray,
    *,
    rank: int,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Approximate one covariance by a positive ``diag + UU^T`` factor."""

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        diagonal = np.maximum(np.diag(covariance), variance_floor)
        return diagonal, np.zeros((len(diagonal), rank)), True
    order = np.argsort(eigenvalues)[::-1][:rank]
    selected = np.maximum(eigenvalues[order], 0.0)
    factor = eigenvectors[:, order] * np.sqrt(selected)[None, :]
    residual = np.diag(covariance - factor @ factor.T)
    fallback = bool(np.any(residual < variance_floor))
    diagonal = np.maximum(residual, variance_floor)
    reconstructed = np.diag(diagonal) + factor @ factor.T
    if not np.all(np.linalg.eigvalsh(reconstructed) > 0.0):
        diagonal = np.maximum(np.diag(covariance), variance_floor)
        factor = np.zeros((len(diagonal), rank), dtype=np.float64)
        fallback = True
    return diagonal, factor, fallback


def _assignment_is_viable(
    assignments: np.ndarray,
    *,
    class_indices: np.ndarray,
    cases: Sequence[str],
    n_components: int,
    minimum_rows: int,
    minimum_cases: int,
) -> bool:
    for component in range(n_components):
        selected = class_indices[assignments == component]
        if len(selected) < minimum_rows:
            return False
        if len({cases[index] for index in selected}) < minimum_cases:
            return False
    return True


def _balanced_case_projection_split(
    values: np.ndarray,
    *,
    case_ids: Sequence[str],
    minimum_rows: int,
    minimum_cases: int,
) -> np.ndarray:
    """Deterministically repair a collapsed K=2 assignment at case level."""

    cases = tuple(str(value) for value in case_ids)
    unique_cases = sorted(set(cases))
    if len(unique_cases) < 2 * minimum_cases:
        raise ValueError("Too few distinct cases for constrained K=2 repair.")
    case_means = np.stack(
        [values[np.asarray([case == value for value in cases])].mean(axis=0) for case in unique_cases]
    )
    centered = case_means - case_means.mean(axis=0)
    if np.allclose(centered, 0.0):
        projections = np.zeros(len(unique_cases), dtype=np.float64)
    else:
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        direction = right[0]
        nonzero = np.flatnonzero(np.abs(direction) > 1e-12)
        if len(nonzero) and direction[nonzero[0]] < 0.0:
            direction = -direction
        projections = centered @ direction
    ordered_cases = sorted(
        range(len(unique_cases)),
        key=lambda index: (float(projections[index]), unique_cases[index]),
    )
    candidates: list[tuple[int, int, set[str]]] = []
    for cut in range(minimum_cases, len(unique_cases) - minimum_cases + 1):
        left = {unique_cases[index] for index in ordered_cases[:cut]}
        left_rows = sum(case in left for case in cases)
        right_rows = len(cases) - left_rows
        if left_rows >= minimum_rows and right_rows >= minimum_rows:
            candidates.append((abs(left_rows - right_rows), cut, left))
    if not candidates:
        raise ValueError(
            "No deterministic case-level split satisfies row/case/weight floors."
        )
    _, _, selected_left = min(candidates, key=lambda value: (value[0], value[1]))
    return np.asarray(
        [0 if case in selected_left else 1 for case in cases],
        dtype=np.int64,
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    if bool((value <= 0.0).any()):
        raise ValueError("inverse softplus requires positive values.")
    return value + torch.log(-torch.expm1(-value))


def _validated_labels(
    labels: torch.Tensor,
    *,
    n_rows: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if torch.is_floating_point(labels) or labels.dtype == torch.bool:
        raise ValueError("Class labels must use an integer dtype.")
    y = labels.to(device=device, dtype=torch.long).reshape(-1)
    if n_rows is not None and y.numel() != int(n_rows):
        raise ValueError("Class labels are not aligned with rows.")
    if y.numel() and (int(y.min()) < 0 or int(y.max()) >= N_CLASSES):
        raise ValueError("Class labels must be binary values 0/1.")
    return y
