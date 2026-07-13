"""Standard and source-fitted aggregate-posterior latent samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..real_features.classifier_reference.artifacts import stable_hash


STANDARD_SAMPLER = "standard_normal"
DIAGONAL_SAMPLER = "class_conditional_diagonal_total_moment"
FULL_SAMPLER = "class_conditional_shrinkage_full_total_moment"
VALID_SAMPLERS = frozenset({STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER})


@dataclass(frozen=True)
class ClassSamplerState:
    class_label: int
    requested_family: str
    realized_family: str
    mean: object
    covariance: object
    cholesky: object
    n_rows: int
    raw_between_covariance: object
    within_posterior_diagonal: object
    shrinkage: float | None
    shrinkage_target: float | None
    jitter: float
    condition_number: float
    eigenvalues: tuple[float, ...]
    fallback_reason: str

    def to_payload(self) -> dict[str, object]:
        def _tolist(value: object) -> object:
            return value.tolist() if hasattr(value, "tolist") else value

        return {
            "class_label": self.class_label,
            "requested_family": self.requested_family,
            "realized_family": self.realized_family,
            "mean": _tolist(self.mean),
            "covariance": _tolist(self.covariance),
            "n_rows": self.n_rows,
            "raw_between_covariance": _tolist(self.raw_between_covariance),
            "within_posterior_diagonal": _tolist(self.within_posterior_diagonal),
            "shrinkage": self.shrinkage,
            "shrinkage_target": self.shrinkage_target,
            "jitter": self.jitter,
            "condition_number": self.condition_number,
            "eigenvalues": list(self.eigenvalues),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class AggregatePosteriorSampler:
    requested_family: str
    classes: Mapping[int, ClassSamplerState]
    latent_dim: int
    source_row_hash: str

    @property
    def state_hash(self) -> str:
        return stable_hash(
            {
                "requested_family": self.requested_family,
                "latent_dim": self.latent_dim,
                "source_row_hash": self.source_row_hash,
                "classes": {str(key): value.to_payload() for key, value in sorted(self.classes.items())},
            }
        )

    def realized_family_by_class(self) -> dict[str, str]:
        return {str(key): value.realized_family for key, value in sorted(self.classes.items())}

    def fallback_reason_by_class(self) -> dict[str, str]:
        return {str(key): value.fallback_reason for key, value in sorted(self.classes.items())}

    @property
    def requested_family_realized_for_both_classes(self) -> bool:
        return all(state.realized_family == self.requested_family for state in self.classes.values())


def fit_aggregate_posterior_sampler(
    mu: Sequence[Sequence[float]],
    logvar: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    family: str,
    source_row_hash: str,
    min_class_count: int = 64,
    max_condition_number: float = 1e6,
) -> AggregatePosteriorSampler:
    """Fit per-class source-only sampler statistics with fail-closed fallbacks."""

    import numpy as np

    if family not in VALID_SAMPLERS:
        raise ValueError(f"Unsupported sampler family: {family!r}")
    mu_np = np.asarray(mu, dtype=np.float64)
    logvar_np = np.asarray(logvar, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    if mu_np.ndim != 2 or logvar_np.shape != mu_np.shape or len(labels_np) != len(mu_np):
        raise ValueError("mu, logvar, and labels must be aligned arrays.")
    if not np.isfinite(mu_np).all() or not np.isfinite(logvar_np).all():
        raise ValueError("Posterior parameters contain nonfinite values.")
    states: dict[int, ClassSamplerState] = {}
    for class_label in (0, 1):
        indices = np.flatnonzero(labels_np == class_label)
        states[class_label] = _fit_class_state(
            mu_np[indices],
            logvar_np[indices],
            class_label=class_label,
            requested_family=family,
            min_class_count=int(min_class_count),
            max_condition_number=float(max_condition_number),
        )
    return AggregatePosteriorSampler(
        requested_family=family,
        classes=states,
        latent_dim=int(mu_np.shape[1]),
        source_row_hash=str(source_row_hash),
    )


def standard_normal_sampler(*, latent_dim: int, source_row_hash: str = "not_applicable") -> AggregatePosteriorSampler:
    import numpy as np

    states = {}
    for class_label in (0, 1):
        identity = np.eye(int(latent_dim), dtype=np.float64)
        states[class_label] = ClassSamplerState(
            class_label=class_label,
            requested_family=STANDARD_SAMPLER,
            realized_family=STANDARD_SAMPLER,
            mean=np.zeros(int(latent_dim), dtype=np.float64),
            covariance=identity,
            cholesky=identity,
            n_rows=0,
            raw_between_covariance=np.zeros_like(identity),
            within_posterior_diagonal=np.ones(int(latent_dim), dtype=np.float64),
            shrinkage=None,
            shrinkage_target=None,
            jitter=0.0,
            condition_number=1.0,
            eigenvalues=tuple(1.0 for _ in range(int(latent_dim))),
            fallback_reason="",
        )
    return AggregatePosteriorSampler(
        requested_family=STANDARD_SAMPLER,
        classes=states,
        latent_dim=int(latent_dim),
        source_row_hash=str(source_row_hash),
    )


def sample_latents(
    sampler: AggregatePosteriorSampler,
    labels: Sequence[int],
    *,
    seed: int,
) -> object:
    import numpy as np

    labels_np = np.asarray(labels, dtype=np.int64)
    if not set(int(value) for value in labels_np.tolist()).issubset({0, 1}):
        raise ValueError("Latent sampler supports binary labels 0/1 only.")
    rng = np.random.default_rng(int(seed))
    result = np.empty((len(labels_np), sampler.latent_dim), dtype=np.float32)
    for class_label in (0, 1):
        idx = np.flatnonzero(labels_np == class_label)
        state = sampler.classes[class_label]
        noise = rng.normal(size=(len(idx), sampler.latent_dim))
        result[idx] = (state.mean + noise @ state.cholesky.T).astype(np.float32)
    return result


def aggregate_posterior_moments(mu: object, logvar: object) -> tuple[object, object, object, object]:
    """Return mean, total covariance, between covariance, and mean posterior variance."""

    import numpy as np

    mu_np = np.asarray(mu, dtype=np.float64)
    logvar_np = np.asarray(logvar, dtype=np.float64)
    if mu_np.ndim != 2 or logvar_np.shape != mu_np.shape or len(mu_np) == 0:
        raise ValueError("Class posterior arrays must be aligned and nonempty.")
    mean = mu_np.mean(axis=0)
    centered = mu_np - mean
    between = centered.T @ centered / float(len(mu_np))
    within = np.exp(logvar_np).mean(axis=0)
    total = between + np.diag(within)
    return mean, total, between, within


def _fit_class_state(
    mu: object,
    logvar: object,
    *,
    class_label: int,
    requested_family: str,
    min_class_count: int,
    max_condition_number: float,
) -> ClassSamplerState:
    import numpy as np
    from sklearn.covariance import LedoitWolf

    mu_np = np.asarray(mu, dtype=np.float64)
    logvar_np = np.asarray(logvar, dtype=np.float64)
    latent_dim = int(mu_np.shape[1]) if mu_np.ndim == 2 else int(logvar_np.shape[1])
    if len(mu_np) == 0:
        return _standard_fallback(class_label, requested_family, latent_dim, 0, "empty_class")
    mean, total, between, within = aggregate_posterior_moments(mu_np, logvar_np)
    jitter = 1e-6 * max(float(np.trace(total)) / float(latent_dim), 1.0)
    if requested_family == STANDARD_SAMPLER:
        return _standard_fallback(class_label, requested_family, latent_dim, len(mu_np), "")
    if len(mu_np) < min_class_count:
        return _standard_fallback(class_label, requested_family, latent_dim, len(mu_np), "class_count_below_minimum")
    if requested_family == FULL_SAMPLER:
        shrinkage = LedoitWolf(store_precision=False, assume_centered=False).fit(mu_np)
        full_covariance = np.asarray(shrinkage.covariance_, dtype=np.float64) + np.diag(within) + jitter * np.eye(latent_dim)
        state = _validated_state(
            class_label=class_label,
            requested_family=requested_family,
            realized_family=FULL_SAMPLER,
            mean=mean,
            covariance=full_covariance,
            n_rows=len(mu_np),
            between=between,
            within=within,
            shrinkage=float(shrinkage.shrinkage_),
            shrinkage_target=float(np.trace(between) / latent_dim),
            jitter=jitter,
            max_condition_number=max_condition_number,
        )
        if state is not None:
            return state
        fallback_reason = "full_covariance_invalid"
    else:
        fallback_reason = ""
    diagonal_covariance = np.diag(np.diag(total)) + jitter * np.eye(latent_dim)
    diagonal = _validated_state(
        class_label=class_label,
        requested_family=requested_family,
        realized_family=DIAGONAL_SAMPLER,
        mean=mean,
        covariance=diagonal_covariance,
        n_rows=len(mu_np),
        between=between,
        within=within,
        shrinkage=None,
        shrinkage_target=None,
        jitter=jitter,
        max_condition_number=max_condition_number,
        fallback_reason=fallback_reason,
    )
    if diagonal is not None:
        return diagonal
    return _standard_fallback(
        class_label,
        requested_family,
        latent_dim,
        len(mu_np),
        fallback_reason or "diagonal_covariance_invalid",
        between=between,
        within=within,
    )


def _validated_state(
    *,
    class_label: int,
    requested_family: str,
    realized_family: str,
    mean: object,
    covariance: object,
    n_rows: int,
    between: object,
    within: object,
    shrinkage: float | None,
    shrinkage_target: float | None,
    jitter: float,
    max_condition_number: float,
    fallback_reason: str = "",
) -> ClassSamplerState | None:
    import numpy as np

    covariance_np = np.asarray(covariance, dtype=np.float64)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance_np)
        condition_number = float(np.linalg.cond(covariance_np))
        cholesky = np.linalg.cholesky(covariance_np)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(covariance_np).all() or not np.isfinite(condition_number):
        return None
    if float(eigenvalues.min()) <= 0.0 or condition_number > max_condition_number:
        return None
    return ClassSamplerState(
        class_label=class_label,
        requested_family=requested_family,
        realized_family=realized_family,
        mean=np.asarray(mean, dtype=np.float64),
        covariance=covariance_np,
        cholesky=cholesky,
        n_rows=int(n_rows),
        raw_between_covariance=np.asarray(between, dtype=np.float64),
        within_posterior_diagonal=np.asarray(within, dtype=np.float64),
        shrinkage=shrinkage,
        shrinkage_target=shrinkage_target,
        jitter=float(jitter),
        condition_number=condition_number,
        eigenvalues=tuple(float(value) for value in eigenvalues.tolist()),
        fallback_reason=fallback_reason,
    )


def _standard_fallback(
    class_label: int,
    requested_family: str,
    latent_dim: int,
    n_rows: int,
    reason: str,
    *,
    between: object | None = None,
    within: object | None = None,
) -> ClassSamplerState:
    import numpy as np

    identity = np.eye(int(latent_dim), dtype=np.float64)
    return ClassSamplerState(
        class_label=int(class_label),
        requested_family=requested_family,
        realized_family=STANDARD_SAMPLER,
        mean=np.zeros(int(latent_dim), dtype=np.float64),
        covariance=identity,
        cholesky=identity,
        n_rows=int(n_rows),
        raw_between_covariance=(
            np.zeros_like(identity) if between is None else np.asarray(between, dtype=np.float64)
        ),
        within_posterior_diagonal=(
            np.ones(int(latent_dim), dtype=np.float64) if within is None else np.asarray(within, dtype=np.float64)
        ),
        shrinkage=None,
        shrinkage_target=None,
        jitter=0.0,
        condition_number=1.0,
        eigenvalues=tuple(1.0 for _ in range(int(latent_dim))),
        fallback_reason=reason,
    )
