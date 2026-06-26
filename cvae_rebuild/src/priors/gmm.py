from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ClassConditionalPrior:
    classes: tuple[int, ...]
    latent_dim: int
    models: Mapping[int, object]
    prior_type: str

    def sample(
        self,
        *,
        labels: Sequence[int],
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        chunks: list[np.ndarray] = []
        for label in labels:
            cls = int(label)
            if cls not in self.models:
                raise ValueError(f"No latent prior fitted for class {cls}.")
            model = self.models[cls]
            if isinstance(model, GaussianMixture):
                chunks.append(_sample_gmm_row(model, rng=rng))
            else:
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                gaussian = _GaussianParameters.from_object(model)
                chunks.append(gaussian.sample(rng=np.random.default_rng(seed)))
        return np.vstack(chunks).astype(np.float32)


@dataclass(frozen=True)
class _GaussianParameters:
    mean: np.ndarray
    variance: np.ndarray

    @classmethod
    def from_object(cls, value: object) -> "_GaussianParameters":
        if not isinstance(value, _GaussianParameters):
            raise TypeError("Expected Gaussian prior parameters.")
        return value

    def sample(self, *, rng: np.random.Generator) -> np.ndarray:
        std = np.sqrt(np.maximum(self.variance, 1.0e-12))
        return rng.normal(loc=self.mean, scale=std).astype(np.float32)


def fit_class_conditional_gmm_prior(
    latents: object,
    labels: Sequence[int],
    *,
    n_components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1.0e-6,
    random_state: int | None = None,
    n_init: int = 1,
    max_iter: int = 200,
    min_class_count: int = 2,
) -> ClassConditionalPrior:
    z = _as_2d_latents(latents)
    y = np.asarray(labels, dtype=np.int64)
    if z.shape[0] != y.shape[0]:
        raise ValueError("Latent row count must match label count.")
    if covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise ValueError("Unsupported GMM covariance_type.")
    models: dict[int, object] = {}
    for cls in tuple(sorted(int(v) for v in np.unique(y))):
        cls_z = z[y == cls]
        if cls_z.shape[0] < int(min_class_count):
            raise ValueError(f"Class {cls} has too few fit rows for prior fitting.")
        effective_components = min(int(n_components), int(cls_z.shape[0]))
        gmm = GaussianMixture(
            n_components=effective_components,
            covariance_type=covariance_type,
            reg_covar=float(reg_covar),
            random_state=random_state,
            n_init=int(n_init),
            max_iter=int(max_iter),
        )
        gmm.fit(cls_z)
        models[cls] = gmm
    return ClassConditionalPrior(
        classes=tuple(sorted(models)),
        latent_dim=int(z.shape[1]),
        models=models,
        prior_type="class_conditional_gmm",
    )


def fit_class_conditional_gaussian_prior(
    latents: object,
    labels: Sequence[int],
    *,
    min_class_count: int = 2,
) -> ClassConditionalPrior:
    z = _as_2d_latents(latents)
    y = np.asarray(labels, dtype=np.int64)
    if z.shape[0] != y.shape[0]:
        raise ValueError("Latent row count must match label count.")
    models: dict[int, object] = {}
    for cls in tuple(sorted(int(v) for v in np.unique(y))):
        cls_z = z[y == cls]
        if cls_z.shape[0] < int(min_class_count):
            raise ValueError(f"Class {cls} has too few fit rows for prior fitting.")
        models[cls] = _GaussianParameters(
            mean=np.mean(cls_z, axis=0).astype(np.float32),
            variance=np.var(cls_z, axis=0).astype(np.float32),
        )
    return ClassConditionalPrior(
        classes=tuple(sorted(models)),
        latent_dim=int(z.shape[1]),
        models=models,
        prior_type="class_conditional_gaussian",
    )


def _as_2d_latents(latents: object) -> np.ndarray:
    z = np.asarray(latents, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("Latents must be a non-empty 2D array.")
    return z


def _sample_gmm_row(model: GaussianMixture, *, rng: np.random.Generator) -> np.ndarray:
    weights = np.maximum(np.asarray(model.weights_, dtype=np.float64), 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("GMM weights must sum to a positive finite value.")
    component = int(rng.choice(model.n_components, p=weights / total))
    mean = np.asarray(model.means_[component], dtype=np.float64)
    cov = _component_covariance(model, component)
    return rng.multivariate_normal(mean, cov).astype(np.float32)


def _component_covariance(model: GaussianMixture, component: int) -> np.ndarray:
    covariances = np.asarray(model.covariances_, dtype=np.float64)
    if model.covariance_type == "full":
        cov = covariances[component]
    elif model.covariance_type == "tied":
        cov = covariances
    elif model.covariance_type == "diag":
        cov = np.diag(np.maximum(covariances[component], 1.0e-12))
    elif model.covariance_type == "spherical":
        cov = np.eye(model.means_.shape[1], dtype=np.float64) * max(float(covariances[component]), 1.0e-12)
    else:  # pragma: no cover - validated during fitting.
        raise ValueError("Unsupported GMM covariance_type.")
    return np.asarray(cov, dtype=np.float64)
