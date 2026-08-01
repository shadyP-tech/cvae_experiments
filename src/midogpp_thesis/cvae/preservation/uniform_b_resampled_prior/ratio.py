"""Source-only class-conditional posterior/base density-ratio estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import warnings

import numpy as np
import torch

from ....common.hashing import stable_hash
from ...keyed_training import derived_seed, torch_generator
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from .config import UniformBResampledPriorConfig


@dataclass(frozen=True)
class RatioClassState:
    class_label: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    crossfit_auc: float
    crossfit_log_loss: float
    baseline_log_loss: float
    log_loss_gain: float
    converged: bool
    reliable: bool
    n_source_rows: int
    n_source_cases: int

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_posterior_ratio_class_state_v1",
            "class_label": self.class_label,
            "feature_family": "z_plus_diagonal_z_squared",
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "crossfit_auc": self.crossfit_auc,
            "crossfit_log_loss": self.crossfit_log_loss,
            "baseline_log_loss": self.baseline_log_loss,
            "log_loss_gain": self.log_loss_gain,
            "converged": self.converged,
            "reliable": self.reliable,
            "n_source_rows": self.n_source_rows,
            "n_source_cases": self.n_source_cases,
            "positive_distribution": "source_aggregate_posterior",
            "negative_distribution": "standard_normal",
            "outer_or_inner_rows_used": False,
        }

    def acceptance(self, z: np.ndarray, *, config: UniformBResampledPriorConfig) -> np.ndarray:
        values = _ratio_features(np.asarray(z, dtype=np.float64))
        scaled = (values - self.scaler_mean) / self.scaler_scale
        logits = scaled @ self.coefficient + self.intercept
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(config.ratio_lambda * logits, -40.0, 40.0)))
        return config.acceptance_floor + (1.0 - config.acceptance_floor) * sigmoid


@dataclass(frozen=True)
class PosteriorRatioState:
    source_center: str
    training_seed: int
    checkpoint_hash: str
    source_row_hash: str
    source_case_hash: str
    classes: Mapping[int, RatioClassState]

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_posterior_ratio_state_v1",
            "source_center": self.source_center,
            "training_seed": self.training_seed,
            "checkpoint_hash": self.checkpoint_hash,
            "source_row_hash": self.source_row_hash,
            "source_case_hash": self.source_case_hash,
            "fit_scope": "source_center_cases_only",
            "outer_or_inner_rows_used": False,
            "target_labels_used": False,
            "classes": {
                str(label): state.to_payload()
                for label, state in sorted(self.classes.items())
            },
        }


def fit_posterior_ratio_state(
    model: ClassConditionedCVAE,
    projected: Sequence[Sequence[float]],
    labels: Sequence[int],
    case_ids: Sequence[str],
    *,
    source_center: str,
    training_seed: int,
    checkpoint_hash: str,
    source_row_hash: str,
    source_case_hash: str,
    config: UniformBResampledPriorConfig,
    device: str,
) -> PosteriorRatioState:
    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray([str(value) for value in case_ids], dtype=str)
    if x.ndim != 2 or x.shape[1] != 128 or len(x) != len(y) or len(x) != len(cases):
        raise ProtocolError("Posterior-ratio source arrays are invalid.")
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32, device=device)
        yb = torch.as_tensor(y, dtype=torch.long, device=device)
        mu, logvar = model.encode(xb, yb)
    classes: dict[int, RatioClassState] = {}
    for class_label in (0, 1):
        mask = y == class_label
        class_mu = mu[torch.as_tensor(mask, dtype=torch.bool, device=device)]
        class_logvar = logvar[torch.as_tensor(mask, dtype=torch.bool, device=device)]
        epsilon = torch.randn(
            class_mu.shape,
            generator=torch_generator(
                device,
                derived_seed(checkpoint_hash, class_label, "ratio_posterior_epsilon"),
            ),
            dtype=class_mu.dtype,
            device=device,
        )
        posterior = class_mu + epsilon * torch.exp(0.5 * class_logvar)
        base = torch.randn(
            class_mu.shape,
            generator=torch_generator(
                device,
                derived_seed(checkpoint_hash, class_label, "ratio_standard_normal"),
            ),
            dtype=class_mu.dtype,
            device=device,
        )
        classes[class_label] = _fit_class_state(
            posterior.detach().cpu().numpy(),
            base.detach().cpu().numpy(),
            cases[mask],
            class_label=class_label,
            seed=derived_seed(checkpoint_hash, class_label, "ratio_crossfit"),
            config=config,
        )
    return PosteriorRatioState(
        source_center=str(source_center),
        training_seed=int(training_seed),
        checkpoint_hash=str(checkpoint_hash),
        source_row_hash=str(source_row_hash),
        source_case_hash=str(source_case_hash),
        classes=classes,
    )


def _fit_class_state(
    posterior: np.ndarray,
    base: np.ndarray,
    source_cases: np.ndarray,
    *,
    class_label: int,
    seed: int,
    config: UniformBResampledPriorConfig,
) -> RatioClassState:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    n = len(posterior)
    if n != len(base) or n != len(source_cases) or len(set(source_cases.tolist())) < config.ratio_crossfit_folds:
        raise ProtocolError("Posterior-ratio class has insufficient aligned source cases.")
    values = np.concatenate([posterior, base], axis=0)
    targets = np.asarray([1] * n + [0] * n, dtype=np.int64)
    groups = np.concatenate([source_cases, source_cases], axis=0)
    features = _ratio_features(values)
    splitter = StratifiedGroupKFold(
        n_splits=config.ratio_crossfit_folds,
        shuffle=True,
        random_state=int(seed),
    )
    probabilities = np.zeros(len(targets), dtype=np.float64)
    converged = True
    for fit_idx, eval_idx in splitter.split(features, targets, groups):
        scaler = StandardScaler().fit(features[fit_idx])
        classifier = LogisticRegression(
            C=config.ratio_classifier_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=config.ratio_classifier_max_iter,
            random_state=int(seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(scaler.transform(features[fit_idx]), targets[fit_idx])
        converged &= not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        probabilities[eval_idx] = classifier.predict_proba(scaler.transform(features[eval_idx]))[:, 1]
    auc = float(roc_auc_score(targets, probabilities))
    observed_log_loss = float(log_loss(targets, probabilities, labels=[0, 1]))
    baseline = float(math.log(2.0))
    gain = baseline - observed_log_loss
    final_scaler = StandardScaler().fit(features)
    final_classifier = LogisticRegression(
        C=config.ratio_classifier_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=config.ratio_classifier_max_iter,
        random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_classifier.fit(final_scaler.transform(features), targets)
    converged &= not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    reliable = bool(
        converged
        and auc >= config.min_ratio_auc
        and gain >= config.min_log_loss_gain
    )
    scale = np.asarray(final_scaler.scale_, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    return RatioClassState(
        class_label=int(class_label),
        scaler_mean=np.asarray(final_scaler.mean_, dtype=np.float64),
        scaler_scale=scale,
        coefficient=np.asarray(final_classifier.coef_[0], dtype=np.float64),
        intercept=float(final_classifier.intercept_[0]),
        crossfit_auc=auc,
        crossfit_log_loss=observed_log_loss,
        baseline_log_loss=baseline,
        log_loss_gain=gain,
        converged=converged,
        reliable=reliable,
        n_source_rows=n,
        n_source_cases=len(set(source_cases.tolist())),
    )


def _ratio_features(z: np.ndarray) -> np.ndarray:
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ProtocolError("Ratio features require a finite matrix.")
    return np.concatenate([z, z * z], axis=1).astype(np.float64)


__all__ = (
    "PosteriorRatioState",
    "RatioClassState",
    "fit_posterior_ratio_state",
)
