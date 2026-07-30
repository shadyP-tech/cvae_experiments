"""Cross-fitted source-only task geometry for prior-sample regularization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence
import warnings

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..independent_source import deterministic_case_folds
from .config import UniformBTaskGeometryConfig


@dataclass(frozen=True)
class FoldTaskGeometry:
    fold: int
    fit_row_hash: str
    reference_row_hash: str
    fit_case_hash: str
    reference_case_hash: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    nystrom_components: np.ndarray
    nystrom_normalization: np.ndarray
    nystrom_gamma: float
    teacher_coef: np.ndarray
    teacher_intercept: float
    hessian_inverse_sqrt: np.ndarray
    hessian_eigenvalues: np.ndarray
    reference_projected: np.ndarray
    reference_labels: np.ndarray
    reference_gradient: np.ndarray
    cdf_grids: np.ndarray
    cdf_targets: np.ndarray
    mmd_bandwidths: np.ndarray
    reference_per_class: int

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_fold_task_geometry_v1",
            "fold": self.fold,
            "fit_row_hash": self.fit_row_hash,
            "reference_row_hash": self.reference_row_hash,
            "fit_case_hash": self.fit_case_hash,
            "reference_case_hash": self.reference_case_hash,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "nystrom_components": self.nystrom_components.tolist(),
            "nystrom_normalization": self.nystrom_normalization.tolist(),
            "nystrom_gamma": self.nystrom_gamma,
            "teacher_coef": self.teacher_coef.tolist(),
            "teacher_intercept": self.teacher_intercept,
            "hessian_inverse_sqrt": self.hessian_inverse_sqrt.tolist(),
            "hessian_eigenvalues": self.hessian_eigenvalues.tolist(),
            "reference_projected": self.reference_projected.tolist(),
            "reference_labels": self.reference_labels.tolist(),
            "reference_gradient": self.reference_gradient.tolist(),
            "cdf_grids": self.cdf_grids.tolist(),
            "cdf_targets": self.cdf_targets.tolist(),
            "mmd_bandwidths": self.mmd_bandwidths.tolist(),
            "reference_per_class": self.reference_per_class,
            "teacher_fit_uses_reference_cases": False,
            "outer_or_inner_rows_used": False,
            "intercept_in_gradient_and_hessian": True,
        }


@dataclass(frozen=True)
class TaskGeometryState:
    source_center: str
    source_row_hash: str
    frame_hash: str
    folds: tuple[FoldTaskGeometry, ...]
    config_identity: dict[str, object]

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_task_geometry_state_v1",
            "source_center": self.source_center,
            "source_row_hash": self.source_row_hash,
            "frame_hash": self.frame_hash,
            "fit_scope": "source_center_case_crossfit_only",
            "outer_or_inner_rows_used": False,
            "config_identity": dict(self.config_identity),
            "folds": [fold.to_payload() for fold in self.folds],
        }


def fit_task_geometry(
    projected: Sequence[Sequence[float]],
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    source_center: str,
    source_row_hash: str,
    frame_hash: str,
    config: UniformBTaskGeometryConfig,
    seed: int,
) -> TaskGeometryState:
    """Fit frozen fold teachers; every reference case is out-of-fold."""

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(projected, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray([str(value) for value in case_ids], dtype=str)
    samples = np.asarray([str(value) for value in sample_ids], dtype=str)
    if (
        x.ndim != 2
        or x.shape[1] != 128
        or len(x) != len(y)
        or len(x) != len(cases)
        or len(x) != len(samples)
        or set(int(value) for value in y.tolist()) != {0, 1}
    ):
        raise ProtocolError("Task-geometry source arrays are invalid.")
    folds = deterministic_case_folds(
        y,
        cases,
        n_splits=config.crossfit_folds,
        seed=int(seed),
    )
    states: list[FoldTaskGeometry] = []
    for fold in folds:
        fit_idx = np.asarray(fold.fit_indices, dtype=np.int64)
        ref_idx = np.asarray(fold.reference_indices, dtype=np.int64)
        x_fit = x[fit_idx]
        y_fit = y[fit_idx]
        if len(x_fit) < config.nystrom_components:
            raise ProtocolError(
                "Source fold has fewer rows than locked Nyström components."
            )
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(x_fit)
        nystrom = Nystroem(
            kernel="rbf",
            gamma=config.nystrom_gamma,
            n_components=config.nystrom_components,
            random_state=int(seed) + fold.fold,
        )
        phi_fit = np.asarray(nystrom.fit_transform(fit_scaled), dtype=np.float64)
        teacher = LogisticRegression(
            C=config.teacher_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=config.teacher_max_iter,
            class_weight="balanced",
            random_state=int(seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            teacher.fit(phi_fit, y_fit)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ProtocolError("Source-only task teacher did not converge.")
        if tuple(int(value) for value in teacher.classes_.tolist()) != (0, 1):
            raise ProtocolError("Task teacher has unexpected class order.")

        selected = _balanced_reference_indices(
            ref_idx,
            y,
            samples,
            requested=config.reference_per_class,
            seed=int(seed) + fold.fold,
        )
        x_ref = x[selected]
        y_ref = y[selected]
        ref_scaled_selected = scaler.transform(x_ref)
        phi_ref = np.asarray(
            nystrom.transform(ref_scaled_selected),
            dtype=np.float64,
        )
        coef = np.asarray(teacher.coef_[0], dtype=np.float64)
        intercept = float(teacher.intercept_[0])
        augmented_fit = _augment_intercept(phi_fit)
        probabilities_fit = _sigmoid(augmented_fit @ np.r_[coef, intercept])
        fit_weights = _balanced_sample_weights(y_fit)
        curvature = fit_weights * probabilities_fit * (1.0 - probabilities_fit)
        hessian = (
            augmented_fit.T
            @ (augmented_fit * curvature[:, None])
            / float(fit_weights.sum())
        )
        hessian = 0.5 * (hessian + hessian.T)
        hessian += config.hessian_ridge * np.eye(hessian.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floored = np.maximum(eigenvalues, config.hessian_eigenfloor)
        inverse_sqrt = (
            eigenvectors
            @ np.diag(floored ** -0.5)
            @ eigenvectors.T
        )
        if not np.isfinite(inverse_sqrt).all():
            raise ProtocolError("Task Hessian inverse square root is nonfinite.")
        augmented_ref = _augment_intercept(phi_ref)
        ref_gradient = _balanced_logistic_gradient(
            augmented_ref,
            y_ref,
            np.r_[coef, intercept],
        )
        margins = phi_ref @ coef + intercept
        grids = []
        targets = []
        for cls in (0, 1):
            class_margins = margins[y_ref == cls]
            grid = np.quantile(class_margins, config.cdf_grid_quantiles)
            target = np.mean(
                _sigmoid(
                    (grid[:, None] - class_margins[None, :])
                    / config.cdf_temperature
                ),
                axis=1,
            )
            grids.append(grid)
            targets.append(target)
        median_distance = _median_pairwise_distance(x_ref)
        bandwidths = np.asarray(
            [
                median_distance * multiplier
                for multiplier in config.mmd_bandwidth_multipliers
            ],
            dtype=np.float64,
        )
        if np.any(bandwidths <= 0.0) or not np.isfinite(bandwidths).all():
            raise ProtocolError("Source-only MMD bandwidths are invalid.")
        state = FoldTaskGeometry(
            fold=fold.fold,
            fit_row_hash=_row_hash(samples[fit_idx]),
            reference_row_hash=_row_hash(samples[selected]),
            fit_case_hash=_row_hash(np.asarray(fold.fit_cases)),
            reference_case_hash=_row_hash(np.asarray(fold.reference_cases)),
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            nystrom_components=np.asarray(
                nystrom.components_,
                dtype=np.float64,
            ),
            nystrom_normalization=np.asarray(
                nystrom.normalization_,
                dtype=np.float64,
            ),
            nystrom_gamma=float(config.nystrom_gamma),
            teacher_coef=coef,
            teacher_intercept=intercept,
            hessian_inverse_sqrt=np.asarray(inverse_sqrt, dtype=np.float64),
            hessian_eigenvalues=np.asarray(floored, dtype=np.float64),
            reference_projected=np.asarray(x_ref, dtype=np.float64),
            reference_labels=np.asarray(y_ref, dtype=np.int64),
            reference_gradient=np.asarray(ref_gradient, dtype=np.float64),
            cdf_grids=np.asarray(grids, dtype=np.float64),
            cdf_targets=np.asarray(targets, dtype=np.float64),
            mmd_bandwidths=bandwidths,
            reference_per_class=int((y_ref == 0).sum()),
        )
        if set(fold.fit_cases).intersection(fold.reference_cases):
            raise ProtocolError("Task-geometry teacher/reference case leakage.")
        states.append(state)
    return TaskGeometryState(
        source_center=str(source_center),
        source_row_hash=str(source_row_hash),
        frame_hash=str(frame_hash),
        folds=tuple(states),
        config_identity={
            "crossfit_folds": config.crossfit_folds,
            "nystrom_components": config.nystrom_components,
            "nystrom_gamma": config.nystrom_gamma,
            "teacher_c": config.teacher_c,
            "hessian_ridge": config.hessian_ridge,
            "hessian_eigenfloor": config.hessian_eigenfloor,
            "reference_per_class_requested": config.reference_per_class,
            "mmd_bandwidth_multipliers": list(
                config.mmd_bandwidth_multipliers
            ),
            "cdf_grid_quantiles": list(config.cdf_grid_quantiles),
            "cdf_temperature": config.cdf_temperature,
        },
    )


def _balanced_reference_indices(
    fold_indices: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    requested: int,
    seed: int,
) -> np.ndarray:
    available = {
        cls: fold_indices[labels[fold_indices] == cls] for cls in (0, 1)
    }
    budget = min(requested, len(available[0]), len(available[1]))
    if budget < 4:
        raise ProtocolError("Cross-fit reference bag is too small per class.")
    selected: list[int] = []
    for cls in (0, 1):
        ordered = sorted(
            (int(index) for index in available[cls].tolist()),
            key=lambda index: hashlib.sha256(
                f"{seed}|{sample_ids[index]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:budget])
    return np.asarray(selected, dtype=np.int64)


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    for cls in (0, 1):
        mask = labels == cls
        weights[mask] = 0.5 / float(mask.sum())
    return weights


def _balanced_logistic_gradient(
    augmented: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    probabilities = _sigmoid(augmented @ parameters)
    weights = _balanced_sample_weights(labels)
    return augmented.T @ (weights * (probabilities - labels))


def _augment_intercept(features: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _median_pairwise_distance(values: np.ndarray) -> float:
    subset = values[: min(512, len(values))]
    differences = subset[:, None, :] - subset[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper = distances[np.triu_indices(len(subset), k=1)]
    positive = upper[upper > 0.0]
    if len(positive) == 0:
        raise ProtocolError("Reference geometry has zero pairwise distance.")
    return float(np.median(positive))


def _row_hash(values: np.ndarray) -> str:
    return stable_hash([str(value) for value in values.tolist()])


__all__ = (
    "FoldTaskGeometry",
    "TaskGeometryState",
    "fit_task_geometry",
)
