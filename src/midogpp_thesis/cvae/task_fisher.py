"""Source-only expected input-Fisher reconstruction metric."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Sequence

from ..real_features.classifier_reference.artifacts import stable_hash
from ..real_features.classifier_reference.classifiers import ClassifierSpec, validate_classifier_spec
from .objectives import validate_trace_normalized_metric


@dataclass(frozen=True)
class TaskFisherMetric:
    metric: object
    raw_fisher: object
    valid: bool
    reason: str
    alpha: float
    trace_raw: float
    rank: int
    eigenvalues: tuple[float, ...]
    probe_config_hash: str
    probe_scale: tuple[float, ...]
    probe_weight_pca: tuple[float, ...]

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        def _tolist(value: object) -> object:
            return value.tolist() if hasattr(value, "tolist") else value

        return {
            "schema_version": "midogpp_task_fisher_state_v1",
            "metric": _tolist(self.metric),
            "raw_fisher": _tolist(self.raw_fisher),
            "valid": self.valid,
            "reason": self.reason,
            "alpha": self.alpha,
            "trace_raw": self.trace_raw,
            "rank": self.rank,
            "eigenvalues": list(self.eigenvalues),
            "probe_config_hash": self.probe_config_hash,
            "probe_scale": list(self.probe_scale),
            "probe_weight_pca": list(self.probe_weight_pca),
        }


def fit_task_fisher_metric(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    spec: ClassifierSpec,
    alpha: float = 1.0,
    minimum_trace: float = 1e-12,
) -> TaskFisherMetric:
    """Fit a binary logistic probe and map its Fisher direction to PCA coordinates."""

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    validate_classifier_spec(spec)
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("Task-Fisher embeddings/labels must be aligned nonempty arrays.")
    if sorted(set(int(value) for value in y.tolist())) != [0, 1]:
        raise ValueError("Task-Fisher probe requires both binary classes.")
    if alpha < 0.0:
        raise ValueError("Task-Fisher alpha must be nonnegative.")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    classifier = LogisticRegression(**spec.to_sklearn_kwargs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_scaled, y)
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence_warnings:
        return _invalid_metric(x.shape[1], spec=spec, alpha=alpha, reason="probe_nonconverged")
    if tuple(int(value) for value in classifier.classes_.tolist()) != (0, 1):
        return _invalid_metric(x.shape[1], spec=spec, alpha=alpha, reason="unexpected_probe_classes")
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        return _invalid_metric(x.shape[1], spec=spec, alpha=alpha, reason="invalid_probe_scale")
    weight_scaled = np.asarray(classifier.coef_[0], dtype=np.float64)
    weight_pca = weight_scaled / scale
    probabilities = np.asarray(classifier.predict_proba(x_scaled)[:, 1], dtype=np.float64)
    scalar = float(np.mean(probabilities * (1.0 - probabilities)))
    fisher = scalar * np.outer(weight_pca, weight_pca)
    trace_raw = float(np.trace(fisher))
    if not np.isfinite(fisher).all() or trace_raw <= float(minimum_trace):
        return _invalid_metric(
            x.shape[1],
            spec=spec,
            alpha=alpha,
            reason="fisher_trace_below_minimum",
            fisher=fisher,
            weight=weight_pca,
            scale=scale,
        )
    normalized = float(x.shape[1]) * fisher / trace_raw
    metric = (np.eye(x.shape[1], dtype=np.float64) + float(alpha) * normalized) / (1.0 + float(alpha))
    validate_trace_normalized_metric(metric, input_dim=x.shape[1])
    eigenvalues = np.linalg.eigvalsh(fisher)
    return TaskFisherMetric(
        metric=metric,
        raw_fisher=fisher,
        valid=True,
        reason="ok",
        alpha=float(alpha),
        trace_raw=trace_raw,
        rank=int(np.linalg.matrix_rank(fisher, tol=minimum_trace)),
        eigenvalues=tuple(float(value) for value in eigenvalues.tolist()),
        probe_config_hash=spec.config_hash,
        probe_scale=tuple(float(value) for value in scale.tolist()),
        probe_weight_pca=tuple(float(value) for value in weight_pca.tolist()),
    )


def _invalid_metric(
    input_dim: int,
    *,
    spec: ClassifierSpec,
    alpha: float,
    reason: str,
    fisher: object | None = None,
    weight: object | None = None,
    scale: object | None = None,
) -> TaskFisherMetric:
    import numpy as np

    raw = np.zeros((input_dim, input_dim), dtype=np.float64) if fisher is None else np.asarray(fisher, dtype=np.float64)
    probe_weight = np.zeros(input_dim, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    probe_scale = np.ones(input_dim, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)
    return TaskFisherMetric(
        metric=np.eye(input_dim, dtype=np.float64),
        raw_fisher=raw,
        valid=False,
        reason=reason,
        alpha=float(alpha),
        trace_raw=float(np.trace(raw)),
        rank=int(np.linalg.matrix_rank(raw)),
        eigenvalues=tuple(float(value) for value in np.linalg.eigvalsh(raw).tolist()),
        probe_config_hash=spec.config_hash,
        probe_scale=tuple(float(value) for value in probe_scale.tolist()),
        probe_weight_pca=tuple(float(value) for value in probe_weight.tolist()),
    )
