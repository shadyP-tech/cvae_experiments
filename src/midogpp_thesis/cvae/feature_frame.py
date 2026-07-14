from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..real_features.classifier_reference.artifacts import stable_hash


@dataclass
class ExpertFeatureFrame:
    expert_id: str
    scaler_mean: object
    scaler_scale: object
    pca_components: object
    pca_mean: object
    pca_explained_variance: object
    requested_dim: int
    effective_dim: int
    explained_variance_ratio_sum: float
    fit_scope: str = "per_expert_source_train"

    def transform(self, embeddings: Sequence[Sequence[float]]) -> object:
        import numpy as np

        x = np.asarray(embeddings, dtype=float)
        scaled = (x - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        return (scaled - np.asarray(self.pca_mean)) @ np.asarray(self.pca_components).T

    def inverse_transform(self, projected: Sequence[Sequence[float]]) -> object:
        """Map PCA coordinates back to the original embedding frame."""

        import numpy as np

        projected_array = np.asarray(projected, dtype=float)
        scaled = projected_array @ np.asarray(self.pca_components) + np.asarray(self.pca_mean)
        return scaled * np.asarray(self.scaler_scale) + np.asarray(self.scaler_mean)

    def state_payload(self) -> Mapping[str, object]:
        """Return the fitted frame state used for cache and artifact identity."""

        def _tolist(value: object) -> object:
            return value.tolist() if hasattr(value, "tolist") else value

        return {
            "expert_id": self.expert_id,
            "requested_dim": int(self.requested_dim),
            "effective_dim": int(self.effective_dim),
            "explained_variance_ratio_sum": float(self.explained_variance_ratio_sum),
            "fit_scope": self.fit_scope,
            "scaler_mean": _tolist(self.scaler_mean),
            "scaler_scale": _tolist(self.scaler_scale),
            "pca_components": _tolist(self.pca_components),
            "pca_mean": _tolist(self.pca_mean),
            "pca_explained_variance": _tolist(self.pca_explained_variance),
        }

    @property
    def state_hash(self) -> str:
        return stable_hash(self.state_payload())


def fit_expert_frame(
    *,
    expert_id: str,
    source_train_embeddings: Sequence[Sequence[float]],
    requested_dim: int = 256,
) -> ExpertFeatureFrame:
    try:
        import numpy as np  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Feature-frame fitting requires numpy and scikit-learn.") from exc

    x = np.asarray(source_train_embeddings, dtype=float)
    if x.ndim != 2:
        raise ValueError("source_train_embeddings must be a 2D array.")
    effective = min(int(requested_dim), int(x.shape[0]), int(x.shape[1]))
    if effective <= 0:
        raise ValueError("Cannot fit PCA on empty embeddings.")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=effective, svd_solver="auto", random_state=0)
    pca.fit(x_scaled)
    return ExpertFeatureFrame(
        expert_id=str(expert_id),
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
        pca_components=np.asarray(pca.components_, dtype=float),
        pca_mean=np.asarray(pca.mean_, dtype=float),
        pca_explained_variance=np.asarray(pca.explained_variance_, dtype=float),
        requested_dim=int(requested_dim),
        effective_dim=int(effective),
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
    )
