from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ExpertFeatureFrame:
    expert_id: str
    scaler: object
    pca: object
    requested_dim: int
    effective_dim: int
    fit_scope: str = "per_expert_source_train"

    def transform(self, embeddings: Sequence[Sequence[float]]) -> object:
        x = self.scaler.transform(embeddings)
        return self.pca.transform(x)


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
        scaler=scaler,
        pca=pca,
        requested_dim=int(requested_dim),
        effective_dim=int(effective),
    )
