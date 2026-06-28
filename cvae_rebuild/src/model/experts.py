from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from config import RebuildConfig
from feature_frame import ExpertFeatureFrame, fit_expert_frame
from features import FeatureCache, select_rows
from splits import SourceTrainValSplit, stratified_source_train_val_split
from train import train_class_conditioned_expert


@dataclass
class ExpertRuntime:
    expert_id: str
    frame: ExpertFeatureFrame
    model: object
    calibration: object
    source_train_embeddings: object
    source_train_labels: tuple[int, ...]
    source_train_sample_ids: tuple[str, ...]
    source_val_split: SourceTrainValSplit
    n_train: int
    n_val: int
    effective_dim: int


def train_seed_experts(
    cfg: RebuildConfig,
    *,
    train_cache: FeatureCache,
    experiment_seed: int,
) -> dict[str, ExpertRuntime]:
    import numpy as np  # type: ignore

    experts: dict[str, ExpertRuntime] = {}
    for expert_id in cfg.heldout_centers:
        split = stratified_source_train_val_split(
            train_cache.metadata,
            center=str(expert_id),
            experiment_seed=int(experiment_seed),
        )
        source_train_raw, source_train_meta = select_rows(train_cache.embeddings, train_cache.metadata, split.train_indices)
        source_val_raw, _source_val_meta = select_rows(train_cache.embeddings, train_cache.metadata, split.val_indices)
        source_train_labels = tuple(label(row) for row in source_train_meta)
        if set(source_train_labels) != {0, 1}:
            raise RuntimeError(f"Expert {expert_id} train split must contain classes 0 and 1.")
        frame = fit_expert_frame(
            expert_id=str(expert_id),
            source_train_embeddings=to_numpy(source_train_raw),
            requested_dim=cfg.pca_dim,
        )
        source_train_x = frame.transform(to_numpy(source_train_raw))
        source_val_x = frame.transform(to_numpy(source_val_raw))
        trained = train_class_conditioned_expert(
            expert_id=str(expert_id),
            train_embeddings=source_train_x,
            train_labels=source_train_labels,
            val_embeddings=source_val_x,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            epochs=cfg.train_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.learning_rate,
            seed=int(experiment_seed),
        )
        experts[str(expert_id)] = ExpertRuntime(
            expert_id=str(expert_id),
            frame=frame,
            model=trained.model,
            calibration=trained.calibration,
            source_train_embeddings=np.asarray(source_train_x, dtype=float),
            source_train_labels=source_train_labels,
            source_train_sample_ids=tuple(sample_id(row, idx) for idx, row in enumerate(source_train_meta)),
            source_val_split=split,
            n_train=trained.n_train,
            n_val=trained.n_val,
            effective_dim=frame.effective_dim,
        )
    return experts


def source_refs_by_class(expert: ExpertRuntime) -> dict[int, object]:
    import numpy as np  # type: ignore

    x = np.asarray(expert.source_train_embeddings, dtype=float)
    y = np.asarray(expert.source_train_labels, dtype=int)
    return {class_label: x[y == class_label] for class_label in (0, 1)}


def to_numpy(value: object) -> object:
    import numpy as np  # type: ignore

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def label(row: Mapping[str, object]) -> int:
    return int(float(str(row.get("label", 0))))


def sample_id(row: Mapping[str, object], fallback_idx: int) -> str:
    value = row.get("sample_id", "")
    return str(value) if str(value) else f"row_{int(fallback_idx)}"
