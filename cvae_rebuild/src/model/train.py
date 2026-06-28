from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from model.models import ClassConditionedCVAE, loss_for_batch
from evaluation.support_nelbo import CalibrationStats, calibration_stats


@dataclass
class TrainedExpert:
    expert_id: str
    model: ClassConditionedCVAE
    calibration: CalibrationStats
    n_train: int
    n_val: int


def train_class_conditioned_expert(
    *,
    expert_id: str,
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    val_embeddings: Sequence[Sequence[float]],
    hidden_dim: int = 512,
    latent_dim: int = 64,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    seed: int = 42,
) -> TrainedExpert:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("CVAE expert training requires numpy and torch.") from exc

    torch.manual_seed(int(seed))
    x_train = torch.as_tensor(np.asarray(train_embeddings, dtype=np.float32))
    y_train = torch.as_tensor(np.asarray(train_labels, dtype=np.int64))
    x_val = torch.as_tensor(np.asarray(val_embeddings, dtype=np.float32))
    model = ClassConditionedCVAE(
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        n_classes=2,
        num_hidden_layers=2,
    )
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(batch_size), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    for _epoch in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_for_batch(model, xb, yb).nelbo
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        val_nelbo = model.marginal_nelbo(x_val).detach().cpu().numpy().tolist()
    return TrainedExpert(
        expert_id=str(expert_id),
        model=model,
        calibration=calibration_stats(str(expert_id), val_nelbo),
        n_train=int(x_train.shape[0]),
        n_val=int(x_val.shape[0]),
    )
