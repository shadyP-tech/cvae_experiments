"""Weighted Nyström objectives and GPU low-rank bilinear classifier."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..protocol import ProtocolError


def group_sample_weights(
    centers: np.ndarray, labels: np.ndarray, group_mass: dict[tuple[str, int], float]
) -> np.ndarray:
    groups = [(str(center), int(label)) for center, label in zip(centers, labels)]
    counts = Counter(groups)
    raw = np.asarray([group_mass[group] / counts[group] for group in groups])
    return raw * len(raw) / np.sum(raw)


def fit_weighted_logistic(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    *,
    objective: str,
    c_value: float,
    dro_iterations: int,
) -> LogisticRegression:
    groups = sorted({(str(center), int(label)) for center, label in zip(centers, y)})
    mass = {group: 1.0 / len(groups) for group in groups}
    eta = 0.0
    if objective.startswith("group_dro_eta_"):
        eta = float(objective.rsplit("_", 1)[1])
    elif objective != "equal_group":
        raise ProtocolError(f"Unknown robust objective: {objective}.")
    model = None
    iterations = dro_iterations if eta > 0.0 else 1
    cumulative = {group: 0.0 for group in groups}
    for _ in range(iterations):
        weights = group_sample_weights(centers, y, mass)
        model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000)
        model.fit(x, y, sample_weight=weights)
        if eta > 0.0:
            probability = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
            losses = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
            for group in groups:
                mask = (centers == group[0]) & (y == group[1])
                cumulative[group] += float(np.mean(losses[mask]))
            logits = np.asarray([eta * cumulative[group] for group in groups])
            logits -= np.max(logits)
            normalized = np.exp(logits)
            normalized /= np.sum(normalized)
            mass = dict(zip(groups, normalized.tolist()))
    if model is None:
        raise ProtocolError("Robust logistic fitting produced no model.")
    return model


@dataclass
class BilinearFit:
    model: object
    mean: np.ndarray
    scale: np.ndarray
    device: str
    final_loss: float

    def predict_proba(self, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        import torch

        normalized = ((x - self.mean) / self.scale).astype(np.float32)
        output = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                batch = torch.from_numpy(normalized[start : start + batch_size]).to(
                    self.device
                )
                output.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(output)


def fit_bilinear(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    *,
    global_dim: int,
    local_dim: int,
    rank: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device_index: int,
) -> BilinearFit:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise ProtocolError("The frozen bilinear protocol requires CUDA.")
    device = f"cuda:{device_index}"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((x - mean) / scale).astype(np.float32)
    groups = sorted({(str(center), int(label)) for center, label in zip(centers, y)})
    mass = {group: 1.0 / len(groups) for group in groups}
    weights = group_sample_weights(centers, y, mass).astype(np.float32)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(global_dim + local_dim, 1)
            self.u = nn.Linear(global_dim, rank, bias=False)
            self.v = nn.Linear(local_dim, rank, bias=False)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            nn.init.normal_(self.u.weight, std=0.01 / math.sqrt(global_dim))
            nn.init.normal_(self.v.weight, std=0.01 / math.sqrt(local_dim))

        def forward(self, values: object) -> object:
            global_values = values[:, :global_dim]
            local_values = values[:, global_dim:]
            interaction = torch.sum(self.u(global_values) * self.v(local_values), dim=1)
            return self.linear(values).squeeze(1) + interaction

    model = Model().to(device)
    linear_init = LogisticRegression(C=0.01, solver="lbfgs", max_iter=5000)
    linear_init.fit(normalized, y, sample_weight=weights)
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.from_numpy(linear_init.coef_.astype(np.float32)).to(device)
        )
        model.linear.bias.copy_(
            torch.from_numpy(linear_init.intercept_.astype(np.float32)).to(device)
        )
    model.linear.weight.requires_grad_(False)
    model.linear.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(normalized), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = torch.from_numpy(normalized[indices]).to(device)
            batch_y = torch.from_numpy(y[indices].astype(np.float32)).to(device)
            batch_w = torch.from_numpy(weights[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch_y, reduction="none"
            )
            loss = torch.mean(losses * batch_w)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return BilinearFit(model=model, mean=mean, scale=scale, device=device, final_loss=final_loss)
