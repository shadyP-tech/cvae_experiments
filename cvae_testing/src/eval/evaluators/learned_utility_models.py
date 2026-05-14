from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class _LinearRegressor:
    l2: float = 1e-4
    w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        xtx = x_aug.T @ x_aug
        xtx += float(self.l2) * np.eye(xtx.shape[0], dtype=np.float64)
        self.w = np.linalg.solve(xtx, x_aug.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Linear regressor is not fitted")
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        return x_aug @ self.w


@dataclass
class _MLPRegressor:
    seed: int
    hidden_dim: int = 128
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 2048
    device: str = "auto"
    model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(int(self.seed))
        if self.device == "auto":
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            run_device = torch.device(self.device)

        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], int(self.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim), 1),
        ).to(run_device)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr))
        loss_fn = torch.nn.MSELoss()

        x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(run_device)
        y_t = torch.from_numpy(y.astype(np.float32, copy=False)).view(-1, 1).to(run_device)

        n = int(x_t.shape[0])
        for _ in range(int(self.epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, int(self.batch_size)):
                idx = perm[i : i + int(self.batch_size)]
                pred = net(x_t[idx])
                loss = loss_fn(pred, y_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model = net.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLP regressor is not fitted")
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(model_device)
            pred = self.model(x_t).view(-1)
        return pred.detach().cpu().numpy().astype(np.float64, copy=False)


@dataclass
class _PairwiseRanker:
    seed: int
    hidden_dim: int = 128
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 2048
    margin: float = 1.0
    device: str = "auto"
    model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, pairs: List[Tuple[int, int]]) -> None:
        if not pairs:
            raise RuntimeError("Pairwise ranker received zero training pairs")

        torch.manual_seed(int(self.seed))
        if self.device == "auto":
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            run_device = torch.device(self.device)

        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], int(self.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim), 1),
        ).to(run_device)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr))

        x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(run_device)
        pair_t = torch.tensor(pairs, dtype=torch.long, device=run_device)

        n_pairs = int(pair_t.shape[0])
        for _ in range(int(self.epochs)):
            perm = torch.randperm(n_pairs, device=run_device)
            for i in range(0, n_pairs, int(self.batch_size)):
                idx = perm[i : i + int(self.batch_size)]
                pair_batch = pair_t[idx]

                better_x = x_t[pair_batch[:, 0]]
                worse_x = x_t[pair_batch[:, 1]]
                pred_better = net(better_x).view(-1)
                pred_worse = net(worse_x).view(-1)

                # Lower score means better expert; enforce pred_worse - pred_better >= margin.
                loss = torch.relu(float(self.margin) - (pred_worse - pred_better)).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self.model = net.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Pairwise ranker is not fitted")
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(model_device)
            pred = self.model(x_t).view(-1)
        return pred.detach().cpu().numpy().astype(np.float64, copy=False)


@dataclass
class _LogisticRidgePairprob:
    l2: float = 1e-3
    max_iter: int = 100
    device: str = "auto"
    w: np.ndarray | None = None
    b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        if x.ndim != 2:
            raise RuntimeError("Logistic pair-prob model requires a 2D feature matrix")
        if int(x.shape[0]) == 0:
            raise RuntimeError("Logistic pair-prob model received zero training rows")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if int(y_arr.shape[0]) != int(x.shape[0]):
            raise RuntimeError("Logistic pair-prob labels do not match feature rows")
        weights = (
            np.ones((x.shape[0],), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if int(weights.shape[0]) != int(x.shape[0]):
            raise RuntimeError("Logistic pair-prob weights do not match feature rows")

        if self.device == "auto":
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            run_device = torch.device(self.device)

        layer = torch.nn.Linear(int(x.shape[1]), 1).to(run_device)
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

        x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(run_device)
        y_t = torch.from_numpy(y_arr.astype(np.float32, copy=False)).to(run_device)
        w_t = torch.from_numpy(weights.astype(np.float32, copy=False)).to(run_device)
        denom = torch.clamp(w_t.sum(), min=1.0)

        opt = torch.optim.LBFGS(
            layer.parameters(),
            lr=1.0,
            max_iter=int(self.max_iter),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            logits = layer(x_t).view(-1)
            data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_t,
                weight=w_t,
                reduction="sum",
            ) / denom
            penalty = 0.5 * float(self.l2) * torch.sum(layer.weight.view(-1) ** 2)
            loss = data_loss + penalty
            loss.backward()
            return loss

        opt.step(closure)
        self.w = layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        self.b = float(layer.bias.detach().cpu().numpy().reshape(-1)[0])

    def predict_logit(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Logistic pair-prob model is not fitted")
        x_arr = np.asarray(x, dtype=np.float64)
        return (x_arr @ self.w + float(self.b)).astype(np.float64, copy=False)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logit(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64, copy=False)
