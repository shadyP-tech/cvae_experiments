from __future__ import annotations

from pathlib import Path
import importlib
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cvae_expert import negative_elbo
from src.torch_utils import safe_torch_load
from src.train.checkpoint_utils import load_resume_state, save_resume_state, training_state_path
from src.train.hybrid.checkpointing import save_hybrid_checkpoint
from src.train.hybrid.checkpointing import build_hybrid_checkpoint_payload
from src.train.hybrid.variants import (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_POOLED,
    HybridModuleBundle,
    build_hybrid_modules,
)

try:
    tqdm = getattr(importlib.import_module("tqdm"), "tqdm")
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


class HybridAblationTrainer:
    def __init__(
        self,
        train_payload: Dict[str, object],
        val_payload: Dict[str, object],
        domains: List[int],
        projection_dim: int,
        head_hidden_dim: int,
        cvae_hidden_dim: int,
        latent_dim: int,
        lr: float,
        epochs: int,
        patience: int,
        batch_size: int,
        seed: int,
        variant: str,
    ) -> None:
        self.train_x = train_payload["embeddings"]
        self.val_x = val_payload["embeddings"]
        self.train_meta = train_payload["metadata"]
        self.val_meta = val_payload["metadata"]

        self.domains = [int(d) for d in domains]
        self.input_dim = int(self.train_x.shape[1])
        self.projection_dim = int(projection_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.cvae_hidden_dim = int(cvae_hidden_dim)
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.variant = str(variant)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.bundle: HybridModuleBundle = build_hybrid_modules(
            variant=self.variant,
            device=self.device,
            input_dim=self.input_dim,
            projection_dim=self.projection_dim,
            head_hidden_dim=self.head_hidden_dim,
            cvae_hidden_dim=self.cvae_hidden_dim,
            latent_dim=self.latent_dim,
            domains=self.domains,
        )

    def _domains_tensor(self, metadata: List[dict]) -> torch.Tensor:
        return torch.tensor([int(m["magnification"]) for m in metadata], dtype=torch.long)

    def _make_loader(self, x: torch.Tensor, metadata: List[dict], shuffle: bool) -> DataLoader:
        d = self._domains_tensor(metadata)
        ds = TensorDataset(x, d)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        if self.bundle.shared_head is not None:
            params.extend(list(self.bundle.shared_head.parameters()))
        if self.bundle.shared_cvae is not None:
            params.extend(list(self.bundle.shared_cvae.parameters()))
        for m in self.bundle.heads.values():
            params.extend(list(m.parameters()))
        for m in self.bundle.cvaes.values():
            params.extend(list(m.parameters()))
        return params

    def _forward_variant(self, xb: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        total_count = 0

        for d in self.domains:
            mask = db == d
            if not torch.any(mask):
                continue
            x_d = xb[mask]

            if self.variant == VARIANT_A:
                assert self.bundle.shared_head is not None
                proj = self.bundle.shared_head(x_d)
                recon, mu, logvar = self.bundle.cvaes[d](proj)
            elif self.variant == VARIANT_POOLED:
                assert self.bundle.shared_head is not None
                assert self.bundle.shared_cvae is not None
                proj = self.bundle.shared_head(x_d)
                recon, mu, logvar = self.bundle.shared_cvae(proj)
            elif self.variant == VARIANT_B:
                assert self.bundle.shared_cvae is not None
                proj = self.bundle.heads[d](x_d)
                recon, mu, logvar = self.bundle.shared_cvae(proj)
            elif self.variant == VARIANT_C:
                proj = self.bundle.heads[d](x_d)
                recon, mu, logvar = self.bundle.cvaes[d](proj)
            else:
                raise ValueError(f"Unsupported hybrid variant: {self.variant}")

            loss = negative_elbo(recon, proj, mu, logvar)
            count = int(x_d.shape[0])
            total_loss = total_loss + loss * count
            total_count += count

        if total_count == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / total_count

    def _set_train(self, train_mode: bool) -> None:
        modules = []
        if self.bundle.shared_head is not None:
            modules.append(self.bundle.shared_head)
        if self.bundle.shared_cvae is not None:
            modules.append(self.bundle.shared_cvae)
        modules.extend(self.bundle.heads.values())
        modules.extend(self.bundle.cvaes.values())
        for m in modules:
            m.train(mode=train_mode)

    def save_checkpoint(self, path: Path) -> None:
        save_hybrid_checkpoint(
            path=path,
            variant=self.variant,
            domains=self.domains,
            input_dim=self.input_dim,
            projection_dim=self.projection_dim,
            head_hidden_dim=self.head_hidden_dim,
            cvae_hidden_dim=self.cvae_hidden_dim,
            latent_dim=self.latent_dim,
            bundle=self.bundle,
        )

    def _load_checkpoint_payload(self, payload: Dict[str, object]) -> None:
        self._validate_checkpoint_payload(payload)
        shared_head = payload.get("shared_head")
        shared_cvae = payload.get("shared_cvae")
        if self.bundle.shared_head is not None and shared_head is not None:
            self.bundle.shared_head.load_state_dict(shared_head)
        if self.bundle.shared_cvae is not None and shared_cvae is not None:
            self.bundle.shared_cvae.load_state_dict(shared_cvae)

        for k, state in payload.get("heads", {}).items():
            d = int(k)
            if d in self.bundle.heads:
                self.bundle.heads[d].load_state_dict(state)

        for k, state in payload.get("cvaes", {}).items():
            d = int(k)
            if d in self.bundle.cvaes:
                self.bundle.cvaes[d].load_state_dict(state)

    def _validate_checkpoint_payload(self, payload: Dict[str, object]) -> None:
        payload_variant = str(payload.get("variant", ""))
        if payload_variant and payload_variant != self.variant:
            raise ValueError(
                f"Resume checkpoint variant mismatch: expected {self.variant}, got {payload_variant}."
            )

        payload_domains = sorted(int(d) for d in payload.get("domains", []))
        expected_domains = sorted(int(d) for d in self.domains)
        if payload_domains and payload_domains != expected_domains:
            raise ValueError(
                f"Resume checkpoint domain mismatch: expected {expected_domains}, got {payload_domains}."
            )

        heads = {int(k) for k in payload.get("heads", {}).keys()}
        cvaes = {int(k) for k in payload.get("cvaes", {}).keys()}
        expected = set(expected_domains)
        shared_head = payload.get("shared_head")
        shared_cvae = payload.get("shared_cvae")

        if self.variant == VARIANT_A:
            if shared_head is None:
                raise ValueError("Resume checkpoint missing shared_head for variant A.")
            if cvaes != expected:
                raise ValueError(f"Resume checkpoint must include CVAE states for all domains: {expected_domains}.")
        elif self.variant == VARIANT_B:
            if shared_cvae is None:
                raise ValueError("Resume checkpoint missing shared_cvae for variant B.")
            if heads != expected:
                raise ValueError(f"Resume checkpoint must include head states for all domains: {expected_domains}.")
        elif self.variant == VARIANT_C:
            if heads != expected:
                raise ValueError(f"Resume checkpoint must include head states for all domains: {expected_domains}.")
            if cvaes != expected:
                raise ValueError(f"Resume checkpoint must include CVAE states for all domains: {expected_domains}.")
        elif self.variant == VARIANT_POOLED:
            if shared_head is None or shared_cvae is None:
                raise ValueError("Resume checkpoint missing shared modules for pooled variant.")

    def train(self, out_dir: Path, model_name: str, resume_from: Path | None = None) -> Tuple[Path, Dict[str, List[float]]]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"{model_name}.pt"
        state_ckpt = training_state_path(ckpt_path)

        optimizer = torch.optim.Adam(self._parameters(), lr=self.lr)
        train_loader = self._make_loader(self.train_x, self.train_meta, shuffle=True)
        val_loader = self._make_loader(self.val_x, self.val_meta, shuffle=False)

        best_val = float("inf")
        bad_epochs = 0
        history = {"train": [], "val": []}
        start_epoch = 0

        resume_state_path = None
        if resume_from is not None:
            resume_state_path = resume_from if resume_from.name.endswith(".training.pt") else training_state_path(resume_from)
        elif state_ckpt.exists():
            resume_state_path = state_ckpt

        if resume_state_path is not None and resume_state_path.exists():
            state = load_resume_state(resume_state_path)
            self._load_checkpoint_payload(state["model_payload"])
            optimizer.load_state_dict(state["optimizer_state"])
            history = state.get("history", history)
            start_epoch = int(state.get("epoch", -1)) + 1
            best_val = float(state.get("best_metric", best_val))
            bad_epochs = int(state.get("bad_epochs", bad_epochs))
        elif ckpt_path.exists():
            self._load_checkpoint_payload(safe_torch_load(ckpt_path, map_location="cpu"))

        epoch_iter = range(start_epoch, self.epochs)
        epoch_bar = tqdm(epoch_iter, desc=f"train:{model_name}", unit="epoch") if tqdm is not None else None

        for epoch in epoch_iter:
            self._set_train(True)
            train_loss_sum = 0.0
            train_count = 0
            for xb, db in train_loader:
                xb = xb.to(self.device)
                db = db.to(self.device)
                loss = self._forward_variant(xb, db)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_n = int(xb.shape[0])
                train_loss_sum += float(loss.item()) * batch_n
                train_count += batch_n

            self._set_train(False)
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, db in val_loader:
                    xb = xb.to(self.device)
                    db = db.to(self.device)
                    loss = self._forward_variant(xb, db)
                    batch_n = int(xb.shape[0])
                    val_loss_sum += float(loss.item()) * batch_n
                    val_count += batch_n

            train_epoch = train_loss_sum / max(train_count, 1)
            val_epoch = val_loss_sum / max(val_count, 1)
            history["train"].append(train_epoch)
            history["val"].append(val_epoch)

            if epoch_bar is not None:
                epoch_bar.set_postfix(
                    train=f"{train_epoch:.4f}",
                    val=f"{val_epoch:.4f}",
                    best=f"{best_val:.4f}",
                    bad=f"{bad_epochs}/{self.patience}",
                )
                epoch_bar.update(1)

            if val_epoch < best_val:
                best_val = val_epoch
                bad_epochs = 0
                self.save_checkpoint(ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    save_resume_state(
                        state_ckpt,
                        model_payload=build_hybrid_checkpoint_payload(
                            variant=self.variant,
                            domains=self.domains,
                            input_dim=self.input_dim,
                            projection_dim=self.projection_dim,
                            head_hidden_dim=self.head_hidden_dim,
                            cvae_hidden_dim=self.cvae_hidden_dim,
                            latent_dim=self.latent_dim,
                            bundle=self.bundle,
                        ),
                        optimizer_state=optimizer.state_dict(),
                        history=history,
                        epoch=epoch,
                        best_metric=best_val,
                        bad_epochs=bad_epochs,
                        meta={"model_name": model_name, "variant": self.variant},
                    )
                    break

            save_resume_state(
                state_ckpt,
                model_payload=build_hybrid_checkpoint_payload(
                    variant=self.variant,
                    domains=self.domains,
                    input_dim=self.input_dim,
                    projection_dim=self.projection_dim,
                    head_hidden_dim=self.head_hidden_dim,
                    cvae_hidden_dim=self.cvae_hidden_dim,
                    latent_dim=self.latent_dim,
                    bundle=self.bundle,
                ),
                optimizer_state=optimizer.state_dict(),
                history=history,
                epoch=epoch,
                best_metric=best_val,
                bad_epochs=bad_epochs,
                meta={"model_name": model_name, "variant": self.variant},
            )

        if epoch_bar is not None:
            epoch_bar.close()

        return ckpt_path, history
