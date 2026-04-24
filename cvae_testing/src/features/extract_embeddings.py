from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.data.datasets.breakhis import BreakHisRecord


class RecordImageDataset(Dataset):
    def __init__(self, records: List[BreakHisRecord], image_size: int) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        return self.transform(image), rec


def _build_resnet18() -> torch.nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def _feature_protocol(feature_config: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = feature_config or {}
    name = str(
        cfg.get("feature_extractor_name")
        or cfg.get("extractor")
        or cfg.get("backbone")
        or "resnet18"
    )
    if name in {"dinov2", "dinov2_base", "dinov2_vitb14"}:
        return {
            "feature_extractor_name": "dinov2_vitb14",
            "feature_extractor_checkpoint": str(
                cfg.get("feature_extractor_checkpoint", "facebook/dinov2-base")
            ),
            "feature_extractor_layer": str(cfg.get("feature_extractor_layer", "final_norm_cls")),
            "embedding_pooling": str(cfg.get("embedding_pooling", "cls_token")),
            "embedding_dim": int(cfg.get("embedding_dim", 768)),
        }

    return {
        "feature_extractor_name": "resnet18",
        "feature_extractor_checkpoint": str(
            cfg.get("feature_extractor_checkpoint", "torchvision/resnet18_default")
        ),
        "feature_extractor_layer": str(cfg.get("feature_extractor_layer", "avgpool")),
        "embedding_pooling": str(cfg.get("embedding_pooling", "global_avg_pool")),
        "embedding_dim": int(cfg.get("embedding_dim", 512)),
    }


class _DinoV2Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "last_hidden_state" in out:
                return out["last_hidden_state"][:, 0]
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            return out[:, 0]
        return out


def _build_dinov2_base() -> torch.nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2 Base via torch.hub. Ensure facebookresearch/dinov2 "
            "is available in the torch hub cache or allow the run environment to download it."
        ) from exc
    model.eval()
    return _DinoV2Wrapper(model)


def _build_feature_extractor(feature_config: Dict[str, Any] | None) -> tuple[torch.nn.Module, Dict[str, Any]]:
    protocol = _feature_protocol(feature_config)
    name = str(protocol["feature_extractor_name"])
    if name == "dinov2_vitb14":
        return _build_dinov2_base(), protocol
    if name == "resnet18":
        return _build_resnet18(), protocol
    raise ValueError(f"Unsupported feature extractor: {name}")


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    records = [b[1] for b in batch]
    return imgs, records


def extract_and_cache_embeddings(
    records: List[BreakHisRecord],
    cache_dir: Path,
    image_size: int,
    batch_size: int,
    feature_config: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: cache_dir / f"{split}.pt" for split in ["train", "val", "test"]
    }
    requested_protocol = _feature_protocol(feature_config)
    if all(p.exists() for p in paths.values()):
        # Do not silently reuse stale empty caches.
        reusable = True
        expected_by_split = {
            split: sorted([r.image_path for r in records if r.split == split])
            for split in ["train", "val", "test"]
        }
        for split, p in paths.items():
            payload = torch.load(p, map_location="cpu")
            if int(payload["embeddings"].shape[0]) <= 0:
                reusable = False
                break
            if payload.get("feature_extractor") != requested_protocol:
                reusable = False
                break
            cached_paths = sorted([m["image_path"] for m in payload["metadata"]])
            if cached_paths != expected_by_split[split]:
                reusable = False
                break
        if reusable:
            return paths

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, feature_protocol = _build_feature_extractor(feature_config)
    model = model.to(device)

    by_split = {
        split: [r for r in records if r.split == split] for split in ["train", "val", "test"]
    }

    for split, split_records in by_split.items():
        ds = RecordImageDataset(split_records, image_size=image_size)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

        all_embeddings = []
        all_meta = []
        with torch.no_grad():
            for x, batch_records in dl:
                x = x.to(device)
                feats = model(x)
                if feats.ndim == 4:
                    feats = feats.squeeze(-1).squeeze(-1)
                feats = feats.cpu()
                all_embeddings.append(feats)
                all_meta.extend([asdict(r) for r in batch_records])

        expected_dim = int(feature_protocol["embedding_dim"])
        embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, expected_dim))
        payload = {
            "embeddings": embeddings,
            "metadata": all_meta,
            "feature_extractor": feature_protocol,
        }
        torch.save(payload, paths[split])

        with (cache_dir / f"{split}_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2)

    return paths


def validate_embedding_cache(cache_paths: Dict[str, Path], expected_dim: int = 512) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for split, path in cache_paths.items():
        payload = torch.load(path, map_location="cpu")
        embeddings = payload["embeddings"]
        metadata = payload["metadata"]

        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(f"{split}: expected [N,{expected_dim}] got {tuple(embeddings.shape)}")
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise ValueError(f"{split}: embeddings contain NaN or Inf")
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"{split}: embedding count and metadata count mismatch")

        report[split] = {
            "num_samples": embeddings.shape[0],
            "shape": tuple(embeddings.shape),
        }
    return report
