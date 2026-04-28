from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.data.datasets.breakhis import BreakHisRecord
from src.torch_utils import safe_torch_load


def _normalize_extractor_config(backbone_type: str, extractor_config: Dict[str, object] | None) -> Dict[str, str]:
    cfg = dict(extractor_config or {})
    backbone_norm = str(backbone_type).strip().lower()

    feature_extractor_name = str(cfg.get("feature_extractor_name", backbone_norm)).strip().lower()
    if not feature_extractor_name:
        feature_extractor_name = backbone_norm

    feature_extractor_checkpoint = str(cfg.get("feature_extractor_checkpoint", "")).strip()

    default_layer = "final_norm_cls" if feature_extractor_name.startswith("dinov2") else "global_avg_pool"
    feature_extractor_layer = str(cfg.get("feature_extractor_layer", default_layer)).strip().lower()

    default_pooling = "cls_token" if feature_extractor_name.startswith("dinov2") else "global_avg"
    embedding_pooling = str(cfg.get("embedding_pooling", default_pooling)).strip().lower()

    return {
        "feature_extractor_name": feature_extractor_name,
        "feature_extractor_checkpoint": feature_extractor_checkpoint,
        "feature_extractor_layer": feature_extractor_layer,
        "embedding_pooling": embedding_pooling,
    }


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


class _DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, feature_layer: str, pooling: str) -> None:
        super().__init__()
        self.model = model
        self.feature_layer = str(feature_layer).strip().lower()
        self.pooling = str(pooling).strip().lower()

        allowed_layers = {"final_norm_cls", "final_norm_patch_mean", "prenorm_cls"}
        if self.feature_layer not in allowed_layers:
            raise ValueError(
                f"Unsupported DINOv2 feature layer '{self.feature_layer}'. Allowed: {sorted(allowed_layers)}"
            )

        allowed_pooling = {"cls_token", "patch_mean"}
        if self.pooling not in allowed_pooling:
            raise ValueError(
                f"Unsupported DINOv2 embedding pooling '{self.pooling}'. Allowed: {sorted(allowed_pooling)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        if isinstance(feats, dict):
            if self.feature_layer == "final_norm_cls" and self.pooling == "cls_token" and "x_norm_clstoken" in feats:
                return feats["x_norm_clstoken"]
            if (
                self.feature_layer in {"final_norm_cls", "final_norm_patch_mean"}
                and self.pooling == "patch_mean"
                and "x_norm_patchtokens" in feats
            ):
                return feats["x_norm_patchtokens"].mean(dim=1)
            if self.feature_layer == "prenorm_cls" and "x_prenorm" in feats:
                prenorm = feats["x_prenorm"]
                if prenorm.ndim == 3:
                    return prenorm[:, 0]
                return prenorm

        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.ndim == 3:
            return feats[:, 0]
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3))
        return feats


def _build_backbone(
    backbone_type: str,
    extractor_config: Dict[str, object] | None = None,
) -> Tuple[nn.Module, int, str, Dict[str, str]]:
    backbone = str(backbone_type).strip().lower()
    resolved_extractor_cfg = _normalize_extractor_config(backbone, extractor_config)

    if backbone == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor, 512, "resnet18", resolved_extractor_cfg

    if backbone == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet50(pretrained=True)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor, 2048, "resnet50", resolved_extractor_cfg

    if backbone == "dinov2_vitb14":
        try:
            import timm
        except Exception as exc:
            raise RuntimeError(
                "Backbone 'dinov2_vitb14' requires the optional dependency 'timm'. "
                "Install it with: pip install timm"
            ) from exc

        ckpt = str(resolved_extractor_cfg.get("feature_extractor_checkpoint", "")).strip().lower()
        if ckpt not in {"", "facebook/dinov2-base"}:
            raise ValueError(
                "features.feature_extractor_checkpoint currently supports only 'facebook/dinov2-base' "
                f"for dinov2_vitb14 (got: {ckpt})"
            )

        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
        feature_extractor = _DinoV2FeatureExtractor(
            model,
            feature_layer=resolved_extractor_cfg["feature_extractor_layer"],
            pooling=resolved_extractor_cfg["embedding_pooling"],
        )
        feature_extractor.eval()
        resolved_dim = int(getattr(model, "num_features", 768))
        return feature_extractor, resolved_dim, "dinov2_vitb14", resolved_extractor_cfg

    raise ValueError(
        f"Unsupported features.backbone_type '{backbone_type}'. "
        "Expected one of ['resnet18', 'resnet50', 'dinov2_vitb14']."
    )


def _to_2d_embeddings(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = features[0]

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Expected tensor features, got {type(features)}")

    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim == 3:
        features = features[:, 0]
    elif features.ndim != 2:
        raise ValueError(f"Expected feature tensor with 2-4 dims, got shape={tuple(features.shape)}")

    return features


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    records = [b[1] for b in batch]
    return imgs, records


def extract_and_cache_embeddings(
    records: List[BreakHisRecord],
    cache_dir: Path,
    image_size: int,
    batch_size: int,
    backbone_type: str = "resnet18",
    expected_dim: int | None = None,
    extractor_config: Dict[str, object] | None = None,
) -> Tuple[Dict[str, Path], Dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: cache_dir / f"{split}.pt" for split in ["train", "val", "test"]
    }

    model, resolved_dim, resolved_backbone, resolved_extractor_cfg = _build_backbone(
        backbone_type,
        extractor_config=extractor_config,
    )
    extraction_info: Dict[str, object] = {
        "backbone_type": resolved_backbone,
        "resolved_embedding_dim": int(resolved_dim),
        "expected_embedding_dim": int(expected_dim) if expected_dim is not None else None,
        "image_size": int(image_size),
        "feature_extractor_name": str(resolved_extractor_cfg["feature_extractor_name"]),
        "feature_extractor_checkpoint": str(resolved_extractor_cfg["feature_extractor_checkpoint"]),
        "feature_extractor_layer": str(resolved_extractor_cfg["feature_extractor_layer"]),
        "embedding_pooling": str(resolved_extractor_cfg["embedding_pooling"]),
    }
    if expected_dim is not None and int(expected_dim) != int(resolved_dim):
        raise ValueError(
            f"Configured features.embedding_dim={int(expected_dim)} does not match "
            f"resolved dimension {int(resolved_dim)} for backbone '{resolved_backbone}'."
        )

    if all(p.exists() for p in paths.values()):
        # Do not silently reuse stale empty caches.
        reusable = True
        expected_by_split = {
            split: sorted([r.image_path for r in records if r.split == split])
            for split in ["train", "val", "test"]
        }
        for split, p in paths.items():
            payload = safe_torch_load(p, map_location="cpu")
            if int(payload["embeddings"].shape[0]) <= 0:
                reusable = False
                break
            extractor_meta = payload.get("feature_extractor", {})
            cached_backbone = str(extractor_meta.get("backbone_type", "")).strip().lower()
            cached_dim = int(extractor_meta.get("embedding_dim", -1))
            if cached_backbone != resolved_backbone or cached_dim != int(resolved_dim):
                reusable = False
                break
            if str(extractor_meta.get("feature_extractor_name", "")).strip().lower() != str(
                resolved_extractor_cfg["feature_extractor_name"]
            ).strip().lower():
                reusable = False
                break
            if str(extractor_meta.get("feature_extractor_checkpoint", "")).strip().lower() != str(
                resolved_extractor_cfg["feature_extractor_checkpoint"]
            ).strip().lower():
                reusable = False
                break
            if str(extractor_meta.get("feature_extractor_layer", "")).strip().lower() != str(
                resolved_extractor_cfg["feature_extractor_layer"]
            ).strip().lower():
                reusable = False
                break
            if str(extractor_meta.get("embedding_pooling", "")).strip().lower() != str(
                resolved_extractor_cfg["embedding_pooling"]
            ).strip().lower():
                reusable = False
                break
            cached_paths = sorted([m["image_path"] for m in payload["metadata"]])
            if cached_paths != expected_by_split[split]:
                reusable = False
                break
        if reusable:
            return paths, extraction_info

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
                feats = _to_2d_embeddings(model(x)).cpu()
                all_embeddings.append(feats)
                all_meta.extend([asdict(r) for r in batch_records])

        embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, int(resolved_dim)))
        payload = {
            "embeddings": embeddings,
            "metadata": all_meta,
            "feature_extractor": {
                "backbone_type": resolved_backbone,
                "embedding_dim": int(resolved_dim),
                "image_size": int(image_size),
                "feature_extractor_name": str(resolved_extractor_cfg["feature_extractor_name"]),
                "feature_extractor_checkpoint": str(resolved_extractor_cfg["feature_extractor_checkpoint"]),
                "feature_extractor_layer": str(resolved_extractor_cfg["feature_extractor_layer"]),
                "embedding_pooling": str(resolved_extractor_cfg["embedding_pooling"]),
            },
        }
        torch.save(payload, paths[split])

        with (cache_dir / f"{split}_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2)

    return paths, extraction_info


def validate_embedding_cache(
    cache_paths: Dict[str, Path],
    expected_dim: int,
    expected_backbone_type: str | None = None,
    expected_feature_extractor: Dict[str, object] | None = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for split, path in cache_paths.items():
        payload = safe_torch_load(path, map_location="cpu")
        embeddings = payload["embeddings"]
        metadata = payload["metadata"]
        extractor_meta = payload.get("feature_extractor", {})
        cache_backbone = str(extractor_meta.get("backbone_type", "unknown"))
        cache_dim = int(extractor_meta.get("embedding_dim", embeddings.shape[1] if embeddings.ndim == 2 else -1))

        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(f"{split}: expected [N,{expected_dim}] got {tuple(embeddings.shape)}")
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise ValueError(f"{split}: embeddings contain NaN or Inf")
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"{split}: embedding count and metadata count mismatch")
        if cache_dim != expected_dim:
            raise ValueError(f"{split}: cache metadata embedding_dim={cache_dim} does not match expected {expected_dim}")
        if expected_backbone_type is not None and cache_backbone != str(expected_backbone_type):
            raise ValueError(
                f"{split}: cache metadata backbone_type='{cache_backbone}' does not match expected '{expected_backbone_type}'"
            )

        if expected_feature_extractor is not None:
            expected_name = str(expected_feature_extractor.get("feature_extractor_name", "")).strip().lower()
            expected_ckpt = str(expected_feature_extractor.get("feature_extractor_checkpoint", "")).strip().lower()
            expected_layer = str(expected_feature_extractor.get("feature_extractor_layer", "")).strip().lower()
            expected_pooling = str(expected_feature_extractor.get("embedding_pooling", "")).strip().lower()

            cache_name = str(extractor_meta.get("feature_extractor_name", "")).strip().lower()
            cache_ckpt = str(extractor_meta.get("feature_extractor_checkpoint", "")).strip().lower()
            cache_layer = str(extractor_meta.get("feature_extractor_layer", "")).strip().lower()
            cache_pooling = str(extractor_meta.get("embedding_pooling", "")).strip().lower()

            if expected_name and cache_name != expected_name:
                raise ValueError(
                    f"{split}: cache feature_extractor_name='{cache_name}' does not match expected '{expected_name}'"
                )
            if cache_ckpt != expected_ckpt:
                raise ValueError(
                    f"{split}: cache feature_extractor_checkpoint='{cache_ckpt}' does not match expected '{expected_ckpt}'"
                )
            if expected_layer and cache_layer != expected_layer:
                raise ValueError(
                    f"{split}: cache feature_extractor_layer='{cache_layer}' does not match expected '{expected_layer}'"
                )
            if expected_pooling and cache_pooling != expected_pooling:
                raise ValueError(
                    f"{split}: cache embedding_pooling='{cache_pooling}' does not match expected '{expected_pooling}'"
                )

        report[split] = {
            "num_samples": embeddings.shape[0],
            "shape": tuple(embeddings.shape),
            "backbone_type": cache_backbone,
            "embedding_dim": cache_dim,
            "image_size": int(extractor_meta.get("image_size", 0) or 0),
            "feature_extractor_name": str(extractor_meta.get("feature_extractor_name", "")),
            "feature_extractor_checkpoint": str(extractor_meta.get("feature_extractor_checkpoint", "")),
            "feature_extractor_layer": str(extractor_meta.get("feature_extractor_layer", "")),
            "embedding_pooling": str(extractor_meta.get("embedding_pooling", "")),
        }
    return report
