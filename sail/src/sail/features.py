"""Feature cache loading and optional Virchow2 cache creation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PipelineConfig
from .protocol import ProtocolError
from .splits import sample_id


@dataclass(frozen=True)
class FeatureCache:
    embeddings: Any
    metadata: tuple[Mapping[str, object], ...]
    feature_extractor: Mapping[str, object]


@dataclass(frozen=True)
class CacheBuildRequest:
    samples_manifest: Path
    output_root: Path
    experiment_seed: int
    backbone_name: str = "virchow2"
    model_ref: str = "hf-hub:paige-ai/Virchow2"
    batch_size: int = 32
    device: str = "auto"
    splits: tuple[str, ...] = ("train", "val", "test")
    limit_samples_per_split: int | None = None
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class CacheBuildResult:
    status: str
    output_paths: Mapping[str, Path]
    report_path: Path
    split_counts: Mapping[str, int]


def cache_path(config: PipelineConfig, repo_root: Path, *, seed: int, split: str) -> Path:
    return repo_root / config.cache_path_template.format(
        cache_root=config.cache_root,
        backbone=config.primary_backbone,
        seed=int(seed),
        split=str(split),
    )


def load_feature_cache(path: Path) -> FeatureCache:
    path = Path(path)
    if path.suffix == ".npz":
        return _load_npz_cache(path)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError(f"Loading torch feature caches requires torch: {path}") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Feature cache is not a mapping: {path}")
    return _cache_from_payload(payload, path)


def write_npz_cache(path: Path, embeddings: Any, metadata: Sequence[Mapping[str, object]]) -> None:
    """Small test helper; production caches are usually torch ``.pt`` files."""

    import numpy as np  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=float),
        metadata_json=json.dumps([dict(row) for row in metadata], sort_keys=True),
    )


def build_virchow2_cache(request: CacheBuildRequest) -> CacheBuildResult:
    rows_by_split = read_manifest_rows_by_split(
        request.samples_manifest,
        splits=request.splits,
        limit_samples_per_split=request.limit_samples_per_split,
    )
    output_paths = {
        split: request.output_root / request.backbone_name / f"seed{int(request.experiment_seed)}" / "embeddings" / f"{split}.pt"
        for split in request.splits
    }
    report_path = request.output_root / request.backbone_name / f"seed{int(request.experiment_seed)}" / "reports" / "cache_builder_report.json"
    split_counts = {split: len(rows_by_split.get(split, ())) for split in request.splits}
    if request.dry_run:
        _write_cache_report(report_path, request, output_paths, split_counts, "dry_run_passed", {})
        return CacheBuildResult("dry_run_passed", output_paths, report_path, split_counts)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not request.overwrite:
        _write_cache_report(report_path, request, output_paths, split_counts, "reused_existing", {})
        return CacheBuildResult("reused_existing", output_paths, report_path, split_counts)
    extractor = _load_virchow2_extractor(request)
    for split, rows in rows_by_split.items():
        embeddings = extractor.extract(rows, batch_size=int(request.batch_size))
        payload = {
            "embeddings": embeddings,
            "metadata": [canonical_metadata(row, split=split) for row in rows],
            "feature_extractor": {
                "backbone_type": "virchow2",
                "model_ref": request.model_ref,
                "loader": "timm_hf",
                "pooling_policy": "class_token_plus_mean_patch_tokens_skip_registers",
                "dataset_level_normalization": "forbidden_not_used",
                "target_labels_used_for_extractor_fitting": False,
                "cache_builder": "sail_virchow2_cache_builder_v1",
            },
        }
        _assert_cache_payload(payload, expected_rows=len(rows), split=split)
        path = output_paths[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        _torch_save(payload, path)
    _write_cache_report(report_path, request, output_paths, split_counts, "complete", {"model_ref": request.model_ref})
    return CacheBuildResult("complete", output_paths, report_path, split_counts)


def read_manifest_rows_by_split(
    manifest_path: Path,
    *,
    splits: Sequence[str],
    limit_samples_per_split: int | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty samples manifest: {manifest_path}")
        required = {"sample_id", "image_path", "label", "split"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ProtocolError(f"samples manifest is missing required fields {missing}")
        by_split: dict[str, list[dict[str, object]]] = {str(split): [] for split in splits}
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            if split not in by_split:
                continue
            resolved = dict(row)
            resolved["image_path"] = str(_resolve_image_path(manifest_path, str(row.get("image_path", ""))))
            resolved.setdefault("center", row.get("magnification", ""))
            by_split[split].append(resolved)
    out = {}
    for split in splits:
        rows = by_split[str(split)]
        if limit_samples_per_split is not None:
            rows = rows[: int(limit_samples_per_split)]
        if not rows:
            raise ProtocolError(f"No rows for split={split} in {manifest_path}")
        out[str(split)] = tuple(rows)
    return out


def canonical_metadata(row: Mapping[str, object], *, split: str) -> dict[str, object]:
    center = str(row.get("center", "") or row.get("magnification", "")).strip()
    if not center:
        raise ProtocolError(f"Manifest row lacks center/magnification: {row}")
    return {
        **dict(row),
        "sample_id": sample_id(row),
        "label": int(float(str(row.get("label", "")))),
        "split": str(split),
        "center": center,
        "magnification": str(row.get("magnification", center)),
    }


def _load_npz_cache(path: Path) -> FeatureCache:
    import numpy as np  # type: ignore

    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata_json"].item()))
    return FeatureCache(
        embeddings=payload["embeddings"],
        metadata=tuple(dict(row) for row in metadata),
        feature_extractor={"loader": "npz_test_or_lightweight_cache"},
    )


def _cache_from_payload(payload: Mapping[str, Any], path: Path) -> FeatureCache:
    if "embeddings" not in payload or "metadata" not in payload:
        raise ProtocolError(f"Feature cache must contain embeddings and metadata: {path}")
    return FeatureCache(
        embeddings=payload["embeddings"],
        metadata=tuple(payload["metadata"]),
        feature_extractor=payload.get("feature_extractor", {}) if isinstance(payload.get("feature_extractor", {}), Mapping) else {},
    )


def _assert_cache_payload(payload: Mapping[str, Any], *, expected_rows: int, split: str) -> None:
    embeddings = payload.get("embeddings")
    metadata = payload.get("metadata")
    if embeddings is None or metadata is None:
        raise ProtocolError(f"{split}: cache payload must contain embeddings and metadata")
    if int(getattr(embeddings, "ndim", 0)) != 2:
        raise ProtocolError(f"{split}: embeddings must be 2D")
    if int(embeddings.shape[0]) != int(expected_rows) or int(embeddings.shape[0]) != len(metadata):
        raise ProtocolError(f"{split}: embedding row count does not match metadata")


class _Virchow2Extractor:
    def __init__(self, model: Any, transform: Any, device: Any) -> None:
        self.model = model
        self.transform = transform
        self.device = device

    def extract(self, rows: Sequence[Mapping[str, object]], *, batch_size: int) -> Any:
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        chunks = []
        with torch.no_grad():
            for start in range(0, len(rows), int(batch_size)):
                batch = rows[start : start + int(batch_size)]
                images = [Image.open(str(row["image_path"])).convert("RGB") for row in batch]
                tensors = [self.transform(image) for image in images]
                inputs = torch.stack(tensors, dim=0).to(self.device)
                outputs = self.model(inputs)
                chunks.append(_virchow2_embedding_from_tokens(outputs).detach().cpu().float())
                for image in images:
                    image.close()
        return torch.cat(chunks, dim=0)


def _load_virchow2_extractor(request: CacheBuildRequest) -> _Virchow2Extractor:
    try:
        import torch  # type: ignore
        import timm  # type: ignore
        from timm.data import resolve_data_config  # type: ignore
        from timm.data.transforms_factory import create_transform  # type: ignore
        from timm.layers import SwiGLUPacked  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Virchow2 cache creation requires torch, timm, and Pillow.") from exc
    device = _resolve_device(torch, request.device)
    model = timm.create_model(
        request.model_ref,
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval()
    model.to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return _Virchow2Extractor(model=model, transform=transform, device=device)


def _virchow2_embedding_from_tokens(outputs: Any) -> Any:
    tokens = _resolve_tensor(outputs)
    if getattr(tokens, "ndim", 0) != 3 or int(tokens.shape[1]) <= 5:
        raise RuntimeError(f"Virchow2 expected class/register/patch tokens, got shape={getattr(tokens, 'shape', None)}")
    return _torch_cat((tokens[:, 0], tokens[:, 5:].mean(dim=1)), dim=-1)


def _resolve_tensor(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise RuntimeError(f"Could not resolve tensor from output type {type(value)}")
    if getattr(value, "ndim", None) is not None:
        return value
    if isinstance(value, Mapping):
        for candidate in value.values():
            try:
                return _resolve_tensor(candidate, depth=depth + 1)
            except RuntimeError:
                continue
    if hasattr(value, "to_tuple"):
        return _resolve_tensor(value.to_tuple(), depth=depth + 1)
    if isinstance(value, (tuple, list)):
        for candidate in value:
            try:
                return _resolve_tensor(candidate, depth=depth + 1)
            except RuntimeError:
                continue
    raise RuntimeError(f"Could not resolve tensor from output type {type(value)}")


def _resolve_device(torch: Any, raw: str) -> Any:
    requested = str(raw or "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def _torch_cat(values: Sequence[Any], *, dim: int) -> Any:
    import torch  # type: ignore

    return torch.cat(tuple(values), dim=int(dim))


def _torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch  # type: ignore

    torch.save(dict(payload), path)


def _resolve_image_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return Path(manifest_path).resolve().parent.parent / path


def _write_cache_report(
    path: Path,
    request: CacheBuildRequest,
    output_paths: Mapping[str, Path],
    split_counts: Mapping[str, int],
    status: str,
    feature_extractor: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "sail_virchow2_cache_builder_report_v1",
        "status": status,
        "backbone_name": request.backbone_name,
        "experiment_seed": int(request.experiment_seed),
        "samples_manifest": str(request.samples_manifest),
        "output_paths": {split: str(item) for split, item in output_paths.items()},
        "split_counts": dict(split_counts),
        "dry_run": bool(request.dry_run),
        "feature_extractor": dict(feature_extractor),
        "protocol": {
            "target_eval_metrics_used": False,
            "target_labels_used_for_extractor_fitting": False,
            "dataset_level_normalization": "forbidden_not_used",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
