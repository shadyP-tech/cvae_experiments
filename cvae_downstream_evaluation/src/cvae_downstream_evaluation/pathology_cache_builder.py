"""Build frozen pathology foundation embedding caches for R1.2.

The cache builder is an extraction-only preprocessing step. It reads the
authoritative Camelyon17 ``samples.csv`` manifest from a synced support run,
loads a frozen vision backbone, writes split-level embedding caches, and emits
provenance. It must not fit dataset-level normalization, inspect target
evaluation scores, update CVAE experts, or choose a backbone.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .pathology_embedding_screen import R12Config, default_r12_config
from .protocol import ProtocolError


SUPPORTED_HF_AUTO_BACKBONES = {"phikon", "plip"}
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CacheBuildRequest:
    backbone_name: str
    model_dir: Path
    experiment_seed: int
    support_run_dir: Path
    output_root: Path
    batch_size: int = 32
    device: str = "auto"
    splits: tuple[str, ...] = DEFAULT_SPLITS
    image_size: int | None = None
    limit_samples_per_split: int | None = None
    overwrite: bool = False
    dry_run: bool = False
    local_files_only: bool = True
    loader: str = "hf_auto"


@dataclass(frozen=True)
class CacheBuildResult:
    status: str
    output_paths: Mapping[str, Path]
    report_path: Path
    split_counts: Mapping[str, int]


def build_r12_pathology_embedding_cache(request: CacheBuildRequest) -> CacheBuildResult:
    """Build split-level R1.2 embedding caches for one backbone/seed."""

    manifest_path = request.support_run_dir / "manifests" / "samples.csv"
    if not manifest_path.exists():
        raise ProtocolError(f"Missing support-run samples manifest: {manifest_path}")
    if not request.model_dir.exists():
        raise ProtocolError(f"Missing model directory: {request.model_dir}")

    rows_by_split = read_manifest_rows_by_split(
        manifest_path,
        splits=request.splits,
        limit_samples_per_split=request.limit_samples_per_split,
    )
    output_paths = {
        split: request.output_root / request.backbone_name / f"seed{int(request.experiment_seed)}" / "embeddings" / f"{split}.pt"
        for split in request.splits
    }
    report_path = request.output_root / request.backbone_name / f"seed{int(request.experiment_seed)}" / "reports" / "cache_builder_report.json"

    if request.dry_run:
        report = _report_payload(
            request=request,
            manifest_path=manifest_path,
            output_paths=output_paths,
            split_counts={split: len(rows_by_split.get(split, ())) for split in request.splits},
            status="dry_run_passed",
            feature_extractor={},
        )
        return CacheBuildResult(
            status=str(report["status"]),
            output_paths=output_paths,
            report_path=report_path,
            split_counts=report["split_counts"],
        )

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not request.overwrite:
        missing = [path for path in output_paths.values() if not path.exists()]
        if missing:
            raise ProtocolError(
                "Partial pathology cache exists. Use --overwrite after inspecting: "
                + ", ".join(str(path) for path in sorted(existing + missing))
            )
        feature = read_existing_feature_extractor(next(iter(output_paths.values())))
        write_cache_builder_report(
            report_path,
            request=request,
            manifest_path=manifest_path,
            output_paths=output_paths,
            split_counts={split: len(rows_by_split.get(split, ())) for split in request.splits},
            status="reused_existing",
            feature_extractor=feature,
        )
        return CacheBuildResult(
            status="reused_existing",
            output_paths=output_paths,
            report_path=report_path,
            split_counts={split: len(rows_by_split.get(split, ())) for split in request.splits},
        )

    extractor = load_pathology_feature_extractor(request)
    for split in request.splits:
        rows = rows_by_split.get(split, ())
        output_path = output_paths[split]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = extractor.extract(rows, batch_size=int(request.batch_size))
        payload = {
            "embeddings": embeddings,
            "metadata": [canonical_cache_metadata(row, split=split) for row in rows],
            "feature_extractor": {
                **extractor.feature_metadata,
                "backbone_type": request.backbone_name,
                "embedding_dim": int(embeddings.shape[1]) if int(embeddings.ndim) == 2 else 0,
                "image_size": int(request.image_size) if request.image_size is not None else "",
                "local_files_only": bool(request.local_files_only),
                "cache_builder": "r12_pathology_cache_builder_v1",
                "dataset_level_normalization": "forbidden_not_used",
                "extractor_fitted_on_camelyon17_labels": False,
            },
        }
        assert_cache_payload(payload, expected_rows=len(rows), split=split)
        _torch_save(payload, output_path)

    write_cache_builder_report(
        report_path,
        request=request,
        manifest_path=manifest_path,
        output_paths=output_paths,
        split_counts={split: len(rows_by_split.get(split, ())) for split in request.splits},
        status="complete",
        feature_extractor=extractor.feature_metadata,
    )
    return CacheBuildResult(
        status="complete",
        output_paths=output_paths,
        report_path=report_path,
        split_counts={split: len(rows_by_split.get(split, ())) for split in request.splits},
    )


def default_support_run_dir(config: R12Config, repo_root: Path, experiment_seed: int) -> Path:
    return (
        repo_root
        / config.z11_config.expected_support_run_root
        / config.z11_config.expected_support_run_dir_pattern.format(seed=int(experiment_seed))
    )


def default_output_root(config: R12Config, repo_root: Path) -> Path:
    return repo_root / config.cache_root


def read_manifest_rows_by_split(
    manifest_path: Path,
    *,
    splits: Sequence[str],
    limit_samples_per_split: int | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty samples manifest: {manifest_path}")
        required = {"sample_id", "image_path", "label", "split"}
        missing = sorted(required.difference(set(reader.fieldnames)))
        if missing:
            raise ProtocolError(f"samples.csv is missing required fields {missing}: {manifest_path}")
        rows_by_split = {str(split): [] for split in splits}
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            if split not in rows_by_split:
                continue
            resolved = dict(row)
            resolved["image_path"] = str(resolve_manifest_image_path(manifest_path, str(row.get("image_path", ""))))
            resolved.setdefault("center", row.get("magnification", ""))
            rows_by_split[split].append(resolved)
    out: dict[str, tuple[dict[str, object], ...]] = {}
    for split in splits:
        rows = rows_by_split[str(split)]
        if limit_samples_per_split is not None:
            rows = rows[: int(limit_samples_per_split)]
        if not rows:
            raise ProtocolError(f"No manifest rows found for split={split} in {manifest_path}")
        out[str(split)] = tuple(rows)
    return out


def resolve_manifest_image_path(manifest_path: Path, raw_path: str) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ProtocolError(f"Manifest row has empty image_path in {manifest_path}")
    path = Path(text)
    if path.is_absolute():
        return path
    repo_root = _infer_repo_root_from_manifest(manifest_path)
    return repo_root / path


def canonical_cache_metadata(row: Mapping[str, object], *, split: str) -> dict[str, object]:
    sample_id = str(row.get("sample_id", "")).strip()
    if not sample_id:
        raise ProtocolError(f"Manifest row lacks sample_id: {row}")
    label = int(float(str(row.get("label", "")).strip()))
    center = str(row.get("center", "") or row.get("magnification", "")).strip()
    if not center:
        raise ProtocolError(f"Manifest row lacks center/magnification: {row}")
    return {
        **dict(row),
        "sample_id": sample_id,
        "image_path": str(row.get("image_path", "")),
        "label": label,
        "split": str(split),
        "magnification": str(row.get("magnification", center)),
        "center": center,
    }


def assert_cache_payload(payload: Mapping[str, Any], *, expected_rows: int, split: str) -> None:
    embeddings = payload.get("embeddings")
    metadata = payload.get("metadata")
    if embeddings is None or metadata is None:
        raise ProtocolError(f"{split}: cache payload must contain embeddings and metadata")
    if int(getattr(embeddings, "ndim", 0)) != 2:
        raise ProtocolError(f"{split}: embeddings must be 2D, got shape={getattr(embeddings, 'shape', None)}")
    if int(embeddings.shape[0]) != int(expected_rows):
        raise ProtocolError(f"{split}: embedding row count does not match manifest count")
    if int(embeddings.shape[0]) != len(metadata):
        raise ProtocolError(f"{split}: embedding row count does not match metadata count")
    if _torch_is_bad(embeddings):
        raise ProtocolError(f"{split}: embeddings contain NaN or Inf")
    for idx, row in enumerate(metadata):
        for key in ("sample_id", "image_path", "label", "split", "magnification"):
            if str(row.get(key, "")).strip() == "":
                raise ProtocolError(f"{split}: metadata row {idx} lacks {key}")


class PathologyFeatureExtractor:
    def __init__(self, *, model: Any, processor: Any, device: Any, feature_metadata: Mapping[str, object]) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.feature_metadata = dict(feature_metadata)

    def extract(self, rows: Sequence[Mapping[str, object]], *, batch_size: int) -> Any:
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        chunks = []
        with torch.no_grad():
            for start in range(0, len(rows), int(batch_size)):
                batch = rows[start : start + int(batch_size)]
                images = [Image.open(str(row["image_path"])).convert("RGB") for row in batch]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                feats = extract_image_features(self.model, inputs)
                chunks.append(to_2d_tensor(feats).detach().cpu().float())
                for image in images:
                    image.close()
        if not chunks:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


def load_pathology_feature_extractor(request: CacheBuildRequest) -> PathologyFeatureExtractor:
    loader = str(request.loader).strip().lower()
    if loader != "hf_auto":
        raise ProtocolError(f"Unsupported R1.2 cache loader '{request.loader}'. Supported: hf_auto")
    if str(request.backbone_name).strip().lower() not in SUPPORTED_HF_AUTO_BACKBONES:
        raise ProtocolError(
            f"Backbone '{request.backbone_name}' is not enabled for the HF auto cache builder yet. "
            f"Supported smoke backbones: {sorted(SUPPORTED_HF_AUTO_BACKBONES)}"
        )
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "R1.2 pathology cache extraction requires torch and transformers in the workstation venv."
        ) from exc

    device = resolve_device(request.device)
    model_dir = str(request.model_dir)
    processor = None
    processor_error = None
    try:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=bool(request.local_files_only))
    except Exception as exc:
        processor_error = exc
    if processor is None:
        try:
            processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=bool(request.local_files_only))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load image processor from {model_dir}. "
                f"AutoProcessor error: {processor_error}; AutoImageProcessor error: {exc}"
            ) from exc
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=bool(request.local_files_only),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    metadata = {
        "backbone_type": str(request.backbone_name),
        "feature_extractor_name": str(request.backbone_name),
        "feature_extractor_checkpoint": str(request.model_dir),
        "feature_extractor_layer": "auto_image_features",
        "embedding_pooling": "get_image_features_or_pooler_or_cls",
        "loader": "hf_auto",
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "device": str(device),
    }
    try:
        first_param = next(model.parameters())
        metadata["torch_dtype"] = str(first_param.dtype)
    except StopIteration:
        metadata["torch_dtype"] = ""
    return PathologyFeatureExtractor(model=model, processor=processor, device=device, feature_metadata=metadata)


def resolve_device(raw: str) -> Any:
    import torch  # type: ignore

    requested = str(raw).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def extract_image_features(model: Any, inputs: Mapping[str, Any]) -> Any:
    if hasattr(model, "get_image_features") and "pixel_values" in inputs:
        return model.get_image_features(pixel_values=inputs["pixel_values"])
    outputs = model(**inputs)
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0]
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if getattr(first, "ndim", 0) == 3:
            return first[:, 0]
        return first
    raise RuntimeError(f"Could not extract image features from model output type {type(outputs)}")


def to_2d_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if getattr(value, "ndim", 0) == 3:
        value = value[:, 0]
    if getattr(value, "ndim", 0) == 4:
        value = value.mean(dim=(2, 3))
    if getattr(value, "ndim", 0) != 2:
        raise RuntimeError(f"Expected 2D feature tensor, got shape={getattr(value, 'shape', None)}")
    return value


def read_existing_feature_extractor(path: Path) -> Mapping[str, object]:
    import torch  # type: ignore

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    feature = payload.get("feature_extractor", {}) if isinstance(payload, Mapping) else {}
    return feature if isinstance(feature, Mapping) else {}


def write_cache_builder_report(
    path: Path,
    *,
    request: CacheBuildRequest,
    manifest_path: Path,
    output_paths: Mapping[str, Path],
    split_counts: Mapping[str, int],
    status: str,
    feature_extractor: Mapping[str, object],
) -> None:
    payload = _report_payload(
        request=request,
        manifest_path=manifest_path,
        output_paths=output_paths,
        split_counts=split_counts,
        status=status,
        feature_extractor=feature_extractor,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_payload(
    *,
    request: CacheBuildRequest,
    manifest_path: Path,
    output_paths: Mapping[str, Path],
    split_counts: Mapping[str, int],
    status: str,
    feature_extractor: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "r12_pathology_cache_builder_report_v1",
        "status": status,
        "backbone_name": request.backbone_name,
        "experiment_seed": int(request.experiment_seed),
        "model_dir": str(request.model_dir),
        "support_run_dir": str(request.support_run_dir),
        "samples_manifest": str(manifest_path),
        "output_paths": {split: str(path) for split, path in output_paths.items()},
        "split_counts": dict(split_counts),
        "batch_size": int(request.batch_size),
        "device": str(request.device),
        "splits": list(request.splits),
        "limit_samples_per_split": request.limit_samples_per_split,
        "overwrite": bool(request.overwrite),
        "dry_run": bool(request.dry_run),
        "local_files_only": bool(request.local_files_only),
        "loader": request.loader,
        "feature_extractor": dict(feature_extractor),
        "protocol": {
            "dataset_level_normalization": "forbidden_not_used",
            "target_eval_metrics_used": False,
            "target_labels_used_for_model_or_processor_fitting": False,
            "cvae_experts_modified": False,
        },
    }


def parse_csv_list(raw: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return tuple(default)
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


def _infer_repo_root_from_manifest(manifest_path: Path) -> Path:
    marker = Path("cvae_testing") / "outputs"
    parts = manifest_path.resolve().parts
    marker_parts = marker.parts
    for idx in range(0, len(parts) - len(marker_parts) + 1):
        if tuple(parts[idx : idx + len(marker_parts)]) == marker_parts:
            return Path(*parts[:idx])
    return manifest_path.resolve().parents[5]


def _torch_is_bad(value: Any) -> bool:
    import torch  # type: ignore

    return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())


def _torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch  # type: ignore

    torch.save(dict(payload), path)


def _non_nan_mean(values: Iterable[object]) -> float:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    parsed = [value for value in parsed if not math.isnan(value)]
    return sum(parsed) / float(len(parsed)) if parsed else math.nan
