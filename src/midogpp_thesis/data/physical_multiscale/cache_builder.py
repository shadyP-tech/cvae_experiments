"""Workstation-only B/C cache construction from the immutable physical contract."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.paths import resolve_contract_path
from midogpp_thesis.data.features.cache_io import load_cache_rows, write_center_shard
from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor

from .bridge import evaluate_jpeg_task_bridge
from .config import PhysicalMultiscaleBuildConfig
from .slide_reader import open_slide as _open_slide


def build_physical_multiscale_caches(
    config: PhysicalMultiscaleBuildConfig,
) -> tuple[Path, Path]:
    """Build, independently validate, and atomically publish B/C cache roots."""

    from .validation import (
        validate_cache_bundle,
        validate_cache_pair,
        validate_contract_bundle,
    )

    validate_contract_bundle(config.contract_root)
    _assert_absent(config.b_cache_root)
    _assert_absent(config.c_cache_root)
    b_stage = config.b_cache_root.with_name(f".{config.b_cache_root.name}.building")
    c_stage = config.c_cache_root.with_name(f".{config.c_cache_root.name}.building")
    _assert_absent(b_stage)
    _assert_absent(c_stage)
    staging = replace(config, b_cache_root=b_stage, c_cache_root=c_stage)
    try:
        _build_physical_multiscale_caches_in_place(staging)
        validate_cache_bundle(
            b_stage,
            expected_dim=3840,
            config=staging,
        )
        validate_cache_bundle(
            c_stage,
            expected_dim=11520,
            config=staging,
        )
        validate_cache_pair(
            b_stage,
            c_stage,
            contract_root=config.contract_root,
            canonical_cache_path=config.canonical_cache_path,
            canonical_reference_root=config.canonical_reference_root,
            config=staging,
        )
        b_stage.rename(config.b_cache_root)
        try:
            c_stage.rename(config.c_cache_root)
        except Exception:
            config.b_cache_root.rename(b_stage)
            raise
    except Exception:
        _quarantine_staging(b_stage)
        _quarantine_staging(c_stage)
        raise
    return config.b_cache_root, config.c_cache_root


def _build_physical_multiscale_caches_in_place(
    config: PhysicalMultiscaleBuildConfig,
) -> tuple[Path, Path]:
    """Materialize complete cache roots at isolated staging paths."""

    _assert_empty(config.b_cache_root)
    _assert_empty(config.c_cache_root)
    _seed_extraction(config.experiment_seed)
    contract_rows = _read_csv(
        config.contract_root / "physical_multiscale_manifest.csv"
    )
    base_rows = _read_csv(config.base_manifest_path)
    base_by_id = {str(row["sample_id"]): row for row in base_rows}
    canonical = load_cache_rows(config.canonical_cache_path, expected_dim=2560)
    canonical_by_id = {
        sample_id: canonical.embeddings[index]
        for index, sample_id in enumerate(canonical.sample_ids)
    }
    expected_ids = tuple(str(row["sample_id"]) for row in contract_rows)
    expected_id_set = set(expected_ids)
    if tuple(
        sample_id
        for sample_id in canonical.sample_ids
        if sample_id in expected_id_set
    ) != expected_ids:
        raise ValueError("Physical contract order differs from canonical cache order.")

    extractor = Virchow2TokenExtractor(
        model_ref=config.model_ref,
        model_revision=config.model_revision,
        device=config.device,
        expected_model_config_sha256=config.expected_model_config_sha256,
        expected_checkpoint_file_sha256=config.expected_checkpoint_file_sha256,
        expected_state_dict_sha256=config.expected_state_dict_sha256,
        expected_preprocessing_config_hash=config.expected_preprocessing_config_hash,
    )
    grouped: dict[str, list[Mapping[str, str]]] = {
        center: [] for center in config.eligible_centers
    }
    for row in contract_rows:
        center = str(row["center"])
        if center not in grouped:
            raise ValueError(f"Physical contract contains unexpected center: {center}")
        grouped[center].append(row)

    bridge_cosines: list[float] = []
    bridge_relative_l2: list[float] = []
    for center in config.eligible_centers:
        rows = grouped[center]
        b_chunks = []
        c_chunks = []
        a_chunks = []
        metadata_rows: list[dict[str, object]] = []
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start : start + config.batch_size]
            jpeg_images = []
            c_images: list[Image.Image] = []
            for row in batch:
                sample_id = str(row["sample_id"])
                base = base_by_id[sample_id]
                jpeg_path = resolve_contract_path(
                    config.repo_root,
                    Path(str(base["image_path"])),
                )
                jpeg_images.append(Image.open(jpeg_path).convert("RGB"))
                with _open_slide(
                    _repo_path(config.repo_root, row["raw_tiff_path"]),
                    require_tiled=config.require_tiled_reader,
                ) as slide:
                    geometries = json.loads(str(row["scale_geometry_json"]))
                    for key in ("28um", "56um", "112um"):
                        c_images.append(
                            slide.read_geometry(
                                geometries[key],
                                padding_rgb=config.padding_rgb,
                            )
                        )
            try:
                b = extractor.extract_images(jpeg_images, include_center=True)
                scale_blocks = []
                for offset in range(3):
                    scale_blocks.append(
                        extractor.extract_images(
                            c_images[offset::3],
                            include_center=True,
                        )
                    )
                import torch  # type: ignore

                c = torch.cat(tuple(scale_blocks), dim=1)
                a = torch.stack(
                    tuple(canonical_by_id[str(row["sample_id"])] for row in batch),
                    dim=0,
                ).detach().cpu().float()
                prefix = b[:, :2560]
                bridge_cosines.extend(
                    torch.nn.functional.cosine_similarity(prefix, a, dim=1).tolist()
                )
                bridge_relative_l2.extend(
                    (
                        torch.linalg.vector_norm(prefix - a, dim=1)
                        / torch.clamp(torch.linalg.vector_norm(a, dim=1), min=1.0e-12)
                    ).tolist()
                )
                b_chunks.append(b)
                c_chunks.append(c)
                a_chunks.append(a)
                for row in batch:
                    metadata_rows.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "case_id": str(row["case_id"]),
                            "label": int(row["label"]),
                            "split": "train",
                            "center": center,
                            "contract_row_index": int(row["row_index"]),
                        }
                    )
            finally:
                for image in (*jpeg_images, *c_images):
                    image.close()
        import torch  # type: ignore

        b_tensor = torch.cat(tuple(b_chunks), dim=0)
        c_tensor = torch.cat(tuple(c_chunks), dim=0)
        a_tensor = torch.cat(tuple(a_chunks), dim=0)
        if int(b_tensor.shape[1]) != 3840 or int(c_tensor.shape[1]) != 11520:
            raise ValueError("B/C cache dimension drift.")
        write_center_shard(
            config.b_cache_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=b_tensor,
            canonical_a_embeddings=a_tensor,
            metadata=metadata_rows,
            feature_extractor=_extractor_payload(
                config,
                "jpeg_center_b",
                3840,
                extractor.identity,
            ),
        )
        write_center_shard(
            config.c_cache_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=c_tensor,
            metadata=metadata_rows,
            feature_extractor=_extractor_payload(
                config,
                "physical_multiscale_center_c",
                11520,
                extractor.identity,
            ),
        )
    bridge = {
        "schema_version": "midogpp_virchow2_jpeg_bridge_v1",
        "status": "PASS",
        "row_count": len(contract_rows),
        "minimum_cosine": min(bridge_cosines),
        "maximum_relative_l2": max(bridge_relative_l2),
        "required_minimum_cosine": config.bridge_minimum_cosine,
        "required_maximum_relative_l2": config.bridge_maximum_relative_l2,
    }
    if bridge["minimum_cosine"] < bridge["required_minimum_cosine"] or bridge[
        "maximum_relative_l2"
    ] > bridge["required_maximum_relative_l2"]:
        raise ValueError(f"Canonical JPEG extractor bridge failed: {bridge}")
    bridge["task_semantic_bridge"] = evaluate_jpeg_task_bridge(
        config.b_cache_root,
        config.canonical_reference_root,
        minimum_prediction_agreement=config.bridge_minimum_prediction_agreement,
        maximum_equal_center_bacc_delta=(
            config.bridge_maximum_equal_center_bacc_delta
        ),
    )
    _write_cache_reports(
        config,
        contract_rows,
        canonical.cache_sha256,
        bridge,
        extractor.identity,
    )
    return config.b_cache_root, config.c_cache_root


def _extractor_payload(
    config: PhysicalMultiscaleBuildConfig,
    representation_id: str,
    dimension: int,
    model_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_physical_multiscale_feature_extractor_v1",
        "representation_id": representation_id,
        "feature_dim": dimension,
        "model_ref": config.model_ref,
        "model_revision": config.model_revision,
        "model_identity": dict(model_identity),
        "experiment_seed": config.experiment_seed,
        "pooling": "cls_global_center_rows6to9_cols6to9",
        "register_tokens_excluded": True,
        "target_labels_used_for_extractor_fitting": False,
    }


def _write_cache_reports(
    config: PhysicalMultiscaleBuildConfig,
    rows: Sequence[Mapping[str, object]],
    canonical_hash: str,
    bridge: Mapping[str, object],
    model_identity: Mapping[str, object],
) -> None:
    contract_payload = json.loads(
        (
            config.contract_root / "physical_multiscale_contract.json"
        ).read_text(encoding="utf-8")
    )
    alignment = {
        "schema_version": "midogpp_physical_multiscale_cache_alignment_v1",
        "status": "PASS",
        "row_count": len(rows),
        "sample_id_order_hash": stable_hash([row["sample_id"] for row in rows]),
        "eligible_centers": list(config.eligible_centers),
        "center_4_present": False,
        "canonical_a_cache_sha256": canonical_hash,
    }
    for root, representation_id, dimension in (
        (config.b_cache_root, "jpeg_center_b", 3840),
        (config.c_cache_root, "physical_multiscale_center_c", 11520),
    ):
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        _write_json(root / "manifests" / "row_alignment.json", alignment)
        _write_json(
            root / "reports" / "cache_builder_report.json",
            {
                "schema_version": "midogpp_physical_multiscale_cache_builder_v1",
                "status": "PASS",
                "representation_id": representation_id,
                "feature_dim": dimension,
                "row_count": len(rows),
                "model_ref": config.model_ref,
                "model_revision": config.model_revision,
                "model_identity": dict(model_identity),
                "physical_contract_hash": contract_payload["contract_hash"],
                "canonical_a_cache_sha256": canonical_hash,
                "bridge": dict(bridge),
            },
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _assert_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite immutable cache root: {path}")


def _assert_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable cache path already exists: {path}")


def _quarantine_staging(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    failed = path.with_name(f"{path.name}.failed-{stamp}-{os.getpid()}")
    path.rename(failed)


def _repo_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_extraction(seed: int) -> None:
    """Freeze all available inference RNGs before model construction."""

    import random

    import numpy as np  # type: ignore
    import torch  # type: ignore

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
