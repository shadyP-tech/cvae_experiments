"""Dual-GPU central-token extraction and immutable B-spatial cache assembly."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.contract.paths import resolve_contract_path
from midogpp_thesis.data.features.cache_io import load_cache_rows
from midogpp_thesis.data.features.virchow2_tokens import pool_central_quadrants

from ..protocol import ProtocolError
from .config import (
    CANONICAL_DIM, EXPECTED_ROWS, GLOBAL_DIM, SPATIAL_DIM, TOKEN_GRID_SIDE,
    TOKEN_WIDTH, SpatialCacheConfig,
)

def assemble_b_spatial_features(canonical_b: Any, central_tokens: Any) -> Any:
    """Return ``[CLS+global, TL, TR, BL, BR]`` as float32."""

    import torch
    canonical = torch.as_tensor(canonical_b)
    tokens = torch.as_tensor(central_tokens)
    if canonical.ndim != 2 or tuple(canonical.shape[1:]) != (CANONICAL_DIM,):
        raise ValueError(f"Canonical B must be [n,{CANONICAL_DIM}], got {tuple(canonical.shape)}.")
    if tokens.ndim != 4 or tuple(tokens.shape[1:]) != (TOKEN_GRID_SIDE, TOKEN_GRID_SIDE, TOKEN_WIDTH):
        raise ValueError(
            "Central tokens must be "
            f"[n,{TOKEN_GRID_SIDE},{TOKEN_GRID_SIDE},{TOKEN_WIDTH}], got {tuple(tokens.shape)}."
        )
    if int(canonical.shape[0]) != int(tokens.shape[0]) or int(tokens.shape[0]) == 0:
        raise ValueError("Canonical B and central tokens must have equal nonzero rows.")
    quadrants = pool_central_quadrants(tokens.float())
    spatial = torch.cat((canonical[:, :GLOBAL_DIM].float(), quadrants), dim=1)
    if tuple(spatial.shape) != (int(canonical.shape[0]), SPATIAL_DIM):
        raise RuntimeError(f"B-spatial dimension drift: {tuple(spatial.shape)}.")
    return spatial.contiguous()

def build_uniform_b_spatial_cache(config: SpatialCacheConfig) -> Path:
    config.root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(config.root) as stage:
        _build_in_place(config, stage)
        pending = validate_uniform_b_spatial_cache(stage, config=config, allow_pending=True)
        builder = _read_json(stage / "reports/cache_builder_report.json")
        builder["status"] = "PASS"
        builder["independent_validation_status"] = "PASS"
        _write_json(stage / "reports/cache_builder_report.json", builder)
        _write_json(stage / "reports/validation_report.json", {
            "schema_version": "midogpp_uniform_b_spatial_cache_validation_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_spatial_cache",
            "checks": pending,
        })
        _write_content_index(stage)
        validate_uniform_b_spatial_cache(stage, config=config)
    return config.root

def _build_in_place(config: SpatialCacheConfig, root: Path) -> None:
    import torch
    started = time.perf_counter()
    rows = _eligible_manifest_rows(config)
    canonical = load_cache_rows(config.canonical_b_cache_path, expected_dim=CANONICAL_DIM)
    canonical_by_id = {sample_id: canonical.embeddings[i] for i, sample_id in enumerate(canonical.sample_ids)}
    expected_ids = tuple(str(row["sample_id"]) for row in rows)
    if tuple(canonical.sample_ids) != expected_ids:
        raise ProtocolError("Canonical B and B-spatial manifest order differ.")

    worker_root = root / "worker_shards"
    worker_root.mkdir(parents=True)
    assignments = _balanced_center_assignments(rows, config.devices)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(config.devices), mp_context=context) as pool:
        futures = [
            pool.submit(_extract_worker, index, device, assigned, config, worker_root)
            for index, (device, assigned) in enumerate(assignments)
        ]
        worker_paths = [Path(future.result()) for future in futures]

    token_by_id: dict[str, Any] = {}
    identities = []
    for worker_path in worker_paths:
        payload = torch.load(worker_path, map_location="cpu", weights_only=True)
        identities.append(payload["model_identity"])
        for index, sample_id in enumerate(payload["sample_ids"]):
            if sample_id in token_by_id:
                raise ProtocolError(f"Duplicate B-spatial token row: {sample_id}.")
            token_by_id[str(sample_id)] = payload["central_tokens"][index]
    if len(token_by_id) != config.expected_rows or any(identity != identities[0] for identity in identities[1:]):
        raise ProtocolError("B-spatial worker coverage or model identity drifted.")

    tokens = torch.stack(tuple(token_by_id[sample_id] for sample_id in expected_ids)).half()
    canonical_matrix = torch.stack(tuple(torch.as_tensor(canonical_by_id[sample_id]) for sample_id in expected_ids)).float()
    spatial = assemble_b_spatial_features(canonical_matrix, tokens)
    metadata = [
        {
            "sample_id": str(row["sample_id"]), "case_id": str(row["case_id"]),
            "label": int(row["label"]), "split": "train", "center": str(row["center"]),
            "contract_row_index": int(row["contract_row_index"]),
        }
        for row in rows
    ]
    local = tokens.float().mean(dim=(1, 2))
    canonical_local = canonical_matrix[:, GLOBAL_DIM:]
    cosine = torch.nn.functional.cosine_similarity(local, canonical_local, dim=1)
    relative = torch.linalg.vector_norm(local - canonical_local, dim=1) / torch.clamp(
        torch.linalg.vector_norm(canonical_local, dim=1), min=1.0e-12
    )
    bridge = {
        "minimum_cosine": float(cosine.min()), "maximum_relative_l2": float(relative.max()),
        "required_minimum_cosine": 0.99999, "required_maximum_relative_l2": 0.001,
    }
    bridge["status"] = "PASS" if bridge["minimum_cosine"] >= bridge["required_minimum_cosine"] and bridge["maximum_relative_l2"] <= bridge["required_maximum_relative_l2"] else "FAIL"
    if bridge["status"] != "PASS":
        raise ProtocolError(f"Float16 central-token bridge to canonical B failed: {bridge}.")

    frozen = {
        "schema_version": "midogpp_uniform_b_spatial_cache_protocol_v1",
        "cache_name": config.name, "representation_id": "annotation_jpeg_fixed_center_b_spatial_quadrants_v1",
        "row_count": config.expected_rows, "token_shape": [4, 4, TOKEN_WIDTH],
        "spatial_feature_dim": SPATIAL_DIM, "quadrant_order": ["TL", "TR", "BL", "BR"],
        "central_window": "rows6to9_cols6to9", "token_storage_dtype": config.token_storage_dtype,
        "extraction_precision": config.extraction_precision, "devices": list(config.devices),
        "batch_size_per_device": config.batch_size_per_device,
        "hf_hub_cache_path": str(config.hf_hub_cache_path),
        "hf_hub_local_files_only": config.hf_hub_local_files_only,
        "canonical_b_cache_sha256": canonical.cache_sha256,
        "labels_used_for_feature_construction": False, "validation_rows_present": False, "test_rows_present": False,
        "model_identity": identities[0],
    }
    frozen["protocol_hash"] = stable_hash(frozen)
    (root / "tokens").mkdir(); (root / "embeddings").mkdir(); (root / "manifests").mkdir(); (root / "reports").mkdir()
    extractor = {
        "schema_version": "midogpp_uniform_b_spatial_feature_extractor_v1",
        "representation_id": "annotation_jpeg_fixed_center_b_spatial_quadrants_v1",
        "feature_dim": SPATIAL_DIM, "pooling": "global_prefix_plus_ordered_2x2_quadrant_means",
        "source_protocol_hash": frozen["protocol_hash"], "model_identity": identities[0],
    }
    torch.save({"central_tokens": tokens, "metadata": metadata, "feature_extractor": extractor}, root / "tokens/train.pt")
    torch.save({"embeddings": spatial, "metadata": metadata, "feature_extractor": extractor}, root / "embeddings/train.pt")
    _write_json(root / "manifests/frozen_cache_protocol.json", frozen)
    _write_json(root / "reports/cache_builder_report.json", {
        "schema_version": "midogpp_uniform_b_spatial_cache_builder_v1", "status": "PENDING_INDEPENDENT_VALIDATION",
        "row_count": config.expected_rows, "spatial_feature_dim": SPATIAL_DIM, "token_storage_dtype": "float16",
        "worker_processes": len(config.devices), "gpu_devices": list(config.devices),
        "hf_hub_cache_path": str(config.hf_hub_cache_path),
        "hf_hub_local_files_only": config.hf_hub_local_files_only,
        "gpu_shards": [
            {"device": device, "rows": len(assigned)}
            for device, assigned in assignments
        ],
        "elapsed_seconds": time.perf_counter() - started, "canonical_local_bridge": bridge,
    })
    for worker_path in worker_paths:
        worker_path.unlink()
    worker_root.rmdir()
    _write_content_index(root)

def _extract_worker(index: int, device: str, rows: Sequence[Mapping[str, object]], config: SpatialCacheConfig, worker_root: Path) -> str:
    import torch
    from PIL import Image
    from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor
    random.seed(config.experiment_seed + index); np.random.seed(config.experiment_seed + index); torch.manual_seed(config.experiment_seed + index)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":")[-1])); torch.cuda.manual_seed_all(config.experiment_seed + index)
    extractor = Virchow2TokenExtractor(
        model_ref=config.model_ref, model_revision=config.model_revision, device=device,
        expected_model_config_sha256=config.expected_model_config_sha256,
        expected_checkpoint_file_sha256=config.expected_checkpoint_file_sha256,
        expected_state_dict_sha256=config.expected_state_dict_sha256,
        expected_preprocessing_config_hash=config.expected_preprocessing_config_hash,
        hf_hub_cache_path=config.hf_hub_cache_path,
        hf_hub_local_files_only=config.hf_hub_local_files_only,
    )
    chunks = []; sample_ids = []
    for start in range(0, len(rows), config.batch_size_per_device):
        batch = rows[start:start + config.batch_size_per_device]; images = []
        try:
            for row in batch:
                path = resolve_contract_path(config.repo_root, str(row["image_path"]))
                images.append(Image.open(path).convert("RGB"))
            chunks.append(extractor.extract_central_token_grid(images).half())
            sample_ids.extend(str(row["sample_id"]) for row in batch)
        finally:
            for image in images: image.close()
    output = worker_root / f"worker_{index}.pt"
    torch.save({"central_tokens": torch.cat(chunks), "sample_ids": sample_ids, "model_identity": extractor.identity}, output)
    return str(output)

def _balanced_center_assignments(rows: Sequence[Mapping[str, object]], devices: Sequence[str]) -> list[tuple[str, list[Mapping[str, object]]]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows: groups.setdefault(str(row["center"]), []).append(row)
    bins = [{"device": device, "rows": [], "n": 0} for device in devices]
    for _, group in sorted(groups.items(), key=lambda item: (-len(item[1]), int(item[0]))):
        target = min(bins, key=lambda item: (int(item["n"]), str(item["device"])))
        target["rows"].extend(group); target["n"] = int(target["n"]) + len(group)
    return [(str(item["device"]), list(item["rows"])) for item in bins]

def validate_uniform_b_spatial_cache(root: str | Path, *, config: SpatialCacheConfig, allow_pending: bool = False) -> dict[str, object]:
    import torch
    path = Path(root)
    required = {"tokens/train.pt", "embeddings/train.pt", "manifests/frozen_cache_protocol.json", "manifests/content_index.json", "reports/cache_builder_report.json"}
    if not allow_pending: required.add("reports/validation_report.json")
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing: raise ProtocolError(f"Uniform-B spatial cache is incomplete: {missing}.")
    token_payload = torch.load(path / "tokens/train.pt", map_location="cpu", weights_only=True)
    feature = load_cache_rows(path / "embeddings/train.pt", expected_dim=SPATIAL_DIM)
    canonical = load_cache_rows(config.canonical_b_cache_path, expected_dim=CANONICAL_DIM)
    tokens = token_payload.get("central_tokens")
    metadata = token_payload.get("metadata")
    if (tokens is None or tuple(tokens.shape) != (EXPECTED_ROWS, 4, 4, TOKEN_WIDTH) or tokens.dtype != torch.float16
        or not isinstance(metadata, Sequence) or len(metadata) != EXPECTED_ROWS
        or feature.sample_ids != canonical.sample_ids or feature.sample_ids != tuple(str(row["sample_id"]) for row in metadata)):
        raise ProtocolError("Uniform-B spatial cache tensor or row identity drifted.")
    recomputed = assemble_b_spatial_features(canonical.embeddings, tokens)
    if not torch.equal(recomputed, torch.as_tensor(feature.embeddings)):
        raise ProtocolError("Uniform-B spatial features do not recompute exactly from tokens and canonical B.")
    frozen = _read_json(path / "manifests/frozen_cache_protocol.json")
    unhashed = {k: v for k, v in frozen.items() if k != "protocol_hash"}
    report = _read_json(path / "reports/cache_builder_report.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    if (stable_hash(unhashed) != frozen.get("protocol_hash") or frozen.get("row_count") != EXPECTED_ROWS
        or frozen.get("spatial_feature_dim") != SPATIAL_DIM or frozen.get("labels_used_for_feature_construction") is not False
        or frozen.get("validation_rows_present") is not False or frozen.get("test_rows_present") is not False
        or report.get("status") != expected_status or report.get("canonical_local_bridge", {}).get("status") != "PASS"):
        raise ProtocolError("Uniform-B spatial cache protocol validation failed.")
    _validate_content_index(path)
    checks = {"status": "PASS", "row_count": EXPECTED_ROWS, "token_shape": [4, 4, TOKEN_WIDTH], "feature_dim": SPATIAL_DIM, "numeric_recomputation": "EXACT", "gpu_workers": 2}
    if not allow_pending:
        validation = _read_json(path / "reports/validation_report.json")
        if validation.get("status") != "PASS" or validation.get("checks") != checks:
            raise ProtocolError("Uniform-B spatial cache validation report drifted.")
    return checks

def _eligible_manifest_rows(config: SpatialCacheConfig) -> list[dict[str, object]]:
    with config.manifest_path.open(newline="", encoding="utf-8") as handle:
        raw = [dict(row) for row in csv.DictReader(handle)]
    eligible = set(config.eligible_centers); rows = []
    for index, row in enumerate(raw):
        if str(row.get("split", "")).lower() == "train" and str(row.get("center")) in eligible:
            rows.append({**row, "contract_row_index": index})
    if len(rows) != config.expected_rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ProtocolError("Uniform-B spatial manifest coverage drifted.")
    return rows

def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text());
    if not isinstance(value, dict): raise ProtocolError(f"Expected JSON object: {path}.")
    return value

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def _write_content_index(root: Path) -> None:
    files = [{"path": str(path.relative_to(root)), "sha256": _sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "content_index.json"]
    payload = {"schema_version": "midogpp_uniform_b_spatial_cache_content_index_v1", "files": files}; payload["content_hash"] = stable_hash(payload)
    _write_json(root / "manifests/content_index.json", payload)

def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json"); unhashed = {k: v for k, v in payload.items() if k != "content_hash"}
    if stable_hash(unhashed) != payload.get("content_hash"): raise ProtocolError("Uniform-B spatial content hash drifted.")
    expected = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "content_index.json"}
    observed = set()
    for row in payload.get("files", []):
        member = root / str(row["path"]); observed.add(str(row["path"]))
        if not member.is_file() or _sha256_file(member) != row.get("sha256"): raise ProtocolError(f"Uniform-B spatial cache member drifted: {member}.")
    if observed != expected: raise ProtocolError("Uniform-B spatial cache index coverage drifted.")
