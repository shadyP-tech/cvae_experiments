"""Immutable B-only feature cache for the untouched MIDOG++ test split."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Mapping

from PIL import Image

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.contract.paths import resolve_contract_path
from midogpp_thesis.data.features.cache_io import load_cache_rows, write_center_shard
from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor
from midogpp_thesis.data.features.virchow2_tokens import assert_preprocessing_spatial_identity

from ..protocol import ProtocolError
from .config import (
    CANONICAL_A,
    EVALUATION_SPLIT,
    EXPECTED_RUNTIME,
    EXPECTED_TEST_ROWS_BY_CENTER,
    EXPECTED_TRAIN_ROWS,
    TRAIN_SPLIT,
    UNIFORM_B,
    UniformBTestCacheConfig,
)


CACHE_REQUIRED_FILES = (
    "manifests/frozen_build_protocol.json",
    "manifests/row_alignment.json",
    "manifests/content_index.json",
    "reports/cache_builder_report.json",
    "reports/validation_report.json",
)


def build_uniform_b_test_cache(config: UniformBTestCacheConfig) -> Path:
    config.cache_root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(config.cache_root) as stage:
        _build_in_place(replace(config, cache_root=stage))
        validate_uniform_b_test_cache(stage, expected_config=config, allow_pending=True)
        _finalize_cache(stage, config)
        validate_uniform_b_test_cache(stage, expected_config=config)
    return config.cache_root


def _build_in_place(config: UniformBTestCacheConfig) -> None:
    started = time.perf_counter()
    _seed(config.experiment_seed)
    manifest = _read_csv(config.manifest_path)
    manifest_index = {str(row["sample_id"]): index for index, row in enumerate(manifest)}
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    canonical_train = load_cache_rows(config.canonical_train_cache_path, expected_dim=2560)
    canonical_test = load_cache_rows(config.canonical_test_cache_path, expected_dim=2560)
    eligible = set(config.eligible_centers)
    train_rows = [
        row for row in canonical_train.metadata
        if str(row.get("split")) == TRAIN_SPLIT and str(row.get("center")) in eligible
    ]
    test_pairs = [
        (row, canonical_test.embeddings[index])
        for index, row in enumerate(canonical_test.metadata)
        if str(row.get("split")) == EVALUATION_SPLIT and str(row.get("center")) in eligible
    ]
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(test_pairs) != config.expected_test_rows:
        raise ProtocolError("Uniform-B prospective train/test row coverage drifted.")
    _validate_disjoint(train_rows, [row for row, _embedding in test_pairs])
    counts = {center: 0 for center in config.eligible_centers}
    for row, _embedding in test_pairs:
        counts[str(row["center"])] += 1
    if counts != EXPECTED_TEST_ROWS_BY_CENTER:
        raise ProtocolError("Uniform-B prospective test-center coverage drifted.")

    extractor = Virchow2TokenExtractor(
        model_ref=config.model_ref,
        model_revision=config.model_revision,
        device=config.device,
        expected_model_config_sha256=config.expected_model_config_sha256,
        expected_checkpoint_file_sha256=config.expected_checkpoint_file_sha256,
        expected_state_dict_sha256=config.expected_state_dict_sha256,
        expected_preprocessing_config_hash=config.expected_preprocessing_config_hash,
    )
    runtime = _runtime_identity()
    if runtime != dict(config.expected_runtime):
        raise ProtocolError(f"Uniform-B prospective runtime identity drift: {runtime}.")
    preprocessing = extractor.identity.get("preprocessing_config")
    if not isinstance(preprocessing, Mapping):
        raise ProtocolError("Uniform-B prospective preprocessing identity is missing.")
    spatial_identity = assert_preprocessing_spatial_identity(preprocessing)
    source_report_path = config.source_train_b_cache_root / "reports/cache_builder_report.json"
    source_report = _read_json(source_report_path)
    if (
        source_report.get("status") != "PASS"
        or source_report.get("representation_id") != UNIFORM_B
        or source_report.get("pooling") != "fixed_center_rows6to9_cols6to9"
        or stable_hash(source_report.get("model_identity"))
        != stable_hash(dict(extractor.identity))
    ):
        raise ProtocolError("Uniform-B prospective extractor differs from source B.")
    frozen = _frozen_build_protocol(
        config,
        runtime,
        extractor.identity,
        source_report_sha256=_sha256_file(source_report_path),
    )
    _write_json(config.cache_root / "manifests/frozen_build_protocol.json", frozen)

    grouped: dict[str, list[tuple[Mapping[str, object], object]]] = {
        center: [] for center in config.eligible_centers
    }
    for row, embedding in test_pairs:
        grouped[str(row["center"])].append((row, embedding))
    bridge_cosines: list[float] = []
    bridge_relative_l2: list[float] = []
    sample_order: list[str] = []
    for center in config.eligible_centers:
        chunks = []
        a_chunks = []
        metadata = []
        rows = grouped[center]
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start : start + config.batch_size]
            images = []
            try:
                for row, _a in batch:
                    manifest_row = manifest_by_id.get(str(row["sample_id"]))
                    if manifest_row is None:
                        raise ProtocolError("Uniform-B test sample is absent from the manifest.")
                    image_path = resolve_contract_path(
                        config.repo_root, Path(str(manifest_row["image_path"]))
                    )
                    if not image_path.is_file():
                        raise ProtocolError(f"Uniform-B JPEG is missing: {image_path}.")
                    images.append(Image.open(image_path).convert("RGB"))
                b = extractor.extract_spatial_windows(
                    images, window_starts=[(6, 6)] * len(images)
                )
                import torch

                a = torch.stack(tuple(item[1] for item in batch), dim=0).detach().cpu().float()
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
                chunks.append(b)
                a_chunks.append(a)
                for row, _embedding in batch:
                    sample_id = str(row["sample_id"])
                    sample_order.append(sample_id)
                    metadata.append(
                        {
                            "sample_id": sample_id,
                            "case_id": str(row["case_id"]),
                            "label": int(row["label"]),
                            "split": EVALUATION_SPLIT,
                            "center": center,
                            "contract_row_index": manifest_index[sample_id],
                        }
                    )
            finally:
                for image in images:
                    image.close()
        import torch

        write_center_shard(
            config.cache_root / "embeddings/by_center" / f"center_{center}.pt",
            embeddings=torch.cat(tuple(chunks), dim=0),
            canonical_a_embeddings=torch.cat(tuple(a_chunks), dim=0),
            metadata=metadata,
            feature_extractor={
                "schema_version": "midogpp_uniform_b_test_feature_extractor_v1",
                "representation_id": UNIFORM_B,
                "feature_dim": 3840,
                "pooling": "fixed_center_rows6to9_cols6to9",
                "model_identity": dict(extractor.identity),
                "runtime_identity": runtime,
                "preprocessing_spatial_identity": spatial_identity,
                "frozen_build_protocol_hash": frozen["frozen_build_protocol_hash"],
            },
        )
    minimum_cosine = min(bridge_cosines)
    maximum_relative_l2 = max(bridge_relative_l2)
    if minimum_cosine < 0.99999 or maximum_relative_l2 > 0.001:
        raise ProtocolError("Uniform-B prospective canonical-A numeric bridge failed.")
    order_hash = stable_hash(sample_order)
    _write_json(
        config.cache_root / "manifests/row_alignment.json",
        {
            "schema_version": "midogpp_uniform_b_test_row_alignment_v1",
            "status": "PASS",
            "split": EVALUATION_SPLIT,
            "row_count": len(sample_order),
            "rows_by_center": counts,
            "sample_id_order_hash": order_hash,
            "train_test_sample_overlap": 0,
            "train_test_case_overlap": 0,
            "center_4_present": False,
        },
    )
    _write_json(
        config.cache_root / "reports/cache_builder_report.json",
        {
            "schema_version": "midogpp_uniform_b_test_cache_builder_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "representation_id": UNIFORM_B,
            "split": EVALUATION_SPLIT,
            "row_count": len(sample_order),
            "feature_dim": 3840,
            "model_identity": dict(extractor.identity),
            "runtime_identity": runtime,
            "minimum_canonical_a_prefix_cosine": minimum_cosine,
            "maximum_canonical_a_prefix_relative_l2": maximum_relative_l2,
            "outcome_metric_computed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _write_content_index(config.cache_root)


def validate_uniform_b_test_cache(
    root: str | Path,
    *,
    expected_config: UniformBTestCacheConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(CACHE_REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    required.update(
        f"embeddings/by_center/center_{center}.pt"
        for center in expected_config.eligible_centers
    )
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B prospective test cache is incomplete: {missing}.")
    frozen = _read_json(path / "manifests/frozen_build_protocol.json")
    alignment = _read_json(path / "manifests/row_alignment.json")
    report = _read_json(path / "reports/cache_builder_report.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    unhashed = {key: value for key, value in frozen.items() if key != "frozen_build_protocol_hash"}
    if (
        stable_hash(unhashed) != frozen.get("frozen_build_protocol_hash")
        or frozen.get("representation_id") != UNIFORM_B
        or frozen.get("evaluation_split") != EVALUATION_SPLIT
        or frozen.get("test_outcomes_observed_before_lock") is not False
        or report.get("status") != expected_status
        or report.get("outcome_metric_computed") is not False
        or alignment.get("status") != "PASS"
        or alignment.get("row_count") != expected_config.expected_test_rows
        or alignment.get("rows_by_center") != EXPECTED_TEST_ROWS_BY_CENTER
        or alignment.get("train_test_sample_overlap") != 0
        or alignment.get("train_test_case_overlap") != 0
        or alignment.get("center_4_present") is not False
    ):
        raise ProtocolError("Uniform-B prospective cache protocol/alignment failed.")
    total = 0
    sample_ids: list[str] = []
    for center in expected_config.eligible_centers:
        shard = load_cache_rows(
            path / "embeddings/by_center" / f"center_{center}.pt", expected_dim=3840
        )
        try:
            import numpy as np
            import torch

            payload = torch.load(
                path / "embeddings/by_center" / f"center_{center}.pt",
                map_location="cpu",
                weights_only=True,
            )
            canonical = np.asarray(payload["canonical_a_embeddings"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Uniform-B test cache lacks embedded canonical A.") from exc
        if canonical.shape != (len(shard.metadata), 2560):
            raise ProtocolError("Uniform-B test cache embedded-A shape drifted.")
        if any(
            str(row.get("center")) != center or str(row.get("split")) != EVALUATION_SPLIT
            for row in shard.metadata
        ):
            raise ProtocolError("Uniform-B test-cache center/split metadata drifted.")
        total += len(shard.metadata)
        sample_ids.extend(shard.sample_ids)
    if total != expected_config.expected_test_rows or len(set(sample_ids)) != total:
        raise ProtocolError("Uniform-B prospective test-cache coverage is invalid.")
    _validate_content_index(path)
    checks = {"status": "PASS", "row_count": total, "center_count": 9, "split": "test"}
    if not allow_pending:
        validation = _read_json(path / "reports/validation_report.json")
        if validation.get("checks") != checks or validation.get("status") != "PASS":
            raise ProtocolError("Uniform-B prospective cache validation report failed.")
    return checks


def _frozen_build_protocol(
    config: UniformBTestCacheConfig,
    runtime: Mapping[str, str],
    model_identity: Mapping[str, object],
    *,
    source_report_sha256: str,
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_test_frozen_build_protocol_v1",
        "cache_name": config.name,
        "representation_id": UNIFORM_B,
        "canonical_comparator_id": CANONICAL_A,
        "training_split": TRAIN_SPLIT,
        "evaluation_split": EVALUATION_SPLIT,
        "eligible_centers": list(config.eligible_centers),
        "expected_test_rows": config.expected_test_rows,
        "expected_test_rows_by_center": EXPECTED_TEST_ROWS_BY_CENTER,
        "pooling": "fixed_center_rows6to9_cols6to9",
        "feature_dim": 3840,
        "model_identity": dict(model_identity),
        "runtime_identity": dict(runtime),
        "source_train_b_cache_report_sha256": source_report_sha256,
        "test_outcomes_observed_before_lock": False,
        "test_labels_used_for_feature_extraction": False,
        "validation_split_used": False,
    }
    payload["frozen_build_protocol_hash"] = stable_hash(payload)
    return payload


def _finalize_cache(root: Path, config: UniformBTestCacheConfig) -> None:
    checks = validate_uniform_b_test_cache(root, expected_config=config, allow_pending=True)
    report_path = root / "reports/cache_builder_report.json"
    report = _read_json(report_path)
    report["status"] = "PASS"
    report["independent_validation_status"] = "PASS"
    _write_json(report_path, report)
    _write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_test_cache_validation_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_test_cache",
            "checks": checks,
        },
    )
    _write_content_index(root)


def _validate_disjoint(
    train_rows: list[Mapping[str, object]], test_rows: list[Mapping[str, object]]
) -> None:
    train_samples = {str(row["sample_id"]) for row in train_rows}
    test_samples = {str(row["sample_id"]) for row in test_rows}
    train_cases = {str(row["case_id"]) for row in train_rows}
    test_cases = {str(row["case_id"]) for row in test_rows}
    if train_samples & test_samples or train_cases & test_cases:
        raise ProtocolError("Uniform-B prospective train/test identities overlap.")


def _runtime_identity() -> dict[str, str]:
    import PIL
    import timm
    import torch

    return {
        "timm": str(timm.__version__),
        "torch": str(torch.__version__),
        "pillow": str(PIL.__version__),
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Uniform-B JSON must be an object: {path}.")
    return payload


def _write_content_index(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    payload = {
        "schema_version": "midogpp_uniform_b_test_cache_content_index_v1",
        "files": rows,
    }
    payload["content_hash"] = stable_hash(payload)
    _write_json(root / "manifests/content_index.json", payload)


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if stable_hash(unhashed) != payload.get("content_hash"):
        raise ProtocolError("Uniform-B prospective cache content hash drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Uniform-B prospective cache content index is invalid.")
    expected = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Uniform-B prospective cache index row is invalid.")
        relative = str(row.get("path", ""))
        path = root / relative
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ProtocolError(f"Uniform-B prospective cache member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Uniform-B prospective cache index coverage drifted.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
