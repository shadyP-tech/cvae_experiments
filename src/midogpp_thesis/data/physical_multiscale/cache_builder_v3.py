"""Workstation-only atomic B/C cache publication for clipped-bbox v3."""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from PIL import Image

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.staged_directory import staged_directory
from midogpp_thesis.data.contract.paths import resolve_contract_path
from midogpp_thesis.data.features.cache_io import load_cache_rows, write_center_shard
from midogpp_thesis.data.features.virchow2 import Virchow2TokenExtractor
from midogpp_thesis.data.features.virchow2_tokens import (
    VIRCHOW2_TOKEN_LAYOUT,
    assert_preprocessing_spatial_identity,
    normalized_position_to_window_start,
)

from .bridge import evaluate_jpeg_task_bridge
from .config_v3 import (
    ANNOTATION_JPEG_DECODER,
    B_FEATURE_DIM,
    B_INPUT_DECODER,
    B_REPRESENTATION_ID,
    C_FEATURE_DIM,
    C_INPUT_DECODER,
    C_REPRESENTATION_ID,
    PROFILE_ID,
    PhysicalMultiscaleV3BuildConfig,
)
from .contract_inputs import read_csv
from .contract_validation_v3 import validate_contract_bundle_v3
from .slide_reader import open_slide


def build_physical_multiscale_caches_v3(
    config: PhysicalMultiscaleV3BuildConfig,
) -> Path:
    """Build, independently validate, and publish B/C through one rename."""

    from .cache_validation_v3 import validate_cache_bundle_v3

    validate_contract_bundle_v3(
        config.contract_root,
        verify_raw_files=True,
        expected_config=config,
    )
    config.cache_bundle_root.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(config.cache_bundle_root) as stage:
        staging = replace(config, cache_bundle_root=stage)
        _build_cache_bundle_v3_in_place(staging)
        validate_cache_bundle_v3(
            stage,
            contract_root=config.contract_root,
            canonical_cache_path=config.canonical_cache_path,
            canonical_reference_root=config.canonical_reference_root,
            expected_config=config,
        )
    return config.cache_bundle_root


def _build_cache_bundle_v3_in_place(
    config: PhysicalMultiscaleV3BuildConfig,
) -> None:
    if any(config.cache_bundle_root.iterdir()):
        raise FileExistsError(
            f"Physical multiscale v3 staging root is not empty: "
            f"{config.cache_bundle_root}"
        )
    _seed_extraction(config.experiment_seed)
    contract_rows = read_csv(
        config.contract_root / "physical_multiscale_manifest.csv"
    )
    base_rows = read_csv(config.base_manifest_path)
    base_by_id = {str(row["sample_id"]): row for row in base_rows}
    canonical = load_cache_rows(config.canonical_cache_path, expected_dim=2560)
    canonical_by_id = {
        sample_id: canonical.embeddings[index]
        for index, sample_id in enumerate(canonical.sample_ids)
    }
    expected_ids = tuple(str(row["sample_id"]) for row in contract_rows)
    expected_id_set = set(expected_ids)
    if tuple(
        sample_id for sample_id in canonical.sample_ids if sample_id in expected_id_set
    ) != expected_ids:
        raise ValueError("Physical multiscale v3 contract/canonical order differs.")
    extractor = Virchow2TokenExtractor(
        model_ref=config.model_ref,
        model_revision=config.model_revision,
        device=config.device,
        expected_model_config_sha256=config.expected_model_config_sha256,
        expected_checkpoint_file_sha256=config.expected_checkpoint_file_sha256,
        expected_state_dict_sha256=config.expected_state_dict_sha256,
        expected_preprocessing_config_hash=config.expected_preprocessing_config_hash,
    )
    preprocessing = extractor.identity.get("preprocessing_config")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("Virchow2 v3 preprocessing identity is missing.")
    spatial_identity = assert_preprocessing_spatial_identity(preprocessing)
    runtime_identity = _runtime_identity()
    expected_runtime = {
        "timm": config.expected_timm_version,
        "torch": config.expected_torch_version,
        "pillow": config.expected_pillow_version,
        "pyvips": config.expected_pyvips_version,
        "libvips": config.expected_libvips_version,
    }
    if runtime_identity != expected_runtime:
        raise ValueError(
            f"Physical multiscale v3 runtime identity drift: "
            f"expected={expected_runtime}, actual={runtime_identity}"
        )
    grouped = {center: [] for center in config.eligible_centers}
    for row in contract_rows:
        center = str(row["center"])
        if center not in grouped:
            raise ValueError(f"Physical multiscale v3 has unexpected center: {center}")
        grouped[center].append(row)

    bridge_cosines: list[float] = []
    bridge_relative_l2: list[float] = []
    pooling_audit_rows: list[dict[str, object]] = []
    for center in config.eligible_centers:
        rows = grouped[center]
        b_chunks = []
        c_chunks = []
        a_chunks = []
        metadata_rows: list[dict[str, object]] = []
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start : start + config.batch_size]
            jpeg_images: list[Image.Image] = []
            c_images_by_scale: list[list[Image.Image]] = [[], [], []]
            c_starts_by_scale: list[list[tuple[int, int]]] = [[], [], []]
            for row in batch:
                sample_id = str(row["sample_id"])
                base = base_by_id.get(sample_id)
                if base is None:
                    raise ValueError(
                        f"Physical multiscale v3 sample missing from base manifest: "
                        f"{sample_id}"
                    )
                jpeg_path = resolve_contract_path(
                    config.repo_root,
                    Path(str(base["image_path"])),
                )
                jpeg_images.append(Image.open(jpeg_path).convert("RGB"))
                geometries = json.loads(str(row["scale_geometry_json"]))
                with open_slide(
                    _repo_path(config.repo_root, row["raw_tiff_path"]),
                    require_tiled=config.require_tiled_reader,
                    required_backend=config.required_slide_reader_backend,
                ) as slide:
                    for scale_index, key in enumerate(("28um", "56um", "112um")):
                        geometry = geometries[key]
                        image = slide.read_exact_square(
                            geometry,
                            output_size=config.output_size_px,
                        )
                        c_images_by_scale[scale_index].append(image)
                        token_start = normalized_position_to_window_start(
                            x=float(geometry["p_x"]),
                            y=float(geometry["p_y"]),
                        )
                        expected_start = (
                            int(geometry["token_start_row"]),
                            int(geometry["token_start_col"]),
                        )
                        if token_start != expected_start:
                            raise ValueError(
                                f"Physical multiscale v3 contract token start drift: "
                                f"{sample_id} {key}."
                            )
                        c_starts_by_scale[scale_index].append(token_start)
                        pooling_audit_rows.append(
                            {
                                "sample_id": sample_id,
                                "contract_row_index": int(row["row_index"]),
                                "center": center,
                                "label": int(row["label"]),
                                "fov_um": float(geometry["fov_um"]),
                                "annotation_anchor_policy_id": row["policy_id"],
                                "anchor_x": float(row["anchor_x"]),
                                "anchor_y": float(row["anchor_y"]),
                                "p_x": float(geometry["p_x"]),
                                "p_y": float(geometry["p_y"]),
                                "token_start_row": token_start[0],
                                "token_start_col": token_start[1],
                                "shift_x": int(geometry["shift_x"]),
                                "shift_y": int(geometry["shift_y"]),
                            }
                        )
            try:
                b = extractor.extract_spatial_windows(
                    jpeg_images,
                    window_starts=[(6, 6)] * len(jpeg_images),
                )
                scale_blocks = [
                    extractor.extract_spatial_windows(
                        c_images_by_scale[index],
                        window_starts=c_starts_by_scale[index],
                    )
                    for index in range(3)
                ]
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
                        / torch.clamp(
                            torch.linalg.vector_norm(a, dim=1),
                            min=1.0e-12,
                        )
                    ).tolist()
                )
                b_chunks.append(b)
                c_chunks.append(c)
                a_chunks.append(a)
                metadata_rows.extend(
                    {
                        "sample_id": str(row["sample_id"]),
                        "case_id": str(row["case_id"]),
                        "label": int(row["label"]),
                        "split": "train",
                        "center": center,
                        "contract_row_index": int(row["row_index"]),
                    }
                    for row in batch
                )
            finally:
                for image in jpeg_images:
                    image.close()
                for images in c_images_by_scale:
                    for image in images:
                        image.close()
        import torch  # type: ignore

        b_tensor = torch.cat(tuple(b_chunks), dim=0)
        c_tensor = torch.cat(tuple(c_chunks), dim=0)
        a_tensor = torch.cat(tuple(a_chunks), dim=0)
        if (
            int(b_tensor.shape[1]) != B_FEATURE_DIM
            or int(c_tensor.shape[1]) != C_FEATURE_DIM
        ):
            raise ValueError("Physical multiscale v3 B/C dimension drift.")
        write_center_shard(
            config.b_cache_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=b_tensor,
            canonical_a_embeddings=a_tensor,
            metadata=metadata_rows,
            feature_extractor=_extractor_payload(
                config,
                B_REPRESENTATION_ID,
                B_FEATURE_DIM,
                extractor.identity,
                runtime_identity,
                spatial_identity,
                pooling="fixed_center_rows6to9_cols6to9",
            ),
        )
        write_center_shard(
            config.c_cache_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=c_tensor,
            metadata=metadata_rows,
            feature_extractor=_extractor_payload(
                config,
                C_REPRESENTATION_ID,
                C_FEATURE_DIM,
                extractor.identity,
                runtime_identity,
                spatial_identity,
                pooling="annotation_local_start_clamp_floor_16p_minus2_window4",
            ),
        )
    bridge = {
        "schema_version": "midogpp_virchow2_jpeg_bridge_v3",
        "status": "PASS",
        "row_count": len(contract_rows),
        "minimum_cosine": min(bridge_cosines),
        "maximum_relative_l2": max(bridge_relative_l2),
        "required_minimum_cosine": config.bridge_minimum_cosine,
        "required_maximum_relative_l2": config.bridge_maximum_relative_l2,
        "official_forward_features_prefix_used": True,
    }
    if (
        bridge["minimum_cosine"] < bridge["required_minimum_cosine"]
        or bridge["maximum_relative_l2"] > bridge["required_maximum_relative_l2"]
    ):
        raise ValueError(f"Physical multiscale v3 canonical-A bridge failed: {bridge}")
    bridge["task_semantic_bridge"] = evaluate_jpeg_task_bridge(
        config.b_cache_root,
        config.canonical_reference_root,
        minimum_prediction_agreement=config.bridge_minimum_prediction_agreement,
        maximum_equal_center_bacc_delta=config.bridge_maximum_equal_center_bacc_delta,
    )
    _write_cache_reports(
        config,
        contract_rows,
        pooling_audit_rows,
        canonical.cache_sha256,
        bridge,
        extractor.identity,
        runtime_identity,
        spatial_identity,
    )


def _extractor_payload(
    config: PhysicalMultiscaleV3BuildConfig,
    representation_id: str,
    dimension: int,
    model_identity: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    spatial_identity: Mapping[str, object],
    *,
    pooling: str,
) -> dict[str, object]:
    pooling_policy = {
        "representation_id": representation_id,
        "operator_order": ["cls", "global_patch_mean", "local_4x4_patch_mean"],
        "patch_grid": [16, 16],
        "patch_order": "row_major",
        "patch_token_start": 5,
        "register_token_count": 4,
        "register_tokens_excluded": True,
        "window_side": 4,
        "pooling": pooling,
        "patch_pool": "uniform_arithmetic_mean_16_tokens",
        "combination": "feature_concatenation_not_mixture",
    }
    return {
        "schema_version": "midogpp_physical_multiscale_feature_extractor_v3",
        "profile_id": PROFILE_ID,
        "representation_id": representation_id,
        "feature_dim": dimension,
        "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
        "model_ref": config.model_ref,
        "model_revision": config.model_revision,
        "model_identity": dict(model_identity),
        "runtime_identity": dict(runtime_identity),
        "input_decoder": _input_decoder(representation_id),
        "preprocessing_spatial_identity": dict(spatial_identity),
        "experiment_seed": config.experiment_seed,
        "forward_path": "forward_features",
        "token_layout": asdict(VIRCHOW2_TOKEN_LAYOUT),
        "pooling_policy": pooling_policy,
        "pooling_policy_hash": stable_hash(pooling_policy),
        "target_labels_used_for_extractor_fitting": False,
        "uses_experts": False,
        "performs_expert_aggregation": False,
        "uses_mixture_model": False,
    }


def _write_cache_reports(
    config: PhysicalMultiscaleV3BuildConfig,
    rows: Sequence[Mapping[str, object]],
    pooling_rows: Sequence[Mapping[str, object]],
    canonical_hash: str,
    bridge: Mapping[str, object],
    model_identity: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    spatial_identity: Mapping[str, object],
) -> None:
    contract = json.loads(
        (config.contract_root / "physical_multiscale_contract.json").read_text(
            encoding="utf-8"
        )
    )
    sample_order_hash = stable_hash([row["sample_id"] for row in rows])
    alignment = {
        "schema_version": "midogpp_physical_multiscale_cache_alignment_v3",
        "status": "PASS",
        "profile_id": PROFILE_ID,
        "row_count": len(rows),
        "sample_id_order_hash": sample_order_hash,
        "eligible_centers": list(config.eligible_centers),
        "center_4_present": False,
        "canonical_a_cache_sha256": canonical_hash,
        "physical_contract_hash": contract["contract_hash"],
        "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
    }
    for root, representation_id, dimension, pooling in (
        (
            config.b_cache_root,
            B_REPRESENTATION_ID,
            B_FEATURE_DIM,
            "fixed_center_rows6to9_cols6to9",
        ),
        (
            config.c_cache_root,
            C_REPRESENTATION_ID,
            C_FEATURE_DIM,
            "annotation_local_start_clamp_floor_16p_minus2_window4",
        ),
    ):
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        _write_json(root / "manifests" / "row_alignment.json", alignment)
        _write_json(
            root / "reports" / "cache_builder_report.json",
            {
                "schema_version": "midogpp_physical_multiscale_cache_builder_v3",
                "status": "PASS",
                "profile_id": PROFILE_ID,
                "representation_id": representation_id,
                "feature_dim": dimension,
                "row_count": len(rows),
                "sample_id_order_hash": sample_order_hash,
                "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
                "model_ref": config.model_ref,
                "model_revision": config.model_revision,
                "model_identity": dict(model_identity),
                "runtime_identity": dict(runtime_identity),
                "input_decoder": _input_decoder(representation_id),
                "preprocessing_spatial_identity": dict(spatial_identity),
                "pooling": pooling,
                "physical_contract_hash": contract["contract_hash"],
                "canonical_a_cache_sha256": canonical_hash,
                "bridge": dict(bridge),
            },
        )
    (config.cache_bundle_root / "manifests").mkdir(parents=True, exist_ok=True)
    (config.cache_bundle_root / "reports").mkdir(parents=True, exist_ok=True)
    _write_csv(
        config.cache_bundle_root / "manifests" / "pooling_audit.csv",
        pooling_rows,
    )
    bundle_manifest = {
        "schema_version": "midogpp_physical_multiscale_cache_bundle_v3",
        "status": "PASS",
        "profile_id": PROFILE_ID,
        "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
        "physical_contract_hash": contract["contract_hash"],
        "canonical_a_cache_sha256": canonical_hash,
        "row_count": len(rows),
        "sample_id_order_hash": sample_order_hash,
        "representations": [
            {
                "representation_id": B_REPRESENTATION_ID,
                "relative_root": "b_3840",
                "feature_dim": B_FEATURE_DIM,
            },
            {
                "representation_id": C_REPRESENTATION_ID,
                "relative_root": "c_11520",
                "feature_dim": C_FEATURE_DIM,
            },
        ],
        "c_scale_order_um": [28.0, 56.0, 112.0],
        "representation_c_combination": "feature_concatenation_not_mixture",
        "patch_pool": "uniform_arithmetic_mean_16_tokens",
        "annotation_jpeg_decoder": ANNOTATION_JPEG_DECODER,
        "raw_tiff_slide_reader_backend": config.required_slide_reader_backend,
        "feature_extraction_stochastic": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_likelihood": False,
        "uses_nelbo": False,
        "uses_mixture_model": False,
        "uses_experts": False,
        "performs_expert_aggregation": False,
        "uses_generative_sampling": False,
    }
    _write_json(
        config.cache_bundle_root / "manifests" / "bundle_manifest.json",
        bundle_manifest,
    )
    _write_json(
        config.cache_bundle_root / "reports" / "cache_bundle_report.json",
        {
            **bundle_manifest,
            "schema_version": "midogpp_physical_multiscale_cache_bundle_report_v3",
            "pooling_audit_row_count": len(pooling_rows),
            "pooling_audit_hash": stable_hash(
                [
                    {str(key): str(value) for key, value in row.items()}
                    for row in pooling_rows
                ]
            ),
            "model_identity": dict(model_identity),
            "runtime_identity": dict(runtime_identity),
            "preprocessing_spatial_identity": dict(spatial_identity),
            "bridge": dict(bridge),
        },
    )
    _write_json(
        config.cache_bundle_root / "manifests" / "content_index.json",
        {
            "schema_version": "midogpp_physical_multiscale_content_index_v3",
            "status": "PASS",
            "annotation_anchor_policy_id": config.annotation_anchor_policy_id,
            "physical_contract_hash": contract["contract_hash"],
            "files": {
                str(path.relative_to(config.cache_bundle_root)): _sha256(path)
                for path in sorted(config.cache_bundle_root.rglob("*"))
                if path.is_file() and path.name != "content_index.json"
            },
        },
    )


def _input_decoder(representation_id: str) -> str:
    if representation_id == B_REPRESENTATION_ID:
        return B_INPUT_DECODER
    if representation_id == C_REPRESENTATION_ID:
        return C_INPUT_DECODER
    raise ValueError(
        f"Unsupported physical multiscale v3 representation: {representation_id}."
    )


def _runtime_identity() -> dict[str, str]:
    import PIL  # type: ignore
    import pyvips  # type: ignore
    import timm  # type: ignore
    import torch  # type: ignore

    return {
        "timm": str(timm.__version__),
        "torch": str(torch.__version__),
        "pillow": str(PIL.__version__),
        "pyvips": str(pyvips.__version__),
        "libvips": ".".join(str(pyvips.version(index)) for index in range(3)),
    }


def _repo_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_extraction(seed: int) -> None:
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
