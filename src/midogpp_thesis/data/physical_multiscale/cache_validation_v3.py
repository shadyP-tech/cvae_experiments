"""Independent validation of the atomically published v3 B/C cache bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.data.features.cache_io import load_cache_rows
from midogpp_thesis.data.features.virchow2_tokens import (
    VIRCHOW2_TOKEN_LAYOUT,
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
from .contract_validation_v3 import validate_contract_bundle_v3


def validate_cache_bundle_v3(
    root: str | Path,
    *,
    contract_root: str | Path,
    canonical_cache_path: str | Path,
    canonical_reference_root: str | Path,
    expected_config: PhysicalMultiscaleV3BuildConfig,
) -> Mapping[str, object]:
    """Recompute cache, row, pooling, model, bridge, and content identities."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Physical cache v3 validation requires torch.") from exc
    bundle = Path(root)
    b_root = bundle / "b_3840"
    c_root = bundle / "c_11520"
    _require(
        bundle,
        (
            "manifests/bundle_manifest.json",
            "manifests/pooling_audit.csv",
            "manifests/content_index.json",
            "reports/cache_bundle_report.json",
        ),
    )
    contract_validation = validate_contract_bundle_v3(
        contract_root,
        verify_raw_files=False,
        expected_config=expected_config,
    )
    contract = _json(Path(contract_root) / "physical_multiscale_contract.json")
    contract_rows = _csv(Path(contract_root) / "physical_multiscale_manifest.csv")
    manifest = _json(bundle / "manifests" / "bundle_manifest.json")
    report = _json(bundle / "reports" / "cache_bundle_report.json")
    pooling_rows = _csv(bundle / "manifests" / "pooling_audit.csv")
    content_index = _json(bundle / "manifests" / "content_index.json")
    _validate_parent_decoders(
        manifest,
        report,
        expected_raw_tiff_backend=expected_config.required_slide_reader_backend,
    )
    expected_ids = tuple(row["sample_id"] for row in contract_rows)
    expected_id_set = set(expected_ids)
    canonical = load_cache_rows(canonical_cache_path, expected_dim=2560)
    canonical_ids = tuple(
        sample_id
        for sample_id, metadata in zip(
            canonical.sample_ids,
            canonical.metadata,
            strict=True,
        )
        if sample_id in expected_id_set
        and str(metadata.get("split", "")).lower() == "train"
        and str(metadata.get("center", "")) in MIDOGPP_ELIGIBLE_CENTERS
    )
    if canonical_ids != expected_ids:
        raise ValueError("Physical multiscale v3 canonical/contract order differs.")
    expected_runtime = {
        "timm": expected_config.expected_timm_version,
        "torch": expected_config.expected_torch_version,
        "pillow": expected_config.expected_pillow_version,
        "pyvips": expected_config.expected_pyvips_version,
        "libvips": expected_config.expected_libvips_version,
    }
    expected_representations = [
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
    ]
    if (
        manifest.get("status") != "PASS"
        or report.get("status") != "PASS"
        or manifest.get("profile_id") != PROFILE_ID
        or report.get("profile_id") != PROFILE_ID
        or manifest.get("annotation_anchor_policy_id")
        != expected_config.annotation_anchor_policy_id
        or report.get("annotation_anchor_policy_id")
        != expected_config.annotation_anchor_policy_id
        or manifest.get("physical_contract_hash") != contract["contract_hash"]
        or report.get("physical_contract_hash") != contract["contract_hash"]
        or manifest.get("canonical_a_cache_sha256") != canonical.cache_sha256
        or report.get("canonical_a_cache_sha256") != canonical.cache_sha256
        or manifest.get("representations") != expected_representations
        or manifest.get("c_scale_order_um") != [28.0, 56.0, 112.0]
        or manifest.get("representation_c_combination")
        != "feature_concatenation_not_mixture"
        or manifest.get("patch_pool") != "uniform_arithmetic_mean_16_tokens"
        or manifest.get("uses_mixture_model") is not False
        or manifest.get("uses_experts") is not False
        or manifest.get("performs_expert_aggregation") is not False
        or manifest.get("uses_nelbo") is not False
        or manifest.get("may_feed_recipe_selection") is not False
        or manifest.get("may_feed_deployable_selection") is not False
        or report.get("runtime_identity") != expected_runtime
    ):
        raise ValueError("Physical multiscale v3 bundle manifest/firewall drifted.")
    _validate_model_identity(report.get("model_identity"), expected_config)
    spatial_identity = report.get("preprocessing_spatial_identity")
    if not isinstance(spatial_identity, Mapping) or spatial_identity.get(
        "spatial_identity"
    ) is not True:
        raise ValueError("Physical multiscale v3 preprocessing is not spatially identity.")
    pooling_index = _validate_pooling_audit(
        pooling_rows,
        contract_rows,
        anchor_policy_id=expected_config.annotation_anchor_policy_id,
    )
    expected_sample_hash = stable_hash(expected_ids)
    observed_ids: list[str] = []
    minimum_cosine = 1.0
    maximum_relative_l2 = 0.0
    canonical_index = {
        sample_id: index for index, sample_id in enumerate(canonical.sample_ids)
    }
    extractor_hashes: dict[str, set[str]] = {
        B_REPRESENTATION_ID: set(),
        C_REPRESENTATION_ID: set(),
    }
    expected_center_grouped_ids = _center_grouped_sample_ids(contract_rows)
    for center in MIDOGPP_ELIGIBLE_CENTERS:
        expected_rows = [row for row in contract_rows if row["center"] == center]
        b_path = b_root / "embeddings" / "by_center" / f"center_{center}.pt"
        c_path = c_root / "embeddings" / "by_center" / f"center_{center}.pt"
        b_payload = _torch_payload(torch, b_path)
        c_payload = _torch_payload(torch, c_path)
        b_embeddings = torch.as_tensor(b_payload["embeddings"]).detach().cpu()
        c_embeddings = torch.as_tensor(c_payload["embeddings"]).detach().cpu()
        if tuple(b_embeddings.shape) != (len(expected_rows), B_FEATURE_DIM):
            raise ValueError(f"Physical multiscale v3 B shape drift: center {center}.")
        if tuple(c_embeddings.shape) != (len(expected_rows), C_FEATURE_DIM):
            raise ValueError(f"Physical multiscale v3 C shape drift: center {center}.")
        b_metadata = tuple(dict(row) for row in b_payload["metadata"])
        c_metadata = tuple(dict(row) for row in c_payload["metadata"])
        if b_metadata != c_metadata:
            raise ValueError(f"Physical multiscale v3 B/C metadata differ: center {center}.")
        expected_identity = tuple(
            (
                row["sample_id"],
                row["case_id"],
                int(row["label"]),
                row["split"],
                row["center"],
                int(row["row_index"]),
            )
            for row in expected_rows
        )
        actual_identity = tuple(
            (
                str(row.get("sample_id", "")),
                str(row.get("case_id", "")),
                int(row.get("label", -1)),
                str(row.get("split", "")),
                str(row.get("center", "")),
                int(row.get("contract_row_index", -1)),
            )
            for row in b_metadata
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"Physical multiscale v3 cache/contract identity differs: center {center}."
            )
        _validate_extractor(
            b_payload.get("feature_extractor"),
            representation_id=B_REPRESENTATION_ID,
            feature_dim=B_FEATURE_DIM,
            expected_config=expected_config,
            expected_runtime=expected_runtime,
        )
        _validate_extractor(
            c_payload.get("feature_extractor"),
            representation_id=C_REPRESENTATION_ID,
            feature_dim=C_FEATURE_DIM,
            expected_config=expected_config,
            expected_runtime=expected_runtime,
        )
        extractor_hashes[B_REPRESENTATION_ID].add(
            stable_hash(b_payload["feature_extractor"])
        )
        extractor_hashes[C_REPRESENTATION_ID].add(
            stable_hash(c_payload["feature_extractor"])
        )
        embedded = b_payload.get("canonical_a_embeddings")
        if embedded is None:
            raise ValueError(f"Physical multiscale v3 B lacks canonical A: center {center}.")
        canonical_rows = torch.stack(
            tuple(
                torch.as_tensor(canonical.embeddings[canonical_index[row["sample_id"]]])
                for row in expected_rows
            ),
            dim=0,
        ).detach().cpu()
        embedded_tensor = torch.as_tensor(embedded).detach().cpu()
        if not torch.equal(embedded_tensor, canonical_rows):
            raise ValueError(
                f"Physical multiscale v3 embedded canonical A differs: center {center}."
            )
        prefix = b_embeddings[:, :2560]
        cosines = torch.nn.functional.cosine_similarity(prefix, embedded_tensor, dim=1)
        relative_l2 = torch.linalg.vector_norm(
            prefix - embedded_tensor,
            dim=1,
        ) / torch.clamp(
            torch.linalg.vector_norm(embedded_tensor, dim=1),
            min=1.0e-12,
        )
        minimum_cosine = min(minimum_cosine, float(cosines.min().item()))
        maximum_relative_l2 = max(
            maximum_relative_l2,
            float(relative_l2.max().item()),
        )
        observed_ids.extend(row["sample_id"] for row in expected_rows)
        for row in expected_rows:
            for fov in ("28", "56", "112"):
                if (row["sample_id"], fov) not in pooling_index:
                    raise ValueError(
                        f"Physical multiscale v3 pooling audit is incomplete: "
                        f"{row['sample_id']} {fov}um."
                    )
    if (
        tuple(observed_ids) != expected_center_grouped_ids
        or any(len(values) != 1 for values in extractor_hashes.values())
        or minimum_cosine < expected_config.bridge_minimum_cosine
        or maximum_relative_l2 > expected_config.bridge_maximum_relative_l2
    ):
        raise ValueError("Physical multiscale v3 row/extractor/bridge validation failed.")
    for cache_root, representation_id, feature_dim in (
        (b_root, B_REPRESENTATION_ID, B_FEATURE_DIM),
        (c_root, C_REPRESENTATION_ID, C_FEATURE_DIM),
    ):
        alignment = _json(cache_root / "manifests" / "row_alignment.json")
        cache_report = _json(cache_root / "reports" / "cache_builder_report.json")
        _validate_child_input_decoder(
            cache_report,
            representation_id=representation_id,
        )
        bridge = cache_report.get("bridge")
        if (
            alignment.get("status") != "PASS"
            or alignment.get("profile_id") != PROFILE_ID
            or alignment.get("sample_id_order_hash") != expected_sample_hash
            or alignment.get("physical_contract_hash") != contract["contract_hash"]
            or alignment.get("canonical_a_cache_sha256") != canonical.cache_sha256
            or alignment.get("annotation_anchor_policy_id")
            != expected_config.annotation_anchor_policy_id
            or cache_report.get("representation_id") != representation_id
            or int(cache_report.get("feature_dim", -1)) != feature_dim
            or cache_report.get("physical_contract_hash") != contract["contract_hash"]
            or cache_report.get("runtime_identity") != expected_runtime
            or cache_report.get("annotation_anchor_policy_id")
            != expected_config.annotation_anchor_policy_id
            or not isinstance(bridge, Mapping)
            or bridge.get("status") != "PASS"
        ):
            raise ValueError(
                f"Physical multiscale v3 subcache report drifted: {representation_id}."
            )
    task_bridge = evaluate_jpeg_task_bridge(
        b_root,
        canonical_reference_root,
        minimum_prediction_agreement=expected_config.bridge_minimum_prediction_agreement,
        maximum_equal_center_bacc_delta=(
            expected_config.bridge_maximum_equal_center_bacc_delta
        ),
    )
    report_bridge = report.get("bridge")
    if (
        not isinstance(report_bridge, Mapping)
        or report_bridge.get("task_semantic_bridge") != task_bridge
    ):
        raise ValueError("Physical multiscale v3 task-semantic bridge drifted.")
    normalized_pooling = [
        {str(key): str(value) for key, value in row.items()} for row in pooling_rows
    ]
    if (
        int(report.get("pooling_audit_row_count", -1)) != len(pooling_rows)
        or report.get("pooling_audit_hash") != stable_hash(normalized_pooling)
    ):
        raise ValueError("Physical multiscale v3 pooling-audit identity drifted.")
    _validate_content_index(
        bundle,
        content_index,
        anchor_policy_id=expected_config.annotation_anchor_policy_id,
        contract_hash=str(contract["contract_hash"]),
    )
    return {
        "status": "PASS",
        "profile_id": PROFILE_ID,
        "row_count": len(observed_ids),
        "pooling_audit_row_count": len(pooling_rows),
        "contract_hash": contract_validation["contract_hash"],
        "minimum_cosine": minimum_cosine,
        "maximum_relative_l2": maximum_relative_l2,
        "task_semantic_bridge": task_bridge,
    }


def _center_grouped_sample_ids(
    contract_rows: list[dict[str, str]],
) -> tuple[str, ...]:
    """Return the exact traversal order used by the per-center cache shards."""

    return tuple(
        row["sample_id"]
        for center in MIDOGPP_ELIGIBLE_CENTERS
        for row in contract_rows
        if row["center"] == center
    )


def _validate_pooling_audit(
    pooling_rows: list[dict[str, str]],
    contract_rows: list[dict[str, str]],
    *,
    anchor_policy_id: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    expected_count = 3 * len(contract_rows)
    if len(pooling_rows) != expected_count:
        raise ValueError(
            f"Physical multiscale v3 pooling row count drift: "
            f"expected={expected_count}, actual={len(pooling_rows)}"
        )
    contract_by_id = {row["sample_id"]: row for row in contract_rows}
    index: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in pooling_rows:
        sample_id = row["sample_id"]
        fov_value = float(row["fov_um"])
        fov = f"{int(fov_value)}"
        key = (sample_id, fov)
        if key in index or sample_id not in contract_by_id:
            raise ValueError("Physical multiscale v3 pooling audit duplicates/unknown row.")
        contract_row = contract_by_id[sample_id]
        geometry = json.loads(contract_row["scale_geometry_json"])[f"{fov}um"]
        start = normalized_position_to_window_start(
            x=float(geometry["p_x"]),
            y=float(geometry["p_y"]),
        )
        if (
            int(row["contract_row_index"]) != int(contract_row["row_index"])
            or row["center"] != contract_row["center"]
            or int(row["label"]) != int(contract_row["label"])
            or row["annotation_anchor_policy_id"] != anchor_policy_id
            or row["annotation_anchor_policy_id"] != contract_row["policy_id"]
            or float(row["anchor_x"]) != float(contract_row["anchor_x"])
            or float(row["anchor_y"]) != float(contract_row["anchor_y"])
            or float(row["p_x"]) != float(geometry["p_x"])
            or float(row["p_y"]) != float(geometry["p_y"])
            or int(row["token_start_row"]) != start[0]
            or int(row["token_start_col"]) != start[1]
            or int(row["token_start_row"]) != int(geometry["token_start_row"])
            or int(row["token_start_col"]) != int(geometry["token_start_col"])
            or int(row["shift_x"]) != int(geometry["shift_x"])
            or int(row["shift_y"]) != int(geometry["shift_y"])
        ):
            raise ValueError(
                f"Physical multiscale v3 pooling recomputation drift: "
                f"{sample_id} {fov}um."
            )
        index[key] = row
    return index


def _validate_extractor(
    raw: object,
    *,
    representation_id: str,
    feature_dim: int,
    expected_config: PhysicalMultiscaleV3BuildConfig,
    expected_runtime: Mapping[str, str],
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("Physical multiscale v3 shard extractor is missing.")
    model_identity = raw.get("model_identity")
    pooling = raw.get("pooling_policy")
    if (
        raw.get("schema_version")
        != "midogpp_physical_multiscale_feature_extractor_v3"
        or raw.get("profile_id") != PROFILE_ID
        or raw.get("representation_id") != representation_id
        or int(raw.get("feature_dim", -1)) != feature_dim
        or raw.get("annotation_anchor_policy_id")
        != expected_config.annotation_anchor_policy_id
        or raw.get("runtime_identity") != expected_runtime
        or raw.get("input_decoder")
        != _expected_input_decoder(representation_id)
        or raw.get("forward_path") != "forward_features"
        or raw.get("token_layout")
        != {
            "width": VIRCHOW2_TOKEN_LAYOUT.width,
            "cls_token_count": VIRCHOW2_TOKEN_LAYOUT.cls_token_count,
            "register_token_count": VIRCHOW2_TOKEN_LAYOUT.register_token_count,
            "patch_grid_side": VIRCHOW2_TOKEN_LAYOUT.patch_grid_side,
            "window_side": VIRCHOW2_TOKEN_LAYOUT.window_side,
            "patch_order": VIRCHOW2_TOKEN_LAYOUT.patch_order,
        }
        or not isinstance(pooling, Mapping)
        or raw.get("pooling_policy_hash") != stable_hash(pooling)
        or pooling.get("patch_token_start") != 5
        or pooling.get("register_tokens_excluded") is not True
        or pooling.get("operator_order")
        != ["cls", "global_patch_mean", "local_4x4_patch_mean"]
        or pooling.get("patch_pool") != "uniform_arithmetic_mean_16_tokens"
        or raw.get("uses_experts") is not False
        or raw.get("performs_expert_aggregation") is not False
        or raw.get("uses_mixture_model") is not False
    ):
        raise ValueError(f"Physical multiscale v3 extractor drift: {representation_id}.")
    _validate_model_identity(model_identity, expected_config)


def _validate_model_identity(
    raw: object,
    config: PhysicalMultiscaleV3BuildConfig,
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("Physical multiscale v3 model identity is missing.")
    expected = {
        "model_ref": config.model_ref,
        "requested_revision": config.model_revision,
        "resolved_revision": config.model_revision,
        "model_config_sha256": config.expected_model_config_sha256,
        "checkpoint_file_sha256": config.expected_checkpoint_file_sha256,
        "state_dict_sha256": config.expected_state_dict_sha256,
        "preprocessing_config_hash": config.expected_preprocessing_config_hash,
    }
    preprocessing = raw.get("preprocessing_config")
    if (
        raw.get("schema_version") != "midogpp_virchow2_pinned_identity_v1"
        or any(raw.get(key) != value for key, value in expected.items())
        or not isinstance(preprocessing, Mapping)
        or stable_hash(preprocessing) != config.expected_preprocessing_config_hash
    ):
        raise ValueError("Physical multiscale v3 pinned model identity drifted.")


def _validate_content_index(
    root: Path,
    raw: Mapping[str, object],
    *,
    anchor_policy_id: str,
    contract_hash: str,
) -> None:
    files = raw.get("files")
    if (
        raw.get("schema_version") != "midogpp_physical_multiscale_content_index_v3"
        or raw.get("status") != "PASS"
        or raw.get("annotation_anchor_policy_id") != anchor_policy_id
        or raw.get("physical_contract_hash") != contract_hash
        or not isinstance(files, Mapping)
    ):
        raise ValueError("Physical multiscale v3 content index is malformed.")
    actual = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "content_index.json"
    }
    if dict(files) != actual:
        raise ValueError("Physical multiscale v3 content index differs from bundle bytes.")


def _validate_parent_decoders(
    *documents: Mapping[str, object],
    expected_raw_tiff_backend: str,
) -> None:
    if expected_raw_tiff_backend != "pyvips" or any(
        document.get("annotation_jpeg_decoder") != ANNOTATION_JPEG_DECODER
        or document.get("raw_tiff_slide_reader_backend")
        != expected_raw_tiff_backend
        for document in documents
    ):
        raise ValueError("Physical multiscale v3 parent decoder lineage drifted.")


def _validate_child_input_decoder(
    document: Mapping[str, object],
    *,
    representation_id: str,
) -> None:
    if document.get("input_decoder") != _expected_input_decoder(representation_id):
        raise ValueError("Physical multiscale v3 child input decoder drifted.")


def _expected_input_decoder(representation_id: str) -> str:
    if representation_id == B_REPRESENTATION_ID:
        return B_INPUT_DECODER
    if representation_id == C_REPRESENTATION_ID:
        return C_INPUT_DECODER
    raise ValueError(
        f"Unsupported physical multiscale v3 representation: {representation_id}."
    )


def _torch_payload(torch: object, path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"Physical multiscale v3 cache shard is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[attr-defined]
    except TypeError:
        payload = torch.load(path, map_location="cpu")  # type: ignore[attr-defined]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Physical multiscale v3 shard must be a mapping: {path}")
    if payload.get("embeddings") is None or not isinstance(
        payload.get("metadata"),
        list,
    ):
        raise ValueError(f"Physical multiscale v3 shard is incomplete: {path}")
    return payload


def _require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Physical multiscale v3 cache is missing files: {missing}")


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
