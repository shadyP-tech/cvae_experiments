"""Complete immutable input-file bindings for per-H decision locks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.physical_multiscale.config_v3 import (
    ANNOTATION_JPEG_DECODER,
    B_INPUT_DECODER,
    C_INPUT_DECODER,
)

from .config import PhysicalMultiscalePilotConfig


_V3_PARENT_FIREWALL = {
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
_V3_CONTRACT_FIREWALL = {
    **_V3_PARENT_FIREWALL,
    "geometry_uses_labels": False,
    "geometry_uses_center_identity": False,
    "uses_latent_prior": False,
    "uses_posterior": False,
}


def input_files(config: PhysicalMultiscalePilotConfig) -> dict[str, Path]:
    files = {
        "base_manifest": config.base_manifest_path,
        "canonical_a_cache": config.canonical_a_cache_path,
        "physical_contract": config.physical_contract_root
        / "physical_multiscale_contract.json",
        "physical_manifest": config.physical_contract_root
        / "physical_multiscale_manifest.csv",
        "physical_resolution_audit": config.physical_contract_root
        / "resolution_audit.csv",
        "b_alignment": config.b_cache_root / "manifests" / "row_alignment.json",
        "b_builder_report": config.b_cache_root
        / "reports"
        / "cache_builder_report.json",
        "c_alignment": config.c_cache_root / "manifests" / "row_alignment.json",
        "c_builder_report": config.c_cache_root
        / "reports"
        / "cache_builder_report.json",
        "canonical_reference_protocol": config.canonical_reference_root
        / "manifests"
        / "protocol_manifest.json",
        "canonical_reference_results": config.canonical_reference_root
        / "tables"
        / "classifier_tuned_source_results.csv",
        "canonical_reference_predictions": config.canonical_reference_root
        / "tables"
        / "classifier_tuned_predictions.csv",
    }
    for center in config.heldout_centers:
        files[f"b_center_{center}"] = (
            config.b_cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
        files[f"c_center_{center}"] = (
            config.c_cache_root
            / "embeddings"
            / "by_center"
            / f"center_{center}.pt"
        )
    if config.cache_bundle_root is not None:
        files.update(
            {
                "cache_bundle_manifest": config.cache_bundle_root
                / "manifests"
                / "bundle_manifest.json",
                "cache_bundle_pooling_audit": config.cache_bundle_root
                / "manifests"
                / "pooling_audit.csv",
                "cache_bundle_content_index": config.cache_bundle_root
                / "manifests"
                / "content_index.json",
                "cache_bundle_report": config.cache_bundle_root
                / "reports"
                / "cache_bundle_report.json",
            }
        )
    return files


def compute_input_hashes(config: PhysicalMultiscalePilotConfig) -> dict[str, str]:
    files = input_files(config)
    missing = [label for label, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Physical multiscale inputs are incomplete: {missing}")
    if config.cache_bundle_root is not None:
        _validate_atomic_cache_bundle(config)
    return {label: _sha256(path) for label, path in files.items()}


def _validate_atomic_cache_bundle(config: PhysicalMultiscalePilotConfig) -> None:
    root = config.cache_bundle_root
    if root is None:
        return
    if (
        config.b_cache_root.parent.resolve() != root.resolve()
        or config.c_cache_root.parent.resolve() != root.resolve()
    ):
        raise ValueError("Physical multiscale B/C roots are outside the atomic parent.")
    contract = _json(
        config.physical_contract_root / "physical_multiscale_contract.json"
    )
    manifest = _json(root / "manifests" / "bundle_manifest.json")
    report = _json(root / "reports" / "cache_bundle_report.json")
    content_index = _json(root / "manifests" / "content_index.json")
    contract_rows = _csv(
        config.physical_contract_root / "physical_multiscale_manifest.csv"
    )
    b_report = _json(config.b_cache_root / "reports" / "cache_builder_report.json")
    c_report = _json(config.c_cache_root / "reports" / "cache_builder_report.json")
    b_alignment = _json(config.b_cache_root / "manifests" / "row_alignment.json")
    c_alignment = _json(config.c_cache_root / "manifests" / "row_alignment.json")
    geometry_policy = contract.get("geometry_policy")
    if not isinstance(geometry_policy, Mapping):
        raise ValueError("Physical multiscale v3 contract geometry policy is missing.")
    anchor_policy_id = geometry_policy.get("annotation_anchor_policy_id")
    sample_ids = tuple(row["sample_id"] for row in contract_rows)
    sample_order_hash = stable_hash(sample_ids)
    canonical_cache_sha256 = _sha256(config.canonical_a_cache_path)
    expected_representations = [
        {
            "representation_id": config.representation_order[1],
            "relative_root": config.b_cache_root.name,
            "feature_dim": config.representation_dims[
                config.representation_order[1]
            ],
        },
        {
            "representation_id": config.representation_order[2],
            "relative_root": config.c_cache_root.name,
            "feature_dim": config.representation_dims[
                config.representation_order[2]
            ],
        },
    ]
    contract_hash = contract.get("contract_hash")
    expected_manifest = {
        "schema_version": "midogpp_physical_multiscale_cache_bundle_v3",
        "status": "PASS",
        "profile_id": config.profile.profile_id,
        "annotation_anchor_policy_id": anchor_policy_id,
        "physical_contract_hash": contract_hash,
        "canonical_a_cache_sha256": canonical_cache_sha256,
        "row_count": len(contract_rows),
        "sample_id_order_hash": sample_order_hash,
        "representations": expected_representations,
        "c_scale_order_um": [28.0, 56.0, 112.0],
        "representation_c_combination": "feature_concatenation_not_mixture",
        "patch_pool": "uniform_arithmetic_mean_16_tokens",
        "annotation_jpeg_decoder": ANNOTATION_JPEG_DECODER,
        "raw_tiff_slide_reader_backend": "pyvips",
        **_V3_PARENT_FIREWALL,
    }
    expected_report_keys = set(expected_manifest) | {
        "pooling_audit_row_count",
        "pooling_audit_hash",
        "model_identity",
        "runtime_identity",
        "preprocessing_spatial_identity",
        "bridge",
    }
    report_parity = {
        key: value
        for key, value in manifest.items()
        if key != "schema_version"
    }
    if (
        dict(manifest) != expected_manifest
        or report.get("schema_version")
        != "midogpp_physical_multiscale_cache_bundle_report_v3"
        or set(report) != expected_report_keys
        or any(report.get(key) != value for key, value in report_parity.items())
        or contract.get("canonical_cache_sha256") != canonical_cache_sha256
        or contract.get("row_count") != len(contract_rows)
        or contract.get("fov_um") != [28.0, 56.0, 112.0]
        or contract.get("claim_firewall") != _V3_CONTRACT_FIREWALL
    ):
        raise ValueError("Physical multiscale atomic cache parent lineage drifted.")
    expected_child = (
        (
            b_report,
            b_alignment,
            config.representation_order[1],
            config.representation_dims[config.representation_order[1]],
            "fixed_center_rows6to9_cols6to9",
            B_INPUT_DECODER,
        ),
        (
            c_report,
            c_alignment,
            config.representation_order[2],
            config.representation_dims[config.representation_order[2]],
            "annotation_local_start_clamp_floor_16p_minus2_window4",
            C_INPUT_DECODER,
        ),
    )
    parent_model_identity = report.get("model_identity")
    if not isinstance(parent_model_identity, Mapping):
        raise ValueError("Physical multiscale parent model identity is missing.")
    expected_alignment = {
        "schema_version": "midogpp_physical_multiscale_cache_alignment_v3",
        "status": "PASS",
        "profile_id": config.profile.profile_id,
        "row_count": len(contract_rows),
        "sample_id_order_hash": sample_order_hash,
        "eligible_centers": contract.get("eligible_centers"),
        "center_4_present": False,
        "canonical_a_cache_sha256": canonical_cache_sha256,
        "physical_contract_hash": contract_hash,
        "annotation_anchor_policy_id": anchor_policy_id,
    }
    if any(
        dict(child)
        != {
            "schema_version": "midogpp_physical_multiscale_cache_builder_v3",
            "status": "PASS",
            "profile_id": config.profile.profile_id,
            "representation_id": representation_id,
            "feature_dim": feature_dim,
            "row_count": len(contract_rows),
            "sample_id_order_hash": sample_order_hash,
            "annotation_anchor_policy_id": anchor_policy_id,
            "model_ref": parent_model_identity.get("model_ref"),
            "model_revision": parent_model_identity.get("requested_revision"),
            "model_identity": dict(parent_model_identity),
            "runtime_identity": report.get("runtime_identity"),
            "input_decoder": input_decoder,
            "preprocessing_spatial_identity": report.get(
                "preprocessing_spatial_identity"
            ),
            "pooling": pooling,
            "physical_contract_hash": contract_hash,
            "canonical_a_cache_sha256": canonical_cache_sha256,
            "bridge": report.get("bridge"),
        }
        or dict(alignment) != expected_alignment
        for (
            child,
            alignment,
            representation_id,
            feature_dim,
            pooling,
            input_decoder,
        ) in expected_child
    ):
        raise ValueError("Physical multiscale B/C child lineage is mixed or invalid.")
    pooling_rows = _csv(root / "manifests" / "pooling_audit.csv")
    pooling_keys = [
        (row.get("sample_id", ""), float(row.get("fov_um", "nan")))
        for row in pooling_rows
    ]
    expected_pooling_keys = _center_grouped_pooling_keys(
        contract_rows,
        center_order=config.heldout_centers,
    )
    if (
        pooling_keys != expected_pooling_keys
        or int(report.get("pooling_audit_row_count", -1)) != len(pooling_rows)
        or report.get("pooling_audit_hash")
        != stable_hash(
            [
                {str(key): str(value) for key, value in row.items()}
                for row in pooling_rows
            ]
        )
    ):
        raise ValueError("Physical multiscale parent pooling audit drifted.")
    indexed_files = content_index.get("files")
    actual_files = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "content_index.json"
    }
    if (
        content_index.get("schema_version")
        != "midogpp_physical_multiscale_content_index_v3"
        or content_index.get("status") != "PASS"
        or content_index.get("annotation_anchor_policy_id") != anchor_policy_id
        or content_index.get("physical_contract_hash") != contract_hash
        or not isinstance(indexed_files, Mapping)
        or dict(indexed_files) != actual_files
    ):
        raise ValueError("Physical multiscale atomic cache content index drifted.")


def _center_grouped_pooling_keys(
    contract_rows: list[dict[str, str]],
    *,
    center_order: tuple[str, ...],
) -> list[tuple[str, float]]:
    """Return the exact key order emitted by the center-sharded cache builder."""

    return [
        (row["sample_id"], fov)
        for center in center_order
        for row in contract_rows
        if row["center"] == center
        for fov in (28.0, 56.0, 112.0)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
