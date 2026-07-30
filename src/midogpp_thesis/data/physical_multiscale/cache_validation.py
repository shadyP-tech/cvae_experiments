"""Independent validation for physical B/C center-sharded caches."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.common.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from midogpp_thesis.data.features.cache_io import load_cache_rows

from .bridge import evaluate_jpeg_task_bridge
if TYPE_CHECKING:
    from .config import PhysicalMultiscaleBuildConfig


def validate_cache_bundle(
    root: str | Path,
    *,
    expected_dim: int,
    config: "PhysicalMultiscaleBuildConfig | None" = None,
) -> Mapping[str, object]:
    path = Path(root)
    _require(
        path,
        ("manifests/row_alignment.json", "reports/cache_builder_report.json"),
    )
    report = _json(path / "reports/cache_builder_report.json")
    alignment = _json(path / "manifests/row_alignment.json")
    if (
        report.get("status") != "PASS"
        or alignment.get("status") != "PASS"
        or int(report.get("feature_dim", -1)) != int(expected_dim)
    ):
        raise ValueError("Physical multiscale cache report failed validation.")
    sample_ids: list[str] = []
    extractor_hashes: set[str] = set()
    expected_representation = {
        3840: "jpeg_center_b",
        11520: "physical_multiscale_center_c",
    }.get(int(expected_dim))
    if expected_representation is None:
        raise ValueError(f"Unsupported physical cache dimension: {expected_dim}")
    for center in MIDOGPP_ELIGIBLE_CENTERS:
        shard_path = path / "embeddings" / "by_center" / f"center_{center}.pt"
        shard = load_cache_rows(
            shard_path,
            expected_dim=expected_dim,
        )
        payload = _torch_payload_file(shard_path)
        extractor = payload.get("feature_extractor")
        if (
            not isinstance(extractor, Mapping)
            or extractor.get("representation_id") != expected_representation
            or int(extractor.get("feature_dim", -1)) != int(expected_dim)
        ):
            raise ValueError(f"Cache shard extractor identity drift: {center}")
        model_identity = extractor.get("model_identity")
        if not isinstance(model_identity, Mapping):
            raise ValueError(f"Cache shard lacks pinned model identity: {center}")
        if config is not None:
            _validate_model_identity(model_identity, config)
            if int(extractor.get("experiment_seed", -1)) != config.experiment_seed:
                raise ValueError(f"Cache shard experiment seed drift: {center}")
        extractor_hashes.add(stable_hash(extractor))
        if any(str(row.get("center")) != center for row in shard.metadata):
            raise ValueError(f"Cache shard center metadata drift: {center}")
        sample_ids.extend(shard.sample_ids)
    if (
        len(sample_ids) != len(set(sample_ids))
        or len(sample_ids) != int(alignment.get("row_count", -1))
        or len(extractor_hashes) != 1
    ):
        raise ValueError("Physical multiscale cache row identity drift.")
    if config is not None:
        contract = _json(
            config.contract_root / "physical_multiscale_contract.json"
        )
        report_identity = report.get("model_identity")
        if (
            report.get("physical_contract_hash") != contract.get("contract_hash")
            or not isinstance(report_identity, Mapping)
        ):
            raise ValueError("Cache report physical/model lineage drifted.")
        _validate_model_identity(report_identity, config)
    return {"status": "PASS", "row_count": len(sample_ids), "feature_dim": expected_dim}


def validate_cache_pair(
    b_root: str | Path,
    c_root: str | Path,
    *,
    contract_root: str | Path,
    canonical_cache_path: str | Path,
    canonical_reference_root: str | Path | None = None,
    config: "PhysicalMultiscaleBuildConfig | None" = None,
) -> Mapping[str, object]:
    """Reconstruct B/C row alignment and the embedded canonical-A bridge."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Physical cache-pair validation requires torch.") from exc

    b_path = Path(b_root)
    c_path = Path(c_root)
    contract_payload = _json(
        Path(contract_root) / "physical_multiscale_contract.json"
    )
    contract_rows = _csv(Path(contract_root) / "physical_multiscale_manifest.csv")
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
        raise ValueError("Canonical A and physical contract row order differ.")
    canonical_index = {
        sample_id: index for index, sample_id in enumerate(canonical.sample_ids)
    }
    minimum_cosine = 1.0
    maximum_relative_l2 = 0.0
    observed_ids: list[str] = []
    for center in MIDOGPP_ELIGIBLE_CENTERS:
        expected_rows = [row for row in contract_rows if row["center"] == center]
        b_payload = _torch_payload(
            torch,
            b_path / "embeddings" / "by_center" / f"center_{center}.pt",
        )
        c_payload = _torch_payload(
            torch,
            c_path / "embeddings" / "by_center" / f"center_{center}.pt",
        )
        b_metadata = tuple(dict(row) for row in b_payload["metadata"])
        c_metadata = tuple(dict(row) for row in c_payload["metadata"])
        if b_metadata != c_metadata:
            raise ValueError(f"B/C metadata differ for center {center}.")
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
            raise ValueError(f"Cache/contract row identity differs for center {center}.")
        a_embedded = b_payload.get("canonical_a_embeddings")
        if a_embedded is None:
            raise ValueError(f"B shard lacks embedded canonical A rows: center {center}.")
        canonical_rows = torch.stack(
            tuple(
                torch.as_tensor(canonical.embeddings[canonical_index[row["sample_id"]]])
                for row in expected_rows
            ),
            dim=0,
        ).detach().cpu()
        embedded = torch.as_tensor(a_embedded).detach().cpu()
        if embedded.shape != canonical_rows.shape or not torch.equal(
            embedded,
            canonical_rows,
        ):
            raise ValueError(f"Embedded canonical A differs byte-for-value: center {center}.")
        b_embeddings = torch.as_tensor(b_payload["embeddings"]).detach().cpu()
        prefix = b_embeddings[:, :2560]
        cosines = torch.nn.functional.cosine_similarity(prefix, embedded, dim=1)
        relative_l2 = torch.linalg.vector_norm(prefix - embedded, dim=1) / torch.clamp(
            torch.linalg.vector_norm(embedded, dim=1),
            min=1.0e-12,
        )
        minimum_cosine = min(minimum_cosine, float(cosines.min().item()))
        maximum_relative_l2 = max(
            maximum_relative_l2,
            float(relative_l2.max().item()),
        )
        observed_ids.extend(row["sample_id"] for row in expected_rows)
    if tuple(observed_ids) != expected_ids:
        raise ValueError("Center-sharded caches do not reconstruct contract order.")
    if minimum_cosine < 0.99999 or maximum_relative_l2 > 0.001:
        raise ValueError(
            "Recomputed canonical-A JPEG bridge failed: "
            f"minimum_cosine={minimum_cosine}, maximum_relative_l2={maximum_relative_l2}"
        )
    for root in (b_path, c_path):
        alignment = _json(root / "manifests" / "row_alignment.json")
        report = _json(root / "reports" / "cache_builder_report.json")
        bridge = report.get("bridge")
        if (
            alignment.get("canonical_a_cache_sha256") != canonical.cache_sha256
            or not isinstance(bridge, Mapping)
            or bridge.get("status") != "PASS"
            or float(bridge.get("minimum_cosine", -1.0)) < 0.99999
            or float(bridge.get("maximum_relative_l2", 1.0)) > 0.001
            or report.get("physical_contract_hash")
            != contract_payload.get("contract_hash")
        ):
            raise ValueError("Cache report does not bind the validated canonical-A bridge.")
        if config is not None:
            model_identity = report.get("model_identity")
            if not isinstance(model_identity, Mapping):
                raise ValueError("Cache report lacks pinned model identity.")
            _validate_model_identity(model_identity, config)
    task_bridge = None
    if canonical_reference_root is not None:
        task_bridge = evaluate_jpeg_task_bridge(
            b_path,
            canonical_reference_root,
        )
        for root in (b_path, c_path):
            report = _json(root / "reports" / "cache_builder_report.json")
            bridge = report.get("bridge")
            if (
                not isinstance(bridge, Mapping)
                or bridge.get("task_semantic_bridge") != task_bridge
            ):
                raise ValueError(
                    "Cache report does not bind the recomputed task-semantic bridge."
                )
    result = {
        "status": "PASS",
        "row_count": len(observed_ids),
        "canonical_a_cache_sha256": canonical.cache_sha256,
        "minimum_cosine": minimum_cosine,
        "maximum_relative_l2": maximum_relative_l2,
    }
    if task_bridge is not None:
        result["task_semantic_bridge"] = task_bridge
    return result


def _require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Physical multiscale bundle is missing files: {missing}")


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _torch_payload(torch: object, path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[attr-defined]
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")  # type: ignore[attr-defined]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Cache shard must contain a mapping: {path}")
    if "embeddings" not in payload or "metadata" not in payload:
        raise ValueError(f"Cache shard lacks embeddings/metadata: {path}")
    return payload


def _torch_payload_file(path: Path) -> Mapping[str, object]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Physical cache validation requires torch.") from exc
    return _torch_payload(torch, path)


def _validate_model_identity(
    identity: Mapping[str, object],
    config: "PhysicalMultiscaleBuildConfig",
) -> None:
    expected = {
        "model_ref": config.model_ref,
        "requested_revision": config.model_revision,
        "resolved_revision": config.model_revision,
        "model_config_sha256": config.expected_model_config_sha256,
        "checkpoint_file_sha256": config.expected_checkpoint_file_sha256,
        "state_dict_sha256": config.expected_state_dict_sha256,
        "preprocessing_config_hash": config.expected_preprocessing_config_hash,
    }
    if (
        identity.get("schema_version") != "midogpp_virchow2_pinned_identity_v1"
        or any(identity.get(key) != value for key, value in expected.items())
        or not isinstance(identity.get("preprocessing_config"), Mapping)
        or stable_hash(identity["preprocessing_config"])
        != config.expected_preprocessing_config_hash
    ):
        raise ValueError("Pinned Virchow2 model identity drifted.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
