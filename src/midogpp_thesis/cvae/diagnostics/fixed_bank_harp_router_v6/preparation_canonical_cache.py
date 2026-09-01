"""Authentication and loading of the immutable consumed-test source cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .preparation_contracts import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_NAME,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_REPRESENTATION,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASES_BY_CENTER,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROWS_BY_CENTER,
    LEGACY_LABEL,
    METADATA_FIELDS,
    CanonicalFrameRow,
    CanonicalLabelBlindCacheIdentity,
    CanonicalLabelBlindFrame,
)
from .preparation_durable_io import single_inventory
from .safe_paths import safe_existing_member


def load_canonical_label_blind_cache(root: Path) -> CanonicalLabelBlindFrame:
    """Authenticate and load the canonical consumed-test PT shards."""

    source_identity = validate_canonical_label_blind_cache_identity(root)
    cache_root = source_identity.root
    indexed = dict(source_identity.member_sha256)
    frozen = read_json(cache_root / "manifests/frozen_build_protocol.json")
    alignment = read_json(cache_root / "manifests/row_alignment.json")
    report = read_json(cache_root / "reports/cache_builder_report.json")
    validation = read_json(cache_root / "reports/validation_report.json")
    extractor = frozen.get("cache_extractor_protocol")
    if (
        not isinstance(extractor, Mapping)
        or frozen.get("cache_name") != CANONICAL_CACHE_NAME
        or extractor.get("representation_id") != CANONICAL_REPRESENTATION
        or frozen.get("scoring_manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or alignment.get("row_order_hash") != CANONICAL_CACHE_ROW_ORDER_HASH
        or report.get("row_order_hash") != CANONICAL_CACHE_ROW_ORDER_HASH
        or report.get("row_count") != EXPECTED_ROW_COUNT
        or report.get("fresh_evidence") is not False
        or validation.get("status") != "PASS"
    ):
        raise ProtocolError("HARP canonical cache protocol drifted.")
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP cache preparation requires torch.") from exc
    rows_by_center: dict[str, tuple[CanonicalFrameRow, ...]] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        relative = f"embeddings/by_center/center_{center}.pt"
        path = safe_existing_member(cache_root, relative, role="canonical cache")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old workstation torch
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise ProtocolError("HARP canonical cache shard is unreadable.") from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "embeddings",
            "metadata",
            "feature_extractor",
        }:
            raise ProtocolError("HARP canonical cache shard schema drifted.")
        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, Sequence) or isinstance(
            raw_metadata, (str, bytes)
        ):
            raise ProtocolError("HARP canonical cache metadata is malformed.")
        center_rows: list[CanonicalFrameRow] = []
        for ordinal, raw in enumerate(raw_metadata):
            if not isinstance(raw, Mapping) or {
                str(key) for key in raw
            } != METADATA_FIELDS:
                raise ProtocolError("HARP canonical cache metadata firewall failed.")
            sample = str(raw["evaluation_row_id"])
            if (
                not sample.startswith("eval_")
                or len(sample) != 69
                or LEGACY_LABEL.search(sample)
                or str(raw["center"]) != center
                or str(raw["split"]) != "test"
                or not str(raw["case_id"])
                or type(raw["contract_row_index"]) is not int
                or int(raw["contract_row_index"]) < 0
            ):
                raise ProtocolError("HARP canonical cache row identity drifted.")
            center_rows.append(
                CanonicalFrameRow(
                    center=center,
                    case_id=str(raw["case_id"]),
                    sample_id=sample,
                    contract_row_index=int(raw["contract_row_index"]),
                    center_row_index=ordinal,
                )
            )
        values = np.ascontiguousarray(
            torch.as_tensor(payload["embeddings"]).detach().cpu().float().numpy(),
            dtype=np.float32,
        )
        if (
            values.shape != (len(center_rows), COMMON_OUTPUT_DIM)
            or len(center_rows) != EXPECTED_ROWS_BY_CENTER[center]
            or len({row.case_id for row in center_rows})
            != EXPECTED_CASES_BY_CENTER[center]
            or not np.isfinite(values).all()
        ):
            raise ProtocolError("HARP canonical cache shard geometry drifted.")
        rows_by_center[center] = tuple(center_rows)
        embeddings_by_center[center] = values
    rows = tuple(row for center in CENTERS for row in rows_by_center[center])
    if (
        len(rows) != EXPECTED_ROW_COUNT
        or len({(row.center, row.case_id) for row in rows}) != EXPECTED_CASE_COUNT
        or len({row.sample_id for row in rows}) != EXPECTED_ROW_COUNT
        or len({row.contract_row_index for row in rows}) != EXPECTED_ROW_COUNT
    ):
        raise ProtocolError("HARP canonical cache global geometry drifted.")
    return CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash=source_identity.content_hash,
        row_order_hash=CANONICAL_CACHE_ROW_ORDER_HASH,
        source_member_sha256=indexed,
    )


def validate_canonical_label_blind_cache_identity(
    root: Path,
) -> CanonicalLabelBlindCacheIdentity:
    """Reauthenticate every canonical source byte without loading tensors."""

    if root.is_symlink():
        raise ProtocolError("HARP canonical cache root is unsafe.")
    try:
        cache_root = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("HARP canonical cache root is absent.") from exc
    if not cache_root.is_dir():
        raise ProtocolError("HARP canonical cache root is unsafe.")
    content = read_json(cache_root / "manifests/content_index.json")
    files = content.get("files")
    content_base = {
        key: value for key, value in content.items() if key != "content_hash"
    }
    if (
        set(content) != {"schema_version", "files", "content_hash"}
        or not isinstance(files, list)
        or content.get("content_hash") != canonical_hash(content_base)
        or content.get("content_hash") != CANONICAL_CACHE_CONTENT_HASH
    ):
        raise ProtocolError("HARP canonical cache content index drifted.")
    indexed: dict[str, str] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ProtocolError("HARP canonical cache content member is malformed.")
        relative = str(row["path"])
        member = safe_existing_member(cache_root, relative, role="canonical cache")
        if relative in indexed or sha256_file(member) != row["sha256"]:
            raise ProtocolError("HARP canonical cache member bytes drifted.")
        indexed[relative] = str(row["sha256"])
    # Take one closed-world snapshot. Repeated traversal of this large tree is
    # needlessly expensive on the workstation.
    inventory = single_inventory(cache_root, role="canonical cache")
    actual = {
        path.relative_to(cache_root).as_posix()
        for path in inventory
        if path.is_file()
        and path.relative_to(cache_root).as_posix()
        != "manifests/content_index.json"
    }
    if actual != set(indexed):
        raise ProtocolError("HARP canonical cache closed-world inventory drifted.")
    return CanonicalLabelBlindCacheIdentity(
        root=cache_root,
        content_hash=str(content["content_hash"]),
        member_sha256=indexed,
    )


__all__ = (
    "load_canonical_label_blind_cache",
    "validate_canonical_label_blind_cache_identity",
)
