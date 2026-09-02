"""Label-blind projection and post-barrier identity for canonical train rows."""

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
    CANONICAL_REPRESENTATION,
    CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
    CANONICAL_SOURCE_TRAIN_CACHE_NAME,
    CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
    CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
    EXPECTED_SOURCE_TRAIN_CASE_COUNT,
    EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER,
    EXPECTED_SOURCE_TRAIN_ROW_COUNT,
    EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER,
    CanonicalFrameRow,
    CanonicalLabelBlindFrame,
)
from .preparation_durable_io import single_inventory
from .safe_paths import safe_existing_member


_SOURCE_TRAIN_MEMBERS = {
    "embeddings/train.pt": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    "manifests/frozen_cache_protocol.json": CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
    "manifests/content_index.json": CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
    "reports/cache_builder_report.json": CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
    "reports/validation_report.json": CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
}


def source_train_row_id(raw_sample_id: str, contract_row_index: int) -> str:
    """Return a stable identity that cannot disclose the legacy label-bearing ID."""

    return "src_" + canonical_hash(
        {
            "namespace": "midogpp_harp_v10_source_train_row_v1",
            "train_tensor_sha256": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
            "contract_row_index": int(contract_row_index),
            "raw_sample_id": str(raw_sample_id),
        }
    )


def load_canonical_source_train_label_blind_cache(
    root: Path,
) -> CanonicalLabelBlindFrame:
    """Project all 9,648 train rows without reading their metadata labels.

    The legacy tensor physically contains source labels.  This projection
    deliberately never indexes the ``label`` field and replaces the legacy
    sample identifier (which may itself contain a label suffix) with an opaque
    revision-owned identifier before any prepared byte is written.
    """

    cache_root = _validate_source_train_cache_identity(root)
    frozen = read_json(cache_root / "manifests/frozen_cache_protocol.json")
    report = read_json(cache_root / "reports/cache_builder_report.json")
    validation = read_json(cache_root / "reports/validation_report.json")
    if (
        frozen.get("cache_name") != CANONICAL_SOURCE_TRAIN_CACHE_NAME
        or frozen.get("representation_id") != CANONICAL_REPRESENTATION
        or frozen.get("split") != "train"
        or frozen.get("row_count") != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or frozen.get("feature_dim") != COMMON_OUTPUT_DIM
        or frozen.get("labels_used_for_feature_construction") is not False
        or frozen.get("test_rows_present") is not False
        or report.get("status") != "PASS"
        or report.get("row_count") != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or validation.get("status") != "PASS"
    ):
        raise ProtocolError("HARP v10 canonical source-train protocol drifted.")
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP cache preparation requires torch.") from exc
    tensor_path = cache_root / "embeddings/train.pt"
    try:
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old workstation torch
        payload = torch.load(tensor_path, map_location="cpu")
    except Exception as exc:
        raise ProtocolError("HARP v10 canonical source-train tensor is unreadable.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "embeddings",
        "metadata",
        "feature_extractor",
    }:
        raise ProtocolError("HARP v10 canonical source-train tensor schema drifted.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise ProtocolError("HARP v10 canonical source-train metadata is malformed.")
    values = np.ascontiguousarray(
        torch.as_tensor(payload["embeddings"]).detach().cpu().float().numpy(),
        dtype=np.float32,
    )
    if (
        values.shape != (EXPECTED_SOURCE_TRAIN_ROW_COUNT, COMMON_OUTPUT_DIM)
        or not np.isfinite(values).all()
        or len(metadata) != EXPECTED_SOURCE_TRAIN_ROW_COUNT
    ):
        raise ProtocolError("HARP v10 canonical source-train geometry drifted.")

    indices_by_center: dict[str, list[int]] = {center: [] for center in CENTERS}
    projected: dict[int, CanonicalFrameRow] = {}
    raw_ids: set[str] = set()
    opaque_ids: set[str] = set()
    for tensor_index, raw in enumerate(metadata):
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v10 canonical source-train metadata is malformed.")
        # Do not read raw["label"].  Only the later source-label capability is
        # allowed to access that outcome after the label-free cache is sealed.
        raw_id = str(raw.get("sample_id", ""))
        center = str(raw.get("center", ""))
        case_id = str(raw.get("case_id", ""))
        split = str(raw.get("split", ""))
        contract_index = raw.get("contract_row_index")
        if (
            not raw_id
            or raw_id in raw_ids
            or center not in CENTERS
            or not case_id
            or split != "train"
            or type(contract_index) is not int
            or int(contract_index) < 0
        ):
            raise ProtocolError("HARP v10 source-train row identity drifted.")
        opaque_id = source_train_row_id(raw_id, int(contract_index))
        if opaque_id in opaque_ids:
            raise ProtocolError("HARP v10 opaque source-train identity collided.")
        raw_ids.add(raw_id)
        opaque_ids.add(opaque_id)
        indices_by_center[center].append(tensor_index)
        projected[tensor_index] = CanonicalFrameRow(
            center=center,
            case_id=case_id,
            sample_id=opaque_id,
            contract_row_index=int(contract_index),
            center_row_index=-1,
            source_split="train",
        )

    rows_by_center: dict[str, tuple[CanonicalFrameRow, ...]] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        indices = indices_by_center[center]
        center_rows = tuple(
            CanonicalFrameRow(
                center=projected[index].center,
                case_id=projected[index].case_id,
                sample_id=projected[index].sample_id,
                contract_row_index=projected[index].contract_row_index,
                center_row_index=ordinal,
                source_split="train",
            )
            for ordinal, index in enumerate(indices)
        )
        if (
            len(center_rows) != EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER[center]
            or len({row.case_id for row in center_rows})
            != EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER[center]
        ):
            raise ProtocolError("HARP v10 source-train center geometry drifted.")
        rows_by_center[center] = center_rows
        embeddings_by_center[center] = np.ascontiguousarray(
            values[indices], dtype=np.float32
        )
    all_rows = tuple(row for center in CENTERS for row in rows_by_center[center])
    if (
        len(all_rows) != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or len({(row.center, row.case_id) for row in all_rows})
        != EXPECTED_SOURCE_TRAIN_CASE_COUNT
    ):
        raise ProtocolError("HARP v10 source-train global geometry drifted.")
    return CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash=CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
        row_order_hash=canonical_hash(
            {
                "ordered_source_rows": [
                    [row.center, row.case_id, row.sample_id, row.contract_row_index]
                    for row in all_rows
                ]
            }
        ),
        source_member_sha256=dict(_SOURCE_TRAIN_MEMBERS),
    )


def _validate_source_train_cache_identity(root: Path) -> Path:
    if root.is_symlink():
        raise ProtocolError("HARP v10 canonical source-train root is unsafe.")
    try:
        cache_root = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("HARP v10 canonical source-train root is absent.") from exc
    if not cache_root.is_dir():
        raise ProtocolError("HARP v10 canonical source-train root is unsafe.")
    inventory = single_inventory(cache_root, role="canonical source-train cache")
    actual = {
        path.relative_to(cache_root).as_posix()
        for path in inventory
        if path.is_file()
    }
    if actual != set(_SOURCE_TRAIN_MEMBERS):
        raise ProtocolError("HARP v10 source-train closed-world inventory drifted.")
    for relative, digest in _SOURCE_TRAIN_MEMBERS.items():
        member = safe_existing_member(
            cache_root, relative, role="canonical source-train cache"
        )
        if sha256_file(member) != digest:
            raise ProtocolError("HARP v10 canonical source-train bytes drifted.")
    return cache_root


__all__ = (
    "load_canonical_source_train_label_blind_cache",
    "source_train_row_id",
)
