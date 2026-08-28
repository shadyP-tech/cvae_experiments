"""Exact canonical source-cache admission and producer-local outcome capability."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS
from ..source_bundle.constants import (
    RAW_SOURCE_CASE_COUNT,
    RAW_SOURCE_ROW_COUNT,
    SOURCE_CACHE_ARTIFACT_ID,
    SOURCE_CACHE_FILE_HASHES,
    SOURCE_FEATURE_DIM,
    SOURCE_REPRESENTATION_ID,
    SOURCE_SPLIT,
)
from ..source_bundle.hashing import exact_keys, file_sha256, read_json


_HEX_LOCK = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
_OUTCOME_GATE = object()
_PROBABILITY_SEAL_GATE = object()


@dataclass(frozen=True, order=True, slots=True)
class SourceRowIdentity:
    source_cache_row_index: int
    source_row_id: str
    case_id: str
    center: str
    split: str = SOURCE_SPLIT

    def __post_init__(self) -> None:
        if (
            type(self.source_cache_row_index) is not int
            or self.source_cache_row_index < 0
            or not self.source_row_id.startswith("source_row_")
            or not self.case_id
            or self.center not in CENTERS
            or self.split != SOURCE_SPLIT
        ):
            raise ProtocolError("OE-PPUR v3 source row identity drifted.")


@dataclass(frozen=True, slots=True)
class LabelFreeSourceFrame:
    embeddings: np.ndarray
    rows: tuple[SourceRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[SourceRowIdentity, ...]]
    cache_file_hashes: tuple[tuple[str, str], ...]
    frame_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {str(center): tuple(items) for center, items in self.rows_by_center.items()}
        if (
            values.shape != (RAW_SOURCE_ROW_COUNT, SOURCE_FEATURE_DIM)
            or values.dtype != np.float32
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
            or len(rows) != RAW_SOURCE_ROW_COUNT
            or tuple(row.source_cache_row_index for row in rows) != tuple(range(RAW_SOURCE_ROW_COUNT))
            or len({row.source_row_id for row in rows}) != RAW_SOURCE_ROW_COUNT
            or tuple(by_center) != CENTERS
            or any(row.center != center for center in CENTERS for row in by_center[center])
            or tuple(
                sorted(
                    (row for center in CENTERS for row in by_center[center]),
                    key=lambda item: item.source_cache_row_index,
                )
            )
            != rows
            or len({(row.center, row.case_id) for row in rows}) != RAW_SOURCE_CASE_COUNT
            or tuple(self.cache_file_hashes) != SOURCE_CACHE_FILE_HASHES
        ):
            raise ProtocolError("OE-PPUR v3 label-free source frame drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_file_hashes", SOURCE_CACHE_FILE_HASHES)
        object.__setattr__(
            self,
            "frame_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_label_free_source_frame_v1",
                    "source_cache_artifact_id": SOURCE_CACHE_ARTIFACT_ID,
                    "source_cache_file_hashes": SOURCE_CACHE_FILE_HASHES,
                    "row_identity": tuple(
                        (
                            row.source_cache_row_index,
                            row.source_row_id,
                            row.case_id,
                            row.center,
                            row.split,
                        )
                        for row in rows
                    ),
                    "row_count": RAW_SOURCE_ROW_COUNT,
                    "case_count": RAW_SOURCE_CASE_COUNT,
                    "feature_dim": SOURCE_FEATURE_DIM,
                    "labels_present": False,
                    "target_rows_present": False,
                }
            ),
        )

    def embeddings_for_center(self, center: object) -> np.ndarray:
        key = str(center)
        if key not in self.rows_by_center:
            raise ProtocolError("OE-PPUR v3 source center is outside C.")
        indices = np.fromiter(
            (row.source_cache_row_index for row in self.rows_by_center[key]),
            dtype=np.int64,
        )
        return np.ascontiguousarray(self.embeddings[indices], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class SourceOutcomeRow:
    source_cache_row_index: int
    source_row_id: str
    case_id: str
    center: str
    outcome: int
    split: str = SOURCE_SPLIT

    def __post_init__(self) -> None:
        if (
            type(self.source_cache_row_index) is not int
            or self.source_cache_row_index < 0
            or not self.source_row_id.startswith("source_row_")
            or not self.case_id
            or self.center not in CENTERS
            or self.outcome not in (0, 1)
            or self.split != SOURCE_SPLIT
        ):
            raise ProtocolError("OE-PPUR v3 source outcome row drifted.")


@dataclass(frozen=True, slots=True)
class SourceProbabilitySeal:
    source_frame_hash: str
    source_stream_lock_hash: str
    held_action_library_sha256: str
    held_mass_policy_receipt_sha256: str
    oriented_block_receipts: tuple[tuple[str, str, str], ...]
    labels_used: bool = False
    _factory_token: InitVar[object] = None
    _factory_validated: bool = field(init=False, repr=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PROBABILITY_SEAL_GATE:
            raise ProtocolError(
                "OE-PPUR v3 source probability seal is production-factory-only."
            )
        frame_hash = require_sha256(self.source_frame_hash, "source frame hash")
        stream_lock = str(self.source_stream_lock_hash).lower()
        library = require_sha256(self.held_action_library_sha256, "held action library hash")
        mass = require_sha256(self.held_mass_policy_receipt_sha256, "held mass policy receipt hash")
        blocks = tuple(self.oriented_block_receipts)
        expected = tuple((h, q) for h in CENTERS for q in CENTERS if h != q)
        if (
            _HEX_LOCK.fullmatch(stream_lock) is None
            or tuple((h, q) for h, q, _ in blocks) != expected
            or any(require_sha256(digest, "held prediction block hash") != digest for _, _, digest in blocks)
            or self.labels_used is not False
        ):
            raise ProtocolError("OE-PPUR v3 source probability seal coverage drifted.")
        object.__setattr__(self, "source_frame_hash", frame_hash)
        object.__setattr__(self, "source_stream_lock_hash", stream_lock)
        object.__setattr__(self, "held_action_library_sha256", library)
        object.__setattr__(self, "held_mass_policy_receipt_sha256", mass)
        object.__setattr__(self, "oriented_block_receipts", blocks)
        object.__setattr__(self, "_factory_validated", True)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_source_probability_seal_v1",
                    "source_frame_hash": frame_hash,
                    "source_stream_lock_hash": stream_lock,
                    "held_action_library_sha256": library,
                    "held_mass_policy_receipt_sha256": mass,
                    "oriented_block_receipts": blocks,
                    "oriented_block_count": 72,
                    "seed_pair_count_per_block": 9,
                    "labels_used": False,
                    "target_rows_present": False,
                }
            ),
        )


class _SourceOutcomeCapability:
    __slots__ = ("_frame_hash", "_rows", "_outcomes")

    def __init__(
        self,
        *,
        gate: object,
        frame_hash: str,
        rows: Sequence[SourceRowIdentity],
        outcomes: Sequence[int],
    ) -> None:
        if gate is not _OUTCOME_GATE:
            raise ProtocolError("OE-PPUR v3 source outcome capability is factory-only.")
        self._frame_hash = require_sha256(frame_hash, "source frame hash")
        self._rows = tuple(rows)
        self._outcomes = tuple(int(value) for value in outcomes)
        if len(self._rows) != RAW_SOURCE_ROW_COUNT or set(self._outcomes) - {0, 1}:
            raise ProtocolError("OE-PPUR v3 source outcome capability drifted.")

    def open_after_probability_seal(
        self, seal: SourceProbabilitySeal
    ) -> tuple[SourceOutcomeRow, ...]:
        if (
            not isinstance(seal, SourceProbabilitySeal)
            or seal._factory_validated is not True
            or seal.source_frame_hash != self._frame_hash
        ):
            raise ProtocolError("OE-PPUR v3 source outcomes cannot open before prediction sealing.")
        return tuple(
            SourceOutcomeRow(
                source_cache_row_index=row.source_cache_row_index,
                source_row_id=row.source_row_id,
                case_id=row.case_id,
                center=row.center,
                outcome=outcome,
            )
            for row, outcome in zip(self._rows, self._outcomes, strict=True)
        )


@dataclass(frozen=True, slots=True)
class AdmittedSourceCache:
    frame: LabelFreeSourceFrame
    _outcome_capability: _SourceOutcomeCapability = field(repr=False)
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, LabelFreeSourceFrame) or not isinstance(
            self._outcome_capability, _SourceOutcomeCapability
        ):
            raise ProtocolError("OE-PPUR v3 source cache admission is untyped.")
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_source_cache_admission_v1",
                    "source_frame_hash": self.frame.frame_hash,
                    "source_cache_file_hashes": SOURCE_CACHE_FILE_HASHES,
                    "source_outcomes_sealed": True,
                    "target_rows_present": False,
                }
            ),
        )

    def open_source_outcomes(
        self, seal: SourceProbabilitySeal
    ) -> tuple[SourceOutcomeRow, ...]:
        return self._outcome_capability.open_after_probability_seal(seal)


def load_canonical_source_cache(root: str | Path) -> AdmittedSourceCache:
    """Load the exact five-file train cache while sealing source outcomes."""

    path = _safe_exact_root(root)
    files = {member: path / member for member, _ in SOURCE_CACHE_FILE_HASHES}
    for member, expected in SOURCE_CACHE_FILE_HASHES:
        if file_sha256(files[member]) != expected:
            raise ProtocolError(f"OE-PPUR v3 source cache member drifted: {member}")
    source_protocol_hash = _validate_cache_json(files)
    try:
        import torch

        payload = torch.load(files["embeddings/train.pt"], map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ProtocolError("OE-PPUR v3 source cache tensor payload is unreadable.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("OE-PPUR v3 source cache tensor payload is untyped.")
    exact_keys(payload, ("embeddings", "metadata", "feature_extractor"), role="source cache tensor")
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    metadata = payload["metadata"]
    extractor = payload["feature_extractor"]
    if (
        embeddings.shape != (RAW_SOURCE_ROW_COUNT, SOURCE_FEATURE_DIM)
        or not np.isfinite(embeddings).all()
        or not isinstance(metadata, list)
        or len(metadata) != RAW_SOURCE_ROW_COUNT
        or not isinstance(extractor, Mapping)
        or extractor.get("schema_version") != "midogpp_uniform_b_canonical_feature_extractor_v1"
        or extractor.get("model") != "Virchow2"
        or extractor.get("dataset") != "MIDOG++"
        or extractor.get("representation_id") != SOURCE_REPRESENTATION_ID
        or extractor.get("feature_dim") != SOURCE_FEATURE_DIM
        or extractor.get("pooling") != "fixed_center_rows6to9_cols6to9"
        or extractor.get("source_protocol_hash") != source_protocol_hash
    ):
        raise ProtocolError("OE-PPUR v3 source cache tensor geometry/lineage drifted.")
    rows: list[SourceRowIdentity] = []
    outcomes: list[int] = []
    contract_indices: set[int] = set()
    raw_sample_ids: set[str] = set()
    for index, raw in enumerate(metadata):
        if not isinstance(raw, Mapping):
            raise ProtocolError("OE-PPUR v3 source cache metadata row is untyped.")
        exact_keys(
            raw,
            ("sample_id", "case_id", "label", "split", "center", "contract_row_index"),
            role="source cache metadata row",
        )
        sample_id = str(raw["sample_id"])
        case_id = str(raw["case_id"])
        center = str(raw["center"])
        split = str(raw["split"])
        if isinstance(raw["label"], bool) or isinstance(
            raw["contract_row_index"], bool
        ):
            raise ProtocolError("OE-PPUR v3 source cache boolean integer drifted.")
        try:
            label = int(raw["label"])
            contract_index = int(raw["contract_row_index"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR v3 source cache metadata integer drifted.") from exc
        if (
            not sample_id
            or sample_id in raw_sample_ids
            or not case_id
            or center not in CENTERS
            or split != SOURCE_SPLIT
            or label not in (0, 1)
            or contract_index < 0
            or contract_index in contract_indices
        ):
            raise ProtocolError("OE-PPUR v3 source cache metadata identity drifted.")
        raw_sample_ids.add(sample_id)
        contract_indices.add(contract_index)
        opaque = "source_row_" + canonical_hash(
            {
                "schema_version": "oe_ppur_v3_opaque_source_row_id_v1",
                "source_cache_tensor_sha256": dict(SOURCE_CACHE_FILE_HASHES)["embeddings/train.pt"],
                "source_cache_row_index": index,
                "sample_id": sample_id,
            }
        )
        rows.append(SourceRowIdentity(index, opaque, case_id, center))
        outcomes.append(label)
    frozen = np.ascontiguousarray(embeddings, dtype=np.float32)
    by_center = {center: tuple(row for row in rows if row.center == center) for center in CENTERS}
    frame = LabelFreeSourceFrame(frozen, tuple(rows), by_center, SOURCE_CACHE_FILE_HASHES)
    return AdmittedSourceCache(
        frame,
        _SourceOutcomeCapability(
            gate=_OUTCOME_GATE,
            frame_hash=frame.frame_hash,
            rows=frame.rows,
            outcomes=outcomes,
        ),
    )


def _safe_exact_root(value: str | Path) -> Path:
    candidate = Path(os.path.abspath(Path(value)))
    current = candidate
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 source cache path contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 source cache root is absent.") from exc
    if root != candidate or root.is_symlink() or not root.is_dir() or root == Path(root.anchor):
        raise ProtocolError("OE-PPUR v3 source cache root is unsafe.")
    observed = tuple(
        sorted(item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file())
    )
    observed_directories = tuple(
        sorted(
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_dir()
        )
    )
    expected = tuple(sorted(member for member, _ in SOURCE_CACHE_FILE_HASHES))
    if (
        observed != expected
        or observed_directories != ("embeddings", "manifests", "reports")
        or any(item.is_symlink() for item in root.rglob("*"))
    ):
        raise ProtocolError("OE-PPUR v3 source cache must contain exactly five plain files.")
    return root


def _validate_cache_json(files: Mapping[str, Path]) -> str:
    frozen = read_json(files["manifests/frozen_cache_protocol.json"])
    report = read_json(files["reports/cache_builder_report.json"])
    validation = read_json(files["reports/validation_report.json"])
    index = read_json(files["manifests/content_index.json"])
    unhashed_frozen = {key: value for key, value in frozen.items() if key != "protocol_hash"}
    unhashed_index = {key: value for key, value in index.items() if key != "content_hash"}
    indexed = index.get("files")
    expected_indexed = tuple(member for member, _ in SOURCE_CACHE_FILE_HASHES if member != "manifests/content_index.json")
    if (
        frozen.get("schema_version") != "midogpp_uniform_b_canonical_cache_protocol_v1"
        or frozen.get("representation_id") != SOURCE_REPRESENTATION_ID
        or frozen.get("transformation") != "lossless_center_shard_concatenation_in_manifest_order"
        or frozen.get("split") != SOURCE_SPLIT
        or tuple(frozen.get("eligible_centers", ())) != CENTERS
        or frozen.get("row_count") != RAW_SOURCE_ROW_COUNT
        or frozen.get("feature_dim") != SOURCE_FEATURE_DIM
        or frozen.get("labels_used_for_feature_construction") is not False
        or frozen.get("test_rows_present") is not False
        or frozen.get("validation_rows_present") is not False
        or frozen.get("protocol_hash") != stable_hash(unhashed_frozen)
        or report.get("schema_version") != "midogpp_uniform_b_canonical_cache_builder_v1"
        or report.get("status") != "PASS"
        or report.get("representation_id") != SOURCE_REPRESENTATION_ID
        or report.get("split") != SOURCE_SPLIT
        or report.get("row_count") != RAW_SOURCE_ROW_COUNT
        or report.get("feature_dim") != SOURCE_FEATURE_DIM
        or report.get("source_shards") != len(CENTERS)
        or report.get("numeric_transformation") != "none"
        or report.get("independent_validation_status") != "PASS"
        or validation.get("schema_version") != "midogpp_uniform_b_canonical_cache_validation_v1"
        or validation.get("status") != "PASS"
        or validation.get("validator")
        != "validate_uniform_b_canonical_train_cache"
        or validation.get("checks")
        != {
            "status": "PASS",
            "row_count": RAW_SOURCE_ROW_COUNT,
            "center_count": len(CENTERS),
            "feature_dim": SOURCE_FEATURE_DIM,
            "numeric_identity": "EXACT",
        }
        or index.get("schema_version") != "midogpp_uniform_b_canonical_cache_content_index_v1"
        or index.get("content_hash") != stable_hash(unhashed_index)
        or not isinstance(indexed, list)
    ):
        raise ProtocolError("OE-PPUR v3 source cache JSON protocol drifted.")
    observed_indexed = []
    for raw in indexed:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ProtocolError("OE-PPUR v3 source cache content-index row drifted.")
        member = str(raw["path"])
        digest = str(raw["sha256"])
        if member not in expected_indexed or file_sha256(files[member]) != digest:
            raise ProtocolError("OE-PPUR v3 source cache content-index member drifted.")
        observed_indexed.append(member)
    if tuple(observed_indexed) != expected_indexed:
        raise ProtocolError("OE-PPUR v3 source cache content-index order/coverage drifted.")
    return str(frozen["protocol_hash"])


__all__ = (
    "AdmittedSourceCache",
    "LabelFreeSourceFrame",
    "SourceOutcomeRow",
    "SourceProbabilitySeal",
    "SourceRowIdentity",
    "load_canonical_source_cache",
)
