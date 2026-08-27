"""Exact parser and contracts for OE-PPUR v3 direct input number three.

The bundle is a future, source-only artifact. This module can validate a
materialized bundle but never creates it and never reads unregistered raw
training data. Every logical source row is tied to exact ``C\\{H,q}`` pool
and sealed producer/compiler lineage before it can enter science code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility.contracts import canonical_sha256
from .candidate_pools import (
    ALL_ACTION_IDS,
    CompiledActionSurfaceReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
    build_held_center_candidate_pool,
)
from .identity import (
    CENTERS,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)


SOURCE_CACHE_ARTIFACT_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
SOURCE_SPLIT = "train"
SOURCE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
SOURCE_FEATURE_DIM = 3_840
DERIVED_FEATURE_DIM = 6
RAW_SOURCE_ROW_COUNT = 9_648
RAW_SOURCE_CASE_COUNT = 216
HELD_POOL_BLOCK_COUNT = 72
LOGICAL_SOURCE_ROW_COUNT = 77_184
LOGICAL_SOURCE_CASE_GROUP_COUNT = 1_728
PROBABILITY_COLUMN_COUNT = 7
PROBABILITY_DTYPE = "<f4"

SOURCE_CACHE_FILE_HASHES = (
    ("embeddings/train.pt", "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"),
    ("manifests/frozen_cache_protocol.json", "a4faf27a427cfb424789e5592048aa748a057f37124566d46b8b6c557e2bfe69"),
    ("manifests/content_index.json", "307991668f11454da69e3798feb23a2e899e1a00c2ee5132b031e7f7fb9ab82e"),
    ("reports/cache_builder_report.json", "3e3c40449196dc6db9fe0ab982defa86afb1094e3d958e944875396bc363b0ec"),
    ("reports/validation_report.json", "e8b69f557ea92ac8e70a20e504150aba1c947f2b47f735b34e3ca7147efcf6b7"),
)

SOURCE_SUPERVISION_MEMBERS = tuple(SOURCE_SUPERVISION_REQUIRED_MEMBERS)
INDEXED_MEMBERS = SOURCE_SUPERVISION_MEMBERS[:4]
SOURCE_ROW_COLUMNS = (
    "matrix_row_index",
    "outer_target_center",
    "query_center",
    "source_cache_row_index",
    "source_row_id",
    "case_id",
    "split",
    "outcome",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _text(value: object, *, role: str) -> str:
    result = str(value).strip()
    if not result:
        raise ProtocolError(f"OE-PPUR v3 requires non-empty {role}.")
    return result


def _sha256(value: object, *, role: str) -> str:
    result = _text(value, role=role).lower()
    if _SHA256.fullmatch(result) is None:
        raise ProtocolError(f"OE-PPUR v3 {role} is not a SHA-256 digest.")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: object, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def _no_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"OE-PPUR v3 JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"OE-PPUR v3 JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"OE-PPUR v3 JSON root is not an object: {path.name}")
    return value


def _exact_keys(value: Mapping[str, object], expected: Sequence[str], *, role: str) -> None:
    if set(value) != set(expected):
        raise ProtocolError(f"OE-PPUR v3 {role} keys drifted.")


@dataclass(frozen=True, slots=True)
class SourceSupervisionContractReceipt:
    """Pre-output contract hash; this breaks the pool/physical-receipt cycle."""

    compiler_receipt_hash: str
    producer_source_seal_sha256: str
    artifact_id: str = SOURCE_SUPERVISION_ARTIFACT_ID
    split: str = SOURCE_SPLIT
    representation_id: str = SOURCE_REPRESENTATION_ID
    all_center_ids: tuple[str, ...] = CENTERS
    representation_feature_dim: int = SOURCE_FEATURE_DIM
    derived_feature_dim: int = DERIVED_FEATURE_DIM
    exact_members: tuple[str, ...] = SOURCE_SUPERVISION_MEMBERS
    source_cache_file_hashes: tuple[tuple[str, str], ...] = SOURCE_CACHE_FILE_HASHES
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.artifact_id != SOURCE_SUPERVISION_ARTIFACT_ID
            or self.split != SOURCE_SPLIT
            or self.representation_id != SOURCE_REPRESENTATION_ID
            or tuple(self.all_center_ids) != CENTERS
            or int(self.representation_feature_dim) != SOURCE_FEATURE_DIM
            or int(self.derived_feature_dim) != DERIVED_FEATURE_DIM
            or tuple(self.exact_members) != SOURCE_SUPERVISION_MEMBERS
            or tuple(self.source_cache_file_hashes) != SOURCE_CACHE_FILE_HASHES
        ):
            raise ProtocolError("OE-PPUR v3 source-supervision contract identity drifted.")
        object.__setattr__(
            self, "compiler_receipt_hash",
            _sha256(self.compiler_receipt_hash, role="compiler receipt hash"),
        )
        object.__setattr__(
            self, "producer_source_seal_sha256",
            _sha256(self.producer_source_seal_sha256, role="producer source seal"),
        )
        object.__setattr__(self, "all_center_ids", CENTERS)
        object.__setattr__(self, "exact_members", SOURCE_SUPERVISION_MEMBERS)
        object.__setattr__(self, "source_cache_file_hashes", SOURCE_CACHE_FILE_HASHES)
        object.__setattr__(
            self, "contract_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_supervision_contract_v2",
                    "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
                    "split": SOURCE_SPLIT,
                    "representation_id": SOURCE_REPRESENTATION_ID,
                    "all_centers": CENTERS,
                    "representation_feature_dim": SOURCE_FEATURE_DIM,
                    "derived_feature_dim": DERIVED_FEATURE_DIM,
                    "source_cache_artifact_id": SOURCE_CACHE_ARTIFACT_ID,
                    "source_cache_file_hashes": SOURCE_CACHE_FILE_HASHES,
                    "expert_bank": (
                        EXPERT_BANK_ARTIFACT_ID,
                        EXPECTED_BANK_LOCK_HASH,
                        EXPECTED_BANK_CONTENT_INDEX_SHA256,
                    ),
                    "generation_lock": (
                        GENERATION_LOCK_ARTIFACT_ID,
                        EXPECTED_GENERATION_LOCK_HASH,
                        EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
                    ),
                    "producer_source_seal_sha256": self.producer_source_seal_sha256,
                    "compiler_receipt_hash": self.compiler_receipt_hash,
                    "exact_members": SOURCE_SUPERVISION_MEMBERS,
                    "raw_counts": (RAW_SOURCE_ROW_COUNT, RAW_SOURCE_CASE_COUNT),
                    "logical_counts": (
                        HELD_POOL_BLOCK_COUNT,
                        LOGICAL_SOURCE_ROW_COUNT,
                        LOGICAL_SOURCE_CASE_GROUP_COUNT,
                    ),
                    "matrix_contract": (
                        LOGICAL_SOURCE_ROW_COUNT,
                        PROBABILITY_COLUMN_COUNT,
                        PROBABILITY_DTYPE,
                    ),
                    "held_pool_policy": "C_MINUS_H_MINUS_q",
                    "final_pool_policy": "C_MINUS_H",
                    "source_outcomes_present": True,
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_source_training_surface_manifest_v2",
            "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
            "split": SOURCE_SPLIT,
            "representation_id": SOURCE_REPRESENTATION_ID,
            "representation_feature_dim": SOURCE_FEATURE_DIM,
            "derived_feature_dim": DERIVED_FEATURE_DIM,
            "all_center_ids": list(CENTERS),
            "source_cache_artifact_id": SOURCE_CACHE_ARTIFACT_ID,
            "source_cache_file_hashes": [
                {"member": member, "sha256": digest}
                for member, digest in SOURCE_CACHE_FILE_HASHES
            ],
            "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
            "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_bank_content_index_sha256": EXPECTED_BANK_CONTENT_INDEX_SHA256,
            "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
            "generation_content_index_sha256": EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
            "producer_source_seal_sha256": self.producer_source_seal_sha256,
            "compiler_receipt_hash": self.compiler_receipt_hash,
            "source_supervision_contract_hash": self.contract_hash,
            "action_ids": list(ALL_ACTION_IDS),
            "raw_source_row_count": RAW_SOURCE_ROW_COUNT,
            "raw_source_case_count": RAW_SOURCE_CASE_COUNT,
            "logical_block_count": HELD_POOL_BLOCK_COUNT,
            "logical_source_row_count": LOGICAL_SOURCE_ROW_COUNT,
            "logical_source_case_group_count": LOGICAL_SOURCE_CASE_GROUP_COUNT,
            "probability_shape": [LOGICAL_SOURCE_ROW_COUNT, PROBABILITY_COLUMN_COUNT],
            "probability_dtype": PROBABILITY_DTYPE,
            "source_outcomes_present": True,
            "target_rows_present": False,
            "target_labels_used": False,
        }


@dataclass(frozen=True, slots=True)
class SourceSupervisionRow:
    """One parsed matrix/CSV row; outcomes are source-only."""

    matrix_row_index: int
    outer_target_center: str
    query_center: str
    source_cache_row_index: int
    source_row_id: str
    case_id: str
    split: str
    outcome: int
    action_probabilities: tuple[float, ...]
    candidate_pool_receipt_hash: str
    compiled_surface_receipt_hash: str
    compiler_receipt_hash: str
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        matrix_index = int(self.matrix_row_index)
        cache_index = int(self.source_cache_row_index)
        h = _text(self.outer_target_center, role="outer target H")
        q = _text(self.query_center, role="source query q")
        row_id = _text(self.source_row_id, role="source row id")
        case_id = _text(self.case_id, role="source case id")
        outcome = int(self.outcome)
        probabilities = tuple(float(value) for value in self.action_probabilities)
        if (
            matrix_index < 0
            or not 0 <= cache_index < RAW_SOURCE_ROW_COUNT
            or h not in CENTERS
            or q not in CENTERS
            or h == q
            or self.split != SOURCE_SPLIT
            or outcome not in (0, 1)
            or len(probabilities) != PROBABILITY_COLUMN_COUNT
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in probabilities
            )
        ):
            raise ProtocolError("OE-PPUR v3 parsed source row violates its firewall.")
        object.__setattr__(self, "matrix_row_index", matrix_index)
        object.__setattr__(self, "source_cache_row_index", cache_index)
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "query_center", q)
        object.__setattr__(self, "source_row_id", row_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "action_probabilities", probabilities)
        for name in (
            "candidate_pool_receipt_hash",
            "compiled_surface_receipt_hash",
            "compiler_receipt_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), role=name))
        object.__setattr__(
            self, "row_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_supervision_row_v2",
                    "matrix_row_index": matrix_index,
                    "H": h,
                    "q": q,
                    "source_cache_row_index": cache_index,
                    "source_row_id": row_id,
                    "case_id": case_id,
                    "split": SOURCE_SPLIT,
                    "outcome": outcome,
                    "probabilities_f32_sha256": _array_sha256(
                        probabilities, dtype=PROBABILITY_DTYPE
                    ),
                    "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash,
                    "compiled_surface_receipt_hash": self.compiled_surface_receipt_hash,
                    "compiler_receipt_hash": self.compiler_receipt_hash,
                    "target_label": False,
                }
            ),
        )

    @property
    def held_center(self) -> str:
        return self.query_center

    @property
    def source_center(self) -> str:
        return self.query_center

    @property
    def row_id(self) -> str:
        return self.source_row_id


def _canonical_rows(
    values: Sequence[SourceSupervisionRow],
) -> tuple[SourceSupervisionRow, ...]:
    rows = tuple(sorted(tuple(values), key=lambda row: row.matrix_row_index))
    if (
        not rows
        or any(not isinstance(row, SourceSupervisionRow) for row in rows)
        or tuple(row.matrix_row_index for row in rows) != tuple(range(len(rows)))
    ):
        raise ProtocolError("OE-PPUR v3 source-supervision row order is not canonical.")
    return rows


def source_row_order_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return canonical_sha256(
        tuple(
            (
                row.matrix_row_index,
                row.outer_target_center,
                row.query_center,
                row.source_cache_row_index,
                row.source_row_id,
                row.case_id,
                row.split,
            )
            for row in _canonical_rows(rows)
        )
    )


def source_probability_matrix_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return _array_sha256(
        [row.action_probabilities for row in _canonical_rows(rows)],
        dtype=PROBABILITY_DTYPE,
    )


def source_outcome_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return _array_sha256([row.outcome for row in _canonical_rows(rows)], dtype="u1")


@dataclass(frozen=True, slots=True)
class SourceTrainingSurfaceReceipt:
    """Physical six-member receipt with no self-hash in any preimage."""

    contract: SourceSupervisionContractReceipt
    member_hashes: tuple[tuple[str, str], ...]
    row_order_sha256: str
    probability_matrix_sha256: str
    source_outcome_sha256: str
    pool_lineage_sha256: str
    compiled_surface_lineage_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, SourceSupervisionContractReceipt):
            raise ProtocolError("OE-PPUR v3 physical receipt lacks its source contract.")
        members = tuple(
            (str(member), _sha256(digest, role=f"member {member}"))
            for member, digest in self.member_hashes
        )
        if tuple(member for member, _ in members) != SOURCE_SUPERVISION_MEMBERS:
            raise ProtocolError("OE-PPUR v3 physical receipt member inventory drifted.")
        object.__setattr__(self, "member_hashes", members)
        for name in (
            "row_order_sha256",
            "probability_matrix_sha256",
            "source_outcome_sha256",
            "pool_lineage_sha256",
            "compiled_surface_lineage_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), role=name))
        object.__setattr__(
            self, "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_training_surface_receipt_v2",
                    "source_supervision_contract_hash": self.contract.contract_hash,
                    "member_hashes": members,
                    "row_order_sha256": self.row_order_sha256,
                    "probability_matrix_sha256": self.probability_matrix_sha256,
                    "source_outcome_sha256": self.source_outcome_sha256,
                    "pool_lineage_sha256": self.pool_lineage_sha256,
                    "compiled_surface_lineage_sha256": self.compiled_surface_lineage_sha256,
                    "counts": (
                        RAW_SOURCE_ROW_COUNT,
                        RAW_SOURCE_CASE_COUNT,
                        HELD_POOL_BLOCK_COUNT,
                        LOGICAL_SOURCE_ROW_COUNT,
                        LOGICAL_SOURCE_CASE_GROUP_COUNT,
                    ),
                    "matrix_shape": (
                        LOGICAL_SOURCE_ROW_COUNT,
                        PROBABILITY_COLUMN_COUNT,
                    ),
                    "matrix_dtype": PROBABILITY_DTYPE,
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )

    artifact_id = property(lambda self: self.contract.artifact_id)
    split = property(lambda self: self.contract.split)
    representation_id = property(lambda self: self.contract.representation_id)
    all_center_ids = property(lambda self: self.contract.all_center_ids)
    representation_feature_dim = property(lambda self: SOURCE_FEATURE_DIM)
    derived_feature_dim = property(lambda self: DERIVED_FEATURE_DIM)
    row_count = property(lambda self: LOGICAL_SOURCE_ROW_COUNT)
    case_count = property(lambda self: LOGICAL_SOURCE_CASE_GROUP_COUNT)
    target_rows_present = property(lambda self: False)
    target_labels_used = property(lambda self: False)


@dataclass(frozen=True, slots=True)
class SourceTrainingSurface:
    """Typed parsed bundle consumed by the canonical scientific service."""

    receipt: SourceTrainingSurfaceReceipt
    rows: tuple[SourceSupervisionRow, ...]
    held_pool_receipts: tuple[HeldCenterCandidatePoolReceipt, ...]
    compiled_surface_receipts: tuple[CompiledActionSurfaceReceipt, ...]
    compiler: PoolInvariantActionCompilerReceipt
    pool_lineage_hash: str = field(init=False)
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = _canonical_rows(self.rows)
        order = {center: index for index, center in enumerate(CENTERS)}
        pools = tuple(sorted(
            self.held_pool_receipts,
            key=lambda row: (order[row.outer_target_center], order[row.held_center]),
        ))
        surfaces = tuple(sorted(
            self.compiled_surface_receipts,
            key=lambda row: (order[row.outer_target_center], order[row.evaluated_center]),
        ))
        if (
            not isinstance(self.receipt, SourceTrainingSurfaceReceipt)
            or not isinstance(self.compiler, PoolInvariantActionCompilerReceipt)
            or len(rows) != LOGICAL_SOURCE_ROW_COUNT
            or len(pools) != HELD_POOL_BLOCK_COUNT
            or len(surfaces) != HELD_POOL_BLOCK_COUNT
            or self.receipt.contract.compiler_receipt_hash != self.compiler.receipt_hash
            or source_row_order_sha256(rows) != self.receipt.row_order_sha256
            or source_probability_matrix_sha256(rows)
            != self.receipt.probability_matrix_sha256
            or source_outcome_sha256(rows) != self.receipt.source_outcome_sha256
        ):
            raise ProtocolError("OE-PPUR v3 parsed source surface drifted from its receipt.")
        expected_keys = {(h, q) for h in CENTERS for q in CENTERS if q != h}
        pool_by_key = {(row.outer_target_center, row.held_center): row for row in pools}
        surface_by_key = {
            (row.outer_target_center, row.evaluated_center): row for row in surfaces
        }
        if set(pool_by_key) != expected_keys or set(surface_by_key) != expected_keys:
            raise ProtocolError("OE-PPUR v3 parsed source surface lacks exact 72 H/q blocks.")
        raw_by_index: dict[int, tuple[str, str, str, int, str]] = {}
        repeats: dict[int, int] = {}
        for row in rows:
            key = (row.outer_target_center, row.query_center)
            pool, surface = pool_by_key[key], surface_by_key[key]
            raw_identity = (
                row.query_center,
                row.source_row_id,
                row.case_id,
                row.outcome,
                row.split,
            )
            previous = raw_by_index.setdefault(row.source_cache_row_index, raw_identity)
            repeats[row.source_cache_row_index] = repeats.get(row.source_cache_row_index, 0) + 1
            if (
                previous != raw_identity
                or row.candidate_pool_receipt_hash != pool.receipt_hash
                or row.compiled_surface_receipt_hash != surface.receipt_hash
                or row.compiler_receipt_hash != self.compiler.receipt_hash
                or pool.source_supervision_contract_hash
                != self.receipt.contract.contract_hash
                or surface.pool_receipt_hash != pool.receipt_hash
                or surface.compiler_receipt_hash != self.compiler.receipt_hash
            ):
                raise ProtocolError("OE-PPUR v3 source row/pool/compiler lineage drifted.")
        if (
            set(raw_by_index) != set(range(RAW_SOURCE_ROW_COUNT))
            or set(repeats.values()) != {len(CENTERS) - 1}
            or len({(q, case) for q, _row, case, _outcome, _split in raw_by_index.values()})
            != RAW_SOURCE_CASE_COUNT
            or len({(row.outer_target_center, row.query_center, row.case_id) for row in rows})
            != LOGICAL_SOURCE_CASE_GROUP_COUNT
        ):
            raise ProtocolError("OE-PPUR v3 canonical source-cache expansion drifted.")
        pool_lineage = canonical_sha256(tuple(row.receipt_hash for row in pools))
        surface_lineage = canonical_sha256(tuple(row.receipt_hash for row in surfaces))
        if (
            pool_lineage != self.receipt.pool_lineage_sha256
            or surface_lineage != self.receipt.compiled_surface_lineage_sha256
        ):
            raise ProtocolError("OE-PPUR v3 parsed lineage hashes drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "held_pool_receipts", pools)
        object.__setattr__(self, "compiled_surface_receipts", surfaces)
        object.__setattr__(self, "pool_lineage_hash", pool_lineage)
        object.__setattr__(
            self, "surface_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_parsed_source_training_surface_v2",
                    "physical_receipt_hash": self.receipt.receipt_hash,
                    "pool_lineage_hash": pool_lineage,
                    "compiled_surface_lineage_hash": surface_lineage,
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )

    def rows_for_outer(self, outer_target_center: object) -> tuple[SourceSupervisionRow, ...]:
        h = str(outer_target_center)
        result = tuple(row for row in self.rows if row.outer_target_center == h)
        if not result:
            raise ProtocolError(f"OE-PPUR v3 source surface has no outer H={h}.")
        return result

    def rows_for_held_center(
        self, *, outer_target_center: object, held_center: object
    ) -> tuple[SourceSupervisionRow, ...]:
        h, q = str(outer_target_center), str(held_center)
        result = tuple(
            row for row in self.rows
            if row.outer_target_center == h and row.query_center == q
        )
        if not result:
            raise ProtocolError(f"OE-PPUR v3 source surface has no H/q={h}/{q}.")
        return result


def _validate_tree(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("OE-PPUR v3 source-supervision root is not a real directory.")
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("OE-PPUR v3 source-supervision tree contains a symlink.")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    if tuple(sorted(files)) != tuple(sorted(SOURCE_SUPERVISION_MEMBERS)):
        raise ProtocolError("OE-PPUR v3 source-supervision member inventory drifted.")
    return files


def _parse_content_index(files: Mapping[str, Path]) -> tuple[tuple[str, str], ...]:
    payload = _read_json(files["manifests/content_index.json"])
    _exact_keys(payload, ("schema_version", "members"), role="content index")
    if (
        payload["schema_version"] != "oe_ppur_v3_source_content_index_v1"
        or not isinstance(payload["members"], list)
    ):
        raise ProtocolError("OE-PPUR v3 source content-index schema drifted.")
    entries: list[tuple[str, str]] = []
    for value in payload["members"]:
        if not isinstance(value, dict):
            raise ProtocolError("OE-PPUR v3 source content-index entry is untyped.")
        _exact_keys(value, ("member", "sha256"), role="content-index entry")
        entries.append((
            str(value["member"]),
            _sha256(value["sha256"], role="indexed member hash"),
        ))
    if tuple(member for member, _ in entries) != INDEXED_MEMBERS:
        raise ProtocolError("OE-PPUR v3 content index must bind members 1-4 only.")
    for member, digest in entries:
        if _file_sha256(files[member]) != digest:
            raise ProtocolError(f"OE-PPUR v3 indexed source member drifted: {member}")
    return tuple(entries)


def _parse_manifest(
    files: Mapping[str, Path], compiler: PoolInvariantActionCompilerReceipt
) -> SourceSupervisionContractReceipt:
    payload = _read_json(files["manifests/source_training_surface.json"])
    contract = SourceSupervisionContractReceipt(
        compiler_receipt_hash=compiler.receipt_hash,
        producer_source_seal_sha256=str(payload.get("producer_source_seal_sha256")),
    )
    if payload != contract.manifest_payload():
        raise ProtocolError("OE-PPUR v3 source-training manifest drifted.")
    return contract


def _parse_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_ROW_COLUMNS:
                raise ProtocolError("OE-PPUR v3 source CSV header drifted.")
            values = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ProtocolError("OE-PPUR v3 source CSV is unreadable.") from exc
    if (
        len(values) != LOGICAL_SOURCE_ROW_COUNT
        or any(set(row) != set(SOURCE_ROW_COLUMNS) or None in row for row in values)
    ):
        raise ProtocolError("OE-PPUR v3 source CSV row count or width drifted.")
    return values


def _parse_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 source probability matrix is unreadable.") from exc
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (LOGICAL_SOURCE_ROW_COUNT, PROBABILITY_COLUMN_COUNT)
        or matrix.dtype.str != PROBABILITY_DTYPE
        or not matrix.flags.c_contiguous
        or not np.isfinite(matrix).all()
        or np.any((matrix < 0.0) | (matrix > 1.0))
    ):
        raise ProtocolError("OE-PPUR v3 source probability matrix contract drifted.")
    matrix.setflags(write=False)
    return matrix


def _parse_inventory(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ProtocolError("OE-PPUR v3 source expert inventory is untyped.")
    rows = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ProtocolError("OE-PPUR v3 source expert inventory entry is untyped.")
        _exact_keys(entry, ("expert_id", "source_center"), role="expert inventory")
        rows.append((str(entry["expert_id"]), str(entry["source_center"])))
    if tuple(sorted(center for _, center in rows)) != tuple(sorted(CENTERS)):
        raise ProtocolError("OE-PPUR v3 source expert inventory does not cover C.")
    return tuple(rows)


def _parse_pool_lineage(
    files: Mapping[str, Path],
    *,
    contract: SourceSupervisionContractReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
    csv_rows: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
) -> tuple[
    tuple[HeldCenterCandidatePoolReceipt, ...],
    tuple[CompiledActionSurfaceReceipt, ...],
]:
    payload = _read_json(files["manifests/source_pool_lineage.json"])
    _exact_keys(
        payload,
        (
            "schema_version", "source_supervision_contract_hash",
            "compiler_receipt_hash", "bank_lock_hash", "expert_inventory",
            "held_pool_policy", "blocks",
        ),
        role="source pool lineage",
    )
    if (
        payload["schema_version"] != "oe_ppur_v3_source_pool_lineage_v2"
        or payload["source_supervision_contract_hash"] != contract.contract_hash
        or payload["compiler_receipt_hash"] != compiler.receipt_hash
        or payload["bank_lock_hash"] != EXPECTED_BANK_LOCK_HASH
        or payload["held_pool_policy"] != "C_MINUS_H_MINUS_q"
        or not isinstance(payload["blocks"], list)
        or len(payload["blocks"]) != HELD_POOL_BLOCK_COUNT
    ):
        raise ProtocolError("OE-PPUR v3 source pool-lineage header drifted.")
    inventory = _parse_inventory(payload["expert_inventory"])
    pools: list[HeldCenterCandidatePoolReceipt] = []
    surfaces: list[CompiledActionSurfaceReceipt] = []
    cursor = ordinal = 0
    for h in CENTERS:
        for q in (center for center in CENTERS if center != h):
            raw = payload["blocks"][ordinal]
            if not isinstance(raw, dict):
                raise ProtocolError("OE-PPUR v3 source pool block is untyped.")
            _exact_keys(
                raw,
                (
                    "block_ordinal", "outer_target_center", "query_center",
                    "matrix_start", "matrix_stop", "row_count", "case_count",
                    "candidate_center_ids", "pool_receipt_hash",
                    "base_surface_sha256", "row_index_sha256",
                    "action_probability_hashes", "compiled_surface_receipt_hash",
                ),
                role="source pool block",
            )
            start, stop = int(raw["matrix_start"]), int(raw["matrix_stop"])
            block_csv = csv_rows[start:stop]
            if (
                int(raw["block_ordinal"]) != ordinal
                or raw["outer_target_center"] != h
                or raw["query_center"] != q
                or start != cursor
                or stop <= start
                or int(raw["row_count"]) != stop - start
                or len(block_csv) != stop - start
                or {row["outer_target_center"] for row in block_csv} != {h}
                or {row["query_center"] for row in block_csv} != {q}
                or int(raw["case_count"])
                != len({row["case_id"] for row in block_csv})
            ):
                raise ProtocolError("OE-PPUR v3 source pool block range/order drifted.")
            pool = build_held_center_candidate_pool(
                outer_target_center=h,
                held_center=q,
                all_center_ids=CENTERS,
                expert_inventory=inventory,
                bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
                source_supervision_contract_hash=contract.contract_hash,
                compiler=compiler,
            )
            if (
                list(pool.candidate_center_ids) != raw["candidate_center_ids"]
                or pool.receipt_hash != raw["pool_receipt_hash"]
            ):
                raise ProtocolError("OE-PPUR v3 source pool receipt drifted.")
            action_hashes = tuple(
                (
                    action_id,
                    _array_sha256(matrix[start:stop, index], dtype=PROBABILITY_DTYPE),
                )
                for index, action_id in enumerate(ALL_ACTION_IDS)
            )
            expected_hash_payload = [
                {"action_id": action_id, "sha256": digest}
                for action_id, digest in action_hashes
            ]
            row_ids = tuple(row["source_row_id"] for row in block_csv)
            surface = CompiledActionSurfaceReceipt(
                outer_target_center=h,
                evaluated_center=q,
                pool_receipt_hash=pool.receipt_hash,
                compiler_receipt_hash=compiler.receipt_hash,
                row_index_sha256=canonical_sha256(row_ids),
                base_surface_sha256=str(raw["base_surface_sha256"]),
                action_probability_hashes=action_hashes,
            )
            if (
                raw["row_index_sha256"] != surface.row_index_sha256
                or raw["action_probability_hashes"] != expected_hash_payload
                or raw["compiled_surface_receipt_hash"] != surface.receipt_hash
            ):
                raise ProtocolError("OE-PPUR v3 compiled source-surface lineage drifted.")
            pools.append(pool)
            surfaces.append(surface)
            cursor, ordinal = stop, ordinal + 1
    if cursor != LOGICAL_SOURCE_ROW_COUNT:
        raise ProtocolError("OE-PPUR v3 source pool ranges do not cover the matrix.")
    return tuple(pools), tuple(surfaces)


def _build_rows(
    csv_rows: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
    *,
    pools: Sequence[HeldCenterCandidatePoolReceipt],
    surfaces: Sequence[CompiledActionSurfaceReceipt],
    compiler: PoolInvariantActionCompilerReceipt,
) -> tuple[SourceSupervisionRow, ...]:
    pool_by_key = {(row.outer_target_center, row.held_center): row for row in pools}
    surface_by_key = {
        (row.outer_target_center, row.evaluated_center): row for row in surfaces
    }
    rows = []
    for expected_index, raw in enumerate(csv_rows):
        try:
            matrix_index = int(raw["matrix_row_index"])
            cache_index = int(raw["source_cache_row_index"])
            outcome = int(raw["outcome"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR v3 source CSV integer field drifted.") from exc
        h, q = raw["outer_target_center"], raw["query_center"]
        if matrix_index != expected_index or (h, q) not in pool_by_key:
            raise ProtocolError("OE-PPUR v3 source CSV canonical order drifted.")
        rows.append(SourceSupervisionRow(
            matrix_row_index=matrix_index,
            outer_target_center=h,
            query_center=q,
            source_cache_row_index=cache_index,
            source_row_id=raw["source_row_id"],
            case_id=raw["case_id"],
            split=raw["split"],
            outcome=outcome,
            action_probabilities=tuple(float(value) for value in matrix[matrix_index]),
            candidate_pool_receipt_hash=pool_by_key[(h, q)].receipt_hash,
            compiled_surface_receipt_hash=surface_by_key[(h, q)].receipt_hash,
            compiler_receipt_hash=compiler.receipt_hash,
        ))
    return tuple(rows)


def _validate_report(
    files: Mapping[str, Path],
    *,
    contract: SourceSupervisionContractReceipt,
    indexed: tuple[tuple[str, str], ...],
    rows: Sequence[SourceSupervisionRow],
    pools: Sequence[HeldCenterCandidatePoolReceipt],
    surfaces: Sequence[CompiledActionSurfaceReceipt],
) -> None:
    payload = _read_json(files["reports/validation_report.json"])
    expected = {
        "schema_version": "oe_ppur_v3_source_validation_report_v2",
        "status": "PASS",
        "source_supervision_contract_hash": contract.contract_hash,
        "content_index_sha256": _file_sha256(files["manifests/content_index.json"]),
        "indexed_member_hashes": [
            {"member": member, "sha256": digest} for member, digest in indexed
        ],
        "producer_source_seal_sha256": contract.producer_source_seal_sha256,
        "raw_source_row_count": RAW_SOURCE_ROW_COUNT,
        "raw_source_case_count": RAW_SOURCE_CASE_COUNT,
        "logical_block_count": HELD_POOL_BLOCK_COUNT,
        "logical_source_row_count": LOGICAL_SOURCE_ROW_COUNT,
        "logical_source_case_group_count": LOGICAL_SOURCE_CASE_GROUP_COUNT,
        "row_order_sha256": source_row_order_sha256(rows),
        "probability_matrix_sha256": source_probability_matrix_sha256(rows),
        "source_outcome_sha256": source_outcome_sha256(rows),
        "pool_lineage_sha256": canonical_sha256(
            tuple(row.receipt_hash for row in pools)
        ),
        "compiled_surface_lineage_sha256": canonical_sha256(
            tuple(row.receipt_hash for row in surfaces)
        ),
        "base_surfaces_sealed_by_producer": True,
        "source_outcomes_present": True,
        "target_rows_present": False,
        "target_labels_used": False,
    }
    if payload != expected:
        raise ProtocolError("OE-PPUR v3 source validation report drifted.")


def parse_source_training_bundle(
    root: str | Path,
    *,
    compiler: PoolInvariantActionCompilerReceipt,
) -> SourceTrainingSurface:
    """Parse and fully validate immutable source-only direct input number three."""

    if not isinstance(compiler, PoolInvariantActionCompilerReceipt):
        raise ProtocolError("OE-PPUR v3 source parser requires its typed compiler.")
    files = _validate_tree(Path(root))
    indexed = _parse_content_index(files)
    contract = _parse_manifest(files, compiler)
    csv_rows = _parse_csv(files["tables/source_rows.csv"])
    matrix = _parse_matrix(files["arrays/source_action_probabilities.npy"])
    pools, surfaces = _parse_pool_lineage(
        files,
        contract=contract,
        compiler=compiler,
        csv_rows=csv_rows,
        matrix=matrix,
    )
    rows = _build_rows(
        csv_rows, matrix, pools=pools, surfaces=surfaces, compiler=compiler
    )
    _validate_report(
        files,
        contract=contract,
        indexed=indexed,
        rows=rows,
        pools=pools,
        surfaces=surfaces,
    )
    member_hashes = tuple(
        (member, _file_sha256(files[member])) for member in SOURCE_SUPERVISION_MEMBERS
    )
    receipt = SourceTrainingSurfaceReceipt(
        contract=contract,
        member_hashes=member_hashes,
        row_order_sha256=source_row_order_sha256(rows),
        probability_matrix_sha256=source_probability_matrix_sha256(rows),
        source_outcome_sha256=source_outcome_sha256(rows),
        pool_lineage_sha256=canonical_sha256(tuple(row.receipt_hash for row in pools)),
        compiled_surface_lineage_sha256=canonical_sha256(
            tuple(row.receipt_hash for row in surfaces)
        ),
    )
    return SourceTrainingSurface(
        receipt=receipt,
        rows=rows,
        held_pool_receipts=pools,
        compiled_surface_receipts=surfaces,
        compiler=compiler,
    )


def build_source_training_surface(
    receipt: SourceTrainingSurfaceReceipt,
    rows: Sequence[SourceSupervisionRow],
    *,
    held_pool_receipts: Sequence[HeldCenterCandidatePoolReceipt],
    compiled_surface_receipts: Sequence[CompiledActionSurfaceReceipt],
    compiler: PoolInvariantActionCompilerReceipt,
) -> SourceTrainingSurface:
    """Pure builder for values already parsed and hash-validated by a producer."""

    return SourceTrainingSurface(
        receipt=receipt,
        rows=tuple(rows),
        held_pool_receipts=tuple(held_pool_receipts),
        compiled_surface_receipts=tuple(compiled_surface_receipts),
        compiler=compiler,
    )


validate_source_training_surface = build_source_training_surface


__all__ = (
    "DERIVED_FEATURE_DIM",
    "HELD_POOL_BLOCK_COUNT",
    "INDEXED_MEMBERS",
    "LOGICAL_SOURCE_CASE_GROUP_COUNT",
    "LOGICAL_SOURCE_ROW_COUNT",
    "PROBABILITY_COLUMN_COUNT",
    "PROBABILITY_DTYPE",
    "RAW_SOURCE_CASE_COUNT",
    "RAW_SOURCE_ROW_COUNT",
    "SOURCE_CACHE_FILE_HASHES",
    "SOURCE_ROW_COLUMNS",
    "SOURCE_SUPERVISION_MEMBERS",
    "SourceSupervisionContractReceipt",
    "SourceSupervisionRow",
    "SourceTrainingSurface",
    "SourceTrainingSurfaceReceipt",
    "build_source_training_surface",
    "parse_source_training_bundle",
    "source_outcome_sha256",
    "source_probability_matrix_sha256",
    "source_row_order_sha256",
    "validate_source_training_surface",
)
