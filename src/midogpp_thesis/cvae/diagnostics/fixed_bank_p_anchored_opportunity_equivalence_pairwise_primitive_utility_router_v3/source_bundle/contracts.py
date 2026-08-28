"""Typed immutable contracts for the OE-PPUR v3 source-only surface."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility.contracts import canonical_sha256
from ..candidate_pools import (
    ALL_ACTION_IDS,
    CompiledActionSurfaceReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
)
from ..identity import (
    CENTERS,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
)
from .constants import (
    DERIVED_FEATURE_DIM,
    HELD_POOL_BLOCK_COUNT,
    LOGICAL_SOURCE_CASE_GROUP_COUNT,
    LOGICAL_SOURCE_ROW_COUNT,
    PROBABILITY_COLUMN_COUNT,
    PROBABILITY_DTYPE,
    RAW_SOURCE_CASE_COUNT,
    RAW_SOURCE_ROW_COUNT,
    SOURCE_CACHE_ARTIFACT_ID,
    SOURCE_CACHE_FILE_HASHES,
    SOURCE_FEATURE_DIM,
    SOURCE_REPRESENTATION_ID,
    SOURCE_SPLIT,
    SOURCE_SUPERVISION_MEMBERS,
)
from .hashing import array_sha256, sha256, text


@dataclass(frozen=True, slots=True)
class SourceSupervisionContractReceipt:
    """Pre-output contract which breaks the pool/physical-receipt hash cycle."""

    compiler_receipt_hash: str
    producer_source_seal_sha256: str
    held_action_library_sha256: str
    held_mass_policy_receipt_sha256: str
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
        compiler_hash = sha256(self.compiler_receipt_hash, role="compiler receipt hash")
        producer_seal = sha256(self.producer_source_seal_sha256, role="producer source seal")
        action_library = sha256(
            self.held_action_library_sha256, role="held action library hash"
        )
        mass_policy = sha256(
            self.held_mass_policy_receipt_sha256,
            role="held mass policy receipt hash",
        )
        object.__setattr__(self, "compiler_receipt_hash", compiler_hash)
        object.__setattr__(self, "producer_source_seal_sha256", producer_seal)
        object.__setattr__(self, "held_action_library_sha256", action_library)
        object.__setattr__(
            self, "held_mass_policy_receipt_sha256", mass_policy
        )
        object.__setattr__(self, "all_center_ids", CENTERS)
        object.__setattr__(self, "exact_members", SOURCE_SUPERVISION_MEMBERS)
        object.__setattr__(self, "source_cache_file_hashes", SOURCE_CACHE_FILE_HASHES)
        object.__setattr__(
            self,
            "contract_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_supervision_contract_v4",
                    "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
                    "split": SOURCE_SPLIT,
                    "representation_id": SOURCE_REPRESENTATION_ID,
                    "all_centers": CENTERS,
                    "representation_feature_dim": SOURCE_FEATURE_DIM,
                    "derived_feature_dim": DERIVED_FEATURE_DIM,
                    "source_cache_artifact_id": SOURCE_CACHE_ARTIFACT_ID,
                    "source_cache_file_hashes": SOURCE_CACHE_FILE_HASHES,
                    "expert_bank": (EXPERT_BANK_ARTIFACT_ID, EXPECTED_BANK_LOCK_HASH, EXPECTED_BANK_CONTENT_INDEX_SHA256),
                    "generation_lock": (GENERATION_LOCK_ARTIFACT_ID, EXPECTED_GENERATION_LOCK_HASH, EXPECTED_GENERATION_CONTENT_INDEX_SHA256),
                    "producer_source_seal_sha256": producer_seal,
                    "compiler_receipt_hash": compiler_hash,
                    "held_action_library_sha256": action_library,
                    "held_mass_policy_receipt_sha256": mass_policy,
                    "exact_members": SOURCE_SUPERVISION_MEMBERS,
                    "raw_counts": (RAW_SOURCE_ROW_COUNT, RAW_SOURCE_CASE_COUNT),
                    "logical_counts": (HELD_POOL_BLOCK_COUNT, LOGICAL_SOURCE_ROW_COUNT, LOGICAL_SOURCE_CASE_GROUP_COUNT),
                    "matrix_contract": (LOGICAL_SOURCE_ROW_COUNT, PROBABILITY_COLUMN_COUNT, PROBABILITY_DTYPE),
                    "held_pool_policy": "C_MINUS_H_MINUS_q",
                    "final_pool_policy": "C_MINUS_H",
                    "source_outcomes_present": True,
                    "target_rows_present": False,
                    "target_labels_used": False,
                }
            ),
        )

    def manifest_payload(
        self, *, compiler_recomputation_receipt_sha256: object
    ) -> dict[str, object]:
        recomputation = sha256(
            compiler_recomputation_receipt_sha256,
            role="compiler recomputation receipt",
        )
        return {
            "schema_version": "oe_ppur_v3_source_training_surface_manifest_v4",
            "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
            "split": SOURCE_SPLIT,
            "representation_id": SOURCE_REPRESENTATION_ID,
            "representation_feature_dim": SOURCE_FEATURE_DIM,
            "derived_feature_dim": DERIVED_FEATURE_DIM,
            "all_center_ids": list(CENTERS),
            "source_cache_artifact_id": SOURCE_CACHE_ARTIFACT_ID,
            "source_cache_file_hashes": [{"member": member, "sha256": digest} for member, digest in SOURCE_CACHE_FILE_HASHES],
            "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
            "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_bank_content_index_sha256": EXPECTED_BANK_CONTENT_INDEX_SHA256,
            "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
            "generation_content_index_sha256": EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
            "producer_source_seal_sha256": self.producer_source_seal_sha256,
            "held_action_library_sha256": self.held_action_library_sha256,
            "held_mass_policy_receipt_sha256": (
                self.held_mass_policy_receipt_sha256
            ),
            "producer_compiler_recomputation_receipt_sha256": recomputation,
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
class BaseProbabilityLineageReceipt:
    """Per-(H,q) proof of the exact sealed label-free compiler inputs."""

    outer_target_center: str
    query_center: str
    pool_receipt_hash: str
    row_index_sha256: str
    base_surface_sha256: str
    base_probability_hashes: tuple[tuple[str, str], ...]
    producer_source_seal_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = text(self.outer_target_center, role="outer target H")
        q = text(self.query_center, role="query center q")
        if h not in CENTERS or q not in CENTERS or h == q:
            raise ProtocolError("OE-PPUR v3 base-probability H/q scope drifted.")
        hashes = tuple(
            (text(name, role="base probability id"), sha256(value, role="base probability hash"))
            for name, value in self.base_probability_hashes
        )
        expected_ids = ("B", "U", *(f"A1::source={c}" for c in CENTERS if c not in {h, q}))
        if tuple(name for name, _ in hashes) != expected_ids:
            raise ProtocolError("OE-PPUR v3 base-probability inventory/order drifted.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "query_center", q)
        object.__setattr__(self, "pool_receipt_hash", sha256(self.pool_receipt_hash, role="pool receipt hash"))
        object.__setattr__(self, "row_index_sha256", sha256(self.row_index_sha256, role="row index hash"))
        object.__setattr__(self, "base_surface_sha256", sha256(self.base_surface_sha256, role="base surface hash"))
        object.__setattr__(self, "base_probability_hashes", hashes)
        object.__setattr__(self, "producer_source_seal_sha256", sha256(self.producer_source_seal_sha256, role="producer source seal"))
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_base_probability_lineage_v1",
                    "H": h,
                    "q": q,
                    "pool_receipt_hash": self.pool_receipt_hash,
                    "row_index_sha256": self.row_index_sha256,
                    "base_surface_sha256": self.base_surface_sha256,
                    "base_probability_hashes": hashes,
                    "producer_source_seal_sha256": self.producer_source_seal_sha256,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceSupervisionRow:
    """One parsed matrix/CSV row; its outcome is source-train only."""

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
        matrix_index, cache_index = int(self.matrix_row_index), int(self.source_cache_row_index)
        h, q = text(self.outer_target_center, role="outer target H"), text(self.query_center, role="source query q")
        row_id, case_id = text(self.source_row_id, role="source row id"), text(self.case_id, role="source case id")
        outcome = int(self.outcome)
        probabilities = tuple(float(value) for value in self.action_probabilities)
        if (matrix_index < 0 or not 0 <= cache_index < RAW_SOURCE_ROW_COUNT or h not in CENTERS or q not in CENTERS or h == q or self.split != SOURCE_SPLIT or outcome not in (0, 1) or len(probabilities) != PROBABILITY_COLUMN_COUNT or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in probabilities)):
            raise ProtocolError("OE-PPUR v3 parsed source row violates its firewall.")
        for name, value in (("matrix_row_index", matrix_index), ("source_cache_row_index", cache_index), ("outer_target_center", h), ("query_center", q), ("source_row_id", row_id), ("case_id", case_id), ("outcome", outcome), ("action_probabilities", probabilities)):
            object.__setattr__(self, name, value)
        for name in ("candidate_pool_receipt_hash", "compiled_surface_receipt_hash", "compiler_receipt_hash"):
            object.__setattr__(self, name, sha256(getattr(self, name), role=name))
        object.__setattr__(self, "row_hash", canonical_sha256({"schema": "oe_ppur_v3_source_supervision_row_v3", "matrix_row_index": matrix_index, "H": h, "q": q, "source_cache_row_index": cache_index, "source_row_id": row_id, "case_id": case_id, "split": SOURCE_SPLIT, "outcome": outcome, "probabilities_f32_sha256": array_sha256(probabilities, dtype=PROBABILITY_DTYPE), "candidate_pool_receipt_hash": self.candidate_pool_receipt_hash, "compiled_surface_receipt_hash": self.compiled_surface_receipt_hash, "compiler_receipt_hash": self.compiler_receipt_hash, "target_label": False}))

    held_center = property(lambda self: self.query_center)
    source_center = property(lambda self: self.query_center)
    row_id = property(lambda self: self.source_row_id)


def canonical_rows(values: Sequence[SourceSupervisionRow]) -> tuple[SourceSupervisionRow, ...]:
    rows = tuple(sorted(tuple(values), key=lambda row: row.matrix_row_index))
    if not rows or any(not isinstance(row, SourceSupervisionRow) for row in rows) or tuple(r.matrix_row_index for r in rows) != tuple(range(len(rows))):
        raise ProtocolError("OE-PPUR v3 source-supervision row order is not canonical.")
    return rows


def source_row_order_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return canonical_sha256(tuple((r.matrix_row_index, r.outer_target_center, r.query_center, r.source_cache_row_index, r.source_row_id, r.case_id, r.split) for r in canonical_rows(rows)))


def source_probability_matrix_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return array_sha256([r.action_probabilities for r in canonical_rows(rows)], dtype=PROBABILITY_DTYPE)


def source_outcome_sha256(rows: Sequence[SourceSupervisionRow]) -> str:
    return array_sha256([r.outcome for r in canonical_rows(rows)], dtype="u1")


@dataclass(frozen=True, slots=True)
class SourceTrainingSurfaceReceipt:
    """Physical six-member receipt; no self-hash occurs in its preimage."""

    contract: SourceSupervisionContractReceipt
    member_hashes: tuple[tuple[str, str], ...]
    row_order_sha256: str
    probability_matrix_sha256: str
    source_outcome_sha256: str
    pool_lineage_sha256: str
    compiled_surface_lineage_sha256: str
    base_probability_lineage_sha256: str
    compiler_recomputation_receipt_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, SourceSupervisionContractReceipt):
            raise ProtocolError("OE-PPUR v3 physical receipt lacks its source contract.")
        members = tuple((str(member), sha256(digest, role=f"member {member}")) for member, digest in self.member_hashes)
        if tuple(member for member, _ in members) != SOURCE_SUPERVISION_MEMBERS:
            raise ProtocolError("OE-PPUR v3 physical receipt member inventory drifted.")
        object.__setattr__(self, "member_hashes", members)
        for name in ("row_order_sha256", "probability_matrix_sha256", "source_outcome_sha256", "pool_lineage_sha256", "compiled_surface_lineage_sha256", "base_probability_lineage_sha256", "compiler_recomputation_receipt_sha256"):
            object.__setattr__(self, name, sha256(getattr(self, name), role=name))
        object.__setattr__(self, "receipt_hash", canonical_sha256({"schema": "oe_ppur_v3_source_training_surface_receipt_v4", "source_supervision_contract_hash": self.contract.contract_hash, "member_hashes": members, "row_order_sha256": self.row_order_sha256, "probability_matrix_sha256": self.probability_matrix_sha256, "source_outcome_sha256": self.source_outcome_sha256, "pool_lineage_sha256": self.pool_lineage_sha256, "compiled_surface_lineage_sha256": self.compiled_surface_lineage_sha256, "base_probability_lineage_sha256": self.base_probability_lineage_sha256, "compiler_recomputation_receipt_sha256": self.compiler_recomputation_receipt_sha256, "held_action_library_sha256": self.contract.held_action_library_sha256, "held_mass_policy_receipt_sha256": self.contract.held_mass_policy_receipt_sha256, "counts": (RAW_SOURCE_ROW_COUNT, RAW_SOURCE_CASE_COUNT, HELD_POOL_BLOCK_COUNT, LOGICAL_SOURCE_ROW_COUNT, LOGICAL_SOURCE_CASE_GROUP_COUNT), "matrix_shape": (LOGICAL_SOURCE_ROW_COUNT, PROBABILITY_COLUMN_COUNT), "matrix_dtype": PROBABILITY_DTYPE, "target_rows_present": False, "target_labels_used": False}))

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
    base_probability_lineage_receipts: tuple[BaseProbabilityLineageReceipt, ...]
    compiler: PoolInvariantActionCompilerReceipt
    pool_lineage_hash: str = field(init=False)
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, SourceTrainingSurfaceReceipt) or not isinstance(self.compiler, PoolInvariantActionCompilerReceipt):
            raise ProtocolError("OE-PPUR v3 parsed surface lacks typed receipts.")
        rows, pools, surfaces, bases = canonical_rows(self.rows), tuple(self.held_pool_receipts), tuple(self.compiled_surface_receipts), tuple(self.base_probability_lineage_receipts)
        expected_keys = tuple((h, q) for h in CENTERS for q in CENTERS if q != h)
        if len(rows) != LOGICAL_SOURCE_ROW_COUNT or tuple((p.outer_target_center, p.held_center) for p in pools) != expected_keys or tuple((s.outer_target_center, s.evaluated_center) for s in surfaces) != expected_keys or tuple((b.outer_target_center, b.query_center) for b in bases) != expected_keys:
            raise ProtocolError("OE-PPUR v3 parsed source topology drifted.")
        pool_by_key = {(p.outer_target_center, p.held_center): p for p in pools}
        surface_by_key = {(s.outer_target_center, s.evaluated_center): s for s in surfaces}
        base_by_key = {(b.outer_target_center, b.query_center): b for b in bases}
        raw_by_index: dict[int, tuple[str, str, str, int, str]] = {}
        repeats: dict[int, int] = {}
        for row in rows:
            key = (row.outer_target_center, row.query_center)
            pool, surface, base = pool_by_key[key], surface_by_key[key], base_by_key[key]
            raw_identity = (row.query_center, row.source_row_id, row.case_id, row.outcome, row.split)
            previous = raw_by_index.setdefault(row.source_cache_row_index, raw_identity)
            repeats[row.source_cache_row_index] = repeats.get(row.source_cache_row_index, 0) + 1
            if previous != raw_identity or row.candidate_pool_receipt_hash != pool.receipt_hash or row.compiled_surface_receipt_hash != surface.receipt_hash or row.compiler_receipt_hash != self.compiler.receipt_hash or pool.source_supervision_contract_hash != self.receipt.contract.contract_hash or surface.pool_receipt_hash != pool.receipt_hash or surface.compiler_receipt_hash != self.compiler.receipt_hash or surface.base_surface_sha256 != base.base_surface_sha256 or base.pool_receipt_hash != pool.receipt_hash:
                raise ProtocolError("OE-PPUR v3 source row/pool/compiler lineage drifted.")
        if set(raw_by_index) != set(range(RAW_SOURCE_ROW_COUNT)) or set(repeats.values()) != {len(CENTERS) - 1} or len({(q, case) for q, _row, case, _outcome, _split in raw_by_index.values()}) != RAW_SOURCE_CASE_COUNT or len({(r.outer_target_center, r.query_center, r.case_id) for r in rows}) != LOGICAL_SOURCE_CASE_GROUP_COUNT:
            raise ProtocolError("OE-PPUR v3 canonical source-cache expansion drifted.")
        pool_hash, surface_hash, base_hash = canonical_sha256(tuple(p.receipt_hash for p in pools)), canonical_sha256(tuple(s.receipt_hash for s in surfaces)), canonical_sha256(tuple(b.receipt_hash for b in bases))
        if pool_hash != self.receipt.pool_lineage_sha256 or surface_hash != self.receipt.compiled_surface_lineage_sha256 or base_hash != self.receipt.base_probability_lineage_sha256:
            raise ProtocolError("OE-PPUR v3 parsed lineage hashes drifted.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "held_pool_receipts", pools)
        object.__setattr__(self, "compiled_surface_receipts", surfaces)
        object.__setattr__(self, "base_probability_lineage_receipts", bases)
        object.__setattr__(self, "pool_lineage_hash", pool_hash)
        object.__setattr__(self, "surface_hash", canonical_sha256({"schema": "oe_ppur_v3_parsed_source_training_surface_v4", "physical_receipt_hash": self.receipt.receipt_hash, "pool_lineage_hash": pool_hash, "compiled_surface_lineage_hash": surface_hash, "base_probability_lineage_hash": base_hash, "held_action_library_sha256": self.receipt.contract.held_action_library_sha256, "held_mass_policy_receipt_sha256": self.receipt.contract.held_mass_policy_receipt_sha256, "target_rows_present": False, "target_labels_used": False}))

    def rows_for_outer(self, outer_target_center: object) -> tuple[SourceSupervisionRow, ...]:
        h = str(outer_target_center)
        result = tuple(row for row in self.rows if row.outer_target_center == h)
        if not result:
            raise ProtocolError(f"OE-PPUR v3 source surface has no outer H={h}.")
        return result

    def rows_for_held_center(self, *, outer_target_center: object, held_center: object) -> tuple[SourceSupervisionRow, ...]:
        h, q = str(outer_target_center), str(held_center)
        result = tuple(row for row in self.rows if row.outer_target_center == h and row.query_center == q)
        if not result:
            raise ProtocolError(f"OE-PPUR v3 source surface has no H/q={h}/{q}.")
        return result


__all__ = (
    "BaseProbabilityLineageReceipt",
    "SourceSupervisionContractReceipt",
    "SourceSupervisionRow",
    "SourceTrainingSurface",
    "SourceTrainingSurfaceReceipt",
    "canonical_rows",
    "source_outcome_sha256",
    "source_probability_matrix_sha256",
    "source_row_order_sha256",
)
