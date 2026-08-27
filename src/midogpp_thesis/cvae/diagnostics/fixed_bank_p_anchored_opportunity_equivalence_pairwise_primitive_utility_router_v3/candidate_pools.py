"""Pool-indexed expert-candidate contracts for OE-PPUR v3.

The predecessor represented every source pseudo-target with the final
``C\\{H}`` inventory.  That is not a valid source-supervision surface: when
source center ``q`` supplies outcomes, its own expert must also be held out.
This module makes that distinction structural and keeps the final target pool
separate from every source held-center pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Sequence

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility.contracts import (
    CandidatePoolReceipt,
    canonical_sha256,
)


P_ACTION_ID = "P_PROTECTED"
ACTION_FAMILIES = ("B", "I", "R")
DIRECTIONS = ("zero_to_one", "one_to_zero")
CANDIDATE_ACTION_IDS = tuple(
    f"{family}::{direction}"
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
ALL_ACTION_IDS = (P_ACTION_ID, *CANDIDATE_ACTION_IDS)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BANK_LOCK = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


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


def _bank_lock_hash(value: object) -> str:
    result = _text(value, role="bank lock hash").lower()
    if _BANK_LOCK.fullmatch(result) is None:
        raise ProtocolError("OE-PPUR v3 bank lock hash is not canonical hex.")
    return result


def _centers(values: Iterable[object], *, role: str) -> tuple[str, ...]:
    raw = tuple(_text(value, role=role) for value in values)
    result = tuple(sorted(raw))
    if len(result) < 4 or len(set(result)) != len(result):
        raise ProtocolError(f"OE-PPUR v3 {role} inventory is invalid.")
    return result


def _inventory(
    values: Sequence[tuple[object, object]],
) -> tuple[tuple[str, str], ...]:
    result = tuple(
        sorted(
            (
                _text(expert_id, role="expert id"),
                _text(center_id, role="expert source center"),
            )
            for expert_id, center_id in values
        )
    )
    if (
        not result
        or len({expert_id for expert_id, _ in result}) != len(result)
        or len({center_id for _, center_id in result}) != len(result)
    ):
        raise ProtocolError(
            "OE-PPUR v3 requires one unique fixed-bank expert per candidate center."
        )
    return result


@dataclass(frozen=True, slots=True)
class PoolInvariantActionCompilerReceipt:
    """Frozen label-free rule shared by every held and final candidate pool.

    Pool-invariant means that the functional rule and constants are identical;
    it does *not* mean that outputs from different candidate inventories must
    be numerically identical.
    """

    protected_b_weight: float = 3.0 / 5.0
    protected_u_weight: float = 2.0 / 5.0
    threshold: float = 0.5
    action_ids: tuple[str, ...] = ALL_ACTION_IDS
    compiler_version: str = "OE_PPUR_V3_POOL_INVARIANT_COMPILER_V1"
    labels_used: bool = False
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.compiler_version != "OE_PPUR_V3_POOL_INVARIANT_COMPILER_V1"
            or tuple(self.action_ids) != ALL_ACTION_IDS
            or float(self.protected_b_weight) != 3.0 / 5.0
            or float(self.protected_u_weight) != 2.0 / 5.0
            or float(self.threshold) != 0.5
            or type(self.labels_used) is not bool
            or self.labels_used
        ):
            raise ProtocolError("OE-PPUR v3 action-compiler contract drifted.")
        object.__setattr__(self, "protected_b_weight", 3.0 / 5.0)
        object.__setattr__(self, "protected_u_weight", 2.0 / 5.0)
        object.__setattr__(self, "threshold", 0.5)
        object.__setattr__(self, "action_ids", ALL_ACTION_IDS)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_pool_invariant_action_compiler_v1",
                    "compiler_version": self.compiler_version,
                    "protected_formula": "P=0.6*B+0.4*U",
                    "I_zero_to_one": "rowwise_max_A1",
                    "I_one_to_zero": "rowwise_min_A1",
                    "R_both_directions": "rowwise_median_of_U_and_A1",
                    "direction_projection": (
                        "replace_P_only_on_candidate_crossing_at_threshold_0.5"
                    ),
                    "actions": ALL_ACTION_IDS,
                    "pool_invariant": True,
                    "permutation_equivariant": True,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class HeldCenterCandidatePoolReceipt:
    """Exact source pseudo-target pool ``C\\{H,q}``."""

    outer_target_center: str
    held_center: str
    all_center_ids: tuple[str, ...]
    candidate_center_ids: tuple[str, ...]
    expert_inventory: tuple[tuple[str, str], ...]
    bank_lock_hash: str
    source_supervision_contract_hash: str
    compiler_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = _text(self.outer_target_center, role="outer target H")
        q = _text(self.held_center, role="source held center q")
        all_centers = _centers(self.all_center_ids, role="bank center")
        raw_candidates = tuple(
            _text(value, role="held-pool candidate center")
            for value in self.candidate_center_ids
        )
        candidates = tuple(sorted(raw_candidates))
        inventory = _inventory(self.expert_inventory)
        expected = tuple(center for center in all_centers if center not in {h, q})
        if (
            h == q
            or h not in all_centers
            or q not in all_centers
            or len(set(raw_candidates)) != len(raw_candidates)
            or candidates != expected
            or tuple(sorted(center for _, center in inventory)) != expected
        ):
            raise ProtocolError(
                "OE-PPUR v3 source candidate pool is not exact C-minus-H-minus-q."
            )
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "held_center", q)
        object.__setattr__(self, "all_center_ids", all_centers)
        object.__setattr__(self, "candidate_center_ids", candidates)
        object.__setattr__(self, "expert_inventory", inventory)
        object.__setattr__(self, "bank_lock_hash", _bank_lock_hash(self.bank_lock_hash))
        object.__setattr__(
            self,
            "source_supervision_contract_hash",
            _sha256(
                self.source_supervision_contract_hash,
                role="source-supervision contract hash",
            ),
        )
        object.__setattr__(
            self,
            "compiler_receipt_hash",
            _sha256(self.compiler_receipt_hash, role="action compiler receipt hash"),
        )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_source_pool_C_minus_H_minus_q_v1",
                    "H": h,
                    "q": q,
                    "all_centers": all_centers,
                    "candidate_centers": candidates,
                    "expert_inventory": inventory,
                    "bank_lock_hash": self.bank_lock_hash,
                    "source_supervision_contract_hash": (
                        self.source_supervision_contract_hash
                    ),
                    "compiler_receipt_hash": self.compiler_receipt_hash,
                    "q_expert_excluded": True,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FinalOuterCandidatePoolReceipt:
    """Exact final target pool ``C\\{H}``, distinct from held-center pools."""

    outer_target_center: str
    all_center_ids: tuple[str, ...]
    candidate_center_ids: tuple[str, ...]
    expert_inventory: tuple[tuple[str, str], ...]
    bank_lock_hash: str
    source_supervision_contract_hash: str
    compiler_receipt_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = _text(self.outer_target_center, role="outer target H")
        all_centers = _centers(self.all_center_ids, role="bank center")
        raw_candidates = tuple(
            _text(value, role="final-pool candidate center")
            for value in self.candidate_center_ids
        )
        candidates = tuple(sorted(raw_candidates))
        inventory = _inventory(self.expert_inventory)
        expected = tuple(center for center in all_centers if center != h)
        if (
            h not in all_centers
            or len(set(raw_candidates)) != len(raw_candidates)
            or candidates != expected
            or tuple(sorted(center for _, center in inventory)) != expected
        ):
            raise ProtocolError("OE-PPUR v3 final candidate pool is not exact C-minus-H.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "all_center_ids", all_centers)
        object.__setattr__(self, "candidate_center_ids", candidates)
        object.__setattr__(self, "expert_inventory", inventory)
        object.__setattr__(self, "bank_lock_hash", _bank_lock_hash(self.bank_lock_hash))
        object.__setattr__(
            self,
            "source_supervision_contract_hash",
            _sha256(
                self.source_supervision_contract_hash,
                role="source-supervision contract hash",
            ),
        )
        object.__setattr__(
            self,
            "compiler_receipt_hash",
            _sha256(self.compiler_receipt_hash, role="action compiler receipt hash"),
        )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_final_pool_C_minus_H_v1",
                    "H": h,
                    "all_centers": all_centers,
                    "candidate_centers": candidates,
                    "expert_inventory": inventory,
                    "bank_lock_hash": self.bank_lock_hash,
                    "source_supervision_contract_hash": (
                        self.source_supervision_contract_hash
                    ),
                    "compiler_receipt_hash": self.compiler_receipt_hash,
                    "target_labels_used": False,
                }
            ),
        )

    def to_neutral(self) -> CandidatePoolReceipt:
        """Return the stage-neutral final-pool contract used for selection."""

        return CandidatePoolReceipt(
            outer_target_center=self.outer_target_center,
            all_center_ids=self.all_center_ids,
            candidate_center_ids=self.candidate_center_ids,
            expert_inventory=self.expert_inventory,
            bank_lock_hash=self.bank_lock_hash,
            source_surface_receipt_hash=self.source_supervision_contract_hash,
        )


@dataclass(frozen=True, slots=True)
class CompiledActionSurfaceReceipt:
    """Hash-only lineage for one compiler output on one exact pool."""

    outer_target_center: str
    evaluated_center: str
    pool_receipt_hash: str
    compiler_receipt_hash: str
    row_index_sha256: str
    base_surface_sha256: str
    action_probability_hashes: tuple[tuple[str, str], ...]
    labels_used: bool = False
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple(
            (str(action_id), _sha256(value, role="compiled probability hash"))
            for action_id, value in self.action_probability_hashes
        )
        if (
            tuple(action_id for action_id, _ in hashes) != ALL_ACTION_IDS
            or len({action_id for action_id, _ in hashes}) != len(ALL_ACTION_IDS)
            or type(self.labels_used) is not bool
            or self.labels_used
        ):
            raise ProtocolError("OE-PPUR v3 compiled action inventory drifted.")
        object.__setattr__(
            self, "outer_target_center", _text(self.outer_target_center, role="outer H")
        )
        object.__setattr__(
            self, "evaluated_center", _text(self.evaluated_center, role="evaluated center")
        )
        for name in (
            "pool_receipt_hash",
            "compiler_receipt_hash",
            "row_index_sha256",
            "base_surface_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), role=name.replace("_", " "))
            )
        object.__setattr__(self, "action_probability_hashes", hashes)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_compiled_action_surface_receipt_v1",
                    "H": self.outer_target_center,
                    "evaluated_center": self.evaluated_center,
                    "pool_receipt_hash": self.pool_receipt_hash,
                    "compiler_receipt_hash": self.compiler_receipt_hash,
                    "row_index_sha256": self.row_index_sha256,
                    "base_surface_sha256": self.base_surface_sha256,
                    "action_probability_hashes": hashes,
                    "labels_used": False,
                }
            ),
        )


def build_held_center_candidate_pool(
    *,
    outer_target_center: object,
    held_center: object,
    all_center_ids: Sequence[object],
    expert_inventory: Sequence[tuple[object, object]],
    bank_lock_hash: object,
    source_supervision_contract_hash: object,
    compiler: PoolInvariantActionCompilerReceipt,
) -> HeldCenterCandidatePoolReceipt:
    """Filter the fixed inventory to exact ``C\\{H,q}``."""

    if not isinstance(compiler, PoolInvariantActionCompilerReceipt):
        raise ProtocolError("OE-PPUR v3 held pool requires its typed compiler.")
    h, q = str(outer_target_center), str(held_center)
    candidates = tuple(
        str(center) for center in all_center_ids if str(center) not in {h, q}
    )
    candidate_set = set(candidates)
    inventory = tuple(
        (str(expert), str(center))
        for expert, center in expert_inventory
        if str(center) in candidate_set
    )
    return HeldCenterCandidatePoolReceipt(
        outer_target_center=h,
        held_center=q,
        all_center_ids=tuple(str(value) for value in all_center_ids),
        candidate_center_ids=candidates,
        expert_inventory=inventory,
        bank_lock_hash=str(bank_lock_hash),
        source_supervision_contract_hash=str(source_supervision_contract_hash),
        compiler_receipt_hash=compiler.receipt_hash,
    )


def build_final_outer_candidate_pool(
    *,
    outer_target_center: object,
    all_center_ids: Sequence[object],
    expert_inventory: Sequence[tuple[object, object]],
    bank_lock_hash: object,
    source_supervision_contract_hash: object,
    compiler: PoolInvariantActionCompilerReceipt,
) -> FinalOuterCandidatePoolReceipt:
    """Filter the fixed inventory to exact ``C\\{H}``."""

    if not isinstance(compiler, PoolInvariantActionCompilerReceipt):
        raise ProtocolError("OE-PPUR v3 final pool requires its typed compiler.")
    h = str(outer_target_center)
    candidates = tuple(str(center) for center in all_center_ids if str(center) != h)
    candidate_set = set(candidates)
    inventory = tuple(
        (str(expert), str(center))
        for expert, center in expert_inventory
        if str(center) in candidate_set
    )
    return FinalOuterCandidatePoolReceipt(
        outer_target_center=h,
        all_center_ids=tuple(str(value) for value in all_center_ids),
        candidate_center_ids=candidates,
        expert_inventory=inventory,
        bank_lock_hash=str(bank_lock_hash),
        source_supervision_contract_hash=str(source_supervision_contract_hash),
        compiler_receipt_hash=compiler.receipt_hash,
    )


def validate_complete_pool_lineage(
    held_pools: Sequence[HeldCenterCandidatePoolReceipt],
    *,
    final_pool: FinalOuterCandidatePoolReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
) -> tuple[HeldCenterCandidatePoolReceipt, ...]:
    """Prove one ``C\\{H,q}`` pool per legal source q plus final ``C\\{H}``."""

    rows = tuple(held_pools)
    if (
        not isinstance(final_pool, FinalOuterCandidatePoolReceipt)
        or not isinstance(compiler, PoolInvariantActionCompilerReceipt)
        or len(rows) != len(final_pool.candidate_center_ids)
        or any(not isinstance(row, HeldCenterCandidatePoolReceipt) for row in rows)
        or len({row.held_center for row in rows}) != len(rows)
        or {row.held_center for row in rows} != set(final_pool.candidate_center_ids)
        or any(row.outer_target_center != final_pool.outer_target_center for row in rows)
        or any(row.all_center_ids != final_pool.all_center_ids for row in rows)
        or any(row.bank_lock_hash != final_pool.bank_lock_hash for row in rows)
        or any(
            row.source_supervision_contract_hash
            != final_pool.source_supervision_contract_hash
            for row in rows
        )
        or any(row.compiler_receipt_hash != compiler.receipt_hash for row in rows)
        or final_pool.compiler_receipt_hash != compiler.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 pool-indexed lineage is incomplete or mixed.")
    return tuple(sorted(rows, key=lambda row: row.held_center))


__all__ = (
    "ACTION_FAMILIES",
    "ALL_ACTION_IDS",
    "CANDIDATE_ACTION_IDS",
    "DIRECTIONS",
    "P_ACTION_ID",
    "CompiledActionSurfaceReceipt",
    "FinalOuterCandidatePoolReceipt",
    "HeldCenterCandidatePoolReceipt",
    "PoolInvariantActionCompilerReceipt",
    "build_final_outer_candidate_pool",
    "build_held_center_candidate_pool",
    "validate_complete_pool_lineage",
)
