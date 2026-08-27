"""Pure validation of routed-family and exact-B generation budgets.

This module inspects frozen plans only.  It never loads an expert, samples a
latent, trains a classifier, or composes a replacement control.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import (
    SOURCE_BUDGET_PER_CLASS,
    TOTAL_PER_CLASS,
    GenerationLock,
)
from ...generation.generation import (
    equal_union_replicate_plan,
    source_generation_plan,
)
from ...protocol import ProtocolError
from .contracts import CandidateMenu


CANDIDATE_BUDGET_PER_CLASS = 1024
B_SOURCE_BUDGET_PER_CLASS = 128
B_SOURCE_COUNT = 8
B_TOTAL_PER_CLASS = 1024


@dataclass(frozen=True)
class ControlValidationReceipt:
    target_center: str
    generation_lock_hash: str
    candidate_menu_hash: str
    candidate_sources: tuple[str, ...]
    candidate_budget_per_class: int
    b_source_budget_per_class: int
    b_source_count: int
    b_total_per_class: int
    b_replicate_ids: tuple[str, ...]
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not self.target_center
            or not self.generation_lock_hash
            or not self.candidate_menu_hash
            or len(self.candidate_sources) != B_SOURCE_COUNT
            or self.target_center in self.candidate_sources
            or self.candidate_budget_per_class != CANDIDATE_BUDGET_PER_CLASS
            or self.b_source_budget_per_class != B_SOURCE_BUDGET_PER_CLASS
            or self.b_source_count != B_SOURCE_COUNT
            or self.b_total_per_class != B_TOTAL_PER_CLASS
            or len(self.b_replicate_ids) != 9
            or len(set(self.b_replicate_ids)) != 9
        ):
            raise ProtocolError("SCEPTRE control-validation receipt drifted.")
        unhashed = self._payload_without_hash()
        expected_hash = stable_hash(unhashed)
        if self.receipt_hash and self.receipt_hash != expected_hash:
            raise ProtocolError("SCEPTRE control-validation receipt hash drifted.")
        object.__setattr__(self, "receipt_hash", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_control_validation_v1",
            "target_center": self.target_center,
            "generation_lock_hash": self.generation_lock_hash,
            "candidate_menu_hash": self.candidate_menu_hash,
            "candidate_sources": list(self.candidate_sources),
            "candidate_budget_per_class": self.candidate_budget_per_class,
            "b_source_budget_per_class": self.b_source_budget_per_class,
            "b_source_count": self.b_source_count,
            "b_total_per_class": self.b_total_per_class,
            "b_replicate_ids": list(self.b_replicate_ids),
            "generation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_hash": self.receipt_hash}


def validate_candidate_and_b_control(
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> ControlValidationReceipt:
    """Validate 1024/class candidates and the canonical 8x128/class B plan."""

    if generation_lock.generation_lock_hash != candidate_menu.generation_lock_hash:
        raise ProtocolError("SCEPTRE control validation mixed GenerationLock identities.")
    stream_rows = source_generation_plan(generation_lock)
    stream_by_cell = {
        (row.source_center, row.training_seed, row.generation_seed): row
        for row in stream_rows
    }
    if len(stream_by_cell) != len(stream_rows):
        raise ProtocolError("SCEPTRE GenerationLock source streams are not unique.")

    expected_replica_grid = set(product(TRAINING_SEEDS, GENERATION_SEEDS))
    for family in candidate_menu.families:
        if {
            (key.training_seed, key.generation_seed) for key in family.stream_keys
        } != expected_replica_grid:
            raise ProtocolError("SCEPTRE routed-family seed grid drifted.")
        for key in family.stream_keys:
            frozen = stream_by_cell.get(
                (family.source_center, key.training_seed, key.generation_seed)
            )
            if frozen is None or frozen.to_payload() != key.to_payload():
                raise ProtocolError("SCEPTRE candidate stream differs from GenerationLock.")
            if (
                key.max_samples_per_class != CANDIDATE_BUDGET_PER_CLASS
                or key.equal_union_prefix_per_class != B_SOURCE_BUDGET_PER_CLASS
            ):
                raise ProtocolError("SCEPTRE candidate or B stream budget drifted.")

    target_rows = tuple(
        row
        for row in equal_union_replicate_plan(generation_lock)
        if row.target_center == candidate_menu.target_center
    )
    if len(target_rows) != 9 or {
        (row.training_seed, row.generation_seed) for row in target_rows
    } != expected_replica_grid:
        raise ProtocolError("SCEPTRE exact-B replicate coverage drifted.")
    ordered_rows = tuple(
        sorted(target_rows, key=lambda row: (row.training_seed, row.generation_seed))
    )
    for row in ordered_rows:
        expected_stream_ids = tuple(
            stream_by_cell[(source, row.training_seed, row.generation_seed)].stream_id
            for source in candidate_menu.candidate_sources
        )
        if (
            row.candidate_source_centers != candidate_menu.candidate_sources
            or row.source_stream_ids != expected_stream_ids
            or row.target_center in row.candidate_source_centers
            or row.source_budget_per_class != B_SOURCE_BUDGET_PER_CLASS
            or len(row.source_stream_ids) != B_SOURCE_COUNT
            or row.total_per_class != B_TOTAL_PER_CLASS
            or row.source_budget_per_class * len(row.source_stream_ids)
            != row.total_per_class
        ):
            raise ProtocolError("SCEPTRE exact-B composition or budget drifted.")
    if (
        SOURCE_BUDGET_PER_CLASS != B_SOURCE_BUDGET_PER_CLASS
        or TOTAL_PER_CLASS != B_TOTAL_PER_CLASS
    ):
        raise ProtocolError("SCEPTRE compiled generation-budget constants drifted.")

    return ControlValidationReceipt(
        target_center=candidate_menu.target_center,
        generation_lock_hash=generation_lock.generation_lock_hash,
        candidate_menu_hash=candidate_menu.menu_hash,
        candidate_sources=candidate_menu.candidate_sources,
        candidate_budget_per_class=CANDIDATE_BUDGET_PER_CLASS,
        b_source_budget_per_class=B_SOURCE_BUDGET_PER_CLASS,
        b_source_count=B_SOURCE_COUNT,
        b_total_per_class=B_TOTAL_PER_CLASS,
        b_replicate_ids=tuple(row.replicate_id for row in ordered_rows),
    )


validate_control_plan = validate_candidate_and_b_control
validate_exact_b_fallback = validate_candidate_and_b_control


__all__ = (
    "B_SOURCE_BUDGET_PER_CLASS",
    "B_SOURCE_COUNT",
    "B_TOTAL_PER_CLASS",
    "CANDIDATE_BUDGET_PER_CLASS",
    "ControlValidationReceipt",
    "validate_candidate_and_b_control",
    "validate_control_plan",
    "validate_exact_b_fallback",
)
