"""Deterministic, label-free, pool-invariant action compilation.

The compiler consumes only probability surfaces and an exact candidate-pool
receipt.  Its output is row-permutation equivariant and expert-order invariant;
source labels are neither accepted nor reachable through this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.pairwise_primitive_utility.contracts import ActionSurface, canonical_sha256
from .candidate_pools import (
    ALL_ACTION_IDS,
    CANDIDATE_ACTION_IDS,
    DIRECTIONS,
    P_ACTION_ID,
    CompiledActionSurfaceReceipt,
    FinalOuterCandidatePoolReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
)


CandidatePoolV4 = HeldCenterCandidatePoolReceipt | FinalOuterCandidatePoolReceipt


def _probabilities(
    values: Sequence[float], *, expected_length: int | None = None, role: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (
        not result
        or (expected_length is not None and len(result) != expected_length)
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result)
    ):
        raise ProtocolError(f"OE-PPUR v4 historical-lineage {role} probability vector drifted.")
    return result


def _array_sha256(values: object, *, dtype: str = "<f4") -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


@dataclass(frozen=True, slots=True)
class BasePredictionSurface:
    """The label-free B/U/A1 inputs for one evaluated center and case shard."""

    outer_target_center: str
    evaluated_center: str
    row_ids: tuple[str, ...]
    equal_union_probabilities: tuple[float, ...]
    union_probabilities: tuple[float, ...]
    expert_probabilities: tuple[tuple[str, tuple[float, ...]], ...]
    candidate_pool_receipt_hash: str
    labels_present: bool = False
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_center).strip()
        q = str(self.evaluated_center).strip()
        row_ids = tuple(str(value).strip() for value in self.row_ids)
        if (
            not h
            or not q
            or not row_ids
            or any(not value for value in row_ids)
            or len(set(row_ids)) != len(row_ids)
            or type(self.labels_present) is not bool
            or self.labels_present
        ):
            raise ProtocolError("OE-PPUR v4 historical-lineage base probability-surface identity drifted.")
        b = _probabilities(
            self.equal_union_probabilities,
            expected_length=len(row_ids),
            role="equal-union B",
        )
        u = _probabilities(
            self.union_probabilities,
            expected_length=len(row_ids),
            role="union U",
        )
        experts = tuple(
            sorted(
                (
                    str(center).strip(),
                    _probabilities(
                        values,
                        expected_length=len(row_ids),
                        role=f"A1 source {center}",
                    ),
                )
                for center, values in self.expert_probabilities
            )
        )
        if (
            not experts
            or any(not center for center, _ in experts)
            or len({center for center, _ in experts}) != len(experts)
        ):
            raise ProtocolError("OE-PPUR v4 historical-lineage base A1 expert inventory drifted.")
        pool_hash = str(self.candidate_pool_receipt_hash).strip().lower()
        if len(pool_hash) != 64 or any(value not in "0123456789abcdef" for value in pool_hash):
            raise ProtocolError("OE-PPUR v4 historical-lineage base surface pool hash drifted.")
        matrix = np.column_stack(
            (
                np.asarray(b, dtype=np.float64),
                np.asarray(u, dtype=np.float64),
                *(np.asarray(values, dtype=np.float64) for _, values in experts),
            )
        )
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "evaluated_center", q)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "equal_union_probabilities", b)
        object.__setattr__(self, "union_probabilities", u)
        object.__setattr__(self, "expert_probabilities", experts)
        object.__setattr__(self, "candidate_pool_receipt_hash", pool_hash)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_base_prediction_surface_v1",
                    "H": h,
                    "evaluated_center": q,
                    "row_ids": row_ids,
                    "candidate_pool_receipt_hash": pool_hash,
                    "expert_center_ids": tuple(center for center, _ in experts),
                    "matrix_f32_sha256": _array_sha256(matrix),
                    "labels_present": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledActionSurface:
    """Seven immutable probability columns in canonical row/action order."""

    row_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    probabilities_by_action: tuple[tuple[str, tuple[float, ...]], ...]
    receipt: CompiledActionSurfaceReceipt
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(str(value) for value in self.row_ids)
        actions = tuple(self.action_ids)
        values = tuple(
            (
                str(action_id),
                _probabilities(
                    probabilities,
                    expected_length=len(rows),
                    role=f"compiled {action_id}",
                ),
            )
            for action_id, probabilities in self.probabilities_by_action
        )
        if (
            not isinstance(self.receipt, CompiledActionSurfaceReceipt)
            or actions != ALL_ACTION_IDS
            or tuple(action for action, _ in values) != ALL_ACTION_IDS
            or not rows
            or len(set(rows)) != len(rows)
        ):
            raise ProtocolError("OE-PPUR v4 historical-lineage compiled action surface drifted.")
        matrix = np.column_stack(
            tuple(np.asarray(column, dtype=np.float32) for _, column in values)
        )
        action_hashes = tuple(
            (action, _array_sha256(column)) for action, column in values
        )
        row_hash = canonical_sha256(rows)
        if (
            self.receipt.row_index_sha256 != row_hash
            or self.receipt.action_probability_hashes != action_hashes
        ):
            raise ProtocolError("OE-PPUR v4 historical-lineage compiled surface differs from its receipt.")
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "probabilities_by_action", values)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_compiled_action_surface_v1",
                    "receipt_hash": self.receipt.receipt_hash,
                    "matrix_f32_sha256": _array_sha256(matrix),
                    "shape": matrix.shape,
                    "dtype": "<f4",
                }
            ),
        )

    def probabilities(self, action_id: object) -> tuple[float, ...]:
        key = str(action_id)
        for action, values in self.probabilities_by_action:
            if action == key:
                return values
        raise ProtocolError(f"Unknown OE-PPUR v4 historical-lineage compiled action: {key}")

    def probability_matrix(self, *, dtype: str = "<f4") -> np.ndarray:
        """Materialize row-major ``N x 7`` bytes for workstation shards."""

        if np.dtype(dtype) not in (np.dtype("<f4"), np.dtype("<f8")):
            raise ProtocolError("OE-PPUR v4 historical-lineage compiled matrix dtype must be little-endian f4/f8.")
        result = np.ascontiguousarray(
            np.column_stack(
                tuple(
                    np.asarray(values, dtype=np.dtype(dtype))
                    for _, values in self.probabilities_by_action
                )
            ),
            dtype=np.dtype(dtype),
        )
        result.setflags(write=False)
        return result

    def candidate_surfaces(self) -> tuple[ActionSurface, ...]:
        """Adapt the six challengers to the stage-neutral opportunity core."""

        result = []
        for action_id in CANDIDATE_ACTION_IDS:
            family, direction = action_id.split("::", maxsplit=1)
            result.append(
                ActionSurface(
                    action_id=action_id,
                    family=family,
                    direction=direction,
                    probabilities=self.probabilities(action_id),
                )
            )
        return tuple(result)


def canonical_compiler_receipt() -> PoolInvariantActionCompilerReceipt:
    """Return the hash-exact historical compiler rule used by v4's alias."""

    return PoolInvariantActionCompilerReceipt()


def _direction_project(
    protected: np.ndarray,
    candidate: np.ndarray,
    *,
    direction: str,
    threshold: float,
) -> np.ndarray:
    if direction == "zero_to_one":
        crossing = (protected < threshold) & (candidate >= threshold)
    elif direction == "one_to_zero":
        crossing = (protected >= threshold) & (candidate < threshold)
    else:
        raise ProtocolError("OE-PPUR v4 historical-lineage direction is outside the frozen menu.")
    return np.where(crossing, candidate, protected)


def compile_action_surface(
    base: BasePredictionSurface,
    *,
    candidate_pool: CandidatePoolV4,
    compiler: PoolInvariantActionCompilerReceipt,
) -> CompiledActionSurface:
    """Compile exact P and six directional challengers without labels."""

    if (
        not isinstance(base, BasePredictionSurface)
        or not isinstance(
            candidate_pool,
            (HeldCenterCandidatePoolReceipt, FinalOuterCandidatePoolReceipt),
        )
        or not isinstance(compiler, PoolInvariantActionCompilerReceipt)
    ):
        raise ProtocolError("OE-PPUR v4 historical-lineage compiler requires typed label-free inputs.")
    expected_evaluated = (
        candidate_pool.held_center
        if isinstance(candidate_pool, HeldCenterCandidatePoolReceipt)
        else candidate_pool.outer_target_center
    )
    expert_centers = tuple(center for center, _ in base.expert_probabilities)
    if (
        base.outer_target_center != candidate_pool.outer_target_center
        or base.evaluated_center != expected_evaluated
        or base.candidate_pool_receipt_hash != candidate_pool.receipt_hash
        or expert_centers != candidate_pool.candidate_center_ids
        or candidate_pool.compiler_receipt_hash != compiler.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 historical-lineage compiler input/pool lineage drifted.")

    b = np.asarray(base.equal_union_probabilities, dtype=np.float64)
    u = np.asarray(base.union_probabilities, dtype=np.float64)
    experts = np.asarray(
        [values for _, values in base.expert_probabilities], dtype=np.float64
    )
    protected = compiler.protected_b_weight * b + compiler.protected_u_weight * u
    candidate_by_family_direction = {
        ("B", "zero_to_one"): b,
        ("B", "one_to_zero"): b,
        ("I", "zero_to_one"): np.max(experts, axis=0),
        ("I", "one_to_zero"): np.min(experts, axis=0),
    }
    robust = np.median(np.vstack((u[None, :], experts)), axis=0)
    for direction in DIRECTIONS:
        candidate_by_family_direction[("R", direction)] = robust

    compiled: list[tuple[str, tuple[float, ...]]] = [
        (P_ACTION_ID, tuple(float(value) for value in protected))
    ]
    for action_id in CANDIDATE_ACTION_IDS:
        family, direction = action_id.split("::", maxsplit=1)
        projected = _direction_project(
            protected,
            candidate_by_family_direction[(family, direction)],
            direction=direction,
            threshold=compiler.threshold,
        )
        compiled.append((action_id, tuple(float(value) for value in projected)))
    action_hashes = tuple(
        (action_id, _array_sha256(values)) for action_id, values in compiled
    )
    receipt = CompiledActionSurfaceReceipt(
        outer_target_center=base.outer_target_center,
        evaluated_center=base.evaluated_center,
        pool_receipt_hash=candidate_pool.receipt_hash,
        compiler_receipt_hash=compiler.receipt_hash,
        row_index_sha256=canonical_sha256(base.row_ids),
        base_surface_sha256=base.surface_hash,
        action_probability_hashes=action_hashes,
    )
    return CompiledActionSurface(
        row_ids=base.row_ids,
        action_ids=ALL_ACTION_IDS,
        probabilities_by_action=tuple(compiled),
        receipt=receipt,
    )


__all__ = (
    "BasePredictionSurface",
    "CompiledActionSurface",
    "canonical_compiler_receipt",
    "compile_action_surface",
)
