"""Adapt exact-nine fixed-bank cells into the historical-compatible action compiler."""

from __future__ import annotations

import numpy as np

from ....protocol import ProtocolError
from ....runtime.fixed_bank_a1_prediction_contracts import PredictionStore
from ..action_compiler import BasePredictionSurface, CompiledActionSurface, compile_action_surface
from ..candidate_pools import (
    FinalOuterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
)


def build_final_compiled_surface(
    store: PredictionStore,
    *,
    candidate_pool: FinalOuterCandidatePoolReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
) -> CompiledActionSurface:
    """Compile final ``C-minus-H`` probabilities without target labels."""

    if (
        type(store) is not PredictionStore
        or not isinstance(candidate_pool, FinalOuterCandidatePoolReceipt)
        or not isinstance(compiler, PoolInvariantActionCompilerReceipt)
    ):
        raise ProtocolError("OE-PPUR v4 final surface inputs are untyped.")
    target = candidate_pool.outer_target_center
    row_ids = tuple(store.rows_by_center[target])
    if not row_ids:
        raise ProtocolError("OE-PPUR v4 target probability rows are absent.")
    base = BasePredictionSurface(
        outer_target_center=target,
        evaluated_center=target,
        row_ids=row_ids,
        equal_union_probabilities=_float_tuple(store.exact_nine(target, "B")),
        union_probabilities=_float_tuple(store.exact_nine(target, "U")),
        expert_probabilities=tuple(
            (
                source,
                _float_tuple(store.exact_nine(target, f"A1::source={source}")),
            )
            for source in candidate_pool.candidate_center_ids
        ),
        candidate_pool_receipt_hash=candidate_pool.receipt_hash,
    )
    return compile_action_surface(
        base,
        candidate_pool=candidate_pool,
        compiler=compiler,
    )


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("OE-PPUR v4 exact-nine probability vector drifted.")
    return tuple(float(value) for value in array)


__all__ = ("build_final_compiled_surface",)
