"""Label-free producer and recomputation receipts for direct input #3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility.contracts import canonical_sha256
from ..action_compiler import (
    BasePredictionSurface,
    CandidatePoolV4,
    CompiledActionSurface,
    compile_action_surface,
)
from ..candidate_pools import (
    CompiledActionSurfaceReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
)
from ..identity import CENTERS
from .constants import PROBABILITY_DTYPE
from .contracts import (
    BaseProbabilityLineageReceipt,
    SourceSupervisionRow,
    SourceTrainingSurface,
    SourceTrainingSurfaceReceipt,
)
from .hashing import array_sha256, sha256


@dataclass(frozen=True, slots=True)
class CompilerRecomputationReceipt:
    """Aggregate proof derived from all exact typed base/compiled block pairs."""

    producer_source_seal_sha256: str
    compiler_receipt_hash: str
    block_pairs: tuple[tuple[BaseProbabilityLineageReceipt, CompiledActionSurfaceReceipt], ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        producer_seal = sha256(self.producer_source_seal_sha256, role="producer source seal")
        compiler_hash = sha256(self.compiler_receipt_hash, role="compiler receipt hash")
        pairs = tuple(self.block_pairs)
        exact_keys = tuple((h, q) for h in CENTERS for q in CENTERS if q != h)
        if (
            len(pairs) != len(exact_keys)
            or any(not isinstance(base, BaseProbabilityLineageReceipt) or not isinstance(compiled, CompiledActionSurfaceReceipt) for base, compiled in pairs)
            or tuple((base.outer_target_center, base.query_center) for base, _compiled in pairs) != exact_keys
        ):
            raise ProtocolError("OE-PPUR v4 historical-lineage compiler recomputation block inventory drifted.")
        for base, compiled in pairs:
            if (
                compiled.outer_target_center != base.outer_target_center
                or compiled.evaluated_center != base.query_center
                or compiled.pool_receipt_hash != base.pool_receipt_hash
                or compiled.row_index_sha256 != base.row_index_sha256
                or compiled.base_surface_sha256 != base.base_surface_sha256
                or compiled.compiler_receipt_hash != compiler_hash
                or base.producer_source_seal_sha256 != producer_seal
            ):
                raise ProtocolError("OE-PPUR v4 historical-lineage compiler recomputation pair lineage drifted.")
        object.__setattr__(self, "producer_source_seal_sha256", producer_seal)
        object.__setattr__(self, "compiler_receipt_hash", compiler_hash)
        object.__setattr__(self, "block_pairs", pairs)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_sha256(
                {
                    "schema": "oe_ppur_v3_compiler_recomputation_receipt_v1",
                    "producer_source_seal_sha256": producer_seal,
                    "compiler_receipt_hash": compiler_hash,
                    "block_pairs": tuple(
                        (base.receipt_hash, compiled.receipt_hash)
                        for base, compiled in pairs
                    ),
                    "block_count": len(exact_keys),
                    "compiler_inputs": "SEALED_B_U_A1_PROBABILITY_ARRAYS",
                    "labels_seen_by_compiler": False,
                }
            ),
        )


# Backward-compatible descriptive name without a second trust-bearing type.
SourceBundleProducerReceipt = CompilerRecomputationReceipt


def compile_verified_source_block(
    base: BasePredictionSurface,
    *,
    candidate_pool: CandidatePoolV4,
    compiler: PoolInvariantActionCompilerReceipt,
    producer_source_seal_sha256: object,
) -> tuple[BaseProbabilityLineageReceipt, CompiledActionSurface]:
    """Compile actual typed B/U/A1 arrays before outcomes can be attached."""

    if not isinstance(base, BasePredictionSurface) or base.labels_present:
        raise ProtocolError("OE-PPUR v4 historical-lineage producer requires a typed label-free base surface.")
    compiled = compile_action_surface(base, candidate_pool=candidate_pool, compiler=compiler)
    base_hashes = (
        ("B", array_sha256(base.equal_union_probabilities, dtype=PROBABILITY_DTYPE)),
        ("U", array_sha256(base.union_probabilities, dtype=PROBABILITY_DTYPE)),
        *(
            (f"A1::source={center}", array_sha256(values, dtype=PROBABILITY_DTYPE))
            for center, values in base.expert_probabilities
        ),
    )
    lineage = BaseProbabilityLineageReceipt(
        outer_target_center=base.outer_target_center,
        query_center=base.evaluated_center,
        pool_receipt_hash=candidate_pool.receipt_hash,
        row_index_sha256=compiled.receipt.row_index_sha256,
        base_surface_sha256=base.surface_hash,
        base_probability_hashes=base_hashes,
        producer_source_seal_sha256=str(producer_source_seal_sha256),
    )
    if compiled.receipt.base_surface_sha256 != lineage.base_surface_sha256:
        raise ProtocolError("OE-PPUR v4 historical-lineage producer compiler lost its base-surface lineage.")
    return lineage, compiled


def reconstruct_compiler_recomputation_receipt(
    *,
    producer_source_seal_sha256: object,
    compiler: PoolInvariantActionCompilerReceipt,
    base_probability_lineage_receipts: Sequence[BaseProbabilityLineageReceipt],
    compiled_surface_receipts: Sequence[CompiledActionSurfaceReceipt],
) -> CompilerRecomputationReceipt:
    """Reconstruct the aggregate receipt from canonical typed block receipts."""

    if not isinstance(compiler, PoolInvariantActionCompilerReceipt):
        raise ProtocolError("OE-PPUR v4 historical-lineage recomputation receipt requires its typed compiler.")
    bases = tuple(base_probability_lineage_receipts)
    compiled = tuple(compiled_surface_receipts)
    if len(bases) != len(compiled):
        raise ProtocolError("OE-PPUR v4 historical-lineage recomputation base/compiled counts differ.")
    return CompilerRecomputationReceipt(
        producer_source_seal_sha256=str(producer_source_seal_sha256),
        compiler_receipt_hash=compiler.receipt_hash,
        block_pairs=tuple(zip(bases, compiled, strict=True)),
    )


def build_source_training_surface(
    receipt: SourceTrainingSurfaceReceipt,
    rows: Sequence[SourceSupervisionRow],
    *,
    held_pool_receipts: Sequence[HeldCenterCandidatePoolReceipt],
    compiled_surface_receipts: Sequence[CompiledActionSurfaceReceipt],
    base_probability_lineage_receipts: Sequence[BaseProbabilityLineageReceipt],
    compiler: PoolInvariantActionCompilerReceipt,
) -> SourceTrainingSurface:
    """Build from values already parsed and hash-validated by a sealed producer."""

    recomputation = reconstruct_compiler_recomputation_receipt(
        producer_source_seal_sha256=receipt.contract.producer_source_seal_sha256,
        compiler=compiler,
        base_probability_lineage_receipts=base_probability_lineage_receipts,
        compiled_surface_receipts=compiled_surface_receipts,
    )
    if recomputation.receipt_hash != receipt.compiler_recomputation_receipt_sha256:
        raise ProtocolError("OE-PPUR v4 historical-lineage physical receipt lost compiler recomputation lineage.")
    return SourceTrainingSurface(
        receipt=receipt,
        rows=tuple(rows),
        held_pool_receipts=tuple(held_pool_receipts),
        compiled_surface_receipts=tuple(compiled_surface_receipts),
        base_probability_lineage_receipts=tuple(base_probability_lineage_receipts),
        compiler=compiler,
    )


validate_source_training_surface = build_source_training_surface


__all__ = (
    "CompilerRecomputationReceipt",
    "SourceBundleProducerReceipt",
    "build_source_training_surface",
    "compile_verified_source_block",
    "reconstruct_compiler_recomputation_receipt",
    "validate_source_training_surface",
)
