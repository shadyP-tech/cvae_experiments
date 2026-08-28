"""Exact-nine reduction into the 72 oriented label-free B/U/A1 blocks."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from itertools import product
import re
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ....runtime.frozen_source_streams import FrozenSourceStreamCache
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS
from ..source_bundle.constants import PROBABILITY_DTYPE
from ..source_bundle.hashing import array_sha256
from .held_actions import (
    B_ACTION_ID,
    U_ACTION_ID,
    a1_action_id,
    canonical_held_action_library,
    held_candidate_sources,
    TRAINING_SEEDS,
    GENERATION_SEEDS,
)
from .source_frame import (
    LabelFreeSourceFrame,
    SourceProbabilitySeal,
    _PROBABILITY_SEAL_GATE,
)
from .worker import load_held_checkpoint_arrays, load_held_prediction_checkpoint


_HEX_LOCK = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
_BLOCK_GATE = object()
_INVENTORY_GATE = object()


@dataclass(frozen=True, slots=True)
class HeldBaseProbabilityBlock:
    outer_target_center: str
    query_center: str
    row_ids: tuple[str, ...]
    source_cache_row_indices: tuple[int, ...]
    probabilities_by_base: tuple[tuple[str, np.ndarray], ...]
    seed_task_hashes: tuple[str, ...]
    source_frame_hash: str
    source_stream_lock_hash: str
    held_action_library_sha256: str
    held_mass_policy_receipt_sha256: str
    _factory_token: InitVar[object] = None
    _factory_validated: bool = field(init=False, repr=False)
    block_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BLOCK_GATE:
            raise ProtocolError(
                "OE-PPUR v3 held probability block is production-factory-only."
            )
        h, q = str(self.outer_target_center), str(self.query_center)
        rows = tuple(str(value) for value in self.row_ids)
        indices = tuple(int(value) for value in self.source_cache_row_indices)
        values = tuple((str(base_id), np.ascontiguousarray(array, dtype=np.float32)) for base_id, array in self.probabilities_by_base)
        expected_ids = (
            B_ACTION_ID,
            U_ACTION_ID,
            *(a1_action_id(center) for center in CENTERS if center not in {h, q}),
        )
        task_hashes = tuple(require_sha256(value, "held seed task hash") for value in self.seed_task_hashes)
        if (
            h not in CENTERS
            or q not in CENTERS
            or h == q
            or not rows
            or len(rows) != len(indices)
            or len(set(rows)) != len(rows)
            or len(set(indices)) != len(indices)
            or tuple(base_id for base_id, _ in values) != expected_ids
            or any(array.shape != (len(rows),) or not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)) for _, array in values)
            or len(task_hashes) != 9
        ):
            raise ProtocolError("OE-PPUR v3 held base-probability block drifted.")
        for _, array in values:
            array.setflags(write=False)
        frame_hash = require_sha256(self.source_frame_hash, "source frame hash")
        library = require_sha256(self.held_action_library_sha256, "held action library hash")
        mass = require_sha256(self.held_mass_policy_receipt_sha256, "held mass policy receipt hash")
        stream_lock = str(self.source_stream_lock_hash).lower()
        if _HEX_LOCK.fullmatch(stream_lock) is None:
            raise ProtocolError("OE-PPUR v3 held source-stream lock hash drifted.")
        object.__setattr__(self, "outer_target_center", h)
        object.__setattr__(self, "query_center", q)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "source_cache_row_indices", indices)
        object.__setattr__(self, "probabilities_by_base", values)
        object.__setattr__(self, "seed_task_hashes", task_hashes)
        object.__setattr__(self, "source_frame_hash", frame_hash)
        object.__setattr__(self, "held_action_library_sha256", library)
        object.__setattr__(self, "held_mass_policy_receipt_sha256", mass)
        object.__setattr__(self, "source_stream_lock_hash", stream_lock)
        object.__setattr__(self, "_factory_validated", True)
        object.__setattr__(
            self,
            "block_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_held_base_probability_block_v1",
                    "H": h,
                    "q": q,
                    "row_ids": rows,
                    "source_cache_row_indices": indices,
                    "base_probability_hashes": tuple(
                        (base_id, array_sha256(array, dtype=PROBABILITY_DTYPE))
                        for base_id, array in values
                    ),
                    "seed_task_hashes": task_hashes,
                    "source_frame_hash": frame_hash,
                    "source_stream_lock_hash": self.source_stream_lock_hash,
                    "held_action_library_sha256": library,
                    "held_mass_policy_receipt_sha256": mass,
                    "exact_nine_mean_reduction_dtype": "float64",
                    "persisted_dtype": "<f4",
                    "labels_used": False,
                }
            ),
        )

    def probabilities(self, base_id: object) -> np.ndarray:
        key = str(base_id)
        for observed, values in self.probabilities_by_base:
            if observed == key:
                return values
        raise ProtocolError(f"Unknown OE-PPUR v3 held base id: {key}")


@dataclass(frozen=True, slots=True)
class HeldPredictionInventory:
    blocks: tuple[HeldBaseProbabilityBlock, ...]
    probability_seal: SourceProbabilitySeal
    _factory_token: InitVar[object] = None
    _factory_validated: bool = field(init=False, repr=False)
    inventory_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _INVENTORY_GATE:
            raise ProtocolError(
                "OE-PPUR v3 held prediction inventory is production-factory-only."
            )
        blocks = tuple(self.blocks)
        expected = tuple((h, q) for h in CENTERS for q in CENTERS if h != q)
        if (
            tuple((block.outer_target_center, block.query_center) for block in blocks) != expected
            or not isinstance(self.probability_seal, SourceProbabilitySeal)
            or tuple((h, q, block.block_hash) for (h, q), block in zip(expected, blocks, strict=True))
            != self.probability_seal.oriented_block_receipts
            or any(
                block.source_frame_hash != self.probability_seal.source_frame_hash
                or block.source_stream_lock_hash
                != self.probability_seal.source_stream_lock_hash
                or block.held_action_library_sha256
                != self.probability_seal.held_action_library_sha256
                or block.held_mass_policy_receipt_sha256
                != self.probability_seal.held_mass_policy_receipt_sha256
                for block in blocks
            )
        ):
            raise ProtocolError("OE-PPUR v3 held prediction inventory drifted.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "_factory_validated", True)
        object.__setattr__(
            self,
            "inventory_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_held_prediction_inventory_v1",
                    "probability_seal_hash": self.probability_seal.receipt_hash,
                    "block_hashes": tuple(block.block_hash for block in blocks),
                    "labels_used": False,
                }
            ),
        )


def assemble_held_prediction_inventory(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
    *,
    frame: LabelFreeSourceFrame,
    source: FrozenSourceStreamCache,
) -> HeldPredictionInventory:
    rows = tuple(tasks)
    if len(rows) != 324 or len(completed) != 324 or not isinstance(frame, LabelFreeSourceFrame) or type(source) is not FrozenSourceStreamCache:
        raise ProtocolError("OE-PPUR v3 held prediction reduction inputs drifted.")
    by_pair: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for task in rows:
        if load_held_prediction_checkpoint(task) is None or str(task["task_id"]) not in completed:
            raise ProtocolError("OE-PPUR v3 held prediction checkpoint coverage drifted.")
        pair = tuple(str(value) for value in task["excluded_centers"])
        by_pair.setdefault((pair[0], pair[1]), []).append(task)
    library = canonical_held_action_library()
    blocks_by_key: dict[tuple[str, str], HeldBaseProbabilityBlock] = {}
    for pair, pair_tasks in by_pair.items():
        expected_seed_grid = tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))
        observed_seed_grid = tuple(
            (int(task["training_seed"]), int(task["generation_seed"]))
            for task in pair_tasks
        )
        task_hashes = tuple(str(task["task_hash"]) for task in pair_tasks)
        if (
            len(pair_tasks) != 9
            or observed_seed_grid != expected_seed_grid
            or len(set(task_hashes)) != 9
            or any(require_sha256(value, "held seed task hash") != value for value in task_hashes)
        ):
            raise ProtocolError("OE-PPUR v3 held pair lacks exact-nine seed coverage.")
        first_matrices = []
        second_matrices = []
        for task in pair_tasks:
            first, second = load_held_checkpoint_arrays(task)
            first_matrices.append(first)
            second_matrices.append(second)
        first_mean = np.ascontiguousarray(np.mean(np.stack(first_matrices).astype(np.float64), axis=0), dtype=np.float32)
        second_mean = np.ascontiguousarray(np.mean(np.stack(second_matrices).astype(np.float64), axis=0), dtype=np.float32)
        for evaluated, outer, matrix in (
            (pair[0], pair[1], first_mean),
            (pair[1], pair[0], second_mean),
        ):
            source_rows = frame.rows_by_center[evaluated]
            ids = (B_ACTION_ID, U_ACTION_ID, *(a1_action_id(center) for center in held_candidate_sources(pair)))
            block = HeldBaseProbabilityBlock(
                outer_target_center=outer,
                query_center=evaluated,
                row_ids=tuple(row.source_row_id for row in source_rows),
                source_cache_row_indices=tuple(row.source_cache_row_index for row in source_rows),
                probabilities_by_base=tuple((base_id, matrix[index]) for index, base_id in enumerate(ids)),
                seed_task_hashes=task_hashes,
                source_frame_hash=frame.frame_hash,
                source_stream_lock_hash=source.lock_hash,
                held_action_library_sha256=library.library_hash,
                held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
                _factory_token=_BLOCK_GATE,
            )
            blocks_by_key[(outer, evaluated)] = block
    canonical_blocks = tuple(blocks_by_key[(h, q)] for h in CENTERS for q in CENTERS if h != q)
    seal = SourceProbabilitySeal(
        source_frame_hash=frame.frame_hash,
        source_stream_lock_hash=source.lock_hash,
        held_action_library_sha256=library.library_hash,
        held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        oriented_block_receipts=tuple((block.outer_target_center, block.query_center, block.block_hash) for block in canonical_blocks),
        _factory_token=_PROBABILITY_SEAL_GATE,
    )
    return HeldPredictionInventory(
        canonical_blocks, seal, _factory_token=_INVENTORY_GATE
    )


__all__ = (
    "HeldBaseProbabilityBlock",
    "HeldPredictionInventory",
    "assemble_held_prediction_inventory",
)
