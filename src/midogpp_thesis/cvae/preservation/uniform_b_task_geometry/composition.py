"""Pure fixed-budget generated-data composition; never routing or selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    SINGLE_BASE,
    SINGLE_BUDGET_MATCHED,
    UNION_EQUAL_TOTAL,
    UNION_EXPANDED,
)
from .generation import GeneratedBlock


@dataclass(frozen=True)
class ComposedSynthetic:
    mode: str
    generation_kind: str
    embeddings: np.ndarray
    labels: np.ndarray
    source_counts: Mapping[str, Mapping[int, int]]
    source_order: tuple[str, ...]
    selected_source: str | None
    input_block_hashes: Mapping[str, str]
    shuffle_hash: str

    @property
    def composition_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_uniform_b_composition_v1",
                "mode": self.mode,
                "generation_kind": self.generation_kind,
                "source_counts": {
                    source: {
                        str(label): int(count)
                        for label, count in counts.items()
                    }
                    for source, counts in self.source_counts.items()
                },
                "source_order": list(self.source_order),
                "selected_source": self.selected_source,
                "input_block_hashes": dict(self.input_block_hashes),
                "shuffle_hash": self.shuffle_hash,
                "shape": list(self.embeddings.shape),
                "fixed_empirical_design_weights": True,
                "routing_or_compatibility_weights": False,
            }
        )


def compose_generated_blocks(
    blocks: Mapping[str, GeneratedBlock],
    *,
    mode: str,
    base_per_class: int,
    shuffle_seed: int,
    selected_source: str | None = None,
) -> ComposedSynthetic:
    sources = tuple(sorted(str(value) for value in blocks))
    if not sources or base_per_class <= 0:
        raise ProtocolError("Composition requires source blocks and a positive budget.")
    first = blocks[sources[0]]
    if any(
        block.arm != first.arm
        or block.training_seed != first.training_seed
        or block.generation_seed != first.generation_seed
        or block.kind != first.kind
        or block.embeddings.shape[1] != 3840
        for block in blocks.values()
    ):
        raise ProtocolError("Composition input blocks do not share one identity.")
    k = len(sources)
    allocations: dict[str, int]
    if mode == SINGLE_BASE:
        if selected_source not in blocks:
            raise ProtocolError("single_base requires one legal selected source.")
        allocations = {str(selected_source): base_per_class}
    elif mode == SINGLE_BUDGET_MATCHED:
        if selected_source not in blocks:
            raise ProtocolError(
                "single_budget_matched requires one legal selected source."
            )
        allocations = {str(selected_source): k * base_per_class}
    elif mode == UNION_EXPANDED:
        if selected_source is not None:
            raise ProtocolError("Union composition cannot select a source.")
        allocations = {source: base_per_class for source in sources}
    elif mode == UNION_EQUAL_TOTAL:
        if selected_source is not None:
            raise ProtocolError("Union composition cannot select a source.")
        quotient, remainder = divmod(base_per_class, k)
        allocations = {
            source: quotient + (1 if index < remainder else 0)
            for index, source in enumerate(sources)
        }
        if any(value <= 0 for value in allocations.values()):
            raise ProtocolError(
                "Equal-total budget is too small for all legal sources."
            )
    else:
        raise ProtocolError(f"Unknown composition mode: {mode!r}")
    pieces: list[np.ndarray] = []
    label_pieces: list[np.ndarray] = []
    counts: dict[str, dict[int, int]] = {}
    for source in sources:
        count = allocations.get(source, 0)
        if count == 0:
            continue
        block = blocks[source]
        if block.per_class < count:
            raise ProtocolError("Generated block is smaller than composition budget.")
        source_pieces = []
        source_labels = []
        for cls in (0, 1):
            indices = np.flatnonzero(block.labels == cls)[:count]
            if len(indices) != count:
                raise ProtocolError("Generated block violates its class budget.")
            source_pieces.append(block.embeddings[indices])
            source_labels.append(block.labels[indices])
        pieces.extend(source_pieces)
        label_pieces.extend(source_labels)
        counts[source] = {0: count, 1: count}
    embeddings = np.concatenate(pieces, axis=0).astype(np.float32)
    labels = np.concatenate(label_pieces, axis=0).astype(np.int64)
    rng = np.random.default_rng(int(shuffle_seed))
    permutation = rng.permutation(len(labels))
    shuffled_embeddings = embeddings[permutation]
    shuffled_labels = labels[permutation]
    shuffle_hash = stable_hash(
        {
            "seed": int(shuffle_seed),
            "permutation": permutation.tolist(),
        }
    )
    return ComposedSynthetic(
        mode=mode,
        generation_kind=first.kind,
        embeddings=shuffled_embeddings,
        labels=shuffled_labels,
        source_counts=counts,
        source_order=sources,
        selected_source=selected_source,
        input_block_hashes={
            source: blocks[source].block_hash for source in sources
        },
        shuffle_hash=shuffle_hash,
    )


__all__ = ("ComposedSynthetic", "compose_generated_blocks")
