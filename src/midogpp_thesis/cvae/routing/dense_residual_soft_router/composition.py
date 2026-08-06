"""Deterministic classwise prefix composition for dense router actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .allocation import DEFAULT_TOTAL_PER_CLASS


COMPOSITION_SEMANTICS = (
    "canonical_source_order_class_prefix_then_fixed_classwise_shuffle"
)


@dataclass(frozen=True)
class PrefixComposition:
    embeddings: np.ndarray
    labels: np.ndarray
    source_by_row: tuple[str, ...]
    source_order: tuple[str, ...]
    allocation_per_class: Mapping[str, int]
    shuffle_seed_by_class: Mapping[str, int]
    permutation_by_class: Mapping[int, np.ndarray]
    prefix_sha256_by_source_class: Mapping[str, str]
    pre_shuffle_sha256_by_class: Mapping[str, str]
    post_shuffle_sha256_by_class: Mapping[str, str]
    total_per_class: int
    composition_semantics: str = COMPOSITION_SEMANTICS

    @property
    def composition_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_dense_residual_prefix_composition_v1",
                "source_order": list(self.source_order),
                "allocation_per_class": dict(self.allocation_per_class),
                "shuffle_seed_by_class": dict(self.shuffle_seed_by_class),
                "prefix_sha256_by_source_class": dict(
                    self.prefix_sha256_by_source_class
                ),
                "pre_shuffle_sha256_by_class": dict(
                    self.pre_shuffle_sha256_by_class
                ),
                "post_shuffle_sha256_by_class": dict(
                    self.post_shuffle_sha256_by_class
                ),
                "output_sha256": _array_bundle_sha256(
                    self.embeddings, self.labels
                ),
                "total_per_class": self.total_per_class,
                "composition_semantics": self.composition_semantics,
            }
        )


def compose_prefix_blocks(
    source_blocks: Mapping[str, object],
    allocation_per_class: Mapping[str, int],
    *,
    shuffle_seed_by_class: Mapping[str | int, int],
    total_per_class: int = DEFAULT_TOTAL_PER_CLASS,
) -> PrefixComposition:
    """Take canonical class prefixes and shuffle with fixed per-class seeds."""

    blocks = _normalize_source_mapping(source_blocks, role="source block")
    allocations_raw = _normalize_source_mapping(
        allocation_per_class, role="source allocation"
    )
    sources = tuple(sorted(blocks))
    if set(allocations_raw) != set(sources):
        raise ProtocolError("Prefix composition blocks and allocations must match exactly.")
    allocations = {source: int(allocations_raw[source]) for source in sources}
    expected_total = int(total_per_class)
    if (
        expected_total <= 0
        or any(value <= 0 for value in allocations.values())
        or sum(allocations.values()) != expected_total
    ):
        raise ProtocolError(
            "Prefix composition requires positive allocations with one fixed class total."
        )
    seeds = _class_seeds(shuffle_seed_by_class)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_dim: int | None = None
    for source in sources:
        embeddings, labels = _block_arrays(blocks[source])
        block_key = getattr(blocks[source], "key", None)
        if block_key is not None and str(getattr(block_key, "source_center", "")) != source:
            raise ProtocolError("Prefix composition source block identity drifted.")
        if feature_dim is None:
            feature_dim = int(embeddings.shape[1])
        if embeddings.shape[1] != feature_dim:
            raise ProtocolError("Prefix composition feature dimensions do not align.")
        count = allocations[source]
        for class_label in (0, 1):
            if int(np.sum(labels == class_label)) < count:
                raise ProtocolError(
                    "Prefix composition source block is shorter than its allocation."
                )
        arrays[source] = (embeddings, labels)

    class_outputs: list[np.ndarray] = []
    class_sources: list[np.ndarray] = []
    permutations: dict[int, np.ndarray] = {}
    prefix_hashes: dict[str, str] = {}
    pre_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}
    for class_label in (0, 1):
        prefixes: list[np.ndarray] = []
        prefix_sources: list[np.ndarray] = []
        for source in sources:
            embeddings, labels = arrays[source]
            count = allocations[source]
            indices = np.flatnonzero(labels == class_label)[:count]
            prefix = np.ascontiguousarray(embeddings[indices], dtype=np.float32)
            prefixes.append(prefix)
            prefix_sources.append(np.full(count, source, dtype=object))
            prefix_hashes[f"{source}:{class_label}"] = _array_sha256(prefix)
        unshuffled = np.ascontiguousarray(np.concatenate(prefixes, axis=0), dtype=np.float32)
        unshuffled_sources = np.concatenate(prefix_sources, axis=0)
        if unshuffled.shape != (expected_total, feature_dim):
            raise ProtocolError("Prefix composition class total or geometry drifted.")
        permutation = np.random.default_rng(seeds[class_label]).permutation(
            expected_total
        )
        permutation = np.ascontiguousarray(permutation, dtype=np.int64)
        permutation.setflags(write=False)
        shuffled = np.ascontiguousarray(unshuffled[permutation], dtype=np.float32)
        class_outputs.append(shuffled)
        class_sources.append(unshuffled_sources[permutation])
        permutations[class_label] = permutation
        pre_hashes[str(class_label)] = _array_sha256(unshuffled)
        post_hashes[str(class_label)] = _array_sha256(shuffled)

    embeddings = np.ascontiguousarray(np.concatenate(class_outputs, axis=0), dtype=np.float32)
    labels = np.concatenate(
        (
            np.zeros(expected_total, dtype=np.int64),
            np.ones(expected_total, dtype=np.int64),
        )
    )
    if embeddings.shape != (2 * expected_total, feature_dim):
        raise ProtocolError("Prefix composition output geometry drifted.")
    return PrefixComposition(
        embeddings=embeddings,
        labels=labels,
        source_by_row=tuple(
            str(value) for value in np.concatenate(class_sources, axis=0).tolist()
        ),
        source_order=sources,
        allocation_per_class=allocations,
        shuffle_seed_by_class={str(label): seeds[label] for label in (0, 1)},
        permutation_by_class=permutations,
        prefix_sha256_by_source_class=prefix_hashes,
        pre_shuffle_sha256_by_class=pre_hashes,
        post_shuffle_sha256_by_class=post_hashes,
        total_per_class=expected_total,
    )


def _normalize_source_mapping(
    values: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for raw_source, value in values.items():
        source = str(raw_source)
        if not source or source in normalized:
            raise ProtocolError(f"Prefix composition {role} keys are invalid.")
        normalized[source] = value
    if not normalized:
        raise ProtocolError(f"Prefix composition requires at least one {role}.")
    return normalized


def _class_seeds(values: Mapping[str | int, int]) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for raw_label, raw_seed in values.items():
        try:
            label = int(raw_label)
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Prefix shuffle class seeds must be integers.") from exc
        if label not in (0, 1) or label in normalized or seed < 0:
            raise ProtocolError("Prefix shuffle requires one nonnegative seed per class.")
        normalized[label] = seed
    if set(normalized) != {0, 1}:
        raise ProtocolError("Prefix shuffle requires exactly class 0 and class 1 seeds.")
    return normalized


def _block_arrays(block: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(block, Mapping):
        raw_embeddings = block.get("embeddings")
        raw_labels = block.get("labels")
    else:
        raw_embeddings = getattr(block, "embeddings", None)
        raw_labels = getattr(block, "labels", None)
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    labels = np.asarray(raw_labels, dtype=np.int64)
    if (
        embeddings.ndim != 2
        or not len(embeddings)
        or labels.shape != (len(embeddings),)
        or set(int(value) for value in labels.tolist()) != {0, 1}
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Prefix composition source block arrays are invalid.")
    return embeddings, labels


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(_array_sha256(embeddings).encode("ascii"))
    digest.update(_array_sha256(labels).encode("ascii"))
    return digest.hexdigest()


__all__ = (
    "COMPOSITION_SEMANTICS",
    "PrefixComposition",
    "compose_prefix_blocks",
)
