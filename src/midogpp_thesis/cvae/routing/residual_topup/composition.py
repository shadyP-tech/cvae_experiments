"""Deterministic composition of disjoint base and residual source windows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .contracts import ResidualTopupAction, TopupGeometry
from .hashing import array_bundle_sha256, array_sha256, canonical_sha256


COMPOSITION_SEMANTICS = (
    "canonical_equal_union_base_then_canonical_topup_fixed_classwise_shuffle"
)
BASE_ONLY_COMPOSITION_SEMANTICS = (
    "canonical_equal_union_base_prefix_fixed_classwise_shuffle"
)


@dataclass(frozen=True)
class EqualUnionBaseComposition:
    """Materialized immutable equal-union budget-reference arm."""

    embeddings: np.ndarray
    labels: np.ndarray
    source_by_row: tuple[str, ...]
    component_by_row: tuple[str, ...]
    source_order: tuple[str, ...]
    shuffle_seed_by_class: Mapping[int, int]
    permutation_by_class: Mapping[int, np.ndarray]
    window_array_sha256_by_source_class: Mapping[str, str]
    pre_shuffle_sha256_by_class: Mapping[int, str]
    post_shuffle_sha256_by_class: Mapping[int, str]
    allocation_hash: str
    window_hash: str
    output_sha256: str
    composition_hash: str
    total_per_class: int
    composition_semantics: str = BASE_ONLY_COMPOSITION_SEMANTICS

    def to_payload(self) -> dict[str, object]:
        return {
            "source_order": list(self.source_order),
            "shuffle_seed_by_class": {
                str(label): self.shuffle_seed_by_class[label]
                for label in (0, 1)
            },
            "window_array_sha256_by_source_class": dict(
                self.window_array_sha256_by_source_class
            ),
            "pre_shuffle_sha256_by_class": {
                str(label): self.pre_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "post_shuffle_sha256_by_class": {
                str(label): self.post_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "allocation_hash": self.allocation_hash,
            "window_hash": self.window_hash,
            "output_sha256": self.output_sha256,
            "composition_hash": self.composition_hash,
            "total_per_class": self.total_per_class,
            "composition_semantics": self.composition_semantics,
        }


@dataclass(frozen=True)
class ResidualTopupComposition:
    """Materialized balanced mixture with complete deterministic hash audit."""

    embeddings: np.ndarray
    labels: np.ndarray
    source_by_row: tuple[str, ...]
    component_by_row: tuple[str, ...]
    source_order: tuple[str, ...]
    shuffle_seed_by_class: Mapping[int, int]
    permutation_by_class: Mapping[int, np.ndarray]
    window_array_sha256_by_source_class_component: Mapping[str, str]
    base_pre_shuffle_sha256_by_class: Mapping[int, str]
    topup_pre_shuffle_sha256_by_class: Mapping[int, str]
    pre_shuffle_sha256_by_class: Mapping[int, str]
    post_shuffle_sha256_by_class: Mapping[int, str]
    allocation_hash: str
    window_hash: str
    output_sha256: str
    composition_hash: str
    total_per_class: int
    composition_semantics: str = COMPOSITION_SEMANTICS

    def to_payload(self) -> dict[str, object]:
        return {
            "source_order": list(self.source_order),
            "shuffle_seed_by_class": {
                str(label): self.shuffle_seed_by_class[label]
                for label in (0, 1)
            },
            "window_array_sha256_by_source_class_component": dict(
                self.window_array_sha256_by_source_class_component
            ),
            "base_pre_shuffle_sha256_by_class": {
                str(label): self.base_pre_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "topup_pre_shuffle_sha256_by_class": {
                str(label): self.topup_pre_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "pre_shuffle_sha256_by_class": {
                str(label): self.pre_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "post_shuffle_sha256_by_class": {
                str(label): self.post_shuffle_sha256_by_class[label]
                for label in (0, 1)
            },
            "allocation_hash": self.allocation_hash,
            "window_hash": self.window_hash,
            "output_sha256": self.output_sha256,
            "composition_hash": self.composition_hash,
            "total_per_class": self.total_per_class,
            "composition_semantics": self.composition_semantics,
        }


def compose_equal_union_base_blocks(
    source_blocks: Mapping[object, object],
    geometry: TopupGeometry,
    *,
    shuffle_seed_by_class: Mapping[object, int],
) -> EqualUnionBaseComposition:
    """Compose only ``[0, base_per_source)`` from every source/class.

    This is the lower-budget reference arm.  It deliberately has no top-up
    allocation or rows, even though it shares source blocks and shuffle logic
    with the matched-budget top-up arms.
    """

    if not isinstance(geometry, TopupGeometry):
        raise ProtocolError("Equal-union base composition geometry is invalid.")
    blocks = _normalize_source_mapping(source_blocks)
    if tuple(sorted(blocks)) != geometry.source_order:
        raise ProtocolError(
            "Equal-union base source blocks must match the geometry exactly."
        )
    seeds = _class_seeds(shuffle_seed_by_class)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_dim: int | None = None
    for source in geometry.source_order:
        embeddings, labels = _block_arrays(blocks[source])
        block_key = getattr(blocks[source], "key", None)
        if block_key is not None:
            identity = str(getattr(block_key, "source_center", ""))
            if identity != source:
                raise ProtocolError("Equal-union base source block identity drifted.")
        if feature_dim is None:
            feature_dim = int(embeddings.shape[1])
        if int(embeddings.shape[1]) != feature_dim:
            raise ProtocolError(
                "Equal-union base source block feature dimensions do not align."
            )
        for class_label in geometry.class_labels:
            if int(np.sum(labels == class_label)) < geometry.base_per_source:
                raise ProtocolError(
                    "Equal-union base source block has insufficient class capacity."
                )
        arrays[source] = (embeddings, labels)
    assert feature_dim is not None

    class_outputs: list[np.ndarray] = []
    class_sources: list[np.ndarray] = []
    permutations: dict[int, np.ndarray] = {}
    window_hashes: dict[str, str] = {}
    pre_hashes: dict[int, str] = {}
    post_hashes: dict[int, str] = {}
    for class_label in geometry.class_labels:
        pieces: list[np.ndarray] = []
        source_rows: list[np.ndarray] = []
        for source in geometry.source_order:
            embeddings, labels = arrays[source]
            class_indices = np.flatnonzero(labels == class_label)
            indices = class_indices[: geometry.base_per_source]
            if len(indices) != geometry.base_per_source:
                raise ProtocolError("Equal-union base classwise window drifted.")
            piece = np.ascontiguousarray(embeddings[indices], dtype=np.float32)
            pieces.append(piece)
            source_rows.append(
                np.full(geometry.base_per_source, source, dtype=object)
            )
            window_hashes[f"{class_label}:{source}:base"] = array_sha256(piece)
        unshuffled = np.ascontiguousarray(
            np.concatenate(pieces, axis=0), dtype=np.float32
        )
        unshuffled_sources = np.concatenate(source_rows, axis=0)
        if unshuffled.shape != (geometry.base_total_per_class, feature_dim):
            raise ProtocolError("Equal-union base class geometry drifted.")
        permutation = np.ascontiguousarray(
            np.random.default_rng(seeds[class_label]).permutation(
                geometry.base_total_per_class
            ),
            dtype=np.int64,
        )
        permutation.setflags(write=False)
        shuffled = np.ascontiguousarray(
            unshuffled[permutation], dtype=np.float32
        )
        shuffled.setflags(write=False)
        class_outputs.append(shuffled)
        class_sources.append(unshuffled_sources[permutation])
        permutations[class_label] = permutation
        pre_hashes[class_label] = array_sha256(unshuffled)
        post_hashes[class_label] = array_sha256(shuffled)

    embeddings = np.ascontiguousarray(
        np.concatenate(class_outputs, axis=0), dtype=np.float32
    )
    labels = np.concatenate(
        (
            np.zeros(geometry.base_total_per_class, dtype=np.int64),
            np.ones(geometry.base_total_per_class, dtype=np.int64),
        )
    )
    embeddings.setflags(write=False)
    labels.setflags(write=False)
    allocation_payload = {
        "schema_version": "midogpp_equal_union_base_allocation_v1",
        "source_order": list(geometry.source_order),
        "base_per_source": geometry.base_per_source,
        "total_per_class": geometry.base_total_per_class,
    }
    allocation_hash = canonical_sha256(allocation_payload)
    window_payload = {
        "schema_version": "midogpp_equal_union_base_windows_v1",
        "source_order": list(geometry.source_order),
        "windows_by_class": {
            str(label): {
                source: [0, geometry.base_per_source]
                for source in geometry.source_order
            }
            for label in geometry.class_labels
        },
    }
    window_hash = canonical_sha256(window_payload)
    output_sha256 = array_bundle_sha256(embeddings, labels)
    hash_payload = {
        "schema_version": "midogpp_equal_union_base_composition_v1",
        "source_order": list(geometry.source_order),
        "shuffle_seed_by_class": {
            str(label): seeds[label] for label in geometry.class_labels
        },
        "window_array_sha256_by_source_class": window_hashes,
        "pre_shuffle_sha256_by_class": {
            str(label): pre_hashes[label] for label in geometry.class_labels
        },
        "post_shuffle_sha256_by_class": {
            str(label): post_hashes[label] for label in geometry.class_labels
        },
        "allocation_hash": allocation_hash,
        "window_hash": window_hash,
        "output_sha256": output_sha256,
        "total_per_class": geometry.base_total_per_class,
        "composition_semantics": BASE_ONLY_COMPOSITION_SEMANTICS,
    }
    row_count = 2 * geometry.base_total_per_class
    return EqualUnionBaseComposition(
        embeddings=embeddings,
        labels=labels,
        source_by_row=tuple(
            str(value)
            for value in np.concatenate(class_sources, axis=0).tolist()
        ),
        component_by_row=tuple("base" for _ in range(row_count)),
        source_order=geometry.source_order,
        shuffle_seed_by_class=MappingProxyType(dict(seeds)),
        permutation_by_class=MappingProxyType(permutations),
        window_array_sha256_by_source_class=MappingProxyType(window_hashes),
        pre_shuffle_sha256_by_class=MappingProxyType(pre_hashes),
        post_shuffle_sha256_by_class=MappingProxyType(post_hashes),
        allocation_hash=allocation_hash,
        window_hash=window_hash,
        output_sha256=output_sha256,
        composition_hash=canonical_sha256(hash_payload),
        total_per_class=geometry.base_total_per_class,
    )


def compose_residual_topup_blocks(
    source_blocks: Mapping[object, object],
    action: ResidualTopupAction,
    *,
    shuffle_seed_by_class: Mapping[object, int],
) -> ResidualTopupComposition:
    """Compose the immutable base and disjoint additive top-up windows."""

    blocks = _normalize_source_mapping(source_blocks)
    if tuple(sorted(blocks)) != action.geometry.source_order:
        raise ProtocolError(
            "Residual top-up source blocks must match the action exactly."
        )
    seeds = _class_seeds(shuffle_seed_by_class)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_dim: int | None = None
    for source in action.geometry.source_order:
        embeddings, labels = _block_arrays(blocks[source])
        block_key = getattr(blocks[source], "key", None)
        if block_key is not None:
            identity = str(getattr(block_key, "source_center", ""))
            if identity != source:
                raise ProtocolError("Residual top-up source block identity drifted.")
        if feature_dim is None:
            feature_dim = int(embeddings.shape[1])
        if int(embeddings.shape[1]) != feature_dim:
            raise ProtocolError(
                "Residual top-up source block feature dimensions do not align."
            )
        for class_label in action.geometry.class_labels:
            required = action.windows_by_class[class_label][
                source
            ].required_capacity
            if int(np.sum(labels == class_label)) < required:
                raise ProtocolError(
                    "Residual top-up source block has insufficient class capacity."
                )
        arrays[source] = (embeddings, labels)
    assert feature_dim is not None

    class_outputs: list[np.ndarray] = []
    class_sources: list[np.ndarray] = []
    class_components: list[np.ndarray] = []
    permutations: dict[int, np.ndarray] = {}
    window_hashes: dict[str, str] = {}
    base_hashes: dict[int, str] = {}
    topup_hashes: dict[int, str] = {}
    pre_hashes: dict[int, str] = {}
    post_hashes: dict[int, str] = {}

    for class_label in action.geometry.class_labels:
        base_pieces: list[np.ndarray] = []
        topup_pieces: list[np.ndarray] = []
        base_sources: list[np.ndarray] = []
        topup_sources: list[np.ndarray] = []
        for source in action.geometry.source_order:
            embeddings, labels = arrays[source]
            class_indices = np.flatnonzero(labels == class_label)
            window = action.windows_by_class[class_label][source]
            base_indices = class_indices[window.base_start : window.base_stop]
            topup_indices = class_indices[window.topup_start : window.topup_stop]
            if (
                len(base_indices) != window.base_count
                or len(topup_indices) != window.topup_count
                or np.intersect1d(base_indices, topup_indices).size
            ):
                raise ProtocolError(
                    "Residual top-up classwise windows overlap or drifted."
                )
            base = np.ascontiguousarray(embeddings[base_indices], dtype=np.float32)
            topup = np.ascontiguousarray(
                embeddings[topup_indices], dtype=np.float32
            )
            base_pieces.append(base)
            topup_pieces.append(topup)
            base_sources.append(
                np.full(window.base_count, source, dtype=object)
            )
            topup_sources.append(
                np.full(window.topup_count, source, dtype=object)
            )
            window_hashes[f"{class_label}:{source}:base"] = array_sha256(base)
            window_hashes[f"{class_label}:{source}:topup"] = array_sha256(topup)

        base_unshuffled = np.ascontiguousarray(
            np.concatenate(base_pieces, axis=0), dtype=np.float32
        )
        topup_unshuffled = np.ascontiguousarray(
            np.concatenate(topup_pieces, axis=0), dtype=np.float32
        )
        unshuffled = np.ascontiguousarray(
            np.concatenate((base_unshuffled, topup_unshuffled), axis=0),
            dtype=np.float32,
        )
        unshuffled_sources = np.concatenate(
            (
                np.concatenate(base_sources, axis=0),
                np.concatenate(topup_sources, axis=0),
            ),
            axis=0,
        )
        unshuffled_components = np.concatenate(
            (
                np.full(action.geometry.base_total_per_class, "base", dtype=object),
                np.full(
                    action.geometry.topup_total_per_class,
                    "topup",
                    dtype=object,
                ),
            ),
            axis=0,
        )
        expected_shape = (
            action.geometry.final_total_per_class,
            feature_dim,
        )
        if (
            base_unshuffled.shape
            != (action.geometry.base_total_per_class, feature_dim)
            or topup_unshuffled.shape
            != (action.geometry.topup_total_per_class, feature_dim)
            or unshuffled.shape != expected_shape
        ):
            raise ProtocolError(
                "Residual top-up composed class geometry drifted."
            )
        permutation = np.ascontiguousarray(
            np.random.default_rng(seeds[class_label]).permutation(
                action.geometry.final_total_per_class
            ),
            dtype=np.int64,
        )
        permutation.setflags(write=False)
        shuffled = np.ascontiguousarray(
            unshuffled[permutation], dtype=np.float32
        )
        shuffled.setflags(write=False)
        class_outputs.append(shuffled)
        class_sources.append(unshuffled_sources[permutation])
        class_components.append(unshuffled_components[permutation])
        permutations[class_label] = permutation
        base_hashes[class_label] = array_sha256(base_unshuffled)
        topup_hashes[class_label] = array_sha256(topup_unshuffled)
        pre_hashes[class_label] = array_sha256(unshuffled)
        post_hashes[class_label] = array_sha256(shuffled)

    embeddings = np.ascontiguousarray(
        np.concatenate(class_outputs, axis=0), dtype=np.float32
    )
    labels = np.concatenate(
        (
            np.zeros(action.geometry.final_total_per_class, dtype=np.int64),
            np.ones(action.geometry.final_total_per_class, dtype=np.int64),
        )
    )
    embeddings.setflags(write=False)
    labels.setflags(write=False)
    output_sha256 = array_bundle_sha256(embeddings, labels)
    hash_payload = {
        "schema_version": "midogpp_residual_topup_composition_v1",
        "source_order": list(action.geometry.source_order),
        "shuffle_seed_by_class": {
            str(label): seeds[label] for label in action.geometry.class_labels
        },
        "window_array_sha256_by_source_class_component": window_hashes,
        "base_pre_shuffle_sha256_by_class": {
            str(label): base_hashes[label]
            for label in action.geometry.class_labels
        },
        "topup_pre_shuffle_sha256_by_class": {
            str(label): topup_hashes[label]
            for label in action.geometry.class_labels
        },
        "pre_shuffle_sha256_by_class": {
            str(label): pre_hashes[label]
            for label in action.geometry.class_labels
        },
        "post_shuffle_sha256_by_class": {
            str(label): post_hashes[label]
            for label in action.geometry.class_labels
        },
        "allocation_hash": action.allocation_hash,
        "window_hash": action.window_hash,
        "output_sha256": output_sha256,
        "total_per_class": action.geometry.final_total_per_class,
        "composition_semantics": COMPOSITION_SEMANTICS,
    }
    return ResidualTopupComposition(
        embeddings=embeddings,
        labels=labels,
        source_by_row=tuple(
            str(value)
            for value in np.concatenate(class_sources, axis=0).tolist()
        ),
        component_by_row=tuple(
            str(value)
            for value in np.concatenate(class_components, axis=0).tolist()
        ),
        source_order=action.geometry.source_order,
        shuffle_seed_by_class=MappingProxyType(dict(seeds)),
        permutation_by_class=MappingProxyType(permutations),
        window_array_sha256_by_source_class_component=MappingProxyType(
            window_hashes
        ),
        base_pre_shuffle_sha256_by_class=MappingProxyType(base_hashes),
        topup_pre_shuffle_sha256_by_class=MappingProxyType(topup_hashes),
        pre_shuffle_sha256_by_class=MappingProxyType(pre_hashes),
        post_shuffle_sha256_by_class=MappingProxyType(post_hashes),
        allocation_hash=action.allocation_hash,
        window_hash=action.window_hash,
        output_sha256=output_sha256,
        composition_hash=canonical_sha256(hash_payload),
        total_per_class=action.geometry.final_total_per_class,
    )


def _normalize_source_mapping(
    values: Mapping[object, object],
) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Residual top-up source blocks must be a mapping.")
    normalized: dict[str, object] = {}
    for raw_source, block in values.items():
        source = str(raw_source)
        if not source or source.strip() != source or source in normalized:
            raise ProtocolError("Residual top-up source block keys are invalid.")
        normalized[source] = block
    if not normalized:
        raise ProtocolError("Residual top-up requires source blocks.")
    return normalized


def _class_seeds(values: Mapping[object, int]) -> dict[int, int]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Residual top-up shuffle seeds must be a mapping.")
    normalized: dict[int, int] = {}
    for raw_label, raw_seed in values.items():
        if isinstance(raw_label, bool) or isinstance(raw_seed, bool):
            raise ProtocolError("Residual top-up shuffle seeds must be integers.")
        try:
            label = int(raw_label)
            seed = int(raw_seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError(
                "Residual top-up shuffle seeds must be integers."
            ) from exc
        if label not in (0, 1) or label in normalized or seed < 0:
            raise ProtocolError(
                "Residual top-up requires one nonnegative seed per class."
            )
        normalized[label] = seed
    if set(normalized) != {0, 1}:
        raise ProtocolError(
            "Residual top-up requires exactly class 0 and class 1 seeds."
        )
    return normalized


def _block_arrays(block: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(block, Mapping):
        raw_embeddings = block.get("embeddings")
        raw_labels = block.get("labels")
    else:
        raw_embeddings = getattr(block, "embeddings", None)
        raw_labels = getattr(block, "labels", None)
    try:
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        labels = np.asarray(raw_labels, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Residual top-up source block arrays are invalid.") from exc
    if (
        embeddings.ndim != 2
        or not len(embeddings)
        or labels.shape != (len(embeddings),)
        or set(int(value) for value in labels.tolist()) != {0, 1}
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Residual top-up source block arrays are invalid.")
    return embeddings, labels


__all__ = (
    "BASE_ONLY_COMPOSITION_SEMANTICS",
    "COMPOSITION_SEMANTICS",
    "EqualUnionBaseComposition",
    "ResidualTopupComposition",
    "compose_equal_union_base_blocks",
    "compose_residual_topup_blocks",
)
