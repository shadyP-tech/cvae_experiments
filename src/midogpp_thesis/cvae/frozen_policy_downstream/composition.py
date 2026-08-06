"""Exact classwise materialization of frozen Stage-70 policy assignments."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...common.hashing import stable_hash
from ..generation.generation import GeneratedBlock
from ..protocol import ProtocolError
from .contracts import (
    FEATURE_DIM,
    SYNTHETIC_PER_CLASS,
    PolicyReplicate,
    SyntheticComposition,
    array_bundle_sha256,
    array_sha256,
)


def compose_policy_replicate(
    replicate: PolicyReplicate,
    source_blocks: Mapping[str, GeneratedBlock],
) -> SyntheticComposition:
    """Slice prefixes within each class, then apply the frozen class shuffles."""

    # The store is deliberately lazy.  Resolve each selected source only once
    # for this composition rather than decompressing it once per class.
    resolved_blocks: dict[str, GeneratedBlock] = {}
    for assignment in replicate.assignments:
        try:
            resolved_blocks[assignment.source_stream_id] = source_blocks[
                assignment.source_stream_id
            ]
        except KeyError as exc:
            raise ProtocolError(
                f"Missing Stage-70 source stream {assignment.source_stream_id}."
            ) from exc

    shuffled_by_label: list[np.ndarray] = []
    pre_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}
    source_rows: list[dict[str, object]] = []
    for class_label in (0, 1):
        prefixes: list[np.ndarray] = []
        for assignment in replicate.assignments:
            block = resolved_blocks[assignment.source_stream_id]
            _validate_block_identity(block, assignment.source_center)
            class_rows = np.asarray(block.embeddings)[np.asarray(block.labels) == class_label]
            stop = int(assignment.source_budget_per_class)
            if class_rows.shape[0] < stop:
                raise ProtocolError("Stage-70 source block is shorter than its frozen prefix.")
            prefix = np.asarray(class_rows[:stop], dtype=np.float32)
            if prefix.shape != (stop, FEATURE_DIM) or not np.isfinite(prefix).all():
                raise ProtocolError("Stage-70 source prefix failed geometry/finiteness checks.")
            prefixes.append(prefix)
            source_rows.append(
                {
                    "assignment_id": assignment.assignment_id,
                    "source_center": assignment.source_center,
                    "source_stream_id": assignment.source_stream_id,
                    "source_ordinal": assignment.source_ordinal,
                    "class_label": class_label,
                    "prefix_start": 0,
                    "prefix_stop": stop,
                    "prefix_sha256": array_sha256(prefix),
                }
            )
        unshuffled = np.concatenate(prefixes, axis=0)
        if unshuffled.shape != (SYNTHETIC_PER_CLASS, FEATURE_DIM):
            raise ProtocolError("Stage-70 per-class composition budget drifted.")
        seed = int(replicate.class_shuffle_seed_by_label[str(class_label)])
        order = np.random.default_rng(seed).permutation(SYNTHETIC_PER_CLASS)
        shuffled = np.ascontiguousarray(unshuffled[order], dtype=np.float32)
        pre_hashes[str(class_label)] = array_sha256(unshuffled)
        post_hashes[str(class_label)] = array_sha256(shuffled)
        shuffled_by_label.append(shuffled)

    embeddings = np.concatenate(shuffled_by_label, axis=0)
    labels = np.concatenate(
        (
            np.zeros(SYNTHETIC_PER_CLASS, dtype=np.int64),
            np.ones(SYNTHETIC_PER_CLASS, dtype=np.int64),
        )
    )
    train_hash = array_bundle_sha256(embeddings, labels)
    manifest = {
        "schema_version": "midogpp_stage70_composition_manifest_v1",
        "policy_id": replicate.policy_id,
        "policy_lock_hash": replicate.policy_lock_hash,
        "policy_plan_hash": replicate.policy_plan_hash,
        "assignment_table_hash": replicate.assignment_table_hash,
        "replicate_id": replicate.replicate_id,
        "target_center": replicate.target_center,
        "training_seed": replicate.training_seed,
        "generation_seed": replicate.generation_seed,
        "class_shuffle_seed_by_label": dict(replicate.class_shuffle_seed_by_label),
        "source_prefixes": source_rows,
        "pre_shuffle_sha256_by_label": pre_hashes,
        "post_shuffle_sha256_by_label": post_hashes,
        "train_content_sha256": train_hash,
        "synthetic_rows_per_class": SYNTHETIC_PER_CLASS,
        "target_expert_excluded": True,
    }
    return SyntheticComposition(
        replicate=replicate,
        embeddings=embeddings,
        labels=labels,
        pre_shuffle_sha256_by_label=pre_hashes,
        post_shuffle_sha256_by_label=post_hashes,
        train_content_sha256=train_hash,
        composition_manifest_hash=stable_hash(manifest),
    )


def _validate_block_identity(block: GeneratedBlock, expected_center: str) -> None:
    if block.key.source_center != expected_center:
        raise ProtocolError("Stage-70 source block center differs from its assignment.")
    embeddings = np.asarray(block.embeddings)
    labels = np.asarray(block.labels)
    if embeddings.ndim != 2 or embeddings.shape[1] != FEATURE_DIM:
        raise ProtocolError("Stage-70 source block feature geometry drifted.")
    if labels.shape != (embeddings.shape[0],):
        raise ProtocolError("Stage-70 source block labels do not align.")
    for class_label in (0, 1):
        if int(np.sum(labels == class_label)) < SYNTHETIC_PER_CLASS:
            raise ProtocolError("Stage-70 maximum source block lacks a class prefix.")
    if not np.isfinite(embeddings).all():
        raise ProtocolError("Stage-70 source block contains non-finite values.")


__all__ = ("compose_policy_replicate",)
