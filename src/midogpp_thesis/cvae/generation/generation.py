"""Target-independent generation plans and source-stream realization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import torch

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.serialization import RoutingAuthorizedExpert
from ..generation_samplers import sample_latents
from ..preservation.uniform_b_optimized_prior.core import TorchOptimizedFrame
from ..protocol import ProtocolError
from .contracts import (
    COMMON_OUTPUT_DIM,
    COMPOSITION_SHUFFLE_NAMESPACE,
    ControlReplicate,
    GenerationLock,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    SourceGenerationKey,
    TOTAL_PER_CLASS,
)


@dataclass(frozen=True)
class GeneratedBlock:
    key: SourceGenerationKey
    embeddings: np.ndarray
    labels: np.ndarray
    output_sha256: str


def derived_generation_seed(
    *,
    namespace: str,
    bank_lock_hash: str,
    expert_lock_hash: str,
    generation_seed: int,
    class_label: int,
) -> int:
    """Derive a target- and policy-independent seed for one class stream."""

    if int(class_label) not in (0, 1):
        raise ProtocolError("Uniform-B v2 generation supports class labels 0/1 only.")
    payload = {
        "namespace": str(namespace),
        "bank_lock_hash": str(bank_lock_hash),
        "expert_lock_hash": str(expert_lock_hash),
        "generation_seed": int(generation_seed),
        "class_label": int(class_label),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def derived_composition_seed(
    *,
    generation_lock_hash: str,
    target_center: str,
    training_seed: int,
    generation_seed: int,
    class_label: int,
) -> int:
    """Derive the frozen equal-union shuffle seed for one target replicate."""

    if int(class_label) not in (0, 1):
        raise ProtocolError("Uniform-B v2 composition supports class labels 0/1 only.")
    payload = {
        "namespace": COMPOSITION_SHUFFLE_NAMESPACE,
        "generation_lock_hash": str(generation_lock_hash),
        "target_center": str(target_center),
        "training_seed": int(training_seed),
        "generation_seed": int(generation_seed),
        "class_label": int(class_label),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def source_generation_plan(lock: GenerationLock) -> tuple[SourceGenerationKey, ...]:
    payload = lock.to_payload()
    bank = _mapping(payload, "bank")
    generation = _mapping(payload, "generation")
    raw_experts = bank.get("expert_locks")
    if not isinstance(raw_experts, list):
        raise ProtocolError("Generation lock lacks expert identities.")
    generation_seeds = _int_tuple(generation.get("generation_seeds"))
    namespace = str(generation.get("source_stream_namespace", ""))
    if namespace != SOURCE_STREAM_NAMESPACE:
        raise ProtocolError("Generation source-stream namespace drifted.")
    keys: list[SourceGenerationKey] = []
    for raw in raw_experts:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Generation lock contains an invalid expert identity.")
        center = str(raw.get("source_center"))
        training_seed = int(raw.get("training_seed", -1))
        expert_lock_hash = str(raw.get("expert_lock_hash", ""))
        for generation_seed in generation_seeds:
            seed_by_label = {
                str(label): derived_generation_seed(
                    namespace=namespace,
                    bank_lock_hash=lock.bank_lock_hash,
                    expert_lock_hash=expert_lock_hash,
                    generation_seed=generation_seed,
                    class_label=label,
                )
                for label in (0, 1)
            }
            identity = {
                "namespace": namespace,
                "bank_lock_hash": lock.bank_lock_hash,
                "expert_lock_hash": expert_lock_hash,
                "source_center": center,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
            }
            keys.append(
                SourceGenerationKey(
                    source_center=center,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    expert_lock_hash=expert_lock_hash,
                    stream_id=stable_hash(identity),
                    class_seed_by_label=seed_by_label,
                    max_samples_per_class=int(generation["max_source_block_per_class"]),
                    equal_union_prefix_per_class=int(
                        generation["equal_union_source_budget_per_class"]
                    ),
                )
            )
    return tuple(
        sorted(keys, key=lambda key: (key.source_center, key.training_seed, key.generation_seed))
    )


def equal_union_replicate_plan(lock: GenerationLock) -> tuple[ControlReplicate, ...]:
    payload = lock.to_payload()
    bank = _mapping(payload, "bank")
    generation = _mapping(payload, "generation")
    candidate_raw = bank.get("candidate_sources_by_target")
    if not isinstance(candidate_raw, Mapping):
        raise ProtocolError("Generation lock lacks target-excluded source pools.")
    streams = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in source_generation_plan(lock)
    }
    training_seeds = _int_tuple(generation.get("training_seeds"))
    generation_seeds = _int_tuple(generation.get("generation_seeds"))
    rows: list[ControlReplicate] = []
    for target, raw_sources in sorted(candidate_raw.items(), key=lambda item: str(item[0])):
        sources = _str_tuple(raw_sources)
        if str(target) in sources:
            raise ProtocolError("Equal-union generation plan includes the target expert.")
        for training_seed in training_seeds:
            for generation_seed in generation_seeds:
                source_stream_ids = tuple(
                    streams[(source, training_seed, generation_seed)].stream_id
                    for source in sources
                )
                identity = {
                    "generation_lock_hash": lock.generation_lock_hash,
                    "target_center": str(target),
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "source_stream_ids": list(source_stream_ids),
                }
                rows.append(
                    ControlReplicate(
                        target_center=str(target),
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        replicate_id=stable_hash(identity),
                        candidate_source_centers=sources,
                        source_stream_ids=source_stream_ids,
                        class_shuffle_seed_by_label={
                            str(class_label): derived_composition_seed(
                                generation_lock_hash=lock.generation_lock_hash,
                                target_center=str(target),
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                class_label=class_label,
                            )
                            for class_label in (0, 1)
                        },
                        source_budget_per_class=int(
                            generation["equal_union_source_budget_per_class"]
                        ),
                        total_per_class=int(generation["total_per_class"]),
                    )
                )
    return tuple(rows)


def generate_source_block(
    expert: RoutingAuthorizedExpert,
    key: SourceGenerationKey,
    *,
    per_class: int,
    device: str,
) -> GeneratedBlock:
    """Generate one deterministic maximum-prefix-compatible source block."""

    count = int(per_class)
    if count <= 0 or count > key.max_samples_per_class:
        raise ProtocolError("Requested source block exceeds the frozen per-class budget.")
    if (
        expert.source_center != key.source_center
        or expert.training_seed != key.training_seed
        or expert.expert_lock_hash != key.expert_lock_hash
    ):
        raise ProtocolError("Loaded expert does not match the frozen source-generation key.")
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    frame = TorchOptimizedFrame(expert.source_frame, device=device)
    expert.model.eval()
    with torch.no_grad():
        for class_label in (0, 1):
            class_labels = np.full(count, class_label, dtype=np.int64)
            latents = sample_latents(
                expert.sampler,
                class_labels,
                seed=int(key.class_seed_by_label[str(class_label)]),
            )
            if np.asarray(latents).shape != (count, expert.sampler.latent_dim) or not np.isfinite(
                latents
            ).all():
                raise ProtocolError("Generated latent block failed shape or finiteness checks.")
            z = torch.as_tensor(latents, dtype=torch.float32, device=device)
            y = torch.full((count,), class_label, dtype=torch.long, device=device)
            projected = expert.model.decode(z, y)
            if projected.shape != (count, 256) or not bool(torch.isfinite(projected).all()):
                raise ProtocolError("Decoded model-space block failed shape or finiteness checks.")
            raw = frame.inverse_transform(projected)
            if raw.shape != (count, COMMON_OUTPUT_DIM) or not bool(torch.isfinite(raw).all()):
                raise ProtocolError("Reconstructed embedding block failed shape or finiteness checks.")
            blocks.append(raw.detach().cpu().numpy().astype(np.float32, copy=False))
            labels.append(class_labels)
    embeddings = np.concatenate(blocks, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    if embeddings.shape != (2 * count, COMMON_OUTPUT_DIM) or not np.isfinite(embeddings).all():
        raise ProtocolError("Generated source block failed shape or finiteness checks.")
    return GeneratedBlock(
        key=key,
        embeddings=embeddings,
        labels=labels_np,
        output_sha256=_array_bundle_sha256(embeddings, labels_np),
    )


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (embeddings, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Generation payload lacks mapping {key!r}.")
    return value


def _int_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Generation payload expected an integer list.")
    return tuple(int(item) for item in value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Generation payload expected a source-center list.")
    return tuple(str(item) for item in value)


__all__ = (
    "GeneratedBlock",
    "derived_composition_seed",
    "derived_generation_seed",
    "equal_union_replicate_plan",
    "generate_source_block",
    "source_generation_plan",
)
