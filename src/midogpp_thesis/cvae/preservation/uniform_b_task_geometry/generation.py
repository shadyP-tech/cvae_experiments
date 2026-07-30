"""Deterministic balanced prior/posterior generation in common Uniform-B space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ....common.hashing import stable_hash
from ...keyed_training import derived_seed, torch_generator
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from .frame import SourceBlockFrame, TorchBlockFrame


@dataclass(frozen=True)
class GeneratedBlock:
    source_center: str
    arm: str
    training_seed: int
    generation_seed: int
    embeddings: np.ndarray
    labels: np.ndarray
    per_class: int
    checkpoint_hash: str
    frame_hash: str
    stream_hash: str
    kind: str = "prior"

    @property
    def block_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_uniform_b_generated_block_v1",
                "source_center": self.source_center,
                "arm": self.arm,
                "training_seed": self.training_seed,
                "generation_seed": self.generation_seed,
                "per_class": self.per_class,
                "checkpoint_hash": self.checkpoint_hash,
                "frame_hash": self.frame_hash,
                "stream_hash": self.stream_hash,
                "kind": self.kind,
                "embedding_shape": list(self.embeddings.shape),
                "embedding_sha256": _array_hash(self.embeddings),
                "label_sha256": _array_hash(self.labels),
            }
        )


def generate_prior_block(
    model: ClassConditionedCVAE,
    source_frame: SourceBlockFrame,
    *,
    source_center: str,
    arm: str,
    training_seed: int,
    generation_seed: int,
    per_class: int,
    checkpoint_hash: str,
    device: str,
) -> GeneratedBlock:
    if per_class <= 0:
        raise ProtocolError("Generation per-class budget must be positive.")
    labels = np.asarray(
        [0] * per_class + [1] * per_class,
        dtype=np.int64,
    )
    stream_identity = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_prior_stream_v1",
            "source_center": source_center,
            "checkpoint_hash": checkpoint_hash,
            "arm": arm,
            "training_seed": int(training_seed),
            "generation_seed": int(generation_seed),
            "labels": labels.tolist(),
            "outer_or_inner_identity_present": False,
        }
    )
    generator = torch_generator(
        device,
        derived_seed(stream_identity, "standard_normal"),
    )
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    z = torch.randn(
        (len(labels), model.latent_dim),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    adapter = TorchBlockFrame(source_frame, device=device)
    model.eval()
    with torch.no_grad():
        projected = model.decode(z, y)
        common = adapter.inverse_transform(projected)
    if common.shape != (2 * per_class, 3840) or not torch.isfinite(common).all():
        raise ProtocolError("Prior generation produced an invalid common-frame block.")
    return GeneratedBlock(
        source_center=str(source_center),
        arm=str(arm),
        training_seed=int(training_seed),
        generation_seed=int(generation_seed),
        embeddings=common.detach().cpu().numpy().astype(np.float32),
        labels=labels,
        per_class=int(per_class),
        checkpoint_hash=str(checkpoint_hash),
        frame_hash=source_frame.state_hash,
        stream_hash=stream_identity,
    )


def generate_posterior_block(
    model: ClassConditionedCVAE,
    source_frame: SourceBlockFrame,
    source_projected: np.ndarray,
    source_labels: np.ndarray,
    *,
    source_center: str,
    arm: str,
    training_seed: int,
    generation_seed: int,
    per_class: int,
    checkpoint_hash: str,
    device: str,
) -> GeneratedBlock:
    """Generate a matched source-posterior diagnostic with the same budget."""

    if (
        source_projected.ndim != 2
        or source_projected.shape[1] != 128
        or len(source_projected) != len(source_labels)
        or set(source_labels.tolist()) != {0, 1}
        or per_class <= 0
    ):
        raise ProtocolError("Posterior generation source arrays are invalid.")
    stream_identity = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_posterior_stream_v1",
            "source_center": source_center,
            "checkpoint_hash": checkpoint_hash,
            "arm": arm,
            "training_seed": int(training_seed),
            "generation_seed": int(generation_seed),
            "per_class": int(per_class),
            "outer_or_inner_identity_present": False,
        }
    )
    rng = np.random.default_rng(derived_seed(stream_identity, "source_rows"))
    indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(source_labels == cls),
                size=per_class,
                replace=True,
            )
            for cls in (0, 1)
        ]
    ).astype(np.int64)
    labels = source_labels[indices].astype(np.int64)
    x = torch.as_tensor(
        source_projected[indices],
        dtype=torch.float32,
        device=device,
    )
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    epsilon_generator = torch_generator(
        device,
        derived_seed(stream_identity, "posterior_epsilon"),
    )
    adapter = TorchBlockFrame(source_frame, device=device)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x, y)
        epsilon = torch.randn(
            mu.shape,
            generator=epsilon_generator,
            dtype=mu.dtype,
            device=mu.device,
        )
        projected = model.decode(
            mu + epsilon * torch.exp(0.5 * logvar),
            y,
        )
        common = adapter.inverse_transform(projected)
    if common.shape != (2 * per_class, 3840) or not torch.isfinite(common).all():
        raise ProtocolError(
            "Posterior generation produced an invalid common-frame block."
        )
    return GeneratedBlock(
        source_center=str(source_center),
        arm=str(arm),
        training_seed=int(training_seed),
        generation_seed=int(generation_seed),
        embeddings=common.detach().cpu().numpy().astype(np.float32),
        labels=labels,
        per_class=int(per_class),
        checkpoint_hash=str(checkpoint_hash),
        frame_hash=source_frame.state_hash,
        stream_hash=stream_identity,
        kind="posterior",
    )


def _array_hash(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(tuple(array.shape)).encode("ascii"))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


__all__ = (
    "GeneratedBlock",
    "generate_posterior_block",
    "generate_prior_block",
)
