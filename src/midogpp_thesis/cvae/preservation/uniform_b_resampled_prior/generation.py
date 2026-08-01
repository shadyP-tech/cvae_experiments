"""Paired P0/Pq generation from one deterministic Gaussian candidate bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from ....common.hashing import stable_hash
from ...keyed_training import derived_seed, torch_generator
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from ..uniform_b_task_geometry.frame import SourceBlockFrame, TorchBlockFrame
from ..uniform_b_task_geometry.generation import GeneratedBlock
from .config import UniformBResampledPriorConfig
from .contracts import P0, PQ
from .ratio import PosteriorRatioState


@dataclass(frozen=True)
class GenerationAudit:
    source_center: str
    training_seed: int
    generation_seed: int
    class_label: int
    prior: str
    candidate_count: int
    requested_count: int
    empirical_acceptance_rate: float
    expected_acceptance_rate: float
    ess_ratio: float
    ratio_reliable: bool
    fallback_to_p0: bool
    candidate_bank_hash: str
    ratio_state_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_resampled_prior_generation_audit_v1",
            **self.__dict__,
            "outer_or_inner_identity_present": False,
        }


def generate_paired_prior_blocks(
    model: ClassConditionedCVAE,
    source_frame: SourceBlockFrame,
    ratio_state: PosteriorRatioState,
    *,
    source_center: str,
    training_seed: int,
    generation_seed: int,
    per_class: int,
    checkpoint_hash: str,
    config: UniformBResampledPriorConfig,
    device: str,
) -> tuple[dict[str, GeneratedBlock], tuple[GenerationAudit, ...]]:
    if ratio_state.checkpoint_hash != checkpoint_hash or ratio_state.source_center != str(source_center):
        raise ProtocolError("Ratio state/checkpoint identity mismatch.")
    selected: dict[str, list[torch.Tensor]] = {P0: [], PQ: []}
    labels_by_prior: dict[str, list[np.ndarray]] = {P0: [], PQ: []}
    audits: list[GenerationAudit] = []
    stream_identity = stable_hash(
        {
            "schema_version": "midogpp_resampled_prior_candidate_stream_v1",
            "source_center": source_center,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "checkpoint_hash": checkpoint_hash,
            "ratio_state_hash": ratio_state.state_hash,
            "per_class": per_class,
            "proposal_multiplier": config.proposal_multiplier,
            "outer_or_inner_identity_present": False,
        }
    )
    candidate_count = config.proposal_multiplier * per_class
    for class_label in (0, 1):
        generator = torch_generator(
            device,
            derived_seed(stream_identity, class_label, "candidate_bank"),
        )
        candidates = torch.randn(
            (candidate_count, model.latent_dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        candidate_array = candidates.detach().cpu().numpy()
        contiguous_candidates = np.ascontiguousarray(candidate_array)
        bank_hash = stable_hash(
            {
                "stream": stream_identity,
                "class_label": class_label,
                "shape": list(candidate_array.shape),
                "dtype": str(candidate_array.dtype),
                "array_sha256": hashlib.sha256(
                    contiguous_candidates.tobytes()
                ).hexdigest(),
            }
        )
        class_state = ratio_state.classes[class_label]
        weights = class_state.acceptance(candidate_array, config=config)
        expected_rate = float(weights.mean())
        ess_ratio = float((weights.sum() ** 2) / max(float(np.square(weights).sum()) * len(weights), 1e-12))
        reliable = bool(
            class_state.reliable
            and expected_rate >= config.min_acceptance_rate
            and ess_ratio >= config.min_ess_ratio
        )
        rng = np.random.default_rng(derived_seed(stream_identity, class_label, "accept_uniform"))
        accepted = np.flatnonzero(rng.random(candidate_count) < weights)
        fallback = not reliable or len(accepted) < per_class
        pq_indices = np.arange(per_class) if fallback else accepted[:per_class]
        selected[P0].append(candidates[:per_class])
        selected[PQ].append(candidates[torch.as_tensor(pq_indices, dtype=torch.long, device=device)])
        for prior in (P0, PQ):
            labels_by_prior[prior].append(np.full(per_class, class_label, dtype=np.int64))
            audits.append(
                GenerationAudit(
                    source_center=str(source_center),
                    training_seed=int(training_seed),
                    generation_seed=int(generation_seed),
                    class_label=class_label,
                    prior=prior,
                    candidate_count=candidate_count,
                    requested_count=per_class,
                    empirical_acceptance_rate=(1.0 if prior == P0 else float(len(accepted) / candidate_count)),
                    expected_acceptance_rate=(1.0 if prior == P0 else expected_rate),
                    ess_ratio=(1.0 if prior == P0 else ess_ratio),
                    ratio_reliable=(True if prior == P0 else class_state.reliable),
                    fallback_to_p0=(False if prior == P0 else fallback),
                    candidate_bank_hash=bank_hash,
                    ratio_state_hash=ratio_state.state_hash,
                )
            )
    adapter = TorchBlockFrame(source_frame, device=device)
    model.eval()
    blocks: dict[str, GeneratedBlock] = {}
    with torch.no_grad():
        for prior in (P0, PQ):
            z = torch.cat(selected[prior], dim=0)
            labels = np.concatenate(labels_by_prior[prior]).astype(np.int64)
            y = torch.as_tensor(labels, dtype=torch.long, device=device)
            common = adapter.inverse_transform(model.decode(z, y))
            if common.shape != (2 * per_class, 3840) or not torch.isfinite(common).all():
                raise ProtocolError("P0/Pq generation produced an invalid block.")
            blocks[prior] = GeneratedBlock(
                source_center=str(source_center),
                arm=prior,
                training_seed=int(training_seed),
                generation_seed=int(generation_seed),
                embeddings=common.detach().cpu().numpy().astype(np.float32),
                labels=labels,
                per_class=int(per_class),
                checkpoint_hash=str(checkpoint_hash),
                frame_hash=source_frame.state_hash,
                stream_hash=stable_hash({"candidate_stream": stream_identity, "prior": prior}),
                kind="prior",
            )
    return blocks, tuple(audits)


__all__ = ("GenerationAudit", "generate_paired_prior_blocks")
