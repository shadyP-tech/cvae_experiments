from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from protocol import ProtocolError, split_budget


GENERATION_MODE = "class_stratified_reference_posterior"


@dataclass(frozen=True)
class SyntheticBatch:
    expert_id: str
    embeddings: object
    labels: tuple[int, ...]
    generation_mode: str = GENERATION_MODE


def generate_reference_posterior(
    *,
    model: object,
    expert_id: str,
    source_embeddings_by_class: Mapping[int, object],
    budget_per_class: int,
    generation_seed: int,
) -> SyntheticBatch:
    """Generate class-stratified synthetic embeddings from source posterior refs."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reference-posterior generation requires numpy and torch.") from exc

    rng = np.random.default_rng(int(generation_seed))
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(generation_seed))
    chunks = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for cls in sorted(int(v) for v in source_embeddings_by_class):
            refs = np.asarray(source_embeddings_by_class[cls], dtype=np.float32)
            if refs.ndim != 2 or refs.shape[0] == 0:
                raise ProtocolError(f"Empty source reference pool for class {cls}.")
            chosen = refs[rng.integers(0, refs.shape[0], size=int(budget_per_class))]
            x = torch.as_tensor(chosen, dtype=torch.float32)
            y = torch.full((x.shape[0],), int(cls), dtype=torch.long)
            mu, logvar = model.encode(x, y)
            noise = torch.randn(mu.shape, generator=torch_generator, dtype=mu.dtype, device=mu.device)
            z = mu + (noise * torch.exp(0.5 * logvar))
            generated = model.decode(z, y).detach().cpu().numpy()
            chunks.append(generated)
            labels.extend([int(cls)] * int(budget_per_class))
    return SyntheticBatch(
        expert_id=str(expert_id),
        embeddings=np.vstack(chunks),
        labels=tuple(labels),
    )


def generation_budgets(total_per_class: int, ranked_experts: Sequence[str], k: int) -> dict[str, int]:
    selected = tuple(str(v) for v in ranked_experts[: int(k)])
    return split_budget(int(total_per_class), selected)
