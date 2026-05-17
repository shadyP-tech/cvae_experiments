"""Synthetic embedding generation boundary.

The concrete CVAE backend is injected so this package remains a consumer of
frozen expert artifacts. The primary v1 mode is not true class-conditional
generation: labels enter through the source reference pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .protocol import ProtocolError
from .schemas import (
    NEGATIVE_CONTROL_GENERATION_MODE,
    PRIMARY_BUDGET_PER_CLASS,
    PRIMARY_GENERATION_MODE,
)


@dataclass(frozen=True)
class GenerationRequest:
    expert_domain: str
    generation_mode: str
    budget_per_class: int
    generation_seed: int
    reference_pool_scope: str = "expert_source_train"


@dataclass(frozen=True)
class SyntheticBatch:
    expert_domain: str
    generation_mode: str
    projection_frame: str
    embeddings: object
    labels: object


class ReferencePosteriorBackend(Protocol):
    """Minimal backend expected from the frozen CVAE expert bank."""

    def generate_from_reference(self, domain: int, x_ref: object, n_samples: int, seed: int) -> object:
        ...

    def sample_prior(self, domain: int, n_samples: int, seed: int) -> object:
        ...


def validate_generation_request(request: GenerationRequest) -> None:
    if request.generation_mode not in {
        PRIMARY_GENERATION_MODE,
        NEGATIVE_CONTROL_GENERATION_MODE,
    }:
        raise ProtocolError(f"Unknown generation mode: {request.generation_mode}")
    if request.budget_per_class <= 0:
        raise ProtocolError("budget_per_class must be positive.")
    if request.generation_mode == PRIMARY_GENERATION_MODE:
        if request.reference_pool_scope != "expert_source_train":
            raise ProtocolError("Primary generation must use expert_source_train reference pools.")
    if request.generation_mode == NEGATIVE_CONTROL_GENERATION_MODE:
        if request.budget_per_class == PRIMARY_BUDGET_PER_CLASS:
            # Allowed, but downstream gates must still exclude this mode. The
            # guard lives here to keep the diagnostic status visible in code.
            return


def allocate_equal_budget_per_class(class_labels: Sequence[int], budget_per_class: int) -> dict[int, int]:
    labels = sorted({int(label) for label in class_labels})
    if not labels:
        raise ProtocolError("Cannot allocate generation budget without class labels.")
    if budget_per_class <= 0:
        raise ProtocolError("budget_per_class must be positive.")
    return {label: int(budget_per_class) for label in labels}


def allocate_equal_total_ensemble_budget(
    *,
    total_per_class: int,
    candidate_experts: Sequence[str],
) -> dict[str, int]:
    """Split a single-expert-equivalent per-class budget across experts."""

    if total_per_class <= 0:
        raise ProtocolError("total_per_class must be positive.")
    if not candidate_experts:
        raise ProtocolError("candidate_experts is empty.")
    n = len(candidate_experts)
    base = total_per_class // n
    remainder = total_per_class % n
    allocation: dict[str, int] = {}
    for index, expert in enumerate(sorted(str(v) for v in candidate_experts)):
        allocation[expert] = base + (1 if index < remainder else 0)
    return allocation


def generate_reference_posterior_resampled_embeddings(
    backend: ReferencePosteriorBackend,
    request: GenerationRequest,
    *,
    source_reference_embeddings_by_class: Mapping[int, object],
) -> SyntheticBatch:
    """Generate primary v1 synthetic embeddings from labeled source refs.

    The backend returns embeddings in the selected expert's projected CVAE
    feature frame. Target evaluation must be projected through the same expert
    head before downstream scoring.
    """

    validate_generation_request(request)
    if request.generation_mode != PRIMARY_GENERATION_MODE:
        raise ProtocolError("Reference-posterior resampling requires the primary generation mode.")

    chunks: list[object] = []
    labels: list[int] = []
    for class_label in sorted(int(v) for v in source_reference_embeddings_by_class):
        refs = source_reference_embeddings_by_class[class_label]
        generated = backend.generate_from_reference(
            int(request.expert_domain),
            refs,
            int(request.budget_per_class),
            seed=int(request.generation_seed) + int(class_label),
        )
        chunks.append(generated)
        labels.extend([int(class_label)] * int(request.budget_per_class))
    return SyntheticBatch(
        expert_domain=str(request.expert_domain),
        generation_mode=request.generation_mode,
        projection_frame=str(request.expert_domain),
        embeddings=chunks,
        labels=labels,
    )


def generate_unconditional_prior_negative_control(
    backend: ReferencePosteriorBackend,
    request: GenerationRequest,
    *,
    class_labels: Sequence[int],
) -> SyntheticBatch:
    """Diagnostic negative control with externally assigned balanced labels."""

    validate_generation_request(request)
    if request.generation_mode != NEGATIVE_CONTROL_GENERATION_MODE:
        raise ProtocolError("Prior negative control requires the negative-control generation mode.")
    counts = allocate_equal_budget_per_class(class_labels, request.budget_per_class)
    total = sum(counts.values())
    embeddings = backend.sample_prior(int(request.expert_domain), total, int(request.generation_seed))
    labels: list[int] = []
    for class_label, count in counts.items():
        labels.extend([int(class_label)] * int(count))
    return SyntheticBatch(
        expert_domain=str(request.expert_domain),
        generation_mode=request.generation_mode,
        projection_frame=str(request.expert_domain),
        embeddings=embeddings,
        labels=labels,
    )
