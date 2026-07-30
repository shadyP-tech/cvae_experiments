"""Synthetic-only TSTR scoring and anti-prototype diversity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ..scoring import score_representation
from .composition import ComposedSynthetic


@dataclass(frozen=True)
class TSTRDiagnostic:
    bacc: float
    macro_f1: float
    converged: bool
    classifier_spec_hash: str
    diversity: Mapping[str, float]


def evaluate_tstr_diagnostic(
    synthetic: ComposedSynthetic,
    eval_embeddings: Sequence[Sequence[float]],
    eval_labels: Sequence[int],
    *,
    classifier_spec: ClassifierSpec,
) -> TSTRDiagnostic:
    """Fit only synthetic rows; I labels are consumed only by metric scoring."""

    x_eval = np.asarray(eval_embeddings, dtype=np.float32)
    y_eval = np.asarray(eval_labels, dtype=np.int64)
    if (
        synthetic.embeddings.ndim != 2
        or synthetic.embeddings.shape[1] != 3840
        or x_eval.ndim != 2
        or x_eval.shape[1] != 3840
        or set(synthetic.labels.tolist()) != {0, 1}
        or set(y_eval.tolist()) != {0, 1}
    ):
        raise ProtocolError("TSTR diagnostic arrays are invalid.")
    score = score_representation(
        synthetic.embeddings,
        synthetic.labels,
        x_eval,
        y_eval,
        spec=classifier_spec,
    )
    diversity = diversity_diagnostics(
        synthetic.embeddings,
        synthetic.labels,
        x_eval,
        y_eval,
    )
    return TSTRDiagnostic(
        bacc=score.bacc,
        macro_f1=score.macro_f1,
        converged=score.converged,
        classifier_spec_hash=score.classifier_spec_hash,
        diversity=diversity,
    )


def diversity_diagnostics(
    generated: np.ndarray,
    generated_labels: np.ndarray,
    real: np.ndarray,
    real_labels: np.ndarray,
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for cls in (0, 1):
        gen = generated[generated_labels == cls]
        ref = real[real_labels == cls]
        gen_rank = _effective_rank(gen)
        ref_rank = _effective_rank(ref)
        rows[f"class_{cls}_effective_rank_ratio"] = gen_rank / max(ref_rank, 1e-12)
        gen_quantiles = _distance_quantiles(gen)
        ref_quantiles = _distance_quantiles(ref)
        for name, gen_value, ref_value in zip(
            ("q10", "q50", "q90"),
            gen_quantiles,
            ref_quantiles,
        ):
            rows[f"class_{cls}_pairwise_{name}_ratio"] = gen_value / max(
                ref_value,
                1e-12,
            )
    return rows


def _effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(
        centered[: min(len(centered), 512)],
        compute_uv=False,
        full_matrices=False,
    )
    energy = singular**2
    probabilities = energy / max(float(energy.sum()), 1e-12)
    entropy = -float(
        np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
    )
    return float(np.exp(entropy))


def _distance_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    subset = values[: min(len(values), 512)]
    if len(subset) < 2:
        return (0.0, 0.0, 0.0)
    differences = subset[:, None, :] - subset[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper = distances[np.triu_indices(len(subset), k=1)]
    return tuple(float(value) for value in np.quantile(upper, (0.1, 0.5, 0.9)))


__all__ = (
    "TSTRDiagnostic",
    "diversity_diagnostics",
    "evaluate_tstr_diagnostic",
)
