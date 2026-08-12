"""Experiment-local adapter to the neutral variational compatibility score."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...routing.dense_residual_soft_router import score_variational_compatibility


def score_label_free_support(
    expert: object,
    common_embeddings: np.ndarray,
    case_ids: Sequence[str],
) -> object:
    """Score opaque support embeddings under both fixed class hypotheses."""

    return score_variational_compatibility(expert, common_embeddings, case_ids)


__all__ = ("score_label_free_support",)
