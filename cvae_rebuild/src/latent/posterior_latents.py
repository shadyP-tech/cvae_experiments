from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PosteriorLatentRows:
    latents: np.ndarray
    labels: np.ndarray
    row_ids: tuple[str, ...]
    split_names: tuple[str, ...]

    def select_split(self, split_name: str) -> "PosteriorLatentRows":
        mask = np.asarray([value == split_name for value in self.split_names], dtype=bool)
        return self.select_mask(mask)

    def select_mask(self, mask: Sequence[bool]) -> "PosteriorLatentRows":
        values = np.asarray(mask, dtype=bool)
        if values.shape != (self.latents.shape[0],):
            raise ValueError("Selection mask must match latent row count.")
        return PosteriorLatentRows(
            latents=self.latents[values],
            labels=self.labels[values],
            row_ids=tuple(row_id for row_id, keep in zip(self.row_ids, values) if bool(keep)),
            split_names=tuple(split for split, keep in zip(self.split_names, values) if bool(keep)),
        )


def build_posterior_latent_rows(
    *,
    latents: object,
    labels: Sequence[int],
    row_ids: Sequence[str],
    split_names: Sequence[str],
) -> PosteriorLatentRows:
    z = np.asarray(latents, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    rows = tuple(str(value) for value in row_ids)
    splits = tuple(str(value) for value in split_names)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("Posterior latents must be a non-empty 2D array.")
    if y.shape != (z.shape[0],) or len(rows) != z.shape[0] or len(splits) != z.shape[0]:
        raise ValueError("Latents, labels, row_ids, and split_names must have matching row counts.")
    if len(set(rows)) != len(rows):
        raise ValueError("Posterior latent row_ids must be unique.")
    return PosteriorLatentRows(latents=z, labels=y, row_ids=rows, split_names=splits)


def split_fit_eval_latents(
    rows: PosteriorLatentRows,
    *,
    fit_split: str = "fit",
    eval_split: str = "eval",
) -> tuple[PosteriorLatentRows, PosteriorLatentRows]:
    fit_rows = rows.select_split(fit_split)
    eval_rows = rows.select_split(eval_split)
    if fit_rows.latents.shape[0] == 0:
        raise ValueError("No fit rows are available for latent-prior fitting.")
    if eval_rows.latents.shape[0] == 0:
        raise ValueError("No eval rows are available for held-out scoring.")
    overlap = set(fit_rows.row_ids).intersection(eval_rows.row_ids)
    if overlap:
        raise ValueError("Fit and eval posterior latent row_ids must be disjoint.")
    return fit_rows, eval_rows
