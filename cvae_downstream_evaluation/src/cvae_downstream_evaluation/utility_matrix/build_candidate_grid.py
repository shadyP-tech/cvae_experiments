"""Candidate-grid expansion helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..schemas import CandidateManifestRow
from .candidate_registry import build_candidate_manifest


def build_candidate_grid(
    *,
    heldout_target: str,
    checkpoint_rows: Sequence[Mapping[str, object]],
    generation_modes: Sequence[str],
    synthetic_budgets: Sequence[int],
    generation_seeds: Sequence[int],
    classifier_seeds: Sequence[int],
    config_hash: str,
    protocol_hash: str,
    latent_sampling_setting: str = "frozen_config",
    class_prior_rule: str = "class_balanced",
    include_target_oracle: bool = False,
) -> tuple[CandidateManifestRow, ...]:
    rows: list[CandidateManifestRow] = []
    for generation_mode in generation_modes:
        for synthetic_budget in synthetic_budgets:
            for generation_seed in generation_seeds:
                for classifier_seed in classifier_seeds:
                    rows.extend(
                        build_candidate_manifest(
                            heldout_target=heldout_target,
                            checkpoint_rows=checkpoint_rows,
                            generation_mode=generation_mode,
                            latent_sampling_setting=latent_sampling_setting,
                            class_prior_rule=class_prior_rule,
                            synthetic_budget=int(synthetic_budget),
                            generation_seed=int(generation_seed),
                            classifier_seed=int(classifier_seed),
                            config_hash=config_hash,
                            protocol_hash=protocol_hash,
                            include_target_oracle=include_target_oracle,
                        )
                    )
    return tuple(rows)
