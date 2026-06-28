"""Protocol-aware candidate registry for downstream selection experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..baselines import assert_deployable_candidate_pool, assert_oracle_rows_diagnostic_only
from ..protocol import ProtocolError
from ..schemas import CandidateManifestRow, DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE


def build_candidate_manifest(
    *,
    heldout_target: str,
    checkpoint_rows: Sequence[Mapping[str, object]],
    generation_mode: str,
    latent_sampling_setting: str,
    class_prior_rule: str,
    synthetic_budget: int,
    generation_seed: int,
    classifier_seed: int,
    config_hash: str,
    protocol_hash: str,
    include_target_oracle: bool = False,
) -> tuple[CandidateManifestRow, ...]:
    """Build atomic candidate rows with held-out target exclusion by default."""

    candidates: list[CandidateManifestRow] = []
    for row in checkpoint_rows:
        source_domain = str(row.get("source_domain") or row.get("expert_domain") or "")
        checkpoint_id = str(row.get("expert_checkpoint_id") or row.get("checkpoint_path") or "")
        if not source_domain or not checkpoint_id:
            raise ProtocolError(f"Checkpoint row lacks source domain/checkpoint id: {row}")
        is_target = source_domain == str(heldout_target)
        if is_target and not include_target_oracle:
            continue
        eligibility = DIAGNOSTIC_ONLY if is_target else SELECTION_ELIGIBLE
        candidate_id = "|".join(
            [
                f"expert={checkpoint_id}",
                f"source={source_domain}",
                f"mode={generation_mode}",
                f"budget={synthetic_budget}",
                f"gseed={generation_seed}",
                f"cseed={classifier_seed}",
            ]
        )
        candidates.append(
            CandidateManifestRow(
                candidate_id=candidate_id,
                expert_checkpoint_id=checkpoint_id,
                source_domain=source_domain,
                checkpoint_seed=int(row.get("checkpoint_seed") or row.get("experiment_seed") or 0),
                generation_mode=generation_mode,
                latent_sampling_setting=latent_sampling_setting,
                class_prior_rule=class_prior_rule,
                synthetic_budget=int(synthetic_budget),
                generation_seed=int(generation_seed),
                aggregation_recipe=str(row.get("aggregation_recipe") or "single_candidate"),
                classifier_seed=int(classifier_seed),
                expert_checkpoint_hash=str(row.get("expert_checkpoint_hash") or row.get("checkpoint_hash") or ""),
                config_hash=config_hash,
                protocol_hash=protocol_hash,
                eligibility=eligibility,
            )
        )
    candidate_dicts = [candidate.to_row() for candidate in candidates]
    assert_deployable_candidate_pool(heldout_target=str(heldout_target), candidate_rows=candidate_dicts)
    assert_oracle_rows_diagnostic_only(
        [row | {"role": "oracle_reference"} for row in candidate_dicts if row["eligibility"] == DIAGNOSTIC_ONLY]
    )
    return tuple(candidates)


def selection_eligible_candidates(
    candidates: Sequence[CandidateManifestRow],
) -> tuple[CandidateManifestRow, ...]:
    return tuple(candidate for candidate in candidates if candidate.eligibility == SELECTION_ELIGIBLE)
