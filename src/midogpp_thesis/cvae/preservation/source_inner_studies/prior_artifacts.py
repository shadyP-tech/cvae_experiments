"""Artifact writer for the learned conditional-prior source-inner study."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import torch

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError

from ...reporting import write_json
from .validation_common import write_common_artifacts


PRIOR_STATE_INDEX_SCHEMA = "midogpp_learned_conditional_prior_state_index_v2"


def learned_prior_partition_hash(prior_mu: object, prior_rho: object) -> str:
    """Hash learned-prior parameters with a canonical CPU derivation."""

    try:
        canonical_mu = torch.as_tensor(
            prior_mu, dtype=torch.float32, device="cpu"
        ).detach()
        canonical_rho = torch.as_tensor(
            prior_rho, dtype=torch.float32, device="cpu"
        ).detach()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProtocolError("Cannot canonicalize learned-prior parameters.") from exc
    if (
        canonical_mu.shape != canonical_rho.shape
        or canonical_mu.ndim != 2
        or not torch.isfinite(canonical_mu).all()
        or not torch.isfinite(canonical_rho).all()
    ):
        raise ProtocolError("Learned-prior parameters are malformed.")
    effective_logvar = 6.0 * torch.tanh(canonical_rho / 6.0)
    return stable_hash(
        {
            "prior_mu": canonical_mu.tolist(),
            "prior_rho": canonical_rho.tolist(),
            "effective_logvar": effective_logvar.tolist(),
        }
    )


def write_prior_study_bundle(
    root: Path,
    *,
    learned_prior_state_index: Mapping[str, object],
    metric_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    checkpoint_reuse_rows: Sequence[Mapping[str, object]],
    initialization_pairing_rows: Sequence[Mapping[str, object]],
    generation_budget_rows: Sequence[Mapping[str, object]],
    rng_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    protocol_manifest: Mapping[str, object],
    coverage_manifest: Mapping[str, object],
    selection_evidence_manifest: Mapping[str, object],
    embedded_preparation_lineage: Mapping[str, object],
    generation_budget_manifest: Mapping[str, object],
    child_decisions: Mapping[tuple[int, str], Mapping[str, object]],
    consensus_decisions: Mapping[str, Mapping[str, object]],
    study_decision: Mapping[str, object],
    leakage_report: Mapping[str, object],
) -> Path:
    root = write_common_artifacts(
        root,
        metric_rows=metric_rows,
        paired_delta_rows=paired_delta_rows,
        nested_reference_rows=nested_reference_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        checkpoint_reuse_rows=checkpoint_reuse_rows,
        initialization_pairing_rows=initialization_pairing_rows,
        generation_budget_rows=generation_budget_rows,
        rng_rows=rng_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol_manifest,
        coverage_manifest=coverage_manifest,
        selection_evidence_manifest=selection_evidence_manifest,
        embedded_preparation_lineage=embedded_preparation_lineage,
        generation_budget_manifest=generation_budget_manifest,
        child_decisions=child_decisions,
        consensus_decisions=consensus_decisions,
        study_decision=study_decision,
        leakage_report=leakage_report,
    )
    write_json(
        root / "manifests/learned_prior_state_index.json",
        learned_prior_state_index,
    )
    return root

