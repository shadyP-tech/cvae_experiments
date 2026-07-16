"""Artifact writer for the shrunk Task-Fisher source-inner study."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...reporting import write_json
from .validation_common import write_common_artifacts


FISHER_STATE_INDEX_SCHEMA = "midogpp_task_fisher_shrinkage_state_index_v2"


def write_fisher_study_bundle(
    root: Path,
    *,
    task_fisher_state_index: Mapping[str, object],
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
        root / "manifests/task_fisher_shrinkage_state_index.json",
        task_fisher_state_index,
    )
    return root

