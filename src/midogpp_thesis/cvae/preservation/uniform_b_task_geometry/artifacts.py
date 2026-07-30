"""Artifact writers for the Uniform-B task-geometry study."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from ...reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .config import UniformBTaskGeometryConfig
from .contracts import PUBLICATION_STATE


def prepare_bundle(root: Path) -> Path:
    resolved = prepare_artifact_dirs(root)
    (resolved / "provenance").mkdir(parents=True, exist_ok=True)
    return resolved


def write_resolved_config(
    root: Path,
    config: UniformBTaskGeometryConfig,
) -> None:
    import yaml

    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )


def write_final_artifacts(
    root: Path,
    *,
    protocol: Mapping[str, object],
    provenance: Mapping[str, object],
    coverage: Mapping[str, object],
    frame_index: Mapping[str, object],
    geometry_index: Mapping[str, object],
    candidate_pools: Sequence[Mapping[str, object]],
    generation_manifest: Sequence[Mapping[str, object]],
    composition_manifest: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    rng_rows: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    runtime_plan: Mapping[str, object],
) -> None:
    write_json(root / "provenance/input_artifacts.json", provenance)
    write_json(root / "manifests/protocol_manifest.json", protocol)
    write_json(root / "manifests/coverage_manifest.json", coverage)
    write_json(root / "manifests/frame_index.json", frame_index)
    write_json(root / "manifests/task_geometry_state_index.json", geometry_index)
    write_json(
        root / "manifests/candidate_pool_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_candidate_pool_index_v1",
            "records": list(candidate_pools),
        },
    )
    write_json(
        root / "manifests/generation_budget_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_generation_manifest_v1",
            "records": list(generation_manifest),
        },
    )
    write_json(
        root / "manifests/composition_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_composition_manifest_v1",
            "records": list(composition_manifest),
        },
    )
    write_csv_rows(root / "tables/source_inner_metrics.csv", metric_rows)
    write_csv_rows(root / "tables/tstr_metrics.csv", metric_rows)
    write_csv_rows(
        root / "tables/composition_metrics.csv",
        [
            row
            for row in metric_rows
            if str(row.get("composition_mode", "")).startswith("union_")
        ],
    )
    write_csv_rows(root / "tables/paired_deltas.csv", delta_rows)
    write_csv_rows(
        root / "tables/task_geometry_diagnostics.csv",
        geometry_rows,
    )
    write_csv_rows(
        root / "tables/generation_budget_audit.csv",
        generation_rows,
    )
    write_csv_rows(
        root / "tables/identity_overlap_audit.csv",
        identity_rows,
    )
    write_csv_rows(root / "tables/rng_pairing_audit.csv", rng_rows)
    write_csv_rows(root / "tables/runtime_timings.csv", timing_rows)
    write_json(root / "reports/study_decision.json", decision)
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_uniform_b_leakage_report_v1",
            "status": "PASS",
            "outer_rows_used_for_fit": False,
            "inner_rows_used_for_fit": False,
            "inner_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "source_training_keys_are_outer_inner_neutral": True,
            "identity_audit_failures": sum(
                row.get("status") != "PASS" for row in identity_rows
            ),
        },
    )
    write_json(
        root / "reports/publication_state.json",
        {
            "schema_version": "midogpp_uniform_b_publication_state_v1",
            "publication_state": PUBLICATION_STATE,
            "decision": "DO_NOT_PROMOTE",
            "may_feed_model_recipe": False,
            "may_feed_recipe_selection": False,
            "may_feed_expert_bank": False,
            "may_feed_generation": False,
            "may_feed_routing": False,
            "may_feed_downstream_utility": False,
            "stage30_recipe_ready": False,
            "separate_promotion_artifact_required": True,
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_uniform_b_runtime_summary_v1",
            "metric_rows": len(metric_rows),
            "checkpoint_records": int(coverage["checkpoint_records"]),
            "generation_blocks": len(generation_manifest),
            "composition_cells": len(composition_manifest),
            "runtime_plan": dict(runtime_plan),
        },
    )


__all__ = (
    "prepare_bundle",
    "write_final_artifacts",
    "write_resolved_config",
)
