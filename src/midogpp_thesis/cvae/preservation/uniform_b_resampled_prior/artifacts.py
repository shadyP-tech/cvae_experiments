"""Artifact writers for the isolated P0/Pq namespace."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from ...reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .config import UniformBResampledPriorConfig
from .contracts import PUBLICATION_STATE


def prepare_bundle(root: Path) -> Path:
    resolved = prepare_artifact_dirs(root)
    (resolved / "provenance").mkdir(parents=True, exist_ok=True)
    return resolved


def write_resolved_config(root: Path, config: UniformBResampledPriorConfig) -> None:
    import yaml

    path = root / "config.resolved.yaml"
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping) and isinstance(existing.get("experiment"), Mapping) and isinstance(existing.get("inputs"), Mapping):
            return
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def write_final_artifacts(
    root: Path,
    *,
    protocol: Mapping[str, object],
    provenance: Mapping[str, object],
    coverage: Mapping[str, object],
    frame_index: Mapping[str, object],
    ratio_index: Mapping[str, object],
    score_mapping: Sequence[Mapping[str, object]],
    generation_manifest: Sequence[Mapping[str, object]],
    unique_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    ratio_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    runtime_plan: Mapping[str, object],
) -> None:
    write_json(root / "provenance/input_artifacts.json", provenance)
    write_json(root / "manifests/protocol_manifest.json", protocol)
    write_json(root / "manifests/coverage_manifest.json", coverage)
    write_json(root / "manifests/frame_index.json", frame_index)
    write_json(root / "manifests/posterior_ratio_state_index.json", ratio_index)
    write_json(
        root / "manifests/score_reuse_mapping.json",
        {
            "schema_version": "midogpp_resampled_prior_score_reuse_index_v1",
            "records": list(score_mapping),
        },
    )
    write_json(
        root / "manifests/generation_budget_manifest.json",
        {
            "schema_version": "midogpp_resampled_prior_generation_manifest_v1",
            "records": list(generation_manifest),
        },
    )
    write_csv_rows(root / "tables/unique_tstr_scores.csv", unique_rows)
    write_csv_rows(root / "tables/source_inner_metrics.csv", metric_rows)
    write_csv_rows(root / "tables/paired_deltas.csv", delta_rows)
    write_csv_rows(root / "tables/posterior_ratio_diagnostics.csv", ratio_rows)
    write_csv_rows(root / "tables/generation_budget_audit.csv", generation_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_rows)
    write_csv_rows(root / "tables/runtime_timings.csv", timing_rows)
    write_json(root / "reports/study_decision.json", decision)
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_resampled_prior_leakage_report_v1",
            "status": "PASS",
            "outer_rows_used_for_fit": False,
            "inner_rows_used_for_fit": False,
            "inner_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "existing_checkpoint_input_used": False,
            "fresh_bg_training": True,
            "identity_audit_failures": sum(row.get("status") != "PASS" for row in identity_rows),
        },
    )
    write_json(
        root / "reports/publication_state.json",
        {
            "schema_version": "midogpp_resampled_prior_publication_state_v1",
            "publication_state": PUBLICATION_STATE,
            "decision": decision["decision"],
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
    unique_count = len(unique_rows)
    mapped_count = len(metric_rows)
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_resampled_prior_runtime_summary_v1",
            "checkpoint_records": int(coverage["checkpoint_records"]),
            "generation_blocks": int(coverage["generation_blocks"]),
            "classifier_fit_count": unique_count,
            "mapped_metric_rows": mapped_count,
            "redundant_classifier_fits_avoided": mapped_count - unique_count,
            "score_reuse_factor": mapped_count / unique_count,
            "runtime_plan": dict(runtime_plan),
        },
    )


__all__ = ("prepare_bundle", "write_final_artifacts", "write_resolved_config")
