"""Fail-closed validator for fresh BG, P0/Pq, and unique-score reuse."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import P0, PQ, PRIORS, PUBLICATION_STATE, valid_outer_centers


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/coverage_manifest.json",
    "manifests/frame_index.json",
    "manifests/checkpoint_index.json",
    "manifests/posterior_ratio_state_index.json",
    "manifests/score_reuse_mapping.json",
    "manifests/generation_budget_manifest.json",
    "reports/run_state.json",
    "reports/leakage_report.json",
    "reports/study_decision.json",
    "reports/publication_state.json",
    "reports/runtime_summary.json",
    "tables/unique_tstr_scores.csv",
    "tables/source_inner_metrics.csv",
    "tables/paired_deltas.csv",
    "tables/posterior_ratio_diagnostics.csv",
    "tables/generation_budget_audit.csv",
    "tables/identity_overlap_audit.csv",
    "tables/runtime_timings.csv",
)


def validate_uniform_b_resampled_prior_bundle(root: Path) -> dict[str, object]:
    root = Path(root)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ProtocolError(f"Resampled-prior bundle is incomplete: {missing}")
    protocol = _json(root / "manifests/protocol_manifest.json")
    coverage = _json(root / "manifests/coverage_manifest.json")
    provenance = _json(root / "provenance/input_artifacts.json")
    checkpoints = _json(root / "manifests/checkpoint_index.json")
    ratios = _json(root / "manifests/posterior_ratio_state_index.json")
    mapping = _json(root / "manifests/score_reuse_mapping.json")
    generation = _json(root / "manifests/generation_budget_manifest.json")
    leakage = _json(root / "reports/leakage_report.json")
    publication = _json(root / "reports/publication_state.json")
    unique_rows = _csv(root / "tables/unique_tstr_scores.csv")
    metric_rows = _csv(root / "tables/source_inner_metrics.csv")
    centers = tuple(str(value) for value in coverage.get("centers", ()))
    training_seeds = tuple(int(value) for value in coverage.get("training_seeds", ()))
    generation_seeds = tuple(int(value) for value in coverage.get("generation_seeds", ()))
    if (
        len(centers) < 3
        or set(coverage.get("priors", ())) != set(PRIORS)
        or protocol.get("fresh_bg_training_required") is not True
        or protocol.get("existing_checkpoint_input_allowed") is not False
        or protocol.get("unique_score_reuse_required") is not True
        or protocol.get("score_key_excludes_outer_center") is not True
        or leakage.get("status") != "PASS"
        or leakage.get("outer_rows_used_for_fit") is not False
        or leakage.get("inner_rows_used_for_fit") is not False
        or leakage.get("existing_checkpoint_input_used") is not False
        or provenance.get("completed_uniform_b_task_geometry_artifact_used") is not False
        or provenance.get("existing_checkpoint_input_used") is not False
        or publication.get("publication_state") != PUBLICATION_STATE
        or publication.get("may_feed_model_recipe") is not False
        or publication.get("may_feed_expert_bank") is not False
        or publication.get("separate_promotion_artifact_required") is not True
    ):
        raise ProtocolError("Resampled-prior protocol/publication firewall failed.")
    forbidden_input = "midogpp_output_cvae_uniform_b_geco_task_geometry_source_inner_v1"
    if forbidden_input in set(provenance.get("input_artifact_ids", ())):
        raise ProtocolError("Completed task-geometry output is a forbidden input.")
    expected_checkpoints = len(centers) * len(training_seeds)
    _validate_checkpoints(root, checkpoints, expected=expected_checkpoints)
    _validate_ratio_states(ratios, expected=expected_checkpoints)
    expected_unique = (
        len(centers) * (len(centers) - 1)
        * len(training_seeds) * len(generation_seeds) * len(PRIORS)
    )
    expected_mapped = expected_unique * (len(centers) - 2)
    if len(unique_rows) != expected_unique or len(metric_rows) != expected_mapped:
        raise ProtocolError("Unique/mapped score coverage is incorrect.")
    unique_hashes = [row["score_key_hash"] for row in unique_rows]
    if len(unique_hashes) != len(set(unique_hashes)):
        raise ProtocolError("A unique P0/Pq score key was evaluated more than once.")
    unique_by_hash = {row["score_key_hash"]: row for row in unique_rows}
    mapping_records = mapping.get("records")
    if not isinstance(mapping_records, list) or len(mapping_records) != expected_unique:
        raise ProtocolError("Score reuse mapping coverage is malformed.")
    mapping_by_hash = {}
    for record in mapping_records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Score reuse mapping record is malformed.")
        key = str(record.get("score_key_hash", ""))
        if key in mapping_by_hash or key not in unique_by_hash:
            raise ProtocolError("Score reuse mapping key is duplicate or unknown.")
        expected_outers = valid_outer_centers(
            centers,
            source_center=str(record["source_center"]),
            inner_center=str(record["inner_center"]),
        )
        if tuple(record.get("mapped_outer_centers", ())) != expected_outers or int(record.get("mapping_count", -1)) != len(expected_outers):
            raise ProtocolError("Unique score did not map to its exact legal outers.")
        mapping_by_hash[key] = expected_outers
    mapped_by_hash: dict[str, list[dict[str, str]]] = {}
    for row in metric_rows:
        mapped_by_hash.setdefault(row["score_key_hash"], []).append(row)
        if (
            row.get("prior") not in PRIORS
            or row.get("mapped_from_unique_score") != "True"
            or row.get("inner_labels_used_for_scoring_only") != "True"
            or row.get("outer_rows_used") != "False"
            or row.get("routing_or_compatibility") != "False"
        ):
            raise ProtocolError("Mapped metric row violates the score firewall.")
    for key, outers in mapping_by_hash.items():
        rows = mapped_by_hash.get(key, [])
        if tuple(sorted((row["outer_center"] for row in rows), key=_center_key)) != tuple(sorted(outers, key=_center_key)):
            raise ProtocolError("Mapped metric rows do not match legal outer centers.")
        unique = unique_by_hash[key]
        if any(
            row["bacc"] != unique["bacc"]
            or row["macro_f1"] != unique["macro_f1"]
            for row in rows
        ):
            raise ProtocolError("Outer mapping changed a unique score value.")
    _validate_generation(generation, centers, training_seeds, generation_seeds)
    report = {
        "schema_version": "midogpp_resampled_prior_validation_v1",
        "status": "PASS",
        "required_files": len(REQUIRED_FILES),
        "checkpoint_records": expected_checkpoints,
        "unique_score_rows": expected_unique,
        "mapped_metric_rows": expected_mapped,
        "score_reuse_factor": len(centers) - 2,
        "publication_state": PUBLICATION_STATE,
        "claim_eligible": False,
    }
    from ...reporting import write_json

    write_json(root / "reports/validation_report.json", report)
    return report


def _validate_checkpoints(root: Path, index: Mapping[str, object], *, expected: int) -> None:
    records = index.get("records")
    if not isinstance(records, list) or len(records) != expected:
        raise ProtocolError("Fresh BG checkpoint coverage is incorrect.")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Checkpoint record is malformed.")
        path = root / str(record.get("relative_path", ""))
        if (
            record.get("training_arm") != "BG"
            or record.get("fresh_training") is not True
            or record.get("parent_checkpoint_used") is not False
            or record.get("parent_checkpoint_hash") != "none"
            or record.get("source_only_training") is not True
            or record.get("outer_or_inner_identity_present") is not False
            or not path.is_file()
            or _file_sha256(path) != record.get("file_sha256")
        ):
            raise ProtocolError("Checkpoint provenance is not fresh/source-only.")


def _validate_ratio_states(index: Mapping[str, object], *, expected: int) -> None:
    records = index.get("records")
    if not isinstance(records, list) or len(records) != expected:
        raise ProtocolError("Posterior-ratio state coverage is incorrect.")
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("state"), Mapping):
            raise ProtocolError("Posterior-ratio state is malformed.")
        state = record["state"]
        if stable_hash(state) != record.get("state_hash") or state.get("outer_or_inner_rows_used") is not False or state.get("target_labels_used") is not False:
            raise ProtocolError("Posterior-ratio state hash/firewall failed.")


def _validate_generation(
    index: Mapping[str, object],
    centers: tuple[str, ...],
    training_seeds: tuple[int, ...],
    generation_seeds: tuple[int, ...],
) -> None:
    records = index.get("records")
    expected = len(centers) * len(training_seeds) * len(generation_seeds) * len(PRIORS)
    if not isinstance(records, list) or len(records) != expected:
        raise ProtocolError("P0/Pq generation block coverage is incorrect.")
    observed = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"]), str(row["prior"]))
        for row in records
        if isinstance(row, Mapping)
    }
    wanted = {
        (center, training, generation, prior)
        for center in centers
        for training in training_seeds
        for generation in generation_seeds
        for prior in PRIORS
    }
    if observed != wanted or any(
        not isinstance(row, Mapping)
        or row.get("outer_or_inner_identity_present") is not False
        or row.get("class_balanced") is not True
        for row in records
    ):
        raise ProtocolError("P0/Pq generation manifest violates coverage/firewall.")


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _center_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


__all__ = ("REQUIRED_FILES", "validate_uniform_b_resampled_prior_bundle")
