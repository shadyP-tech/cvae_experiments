"""Independent fail-closed validation of completed study bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import ARMS, COMPOSITION_MODES, PUBLICATION_STATE
from .protocol import validate_candidate_pool


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/coverage_manifest.json",
    "manifests/frame_index.json",
    "manifests/task_geometry_state_index.json",
    "manifests/checkpoint_index.json",
    "manifests/candidate_pool_manifest.json",
    "manifests/generation_budget_manifest.json",
    "manifests/composition_manifest.json",
    "reports/run_state.json",
    "reports/leakage_report.json",
    "reports/study_decision.json",
    "reports/publication_state.json",
    "reports/runtime_summary.json",
    "tables/source_inner_metrics.csv",
    "tables/paired_deltas.csv",
    "tables/task_geometry_diagnostics.csv",
    "tables/generation_budget_audit.csv",
    "tables/composition_metrics.csv",
    "tables/tstr_metrics.csv",
    "tables/identity_overlap_audit.csv",
    "tables/rng_pairing_audit.csv",
    "tables/runtime_timings.csv",
)


def validate_uniform_b_task_geometry_bundle(root: Path) -> Mapping[str, object]:
    root = Path(root)
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise ProtocolError(f"Uniform-B study bundle is incomplete: {missing}")
    protocol = _json(root / "manifests/protocol_manifest.json")
    coverage = _json(root / "manifests/coverage_manifest.json")
    publication = _json(root / "reports/publication_state.json")
    leakage = _json(root / "reports/leakage_report.json")
    run_state = _json(root / "reports/run_state.json")
    pools = _json(root / "manifests/candidate_pool_manifest.json")
    metrics = _csv(root / "tables/source_inner_metrics.csv")
    checkpoints = _json(root / "manifests/checkpoint_index.json")
    frames = _json(root / "manifests/frame_index.json")
    geometries = _json(root / "manifests/task_geometry_state_index.json")
    generation = _json(root / "manifests/generation_budget_manifest.json")
    compositions = _json(root / "manifests/composition_manifest.json")
    if (
        protocol.get("expected_feature_dim") != 3840
        or protocol.get("arms") != list(ARMS)
        or protocol.get("composition_modes") != list(COMPOSITION_MODES)
        or protocol.get("inner_labels_used_for_scoring_only") is not True
        or protocol.get("outer_rows_used_for_fit") is not False
        or protocol.get("inner_rows_used_for_fit") is not False
    ):
        raise ProtocolError("Uniform-B protocol manifest violates its firewall.")
    if (
        publication.get("publication_state") != PUBLICATION_STATE
        or publication.get("decision") != "DO_NOT_PROMOTE"
        or any(
            publication.get(key) is not False
            for key in (
                "may_feed_model_recipe",
                "may_feed_recipe_selection",
                "may_feed_expert_bank",
                "may_feed_generation",
                "may_feed_routing",
                "may_feed_downstream_utility",
                "stage30_recipe_ready",
            )
        )
        or publication.get("separate_promotion_artifact_required") is not True
    ):
        raise ProtocolError("Uniform-B publication boundary is consumable.")
    if (
        leakage.get("status") != "PASS"
        or int(leakage.get("identity_audit_failures", -1)) != 0
        or run_state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("Uniform-B leakage/run state did not pass.")
    centers = tuple(str(value) for value in coverage.get("centers", ()))
    records = pools.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Candidate-pool index is malformed.")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Candidate-pool record is malformed.")
        validate_candidate_pool(record, centers)
    expected_metrics = int(coverage.get("metric_rows", -1))
    if expected_metrics != len(metrics) or not metrics:
        raise ProtocolError("Metric coverage does not match the table.")
    if int(checkpoints.get("n_records", -1)) != int(
        coverage.get("checkpoint_records", -2)
    ):
        raise ProtocolError("Checkpoint coverage does not match the index.")
    _validate_hashed_states(frames, label="frame")
    _validate_hashed_states(geometries, label="task geometry")
    _validate_checkpoints(root, checkpoints)
    generation_records = generation.get("records")
    if not isinstance(generation_records, list) or any(
        not isinstance(record, Mapping)
        or record.get("outer_or_inner_identity_present") is not False
        or record.get("class_balanced") is not True
        or record.get("target_or_source_prevalence_used") is not False
        for record in generation_records
    ):
        raise ProtocolError("Generation manifest violates H/I-neutral balance.")
    composition_records = compositions.get("records")
    if not isinstance(composition_records, list) or any(
        not isinstance(record, Mapping)
        or record.get("sealed_before_inner_rows_loaded") is not True
        or record.get("routing_or_selection") is not False
        for record in composition_records
    ):
        raise ProtocolError("Composition manifest violates its sealed boundary.")
    observed_arms = {row.get("arm") for row in metrics}
    observed_modes = {row.get("composition_mode") for row in metrics}
    observed_kinds = {row.get("generation_kind") for row in metrics}
    if (
        observed_arms != set(ARMS)
        or observed_modes != set(COMPOSITION_MODES)
        or observed_kinds != {"prior", "posterior"}
        or any(
            row.get("inner_labels_used_for_scoring_only") != "True"
            or row.get("outer_rows_used") != "False"
            or row.get("routing_or_compatibility") != "False"
            for row in metrics
        )
    ):
        raise ProtocolError("Metric table lacks required arms or budget modes.")
    report = {
        "schema_version": "midogpp_uniform_b_task_geometry_validation_v1",
        "status": "PASS",
        "required_files": len(REQUIRED_FILES),
        "metric_rows": len(metrics),
        "candidate_pools": len(records),
        "publication_state": PUBLICATION_STATE,
        "claim_eligible": False,
    }
    from ...reporting import write_json

    write_json(root / "reports/validation_report.json", report)
    return report


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_hashed_states(
    index: Mapping[str, object],
    *,
    label: str,
) -> None:
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ProtocolError(f"{label.title()} index is empty or malformed.")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError(f"{label.title()} index record is malformed.")
        state = record.get("state")
        if (
            not isinstance(state, Mapping)
            or stable_hash(state) != record.get("state_hash")
            or state.get("outer_or_inner_rows_used") is not False
        ):
            raise ProtocolError(f"{label.title()} state hash/firewall mismatch.")


def _validate_checkpoints(
    root: Path,
    index: Mapping[str, object],
) -> None:
    records = index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Checkpoint index records are malformed.")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Checkpoint record is malformed.")
        path = root / str(record.get("relative_path", ""))
        if (
            record.get("outer_or_inner_identity_present") is not False
            or record.get("source_only_training") is not True
            or not path.is_file()
            or _file_sha256(path) != record.get("file_sha256")
        ):
            raise ProtocolError("Checkpoint file/firewall validation failed.")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ("REQUIRED_FILES", "validate_uniform_b_task_geometry_bundle")
