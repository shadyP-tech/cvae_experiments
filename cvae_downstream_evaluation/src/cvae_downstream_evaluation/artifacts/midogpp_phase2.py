"""Artifact helpers for MIDOG++ phase-2 target-support adaptation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..artifacts import stable_hash
from ..schemas.midogpp_phase2 import (
    PHASE2_PREFLIGHT_FORBIDDEN_FILES,
    PHASE2_PREFLIGHT_REPORT_SCHEMA_VERSION,
    PHASE2_SCORE_FUNCTIONAL_ID,
    PHASE2_SCHEMA_VERSION,
    PHASE2_REQUIRED_DIRS,
    PHASE2_ROOT_NAME,
    assert_phase2_artifact_contract,
    assert_phase2_artifact_root,
    assert_phase2_candidate_manifest,
    assert_phase2_feature_provenance,
    assert_phase2_nelbo_comparability,
    assert_phase2_preflight_artifact_contract,
    assert_phase2_preflight_snapshot,
    assert_phase2_preflight_config,
    assert_phase2_routing_decisions,
    assert_phase2_routing_firewall,
    assert_phase2_selected_sources,
    assert_phase2_split_manifests,
    assert_phase2_support_score_matrix,
    build_phase2_candidate_manifest,
    build_phase2_routing_decisions,
    build_phase2_selected_sources,
    build_locked_phase2_support_eval_split,
)


def create_phase2_artifact_root(root: Path) -> dict[str, Path]:
    """Create the approved empty phase-2 artifact directory scaffold."""

    assert_phase2_artifact_root(root)
    paths: dict[str, Path] = {}
    for name in PHASE2_REQUIRED_DIRS:
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    assert_phase2_artifact_contract(root)
    return paths


def default_phase2_artifact_root(artifacts_root: Path) -> Path:
    """Return ``artifacts/midogpp/phase2_target_support_adaptation_virchow2_seed42``."""

    return Path(artifacts_root) / "midogpp" / PHASE2_ROOT_NAME


def write_phase2_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a phase-2 CSV after rejecting empty rows and forbidden matrix names."""

    _assert_phase2_output_path(path)
    if not rows:
        raise ProtocolError(f"Refusing to write empty phase-2 CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(str(column))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_phase2_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a phase-2 JSON report after rejecting forbidden artifact names."""

    _assert_phase2_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_phase2_preflight_freeze(
    *,
    root: Path,
    source_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
    support_score_inputs: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_size: int,
    support_seed: int,
    replicate: str,
    freeze_run_id: str,
    freeze_timestamp: str,
    snapshot_fields: Mapping[str, object],
    center_column: str = "center",
) -> dict[str, object]:
    """Write and validate only the phase-2 routing-freeze artifacts."""

    assert_phase2_preflight_config(snapshot_fields)
    paths = create_phase2_artifact_root(root)
    candidates = build_phase2_candidate_manifest(source_rows, heldout_center=heldout_center)
    candidate_path = paths["manifests"] / "candidate_sources.csv"
    write_phase2_csv(candidate_path, candidates)

    support_rows, eval_rows = build_locked_phase2_support_eval_split(
        target_rows,
        heldout_center=heldout_center,
        support_size=support_size,
        support_seed=support_seed,
        center_column=center_column,
    )
    write_phase2_csv(paths["manifests"] / "support_sets.csv", support_rows)
    write_phase2_csv(paths["manifests"] / "eval_sets.csv", eval_rows)
    support_split_ids = sorted({str(row["split_id"]) for row in support_rows})
    if len(support_split_ids) != 1:
        raise ProtocolError(f"Expected one support split id; got {support_split_ids}")

    support_score_rows = _phase2_support_score_rows(
        candidates=candidates,
        support_score_inputs=support_score_inputs,
        heldout_center=heldout_center,
        support_seed=support_seed,
        replicate=replicate,
        support_split_id=support_split_ids[0],
        support_n=len(support_rows),
    )
    write_phase2_csv(paths["tables"] / "support_score_matrix.csv", support_score_rows)
    decisions = build_phase2_routing_decisions(support_score_rows, freeze_run_id=freeze_run_id)
    write_phase2_csv(paths["tables"] / "routing_decisions.csv", decisions)
    selected = build_phase2_selected_sources(decisions)
    write_phase2_csv(paths["tables"] / "selected_sources.csv", selected)

    snapshot = _phase2_preflight_snapshot(
        root=root,
        candidates=candidates,
        support_rows=support_rows,
        eval_rows=eval_rows,
        snapshot_fields=snapshot_fields,
        class_prior_value_hash=_snapshot_prior_hash(candidates),
        freeze_run_id=freeze_run_id,
        freeze_timestamp=freeze_timestamp,
    )
    write_phase2_json(paths["configs"] / "frozen_protocol_snapshot.json", snapshot)
    report_path = write_phase2_preflight_freeze_report(root)
    return read_phase2_json(report_path)


def read_phase2_csv(path: Path) -> list[dict[str, str]]:
    """Read a phase-2 CSV artifact with path firewall checks."""

    _assert_phase2_routing_input_path(path)
    if not Path(path).exists():
        raise ProtocolError(f"Missing phase-2 CSV: {path}")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_phase2_json(path: Path) -> dict[str, object]:
    """Read a phase-2 JSON artifact with path firewall checks."""

    _assert_phase2_routing_input_path(path)
    if not Path(path).exists():
        raise ProtocolError(f"Missing phase-2 JSON: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed phase-2 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Phase-2 JSON payload must be an object: {path}")
    return payload


def validate_phase2_preflight_freeze(root: Path) -> dict[str, object]:
    """Validate the frozen routing artifacts before downstream utility exists."""

    root = Path(root)
    assert_phase2_preflight_artifact_contract(root)
    paths = {
        "candidate_sources": root / "manifests" / "candidate_sources.csv",
        "support_sets": root / "manifests" / "support_sets.csv",
        "eval_sets": root / "manifests" / "eval_sets.csv",
        "support_score_matrix": root / "tables" / "support_score_matrix.csv",
        "routing_decisions": root / "tables" / "routing_decisions.csv",
        "selected_sources": root / "tables" / "selected_sources.csv",
        "frozen_protocol_snapshot": root / "configs" / "frozen_protocol_snapshot.json",
    }
    candidate_rows = read_phase2_csv(paths["candidate_sources"])
    support_rows = read_phase2_csv(paths["support_sets"])
    eval_rows = read_phase2_csv(paths["eval_sets"])
    support_score_rows = read_phase2_csv(paths["support_score_matrix"])
    routing_decision_rows = read_phase2_csv(paths["routing_decisions"])
    selected_source_rows = read_phase2_csv(paths["selected_sources"])
    snapshot = read_phase2_json(paths["frozen_protocol_snapshot"])

    heldouts = sorted({str(row.get("heldout_center", "")) for row in candidate_rows})
    if len(heldouts) != 1 or not heldouts[0]:
        raise ProtocolError(f"Phase-2 preflight expects one heldout center per freeze root; got {heldouts}")
    assert_phase2_candidate_manifest(candidate_rows, heldout_center=heldouts[0])
    assert_phase2_split_manifests(support_rows=support_rows, eval_rows=eval_rows)
    assert_phase2_support_score_matrix(support_score_rows)
    assert_phase2_routing_decisions(routing_decision_rows, support_score_rows=support_score_rows)
    assert_phase2_selected_sources(selected_source_rows, routing_decision_rows=routing_decision_rows)
    assert_phase2_preflight_snapshot(snapshot, candidate_rows=candidate_rows)
    assert_phase2_feature_provenance(candidate_rows=candidate_rows, snapshot=snapshot)
    assert_phase2_nelbo_comparability(candidate_rows)
    _assert_snapshot_artifact_hashes(snapshot, paths)

    return phase2_preflight_freeze_payload(
        artifacts_root=root,
        status="PASS",
        checks={
            "candidate_rows": len(candidate_rows),
            "support_rows": len(support_rows),
            "eval_rows": len(eval_rows),
            "support_score_rows": len(support_score_rows),
            "routing_decision_rows": len(routing_decision_rows),
            "selected_source_rows": len(selected_source_rows),
            "heldout_center": heldouts[0],
            "downstream_artifacts_absent": True,
            "claim_boundary": "support-NELBO selected expert routing freeze only",
        },
    )


def phase2_preflight_freeze_payload(
    *,
    artifacts_root: Path,
    status: str,
    checks: Mapping[str, object],
    error_message: str = "",
) -> dict[str, object]:
    """Build the preflight freeze report payload."""

    return {
        "schema_version": PHASE2_PREFLIGHT_REPORT_SCHEMA_VERSION,
        "artifacts_root": str(artifacts_root),
        "status": status,
        "checks": dict(checks),
        "error_message": error_message,
    }


def write_phase2_preflight_freeze_report(root: Path) -> Path:
    """Validate and write reports/phase2_preflight_freeze_report.json."""

    try:
        payload = validate_phase2_preflight_freeze(root)
    except ProtocolError as exc:
        payload = phase2_preflight_freeze_payload(
            artifacts_root=root,
            status="FAIL",
            checks={},
            error_message=str(exc),
        )
        report_path = Path(root) / "reports" / "phase2_preflight_freeze_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    report_path = Path(root) / "reports" / "phase2_preflight_freeze_report.json"
    write_phase2_json(report_path, payload)
    return report_path


def write_locked_phase2_support_eval_split(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> tuple[Path, Path]:
    """Materialize locked support/eval manifests before any phase-2 scorer runs."""

    assert_phase2_artifact_root(root)
    support_rows, eval_rows = build_locked_phase2_support_eval_split(
        rows,
        heldout_center=heldout_center,
        support_size=support_size,
        support_seed=support_seed,
    )
    support_path = Path(root) / "manifests" / "support_sets.csv"
    eval_path = Path(root) / "manifests" / "eval_sets.csv"
    write_phase2_csv(support_path, support_rows)
    write_phase2_csv(eval_path, eval_rows)
    return support_path, eval_path


def phase2_validation_payload(
    *,
    artifacts_root: Path,
    status: str,
    checks: Mapping[str, object],
    error_message: str = "",
) -> dict[str, object]:
    """Build the standard phase-2 validation report payload."""

    return {
        "schema_version": "midogpp_phase2_validation_report_v1",
        "artifacts_root": str(artifacts_root),
        "status": status,
        "checks": dict(checks),
        "error_message": error_message,
    }


def _assert_phase2_output_path(path: Path) -> None:
    if Path(path).name in PHASE2_PREFLIGHT_FORBIDDEN_FILES:
        raise ProtocolError(f"{Path(path).name} is forbidden before MIDOG++ phase-2 preflight freeze.")


def _assert_phase2_routing_input_path(path: Path) -> None:
    assert_phase2_routing_firewall(input_paths=[path])


def _assert_snapshot_artifact_hashes(snapshot: Mapping[str, object], paths: Mapping[str, Path]) -> None:
    expected = {
        "candidate_sources_hash": stable_hash(_file_payload(paths["candidate_sources"])),
        "support_sets_hash": stable_hash(_file_payload(paths["support_sets"])),
        "eval_sets_hash": stable_hash(_file_payload(paths["eval_sets"])),
        "support_score_matrix_hash": stable_hash(_file_payload(paths["support_score_matrix"])),
        "routing_decisions_hash": stable_hash(_file_payload(paths["routing_decisions"])),
        "selected_sources_hash": stable_hash(_file_payload(paths["selected_sources"])),
    }
    mismatches = {
        key: {"observed": snapshot.get(key), "expected": value}
        for key, value in expected.items()
        if str(snapshot.get(key, "")) != value
    }
    if mismatches:
        raise ProtocolError(f"Frozen protocol snapshot artifact hash mismatch: {mismatches}")


def _file_payload(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _phase2_support_score_rows(
    *,
    candidates: Sequence[Mapping[str, object]],
    support_score_inputs: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_seed: int,
    replicate: str,
    support_split_id: str,
    support_n: int,
) -> list[dict[str, object]]:
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    rows: list[dict[str, object]] = []
    for raw in support_score_inputs:
        candidate_id = str(raw.get("candidate_id", ""))
        if candidate_id not in by_candidate:
            raise ProtocolError(f"Support score references unknown candidate_id: {candidate_id!r}")
        candidate = by_candidate[candidate_id]
        rows.append(
            {
                "schema_version": PHASE2_SCHEMA_VERSION,
                "heldout_center": heldout_center,
                "support_seed": str(support_seed),
                "replicate": str(replicate),
                "support_split_id": support_split_id,
                "candidate_id": candidate_id,
                "candidate_source_center": str(candidate["candidate_source_center"]),
                "stable_candidate_id": str(candidate["stable_candidate_id"]),
                "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
                "score_direction": "lower_is_better",
                "support_aggregation": "mean_over_support_samples",
                "support_n": int(raw.get("support_n", support_n)),
                "support_score": float(raw["support_score"]),
                "support_score_variance_or_se": float(raw.get("support_score_variance_or_se", 0.0)),
                "class_order": str(candidate["class_order"]),
                "class_prior_value_hash": str(candidate["class_prior_value_hash"]),
                "checkpoint_hash": str(candidate["checkpoint_hash"]),
                "config_hash": str(candidate["config_hash"]),
                "scorer_implementation_hash": str(candidate["scorer_implementation_hash"]),
                "encoder_mode": str(raw.get("encoder_mode", "deterministic")),
                "tie_or_near_tie": bool(raw.get("tie_or_near_tie", False)),
            }
        )
    assert_phase2_support_score_matrix(rows)
    return rows


def _phase2_preflight_snapshot(
    *,
    root: Path,
    candidates: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
    snapshot_fields: Mapping[str, object],
    class_prior_value_hash: str,
    freeze_run_id: str,
    freeze_timestamp: str,
) -> dict[str, object]:
    payload = {
        "candidate_pool_hash": stable_hash(candidates),
        "support_split_hash": stable_hash(support_rows),
        "eval_split_hash": stable_hash(eval_rows),
        "checkpoint_cache_hash": snapshot_fields.get("checkpoint_cache_hash", stable_hash(_candidate_field(candidates, "checkpoint_hash"))),
        "generation_config_hash": snapshot_fields.get("generation_config_hash", stable_hash(_candidate_field(candidates, "generation_mode"))),
        "classifier_config_hash": snapshot_fields.get("classifier_config_hash", stable_hash(_candidate_field(candidates, "classifier_seed"))),
        "metric_config_hash": snapshot_fields.get("metric_config_hash", "predeclared_metric_config"),
        "feature_whitelist_hash": snapshot_fields.get("feature_whitelist_hash", stable_hash(_candidate_field(candidates, "feature_frame_hash"))),
        "routing_rule": "argmin_support_score",
        "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
        "class_prior_value_hash": snapshot_fields.get("class_prior_value_hash", class_prior_value_hash),
        "score_direction": "lower_is_better",
        "support_aggregation": "mean_over_support_samples",
        "tie_breaker": "stable_candidate_id",
        "freeze_run_id": freeze_run_id,
        "freeze_timestamp": freeze_timestamp,
        "protocol_hash": snapshot_fields.get("protocol_hash", "midogpp_phase2_preflight_v1"),
    } | {
        "candidate_sources_hash": stable_hash(_file_payload(Path(root) / "manifests" / "candidate_sources.csv")),
        "support_sets_hash": stable_hash(_file_payload(Path(root) / "manifests" / "support_sets.csv")),
        "eval_sets_hash": stable_hash(_file_payload(Path(root) / "manifests" / "eval_sets.csv")),
        "support_score_matrix_hash": stable_hash(_file_payload(Path(root) / "tables" / "support_score_matrix.csv")),
        "routing_decisions_hash": stable_hash(_file_payload(Path(root) / "tables" / "routing_decisions.csv")),
        "selected_sources_hash": stable_hash(_file_payload(Path(root) / "tables" / "selected_sources.csv")),
    }
    return payload


def _snapshot_prior_hash(candidates: Sequence[Mapping[str, object]]) -> str:
    prior_hashes = sorted({str(row.get("class_prior_value_hash", "")) for row in candidates})
    if not prior_hashes or "" in prior_hashes:
        raise ProtocolError("Candidate rows must have class_prior_value_hash before snapshot freeze.")
    return prior_hashes[0] if len(prior_hashes) == 1 else "per_candidate"


def _candidate_field(candidates: Sequence[Mapping[str, object]], field: str) -> list[str]:
    return [str(row.get(field, "")) for row in candidates]
