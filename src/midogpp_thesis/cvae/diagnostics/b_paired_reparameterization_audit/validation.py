"""Fail-closed validation for the Stage-90 paired reparameterization audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError

from .artifacts import file_sha256, validate_content_index
from .comparison import (
    FIXED_ANTITHETIC,
    FIXED_ONE_EPSILON,
    audit_decision,
    prediction_digest,
)
from .config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CLAIM_FIREWALL_FIELDS,
    CLAIM_SCOPE,
    CONTROLLED_CANDIDATES,
    EVIDENCE_LABEL,
    INITIALIZATION_SEEDS,
    LEGACY_CANDIDATE,
    SNAPSHOT_ARTIFACT_ID,
    STAGE,
    ClaimFirewall,
    DecisionThresholds,
    FrozenBRecipe,
)
from .protocol import (
    AuditKeyRecord,
    comparison_pairs,
    key_inventory_hash,
    key_record_from_mapping,
    validate_key_inventory,
)
from .snapshot import HASH_PROMOTED
from .snapshot_io import canonical_mapping_hash


VALIDATION_SCHEMA = "midogpp_b_paired_reparameterization_validation_report_v1"
PROTOCOL_SCHEMA = "midogpp_b_paired_reparameterization_protocol_manifest_v1"
SNAPSHOT_BINDING_SCHEMA = "midogpp_b_paired_reparameterization_snapshot_binding_v1"
RUNTIME_SUMMARY_SCHEMA = "midogpp_b_paired_reparameterization_runtime_summary_v1"

AUDIT_REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/snapshot_binding.json",
    "manifests/key_inventory.json",
    "manifests/content_index.json",
    "reports/run_state.json",
    "reports/leakage_provenance_report.json",
    "reports/validation_report.json",
    "reports/audit_decision.json",
    "reports/runtime_summary.json",
    "tables/job_inventory.csv",
    "tables/replay_trace_audit.csv",
    "tables/legacy_v2_validation.csv",
    "tables/controlled_metrics.csv",
    "tables/paired_comparison.csv",
    "tables/consumption_audit.csv",
    "tables/decoded_predictions.csv",
)

EXPECTED_JOB_COUNT = 36
EXPECTED_LEGACY_JOB_COUNT = 12
EXPECTED_CONTROLLED_JOB_COUNT = 24
EXPECTED_PAIR_COUNT = 12
EXPECTED_OPTIMIZER_UPDATES = 36_000
EXPECTED_LEGACY_DECODER_FORWARDS = 12_000
EXPECTED_FIXED_ONE_DECODER_FORWARDS = 12_000
EXPECTED_ANTITHETIC_DECODER_FORWARDS = 24_000
EXPECTED_DECODER_FORWARDS = 48_000
EXPECTED_EPSILON_CONSUMPTIONS = 36

_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_HASH = re.compile(r"^[0-9a-f]{16}$")

_JOB_COLUMNS = frozenset(
    {
        "center",
        "initialization_seed",
        "candidate",
        "execution_device",
        "key_hash",
        "pair_id",
        "prepared_sha256",
        "prepared_content_hash",
        "schedule_sha256",
        "schedule_content_hash",
        "epsilon_trace_sha256",
        "epsilon_trace_content_hash",
        "initialization_hash",
        "checkpoint_hash",
        "schedule_hash",
        "posterior_stream_hash",
        "optimizer_steps",
        "decoder_forwards",
        "posterior_estimator",
        "epsilon_consumptions",
        "cache_status",
        "status",
        "claim_scope",
    }
)

_REPLAY_COLUMNS = frozenset(
    {
        "center",
        "initialization_seed",
        "candidate",
        "key_hash",
        "prepared_file_match",
        "prepared_content_match",
        "schedule_file_match",
        "schedule_content_match",
        "epsilon_file_match",
        "epsilon_content_match",
        "trace_consumption_count",
        "status",
    }
)

_LEGACY_FIELDS = (
    "initialization",
    "checkpoint",
    "prediction",
    "metric",
    "schedule",
    "posterior",
)
_LEGACY_COLUMNS = frozenset(
    {
        "center",
        "initialization_seed",
        "candidate",
        "key_hash",
        "status",
        *(
            column
            for field in _LEGACY_FIELDS
            for column in (
                f"expected_{field}_hash",
                f"observed_{field}_hash",
                f"{field}_match",
            )
        ),
    }
)

_CONSUMPTION_COLUMNS = frozenset(
    {
        "center",
        "initialization_seed",
        "candidate",
        "key_hash",
        "epsilon_consumption_count",
        "optimizer_steps",
        "decoder_forwards",
        "status",
    }
)

_LEAKAGE_FLAGS = {
    "historical_paths_read": False,
    "eval_labels_used_for_cvae_fit": False,
    "eval_labels_used_for_classifier_fit": False,
    "eval_labels_used_for_selection": False,
    "eval_labels_used_for_decode_condition": True,
    "eval_labels_used_for_final_diagnostic_scoring": True,
}


def validate_audit_bundle(root: Path) -> dict[str, object]:
    """Independently validate one complete audit bundle.

    The validator trusts neither success booleans nor the runner's aggregate
    counters in isolation. It reconciles the immutable key inventory, every
    per-job table, metric arithmetic, controlled-pair deltas, the decision,
    and the byte-level content index.
    """

    path = Path(root).resolve()
    errors: list[str] = []
    counts: dict[str, int] = {}

    missing = [
        relative for relative in AUDIT_REQUIRED_FILES if not (path / relative).is_file()
    ]
    errors.extend(f"missing required member: {relative}" for relative in missing)
    if missing:
        return _report(errors, counts)

    try:
        inventory_payload = _json(path / "manifests/key_inventory.json")
        records = _parse_key_inventory(inventory_payload)
        inventory_hash = key_inventory_hash(records)
        counts["key_count"] = len(records)
        counts["controlled_pair_count"] = len(comparison_pairs(records))
    except Exception as exc:
        errors.append(f"key inventory validation failed: {exc}")
        return _report(errors, counts)

    try:
        protocol = _json(path / "manifests/protocol_manifest.json")
        _validate_protocol_manifest(
            path,
            protocol,
            inventory_hash=inventory_hash,
            records=records,
        )
    except Exception as exc:
        errors.append(f"protocol manifest validation failed: {exc}")

    try:
        binding = _json(path / "manifests/snapshot_binding.json")
        _validate_snapshot_binding(
            binding,
            protocol=protocol if "protocol" in locals() else {},
            inventory_hash=inventory_hash,
            records=records,
        )
        eval_row_inventory_hashes = _eval_row_inventory_hashes(binding)
    except Exception as exc:
        errors.append(f"snapshot binding validation failed: {exc}")
        eval_row_inventory_hashes = {}

    try:
        _validate_dynamic_member_coverage(path, records)
    except Exception as exc:
        errors.append(f"per-key member coverage failed: {exc}")

    records_by_coordinate = {
        _record_coordinate(record): record for record in records
    }
    try:
        jobs = _csv(path / "tables/job_inventory.csv")
        jobs_by_coordinate, job_counts = _validate_jobs(
            jobs, records_by_coordinate=records_by_coordinate
        )
        counts.update(job_counts)
    except Exception as exc:
        errors.append(f"job inventory validation failed: {exc}")
        jobs = []
        jobs_by_coordinate = {}

    try:
        replay = _csv(path / "tables/replay_trace_audit.csv")
        _validate_replay_trace_audit(
            replay,
            records_by_coordinate=records_by_coordinate,
        )
        counts["replay_trace_rows"] = len(replay)
    except Exception as exc:
        errors.append(f"replay trace audit validation failed: {exc}")

    try:
        legacy = _csv(path / "tables/legacy_v2_validation.csv")
        _validate_legacy_replay(
            legacy,
            records_by_coordinate=records_by_coordinate,
            jobs_by_coordinate=jobs_by_coordinate,
        )
        counts["legacy_validation_rows"] = len(legacy)
    except Exception as exc:
        errors.append(f"legacy-v2 replay validation failed: {exc}")
        legacy = []

    try:
        metrics = _csv(path / "tables/controlled_metrics.csv")
        typed_metrics = _validate_controlled_metrics(
            metrics,
            records_by_coordinate=records_by_coordinate,
        )
        counts["controlled_metric_rows"] = len(metrics)
    except Exception as exc:
        errors.append(f"controlled metric validation failed: {exc}")
        typed_metrics = []

    try:
        decoded_predictions = _csv(path / "tables/decoded_predictions.csv")
        _validate_decoded_predictions(
            decoded_predictions,
            records_by_coordinate=records_by_coordinate,
            typed_metrics=typed_metrics,
            legacy_rows=legacy,
            eval_row_inventory_hashes=eval_row_inventory_hashes,
        )
        counts["decoded_prediction_rows"] = len(decoded_predictions)
    except Exception as exc:
        errors.append(f"decoded prediction validation failed: {exc}")

    try:
        paired = _csv(path / "tables/paired_comparison.csv")
        _validate_paired_comparisons(
            paired,
            typed_metrics=typed_metrics,
            records_by_coordinate=records_by_coordinate,
        )
        counts["paired_comparison_rows"] = len(paired)
    except Exception as exc:
        errors.append(f"paired comparison validation failed: {exc}")

    try:
        consumption = _csv(path / "tables/consumption_audit.csv")
        _validate_consumption_audit(
            consumption,
            jobs_by_coordinate=jobs_by_coordinate,
            records_by_coordinate=records_by_coordinate,
        )
        counts["consumption_rows"] = len(consumption)
    except Exception as exc:
        errors.append(f"epsilon consumption audit validation failed: {exc}")

    try:
        decision = _json(path / "reports/audit_decision.json")
        _validate_decision(decision, typed_metrics=typed_metrics)
    except Exception as exc:
        errors.append(f"audit decision validation failed: {exc}")

    try:
        leakage = _json(path / "reports/leakage_provenance_report.json")
        _validate_leakage_report(leakage)
    except Exception as exc:
        errors.append(f"leakage/provenance validation failed: {exc}")

    try:
        runtime = _json(path / "reports/runtime_summary.json")
        _validate_runtime_summary(runtime)
    except Exception as exc:
        errors.append(f"runtime summary validation failed: {exc}")

    try:
        run_state = _json(path / "reports/run_state.json")
        if run_state.get("status") not in {"VALIDATING", "COMPLETE"}:
            raise ProtocolError("run_state is not on a successful validation path.")
        _require_audit_identity(run_state, require_all=False)
        _assert_all_may_flags_false(run_state, "run_state")
    except Exception as exc:
        errors.append(f"run-state validation failed: {exc}")

    try:
        index = _json(path / "manifests/content_index.json")
        if (
            index.get("schema_version")
            != "midogpp_b_paired_reparameterization_content_index_v1"
            or index.get("claim_scope") != CLAIM_SCOPE
            or index.get("may_feed_deployable_selection") is not False
        ):
            raise ProtocolError("content-index identity or claim firewall drifted.")
        index_errors = validate_content_index(path)
        if index_errors:
            raise ProtocolError("; ".join(index_errors))
    except Exception as exc:
        errors.append(f"content-index validation failed: {exc}")

    return _report(errors, counts)


def assert_valid_audit_bundle(root: Path) -> dict[str, object]:
    """Return a PASS report or raise :class:`ProtocolError` with all failures."""

    report = validate_audit_bundle(root)
    if report["status"] != "PASS":
        errors = report.get("errors", ())
        joined = "; ".join(str(value) for value in errors)
        raise ProtocolError(f"Stage-90 paired audit bundle is invalid: {joined}")
    return report


def _parse_key_inventory(payload: Mapping[str, object]) -> tuple[AuditKeyRecord, ...]:
    records_value: object = payload.get("records")
    if records_value is None and isinstance(payload.get("key_inventory"), Mapping):
        records_value = payload["key_inventory"].get("records")  # type: ignore[index]
    if not isinstance(records_value, list):
        raise ProtocolError("key inventory records must be a list.")
    records = tuple(
        key_record_from_mapping(_as_mapping(value, "key record"))
        for value in records_value
    )
    validate_key_inventory(records, require_publication_hashes=True)
    declared_hash = payload.get("key_inventory_hash")
    if declared_hash is not None and declared_hash != key_inventory_hash(records):
        raise ProtocolError("declared key-inventory hash does not recompute.")
    return records


def _validate_protocol_manifest(
    root: Path,
    payload: Mapping[str, object],
    *,
    inventory_hash: str,
    records: Sequence[AuditKeyRecord],
) -> None:
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise ProtocolError("protocol manifest schema drifted.")
    _require_audit_identity(payload)
    if payload.get("legacy_used_for_decision") is not False:
        raise ProtocolError("legacy replay rows must be excluded from the decision.")
    if payload.get("recipe") != FrozenBRecipe().to_payload():
        raise ProtocolError("protocol manifest does not freeze the exact Variant-B recipe.")
    _validate_firewall(payload.get("claim_firewall"), "protocol claim firewall")
    if payload.get("key_inventory_hash") != inventory_hash:
        raise ProtocolError("protocol manifest key-inventory binding mismatches.")
    manifest_hashes = {record.snapshot_manifest_hash for record in records}
    if len(manifest_hashes) != 1:
        raise ProtocolError("key inventory binds more than one snapshot manifest.")
    if payload.get("snapshot_manifest_hash") != next(iter(manifest_hashes)):
        raise ProtocolError("protocol snapshot-manifest binding mismatches key records.")
    workspace_hashes = _as_mapping(
        payload.get("workspace_snapshot_hashes"), "workspace_snapshot_hashes"
    )
    if (
        workspace_hashes.get("config_resolved_sha256")
        != file_sha256(root / "config.resolved.yaml")
        or workspace_hashes.get("input_artifacts_sha256")
        != file_sha256(root / "provenance/input_artifacts.json")
    ):
        raise ProtocolError("protocol workspace snapshot hashes do not match bundle bytes.")
    observed_protocol_hash = payload.get("protocol_hash")
    if not isinstance(observed_protocol_hash, str) or not _SEMANTIC_HASH.fullmatch(
        observed_protocol_hash
    ):
        raise ProtocolError("protocol_hash must be a canonical semantic hash.")
    unhashed = dict(payload)
    unhashed.pop("protocol_hash", None)
    if stable_hash(unhashed) != observed_protocol_hash:
        raise ProtocolError("protocol manifest self-hash does not recompute.")
    _assert_all_may_flags_false(payload, "protocol manifest")


def _validate_snapshot_binding(
    payload: Mapping[str, object],
    *,
    protocol: Mapping[str, object],
    inventory_hash: str,
    records: Sequence[AuditKeyRecord],
) -> None:
    if payload.get("schema_version") != SNAPSHOT_BINDING_SCHEMA:
        raise ProtocolError("snapshot-binding schema drifted.")
    if (
        payload.get("snapshot_artifact_id") != SNAPSHOT_ARTIFACT_ID
        or payload.get("publication_state") != HASH_PROMOTED
        or payload.get("historical_paths_read") is not False
    ):
        raise ProtocolError("snapshot binding is not the promoted canonical input.")
    manifest_hashes = {record.snapshot_manifest_hash for record in records}
    if (
        len(manifest_hashes) != 1
        or payload.get("snapshot_manifest_hash") != next(iter(manifest_hashes))
        or payload.get("key_inventory_hash") != inventory_hash
    ):
        raise ProtocolError("snapshot binding does not match the consumed key inventory.")
    for key in ("snapshot_hash", "snapshot_manifest_hash", "key_inventory_hash"):
        value = payload.get(key)
        if not isinstance(value, str) or not _SEMANTIC_HASH.fullmatch(value):
            raise ProtocolError(f"snapshot binding {key} must be a semantic hash.")
    if (
        protocol
        and payload.get("snapshot_manifest_hash")
        != protocol.get("snapshot_manifest_hash")
    ):
        raise ProtocolError("protocol and snapshot binding disagree.")
    if protocol and _eval_row_inventory_hashes(payload) != _eval_row_inventory_hashes(
        protocol
    ):
        raise ProtocolError("protocol and snapshot eval-row inventories disagree.")
    _assert_all_may_flags_false(payload, "snapshot binding")


def _eval_row_inventory_hashes(
    payload: Mapping[str, object],
) -> dict[str, str]:
    mapping = _as_mapping(
        payload.get("eval_row_inventory_hashes"),
        "eval_row_inventory_hashes",
    )
    if set(mapping) != set(AUDIT_CENTERS):
        raise ProtocolError("eval-row inventory hashes must cover exactly four centers.")
    result = {str(center): str(value) for center, value in mapping.items()}
    for center, value in result.items():
        _require_digest(value, f"eval-row inventory hash for center {center}", full=True)
    return result


def _validate_dynamic_member_coverage(
    root: Path,
    records: Sequence[AuditKeyRecord],
) -> None:
    expected_stems = {record.key_hash for record in records}
    for relative, suffix in (
        ("checkpoints", ".pt"),
        ("jobs", ".json"),
        ("reports/training_diagnostics", ".json"),
    ):
        directory = root / relative
        if not directory.exists():
            continue
        observed = {
            member.stem
            for member in directory.iterdir()
            if member.is_file() and member.suffix == suffix
        }
        extras = sorted(observed.difference(expected_stems))
        if extras:
            raise ProtocolError(
                f"{relative} contains stale or undeclared per-key members: {extras[:3]}"
            )


def _validate_jobs(
    rows: Sequence[Mapping[str, str]],
    *,
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
) -> tuple[
    dict[tuple[str, int, str], Mapping[str, str]],
    dict[str, int],
]:
    _require_columns(rows, _JOB_COLUMNS, "job inventory")
    coordinates = [_row_coordinate(row) for row in rows]
    if (
        len(rows) != EXPECTED_JOB_COUNT
        or set(coordinates) != set(records_by_coordinate)
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("job inventory is not the exact 4x3x3 key product.")
    by_coordinate = dict(zip(coordinates, rows, strict=True))
    optimizer_updates = 0
    decoder_total = 0
    decoder_by_candidate = {candidate: 0 for candidate in AUDIT_CANDIDATES}
    consumptions = 0
    for coordinate, row in by_coordinate.items():
        record = records_by_coordinate[coordinate]
        _assert_csv_may_flags_false(row, f"job {coordinate}")
        if row["key_hash"] != record.key_hash:
            raise ProtocolError(f"job key hash mismatch for {coordinate}.")
        expected_pair = "" if record.is_legacy else str(record.pair_id)
        if row["pair_id"] != expected_pair:
            raise ProtocolError(f"job pair ID mismatch for {coordinate}.")
        if (
            row["execution_device"] not in {"cuda:0", "cuda:1"}
            or row["execution_device"] != record.execution_device
        ):
            raise ProtocolError(f"job device is not a frozen workstation GPU: {coordinate}.")
        for field, expected in (
            ("prepared_sha256", record.prepared_sha256),
            ("prepared_content_hash", record.prepared_content_hash),
            ("schedule_sha256", record.schedule_sha256),
            ("schedule_content_hash", record.schedule_content_hash),
            ("epsilon_trace_sha256", record.epsilon_trace_sha256),
            ("epsilon_trace_content_hash", record.epsilon_trace_content_hash),
        ):
            if row[field] != expected:
                raise ProtocolError(f"job {field} mismatches its key record: {coordinate}.")
        _require_digest(row["initialization_hash"], "initialization_hash", full=True)
        _require_digest(row["checkpoint_hash"], "checkpoint_hash", full=True)
        _require_digest(row["schedule_hash"], "schedule_hash", full=False)
        _require_digest(
            row["posterior_stream_hash"], "posterior_stream_hash", full=False
        )
        expected_estimator = (
            "antithetic_epsilon"
            if coordinate[2] == FIXED_ANTITHETIC
            else "one_epsilon"
        )
        expected_forwards = 2000 if coordinate[2] == FIXED_ANTITHETIC else 1000
        if (
            row["posterior_estimator"] != expected_estimator
            or _integer(row["optimizer_steps"], "optimizer_steps") != 1000
            or _integer(row["decoder_forwards"], "decoder_forwards")
            != expected_forwards
            or _integer(row["epsilon_consumptions"], "epsilon_consumptions") != 1
            or row["cache_status"] != "COMPLETED"
            or row["status"] != "PASS"
            or row["claim_scope"] != CLAIM_SCOPE
        ):
            raise ProtocolError(f"job execution accounting mismatches: {coordinate}.")
        optimizer_updates += int(row["optimizer_steps"])
        decoder_total += int(row["decoder_forwards"])
        decoder_by_candidate[coordinate[2]] += int(row["decoder_forwards"])
        consumptions += int(row["epsilon_consumptions"])

    _validate_controlled_job_invariance(by_coordinate)
    if (
        optimizer_updates != EXPECTED_OPTIMIZER_UPDATES
        or decoder_by_candidate[LEGACY_CANDIDATE]
        != EXPECTED_LEGACY_DECODER_FORWARDS
        or decoder_by_candidate[FIXED_ONE_EPSILON]
        != EXPECTED_FIXED_ONE_DECODER_FORWARDS
        or decoder_by_candidate[FIXED_ANTITHETIC]
        != EXPECTED_ANTITHETIC_DECODER_FORWARDS
        or decoder_total != EXPECTED_DECODER_FORWARDS
        or consumptions != EXPECTED_EPSILON_CONSUMPTIONS
    ):
        raise ProtocolError("aggregate optimizer/decoder/epsilon accounting mismatches.")
    return by_coordinate, {
        "job_count": len(rows),
        "optimizer_updates": optimizer_updates,
        "decoder_forwards": decoder_total,
        "epsilon_consumptions": consumptions,
    }


def _validate_controlled_job_invariance(
    jobs: Mapping[tuple[str, int, str], Mapping[str, str]]
) -> None:
    for center in AUDIT_CENTERS:
        controlled = [
            row
            for (row_center, _, candidate), row in jobs.items()
            if row_center == center and candidate in CONTROLLED_CANDIDATES
        ]
        for field in (
            "prepared_sha256",
            "prepared_content_hash",
            "schedule_sha256",
            "schedule_content_hash",
            "epsilon_trace_sha256",
            "epsilon_trace_content_hash",
        ):
            if len({row[field] for row in controlled}) != 1:
                raise ProtocolError(
                    f"controlled {field} is not seed/candidate invariant for center={center}."
                )
        initialization_by_seed: dict[int, str] = {}
        for seed in INITIALIZATION_SEEDS:
            pair = [
                jobs[(center, seed, candidate)]
                for candidate in CONTROLLED_CANDIDATES
            ]
            if len({row["initialization_hash"] for row in pair}) != 1:
                raise ProtocolError(
                    f"controlled initialization differs within pair center={center}, seed={seed}."
                )
            initialization_by_seed[seed] = pair[0]["initialization_hash"]
        if len(set(initialization_by_seed.values())) != len(INITIALIZATION_SEEDS):
            raise ProtocolError(
                f"controlled initialization seeds are not distinct for center={center}."
            )


def _validate_replay_trace_audit(
    rows: Sequence[Mapping[str, str]],
    *,
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
) -> None:
    _require_columns(rows, _REPLAY_COLUMNS, "replay trace audit")
    coordinates = [_row_coordinate(row) for row in rows]
    if (
        len(rows) != EXPECTED_JOB_COUNT
        or set(coordinates) != set(records_by_coordinate)
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("replay trace audit coverage is not exact.")
    for coordinate, row in zip(coordinates, rows, strict=True):
        _assert_csv_may_flags_false(row, f"replay trace {coordinate}")
        if row["key_hash"] != records_by_coordinate[coordinate].key_hash:
            raise ProtocolError(f"replay trace key mismatch for {coordinate}.")
        for field in (
            "prepared_file_match",
            "prepared_content_match",
            "schedule_file_match",
            "schedule_content_match",
            "epsilon_file_match",
            "epsilon_content_match",
        ):
            if not _boolean(row[field], field):
                raise ProtocolError(f"replay trace mismatch for {coordinate}/{field}.")
        if (
            _integer(row["trace_consumption_count"], "trace_consumption_count") != 1
            or row["status"] != "PASS"
        ):
            raise ProtocolError(f"replay trace consumption failed for {coordinate}.")


def _validate_legacy_replay(
    rows: Sequence[Mapping[str, str]],
    *,
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
    jobs_by_coordinate: Mapping[tuple[str, int, str], Mapping[str, str]],
) -> None:
    _require_columns(rows, _LEGACY_COLUMNS, "legacy-v2 validation")
    expected_coordinates = {
        (center, seed, LEGACY_CANDIDATE)
        for center in AUDIT_CENTERS
        for seed in INITIALIZATION_SEEDS
    }
    coordinates = [_row_coordinate(row) for row in rows]
    if (
        len(rows) != EXPECTED_LEGACY_JOB_COUNT
        or set(coordinates) != expected_coordinates
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("legacy replay coverage is not exact 4x3.")
    for coordinate, row in zip(coordinates, rows, strict=True):
        _assert_csv_may_flags_false(row, f"legacy replay {coordinate}")
        record = records_by_coordinate[coordinate]
        if row["key_hash"] != record.key_hash or row["status"] != "PASS":
            raise ProtocolError(f"legacy replay key/status mismatch for {coordinate}.")
        for field in _LEGACY_FIELDS:
            expected = row[f"expected_{field}_hash"]
            observed = row[f"observed_{field}_hash"]
            _require_digest(expected, f"legacy expected {field}", full=None)
            _require_digest(observed, f"legacy observed {field}", full=None)
            if (
                expected != observed
                or not _boolean(row[f"{field}_match"], f"{field}_match")
            ):
                raise ProtocolError(f"legacy {field} replay mismatch for {coordinate}.")
        if (
            row["expected_initialization_hash"]
            != record.legacy_expected_initialization_hash
            or row["expected_schedule_hash"]
            != record.legacy_historical_schedule_hash
            or row["expected_posterior_hash"]
            != record.legacy_historical_posterior_stream_hash
            or
            row["expected_checkpoint_hash"]
            != record.legacy_expected_checkpoint_hash
            or row["expected_prediction_hash"]
            != record.legacy_expected_prediction_hash
            or row["expected_metric_hash"] != record.legacy_expected_metric_hash
        ):
            raise ProtocolError(
                f"legacy promoted expected hashes mismatch key record for {coordinate}."
            )
        job = jobs_by_coordinate.get(coordinate)
        if job is None:
            raise ProtocolError(f"legacy job evidence is absent for {coordinate}.")
        if (
            row["observed_initialization_hash"] != job["initialization_hash"]
            or row["observed_checkpoint_hash"] != job["checkpoint_hash"]
            or row["observed_schedule_hash"] != job["schedule_hash"]
            or row["observed_posterior_hash"] != job["posterior_stream_hash"]
        ):
            raise ProtocolError(f"legacy replay hashes disagree with job for {coordinate}.")
        if (
            not _boolean(row.get("metric_values_match", ""), "metric_values_match")
            or _boolean(row.get("comparison_eligible", ""), "comparison_eligible")
            or not _boolean(
                row.get("replay_validation_only", ""), "replay_validation_only"
            )
            or row.get("claim_scope") != CLAIM_SCOPE
        ):
            raise ProtocolError(
                f"legacy replay role/firewall mismatch for {coordinate}."
            )


def _validate_controlled_metrics(
    rows: Sequence[Mapping[str, str]],
    *,
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
) -> list[dict[str, object]]:
    expected = {
        (center, seed, candidate)
        for center in AUDIT_CENTERS
        for seed in INITIALIZATION_SEEDS
        for candidate in CONTROLLED_CANDIDATES
    }
    coordinates = [
        (
            str(row.get("center", "")),
            _integer(row.get("training_seed", ""), "training_seed"),
            str(row.get("candidate", "")),
        )
        for row in rows
    ]
    if (
        len(rows) != EXPECTED_CONTROLLED_JOB_COUNT
        or set(coordinates) != expected
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("controlled metric coverage is not exact 4x3x2.")
    typed: list[dict[str, object]] = []
    for coordinate, row in zip(coordinates, rows, strict=True):
        _assert_csv_may_flags_false(row, f"controlled metric {coordinate}")
        record = records_by_coordinate[coordinate]
        if (
            row.get("key_hash") != record.key_hash
            or row.get("pair_id") != record.pair_id
        ):
            raise ProtocolError(
                f"controlled metric key/pair binding mismatch for {coordinate}."
            )
        tp, fn, tn, fp = (
            _integer(row[name], name) for name in ("tp", "fn", "tn", "fp")
        )
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        bacc = 0.5 * (recall + specificity)
        macro_f1 = 0.5 * (
            (2.0 * tp / max(1, 2 * tp + fp + fn))
            + (2.0 * tn / max(1, 2 * tn + fp + fn))
        )
        real_bacc = _finite_float(row["real_reference_bacc"], "real_reference_bacc")
        if real_bacc < 0.60:
            raise ProtocolError(f"real BACC denominator floor failed for {coordinate}.")
        preservation = (bacc - 0.5) / (real_bacc - 0.5)
        if max(
            abs(_finite_float(row["positive_recall"], "positive_recall") - recall),
            abs(_finite_float(row["specificity"], "specificity") - specificity),
            abs(_finite_float(row["bacc"], "bacc") - bacc),
            abs(_finite_float(row["macro_f1"], "macro_f1") - macro_f1),
            abs(_finite_float(row["preservation_ratio"], "preservation_ratio") - preservation),
        ) > 1e-12:
            raise ProtocolError(f"metric/confusion arithmetic mismatch for {coordinate}.")
        if (
            not _boolean(row["eval_labels_used_for_scoring"], "eval_labels_used_for_scoring")
            or not _boolean(
                row["eval_labels_used_for_decode_condition"],
                "eval_labels_used_for_decode_condition",
            )
            or _boolean(row["eval_labels_used_for_fit"], "eval_labels_used_for_fit")
            or _boolean(
                row["eval_labels_used_for_selection"],
                "eval_labels_used_for_selection",
            )
            or _boolean(row["oracle_eligible"], "oracle_eligible")
            or row["selection_source"] != "none"
            or row["claim_scope"] != CLAIM_SCOPE
        ):
            raise ProtocolError(f"controlled metric label/firewall mismatch for {coordinate}.")
        typed.append(
            {
                key: _coerce_metric_value(key, value)
                for key, value in row.items()
            }
        )
    return typed


def _validate_decoded_predictions(
    rows: Sequence[Mapping[str, str]],
    *,
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
    typed_metrics: Sequence[Mapping[str, object]],
    legacy_rows: Sequence[Mapping[str, str]],
    eval_row_inventory_hashes: Mapping[str, str],
) -> None:
    required = {
        "center",
        "training_seed",
        "candidate",
        "key_hash",
        "pair_id",
        "sample_id",
        "case_id",
        "y_true",
        "y_pred",
        "real_reference_y_pred",
        "representation_role",
        "eval_label_role",
        "selection_source",
        "oracle_eligible",
        "claim_scope",
    }
    _require_columns(rows, frozenset(required), "decoded predictions")
    grouped: dict[tuple[str, int, str], list[Mapping[str, str]]] = {}
    for row in rows:
        coordinate = (
            str(row["center"]),
            _integer(row["training_seed"], "training_seed"),
            str(row["candidate"]),
        )
        if coordinate not in records_by_coordinate:
            raise ProtocolError(f"Decoded prediction has undeclared key {coordinate}.")
        grouped.setdefault(coordinate, []).append(row)
    if set(grouped) != set(records_by_coordinate):
        raise ProtocolError("Decoded prediction coverage is not the exact 36-key panel.")

    inventory_by_center: dict[str, tuple[tuple[str, str, int], ...]] = {}
    real_predictions_by_center: dict[str, tuple[tuple[str, int], ...]] = {}
    controlled_metrics = {
        (
            str(row["center"]),
            int(row["training_seed"]),
            str(row["candidate"]),
        ): row
        for row in typed_metrics
    }
    legacy_by_coordinate = {
        _row_coordinate(row): row
        for row in legacy_rows
    }
    for coordinate, group in grouped.items():
        record = records_by_coordinate[coordinate]
        expected_pair = "" if record.is_legacy else str(record.pair_id)
        samples = [str(row["sample_id"]) for row in group]
        if not group or len(samples) != len(set(samples)) or any(not value for value in samples):
            raise ProtocolError(f"Decoded prediction rows are empty/duplicate for {coordinate}.")
        for row in group:
            if (
                row["key_hash"] != record.key_hash
                or row["pair_id"] != expected_pair
                or not row["case_id"]
                or row["representation_role"] != "decode_mu"
                or row["eval_label_role"]
                != "final_diagnostic_scoring_and_decode_condition_only"
                or row["selection_source"] != "none"
                or _boolean(row["oracle_eligible"], "oracle_eligible")
                or row["claim_scope"] != CLAIM_SCOPE
            ):
                raise ProtocolError(
                    f"Decoded prediction identity/firewall mismatch for {coordinate}."
                )
            if (
                _integer(row["y_true"], "y_true") not in {0, 1}
                or _integer(row["y_pred"], "y_pred") not in {0, 1}
                or _integer(
                    row["real_reference_y_pred"],
                    "real_reference_y_pred",
                )
                not in {0, 1}
            ):
                raise ProtocolError(f"Decoded prediction is nonbinary for {coordinate}.")
        observed_row_hash = canonical_mapping_hash(
            {
                "schema_version": "midogpp_b_prepared_row_inventory_v1",
                "rows": [
                    {
                        "sample_id": str(row["sample_id"]),
                        "case_id": str(row["case_id"]),
                        "label": _integer(row["y_true"], "y_true"),
                    }
                    for row in group
                ],
            }
        )
        if observed_row_hash != eval_row_inventory_hashes.get(coordinate[0]):
            raise ProtocolError(
                f"Decoded prediction inventory does not match the promoted "
                f"snapshot eval rows for center {coordinate[0]}."
            )
        inventory = tuple(
            sorted(
                (
                    str(row["sample_id"]),
                    str(row["case_id"]),
                    _integer(row["y_true"], "y_true"),
                )
                for row in group
            )
        )
        previous = inventory_by_center.setdefault(coordinate[0], inventory)
        if previous != inventory:
            raise ProtocolError(
                f"Decoded evaluation row inventory drifted within center {coordinate[0]}."
            )
        real_inventory = tuple(
            sorted(
                (
                    str(row["sample_id"]),
                    _integer(
                        row["real_reference_y_pred"],
                        "real_reference_y_pred",
                    ),
                )
                for row in group
            )
        )
        previous_real = real_predictions_by_center.setdefault(
            coordinate[0],
            real_inventory,
        )
        if previous_real != real_inventory:
            raise ProtocolError(
                f"Real-reference predictions drifted within center {coordinate[0]}."
            )
        real_reference_bacc = _balanced_accuracy_from_rows(
            group,
            prediction_field="real_reference_y_pred",
        )
        if real_reference_bacc < 0.60:
            raise ProtocolError(
                f"Real-reference BACC floor failed for center {coordinate[0]}."
            )
        metric_subset = _prediction_metric_subset(group)
        if record.is_legacy:
            legacy_row = legacy_by_coordinate.get(coordinate)
            if legacy_row is None:
                raise ProtocolError(f"Legacy prediction lacks replay row for {coordinate}.")
            if (
                prediction_digest(group) != legacy_row["observed_prediction_hash"]
                or canonical_mapping_hash(metric_subset)
                != legacy_row["observed_metric_hash"]
            ):
                raise ProtocolError(
                    f"Legacy prediction/hash evidence does not recompute for {coordinate}."
                )
        else:
            metric = controlled_metrics.get(coordinate)
            if metric is None:
                raise ProtocolError(
                    f"Controlled prediction lacks metric row for {coordinate}."
                )
            for field in ("tp", "fn", "tn", "fp"):
                if int(metric[field]) != int(metric_subset[field]):
                    raise ProtocolError(
                        f"Prediction confusion count disagrees for {coordinate}/{field}."
                    )
            for field in ("bacc", "positive_recall", "specificity"):
                if abs(float(metric[field]) - float(metric_subset[field])) > 1e-12:
                    raise ProtocolError(
                        f"Prediction metric disagrees for {coordinate}/{field}."
                    )
            if (
                abs(
                    float(metric["real_reference_bacc"])
                    - real_reference_bacc
                )
                > 1e-12
            ):
                raise ProtocolError(
                    f"Real-reference BACC disagrees with persisted predictions "
                    f"for {coordinate}."
                )


def _prediction_metric_subset(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, int | float]:
    truth = [_integer(row["y_true"], "y_true") for row in rows]
    predicted = [_integer(row["y_pred"], "y_pred") for row in rows]
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted, strict=True))
    positive_recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "bacc": 0.5 * (positive_recall + specificity),
        "positive_recall": positive_recall,
        "specificity": specificity,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tp": tp,
    }


def _balanced_accuracy_from_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    prediction_field: str,
) -> float:
    truth = [_integer(row["y_true"], "y_true") for row in rows]
    predicted = [
        _integer(row[prediction_field], prediction_field)
        for row in rows
    ]
    positive = [prediction for label, prediction in zip(truth, predicted, strict=True) if label == 1]
    negative = [prediction for label, prediction in zip(truth, predicted, strict=True) if label == 0]
    if not positive or not negative:
        raise ProtocolError("Balanced-accuracy evidence must contain both classes.")
    recall = sum(value == 1 for value in positive) / len(positive)
    specificity = sum(value == 0 for value in negative) / len(negative)
    return 0.5 * (recall + specificity)


def _validate_paired_comparisons(
    rows: Sequence[Mapping[str, str]],
    *,
    typed_metrics: Sequence[Mapping[str, object]],
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
) -> None:
    expected_coordinates = {
        (center, seed) for center in AUDIT_CENTERS for seed in INITIALIZATION_SEEDS
    }
    coordinates = [
        (
            str(row.get("center", "")),
            _integer(row.get("training_seed", ""), "training_seed"),
        )
        for row in rows
    ]
    if (
        len(rows) != EXPECTED_PAIR_COUNT
        or set(coordinates) != expected_coordinates
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("paired comparison coverage is not exact 4x3.")
    metric_by_coordinate = {
        (
            str(row["center"]),
            int(row["training_seed"]),
            str(row["candidate"]),
        ): row
        for row in typed_metrics
    }
    for coordinate, row in zip(coordinates, rows, strict=True):
        _assert_csv_may_flags_false(row, f"paired comparison {coordinate}")
        baseline_key = (*coordinate, FIXED_ONE_EPSILON)
        proposed_key = (*coordinate, FIXED_ANTITHETIC)
        baseline = metric_by_coordinate.get(baseline_key)
        proposed = metric_by_coordinate.get(proposed_key)
        if baseline is None or proposed is None:
            raise ProtocolError(f"paired comparison lacks metric evidence for {coordinate}.")
        canonical_pair = records_by_coordinate[baseline_key].pair_id
        if (
            canonical_pair is None
            or records_by_coordinate[proposed_key].pair_id != canonical_pair
            or row.get("pair_id") != canonical_pair
        ):
            raise ProtocolError(f"paired comparison pair ID mismatch for {coordinate}.")
        if (
            row.get("baseline") != FIXED_ONE_EPSILON
            or row.get("candidate") != FIXED_ANTITHETIC
            or row.get("comparison_role") != "controlled_common_random_numbers"
            or _boolean(row.get("legacy_v2_included", ""), "legacy_v2_included")
            or row.get("claim_scope") != CLAIM_SCOPE
        ):
            raise ProtocolError(f"paired comparison role/firewall mismatch for {coordinate}.")
        for metric in (
            "bacc",
            "macro_f1",
            "positive_recall",
            "specificity",
            "preservation_ratio",
        ):
            expected_delta = float(proposed[metric]) - float(baseline[metric])
            observed = _finite_float(row[f"delta_{metric}"], f"delta_{metric}")
            if abs(observed - expected_delta) > 1e-12:
                raise ProtocolError(
                    f"paired delta_{metric} does not recompute for {coordinate}."
                )


def _validate_consumption_audit(
    rows: Sequence[Mapping[str, str]],
    *,
    jobs_by_coordinate: Mapping[tuple[str, int, str], Mapping[str, str]],
    records_by_coordinate: Mapping[tuple[str, int, str], AuditKeyRecord],
) -> None:
    _require_columns(rows, _CONSUMPTION_COLUMNS, "consumption audit")
    coordinates = [_row_coordinate(row) for row in rows]
    if (
        len(rows) != EXPECTED_JOB_COUNT
        or set(coordinates) != set(records_by_coordinate)
        or len(coordinates) != len(set(coordinates))
    ):
        raise ProtocolError("epsilon consumption audit coverage is not exact.")
    totals = {"optimizer_steps": 0, "decoder_forwards": 0, "epsilon": 0}
    for coordinate, row in zip(coordinates, rows, strict=True):
        _assert_csv_may_flags_false(row, f"consumption audit {coordinate}")
        job = jobs_by_coordinate.get(coordinate)
        if job is None or row["key_hash"] != records_by_coordinate[coordinate].key_hash:
            raise ProtocolError(f"consumption audit key mismatch for {coordinate}.")
        expected_forwards = 2000 if coordinate[2] == FIXED_ANTITHETIC else 1000
        values = {
            "epsilon": _integer(
                row["epsilon_consumption_count"], "epsilon_consumption_count"
            ),
            "optimizer_steps": _integer(row["optimizer_steps"], "optimizer_steps"),
            "decoder_forwards": _integer(row["decoder_forwards"], "decoder_forwards"),
        }
        if (
            values != {
                "epsilon": 1,
                "optimizer_steps": 1000,
                "decoder_forwards": expected_forwards,
            }
            or row["status"] != "PASS"
            or str(values["epsilon"]) != job["epsilon_consumptions"]
            or str(values["optimizer_steps"]) != job["optimizer_steps"]
            or str(values["decoder_forwards"]) != job["decoder_forwards"]
        ):
            raise ProtocolError(f"consumption audit mismatch for {coordinate}.")
        for name, value in values.items():
            totals[name] += value
    if totals != {
        "epsilon": EXPECTED_EPSILON_CONSUMPTIONS,
        "optimizer_steps": EXPECTED_OPTIMIZER_UPDATES,
        "decoder_forwards": EXPECTED_DECODER_FORWARDS,
    }:
        raise ProtocolError("consumption audit aggregate totals mismatch.")


def _validate_decision(
    payload: Mapping[str, object],
    *,
    typed_metrics: Sequence[Mapping[str, object]],
) -> None:
    expected = audit_decision(
        typed_metrics,
        thresholds=DecisionThresholds().to_payload(),
    )
    if payload != expected:
        raise ProtocolError("audit decision does not independently recompute.")
    if (
        payload.get("legacy_v2_used_for_decision") is not False
        or payload.get("legacy_v2_role") != "exact_replay_validation_only"
    ):
        raise ProtocolError("legacy-v2 rows crossed into the controlled decision.")
    _assert_all_may_flags_false(payload, "audit decision")


def _validate_leakage_report(payload: Mapping[str, object]) -> None:
    if payload.get("status") != "PASS":
        raise ProtocolError("leakage report status is not PASS.")
    _require_audit_identity(payload, require_all=False)
    for field, expected in _LEAKAGE_FLAGS.items():
        if payload.get(field) is not expected:
            raise ProtocolError(f"leakage report flag {field!r} drifted.")
    _validate_firewall(payload.get("claim_firewall"), "leakage claim firewall")
    _assert_all_may_flags_false(payload, "leakage report")


def _validate_runtime_summary(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != RUNTIME_SUMMARY_SCHEMA:
        raise ProtocolError("runtime-summary schema drifted.")
    expected = {
        "job_count": EXPECTED_JOB_COUNT,
        "legacy_job_count": EXPECTED_LEGACY_JOB_COUNT,
        "controlled_job_count": EXPECTED_CONTROLLED_JOB_COUNT,
        "controlled_pair_count": EXPECTED_PAIR_COUNT,
        "optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "legacy_decoder_forwards": EXPECTED_LEGACY_DECODER_FORWARDS,
        "fixed_one_epsilon_decoder_forwards": EXPECTED_FIXED_ONE_DECODER_FORWARDS,
        "antithetic_decoder_forwards": EXPECTED_ANTITHETIC_DECODER_FORWARDS,
        "decoder_forwards": EXPECTED_DECODER_FORWARDS,
        "epsilon_consumptions": EXPECTED_EPSILON_CONSUMPTIONS,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ProtocolError(f"runtime-summary counter {field!r} mismatches.")
    _require_audit_identity(payload, require_all=False)
    _validate_firewall(payload.get("claim_firewall"), "runtime claim firewall")
    _assert_all_may_flags_false(payload, "runtime summary")


def _validate_firewall(value: object, label: str) -> None:
    mapping = _as_mapping(value, label)
    if set(mapping) != set(CLAIM_FIREWALL_FIELDS):
        raise ProtocolError(f"{label} must contain the exact frozen flag set.")
    if any(mapping[field] is not False for field in CLAIM_FIREWALL_FIELDS):
        raise ProtocolError(f"{label} contains an enabled flag.")
    if mapping != ClaimFirewall().to_payload():
        raise ProtocolError(f"{label} differs from the canonical firewall.")


def _assert_all_may_flags_false(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).startswith("may_") and child is not False:
                raise ProtocolError(f"{location} enables {key!r}.")
            _assert_all_may_flags_false(child, location)
    elif isinstance(value, list):
        for child in value:
            _assert_all_may_flags_false(child, location)


def _assert_csv_may_flags_false(row: Mapping[str, str], location: str) -> None:
    for key, value in row.items():
        if key.startswith("may_") and _boolean(value, key):
            raise ProtocolError(f"{location} enables {key!r}.")


def _require_audit_identity(
    payload: Mapping[str, object], *, require_all: bool = True
) -> None:
    expected = {
        "stage": STAGE,
        "evidence_label": EVIDENCE_LABEL,
        "claim_scope": CLAIM_SCOPE,
    }
    for field, value in expected.items():
        if field in payload or require_all:
            if payload.get(field) != value:
                raise ProtocolError(f"audit identity field {field!r} drifted.")


def _record_coordinate(record: AuditKeyRecord) -> tuple[str, int, str]:
    return (record.center, record.initialization_seed, record.candidate)


def _row_coordinate(row: Mapping[str, str]) -> tuple[str, int, str]:
    return (
        str(row.get("center", "")),
        _integer(row.get("initialization_seed", ""), "initialization_seed"),
        str(row.get("candidate", "")),
    )


def _require_columns(
    rows: Sequence[Mapping[str, str]], required: Iterable[str], label: str
) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    columns = set(rows[0])
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ProtocolError(f"{label} misses required columns: {missing}")
    if any(set(row) != columns for row in rows):
        raise ProtocolError(f"{label} rows do not share one schema.")


def _integer(value: object, label: str) -> int:
    text = str(value)
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ProtocolError(f"{label} must be an integer.")
    return int(text)


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite.")
    return result


def _boolean(value: object, label: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ProtocolError(f"{label} must be a strict boolean.")


def _require_digest(value: object, label: str, *, full: bool | None) -> None:
    text = str(value)
    valid = (
        bool(_FULL_SHA256.fullmatch(text))
        if full is True
        else bool(_SEMANTIC_HASH.fullmatch(text))
        if full is False
        else bool(_FULL_SHA256.fullmatch(text) or _SEMANTIC_HASH.fullmatch(text))
    )
    if not valid:
        raise ProtocolError(f"{label} is not a canonical digest.")


def _coerce_metric_value(key: str, value: str) -> object:
    if key in {
        "training_seed",
        "tp",
        "fn",
        "tn",
        "fp",
        "n_positive",
        "n_negative",
    }:
        return _integer(value, key)
    if key in {
        "bacc",
        "macro_f1",
        "positive_recall",
        "specificity",
        "preservation_ratio",
        "real_reference_bacc",
    }:
        return _finite_float(value, key)
    if key in {
        "eval_labels_used_for_fit",
        "eval_labels_used_for_selection",
        "eval_labels_used_for_scoring",
        "eval_labels_used_for_decode_condition",
        "oracle_eligible",
    }:
        return _boolean(value, key)
    return value


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be a mapping.")
    return value


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected a JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"cannot read CSV table: {path}") from exc


def _report(
    errors: Sequence[str], counts: Mapping[str, int]
) -> dict[str, object]:
    passed = not errors
    status = "PASS" if passed else "FAIL"
    check = "PASS" if passed else "SEE_ERRORS"
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": status,
        "errors": list(errors),
        "counts": dict(counts),
        "checks": {
            "required_members": check,
            "snapshot_and_key_bindings": check,
            "exact_job_and_pair_coverage": check,
            "optimizer_and_decoder_accounting": check,
            "single_epsilon_consumption": check,
            "legacy_replay_hashes_and_metrics": check,
            "controlled_common_random_numbers": check,
            "snapshot_eval_row_inventory_binding": check,
            "real_reference_denominator_recomputation": check,
            "metric_and_pair_recomputation": check,
            "per_key_member_coverage": check,
            "legacy_excluded_from_decision": check,
            "claim_firewall": check,
            "leakage_provenance": check,
            "content_index": check,
        },
        "claim_scope": CLAIM_SCOPE,
        "evidence_label": EVIDENCE_LABEL,
        "may_feed_deployable_selection": False,
    }


__all__ = (
    "AUDIT_REQUIRED_FILES",
    "VALIDATION_SCHEMA",
    "assert_valid_audit_bundle",
    "validate_audit_bundle",
)
