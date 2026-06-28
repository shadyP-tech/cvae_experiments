"""MIDOG++ adapter for diagnostic downstream utility artifacts.

The adapter is translation-only: it builds candidate/provenance rows from
source manifests and writes diagnostic downstream matrices. It intentionally
does not compute target-metric-derived selections, ranks, or top-1 choices.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import assert_frozen_snapshot_exists
from ..protocol import ProtocolError
from ..schemas import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE
from ..schemas.midogpp import (
    MIDOGPP_DATASET_NAME,
    MIDOGPP_DOWNSTREAM_COLUMNS,
    MIDOGPP_DOWNSTREAM_PRIMARY_KEY,
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_METHOD_BASELINE_ROW_TYPE,
    MIDOGPP_MATRIX_SCHEMA_VERSION,
    MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
    MidogppDownstreamRow,
    assert_midogpp_candidate_pool,
    midogpp_row_from_mapping,
)
from ..reports.tables import write_rows
from ..utility_matrix import assert_diagnostic_matrix_path, diagnostic_matrix_path

MIDOGPP_ORACLE_SUMMARY_COLUMNS = (
    "dataset",
    "domain_regime",
    "heldout_center",
    "experiment_seed",
    "replicate_seed",
    "support_size",
    "support_seed",
    "support_set_id",
    "eval_set_id",
    "generation_seed",
    "classifier_seed",
    "synthetic_per_class_total",
    "config_hash",
    "protocol_hash",
    "feature_frame_hash",
    "oracle_candidate_id",
    "oracle_candidate_source_center",
    "oracle_candidate_method",
    "oracle_bacc",
    "oracle_macro_f1",
    "mean_single_source_bacc",
    "min_single_source_bacc",
    "max_single_source_bacc",
    "spread_max_minus_mean_bacc",
    "spread_max_minus_min_bacc",
    "n_ok_single_source_candidates",
    "claim_role",
)

MIDOGPP_BASELINE_COMPARISON_COLUMNS = (
    "baseline_method",
    "heldout_center",
    "mean_bacc",
    "mean_macro_f1",
    "mean_oracle_gap_bacc",
    "mean_oracle_gap_macro_f1",
    "n_rows",
    "claim_role",
)


def midogpp_diagnostic_matrix_path(artifacts_root: Path, *, suffix: str = ".csv") -> Path:
    """Return the canonical quarantined MIDOG++ diagnostic matrix path."""

    return diagnostic_matrix_path(artifacts_root, suffix=suffix)


def build_candidate_manifest_from_source_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    candidate_method: str = "single_source_adaptive_k",
) -> list[dict[str, object]]:
    """Build a manifest-driven candidate surface from source summary rows.

    One deployable candidate is emitted for each successful source center. The
    held-out center and domain 4 are excluded from deployable rows.
    """

    by_source: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        source = str(row.get("source_center", ""))
        by_source.setdefault(source, []).append(row)

    candidates: list[dict[str, object]] = []
    for source in sorted(by_source, key=_center_sort_key):
        if source == str(heldout_center) or source not in MIDOGPP_ELIGIBLE_CENTERS:
            continue
        source_rows = by_source[source]
        summary_hashes = sorted(str(row.get("summary_hash", "")) for row in source_rows)
        config_hashes = sorted(str(row.get("expert_config_hash", "")) for row in source_rows)
        experiment_seeds = sorted({int(row.get("experiment_seed", 0)) for row in source_rows})
        candidates.append(
            {
                "dataset": MIDOGPP_DATASET_NAME,
                "domain_regime": "heldout_center",
                "heldout_center": str(heldout_center),
                "candidate_source_center": source,
                "candidate_id": f"midogpp_source_{source}_{candidate_method}",
                "candidate_method": candidate_method,
                "expert_pool_type": "single_source",
                "row_type": MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
                "eligibility": SELECTION_ELIGIBLE,
                "summary_hashes": "|".join(summary_hashes),
                "expert_config_hashes": "|".join(config_hashes),
                "experiment_seeds": "|".join(str(seed) for seed in experiment_seeds),
                "source_summary_row_count": len(source_rows),
            }
        )
    assert_midogpp_candidate_pool(heldout_center=str(heldout_center), candidate_rows=candidates)
    return candidates


def write_midogpp_diagnostic_matrix(
    path: Path,
    rows: Sequence[MidogppDownstreamRow],
) -> None:
    """Write a MIDOG++ diagnostic matrix and schema sidecar."""

    assert_diagnostic_matrix_path(path)
    _validate_midogpp_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_midogpp_schema(path.with_suffix(".schema.json"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MIDOGPP_DOWNSTREAM_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def read_midogpp_diagnostic_matrix(path: Path) -> list[MidogppDownstreamRow]:
    """Read a quarantined MIDOG++ diagnostic matrix."""

    assert_diagnostic_matrix_path(path)
    _assert_midogpp_schema(path.with_suffix(".schema.json"))
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [midogpp_row_from_mapping(row) for row in csv.DictReader(handle)]
    _validate_midogpp_rows(rows)
    return rows


def read_midogpp_scored_rows(path: Path) -> list[MidogppDownstreamRow]:
    """Read pre-scored MIDOG++ rows before writing canonical artifacts."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [midogpp_row_from_mapping(row) for row in csv.DictReader(handle)]
    _validate_midogpp_rows(rows)
    return rows


def read_candidate_manifest_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_midogpp_phase1_artifacts(
    artifacts_root: Path,
    *,
    rows: Sequence[MidogppDownstreamRow],
    candidate_manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    """Write phase-1 diagnostic matrix, summaries, and protocol reports.

    This function is report-only. It does not produce deployable selections and
    it never converts downstream utility into feature or selection artifacts.
    """

    root = Path(artifacts_root)
    matrix_path = midogpp_diagnostic_matrix_path(root)
    write_midogpp_diagnostic_matrix(matrix_path, rows)

    candidate_manifest_path = root / "tables" / "candidate_manifest.csv"
    if candidate_manifest_rows:
        columns = tuple(dict.fromkeys(key for row in candidate_manifest_rows for key in row))
        write_rows(candidate_manifest_path, columns, candidate_manifest_rows)

    oracle_rows = build_midogpp_oracle_summary(rows)
    oracle_path = root / "tables" / "candidate_oracle_summary.csv"
    write_rows(oracle_path, MIDOGPP_ORACLE_SUMMARY_COLUMNS, oracle_rows)

    baseline_rows = build_midogpp_baseline_comparison(rows, oracle_rows)
    baseline_path = root / "tables" / "baseline_comparison.csv"
    write_rows(baseline_path, MIDOGPP_BASELINE_COMPARISON_COLUMNS, baseline_rows)

    leakage_report = build_midogpp_phase1_leakage_report(
        rows=rows,
        candidate_manifest_rows=candidate_manifest_rows,
    )
    leakage_path = root / "reports" / "leakage_report.json"
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text(json.dumps(leakage_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decision_path = root / "reports" / "decision_summary.md"
    decision_path.write_text(_decision_summary_text(oracle_rows, baseline_rows), encoding="utf-8")
    return {
        "diagnostic_matrix": matrix_path,
        "candidate_manifest": candidate_manifest_path,
        "candidate_oracle_summary": oracle_path,
        "baseline_comparison": baseline_path,
        "leakage_report": leakage_path,
        "decision_summary": decision_path,
    }


def validate_midogpp_phase1_artifacts(
    artifacts_root: Path,
    *,
    expected_heldout_centers: Sequence[str] = (),
    expected_baseline_methods: Sequence[str] = (),
    require_preflight_reports: bool = False,
) -> dict[str, object]:
    """Validate a materialized MIDOG++ phase-1 artifact directory."""

    root = Path(artifacts_root)
    required_paths = {
        "diagnostic_matrix": midogpp_diagnostic_matrix_path(root),
        "diagnostic_schema": midogpp_diagnostic_matrix_path(root).with_suffix(".schema.json"),
        "candidate_manifest": root / "tables" / "candidate_manifest.csv",
        "candidate_oracle_summary": root / "tables" / "candidate_oracle_summary.csv",
        "baseline_comparison": root / "tables" / "baseline_comparison.csv",
        "leakage_report": root / "reports" / "leakage_report.json",
        "decision_summary": root / "reports" / "decision_summary.md",
    }
    if require_preflight_reports:
        required_paths["source_summary_preflight_report"] = root / "reports" / "source_summary_preflight_report.json"
        required_paths["run_hashes_report"] = root / "reports" / "run_hashes_report.json"
        required_paths["frozen_protocol_snapshot"] = root / "configs" / "frozen_protocol_snapshot.json"
        if expected_baseline_methods:
            required_paths["baseline_preflight_report"] = root / "reports" / "baseline_preflight_report.json"
    missing = [label for label, path in required_paths.items() if not path.exists()]
    if missing:
        raise ProtocolError(f"MIDOG++ phase-1 artifacts are missing required files: {missing}")

    rows = read_midogpp_diagnostic_matrix(required_paths["diagnostic_matrix"])
    candidate_manifest = read_candidate_manifest_rows(required_paths["candidate_manifest"])
    oracle_rows = _read_csv_rows(required_paths["candidate_oracle_summary"])
    baseline_rows = _read_csv_rows(required_paths["baseline_comparison"])
    leakage_report = _read_json(required_paths["leakage_report"])
    decision_text = required_paths["decision_summary"].read_text(encoding="utf-8")

    rebuilt_oracle = build_midogpp_oracle_summary(rows)
    rebuilt_baseline = build_midogpp_baseline_comparison(rows, rebuilt_oracle)
    rebuilt_leakage = build_midogpp_phase1_leakage_report(
        rows=rows,
        candidate_manifest_rows=candidate_manifest,
    )
    _assert_all_candidate_coverage(rows)
    _assert_all_single_source_rows_ok(rows)
    _assert_candidate_manifest_matches_matrix(rows=rows, candidate_manifest_rows=candidate_manifest)
    if len(oracle_rows) != len(rebuilt_oracle):
        raise ProtocolError(
            f"MIDOG++ oracle summary row count mismatch: {len(oracle_rows)} != {len(rebuilt_oracle)}"
        )
    _assert_csv_rows_match(
        observed=oracle_rows,
        expected=rebuilt_oracle,
        label="oracle summary",
    )
    if len(baseline_rows) != len(rebuilt_baseline):
        raise ProtocolError(
            f"MIDOG++ baseline comparison row count mismatch: {len(baseline_rows)} != {len(rebuilt_baseline)}"
        )
    _assert_csv_rows_match(
        observed=baseline_rows,
        expected=rebuilt_baseline,
        label="baseline comparison",
    )
    _assert_leakage_report_matches(leakage_report, rebuilt_leakage)
    _assert_expected_baseline_methods(rows, expected_baseline_methods)
    _assert_expected_baseline_methods_in_summary(baseline_rows, expected_baseline_methods)
    if "DIAGNOSTIC ONLY" not in decision_text:
        raise ProtocolError("MIDOG++ decision summary must state DIAGNOSTIC ONLY.")
    heldouts = sorted({row.heldout_center for row in rows})
    _assert_expected_heldout_centers(heldouts, expected_heldout_centers)
    preflight_status: dict[str, str] = {}
    if require_preflight_reports:
        preflight_status["source_summary"] = _assert_report_passes(required_paths["source_summary_preflight_report"])
        run_hashes_report = _assert_run_hashes_report(required_paths["run_hashes_report"], rows=rows)
        assert_frozen_snapshot_exists(required_paths["frozen_protocol_snapshot"])
        _assert_frozen_snapshot_matches_run_hashes(
            required_paths["frozen_protocol_snapshot"],
            run_hashes_report=run_hashes_report,
        )
        if expected_baseline_methods:
            preflight_status["baseline"] = _assert_baseline_preflight_report(required_paths["baseline_preflight_report"])
    return {
        "schema_version": "midogpp_phase1_validation_report_v1",
        "status": "PASS",
        "artifacts_root": str(root),
        "diagnostic_rows": len(rows),
        "candidate_manifest_rows": len(candidate_manifest),
        "oracle_summary_rows": len(oracle_rows),
        "baseline_comparison_rows": len(baseline_rows),
        "heldout_centers": heldouts,
        "expected_heldout_centers": [str(center) for center in expected_heldout_centers],
        "expected_baseline_methods": [str(method) for method in expected_baseline_methods],
        "preflight_status": preflight_status,
    }


def build_midogpp_oracle_summary(rows: Sequence[MidogppDownstreamRow]) -> list[dict[str, object]]:
    """Summarize diagnostic single-source oracle winners per context."""

    groups: dict[tuple[object, ...], list[MidogppDownstreamRow]] = {}
    for row in rows:
        if row.row_type != MIDOGPP_SINGLE_SOURCE_ROW_TYPE or row.status != "ok":
            continue
        groups.setdefault(_oracle_context_key(row), []).append(row)

    summaries: list[dict[str, object]] = []
    for _, group in sorted(groups.items(), key=lambda item: item[0]):
        winner = max(group, key=lambda row: (float(row.bacc), float(row.macro_f1), _reverse_lex(row.candidate_id)))
        baccs = [float(row.bacc) for row in group]
        template = group[0]
        summaries.append(
            {
                "dataset": template.dataset,
                "domain_regime": template.domain_regime,
                "heldout_center": template.heldout_center,
                "experiment_seed": template.experiment_seed,
                "replicate_seed": template.replicate_seed,
                "support_size": template.support_size,
                "support_seed": template.support_seed,
                "support_set_id": template.support_set_id,
                "eval_set_id": template.eval_set_id,
                "generation_seed": template.generation_seed,
                "classifier_seed": template.classifier_seed,
                "synthetic_per_class_total": template.synthetic_per_class_total,
                "config_hash": template.config_hash,
                "protocol_hash": template.protocol_hash,
                "feature_frame_hash": template.feature_frame_hash,
                "oracle_candidate_id": winner.candidate_id,
                "oracle_candidate_source_center": winner.candidate_source_center,
                "oracle_candidate_method": winner.candidate_method,
                "oracle_bacc": winner.bacc,
                "oracle_macro_f1": winner.macro_f1,
                "mean_single_source_bacc": _mean(baccs),
                "min_single_source_bacc": min(baccs),
                "max_single_source_bacc": max(baccs),
                "spread_max_minus_mean_bacc": max(baccs) - _mean(baccs),
                "spread_max_minus_min_bacc": max(baccs) - min(baccs),
                "n_ok_single_source_candidates": len(group),
                "claim_role": "oracle_diagnostic",
            }
        )
    return summaries


def build_midogpp_baseline_comparison(
    rows: Sequence[MidogppDownstreamRow],
    oracle_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare method baselines to single-source diagnostic oracle rows."""

    oracle_by_context = {_baseline_context_key(row): row for row in oracle_rows}
    grouped: dict[tuple[str, str], list[tuple[MidogppDownstreamRow, Mapping[str, object]]]] = {}
    for row in rows:
        if row.row_type != MIDOGPP_METHOD_BASELINE_ROW_TYPE or row.status != "ok":
            continue
        oracle = oracle_by_context.get(_baseline_context_key(row))
        if oracle is None:
            continue
        grouped.setdefault((row.candidate_method, row.heldout_center), []).append((row, oracle))

    comparison: list[dict[str, object]] = []
    for (method, heldout), pairs in sorted(grouped.items()):
        baccs = [float(row.bacc) for row, _ in pairs]
        macro_f1s = [float(row.macro_f1) for row, _ in pairs]
        gaps_bacc = [float(oracle["oracle_bacc"]) - float(row.bacc) for row, oracle in pairs]
        gaps_f1 = [float(oracle["oracle_macro_f1"]) - float(row.macro_f1) for row, oracle in pairs]
        comparison.append(
            {
                "baseline_method": method,
                "heldout_center": heldout,
                "mean_bacc": _mean(baccs),
                "mean_macro_f1": _mean(macro_f1s),
                "mean_oracle_gap_bacc": _mean(gaps_bacc),
                "mean_oracle_gap_macro_f1": _mean(gaps_f1),
                "n_rows": len(pairs),
                "claim_role": "baseline_diagnostic",
            }
        )
    return comparison


def build_midogpp_phase1_leakage_report(
    *,
    rows: Sequence[MidogppDownstreamRow],
    candidate_manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a phase-1 leakage report for diagnostic-only MIDOG++ artifacts."""

    _validate_midogpp_rows(rows)
    for heldout in sorted({row.heldout_center for row in rows}):
        assert_midogpp_candidate_pool(
            heldout_center=heldout,
            candidate_rows=[
                row
                for row in candidate_manifest_rows
                if str(row.get("heldout_center", heldout)) == heldout
            ],
        )
    report = {
        "schema_version": "midogpp_phase1_leakage_report_v1",
        "support_eval_overlap": False,
        "target_expert_in_candidate_pool": False,
        "selection_read_downstream_matrix": False,
        "selection_used_target_eval_labels": False,
        "target_eval_metric_used_in_selection": False,
        "classifier_tuned_on_target_eval": False,
        "generation_tuned_on_target_eval": False,
        "feature_normalization_used_target_eval": False,
        "generation_settings_frozen_before_eval": True,
        "classifier_settings_frozen_before_eval": True,
        "diagnostic_rows": len(rows),
        "candidate_manifest_rows": len(candidate_manifest_rows),
        "all_rows_diagnostic_only": all(row.eligibility == DIAGNOSTIC_ONLY for row in rows),
        "all_rows_oracle_diagnostic": all(row.claim_role == "oracle_diagnostic" for row in rows),
        "selection_rows_written": 0,
    }
    if not report["all_rows_diagnostic_only"] or not report["all_rows_oracle_diagnostic"]:
        raise ProtocolError("MIDOG++ phase-1 rows must remain diagnostic-only oracle evidence.")
    return report


def _validate_midogpp_rows(rows: Sequence[MidogppDownstreamRow]) -> None:
    if not rows:
        raise ProtocolError("MIDOG++ diagnostic matrix is empty.")
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = row.primary_key()
        if key in seen:
            raise ProtocolError(f"Duplicate MIDOG++ downstream row: {key}")
        seen.add(key)
        assert_midogpp_candidate_pool(
            heldout_center=row.heldout_center,
            candidate_rows=[
                {
                    "candidate_source_center": row.candidate_source_center,
                    "eligibility": DIAGNOSTIC_ONLY,
                }
            ],
        )


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed MIDOG++ JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"MIDOG++ JSON artifact must be an object: {path}")
    return payload


def _assert_leakage_report_matches(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    required = (
        "schema_version",
        "support_eval_overlap",
        "target_expert_in_candidate_pool",
        "selection_read_downstream_matrix",
        "selection_used_target_eval_labels",
        "target_eval_metric_used_in_selection",
        "classifier_tuned_on_target_eval",
        "generation_tuned_on_target_eval",
        "feature_normalization_used_target_eval",
        "generation_settings_frozen_before_eval",
        "classifier_settings_frozen_before_eval",
        "diagnostic_rows",
        "candidate_manifest_rows",
        "all_rows_diagnostic_only",
        "all_rows_oracle_diagnostic",
        "selection_rows_written",
    )
    mismatched = [key for key in required if observed.get(key) != expected.get(key)]
    if mismatched:
        raise ProtocolError(f"MIDOG++ leakage report mismatch for fields: {mismatched}")


def _assert_candidate_manifest_matches_matrix(
    *,
    rows: Sequence[MidogppDownstreamRow],
    candidate_manifest_rows: Sequence[Mapping[str, object]],
) -> None:
    expected = {
        (row.heldout_center, row.candidate_source_center, row.candidate_id)
        for row in rows
        if row.row_type == MIDOGPP_SINGLE_SOURCE_ROW_TYPE
    }
    observed = {
        (
            str(row.get("heldout_center", "")),
            str(row.get("candidate_source_center", row.get("source_center", ""))),
            str(row.get("candidate_id", "")),
        )
        for row in candidate_manifest_rows
        if str(row.get("eligibility", SELECTION_ELIGIBLE)) == SELECTION_ELIGIBLE
    }
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ProtocolError(
            "MIDOG++ candidate manifest does not match single-source diagnostic rows; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def _assert_all_candidate_coverage(rows: Sequence[MidogppDownstreamRow]) -> None:
    observed_by_heldout: dict[str, set[str]] = {}
    for row in rows:
        if row.row_type != MIDOGPP_SINGLE_SOURCE_ROW_TYPE:
            continue
        observed_by_heldout.setdefault(row.heldout_center, set()).add(row.candidate_source_center)
    for heldout, observed in sorted(observed_by_heldout.items()):
        expected = set(MIDOGPP_ELIGIBLE_CENTERS).difference({heldout})
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise ProtocolError(
                "MIDOG++ phase-1 matrix must score every eligible non-heldout source; "
                f"heldout={heldout}, missing={missing}, extra={extra}"
            )


def _assert_all_single_source_rows_ok(rows: Sequence[MidogppDownstreamRow]) -> None:
    failed = [
        (row.heldout_center, row.candidate_source_center, row.candidate_id, row.status)
        for row in rows
        if row.row_type == MIDOGPP_SINGLE_SOURCE_ROW_TYPE and row.status != "ok"
    ]
    if failed:
        raise ProtocolError(
            "MIDOG++ phase-1 thesis-facing validation requires every single-source candidate "
            f"to score successfully; failed_rows={failed[:10]}"
        )


def _assert_expected_heldout_centers(
    observed_heldouts: Sequence[str],
    expected_heldout_centers: Sequence[str],
) -> None:
    expected = {str(center) for center in expected_heldout_centers}
    if not expected:
        return
    observed = {str(center) for center in observed_heldouts}
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ProtocolError(
            "MIDOG++ phase-1 matrix heldout coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )


def _assert_csv_rows_match(
    *,
    observed: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
    label: str,
) -> None:
    observed_canonical = [_canonical_csv_row(row) for row in observed]
    expected_canonical = [_canonical_csv_row(row) for row in expected]
    if observed_canonical != expected_canonical:
        raise ProtocolError(f"MIDOG++ {label} contents do not match diagnostic matrix rebuild.")


def _canonical_csv_row(row: Mapping[str, object]) -> dict[str, str]:
    return {
        str(key): _canonical_csv_value(value)
        for key, value in row.items()
    }


def _canonical_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return str(value)
    return str(value)


def _assert_expected_baseline_methods(
    rows: Sequence[MidogppDownstreamRow],
    expected_baseline_methods: Sequence[str],
) -> None:
    observed = {
        row.candidate_method
        for row in rows
        if row.row_type == MIDOGPP_METHOD_BASELINE_ROW_TYPE and row.status == "ok"
    }
    missing = sorted(set(str(method) for method in expected_baseline_methods).difference(observed))
    if missing:
        raise ProtocolError(f"MIDOG++ expected baseline methods are missing from diagnostic rows: {missing}")


def _assert_expected_baseline_methods_in_summary(
    rows: Sequence[Mapping[str, object]],
    expected_baseline_methods: Sequence[str],
) -> None:
    observed = {str(row.get("baseline_method", "")) for row in rows}
    missing = sorted(set(str(method) for method in expected_baseline_methods).difference(observed))
    if missing:
        raise ProtocolError(f"MIDOG++ expected baseline methods are missing from baseline summary: {missing}")


def _assert_report_passes(path: Path) -> str:
    report = _read_json(path)
    status = str(report.get("status", ""))
    if status != "PASS":
        raise ProtocolError(f"MIDOG++ preflight report did not PASS: {path}")
    if report.get("schema_version") != "midogpp_source_summary_preflight_report_v1":
        raise ProtocolError(f"MIDOG++ source-summary preflight report schema mismatch: {path}")
    return status


def _assert_run_hashes_report(path: Path, *, rows: Sequence[MidogppDownstreamRow]) -> Mapping[str, object]:
    report = _read_json(path)
    required = {
        "cache_file_hashes",
        "config_hash",
        "feature_frame_hash",
        "protocol_hash",
        "schema_version",
        "snapshot",
        "source_summary_file_hashes",
        "summary_manifest_hash",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise ProtocolError(f"MIDOG++ run hashes report missing fields: {missing}")
    if report.get("schema_version") != "midogpp_phase1_run_hashes_v1":
        raise ProtocolError(f"MIDOG++ run hashes report schema mismatch: {path}")
    _assert_row_hashes_match_run_hashes(report=report, rows=rows, path=path)
    return report


def _assert_frozen_snapshot_matches_run_hashes(
    path: Path,
    *,
    run_hashes_report: Mapping[str, object],
) -> None:
    snapshot = _read_json(path)
    expected_snapshot = run_hashes_report.get("snapshot")
    if not isinstance(expected_snapshot, Mapping):
        raise ProtocolError("MIDOG++ run hashes report snapshot must be an object.")
    component_keys = (
        "candidate_pool_hash",
        "generation_config_hash",
        "classifier_config_hash",
        "metric_config_hash",
        "feature_config_hash",
        "routing_config_hash",
    )
    mismatched = [
        key
        for key in component_keys
        if str(snapshot.get(key, "")) != str(expected_snapshot.get(key, ""))
    ]
    if mismatched:
        raise ProtocolError(
            f"MIDOG++ frozen protocol snapshot does not match run_hashes_report: {mismatched}"
        )
    if str(snapshot.get("protocol_hash", "")) != str(run_hashes_report.get("protocol_hash", "")):
        raise ProtocolError("MIDOG++ frozen protocol snapshot protocol_hash does not match run_hashes_report.")
    if str(snapshot.get("feature_config_hash", "")) != str(run_hashes_report.get("feature_frame_hash", "")):
        raise ProtocolError("MIDOG++ frozen protocol snapshot feature_config_hash does not match feature_frame_hash.")


def _assert_row_hashes_match_run_hashes(
    *,
    report: Mapping[str, object],
    rows: Sequence[MidogppDownstreamRow],
    path: Path,
) -> None:
    expected = {
        "config_hash": str(report["config_hash"]),
        "protocol_hash": str(report["protocol_hash"]),
        "feature_frame_hash": str(report["feature_frame_hash"]),
    }
    mismatches: list[tuple[str, str, str, str]] = []
    for row in rows:
        observed = {
            "config_hash": row.config_hash,
            "protocol_hash": row.protocol_hash,
            "feature_frame_hash": row.feature_frame_hash,
        }
        for field, expected_value in expected.items():
            observed_value = str(observed[field])
            if observed_value != expected_value:
                mismatches.append((row.heldout_center, row.candidate_id, field, observed_value))
    if mismatches:
        raise ProtocolError(
            "MIDOG++ diagnostic row hashes do not match run_hashes_report; "
            f"path={path}, expected={expected}, mismatches={mismatches[:10]}"
        )


def _assert_baseline_preflight_report(path: Path) -> str:
    report = _read_json(path)
    status = str(report.get("status", ""))
    if status != "PASS":
        raise ProtocolError(f"MIDOG++ baseline preflight report did not PASS: {path}")
    if report.get("schema_version") != "midogpp_baseline_preflight_report_v1":
        raise ProtocolError(f"MIDOG++ baseline preflight report schema mismatch: {path}")
    required = {"baseline_matrix_hashes", "baseline_row_hashes"}
    missing = sorted(required.difference(report))
    if missing:
        raise ProtocolError(f"MIDOG++ baseline preflight report missing fields: {missing}")
    return status


def _oracle_context_key(row: MidogppDownstreamRow) -> tuple[object, ...]:
    return (
        row.dataset,
        row.domain_regime,
        row.heldout_center,
        row.experiment_seed,
        row.replicate_seed,
        row.support_size,
        row.support_seed,
        row.support_set_id,
        row.eval_set_id,
        row.generation_seed,
        row.classifier_seed,
        row.synthetic_per_class_total,
        row.config_hash,
        row.protocol_hash,
        row.feature_frame_hash,
    )


def _baseline_context_key(row: MidogppDownstreamRow | Mapping[str, object]) -> tuple[object, ...]:
    if isinstance(row, MidogppDownstreamRow):
        return _oracle_context_key(row)
    return (
        row["dataset"],
        row["domain_regime"],
        row["heldout_center"],
        row["experiment_seed"],
        row["replicate_seed"],
        row["support_size"],
        row["support_seed"],
        row["support_set_id"],
        row["eval_set_id"],
        row["generation_seed"],
        row["classifier_seed"],
        row["synthetic_per_class_total"],
        row["config_hash"],
        row["protocol_hash"],
        row["feature_frame_hash"],
    )


def _write_midogpp_schema(path: Path) -> None:
    payload = {
        "schema_version": MIDOGPP_MATRIX_SCHEMA_VERSION,
        "dataset": MIDOGPP_DATASET_NAME,
        "primary_key": list(MIDOGPP_DOWNSTREAM_PRIMARY_KEY),
        "oracle_eligible_filter": {
            "claim_role": "oracle_diagnostic",
            "eligibility": DIAGNOSTIC_ONLY,
            "row_type": MIDOGPP_SINGLE_SOURCE_ROW_TYPE,
            "status": "ok",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_midogpp_schema(path: Path) -> None:
    if not path.exists():
        raise ProtocolError(f"Missing MIDOG++ diagnostic matrix schema sidecar: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed MIDOG++ diagnostic matrix schema sidecar: {path}") from exc
    if payload.get("schema_version") != MIDOGPP_MATRIX_SCHEMA_VERSION:
        raise ProtocolError("MIDOG++ diagnostic matrix schema_version mismatch.")
    if list(payload.get("primary_key") or []) != list(MIDOGPP_DOWNSTREAM_PRIMARY_KEY):
        raise ProtocolError("MIDOG++ diagnostic matrix primary_key mismatch.")


def _center_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10_000, value


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if not math.isnan(float(value))]
    return sum(finite) / float(len(finite)) if finite else math.nan


def _reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(ch)) for ch in str(value))


def _decision_summary_text(
    oracle_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# MIDOG++ Phase-1 Diagnostic Summary",
        "",
        "Decision: DIAGNOSTIC ONLY",
        "",
        "This artifact reports held-out downstream utility observations only. It does not claim learned routing improves BACC.",
        "",
        f"- Oracle contexts: {len(oracle_rows)}",
        f"- Baseline comparison rows: {len(baseline_rows)}",
        "- Claim boundary: candidate-level downstream utility variation was/was not present under the annotation-patch regime.",
        "",
    ]
    return "\n".join(lines)
