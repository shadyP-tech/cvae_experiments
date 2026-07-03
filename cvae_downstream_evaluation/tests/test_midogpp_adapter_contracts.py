from pathlib import Path
import csv
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.adapters.midogpp import (  # noqa: E402
    build_midogpp_baseline_comparison,
    build_candidate_manifest_from_source_summary,
    build_midogpp_oracle_summary,
    build_midogpp_phase1_leakage_report,
    midogpp_diagnostic_matrix_path,
    read_midogpp_diagnostic_matrix,
    read_midogpp_scored_rows,
    validate_midogpp_phase1_artifacts,
    write_midogpp_phase1_artifacts,
    write_midogpp_diagnostic_matrix,
)
from cvae_downstream_evaluation.adapters.midogpp_runner import (  # noqa: E402
    MidogppCandidate,
    MidogppRunContext,
    MidogppScoringResult,
    run_midogpp_phase1_scoring,
    score_midogpp_baseline,
    score_midogpp_candidate,
    select_midogpp_source_inner_classifier_spec,
)
from cvae_downstream_evaluation.adapters.midogpp_source_summary_backend import (  # noqa: E402
    SourceSummaryMidogppBackend,
    build_midogpp_phase1_run_hashes,
    load_midogpp_feature_cache,
    preflight_midogpp_external_baselines,
    preflight_midogpp_source_summary_inputs,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.classifiers import ClassifierSpec  # noqa: E402
from cvae_downstream_evaluation.schemas import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import (  # noqa: E402
    MIDOGPP_DOWNSTREAM_PRIMARY_KEY,
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_METHOD_BASELINE_ROW_TYPE,
    MidogppDownstreamRow,
    assert_midogpp_frozen_config_file,
    assert_midogpp_frozen_config_text,
    assert_midogpp_candidate_pool,
    assert_midogpp_feature_table,
    canonical_support_context,
    midogpp_deployable_feature_columns,
)
from cvae_downstream_evaluation.utility_matrix import assert_selection_does_not_read_matrix  # noqa: E402


def test_midogpp_candidate_manifest_is_manifest_driven_and_excludes_target_and_domain4() -> None:
    rows = [
        _source_summary(source_center="0", class_label=0),
        _source_summary(source_center="0", class_label=1),
        _source_summary(source_center="1", class_label=0),
        _source_summary(source_center="4", class_label=0),
        _source_summary(source_center="5", class_label=0, status="failed"),
    ]

    candidates = build_candidate_manifest_from_source_summary(rows, heldout_center="0")

    assert [row["candidate_source_center"] for row in candidates] == ["1"]
    assert candidates[0]["candidate_id"] == "midogpp_source_1_single_source_adaptive_k"
    assert candidates[0]["eligibility"] == SELECTION_ELIGIBLE


def test_midogpp_frozen_config_declares_protocol_guards() -> None:
    path = ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml"
    assert_midogpp_frozen_config_file(path)
    text = path.read_text(encoding="utf-8")
    assert "source_summary_manifest" in text
    assert "diagnostic_downstream_utility.csv" in text

    try:
        assert_midogpp_frozen_config_text(text.replace("support_labels_used: false", "support_labels_used: true"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ config accepted support-label misuse")

    try:
        assert_midogpp_frozen_config_text(text.replace("dataset: midogpp", "dataset: camelyon17"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ config accepted stale Camelyon17 dataset")


def test_midogpp_phase1_runbook_matches_script_and_protocol_contracts() -> None:
    text = (ROOT / "docs" / "midogpp_phase1_runbook.md").read_text(encoding="utf-8")

    required = (
        "run_midogpp_source_summary_phase1.py",
        "validate_midogpp_phase1_artifacts.py",
        "virchow2_midogpp_all_candidates.yaml",
        "--preflight-only",
        "--expected-heldout-center 9",
        "--require-preflight-reports",
        "diagnostic_downstream_utility.csv",
        "phase1_validation_report.json",
        "run_hashes_report.json",
        "frozen_protocol_snapshot.json",
        "summary_manifest_hash",
        "source_summary_file_hashes",
        "cache_file_hashes",
        "baseline_matrix_hashes",
        "baseline_row_hashes",
        "support_size=0",
        "center `4` remains excluded",
        "real_source_embedding_classifier_dense_reference",
    )
    for snippet in required:
        assert snippet in text


def test_midogpp_candidate_pool_rejects_deployable_target_and_domain4() -> None:
    assert_midogpp_candidate_pool(
        heldout_center="2",
        candidate_rows=[
            {"candidate_source_center": "0", "eligibility": SELECTION_ELIGIBLE},
            {"candidate_source_center": "2", "eligibility": DIAGNOSTIC_ONLY},
        ],
    )

    try:
        assert_midogpp_candidate_pool(
            heldout_center="2",
            candidate_rows=[{"candidate_source_center": "2", "eligibility": SELECTION_ELIGIBLE}],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("selection-eligible heldout target was not rejected")

    try:
        assert_midogpp_candidate_pool(
            heldout_center="2",
            candidate_rows=[{"candidate_source_center": "4", "eligibility": DIAGNOSTIC_ONLY}],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("domain4 quarantine violation was not rejected")


def test_midogpp_support_context_sentinels_are_canonical() -> None:
    assert canonical_support_context(
        support_size=0,
        support_seed="none",
        support_set_id="none",
    ) == (0, "none", "none")
    assert canonical_support_context(
        support_size=8,
        support_seed=17,
        support_set_id="support-8-17",
    ) == (8, "17", "support-8-17")

    try:
        canonical_support_context(support_size=0, support_seed=17, support_set_id="support-0-17")
    except ProtocolError:
        pass
    else:
        raise AssertionError("non-canonical no-support sentinel was not rejected")


def test_midogpp_diagnostic_matrix_roundtrip_and_schema_key(tmp_path: Path) -> None:
    path = midogpp_diagnostic_matrix_path(tmp_path)
    rows = [_matrix_row(candidate_source_center="1"), _matrix_row(candidate_source_center="3")]

    write_midogpp_diagnostic_matrix(path, rows)
    loaded = read_midogpp_diagnostic_matrix(path)

    assert path.name == "diagnostic_downstream_utility.csv"
    assert [row.primary_key() for row in loaded] == [row.primary_key() for row in rows]
    schema_text = path.with_suffix(".schema.json").read_text(encoding="utf-8")
    assert "midogpp_all_candidate_downstream_matrix_v1" in schema_text
    assert "support_set_id" in schema_text


def test_midogpp_scored_rows_reader_accepts_presidecar_csv(tmp_path: Path) -> None:
    path = tmp_path / "prescored_midogpp_rows.csv"
    write_midogpp_diagnostic_matrix(tmp_path / "tables" / "diagnostic_downstream_utility.csv", [_matrix_row()])
    canonical = tmp_path / "tables" / "diagnostic_downstream_utility.csv"
    path.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")

    rows = read_midogpp_scored_rows(path)

    assert len(rows) == 1
    assert rows[0].candidate_source_center == "1"


def test_midogpp_phase1_artifacts_write_diagnostic_tables_and_reports(tmp_path: Path) -> None:
    rows = [
        _matrix_row(candidate_source_center="1", bacc=0.61, macro_f1=0.59),
        _matrix_row(candidate_source_center="3", bacc=0.72, macro_f1=0.70),
        _matrix_row(
            candidate_source_center="__dense__",
            candidate_id="dense_late_equal_all_sources_geom",
            candidate_method="dense_late_equal_all_sources_geom",
            row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
            bacc=0.68,
            macro_f1=0.66,
        ),
    ]
    candidate_manifest = [
        {
            "heldout_center": "0",
            "candidate_source_center": "1",
            "candidate_id": "midogpp_source_1_single_source_adaptive_k",
            "eligibility": SELECTION_ELIGIBLE,
        },
        {
            "heldout_center": "0",
            "candidate_source_center": "3",
            "candidate_id": "midogpp_source_3_single_source_adaptive_k",
            "eligibility": SELECTION_ELIGIBLE,
        },
    ]

    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)

    assert paths["diagnostic_matrix"].name == "diagnostic_downstream_utility.csv"
    assert paths["candidate_oracle_summary"].read_text(encoding="utf-8").count("\n") == 2
    assert "dense_late_equal_all_sources_geom" in paths["baseline_comparison"].read_text(encoding="utf-8")
    leakage = paths["leakage_report"].read_text(encoding="utf-8")
    assert '"selection_rows_written": 0' in leakage
    assert "DIAGNOSTIC ONLY" in paths["decision_summary"].read_text(encoding="utf-8")
    assert read_midogpp_diagnostic_matrix(paths["diagnostic_matrix"])[0].claim_role == "oracle_diagnostic"


def test_midogpp_phase1_artifact_validator_passes_and_writes_report(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )
    baseline_path = _write_external_baseline_fixture(tmp_path)
    baseline_preflight = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(baseline_path,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )
    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path,
    )
    rows = [
        *_full_single_source_rows(
            config_hash=run_hashes.config_hash,
            protocol_hash=run_hashes.protocol_hash,
            feature_frame_hash=run_hashes.feature_frame_hash,
        ),
        _matrix_row(
            candidate_source_center="__dense__",
            candidate_id="dense_late_equal_all_sources_geom",
            candidate_method="dense_late_equal_all_sources_geom",
            row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
            bacc=0.68,
            macro_f1=0.66,
            config_hash=run_hashes.config_hash,
            protocol_hash=run_hashes.protocol_hash,
            feature_frame_hash=run_hashes.feature_frame_hash,
        ),
    ]
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    reports_dir = tmp_path / "reports"
    (reports_dir / "source_summary_preflight_report.json").write_text(
        json.dumps(preflight.to_report()),
        encoding="utf-8",
    )
    (reports_dir / "baseline_preflight_report.json").write_text(
        json.dumps(baseline_preflight.to_report()),
        encoding="utf-8",
    )
    (reports_dir / "run_hashes_report.json").write_text(
        json.dumps(run_hashes.to_report()),
        encoding="utf-8",
    )

    report = validate_midogpp_phase1_artifacts(
        tmp_path,
        expected_heldout_centers=("0",),
        expected_baseline_methods=("dense_late_equal_all_sources_geom",),
        require_preflight_reports=True,
    )

    assert report["status"] == "PASS"
    assert report["diagnostic_rows"] == 9
    assert preflight.to_report()["schema_version"] == "midogpp_source_summary_preflight_report_v1"
    assert baseline_preflight.to_report()["schema_version"] == "midogpp_baseline_preflight_report_v1"

    script = ROOT / "scripts" / "validate_midogpp_phase1_artifacts.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifacts-root",
            str(tmp_path),
            "--expected-heldout-center",
            "0",
            "--expected-baseline-method",
            "dense_late_equal_all_sources_geom",
            "--require-preflight-reports",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    validation_report = tmp_path / "reports" / "phase1_validation_report.json"
    assert json.loads(validation_report.read_text(encoding="utf-8"))["status"] == "PASS"


def test_midogpp_phase1_validation_cli_failure_writes_report(tmp_path: Path) -> None:
    rows = _full_single_source_rows()
    candidate_manifest = _candidate_manifest_from_rows(rows)
    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    paths["decision_summary"].write_text("not diagnostic enough\n", encoding="utf-8")
    script = ROOT / "scripts" / "validate_midogpp_phase1_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifacts-root",
            str(tmp_path),
            "--expected-heldout-center",
            "0",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    validation_report = tmp_path / "reports" / "phase1_validation_report.json"
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert report["status"] == "FAIL"
    assert "DIAGNOSTIC ONLY" in report["error_message"]


def test_midogpp_phase1_artifact_validator_rejects_stale_preflight_schema(tmp_path: Path) -> None:
    rows = _full_single_source_rows()
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )
    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path,
    )
    source_report = preflight.to_report()
    source_report["schema_version"] = "old_preflight_schema"
    reports_dir = tmp_path / "reports"
    (reports_dir / "source_summary_preflight_report.json").write_text(
        json.dumps(source_report),
        encoding="utf-8",
    )
    (reports_dir / "run_hashes_report.json").write_text(
        json.dumps(run_hashes.to_report()),
        encoding="utf-8",
    )

    try:
        validate_midogpp_phase1_artifacts(tmp_path, require_preflight_reports=True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("stale source-summary preflight schema was accepted")


def test_midogpp_phase1_artifact_validator_rejects_run_hash_row_mismatch(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )
    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path,
    )
    rows = _full_single_source_rows(
        config_hash=run_hashes.config_hash,
        protocol_hash=run_hashes.protocol_hash,
        feature_frame_hash=run_hashes.feature_frame_hash,
    )
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    reports_dir = tmp_path / "reports"
    (reports_dir / "source_summary_preflight_report.json").write_text(
        json.dumps(preflight.to_report()),
        encoding="utf-8",
    )
    run_hash_report = run_hashes.to_report()
    run_hash_report["config_hash"] = "stale-config-hash"
    (reports_dir / "run_hashes_report.json").write_text(
        json.dumps(run_hash_report),
        encoding="utf-8",
    )

    try:
        validate_midogpp_phase1_artifacts(tmp_path, require_preflight_reports=True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("run_hashes_report mismatch with diagnostic rows was accepted")


def test_midogpp_phase1_artifact_validator_rejects_stale_frozen_snapshot(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )
    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path,
    )
    rows = _full_single_source_rows(
        config_hash=run_hashes.config_hash,
        protocol_hash=run_hashes.protocol_hash,
        feature_frame_hash=run_hashes.feature_frame_hash,
    )
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    reports_dir = tmp_path / "reports"
    (reports_dir / "source_summary_preflight_report.json").write_text(
        json.dumps(preflight.to_report()),
        encoding="utf-8",
    )
    (reports_dir / "run_hashes_report.json").write_text(
        json.dumps(run_hashes.to_report()),
        encoding="utf-8",
    )
    snapshot = json.loads(run_hashes.frozen_snapshot_path.read_text(encoding="utf-8"))
    snapshot["generation_config_hash"] = "stale-generation-hash"
    run_hashes.frozen_snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    try:
        validate_midogpp_phase1_artifacts(tmp_path, require_preflight_reports=True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("stale frozen protocol snapshot was accepted")


def test_midogpp_phase1_artifact_validator_rejects_missing_expected_heldout(tmp_path: Path) -> None:
    rows = _full_single_source_rows(heldout_center="0")
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)

    try:
        validate_midogpp_phase1_artifacts(tmp_path, expected_heldout_centers=("0", "1"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("missing expected MIDOG++ heldout center was accepted")


def test_midogpp_phase1_artifact_validator_rejects_corrupt_leakage_report(tmp_path: Path) -> None:
    rows = _full_single_source_rows()
    candidate_manifest = _candidate_manifest_from_rows(rows)
    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    leakage = json.loads(paths["leakage_report"].read_text(encoding="utf-8"))
    leakage["selection_rows_written"] = 1
    paths["leakage_report"].write_text(json.dumps(leakage), encoding="utf-8")

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("corrupt MIDOG++ leakage report was accepted")


def test_midogpp_phase1_artifact_validator_rejects_corrupt_candidate_manifest(tmp_path: Path) -> None:
    rows = _full_single_source_rows()
    candidate_manifest = _candidate_manifest_from_rows(rows)
    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    with paths["candidate_manifest"].open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_rows[0]["candidate_id"] = "midogpp_source_X_single_source_adaptive_k"
    with paths["candidate_manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("corrupt MIDOG++ candidate manifest was accepted")


def test_midogpp_phase1_artifact_validator_rejects_incomplete_candidate_coverage(tmp_path: Path) -> None:
    rows = [
        row
        for row in _full_single_source_rows()
        if row.candidate_source_center != "9"
    ]
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("incomplete MIDOG++ all-candidate coverage was accepted")


def test_midogpp_phase1_artifact_validator_rejects_failed_single_source_row(tmp_path: Path) -> None:
    rows = [
        _matrix_row(
            candidate_source_center=row.candidate_source_center,
            candidate_id=row.candidate_id,
            bacc=row.bacc,
            macro_f1=row.macro_f1,
            status="failed_empty_reference_pool",
            error_message="empty reference pool",
        )
        if row.candidate_source_center == "9"
        else row
        for row in _full_single_source_rows()
    ]
    candidate_manifest = _candidate_manifest_from_rows(rows)
    write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("failed MIDOG++ single-source row was accepted")


def test_midogpp_phase1_artifact_validator_rejects_corrupt_oracle_summary(tmp_path: Path) -> None:
    rows = _full_single_source_rows()
    candidate_manifest = _candidate_manifest_from_rows(rows)
    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    text = paths["candidate_oracle_summary"].read_text(encoding="utf-8")
    paths["candidate_oracle_summary"].write_text(text.replace("midogpp_source_9", "midogpp_source_X"), encoding="utf-8")

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("corrupt MIDOG++ oracle summary was accepted")


def test_midogpp_phase1_artifact_validator_rejects_corrupt_baseline_summary(tmp_path: Path) -> None:
    rows = [
        *_full_single_source_rows(),
        _matrix_row(
            candidate_source_center="__dense__",
            candidate_id="dense_late_equal_all_sources_geom",
            candidate_method="dense_late_equal_all_sources_geom",
            row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
            bacc=0.68,
            macro_f1=0.66,
        ),
    ]
    candidate_manifest = _candidate_manifest_from_rows(rows)
    paths = write_midogpp_phase1_artifacts(tmp_path, rows=rows, candidate_manifest_rows=candidate_manifest)
    with paths["baseline_comparison"].open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    baseline_rows[0]["mean_bacc"] = "0.12"
    with paths["baseline_comparison"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)

    try:
        validate_midogpp_phase1_artifacts(tmp_path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("corrupt MIDOG++ baseline summary was accepted")


def test_midogpp_oracle_summary_and_baselines_keep_dense_outside_single_source_oracle() -> None:
    rows = [
        _matrix_row(candidate_source_center="1", bacc=0.61, macro_f1=0.59),
        _matrix_row(candidate_source_center="3", bacc=0.72, macro_f1=0.70),
        _matrix_row(
            candidate_source_center="__dense__",
            candidate_id="dense_late_equal_all_sources_geom",
            candidate_method="dense_late_equal_all_sources_geom",
            row_type=MIDOGPP_METHOD_BASELINE_ROW_TYPE,
            bacc=0.95,
            macro_f1=0.94,
        ),
    ]

    oracle = build_midogpp_oracle_summary(rows)
    baseline = build_midogpp_baseline_comparison(rows, oracle)

    assert len(oracle) == 1
    assert oracle[0]["oracle_candidate_source_center"] == "3"
    assert oracle[0]["spread_max_minus_min_bacc"] == 0.10999999999999999
    assert baseline[0]["baseline_method"] == "dense_late_equal_all_sources_geom"
    assert baseline[0]["mean_oracle_gap_bacc"] < 0.0


def test_midogpp_phase1_leakage_report_rejects_bad_candidate_manifest() -> None:
    rows = [_matrix_row(candidate_source_center="1")]
    try:
        build_midogpp_phase1_leakage_report(
            rows=rows,
            candidate_manifest_rows=[
                {
                    "heldout_center": "0",
                    "candidate_source_center": "0",
                    "eligibility": SELECTION_ELIGIBLE,
                }
            ],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("phase-1 leakage report accepted heldout target in candidate manifest")


def test_midogpp_scoring_runner_scores_candidates_and_writes_artifacts(tmp_path: Path) -> None:
    context = _run_context()
    candidates = [
        MidogppCandidate(
            candidate_source_center="1",
            candidate_id="midogpp_source_1_single_source_adaptive_k",
            checkpoint_hash="checkpoint-1",
        ),
        MidogppCandidate(
            candidate_source_center="3",
            candidate_id="midogpp_source_3_single_source_adaptive_k",
            checkpoint_hash="checkpoint-3",
        ),
    ]

    outputs = run_midogpp_phase1_scoring(
        backend=_FakeMidogppBackend(),
        contexts=[context],
        candidates_by_heldout={"0": candidates},
        artifacts_root=tmp_path,
        baseline_methods=("dense_late_equal_all_sources_geom",),
    )

    rows = read_midogpp_diagnostic_matrix(outputs["diagnostic_matrix"])
    assert len(rows) == 3
    assert all(row.claim_role == "oracle_diagnostic" for row in rows)
    assert all(row.selection_used_target_labels is False for row in rows)
    assert "dense_late_equal_all_sources_geom" in outputs["baseline_comparison"].read_text(encoding="utf-8")


def test_midogpp_source_inner_classifier_selection_excludes_outer_target() -> None:
    backend = _TrackingMidogppBackend()
    context = _run_context()

    selection = select_midogpp_source_inner_classifier_spec(
        backend=backend,
        outer_context=context,
        candidate_specs=(ClassifierSpec(C=0.1, random_state=23), ClassifierSpec(C=1.0, random_state=23)),
    )

    assert selection.heldout_center == "0"
    assert "0" not in backend.target_eval_centers
    assert "0" not in backend.synthetic_sources
    for pseudo_target, source in backend.synthetic_pairs:
        assert source != pseudo_target


def test_midogpp_scoring_runner_rejects_missing_requested_baseline(tmp_path: Path) -> None:
    context = _run_context()
    candidates = [
        MidogppCandidate(
            candidate_source_center="1",
            candidate_id="midogpp_source_1_single_source_adaptive_k",
            checkpoint_hash="checkpoint-1",
        )
    ]

    try:
        run_midogpp_phase1_scoring(
            backend=_FakeMidogppBackend(),
            contexts=[context],
            candidates_by_heldout={"0": candidates},
            artifacts_root=tmp_path,
            baseline_methods=("missing_locked_baseline",),
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("missing requested baseline was silently omitted")


def test_midogpp_scoring_runner_records_candidate_failure_without_selection() -> None:
    context = _run_context()
    candidate = MidogppCandidate(
        candidate_source_center="1",
        candidate_id="midogpp_source_1_single_source_adaptive_k",
        checkpoint_hash="checkpoint-1",
    )

    row = score_midogpp_candidate(
        backend=_FailingMidogppBackend(),
        context=context,
        candidate=candidate,
    )

    assert row.status == "failed_empty_reference_pool"
    assert row.claim_role == "oracle_diagnostic"
    assert row.eligibility == DIAGNOSTIC_ONLY


def test_source_summary_backend_loads_npz_summaries_and_eval_cache(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path)
    context = _run_context()
    backend = SourceSummaryMidogppBackend(
        summary_manifest=manifest,
        test_cache_path=cache,
    )
    candidate = backend.candidate_for_source(context=context, source_center="1")

    synthetic, labels = backend.synthetic_train_batch(candidate, context=context)
    eval_x, eval_y = backend.target_eval_batch(context=context)
    row = score_midogpp_candidate(backend=backend, context=context, candidate=candidate)

    assert len(labels) == 8
    assert getattr(synthetic, "shape") == (8, 2)
    assert getattr(eval_x, "shape") == (4, 2)
    assert eval_y == [0, 0, 1, 1]
    assert row.status == "ok"
    assert row.checkpoint_hash == "summary-1-0|summary-1-1"


def test_source_summary_backend_imports_locked_diagnostic_baseline(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path)
    baseline = _write_external_baseline_fixture(tmp_path)
    backend = SourceSummaryMidogppBackend(
        summary_manifest=manifest,
        test_cache_path=cache,
        baseline_matrix_paths=(baseline,),
    )

    row = score_midogpp_baseline(
        backend=backend,
        context=_run_context(),
        baseline_method="real_source_embedding_classifier_dense_reference",
        candidate_sources=("1",),
    )

    assert row is not None
    assert row.row_type == MIDOGPP_METHOD_BASELINE_ROW_TYPE
    assert row.eligibility == DIAGNOSTIC_ONLY
    assert row.bacc == 0.81
    assert row.macro_f1 == 0.79
    assert row.checkpoint_hash.startswith("baseline:real_source_embedding_classifier_dense_reference:")
    assert len(row.checkpoint_hash.rsplit(":", 1)[-1]) == 64


def test_midogpp_external_baseline_preflight_passes_for_requested_context(tmp_path: Path) -> None:
    baseline = _write_external_baseline_fixture(tmp_path)

    report = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(baseline,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )

    assert report.status == "PASS"
    assert report.available_baseline_keys == (
        "method=real_source_embedding_classifier_dense_reference|seed=42|heldout=0|replicate_seed=0",
    )
    assert len(report.baseline_matrix_hashes[str(baseline)]) == 16
    assert len(next(iter(report.baseline_row_hashes.values()))) == 64


def test_midogpp_external_baseline_preflight_hashes_change_with_content(tmp_path: Path) -> None:
    baseline = _write_external_baseline_fixture(tmp_path)
    first = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(baseline,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )
    baseline.write_text(
        baseline.read_text(encoding="utf-8").replace("0.81", "0.82"),
        encoding="utf-8",
    )
    second = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(baseline,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )

    assert first.baseline_matrix_hashes[str(baseline)] != second.baseline_matrix_hashes[str(baseline)]
    assert first.baseline_row_hashes != second.baseline_row_hashes


def test_midogpp_external_baseline_row_hash_is_path_stable(tmp_path: Path) -> None:
    baseline = _write_external_baseline_fixture(tmp_path)
    copy = tmp_path / "copied" / "locked_baseline_matrix.csv"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(baseline.read_bytes())

    first = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(baseline,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )
    second = preflight_midogpp_external_baselines(
        baseline_matrix_paths=(copy,),
        baseline_methods=("real_source_embedding_classifier_dense_reference",),
        experiment_seed=42,
        replicate_seed=0,
        heldout_centers=("0",),
    )

    assert first.baseline_row_hashes == second.baseline_row_hashes
    assert first.baseline_matrix_hashes[str(baseline)] == second.baseline_matrix_hashes[str(copy)]


def test_midogpp_external_baseline_preflight_rejects_missing_requested_context(tmp_path: Path) -> None:
    baseline = _write_external_baseline_fixture(tmp_path)

    try:
        preflight_midogpp_external_baselines(
            baseline_matrix_paths=(baseline,),
            baseline_methods=("real_source_embedding_classifier_dense_reference",),
            experiment_seed=42,
            replicate_seed=0,
            heldout_centers=("1",),
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("baseline preflight accepted missing heldout context")


def test_midogpp_phase1_run_hashes_write_frozen_snapshot(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )

    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path / "out",
    )

    snapshot = json.loads(run_hashes.frozen_snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["protocol_hash"] == run_hashes.protocol_hash
    assert run_hashes.to_report()["schema_version"] == "midogpp_phase1_run_hashes_v1"
    assert len(run_hashes.config_hash) == 16
    assert len(run_hashes.feature_frame_hash) == 16
    assert len(run_hashes.summary_manifest_hash) == 16
    assert run_hashes.source_summary_file_hashes
    assert run_hashes.cache_file_hashes


def test_midogpp_phase1_run_hashes_change_with_manifest_and_cache_content(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    cache = _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")
    preflight = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )
    first = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path / "out1",
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path / "out2",
    )

    assert first.snapshot.candidate_pool_hash != second.snapshot.candidate_pool_hash

    summary_key, summary_path = next(iter(preflight.source_summary_paths.items()))
    Path(summary_path).write_bytes(Path(summary_path).read_bytes() + b"\n")
    summary_changed = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path / "out-summary-changed",
    )

    assert summary_key in summary_changed.source_summary_file_hashes
    assert second.snapshot.candidate_pool_hash != summary_changed.snapshot.candidate_pool_hash
    assert second.snapshot.generation_config_hash != summary_changed.snapshot.generation_config_hash

    cache.write_bytes(cache.read_bytes() + b"\n")
    third = build_midogpp_phase1_run_hashes(
        config_path=ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml",
        summary_manifest=manifest,
        preflight=preflight,
        heldout_centers=("0",),
        experiment_seed=42,
        replicate_seed=0,
        synthetic_per_class_total=128,
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        out_dir=tmp_path / "out3",
    )

    assert second.feature_frame_hash != third.feature_frame_hash


def test_source_summary_backend_rejects_leaky_external_baseline(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path)
    baseline = _write_external_baseline_fixture(tmp_path, selection_used_target_labels="true")

    try:
        SourceSummaryMidogppBackend(
            summary_manifest=manifest,
            test_cache_path=cache,
            baseline_matrix_paths=(baseline,),
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("leaky external baseline matrix was not rejected")


def test_midogpp_phase1_script_preflight_validates_baselines(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    cache = _write_feature_cache_fixture(tmp_path)
    baseline = _write_external_baseline_fixture(tmp_path)
    script = ROOT / "scripts" / "run_midogpp_source_summary_phase1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-manifest",
            str(manifest),
            "--test-cache-path",
            str(cache),
            "--out-dir",
            str(tmp_path / "out"),
            "--experiment-seed",
            "42",
            "--replicate-seed",
            "0",
            "--heldout-centers",
            "0",
            "--baseline-matrix",
            str(baseline),
            "--baseline-method",
            "real_source_embedding_classifier_dense_reference",
            "--preflight-only",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "PASS"' in result.stdout
    assert "real_source_embedding_classifier_dense_reference" in result.stdout
    source_report = tmp_path / "out" / "reports" / "source_summary_preflight_report.json"
    baseline_report = tmp_path / "out" / "reports" / "baseline_preflight_report.json"
    run_hashes_report = tmp_path / "out" / "reports" / "run_hashes_report.json"
    frozen_snapshot = tmp_path / "out" / "configs" / "frozen_protocol_snapshot.json"
    assert json.loads(source_report.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(baseline_report.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(run_hashes_report.read_text(encoding="utf-8"))["schema_version"] == "midogpp_phase1_run_hashes_v1"
    assert "protocol_hash" in json.loads(frozen_snapshot.read_text(encoding="utf-8"))


def test_midogpp_phase1_script_preflight_failure_writes_report(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path)
    script = ROOT / "scripts" / "run_midogpp_source_summary_phase1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-manifest",
            str(manifest),
            "--test-cache-path",
            str(cache),
            "--out-dir",
            str(tmp_path / "out"),
            "--experiment-seed",
            "42",
            "--heldout-centers",
            "0",
            "--preflight-only",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    report_path = tmp_path / "out" / "reports" / "source_summary_preflight_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert report["status"] == "FAIL"
    assert "source summary coverage is incomplete" in report["error_message"]


def test_midogpp_phase1_script_preflight_failure_reports_unresolved_workstation_paths(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    manifest_text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        manifest_text.replace(str(tmp_path), "/home/stud/spark/cvae_experiments/missing_midogpp"),
        encoding="utf-8",
    )
    cache = _write_feature_cache_fixture(tmp_path)
    script = ROOT / "scripts" / "run_midogpp_source_summary_phase1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-manifest",
            str(manifest),
            "--test-cache-path",
            str(cache),
            "--out-dir",
            str(tmp_path / "out"),
            "--experiment-seed",
            "42",
            "--heldout-centers",
            "0",
            "--preflight-only",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    report_path = tmp_path / "out" / "reports" / "source_summary_preflight_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert report["status"] == "FAIL"
    assert "summary file not found" in report["error_message"]
    assert "/home/stud/spark" in report["error_message"]


def test_midogpp_phase1_script_rejects_stale_explicit_hash(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    cache = _write_feature_cache_fixture(tmp_path)
    script = ROOT / "scripts" / "run_midogpp_source_summary_phase1.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-manifest",
            str(manifest),
            "--test-cache-path",
            str(cache),
            "--out-dir",
            str(tmp_path / "out"),
            "--experiment-seed",
            "42",
            "--replicate-seed",
            "0",
            "--heldout-centers",
            "0",
            "--config-hash",
            "stale-config-hash",
            "--preflight-only",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "does not match generated frozen value" in result.stderr


def test_midogpp_late_aggregation_import_script_materializes_valid_phase1_artifacts(tmp_path: Path) -> None:
    late_matrix = _write_late_aggregation_fixture(tmp_path)
    dense_matrix = _write_dense_late_fixture(tmp_path)
    out_dir = tmp_path / "imported"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_midogpp_late_aggregation_phase1.py"),
            "--late-aggregation-matrix",
            str(late_matrix),
            "--dense-matrix",
            str(dense_matrix),
            "--out-dir",
            str(out_dir),
            "--experiment-seed",
            "42",
            "--heldout-centers",
            "0",
            "--synthetic-per-class-total",
            "128",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = validate_midogpp_phase1_artifacts(out_dir, expected_heldout_centers=("0",))
    assert report["status"] == "PASS"
    assert report["diagnostic_rows"] == 9
    assert report["candidate_manifest_rows"] == 8
    assert report["oracle_summary_rows"] == 1
    assert report["baseline_comparison_rows"] == 1
    provenance = json.loads((out_dir / "reports" / "import_provenance_report.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "midogpp_late_aggregation_import_v1"
    assert "no deployable selection" in provenance["claim_boundary"]


def test_source_summary_backend_rejects_missing_target_or_summary(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path, heldout_center="2")
    backend = SourceSummaryMidogppBackend(summary_manifest=manifest, test_cache_path=cache)

    try:
        backend.target_eval_batch(context=_run_context())
    except ProtocolError:
        pass
    else:
        raise AssertionError("missing heldout target eval rows were not rejected")

    try:
        backend.candidate_for_source(context=_run_context(), source_center="3")
    except ProtocolError:
        pass
    else:
        raise AssertionError("missing source summary pair was not rejected")


def test_source_summary_backend_loads_npz_cache_metadata(tmp_path: Path) -> None:
    cache = _write_feature_cache_fixture(tmp_path)
    loaded = load_midogpp_feature_cache(cache)

    assert getattr(loaded.embeddings, "shape") == (4, 2)
    assert loaded.metadata[0]["center"] == "0"


def test_source_summary_preflight_passes_for_complete_fixture_root(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    _write_feature_cache_fixture(tmp_path / "cache_root" / "seed42" / "embeddings", filename="test.npz")

    report = preflight_midogpp_source_summary_inputs(
        summary_manifest=manifest,
        experiment_seeds=(42,),
        heldout_centers=("0",),
        test_cache_root=tmp_path / "cache_root",
    )

    assert report.status == "PASS"
    assert report.cache_eval_counts["seed=42|heldout=0"] == 4
    assert report.cache_label_sets["seed=42|heldout=0"] == (0, 1)
    assert report.source_summary_dims["seed=42|source=1|class=0"] == 2
    assert report.cache_embedding_dims["seed=42|cache=test"] == 2


def test_source_summary_preflight_rejects_missing_summary_pair(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1",))
    cache = _write_feature_cache_fixture(tmp_path)

    try:
        preflight_midogpp_source_summary_inputs(
            summary_manifest=manifest,
            experiment_seeds=(42,),
            heldout_centers=("0",),
            test_cache_path=cache,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("incomplete summary coverage was not rejected")


def test_source_summary_preflight_rejects_mono_class_target_eval(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(tmp_path, source_centers=("1", "2", "3", "5", "6", "7", "8", "9"))
    cache = _write_feature_cache_fixture(tmp_path, labels=(0, 0, 0, 0))

    try:
        preflight_midogpp_source_summary_inputs(
            summary_manifest=manifest,
            experiment_seeds=(42,),
            heldout_centers=("0",),
            test_cache_path=cache,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("mono-class target eval cache was not rejected")


def test_source_summary_preflight_rejects_feature_frame_mismatch(tmp_path: Path) -> None:
    manifest = _write_source_summary_fixture(
        tmp_path,
        source_centers=("1", "2", "3", "5", "6", "7", "8", "9"),
        dim=3,
    )
    cache = _write_feature_cache_fixture(tmp_path)

    try:
        preflight_midogpp_source_summary_inputs(
            summary_manifest=manifest,
            experiment_seeds=(42,),
            heldout_centers=("0",),
            test_cache_path=cache,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("source summary/cache feature-frame mismatch was not rejected")


def test_midogpp_diagnostic_matrix_rejects_deployable_paths_and_duplicate_keys(tmp_path: Path) -> None:
    try:
        write_midogpp_diagnostic_matrix(tmp_path / "features" / "diagnostic_downstream_utility.csv", [_matrix_row()])
    except ProtocolError:
        pass
    else:
        raise AssertionError("diagnostic matrix under features path was not rejected")

    try:
        assert_selection_does_not_read_matrix(tmp_path / "tables" / "diagnostic_downstream_utility.csv")
    except ProtocolError:
        pass
    else:
        raise AssertionError("selection read of diagnostic matrix was not rejected")

    duplicate = _matrix_row(candidate_source_center="1")
    try:
        write_midogpp_diagnostic_matrix(
            tmp_path / "tables" / "diagnostic_downstream_utility.csv",
            [duplicate, duplicate],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("duplicate MIDOG++ context-specific key was not rejected")


def test_midogpp_feature_table_rejects_target_downstream_and_raw_id_features() -> None:
    good = _feature_row() | {"feature_source": "source_manifest_only", "metadata_distance": 0.4}
    assert_midogpp_feature_table([good])
    assert midogpp_deployable_feature_columns(tuple(good)) == ("metadata_distance",)

    bad = good | {"bacc": 0.91, "oracle_rank": 1}
    try:
        assert_midogpp_feature_table([bad])
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ forbidden deployable feature leak was not rejected")

    try:
        midogpp_deployable_feature_columns(("candidate_id", "metadata_distance", "bacc"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ deployable feature column leak was not rejected")

    raw_id = good | {"case_id": "case-1", "sample_path": "/tmp/sample.pt"}
    try:
        assert_midogpp_feature_table([raw_id])
    except ProtocolError:
        pass
    else:
        raise AssertionError("MIDOG++ raw ID feature leak was not rejected")


def test_midogpp_diagnostic_rows_reject_support_label_and_target_label_misuse() -> None:
    try:
        _matrix_row(support_labels_used=True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("support label use was not rejected")

    try:
        _matrix_row(target_eval_labels_used_for_scoring_only=False)
    except ProtocolError:
        pass
    else:
        raise AssertionError("target eval label misuse flag was not rejected")


def _source_summary(*, source_center: str, class_label: int, status: str = "ok") -> dict[str, object]:
    return {
        "experiment_seed": 42,
        "source_center": source_center,
        "class_label": class_label,
        "expert_config_hash": f"config-{source_center}",
        "summary_hash": f"summary-{source_center}-{class_label}",
        "status": status,
    }


def _matrix_row(
    *,
    candidate_source_center: str = "1",
    candidate_id: str | None = None,
    candidate_method: str = "single_source_adaptive_k",
    row_type: str = "single_source",
    support_labels_used: bool = False,
    target_eval_labels_used_for_scoring_only: bool = True,
    bacc: float = 0.62,
    macro_f1: float = 0.60,
    status: str = "ok",
    error_message: str = "",
    config_hash: str = "config-hash",
    protocol_hash: str = "protocol-hash",
    feature_frame_hash: str = "feature-frame-hash",
) -> MidogppDownstreamRow:
    return MidogppDownstreamRow(
        heldout_center="0",
        candidate_source_center=candidate_source_center,
        candidate_id=candidate_id or f"midogpp_source_{candidate_source_center}_single_source_adaptive_k",
        candidate_method=candidate_method,
        experiment_seed=42,
        replicate_seed=0,
        support_size=0,
        support_seed="none",
        support_set_id="none",
        eval_set_id="eval-center-0",
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        synthetic_per_class_total=128,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        checkpoint_hash=f"checkpoint-{candidate_source_center}",
        feature_frame_hash=feature_frame_hash,
        row_type=row_type,
        bacc=bacc,
        macro_f1=macro_f1,
        status=status,
        error_message=error_message,
        support_labels_used=support_labels_used,
        target_eval_labels_used_for_scoring_only=target_eval_labels_used_for_scoring_only,
    )


def _full_single_source_rows(
    *,
    heldout_center: str = "0",
    config_hash: str = "config-hash",
    protocol_hash: str = "protocol-hash",
    feature_frame_hash: str = "feature-frame-hash",
) -> list[MidogppDownstreamRow]:
    return [
        _matrix_row(
            candidate_source_center=center,
            bacc=0.55 + (int(center) / 100.0),
            macro_f1=0.53 + (int(center) / 100.0),
            config_hash=config_hash,
            protocol_hash=protocol_hash,
            feature_frame_hash=feature_frame_hash,
        )
        for center in MIDOGPP_ELIGIBLE_CENTERS
        if center != str(heldout_center)
    ]


def _candidate_manifest_from_rows(rows: list[MidogppDownstreamRow]) -> list[dict[str, object]]:
    return [
        {
            "heldout_center": row.heldout_center,
            "candidate_source_center": row.candidate_source_center,
            "candidate_id": row.candidate_id,
            "eligibility": SELECTION_ELIGIBLE,
        }
        for row in rows
        if row.row_type != MIDOGPP_METHOD_BASELINE_ROW_TYPE
    ]


def _feature_row() -> dict[str, object]:
    return {
        column: _feature_value(column)
        for column in MIDOGPP_DOWNSTREAM_PRIMARY_KEY
    } | {"eligibility": SELECTION_ELIGIBLE}


def _feature_value(column: str) -> object:
    values = {
        "dataset": "midogpp",
        "domain_regime": "heldout_center",
        "heldout_center": "0",
        "candidate_source_center": "1",
        "candidate_id": "midogpp_source_1_single_source_adaptive_k",
        "candidate_method": "single_source_adaptive_k",
        "experiment_seed": 42,
        "replicate_seed": 0,
        "support_size": 0,
        "support_seed": "none",
        "support_set_id": "none",
        "eval_set_id": "eval-center-0",
        "generation_seed": 17,
        "classifier_seed": 23,
        "synthetic_per_class_total": 128,
        "config_hash": "config-hash",
        "protocol_hash": "protocol-hash",
        "checkpoint_hash": "checkpoint-1",
        "feature_frame_hash": "feature-frame-hash",
    }
    return values[column]


def _run_context() -> MidogppRunContext:
    return MidogppRunContext(
        heldout_center="0",
        experiment_seed=42,
        replicate_seed=0,
        support_size=0,
        support_seed="none",
        support_set_id="none",
        eval_set_id="eval-center-0",
        generation_seed=17,
        latent_sample_seed=17,
        classifier_seed=23,
        synthetic_per_class_total=4,
        config_hash="config-hash",
        protocol_hash="protocol-hash",
        feature_frame_hash="feature-frame-hash",
    )


class _FakeMidogppBackend:
    def synthetic_train_batch(self, candidate: MidogppCandidate, *, context: MidogppRunContext):
        _ = context
        if candidate.candidate_source_center == "3":
            return (
                [
                    [-2.0, -1.8],
                    [-1.6, -1.5],
                    [2.0, 1.8],
                    [1.6, 1.5],
                ],
                [0, 0, 1, 1],
            )
        return (
            [
                [-0.2, 0.1],
                [0.1, -0.2],
                [0.2, -0.1],
                [-0.1, 0.2],
            ],
            [0, 0, 1, 1],
        )

    def target_eval_batch(self, *, context: MidogppRunContext):
        _ = context
        return (
            [
                [-2.2, -1.7],
                [-1.4, -1.6],
                [1.9, 1.7],
                [1.7, 1.4],
            ],
            [0, 0, 1, 1],
        )

    def method_baseline_score(self, baseline_method: str, *, context: MidogppRunContext, candidate_sources):
        _ = context, candidate_sources
        if baseline_method == "dense_late_equal_all_sources_geom":
            return MidogppScoringResult(bacc=0.75, macro_f1=0.73)
        return None


class _TrackingMidogppBackend(_FakeMidogppBackend):
    def __init__(self) -> None:
        self.target_eval_centers: list[str] = []
        self.synthetic_sources: list[str] = []
        self.synthetic_pairs: list[tuple[str, str]] = []

    def synthetic_train_batch(self, candidate: MidogppCandidate, *, context: MidogppRunContext):
        self.synthetic_sources.append(str(candidate.candidate_source_center))
        self.synthetic_pairs.append((str(context.heldout_center), str(candidate.candidate_source_center)))
        return super().synthetic_train_batch(candidate, context=context)

    def target_eval_batch(self, *, context: MidogppRunContext):
        self.target_eval_centers.append(str(context.heldout_center))
        return super().target_eval_batch(context=context)


class _FailingMidogppBackend(_FakeMidogppBackend):
    def synthetic_train_batch(self, candidate: MidogppCandidate, *, context: MidogppRunContext):
        _ = candidate, context
        raise ProtocolError("Empty reference pool for expert 1")


def _write_source_summary_fixture(
    tmp_path: Path,
    *,
    source_centers: tuple[str, ...],
    dim: int = 2,
) -> Path:
    import numpy as np

    rows: list[dict[str, object]] = []
    root = tmp_path / "summaries"
    for source in source_centers:
        for cls in (0, 1):
            path = root / f"seed_42/source_{source}/class_{cls}_adaptive_largest_viable_summary.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            mean = -2.0 if cls == 0 else 2.0
            np.savez(
                path,
                weights=np.asarray([1.0], dtype=float),
                means=np.asarray([[mean] * int(dim)], dtype=float),
                diag_vars=np.asarray([[0.01] * int(dim)], dtype=float),
            )
            rows.append(
                {
                    "experiment_seed": 42,
                    "source_center": source,
                    "class_label": cls,
                    "selection_rule": "largest_viable",
                    "summary_path": str(path),
                    "summary_hash": f"summary-{source}-{cls}",
                    "expert_config_hash": f"config-{source}",
                    "status": "ok",
                }
            )
    manifest = tmp_path / "exported_source_summary_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _write_feature_cache_fixture(
    tmp_path: Path,
    *,
    heldout_center: str = "0",
    labels: tuple[int, int, int, int] = (0, 0, 1, 1),
    filename: str = "test_cache.npz",
) -> Path:
    import numpy as np

    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {"sample_id": f"sample_{heldout_center}_{idx}", "center": heldout_center, "label": label}
        for idx, label in enumerate(labels)
    ]
    np.savez(
        path,
        embeddings=np.asarray([[-2.0, -2.1], [-1.9, -1.8], [2.0, 1.9], [1.8, 2.1]], dtype=float),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return path


def _write_external_baseline_fixture(
    tmp_path: Path,
    *,
    selection_used_target_labels: str = "false",
) -> Path:
    path = tmp_path / "locked_baseline_matrix.csv"
    rows = [
        {
            "experiment_seed": 42,
            "heldout_center": "0",
            "prior_method": "real_source_embedding_classifier_dense_reference",
            "replicate_seed": 0,
            "bacc": 0.81,
            "macro_f1": 0.79,
            "status": "ok",
            "selection_source": "diagnostic_only",
            "selection_used_target_labels": selection_used_target_labels,
            "support_labels_used": "false",
            "target_eval_labels_used_for_scoring_only": "true",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_late_aggregation_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "late_aggregation_matrix.csv"
    rows: list[dict[str, object]] = []
    for idx, source in enumerate(("1", "2", "3", "5", "6", "7", "8", "9")):
        rows.append(
            {
                "experiment_seed": 42,
                "heldout_center": "0",
                "expert_id": source,
                "expert_pool_type": "per_source",
                "variant_id": "pca64_beta001",
                "prior_method": "dense_late_equal_all_sources_geom",
                "gmm_components": 31,
                "source_weighting": "equal_source_mass",
                "pooling_rule": "single_source",
                "replicate_seed": 17,
                "latent_sample_seed": 1000 + idx,
                "synthetic_per_class_total": 16,
                "bacc": 0.50 + idx / 100,
                "macro_f1": 0.49 + idx / 100,
                "generated_features_hash": f"generated-{source}",
                "prediction_hash": f"prediction-{source}",
                "composed_prior_hash": f"prior-{source}",
                "summary_set_hash": f"summary-{source}",
                "status": "ok",
                "error_message": "",
                "claim_role": "single_source_component_for_dense_aggregation",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_dense_late_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "dense_late_all_sources_downstream_matrix.csv"
    rows = [
        {
            "experiment_seed": 42,
            "heldout_center": "0",
            "expert_id": "dense_all_sources",
            "expert_pool_type": "decentralized_source_summary",
            "variant_id": "pca64_beta001",
            "prior_method": "dense_late_equal_all_sources_geom",
            "gmm_components": 31,
            "source_weighting": "equal_source_mass",
            "pooling_rule": "geometric",
            "replicate_seed": 17,
            "latent_sample_seed": 2000,
            "synthetic_per_class_total": 128,
            "bacc": 0.57,
            "macro_f1": 0.55,
            "generated_features_hash": "generated-dense",
            "prediction_hash": "prediction-dense",
            "composed_prior_hash": "prior-dense",
            "summary_set_hash": "summary-dense",
            "status": "ok",
            "error_message": "",
            "claim_role": "primary_equal_all_sources_baseline",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
