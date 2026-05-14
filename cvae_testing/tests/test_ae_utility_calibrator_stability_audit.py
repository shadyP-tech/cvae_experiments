from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_ae_utility_calibrator_stability import build_outputs


PRIMARY = "ae_utility_calibrated_safe_override_v1"
AE_ARGMIN = "ae_argmin_zscore"
BASELINES = [
    "metadata_routing",
    "metadata_ae_residual_safe_override_v1",
    "pairwise_ranker_ae_combined",
    "ae_first_margin_gated_v1",
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_run(
    root: Path,
    seed: int,
    specs: dict[int, dict[str, float]],
    *,
    heldout_target_nelbo_used_for_selection: int = 0,
    patient_overlap: bool = False,
    include_overlap_audit: bool = True,
) -> Path:
    reports = root / "outputs" / "camelyon17" / "learned_utility_ae_utility_calibrator_v1" / f"seed{seed}" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result_path = reports / "learned_utility_results.json"
    result_path.write_text("{}\n", encoding="utf-8")

    domain_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    precision_rows: list[dict[str, object]] = []
    source_inner_rows: list[dict[str, object]] = []
    for center, spec in sorted(specs.items()):
        ae_top1 = 0.50
        ae_spearman = 0.40
        ae_gap = 10.0
        primary_top1 = ae_top1 + spec.get("top1_delta", 0.05)
        primary_spearman = ae_spearman + spec.get("spearman_delta", 0.05)
        primary_gap = ae_gap - spec.get("gap_reduction", 2.0)
        active = spec.get("active_override_rate", 0.30)
        precision = spec.get("selected_override_precision", 0.80)
        harmful = spec.get("harmful_vs_ae_argmin_rate", 0.05)
        improving = spec.get("improving_vs_ae_argmin_rate", 0.30)

        domain_rows.append(
            {
                "method": PRIMARY,
                "query_domain": center,
                "top1_oracle_hit": primary_top1,
                "spearman": primary_spearman,
                "mean_oracle_gap_pct": primary_gap,
            }
        )
        domain_rows.append(
            {
                "method": AE_ARGMIN,
                "query_domain": center,
                "top1_oracle_hit": ae_top1,
                "spearman": ae_spearman,
                "mean_oracle_gap_pct": ae_gap,
            }
        )
        for idx, method in enumerate(BASELINES):
            domain_rows.append(
                {
                    "method": method,
                    "query_domain": center,
                    "top1_oracle_hit": 0.35 + idx * 0.01,
                    "spearman": 0.20 + idx * 0.01,
                    "mean_oracle_gap_pct": 12.0 - idx * 0.1,
                }
            )

        precision_value: object = precision
        if active == 0.0:
            precision_value = "nan"
        policy_rows.append(
            {
                "method": PRIMARY,
                "fold_query_domain": center,
                "feature_set": "ae_core",
                "selected_delta_threshold": 0.025,
                "selected_margin_threshold": 0.05,
                "active_override_rate": active,
                "selected_override_precision": precision_value,
                "harmful_vs_ae_argmin_rate": harmful,
                "improving_vs_ae_argmin_rate": improving,
                "override_capture_rate": spec.get("override_capture_rate", 0.35),
                "oracle_improvable_query_rate": spec.get("oracle_improvable_query_rate", 0.60),
                "oracle_headroom_vs_ae_argmin": spec.get("oracle_headroom_vs_ae_argmin", 3.0),
                "net_gain_vs_ae_argmin": spec.get("gap_reduction", 2.0),
                "heldout_target_nelbo_used_for_selection": heldout_target_nelbo_used_for_selection,
                "excluded_target_ae": 1,
                "excluded_target_cvae": 1,
                "excluded_pseudo_query_ae": 1,
                "excluded_pseudo_query_cvae": 1,
            }
        )
        precision_rows.append(
            {
                "method": PRIMARY,
                "fold_query_domain": center,
                "active_overrides": int(active * 100),
                "selected_override_precision": precision_value,
                "active_override_rate": active,
                "override_capture_rate": spec.get("override_capture_rate", 0.35),
            }
        )
        source_inner_rows.append(
            {
                "method": PRIMARY,
                "feature_set": "ae_core",
                "fold_query_domain": center,
                "source_inner_pseudo_query_domain": (center + 1) % 5,
                "delta_threshold": 0.025,
                "margin_threshold": 0.05,
                "selected_feature_set": "ae_core",
                "selected_delta_threshold": 0.025,
                "selected_margin_threshold": 0.05,
                "selected_by_source_inner_validation": 1,
                "macro_top1_oracle_hit": primary_top1,
                "macro_mean_oracle_gap_pct": primary_gap,
                "macro_active_override_rate": active,
                "macro_selected_override_precision": precision_value,
                "heldout_target_nelbo_used_for_selection": heldout_target_nelbo_used_for_selection,
            }
        )

    _write_csv(reports / "learned_utility_domain_breakdown.csv", domain_rows)
    _write_csv(reports / "ae_utility_calibrator_policy_audit.csv", policy_rows)
    _write_csv(reports / "ae_utility_calibrator_override_precision.csv", precision_rows)
    _write_csv(reports / "ae_utility_calibrator_source_inner_validation.csv", source_inner_rows)
    (reports / "leakage_report.json").write_text(
        json.dumps(
            {
                "patient_overlap": {"train_test": ["p1"] if patient_overlap else [], "val_test": []},
                "duplicate_paths": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / "support_free_ae_provenance.json").write_text(
        json.dumps(
            {
                "target_support_used": 0,
                "target_labels_used": 0,
                "target_domain_normalization_statistics_used": 0,
                "target_ae_excluded": 1,
                "source_inner_self_ae_excluded": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if include_overlap_audit:
        _write_csv(
            reports / "support_free_ae_overlap_audit.csv",
            [{"ae_train_query_overlap_count": 0, "ae_val_query_overlap_count": 0}],
        )
    return result_path


def _default_specs(**overrides: float) -> dict[int, dict[str, float]]:
    return {
        center: {
            "gap_reduction": overrides.get("gap_reduction", 2.0),
            "top1_delta": overrides.get("top1_delta", 0.05),
            "spearman_delta": overrides.get("spearman_delta", 0.05),
            "active_override_rate": overrides.get("active_override_rate", 0.30),
            "selected_override_precision": overrides.get("selected_override_precision", 0.80),
            "harmful_vs_ae_argmin_rate": overrides.get("harmful_vs_ae_argmin_rate", 0.05),
            "improving_vs_ae_argmin_rate": overrides.get("improving_vs_ae_argmin_rate", 0.30),
        }
        for center in range(5)
    }


def _run_audit(tmp_path: Path, run_paths: list[Path]) -> tuple[dict[str, object], Path]:
    manifest = tmp_path / "results" / "comparison_tables" / "ae_utility_calibrator_run_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in run_paths) + "\n", encoding="utf-8")
    output_dir = tmp_path / "audit"
    summary_md = tmp_path / "results" / "summaries" / "ae_utility_calibrator_camelyon17_stability_audit.md"
    result = build_outputs(
        manifest=manifest,
        dataset="camelyon17",
        output_dir=output_dir,
        summary_md=summary_md,
    )
    return result, output_dir


def test_camelyon17_stability_audit_detects_one_center_dominance(tmp_path: Path) -> None:
    runs = []
    for seed in [11, 42, 73]:
        specs = _default_specs(gap_reduction=1.0)
        specs[0]["gap_reduction"] = 10.0
        runs.append(_write_run(tmp_path, seed, specs))

    result, _output_dir = _run_audit(tmp_path, runs)

    assert result["summary"]["aggregate_verdict"] == "PASS"
    assert result["summary"]["stability_interpretation"] == "CENTER-SENSITIVE PASS"
    assert result["summary"]["max_center_positive_gap_reduction_share"] > 0.50


def test_camelyon17_stability_audit_detects_seed_instability(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, 11, _default_specs(top1_delta=-0.05)),
        _write_run(tmp_path, 42, _default_specs(top1_delta=0.10)),
        _write_run(tmp_path, 73, _default_specs(top1_delta=0.10)),
    ]

    result, output_dir = _run_audit(tmp_path, runs)
    per_seed = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_per_seed_stability.csv")

    assert result["summary"]["aggregate_verdict"] == "PASS"
    assert result["summary"]["stability_interpretation"] == "SEED-SENSITIVE PASS"
    assert any(row["seed"] == "11" and int(row["material_degradation_units"]) > 0 for row in per_seed)


def test_camelyon17_stability_audit_leave_one_center_fragility(tmp_path: Path) -> None:
    runs = []
    for seed in [11, 42, 73]:
        specs = _default_specs(gap_reduction=0.2, active_override_rate=0.05)
        specs[0]["gap_reduction"] = 10.0
        specs[0]["active_override_rate"] = 0.80
        runs.append(_write_run(tmp_path, seed, specs))

    result, output_dir = _run_audit(tmp_path, runs)
    leave_one = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_leave_one_center_sensitivity.csv")

    assert result["summary"]["aggregate_verdict"] == "PASS"
    assert result["summary"]["leave_one_center_fragile"] == 1
    assert result["summary"]["stability_interpretation"] == "CENTER-SENSITIVE PASS"
    assert any(row["removed_heldout_center"] == "0" and row["verdict_without_center"] == "DIAGNOSTIC ONLY" for row in leave_one)


def test_camelyon17_stability_audit_passes_consistent_multicenter_gain(tmp_path: Path) -> None:
    runs = [_write_run(tmp_path, seed, _default_specs()) for seed in [11, 42, 73]]

    result, output_dir = _run_audit(tmp_path, runs)
    domain_rows = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_per_domain_stability.csv")

    assert result["summary"]["protocol_status"] == "PASS"
    assert result["summary"]["aggregate_verdict"] == "PASS"
    assert result["summary"]["stability_interpretation"] == "STABLE PASS"
    assert len(domain_rows) == 5


def test_camelyon17_stability_audit_rejects_target_nelbo_selection_flag(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, seed, _default_specs(), heldout_target_nelbo_used_for_selection=1)
        for seed in [11, 42, 73]
    ]

    result, output_dir = _run_audit(tmp_path, runs)
    provenance_rows = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_leakage_provenance_audit.csv")

    assert result["summary"]["protocol_status"] == "FAIL"
    assert result["summary"]["stability_interpretation"] == "REJECTED"
    assert all("policy_heldout_target_nelbo_not_used" in row["failed_checks"] for row in provenance_rows)


def test_camelyon17_stability_audit_rejects_patient_overlap(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, seed, _default_specs(), patient_overlap=(seed == 42))
        for seed in [11, 42, 73]
    ]

    result, output_dir = _run_audit(tmp_path, runs)
    provenance_rows = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_leakage_provenance_audit.csv")

    assert result["summary"]["protocol_status"] == "FAIL"
    assert result["summary"]["stability_interpretation"] == "REJECTED"
    assert any("patient_train_test_overlap_zero" in row["failed_checks"] for row in provenance_rows)


def test_camelyon17_stability_audit_reports_missing_overlap_as_needs_evidence(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, seed, _default_specs(), include_overlap_audit=False)
        for seed in [11, 42, 73]
    ]

    result, output_dir = _run_audit(tmp_path, runs)
    provenance_rows = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_leakage_provenance_audit.csv")

    assert result["summary"]["protocol_status"] == "NEEDS EVIDENCE"
    assert result["summary"]["stability_interpretation"] == "DIAGNOSTIC ONLY"
    assert all(row["provenance_status"] == "NEEDS EVIDENCE" for row in provenance_rows)


def test_camelyon17_stability_audit_handles_zero_active_overrides_as_nan_precision(tmp_path: Path) -> None:
    runs = [
        _write_run(
            tmp_path,
            seed,
            _default_specs(active_override_rate=0.0, selected_override_precision=float("nan")),
        )
        for seed in [11, 42, 73]
    ]

    result, output_dir = _run_audit(tmp_path, runs)
    metrics = _read_csv(output_dir / "ae_utility_calibrator_camelyon17_seed_domain_metrics.csv")

    assert result["summary"]["aggregate_verdict"] == "DIAGNOSTIC ONLY"
    assert all(float(row["active_override_rate"]) == 0.0 for row in metrics)
    assert all(math.isnan(float(row["selected_override_precision"])) for row in metrics)
