from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from protocol import ProtocolError, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_json, write_protocol_finalization


VIABILITY_AUDIT_NAME = "virchow2_cvae_covariance_prior_viability_audit_v1"
CONFIRMATION_ARTIFACT_NAME = "virchow2_cvae_covariance_prior_confirmation_v1"
PRIMARY_VARIANT = "pca64_beta001"
POOL_PER_SOURCE = "per_source"
POOL_SOURCE_UNION = "source_union_excluding_target"
PRIMARY_ROW = "cvae_cc_cov_shrinkage_prior_sample"
PRIMARY_SELECTION = "primary"
DIAGNOSTIC_SELECTION = "diagnostic_only"
NA = "NA"


@dataclass(frozen=True)
class CovarianceViabilityConfig:
    name: str
    artifact_root: Path
    covariance_confirmation_artifact_root: Path
    heldout_centers: tuple[str, ...]
    min_viable_cells: int
    min_viable_cells_per_center: int
    min_viable_seeds_per_center: int
    high_real_threshold: float
    viable_real_threshold: float
    borderline_real_threshold: float
    global_center_equal_mean_bacc_min: float
    mean_clipped_preservation_gap_max: float
    mean_preservation_ratio_min: float
    seed_std_max: float
    delta_bacc_vs_standard_prior_min: float
    delta_bacc_vs_diag_prior_min: float
    covariance_beats_diag_cell_fraction_min: float
    covariance_beats_diag_center_fraction_min: float
    worst_delta_vs_diag_prior_min: float
    min_cell_bacc_min: float
    min_center_mean_bacc_min: float


def load_covariance_viability_config(path: str | Path) -> CovarianceViabilityConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_covariance_viability_config(data, base_dir=base_dir)


def parse_covariance_viability_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> CovarianceViabilityConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    audit = _mapping(data, "viability_audit")
    cfg = CovarianceViabilityConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        covariance_confirmation_artifact_root=_path(base, str(inputs["covariance_confirmation_artifact_root"])),
        heldout_centers=tuple(str(v) for v in audit["heldout_centers"]),
        min_viable_cells=int(audit["min_viable_cells"]),
        min_viable_cells_per_center=int(audit["min_viable_cells_per_center"]),
        min_viable_seeds_per_center=int(audit["min_viable_seeds_per_center"]),
        high_real_threshold=float(audit["high_real_threshold"]),
        viable_real_threshold=float(audit["viable_real_threshold"]),
        borderline_real_threshold=float(audit["borderline_real_threshold"]),
        global_center_equal_mean_bacc_min=float(audit["global_center_equal_mean_bacc_min"]),
        mean_clipped_preservation_gap_max=float(audit["mean_clipped_preservation_gap_max"]),
        mean_preservation_ratio_min=float(audit["mean_preservation_ratio_min"]),
        seed_std_max=float(audit["seed_std_max"]),
        delta_bacc_vs_standard_prior_min=float(audit["delta_bacc_vs_standard_prior_min"]),
        delta_bacc_vs_diag_prior_min=float(audit["delta_bacc_vs_diag_prior_min"]),
        covariance_beats_diag_cell_fraction_min=float(audit["covariance_beats_diag_cell_fraction_min"]),
        covariance_beats_diag_center_fraction_min=float(audit["covariance_beats_diag_center_fraction_min"]),
        worst_delta_vs_diag_prior_min=float(audit["worst_delta_vs_diag_prior_min"]),
        min_cell_bacc_min=float(audit["min_cell_bacc_min"]),
        min_center_mean_bacc_min=float(audit["min_center_mean_bacc_min"]),
    )
    validate_covariance_viability_config(cfg)
    return cfg


def validate_covariance_viability_config(cfg: CovarianceViabilityConfig) -> None:
    if cfg.name != VIABILITY_AUDIT_NAME:
        raise ProtocolError(f"Viability audit experiment name must be {VIABILITY_AUDIT_NAME!r}.")
    if cfg.covariance_confirmation_artifact_root.name != CONFIRMATION_ARTIFACT_NAME:
        raise ProtocolError(f"covariance_confirmation_artifact_root must point to {CONFIRMATION_ARTIFACT_NAME!r}.")
    if cfg.min_viable_cells != 30:
        raise ProtocolError("min_viable_cells must be locked to 30.")
    if cfg.min_viable_cells_per_center != 3 or cfg.min_viable_seeds_per_center != 2:
        raise ProtocolError("Center coverage gates must be locked to 3 cells and 2 seeds per center.")
    if not (cfg.high_real_threshold == 0.80 and cfg.viable_real_threshold == 0.75 and cfg.borderline_real_threshold == 0.65):
        raise ProtocolError("variant_real_budget_bacc stratum thresholds must be locked to 0.80/0.75/0.65.")
    if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
        raise ProtocolError("heldout_centers must be locked to ['0', '1', '2', '3', '4'].")


def run_covariance_prior_viability_audit(
    cfg: CovarianceViabilityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    protocol_violations: list[str] = []
    try:
        imported = _load_imported_artifact(cfg)
        cells = _replicate_averaged_cells(cfg, imported.matrix_rows)
        conditional = [row for row in cells if _is_conditional_viability_cell(cfg, row)]
        original = [row for row in cells if _is_original_9_cell(row)]
        source_pool_cells = [
            row for row in cells
            if row["expert_pool_type"] == POOL_SOURCE_UNION
            and row["row_role"] == PRIMARY_ROW
            and row["status"] == "ok"
        ]
        stratum_summary = _stratum_summary(cfg, cells, original)
        center_seed_summary = _center_seed_summary(conditional)
        fallback_rows = _fallback_viability_rows(conditional, imported.fallback_rows)
        source_pool_summary = _source_pool_summary(cfg, source_pool_cells)
        decision = _decision(cfg, conditional, original, source_pool_summary, imported.leakage_status, stratum_summary)
        leakage_status = imported.leakage_status
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        cells = []
        conditional = []
        original = []
        stratum_summary = []
        center_seed_summary = []
        fallback_rows = []
        source_pool_summary = []
        leakage_status = "FAIL"
        decision = _decision(cfg, [], [], [], leakage_status, [])

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    if leakage_status != "PASS" and "imported_leakage_status_not_PASS" not in protocol_violations:
        leakage = build_leakage_report(
            target_support_labels_for_selection=False,
            target_eval_labels_for_scoring_only=True,
            target_expert_excluded=True,
            oracle_rows_diagnostic_only=True,
            extra_violations=tuple(protocol_violations) + ("imported_leakage_status_not_PASS",),
        )
        decision = {**decision, "primary_verdict": "PROTOCOL_FAIL"}
    _write_artifacts(
        root,
        cfg,
        conditional_rows=conditional,
        stratum_rows=stratum_summary,
        original_rows=original,
        center_seed_rows=center_seed_summary,
        fallback_rows=fallback_rows,
        source_pool_rows=source_pool_summary,
        decision=decision,
        leakage=leakage,
    )
    return root


@dataclass(frozen=True)
class ImportedArtifact:
    matrix_rows: tuple[dict[str, str], ...]
    fallback_rows: tuple[dict[str, str], ...]
    leakage_status: str
    original_verdict: str
    decision_cell_set_hash: str


def _load_imported_artifact(cfg: CovarianceViabilityConfig) -> ImportedArtifact:
    root = cfg.covariance_confirmation_artifact_root
    required = (
        "tables/covariance_prior_downstream_matrix.csv",
        "tables/covariance_prior_gap_summary.csv",
        "tables/covariance_fallback_audit.csv",
        "tables/covariance_prior_parameter_manifest.csv",
        "tables/source_pool_covariance_prior_summary.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "manifests/protocol_manifest.json",
    )
    missing = [rel for rel in required if not (root / rel).exists()]
    if missing:
        raise ProtocolError(f"Missing covariance confirmation artifact files: {missing}")
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    if protocol.get("experiment_name") != CONFIRMATION_ARTIFACT_NAME:
        raise ProtocolError("Imported protocol manifest is not the covariance confirmation artifact.")
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = _read_csv(root / "tables" / "covariance_prior_downstream_matrix.csv")
    fallback = _read_csv(root / "tables" / "covariance_fallback_audit.csv")
    _require_columns(
        matrix,
        {
            "experiment_seed",
            "heldout_center",
            "expert_id",
            "expert_pool_type",
            "variant_id",
            "row_role",
            "prior_method",
            "replicate_seed",
            "source_utility_stratum_reference",
            "variant_real_budget_bacc",
            "bacc",
            "macro_f1",
            "total_covariance_prior_gap",
            "delta_bacc_vs_standard_prior",
            "delta_bacc_vs_diag_prior",
            "gap_reduction_vs_standard_prior",
            "gap_reduction_vs_diag_prior",
            "covariance_fallback_used",
            "fallback_reason",
            "selection_source",
            "status",
        },
        "covariance_prior_downstream_matrix.csv",
    )
    decision_text = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")
    return ImportedArtifact(
        matrix_rows=tuple(matrix),
        fallback_rows=tuple(fallback),
        leakage_status=str(leakage.get("status", "")),
        original_verdict=_extract_summary_value(decision_text, "Primary verdict"),
        decision_cell_set_hash=str(protocol.get("decision_cell_set_hash", "")),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _require_columns(rows: Sequence[Mapping[str, str]], required: set[str], name: str) -> None:
    if not rows:
        raise ProtocolError(f"{name} is empty.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ProtocolError(f"{name} is missing fields: {sorted(missing)}")


def _extract_summary_value(text: str, label: str) -> str:
    prefix = f"- {label}: `"
    for line in text.splitlines():
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix):-1]
    return ""


def _replicate_averaged_cells(cfg: CovarianceViabilityConfig, rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("variant_id") not in {PRIMARY_VARIANT, "source_union_pca64_beta001_diagnostic"}:
            continue
        key = (
            str(row["experiment_seed"]),
            str(row["heldout_center"]),
            str(row["expert_id"]),
            str(row["expert_pool_type"]),
            str(row["variant_id"]),
            str(row["row_role"]),
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for key, subset in sorted(grouped.items(), key=lambda item: item[0]):
        real = _mean_field(subset, "variant_real_budget_bacc")
        bacc = _mean_field(subset, "bacc")
        absolute_gap = real - bacc
        clipped_gap = max(0.0, absolute_gap) if math.isfinite(absolute_gap) else math.nan
        preservation_ratio = bacc / real if math.isfinite(bacc) and math.isfinite(real) and real > 0 else math.nan
        fallback_used = any(str(row.get("covariance_fallback_used")) == "True" for row in subset)
        labels = _failure_labels(cfg, real, bacc, clipped_gap, _mean_field(subset, "delta_bacc_vs_diag_prior"), fallback_used)
        row = {
            "experiment_seed": key[0],
            "heldout_center": key[1],
            "expert_id": key[2],
            "expert_pool_type": key[3],
            "variant_id": key[4],
            "row_role": key[5],
            "prior_method": key[5],
            "replicate_count": len(subset),
            "variant_real_budget_bacc": real,
            "variant_real_stratum": _variant_real_stratum(cfg, real),
            "source_utility_stratum_reference": str(subset[0].get("source_utility_stratum_reference", "")),
            "bacc": bacc,
            "macro_f1": _mean_field(subset, "macro_f1"),
            "preservation_ratio": preservation_ratio,
            "absolute_preservation_gap": absolute_gap,
            "clipped_preservation_gap": clipped_gap,
            "exceeds_real_budget_reference": bacc > real if math.isfinite(bacc) and math.isfinite(real) else False,
            "total_covariance_prior_gap": _mean_field(subset, "total_covariance_prior_gap"),
            "delta_bacc_vs_standard_prior": _mean_field(subset, "delta_bacc_vs_standard_prior"),
            "delta_bacc_vs_diag_prior": _mean_field(subset, "delta_bacc_vs_diag_prior"),
            "gap_reduction_vs_standard_prior": _mean_field(subset, "gap_reduction_vs_standard_prior"),
            "gap_reduction_vs_diag_prior": _mean_field(subset, "gap_reduction_vs_diag_prior"),
            "covariance_fallback_used": fallback_used,
            "fallback_reason": "|".join(sorted({str(r.get("fallback_reason", "")) for r in subset if str(r.get("fallback_reason", ""))})),
            "cell_failure_labels": ";".join(labels),
            "primary_failure_label": labels[0] if labels else "",
            "selection_source": str(subset[0].get("selection_source", "")),
            "status": "ok",
        }
        out.append(row)
    return out


def _failure_labels(
    cfg: CovarianceViabilityConfig,
    real: float,
    bacc: float,
    clipped_gap: float,
    delta_diag: float,
    fallback_used: bool,
) -> list[str]:
    labels = []
    if real >= cfg.high_real_threshold and (bacc < 0.60 or clipped_gap > 0.08):
        labels.append("PRIOR_PRESERVATION_FAILURE")
    if delta_diag < 0.0:
        labels.append("DIAGONAL_OUTPERFORMS_COVARIANCE")
    if fallback_used:
        labels.append("FALLBACK_USED")
    if cfg.viable_real_threshold <= real < cfg.high_real_threshold:
        labels.append("VARIANT_REAL_VIABLE_NOT_HIGH")
    if cfg.borderline_real_threshold <= real < cfg.viable_real_threshold:
        labels.append("BORDERLINE_VARIANT_CEILING")
    if real < cfg.borderline_real_threshold:
        labels.append("VARIANT_FRAME_CEILING_LIMITED")
    return labels


def _variant_real_stratum(cfg: CovarianceViabilityConfig, value: float) -> str:
    if value >= cfg.high_real_threshold:
        return "variant_real_high"
    if value >= cfg.viable_real_threshold:
        return "variant_real_viable"
    if value >= cfg.borderline_real_threshold:
        return "variant_real_borderline"
    return "variant_real_weak"


def _is_conditional_viability_cell(cfg: CovarianceViabilityConfig, row: Mapping[str, object]) -> bool:
    return (
        row.get("variant_id") == PRIMARY_VARIANT
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("row_role") == PRIMARY_ROW
        and row.get("selection_source") == PRIMARY_SELECTION
        and row.get("status") == "ok"
        and _float(row.get("variant_real_budget_bacc")) >= cfg.high_real_threshold
    )


def _is_original_9_cell(row: Mapping[str, object]) -> bool:
    return (
        row.get("variant_id") == PRIMARY_VARIANT
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("row_role") == PRIMARY_ROW
        and row.get("selection_source") == PRIMARY_SELECTION
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
    )


def _stratum_summary(
    cfg: CovarianceViabilityConfig,
    cells: Sequence[Mapping[str, object]],
    original: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates = [
        row for row in cells
        if row.get("variant_id") == PRIMARY_VARIANT
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("row_role") == PRIMARY_ROW
        and row.get("selection_source") == PRIMARY_SELECTION
        and row.get("status") == "ok"
    ]
    total = len(candidates)
    original_ids = {_cell_id(row) for row in original}
    rows = []
    for stratum in ("variant_real_high", "variant_real_viable", "variant_real_borderline", "variant_real_weak"):
        subset = [row for row in candidates if row.get("variant_real_stratum") == stratum]
        rows.append(
            {
                "variant_real_stratum": stratum,
                "n_cells": len(subset),
                "fraction_of_candidate_cells": len(subset) / total if total else math.nan,
                "mean_bacc": _mean_field(subset, "bacc"),
                "mean_variant_real_budget_bacc": _mean_field(subset, "variant_real_budget_bacc"),
                "mean_clipped_preservation_gap": _mean_field(subset, "clipped_preservation_gap"),
                "mean_preservation_ratio": _mean_field(subset, "preservation_ratio"),
                "n_by_center": json.dumps(_count_by(subset, "heldout_center"), sort_keys=True),
                "n_by_seed": json.dumps(_count_by(subset, "experiment_seed"), sort_keys=True),
                "n_by_expert": json.dumps(_count_by(subset, "expert_id"), sort_keys=True),
                "excluded_original_9_cells": sum(1 for row in original if _cell_id(row) not in {_cell_id(v) for v in subset}),
                "excluded_center3_cells": sum(1 for row in candidates if row.get("heldout_center") == "3" and row.get("variant_real_stratum") != stratum),
            }
        )
    high = [row for row in candidates if row.get("variant_real_stratum") == "variant_real_high"]
    rows.append(
        {
            "variant_real_stratum": "selection_denominator",
            "n_total_candidate_cells": total,
            "n_high_real_budget_cells": len(high),
            "n_viable_075_080_cells": sum(1 for row in candidates if row.get("variant_real_stratum") == "variant_real_viable"),
            "n_borderline_065_075_cells": sum(1 for row in candidates if row.get("variant_real_stratum") == "variant_real_borderline"),
            "n_weak_lt_065_cells": sum(1 for row in candidates if row.get("variant_real_stratum") == "variant_real_weak"),
            "fraction_high_real_budget": len(high) / total if total else math.nan,
            "n_high_real_budget_by_center": json.dumps(_count_by(high, "heldout_center"), sort_keys=True),
            "n_high_real_budget_by_seed": json.dumps(_count_by(high, "experiment_seed"), sort_keys=True),
            "n_high_real_budget_by_expert": json.dumps(_count_by(high, "expert_id"), sort_keys=True),
            "excluded_original_9_cells": sum(1 for row in original if _cell_id(row) not in {_cell_id(v) for v in high}),
            "excluded_center3_cells": sum(1 for row in original if row.get("heldout_center") == "3" and _cell_id(row) not in {_cell_id(v) for v in high}),
            "original_9_cell_ids_hash": _hash_strings(sorted(original_ids)),
        }
    )
    return rows


def _center_seed_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_center_seed: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        by_center_seed.setdefault((str(row["heldout_center"]), str(row["experiment_seed"])), []).append(row)
    out = []
    for (center, seed), subset in sorted(by_center_seed.items()):
        out.append(
            {
                "heldout_center": center,
                "experiment_seed": seed,
                "n_cells": len(subset),
                "mean_bacc": _mean_field(subset, "bacc"),
                "mean_clipped_preservation_gap": _mean_field(subset, "clipped_preservation_gap"),
                "mean_delta_bacc_vs_diag_prior": _mean_field(subset, "delta_bacc_vs_diag_prior"),
                "min_cell_bacc": _min_field(subset, "bacc"),
            }
        )
    return out


def _fallback_viability_rows(
    conditional: Sequence[Mapping[str, object]],
    _fallback_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": row["experiment_seed"],
            "heldout_center": row["heldout_center"],
            "expert_id": row["expert_id"],
            "covariance_fallback_used": row["covariance_fallback_used"],
            "fallback_reason": row["fallback_reason"],
            "bacc": row["bacc"],
            "variant_real_budget_bacc": row["variant_real_budget_bacc"],
            "primary_failure_label": row["primary_failure_label"],
        }
        for row in conditional
        if bool(row.get("covariance_fallback_used"))
    ]


def _source_pool_summary(cfg: CovarianceViabilityConfig, source_pool_cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    high = [row for row in source_pool_cells if _float(row.get("variant_real_budget_bacc")) >= cfg.high_real_threshold]
    return [
        {
            "population": "source_union_diagnostic_only_all",
            **_metric_summary(cfg, source_pool_cells),
        },
        {
            "population": "source_union_diagnostic_only_high_real_budget",
            **_metric_summary(cfg, high),
        },
    ]


def _decision(
    cfg: CovarianceViabilityConfig,
    conditional: Sequence[Mapping[str, object]],
    original: Sequence[Mapping[str, object]],
    source_pool_summary: Sequence[Mapping[str, object]],
    leakage_status: str,
    stratum_summary: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    stats = _metric_summary(cfg, conditional)
    original_flags = _original_flags(cfg, original)
    flags = list(original_flags)
    expected_centers = set(cfg.heldout_centers)
    observed_centers = set(json.loads(str(stats["per_center_bacc"])).keys()) if str(stats["per_center_bacc"]) not in {"", "nan"} else set()
    coverage_ok = (
        expected_centers.issubset(observed_centers)
        and _int(stats["min_viable_cells_per_center"]) >= cfg.min_viable_cells_per_center
        and _int(stats["min_viable_seeds_per_center"]) >= cfg.min_viable_seeds_per_center
    )
    if not coverage_ok:
        flags.append("CENTER_COVERAGE_INSUFFICIENT_FOR_PASS")
    if any(bool(row.get("covariance_fallback_used")) for row in conditional):
        flags.append("DIAG_FALLBACK_USED_IN_VIABLE_CELL")
    if _low_stratum_has_high_utility(stratum_summary):
        flags.append("LOW_STRATUM_HAS_HIGH_COVARIANCE_UTILITY")
    if any(row.get("population") == "source_union_diagnostic_only_high_real_budget" and _float(row.get("global_center_equal_mean_bacc")) >= 0.85 for row in source_pool_summary):
        flags.append("SOURCE_POOL_VIABLE_STRONG")

    numeric_pass = (
        _int(stats["n_viable_cells"]) >= cfg.min_viable_cells
        and coverage_ok
        and _float(stats["global_center_equal_mean_bacc"]) >= cfg.global_center_equal_mean_bacc_min
        and _float(stats["mean_clipped_preservation_gap"]) <= cfg.mean_clipped_preservation_gap_max
        and _float(stats["mean_preservation_ratio"]) >= cfg.mean_preservation_ratio_min
        and _float(stats["seed_std"]) <= cfg.seed_std_max
        and _float(stats["mean_delta_bacc_vs_standard_prior"]) >= cfg.delta_bacc_vs_standard_prior_min
        and _float(stats["mean_delta_bacc_vs_diag_prior"]) >= cfg.delta_bacc_vs_diag_prior_min
        and _float(stats["covariance_beats_diag_cell_fraction"]) >= cfg.covariance_beats_diag_cell_fraction_min
        and _float(stats["covariance_beats_diag_center_fraction"]) >= cfg.covariance_beats_diag_center_fraction_min
        and _float(stats["worst_delta_vs_diag_prior"]) >= cfg.worst_delta_vs_diag_prior_min
        and _float(stats["min_cell_bacc"]) >= cfg.min_cell_bacc_min
        and _float(stats["min_center_mean_bacc"]) >= cfg.min_center_mean_bacc_min
        and leakage_status == "PASS"
    )
    fallback_used = any(bool(row.get("covariance_fallback_used")) for row in conditional)
    collapse = _float(stats["min_cell_bacc"]) < cfg.min_cell_bacc_min or _float(stats["min_center_mean_bacc"]) < cfg.min_center_mean_bacc_min
    verdict = "VIABLE_CONDITIONAL_COV_PRIOR_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif numeric_pass and not fallback_used:
        verdict = "VIABLE_CONDITIONAL_COV_PRIOR_PASS_DIAGNOSTIC"
    elif numeric_pass and fallback_used:
        verdict = "VIABLE_CONDITIONAL_COV_PRIOR_HYBRID_PASS_DIAGNOSTIC"
    elif _float(stats["global_center_equal_mean_bacc"]) >= cfg.global_center_equal_mean_bacc_min and (
        _float(stats["covariance_beats_diag_cell_fraction"]) < cfg.covariance_beats_diag_cell_fraction_min
        or _float(stats["worst_delta_vs_diag_prior"]) < cfg.worst_delta_vs_diag_prior_min
    ):
        verdict = "VIABLE_CONDITIONAL_COV_PRIOR_PARTIAL_NO_FULLCOV_GAIN"
    elif collapse:
        verdict = "VIABLE_CONDITIONAL_COV_PRIOR_UNSTABLE"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(dict.fromkeys(flags)),
        **stats,
    }


def _metric_summary(cfg: CovarianceViabilityConfig, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    center_seed = _center_seed_means(rows)
    seed_groups: dict[str, list[Mapping[str, object]]] = {}
    center_groups: dict[str, list[Mapping[str, object]]] = {}
    for row in center_seed:
        seed_groups.setdefault(str(row["experiment_seed"]), []).append(row)
        center_groups.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = {seed: _mean_field(values, "mean_bacc") for seed, values in seed_groups.items()}
    center_means = {center: _mean_field(values, "mean_bacc") for center, values in center_groups.items()}
    center_delta_means = {center: _mean_field(values, "mean_delta_bacc_vs_diag_prior") for center, values in center_groups.items()}
    per_center_seed_counts: dict[str, set[str]] = {}
    per_center_counts: dict[str, int] = {}
    for row in rows:
        center = str(row["heldout_center"])
        per_center_seed_counts.setdefault(center, set()).add(str(row["experiment_seed"]))
        per_center_counts[center] = per_center_counts.get(center, 0) + 1
    return {
        "n_viable_cells": len(rows),
        "n_centers": len(center_groups),
        "n_seeds": len(seed_groups),
        "min_viable_cells_per_center": min(per_center_counts.values()) if per_center_counts else 0,
        "min_viable_seeds_per_center": min((len(v) for v in per_center_seed_counts.values()), default=0),
        "global_center_equal_mean_bacc": _mean(list(seed_means.values())),
        "global_cell_weighted_mean_bacc": _mean_field(rows, "bacc"),
        "mean_clipped_preservation_gap": _mean_field(rows, "clipped_preservation_gap"),
        "mean_preservation_ratio": _mean_field(rows, "preservation_ratio"),
        "seed_std": _std(list(seed_means.values())),
        "mean_delta_bacc_vs_standard_prior": _mean_field(rows, "delta_bacc_vs_standard_prior"),
        "mean_delta_bacc_vs_diag_prior": _mean_field(rows, "delta_bacc_vs_diag_prior"),
        "covariance_beats_diag_cell_fraction": _mean([1.0 if _float(row["delta_bacc_vs_diag_prior"]) > 0 else 0.0 for row in rows]),
        "covariance_beats_diag_center_fraction": _mean([1.0 if value > 0 else 0.0 for value in center_delta_means.values()]),
        "worst_delta_vs_diag_prior": _min_field(rows, "delta_bacc_vs_diag_prior"),
        "min_cell_bacc": _min_field(rows, "bacc"),
        "min_center_mean_bacc": min(center_means.values()) if center_means else math.nan,
        "per_seed_bacc": json.dumps(seed_means, sort_keys=True),
        "per_center_bacc": json.dumps(center_means, sort_keys=True),
    }


def _center_seed_means(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    out = []
    for (seed, center), subset in sorted(groups.items()):
        out.append(
            {
                "experiment_seed": seed,
                "heldout_center": center,
                "mean_bacc": _mean_field(subset, "bacc"),
                "mean_delta_bacc_vs_diag_prior": _mean_field(subset, "delta_bacc_vs_diag_prior"),
            }
        )
    return out


def _original_flags(cfg: CovarianceViabilityConfig, original: Sequence[Mapping[str, object]]) -> list[str]:
    flags = []
    ceiling_limited = [
        row for row in original
        if _float(row.get("variant_real_budget_bacc")) < cfg.high_real_threshold
    ]
    prior_fail = [
        row for row in original
        if _float(row.get("variant_real_budget_bacc")) >= cfg.high_real_threshold
        and (_float(row.get("bacc")) < 0.60 or _float(row.get("clipped_preservation_gap")) > 0.08)
    ]
    center3_borderline = [
        row for row in original
        if row.get("heldout_center") == "3"
        and _float(row.get("variant_real_budget_bacc")) < cfg.high_real_threshold
    ]
    if ceiling_limited:
        flags.append("ORIGINAL_9_CELL_CEILING_LIMITED")
    if prior_fail:
        flags.append("ORIGINAL_9_CELL_PRIOR_FAILURE")
    if center3_borderline:
        flags.append("CENTER3_BORDERLINE")
    return flags


def _low_stratum_has_high_utility(stratum_rows: Sequence[Mapping[str, object]]) -> bool:
    for row in stratum_rows:
        if row.get("variant_real_stratum") == "variant_real_weak" and _float(row.get("mean_bacc")) >= 0.80:
            return True
    return False


def _count_by(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _cell_id(row: Mapping[str, object]) -> str:
    return f"{row.get('experiment_seed')}|{row.get('heldout_center')}|{row.get('expert_id')}"


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return _mean([_float(row.get(field)) for row in rows])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else math.nan


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = _mean(finite)
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _float(value: object) -> float:
    if value in ("", NA, None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_float(value: object) -> str:
    number = _float(value)
    return "nan" if math.isnan(number) else f"{number:.4f}"


def _hash_strings(values: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    path = _resolve_config_path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ProtocolError("YAML config parsing requires PyYAML unless the file is JSON syntax.") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ProtocolError("Config root must be a mapping.")
        return data


def _resolve_config_path(path: Path) -> Path:
    if path.exists():
        return path
    parts = path.parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "cvae_rebuild" and parts[idx + 1] == "cvae_rebuild":
            collapsed = Path(*parts[: idx + 1], *parts[idx + 2 :])
            if collapsed.exists():
                return collapsed
    if not parts or parts[0] != "cvae_rebuild":
        return path
    local = Path(*path.parts[1:])
    return local if local.exists() else path


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_artifacts(
    root: Path,
    cfg: CovarianceViabilityConfig,
    *,
    conditional_rows: Sequence[Mapping[str, object]],
    stratum_rows: Sequence[Mapping[str, object]],
    original_rows: Sequence[Mapping[str, object]],
    center_seed_rows: Sequence[Mapping[str, object]],
    fallback_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
) -> None:
    write_csv_rows(root / "tables" / "conditional_viability_cells.csv", conditional_rows, columns=_cell_columns())
    write_csv_rows(root / "tables" / "variant_real_stratum_summary.csv", stratum_rows)
    write_csv_rows(root / "tables" / "original_9_cell_failure_audit.csv", original_rows, columns=_cell_columns())
    write_csv_rows(root / "tables" / "center_seed_stability_summary.csv", center_seed_rows)
    write_csv_rows(root / "tables" / "fallback_viability_audit.csv", fallback_rows)
    write_csv_rows(root / "tables" / "source_pool_viability_summary.csv", source_pool_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest={
            "schema_version": "cvae_rebuild_covariance_prior_viability_audit_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "read_only_variant_ceiling_viability_audit",
            "imported_artifact": str(cfg.covariance_confirmation_artifact_root),
            "target_eval_labels_for_scoring_only": True,
            "target_scored_variant_real_budget_used_for_diagnostic_stratification": True,
            "claim_boundary": "conditional diagnostic viability only; does not replace covariance confirmation verdict and does not evaluate routing",
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage.status)


def _cell_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "variant_real_budget_bacc",
        "variant_real_stratum",
        "source_utility_stratum_reference",
        "bacc",
        "macro_f1",
        "preservation_ratio",
        "absolute_preservation_gap",
        "clipped_preservation_gap",
        "exceeds_real_budget_reference",
        "total_covariance_prior_gap",
        "delta_bacc_vs_standard_prior",
        "delta_bacc_vs_diag_prior",
        "gap_reduction_vs_standard_prior",
        "gap_reduction_vs_diag_prior",
        "covariance_fallback_used",
        "fallback_reason",
        "cell_failure_labels",
        "primary_failure_label",
        "status",
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Covariance Prior Viability Audit v1",
            "",
            "## Summary",
            "",
            f"- Primary verdict: `{decision.get('primary_verdict', 'VIABLE_CONDITIONAL_COV_PRIOR_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Conditional viable cells: {decision.get('n_viable_cells', 0)}",
            f"- Center-equal mean BACC: {_format_float(decision.get('global_center_equal_mean_bacc'))}",
            f"- Cell-weighted mean BACC: {_format_float(decision.get('global_cell_weighted_mean_bacc'))}",
            f"- Mean clipped preservation gap: {_format_float(decision.get('mean_clipped_preservation_gap'))}",
            f"- Mean preservation ratio: {_format_float(decision.get('mean_preservation_ratio'))}",
            f"- Delta BACC vs standard prior: {_format_float(decision.get('mean_delta_bacc_vs_standard_prior'))}",
            f"- Delta BACC vs diagonal prior: {_format_float(decision.get('mean_delta_bacc_vs_diag_prior'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Claim Boundary",
            "",
            "This audit does not replace the original 9-cell covariance-prior verdict.",
            "It appends a conditional viability statement over target-scored high-real-budget cells.",
            "The target-scored real-budget reference is diagnostic evidence and cannot define a deployable router.",
            "This slice does not evaluate routing, support-NELBO selection, metadata selection, top-k composition, or formal privacy.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: CovarianceViabilityConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "covariance_confirmation_artifact_root": str(cfg.covariance_confirmation_artifact_root),
        "heldout_centers": list(cfg.heldout_centers),
        "min_viable_cells": cfg.min_viable_cells,
        "min_viable_cells_per_center": cfg.min_viable_cells_per_center,
        "min_viable_seeds_per_center": cfg.min_viable_seeds_per_center,
        "high_real_threshold": cfg.high_real_threshold,
        "viable_real_threshold": cfg.viable_real_threshold,
        "borderline_real_threshold": cfg.borderline_real_threshold,
        "global_center_equal_mean_bacc_min": cfg.global_center_equal_mean_bacc_min,
        "mean_clipped_preservation_gap_max": cfg.mean_clipped_preservation_gap_max,
        "mean_preservation_ratio_min": cfg.mean_preservation_ratio_min,
        "seed_std_max": cfg.seed_std_max,
        "delta_bacc_vs_standard_prior_min": cfg.delta_bacc_vs_standard_prior_min,
        "delta_bacc_vs_diag_prior_min": cfg.delta_bacc_vs_diag_prior_min,
        "covariance_beats_diag_cell_fraction_min": cfg.covariance_beats_diag_cell_fraction_min,
        "covariance_beats_diag_center_fraction_min": cfg.covariance_beats_diag_center_fraction_min,
        "worst_delta_vs_diag_prior_min": cfg.worst_delta_vs_diag_prior_min,
        "min_cell_bacc_min": cfg.min_cell_bacc_min,
        "min_center_mean_bacc_min": cfg.min_center_mean_bacc_min,
    }
